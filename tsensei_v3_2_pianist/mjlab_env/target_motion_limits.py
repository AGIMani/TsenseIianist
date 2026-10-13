"""Final wrist target velocity/acceleration limits with discrete stopping room.

These bound differences of issued float32 position targets, not physical qvel
or qacc. Position feasibility includes all future braking steps, so reaching a
position endpoint never requires a later clip that violates acceleration.
"""
import torch


def _inward_float32(value, *, lower):
    rounded = value.to(torch.float32)
    outside = rounded.to(torch.float64) < value if lower else rounded.to(torch.float64) > value
    direction = torch.full_like(rounded, float('inf') if lower else -float('inf'))
    return torch.where(outside, torch.nextafter(rounded, direction), rounded)


class TargetMotionLimiter:
    """Stateful final projection; call once, after every other target transform."""

    def __init__(self, contract, *, num_envs, device, reset_action,
                 physical_low=None, physical_high=None, brake_to_target=False):
        self.brake_to_target = bool(brake_to_target)
        rows = {name: row for name, row in contract['channels'].items()
                if 'max_acceleration' in row}
        if not rows:
            raise ValueError('Final target motion limiter requires explicit max_acceleration')
        self.dt = float(contract['control_dt'])
        self.indices = torch.tensor([row['index'] for row in rows.values()], device=device, dtype=torch.long)
        tensor = lambda values: torch.as_tensor(values, device=device, dtype=torch.float64)
        low = tensor([row['min'] for row in rows.values()])
        high = tensor([row['max'] for row in rows.values()])
        if physical_low is not None:
            low = torch.maximum(low, tensor(physical_low)[self.indices])
        if physical_high is not None:
            high = torch.minimum(high, tensor(physical_high)[self.indices])
        self.low = _inward_float32(low, lower=True).to(torch.float64)
        self.high = _inward_float32(high, lower=False).to(torch.float64)
        self.max_velocity = tensor([row['max_velocity'] for row in rows.values()])
        self.max_acceleration = tensor([row['max_acceleration'] for row in rows.values()])
        self.reset_target = torch.as_tensor(reset_action, device=device, dtype=torch.float32)[self.indices].double()
        if (not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all()
                or not (self.low < self.high).all()
                or not ((self.reset_target >= self.low) & (self.reset_target <= self.high)).all()):
            raise ValueError('Final wrist motion bounds must contain the actual float32 reset target')

        # Float32 commands lie on a lattice. Reserve four worst-case position
        # ulps in the braking calculation, so a future maximum-deceleration
        # target can be rounded inward while respecting the requested bound.
        # Actual per-step acceleration still uses the exact user limit below.
        magnitude = torch.maximum(self.low.abs(), self.high.abs()).float()
        spacing = (torch.nextafter(magnitude, torch.full_like(magnitude, float('inf'))) - magnitude).double()
        self.braking_acceleration = self.max_acceleration - 4 * spacing / self.dt**2
        if not (self.braking_acceleration > 0).all():
            raise ValueError('Wrist acceleration/control_dt is too small for float32 target resolution')
        self.previous = self.reset_target.expand(num_envs, -1).clone()
        self.velocity = torch.zeros_like(self.previous)
        self.contract = dict(controller='wrist_target_braking', control_dt=self.dt,
            channels={name: dict(index=row['index'], min=float(self.low[i]), max=float(self.high[i]),
                max_velocity=row['max_velocity'], max_acceleration=row['max_acceleration'],
                braking_acceleration=float(self.braking_acceleration[i]),
                position_unit=row['position_unit'], velocity_unit=row['velocity_unit'],
                acceleration_unit=row['acceleration_unit'],
                reset_target=float(self.reset_target[i])) for i, (name, row) in enumerate(rows.items())},
            units='per-channel: m, m/s, m/s^2 for TX/TY; rad, rad/s, rad/s^2 for rotation', initial_velocity=0.,
            position_feasibility='discrete stop including current step; float32 inward rounding',
            application='after target limiter, coupling, clearance and hard position clip')
        if self.brake_to_target:
            self.contract.update(
                target_feasibility='discrete braking to original wrist request as well as position boundaries',
                braking_input='original wrist request before legacy one-step slew; clipped to final position bounds')

    def reset(self, ids):
        self.previous[ids] = self.reset_target
        self.velocity[ids] = 0.

    def _stopping_speed(self, gap):
        # Invert D(v) = dt * sum_{k>=0} max(v-k*A*dt, 0).
        # n is the segment containing the maximum feasible initial speed.
        gap = gap.clamp_min(0.)
        delta = self.braking_acceleration * self.dt
        n = torch.floor((torch.sqrt(1 + 8 * gap / (delta * self.dt)) - 1) / 2)
        return (gap / self.dt + delta * n * (n + 1) / 2) / (n + 1)

    def apply(self, requested, *, braking_target=None):
        if (requested.dtype != torch.float32 or requested.ndim != 2
                or requested.shape != (self.previous.shape[0], 51)):
            raise ValueError('Final motion limiter requires (num_envs, 51) float32 physical targets')
        delta = self.max_acceleration * self.dt
        lower_velocity = torch.maximum(-self.max_velocity, self.velocity - delta)
        upper_velocity = torch.minimum(self.max_velocity, self.velocity + delta)
        lower_velocity = torch.maximum(lower_velocity, -self._stopping_speed(self.previous - self.low))
        upper_velocity = torch.minimum(upper_velocity, self._stopping_speed(self.high - self.previous))
        lower = torch.maximum(self.low, self.previous + lower_velocity * self.dt)
        upper = torch.minimum(self.high, self.previous + upper_velocity * self.dt)
        lower = _inward_float32(lower, lower=True)
        upper = _inward_float32(upper, lower=False)
        values = requested[:, self.indices]
        desired = values
        if self.brake_to_target:
            if braking_target is None or braking_target.shape != requested.shape:
                raise ValueError('Rigid wrist target braking requires the original pre-slew request')
            # The legacy first slew may hide a distant goal behind a one-step
            # target. Brake toward the real goal, not that artificial endpoint.
            goal = braking_target[:, self.indices].double().clamp(self.low, self.high)
            gap = goal-self.previous
            speed = torch.minimum(self.max_velocity, self._stopping_speed(gap.abs()))
            desired = (self.previous + gap.sign()*speed*self.dt).float()
        issued = torch.maximum(lower, torch.minimum(upper, desired))
        applied = requested.clone()
        applied[:, self.indices] = issued
        limited = torch.zeros_like(requested, dtype=torch.bool)
        limited[:, self.indices] = issued != values
        # Commit the real quantized command, never an intermediate or requested
        # velocity. No transform may change these channels after this call.
        exact = issued.double()
        self.velocity.copy_((exact - self.previous) / self.dt)
        self.previous.copy_(exact)
        return applied, limited
