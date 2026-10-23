"""Continuous Cartesian palm IK and hard-limited joint trajectories.

All stepping is Torch on the simulation device. CPU construction reads only
project-local URDFs. q/v histories are float64 so finite differences at 200Hz
are not dominated by float32 rounding. Rendered transforms use MuJoCo's dtype.
"""
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .arm_options import ASSETS, SIDES, canonical
from .arm_ik import rot, quaternion


def matrix_rpy(value):
    return Rotation.from_euler('xyz', value).as_matrix()


def chain_constants(side, options):
    source = ET.parse(ASSETS / side / 'arm.urdf').getroot()
    nodes = {j.get('name'): j for j in source.findall('joint')}
    result = dict(joint_pos=[], joint_rot=[], axis=[])
    for index in range(1, 8):
        joint = nodes[f'Joint{index}_{side[0].upper()}']
        origin = joint.find('origin')
        result['joint_pos'].append(np.fromstring(origin.get('xyz', '0 0 0'), sep=' '))
        result['joint_rot'].append(matrix_rpy(np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')))
        result['axis'].append(np.fromstring(joint.find('axis').get('xyz'), sep=' '))
    tool = options['link7_to_hand'][side]
    return dict(**{k: np.asarray(v) for k, v in result.items()},
                tool_pos=np.asarray(tool['xyz_m']), tool_rot=matrix_rpy(tool['rpy_rad']))


def rotation_error(actual, target):
    q = quaternion(target @ actual.transpose(-1, -2))
    q = torch.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    length = xyz.norm(dim=-1, keepdim=True)
    return xyz * (2 * torch.atan2(length, q[..., :1].clamp_min(0)) / length.clamp_min(1e-12))


class ArmFollower:
    """World-frame palm targets lead a single rigid arm/tool/hand chain.

    IK maintains a continuous reference independently of the rate-limited
    trajectory. The executed trajectory applies only the current-step
    position, velocity, acceleration and coupling bounds; it does not add a
    goal-distance or anticipatory target-braking rule. Pose error is between
    request and executed assembly, never an arm/hand attachment gap.
    """

    observation_size = 54

    def __init__(self, options, initial_poses, *, num_envs, dt, device='cpu'):
        self.options = canonical(options)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.num_envs = num_envs
        self.tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.device)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive arm physics timestep required')
        self.sides = {}
        self.weight = options['orientation_weight_m']
        self.coupling = options['coupling']
        self.coupling_j6 = int(self.coupling['j6_index'])
        self.coupling_j7 = int(self.coupling['j7_index'])
        self.coupling_coefficient = float(self.coupling['coefficient'])
        self.coupling_sum_limit = float(self.coupling['sum_limit_rad'])
        self.eye = torch.eye(6, dtype=torch.float64, device=self.device)
        self.fractions = self.tensor([1., .5, .25, .125, 0.])
        self.contract = dict(controller=options['controller'], physics_dt=dt,
            wrist_execution='policy_cartesian_palm_displacement_to_IK_to_single_rigid_assembly', initial_velocity='zero',
            trajectory='discrete_rate_limited_with_position_velocity_acceleration_bounds',
            reference_seed='previous_reference_not_rate_limited_execution',
            observation_fields=[f'{s}_{field}' for s in SIDES for field in
                ('q_J1_J7_rad', 'v_J1_J7_rad_s', 'reference_J1_J7_rad', 'position_error_xyz_m', 'rotation_error_xyz_rad')],
            base={}, options=options)
        for side in SIDES:
            constants = chain_constants(side, options)
            a = {k: self.tensor(v) for k, v in constants.items()}
            a['base_pos'] = self.tensor([0., 0., 0.])
            a['base_rot'] = self.tensor(np.eye(3))
            a['q0'] = self.tensor(options['initial_q_rad'][side])
            # Calibrate on CPU once so serialized bases do not depend on CUDA
            # architecture, world count or backend transcendental rounding.
            p, r = np.zeros(3), np.eye(3)
            for i, angle in enumerate(options['initial_q_rad'][side]):
                p = p + r @ constants['joint_pos'][i]
                r = r @ constants['joint_rot'][i] @ Rotation.from_rotvec(constants['axis'][i] * angle).as_matrix()
            p = p + r @ constants['tool_pos']; r = r @ constants['tool_rot']
            target_p, target_r = initial_poses[side]
            base_r = np.asarray(target_r) @ r.T
            a['base_rot'] = self.tensor(base_r)
            a['base_pos'] = self.tensor(np.asarray(target_p) - base_r @ p)
            rows = options['joint_limits'][side]
            for name, key in (('low', 'min'), ('high', 'max'), ('vmax', 'velocity'), ('amax', 'acceleration')):
                a[name] = self.tensor([row[key] for row in rows])
            a['q'] = a['q0'].expand(num_envs, -1).clone()
            a['reference'] = a['q'].clone()
            a['v'] = torch.zeros_like(a['q'])
            a['acceleration'] = torch.zeros_like(a['q'])
            self.sides[side] = a
            self.contract['base'][side] = dict(position_m=a['base_pos'].tolist(),
                                              rotation=a['base_rot'].tolist())
        self.excess = torch.zeros(num_envs, dtype=torch.float64, device=self.device)
        self.last = {}

    @staticmethod
    def fk(a, q):
        p = a['base_pos'].expand(*q.shape[:-1], 3)
        r = a['base_rot'].expand(*q.shape[:-1], 3, 3)
        positions, rotations, axes, anchors = [p], [r], [], []
        for i in range(7):
            p = p + (r @ a['joint_pos'][i, :, None]).squeeze(-1)
            r = r @ a['joint_rot'][i]
            axes.append((r @ a['axis'][i, :, None]).squeeze(-1))
            anchors.append(p)
            r = r @ rot(a['axis'][i], q[..., i])
            positions.append(p); rotations.append(r)
        tip = p + (r @ a['tool_pos'][:, None]).squeeze(-1)
        axis, anchor = torch.stack(axes, -1), torch.stack(anchors, -1)
        jacp = torch.cross(axis.transpose(-1, -2),
                           (tip[..., None] - anchor).transpose(-1, -2), dim=-1).transpose(-1, -2)
        return tip, r @ a['tool_rot'], jacp, axis, torch.stack(positions, -2), torch.stack(rotations, -3)

    def _error(self, p, r, target_p, target_r):
        return torch.cat((target_p - p, self.weight * rotation_error(r, target_r)), -1)

    def _coupling_value(self, q):
        """Return the CCS-696 J6/J7 coupling expression in radians."""
        return (self.coupling_coefficient * q[..., self.coupling_j6].abs()
                + q[..., self.coupling_j7].abs())

    def _coupling_valid(self, q):
        # Keep exact boundary points accepted despite float64 roundoff.
        return self._coupling_value(q) <= self.coupling_sum_limit + 1e-12

    def _coupled_segment(self, start, candidate):
        """Take the largest feasible prefix of a per-step arm update.

        The SDK region is convex in (J6,J7). Both endpoints already satisfy
        the independent position, speed and acceleration boxes, so bisection
        on their segment preserves those boxes while enforcing the coupling.
        """
        valid = self._coupling_valid(candidate)
        lo = torch.zeros(start.shape[0], dtype=start.dtype, device=start.device)
        hi = torch.ones_like(lo)
        delta = candidate - start
        for _ in range(48):
            mid = (lo + hi) * .5
            probe = start + mid[:, None] * delta
            ok = self._coupling_valid(probe)
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        projected = start + lo[:, None] * delta
        return torch.where(valid[:, None], candidate, projected)

    def solve(self, a, initial, target_p, target_r, low, high):
        q = initial.clamp(low, high)
        for _ in range(self.options['iterations']):
            p, r, jp, jr, *_ = self.fk(a, q)
            error = self._error(p, r, target_p, target_r)
            jac = torch.cat((jp, self.weight * jr), -2)
            delta = (jac.transpose(-1, -2) @ torch.linalg.solve(
                jac @ jac.transpose(-1, -2) + self.options['damping']**2 * self.eye,
                error[..., None])).squeeze(-1)
            # Bounded line search includes the current feasible iterate, so the
            # objective cannot increase merely because of a saturated joint.
            candidates = (q[:, None] + self.fractions[None, :, None] * delta[:, None]).clamp(low[:, None], high[:, None])
            cp, cr, *_ = self.fk(a, candidates)
            cost = self._error(cp, cr, target_p[:, None], target_r[:, None]).square().sum(-1)
            cost = cost.masked_fill(~self._coupling_valid(candidates), float('inf'))
            index = cost.argmin(-1)
            q = candidates[torch.arange(self.num_envs, device=self.device), index]
        return q

    def _boundary_speed_cap(self, a, gap):
        # Endpoint feasibility reserves room beyond the immediate acceleration
        # box so a position boundary cannot force a later hard-limit violation.
        # equal margins can make their intersection empty by one float64 ulp
        # after repeated saturated updates near a position boundary.
        delta = a['amax'] * self.dt * (1 - 1e-7)
        gap = gap.clamp_min(0)
        n = torch.floor((torch.sqrt(1 + 8 * gap / (delta * self.dt)) - 1) / 2)
        return (gap / self.dt + delta * n * (n + 1) / 2) / (n + 1)

    def bounds(self, a):
        # Reserve tiny arithmetic margins while using the SDK limits for audit.
        dv = a['amax'] * self.dt * (1 - 1e-10)
        lo = torch.maximum(-a['vmax'], a['v'] - dv)
        hi = torch.minimum(a['vmax'], a['v'] + dv)
        lo = torch.maximum(lo, -self._boundary_speed_cap(a, a['q'] - a['low']))
        hi = torch.minimum(hi, self._boundary_speed_cap(a, a['high'] - a['q']))
        return torch.maximum(a['low'], a['q'] + lo * self.dt), torch.minimum(a['high'], a['q'] + hi * self.dt)

    def begin_control(self):
        self.excess.zero_()

    def reset(self, ids):
        for a in self.sides.values():
            a['q'][ids] = a['q0']; a['reference'][ids] = a['q0']
            a['v'][ids] = 0; a['acceleration'][ids] = 0
        self.excess[ids] = 0
        for rows in self.last.values():
            for value in rows.values():
                value[ids] = 0

    def step(self, targets):
        pending, diagnostics = {}, {}
        for side, a in self.sides.items():
            tp, tr = (x.to(device=self.device, dtype=torch.float64) for x in targets[side])
            if tp.shape != (self.num_envs, 3) or tr.shape != (self.num_envs, 3, 3) or not (torch.isfinite(tp).all() and torch.isfinite(tr).all()):
                raise ValueError('Invalid wrist pose request: ' + side)
            low, high = self.bounds(a)
            if not torch.all(low <= high):
                raise RuntimeError('Arm has no feasible hard-limit interval; state not advanced: ' + side)
            raw = self.solve(a, a['reference'], tp, tr, a['low'].expand_as(low), a['high'].expand_as(high))
            requested_v = (raw - a['q']) / self.dt
            requested_a = (requested_v - a['v']) / self.dt
            excess = torch.maximum((requested_v.abs() / (a['vmax'] * self.options['soft_limit_ratio']) - 1).amax(-1),
                                   (requested_a.abs() / (a['amax'] * self.options['soft_limit_ratio']) - 1).amax(-1)).clamp_min(0)
            # Follow the current IK reference as far as the current-step
            # feasible box permits. No remaining-distance target braking is
            # applied here; bounds() still protects the URDF endpoint.
            issued = raw.clamp(low, high)
            issued = self._coupled_segment(a['q'], issued)
            velocity = (issued - a['q']) / self.dt
            acceleration = (velocity - a['v']) / self.dt
            if not (torch.isfinite(issued).all() and torch.isfinite(raw).all()):
                raise FloatingPointError('Arm IK numerical failure; state not advanced: ' + side)
            if (torch.any(velocity.abs() > a['vmax'] + 1e-8)
                    or torch.any(acceleration.abs() > a['amax'] + 1e-7)):
                raise RuntimeError('Arm hard-limit verification failed; state not advanced: ' + side)
            if not torch.all(self._coupling_valid(issued)):
                raise RuntimeError('Arm CCS-696 J6/J7 coupling verification failed; state not advanced: ' + side)
            p, r, *_ = self.fk(a, issued)
            diagnostics[side] = dict(target_position=tp.clone(), target_rotation=tr.clone(),
                requested_q=raw, requested_velocity=requested_v,
                requested_acceleration=requested_a, position_error=(tp-p).norm(dim=-1),
                orientation_error=rotation_error(r, tr).norm(dim=-1),
                position_error_vector=tp-p, rotation_error_vector=rotation_error(r, tr), excess=excess)
            pending[side] = (issued, velocity, acceleration, p, r)
        # Commit both arms together only after both have a verified finite result.
        for side, (q, v, acc, _, _) in pending.items():
            a = self.sides[side]
            a['q'].copy_(q); a['v'].copy_(v); a['acceleration'].copy_(acc)
            a['reference'].copy_(diagnostics[side]['requested_q'])
            self.excess = torch.maximum(self.excess, diagnostics[side]['excess'])
        self.last = diagnostics
        return {s: (values[3], values[4]) for s, values in pending.items()}

    def observation(self):
        values = []
        for side, a in self.sides.items():
            values.extend(a[field] for field in ('q', 'v', 'reference'))
            for field in ('position_error_vector', 'rotation_error_vector'):
                values.append(self.last[side][field] if side in self.last else torch.zeros_like(a['q'][:, :3]))
        return torch.cat(values, -1).float()

    def penalty(self):
        return (-self.options['maximum_penalty_per_step'] *
                (self.excess / self.options['full_penalty_excess']).clamp(0, 1)).float()
