"""Read-only physical-substep samples for optional control-cycle rewards.

Key travel is recorded after every integrated physics step. Actuator power and
contacts are the quantities evaluated by that step's solver. Power is averaged
over the control interval; contacts are a union, never ten duplicated penalties.
No forward/step call, state correction or model change is made by this module.
"""
import numpy as np
import torch
import warp as wp
from ..key_activation import KEY_ACTIVATION_FRACTION, validate_fraction


@wp.kernel
def sample_positions(positions: wp.array2d(dtype=wp.vec3),
                     ids: wp.array(dtype=wp.int32),
                     offsets: wp.array(dtype=wp.vec3), substep: int,
                     history: wp.array3d(dtype=wp.vec3)):
    world, column = wp.tid()
    history[world, substep, column] = positions[world, ids[column]] + offsets[column]


@wp.kernel
def sample_key_travel(qpos: wp.array2d(dtype=wp.float32),
                      key_qadr: wp.array(dtype=wp.int32),
                      key_upper: wp.array(dtype=wp.float32),
                      key_activation_fraction: float,
                      substep: int,
                      history: wp.array3d(dtype=wp.float32),
                      activation_history: wp.array3d(dtype=wp.bool)):
    world, key = wp.tid()
    value = wp.clamp(qpos[world, key_qadr[key]], 0.0, key_upper[key])
    history[world, substep, key] = value / key_upper[key]
    # Inclusive 100% cutoff: avoid a division round trip moving the boundary.
    activation_history[world, substep, key] = value >= key_upper[key] * key_activation_fraction


@wp.kernel
def sample_actuator_power(force: wp.array2d(dtype=wp.float32),
                          velocity: wp.array2d(dtype=wp.float32),
                          action_ids: wp.array(dtype=wp.int32),
                          inverse_substeps: float,
                          absolute_mean: wp.array(dtype=wp.float32),
                          signed_mean: wp.array2d(dtype=wp.float32)):
    world = wp.tid()
    absolute = float(0.0)
    for column in range(action_ids.shape[0]):
        actuator = action_ids[column]
        power = force[world, actuator] * velocity[world, actuator]
        absolute = absolute + wp.abs(power)
        signed_mean[world, column] = signed_mean[world, column] + power * inverse_substeps
    absolute_mean[world] = absolute_mean[world] + absolute * inverse_substeps


@wp.kernel
def sample_palm_contacts(nacon: wp.array(dtype=wp.int32),
                          geom: wp.array(dtype=wp.vec2i),
                          distance: wp.array(dtype=wp.float32),
                          world: wp.array(dtype=wp.int32),
                          pair_mask: wp.array2d(dtype=wp.int32),
                          margin: float,
                          substep: int,
                          collision_any: wp.array(dtype=wp.int32),
                          collision_substeps: wp.array2d(dtype=wp.int32)):
    contact = wp.tid()
    if contact < nacon[0] and distance[contact] <= margin:
        a, b = geom[contact][0], geom[contact][1]
        env = world[contact]
        if (a >= 0 and b >= 0 and a < pair_mask.shape[0]
                and b < pair_mask.shape[1] and env >= 0
                and env < collision_any.shape[0]):
            if pair_mask[a, b] != 0:
                wp.atomic_max(collision_any, env, 1)
                # The endpoint refresh is passed as -1.  It remains useful for
                # the legacy `*_any` diagnostic, but is deliberately excluded
                # from the per-control-step reward history.
                if substep >= 0:
                    wp.atomic_max(collision_substeps, env, substep, 1)


class SubstepRewardCollector:
    """Fixed-size buffers that can be written inside the captured physics graph."""

    def __init__(self, *, num_envs, decimation, device, key_qadr, key_limits,
                 action_ids, geom_pair_mask, collision_margin=0.0, key_activation_fraction=KEY_ACTIVATION_FRACTION,
                 tip_ids=None, key_site_ids=None, key_offsets=None,
                 key_hold_model=None, key_hold_geom_ids=None):
        if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
            raise ValueError('decimation must be a positive integer')
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError('num_envs must be a positive integer')
        self.device = str(device)
        self.decimation = decimation
        limits = np.asarray(key_limits, dtype=np.float32)
        keys = np.asarray(key_qadr, dtype=np.int32)
        actions = np.asarray(action_ids, dtype=np.int32)
        mask = np.asarray(geom_pair_mask, dtype=np.int32)
        if (limits.shape != (len(keys), 2) or not np.isfinite(limits).all()
                or np.any(limits[:, 0] != 0) or np.any(limits[:, 1] <= 0)):
            raise ValueError('Substep key travel requires finite [0, positive] key limits')
        if keys.ndim != 1 or actions.ndim != 1 or np.any(keys < 0) or np.any(actions < 0):
            raise ValueError('Expected one-dimensional nonnegative key/actuator addresses')
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1] or not np.array_equal(mask, mask.T):
            raise ValueError('Expected a symmetric square geom-pair mask')
        if not np.isfinite(collision_margin):
            raise ValueError('collision_margin must be finite')
        validate_fraction(key_activation_fraction)
        self.collision_margin = float(collision_margin)
        self.key_activation_fraction = float(key_activation_fraction)
        self.key_qadr = wp.array(keys, dtype=wp.int32, device=self.device)
        self.key_upper = wp.array(limits[:, 1], dtype=wp.float32, device=self.device)
        self.action_ids = wp.array(actions, dtype=wp.int32, device=self.device)
        self.geom_pair_mask = wp.array(mask, dtype=wp.int32, device=self.device)
        self.key_state_history = torch.zeros((num_envs, decimation, len(keys)), device=device)
        self.key_activation_history = torch.zeros_like(self.key_state_history, dtype=torch.bool)
        self.power_mean = torch.zeros(num_envs, device=device)
        self.signed_power_mean = torch.zeros((num_envs, len(actions)), device=device)
        self._palm_collision_any = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self._palm_collision_substeps = torch.zeros(
            (num_envs, decimation), dtype=torch.int32, device=device)
        self._key_history_wp = wp.from_torch(self.key_state_history)
        self._key_activation_history_wp = wp.from_torch(self.key_activation_history)
        self._power_wp = wp.from_torch(self.power_mean)
        self._signed_power_wp = wp.from_torch(self.signed_power_mean)
        self._palm_wp = wp.from_torch(self._palm_collision_any)
        self._palm_substeps_wp = wp.from_torch(self._palm_collision_substeps)
        self.key_hold_lateral = None
        if key_hold_model is not None:
            from .key_hold_lateral import KeyHoldLateralCollector
            if np.shape(key_hold_geom_ids) != keys.shape:
                raise ValueError('Key-hold geometry must match activation history order/size')
            self.key_hold_lateral = KeyHoldLateralCollector(key_hold_model, key_hold_geom_ids,
                num_envs=num_envs, decimation=decimation, device=device)
        elif key_hold_geom_ids is not None:
            raise ValueError('Key-hold geometry requires a compiled model')
        self.fingertip_history = self.key_target_history = None
        if tip_ids is not None:
            if np.shape(tip_ids) != (10,) or np.shape(key_site_ids) != (88,) or np.shape(key_offsets) != (88, 3):
                raise ValueError('Fingering samples require ten sites and 88 key reference sites/offsets')
            self.tip_ids = wp.array(tip_ids, dtype=wp.int32, device=self.device)
            self.key_site_ids = wp.array(key_site_ids, dtype=wp.int32, device=self.device)
            self.tip_offsets = wp.zeros(10, dtype=wp.vec3, device=self.device)
            self.key_offsets = wp.array(key_offsets, dtype=wp.vec3, device=self.device)
            self.fingertip_history = torch.zeros((num_envs, decimation, 10, 3), device=device)
            self.key_target_history = torch.zeros((num_envs, decimation, 88, 3), device=device)
            self._tips_wp = wp.from_torch(self.fingertip_history, dtype=wp.vec3)
            self._targets_wp = wp.from_torch(self.key_target_history, dtype=wp.vec3)
        elif key_site_ids is not None or key_offsets is not None:
            raise ValueError('Incomplete fingering geometry configuration')

    @property
    def palm_collision_any(self):
        return self._palm_collision_any.bool()

    @property
    def palm_collision_substeps(self):
        """Number of sampled physical substeps with any cross-hand contact."""
        return self._palm_collision_substeps.sum(-1)

    def clear(self, env_ids=None):
        if self.key_hold_lateral is not None:
            self.key_hold_lateral.clear(env_ids)
        for buffer in (self.fingertip_history, self.key_target_history,
                       self.key_state_history, self.key_activation_history, self.power_mean,
                       self.signed_power_mean, self._palm_collision_any,
                       self._palm_collision_substeps):
            if buffer is None:
                continue
            if env_ids is None:
                buffer.zero_()
            else:
                buffer[env_ids] = 0

    def begin_control(self, sim):
        if self.key_hold_lateral is not None:
            self.key_hold_lateral.begin_control(sim.wp_data)

    def sample(self, sim, substep, *, contact_data=None):
        if not 0 <= substep < self.decimation:
            raise ValueError('Substep index outside the control interval')
        data = sim.wp_data
        if self.fingertip_history is not None:
            for positions, ids, offsets, history, count in (
                    (data.site_xpos, self.tip_ids, self.tip_offsets, self._tips_wp, 10),
                    (data.site_xpos, self.key_site_ids, self.key_offsets, self._targets_wp, 88)):
                wp.launch(sample_positions, dim=(self.key_state_history.shape[0], count),
                          inputs=[positions, ids, offsets, substep, history], device=self.device)
        wp.launch(sample_key_travel, dim=(self.key_state_history.shape[0], self.key_state_history.shape[2]),
                  inputs=[data.qpos, self.key_qadr, self.key_upper, self.key_activation_fraction,
                          substep, self._key_history_wp, self._key_activation_history_wp],
                  device=self.device)
        wp.launch(sample_actuator_power, dim=self.key_state_history.shape[0],
                  inputs=[data.actuator_force, data.actuator_velocity, self.action_ids,
                          1.0 / self.decimation, self._power_wp, self._signed_power_wp],
                  device=self.device)
        if self.key_hold_lateral is not None:
            # Always use real main-world contacts, never detection-only queries.
            self.key_hold_lateral.sample(data, self.key_activation_history, substep)
        self.sample_contacts(sim, substep=substep, contact_data=contact_data)

    def sample_contacts(self, sim, *, substep=-1, contact_data=None):
        """Also union the final forward contacts, without an extra energy sample."""
        contacts = sim.wp_data if contact_data is None else contact_data
        wp.launch(sample_palm_contacts, dim=contacts.naconmax,
                  inputs=[contacts.nacon, contacts.contact.geom, contacts.contact.dist,
                          contacts.contact.worldid, self.geom_pair_mask,
                          self.collision_margin, int(substep), self._palm_wp,
                          self._palm_substeps_wp], device=self.device)
