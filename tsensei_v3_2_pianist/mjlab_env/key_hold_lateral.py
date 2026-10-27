"""Read-only, per-hand active-key contact and actual palm-origin lateral velocity.

Current scene contract: fixed piano, world Y runs from low to high notes.
Use the ancestor Jacobian at the palm origin, not target differences, body COM
velocity, or the last joint in an implementation-dependent joint list.
"""
import mujoco
import numpy as np
import torch
import warp as wp


@wp.kernel
def seed_palm_positions(xpos: wp.array2d(dtype=wp.vec3),
                        bodies: wp.array(dtype=wp.int32), previous: wp.array2d(dtype=wp.vec3)):
    world, side = wp.tid()
    previous[world,side] = xpos[world,bodies[side]]


@wp.kernel
def sample_palm_lateral(qvel: wp.array2d(dtype=wp.float32),
                        xpos: wp.array2d(dtype=wp.vec3),
                        xaxis: wp.array2d(dtype=wp.vec3),
                        xanchor: wp.array2d(dtype=wp.vec3),
                        bodies: wp.array(dtype=wp.int32),
                        joints: wp.array2d(dtype=wp.int32),
                        dofs: wp.array2d(dtype=wp.int32),
                        hinges: wp.array2d(dtype=wp.int32),
                        mocap: wp.array(dtype=wp.int32),
                        previous: wp.array2d(dtype=wp.vec3),
                        inverse_dt: float, substep: int,
                        speed: wp.array3d(dtype=wp.float32),
                        held: wp.array3d(dtype=wp.int32)):
    world, side = wp.tid()
    position = xpos[world, bodies[side]]
    velocity = wp.vec3(0.0, 0.0, 0.0)
    for column in range(joints.shape[1]):
        joint = joints[side, column]
        if joint >= 0:
            direction = xaxis[world, joint]
            if hinges[side, column] != 0:
                direction = wp.cross(direction, position - xanchor[world, joint])
            velocity = velocity + direction * qvel[world, dofs[side, column]]
    if mocap[side] != 0:
        # Rigid arms publish FK through mocap, which has no dynamic qvel.
        # Use the actual applied palm-origin displacement over this substep.
        velocity = (position - previous[world,side]) * inverse_dt
    previous[world,side] = position
    speed[world, substep, side] = velocity[1]
    held[world, substep, side] = 0


@wp.kernel
def mark_active_key_contact(nacon: wp.array(dtype=wp.int32),
                           geom: wp.array(dtype=wp.vec2i),
                           distance: wp.array(dtype=wp.float32),
                           world: wp.array(dtype=wp.int32),
                           key_index: wp.array(dtype=wp.int32),
                           hand_side: wp.array(dtype=wp.int32),
                           active: wp.array3d(dtype=wp.bool),
                           substep: int,
                           held: wp.array3d(dtype=wp.int32)):
    i = wp.tid()
    if i < nacon[0] and distance[i] <= 0.0:
        a, b = geom[i][0], geom[i][1]
        env = world[i]
        if (a >= 0 and b >= 0 and a < key_index.shape[0] and b < key_index.shape[0]
                and env >= 0 and env < held.shape[0]):
            key, side = int(-1), int(-1)
            if key_index[a] >= 0 and hand_side[b] >= 0:
                key = key_index[a]
                side = hand_side[b]
            elif key_index[b] >= 0 and hand_side[a] >= 0:
                key = key_index[b]
                side = hand_side[a]
            if key >= 0 and active[env, substep, key]:
                wp.atomic_max(held, env, substep, side, 1)


class KeyHoldLateralCollector:
    def __init__(self, model, key_geom_ids, *, num_envs, decimation, device):
        self.device = str(device)
        self.decimation = decimation
        keys = np.asarray(key_geom_ids, dtype=np.int32)
        if (keys.ndim != 1 or not len(keys) or len(set(keys.tolist())) != len(keys)
                or np.any(keys < 0) or np.any(keys >= model.ngeom)):
            raise ValueError('Key-hold collection needs unique existing key geom IDs in activation order')
        if num_envs < 1 or decimation < 1:
            raise ValueError('Key-hold collection requires positive worlds/substeps')
        key_index = np.full(model.ngeom, -1, dtype=np.int32)
        key_index[keys] = np.arange(len(keys), dtype=np.int32)
        sides = np.full(model.ngeom, -1, dtype=np.int32)
        bodies, ancestors, mocap_sides = [], [], []
        # A3.2 explicitly marks every physical URDF collider; include the entire
        # hand/mount, exclude visual meshes and unrelated arm collision queries.
        for side, prefix in enumerate(('right_/', 'left_/')):
            body = model.body(prefix + 'base_link').id
            bodies.append(body)
            for g in range(model.ngeom):
                name = model.geom(g).name or ''
                if name.startswith(prefix) and '_collision_' in name:
                    sides[g] = side
            if not np.any(sides == side):
                raise ValueError(f'Missing physical collision geometry for {prefix}')
            chain = []
            has_mocap = False
            while body > 0:
                has_mocap = has_mocap or int(model.body_mocapid[body]) >= 0
                for joint in range(int(model.body_jntadr[body]),
                                   int(model.body_jntadr[body] + model.body_jntnum[body])):
                    kind = model.jnt_type[joint]
                    if kind not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
                        raise ValueError('Palm velocity requires slide/hinge ancestors; free/ball roots unsupported')
                    chain.append((joint, int(model.jnt_dofadr[joint]), int(kind == mujoco.mjtJoint.mjJNT_HINGE)))
                body = int(model.body_parentid[body])
            ancestors.append(chain)
            mocap_sides.append(int(has_mocap))
        if np.any(sides[keys] >= 0):
            raise ValueError('Piano key and hand geometry cannot overlap')
        count = max(1, *(len(chain) for chain in ancestors))
        joints = np.full((2,count), -1, dtype=np.int32)
        dofs = np.zeros_like(joints); hinges = np.zeros_like(joints)
        for side, chain in enumerate(ancestors):
            for column, (joint,dof,hinge) in enumerate(chain):
                joints[side,column], dofs[side,column], hinges[side,column] = joint,dof,hinge
        self.bodies = wp.array(bodies, dtype=wp.int32, device=self.device)
        self.has_mocap = any(mocap_sides)
        self.mocap = wp.array(mocap_sides, dtype=wp.int32, device=self.device)
        self.inverse_dt = 1. / float(model.opt.timestep)
        self.previous_positions = wp.zeros((num_envs,2), dtype=wp.vec3, device=self.device)
        self.joints = wp.array(joints, dtype=wp.int32, device=self.device)
        self.dofs = wp.array(dofs, dtype=wp.int32, device=self.device)
        self.hinges = wp.array(hinges, dtype=wp.int32, device=self.device)
        self.key_index = wp.array(key_index, dtype=wp.int32, device=self.device)
        self.hand_side = wp.array(sides, dtype=wp.int32, device=self.device)
        self.velocity_history = torch.zeros((num_envs, decimation, 2), device=device)
        self.contact_history = torch.zeros((num_envs, decimation, 2), dtype=torch.int32, device=device)
        self._speed = wp.from_torch(self.velocity_history)
        self._held = wp.from_torch(self.contact_history)
        self.contract = dict(axis='piano_world_y', velocity='actual_palm_origin',
            velocity_sources=['physical_substep_position_difference' if m else 'jacobian_qvel' for m in mocap_sides],
            palm_bodies=['right_/base_link','left_/base_link'], contact_distance_max_m=0.,
            activation='same_substep_actual_key_activation', side_order=['right','left'],
            contact_source='main_physics', aggregation='max_sides_then_mean_substeps')

    def clear(self, env_ids=None):
        for buffer in (self.velocity_history, self.contact_history):
            if env_ids is None:
                buffer.zero_()
            else:
                buffer[env_ids] = 0

    def begin_control(self, data):
        # Seed once before the first mocap update in EVERY control step, including
        # the first step after partial reset. No cross-episode position difference.
        if self.has_mocap:
            wp.launch(seed_palm_positions, dim=(self.velocity_history.shape[0],2),
                inputs=[data.xpos,self.bodies,self.previous_positions],device=self.device)

    def sample(self, data, activation_history, substep):
        if not 0 <= substep < self.decimation:
            raise ValueError('Key-hold substep outside control interval')
        wp.launch(sample_palm_lateral, dim=(self.velocity_history.shape[0],2),
            inputs=[data.qvel,data.xpos,data.xaxis,data.xanchor,self.bodies,
                    self.joints,self.dofs,self.hinges,self.mocap,self.previous_positions,
                    self.inverse_dt,substep,self._speed,self._held], device=self.device)
        wp.launch(mark_active_key_contact, dim=data.naconmax,
            inputs=[data.nacon,data.contact.geom,data.contact.dist,data.contact.worldid,
                    self.key_index,self.hand_side,wp.from_torch(activation_history),substep,self._held],
            device=self.device)
