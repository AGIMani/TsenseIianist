"""Whole-finger geometric target guard, independent of score-specific wrist limits.

This constrains commanded kinematic endpoints, never physical qpos/qvel. Servo
lag, momentum and contact can still violate the envelope; actual-state deficits
are reported separately and must be audited at the physical sampling rate.
"""
from copy import deepcopy
import math

import mujoco
import numpy as np
import torch


MODE = 'full_finger_clearance_v1'
CONTROL_SWITCHES = ('idle_finger_protection', 'active_press_permission')
DEFAULTS = dict(clearance_m=.002, contact_slack_m=.001,
                projection_iterations=6, max_joint_correction_rad=.25,
                active_command_mode='strict_v1', active_contact_slack_m=.005,
                active_support_xy_margin_m=.002, active_recovery_delta_rad=.2,
                wrist_clearance_interlock=False)


def canonical(options):
    if options is None:
        return None
    if not isinstance(options, dict) or set(options)-{'schema', 'mode', *DEFAULTS, *CONTROL_SWITCHES}:
        raise ValueError('Unknown finger clearance options')
    if type(options.get('schema')) is not int or options['schema'] != 1 or options.get('mode') != MODE:
        raise ValueError('Expected schema 1 full_finger_clearance_v1')
    result = dict(schema=1, mode=MODE, **DEFAULTS)
    result.update({k: v for k, v in options.items() if k in DEFAULTS})
    # Preserve the exact schema-1 contract for archived runs. Historical configs
    # without switches enabled both controls; new launchers write explicit False.
    for name in CONTROL_SWITCHES:
        if name in options:
            if type(options[name]) is not bool:
                raise ValueError(name+' must be a bool')
            result[name] = options[name]
    for name in ('clearance_m', 'contact_slack_m', 'max_joint_correction_rad',
                 'active_contact_slack_m', 'active_support_xy_margin_m', 'active_recovery_delta_rad'):
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name+' must be finite and positive')
        result[name] = float(value)
    count = result['projection_iterations']
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('projection_iterations must be an integer in [1,32]')
    if result['active_command_mode'] not in ('strict_v1', 'idle_only_probe', 'active_contact_v1'):
        raise ValueError('Unknown active_command_mode')
    if type(result['wrist_clearance_interlock']) is not bool:
        raise ValueError('wrist_clearance_interlock must be a bool')
    return result


def _matrix(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quaternion)
    return result.reshape(3, 3)


def _rotation(axis, angle):
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + angle.sin()[..., None, None]*skew + (1-angle.cos())[..., None, None]*(skew@skew)


class FingerClearanceGuard:
    """Stateless batched filter; all runtime work stays on the tensor device.

    ``assigned_keys`` is the current score's (world, 10, 88) finger/key matrix,
    right thumb first. No future fingering or remembered role enters the filter.
    The terminal capsule is the model's contact pad; its entire XY extent must
    fit an assigned key before it may descend to that key's full-travel plane.
    """
    def __init__(self, bundle, *, device='cuda:0', options):
        self.options = canonical(options)
        if self.options is None:
            raise ValueError('Construct the guard only for explicitly enabled options')
        self.device = torch.device(device)
        self.bundle = bundle
        model = bundle.model
        self.rigid_arm = hasattr(bundle, 'arm_follow_spec')
        expected = (40, 128, 40) if self.rigid_arm else (46, 134, 46)
        if (len(bundle.active_action_indices), model.nq, model.nu) != expected:
            raise ValueError('Whole-finger guard currently requires the no-arm 47-action TZ/Yaw-removed model')
        if self.rigid_arm and (any(self.options.get(n, True) for n in CONTROL_SWITCHES)
                               or self.options['wrist_clearance_interlock']):
            raise ValueError('Rigid arm assembly supports clearance diagnostics with projection controls disabled')
        self.tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.integer = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.int64, device=self.device)
        reference = mujoco.MjData(model)
        reference.qpos[:] = model.qpos0
        reference.mocap_pos[:] = bundle.origin_mocap_pos
        reference.mocap_quat[:] = bundle.origin_mocap_quat
        mujoco.mj_forward(model, reference)
        physical_names = bundle.manifest['action_names']
        joint_to_full = {int(model.actuator_trnid[act, 0]): int(full)
                         for act, full in zip(bundle.action_ids, bundle.active_action_indices)}
        full_qadr = [0]*51
        for joint, full in joint_to_full.items():
            full_qadr[full] = int(model.jnt_qposadr[joint])
        self.full_qadr = self.integer(full_qadr)
        self.present = self.tensor([i in bundle.active_action_indices for i in range(51)])
        self.fixed = self.tensor(bundle.neutral_action)
        groups = []
        for side in ('right', 'left'):
            for finger in ('thumb', 'index', 'mid', 'ring', 'pinky'):
                joint_ids = [j for j in range(model.njnt) if (name := model.joint(j).name)
                             and name.startswith(side+'_/') and '_'+finger+'_' in name]
                geom_ids = [g for g in range(model.ngeom) if (name := model.geom(g).name)
                            and name.startswith(side+'_/') and '_'+finger+'_' in name and '_collision_' in name]
                if len(joint_ids) != 4 or len(geom_ids) != 4:
                    raise ValueError('Expected all four collision links and four joints: '+side+'/'+finger)
                if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE for g in geom_ids):
                    raise ValueError('Whole-finger envelope requires explicit capsule collision geometry')
                # Each collision capsule belongs to one of the four serial links.
                geom_ids = [next(g for g in geom_ids if model.geom_bodyid[g] == model.jnt_bodyid[j])
                            for j in joint_ids]
                body = int(model.jnt_bodyid[joint_ids[-1]])
                ancestors = []
                while body:
                    ancestors.append(list(range(int(model.body_jntadr[body]),
                                                int(model.body_jntadr[body]+model.body_jntnum[body]))))
                    body = int(model.body_parentid[body])
                # Reverse body ancestry while preserving the declared order of
                # multiple joints on one body (the virtual wrist has three).
                chain = [joint for body_joints in reversed(ancestors) for joint in body_joints]
                if len(chain) != (4 if self.rigid_arm else 7) or chain[-4:] != joint_ids:
                    raise ValueError('Unexpected serial wrist/finger joint chain')
                groups.append(dict(side=side, name=side+'_'+finger, joints=joint_ids, geoms=geom_ids, chain=chain))
        self.groups = groups
        local_positions = [];local_rotations = [];joint_positions = [];joint_axes = []
        chain_indices = [];qzero = [];geom_positions = [];geom_axes = [];radii = [];half_lengths = []
        for group in groups:
            previous_p = np.zeros(3);previous_r = np.eye(3)
            if self.rigid_arm:
                root = model.body(bundle.manifest['hands'][group['side']]['body']).id
                previous_p = reference.xpos[root].copy()
                previous_r = reference.xmat[root].reshape(3, 3).copy()
            lp = [];lr = []
            for joint in group['chain']:
                body = int(model.jnt_bodyid[joint]);p = reference.xpos[body];r = reference.xmat[body].reshape(3, 3)
                lp.append(previous_r.T@(p-previous_p));lr.append(previous_r.T@r)
                previous_p = p.copy();previous_r = r.copy()
            local_positions.append(lp);local_rotations.append(lr)
            joint_positions.append(model.jnt_pos[group['chain']]);joint_axes.append(model.jnt_axis[group['chain']])
            chain_indices.append([joint_to_full[j] for j in group['chain']])
            qzero.append(model.qpos0[model.jnt_qposadr[group['chain']]])
            geom_positions.append(model.geom_pos[group['geoms']])
            geom_axes.append([_matrix(model.geom_quat[g])[:, 2] for g in group['geoms']])
            radii.append(model.geom_size[group['geoms'], 0]);half_lengths.append(model.geom_size[group['geoms'], 1])
        kinds = np.asarray([[model.jnt_type[j] for j in group['chain']] for group in groups])
        if not np.all(kinds == kinds[0]) or not np.all(kinds[:, -4:] == 3) or not np.isin(kinds, [2, 3]).all():
            raise ValueError('Unsupported joint types/order in capsule FK chain')
        self.joint_kinds = kinds[0].tolist()
        self.local_positions = self.tensor(local_positions);self.local_rotations = self.tensor(local_rotations)
        self.joint_positions = self.tensor(joint_positions);self.joint_axes = self.tensor(joint_axes)
        self.chain_indices = self.integer(chain_indices);self.qzero = self.tensor(qzero)
        self.finger_indices = self.chain_indices[:, -4:]
        self.geom_positions = self.tensor(geom_positions);self.geom_axes = self.tensor(geom_axes)
        self.radii = self.tensor(radii);self.half_lengths = self.tensor(half_lengths)
        self.ancestor = self.tensor(np.tri(4))[None, None, :, :, None]
        self.side_indices = self.integer([0]*5+[1]*5)
        from .preparation import MOUNTS
        if self.rigid_arm:
            self.root_ids = self.integer([model.body(bundle.manifest['hands'][s]['body']).id
                                          for s in ('right', 'left')])
        else:
            self.mount_ids = self.integer([int(model.body(name).mocapid[0]) for name in MOUNTS.values()])
            if self.mount_ids.numel() != 2:
                raise ValueError('Two preparation mounts required')
            self.origin_mount_z = self.tensor(bundle.origin_mocap_pos)[self.mount_ids, 2]
        # Score ordering is MIDI 21..108, matching the existing task contract.
        keys = []
        for key in range(88):
            candidates = [model.geom(name).id for name in (f'piano/white_key_geom_{key}', f'piano/black_key_geom_{key}')
                          if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0]
            if len(candidates) != 1:
                raise ValueError('Cannot resolve the existing piano key geometry ordering')
            keys.append(candidates[0])
        origin = bundle.reset_cpu();origin.qpos[:] = bundle.origin_qpos
        origin.mocap_pos[:] = bundle.origin_mocap_pos;origin.mocap_quat[:] = bundle.origin_mocap_quat
        mujoco.mj_forward(model, origin)
        min_xy = [];max_xy = [];pressed_center = [];pressed_normal = [];rest_top = [];rest_bottom = []
        for key, geom in enumerate(keys):
            from ..fingering_targets import key_reference
            site, half_size = key_reference(model, geom)
            rotation = origin.site_xmat[site].reshape(3, 3)
            if not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-7):
                raise ValueError('Rest piano keys must share the world-Z normal')
            extent = np.abs(rotation)@half_size
            lower = origin.site_xpos[site, :2]-extent[:2]
            upper = origin.site_xpos[site, :2]+extent[:2]
            if 'black_key_geom_' in model.geom(geom).name:
                # Conservative outer footprint of the sloped sides and front.
                from ..models.piano import piano_constants as pc
                lower[1] -= (pc.BLACK_KEY_WIDTH-pc.BLACK_KEY_TOP_WIDTH)/2
                upper[1] += (pc.BLACK_KEY_WIDTH-pc.BLACK_KEY_TOP_WIDTH)/2
                upper[0] += pc.BLACK_KEY_BOTTOM_LENGTH-pc.BLACK_KEY_LENGTH
            min_xy.append(lower);max_xy.append(upper)
            rest_top.append(float(origin.site_xpos[site, 2]+extent[2]))
            rest_bottom.append(float(origin.site_xpos[site, 2]-extent[2]))
            native = bundle.reset_cpu();native.qpos[:] = bundle.origin_qpos
            jid = int(bundle.key_ids[key]);native.qpos[model.jnt_qposadr[jid]] = model.jnt_range[jid, 1]
            mujoco.mj_forward(model, native)
            pressed_center.append(native.site_xpos[site].copy());pressed_normal.append(native.site_xmat[site].reshape(3, 3)[:, 2].copy())
        self.key_min_xy = self.tensor(min_xy);self.key_max_xy = self.tensor(max_xy)
        self.key_top = self.tensor(rest_top);self.key_half_z = self.tensor([key_reference(model, g)[1][2] for g in keys])
        self.key_bottom = self.tensor(rest_bottom);self.lowest_bottom = float(np.min(rest_bottom))
        self.pressed_center = self.tensor(pressed_center);self.pressed_normal = self.tensor(pressed_normal)
        black = np.asarray(['black' in model.geom(g).name for g in keys])
        self.black_ids = self.integer(np.flatnonzero(black))
        self.white_top = float(np.min(rest_top));self.highest_top = float(np.max(rest_top))
        if not np.allclose(np.asarray(rest_top)[~black], self.white_top):
            raise ValueError('Expected one white-key rest surface')
        low = [];high = [];origin_angles = []
        for group in groups:
            bounds = model.jnt_range[group['joints']].copy()
            for i, joint in enumerate(group['joints']):
                full = joint_to_full[joint]
                bounds[i] = np.maximum(bounds[i, 0], bundle.low[full]), np.minimum(bounds[i, 1], bundle.high[full])
                if 'thumb' not in group['name'] and '_MCP_' in model.joint(joint).name:
                    bounds[i, 1] = min(bounds[i, 1], 1.2)
            low.append(bounds[:, 0]);high.append(bounds[:, 1])
            origin_angles.append(bundle.origin_qpos[model.jnt_qposadr[group['joints']]])
        self.low = self.tensor(low);self.high = self.tensor(high);self.origin_angles = self.tensor(origin_angles)
        safe_angles = self._safe_angles(model, origin, np.asarray(origin_angles), np.asarray(low), np.asarray(high))
        self.safe_angles = self.tensor(safe_angles)
        # Cartesian arm policy exposes TX/TY/TZ; Pitch/Yaw are fixed during
        # bundle compilation and are absent from the 47-action policy.
        self.wrist_indices = self.integer([[20, 21, 22], [45, 46, 47]]) if self.rigid_arm else self.chain_indices[[0, 5], :3]
        self.wrist_low = self.tensor(bundle.low)[self.wrist_indices]
        self.wrist_high = self.tensor(bundle.high)[self.wrist_indices]
        self.contract = dict(schema=1, mode=MODE, options=deepcopy(self.options),
            finger_names=[g['name'] for g in groups],
            capsule_names=[[model.geom(i).name for i in g['geoms']] for g in groups],
            finger_joint_names=[[model.joint(i).name for i in g['joints']] for g in groups],
            origin_angles=np.asarray(origin_angles).tolist(), fallback_angles=safe_angles.tolist(),
            white_rest_top_z=self.white_top, highest_rest_top_z=self.highest_top,
            key_names=[model.geom(i).name for i in keys],
            surface='white rest plane plus local overlapping black-key rest surfaces',
            active_pad='terminal capsule fully inside assigned key XY bounds may reach its full-travel top plane',
            context='current finger-key assignment only; no future fingering or hidden phase memory',
            active_contact_rule='actual terminal overlaps assigned XY support, no deeper than full-travel top minus active_contact_slack_m; whole finger stays above overlapping rest key bottoms plus clearance_m',
            active_contact_evidence=dict(successful_assigned_frames=284, median_actual_below_fullpress_m=.0017313631,
                max_actual_below_fullpress_m=.0047702100,
                source='clearance_contacts_probe_retry_20260927/offline_fk/fixed_baseline_fk.json',
                meaning='5 mm candidate allowance accommodates observed soft top contact; not a hardware safety calibration'),
            wrist_interlock='optional current actual wrist target hold while an idle finger is below its local key surface; full workspace retained, no physical qpos/velocity reset',
            projection_support='origin support height, conservative for all nonnegative preparation lifts',
            application='physical target FK correction; no physical qpos/qvel changes; no wrist workspace restriction')
        if self.rigid_arm:
            self.contract.update(root_frame='actual_rigid_hand_xpos_xmat',
                projection_support='actual rigid hand pose at command issue; projection controls disabled',
                application='diagnostics at actual rigid root; finger targets and physical states unchanged')

    def _safe_angles(self, model, data, start, low, high):
        from scipy.optimize import minimize
        result = start.copy();threshold = self.highest_top+self.options['clearance_m']+.0001
        base = data.qpos.copy()
        for finger, group in enumerate(self.groups):
            qadr = model.jnt_qposadr[group['joints']]
            def constraint(q):
                data.qpos[:] = base;data.qpos[qadr] = q;mujoco.mj_kinematics(model, data)
                z = []
                for geom in group['geoms']:
                    radius, length = model.geom_size[geom, :2]
                    z.append(data.geom_xpos[geom, 2]-abs(data.geom_xmat[geom].reshape(3, 3)[2, 2])*length-radius-threshold)
                return np.asarray(z)
            if constraint(start[finger]).min() < 0:
                solved = minimize(lambda q: float(np.square(q-start[finger]).sum()), start[finger],
                    method='SLSQP', bounds=list(zip(low[finger], high[finger])),
                    constraints=[dict(type='ineq', fun=constraint)], options=dict(maxiter=150, ftol=1e-11))
                if constraint(solved.x).min() < -1e-7:
                    raise ValueError('No verified raised fallback posture for '+group['name'])
                result[finger] = solved.x
        data.qpos[:] = base;mujoco.mj_forward(model, data)
        return result

    def physical_qpos(self, qpos):
        return qpos[:, self.full_qadr]*self.present + self.fixed*(1-self.present)

    def lift(self, mocap_pos):
        if self.rigid_arm:
            raise ValueError('Rigid hand diagnostics require full root poses, not a vertical lift')
        return mocap_pos[:, self.mount_ids, 2]-self.origin_mount_z

    def support(self, mocap_pos, body_xpos, body_xmat):
        """Vertical support for old hand physics; full actual root for rigid arms."""
        if not self.rigid_arm:
            return self.lift(mocap_pos)
        return torch.cat((body_xpos[:, self.root_ids],
                          body_xmat[:, self.root_ids].reshape(-1, 2, 9)), -1)

    def projection_support(self, support):
        return support if self.rigid_arm else torch.zeros_like(support)

    def fk(self, physical, lift):
        """Exact capsule endpoints from compiled serial chains; no collision step."""
        count = physical.shape[0]
        q = physical[:, self.chain_indices]-self.qzero
        if self.rigid_arm:
            p = lift[:, self.side_indices, :3]
            r = lift[:, self.side_indices, 3:].reshape(count, 10, 3, 3)
        else:
            p = torch.zeros((count, 10, 3), device=physical.device, dtype=physical.dtype)
            p[..., 2] = lift[:, self.side_indices]
            r = torch.eye(3, device=physical.device, dtype=physical.dtype).expand(count, 10, 3, 3)
        positions = [];rotations = [];axes = [];anchors = []
        for depth in range(len(self.joint_kinds)):
            p = p+(r@self.local_positions[:, depth, :, None]).squeeze(-1)
            r = r@self.local_rotations[:, depth]
            axis = (r@self.joint_axes[:, depth, :, None]).squeeze(-1)
            anchor = p+(r@self.joint_positions[:, depth, :, None]).squeeze(-1)
            if self.joint_kinds[depth] == 2:
                p = p+axis*q[..., depth, None]
            else:
                r = r@_rotation(self.joint_axes[:, depth], q[..., depth])
                p = anchor-(r@self.joint_positions[:, depth, :, None]).squeeze(-1)
            if depth >= len(self.joint_kinds)-4:
                positions.append(p);rotations.append(r);axes.append(axis);anchors.append(anchor)
        body_p = torch.stack(positions, 2);body_r = torch.stack(rotations, 2)
        centers = body_p+(body_r@self.geom_positions[..., None]).squeeze(-1)
        axis = (body_r@self.geom_axes[..., None]).squeeze(-1)
        span = axis*self.half_lengths[..., None]
        endpoints = torch.stack((centers-span, centers+span), 3)
        return endpoints, torch.stack(axes, 2), torch.stack(anchors, 2)

    def _evaluate(self, physical, lift, assigned_keys, *, jacobian=False, kinematics=None):
        endpoints, joint_axes, anchors = self.fk(physical, lift) if kinematics is None else kinematics
        radius = self.radii[None, :, :, None]
        minimum = endpoints.amin(3)-radius;maximum = endpoints.amax(3)+radius
        black_low = self.key_min_xy[self.black_ids];black_high = self.key_max_xy[self.black_ids]
        overlap = ((maximum[..., None, :2] >= black_low) & (minimum[..., None, :2] <= black_high)).all(-1)
        top = torch.where(overlap.any(-1), self.highest_top, self.white_top)
        required = top+self.options['clearance_m']
        normals = torch.zeros_like(minimum);normals[..., 2] = 1
        active = assigned_keys.bool().any(-1)
        terminal_low = minimum[:, :, -1, None, :2];terminal_high = maximum[:, :, -1, None, :2]
        fits = ((terminal_low >= self.key_min_xy) & (terminal_high <= self.key_max_xy)).all(-1)
        valid = fits & assigned_keys.bool()
        allowed = valid.any(-1)
        key = valid.to(torch.int64).argmax(-1)
        pad_normal = self.pressed_normal[key]
        pad_required = (pad_normal*self.pressed_center[key]).sum(-1)+self.key_half_z[key]-self.options['contact_slack_m']
        normals[:, :, -1] = torch.where(allowed[..., None], pad_normal, normals[:, :, -1])
        required[:, :, -1] = torch.where(allowed, pad_required, required[:, :, -1])
        signed = (endpoints*normals[..., None, :]).sum(-1)-self.radii[None, :, :, None]-required[..., None]
        clearance = signed.amin(-1)
        deficit = (-clearance).clamp_min(0)
        result = dict(finger_clearance_deficit_m=deficit.amax(-1),
                      finger_min_clearance_m=clearance.amin(-1),
                      capsule_clearance_m=clearance, finger_active=active,
                      finger_terminal_allowed=allowed)
        if jacobian:
            worst_end = signed.argmin(-1)
            points = endpoints.gather(3, worst_end[..., None, None].expand(*worst_end.shape, 1, 3)).squeeze(3)
            offset = points[:, :, :, None, :]-anchors[:, :, None, :, :]
            derivative = torch.cross(joint_axes[:, :, None, :, :].expand_as(offset), offset, dim=-1)*self.ancestor
            gradient = (derivative*normals[:, :, :, None, :]).sum(-1)
            worst_capsule = clearance.argmin(-1)
            result['gradient'] = gradient.gather(2, worst_capsule[..., None, None].expand(*worst_capsule.shape, 1, 4)).squeeze(2)
        return result

    def _active_contact(self, physical, lift, assigned_keys, strict, *, endpoints=None):
        if endpoints is None:
            endpoints, _, _ = self.fk(physical, lift)
        minimum = endpoints.amin(3)-self.radii[None, :, :, None]
        maximum = endpoints.amax(3)+self.radii[None, :, :, None]
        active = assigned_keys.bool().any(-1)
        margin = self.options['active_support_xy_margin_m']
        support = ((maximum[:, :, -1, None, :2] >= self.key_min_xy-margin)
                   & (minimum[:, :, -1, None, :2] <= self.key_max_xy+margin)).all(-1)
        eligible = support & assigned_keys.bool()
        supported = eligible.any(-1)
        # If outside the support, still measure against the actual assigned key,
        # not key zero. Empty idle assignments are masked out of this branch.
        key = torch.where(supported, eligible.to(torch.int64).argmax(-1),
                          assigned_keys.to(torch.int64).argmax(-1))
        normal = self.pressed_normal[key]
        required = (normal*self.pressed_center[key]).sum(-1)+self.key_half_z[key]
        pad_plane = (endpoints[:, :, -1]*normal[..., None, :]).sum(-1).amin(-1)-self.radii[None, :, -1]-required
        pad_breach = (-pad_plane-self.options['active_contact_slack_m']).clamp_min(0)
        overlap = ((maximum[..., None, :2] >= self.key_min_xy)
                   & (minimum[..., None, :2] <= self.key_max_xy)).all(-1)
        bottom = torch.where(overlap, self.key_bottom, self.lowest_bottom).amax(-1)
        core_clearance = minimum[..., 2]-bottom-self.options['clearance_m']
        core_deficit = (-core_clearance.amin(-1)).clamp_min(0)
        contact_deficit = torch.maximum(core_deficit, pad_breach)
        # An unsupported active finger must approach above the surfaces. A
        # supported finger can transmit force without requiring its *commanded*
        # FK to lie above a key, but cannot start beneath the physical key slab.
        contact_deficit = torch.where(supported, contact_deficit,
                                      torch.maximum(core_deficit, strict['finger_clearance_deficit_m']))
        permission = active & supported & (pad_breach <= 1e-6) & (core_deficit <= 1e-6)
        return dict(finger_active_force_permitted=permission,
                    finger_assigned_support=supported & active,
                    finger_key_bottom_deficit_m=torch.where(active, core_deficit, torch.zeros_like(core_deficit)),
                    finger_actual_pad_plane_m=pad_plane,
                    finger_active_contact_deficit_m=contact_deficit)

    def evaluate(self, physical, lift, assigned_keys):
        """Actual-state feedback; contact mode separates servo force from geometry."""
        result = self._evaluate(physical, lift, assigned_keys)
        if self.options['active_command_mode'] == 'active_contact_v1':
            state = self._active_contact(physical, lift, assigned_keys, result)
            active = assigned_keys.bool().any(-1)
            result['finger_clearance_deficit_m'] = torch.where(active,
                state['finger_active_contact_deficit_m'], result['finger_clearance_deficit_m'])
            result['finger_terminal_allowed'] = state['finger_active_force_permitted']
            result.update(state)
        return result

    def reward_deficit(self, actual_metrics, requested_deficit, assigned_keys):
        actual = actual_metrics['finger_clearance_deficit_m']
        combined = torch.maximum(actual, requested_deficit)
        if self.options['active_command_mode'] == 'active_contact_v1':
            # Active PD overdrive is not actual penetration. Keep its raw FK in
            # diagnostics, but never silently reinsert it into the safety cost.
            return torch.where(assigned_keys.bool().any(-1), actual, combined)
        return combined

    def upcoming_request_metrics(self, physical, lift, assigned_keys, future_keys,
                                 future_seconds, window, requested_deficit):
        """Reward-only allowance for an imminent, correctly aimed press target.

        All inputs describe the instant the command was issued. ``lift`` must
        use the same convention as the original requested-target measurement
        (the guard uses zero support lift). This function never changes actions
        or actual-state metrics. Its result replaces only the requested part of
        ``reward_deficit``; callers must keep the original actual deficit.

        The terminal capsule must fit fully over one imminent assigned key.
        Its requested depth then uses the existing active-contact allowance.
        Proximal clearance, whole-finger key-bottom risk, and requested contact
        with every other key remain penalized. No general future-finger mask
        can grant this allowance to an incorrectly aimed target.
        """
        if type(window) not in (int, float) or not math.isfinite(window) or window < 0:
            raise ValueError('Upcoming press permission window must be finite and nonnegative')
        count = physical.shape[0]
        support_shape = (count, 2, 12) if self.rigid_arm else (count, 2)
        if (physical.shape != (count, 51) or lift.shape != support_shape
                or assigned_keys.shape != (count, 10, 88)
                or future_keys.shape != assigned_keys.shape
                or future_seconds.shape != (count, 10)
                or requested_deficit.shape != (count, 10)):
            raise ValueError('Invalid shapes for upcoming finger requested-target permission')
        disabled = dict(finger_reward_requested_deficit_m=requested_deficit,
                        finger_upcoming_requested_permission_mask=torch.zeros_like(
                            requested_deficit, dtype=torch.bool))
        if window == 0:
            return disabled
        if (self.options['active_command_mode'] != 'active_contact_v1'
                or any(self.options.get(name, True) for name in CONTROL_SWITCHES)):
            raise ValueError('Upcoming request permission requires active_contact_v1 with both physical controls disabled')

        future = future_keys.bool()
        kinematics = self.fk(physical, lift)
        strict = self._evaluate(physical, lift, future_keys, kinematics=kinematics)
        contact = self._active_contact(physical, lift, future_keys, strict, endpoints=kinematics[0])
        eligible = (~assigned_keys.bool().any(-1)
                    & (future.sum(-1) == 1)
                    & (future_seconds > 0) & (future_seconds <= window + 1e-7)
                    & strict['finger_terminal_allowed'])

        # Only the terminal request can receive the lower contact plane. Keep
        # the ordinary rest-surface clearance requirement for proximal links.
        proximal = (-strict['capsule_clearance_m'][:, :, :-1]).clamp_min(0).amax(-1)
        terminal = kinematics[0][:, :, -1]
        radius = self.radii[None, :, -1, None]
        minimum = terminal.amin(2) - radius
        maximum = terminal.amax(2) + radius
        overlap = ((maximum[:, :, None, :2] >= self.key_min_xy)
                   & (minimum[:, :, None, :2] <= self.key_max_xy)).all(-1)
        other_keys = overlap & ~future
        other_depth = (self.key_top + self.options['clearance_m']
                       - minimum[..., 2, None]).clamp_min(0)
        other_deficit = torch.where(other_keys, other_depth, 0.).amax(-1)
        candidate = torch.maximum(proximal, torch.maximum(
            other_deficit, contact['finger_active_contact_deficit_m']))
        effective = torch.where(eligible, torch.minimum(requested_deficit, candidate),
                                requested_deficit)
        return dict(finger_reward_requested_deficit_m=effective,
                    finger_upcoming_requested_permission_mask=eligible)

    @torch.no_grad()
    def filter(self, physical, *, actual_qpos, mocap_pos, assigned_keys, support=None):
        if physical.ndim != 2 or physical.shape[1] != 51 or assigned_keys.shape != (physical.shape[0], 10, 88):
            raise ValueError('Expected physical[N,51] and current assigned_keys[N,10,88]')
        lift = self.lift(mocap_pos) if support is None else support
        projection_lift = self.projection_support(lift)
        actual = self.physical_qpos(actual_qpos)
        requested_metrics = self._evaluate(physical, projection_lift, assigned_keys)
        actual_metrics = self.evaluate(actual, lift, assigned_keys)
        idle_enabled = self.options.get('idle_finger_protection', True)
        active_enabled = self.options.get('active_press_permission', True)
        active = assigned_keys.bool().any(-1)
        projected = physical.clone()
        # Diagnostics/reward remain usable with both action controls disabled.
        # Skip the iterative solver entirely in that case; never change targets.
        iterations = self.options['projection_iterations'] if idle_enabled or active_enabled else 0
        for _ in range(iterations):
            result = self._evaluate(projected, projection_lift, assigned_keys, jacobian=True)
            gradient = result['gradient'];deficit = result['finger_clearance_deficit_m']
            step = deficit[..., None]*gradient/(gradient.square().sum(-1, keepdim=True)+1e-10)
            step = step.clamp(-self.options['max_joint_correction_rad'], self.options['max_joint_correction_rad'])
            angles = (projected[:, self.finger_indices]+step).clamp(self.low, self.high)
            projected[:, self.finger_indices] = angles
        projection_metrics = self._evaluate(projected, projection_lift, assigned_keys)
        fallback = projection_metrics['finger_clearance_deficit_m'] > 1e-6
        projected[:, self.finger_indices] = torch.where(fallback[..., None], self.safe_angles,
                                                       projected[:, self.finger_indices])
        if self.options['active_command_mode'] == 'idle_only_probe':
            # Explicit ablation: a PD position target may lie through a key to
            # produce pressing force. This bypass is not an active-finger safety
            # guarantee; continue to report complete actual/requested geometry.
            active = assigned_keys.bool().any(-1)
            projected[:, self.finger_indices] = torch.where(active[..., None],
                physical[:, self.finger_indices], projected[:, self.finger_indices])
            fallback = fallback & ~active
        recovery_mask = torch.zeros_like(fallback)
        if self.options['active_command_mode'] == 'active_contact_v1' and active_enabled:
            active = assigned_keys.bool().any(-1)
            permitted = actual_metrics['finger_active_force_permitted']
            recovery_mask = active & (actual_metrics['finger_clearance_deficit_m'] > 1e-6)
            delta = self.options['active_recovery_delta_rad']
            near = actual[:, self.finger_indices]+(projected[:, self.finger_indices]-actual[:, self.finger_indices]).clamp(-delta, delta)
            near = near.clamp(self.low, self.high)
            # Unsupported approach targets remain near the actual finger pose;
            # dangerous fingers recover from that pose, not a remote fallback.
            recovered = actual.clone()
            no_assignment = torch.zeros_like(assigned_keys)
            for _ in range(self.options['projection_iterations']):
                state = self._evaluate(recovered, projection_lift, no_assignment, jacobian=True)
                gradient = state['gradient'];deficit = state['finger_clearance_deficit_m']
                change = deficit[..., None]*gradient/(gradient.square().sum(-1, keepdim=True)+1e-10)
                angles = recovered[:, self.finger_indices]+change.clamp(-.1, .1)
                angles = torch.maximum(actual[:, self.finger_indices]-delta,
                                       torch.minimum(angles, actual[:, self.finger_indices]+delta)).clamp(self.low, self.high)
                recovered[:, self.finger_indices] = angles
            near = torch.where(recovery_mask[..., None], recovered[:, self.finger_indices], near)
            active_targets = torch.where(permitted[..., None], physical[:, self.finger_indices], near)
            projected[:, self.finger_indices] = torch.where(active[..., None], active_targets,
                                                            projected[:, self.finger_indices])
            fallback = fallback & ~active
        controlled = ((~active & idle_enabled) | (active & active_enabled))
        projected[:, self.finger_indices] = torch.where(controlled[..., None],
            projected[:, self.finger_indices], physical[:, self.finger_indices])
        fallback &= controlled
        recovery_mask &= controlled
        interlock = torch.zeros((physical.shape[0], 2), device=physical.device, dtype=torch.bool)
        if self.options['wrist_clearance_interlock'] and idle_enabled:
            # A local 2 mm clearance deficit merely means loss of margin. Hold
            # lateral targets only once an idle finger is below the key surface.
            strict_actual = self._evaluate(actual, lift, assigned_keys)
            below = (~assigned_keys.bool().any(-1)
                     & (strict_actual['finger_clearance_deficit_m'] > self.options['clearance_m']))
            bad_hand = below.reshape(-1, 2, 5).any(-1)
            moving = (physical[:, self.wrist_indices]-actual[:, self.wrist_indices]).abs().amax(-1) > 1e-6
            interlock = bad_hand & moving
            hold = actual[:, self.wrist_indices].clamp(self.wrist_low, self.wrist_high)
            projected[:, self.wrist_indices] = torch.where(interlock[..., None], hold,
                                                          projected[:, self.wrist_indices])
        applied_metrics = self._evaluate(projected, projection_lift, assigned_keys)
        reference = actual.clone();reference[:, self.finger_indices] = self.origin_angles
        reference_metrics = self.evaluate(reference, lift, assigned_keys)
        metrics = dict(finger_clearance_deficit_m=self.reward_deficit(
            actual_metrics, requested_metrics['finger_clearance_deficit_m'], assigned_keys),
            finger_pose_reference_feasible=reference_metrics['finger_clearance_deficit_m'] <= 1e-6,
            finger_actual_deficit_m=actual_metrics['finger_clearance_deficit_m'],
            finger_requested_deficit_m=requested_metrics['finger_clearance_deficit_m'],
            finger_applied_deficit_m=applied_metrics['finger_clearance_deficit_m'],
            finger_projection_mask=(projected[:, self.finger_indices]-physical[:, self.finger_indices]).abs().amax(-1) > 1e-7,
            finger_fallback_mask=fallback, finger_active=actual_metrics['finger_active'],
            finger_terminal_allowed=actual_metrics['finger_terminal_allowed'],
            finger_recovery_mask=recovery_mask, wrist_clearance_interlock_mask=interlock)
        for name in ('finger_active_force_permitted', 'finger_assigned_support',
                     'finger_key_bottom_deficit_m', 'finger_actual_pad_plane_m'):
            if name in actual_metrics:
                metrics[name] = actual_metrics[name]
        return projected, metrics
