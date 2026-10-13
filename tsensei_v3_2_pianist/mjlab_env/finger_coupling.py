"""Optional four-finger servo coupling derived from the compiled hand model.

PIP/DIP remain physical joints. The policy commands MCP and swing; the followers
seek a downward distal direction relative to the resting piano plane. This is
an experimental controller, not a mechanical tendon or a calibrated pad angle.
The distal landmark inherits TODO(V32-07) from the bundled fingertip geometry.
"""
import math

import mujoco
import numpy as np
import torch


MODE = 'distal_60deg_v1'
RANGE_MODE = 'distal_angle_range_v1'
FINGERS = ('index', 'mid', 'ring', 'pinky')
FINGER_NAMES = tuple(side+'_'+finger for side in ('right', 'left') for finger in FINGERS)
FOLLOWER_INDICES = tuple(base+i for base in (0, 25) for i in (6, 7, 10, 11, 14, 15, 18, 19))


def canonical(options):
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError('Unknown finger coupling options')
    mode = options.get('mode')
    if type(options.get('schema')) is not int or options['schema'] != 1 or mode not in (MODE, RANGE_MODE):
        raise ValueError('Expected schema 1 distal_60deg_v1 or distal_angle_range_v1 finger coupling')
    angle_key = 'target_angle_deg' if mode == MODE else 'angle_ranges_deg'
    if set(options)-{'schema', 'mode', angle_key, 'reduce_observation'}:
        raise ValueError('Unknown finger coupling options')
    reduced = options.get('reduce_observation', True)
    if type(reduced) is not bool:
        raise ValueError('finger coupling reduce_observation must be a bool')
    if mode == MODE:
        angle = options.get('target_angle_deg', 60.)
        if type(angle) not in (int, float) or not math.isfinite(angle) or not 0 < angle < 90:
            raise ValueError('finger coupling target_angle_deg must be finite and between 0 and 90')
        # Preserve the exact options/solver contract of existing checkpoints.
        return dict(schema=1, mode=MODE, target_angle_deg=float(angle), reduce_observation=reduced)
    ranges = options.get('angle_ranges_deg', {name: [50., 80.] for name in FINGER_NAMES})
    if not isinstance(ranges, dict) or set(ranges) != set(FINGER_NAMES):
        raise ValueError('angle_ranges_deg must specify all eight fingers: '+', '.join(FINGER_NAMES))
    normalized = {}
    for name in FINGER_NAMES:
        pair = ranges[name]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in pair) or
                not 0 < pair[0] <= pair[1] < 90):
            raise ValueError(f'angle_ranges_deg[{name}] requires finite 0 < lower <= upper < 90')
        normalized[name] = [float(v) for v in pair]
    return dict(schema=1, mode=RANGE_MODE, angle_ranges_deg=normalized, reduce_observation=reduced)


def observation_contract(task_manifest, action_names, options):
    """Experimentally remove follower qpos/action fields, including all stack slots.

    Follower joints retain physical tracking lag and contacts: removing their
    measured qpos creates partial observation, not a lossless state reduction.
    """
    options = canonical(options)
    if options is None:
        return None
    contract = task_manifest['contract']
    stack = contract['environment']['frame_stack']
    fields = contract['observation_fields_before_stack']
    follower_names = {f'{side}_/{side}_hand_{finger}_{kind}_joint'
                      for side in ('right', 'left') for finger in FINGERS for kind in ('PIP', 'DIP')}
    kept = []; removed = []; offset = 0
    for field in fields:
        path = field['path']
        size = int(np.prod(field['shape']))
        drop = set()
        if options['reduce_observation']:
            if path == 'action':
                if size != 51:
                    raise ValueError('Coupled observation requires full physical previous action')
                drop = set(FOLLOWER_INDICES)
            elif path in ('right_/joints_pos', 'left_/joints_pos'):
                side = 0 if path.startswith('right') else 1
                names = task_manifest['hand_joints'][side]
                if len(names) != size:
                    raise ValueError('Coupled observation hand joint layout mismatch')
                drop = {i for i, name in enumerate(names) if name in follower_names}
                if len(drop) != 8:
                    raise ValueError('Cannot resolve eight follower qpos slots per hand')
        for i in range(size):
            (removed if i in drop else kept).extend((offset+i)*stack+j for j in range(stack))
        offset += size
    if offset*stack != contract['observation_size']:
        raise ValueError('Coupled observation stacking contract mismatch')
    return dict(schema=1, mode=options['mode'], reduce_observation=options['reduce_observation'],
                full_base_size=offset*stack, reduced_base_size=len(kept),
                retained_base_indices=kept, removed_base_indices=removed,
                removed_fields='four-finger PIP/DIP actual qpos and previous physical controls only',
                appended_preparation_and_anticipation='preserved')


def _rotate(vector, axis, angle):
    """Rodrigues rotation, with independent batch/finger/candidate dimensions."""
    c = angle.cos()[..., None]; s = angle.sin()[..., None]
    return vector*c + torch.linalg.cross(axis.expand_as(vector), vector, dim=-1)*s + axis*(axis*vector).sum(-1, keepdim=True)*(1-c)


class FingerCoupler:
    """Bounded PIP search and exact one-axis DIP solve on the tensor device.

    PIP candidates cover the allowed range and the origin/actual state, followed
    by a local refinement. For every candidate, all analytic DIP roots in the
    allowed range plus closest clamped extrema are considered. Among solutions
    within 0.05 degrees, minimum squared travel from origin chooses one pose.
    Infeasible means this bounded search did not meet that tolerance; no claim
    of a globally feasible contact trajectory follows from an orientation solve.
    """
    TOLERANCE_DEG = .05
    GRID_POINTS = 17
    REFINEMENT_POINTS = 9

    def __init__(self, bundle, *, device='cuda:0', options, physical_low=None, physical_high=None):
        self.options = canonical(options)
        if self.options is None:
            raise ValueError('Construct coupling only for explicit options')
        if 'm6' in bundle.manifest:
            raise ValueError('Finger coupling currently requires --no-arms')
        self.device = torch.device(device)
        tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        integer = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.int64, device=self.device)
        model = bundle.model
        joint_to_full = {int(model.actuator_trnid[act, 0]): int(full)
                         for act, full in zip(bundle.action_ids, bundle.active_action_indices)}
        low = np.asarray(bundle.low if physical_low is None else physical_low)
        high = np.asarray(bundle.high if physical_high is None else physical_high)
        reference = bundle.reset_cpu()
        reference.qpos[:] = bundle.origin_qpos
        reference.mocap_pos[:] = bundle.origin_mocap_pos
        reference.mocap_quat[:] = bundle.origin_mocap_quat
        # Resting key geometry supplies the plane normal, independent of wrist names.
        reference.qpos[bundle.key_qadr] = model.qpos0[bundle.key_qadr]
        mujoco.mj_forward(model, reference)
        normal = None
        for gid in range(model.ngeom):
            name = model.geom(gid).name or ''
            if name.startswith('piano/') and ('white_key_geom_' in name or 'black_key_geom_' in name):
                current = reference.geom_xmat[gid].reshape(3, 3)[:, 2]
                if normal is None:
                    normal = current.copy()
                if not np.allclose(current, normal, atol=1e-7):
                    raise ValueError('Coupling requires a common resting piano plane')
        if normal is None:
            raise ValueError('Cannot resolve resting piano key plane')
        self.normal = tensor(normal)
        groups = []
        for side in ('right', 'left'):
            for finger in FINGERS:
                joint_names = [f'{side}_/{side}_hand_{finger}_{kind}_joint' for kind in ('MCP', 'PIP', 'DIP')]
                joints = [int(model.joint(name).id) for name in joint_names]
                full = [joint_to_full[j] for j in joints]
                if not set(full).issubset(set(bundle.active_action_indices)):
                    raise ValueError('Coupling requires all MCP/PIP/DIP physical actuators')
                if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
                    raise ValueError('Coupling supports hinge finger joints only')
                for joint, index in zip(joints[1:], full[1:]):
                    if not (model.jnt_range[joint, 0]-1e-8 <= low[index] < high[index] <= model.jnt_range[joint, 1]+1e-8):
                        raise ValueError('Follower targets must lie within physical joint ranges')
                bodies = model.jnt_bodyid[joints]
                # Rotation composition assumes the original serial MCP/swing/PIP/DIP chain.
                def ancestor(first, last):
                    while last and last != first:
                        last = int(model.body_parentid[last])
                    return last == first
                if not ancestor(int(bodies[0]), int(bodies[1])) or int(model.body_parentid[bodies[2]]) != bodies[1]:
                    raise ValueError('Unsupported finger chain for coupling')
                site = model.site(f'{side}_/{finger}_tip').id
                if model.site_bodyid[site] != bodies[2]:
                    raise ValueError('Distal tip site must belong to DIP body')
                direction = model.site_pos[site]-model.jnt_pos[joints[2]]
                if np.linalg.norm(direction) < 1e-6:
                    raise ValueError('Invalid distal landmark direction')
                direction = direction/np.linalg.norm(direction)
                groups.append(dict(name=side+'_'+finger, joint_names=joint_names, full_indices=full,
                    qpos_addresses=model.jnt_qposadr[joints].tolist(), body_ids=bodies.tolist(),
                    local_axes=model.jnt_axis[joints].tolist(), distal_direction_local=direction.tolist(),
                    follower_low=low[full[1:]].tolist(), follower_high=high[full[1:]].tolist(),
                    follower_origin=bundle.origin_qpos[model.jnt_qposadr[joints[1:]]].tolist()))
        self.groups = groups
        if [index for group in groups for index in group['full_indices'][1:]] != list(FOLLOWER_INDICES):
            raise ValueError('Coupling requires the canonical full 51-channel finger ordering')
        self.indices = integer([g['full_indices'] for g in groups])
        self.qadr = integer([g['qpos_addresses'] for g in groups])
        self.bodies = integer([g['body_ids'] for g in groups])
        self.axes = tensor([g['local_axes'] for g in groups])
        self.direction = tensor([g['distal_direction_local'] for g in groups])
        self.low = tensor([g['follower_low'] for g in groups]); self.high = tensor([g['follower_high'] for g in groups])
        self.origin = tensor([g['follower_origin'] for g in groups])
        self.grid = tensor(np.linspace(0, 1, self.GRID_POINTS))
        self.refine = tensor(np.linspace(-1, 1, self.REFINEMENT_POINTS))
        self.turns = tensor([-2*math.pi, 0., 2*math.pi])
        self.range_mode = self.options['mode'] == RANGE_MODE
        if self.range_mode:
            self.angle_bounds = tensor([self.options['angle_ranges_deg'][g['name']] for g in groups])
            self.target_dot = -torch.sin(self.angle_bounds*(math.pi/180))
        else:
            self.target_dot = -math.sin(math.radians(self.options['target_angle_deg']))
        self.contract = dict(**self.options, physical_model='unchanged_servo_joints',
            follower_full_indices=list(FOLLOWER_INDICES), coupled_fingers=groups,
            upstream_state='actual wrist and swing; commanded MCP',
            plane_normal=normal.tolist(), distal_axis='DIP pivot to packaged fingertip landmark; TODO(V32-07)',
            solver=dict(name='bounded_pip_grid_analytic_dip_v1', grid_points=self.GRID_POINTS,
                refinement_points=self.REFINEMENT_POINTS, tolerance_deg=self.TOLERANCE_DEG,
                preference='minimum squared PIP/DIP displacement from origin among feasible candidates'),
            update='once per control frame; no qpos projection',
            angle='signed downward angle of directed distal axis to resting piano plane')
        if self.range_mode:
            self.contract['solver']['name'] = 'bounded_pip_grid_analytic_dip_interval_v1'
            self.contract['solver']['error'] = 'distance outside each finger interval; zero throughout interval'

    def _angle_error(self, angle):
        if not self.range_mode:
            return (angle-self.options['target_angle_deg']).abs()
        shape = (1, len(self.groups))+(1,)*(angle.ndim-2)
        lower = self.angle_bounds[:, 0].reshape(shape)
        upper = self.angle_bounds[:, 1].reshape(shape)
        return (lower-angle).clamp_min(0)+(angle-upper).clamp_min(0)

    def _frames(self, body_xmat):
        rotation = body_xmat[:, self.bodies].reshape(body_xmat.shape[0], 8, 3, 3, 3)
        axes = (rotation@self.axes[None, ..., None]).squeeze(-1)
        distal = (rotation[:, :, 2]@self.direction[None, ..., None]).squeeze(-1)
        return axes, distal

    def actual_diagnostics(self, body_xmat):
        rotation = body_xmat[:, self.bodies[:, 2]].reshape(body_xmat.shape[0], 8, 3, 3)
        direction = (rotation@self.direction[None, ..., None]).squeeze(-1)
        angle = torch.asin(-(direction*self.normal).sum(-1).clamp(-1, 1))* (180/math.pi)
        return dict(coupling_actual_angle_deg=angle,
                    coupling_actual_error_deg=self._angle_error(angle))

    def _solve_candidates(self, pip, actual, axes, direction, mcp_delta):
        # Undo actual DIP, then apply requested PIP and commanded MCP while
        # retaining the actual upstream wrist/swing transformations.
        base = _rotate(direction, axes[:, :, 2], -actual[:, :, 2])
        pip_axis = axes[:, :, 1, None, :]
        delta = pip-actual[:, :, 1, None]
        base = _rotate(base[:, :, None, :].expand(-1, -1, pip.shape[-1], -1), pip_axis, delta)
        dip_axis = _rotate(axes[:, :, 2, None, :].expand_as(base), pip_axis, delta)
        mcp_axis = axes[:, :, 0, None, :]
        base = _rotate(base, mcp_axis, mcp_delta[:, :, None])
        dip_axis = _rotate(dip_axis, mcp_axis, mcp_delta[:, :, None])
        dot_axis = (dip_axis*self.normal).sum(-1)
        parallel = (dip_axis*base).sum(-1)
        c = dot_axis*parallel
        a = (base*self.normal).sum(-1)-c
        b = (torch.linalg.cross(dip_axis, base, dim=-1)*self.normal).sum(-1)
        radius = torch.sqrt(a.square()+b.square()).clamp_min(1e-12)
        phase = torch.atan2(b, a)
        if self.range_mode:
            # Both interval boundaries matter; an interior posture must not be
            # attracted to either boundary or to a shared midpoint angle.
            offset = torch.acos(((self.target_dot[None, :, None, :]-c[..., None])/radius[..., None]).clamp(-1, 1))
            roots = torch.stack((phase[..., None]+offset, phase[..., None]-offset), -1)
            roots = (roots[..., None]+self.turns).flatten(-3)
            # Include stationary points for the best bounded fallback when the
            # interval is unreachable, and origin DIP for feasible interiors.
            extrema = (torch.stack((phase, phase+math.pi), -1)[..., None]+self.turns).flatten(-2)
            stationary = torch.stack((self.origin[:, 1][None].expand_as(actual[:, :, 2]),
                actual[:, :, 2], self.low[:, 1][None].expand_as(actual[:, :, 2]),
                self.high[:, 1][None].expand_as(actual[:, :, 2])), -1)
            dip = torch.cat((roots, extrema, stationary[:, :, None, :].expand(-1, -1, pip.shape[-1], -1)), -1)
            dip = dip.clamp(self.low[:, 1][None, :, None, None], self.high[:, 1][None, :, None, None])
        else:
            offset = torch.acos(((self.target_dot-c)/radius).clamp(-1, 1))
            roots = torch.stack((phase+offset, phase-offset), -1)
            dip = (roots[..., None]+self.turns).flatten(-2).clamp(self.low[:, 1][None, :, None, None], self.high[:, 1][None, :, None, None])
        z = a[..., None]*dip.cos()+b[..., None]*dip.sin()+c[..., None]
        angle = torch.asin(-z.clamp(-1, 1))*(180/math.pi)
        error = self._angle_error(angle)
        effort = (pip[..., None]-self.origin[:, 0][None, :, None, None]).square()+(dip-self.origin[:, 1][None, :, None, None]).square()
        feasible = error <= self.TOLERANCE_DEG
        # Feasible solutions are ordered by excursion, others by angular error.
        # 1000 exceeds the maximum excursion on the validated bounded joints.
        score = torch.where(feasible, effort, 1000+error+effort*1e-6).flatten(-2)
        chosen = score.argmin(-1, keepdim=True)
        take = lambda x: x.flatten(-2).gather(-1, chosen).squeeze(-1)
        pip = pip[..., None].expand_as(dip)
        return take(pip), take(dip), take(error), take(angle)

    def apply(self, physical, *, actual_qpos, body_xmat):
        actual = actual_qpos[:, self.qadr]
        axes, direction = self._frames(body_xmat)
        mcp_delta = physical[:, self.indices[:, 0]]-actual[:, :, 0]
        grid = self.low[:, 0, None]+(self.high[:, 0, None]-self.low[:, 0, None])*self.grid
        pip = torch.cat((grid[None].expand(actual.shape[0], -1, -1),
                         self.origin[:, 0].clamp(self.low[:, 0], self.high[:, 0])[None, :, None].expand(actual.shape[0], -1, -1),
                         actual[:, :, 1, None].clamp(self.low[:, 0][None, :, None], self.high[:, 0][None, :, None])), -1)
        first = self._solve_candidates(pip, actual, axes, direction, mcp_delta)
        spacing = (self.high[:, 0]-self.low[:, 0])/(self.GRID_POINTS-1)
        refine = (first[0][..., None]+spacing[None, :, None]*self.refine).clamp(self.low[:, 0][None, :, None], self.high[:, 0][None, :, None])
        # Include original candidates so refinement cannot lose a valid solution.
        pip = torch.cat((pip, refine), -1)
        p, d, error, angle = self._solve_candidates(pip, actual, axes, direction, mcp_delta)
        output = physical.clone()
        output[:, self.indices[:, 1]] = p
        output[:, self.indices[:, 2]] = d
        follower = torch.stack((p, d), -1)
        at_limit = ((follower-self.low).abs()<1e-5) | ((follower-self.high).abs()<1e-5)
        return output, dict(coupling_target_angle_deg=angle, coupling_target_error_deg=error,
            coupling_infeasible=error>self.TOLERANCE_DEG, coupling_follower_at_limit=at_limit.any(-1))
