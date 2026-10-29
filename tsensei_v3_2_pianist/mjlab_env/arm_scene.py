"""2026-10-25: virtual forearm joint names are now palm_base_*; physical axes unchanged.

Cartesian palm-position IK with a structural flange/connector/hand assembly.

Link7 is a kinematic mocap root. The flange, connector and hand base are its
jointless descendants; only finger/key joints are simulated dynamically.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .arm_options import ASSETS, ROOT, SIDES, CONNECTOR, canonical
from .arm_follow import ArmFollower
from .arm_ik import rot, quaternion
from ..preview_connector import load_config, tool_transforms, offset_transform

FLANGE_IN_LINK7 = np.array([0., -.095, 0.])


def _xml_pose(matrix):
    return dict(pos=' '.join(map(str, matrix[:3, 3])),
                quat=' '.join(map(str, np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1))))


def _preserve_physical_parameters(old, new, hand_roots):
    """Restore exact compiled finger/key inputs after XML's decimal rounding.

    Removing wrist joints and reparenting hands changes all entity addresses.
    Copy by names, never by shared array prefixes. Hand-root local poses are the
    intentional assembly change; masses/inertias and all other physical inputs
    are preserved. Derived mass/reference constants are then recomputed.
    """
    kinds = {'body': ('body', old.nbody, new.nbody), 'jnt': ('joint', old.njnt, new.njnt),
             'geom': ('geom', old.ngeom, new.ngeom), 'site': ('site', old.nsite, new.nsite),
             'mesh': ('mesh', old.nmesh, new.nmesh), 'cam': ('cam', old.ncam, new.ncam),
             'light': ('light', old.nlight, new.nlight), 'actuator': ('actuator', old.nu, new.nu)}
    maps = {}
    for prefix, (kind, count, newcount) in kinds.items():
        names = {getattr(new, kind)(i).name: i for i in range(newcount)}
        maps[prefix] = [(i, names[name]) for i in range(count)
                        if (name := getattr(old, kind)(i).name) in names and name]
    for name in dir(old):
        prefix = name.split('_')[0]
        if prefix not in maps:
            continue
        before, after = getattr(old, name), getattr(new, name)
        if not (isinstance(before, np.ndarray) and np.issubdtype(before.dtype, np.floating)
                and before.ndim and before.shape[0] == kinds[prefix][1]
                and before.shape[1:] == after.shape[1:]):
            continue
        for oi, ni in maps[prefix]:
            if prefix == 'body' and old.body(oi).name in hand_roots and name in ('body_pos', 'body_quat'):
                continue
            after[ni] = before[oi]
    # All retained physical joints are scalar, as required by SceneBundle.
    for oi, ni in maps['jnt']:
        new.qpos0[new.jnt_qposadr[ni]] = old.qpos0[old.jnt_qposadr[oi]]
        for name in ('dof_armature', 'dof_damping', 'dof_frictionloss', 'dof_solref', 'dof_solimp'):
            getattr(new, name)[new.jnt_dofadr[ni]] = getattr(old, name)[old.jnt_dofadr[oi]]
    for kind in ('body', 'mesh'):
        for oi, ni in maps[kind]:
            size = int(getattr(old, kind + '_bvhnum')[oi])
            if size != int(getattr(new, kind + '_bvhnum')[ni]):
                raise ValueError('Original geometry BVH changed: ' + getattr(old, kind)(oi).name)
            if size:
                src, dst = (int(getattr(m, kind + '_bvhadr')[i]) for m, i in ((old, oi), (new, ni)))
                new.bvh_aabb[dst:dst+size] = old.bvh_aabb[src:src+size]
    for oi, ni in maps['mesh']:
        for field in ('vert', 'normal', 'texcoord'):
            size = int(getattr(old, 'mesh_' + field + 'num')[oi])
            if size != int(getattr(new, 'mesh_' + field + 'num')[ni]):
                raise ValueError('Original mesh changed: ' + old.mesh(oi).name)
            if size:
                src, dst = (int(getattr(m, 'mesh_' + field + 'adr')[i]) for m, i in ((old, oi), (new, ni)))
                getattr(new, 'mesh_' + field)[dst:dst+size] = getattr(old, 'mesh_' + field)[src:src+size]
    mujoco.mj_setConst(new, mujoco.MjData(new))


def _restore_parent_collision_filter(old, new, contact):
    """Materialize only native parent exclusions lost by the rigid conversion.

    A mocap hand root belongs to weld 0, which MuJoCo deliberately exempts
    from parent filtering. Retain the original hand-only model's exclusions
    as body pairs, using compiled weld ancestry rather than a guessed list.
    Explicit geom pairs keep their native precedence over body exclusions.
    """
    def filtered(model, a, b):
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT:
            return False
        wa, wb = int(model.body_weldid[a]), int(model.body_weldid[b])
        return bool(wa and wb and (wa == model.body_weldid[model.body_parentid[wb]]
                                  or wb == model.body_weldid[model.body_parentid[wa]]))

    # Only surviving bodies with physical collision geometry can gain a new
    # dynamic contact. Visual-only arm/connector bodies are outside this set.
    bodies = []
    for bid in np.unique(old.geom_bodyid[(old.geom_contype | old.geom_conaffinity) != 0]):
        name = old.body(int(bid)).name
        nid = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
        if nid >= 0:
            bodies.append((int(bid), nid, name))
    existing = {frozenset((p.get('body1'), p.get('body2'))) for p in contact.findall('exclude')}
    restored = []
    for i, (a, na, name_a) in enumerate(bodies):
        for b, nb, name_b in bodies[i+1:]:
            if filtered(old, a, b) and not filtered(new, na, nb):
                pair = sorted((name_a, name_b))
                if frozenset(pair) not in existing:
                    ET.SubElement(contact, 'exclude', body1=pair[0], body2=pair[1])
                    existing.add(frozenset(pair))
                    restored.append(pair)
    return sorted(restored)


def _make_arm_collision_query(xml, model, assets):
    """Compile a contact-only model for arm conflicts.

    The active scene keeps arm meshes visual-only so the kinematic assembly
    cannot inject unmodelled reaction forces.  This copy enables only arm
    meshes and adds explicit pairs for the three supported diagnostic scopes.
    The returned pair list is name-based and therefore stable across compiled
    geom re-numbering.
    """
    root = ET.fromstring(ET.tostring(xml, encoding='unicode'))
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for node in list(contact):
        if node.tag == 'pair':
            contact.remove(node)
    def side(name):
        return 'right' if name.startswith('arm_follow_right_') or name.startswith('arm_connector_right_') else (
            'left' if name.startswith('arm_follow_left_') or name.startswith('arm_connector_left_') else None)
    arm = []
    hand = []
    piano = []
    for gid in range(model.ngeom):
        name = model.geom(gid).name or ''
        body = model.body(int(model.geom_bodyid[gid])).name or ''
        if side(name) is not None:
            arm.append((name, body, side(name)))
        elif body.startswith(('right_/', 'left_/')):
            hand.append((name, body, 'right' if body.startswith('right_/') else 'left'))
        elif name.startswith('piano/'):
            piano.append((name, body, None))
    if not arm:
        raise ValueError('Rigid arm collision query found no arm meshes')
    arm_names = {name for name, _, _ in arm}
    for node in root.iter('geom'):
        if node.get('name') in arm_names:
            node.set('contype', '1'); node.set('conaffinity', '1'); node.set('density', '0')
    pairs = []
    for i, (first, first_body, first_side) in enumerate(arm):
        for second, second_body, second_side in arm[i+1:]:
            if first_side != second_side:
                pairs.append((first, second, 'arm_arm'))
        for second, second_body, second_side in hand:
            # Link7, flange and connector are the approved rigid mounting
            # chain.  Their same-side contacts are structural, not conflicts.
            mounting = (first_side == second_side and
                        any(token in first_body for token in ('Link7_', 'arm_flange_', 'arm_connector_')))
            if not mounting:
                pairs.append((first, second, 'arm_hand'))
        for second, _, _ in piano:
            pairs.append((first, second, 'arm_piano'))
    seen = set()
    for first, second, category in pairs:
        key = (min(first, second), max(first, second))
        if key in seen:
            continue
        seen.add(key)
        ET.SubElement(contact, 'pair', name=f'arm_query_{category}_{len(seen)}',
                       geom1=first, geom2=second, margin='0', gap='0', condim='1')
    query = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)
    return query, pairs


def augment_bundle(bundle, options):
    """Transform a private bundle; historical XML, recordings and archives stay intact."""
    options = canonical(options)
    old = bundle.model
    if 'm6' in bundle.manifest:
        raise ValueError('Cartesian arm assembly requires the hand-only source; old m6 bundles need their archive')
    if bundle.preparation_options is not None and bundle.preparation_options.get('mode') != 'policy_v1':
        raise ValueError('Arm follower supports policy preparation or no preparation')
    if bundle.disable_wrist_tz or 'right_/palm_base_yaw' not in bundle.fixed_joint_values:
        raise ValueError('Cartesian arm follower requires live TX/TY/TZ and structurally fixed Yaw')
    connector = load_config(CONNECTOR)
    installed = tool_transforms(connector)
    for side in SIDES:
        specified = np.eye(4)
        specified[:3, :3] = Rotation.from_euler('xyz', options['link7_to_hand'][side]['rpy_rad']).as_matrix()
        specified[:3, 3] = options['link7_to_hand'][side]['xyz_m']
        if not np.allclose(specified, installed[side], atol=1e-12, rtol=0):
            raise ValueError('Arm mount disagrees with approved four-hole connector: ' + side)
    initial, reference = bundle.reset_cpu(), bundle.reset_cpu()
    reference.mocap_pos[:] = bundle.origin_mocap_pos
    reference.mocap_quat[:] = bundle.origin_mocap_quat
    translation_names = ('palm_base_tx', 'palm_base_ty', 'palm_base_tz')
    orientation_names = ('palm_base_pitch', 'palm_base_roll')
    wrist_ids = {}
    translation_ids = {}
    orientation_ids = {}
    for side in SIDES:
        translation_ids[side] = [j for j in range(old.njnt)
            if (old.joint(j).name or '') in {f'{side}_/{name}' for name in translation_names}]
        orientation_ids[side] = [j for j in range(old.njnt)
            if (old.joint(j).name or '') in {f'{side}_/{name}' for name in orientation_names}]
        if len(translation_ids[side]) != 3 or len(orientation_ids[side]) != 2:
            raise ValueError('Cartesian arm requires TX/TY/TZ plus fixed Pitch/Roll in each hand scene')
        wrist_ids[side] = translation_ids[side] + orientation_ids[side]
        reference.qpos[old.jnt_qposadr[translation_ids[side]]] = 0
    mujoco.mj_forward(old, reference)
    wrists, initial_poses, removed = {}, {}, set()
    indices = list(bundle.active_action_indices)
    orientation_action_indices = set()
    cartesian_policy_indices = []
    for side in SIDES:
        body_name = bundle.manifest['hands'][side]['body']
        body, ids = old.body(body_name).id, wrist_ids[side]
        translation = translation_ids[side]
        actions = [int(full) for aid, full in zip(bundle.action_ids, indices)
                   if old.actuator_trnid[aid, 0] in translation]
        orientation_actions = [int(full) for aid, full in zip(bundle.action_ids, indices)
                         if old.actuator_trnid[aid, 0] in orientation_ids[side]]
        if len(actions) != 3 or len(orientation_actions) != 2 or list(old.jnt_type[translation]) != [2, 2, 2]:
            raise ValueError('Expected TX/TY/TZ Cartesian request coordinates and fixed Pitch/Roll')
        orientation_action_indices.update(orientation_actions)
        cartesian_policy_indices.extend(actions)
        # 2026-10-25: the external world-XYZ initial pose applies to arms too.
        # Zeroing removed slides defines Cartesian request zero, not a second
        # installation position. Rebase the fixed arm to the requested origin.
        source_pose = bundle.manifest.get('source_pose', {})
        rest_pos = reference.xpos[body].copy()
        if source_pose.get('contract') == 'initial_palm_base_world_xyz_v1':
            rest_pos = np.asarray(source_pose[side + '_initial_world_xyz_m'], dtype=float)
            if rest_pos.shape != (3,) or not np.isfinite(rest_pos).all():
                raise ValueError('Invalid initial palm-base world position: ' + side)
        wrists[side] = dict(indices=actions, axis=old.jnt_axis[translation].copy(), kind=old.jnt_type[translation].tolist(),
            rest_pos=rest_pos.copy(), rest_rot=reference.xmat[body].reshape(3, 3).copy(),
            body_name=body_name)
        # Position is absolute world XYZ; requests stay relative world XYZ.
        initial_poses[side] = (rest_pos.copy(), reference.xmat[body].reshape(3, 3).copy())
        for j in ids:
            removed.add(old.joint(j).name)
            # These are placeholders in the historical joint vector only. The
            # current arm observation adds actual q/v, IK reference and pose errors.
            bundle.fixed_joint_values[old.joint(j).name] = float(bundle.reset_qpos[old.jnt_qposadr[j]])
    # The freshly exported music task still uses the verified hand-only
    # coordinate convention. Keep that reference for strict asset validation;
    # runtime observations/rewards use the derived Cartesian reset below.
    bundle.task_source_neutral_action = bundle.neutral_action.copy()
    bundle.neutral_action[indices] = bundle.reset_ctrl[bundle.action_ids]
    # Cartesian rows are displacements from the installed palm pose. The
    # hand-only scene's TY reset (0.04 m) is an absolute slide coordinate and
    # must not become an initial Cartesian request.
    bundle.neutral_action[cartesian_policy_indices] = 0.
    with tempfile.TemporaryDirectory(prefix='tsensei-arm-model-') as folder:
        path = Path(folder) / 'compiled.xml'
        mujoco.mj_saveLastXML(str(path), old)
        xml = ET.parse(path).getroot()
    for parent in xml.iter():
        for node in list(parent):
            if ((node.tag == 'joint' and node.get('name') in removed)
                    or (parent.tag == 'actuator' and node.get('joint') in removed)):
                parent.remove(node)
    world = xml.find('worldbody')
    hands = {}
    rigid_geoms = {}
    for side, w in wrists.items():
        weld = old.body_weldid[old.body(w['body_name']).id]
        rigid_geoms[side] = [i for i in range(old.ngeom)
            if old.body_weldid[old.geom_bodyid[i]] == weld
            and (old.geom_contype[i] or old.geom_conaffinity[i])]
        root = next(b for b in world.findall('body')
                    if any(x.get('name') == w['body_name'] for x in b.iter('body')))
        hand = next(b for b in root.iter('body') if b.get('name') == w['body_name'])
        world.remove(root)
        hand.set('pos', '0 0 0'); hand.set('quat', '1 0 0 0')
        hands[side] = hand
    assets = {name: (bundle.directory / name).read_bytes() for name in bundle.manifest['files']
              if name not in ('scene.xml', 'cpu_reference.npz')}
    arm_names = {}
    for side in SIDES:
        source = ET.parse(ASSETS / side / 'arm.urdf').getroot()
        names = [f'Base_{side[0].upper()}'] + [f'Link{i}_{side[0].upper()}' for i in range(1, 8)]
        links = {n.get('name'): n for n in source.findall('link')}
        arm_names[side] = []
        for name in names:
            body_name = f'arm_follow_{side}_{name}'
            arm_names[side].append(body_name)
            body = ET.SubElement(world, 'body', name=body_name, mocap='true')
            for index, visual in enumerate(links[name].findall('visual')):
                mesh_name = f'{body_name}_{index}'
                filename = Path(visual.find('geometry/mesh').get('filename')).name
                asset_name = f'{side}_{filename}'
                assets[asset_name] = (ASSETS / side / 'meshes' / filename).read_bytes()
                ET.SubElement(xml.find('asset'), 'mesh', name=mesh_name, file=asset_name)
                origin = visual.find('origin')
                xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                quat = np.roll(Rotation.from_euler('xyz', rpy).as_quat(), 1)
                ET.SubElement(body, 'geom', type='mesh', mesh=mesh_name, pos=xyz,
                    quat=' '.join(map(str, quat)), contype='0', conaffinity='0', density='0',
                    group='1', rgba='.9 .92 .95 1')
        flange = ET.SubElement(body, 'body', name=f'arm_flange_{side}',
                              pos=' '.join(map(str, FLANGE_IN_LINK7)))
        # Connector STL coordinates already use the approved hand frame.
        mount = installed[side].copy(); mount[:3, 3] -= FLANGE_IN_LINK7
        fixed = ET.SubElement(flange, 'body', name=f'arm_connector_{side}', **_xml_pose(mount))
        fixed.append(hands[side])
        visual_frame = ET.SubElement(fixed, 'body', name=f'arm_connector_visual_{side}',
                                    **_xml_pose(offset_transform(connector['offsets'][side])))
        for i, name in enumerate(connector['meshes'][side]):
            mesh_name = f'arm_connector_{side}_{i}'
            asset_name = mesh_name + '.stl'
            assets[asset_name] = (ROOT / name).read_bytes()
            ET.SubElement(xml.find('asset'), 'mesh', name=mesh_name, file=asset_name)
            ET.SubElement(visual_frame, 'geom', name=mesh_name, type='mesh', mesh=mesh_name,
                          contype='0', conaffinity='0', density='0', group='1', rgba='.85 .60 .22 1')
    # MuJoCo omits two DOF-less roots from dynamic broadphase. Explicit pairs
    # retain the original cross-hand root/palm detection without inventing an
    # arm DOF or detaching the hand. These kinematic contacts transmit no force.
    contact = xml.find('contact')
    if contact is None:
        contact = ET.SubElement(xml, 'contact')
    existing = {frozenset((p.get('geom1'), p.get('geom2'))) for p in contact.findall('pair')}
    for right in rigid_geoms['right']:
        for left in rigid_geoms['left']:
            if not (old.geom_contype[right] & old.geom_conaffinity[left]
                    or old.geom_contype[left] & old.geom_conaffinity[right]):
                continue
            names = [old.geom(right).name, old.geom(left).name]
            if frozenset(names) not in existing:
                ET.SubElement(contact, 'pair', name=f'rigid_cross_hand_{right}_{left}',
                    geom1=names[0], geom2=names[1],
                    margin=str(max(old.geom_margin[right], old.geom_margin[left])),
                    gap=str(max(old.geom_gap[right], old.geom_gap[left])))
    new = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'), assets)
    restored_parent_pairs = _restore_parent_collision_filter(old, new, contact)
    if restored_parent_pairs:
        new = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'), assets)
    # Build this before preserving compiled physical arrays so the query has
    # exactly the same body/geom layout and mocap addresses as the force model.
    arm_query, arm_query_pairs = _make_arm_collision_query(xml, new, assets)
    _preserve_physical_parameters(old, new, {w['body_name'] for w in wrists.values()})
    _preserve_physical_parameters(old, arm_query, {w['body_name'] for w in wrists.values()})
    for prefix in ('origin', 'reset'):
        for field in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl'):
            attr = prefix + '_' + field
            if not hasattr(bundle, attr):
                continue
            before = getattr(bundle, attr)
            actuator = field == 'ctrl'
            after = np.zeros(new.nu if actuator else (new.nq if field == 'qpos' else new.nv))
            for i in range(new.nu if actuator else new.njnt):
                oldid = old.actuator(new.actuator(i).name).id if actuator else old.joint(new.joint(i).name).id
                addresses = 'jnt_qposadr' if field == 'qpos' else 'jnt_dofadr'
                ni = i if actuator else int(getattr(new, addresses)[i])
                oi = oldid if actuator else int(getattr(old, addresses)[oldid])
                after[ni] = before[oi]
            setattr(bundle, attr, after)
    bundle.model = new
    # Cartesian policy outputs are the six translation slots plus the 40 hand
    # targets.  The two pitch actions are structurally removed from the model
    # and held at their installed reset orientation.
    policy_indices = [i for i in indices if i not in orientation_action_indices]
    if sorted(policy_indices) != sorted(cartesian_policy_indices +
                                       [i for i in indices if '_Palm_Base_' not in bundle.manifest['action_names'][i]]):
        raise ValueError('Cartesian policy action mapping lost a hand or translation channel')
    bundle.policy_action_indices = np.asarray(policy_indices)
    bundle.active_action_indices = np.asarray([i for i in policy_indices
                                                if '_Palm_Base_' not in bundle.manifest['action_names'][i]])
    bundle.action_ids = bundle.ids(mujoco.mjtObj.mjOBJ_ACTUATOR,
        [bundle.manifest['action_names'][i] for i in bundle.active_action_indices])
    bundle.ctrl_ids = np.arange(new.nu); bundle.joint_ids = np.arange(new.njnt)
    bundle.qadr = new.jnt_qposadr.copy(); bundle.vadr = new.jnt_dofadr.copy()
    bundle.key_ids = bundle.ids(mujoco.mjtObj.mjOBJ_JOINT, bundle.manifest['piano_joints'])
    bundle.key_qadr = new.jnt_qposadr[bundle.key_ids]
    for side, w in wrists.items():
        w['body_id'] = int(new.body(w['body_name']).id)
        w['flange_id'] = int(new.body(f'arm_flange_{side}').id)
        w['visual_ids'] = [int(new.body(n).mocapid[0]) for n in arm_names[side]]
    piano = old.body('piano/').id
    bundle.arm_follow_spec = dict(options=options, wrists=wrists, initial_poses=initial_poses,
        piano_pose=(initial.xpos[piano].copy(), initial.xmat[piano].reshape(3, 3).copy()),
        parent_collision_filter=dict(source='hand_only_native_parent_filter',
            source_enabled=not bool(old.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT),
            restored_body_pairs=restored_parent_pairs))
    bundle.arm_collision_query_model = arm_query
    bundle.arm_collision_query_pairs = arm_query_pairs
    zero = mujoco.MjData(new)
    follower = SceneArmFollower(bundle, num_envs=1, device='cpu')
    for prefix in ('origin', 'reset'):
        pos, quat = zero.mocap_pos.copy(), zero.mocap_quat.copy()
        follower.publish_numpy(pos, quat)
        setattr(bundle, prefix + '_mocap_pos', pos)
        setattr(bundle, prefix + '_mocap_quat', quat)
    if bundle.preparation is not None:
        from .preparation import _named_pose
        # The hand-only records define the wrist request reference. They no
        # longer describe independently moving physical support bodies.
        bundle.preparation = dict(bundle.preparation,
            method='policy_wrist_request_preparation_v1',
            pose_records='origin/initial are the hand-only wrist request reference',
            actual_rigid_initial=_named_pose(bundle, bundle.reset_cpu()),
            terminal_support_offset_world=None,
            terminal_request_lift_m=0.,
            support_dynamics='wrist request through arm IK and SDK hard limits; single rigid assembly')
    bundle.self_collision_query_model = None
    return bundle


class SceneArmFollower(ArmFollower):
    def __init__(self, bundle, *, num_envs, device):
        spec = bundle.arm_follow_spec
        super().__init__(spec['options'], spec['initial_poses'], num_envs=num_envs,
                         dt=bundle.manifest['physics_dt'], device=device)
        self.wrists = {s: {k: (self.tensor(v) if k in ('axis', 'rest_pos', 'rest_rot') else v)
                           for k, v in w.items()} for s, w in spec['wrists'].items()}
        self.preparation = bundle.preparation_options
        self.contract['request_channels'] = {s: ['TX', 'TY', 'TZ'] for s in self.wrists}
        self.contract['target_sampling'] = 'policy_cartesian_palm_displacement_and_preparation_at_next_physics_substep'
        piano_p, piano_r = spec['piano_pose']
        self.contract['piano_world_pose'] = dict(position_m=piano_p.tolist(), rotation=piano_r.tolist())
        self.contract['base_in_piano'] = {}
        for side, base in self.contract['base'].items():
            self.contract['base_in_piano'][side] = dict(
                position_m=(piano_r.T @ (np.asarray(base['position_m'])-piano_p)).tolist(),
                rotation=(piano_r.T @ np.asarray(base['rotation'])).tolist())
        self.contract['base_coordinate_frame'] = 'world; base_in_piano stores the same fixed bases in the piano frame'
        self.contract['root_physics'] = 'kinematic_arm_rigid_connector_hand_fingers_dynamic'
        self.contract['request_frame'] = 'world'
        self.contract['request_units'] = 'metres_displacement'
        self.contract['position_bounds_m'] = spec['options']['position_bounds_m']
        self.contract['fixed_orientation'] = 'installed_palm_down'
        self.contract['attachment'] = 'Link7 > fixed_flange > fixed_four_hole_connector > fixed_hand_base'
        self.contract['rigid_hand_collision'] = 'explicit original cross-hand root/palm pairs; no kinematic reaction forces'
        self.contract['arm_collision'] = dict(
            mode='rigid_arm_detection_only_v1',
            categories=['arm_arm', 'arm_hand', 'arm_piano'],
            sample_hz=round(1.0 / bundle.manifest['physics_dt']),
            aggregation='unique_geom_pair_per_physics_substep_then_sum_per_control_step',
            force_solver='query_only; arm meshes remain contype=0/conaffinity=0 in the active model',
            pair_count=len(getattr(bundle, 'arm_collision_query_pairs', ())),
        )
        self.contract['parent_collision_filter'] = spec['parent_collision_filter']
        self.contract['flange_in_link7_m'] = FLANGE_IN_LINK7.tolist()
        self.contract['historical_wrist_observation_slots'] = 'fixed_reset_placeholders; actual_full_pose_from_arm_q'

    def requests(self, physical, times):
        result = {}
        for side, w in self.wrists.items():
            displacement = physical[:, w['indices']].double()
            # ROSELIA Cartesian semantics: the policy sends a world-frame
            # palm displacement, while orientation remains the installed
            # palm-down pose.  The A3.2 connector transform is retained.
            p = w['rest_pos'].expand(self.num_envs, 3) + displacement
            r = w['rest_rot'].expand(self.num_envs, 3, 3)
            if self.preparation is not None:
                u = (times.double() / self.preparation['duration']).clamp(0, 1)
                lift = self.preparation['lift'] * (1 - u**3 * (10 - 15*u + 6*u*u))
                p = p.clone(); p[:, 2] += lift
            result[side] = (p, r)
        return result

    def transforms(self):
        result = {}
        for side, a in self.sides.items():
            _, _, _, _, positions, rotations = self.fk(a, a['q'])
            result[side] = (positions, quaternion(rotations))
        return result

    def publish(self, data):
        for side, (positions, rotations) in self.transforms().items():
            w = self.wrists[side]
            data.mocap_pos[:, w['visual_ids']] = positions.float()
            data.mocap_quat[:, w['visual_ids']] = rotations.float()

    def publish_numpy(self, pos, quat):
        for side, (positions, rotations) in self.transforms().items():
            w = self.wrists[side]
            pos[w['visual_ids']] = positions[0].numpy(); quat[w['visual_ids']] = rotations[0].numpy()
