"""Wrist-command IK with a structural flange/connector/hand rigid assembly.

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


def augment_bundle(bundle, options):
    """Transform a private bundle; historical XML, recordings and archives stay intact."""
    options = canonical(options)
    old = bundle.model
    if 'm6' in bundle.manifest:
        raise ValueError('Rigid arm assembly requires the hand-only source; old m6 bundles need their archive')
    if bundle.preparation_options is not None and bundle.preparation_options.get('mode') != 'policy_v1':
        raise ValueError('Arm follower supports policy preparation or no preparation')
    if not bundle.disable_wrist_tz or 'right_/forearm_yaw' not in bundle.fixed_joint_values:
        raise ValueError('Arm follower requires structurally removed TZ/Yaw command axes')
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
    wrist_ids = {side: [j for j in range(old.njnt)
                 if (old.joint(j).name or '').startswith(side + '_/forearm_')] for side in SIDES}
    for ids in wrist_ids.values():
        reference.qpos[old.jnt_qposadr[ids]] = 0
    mujoco.mj_forward(old, reference)
    wrists, initial_poses, removed = {}, {}, set()
    indices = list(bundle.active_action_indices)
    for side in SIDES:
        body_name = bundle.manifest['hands'][side]['body']
        body, ids = old.body(body_name).id, wrist_ids[side]
        actions = [int(full) for aid, full in zip(bundle.action_ids, indices) if old.actuator_trnid[aid, 0] in ids]
        if len(ids) != 3 or list(old.jnt_type[ids]) != [2, 2, 3]:
            raise ValueError('Expected TX/TY/Pitch wrist request coordinates')
        wrists[side] = dict(indices=actions, axis=old.jnt_axis[ids].copy(), kind=old.jnt_type[ids].tolist(),
            rest_pos=reference.xpos[body].copy(), rest_rot=reference.xmat[body].reshape(3, 3).copy(),
            body_name=body_name)
        initial_poses[side] = (initial.xpos[body].copy(), initial.xmat[body].reshape(3, 3).copy())
        for j in ids:
            removed.add(old.joint(j).name)
            # These are placeholders in the historical joint vector only. The
            # current arm observation adds actual q/v, IK reference and pose errors.
            bundle.fixed_joint_values[old.joint(j).name] = float(bundle.reset_qpos[old.jnt_qposadr[j]])
    bundle.neutral_action[indices] = bundle.reset_ctrl[bundle.action_ids]
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
    _preserve_physical_parameters(old, new, {w['body_name'] for w in wrists.values()})
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
    bundle.policy_action_indices = np.asarray(indices)
    bundle.active_action_indices = np.asarray([i for i in indices if '_Wrist_' not in bundle.manifest['action_names'][i]])
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
        self.contract['request_axes'] = {s: list(w['indices']) for s, w in self.wrists.items()}
        self.contract['target_sampling'] = 'policy_wrist_command_and_preparation_at_next_physics_substep'
        piano_p, piano_r = spec['piano_pose']
        self.contract['piano_world_pose'] = dict(position_m=piano_p.tolist(), rotation=piano_r.tolist())
        self.contract['base_in_piano'] = {}
        for side, base in self.contract['base'].items():
            self.contract['base_in_piano'][side] = dict(
                position_m=(piano_r.T @ (np.asarray(base['position_m'])-piano_p)).tolist(),
                rotation=(piano_r.T @ np.asarray(base['rotation'])).tolist())
        self.contract['base_coordinate_frame'] = 'world; base_in_piano stores the same fixed bases in the piano frame'
        self.contract['root_physics'] = 'kinematic_arm_rigid_connector_hand_fingers_dynamic'
        self.contract['disabled_axes'] = 'no_TZ_or_Yaw_policy_channel; actual_full_pose_is_arm_FK'
        self.contract['attachment'] = 'Link7 > fixed_flange > fixed_four_hole_connector > fixed_hand_base'
        self.contract['rigid_hand_collision'] = 'explicit original cross-hand root/palm pairs; no kinematic reaction forces'
        self.contract['parent_collision_filter'] = spec['parent_collision_filter']
        self.contract['flange_in_link7_m'] = FLANGE_IN_LINK7.tolist()
        self.contract['historical_wrist_observation_slots'] = 'fixed_reset_placeholders; actual_full_pose_from_arm_q'

    def requests(self, physical, times):
        result = {}
        for side, w in self.wrists.items():
            q = physical[:, w['indices']].double()
            p = w['rest_pos'].expand(self.num_envs, 3)
            r = w['rest_rot'].expand(self.num_envs, 3, 3)
            for i, kind in enumerate(w['kind']):
                if kind == 2:
                    p = p + (r @ w['axis'][i, :, None]).squeeze(-1) * q[:, i, None]
                else:
                    r = r @ rot(w['axis'][i], q[:, i])
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
