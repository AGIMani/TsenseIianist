"""Physical integration and independence checks for the new hand adaptation."""
import importlib.abc
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


class BlockOriginal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {"robopianist", "komopianist", "tsensei_pianist", "tsensei_v3_pianist", "external_never_copy_only_reference"}:
            raise ImportError(f"Original project import forbidden: {fullname}")


sys.meta_path.insert(0, BlockOriginal())
from tsensei_v3_2_pianist import suite
from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as consts


@pytest.fixture(scope="module")
def env():
    environment = suite.load()
    yield environment
    environment.close()


def test_reset_pose_and_world_axes(env):
    env.reset()
    assert env.action_spec().shape == (51,)
    assert env.physics.model.nu == 50
    assert env.physics.data.ncon == 0, "Default pose must be free of contact penetration"
    r = env.physics.bind(env.task.right_hand.fingertip_sites).xpos
    l = env.physics.bind(env.task.left_hand.fingertip_sites).xpos
    np.testing.assert_allclose(r * (1,-1,1), l, atol=1e-8)
    for hand in (env.task.right_hand, env.task.left_hand):
        physics = env.physics
        for name, expected_axis in [("forearm_tx",(0,1,0)),("forearm_ty",(-1,0,0)),("forearm_tz",(0,0,1))]:
            joint = hand.mjcf_model.find("joint", name)
            before = physics.bind(hand.root_body).xpos.copy()
            physics.bind(joint).qpos = physics.bind(joint).qpos[0] + .001
            physics.forward()
            delta = physics.bind(hand.root_body).xpos - before
            np.testing.assert_allclose(delta, np.array(expected_axis)*.001, atol=1e-9)
            physics.bind(joint).qpos = 0
            physics.forward()


@pytest.mark.parametrize('joint_name', ['forearm_yaw', 'forearm_pitch'])
@pytest.mark.parametrize('angle', [-.1, .1])
def test_wrist_yaw_same_direction_and_pitch_opposite(env, joint_name, angle):
    env.reset()
    physics = env.physics
    hands = (env.task.right_hand, env.task.left_hand)
    for hand in hands:
        for name in ('forearm_pitch', 'forearm_yaw'):
            physics.bind(hand.mjcf_model.find('joint', name)).qpos = 0
    physics.forward()
    initial = [physics.bind(hand.root_body).xmat.reshape(3, 3).copy() for hand in hands]
    for hand in hands:
        physics.bind(hand.mjcf_model.find('joint', joint_name)).qpos = angle
    physics.forward()
    rotations = [physics.bind(hand.root_body).xmat.reshape(3, 3) @ before.T
                 for hand, before in zip(hands, initial)]
    assert np.linalg.norm(rotations[0] - np.eye(3)) > .1
    expected_left = rotations[0] if joint_name == 'forearm_yaw' else rotations[0].T
    np.testing.assert_allclose(rotations[1], expected_left, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_urdf_forward_kinematics(env, side):
    """Compare every URDF body against independently accumulated rigid transforms."""
    env.reset()
    hand = getattr(env.task, side + "_hand")
    physics = env.physics
    source = ET.parse(consts.ASSETS / "urdf" / f"TsenseI_{side}.urdf").getroot()
    rng = np.random.RandomState(17)
    values = {}
    for j in source.findall("joint"):
        if j.get("type") == "fixed": continue
        limit = j.find("limit")
        value = rng.uniform(float(limit.get("lower")), float(limit.get("upper")))
        values[j.get("name")] = value
        physics.bind(hand.mjcf_model.find("joint", j.get("name"))).qpos = value
    physics.forward()
    root = np.eye(4)
    root[:3,:3] = physics.bind(hand.root_body).xmat.reshape(3,3)
    root[:3,3] = physics.bind(hand.root_body).xpos
    transforms = {"base_link": root}
    pending = list(source.findall("joint"))
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in transforms]
        assert ready, "Disconnected URDF graph"
        for j in ready:
            origin = j.find("origin")
            local = np.eye(4)
            local[:3,3] = np.fromstring(origin.get("xyz"), sep=" ")
            local[:3,:3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
            if j.get("type") != "fixed":
                axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                local[:3,:3] = local[:3,:3] @ Rotation.from_rotvec(axis / np.linalg.norm(axis) * values[j.get("name")]).as_matrix()
            child = j.find("child").get("link")
            transforms[child] = transforms[j.find("parent").get("link")] @ local
            body = hand.mjcf_model.find("body", child)
            np.testing.assert_allclose(physics.bind(body).xpos, transforms[child][:3,3], atol=1e-8)
            np.testing.assert_allclose(physics.bind(body).xmat.reshape(3,3), transforms[child][:3,:3], atol=1e-8)
            pending.remove(j)


def test_step_rewards_reset_and_action_order(env):
    env.reset()
    right, left = env.task.right_hand, env.task.left_hand
    action = np.r_[right.neutral_control, left.neutral_control, 0.]
    action[0] += .1
    step = env.step(action)
    np.testing.assert_allclose(env.physics.bind(right.actuators).ctrl, action[:25])
    np.testing.assert_allclose(env.physics.bind(left.actuators).ctrl, action[25:50])
    for assigned in (env.task._rh_keys_current, env.task._lh_keys_current):
        for key, finger in assigned:
            if not env.task.piano.activation[key]:
                np.testing.assert_allclose(env.physics.bind(env.task.piano.keys[key].geom[0]).rgba,
                                           (*consts.FINGERTIP_COLORS[finger], 1))
    for _ in range(40):
        assert np.isfinite(step.reward)
        assert all(np.isfinite(value).all() for value in step.observation.values())
        if step.last(): break
        step = env.step(action)
    assert step.last()
    env.reset()
    np.testing.assert_allclose(env.physics.bind(right.actuators).ctrl, right.neutral_control)
    assert env.physics.data.ncon == 0


def test_unfingered_midi_and_randomized_reset():
    environment = suite.load(task_kwargs={"disable_fingering_reward":True, "randomize_hand_positions":True})
    try:
        environment.reset()
        for hand in (environment.task.right_hand, environment.task.left_hand):
            joint = hand.mjcf_model.find("joint", "forearm_tx")
            actuator = hand.actuator_for_joint("forearm_tx")
            np.testing.assert_allclose(environment.physics.bind(joint).qpos, environment.physics.bind(actuator).ctrl)
        step = environment.step(np.r_[environment.task.right_hand.neutral_control,
                                      environment.task.left_hand.neutral_control, 0.])
        assert np.isfinite(step.reward)
    finally:
        environment.close()


def test_provisional_tip_is_on_distal_collision_hull(env):
    """Check the V3.2 landmark against independently compiled collision geometry."""
    from scipy.spatial import ConvexHull
    from tsensei_v3_2_pianist.models.hands.fingertip_geometry import distal_direction
    env.reset()
    model, data = env.physics.model.ptr, env.physics.data.ptr
    for hand in (env.task.right_hand, env.task.left_hand):
        for body, site in zip(hand.fingertip_bodies, hand.fingertip_sites):
            geom = next(g for g in body.geom if g.group == 3)
            gid = model.geom(geom.full_identifier).id
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]].astype(float)
            world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            body_rotation = env.physics.bind(body).xmat.reshape(3, 3)
            local = (world - env.physics.bind(body).xpos) @ body_rotation
            point = np.asarray(site.pos)
            side = hand.hand_side.name.lower()
            root = ET.parse(consts.ASSETS / 'urdf' / f'TsenseI_{side}.urdf').getroot()
            direction = distal_direction(root, body.name)
            np.testing.assert_allclose(point @ direction, (local @ direction).max(), atol=1e-8)
            hull = ConvexHull(local)
            assert np.max(hull.equations[:, :3] @ point + hull.equations[:, 3]) < 1e-8
            np.testing.assert_array_equal(site.rgba, (0, 0, 0, 1))


def test_v32_joint_order_limits_and_inertias(env):
    env.reset()
    for side in ("left", "right"):
        hand = getattr(env.task, side + "_hand")
        source = ET.parse(consts.ASSETS / "urdf" / f"TsenseI_{side}.urdf").getroot()
        joints = [j for j in source.findall("joint") if j.get("type") != "fixed"]
        assert len(joints) == 20
        assert [a.joint.name for a in hand.actuators[:20]] == [j.get("name") for j in joints]
        assert hand.actuators[1].joint.name.endswith("thumb_rotaback_joint2")
        # Irregular joint2 suffix must resolve the saved per-hand pose.
        assert hand.neutral_control[1] == consts.INITIAL_ANGLES_BY_SIDE[side]['thumb_rotaback_joint2']
        for actuator, joint in zip(hand.actuators, joints):
            limit = joint.find("limit")
            np.testing.assert_allclose(actuator.ctrlrange, [float(limit.get("lower")), float(limit.get("upper"))])
            assert float(limit.get("effort")) == 0
            assert float(limit.get("velocity")) == 0
            np.testing.assert_allclose(actuator.forcerange, [-consts.PROVISIONAL_FINGER_EFFORT, consts.PROVISIONAL_FINGER_EFFORT])
        for link in source.findall("link"):
            body = hand.mjcf_model.find("body", link.get("name"))
            np.testing.assert_allclose(env.physics.bind(body).mass, float(link.find("inertial/mass").get("value")))


def test_control_panel_uses_short_actuator_labels(env):
    env.reset()
    model = env.physics.model.ptr
    compiled = [model.actuator(index).name for index in range(model.nu)]
    assert "R_Index_SW" in compiled
    assert "L_Index_SW" in compiled
    assert "R_Palm_Base_Tx" in compiled
    assert "L_Palm_Base_Yaw" in compiled
    assert len(compiled) == len(set(compiled)) == 50
    assert all("/" not in name and "_hand_" not in name and "_joint" not in name
               for name in compiled)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_updated_a32_pinky_chain_and_terminal_site(env, side):
    env.reset()
    hand = getattr(env.task, side + '_hand')
    joints = [a.joint for a in hand.actuators if 'pinky' in a.joint.name]
    assert [j.name for j in joints] == [
        f'{side}_hand_pinky_{part}_joint'
        for part in ('MCP', 'swing', 'PIP', 'DIP')]
    np.testing.assert_allclose(joints[1].range, [-.35, .35])
    # URDF axis encodes direction; MuJoCo normalizes non-unit CAD exports.
    # The current vendor source uses an uppercase SWING name, unlike older exports.
    import json
    source = ET.parse(consts.ASSETS / 'source/left_original.urdf').getroot()
    mapping = json.loads((consts.ASSETS / 'model_derivation.json').read_text())['left_source_name_map']
    swing = next(j for j in source.findall('joint')
                 if mapping.get(j.get('name'), j.get('name')) == 'left_hand_pinky_swing_joint')
    expected_axis = np.fromstring(swing.find('axis').get('xyz'), sep=' ')
    expected_axis /= np.linalg.norm(expected_axis)
    if side == 'right':
        expected_axis = -np.diag([1., -1., 1.]) @ expected_axis
    np.testing.assert_allclose(joints[1].axis, expected_axis, atol=1e-12)
    assert hand.fingertip_bodies[-1].name == f'{side}_hand_pinky_DIP_link'
    np.testing.assert_allclose(hand.neutral_control[16:20], [
        consts.INITIAL_ANGLES_BY_SIDE[side]['pinky_' + part]
        for part in ('MCP', 'swing', 'PIP', 'DIP')])
    assert [a.name for a in hand.actuators[16:20]] == [
        side[0].upper() + '_Pinky_' + name for name in ('MCP', 'SW', 'PIP', 'DIP')]


def test_assembly_exclusion_can_be_disabled():
    environment = suite.load(task_kwargs={"exclude_assembly_collisions": False})
    try:
        environment.reset()
        physics = environment.physics
        penetrations = [c for c in physics.data.contact if c.dist < -1e-5]
        assert penetrations, "Keep the uncorrected V3.2 assembly overlap inspectable"
        for contact in penetrations:
            names = {physics.model.id2name(int(g), "geom") for g in (contact.geom1, contact.geom2)}
            assert any("palm_link" in name for name in names)
            assert any("thumb_rotaback_link2" in name for name in names)
    finally:
        environment.close()


def test_preview_switches_do_not_change_physics_or_piano_sites(env):
    import mujoco
    from tsensei_v3_2_pianist.preview_geometry import PadCollisionWireframe, configure_options
    env.reset()
    before = env.physics.get_state().copy()
    rgba = env.physics.model.geom_rgba.copy()
    wire = PadCollisionWireframe(env.physics, (env.task.right_hand, env.task.left_hand))
    assert len(wire.geom_names) == 10
    model = env.physics.model.ptr
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=1000 + wire.edge_count)
    for show in (False, True):
        options = mujoco.MjvOption()
        configure_options(options, show)
        assert options.sitegroup[4] == 0, "Piano sensor boxes must stay hidden"
        mujoco.mjv_updateScene(model, env.physics.data.ptr, options, None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        site_geoms = [g for g in scene.geoms[:scene.ngeom] if g.objtype == mujoco.mjtObj.mjOBJ_SITE
                      and model.site_group[g.objid] == 5]
        assert len(site_geoms) == (10 if show else 0)
        count = scene.ngeom
        wire.append_to_scene(scene)
        assert scene.ngeom == count + wire.edge_count
    np.testing.assert_array_equal(env.physics.get_state(), before)
    np.testing.assert_array_equal(env.physics.model.geom_rgba, rgba)


def _origin_matrix(element):
    origin = element.find('origin')
    result = np.eye(4)
    if origin is not None:
        result[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        result[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return result


def test_normalized_left_preserves_original_urdf_kinematics_and_inertia(env):
    """Compare compiled bodies/COM/inertia to the untouched vendor URDF."""
    import json
    env.reset()
    source = ET.parse(consts.ASSETS / 'source/left_original.urdf').getroot()
    derivation = json.loads((consts.ASSETS / 'model_derivation.json').read_text())
    correction = np.array(derivation['canonical_from_source_root'])
    name_map = derivation['left_source_name_map']
    hand, physics = env.task.left_hand, env.physics
    rng = np.random.RandomState(32)
    for _ in range(3):
        values = {}
        for joint in source.findall('joint'):
            if joint.get('type') == 'fixed':
                continue
            limits = joint.find('limit')
            value = rng.uniform(float(limits.get('lower')), float(limits.get('upper')))
            values[joint.get('name')] = value
            physics.bind(hand.mjcf_model.find('joint', name_map.get(joint.get('name'), joint.get('name')))).qpos = value
        physics.forward()
        mount = np.eye(4)
        mount[:3, :3] = physics.bind(hand.root_body).xmat.reshape(3, 3)
        mount[:3, 3] = physics.bind(hand.root_body).xpos
        frames = {'base_link': mount @ correction}
        for joint in source.findall('joint'):
            local = _origin_matrix(joint)
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                local[:3, :3] = local[:3, :3] @ Rotation.from_rotvec(axis / np.linalg.norm(axis) * values[joint.get('name')]).as_matrix()
            child = joint.find('child').get('link')
            frames[child] = frames[joint.find('parent').get('link')] @ local
            body = hand.mjcf_model.find('body', name_map.get(child, child))
            np.testing.assert_allclose(physics.bind(body).xpos, frames[child][:3, 3], atol=1e-9)
            np.testing.assert_allclose(physics.bind(body).xmat.reshape(3, 3), frames[child][:3, :3], atol=1e-9)
        for link in source.findall('link'):
            body = hand.mjcf_model.find('body', name_map.get(link.get('name'), link.get('name')))
            bid = physics.model.name2id(body.full_identifier, 'body')
            inertial = link.find('inertial')
            world = frames[link.get('name')] @ _origin_matrix(inertial)
            np.testing.assert_allclose(physics.data.xipos[bid], world[:3, 3], atol=1e-9)
            i = inertial.find('inertia').attrib
            tensor = np.array([[float(i['ixx']), float(i['ixy']), float(i['ixz'])],
                               [float(i['ixy']), float(i['iyy']), float(i['iyz'])],
                               [float(i['ixz']), float(i['iyz']), float(i['izz'])]])
            actual_rotation = physics.data.ximat[bid].reshape(3, 3)
            actual = actual_rotation @ np.diag(physics.model.body_inertia[bid]) @ actual_rotation.T
            np.testing.assert_allclose(actual, world[:3, :3] @ tensor @ world[:3, :3].T, atol=1e-10)


def test_provisional_right_mirrors_motion_com_and_inertia(env):
    env.reset()
    reflection = np.diag([1., -1., 1.])
    left, right, physics = env.task.left_hand, env.task.right_hand, env.physics
    rng = np.random.RandomState(320)
    source = ET.parse(consts.ASSETS / 'urdf/TsenseI_left.urdf').getroot()
    for _ in range(3):
        for joint in source.findall('joint'):
            if joint.get('type') == 'fixed':
                continue
            limits = joint.find('limit')
            value = rng.uniform(float(limits.get('lower')), float(limits.get('upper')))
            name = joint.get('name')
            physics.bind(left.mjcf_model.find('joint', name)).qpos = value
            physics.bind(right.mjcf_model.find('joint', name.replace('left_', 'right_'))).qpos = value
        physics.forward()
        for link in source.findall('link'):
            lname = link.get('name')
            rname = lname.replace('left_', 'right_').replace('Left_', 'Right_')
            lb, rb = left.mjcf_model.find('body', lname), right.mjcf_model.find('body', rname)
            np.testing.assert_allclose(physics.bind(rb).xpos, reflection @ physics.bind(lb).xpos, atol=1e-9)
            ids = [physics.model.name2id(b.full_identifier, 'body') for b in (lb, rb)]
            np.testing.assert_allclose(physics.data.xipos[ids[1]], reflection @ physics.data.xipos[ids[0]], atol=1e-9)
            tensors = []
            for bid in ids:
                rot = physics.data.ximat[bid].reshape(3, 3)
                tensors.append(rot @ np.diag(physics.model.body_inertia[bid]) @ rot.T)
            np.testing.assert_allclose(tensors[1], reflection @ tensors[0] @ reflection, atol=1e-10)


def test_mirrored_mesh_vertices_normals_and_winding():
    import struct
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    for left in (consts.ASSETS / 'meshes_left').glob('*.STL'):
        right = consts.ASSETS / 'meshes_right' / left.name.replace('left_', 'right_').replace('Left_', 'Right_')
        arrays = []
        for path in (left, right):
            data = path.read_bytes()
            count = struct.unpack_from('<I', data, 80)[0]
            assert len(data) == 84 + 50 * count
            arrays.append(np.frombuffer(data, dtype=dtype, offset=84))
        np.testing.assert_array_equal(arrays[1]['vertices'], arrays[0]['vertices'][:, [0, 2, 1], :] * (1, -1, 1))
        np.testing.assert_array_equal(arrays[1]['normal'], arrays[0]['normal'] * (1, -1, 1))
