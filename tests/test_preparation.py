"""Physical preparation keeps policy topology and isolated batched resets."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
from tsensei_v3_2_pianist.mjlab_env.preparation import lift_fraction, options


PREPARATION = {'duration': 1., 'lift': .02, 'trajectory': 'quintic_v1'}


def write_fixture(directory):
    """Portable scalar-joint scene with the same wrist/action layout as V3.2."""
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.005', gravity='0 0 0')
    world = ET.SubElement(root, 'worldbody')
    actuators = ET.SubElement(root, 'actuator')
    key_names, action_names, joint_targets = [], [], {}
    hands = {}
    for index in range(88):
        body = ET.SubElement(world, 'body', pos=f'{-2-index*.02} 0 0')
        name = f'piano/key_{index}'
        key_names.append(name)
        ET.SubElement(body, 'joint', name=name, type='hinge', axis='0 1 0',
                      limited='true', range='0 .01', damping='1', stiffness='100')
        ET.SubElement(body, 'geom', name=f'piano/key_geom_{index}', type='sphere', size='.003', mass='.01', contype='0', conaffinity='0')
    for side, y in (('right', .2), ('left', -.2)):
        code = 'R' if side == 'right' else 'L'
        attachment = ET.SubElement(world, 'body', name=side+'_/')
        name = side+'_/base_link'
        body = ET.SubElement(attachment, 'body', name=name, pos=f'.2 {y} .1')
        ET.SubElement(body, 'geom', name=side+'_/palm_collision_0', type='sphere', size='.01', mass='.1', contype='0', conaffinity='0')
        hands[side] = {'body': name}
        wrist = [('tx','slide','1 0 0',-.2,.2,.03),
                 ('ty','slide','0 1 0',0,.15,.04),
                 ('tz','slide','0 0 1',-.025,0,0),
                 ('pitch','hinge','0 1 0',-.2,.2,0),
                 ('yaw','hinge','0 0 1',-.25,.25,.15)]
        for suffix, kind, axis, low, high, value in wrist:
            name = side+'_/forearm_'+suffix
            ET.SubElement(body, 'joint', name=name, type=kind, axis=axis,
                          limited='true', range=f'{low} {high}', damping='1')
            joint_targets[name] = value
        for index in range(20):
            child = ET.SubElement(body, 'body', pos=f'{.001*index} 0 .02')
            name = side+f'_/finger_{index}'
            ET.SubElement(child, 'joint', name=name, type='hinge', range='-.5 .5',
                          limited='true', damping='.1')
            ET.SubElement(child, 'geom', type='sphere', size='.001', mass='.01', contype='0', conaffinity='0')
            if index % 4 == 0:
                ET.SubElement(child,'site',name=side+f'_/tip_{index//4}',size='.001')
            joint_targets[name] = .1
        control_joints = [side+f'_/finger_{index}' for index in range(20)]
        control_joints += [side+'_/forearm_'+w[0] for w in wrist]
        finger_names = [f'{code}_Thumb_{joint}' for joint in ('Rot','Rot2','MCP','PIP')]
        finger_names += [f'{code}_{finger}_{joint}'
                         for finger in ('Index','Mid','Ring','Pinky')
                         for joint in ('MCP','SW','PIP','DIP')]
        wrist_names = [f'{code}_Palm_Base_{joint}' for joint in ('Tx','Ty','Tz','Pitch','Yaw')]
        action_labels = finger_names + wrist_names
        for index, joint in enumerate(control_joints):
            name = action_labels[index]
            low, high = (-.5, .5) if index < 20 else wrist[index-20][3:5]
            ET.SubElement(actuators, 'position', name=name, joint=joint, kp='1',
                          ctrllimited='true', ctrlrange=f'{low} {high}')
            action_names.append(name)
    scene = directory/'scene.xml'
    scene.write_text(ET.tostring(root, encoding='unicode'))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    for joint, value in joint_targets.items():
        data.qpos[model.joint(joint).qposadr[0]] = value
    for index in range(model.nu):
        joint = model.actuator_trnid[index, 0]
        data.ctrl[index] = data.qpos[model.jnt_qposadr[joint]]
    manifest = dict(schema_version=1, files={'scene.xml': sha256(scene)},
        joint_names=[model.joint(i).name for i in range(model.njnt)],
        actuator_names=action_names, action_names=action_names+['sustain'],
        qpos=data.qpos.tolist(), qvel=data.qvel.tolist(), ctrl=data.ctrl.tolist(),
        physics_options={}, exporter_default_options={}, physics_dt=.005, control_dt=.05,
        piano_joints=key_names, key_activation_fraction=1., hands=hands,
        neutral_action=data.ctrl.tolist()+[0.],
        action_min=model.actuator_ctrlrange[:,0].tolist()+[0.],
        action_max=model.actuator_ctrlrange[:,1].tolist()+[1.])
    (directory/'manifest.json').write_text(json.dumps(manifest))


class PreparationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        write_fixture(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_validation_and_smooth_endpoints(self):
        self.assertIsNone(options(None))
        self.assertEqual(options(dict(PREPARATION,mode='fixed_v1')),PREPARATION)
        for config in ({'duration': 0}, {'lift': -.1}, {'duration': float('inf')},
                       {'trajectory': 'linear'}, {'unexpected': True}):
            with self.assertRaises(ValueError):
                options(config)
        fractions = lift_fraction(np.linspace(0, 1, 201))
        self.assertEqual(fractions[0], 1.)
        self.assertEqual(fractions[-1], 0.)
        self.assertTrue(np.all(np.diff(fractions) < 0))
        self.assertLess(abs(fractions[1]-fractions[0]), 2e-6)
        self.assertLess(abs(fractions[-1]-fractions[-2]), 2e-6)

    def test_topology_world_height_and_actual_terminal(self):
        old = SceneBundle(self.directory, disable_wrist_tz=True, disable_wrist_yaw=True)
        bundle = SceneBundle(self.directory, disable_wrist_tz=True, disable_wrist_yaw=True,
                             preparation=PREPARATION)
        model, trace = bundle.model, bundle.preparation_trace
        self.assertEqual((model.nq, model.nv, model.nu, model.nmocap), (134,134,46,2))
        self.assertEqual(len(bundle.active_action_indices)+1, 47)
        self.assertIsNone(old.preparation)
        for side in ('right', 'left'):
            self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,side+'_/forearm_tz'), -1)
            origin = bundle.preparation['origin']['hands'][side]['position']
            initial = bundle.preparation['initial']['hands'][side]['position']
            np.testing.assert_allclose(np.array(initial)-origin, [0,0,.02], atol=1e-15)
        np.testing.assert_allclose(trace['time'], np.arange(201)*.005)
        np.testing.assert_array_equal(trace['qpos'][0], bundle.origin_qpos)
        np.testing.assert_array_equal(trace['qpos'][-1], bundle.reset_qpos)
        np.testing.assert_allclose(trace['hand_positions'][-1,:,2], trace['hand_positions'][0,:,2]-.02, atol=1e-12)
        self.assertFalse(trace['key_activation'].any())
        self.assertLess(bundle.preparation['report']['max_joint_error_rad'], 1e-9)
        self.assertLess(bundle.preparation['report']['max_abs_terminal_velocity'], 1e-9)
        np.testing.assert_allclose(old.reset_cpu().xpos[old.model.body('right_/base_link').id],
                                   bundle.reset_cpu().xpos[model.body('right_/base_link').id])
        json.dumps(bundle.preparation, allow_nan=False)

    def test_preparation_rejects_fractional_physics_steps(self):
        with self.assertRaisesRegex(ValueError, 'whole physics'):
            SceneBundle(self.directory, preparation=dict(PREPARATION, duration=.997))
        with self.assertRaisesRegex(ValueError, 'whole control'):
            SceneBundle(self.directory, preparation=dict(PREPARATION, duration=.025))

    def test_terminal_state_is_not_projected_back_to_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_fixture(directory)
            path = directory/'manifest.json'
            manifest = json.loads(path.read_text())
            index = manifest['joint_names'].index('right_/forearm_tx')
            manifest['qvel'][index] = .1
            path.write_text(json.dumps(manifest))
            bundle = SceneBundle(directory, disable_wrist_tz=True, disable_wrist_yaw=True,
                                 preparation=PREPARATION)
            qadr = bundle.model.joint('right_/forearm_tx').qposadr[0]
            self.assertGreater(abs(bundle.reset_qpos[qadr]-bundle.origin_qpos[qadr]),1e-5)
            self.assertGreater(abs(bundle.reset_qvel[qadr]),1e-5)
            np.testing.assert_array_equal(bundle.reset_qpos,bundle.preparation_trace['qpos'][-1])

    def test_gpu_subset_reset_preserves_other_world(self):
        try:
            import torch
            import mjlab  # noqa: F401
        except ImportError:
            self.skipTest('Requires Torch and MJLab')
        if not torch.cuda.is_available():
            self.skipTest('Requires a CUDA device')
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        scene = PianoScene(self.directory, num_envs=2, disable_wrist_tz=True,
                           disable_wrist_yaw=True, preparation=PREPARATION)
        neutral = scene.policy_action(scene.bundle.neutral_action).expand(2,-1).clone()
        neutral[1,0] = .4
        scene.step(neutral)
        fields = ('qpos','qvel','ctrl','mocap_pos','mocap_quat','qacc_warmstart','time')
        before = {name: getattr(scene.data,name)[1].clone() for name in fields}
        step = scene.control_steps[1].clone()
        scene.reset([0])
        for name in fields:
            torch.testing.assert_close(getattr(scene.data,name)[1], before[name], rtol=0, atol=0)
        torch.testing.assert_close(scene.control_steps[1],step,rtol=0,atol=0)
        torch.testing.assert_close(scene.data.qpos[0], scene.tensor(scene.bundle.reset_qpos), rtol=0, atol=0)
        self.assertEqual(scene.control_steps[0].item(),0)
        self.assertFalse(scene.state()['key_activation'][0].any())


if __name__ == '__main__':
    unittest.main()
