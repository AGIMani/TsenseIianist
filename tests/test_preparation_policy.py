"""Policy-controlled preparation advances physics, music rewards and replay."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from test_preparation import write_fixture, PREPARATION
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
from tsensei_v3_2_pianist.mjlab_env.preparation import lift_fraction
from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle


POLICY = dict(PREPARATION,mode='policy_v1')


def write_task(scene, directory, prefix=20):
    directory.mkdir()
    b = SceneBundle(scene)
    goals = np.zeros((24,89)); fingers = np.zeros((24,10,88))
    goals[prefix:,10] = 1; fingers[prefix:,1,10] = 1
    np.savez(directory/'score.npz',goal=goals,fingers=fingers)
    fields = ['action','reward','goal','fingering','left_/joints_pos','right_/joints_pos',
              'piano/state','piano/sustain_state']
    manifest = dict(schema_version=1,profile='cpu_compatible_fingered_v1',
        scene_manifest_sha256=sha256(scene/'manifest.json'),
        control_dt=b.manifest['control_dt'],
        files={'score.npz':sha256(directory/'score.npz')},length=24,
        contract={'environment':dict(wrong_press_termination=False,disable_fingering_reward=False,
            disable_forearm_reward=False,n_steps_lookahead=10,frame_stack=1,initial_buffer_time=prefix*.05),
            'observation_size':1180,'observation_fields_before_stack':[{'path':name} for name in fields]},
        observation_base_size=1180,reset_action=b.neutral_action.tolist(),reset_reward=0.,
        fingertip_sites=[side+f'_/tip_{i}' for side in ('right','left') for i in range(5)],
        key_geoms=[f'piano/key_geom_{i}' for i in range(88)],
        hand_joints=[[name for name in b.manifest['joint_names'] if name.startswith(side+'_')]
                     for side in ('right','left')],
        palm_geoms=[[side+'_/palm_collision_0'] for side in ('right','left')],
        # Synthetic fixture: its only named hand collision proxies are these
        # spheres; every fixture geom disables physical contacts.  This does
        # not specify the collision scope of the real A3.2 training model.
        forearm_collision_geoms=[[side+'_/palm_collision_0'] for side in ('right','left')],
        collision_margin=0.,key_tolerance=.05,finger_tolerance=.01,energy_coefficient=.005,
        reward_terms=['key_press_reward','sustain_reward','energy_reward','fingering_reward','forearm_reward'])
    (directory/'task.json').write_text(json.dumps(manifest))


class PolicyPreparationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.scene = Path(cls.tmp.name)/'scene';cls.scene.mkdir()
        cls.task = Path(cls.tmp.name)/'task'
        write_fixture(cls.scene);write_task(cls.scene,cls.task)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def bundle(self):
        return SceneBundle(self.scene,disable_wrist_tz=True,disable_wrist_yaw=True,preparation=POLICY)

    def test_policy_reset_is_raised_initial_not_precomputed_terminal(self):
        bundle = self.bundle()
        self.assertIsNone(bundle.preparation_trace)
        self.assertTrue(bundle.preparation['included_in_replay'])
        self.assertTrue(bundle.preparation['included_in_score'])
        self.assertEqual(bundle.preparation['policy_observation']['shape'],[1])
        self.assertEqual((bundle.model.nq,bundle.model.nu,len(bundle.active_action_indices)+1),(134,46,47))
        np.testing.assert_array_equal(bundle.reset_qpos,bundle.origin_qpos)
        np.testing.assert_array_equal(bundle.reset_ctrl,bundle.origin_ctrl)
        np.testing.assert_allclose(bundle.reset_mocap_pos[:,2],.02)
        for side in ('right','left'):
            self.assertEqual(mujoco.mj_name2id(bundle.model,mujoco.mjtObj.mjOBJ_JOINT,side+'_/palm_base_tz'),-1)

    def test_task_requires_actual_empty_preparation_frames(self):
        bundle = self.bundle()
        self.assertEqual(TaskBundle(self.task,bundle).preparation_steps,20)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'task'
            write_task(self.scene,directory,prefix=19)
            with self.assertRaisesRegex(ValueError,'empty score prefix'):
                TaskBundle(directory,bundle)

    def gpu(self):
        try:
            import torch
            import mjlab  # noqa:F401
        except ImportError:
            self.skipTest('Requires Torch and MJLab')
        if not torch.cuda.is_available():
            self.skipTest('Requires CUDA')
        return torch

    def test_gpu_each_substep_graph_phase_and_subset_reset(self):
        torch = self.gpu()
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        for graph in (False,True):
            with self.subTest(graph=graph):
                scene = PianoScene(self.scene,num_envs=2,disable_wrist_tz=True,disable_wrist_yaw=True,
                                   preparation=POLICY,physics_graph=graph)
                action = scene.policy_action(scene.bundle.neutral_action).expand(2,-1).clone()
                action[:,0] = .8
                self.assertEqual(scene.preparation_steps,20)
                heights=[]
                callback = None if graph else lambda s:heights.append(s.data.mocap_pos[0,0,2].item())
                for _ in range(10):
                    scene.step(action,substep_callback=callback)
                self.assertAlmostEqual(scene.preparation_state()[0,0].item(),.5)
                np.testing.assert_allclose(scene.data.mocap_pos[:,0,2].cpu(),.01,atol=1e-8)
                before={name:getattr(scene.data,name)[1].clone() for name in
                        ('qpos','qvel','ctrl','mocap_pos','mocap_quat','time','qacc_warmstart')}
                scene.reset([0])
                for name,value in before.items():
                    torch.testing.assert_close(getattr(scene.data,name)[1],value,atol=0,rtol=0)
                scene.step(action)
                np.testing.assert_allclose(scene.preparation_state().cpu().numpy().ravel(),[.95,.45],atol=1e-7)
                expected=.02*lift_fraction(np.array([.05,.55]))
                np.testing.assert_allclose(scene.data.mocap_pos[:,0,2].cpu(),expected,atol=1e-8)
                for _ in range(19):
                    scene.step(action)
                np.testing.assert_array_equal(scene.data.mocap_pos[:,:,2].cpu().numpy(),np.zeros((2,2)))
                np.testing.assert_array_equal(scene.preparation_state().cpu().numpy(),np.zeros((2,1)))
                qadr=scene.model.joint('right_/finger_0').qposadr[0]
                self.assertGreater(abs(scene.data.qpos[0,qadr].item()-scene.bundle.origin_qpos[qadr]),.05)
                if not graph:
                    np.testing.assert_allclose(heights,.02*lift_fraction(np.arange(1,101)/200),atol=1e-8)

    def test_policy_transition_phase_reward_and_replay_include_preparation(self):
        torch = self.gpu()
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
        env = PianoTaskEnv(self.scene,self.task,num_envs=2,disable_wrist_tz=True,disable_wrist_yaw=True,
                           preparation=POLICY)
        obs = env.reset()
        self.assertEqual((env.observation_size,obs.shape[-1]),(1181,1181))
        self.assertEqual(env.preparation_steps,20)
        self.assertTrue(torch.all(obs[:,-1]==1))
        action = env.scene.policy_action(env.scene.bundle.neutral_action).expand(2,-1).clone()
        action[:,0]=.8
        result=env.step(action)
        self.assertTrue(result.info['preparation_active'].all())
        self.assertTrue(torch.all(result.info['score_index']==0))
        self.assertFalse(result.info['goal_applied'].any())
        self.assertTrue(torch.allclose(result.observation[:,-1],torch.tensor(.95,device=env.device)))
        self.assertTrue(torch.isfinite(result.reward).all())
        replay=Replay(env.observation_size,env.action_size,16,2,device=env.device)
        replay.insert(Transition(obs,action,result.reward,result.discount,result.observation,result.terminated,result.truncated))
        self.assertEqual(len(replay),2)
        batch=replay.sample()
        self.assertTrue(torch.all(batch.state[:,-1]==1))
        self.assertTrue(torch.allclose(batch.next_state[:,-1],torch.tensor(.95,device=env.device)))
        snapshot=env.snapshot()
        safe,_=env.logic.reward_terms(snapshot,torch.zeros(2,dtype=torch.long,device=env.device))
        snapshot['key_activation'][:,0]=True
        wrong,_=env.logic.reward_terms(snapshot,torch.zeros(2,dtype=torch.long,device=env.device))
        torch.testing.assert_close(safe['key_press_reward']-wrong['key_press_reward'],torch.full((2,),.5,device=env.device))
        for _ in range(19):
            result=env.step(action)
        self.assertTrue(result.info['preparation_active'].all())
        self.assertTrue(torch.all(result.observation[:,-1]==0))
        result=env.step(action)
        self.assertFalse(result.info['preparation_active'].any())
        self.assertTrue(result.info['goal_applied'][:,10].all())


if __name__=='__main__':
    unittest.main()
