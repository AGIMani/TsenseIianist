"""GPU graph/callback/reset and same-hand contact counting acceptance."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
try:
    import torch
    import mujoco
    import warp as wp
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
    from tsensei_v3_2_pianist.mjlab_env.self_collision import SelfCollisionTracker
except ImportError:
    raise unittest.SkipTest('Requires training_mjlab/.venv Torch, Warp and MJLab')
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tests.test_torch_sac import batch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AccelerationChecks(unittest.TestCase):
    def test_custom_self_penalty_is_capped_once_without_dt(self):
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.mjlab_env.reward_config import template
        if not (ROOT/'training_mjlab/outputs/task_twinkle/task.json').is_file():
            self.skipTest('Export the GPU task first')
        config=template();config['scale_by_dt']=True
        for name,term in config['terms'].items():
            term['enabled']=name=='self_collision_reward'
        env=PianoTaskEnv(ROOT/'training_mjlab/outputs/scene',ROOT/'training_mjlab/outputs/task_twinkle',
                         reward_config=config)
        self.assertEqual(env.self_collision_weight,.2)
        result=env.step(torch.zeros(1,51,device=env.device))
        expected=(-.2*result.info['self_collision_pair_substeps'].float()).clamp_min(-5.)
        torch.testing.assert_close(result.reward,expected)
        self.assertEqual(env.reward_term_names,['self_collision_reward'])

    def test_graph_matches_substeps_and_callback_after_partial_reset(self):
        if not (ROOT/'training_mjlab/outputs/scene/manifest.json').is_file():
            self.skipTest('Export the GPU scene first')
        scene = PianoScene(ROOT/'training_mjlab/outputs/scene', num_envs=2,
                           track_self_collision=True)
        target = np.r_[scene.bundle.reset_ctrl[scene.bundle.action_ids], 0.]
        action = scene.tensor(2 * (target-scene.bundle.low)/(scene.bundle.high-scene.bundle.low)-1)
        action = action[None].repeat(2, 1)
        # Same object/reset: pointers are reused by both child/outer graphs.
        scene.physics_graph = False
        expected = []
        for i in range(3):
            if i == 2: scene.reset([1])
            scene.step(action)
            expected.append((scene.data.qpos[:].clone(), scene.data.qvel[:].clone(),
                             scene.self_collision.count().clone()))
        scene.reset(); scene.physics_graph = True
        for i, (q, v, count) in enumerate(expected):
            if i == 2: scene.reset([1])
            scene.step(action)
            torch.testing.assert_close(scene.data.qpos[:], q, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(scene.data.qvel[:], v, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(scene.self_collision.count(), count)
        self.assertIsNotNone(scene.control_graph)
        calls = []
        scene.step(action, substep_callback=lambda _: calls.append(1))
        self.assertEqual(len(calls), scene.decimation)
        self.assertEqual(scene.control_steps.tolist(), [4, 2])
        # Recreating the MJLab child graph invalidates the outer graph cache.
        old = scene.control_graph
        with scene.scope(): scene.sim.create_graph()
        scene.step(action)
        self.assertIsNot(scene.control_graph, old)

    def test_distinct_pairs_worlds_distance_and_reset(self):
        bodies = ''.join(f'<body name="{name}" pos="{i} 0 0"><joint type="hinge"/><geom type="sphere" size=".01"/></body>'
                         for i, name in enumerate(['right_/a','right_/b','right_/c','left_/a','left_/b','piano/key']))
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies+'</worldbody></mujoco>')
        wp.init()
        tracker = SelfCollisionTracker(model, 2, torch.device('cuda:0'))
        # right a/b duplicate/reversed; right a/c second pair; left pair;
        # cross-hand, hand/key, same-body, positive-gap and padding ignored.
        pairs = np.array([[0,1],[1,0],[0,2],[3,4],[0,3],[0,5],[0,0],[1,2],[0,1],[3,4]], np.int32)
        distances = np.array([-.001]*7+[.001,0.,-.001],np.float32)
        worlds = np.array([0]*8+[1,0],np.int32)
        stream = wp.get_stream('cuda:0')
        caller = torch.cuda.current_stream(); ts = wp.stream_to_torch(stream)
        ts.wait_stream(caller)
        with torch.cuda.stream(ts), wp.ScopedStream(stream):
            data = SimpleNamespace(naconmax=10,nacon=wp.array([9],dtype=wp.int32,device='cuda:0'),
                contact=SimpleNamespace(geom=wp.array(pairs,dtype=wp.vec2i,device='cuda:0'),
                    dist=wp.array(distances,device='cuda:0'),worldid=wp.array(worlds,device='cuda:0')))
            tracker.sample(SimpleNamespace(wp_data=data,wp_device=wp.get_device('cuda:0')))
        caller.wait_stream(ts)
        self.assertEqual(tracker.count().tolist(), [3,1])
        tracker.seen[1] = 0
        self.assertEqual(tracker.count().tolist(), [3,0])

    def test_fused_checkpoint_can_switch_runtime_modes(self):
        c = SACConfig(actor_hidden=(16,12), critic_hidden=(16,12,10))
        eager = SAC(7,3,c,device='cuda:0',seed=9)
        eager.update(batch(device='cuda:0'))
        fused = SAC(7,3,c,device='cuda:0',seed=9,fused_optimizer=True)
        fused.load_state_dict(deepcopy(eager.state_dict()))
        for opt in (fused.actor_optimizer,fused.critic_optimizer,fused.temp_optimizer):
            self.assertTrue(opt.param_groups[0]['fused'])
            self.assertTrue(all(s['step'].is_cuda for s in opt.state.values()))
        x = eager.update(batch(device='cuda:0')); y = fused.update(batch(device='cuda:0'))
        for k in x: torch.testing.assert_close(x[k], y[k], atol=3e-5, rtol=3e-4)
        restored = SAC(7,3,c,device='cuda:0')
        restored.load_state_dict(deepcopy(fused.state_dict()))
        self.assertFalse(restored.actor_optimizer.param_groups[0]['fused'])
        restored.update(batch(device='cuda:0'))

    def test_explicit_inter_hand_case_categories_and_ignored_pair(self):
        names=['right_/a','right_/b','left_/a','left_/b','piano/base','piano/white_key_0']
        bodies=''.join(f'<body name="{n}" pos="{i} 0 0"><joint/><geom type="sphere" size=".01"/></body>'
                       for i,n in enumerate(names))
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies+'</worldbody></mujoco>')
        wp.init()
        tracker=SelfCollisionTracker(model,2,torch.device('cuda:0'),
            categories=('self','inter_hand','piano_case'),ignored_body_pairs=[['right_/a','right_/b']])
        stream=wp.get_stream('cuda:0');ts=wp.stream_to_torch(stream)
        caller=torch.cuda.current_stream();ts.wait_stream(caller)
        with torch.cuda.stream(ts),wp.ScopedStream(stream):
            data=SimpleNamespace(naconmax=7,nacon=wp.array([7],dtype=wp.int32,device='cuda:0'),
                contact=SimpleNamespace(
                    geom=wp.array(np.array([[0,1],[1,0],[0,2],[0,4],[0,5],[2,3],[0,2]],np.int32),dtype=wp.vec2i,device='cuda:0'),
                    dist=wp.array(np.full(7,-.001,np.float32),device='cuda:0'),
                    worldid=wp.array(np.array([0]*6+[1],np.int32),device='cuda:0')))
            tracker.sample(SimpleNamespace(wp_data=data,wp_device=wp.get_device('cuda:0')))
        caller.wait_stream(ts)
        self.assertEqual(tracker.count().tolist(),[3,1])
        self.assertEqual(tracker.count(('self',)).tolist(),[1,0])
        self.assertEqual(tracker.count(('inter_hand',)).tolist(),[1,1])
        self.assertEqual(tracker.count(('piano_case',)).tolist(),[1,0])

if __name__ == '__main__': unittest.main()
