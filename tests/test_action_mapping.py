"""Mixed policy output contract, entropy, replay and frozen-physics integration."""
from pathlib import Path
import unittest
import torch
from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds, policy_bounds, shifts
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition

ROOT=Path(__file__).resolve().parents[1]

class MappingChecks(unittest.TestCase):
    def test_direct_joint_contract_has_eight_capped_mcps_and_47_layout(self):
        import json
        rows=json.loads((ROOT/'configs/actions/direct_joint_v1.json').read_text())['channels']
        capped={4,8,12,16,29,33,37,41}
        mechanical=[1.57 if i in capped else row['physical_max'] for i,row in enumerate(rows)]
        low,high=physical_bounds('direct_joint_v1',[row['name'] for row in rows],
                                 [row['physical_min'] for row in rows],mechanical)
        for i,(actual,limit) in enumerate(zip(high,mechanical)):
            self.assertAlmostEqual(actual,1.3 if i in capped else limit)
        policy_low,policy_high=policy_bounds('direct_joint_v1',47)
        self.assertEqual(len(policy_low),47)
        self.assertEqual((policy_low[4],policy_high[4]),(0.,2.))
        self.assertEqual((policy_low[5],policy_high[5]),(-1.,1.))
        self.assertEqual((policy_low[-1],policy_high[-1]),(-1.,1.))

    def test_direct_actor_and_replay_use_explicit_policy_bounds(self):
        agent=SAC(7,47,device='cpu',seed=3,action_profile='direct_joint_v1')
        obs=torch.zeros(8,7)
        action=agent.act(obs)
        low,high=policy_bounds('direct_joint_v1',47)
        self.assertTrue(torch.all(action>=torch.tensor(low)))
        self.assertTrue(torch.all(action<=torch.tensor(high)))
        replay=Replay(7,47,16,8,device='cpu',action_profile='direct_joint_v1')
        item=Transition(obs,action,torch.ones(8),torch.ones(8),obs,
                        torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool))
        replay.insert(item)
        invalid=action.clone();invalid[:,4]=2.01
        with self.assertRaisesRegex(ValueError,'policy bounds'):
            replay.insert(item._replace(action=invalid))

    def test_actor_translation_keeps_log_density_and_legacy_checkpoint(self):
        legacy=SAC(7,51,device='cpu',seed=1)
        mixed=SAC(7,51,device='cpu',seed=1,action_profile='mixed_v1')
        obs=torch.zeros(8,7);noise=torch.linspace(-5,5,8*51).reshape(8,51)
        a,lp=legacy.actor.sample(obs,noise=noise)
        b,lq=mixed.actor.sample(obs,noise=noise)
        torch.testing.assert_close(b,a+torch.tensor(shifts('mixed_v1')))
        torch.testing.assert_close(lp,lq)
        torch.testing.assert_close(mixed.act(obs,deterministic=True),legacy.act(obs,deterministic=True)+torch.tensor(shifts('mixed_v1')))
        state=legacy.state_dict();state.pop('action_profile')
        legacy.load_state_dict(state)
        with self.assertRaisesRegex(ValueError,'action profile'):
            mixed.load_state_dict(state)

    def test_replay_and_learner_use_mixed_actions(self):
        agent=SAC(7,51,device='cpu',seed=2,action_profile='mixed_v1')
        obs=torch.randn(8,7);action=agent.act(obs)
        replay=Replay(7,51,16,8,device='cpu',action_profile='mixed_v1')
        item=Transition(obs,action,torch.ones(8),torch.ones(8),obs+.1,torch.zeros(8,dtype=torch.bool),torch.zeros(8,dtype=torch.bool))
        replay.insert(item);metrics=agent.update(replay.sample())
        self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values()))
        invalid=action.clone();invalid[:,21]=-.1
        with self.assertRaisesRegex(ValueError,'policy bounds'):
            replay.insert(item._replace(action=invalid))
        with self.assertRaises(ValueError):shifts('mixed_v1',3)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_gpu_physical_mapping_and_bounds(self):
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        scene_path, task_path, _ = build(make_recipe(initial_buffer_time=0.,
            task_profile='tsensei_collision_v1', drive_profile='shadow_v1'))
        scene_path, _ = without_arms(scene_path, task_path, scene_path.parent/'hand_only')
        scene=PianoScene(scene_path, action_profile='mixed_v1')
        s=scene.action_shift
        self.assertEqual(s[5].item(),0.)
        self.assertEqual(s[0].item(),1.)
        self.assertEqual(s[21].item(),1.)
        self.assertEqual(s[22].item(),-1.)
        for x,want in [(s-1,scene.low),(s+1,scene.high),(s,(scene.low+scene.high)/2)]:
            torch.testing.assert_close(scene.physical_action(x),want)
        self.assertAlmostEqual(scene.high[23].item(),.3,places=6)
        self.assertAlmostEqual(scene.low[48].item(),0.,places=6)
        reset=scene.tensor(scene.bundle.normalize(scene.bundle.neutral_action))+s
        scene.step(reset[None])
        torch.testing.assert_close(scene.data.ctrl[0,scene.action_ids],scene.tensor(scene.bundle.neutral_action[:-1]),atol=1e-6,rtol=1e-6)
        before=scene.data.qpos.clone()
        invalid=reset.clone();invalid[22]=.1
        with self.assertRaisesRegex(ValueError,'bounds'):scene.step(invalid[None])
        torch.testing.assert_close(before,scene.data.qpos[:])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_direct_gpu_mcp_endpoints_and_other_joint_unchanged(self):
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
        from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
        scene_path,task_path,_=build(make_recipe(initial_buffer_time=0.,task_profile='tsensei_collision_v1',
                                                drive_profile='shadow_v1'))
        scene_path,task_path=without_arms(scene_path,task_path,scene_path.parent/'hand_only')
        scene=PianoScene(scene_path,
                         action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True)
        low=scene.action_low[None]
        high=scene.action_high[None]
        midpoint=(low+high)*.5
        p0,p1,pm=(scene.physical_action(value) for value in (low,high,midpoint))
        for full_index in (4,8,12,16,29,33,37,41):
            self.assertAlmostEqual(p0[0,full_index].item(),0.,places=6)
            self.assertAlmostEqual(pm[0,full_index].item(),.65,places=6)
            self.assertAlmostEqual(p1[0,full_index].item(),1.3,places=6)
        self.assertAlmostEqual(p1[0,2].item(),1.2,places=6)
        self.assertAlmostEqual(p1[0,23].item(),.3,places=6)
        torch.testing.assert_close(scene.policy_action(p0),low)
        torch.testing.assert_close(scene.policy_action(p1),high)

if __name__=='__main__':unittest.main()
