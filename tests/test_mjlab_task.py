"""Analytic rule/ownership tests; run with training_mjlab Python, no GUI required."""
import unittest
from types import SimpleNamespace
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Torch task tests require training_mjlab/.venv')
from tsensei_v3_2_pianist.mjlab_env.task_logic import (
    MusicTaskLogic, gaussian_tolerance, binary_metrics, fingering_substep_reward,
)
from tsensei_v3_2_pianist.mjlab_env.task_bundle import (
    RESET_ACTION_SEMANTICS, _replace_initial_action_observation,
)


def fixture(frames=1, *, history=True, forearm=True):
    goal=np.zeros((4,89),dtype=np.float32)
    goal[0,[10,20]]=1
    goal[2,[30,88]]=1
    goal[3,[30,88]]=1
    fingers=np.zeros((4,10,88),dtype=np.float32)
    fingers[0,1,10]=1;fingers[0,6,20]=1
    fingers[2:,2,30]=1
    fields=[('fingering',10),('goal',2*89),('left_/joints_pos',25),
            ('piano/state',88),('piano/sustain_state',1),('right_/joints_pos',25)]
    if history:fields=[('action',51)]+fields[:-1]+[('reward',1)]+fields[-1:]
    size=sum(n for _,n in fields)
    terms=['key_press_reward','sustain_reward','energy_reward','fingering_reward']
    if forearm:terms+=['forearm_reward']
    return SimpleNamespace(goal=goal,fingers=fingers,manifest={
        'contract':{'environment':{'wrong_press_termination':False,'disable_fingering_reward':False,
                   'disable_forearm_reward':not forearm,'n_steps_lookahead':1,'frame_stack':frames},
                    'observation_fields_before_stack':[{'path':k,'size':n} for k,n in fields],
                    'observation_size':size*frames},
        'observation_base_size':size,'reset_action':[.125]*51,
        'reset_action_semantics':RESET_ACTION_SEMANTICS,'reset_reward':0.,
        'reward_terms':terms,'finger_tolerance':.01,'key_tolerance':.05,'energy_coefficient':.005})


def snapshot(n=2):
    key=torch.zeros(n,88);key[:,[10,20]]=1
    activation=key.bool()
    keys=torch.full((n,10,10),-1,dtype=torch.long);keys[:,:,1]=10;keys[:,:,6]=20
    return {'fingertip_history':torch.zeros(n,10,10,3),
            'key_target_history':torch.zeros(n,10,88,3),
            'exact_finger_key_history':keys, 'key_state_history':key[:,None,:].expand(-1,10,-1),
            'key_activation_history':activation[:,None,:].expand(-1,10,-1),
            'right_qpos':torch.zeros(n,25),'left_qpos':torch.ones(n,25)*.2,
            'key_state':key,'key_activation':activation,'sustain':torch.zeros(n,1),
            'fingertips':torch.zeros(n,10,3),'key_targets':torch.zeros(n,88,3),
            'power':torch.ones(n)*2,'palm_collision':torch.zeros(n,dtype=torch.bool)}


class MusicRulesTest(unittest.TestCase):
    def test_extra_penalty_reaches_reward_observation_and_return(self):
        task=MusicTaskLogic(fixture(),num_envs=2,device='cpu')
        s=snapshot(); task.reset(torch.arange(2),s)
        result=task.advance(torch.zeros(2,51),s,
                            extra_terms={'self_collision_reward':torch.tensor([0.,-.4])})
        torch.testing.assert_close(result.reward,torch.tensor([3.49,3.09]))
        torch.testing.assert_close(task.last_reward,result.reward)
        torch.testing.assert_close(result.info['episode_return'],result.reward)
        names=[x['path'] for x in task.manifest['contract']['observation_fields_before_stack']]
        self.assertIn('reward',names)
        self.assertEqual(result.info['reward_terms']['self_collision_reward'].tolist(),[0.,torch.tensor(-.4).item()])
        task.reset(torch.tensor([1]),s)
        self.assertEqual(task.returns[1].item(),0.)
        self.assertAlmostEqual(task.returns[0].item(),3.49,places=5)

    def test_tolerance_and_metric_boundaries(self):
        x=torch.tensor([0.,.01,.11,-.1])
        np.testing.assert_allclose(gaussian_tolerance(x,.01,.1),[1,1,.1,.1],atol=2e-7)
        metric,_=binary_metrics(torch.tensor([[0,0],[1,0],[1,0]]),torch.tensor([[0,0],[1,1],[0,0]]))
        np.testing.assert_allclose(metric,[[1,1,1],[.5,1,2/3],[1,0,0]],atol=1e-7)

    def test_per_finger_targets_override_only_the_assigned_finger(self):
        s = snapshot(1)
        targets = torch.zeros(1, 10, 88, 3)
        targets[:, 0, 10, 0] = .02
        targets[:, 1, 10, 0] = .35
        s['key_target_history'] = targets
        reward = fingering_substep_reward(s, .01)
        self.assertLess(reward.item(), 1.)
        s['fingertips'].fill_(100.) # endpoint cannot substitute for history
        torch.testing.assert_close(fingering_substep_reward(s,.01), reward)

    def test_known_rewards_and_wrong_keys_do_not_terminate(self):
        task=MusicTaskLogic(fixture(),num_envs=2,device='cpu')
        s=snapshot();task.reset(torch.arange(2),s)
        s['key_activation'][1,70]=True;s['key_state'][1,70]=1
        result=task.advance(torch.zeros(2,51),s)
        np.testing.assert_allclose(result.reward,[3.49,2.49+.6+.4*np.exp(-1.25)],atol=1e-6)
        self.assertFalse(result.done.any())
        self.assertEqual(result.info['wrong_key'].tolist(),[False,True])
        self.assertTrue(torch.equal(result.info['goal_applied'],task.goal[0].expand(2,-1)))
        self.assertEqual(task.index.tolist(),[1,1])
        # Frame1 is silent; no assigned fingers must produce zero, not NaN.
        s['key_state'].zero_();s['key_activation'].zero_()
        s['exact_finger_key_history'].fill_(-1)
        result=task.advance(torch.zeros(2,51),s)
        self.assertEqual(result.info['reward_terms']['fingering_reward'].tolist(),[0,0])
        torch.testing.assert_close(result.info['reward_terms']['key_press_reward'],torch.full((2,),.4))

    def test_partial_reset_and_terminal_memory(self):
        task=MusicTaskLogic(fixture(3),num_envs=2,device='cpu')
        s=snapshot();task.reset(torch.arange(2),s)
        self.assertTrue((task.history[:,:,:2]==0).all())
        np.testing.assert_array_equal(task.history[0,:51,-1],[.125]*51)
        first=task.advance(torch.zeros(2,51),s)
        before=first.observation.clone();saved=task.history[1].clone()
        task.reset(torch.tensor([0]),s)
        self.assertTrue(torch.equal(saved,task.history[1]))
        self.assertTrue(torch.equal(first.observation,before))
        self.assertEqual(task.index.tolist(),[0,1])
        # Feature-major interleaving: last axis contains oldest..newest frame.
        task.reset(torch.arange(2),s)
        task.advance(torch.ones(2,51)*.1,s)
        task.advance(torch.ones(2,51)*.2,s)
        np.testing.assert_allclose(task.observation()[0,:3],[.125,.1,.2],atol=1e-7)
        task.advance(torch.zeros(2,51),s)
        final=task.advance(torch.zeros(2,51),s)
        self.assertTrue(final.terminated.all());self.assertTrue((final.discount==1).all())
        obs=final.observation.clone();task.reset(torch.arange(2),s)
        self.assertTrue(torch.equal(final.observation,obs))

    def test_truncation_and_optional_observation_fields(self):
        task=MusicTaskLogic(fixture(history=False,forearm=False),num_envs=2,device='cpu',max_episode_steps=1)
        s=snapshot();task.reset(torch.arange(2),s)
        result=task.advance(torch.zeros(2,51),s)
        self.assertTrue(result.truncated.all());self.assertFalse(result.terminated.any())
        self.assertEqual(result.observation.shape,(2,327))
        self.assertNotIn('forearm_reward',result.info['reward_terms'])
        self.assertTrue((result.discount==1).all())

    def test_exported_reset_reference_replaces_only_newest_action_frame(self):
        contract=fixture(3).manifest['contract']
        size=fixture(3).manifest['observation_base_size']
        observation=np.arange(size*3,dtype=np.float32)
        neutral=np.linspace(-.2,.3,51,dtype=np.float32)
        corrected=_replace_initial_action_observation(observation,contract,neutral)
        before=observation.reshape(size,3)
        after=corrected.reshape(size,3)
        np.testing.assert_array_equal(after[:51,:2],before[:51,:2])
        np.testing.assert_allclose(after[:51,-1],neutral)
        np.testing.assert_array_equal(after[51:],before[51:])

if __name__=='__main__':unittest.main()
