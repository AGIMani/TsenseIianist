"""Replay ownership, SAC boundaries, checkpoint state and paper preset contracts."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest('Requires training_mjlab/.venv Torch')

from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config, evaluate

torch.set_num_threads(1)


def batch(n=8, device='cpu', start=0):
    x=torch.arange(start,start+n,device=device,dtype=torch.float32)
    return Transition(x[:,None].repeat(1,7),torch.zeros(n,3,device=device),x*.1,
                      torch.ones(n,device=device),x[:,None].repeat(1,7)+.5,
                      x.long()%2==0,x.long()%2!=0)


def agent(device='cpu', stage=1):
    c=SACConfig(actor_hidden=(16,12),critic_hidden=(16,12,10),gamma=.85 if stage==1 else .9,
                actor_lr=3e-4 if stage==1 else 1e-4)
    return SAC(7,3,c,device=device,seed=9)


class TorchSACChecks(unittest.TestCase):
    def test_replay_wrap_oversize_and_owned_snapshots(self):
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            r=Replay(7,3,5,3,device=device,seed=2)
            with self.assertRaises(ValueError):r.sample()
            original=batch(3,device);r.insert(original)
            original.next_state.fill_(-100)
            self.assertEqual(r.data.next_state[0,0].item(),.5)
            sampled=r.sample();saved=sampled.next_state.clone()
            r.insert(batch(4,device,3))
            self.assertEqual(r.pointer,2)
            self.assertEqual(set(r.data.state[:,0].cpu().tolist()),set(range(2,7)))
            r.insert(batch(12,device,7))
            self.assertEqual(r.pointer,4)
            self.assertEqual(r.total_inserted,19)
            self.assertEqual(set(r.data.state[:,0].cpu().tolist()),set(range(14,19)))
            self.assertTrue(torch.equal(sampled.next_state,saved))
            for _ in range(4):
                b=r.sample()
                self.assertTrue(torch.equal(b.next_state,b.state+.5))
                self.assertTrue((b.discount==1).all())
                self.assertTrue((b.terminated|b.truncated).all())
            before=[x.clone() for x in r.data]
            invalid=batch(3,device)._replace(reward=torch.full((3,),float('nan'),device=device))
            with self.assertRaises(ValueError):r.insert(invalid)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,r.data)))
            overflow=batch(3,device)._replace(reward=torch.full((3,),1e300,device=device,dtype=torch.float64))
            with self.assertRaises(ValueError):r.insert(overflow)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,r.data)))
            r.insert(batch(0,device));self.assertEqual(r.total_inserted,19)

    def test_terminal_discount_and_stable_squash(self):
        a=agent()
        with torch.no_grad():
            for p in a.actor.parameters():p.zero_()
            for p in a.target_critic.parameters():p.zero_()
            for q in a.target_critic.qs:q.value.bias.fill_(2)
        b=batch(2)._replace(discount=torch.tensor([1.,0.]))
        info=a.update(b,randomness={'next_noise':torch.zeros(2,3),'actor_noise':torch.zeros(2,3)},diagnostics=True)
        expected=b.reward+.85*b.discount*(2-.2*info['diagnostics']['next_log_prob'])
        torch.testing.assert_close(info['diagnostics']['target'],expected)
        self.assertGreater(info['diagnostics']['target'][0],b.reward[0])
        self.assertEqual(info['diagnostics']['target'][1],b.reward[1])
        with torch.no_grad():a.actor.mean.bias.fill_(80)
        act,lp=a.actor.sample(torch.zeros(2,7),noise=torch.zeros(2,3))
        self.assertTrue(torch.isfinite(lp).all());self.assertTrue((act==1).all())
        self.assertEqual(agent().target_entropy,-1.5)
        self.assertEqual(SAC(7,3,SACConfig(target_entropy=0.),device='cpu').target_entropy,0.)

    def test_checkpoint_optimizer_rng_and_stage2_transfer(self):
        a=agent(); other=agent()
        first=a.update(batch())
        second=other.update(batch(),randomness={k:None for k in
                            ('next_noise','target_masks','critic_masks','actor_noise','actor_masks')})
        for k in first:torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
        meta=dict(scene_sha256='scene',task_sha256='task',stage=1)
        rng=torch.Generator().manual_seed(19)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';checkpoint.save(path,a,meta,rng)
            restored=agent();checkpoint.load(path,restored,meta,exploration_rng=rng)
            torch.testing.assert_close(a.act(batch().state),restored.act(batch().state),rtol=0,atol=0)
            x=a.update(batch());y=restored.update(batch())
            for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
            for name in ('actor','critic','target_critic'):
                for x,y in zip(getattr(a,name).parameters(),getattr(restored,name).parameters()):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
            with self.assertRaises(ValueError):checkpoint.load(path,agent(),{**meta,'task_sha256':'changed'})
            with self.assertRaises(ValueError):checkpoint.load(path,agent(),{**meta,'self_collision_weight':.2})
            self.assertEqual(checkpoint.reward_options(path),{'self_collision_weight':0.})
            with self.assertRaises(ValueError):
                checkpoint.load_agent(path,scene_sha256='scene',task_sha256='task',obs_size=7,action_size=3,
                                      device='cpu',self_collision_weight=.2)
            fresh=agent(stage=2);checkpoint.load(path,fresh,meta,weights_only=True)
            self.assertEqual(len(fresh.actor_optimizer.state),0)
            self.assertEqual(fresh.updates,0)
            self.assertAlmostEqual(fresh.temperature.item(),.2)
            self.assertEqual(fresh.config.gamma,.9)

    def test_evaluation_does_not_consume_learner_rng(self):
        a=agent()
        class Env:
            logic=SimpleNamespace(length=2,key_counts=torch.tensor([[2,0,0]]))
            def reset(self):self.t=0;return torch.zeros(1,7)
            def step(self,action):
                self.t+=1
                return SimpleNamespace(observation=torch.ones(1,7)*self.t,done=torch.tensor([self.t==2]),
                    terminated=torch.tensor([self.t==2]),truncated=torch.tensor([False]),
                    info={'episode_metrics':torch.ones(1,6),'episode_return':torch.tensor([1.]),
                          'episode_steps':torch.tensor([self.t]),'key_frame_micro_f1':torch.ones(1)})
        before=[a.action_rng.get_state().clone(),a.update_rng.get_state().clone()]
        result=evaluate(a,Env(),2)
        self.assertEqual(result['f1'],1.)
        self.assertTrue(torch.equal(before[0],a.action_rng.get_state()))
        self.assertTrue(torch.equal(before[1],a.update_rng.get_state()))

    def test_paper_presets_and_vector_budget(self):
        for stage in (1,2):
            args=parser().parse_args(['--stage',str(stage),'--inspect-only'])
            c=effective_config(args)
            self.assertEqual(c['replay_capacity'],262144)
            self.assertEqual(c['batch_size'],4096)
            self.assertEqual(c['steady_updates_per_transition'],.125)
            self.assertEqual(c['steady_sample_ratio'],512)
            self.assertEqual(c['requested_transitions'],5120000)
            self.assertEqual(c['sac']['actor_hidden'],(256,256,256))
            self.assertEqual(c['sac']['gamma'],.8)
            self.assertEqual(c['sac']['init_temperature'],1.)
            self.assertEqual(SACConfig.paper(stage).actor_hidden,(256,256))
            self.assertEqual(SACConfig.paper(stage).gamma,.85 if stage==1 else .9)
            self.assertEqual(c['warmup_vector_steps'],300 if stage==1 else 0)
        legacy = effective_config(parser().parse_args(['--actor-hidden','256','256',
                                  '--init-temperature','.2','--discount','.85']))
        self.assertEqual(legacy['sac'],vars(SACConfig.paper(1)))
        for values in ({'tau':0},{'gamma':float('nan')},{'actor_lr':0},{'critic_dropout':1}):
            with self.assertRaises(ValueError):SACConfig(**values)
        with self.assertRaises(ValueError):effective_config(parser().parse_args(['--stage','2']))

if __name__=='__main__':unittest.main()
