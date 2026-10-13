#!/usr/bin/env python3
"""Freeze real local JAX SAC updates with externally controlled noise/dropout.

Run with training/.venv. Only stochastic sampling sites are patched, never the
losses, optimizer, update order or network computation. Dropout masks are shared
across the two Qs in this deterministic fixture; production draws are independent.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from functools import partial
import hashlib
import json
from unittest.mock import patch

import flax.linen as nn
from flax.training.train_state import TrainState
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tsensei_v3_2_pianist.rl.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl.specs import EnvironmentSpec
from tsensei_v3_2_pianist.rl.replay import Transition
from tsensei_v3_2_pianist.rl.networks import MLP
from tsensei_v3_2_pianist.rl.distributions import TanhNormal, TanhMultivariateNormalDiag


def mapped(params, critic=False):
    result = {}
    def add(tree, prefix):
        for k, v in tree['MLP_0'].items():
            kind, number = k.rsplit('_', 1)
            for name, value in v.items():
                if kind == 'Dense':
                    target = prefix + f'base.layers.{number}.' + ('weight' if name == 'kernel' else 'bias')
                    result[target] = np.asarray(value).T if name == 'kernel' else np.asarray(value)
                else:
                    target = prefix + f'base.norms.{number}.' + ('weight' if name == 'scale' else 'bias')
                    result[target] = np.asarray(value)
        for head, dest in (('Dense_0', 'value'),) if critic else (('OutputDenseMean', 'mean'), ('OutputDenseLogStd', 'log_std')):
            result[prefix + dest + '.weight'] = np.asarray(tree[head]['kernel']).T
            result[prefix + dest + '.bias'] = np.asarray(tree[head]['bias'])
    if critic:
        tree = params['VmapStateActionValue_0']
        for q in range(2):
            add(jax.tree.map(lambda x: x[q], tree), f'qs.{q}.')
    else:
        add(params, '')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'tests/data/torch_sac_reference.npz')
    args = p.parse_args()
    result = {}
    rng = np.random.default_rng(204)
    actor_hidden, critic_hidden = (16, 12), (16, 12, 10)
    batch_size, obs_size, action_size, steps = 8, 7, 3, 5
    batch = Transition(rng.normal(size=(batch_size, obs_size)).astype('f'),
                       rng.uniform(-.8, .8, size=(batch_size, action_size)).astype('f'),
                       rng.normal(size=batch_size).astype('f'), np.array([0, .25, 1, 1, 0, 1, .5, 1], dtype='f'),
                       rng.normal(size=(batch_size, obs_size)).astype('f'))
    for key, value in zip(Transition._fields, batch): result['batch/' + key] = value
    def record(prefix, params, critic=False):
        for k, v in mapped(params, critic).items():result[prefix + k] = v.copy()
    def learner(prefix, agent):
        for name in ('actor', 'critic', 'target_critic'):
            state = getattr(agent, name)
            record(prefix + name + '/', state.params, name != 'actor')
            if name != 'target_critic':
                for moment in ('mu', 'nu'):
                    record(prefix + name + '/' + moment + '/', getattr(state.opt_state[0], moment), name == 'critic')
        result[prefix + 'log_temperature'] = np.asarray(agent.temp.params['log_temp']).copy()
        for moment in ('mu', 'nu'):
            result[prefix + 'temp/' + moment] = np.asarray(getattr(agent.temp.opt_state[0], moment)['log_temp']).copy()
    for rate in (0., .01):
        case = f'dropout_{rate}/'
        spec = EnvironmentSpec(specs.Array((obs_size,), np.float32), specs.BoundedArray((action_size,), np.float32, -1, 1))
        config = SACConfig(hidden_dims=critic_hidden, critic_dropout_rate=rate, critic_layer_norm=True,
                           critic_lr=.001, temp_lr=.001, init_temperature=.2)
        agent = SAC.initialize(spec, config, seed=22, discount=.85)
        definition = TanhNormal(partial(MLP, hidden_dims=actor_hidden, activation=nn.gelu, activate_final=True), action_size)
        actor_params = definition.init(jax.random.PRNGKey(44), jnp.zeros(obs_size))['params']
        agent = agent.replace(actor=TrainState.create(apply_fn=definition.apply, params=actor_params, tx=optax.adam(.0003)))
        learner(case + 'initial/', agent)
        for step in range(steps):
            prefix = case + f'step{step}/'
            noises = [rng.normal(size=(batch_size, action_size)).astype('f') for _ in range(2)]
            masks = [[(rng.random((batch_size, width)) >= rate).astype('f') for width in critic_hidden] for _ in range(3)]
            # Force some actual dropped activations even in a small 0.01 fixture.
            if rate:
                for stage in masks:
                    for mask in stage:mask[0, 0] = 0
            for role, noise in zip(('next', 'actor'), noises):result[prefix + role + '_noise'] = noise
            for role, group in zip(('target', 'critic', 'actor'), masks):
                for layer, mask in enumerate(group):result[prefix + f'{role}_mask{layer}'] = mask
            counters = dict(noise=0, dropout=0)
            def sample_with_noise(dist, noise):
                z = dist.distribution.loc + dist.distribution.scale_diag * jnp.asarray(noise)
                action, jac = dist.bijector.forward_and_log_det(z)
                return action, dist.distribution.log_prob(z) - jac
            def controlled_sample(dist, *args, **kwargs):
                value = sample_with_noise(dist, noises[counters['noise']])
                counters['noise'] += 1
                return value
            def controlled_dropout(module, inputs, deterministic=None, **kwargs):
                if deterministic:return inputs
                call = counters['dropout'];counters['dropout'] += 1
                return inputs * jnp.asarray(masks[call // len(critic_hidden)][call % len(critic_hidden)]) / (1 - module.rate)
            with jax.disable_jit(), patch.object(TanhMultivariateNormalDiag, 'sample_and_log_prob', controlled_sample), patch.object(nn.Dropout, '__call__', controlled_dropout):
                dist = agent.actor.apply_fn({'params':agent.actor.params}, batch.next_state)
                next_action, next_lp = sample_with_noise(dist, noises[0])
                next_q = agent.target_critic.apply_fn({'params':agent.target_critic.params},batch.next_state,next_action,True).min(0)
                target = batch.reward + .85 * batch.discount * (next_q - agent.temp.apply_fn({'params':agent.temp.params}) * next_lp)
                result[prefix+'target'] = np.asarray(target)
                result[prefix+'next_action'] = np.asarray(next_action)
                result[prefix+'next_log_prob'] = np.asarray(next_lp)
                counters['dropout'] = 0
                old = agent
                agent, critic_info = agent.update_critic(batch)
                dist = agent.actor.apply_fn({'params':agent.actor.params},batch.state)
                action, lp = sample_with_noise(dist,noises[1])
                result[prefix+'actor_action'] = np.asarray(action)
                result[prefix+'actor_log_prob'] = np.asarray(lp)
                agent, actor_info = agent.update_actor(batch)
                agent, temp_info = agent.update_temperature(actor_info['entropy'])
                assert counters['noise'] == 2
                assert counters['dropout'] == (9 if rate else 0), counters
            for key,value in {**critic_info,**actor_info,**temp_info}.items():result[prefix+'metric/'+key]=np.asarray(value)
            learner(prefix,agent)
            for name in ('actor','critic'):
                before = getattr(old,name).opt_state[0].mu
                after = getattr(agent,name).opt_state[0].mu
                grads = jax.tree.map(lambda a,b:(a-.9*b)/.1,after,before)
                record(prefix+name+'/gradient/',grads,name=='critic')
            result[prefix+'temperature_gradient'] = np.asarray((agent.temp.opt_state[0].mu['log_temp']-.9*old.temp.opt_state[0].mu['log_temp'])/.1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**result)
    metadata=dict(obs_size=obs_size,action_size=action_size,actor_hidden=actor_hidden,critic_hidden=critic_hidden,
                  steps=steps,rates=[0.,.01],source='Local JAX SAC update_critic/update_actor/update_temperature with controlled stochastic inputs; no loss replacements',
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*(ROOT/'tsensei_v3_2_pianist/rl').glob('*.py')]},
                  sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Exported {len(result)} numerical arrays: {args.output}')

if __name__=='__main__':main()
