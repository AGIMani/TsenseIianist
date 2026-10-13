#!/usr/bin/env python3
"""Validate Torch against frozen, controlled updates of the real local JAX SAC."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from tsensei_v3_2_pianist.rl_torch.sac import SAC,SACConfig
from tsensei_v3_2_pianist.rl_torch.replay import Transition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=ROOT/'tests/data/torch_sac_reference.npz')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--compile-learner',action='store_true')
    p.add_argument('--fused-optimizer',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'training_mjlab/outputs/torch_sac_parity.json')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    meta=json.loads(args.reference.with_suffix('.json').read_text())
    assert hashlib.sha256(args.reference.read_bytes()).hexdigest()==meta['sha256']
    with np.load(args.reference,allow_pickle=False) as f:ref={k:f[k] for k in f.files}
    checks=[];errors={};failed=[]
    start=time.perf_counter()
    def tensor(a):return torch.as_tensor(a,device=args.device,dtype=torch.float32)
    def compare(label,actual,key,category,atol=3e-5,rtol=3e-4):
        a=actual.detach().cpu().numpy();b=ref[key]
        error=float(np.max(np.abs(a-b)))
        errors[category]=max(errors.get(category,0.),error)
        ok=np.allclose(a,b,atol=atol,rtol=rtol)
        checks.append(bool(ok))
        if not ok:failed.append(dict(label=label,max_error=error,key=key))
    b=Transition(*(tensor(ref['batch/'+k]) for k in ('state','action','reward','discount','next_state')),
                 torch.ones(8,device=args.device,dtype=torch.bool),torch.zeros(8,device=args.device,dtype=torch.bool))
    for rate in meta['rates']:
        case=f'dropout_{rate}/'
        config=SACConfig(actor_hidden=tuple(meta['actor_hidden']),critic_hidden=tuple(meta['critic_hidden']),critic_dropout=rate)
        agent=SAC(meta['obs_size'],meta['action_size'],config,device=args.device,
                  compile_learner=args.compile_learner,fused_optimizer=args.fused_optimizer)
        with torch.no_grad():
            for name in ('actor','critic','target_critic'):
                getattr(agent,name).load_state_dict({k:tensor(ref[case+'initial/'+name+'/'+k]) for k in getattr(agent,name).state_dict()})
            agent.log_temperature.copy_(tensor(ref[case+'initial/log_temperature']))
        for step in range(meta['steps']):
            prefix=case+f'step{step}/'
            random={role+'_noise':tensor(ref[prefix+role+'_noise']) for role in ('next','actor')}
            for role in ('target','critic','actor'):
                masks=[tensor(ref[prefix+f'{role}_mask{i}']) for i in range(len(config.critic_hidden))]
                random[role+'_masks']=[masks,masks]
            result=agent.update(b,randomness=random,diagnostics=True)
            diagnostic=result.pop('diagnostics')
            for name,value in result.items():compare(name,value,prefix+'metric/'+name,'losses_and_metrics')
            for name in ('target','next_action','next_log_prob','actor_action','actor_log_prob','temperature_gradient'):
                compare(name,diagnostic[name],prefix+name,name)
            for name in ('actor','critic','target_critic'):
                for key,value in getattr(agent,name).named_parameters():
                    compare(name+'/'+key,value,prefix+name+'/'+key,'parameters',atol=1e-5,rtol=1e-4)
                    if name!='target_critic':
                        compare('gradient/'+key,diagnostic[name+'_gradients'][key],prefix+name+'/gradient/'+key,'gradients')
                        state=getattr(agent,name+'_optimizer').state[value]
                        for a,c in [('exp_avg','mu'),('exp_avg_sq','nu')]:
                            compare('optimizer/'+key,state[a],prefix+name+'/'+c+'/'+key,'optimizer_moments')
                        assert state['step'].item()==step+1
            compare('log_temperature',agent.log_temperature,prefix+'log_temperature','temperature_parameter')
            state=agent.temp_optimizer.state[agent.log_temperature]
            for a,c in [('exp_avg','mu'),('exp_avg_sq','nu')]:compare('temp/'+c,state[a],prefix+'temp/'+c,'optimizer_moments')
    report=dict(status='passed' if all(checks) else 'failed',device=args.device,checks=len(checks),failures=failed,
                compile_learner=args.compile_learner,fused_optimizer=args.fused_optimizer,
                max_abs_errors=errors,reference_sha256=meta['sha256'],steps_per_case=meta['steps'],dropout_rates=meta['rates'],
                atol=3e-5,rtol=3e-4,parameter_atol=1e-5,parameter_rtol=1e-4,elapsed_s=time.perf_counter()-start,
                scope='Real JAX SAC losses and Adam under identical explicit noises/dropout; not stochastic RNG equivalence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
