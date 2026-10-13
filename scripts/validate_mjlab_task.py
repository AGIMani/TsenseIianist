#!/usr/bin/env python3
"""CPU task parity, real GPU rollouts, terminal snapshots and independent resets."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=ROOT/'training_mjlab/outputs/scene')
    p.add_argument('--task',type=Path,default=ROOT/'training_mjlab/outputs/task_toy')
    p.add_argument('--num-envs',type=int,choices=(1,16,64),default=16)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    output=a.output or ROOT/f'training_mjlab/outputs/task_acceptance_{a.task.name}_{a.num_envs}'
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'failed','checks':{},'num_envs':a.num_envs}
    start=time.perf_counter()
    try:
        import numpy as np
        import mujoco
        import torch
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        from tsensei_v3_2_pianist.mjlab_env.task_logic import MusicTaskLogic
        from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
        from tsensei_v3_2_pianist.mjlab_env.acceptance_cases import press_action
        env=PianoTaskEnv(a.scene,a.task,num_envs=a.num_envs,device=a.device)
        scene=env.scene;b=env.bundle;m=b.manifest;n=a.num_envs
        report.update(task_sha256=sha256(a.task/'task.json'),scene_sha256=sha256(a.scene/'manifest.json'),
                      observation_size=env.observation_size,gpu=torch.cuda.get_device_name(scene.device))
        def check(name,ok,**values):report['checks'][name]={'passed':bool(ok),**values}
        def tensor(x):return torch.as_tensor(x,device=scene.device)
        fields=('right_qpos','left_qpos','key_state','key_activation','sustain','fingertips','key_targets','power','palm_collision')
        with np.load(a.task/'cpu_reference.npz',allow_pickle=False) as f:
            refs={k:f[k].copy() for k in f.files}
        fixed_obs=fixed_reward=0.
        fixed_terms={k:0. for k in m['reward_terms']}
        # Replay actual CPU states into Torch logic: isolates task errors from physics divergence.
        for case in m['cpu_reference_cases']:
            logic=MusicTaskLogic(b,num_envs=1,device=scene.device)
            def snap(i):
                return {k:tensor(refs[case+'_'+k][i:i+1]).to(torch.bool if k in ('palm_collision','key_activation') else torch.float32) for k in fields}
            obs=logic.reset(tensor([0]),snap(0))
            fixed_obs=max(fixed_obs,float(np.max(np.abs(obs.cpu().numpy()[0]-refs[case+'_observation'][0]))))
            for i in range(1,len(refs[case+'_reward'])):
                act=tensor(refs[case+'_action'][i:i+1]).float()
                physical=scene.low+(act+1)*.5*(scene.high-scene.low)
                out=logic.advance(physical,snap(i))
                fixed_obs=max(fixed_obs,float(np.max(np.abs(out.observation.cpu().numpy()[0]-refs[case+'_observation'][i]))))
                fixed_reward=max(fixed_reward,abs(out.reward.item()-float(refs[case+'_reward'][i])))
                for j,key in enumerate(m['reward_terms']):
                    fixed_terms[key]=max(fixed_terms[key],abs(out.info['reward_terms'][key].item()-float(refs[case+'_terms'][i,j])))
                check(case+f'_fixed_end_{i}',out.done.item()==bool(refs[case+'_last'][i]) and out.discount.item()==float(refs[case+'_discount'][i]))
        check('fixed_state_observation',fixed_obs<2e-5,max_abs_error=fixed_obs,tolerance=2e-5)
        check('fixed_state_rewards',fixed_reward<2e-5 and max(fixed_terms.values())<2e-5,max_abs_error=fixed_reward,terms=fixed_terms,tolerance=2e-5)
        # GPU performs the same actual controls as the frozen original Composer CPU.
        report['rollout_errors']={}
        slide=scene.model.jnt_qposadr[scene.model.jnt_type==mujoco.mjtJoint.mjJNT_SLIDE]
        hinge=scene.model.jnt_qposadr[scene.model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE]
        for case in m['cpu_reference_cases']:
            env.reset()
            max_obs=max_rew=max_q=max_slide=max_hinge=0.
            metric_total=np.zeros(6)
            tp=fp=fn=0
            term_errors={k:0. for k in m['reward_terms']}
            mismatch=0
            for i in range(1,len(refs[case+'_reward'])):
                action=tensor(refs[case+'_action'][i]).float().repeat(n,1)
                out=env.step(action)
                max_obs=max(max_obs,float((out.observation-tensor(refs[case+'_observation'][i]).float()).abs().max().item()))
                max_rew=max(max_rew,float((out.reward-refs[case+'_reward'][i]).abs().max().item()))
                delta=(scene.data.qpos[:]-tensor(refs[case+'_qpos'][i]).float()).abs()
                max_q=max(max_q,float(delta.max().item()))
                max_slide=max(max_slide,float(delta[:,slide].max().item()))
                max_hinge=max(max_hinge,float(delta[:,hinge].max().item()))
                target=b.goal[i-1]
                actual=refs[case+'_key_activation'][i]
                def metrics(goal,actual):
                    hits=int(np.sum(goal & actual));extra=int(np.sum(~goal & actual));miss=int(np.sum(goal & ~actual))
                    return np.array([hits/(hits+extra) if hits+extra else 1.,hits/(hits+miss) if hits+miss else 1.,2*hits/(2*hits+extra+miss) if 2*hits+extra+miss else 1.])
                goal=target[:88].astype(bool)
                metric_total+=np.r_[metrics(goal,actual),metrics(target[-1:].astype(bool),refs[case+'_sustain'][i]>=.5)]
                tp+=int(np.sum(goal & actual));fp+=int(np.sum(~goal & actual));fn+=int(np.sum(goal & ~actual))
                mismatch+=int((out.info['key_activation']!=tensor(refs[case+'_key_activation'][i])).sum().item())
                for j,k in enumerate(m['reward_terms']):
                    term_errors[k]=max(term_errors[k],float((out.info['reward_terms'][k]-refs[case+'_terms'][i,j]).abs().max().item()))
                check(case+f'_rollout_end_{i}',torch.all(out.done==bool(refs[case+'_last'][i])).item() and torch.all(out.discount==float(refs[case+'_discount'][i])).item())
            report['rollout_errors'][case]={'max_observation_error':max_obs,'max_reward_error':max_rew,'term_errors':term_errors,'max_qpos_error_mixed_units':max_q,'max_slide_error_m':max_slide,'max_hinge_error_rad':max_hinge,'key_activation_mismatches':mismatch}
            check(case+'_physical_rollout',max_obs<.1 and max_rew<.05 and mismatch==0 and max_slide<.002 and max_hinge<.05,
                  observation_tolerance=.1,reward_tolerance=.05,slide_tolerance_m=.002,hinge_tolerance_rad=.05)
            check(case+'_metrics_match_cpu',np.allclose(out.info['episode_metrics'].cpu().numpy(),metric_total/i,atol=2e-6)
                  and np.allclose(out.info['key_frame_micro_f1'].cpu().numpy(),2*tp/max(2*tp+fp+fn,1),atol=2e-6))
        # Deliberately independent music progress, controls and physical state.
        env.reset()
        physical=np.tile(scene.bundle.neutral_action,(n,1))
        for i in range(n):
            if i%2:physical[i]=press_action(scene.bundle)
            physical[i,-1]=(i%3)/2
        action=tensor(scene.bundle.normalize(physical)).float()
        previous_observation=env.logic.observation()
        for _ in range(min(3,m['length']-1)):
            previous_observation=env.step(action).observation
        keep=torch.arange(1,n,2,device=scene.device);ids=torch.arange(0,n,2,device=scene.device)
        logic_fields=('index','done','last_action','last_reward','history','metrics_sum','key_counts','returns')
        saved={f:getattr(env.logic,f).clone() for f in logic_fields}
        physics_fields=('qpos','qvel','ctrl','time','qacc_warmstart')
        saved_physics={f:getattr(scene.data,f)[:].clone() for f in physics_fields}
        out_before=previous_observation.clone()
        env.reset(ids)
        check('partial_reset_others_unchanged',all(torch.equal(getattr(env.logic,f)[keep],v[keep]) for f,v in saved.items())
              and all(torch.equal(getattr(scene.data,f)[keep],v[keep]) for f,v in saved_physics.items()))
        check('partial_reset_music_initial',torch.all(env.logic.index[ids]==0).item() and torch.all(env.logic.last_reward[ids]==m['reset_reward']).item()
              and torch.all(env.logic.last_action[ids]==tensor(m['reset_action']).float()).item())
        check('returned_transition_survives_reset',torch.equal(previous_observation,out_before))
        out=env.step(action)
        check('independent_music_indices',torch.all(out.info['score_index'][ids]==0).item() and torch.equal(out.info['score_index'][keep],saved['index'][keep]))
        # Completed-world guard must precede physical mutation. Retain final obs across reset.
        env.reset()
        neutral=tensor(scene.bundle.normalize(scene.bundle.neutral_action)).float().repeat(n,1)
        for _ in range(m['length']):out=env.step(neutral)
        check('complete_score',out.terminated.all().item() and not out.truncated.any().item() and (out.discount==1).all().item())
        final=out.observation.clone();q=scene.data.qpos[:].clone()
        try:env.step(neutral);rejected=False
        except RuntimeError:rejected=True
        check('requires_explicit_reset',rejected and torch.equal(q,scene.data.qpos[:]))
        env.reset()
        check('terminal_observation_owned',torch.equal(out.observation,final))
        # A time limit is an explicit cutoff, distinct from score completion; discount remains 1.
        limit=min(2,max(1,m['length']-1))
        env.logic.max_episode_steps=limit
        for _ in range(limit):out=env.step(neutral)
        cutoff=limit<m['length']
        check('time_limit_truncation',torch.all(out.truncated==cutoff).item()
              and torch.all(out.terminated!=cutoff).item() and (out.discount==1).all().item())
        report['status']='passed' if all(v['passed'] for v in report['checks'].values()) else 'failed'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_s']=time.perf_counter()-start
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"Music task acceptance: {report['status']}; {output/'report.json'}",flush=True)
    if report['status']!='passed':raise SystemExit('Failed checks: '+','.join(k for k,v in report['checks'].items() if not v['passed']))

if __name__=='__main__':main()
