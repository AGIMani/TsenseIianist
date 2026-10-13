#!/usr/bin/env python3
"""Pinned R6 false-press .1 and R7 false-press .25/actor3e-5 refinements.

Both load only the original A networks, with fresh Adam/RNG/alpha and equal new
sample/update budgets. CPU coordinator only; GPU execution uses submit.sh.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import run_twinkle_f1_finetune as base
import run_twinkle_learner_finetune as learner

ROOT, STEPS, ENVS, UPDATES, METRIC = base.ROOT, base.STEPS, base.ENVS, base.UPDATES, base.METRIC
STAGES = (
    dict(label='R6_false010',mode='networks_only',false_press_weight=.1,activation_weight=.5,gamma=.75,actor_lr=.0003,reward_argument='r6_reward_config'),
    dict(label='R7_false025_actor3e5',mode='networks_only',false_press_weight=.25,activation_weight=.5,gamma=.75,actor_lr=.00003,reward_argument='r7_reward_config'),
)
require, read, write, now = base.require, base.read, base.write, base.now
verify_forward = learner.verify_forward


def reward_path(args,stage):
    return getattr(args,stage['reward_argument'])


def expected_config(template,stage,run,reward,parent):
    expected=base.expected_config(template,stage,run,reward,parent)
    expected['sac']['actor_lr']=stage['actor_lr']
    expected['cli']['actor_lr']=stage['actor_lr']
    return expected


def validate_config(config,template,stage,run,reward,parent):
    expected=expected_config(template,stage,run,reward,parent)
    value=config.get('cli',{}).get('finger_coupling_angle_ranges')
    require(isinstance(value,str) and re.fullmatch(
        re.escape(str(ROOT))+r'/logs/hpc/\d+/steps/\d+/source/configs/finger_coupling/angle_ranges_60_80.json',value),
        'Unexpected archive coupling path')
    expected['cli']['finger_coupling_angle_ranges']=value
    differences=[key for key in set(config)|set(expected) if config.get(key)!=expected.get(key)]
    require(not differences,'Unexpected reward-refinement effective-config difference: '+str(sorted(differences)))


def validate_ready_config(config,*args):
    if any(key not in config for key in ('replay_bytes','observation_size','origin','target_entropy')):
        return False
    validate_config(config,*args)
    return True


def command(args,stage,allocation):
    adjusted=SimpleNamespace(**vars(args))
    adjusted.reward_config=reward_path(args,stage)
    adjusted.false025_reward_config=reward_path(args,stage)
    argv=base.command(adjusted,stage,allocation)
    if stage['actor_lr'] != .0003:
        argv.extend(['--actor-lr',str(stage['actor_lr'])])
    return argv


def validate_reward_jsons(args,baseline):
    for stage in STAGES:
        expected=deepcopy(baseline['reward_config'])
        expected['terms']['false_press_penalty']['weight']=stage['false_press_weight']
        require(read(reward_path(args,stage))==expected,'Unexpected additional reward change: '+stage['label'])


def inputs(args):
    for name in ('source_archive','parent','baseline_config','expected_config','r6_reward_config','r7_reward_config'):
        path=getattr(args,name).resolve();setattr(args,name,path)
        base.pinned(path,getattr(args,name+'_sha256'))
    baseline,template=read(args.baseline_config),read(args.expected_config)
    base.source_contract(template,baseline,args.source_archive)
    base.validate_smoke(template,baseline,args.parent,args.parent_sha256)
    validate_reward_jsons(args,baseline)
    require(len(set(args.allocations))==2 and all(a>0 for a in args.allocations),'Need two distinct allocations')
    import torch
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    require(parent['metadata']['vector_step']==18000 and parent['agent']['updates']==2257280,'Expected original A best parent')
    require(json.loads(json.dumps(parent['agent']['config']))==baseline['sac'],'Parent SAC contract differs')
    require(json.loads(json.dumps({k:parent['metadata'].get(k) for k in baseline}))==baseline,'Parent metadata differs from baseline')
    return template


def summarize(run,template,stage,reward,parent):
    import torch
    from tsensei_v3_2_pianist.rl_torch.checkpoint import exact_music_f1_min
    config,summary=read(run/'config.json'),read(run/'summary.json')
    validate_config(config,template,stage,run,reward,parent)
    learner.validate_budget(summary,stage)
    require({p.name for p in run.glob('*.pt')}=={'forward.pt','best.pt','final.pt'},'Unexpected checkpoints')
    evaluations=[r for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()
                 for r in [json.loads(line)] if 'eval' in r]
    require([r['vector_step'] for r in evaluations]==[0,*range(200,4001,200),STEPS],'Missing evaluation steps')
    for row in evaluations:
        require(row['updates']==learner.updates_at(stage,row['vector_step']),'Evaluation new update count mismatch')
        evaluation=row['eval'];_,reasons=exact_music_f1_min(evaluation)
        require(not reasons,'Invalid exact evaluation: '+str(reasons))
        require(evaluation['note_timing_reference_count']==102,'Incomplete original score')
        for key in ('collision_diagnostics_valid','key_contact_diagnostics_valid','substep_key_diagnostics_valid'):
            require(evaluation.get(key) is True,'Invalid diagnostics: '+key)
        for key in ('collision_diagnostics_coverage','collision_substep_coverage',
                    'key_contact_diagnostics_coverage','key_contact_substep_coverage'):
            require(evaluation.get(key)==1.,'Incomplete diagnostics: '+key)
    best=read(run/'best.json')
    require(best['metric']==METRIC and best['score']==exact_music_f1_min(best['evaluation'])[0],'Incorrect best selection')
    require(best.get('best_eligible') is True and best['evaluation']['key_underside_substeps_max']==0,'Best fails physical qualification')
    best_rows=[r for r in evaluations if r['vector_step']==best['vector_step']]
    require(len(best_rows)==1 and best_rows[0]['eval']==best['evaluation'],'Best differs from logged evaluation')
    forward=verify_forward(run,config,stage,parent)
    for name,step,updates in [('best',best['vector_step'],best['updates']),('final',STEPS,UPDATES)]:
        payload=torch.load(run/(name+'.pt'),map_location='cpu',weights_only=True)
        learner.validate_checkpoint(payload,config,stage,step,updates)
        recorded=best['evaluation'] if name=='best' else summary['final_evaluation']
        require(payload['metadata']['last_evaluation_vector_step']==step and payload['metadata']['last_evaluation']==recorded,
                'Checkpoint evaluation metadata differs: '+name)
        if name=='best':
            require(payload['metadata']['evaluation_selection_state']['best']==best,'Best selector differs from best.json')
    final=summary['final_evaluation']
    return dict(run=str(run),steps=STEPS,transitions=STEPS*ENVS,initial_updates=0,new_updates=UPDATES,total_updates=UPDATES,
        seconds=summary['elapsed_s'],best_step=best['vector_step'],best_new_updates=best['updates'],
        best_involves_new_learning=best['updates']>0,best_exact_min=best['score'],
        final_exact_min=exact_music_f1_min(final)[0],final_exact_pooled=final['exact_original_music_micro_f1'],
        final_music_f1_min=final['music_only_f1_min'],forward_acceptance=forward,
        diagnostics_verified=True,checkpoint_metadata_verified=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allocations',type=int,nargs=2,required=True)
    for name in ('source-archive','parent','baseline-config','expected-config','r6-reward-config','r7-reward-config'):
        p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--inspect-only',action='store_true')
    return p


def main():
    args=parser().parse_args();args.output=args.output.resolve();template=inputs(args)
    stages=[dict(stage,status='pending') for stage in STAGES]
    if args.inspect_only:
        print(json.dumps([command(args,stage,args.allocations[i]) for i,stage in enumerate(stages)],indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.detach:
        with (args.output/'coordinator.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*[s for s in sys.argv[1:] if s!='--detach']],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,output=str(args.output))));return
    lock=(args.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.output/'runtime.json';require(not path.exists(),'Existing runtime; use a fresh directory, no implicit retry')
    for allocation in args.allocations:base.verify_allocation(allocation)
    state=dict(status='running',pid=os.getpid(),session_id=os.getsid(0),started_at=now(),updated_at=now(),
        allocations=args.allocations,stages=stages,
        inputs={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        budget=dict(vector_steps=STEPS,transitions=STEPS*ENVS,new_updates=UPDATES),
        forward_note='step200 precedes first update366: original A networks, fresh Adam/alpha .0006, zero new updates.')
    children,logs={},{}
    def interrupted(signum,frame):raise KeyboardInterrupt('Received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);write(path,state)
    try:
        while True:
            for stage in [s for s in stages if s['status']=='running']:
                child=children[stage['label']]
                try:
                    steps=base.numeric_steps(stage['allocation']);require(len(steps)<=1,'Unexpected concurrent workload')
                    stage['owned_step_ids']=sorted(set(stage.get('owned_step_ids',[]))|set(base.owned_steps(stage,steps)))
                    runs=[r for r in args.output.glob(stage['label']+'-*') if (r/'config.json').is_file()]
                    require(len(runs)<=1,'Duplicate run directories');stage['output_dirs']=list(map(str,runs))
                    if runs and not stage.get('config_verified'):
                        try:config=read(runs[0]/'config.json')
                        except json.JSONDecodeError:pass
                        else:stage['config_verified']=validate_ready_config(config,template,stage,runs[0],reward_path(args,stage),args.parent)
                    if runs and stage.get('config_verified') and not stage.get('forward_verified') and (runs[0]/'forward.pt').is_file():
                        stage['forward_acceptance']=verify_forward(runs[0],read(runs[0]/'config.json'),stage,args.parent)
                        stage['forward_verified']=True
                    if child.poll() is not None:
                        stage['exit_code']=child.returncode;require(child.returncode==0,'Submission failed: '+str(child.returncode))
                        require(len(runs)==1 and (runs[0]/'summary.json').is_file(),'Step ended without summary')
                        stage['result']=summarize(runs[0],template,stage,reward_path(args,stage),args.parent)
                        stage.update(status='completed',finished_at=now());logs.pop(stage['label']).close()
                except Exception as error:
                    base.cancel_stage(stage,child);stage.update(status='failed',error=repr(error),finished_at=now())
                    logs.pop(stage['label']).close()
            for stage,allocation in base.dispatchable(stages,args.allocations):
                try:
                    base.verify_allocation(allocation);inputs(args)
                    argv=command(args,stage,allocation);stage.update(status='running',allocation=allocation,command=argv,started_at=now())
                    write(path,state);log=(args.output/(stage['label']+'.submit.log')).open('w');logs[stage['label']]=log
                    children[stage['label']]=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    stage['submit_pid']=children[stage['label']].pid;stage['submit_pgid']=children[stage['label']].pid
                except Exception as error:
                    stage.update(status='failed',error=repr(error),finished_at=now())
                    if stage['label'] in logs:logs.pop(stage['label']).close()
                    break
            active=[s['label'] for s in stages if s['status']=='running'];state.update(updated_at=now(),active_stages=active)
            if not active:
                state['status']='failed' if any(s['status']=='failed' for s in stages) else 'completed';write(path,state);break
            write(path,state);time.sleep(5)
    except BaseException as error:
        for stage in stages:
            if stage['status']=='running' and stage['label'] in children:
                base.cancel_stage(stage,children[stage['label']]);stage.update(status='failed',error=repr(error),finished_at=now())
        state.update(status='failed',error=repr(error),updated_at=now());write(path,state);raise
    finally:
        for log in logs.values():log.close()
    if state['status']=='failed':raise SystemExit(1)


if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    main()
