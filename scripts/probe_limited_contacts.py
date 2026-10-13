#!/usr/bin/env python3
"""Detailed GPU contact audit after an explicit controller-change migration.

Rebuild the parent's model/score with current source; require identical compiled
scene/task hashes. Load parent networks via the explicit finetune gate, without
learner updates. This is a labelled new-controller probe, not historical replay.
"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--control-limits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MUJOCO_GL','egl')
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch import checkpoint
    from tsensei_v3_2_pianist.rl_torch.sac import SAC,SACConfig
    from audit_checkpoint_trajectories import worker
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);old=payload['metadata']
    recipe=old['source_runtime'];prep=recipe['preparation'] or {}
    if recipe['external_score'] is not None:raise ValueError('This probe supports registered scores only')
    current=make_recipe(**recipe['pose'],**recipe['model'],**recipe['environment'],
                        preparation_time=prep.get('duration',0.),preparation_lift=prep.get('lift',.02),
                        preparation_mode=prep.get('mode','fixed_v1'))
    scene,task,_=build(current)
    if old['cli'].get('no_arms'):scene,task=without_arms(scene,task,scene.parent/'hand_only_bundle')
    if sha256(scene/'manifest.json')!=old['scene_sha256'] or sha256(task/'task.json')!=old['task_sha256']:
        raise ValueError('Controller probe requires the exact parent physical scene and score')
    (a.output/'source_recipe.json').write_text(json.dumps(current,indent=2)+'\n')
    variants=[(path.stem,canonical(json.loads(path.read_text()))) for path in a.control_limits]
    if len({name for name,_ in variants})!=len(variants):raise ValueError('Unique variant names required')
    for name,limits in variants:
        directory=a.output/name;directory.mkdir()
        options=checkpoint.reward_options(a.checkpoint)
        options.update(control_limits=limits,collision_diagnostics=True)
        env=PianoTaskEnv(scene,task,device='cuda:0',**options)
        expected=deepcopy(old)
        expected.update(control_limits=limits,control_limits_contract=env.scene.control_limits,
                        preparation_contract=env.scene.bundle.preparation)
        config=deepcopy(payload['agent']['config'])
        config['init_temperature']=float(payload['agent']['log_temperature'].exp())
        agent=SAC(env.observation_size,env.action_size,SACConfig(**config),device='cuda:0',
                  compile_learner=False,fused_optimizer=False,action_profile=old['action_profile'])
        checkpoint.load(a.checkpoint,agent,expected,weights_only=True,transfer_mode='finetune',
                        allow_control_change=limits!=old.get('control_limits'))
        record_config=deepcopy(expected);record_config['source_runtime']=current
        provenance=dict(parent=str(a.checkpoint),parent_sha256=sha256(a.checkpoint),
                        transfer_mode='finetune',networks_only=True,allow_control_change=limits!=old.get('control_limits'),
                        control_limits=limits,resolved=env.scene.control_limits,source_recipe=current,
                        note='Explicit new controller; original physical model/score/preparation/network strictly checked; zero learner updates')
        (directory/'probe.json').write_text(json.dumps(provenance,indent=2)+'\n')
        arguments=SimpleNamespace(worker_root=ROOT,run=a.checkpoint.parent,source_archive=None,output=directory,
                                  checkpoints=[a.checkpoint.name],repeats=1,detail_repeats=1,extended=True)
        worker(arguments,prepared_probe=dict(config=record_config,env=env,agent=agent,scene_path=scene,
                                             task_path=task,options=options,provenance=provenance))
        print('Detailed controller probe complete:',name,flush=True)
        del env,agent
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
