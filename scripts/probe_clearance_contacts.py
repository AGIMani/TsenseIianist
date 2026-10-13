#!/usr/bin/env python3
"""Explicit same-actor geometry/reward probes with strict physical score hashes.

Baseline keeps all saved options. Candidate limits default to None, so inherited
song-specific wrist ranges are never silently retained. This is a labelled
networks-only migration, not historical exact-source replay; no learner updates.
"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--finger-clearance-config',type=Path)
    p.add_argument('--reward-config',type=Path)
    p.add_argument('--control-limits',type=Path,help='Candidate only; omitted means no inherited limits')
    p.add_argument('--include-baseline',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--candidate-name',default='candidate')
    p.add_argument('--repeats',type=int,default=3,help='Canonical episodes; use 1 only for an integration spot check')
    p.add_argument('--detail-repeats',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    return p


def variant_options(parent_options, *, clearance=None, reward=None, limits=None, candidate=False):
    options=deepcopy(parent_options)
    options['collision_diagnostics']=True
    if candidate:
        options['control_limits']=deepcopy(limits)
        options['finger_clearance']=deepcopy(clearance)
        if reward is not None:
            options.pop('self_collision_weight',None)
            options['reward_config']=deepcopy(reward)
    return options


def check_physical_contract(scene_hash, task_hash, saved):
    if scene_hash!=saved['scene_sha256'] or task_hash!=saved['task_sha256']:
        raise ValueError('Clearance probe requires identical parent physical scene and score hashes')


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.repeats<1 or a.detail_repeats<1:p.error('Repeat counts must be positive')
    candidate=any((a.finger_clearance_config,a.reward_config,a.control_limits))
    if not candidate and not a.include_baseline:p.error('No baseline or candidate requested')
    if a.candidate_name in ('baseline','.','..') or Path(a.candidate_name).name!=a.candidate_name:
        p.error('Candidate name must be one safe, distinct directory name')
    a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MUJOCO_GL','egl')
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical as controls
    from tsensei_v3_2_pianist.mjlab_env.finger_clearance import canonical as clearances
    from tsensei_v3_2_pianist.mjlab_env.reward_config import canonical as rewards
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch import checkpoint
    from tsensei_v3_2_pianist.rl_torch.sac import SAC,SACConfig
    from audit_checkpoint_trajectories import worker
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);old=payload['metadata']
    parent_options=checkpoint.reward_options(a.checkpoint)
    load_json=lambda path:json.loads(path.read_text())if path else None
    clearance=clearances(load_json(a.finger_clearance_config))
    limits=controls(load_json(a.control_limits))
    reward=rewards(load_json(a.reward_config))if a.reward_config else None
    recipe=old['source_runtime'];prep=recipe.get('preparation') or {}
    current=make_recipe(**recipe['pose'],**recipe['model'],**recipe['environment'],
        preparation_time=prep.get('duration',0.),preparation_lift=prep.get('lift',.02),
        preparation_mode=prep.get('mode','fixed_v1'))
    # Embedded scores remain portable; build() validates their saved bytes/hash.
    current['external_score']=deepcopy(recipe.get('external_score'))
    current['environment']['midi']=recipe['environment']['midi']
    scene,task,_=build(current)
    if old.get('cli',{}).get('no_arms'):scene,task=without_arms(scene,task,scene.parent/'hand_only_bundle')
    check_physical_contract(sha256(scene/'manifest.json'),sha256(task/'task.json'),old)
    (a.output/'source_recipe.json').write_text(json.dumps(current,indent=2)+'\n')
    variants=[]
    if a.include_baseline:variants.append(('baseline',variant_options(parent_options)))
    if candidate:variants.append((a.candidate_name,variant_options(parent_options,clearance=clearance,
        reward=reward,limits=limits,candidate=True)))
    summary=[]
    for name,options in variants:
        directory=a.output/name;directory.mkdir()
        env=PianoTaskEnv(scene,task,device='cuda:0',**options)
        expected=deepcopy(old)
        expected.update(control_limits=getattr(env.scene,'control_limits_options',None),
            control_limits_contract=env.scene.control_limits,
            finger_clearance=getattr(env.scene,'finger_clearance_options',None),
            finger_clearance_contract=getattr(env.scene,'finger_clearance',None),
            reward_config=env.reward_config,preparation_contract=env.scene.bundle.preparation)
        control_changed=(checkpoint.control_contract(old)!=checkpoint.control_contract(expected)
            or checkpoint.clearance_contract(old)!=checkpoint.clearance_contract(expected))
        reward_changed=checkpoint.reward_contract(old)!=checkpoint.reward_contract(expected)
        sac=deepcopy(payload['agent']['config'])
        sac['init_temperature']=float(payload['agent']['log_temperature'].exp())
        agent=SAC(env.observation_size,env.action_size,SACConfig(**sac),device='cuda:0',
            compile_learner=False,fused_optimizer=False,action_profile=old['action_profile'],seed=old.get('cli',{}).get('seed',0))
        checkpoint.load(a.checkpoint,agent,expected,weights_only=True,transfer_mode='finetune',
                        allow_control_change=control_changed,allow_reward_change=reward_changed)
        record=deepcopy(expected);record['source_runtime']=current
        provenance=dict(parent=str(a.checkpoint),parent_sha256=sha256(a.checkpoint),variant=name,
            transfer_mode='finetune',networks_only=True,learner_updates=agent.updates,
            allow_control_change=control_changed,allow_reward_change=reward_changed,
            parent_control_limits=old.get('control_limits'),control_limits=expected['control_limits'],
            control_limits_contract=expected['control_limits_contract'],finger_clearance=expected['finger_clearance'],
            finger_clearance_contract=expected['finger_clearance_contract'],reward_config=env.reward_config,
            canonical_repeats=a.repeats,detail_repeats=a.detail_repeats,
            source_recipe=current,parent_source_sha256=recipe['source_sha256'],
            note='Explicit current-source migration probe; original physical scene/score/preparation/network checked; zero gradient updates')
        (directory/'probe.json').write_text(json.dumps(provenance,indent=2)+'\n')
        arguments=SimpleNamespace(worker_root=ROOT,run=a.checkpoint.parent,source_archive=None,output=directory,
            checkpoints=[a.checkpoint.name],repeats=a.repeats,detail_repeats=a.detail_repeats,extended=True)
        worker(arguments,prepared_probe=dict(config=record,env=env,agent=agent,scene_path=scene,
            task_path=task,options=options,provenance=provenance))
        if agent.updates!=0:raise RuntimeError('A read-only probe unexpectedly updated its learner')
        del env,agent
        gc.collect();torch.cuda.empty_cache()
        for script in ('analyze_motion_contacts.py','analyze_finger_clearance.py'):
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--audit',str(directory)],check=True)
        audit=json.loads((directory/'checkpoint_audit.json').read_text())['checkpoints'][0]
        detail=directory/a.checkpoint.stem/'detail-0'
        contacts=json.loads((detail/'motion_contact_report.json').read_text())['contact_categories']
        diagnostic=json.loads((detail/'control_diagnostics.json').read_text())
        item=dict(name=name,path=str(directory),canonical_repeats=audit['canonical_repeats'],
            music_f1_min=audit['min_music_f1'],music_f1_max=audit['max_music_f1'],
            contact_categories=contacts,control_diagnostics=diagnostic,provenance=provenance)
        summary.append(item)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(name=name,music_f1_min=item['music_f1_min'],
            key_underside_substeps=contacts['key_underside']['substeps'],
            key_side_or_under_substeps=contacts['key_side_or_under']['substeps'],
            non_distal_key_substeps=contacts['non_distal_link_key']['substeps'])),flush=True)


if __name__=='__main__':main()
