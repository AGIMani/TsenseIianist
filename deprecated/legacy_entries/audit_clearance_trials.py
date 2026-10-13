#!/usr/bin/env python3
"""Read-only CPU audit of short clearance training contracts and results.

No package import, environment construction, learner execution or GPU access.
May run while training: a partial trailing metrics line is ignored explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import torch

ROOT=Path(__file__).resolve().parents[1]


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):digest.update(data)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',type=Path,action='append')
    a=p.parse_args();archive=a.source_archive.resolve()
    archived={};all_files=0
    with tarfile.open(archive) as handle:
        for member in handle:
            if not member.isfile():continue
            all_files+=1;name=member.name.removeprefix('./');parts=Path(name).parts
            if parts[0]!='tsensei_v3_2_pianist' or '__pycache__' in parts or Path(name).suffix in ('.pyc','.pyo'):continue
            if name in archived:raise ValueError('Duplicate archived source: '+name)
            archived[name]=hashlib.sha256(handle.extractfile(member).read()).hexdigest()
    runs=a.run or sorted((ROOT/'training_mjlab/outputs').glob('clearance_trial_*_20260927-*'))
    result=dict(source_archive=str(archive),archive_sha256=sha256(archive),archive_file_count=all_files,
                archived_package_source_files=len(archived),runs=[])
    for run in runs:
        run=run.resolve();cp=run/'config.json'
        if not cp.exists():continue
        try:c=json.loads(cp.read_text())
        except json.JSONDecodeError:c={}
        if 'origin' not in c:
            result['runs'].append(dict(run=str(run),status='awaiting_resolved_config',checks={},
                latest_logged_vector_step=None,best=None,checkpoints={}))
            continue
        origin=c['origin'];parent_path=Path(origin['checkpoint'])
        parent=torch.load(parent_path,map_location='cpu',weights_only=True)
        pm=parent['metadata'];ps=parent['agent'];expected=c['source_runtime']['source_sha256']
        networks_only=origin['transfer_mode']=='finetune_networks_only'
        warm_restart=origin['transfer_mode']=='learner_warm_restart'
        initial_updates=0 if networks_only else ps['updates']
        if expected!=archived:
            delta=sorted(k for k in expected.keys()|archived.keys() if expected.get(k)!=archived.get(k))
            raise ValueError('Archive source mismatch: '+', '.join(delta[:10]))
        current={k:sha256(ROOT/k) for k in expected if (ROOT/k).is_file()}
        parent_alpha=float(ps['log_temperature'].exp())
        constraints=c['control_limits_contract']['channels']
        reward=c['reward_config']['terms']['humanoid_finger_gesture_reward']
        checks=dict(archive_exact=True,current_package_matches_archive=current==expected,
            parent_checkpoint_sha256_matches=sha256(parent_path)==origin['checkpoint_sha256'],
            scene_same_as_parent=c['scene_sha256']==pm['scene_sha256'],
            task_same_as_parent=c['task_sha256']==pm['task_sha256'],
            preparation_same_as_parent=c['preparation']==pm['preparation'],
            full_preparation_contract_same=c['preparation_contract']==pm['preparation_contract'],
            observation_shape_same=c['observation_size']==ps['obs_size']==1181,
            action_shape_same=c['action_size']==ps['action_size']==47,
            action_contract_same=c['action_contract']==pm['action_contract'],
            alpha_config_matches_transfer=(c['sac']['init_temperature']==(parent_alpha if networks_only else ps['config']['init_temperature'])),
            no_canonical_wrist_position_clip=all(not {'min','max'}&set(v) for v in c['control_limits']['channels'].values()),
            resolved_wrist_ranges_are_original=all(v['min']==v['original_min'] and v['max']==v['original_max'] for v in constraints.values()),
            active_contact_enabled=c['finger_clearance']['active_command_mode']=='active_contact_v1',
            clearance_options_match_resolved=c['finger_clearance']==c['finger_clearance_contract']['options'],
            geometry_reward_enabled=reward['enabled'] and reward['params']['mode']=='geometry_clearance_v1',
            key_underside_best_and_stop_threshold_zero=c['evaluation_selection']['max_key_underside_substeps']==0,
            key_contact_observer_present=bool(c['key_contact_diagnostics_contract']),
            recognized_explicit_transfer=networks_only or warm_restart,
            optimizer_rng_reset_matches_transfer=origin['fresh_optimizer_alpha_rng']==networks_only,
            fresh_replay=origin['fresh_replay'],
            explicit_changes_match_transfer=(c['cli']['allow_control_change'] and c['cli']['allow_reward_change'] if networks_only else
                (not c['cli']['allow_control_change'] and not c['cli']['allow_reward_change'] and
                 all(c[k]==pm[k] for k in ('finger_clearance','finger_clearance_contract','control_limits','control_limits_contract')) and
                 json.dumps(c['reward_config'],sort_keys=True)==json.dumps(pm['reward_config'],sort_keys=True))),
            no_random_exploration=c['warmup_transitions']==0,
            full_song_prefill=c['learning_start_transition_1_based']==512*178+1)
        rows=[];skipped_partial=False
        raw=(run/'metrics.jsonl').read_text() if (run/'metrics.jsonl').exists() else ''
        for i,line in enumerate(raw.splitlines()):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i==len(raw.splitlines())-1 and not raw.endswith('\n'):skipped_partial=True
                else:raise
        evaluations=[]
        for row in rows:
            if 'eval' not in row:continue
            e=row['eval'];s=row['checkpoint_selection']
            evaluations.append(dict(vector_step=row['vector_step'],updates=row['updates'],
                music_f1_min=e.get('music_only_f1_min'),music_micro_f1_min=e.get('music_only_micro_f1_min'),
                prep_wrong_frames=e.get('preparation_wrong_key_frames_max'),
                hand_collision_substeps=e.get('hand_collision_substeps_max'),
                key_underside_substeps=e.get('key_underside_substeps_max'),
                key_side_or_under_substeps=e.get('key_side_or_under_substeps_max'),
                key_non_distal_substeps=e.get('key_non_distal_substeps_max'),
                key_coverage=e.get('key_contact_substep_coverage'),
                guard_projection_frames=e.get('finger_clearance_projection_frames_max'),
                guard_actual_max_deficit_m=e.get('finger_clearance_actual_max_deficit_m_max'),
                selection=s))
        start=c['learning_start_transition_1_based'];ratio=c['steady_updates_per_transition']
        latest=rows[-1] if rows else None
        expected_updates=lambda step:initial_updates+int(max(0,step*c['num_envs']-start+1)*ratio)
        # Episode completions are emitted after collection/reset but before the
        # current vector's learner block; train/eval rows are emitted after it.
        checks['logged_update_budget_matches']=all(r['updates']==expected_updates(
            r['vector_step']-1 if 'episodes' in r and 'train' not in r and 'eval' not in r else r['vector_step']) for r in rows)
        payloads={}
        for filename in ('best.pt','final.pt'):
            path=run/filename
            if not path.exists():continue
            saved=torch.load(path,map_location='cpu',weights_only=True);m=saved['metadata'];network=saved['agent']
            runtime={'vector_step','updates','evaluation_selection_state','last_evaluation','last_evaluation_vector_step'}
            config_keys_match=(all(k in m and json.dumps(m[k],sort_keys=True)==json.dumps(v,sort_keys=True)
                                   for k,v in c.items()) and not set(m)-set(c)-runtime)
            row=dict(checkpoint_sha256=sha256(path),vector_step=m['vector_step'],updates=m['updates'],alpha=float(network['log_temperature'].exp()),
                     full_config_metadata_matches=config_keys_match,source_archive_matches=m['source_runtime']['source_sha256']==archived,
                     optimizer_state_counts={k:len(network[k]['state']) for k in ('actor_optimizer','critic_optimizer','temp_optimizer')},
                     baseline_network_tensor_equality=None)
            if m['vector_step']==0:
                row['baseline_network_tensor_equality']={name:all(torch.equal(t,ps[name][k]) for k,t in network[name].items())
                                                       for name in ('actor','critic','target_critic')}
                row['baseline_alpha_matches_parent']=float(network['log_temperature'].exp())==parent_alpha
                row['baseline_updates_match_transfer']=network['updates']==initial_updates
                if warm_restart:
                    row['baseline_full_optimizer_equality']={name:equal_state(network[name],ps[name]) for name in
                        ('actor_optimizer','critic_optimizer','temp_optimizer')}
                    row['baseline_optimizer_moment_step_equality']={name:equal_state(network[name]['state'],ps[name]['state']) for name in
                        ('actor_optimizer','critic_optimizer','temp_optimizer')}
                    row['baseline_optimizer_hyperparameters_equality']={name:equal_state(
                        [{k:v for k,v in group.items() if k!='capturable'} for group in network[name]['param_groups']],
                        [{k:v for k,v in group.items() if k!='capturable'} for group in ps[name]['param_groups']]) for name in
                        ('actor_optimizer','critic_optimizer','temp_optimizer')}
                    row['optimizer_execution_note']='SAC.load_state_dict resets capturable=False; CUDA graph setup re-enables it. Step/m/v and all learning hyperparameters are compared separately.'
                    row['baseline_rng_equality']={name:equal_state(network[name],ps[name]) for name in ('action_rng','update_rng')}
                    row['baseline_rng_equality']['exploration_rng']=equal_state(saved['exploration_rng'],parent['exploration_rng'])
                snapshot=a.output.parent/f'{run.name}_initial_migration.json'
                if not snapshot.exists() or json.loads(snapshot.read_text())['checkpoint']['checkpoint_sha256']==row['checkpoint_sha256']:
                    snapshot.parent.mkdir(parents=True,exist_ok=True)
                    snapshot.write_text(json.dumps(dict(run=str(run),parent=str(parent_path),transfer=origin['transfer_mode'],
                        parent_sha256=origin['checkpoint_sha256'],checkpoint=row),indent=2)+'\n')
            payloads[filename]=row
        checks['saved_checkpoint_metadata_match']=all(v['full_config_metadata_matches'] and v['source_archive_matches'] for v in payloads.values())
        baseline=[v for v in payloads.values() if v['vector_step']==0]
        if baseline:
            checks['baseline_networks_alpha_updates_verified']=all(all(v['baseline_network_tensor_equality'].values()) and
                v['baseline_alpha_matches_parent'] and v['baseline_updates_match_transfer'] for v in baseline)
            if warm_restart:
                checks['baseline_optimizer_moments_hyperparameters_rng_verified']=all(
                    all(v['baseline_optimizer_moment_step_equality'].values()) and
                    all(v['baseline_optimizer_hyperparameters_equality'].values()) and
                    all(v['baseline_rng_equality'].values()) for v in baseline)
            else:checks['baseline_optimizer_empty_verified']=all(not any(v['optimizer_state_counts'].values()) for v in baseline)
        summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
        best=json.loads((run/'best.json').read_text()) if (run/'best.json').exists() else None
        if summary:
            checks['final_transition_budget_matches']=summary['transitions']==summary['vector_steps']*c['num_envs']
            checks['final_update_budget_matches']=summary['updates_this_run']==expected_updates(summary['vector_steps'])-initial_updates
        speeds=[r['loop_transitions_per_second'] for r in rows if r.get('updates',0)>0 and 'loop_transitions_per_second' in r]
        result['runs'].append(dict(run=str(run),status='completed' if summary else 'in_progress_snapshot',
            checks=checks,all_checks_passed=all(checks.values()),parent_checkpoint=str(parent_path),parent_alpha=parent_alpha,
            transfer_mode=origin['transfer_mode'],initial_updates=initial_updates,
            source_files_verified=len(expected),sac=c['sac'],control_limits=c['control_limits'],finger_clearance=c['finger_clearance'],
            reward=reward,evaluation_selection=c['evaluation_selection'],
            budget=dict(num_envs=c['num_envs'],requested_vector_steps=c['requested_vector_steps'],
                requested_transitions=c['requested_transitions'],batch_size=c['batch_size'],replay_capacity=c['replay_capacity'],
                prefill_transitions=start-1,updates_per_transition=ratio,
                expected_final_updates=expected_updates(c['requested_vector_steps']),
                expected_new_updates=expected_updates(c['requested_vector_steps'])-initial_updates),
            latest_logged_vector_step=latest['vector_step'] if latest else None,
            latest_logged_updates=latest['updates'] if latest else None,
            last_metrics_line_partial_ignored=skipped_partial,evaluations=evaluations,
            unsafe_or_unobserved_best_candidates=[r['vector_step'] for r in evaluations if not r['selection'].get('best_eligible',False)],
            best={k:v for k,v in best.items() if k!='evaluation'} if best else None,checkpoints=payloads,
            completed_summary=summary,mean_reported_loop_transitions_per_second=(sum(speeds)/len(speeds) if speeds else None),
            limitations=['Config/source/checkpoint audit only; no GPU re-evaluation or trajectory simulation.',
                'Loop speed is training log timing, not pure learner/GPU throughput; full elapsed includes initialization and eval.',
                'Initial network equality is only directly checkable when a saved checkpoint has vector_step 0.']))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(archive_sha256=result['archive_sha256'],archive_file_count=all_files,
        source_files=len(archived),runs=[dict(run=r['run'],status=r['status'],checks=r['checks'],
            latest_step=r['latest_logged_vector_step'],best=r['best'],checkpoints=r['checkpoints']) for r in result['runs']]),indent=2))


if __name__=='__main__':main()
