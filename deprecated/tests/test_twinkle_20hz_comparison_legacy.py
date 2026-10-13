"""CPU fake-Slurm lifecycle and strict contract tests; never submit real work."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/hpc'))
import run_twinkle_20hz_comparison as runner


@pytest.fixture
def manifest(tmp_path):
    reward = tmp_path/'reward.json'; runner.write(reward, {'schema': 1, 'terms': {}})
    archive = tmp_path/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        data = b'contract source'; member = tarfile.TarInfo('tsensei_v3_2_pianist/example.py')
        member.size = len(data); tar.addfile(member, io.BytesIO(data))
    data = dict(schema=1, source_archive=str(archive), source_archive_sha256=runner.digest(archive),
        output_root=str(tmp_path/'output'), vector_steps=20, num_envs=512, eval_interval=10,
        groups=[dict(label=label, reward_config=str(reward), reward_config_sha256=runner.digest(reward))
                for label in ('S0', 'S1', 'S2')])
    path = tmp_path/'manifest.json'; runner.write(path, data)
    return path


def configuration(manifest, hashes):
    m = runner.load_manifest(manifest)
    return dict(replay_bytes=1000, observation_size=1407, action_size=31,
        action_contract={'action_size': 31}, no_arms=True, disable_wrist_tz=True, disable_wrist_yaw=True,
        source_runtime={'environment': dict(control_hz=20., physics_hz=200., lookahead=10),
                        'source_sha256': hashes}, source_sha256=hashes,
        cli=dict(seed=42, eval_interval=10, eval_episodes=3),
        requested_vector_steps=20, requested_transitions=10240, num_envs=512,
        reward_config=runner.read(m['groups'][0]['reward_config']),
        execution_layout=dict(collector_pipeline=False, collector_device='cuda:0', learner_device='cuda:0'),
        evaluation_selection=dict(best_metric=runner.METRIC, max_key_underside_substeps=0,
                                  consecutive_evaluations=22),
        learning_start_transition_1_based=5121, scheduling={'update_ratio_fraction': [1, 4]},
        warmup_transitions=9999, sac={'gamma': .75})


def evaluation(n=3, *, tp=95, fp=2, fn=5, underside=0):
    common = dict(collision_diagnostics_valid=True, key_contact_diagnostics_valid=True,
        substep_key_diagnostics_valid=True, collision_diagnostics_coverage=1.,
        collision_substep_coverage=1., key_contact_diagnostics_coverage=1.,
        key_contact_substep_coverage=1., substep_key_diagnostics_coverage=1., substep_key_coverage=1.,
        exact_original_music_micro_f1=2*tp/(2*tp+fp+fn),
        exact_original_music_true_positive_count=tp, exact_original_music_false_positive_count=fp,
        exact_original_music_false_negative_count=fn)
    episodes = [dict(common, episode=i, note_timing_available=True,
        note_timing_records=[{} for _ in range(34)], key_underside_substeps=underside) for i in range(n)]
    return dict(common, evaluation_episodes=n, completed_episodes=n, truncated_episodes=0,
        episode_statistics=episodes, note_timing_reference_count=34*n, substep_key_sample_hz=200.,
        collision_physics_sample_hz=200., key_contact_physics_sample_hz=200.,
        exact_original_music_true_positive_count=tp*n, exact_original_music_false_positive_count=fp*n,
        exact_original_music_false_negative_count=fn*n, hand_collision_substeps=0.,
        key_side_or_under_substeps=2., key_underside_substeps_max=underside)


def test_commands_single_gpu_and_double_archive_replay(manifest):
    m = runner.load_manifest(manifest); argv = runner.train_command(m, m['groups'][0])
    get = lambda k: argv[argv.index(k)+1]
    assert get('--gpus') == '1' and get('--control-hz') == '20' and get('--physics-hz') == '200'
    assert get('--lookahead') == '10' and get('--target-max-key-underside-substeps') == '0'
    assert get('--target-consecutive-evals') == '22' and '--allocation' not in argv
    selected = dict(path='/tmp/best.pt', config='/tmp/config.json')
    replay = runner.replay_command(m, selected, '/tmp/new.json')
    assert replay.count('--source-archive') == 2 and replay[replay.index('--episodes')+1] == '10'
    assert '--reward-config' not in replay and '--actor-lr' not in replay


@pytest.mark.parametrize('flag', ['--warm-start', '--finetune-from=x', '--stage2-from', '--control-hz', '--reward-config'])
def test_reject_injected_transfer_or_controlled_flags(manifest, flag):
    m = runner.read(manifest); m['common_args'] = [flag]; runner.write(manifest, m)
    with pytest.raises(ValueError, match='Controlled/warmstart'): runner.load_manifest(manifest)


def test_pins_and_effective_contract_fail_closed(manifest):
    m = runner.load_manifest(manifest); hashes = runner.archive_hashes(m['source_archive'])
    c = configuration(manifest, hashes)
    assert runner.validate_config(c, m, m['groups'][0], hashes)
    for change in [lambda x: x.update(origin={'checkpoint': 'old'}),
                   lambda x: x['source_runtime']['environment'].update(physics_hz=400),
                   lambda x: x.update(action_size=47),
                   lambda x: x['cli'].update(finetune_from='old'),
                   lambda x: x['reward_config'].update(extra=1),
                   lambda x: x['source_sha256'].update(unexpected='bad'),
                   lambda x: x['evaluation_selection'].update(consecutive_evaluations=3)]:
        bad = deepcopy(c); change(bad)
        with pytest.raises(ValueError): runner.validate_config(bad, m, m['groups'][0], hashes)
    partial = deepcopy(c); partial.pop('replay_bytes')
    assert not runner.validate_config(partial, m, m['groups'][0], hashes)
    Path(m['groups'][0]['reward_config']).write_text('{}')
    with pytest.raises(ValueError, match='Pinned file changed'): runner.load_manifest(manifest)


@pytest.mark.parametrize('mutation', [lambda e:e.update(substep_key_sample_hz=400),
    lambda e:e.update(exact_original_music_micro_f1=float('nan')),
    lambda e:e.update(note_timing_reference_count=99),
    lambda e:e['episode_statistics'][0].update(collision_substep_coverage=.9),
    lambda e:e['episode_statistics'][0].update(exact_original_music_false_negative_count=99),
    lambda e:e.update(key_underside_substeps_max=1)])
def test_exact_diagnostics_fail_closed(mutation):
    e=evaluation(); mutation(e)
    with pytest.raises(ValueError): runner.evaluation_stats(e)


def test_underside_contact_is_ineligible_not_dropped():
    stats=runner.evaluation_stats(evaluation(underside=1))
    assert not stats['eligible'] and stats['underside_max']==1


def test_incremental_metrics_restart_partial_line_and_no_nan(manifest, tmp_path):
    m=runner.load_manifest(manifest);c=configuration(manifest, runner.archive_hashes(m['source_archive']))
    p=tmp_path/'metrics.jsonl';stage={}
    row=dict(vector_step=10, transitions=5120, updates=0, eval=evaluation())
    p.write_text(json.dumps(row)+'\n'+ '{"vector')
    runner.incremental_metrics(p,stage,c,3)
    assert len(stage['curve'])==1 and stage['latest_vector_step']==10
    row2=dict(vector_step=20,transitions=10240,updates=1280,eval=evaluation())
    p.write_text(json.dumps(row)+'\n'+json.dumps(row2)+'\n')
    resumed=json.loads(json.dumps(stage));runner.incremental_metrics(p,resumed,c,3)
    assert [r['vector_step'] for r in resumed['curve']]==[10,20]
    with p.open('a') as stream:stream.write(json.dumps(dict(vector_step=20,transitions=10240,updates=1280,train={'loss':float('nan')}))+'\n')
    with pytest.raises(ValueError,match='Nonfinite'):runner.incremental_metrics(p,resumed,c,3)


@pytest.mark.parametrize('step', [366, 3660])
def test_episode_is_pre_update_but_same_step_train_and_eval_are_post_update(manifest,tmp_path,step):
    m=runner.load_manifest(manifest);c=configuration(manifest,runner.archive_hashes(m['source_archive']))
    # Actual20Hz full-run contract:183-step prefill,128 updates per vector step.
    c['learning_start_transition_1_based']=93697
    assert runner.update_count(c,365)==23296
    assert runner.update_count(c,366)==23424
    before,after=runner.update_count(c,step-1),runner.update_count(c,step)
    #3660 is the first post-prefill intersection of183-step episodes and20-step logs.
    episode=dict(vector_step=step,transitions=step*512,updates=before,
        episodes=dict(count=512,mean_return=392.5284423828125,mean_f1=.1675570011138916))
    post=dict(vector_step=step,transitions=step*512,updates=after,train={'loss':1.})
    evaluated=dict(vector_step=step,transitions=step*512,updates=after,eval=evaluation())
    p=tmp_path/'metrics.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in (episode,post,evaluated)))
    stage={};runner.incremental_metrics(p,stage,c,3)
    assert stage['latest_vector_step']==step and stage['curve'][0]['updates']==after
    # No +/-one-block tolerance: stale train/eval or post-update episode fail.
    bad_rows=[dict(post,updates=before),dict(evaluated,updates=before),dict(episode,updates=after),
              dict(post,updates=before,episodes=episode['episodes']),
              dict(evaluated,updates=before,episodes=episode['episodes'])]
    for bad in bad_rows:
        p.write_text(json.dumps(bad)+'\n')
        with pytest.raises(ValueError,match='Logged new-update mismatch'):
            runner.incremental_metrics(p,{},c,3)


class FakeSlurm:
    def __init__(self):self.submissions=[];self.jobs={};self.cancelled=[]
    def submit(self,path):
        request=runner.read(path);job=str(100+len(self.submissions));self.submissions.append(request)
        self.jobs[job]='PENDING';runner.write(request['receipt'],dict(status='submitted',job_id=job))
        return int(job)+1000
    def states(self,jobs):return {j:self.jobs[j] for j in jobs if j in self.jobs}
    def cancel(self,job):self.cancelled.append(job)


def fake_candidate(label, role, step, score):
    return dict(group=label,role=role,vector_step=step,updates=100,eligible=True,
        path='/tmp/'+label+'/'+role+'.pt',config='/tmp/'+label+'/config.json',
        checkpoint_sha256='a'*64,config_sha256='b'*64,exact_min=score,exact_pooled=score+.001,
        side_substeps=1.,hand_substeps=0.)


def test_durable_receipts_resume_capacity_and_all_best_final(manifest,monkeypatch):
    fake=FakeSlurm();a=runner.Coordinator(manifest,max_parallel=2,backend=fake)
    a.tick();assert len(fake.submissions)==2
    assert all(s['status']=='submitting' for s in a.state['groups'][:2])
    # Simulate parent process exiting: a new coordinator recovers receipts; no resubmit.
    b=runner.Coordinator(manifest,max_parallel=2,backend=fake);b.tick()
    assert len(fake.submissions)==2 and b.state['groups'][0]['job_id']=='100'
    for s in b.state['groups'][:2]:
        s.update(status='completed',candidates=[fake_candidate(s['label'],'best',10,.9),fake_candidate(s['label'],'final',20,.89)])
    b.save();b.tick();assert len(fake.submissions)==3
    s=b.state['groups'][2];s.update(status='completed',candidates=[fake_candidate(s['label'],'best',20,.8),fake_candidate(s['label'],'final',20,.8)])
    b.save();b.tick()
    assert b.state['replays_planned'] and len(b.state['replays'])==5 #same-step best/final collapsed
    assert len(fake.submissions)==5 #at most2 replay jobs concurrently
    assert all(r['kind']=='replay' for r in b.state['replays'])
    assert all(request['argv'].count('--source-archive')==2 for request in fake.submissions[3:])


def test_unknown_submission_never_duplicates_or_cancels_other_jobs(manifest):
    fake=FakeSlurm();a=runner.Coordinator(manifest,backend=fake);a.tick()
    stage=a.state['groups'][0];Path(stage['receipt']).unlink();stage['requested_epoch']=0;a.save()
    b=runner.Coordinator(manifest,backend=fake);assert b.tick()
    assert b.state['status']=='needs_submission_reconciliation' and len(fake.submissions)==2
    assert fake.cancelled==[]


def test_resume_rejects_mutable_manifest(manifest):
    runner.Coordinator(manifest,backend=FakeSlurm())
    m=runner.read(manifest);m['seed']=43;runner.write(manifest,m)
    with pytest.raises(ValueError,match='Manifest changed'):runner.Coordinator(manifest,backend=FakeSlurm())


def test_final_ranking_is_pooled_first():
    a=dict(exact_min=.98,exact_pooled=.981,side_substeps=0,hand_substeps=0)
    b=dict(exact_min=.97,exact_pooled=.99,side_substeps=2,hand_substeps=0)
    assert sorted([a,b],key=runner.rank_key)[0] is a
    assert sorted([a,b],key=runner.independent_rank_key)[0] is b


def test_no_eligible_model_is_not_success(manifest):
    a=runner.Coordinator(manifest,backend=FakeSlurm())
    for stage in a.state['groups']:stage.update(status='completed',candidates=[])
    assert a.tick()
    assert a.state['status']=='completed_with_failures'
    assert not (a.root/'best_model.json').exists()


def test_fake_slurm_replay_completion_writes_referenced_best(manifest,tmp_path):
    fake=FakeSlurm();a=runner.Coordinator(manifest,max_parallel=4,backend=fake)
    for idx,stage in enumerate(a.state['groups']):
        candidate=fake_candidate(stage['label'],'best',20,.9-idx*.01)
        path=tmp_path/(stage['label']+'.pt');path.write_bytes(b'model'+bytes([idx]))
        config=tmp_path/(stage['label']+'.json');runner.write(config,{'group':stage['label']})
        candidate.update(path=str(path),config=str(config),checkpoint_sha256=runner.digest(path),config_sha256=runner.digest(config))
        stage.update(status='completed',candidates=[candidate])
    a.tick();assert len(fake.submissions)==3
    for i,replay in enumerate(a.state['replays']):
        job=str(100+i);fake.jobs[job]='COMPLETED'
        runner.write(replay['output'],dict(checkpoint=replay['selected']['path'],episodes=10,
            metrics=evaluation(10,tp=95+i,fp=2,fn=5-i)))
    assert a.tick() and a.state['status']=='completed'
    best=runner.read(a.root/'best_model.json')
    assert best['selected']['group']=='S2'
    assert best['model_files_copied_or_overwritten'] is False
    assert len(runner.read(a.root/'leaderboard.json')['independent10'])==3


def test_checkpoint_cpu_metadata_and_finite_tensor_checks(manifest,tmp_path):
    import torch
    m=runner.load_manifest(manifest);c=configuration(manifest,runner.archive_hashes(m['source_archive']))
    p=tmp_path/'best.pt';runner.write(tmp_path/'config.json',c);e=evaluation()
    payload=dict(metadata=dict(c,vector_step=20,updates=1280,last_evaluation_vector_step=20,last_evaluation=e),
        agent=dict(updates=1280,config=c['sac'],obs_size=1407,action_size=31,actor={'weight':torch.ones(2)}))
    torch.save(payload,p);assert runner.checkpoint_check(p,c,20,1280,e)['cuda_initialized'] is False
    payload['agent']['actor']['weight'][0]=float('nan');torch.save(payload,p)
    with pytest.raises(ValueError,match='Nonfinite checkpoint'):runner.checkpoint_check(p,c,20,1280,e)
    assert not torch.cuda.is_initialized()
