"""Synthetic CPU-only fixtures for completed/partial coupling result reporting."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('coupling_summary', ROOT/'scripts/summarize_coupling_comparison.py')
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def manifest():
    return dict(requested_samples_per_arm=3000320, expected_updates_per_arm=726656,
        runs=[dict(name='full', job_id=101, output_prefix='full',
                   expected_action_size=31, expected_observation_size=1231)])


def write(path, value):
    path.write_text(json.dumps(value)+'\n')


def create_run(tmp_path, *, complete=True, observation_size=1231):
    run = tmp_path/'full-20260930-120000-000001'
    run.mkdir()
    write(run/'config.json', dict(action_size=31, observation_size=observation_size, target_entropy=-15.5))
    best = dict(music_only_f1=.94, music_only_f1_min=.93,
                hand_collision_substeps=0, key_underside_substeps=0)
    write(run/'best.json', dict(vector_step=5800, updates=718976, evaluation=best))
    evals = [dict(transitions=t, updates=u, vector_step=t//512,
                  eval=dict(music_only_f1=f1, music_only_f1_min=f1-.01))
             for t, u, f1 in ((2400256, 576640, .1), (2560000, 616576, .9), (3000320, 726656, .94))]
    (run/'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in evals)+'\n')
    if complete:
        write(run/'summary.json', dict(status='completed', transitions=3000320,
            replay_insertions=3000320, updates_this_run=726656, updates_total=726656,
            final_evaluation=dict(music_only_f1=.94, music_only_f1_min=.93,
                music_only_substep_key_micro_f1=.91, hand_collision_substeps=3,
                key_underside_substeps=0, finger_coupling_active_actual_error_deg_mean=6.,
                finger_coupling_active_actual_error_deg_max=20., finger_coupling_sample_hz=20.)))
    return run


def test_completed_reads_real_keys_and_fixed_final_window(tmp_path):
    create_run(tmp_path)
    report = reporter.summarize(manifest(), tmp_path, states={'101': 'COMPLETED'})
    run = report['runs'][0]
    assert report['comparison_complete'] is True
    assert report['winner'] is None
    assert run['status'] == 'completed'
    assert run['best']['metrics']['music_only_f1'] == .94
    assert run['best']['metrics']['music_only_f1_min'] == .93
    assert run['final']['music_only_substep_key_micro_f1'] == .91
    assert run['updates_this_run'] == 726656
    assert run['last_500k']['evaluation_count'] == 2
    assert abs(run['last_500k']['mean_music_f1'] - .92) < 1e-12
    assert run['last_500k']['window_complete'] is True
    assert '6.00/20.00' in reporter.markdown(report)


def test_running_never_infers_final_or_winner_even_at_logged_budget(tmp_path):
    create_run(tmp_path, complete=False)
    report = reporter.summarize(manifest(), tmp_path, states={'101': 'RUNNING'})
    run = report['runs'][0]
    assert report['comparison_complete'] is False
    assert report['conclusion'] == 'incomplete_no_winner'
    assert run['status'] == 'running'
    assert run['final'] is None
    assert run['transitions'] == 3000320
    assert run['updates_total'] == 726656
    assert run['updates_this_run'] is None
    assert not run['last_500k']['window_complete']


def test_pending_and_failed_do_not_fabricate_zero_progress(tmp_path):
    pending = reporter.summarize(manifest(), tmp_path, states={'101': 'PENDING'})['runs'][0]
    assert pending['status'] == 'pending'
    assert pending['transitions'] is None
    assert pending['best'] is None
    create_run(tmp_path, complete=False)
    failed = reporter.summarize(manifest(), tmp_path, states={'101': 'FAILED'})['runs'][0]
    assert failed['status'] == 'failed'
    unknown = reporter.summarize(manifest(), tmp_path, states={})['runs'][0]
    assert unknown['status'] == 'incomplete'


def test_duplicate_output_directory_rejected_without_picking_latest(tmp_path):
    create_run(tmp_path)
    (tmp_path/'full-20260930-120000-000002').mkdir()
    report = reporter.summarize(manifest(), tmp_path, states={})
    assert report['runs'][0]['status'] == 'ambiguous'
    assert not report['comparison_complete']
    assert '多个匹配输出目录' in reporter.markdown(report)


def test_invalid_dimensions_or_budget_cannot_pass_completed_comparison(tmp_path):
    run = create_run(tmp_path, observation_size=1199)
    report = reporter.summarize(manifest(), tmp_path, states={})
    assert not report['comparison_complete']
    assert not report['runs'][0]['contract_matches']
    summary = json.loads((run/'summary.json').read_text())
    summary['updates_this_run'] -= 1
    write(run/'summary.json', summary)
    report = reporter.summarize(manifest(), tmp_path, states={})
    assert not report['runs'][0]['budget_complete']


def test_partial_last_record_excluded_and_report_deterministic(tmp_path):
    run = create_run(tmp_path, complete=False)
    with (run/'metrics.jsonl').open('a') as stream:
        stream.write('{"transitions":')
    first = reporter.summarize(manifest(), tmp_path, states={'101': 'RUNNING'})
    second = reporter.summarize(manifest(), tmp_path, states={'101': 'RUNNING'})
    assert first == second
    assert first['runs'][0]['transitions'] == 3000320
    assert first['runs'][0]['data_issues']
    assert reporter.markdown(first) == reporter.markdown(second)
