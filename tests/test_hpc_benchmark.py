"""CPU checks for fixed-budget GPU benchmark commands and result accounting."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('hpc_benchmark', ROOT / 'scripts/hpc/benchmark.py')
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def arguments(**overrides):
    return SimpleNamespace(**dict(samples=262144, num_envs=512, seed=42,
                                 eval_interval=100, **overrides))


def value(command, name):
    return command[command.index(name) + 1]


def test_modes_keep_workload_and_only_change_execution():
    args = arguments()
    for mode in benchmark.MODES:
        command = benchmark.training_command(args, mode, Path('/tmp/run'))
        assert value(command, '--vector-steps') == '512'
        assert value(command, '--eval-interval') == '100'
        assert value(command, '--eval-episodes') == '3'
        assert value(command, '--wrist-tx-max-velocity') == '1'
        assert value(command, '--wrist-ty-max-velocity') == '1'
        assert value(command, '--anticipatory-fingering-time') == '0.15'
        assert value(command, '--eval-num-envs') == ('3' if mode.endswith('batched') else '1')
        assert ('--collector-pipeline' in command) == mode.startswith('pipeline')
        assert int(value(command, '--target-consecutive-evals')) > 512


def test_paired_workers_have_independent_single_device_learners():
    for index in (0, 1):
        command = benchmark.training_command(arguments(), 'paired_batched', Path('/tmp/run'), device=index)
        assert value(command, '--device') == value(command, '--learner-device') == f'cuda:{index}'
        assert '--no-collector-pipeline' in command


def budget_fixture():
    config = dict(scheduling=dict(update_ratio_fraction=[1, 4]),
                  learning_start_transition_1_based=93697, observation_size=1231, action_size=47)
    summary = dict(transitions=262144, replay_insertions=262144, updates_this_run=42112)
    return config, summary


def test_budget_accounts_for_full_score_prefill():
    config, summary = budget_fixture()
    assert benchmark.check_budget(arguments(), 'single_batched', config, summary) == (True, 42112)
    summary['updates_this_run'] -= 1
    assert benchmark.check_budget(arguments(), 'single_batched', config, summary)[0] is False


def test_pipeline_requires_flushed_bounded_actor_lag():
    config, summary = budget_fixture()
    summary['collector_pipeline_state'] = dict(pending_vector_step=None, collected_vector_steps=512,
        learner_updated_through_vector_step=512, max_actor_lag_at_collection_end_updates=128)
    assert benchmark.check_budget(arguments(), 'pipeline_batched', config, summary)[0]
    summary['collector_pipeline_state']['pending_vector_step'] = 512
    assert not benchmark.check_budget(arguments(), 'pipeline_batched', config, summary)[0]
    summary['collector_pipeline_state']['pending_vector_step'] = None
    summary['collector_pipeline_state']['max_actor_lag_at_collection_end_updates'] = 129
    assert not benchmark.check_budget(arguments(), 'pipeline_batched', config, summary)[0]


def test_steady_window_excludes_startup_but_retains_between_log_evaluations(tmp_path):
    rows = [dict(vector_step=100, updates=0, transitions=51200, loop_transitions_per_second=2560),
            dict(vector_step=190, updates=900, transitions=97280, loop_transitions_per_second=972.8),
            dict(vector_step=200, transitions=102400, eval={}),
            dict(vector_step=510, updates=41856, transitions=261120, loop_transitions_per_second=1305.6)]
    (tmp_path / 'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    result = benchmark.steady_metrics(tmp_path)
    assert result['seconds'] == 100
    assert result['transitions'] == 163840
    assert result['transitions_per_second'] == 1638.4
    assert result['includes_periodic_evaluation'] is True


def test_two_policy_comparison_uses_total_samples_not_same_policy_latency():
    rows = [dict(mode='pipeline_batched', wall_seconds=100,
                 aggregate_transitions_per_second=1000, steady_aggregate_transitions_per_second=1200),
            dict(mode='paired_batched', wall_seconds=150,
                 aggregate_transitions_per_second=200000 / 150, steady_aggregate_transitions_per_second=1600)]
    result = benchmark.comparison(rows)
    ratio = result['comparisons']['independent_experiment_throughput_vs_pipeline']
    assert abs(ratio['aggregate_wall_throughput_ratio'] - 4 / 3) < 1e-12
    assert result['by_mode']['paired_batched']['mean_wall_seconds'] == 150
