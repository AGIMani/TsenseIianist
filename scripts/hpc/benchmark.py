#!/usr/bin/env python3
"""Compare GPU execution and evaluation at equal per-policy training budgets.

Run through scripts/hpc/submit.sh benchmark or performance. Paired mode runs
independent policies: aggregate throughput is not single-policy speedup.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = ('single', 'single_batched', 'pipeline', 'pipeline_batched', 'paired_batched')


def read_metrics(run):
    path = run / 'metrics.jsonl'
    return ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.is_file() else [])


def steady_metrics(run):
    """Exclude startup/capture/final eval; retain intervening periodic evals."""
    records = [row for row in read_metrics(run) if row.get('updates', 0) > 0
               and row.get('loop_transitions_per_second', 0) > 0]
    if len(records) < 2:
        return None
    first, last = records[0], records[-1]
    elapsed = (last['transitions'] / last['loop_transitions_per_second']
               - first['transitions'] / first['loop_transitions_per_second'])
    if elapsed <= 0:
        return None
    return dict(start_vector_step=first['vector_step'], end_vector_step=last['vector_step'],
                seconds=elapsed, transitions=last['transitions'] - first['transitions'],
                updates=last['updates'] - first['updates'],
                transitions_per_second=(last['transitions'] - first['transitions']) / elapsed,
                updates_per_second=(last['updates'] - first['updates']) / elapsed,
                includes_periodic_evaluation=True)


def training_command(args, mode, base, *, device=0):
    steps = args.samples // args.num_envs
    command = ['bash', 'run_tsensei_a32.sh', '--device', f'cuda:{device}',
               '--no-swanlab', '--no-progress', '--num-envs', str(args.num_envs),
               '--vector-steps', str(steps), '--eval-interval', str(args.eval_interval or steps),
               '--eval-num-envs', '3' if mode.endswith('batched') else '1',
               '--eval-episodes', '3', '--wrist-tx-max-velocity', '1',
               '--wrist-ty-max-velocity', '1', '--anticipatory-fingering-time', '0.15',
               '--seed', str(args.seed), '--save-interval', '0', '--log-interval', '10',
               # Keep quality/best gates but guarantee the full benchmark budget.
               '--target-consecutive-evals', str(steps + 2), '--output-dir', str(base)]
    if mode.startswith('pipeline'):
        command += ['--learner-device', 'cuda:1', '--collector-pipeline']
    else:
        command += ['--learner-device', f'cuda:{device}', '--no-collector-pipeline']
    return command


def check_budget(args, mode, config, summary):
    steps = args.samples // args.num_envs
    fraction = config.get('scheduling', {}).get('update_ratio_fraction')
    start = config.get('learning_start_transition_1_based')
    expected = (max(0, args.samples - start + 1) * fraction[0] // fraction[1]
                if fraction and start is not None else None)
    pipeline = summary.get('collector_pipeline_state')
    pipeline_valid = (not mode.startswith('pipeline') or bool(
        pipeline and pipeline['pending_vector_step'] is None
        and pipeline['collected_vector_steps'] == steps
        and pipeline['learner_updated_through_vector_step'] == steps
        and fraction and pipeline['max_actor_lag_at_collection_end_updates']
        <= math.ceil(args.num_envs * fraction[0] / fraction[1])))
    valid = (summary.get('transitions') == args.samples
             and summary.get('replay_insertions') == args.samples
             and summary.get('updates_this_run', 0) > 0
             and summary.get('updates_this_run') == expected
             and config.get('observation_size') == 1231 and config.get('action_size') == 47
             and pipeline_valid)
    return valid, expected


def load_run(args, mode, base, returncode):
    row = dict(mode=mode, returncode=returncode)
    runs = [p for p in sorted(args.output.glob(base.name + '-*')) if p.is_dir()]
    if len(runs) == 1:
        run = runs[0]
        row['run'] = str(run)
        for name in ('summary.json', 'config.json'):
            if (run / name).is_file():
                row[name.removesuffix('.json')] = json.loads((run / name).read_text())
        row['steady'] = steady_metrics(run)
    config, summary = row.get('config', {}), row.get('summary', {})
    valid, expected = check_budget(args, mode, config, summary)
    row.update(expected_updates=expected, valid_budget=returncode == 0 and valid)
    records = read_metrics(Path(row['run'])) if 'run' in row else []
    row['evaluation_count'] = sum('eval' in record for record in records)
    row['evaluation_timing'] = summary.get('evaluation_timing')
    row['execution_timing'] = summary.get('execution_timing')
    row['pipeline_timing'] = summary.get('collector_pipeline_state')
    row['final_music_f1'] = (summary.get('final_evaluation') or {}).get('music_only_f1')
    return row


def checkpoint_reload(args, row, label):
    if not row['valid_budget'] or not args.checkpoint_reload:
        return row['valid_budget']
    output = args.output / f'{label}-reload.json'
    with (args.output / f'{label}-reload.log').open('w') as log:
        result = subprocess.run([
            'bash', 'eval.sh', '--checkpoint', str(Path(row['run']) / 'final.pt'),
            '--device', 'cuda:0', '--episodes', '1', '--output', str(output),
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    row['checkpoint_reload_passed'] = result.returncode == 0 and output.is_file()
    return row['checkpoint_reload_passed']


def run_mode(args, mode, repeat):
    label = f'{mode}-{repeat}'
    count = 2 if mode == 'paired_batched' else 1
    bases = [args.output / (label + (f'-gpu{i}' if count == 2 else '')) for i in range(count)]
    commands = [training_command(args, mode, base, device=i) for i, base in enumerate(bases)]
    processes, returncodes = [], []
    started = time.perf_counter()
    with ExitStack() as stack:
        try:
            for base, command in zip(bases, commands):
                log = stack.enter_context(base.with_suffix('.log').open('w'))
                processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            returncodes = [process.wait() for process in processes]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
    wall = time.perf_counter() - started
    members = [load_run(args, mode, base, code) for base, code in zip(bases, returncodes)]
    valid = all(member['valid_budget'] for member in members)
    for index, member in enumerate(members):
        member['command'] = commands[index]
        if valid:
            valid = checkpoint_reload(args, member, label + (f'-gpu{index}' if count == 2 else '')) and valid
    steady = [member.get('steady') for member in members]
    return dict(mode=mode, repeat=repeat, policy_count=count, wall_seconds=wall,
                transitions_across_policies=args.samples * count,
                aggregate_transitions_per_second=args.samples * count / wall,
                # Each process has its own window; this sum is an estimate.
                steady_aggregate_transitions_per_second=(sum(m['transitions_per_second'] for m in steady)
                                                         if all(steady) else None),
                valid=valid, members=members,
                interpretation=('Independent policies; compare experiment throughput, not one policy speed'
                                if count == 2 else 'One policy at the fixed sample/update budget'))


def compact(row):
    return {**{k: v for k, v in row.items() if k != 'members'},
            'members': [{k: v for k, v in member.items() if k not in ('config', 'summary', 'command')}
                        for member in row['members']]}


def comparison(rows):
    by_mode = {}
    for mode in dict.fromkeys(row['mode'] for row in rows):
        members = [row for row in rows if row['mode'] == mode]
        steady = [row['steady_aggregate_transitions_per_second'] for row in members]
        by_mode[mode] = dict(mean_wall_seconds=statistics.mean(row['wall_seconds'] for row in members),
                             mean_aggregate_transitions_per_second=statistics.mean(
                                 row['aggregate_transitions_per_second'] for row in members),
                             mean_steady_aggregate_transitions_per_second=(statistics.mean(steady)
                                                                          if all(steady) else None))
    comparisons = {}
    for label, first, second in (
            ('pipeline_vs_single_serial_eval', 'single', 'pipeline'),
            ('pipeline_vs_single_batched_eval', 'single_batched', 'pipeline_batched'),
            ('batched_eval_single', 'single', 'single_batched'),
            ('batched_eval_pipeline', 'pipeline', 'pipeline_batched'),
            ('independent_experiment_throughput_vs_pipeline', 'pipeline_batched', 'paired_batched')):
        if first in by_mode and second in by_mode:
            a, b = by_mode[first], by_mode[second]
            comparisons[label] = dict(
                aggregate_wall_throughput_ratio=b['mean_aggregate_transitions_per_second'] / a['mean_aggregate_transitions_per_second'],
                aggregate_steady_throughput_ratio=(b['mean_steady_aggregate_transitions_per_second'] / a['mean_steady_aggregate_transitions_per_second']
                    if a['mean_steady_aggregate_transitions_per_second'] and b['mean_steady_aggregate_transitions_per_second'] else None))
    return dict(by_mode=by_mode, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=262144)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-interval', type=int, default=0,
                        help='Vector steps; zero uses full budget, as in legacy short benchmark')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=['single', 'pipeline'])
    parser.add_argument('--checkpoint-reload', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('GPU benchmarks must run in a Slurm allocation')
    if min(args.num_envs, args.samples, args.repeats) <= 0 or args.eval_interval < 0:
        parser.error('samples, num-envs and repeats must be positive; eval interval must be nonnegative')
    if args.samples % args.num_envs:
        parser.error('samples must be divisible by num-envs for an equal budget')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    report = dict(samples_per_policy=args.samples, num_envs=args.num_envs, repeats=args.repeats,
                  eval_interval=args.eval_interval or args.samples // args.num_envs, rows=rows,
                  caveat='Cold/shared caches affect wall time. Steady windows retain periodic evaluation. Paired mode trains independent policies. Short F1 is not convergence evidence.')
    for repeat in range(args.repeats):
        modes = args.modes if repeat % 2 == 0 else args.modes[::-1]
        for mode in modes:
            row = run_mode(args, mode, repeat)
            rows.append(row)
            report['status'] = 'running' if row['valid'] else 'failed'
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(compact(row)), flush=True)
            if not row['valid']:
                raise SystemExit('Benchmark budget/contract/checkpoint validation failed; inspect run logs')
    counts = {member['summary']['updates_this_run'] for row in rows for member in row['members']}
    if len(counts) != 1:
        report.update(status='failed', error='Per-policy update counts differ')
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        raise SystemExit('Update counts differ; cannot report equal-budget performance')
    report.update(status='completed', **comparison(rows))
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
