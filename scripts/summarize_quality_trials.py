#!/usr/bin/env python3
"""Read-only CPU summary of quality pilots; never imports Torch or changes runs."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('music_only_f1_min', 'music_only_micro_f1_min',
           'preparation_wrong_key_frames_max', 'hand_collision_frames_max',
           'self_collision_frames_max', 'inter_hand_collision_frames_max',
           'hand_collision_substeps_max', 'self_collision_substeps_max',
           'inter_hand_collision_substeps_max', 'collision_diagnostics_coverage',
           'collision_substep_coverage', 'collision_physics_sample_hz',
           'completed_episodes', 'truncated_episodes', 'evaluation_episodes')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_rows(path):
    if not path.exists():
        return [], []
    content = path.read_text()
    lines = content.splitlines()
    rows, warnings = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1 or content.endswith('\n'):
                raise ValueError(f'Malformed complete metrics row: {path}:{index+1}')
            warnings.append('Ignored unfinished final metrics row while run may still be writing')
    return rows, warnings


def steady_intervals(rows, config):
    """Infer log timestamps from logged cumulative transitions/second.

    The runner logs before evaluation/checkpoint at that vector step. Reject
    intervals containing either boundary, including the left endpoint. Both
    endpoints must already have learning updates, excluding the first capture.
    This is measured whole training-loop throughput inferred from logs, not a
    separate learner/kernel benchmark or a complete-run wall-clock estimate.
    """
    initial_updates = next((row['updates'] for row in rows if row.get('vector_step') == 0 and 'updates' in row),
                           min((row['updates'] for row in rows if 'updates' in row), default=0))
    logs = [row for row in rows if finite(row.get('loop_transitions_per_second'))
            and row['loop_transitions_per_second'] > 0 and row.get('updates', 0) > initial_updates]
    eval_steps = {row['vector_step'] for row in rows if 'eval' in row}
    save_interval = config.get('cli', {}).get('save_interval', 0) or 0
    intervals = []
    for first, last in zip(logs, logs[1:]):
        lo, hi = first['vector_step'], last['vector_step']
        if any(lo <= step <= hi for step in eval_steps):
            continue
        if save_interval and (lo+save_interval-1)//save_interval*save_interval <= hi:
            continue
        before = first['transitions']/first['loop_transitions_per_second']
        after = last['transitions']/last['loop_transitions_per_second']
        elapsed = after-before
        samples = last['transitions']-first['transitions']
        updates = last['updates']-first['updates']
        if elapsed <= 0 or samples <= 0 or updates <= 0:
            continue
        intervals.append(dict(from_vector_step=lo, to_vector_step=hi, elapsed_s=elapsed,
                              transitions=samples, updates=updates,
                              transitions_per_second=samples/elapsed, updates_per_second=updates/elapsed))
    return dict(method='Difference of transitions / logged cumulative loop_transitions_per_second; '
                       'exclude evaluation/save boundaries and pre-learning/capture intervals',
                initial_updates=initial_updates, interval_count=len(intervals), intervals=intervals,
                median_transitions_per_second=(statistics.median(r['transitions_per_second'] for r in intervals)
                                               if intervals else None),
                median_updates_per_second=(statistics.median(r['updates_per_second'] for r in intervals)
                                           if intervals else None),
                weighted_transitions_per_second=(sum(r['transitions'] for r in intervals)
                                                 / sum(r['elapsed_s'] for r in intervals) if intervals else None))


def summarize_run(path, variant):
    config = json.loads((path/'config.json').read_text())
    rows, warnings = read_rows(path/'metrics.jsonl')
    initial_updates = next((row['updates'] for row in rows if row.get('vector_step') == 0 and 'updates' in row),
                           min((row['updates'] for row in rows if 'updates' in row), default=0))
    summary = json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else None
    evaluations = []
    for row in rows:
        if 'eval' not in row:
            continue
        values = row['eval']
        evaluations.append(dict(variant=variant, vector_step=row['vector_step'],
            transitions=row['transitions'], updates=row['updates'],
            updates_this_run=row['updates']-initial_updates,
            baseline=bool(row.get('baseline', row['updates'] == 0)),
            **{key: values.get(key) for key in METRICS}))
    baseline = next((row for row in evaluations if row['baseline']), None)
    learned = [row for row in evaluations if row['updates'] > 0 and not row['baseline']
               and finite(row['music_only_f1_min']) and row['evaluation_episodes']
               and row['completed_episodes'] == row['evaluation_episodes'] and row['truncated_episodes'] == 0]
    best_learned = max(learned, key=lambda row: row['music_only_f1_min']) if learned else None
    latest = evaluations[-1] if evaluations else None
    improvement = (best_learned['music_only_f1_min'] - baseline['music_only_f1_min']
                   if best_learned and baseline and finite(baseline['music_only_f1_min']) else None)
    enabled_rewards = {name: value for name, value in config.get('reward_config', {}).get('terms', {}).items()
                       if value['enabled']}
    return dict(variant=variant, path=str(path.resolve()),
        status=summary.get('status', 'finished_unknown_status') if summary else 'in_progress_or_interrupted',
        elapsed_s=summary.get('elapsed_s') if summary else None,
        vector_steps=summary['vector_steps'] if summary else rows[-1]['vector_step'] if rows else 0,
        transitions=summary['transitions'] if summary else rows[-1]['transitions'] if rows else 0,
        initial_updates=initial_updates,
        updates=summary['updates_this_run'] if summary else rows[-1]['updates']-initial_updates if rows else 0,
        end_to_end_transitions_per_second=(summary['transitions']/summary['elapsed_s'] if summary else None),
        batch_size=config['batch_size'], num_envs=config.get('num_envs'),
        updates_per_transition=config['steady_updates_per_transition'],
        warmup_transitions=config['warmup_transitions'], learning_start_transition=config['learning_start_transition_1_based'],
        enabled_rewards=enabled_rewards, sac=config['sac'],
        baseline=baseline, latest_evaluation=latest, evaluations=evaluations,
        evaluation_at_steps={str(step): next((row for row in evaluations if row['vector_step'] == step), None)
                             for step in (0, 256, 512)},
        best_checkpoint=summary.get('best_checkpoint') if summary else str(path/'best.pt') if (path/'best.pt').exists() else None,
        best_after_learning=best_learned, best_learned_music_delta_from_initial=improvement,
        learned_improved_over_initial=improvement > 0 if improvement is not None else None,
        steady_learning=steady_intervals(rows, config), warnings=warnings)


def summarize(root, variants=None):
    runs = []
    if variants is None:
        variants = sorted(set('ABCD') | {match[1] for path in root.iterdir()
                         if path.is_dir() and (match := re.match(r'^([A-Z])_', path.name))}) if root.exists() else list('ABCD')
    for variant in variants:
        paths = sorted(path for path in root.glob(variant+'_*')
                       if path.is_dir() and (path/'config.json').exists())
        if not paths:
            runs.append(dict(variant=variant, status='not_started', evaluations=[]))
            continue
        result = summarize_run(paths[-1], variant)
        result['other_matching_runs'] = [str(path.resolve()) for path in paths[:-1]]
        runs.append(result)
    return dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), root=str(root.resolve()), runs=runs,
                caveats=['Initial evaluation is the transferred policy, not a result of new learning.',
                         'No summary.json means progress or interruption, not proof a process is running.',
                         'Steady speeds are inferred from cumulative log timestamps, with eval/save/first-compile intervals excluded.',
                         'elapsed_s starts within the GPU runner; source-launcher preparation time is not included.',
                         'Short pilots and a single seed do not prove final convergence or guarantee F1 >= 0.99.',
                         'Metrics ending in _min/_max are worst cases over evaluated episodes, not means.'])


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(A='#247BA0', B='#4A9654', C='#D17A22', D='#9C4B9A')
    palette = plt.get_cmap('tab10')
    for index, run in enumerate(report['runs']):
        colors.setdefault(run['variant'], palette(index % 10))
    env_counts = {run['num_envs'] for run in report['runs'] if run.get('num_envs') is not None}
    xlabel = ('Vector steps ('+str(next(iter(env_counts)))+' transitions per step)'
              if len(env_counts) == 1 else 'Vector steps')
    ticks = sorted({row['vector_step'] for run in report['runs'] for row in run['evaluations']})
    panels = [('music_only_f1_min', 'Music frame F1: worst episode'),
              ('music_only_micro_f1_min', 'Music micro F1: worst episode'),
              ('hand_collision_frames_max', 'Hand-collision frames: worst episode'),
              ('preparation_wrong_key_frames_max', 'Preparation wrong-key frames: worst episode')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, panels):
        for run in report['runs']:
            values = [row for row in run['evaluations'] if finite(row.get(key))]
            if values:
                ax.plot([row['vector_step'] for row in values], [row[key] for row in values],
                        marker='o', linewidth=2, color=colors[run['variant']], label=run['variant'])
        if 'f1' in key:
            ax.axhline(.99, color='#777777', linestyle='--', linewidth=1, label='Target 0.99')
            present = [row[key] for run in report['runs'] for row in run['evaluations'] if finite(row.get(key))]
            lower = max(0, math.floor((min(present)-.05)*10)/10) if present else 0
            ax.set_ylim(lower, 1.02)
        else:
            ax.set_ylim(bottom=0)
        ax.set(title=title, xlabel=xlabel)
        if len(ticks) <= 8:
            ax.set_xticks(ticks)
        ax.grid(alpha=.22)
    axes[0, 0].legend(loc='lower left', ncol=3, frameon=False)
    fig.suptitle('Quality pilots: transferred baseline at step 0 is not learned improvement', fontsize=12)
    fig.savefig(output/'quality_trials.png', dpi=180)
    fig.savefig(output/'quality_trials.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'training_mjlab/outputs/quality_trials_20260926')
    parser.add_argument('--output', type=Path, default=ROOT/'preview/outputs/diagnostics/quality_trials_20260926')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--variants', nargs='+', help='Optional explicit labels; default A-D plus discovered A-Z runs')
    args = parser.parse_args(argv)
    report = summarize(args.root, args.variants)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    evaluations = [row for run in report['runs'] for row in run['evaluations']]
    write_csv(args.output/'evaluations.csv', evaluations,
              ['variant', 'vector_step', 'transitions', 'updates', 'updates_this_run', 'baseline', *METRICS])
    flat = []
    for run in report['runs']:
        row = {key: run.get(key) for key in ('variant', 'status', 'vector_steps', 'transitions', 'updates',
                    'elapsed_s', 'batch_size', 'updates_per_transition', 'end_to_end_transitions_per_second',
                    'best_learned_music_delta_from_initial', 'learned_improved_over_initial')}
        row['steady_transitions_per_second'] = run.get('steady_learning', {}).get('median_transitions_per_second')
        for step, evaluation in run.get('evaluation_at_steps', {}).items():
            for key in METRICS:
                row[f'step{step}_{key}'] = evaluation.get(key) if evaluation else None
        flat.append(row)
    fields = list(dict.fromkeys(key for row in flat for key in row))
    write_csv(args.output/'runs.csv', flat, fields)
    if not args.no_plot:
        plot(report, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), runs=flat), indent=2))


if __name__ == '__main__':
    main()
