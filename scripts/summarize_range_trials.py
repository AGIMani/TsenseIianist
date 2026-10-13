#!/usr/bin/env python3
"""Read explicit trial logs on CPU; distinguish inherited scores from learning."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_quality_trials import finite, read_rows, steady_intervals

METRICS = (
    'music_only_f1_min', 'music_only_micro_f1_min', 'return',
    'preparation_wrong_key_frames_max', 'hand_collision_frames_max', 'hand_collision_substeps_max',
    'self_collision_frames_max', 'inter_hand_collision_frames_max',
    'collision_diagnostics_coverage', 'collision_substep_coverage', 'collision_physics_sample_hz',
    'completed_episodes', 'truncated_episodes', 'evaluation_episodes', 'metric_sample_hz',
    'motion_max_wrist_tx_target_jump_m', 'motion_max_wrist_ty_target_jump_m',
    'motion_max_finger_target_jump_rad', 'motion_max_wrist_pitch_target_jump_rad',
    'motion_max_wrist_tx_speed_m_s', 'motion_max_wrist_ty_speed_m_s',
    'motion_max_finger_speed_rad_s', 'motion_max_wrist_pitch_speed_rad_s', 'motion_sample_hz',
    'control_limits_enabled', 'control_limited_channel_count', 'control_limits_diagnostics_coverage',
    'control_clipped_frames_max', 'control_slew_limited_frames_max',
    'control_clipped_fraction_max', 'control_slew_limited_fraction_max')


def read_optional(path, warnings):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        warnings.append(f'{path.name} is incomplete or invalid; no completed status inferred')
        return None


def complete(values):
    episodes = values.get('evaluation_episodes')
    return (isinstance(episodes, int) and episodes > 0
            and values.get('completed_episodes') == episodes and values.get('truncated_episodes') == 0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize_run(path):
    path = Path(path).resolve()
    warnings = []
    config = read_optional(path/'config.json', warnings)
    if config is None:
        return dict(name=path.name, path=str(path), status='awaiting_config_or_invalid_config',
                    evaluations=[], warnings=warnings)
    rows, log_warnings = read_rows(path/'metrics.jsonl')
    warnings += log_warnings
    summary = read_optional(path/'summary.json', warnings)
    initial_row = next((r for r in rows if r.get('vector_step') == 0 and 'updates' in r), None)
    initial_updates = initial_row['updates'] if initial_row else None
    if initial_updates is None:
        # Current pilots explicitly start a fresh learner; warm starts without
        # a logged starting counter cannot establish an exact new-update count.
        initial_updates = 0 if config.get('origin', {}).get('fresh_optimizer_alpha_rng') else None
    if initial_updates is None:
        warnings.append('No initial learner update counter: post-learning peak classification unavailable')
    evaluations = []
    for row in rows:
        if 'eval' not in row:
            continue
        new_updates = row['updates'] - initial_updates if initial_updates is not None else None
        phase = ('initial' if row['vector_step'] == 0 else
                 'after_learning' if new_updates is not None and new_updates > 0 else
                 'pre_learning' if new_updates == 0 else 'unknown')
        evaluations.append(dict(name=path.name, vector_step=row['vector_step'],
            transitions=row.get('transitions'), updates=row.get('updates'), updates_this_run=new_updates,
            phase=phase, complete=complete(row['eval']),
            **{key: row['eval'].get(key) for key in METRICS}))
    initial = next((r for r in evaluations if r['phase'] == 'initial'), None)
    learned = [r for r in evaluations if r['phase'] == 'after_learning' and r['complete']
               and finite(r['music_only_f1_min'])]
    peak = max(learned, key=lambda r: r['music_only_f1_min']) if learned else None
    latest = evaluations[-1] if evaluations else None
    final = (latest if summary and latest and latest['vector_step'] == summary.get('vector_steps') else None)
    delta = (peak['music_only_f1_min'] - initial['music_only_f1_min']
             if peak and initial and finite(initial['music_only_f1_min']) else None)
    counters = [r for r in rows if all(k in r for k in ('vector_step', 'transitions', 'updates'))]
    last = counters[-1] if counters else {}
    elapsed_logs = [r for r in rows if finite(r.get('loop_transitions_per_second'))
                    and r['loop_transitions_per_second'] > 0]
    elapsed_last_log = (elapsed_logs[-1]['transitions']/elapsed_logs[-1]['loop_transitions_per_second']
                        if elapsed_logs else None)
    best = (summary or {}).get('evaluation_selection_state', {}).get('best')
    if best is None:
        index = read_optional(path/'best.json', warnings)
        best = index.get('best') if index and isinstance(index.get('best'), dict) else index
    config_keys = ('num_envs', 'batch_size', 'replay_capacity', 'warmup_transitions',
        'learning_start_transition_1_based', 'steady_updates_per_transition', 'steady_sample_ratio',
        'requested_vector_steps', 'requested_transitions', 'scene_sha256', 'task_sha256',
        'source_sha256', 'action_profile', 'preparation', 'control_limits', 'control_limits_contract', 'sac')
    origin = config.get('origin', {})
    elapsed = (summary or {}).get('elapsed_s')
    actual_transitions = (summary or {}).get('transitions', last.get('transitions', 0))
    actual_updates = (summary or {}).get('updates_this_run',
        last.get('updates', initial_updates)-initial_updates if initial_updates is not None else None)
    return dict(name=path.name, path=str(path),
        status=summary.get('status', 'summary_present_status_unknown') if summary else 'in_progress_or_interrupted',
        configuration={key: config.get(key) for key in config_keys} | {'seed': config.get('cli', {}).get('seed')},
        reward_config_sha256=digest(config.get('reward_config')),
        origin={key: origin.get(key) for key in ('checkpoint', 'checkpoint_sha256', 'transfer_mode',
                'fresh_optimizer_alpha_rng', 'fresh_replay', 'allow_control_change', 'previous_vector_step')},
        initial_updates=initial_updates, vector_steps=(summary or {}).get('vector_steps', last.get('vector_step', 0)),
        transitions=actual_transitions, updates_this_run=actual_updates, elapsed_s=elapsed,
        elapsed_s_at_last_training_log_inferred=elapsed_last_log,
        end_to_end_transitions_per_second=actual_transitions/elapsed if finite(elapsed) and elapsed > 0 else None,
        initial_evaluation=initial, best_after_learning=peak, latest_evaluation=latest, final_evaluation=final,
        best_learned_music_delta_from_initial=delta,
        learned_improved_over_initial=delta > 0 if delta is not None else None,
        saved_best_vector_step=best.get('vector_step') if best else None,
        saved_best_is_initial=best.get('vector_step') == 0 if best else None,
        evaluations=evaluations, steady_learning=steady_intervals(rows, config), warnings=warnings)


def summarize(paths):
    runs = [summarize_run(path) for path in paths]
    checks = {}
    for field in ('seed', 'num_envs', 'batch_size', 'requested_vector_steps', 'requested_transitions',
                  'learning_start_transition_1_based', 'steady_updates_per_transition',
                  'scene_sha256', 'task_sha256'):
        values = [r.get('configuration', {}).get(field) for r in runs]
        checks[field] = dict(values=values, equal=all(v is not None for v in values) and len(set(values)) == 1)
    parents = [r.get('origin', {}).get('checkpoint_sha256') for r in runs]
    checks['parent_checkpoint_sha256'] = dict(values=parents, equal=all(parents) and len(set(parents)) == 1)
    rewards = [r.get('reward_config_sha256') for r in runs]
    checks['reward_config_sha256'] = dict(values=rewards, equal=all(rewards) and len(set(rewards)) == 1)
    for field in ('transitions', 'updates_this_run'):
        values = [r.get(field) for r in runs]
        checks['actual_'+field] = dict(values=values, equal=all(v is not None for v in values) and len(set(values)) == 1)
    return dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), runs=runs, comparison_checks=checks,
        caveats=['Initial and prefill-only evaluations are inherited policy behavior, not new learned improvement.',
                 'Post-learning peak uses complete evaluations with a positive new-update count only.',
                 'No summary.json does not prove a process is running; partial final metrics lines are ignored.',
                 'Elapsed summary time excludes source launcher setup; inferred last-log time excludes later work.',
                 'Steady speed is inferred from log timing, excluding eval/save and first learning capture boundaries.',
                 'Motion extrema use the recorded sample rate and are not full physics-substep velocity bounds.',
                 'Clip/slew fractions are among configured limited channels; disabled/missing diagnostics remain null.',
                 'A same-budget single-seed short trial does not establish convergence or guarantee F1 >= 0.99.'])


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['name']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def flat_runs(report):
    flat = []
    for run in report['runs']:
        row = {key: run.get(key) for key in ('name', 'status', 'vector_steps', 'transitions', 'updates_this_run',
            'elapsed_s', 'end_to_end_transitions_per_second', 'saved_best_vector_step', 'saved_best_is_initial',
            'best_learned_music_delta_from_initial', 'learned_improved_over_initial')}
        config = run.get('configuration', {})
        sac = config.get('sac') or {}
        row.update(actor_lr=sac.get('actor_lr'), critic_lr=sac.get('critic_lr'),
                   temp_lr=sac.get('temp_lr'), batch_size=config.get('batch_size'),
                   updates_per_transition=config.get('steady_updates_per_transition'),
                   controller=json.dumps(config.get('control_limits'), sort_keys=True, separators=(',', ':')),
                   steady_transitions_per_second=run.get('steady_learning', {}).get('median_transitions_per_second'))
        for label in ('initial_evaluation', 'best_after_learning', 'latest_evaluation', 'final_evaluation'):
            evaluation = run.get(label) or {}
            for key in ('vector_step', 'updates_this_run', *METRICS):
                row[f'{label}_{key}'] = evaluation.get(key)
        flat.append(row)
    return flat


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('music_only_f1_min', 'Music frame F1, worst episode'),
              ('music_only_micro_f1_min', 'Music micro F1, worst episode'),
              ('hand_collision_frames_max', 'Hand collision frames, worst episode'),
              ('preparation_wrong_key_frames_max', 'Preparation wrong-key frames'),
              ('motion_max_wrist_tx_target_jump_m', 'Maximum wrist TX target jump (m)'),
              ('control_slew_limited_fraction_max', 'Slew-limited channel/frame fraction')]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    colors = plt.get_cmap('tab10')
    for ax, (key, title) in zip(axes.flat, panels):
        plotted = False
        for index, run in enumerate(report['runs']):
            rows = [r for r in run['evaluations'] if finite(r.get(key))]
            label = run['name'].replace('range_trial_', '').split('-20260927-')[0]
            if rows:
                ax.plot([r['transitions'] for r in rows], [r[key] for r in rows],
                        marker='o', label=label, color=colors(index % 10))
                plotted = True
        if not plotted:
            ax.text(.5, .5, 'Disabled or diagnostics unavailable', transform=ax.transAxes, ha='center')
            ax.set_xticks([]); ax.set_yticks([])
        ax.set(title=title, xlabel='Collected transitions'); ax.grid(alpha=.2)
        if 'f1' in key:
            ax.axhline(.99, linestyle='--', color='grey', linewidth=1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Range trials: initial/prefill scores do not count as learned improvement')
    fig.savefig(output/'range_trials.png', dpi=170); fig.savefig(output/'range_trials.svg'); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='Explicit run directories, in display order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Also write PNG/SVG; requires matplotlib')
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in args.runs}:
        parser.error('--output must not overwrite a source run directory')
    report = summarize(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_csv(args.output/'evaluations.csv', [r for run in report['runs'] for r in run['evaluations']])
    write_csv(args.output/'runs.csv', flat_runs(report))
    if args.plot:
        plot(report, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), runs=[{k: r.get(k) for k in
        ('name', 'status', 'transitions', 'updates_this_run', 'elapsed_s', 'saved_best_vector_step',
         'best_learned_music_delta_from_initial', 'learned_improved_over_initial')} for r in report['runs']]), indent=2))


if __name__ == '__main__':
    main()
