#!/usr/bin/env python3
"""Run a baseline and three one-factor reward experiments sequentially.

Uses run.sh and its full source recipe. A new output root is required; old
training directories are never overwritten. --dry-run writes the exact plan
without training; --summarize refreshes scores from an existing manifest.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ARM_NAMES = ('baseline', 'anticipatory', 'staged', 'substep')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--samples', type=int, default=1_000_000,
                        help='Requested transitions per arm, rounded up to a whole vector (default 1M)')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--num-envs', type=int, default=512)
    result.add_argument('--anticipation-seconds', type=float, default=.15)
    result.add_argument('--output-root', type=Path)
    result.add_argument('--arms', choices=ARM_NAMES, nargs='+', default=list(ARM_NAMES),
                        help='Subset/order for smoke tests; default all four arms')
    result.add_argument('--dry-run', action='store_true')
    result.add_argument('--summarize', type=Path, metavar='MANIFEST',
                        help='Refresh report for an existing comparison without starting any training')
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_inventory(root=ROOT):
    """Only replayable source/configuration; exclude outputs and environments."""
    paths = []
    for relative in ('tsensei_v3_2_pianist', 'scripts', 'configs', 'preview/scripts'):
        paths.extend(path for path in (root/relative).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'))
    for relative in ('run.sh', 'eval.sh', 'pyproject.toml', 'training/requirements.txt',
                     'training_mjlab/requirements.txt'):
        if (root/relative).is_file():
            paths.append(root/relative)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def create_source_archive(output, root=ROOT):
    expected = source_inventory(root)
    archive = output/'source.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1, dereference=True) as bundle:
        for relative in expected:
            bundle.add(root/relative, arcname=relative, recursive=False)
        # Documentation is a snapshot, not a runtime input. Later result/log
        # updates may continue without changing the locked comparison sources.
        for relative in ('README.md', 'docs/training_controls_thumb_review_20260928.md'):
            if (root/relative).is_file():
                bundle.add(root/relative, arcname=relative, recursive=False)
    if source_inventory(root) != expected:
        raise RuntimeError('Sources changed while archiving; finish edits before launching comparison')
    return dict(path=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                files_sha256=expected, file_count=len(expected))


def verify_sources(manifest, root=ROOT):
    expected = manifest['source_archive']['files_sha256']
    actual = source_inventory(root)
    if actual != expected:
        changed = sorted(name for name in actual.keys() | expected.keys() if actual.get(name) != expected.get(name))
        raise RuntimeError('Comparison sources/configuration changed; refusing unequal runs: '+', '.join(changed[:12]))


def make_plan(args, output):
    if args.samples < 1 or args.num_envs < 1:
        raise ValueError('samples and num-envs must be positive')
    if (not math.isfinite(args.anticipation_seconds)
            or not 0 < args.anticipation_seconds <= .5):
        raise ValueError('anticipation-seconds must be finite and within (0, .5]')
    if len(set(args.arms)) != len(args.arms):
        raise ValueError('Each comparison arm may appear only once')
    steps = (args.samples + args.num_envs - 1)//args.num_envs
    runs = []
    for name in args.arms:
        options = dict(anticipatory_fingering_time=args.anticipation_seconds if name == 'anticipatory' else 0.,
                       staged_fingering_reward=name == 'staged', substep_rewards=name == 'substep',
                       target_key_substep_reward=False, target_key_activation_weight=0.)
        base = output/name/'run'
        command = ['bash', str(ROOT/'run.sh'), '--seed', str(args.seed), '--num-envs', str(args.num_envs),
                   '--vector-steps', str(steps), '--actor-lr', '.0003', '--critic-lr', '.0003', '--temp-lr', '.0003',
                   '--anticipatory-fingering-time', str(options['anticipatory_fingering_time']),
                   '--staged-fingering-reward' if options['staged_fingering_reward'] else '--no-staged-fingering-reward',
                   '--substep-rewards' if options['substep_rewards'] else '--no-substep-rewards',
                   # This historical comparison varies only the original three
                   # factors, even when run.sh gains new reward defaults.
                   '--no-target-key-substep-reward', '--target-key-activation-weight', '0',
                   '--no-anticipatory-current-priority', '--replay-prefill-episodes', '0',
                   '--score-validation', 'legacy', '--release-tail-time', '0',
                   # Retain best eligibility/quality reporting but prevent a
                   # lucky early stop from giving arms different sample budgets.
                   '--target-consecutive-evals', str(steps+2), '--no-progress', '--output-dir', str(base)]
        runs.append(dict(name=name, status='pending', performance=options, argv=command,
                         output_base=str(base), log=str(output/name/'console.log'),
                         output_dir=None, config_path=None, returncode=None))
    return dict(schema=1, created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                project_root=str(ROOT), output_root=str(output), seed=args.seed,
                requested_samples_per_arm=args.samples, transitions_per_arm=steps*args.num_envs,
                vector_steps_per_arm=steps, num_envs=args.num_envs,
                learning_rates=dict(actor=.0003, critic=.0003, temperature=.0003),
                historical_defaults=dict(anticipatory_current_priority=False, replay_prefill_episodes=0,
                                         score_validation='legacy', release_tail_time=0.),
                initialization='from_scratch_same_seed; anticipation adds observations, so network tensors are not identical',
                comparison='baseline plus independent single-factor arms; unchanged small humanoid budget',
                fixed_budget='target-consecutive-evals exceeds the number of evaluations; best/underside eligibility gates retained',
                run_script_sha256=hashlib.sha256((ROOT/'run.sh').read_bytes()).hexdigest(), runs=runs)


def resolve_output(run):
    base = Path(run['output_base'])
    matches = sorted(path for path in base.parent.glob(base.name+'-*') if path.is_dir())
    if len(matches) > 1:
        raise ValueError(f'Ambiguous output directories for {run["name"]}: {matches}')
    if matches:
        run['output_dir'] = str(matches[0])
        config = matches[0]/'config.json'
        run['config_path'] = str(config) if config.is_file() else None
    return Path(run['output_dir']) if run.get('output_dir') else None


def compact_evaluation(value):
    names = ('music_only_f1', 'music_only_f1_min', 'music_only_micro_f1', 'music_only_micro_f1_min',
             'music_only_substep_key_micro_f1', 'music_only_substep_key_micro_f1_min',
             'wrong_key_substeps', 'wrong_key_key_substeps', 'music_only_wrong_key_substeps',
             'music_only_wrong_key_key_substeps', 'preparation_wrong_key_substeps',
             'preparation_wrong_key_frames_max', 'hand_collision_substeps_max',
             'key_underside_substeps_max', 'key_side_or_under_substeps_max',
             'substep_key_diagnostics_valid', 'substep_key_coverage')
    return {name: value.get(name) for name in names}


def observed_peak(output):
    """Best observed music score regardless of the checkpoint safety gate."""
    path = output/'metrics.jsonl'
    peak = None
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A live writer can leave its final line incomplete.
                continue
            evaluation = record.get('eval') or {}
            score = evaluation.get('music_only_f1_min')
            if (score is None or evaluation.get('completed_episodes') != evaluation.get('evaluation_episodes')
                    or evaluation.get('truncated_episodes', 0)):
                continue
            if peak is None or score > peak['score']:
                peak = dict(score=score, vector_step=record['vector_step'],
                            best_eligible=record.get('checkpoint_selection', {}).get('best_eligible'),
                            evaluation=compact_evaluation(evaluation))
    return peak


def summarize(manifest):
    rows = []
    for run in manifest['runs']:
        output = resolve_output(run)
        row = dict(name=run['name'], status=run['status'], performance=run['performance'], output_dir=run['output_dir'])
        if output:
            row['max_observed'] = observed_peak(output)
        if output and (output/'summary.json').is_file():
            summary = json.loads((output/'summary.json').read_text())
            best = summary.get('evaluation_selection_state', {}).get('best')
            row.update(status=('completed' if run['status'] in ('pending', 'running', 'completed') else run['status']),
                       transitions=summary['transitions'], elapsed_s=summary['elapsed_s'],
                       transitions_per_second=summary['transitions']/max(summary['elapsed_s'], 1e-9),
                       stop_reason=summary['stop_reason'], best_vector_step=best['vector_step'] if best else None,
                       best=compact_evaluation(best['evaluation']) if best else None,
                       final=compact_evaluation(summary.get('final_evaluation') or {}))
        elif output and (output/'best.json').is_file():
            best = json.loads((output/'best.json').read_text())
            row.update(best_vector_step=best['vector_step'], best=compact_evaluation(best['evaluation']))
        rows.append(row)
    report = dict(schema=1, requested_samples_per_arm=manifest['requested_samples_per_arm'],
                  transitions_per_arm=manifest['transitions_per_arm'], seed=manifest['seed'], runs=rows,
                  note='20 Hz music F1 is unchanged. 200 Hz micro-F1 uses held score targets and includes attack/release timing; do not compare the two rates as the same score. One seed/short budget does not establish the final winner.')
    root = Path(manifest['output_root'])
    write_json(root/'comparison.json', report)
    def fmt(value, decimals=4):
        return '—' if value is None else f'{value:.{decimals}f}'
    lines = ['# 奖励单项对照', '',
             f"每组预算：{manifest['transitions_per_arm']:,} transitions；种子：{manifest['seed']}；三个学习率均 3e-4。", '',
             '| 单项 | 状态 | 合格 best F1 | 观测最高 F1 | final F1 | final 200Hz micro-F1 | final 错键子步 | 耗时秒 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        best, final = row.get('best') or {}, row.get('final') or {}
        peak = row.get('max_observed') or {}
        lines.append(f"| {row['name']} | {row['status']} | {fmt(best.get('music_only_f1_min'))} | "
                     f"{fmt(peak.get('score'))} | {fmt(final.get('music_only_f1_min'))} | {fmt(final.get('music_only_substep_key_micro_f1'))} | "
                     f"{fmt(final.get('wrong_key_substeps'), 1)} | {fmt(row.get('elapsed_s'), 1)} |")
    lines += ['', '20 Hz 与 200 Hz 是不同采样口径；后者包含真实起音/离键延迟。提前定位组新增观测，虽然种子相同，网络初始张量并不相同。',
              'F1 使用三回合中最低音乐段得分；合格 best 需满足原键底诊断门槛，观测最高得分不代表满足该门槛。',
              '本表保留所有组的原 20 Hz F1；单种子、短预算只用于首轮筛选。完整命令、输出路径与配置路径见 manifest.json。', '']
    (root/'comparison.md').write_text('\n'.join(lines))
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    if args.summarize:
        manifest = json.loads(args.summarize.read_text())
        summarize(manifest)
        write_json(args.summarize, manifest)
        return 0
    output = (args.output_root or ROOT/'training_mjlab/outputs'/datetime.now(
        ZoneInfo('Asia/Shanghai')).strftime('reward_comparison-%Y%m%d-%H%M%S-%f')).resolve()
    manifest = make_plan(args, output)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'manifest.json'
    write_json(path, manifest)
    if args.dry_run:
        summarize(manifest)
        print(path, flush=True)
        return 0
    manifest['source_archive'] = create_source_archive(output)
    write_json(path, manifest)
    for run in manifest['runs']:
        try:
            verify_sources(manifest)
        except RuntimeError as error:
            run.update(status='source_changed', error=str(error))
            write_json(path, manifest)
            summarize(manifest)
            print(str(error), file=sys.stderr, flush=True)
            return 2
        Path(run['output_base']).parent.mkdir()
        run.update(status='running', started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write_json(path, manifest)
        print(f"Starting {run['name']}: log={run['log']}", flush=True)
        with Path(run['log']).open('w') as log:
            try:
                process = subprocess.Popen(run['argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                run['pid'] = process.pid
                write_json(path, manifest)
                code = process.wait()
            except KeyboardInterrupt:
                if 'process' in locals() and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                run['status'] = 'interrupted'
                write_json(path, manifest)
                summarize(manifest)
                return 130
        run.update(status='completed' if code == 0 else 'failed', returncode=code,
                   completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        try:
            verify_sources(manifest)
        except RuntimeError as error:
            run.update(status='source_changed', error=str(error))
            code = 2
        report = summarize(manifest)
        write_json(path, manifest)
        print(json.dumps(next(row for row in report['runs'] if row['name'] == run['name']),
                         ensure_ascii=False), flush=True)
        if code:
            return code
    print(f'Comparison complete: {output/"comparison.md"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
