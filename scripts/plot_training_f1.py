#!/usr/bin/env python3
"""Plot logged evaluation F1, without loading a checkpoint or changing training."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def portable_path(value, root):
    path = Path(value)
    if not path.is_absolute():
        return root / path
    # Frozen configs can retain the original absolute installation path.
    for marker in ('training_mjlab', 'runs'):
        if marker in path.parts:
            return root.joinpath(*path.parts[path.parts.index(marker):])
    return path


def effective_self_collision_weight(config):
    """The complete reward contract takes precedence over its legacy CLI shortcut."""
    terms = config.get('reward_config', {}).get('terms', {})
    if 'self_collision_reward' in terms:
        term = terms['self_collision_reward']
        return term['weight'] if term['enabled'] else 0.
    return config.get('self_collision_weight', 0.)


def inspect_run(path, root, backend, formal):
    config = read_json(path / 'config.json')
    summary = read_json(path / 'summary.json')
    interruption = read_json(path / 'interruption.json')
    rows = []
    for line_number, line in enumerate((path / 'metrics.jsonl').read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f'{path}/metrics.jsonl:{line_number}: {error}') from error
    points = []
    for row in rows:
        if 'eval' not in row:
            continue
        metric = row['eval']
        f1 = metric['f1']
        if not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError(f'Invalid F1 in {path}: {f1}')
        points.append(dict(step=row.get('vector_step', row.get('step')),
                           transitions=row.get('transitions', row.get('step')),
                           updates=row.get('updates'), **metric))
    # Preserve source order on equal sample counts; never smooth or extrapolate.
    points.sort(key=lambda item: item['transitions'])
    task = {}
    if config.get('cli', {}).get('task'):
        task = read_json(portable_path(config['cli']['task'], root) / 'task.json')
    environment = task.get('contract', config.get('contract', {})).get('environment', {})
    midi = environment.get('midi', 'unknown')
    song = '月光（简化版）' if 'clair' in midi.lower() else ('小星星' if 'twinkle' in midi.lower() else midi)
    control_dt = task.get('control_dt', environment.get('control_timestep'))
    physics_dt = task.get('physics_dt')
    control_hz = round(1 / control_dt) if control_dt else None
    physics_hz = round(1 / physics_dt) if physics_dt else None
    group = (f'{song} · MJLab · {control_hz}/{physics_hz} Hz' if backend == 'mjlab'
             else '历史 JAX · 小星星 · 单独展示')
    status = summary.get('status') or interruption.get('status') or ('completed' if summary else 'unknown')
    latest_samples = max((r.get('transitions', r.get('step', 0)) for r in rows), default=0)
    best = max(points, key=lambda item: item['f1']) if points else None
    checkpoint = None
    if best:
        ext = 'pt' if backend == 'mjlab' else 'msgpack'
        candidate = path / f'step-{best["step"]}.{ext}'
        if not candidate.exists() and summary.get('vector_steps') == best['step']:
            candidate = path / 'final.pt'
        if candidate.exists():
            checkpoint = str(candidate.relative_to(root))
    nonfinite = []
    for row in rows:
        for key, value in row.get('train', {}).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                nonfinite.append(dict(step=row.get('vector_step'), metric=key))
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in ('config.json', 'metrics.jsonl', 'summary.json', 'interruption.json')
              if (path / name).exists()}
    run_name = path.name if path.name != 'training' else path.parent.name
    return dict(run=run_name, path=str(path.relative_to(root)), backend=backend, formal=formal,
                status=status, song=song, group=group, control_hz=control_hz, physics_hz=physics_hz,
                action_profile=config.get('action_profile', 'legacy'),
                self_collision_weight=effective_self_collision_weight(config),
                seed=config.get('cli', {}).get('seed'),
                last_transitions=summary.get('transitions', summary.get('steps', latest_samples)),
                elapsed_s=summary.get('elapsed_s'), evaluation_count=len(points),
                best=best, last=points[-1] if points else None, best_checkpoint=checkpoint,
                nonfinite_train_metrics=nonfinite, points=points, source_sha256=hashes)


def label(run):
    if run['backend'] == 'jax':
        return '09-17 TZ−25mm' if 'tz025' in run['run'] else '09-17 11:19 原始配置'
    pieces = run['run'].split('-')
    date, time = pieces[1:3]
    name = f'{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}'
    return f'{name} | {run["action_profile"]} | 自碰撞 {run["self_collision_weight"]:g}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New directory, to preserve previous reports')
    args = parser.parse_args()
    root = args.root.resolve()
    runs = []
    for metrics in sorted((root / 'training_mjlab' / 'outputs').rglob('metrics.jsonl')):
        path = metrics.parent
        formal = path.parent == root / 'training_mjlab' / 'outputs' and path.name.startswith('sac-')
        runs.append(inspect_run(path, root, 'mjlab', formal))
    for metrics in sorted((root / 'runs').rglob('metrics.jsonl')):
        runs.append(inspect_run(metrics.parent, root, 'jax', False))
    selected = [run for run in runs if (run['formal'] or run['backend'] == 'jax') and run['points']]
    if not selected:
        parser.error('No logged evaluations found')
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(created_at=datetime.now().astimezone().isoformat(),
                  metric='eval.f1: mean frame key F1; raw points, no smoothing',
                  x_axis='environment transitions, not vector steps or gradient updates', runs=runs)
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    fields = ['run', 'backend', 'formal', 'group', 'step', 'transitions', 'updates',
              'f1', 'precision', 'recall', 'key_frame_micro_f1', 'return', 'length']
    with (args.output / 'evaluation_points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for run in runs:
            for point in run['points']:
                writer.writerow({**run, **point})
    inventory_fields = ['run', 'path', 'backend', 'formal', 'status', 'last_transitions',
                        'evaluation_count', 'elapsed_s', 'best_checkpoint']
    with (args.output / 'run_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=inventory_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(runs)
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    groups = list(dict.fromkeys(run['group'] for run in selected))
    fig, axes = plt.subplots(math.ceil(len(groups)/2), 2, figsize=(16, 5*math.ceil(len(groups)/2)),
                             squeeze=False, layout='constrained')
    for ax, group in zip(axes.flat, groups):
        subset = [run for run in selected if run['group'] == group]
        for run in subset:
            points = run['points']
            x = [p['transitions']/1e6 for p in points]
            y = [p['f1'] for p in points]
            suffix = '（主动停止，仅一次评估）' if run['status'].startswith('stopped') else ''
            line, = ax.plot(x, y, marker='o', markersize=4, linewidth=1.8,
                            label=f'{label(run)}{suffix}\n最佳 {run["best"]["f1"]:.4f} / 最后 {y[-1]:.4f}')
            ax.scatter(run['best']['transitions']/1e6, run['best']['f1'], marker='*', s=190,
                       color=line.get_color(), edgecolor='white', linewidth=.6, zorder=5)
        ax.set(title=group, xlabel='累计环境采样数（百万）', ylabel='评估 F1（逐帧平均）', ylim=(0, 1), xlim=(0, None))
        ax.grid(alpha=.2)
        ax.legend(loc='best', fontsize=9)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle('V3.2 训练 F1 总览｜★ 最佳已记录评估｜各面板配置不同，请勿直接作单因素对照', fontsize=16)
    for ext in ('png', 'pdf'):
        fig.savefig(args.output / f'f1_curves.{ext}', dpi=170)
    plt.close(fig)
    clair = [r for r in selected if r['song'] == '月光（简化版）']
    if clair:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), layout='constrained')
        for ax, key, title in zip(axes, ('f1', 'recall'), ('F1 局部放大', 'Recall：目标按键召回率')):
            for run in clair:
                ax.plot([p['transitions']/1e6 for p in run['points']], [p[key] for p in run['points']],
                        marker='o', markersize=4, label=label(run))
            ax.set(title=title, xlabel='累计环境采样数（百万）', ylabel=key, xlim=(0, None), ylim=(0, None))
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
        fig.suptitle('《月光》两轮训练：最终 F1 与召回率均低于首个评估点（纵轴放大）', fontsize=14)
        fig.savefig(args.output / 'clair_detail.png', dpi=170)
        plt.close(fig)
    print(json.dumps(dict(output=str(args.output.resolve()), runs=len(runs),
                          formal_mjlab=sum(r['formal'] for r in runs),
                          plotted_points=sum(r['evaluation_count'] for r in selected)), ensure_ascii=False))


if __name__ == '__main__':
    main()
