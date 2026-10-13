#!/usr/bin/env python3
"""Plot the two current 47-action Twinkle runs and their legacy baseline."""
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
RUNS = (
    ('legacy_initial', 'sac-twinkle-no-yaw47-20260924-120155', 0,
     '无补偿：首轮 5M', '#2663a3'),
    ('legacy_continuation', 'sac-twinkle-no-yaw47-legacy-continue5m-20260924-144738',
     5_000_192, '无补偿：再续训 5M', '#ec8b23'),
    ('gravity_initial', 'sac-twinkle-no-yaw47-gravcomp-20260924', 0,
     '重力补偿：从头 5M', '#269969'),
)


def read_run(root, kind, name, offset):
    directory = root / 'training_mjlab/outputs' / name
    raw = (directory / 'metrics.jsonl').read_bytes()
    lines = raw.splitlines()
    points = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # A live trainer may be writing its final line.
            raise
        if 'eval' not in entry:
            continue
        result = entry['eval']
        f1 = float(result['f1'])
        if not math.isfinite(f1):
            raise ValueError(f'Nonfinite F1 in {directory}')
        points.append(dict(run=kind, run_dir=name, vector_step=entry['vector_step'],
                           run_transitions=entry['transitions'],
                           cumulative_transitions=offset + entry['transitions'],
                           updates=entry['updates'], f1=f1,
                           precision=result['precision'], recall=result['recall'],
                           key_frame_micro_f1=result['key_frame_micro_f1']))
    if not points:
        raise ValueError(f'No evaluation points in {directory}')
    summary = directory / 'summary.json'
    return points, dict(run=kind, path=str(directory),
                        metrics_sha256=hashlib.sha256(raw).hexdigest(),
                        completed=summary.is_file(), evaluations=len(points),
                        best=max(points, key=lambda point: point['f1']),
                        last=points[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory; existing reports are never overwritten')
    args = parser.parse_args()
    output = args.output.resolve()
    all_points, inventory, series = [], [], []
    for kind, name, offset, label, color in RUNS:
        points, row = read_run(ROOT, kind, name, offset)
        all_points.extend(points)
        inventory.append(row)
        series.append((kind, label, color, points))
    output.mkdir(parents=True, exist_ok=False)

    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), layout='constrained')
    for kind, label, color, points in series:
        scores = [point['f1'] for point in points]
        best = max(points, key=lambda point: point['f1'])
        axes[0].plot([point['cumulative_transitions'] / 1e6 for point in points],
                     scores, '-o', label=label, color=color, linewidth=2, markersize=4)
        axes[0].scatter(best['cumulative_transitions'] / 1e6, best['f1'], marker='*',
                        s=180, color=color, edgecolor='white', linewidth=.5, zorder=4)
        if kind != 'legacy_initial':
            axes[1].plot([point['run_transitions'] / 1e6 for point in points],
                         scores, '-o', label=f'{label} · 当前最佳 {best["f1"]:.3f}',
                         color=color, linewidth=2, markersize=4)
            axes[1].scatter(best['run_transitions'] / 1e6, best['f1'], marker='*',
                            s=180, color=color, edgecolor='white', linewidth=.5, zorder=4)
    axes[0].axvline(5.000192, color='#777777', linestyle='--', linewidth=1,
                    label='无补偿首轮结束')
    axes[0].set(title='累计训练采样', xlabel='累计环境采样（百万条）',
                ylabel='评估 F1（逐帧平均）', xlim=(0, 10.1), ylim=(0, 1))
    axes[1].set(title='当前两轮进度（各自起点）', xlabel='本轮环境采样（百万条）',
                ylabel='评估 F1（逐帧平均）', xlim=(0, 5.1), ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.22)
        axis.legend(fontsize=9, loc='lower right')
    fig.suptitle('V3.2 小星星 47 动作训练｜日志原始评估点，★为截至导出时的最佳', fontsize=15)
    fig.text(.5, .005,
             '无补偿续训从已训练 5M 的策略启动；两条当前曲线起点不同，不能单凭图形归因于重力补偿。',
             ha='center', fontsize=9)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'f1_curves.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    with (output / 'evaluation_points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_points[0]))
        writer.writeheader()
        writer.writerows(all_points)
    report = dict(created_at=datetime.now().astimezone().isoformat(),
                  metric='logged eval.f1, mean frame key F1',
                  axis='environment transitions, not vector steps', snapshot=True,
                  runs=inventory,
                  limitations=['Active runs may add evaluations after this snapshot',
                               'Legacy continuation starts from a 5M-sample pretrained policy; gravity run starts from scratch'])
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(output),
                          evaluations={entry['run']: entry['evaluations'] for entry in inventory}),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
