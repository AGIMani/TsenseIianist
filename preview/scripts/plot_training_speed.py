#!/usr/bin/env python3
"""Plot saved GPU benchmark evidence; does not run training or use the GPU."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'preview/outputs/diagnostics/gpu_ceiling_20260926'


def main():
    capacity = []
    for path in (ROOT/'preview/outputs/diagnostics/gpu_capacity_20260926').glob('collection_scan_v2-n*.json'):
        row = json.loads(path.read_text())
        if row.get('status') == 'ok':
            capacity.append(row)
    capacity.sort(key=lambda row: row['num_envs'])
    base = json.loads((OUT/'default512_sustained/analysis.json').read_text())
    learner = json.loads((OUT/'learner_candidates/layout_exact_results.json').read_text())
    ceiling = learner['candidates']['256']['timings'][-1]['median_updates_per_second']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    ax = axes[0]
    ax.plot([r['num_envs'] for r in capacity],
            [r['policy_plus_environment']['samples_per_second']/1000 for r in capacity],
            'o-', color='#176b87', label='Policy + environment (no SAC updates)')
    ax.plot([r['num_envs'] for r in capacity],
            [r['environment_recorded_actions']['samples_per_second']/1000 for r in capacity],
            's--', color='#7a9e9f', label='Environment, recorded actions')
    ax.axhline(ceiling/1000, color='#b84c42', linestyle=':',
               label='Current learner rate: 1 update / transition')
    ax.set_xscale('log', base=2)
    ax.set_xticks([r['num_envs'] for r in capacity], [str(r['num_envs']) for r in capacity])
    ax.set(xlabel='Parallel environments', ylabel='Thousand transitions / second',
           title='Collection capacity is not training throughput')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=.2)
    ax = axes[1]
    labels = ['512: full trainer', '512: ordinary steps', 'Learner only']
    values = [base['trainer_total_samples_per_second'], base['stable_samples_per_second'], ceiling]
    colors = ['#78909c', '#176b87', '#b84c42']
    candidate_path = OUT/'scale2048_same_budget/analysis.json'
    if candidate_path.exists():
        candidate = json.loads(candidate_path.read_text())
        labels.insert(2, '2048: ordinary steps*')
        values.insert(2, candidate['stable_samples_per_second'])
        colors.insert(2, '#4c956c')
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, fmt='%.0f', padding=4)
    ax.invert_yaxis()
    ax.set(xlim=(0, max(values)*1.13), xlabel='Transitions / second (learner: updates / second)',
           title='FP32, batch256, unchanged SAC update ratio')
    ax.grid(axis='x', alpha=.2)
    fig.suptitle('RTX PRO 6000 Blackwell — measured 2026-09-26', fontsize=13)
    fig.supxlabel('*Scale diagnostic only; collection/update grouping changes. Formal default remains 512.', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(OUT/f'throughput_comparison.{suffix}', dpi=160)
    print(OUT/'throughput_comparison.png')


if __name__ == '__main__':
    main()
