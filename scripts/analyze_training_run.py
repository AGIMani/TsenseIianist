#!/usr/bin/env python3
"""CPU-only audit of training metrics, score phases and saved checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run/'metrics.jsonl').read_text().splitlines()]
    evaluations = [row for row in rows if 'eval' in row]
    summary = json.loads((args.run/'summary.json').read_text())
    config = json.loads((args.run/'config.json').read_text())
    final_step = summary['vector_steps']
    records = []
    for row in evaluations:
        step, metrics = row['vector_step'], row['eval']
        checkpoint = args.run/('final.pt' if step == final_step else f'step-{step}.pt')
        records.append(dict(vector_step=step, transitions=row['transitions'], updates=row['updates'],
                            checkpoint=str(checkpoint.resolve()) if checkpoint.exists() else None,
                            **metrics))
    ranking = sorted(records, key=lambda row: row['music_only_f1'], reverse=True)
    saved = [row for row in ranking if row['checkpoint']]
    windows = []
    for low, high in ((0,1000),(1000,3000),(3000,5000),(5000,7000),(7000,9000),(9000,final_step)):
        selected = [row for row in records if low < row['vector_step'] <= high]
        windows.append(dict(from_step_exclusive=low, through_step=high, evaluations=len(selected),
                            mean_music_f1=statistics.mean(row['music_only_f1'] for row in selected),
                            median_music_f1=statistics.median(row['music_only_f1'] for row in selected),
                            min_music_f1=min(row['music_only_f1'] for row in selected),
                            max_music_f1=max(row['music_only_f1'] for row in selected),
                            evaluations_with_preparation_mispresses=sum(row['preparation_wrong_key_frames']>0 for row in selected),
                            mean_preparation_wrong_frames=statistics.mean(row['preparation_wrong_key_frames'] for row in selected)))
    crossings = {str(threshold): next((row['vector_step'] for row in records if row['music_only_f1'] >= threshold), None)
                 for threshold in (.5,.8,.9,.95,.97,.99)}
    total_f1_bias = [row['f1']-row['music_only_f1'] for row in records]
    report = dict(run=str(args.run.resolve()), summary=summary, evaluations=len(records),
                  effective_config={key:config.get(key) for key in (
                      'sac','acceleration','batch_size','num_envs','replay_capacity','steady_updates_per_transition',
                      'preparation','action_profile','reward_config','source_runtime')},
                  metric_definitions=dict(f1='mean per-control-frame F1 over preparation and music',
                      music_only_f1='mean per-control-frame F1 on 158 original music frames',
                      micro_f1='aggregate TP/FP/FN across key frames, not the mean frame F1'),
                  windows=windows, first_reaches_music_f1=crossings,
                  best_logged=ranking[0], best_saved=saved[0], final=records[-1],
                  saved_ranking=saved, top_logged=ranking[:30],
                  final_drop_from_logged_best=ranking[0]['music_only_f1']-records[-1]['music_only_f1'],
                  final_drop_from_saved_best=saved[0]['music_only_f1']-records[-1]['music_only_f1'],
                  mean_preparation_contribution_to_total_f1=statistics.mean(total_f1_bias),
                  candidate_checkpoints=[row['checkpoint'] for row in saved[:3]]+[records[-1]['checkpoint']],
                  limitations=[
                      'Each historical evaluation is one deterministic GPU rollout; floating-point contact variation requires repeated re-evaluation.',
                      'Training transitions were not saved; metrics do not reveal every stochastic training action.',
                      'Motion fields are maxima sampled at 20Hz control endpoints, not 200Hz peak velocities.',
                      'The logged best step has no checkpoint if checkpoint=null; it cannot be reconstructed from nearby saved weights.',
                      'No logged music_only_f1 >= .99 does not prove the task is impossible; it states this run did not meet the target.',
                  ])
    (args.output/'training_curve_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'evaluations.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[row['transitions']/1e6 for row in records]
    fig, axes=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
    axes[0].plot(x,[row['f1'] for row in records],alpha=.5,label='Whole episode F1')
    axes[0].plot(x,[row['music_only_f1'] for row in records],lw=1,label='Music-only F1')
    axes[0].scatter([row['transitions']/1e6 for row in saved],[row['music_only_f1'] for row in saved],s=24,c='black',label='Saved checkpoints',zorder=3)
    axes[0].axhline(.99,color='red',ls='--',label='Requested .99');axes[0].legend(loc='lower right');axes[0].set_ylabel('Mean frame F1')
    axes[1].plot(x,[row['preparation_wrong_key_frames'] for row in records],label='Wrong frames in first 1 s')
    axes[1].set_ylabel('Preparation wrong frames /20');axes[1].legend()
    axes[2].plot(x,[row['motion_max_wrist_tx_speed_m_s'] for row in records],label='TX velocity max')
    axes[2].plot(x,[row['motion_max_wrist_ty_speed_m_s'] for row in records],label='TY velocity max')
    axes[2].set_ylabel('20Hz endpoint velocity (m/s)');axes[2].set_xlabel('Collected transitions (millions)');axes[2].legend()
    for axis in axes:axis.grid(alpha=.25)
    fig.suptitle('Twinkle policy-preparation training: learning, retained weights and motion')
    fig.savefig(args.output/'training_curves.png',dpi=160)
    print(json.dumps({key:report[key] for key in ('first_reaches_music_f1','best_logged','best_saved','final_drop_from_logged_best','candidate_checkpoints')},indent=2))


if __name__ == '__main__':
    main()
