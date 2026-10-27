#!/usr/bin/env python3
"""Compare complete independent recordings on identical original-MIDI bars."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(baseline,candidate,baseline_bars,candidate_bars):
    for name in ('original_midi_sha256','model_sha256','archive_sha256','physics_hz','control_hz','recorded_end_s','original_score_end_s'):
        if baseline[name]!=candidate[name]:raise ValueError('Incomparable '+name)
    if not baseline['entire_score_recorded'] or not candidate['entire_score_recorded']:raise ValueError('Both recordings must cover the entire original score')
    if len(baseline_bars)!=len(candidate_bars):raise ValueError('Mismatched bar partition')
    columns=('micro_f1','tp','fp','fn','same_hand_contact_samples','cross_hand_contact_samples')
    def delta(left,right):return {k:right[k]-left[k] for k in columns}
    rows=[]
    for left,right in zip(baseline_bars,candidate_bars):
        if any(left[k]!=right[k] for k in ('bar','start_s','end_s','samples')):raise ValueError('Mismatched bar partition')
        rows.append(dict(bar=left['bar'],start_s=left['start_s'],end_s=left['end_s'],baseline=left,candidate=right,delta=delta(left,right)))
    return dict(kind='independent_continuous_fullscore_comparison_v1',baseline_trace_sha256=baseline['physical_trace_sha256'],
        candidate_trace_sha256=candidate['physical_trace_sha256'],original_midi_sha256=baseline['original_midi_sha256'],
        baseline_score_sha256=baseline['score_sha256'],candidate_score_sha256=candidate['score_sha256'],
        baseline_metrics=baseline['global_metrics'],candidate_metrics=candidate['global_metrics'],
        global_delta=delta(baseline['global_metrics'],candidate['global_metrics']),
        improved_bar_count=sum(r['delta']['micro_f1']>1e-12 for r in rows),
        regressed_bar_count=sum(r['delta']['micro_f1']<-1e-12 for r in rows),
        equal_bar_count=sum(abs(r['delta']['micro_f1'])<=1e-12 for r in rows),
        biggest_improvements=sorted([r for r in rows if r['delta']['micro_f1']>1e-12],key=lambda r:r['delta']['micro_f1'],reverse=True)[:15],
        biggest_regressions=sorted([r for r in rows if r['delta']['micro_f1']<-1e-12],key=lambda r:r['delta']['micro_f1'])[:15],
        bars=rows,causal_attribution_supported=False,
        limitations='Independent continuous physical runs; candidate changes fingering/reference and ULP seed-boundary lookup. Differences cannot be attributed solely to fingering. No reused physical segments, no mean-of-bar-F1 global score, no dynamic or hardware safety claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=compare(json.loads((a.baseline/'report.json').read_text()),json.loads((a.candidate/'report.json').read_text()),
                   json.loads((a.baseline/'bars.json').read_text()),json.loads((a.candidate/'bars.json').read_text()))
    result['inputs_sha256']={str(p.resolve()):sha(p) for folder in (a.baseline,a.candidate) for p in (folder/'report.json',folder/'bars.json')}
    result['script_sha256']=sha(__file__);a.output.mkdir(parents=True)
    bars=result.pop('bars');(a.output/'per_bar.json').write_text(json.dumps(bars,indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(17,9),sharex=True,layout='constrained')
    x=[(r['start_s']+r['end_s'])/2 for r in bars]
    for label in ('baseline','candidate'):axes[0].plot(x,[r[label]['micro_f1'] for r in bars],label=label)
    axes[0].set_ylabel('Per-bar F1');axes[0].set_ylim(0,1);axes[0].legend()
    axes[1].bar(x,[r['delta']['micro_f1'] for r in bars],width=.6);axes[1].set_ylabel('Candidate − baseline F1')
    for key in ('same_hand_contact_samples','cross_hand_contact_samples'):axes[2].plot(x,[r['delta'][key] for r in bars],label=key)
    axes[2].set_ylabel('Contact sample change');axes[2].set_xlabel('Original MIDI seconds');axes[2].legend()
    for axis in axes:axis.grid(alpha=.2);axis.axhline(0,color='black',linewidth=.4)
    fig.suptitle('Two independent full-score physical recordings | same MIDI/time/model\nFingering + reference + seed-boundary fix differ; no isolated causal attribution or safety claim')
    fig.savefig(a.output/'comparison_summary.png',dpi=150);plt.close(fig)
    (a.output/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}),indent=2)+'\n')
    print(json.dumps(result['global_delta']))


if __name__=='__main__':main()
