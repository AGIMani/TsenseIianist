#!/usr/bin/env python3
"""Plot already recorded IK physics against its verified original MIDI events.

Pure array analysis: never imports MuJoCo or loads a physical model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def edges(mask, times, dt):
    starts=np.flatnonzero(np.diff(np.r_[False,mask].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[mask,False].astype(int))==-1)
    return [[float(times[s]),float(times[e]+dt)] for s,e in zip(starts,ends)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();out=args.directory
    path=out/'physical_trace.npz';z=np.load(path,allow_pickle=False)
    report=json.loads((out/'replay_report.json').read_text())
    times=z['time'];actual=z['actual_keys'].astype(bool);target=z['target_keys'].astype(bool)
    dt=float(np.median(np.diff(times)))
    events=z['original_events'];lo=max(21,int(events[:,2].min()+21)-2);hi=min(108,int(events[:,2].max()+21)+2)
    tp=int((actual&target).sum());fp=int((actual&~target).sum());fn=int((~actual&target).sum())
    f1=2*tp/max(2*tp+fp+fn,1)
    if not np.isclose(f1,report['original_midi_micro_f1'],rtol=0,atol=1e-12):
        raise ValueError('Recorded arrays and report F1 differ')
    measured=[]
    for k in range(88):
        for start,end in edges(actual[:,k],times,dt):measured.append([start,end,k+21])
    provenance=dict(reference='unmodified original MIDI, verified note times and pitches',
        original_midi_sha256=report['source']['original_midi_sha256'],
        score_sha256=report['source']['score_sha256'],
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sample_hz=1/dt,time_origin='original MIDI t=0; negative-time preparation excluded',
        actual_events=measured,original_events=[[float(s),float(e),int(k+21),int(f)] for s,e,k,f in events],
        tp=tp,fp=fp,fn=fn,micro_f1=f1,full_piece=report['full_piece'])
    (out/'comparison_events.json').write_text(json.dumps(provenance,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(18,7),sharex=True,sharey=True,layout='constrained')
    extent=(times[0],times[-1]+dt,20.5,108.5)
    axes[0].imshow(target.T,origin='lower',aspect='auto',interpolation='nearest',extent=extent,
        cmap=ListedColormap(['#f7f9fc','#2475bd']),vmin=0,vmax=1)
    categories=np.zeros(actual.shape,dtype=np.uint8)
    categories[actual&target]=1;categories[actual&~target]=2;categories[~actual&target]=3
    axes[1].imshow(categories.T,origin='lower',aspect='auto',interpolation='nearest',extent=extent,
        cmap=ListedColormap(['#f7f9fc','#26956b','#da4b48','#f3ce80']),vmin=0,vmax=3)
    axes[0].set_title('Original MIDI • exact source times and pitches',loc='left')
    axes[1].set_title('Physical keys • green: correct activation   red: extra key   amber: missed key',loc='left')
    for ax in axes:
        ax.set_ylim(lo-.5,hi+.5);ax.set_ylabel('MIDI pitch');ax.set_yticks(np.arange((lo//12)*12,hi+1,12))
        ax.grid(axis='x',alpha=.15);ax.set_xlim(times[0],times[-1]+dt)
    axes[1].set_xlabel('Original MIDI time (seconds); preparation precedes t=0')
    status='FULL PIECE' if report['full_piece'] else 'PARTIAL DIAGNOSTIC'
    fig.suptitle(f'015-5 | {status} | physical micro-F1 {f1:.5f} | {1/dt:.0f} Hz actual samples\n'
                 f'Trajectory {path.parent.name} · SHA256 {provenance["trace_sha256"][:16]} · no RL checkpoint',fontsize=13)
    fig.savefig(out/'actual_vs_original_midi.png',dpi=200)
    plt.close(fig)
    print(json.dumps({k:provenance[k] for k in ('micro_f1','tp','fp','fn','trace_sha256')},indent=2))


if __name__=='__main__':main()
