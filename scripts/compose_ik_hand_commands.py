#!/usr/bin/env python3
"""Reuse an existing control solution outside explicitly revised thumb passages.

Only actuator command proposals are composed. The result always requires a fresh
physical replay against the revised score, including collision and activation checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(x):
    x=np.clip(x,0,1)
    return x*x*x*(10+x*(-15+6*x))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--revision',type=Path,required=True)
    ap.add_argument('--mode',choices=('wrists','hands'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    base=np.load(a.base/'physical_trace.npz',allow_pickle=False)
    ref=np.load(a.reference/'physical_trace.npz',allow_pickle=False)
    t=base['all_command_times']
    if not np.array_equal(t,ref['all_command_times']):raise ValueError('Command time mismatch')
    if not np.array_equal(base['original_events'][:,:3],ref['original_events'][:,:3]):
        raise ValueError('Original MIDI changed')
    revision=json.loads(a.revision.read_text())
    report=json.loads((a.reference/'replay_report.json').read_text())
    manifest=json.loads((Path(report['source']['build'])/'scene/manifest.json').read_text())
    names=manifest['action_names']
    x=base['all_commands'];y=ref['all_commands']
    if x.shape!=y.shape or x.shape[1]!=len(names):raise ValueError('Command shape mismatch')
    weights=np.zeros((len(t),2))
    for note in revision['selected']:
        side=note['part']//5
        gate=smooth((t-(note['start_s']-.55))/.2)*smooth(((note['end_s']+.35)-t)/.2)
        weights[:,side]=np.maximum(weights[:,side],gate)
    result=x.copy() if a.mode=='wrists' else y.copy()
    for i,name in enumerate(names):
        if not name.startswith(('R_','L_')):continue
        side=int(name.startswith('L_'))
        if a.mode=='wrists':
            if name.rsplit('_',1)[-1] not in ('Tx','Ty','Pitch'):continue
        else:
            if name.startswith(('R_Thumb_','L_Thumb_')) or '_Palm_Base_' in name:continue
        result[:,i]=(1-weights[:,side])*x[:,i]+weights[:,side]*y[:,i]
    if not np.isfinite(result).all():raise ValueError('Nonfinite command')
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'commands.npz',time=t,commands=result,
                        all_command_times=t,all_commands=result,thumb_passage_weights=weights)
    (a.output/'source.json').write_text(json.dumps(dict(
        mode=a.mode,source_sha256=sha(__file__),base_trace_sha256=sha(a.base/'physical_trace.npz'),
        reference_trace_sha256=sha(a.reference/'physical_trace.npz'),
        revision_sha256=sha(a.revision),before_s=.55,after_s=.35,blend_s=.2,
        physical_validation='Pending; command composition is not a trajectory acceptance test'),indent=2)+'\n')


if __name__=='__main__':main()
