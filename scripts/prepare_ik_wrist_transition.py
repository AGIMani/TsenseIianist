#!/usr/bin/env python3
# 2026-10-25: Current palm-base joint names are canonical; archived recordings
# retain explicit legacy-name decoding. Targets here are joint values, not XYZ.
"""Create a smooth preparation command proposal; never modifies physical states."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def palm_base_columns(source, action_names):
    """Decode names from the recording, without renaming or migrating its model."""
    current = any('/palm_base_' in name for name in source['joints'])
    legacy = any('/forearm_' in name for name in source['joints'])
    if current and legacy:
        raise ValueError('Mixed palm-base naming in one recording')
    prefix = '/palm_base_' if current else '/forearm_'
    columns, initial = [], []
    for i, name in enumerate(source['joints']):
        if prefix not in name:
            continue
        side = 'R' if name.startswith('right_') else 'L'
        axis = {'tx': 'Tx', 'ty': 'Ty', 'pitch': 'Pitch'}[name.rsplit('_', 1)[-1]]
        columns.append(action_names.index(f'{side}_Palm_Base_{axis}'))
        initial.append(source['initial_physical_controlled_qpos'][i])
    if len(columns) != 6 or len(set(columns)) != 6:
        raise ValueError('Expected six active palm-base channels')
    return columns, initial, 'palm_base_v1' if current else 'archived_forearm'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recording',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--settle-time',type=float,default=-.5)
    a=ap.parse_args()
    trace=a.recording/'physical_trace.npz'
    z=np.load(trace,allow_pickle=False)
    report=json.loads((a.recording/'replay_report.json').read_text())
    source=report['source']
    manifest=json.loads((Path(source['build'])/'scene/manifest.json').read_text())
    times=z['all_command_times'];commands=z['all_commands'].copy()
    join=int(np.argmin(abs(times-a.settle_time)))
    if not times[0]<times[join]<0:raise ValueError('Transition must finish inside preparation')
    columns, initial, naming = palm_base_columns(source, manifest['action_names'])
    u=np.clip((times-times[0])/(times[join]-times[0]),0,1)
    blend=10*u**3-15*u**4+6*u**5
    mask=np.arange(len(times))<join
    initial=np.asarray(initial)
    commands[np.ix_(mask,columns)]=initial+(commands[join,columns]-initial)*blend[mask,None]
    if not np.array_equal(commands[~mask],z['all_commands'][~mask]):
        raise AssertionError('Commands after preparation changed')
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'commands.npz',time=times,commands=commands)
    metadata=dict(source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  start_s=float(times[0]),join_s=float(times[join]),
                  action_columns=columns,initial_palm_base_joint_targets=initial.tolist(),
                  recorded_joint_naming=naming,
                  physical_validation='Required: command proposal only')
    (a.output/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
