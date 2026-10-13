#!/usr/bin/env python3
"""CPU summary/plots of existing archived-source checkpoint audit recordings."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    a=p.parse_args();audit=json.loads((a.audit/'checkpoint_audit.json').read_text())
    table=[];errors=[];initial_states={}
    for checkpoint in audit['checkpoints']:
        name=Path(checkpoint['checkpoint']).stem;directory=a.audit/name/'detail-0'
        r=json.loads((directory/'report.json').read_text());z=np.load(directory/'rollout.npz')
        with np.load(directory/'initial.npz') as initial:
            initial_states[name]={key:initial[key].copy() for key in initial.files}
        music=r['metrics']['music'];prep=r['metrics']['preparation']
        same=[pair for pair in r['contact_pairs'] if pair['category']=='same_hand']
        row=dict(checkpoint=name,canonical_music_f1_mean=checkpoint['mean_music_f1'],
                 canonical_music_f1_min=checkpoint['min_music_f1'],canonical_music_f1_max=checkpoint['max_music_f1'],
                 detail_music_f1_20hz=music['control_20hz']['mean_frame_f1'],
                 detail_music_f1_200hz=music['diagnostic_200hz']['mean_frame_f1'],
                 music_wrong_key_frames_20hz=music['control_20hz']['wrong_key_frames'],
                 music_missed_key_frames_20hz=music['control_20hz']['missed_key_frames'],
                 music_wrong_key_frames_200hz=music['diagnostic_200hz']['wrong_key_frames'],
                 music_wrong_note_on_events=sum(e['wrong'] for e in music['note_on_events']),
                 preparation_wrong_note_on_events=sum(e['wrong'] for e in prep['note_on_events']),
                 music_self_contact_substeps=music['contact_frames_200hz']['same_hand'],
                 preparation_self_contact_substeps=prep['contact_frames_200hz']['same_hand'],
                 cross_hand_contact_substeps=sum(r['metrics'][phase]['contact_frames_200hz']['cross_hand'] for phase in ('preparation','music')),
                 maximum_same_hand_penetration_mm=max((pair['max_penetration_m']*1000 for pair in same),default=0),
                 first_same_hand_contact_seconds=min((interval['start_seconds'] for pair in same for interval in pair['intervals']),default=None),
                 wrist_tx_velocity_max_m_s=r['motion']['wrist_tx']['max_velocity_200hz'],
                 wrist_ty_velocity_max_m_s=r['motion']['wrist_ty']['max_velocity_200hz'],
                 finger_velocity_max_rad_s=r['motion']['finger']['max_velocity_200hz'],
                 wrist_tx_target_jump_max_cm=r['motion']['wrist_tx']['max_target_jump_20hz']*100,
                 wrist_tx_first_target_jump_cm=r['motion']['wrist_tx']['first_command_max_jump']*100,
                 top_same_hand_pairs=same[:5])
        table.append(row)
        goal=z['goals'][:,:88].astype(bool);actual=z['actual'].astype(bool)
        for i in np.flatnonzero((goal!=actual).any(1)):
            errors.append(dict(checkpoint=name,score_index=int(i),
                               policy_command_time_seconds=float(i*.05),sample_end_seconds=float((i+1)*.05),
                               music_sample_end_seconds=float((i+1)*.05-1),
                               phase='preparation' if i<20 else 'music',
                               target_midi=' '.join(str(int(k)+21) for k in np.flatnonzero(goal[i])),
                               actual_midi=' '.join(str(int(k)+21) for k in np.flatnonzero(actual[i])),
                               missed_midi=' '.join(str(int(k)+21) for k in np.flatnonzero(goal[i]&~actual[i])),
                               wrong_midi=' '.join(str(int(k)+21) for k in np.flatnonzero(~goal[i]&actual[i]))))
    table.sort(key=lambda row:row['canonical_music_f1_mean'],reverse=True)
    initial_reference=table[0]['checkpoint'];base=initial_states[initial_reference]
    consistency=dict(reference_checkpoint=initial_reference,
                     initials_identical_to_reference={name:{key:bool(np.array_equal(value[key],base[key])) for key in base}
                                                      for name,value in initial_states.items()},
                     initial_max_abs_qvel=float(np.abs(base['qvel']).max()),
                     initial_shapes={key:list(value.shape) for key,value in base.items()},
                     limitations=['These recorded arrays do not include the reset contact list; a CPU mj_forward check is reported separately.'])
    (a.audit/'initial_state_consistency.json').write_text(json.dumps(consistency,indent=2)+'\n')
    (a.audit/'comparison.json').write_text(json.dumps(table,indent=2)+'\n')
    with (a.audit/'error_frames_20hz.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors=ListedColormap(['#f8f9fa','#6fbc8b','#d94141','#e6a02c'])
    fig,axes=plt.subplots(2,2,figsize=(15,7),sharex=True,constrained_layout=True)
    for row,name in enumerate((table[0]['checkpoint'],'final')):
        directory=a.audit/name/'detail-0';z=np.load(directory/'rollout.npz');ph=np.load(directory/'physics_200hz.npz')
        goal=z['goals'][:,:88].astype(bool)
        for col,actual in enumerate((z['actual'].astype(bool),ph['key_activation'])):
            target=goal if col==0 else np.repeat(goal,10,axis=0)
            shown=(target&actual).astype(int)+2*(~target&actual)+3*(target&~actual)
            axes[row,col].imshow(shown[:,24:62].T,origin='lower',aspect='auto',extent=(0,8.9,44.5,82.5),cmap=colors,vmin=0,vmax=3,interpolation='nearest')
            axes[row,col].axvline(1,color='black',ls='--',lw=.8)
            axes[row,col].set_title(f'{name}: {20 if col==0 else 200} Hz')
            axes[row,col].set_ylabel('MIDI note');axes[row,col].set_xlabel('Policy time (s), music starts at 1 s')
    fig.suptitle('Green = target pressed; red = unwanted press; orange = missed target\n200 Hz uses the held control-frame target, for diagnosis only')
    fig.savefig(a.audit/'best_final_key_errors.png',dpi=170)
    print(json.dumps([{k:v for k,v in row.items() if k!='top_same_hand_pairs'} for row in table],indent=2))


if __name__=='__main__':main()
