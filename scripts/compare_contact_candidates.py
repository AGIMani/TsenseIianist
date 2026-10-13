#!/usr/bin/env python3
"""Collect already measured baseline/candidate contact audits without GPU use."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate-roots',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();directories=[('baseline',a.baseline)]
    for root in a.candidate_roots:
        directories.extend((directory.name,directory) for directory in sorted(root.iterdir())
                           if (directory/'best/detail-0/motion_contact_report.json').is_file())
    rows=[]
    for name,directory in directories:
        detail=directory/'best/detail-0'
        r=json.loads((detail/'motion_contact_report.json').read_text());m=json.loads((detail/'report.json').read_text())
        j=next(value for value in r['joints'] if value['name']=='L_Thumb_Rot2')
        rows.append(dict(name=name,directory=str(directory.resolve()),music_f1=r['canonical_music_f1'],
            music_micro_f1=m['metrics']['music']['control_20hz']['micro_f1'],
            prep_wrong_control_frames=m['metrics']['preparation']['control_20hz']['frames_with_wrong_keys'],
            prep_wrong_physics_frames=m['metrics']['preparation']['diagnostic_200hz']['frames_with_wrong_keys'],
            prep_wrong_physics_key_samples=m['metrics']['preparation']['diagnostic_200hz']['wrong_key_frames'],
            prep_wrong_note_on_events=[event for event in m['metrics']['preparation']['note_on_events'] if event['wrong']],
            side_under_seconds=r['contact_categories']['key_side_or_under']['duration'],
            underside_seconds=r['contact_categories']['key_underside']['duration'],
            non_distal_seconds=r['contact_categories']['non_distal_link_key']['duration'],
            self_collision_seconds=r['contact_categories']['same_hand']['duration'],
            max_wrist_tx_step=m['motion']['wrist_tx']['max_target_jump_20hz'],
            max_wrist_tx_speed=m['motion']['wrist_tx']['max_velocity_200hz'],
            max_wrist_ty_speed=m['motion']['wrist_ty']['max_velocity_200hz'],
            max_pitch_speed=m['motion']['wrist_pitch']['max_velocity_200hz'],
            max_thumb_rot2_lag=j['absolute_lag']['max'],thumb_rot2_force_saturation_fraction=j['force_saturation_fraction'],
            longest_side_under_interval=max((interval['duration'] for pair in r['hand_piano_contacts'] for interval in pair['side_or_under_intervals']),default=0),
            scope='One detailed rollout per row; explicit changed-controller candidates, no gradient updates; canonical repeats listed separately.'))
    a.output.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
