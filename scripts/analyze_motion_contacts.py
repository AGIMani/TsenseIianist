#!/usr/bin/env python3
"""CPU audit of extended recorded GPU trajectories; no physics re-simulation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def ranges(values):
    return dict(min=float(np.min(values)),p01=float(np.quantile(values,.01)),
                p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),
                p99=float(np.quantile(values,.99)),max=float(np.max(values)))


def intervals(mask,dt):
    changes=np.diff(np.r_[False,mask,False].astype(int))
    return [dict(first_tick=int(s),last_tick=int(e-1),start=(s+1)*dt,end=min(e+1,len(mask))*dt,
                 duration=(e-s)*dt) for s,e in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    args=p.parse_args();root=args.audit
    meta=json.loads((root/'model_indices.json').read_text());audit=json.loads((root/'checkpoint_audit.json').read_text())
    contract={c['name']:c for c in meta['action_contract']['channels']}
    summaries=[]
    for checkpoint in audit['checkpoints']:
        name=Path(checkpoint['checkpoint']).stem;directory=root/name/'detail-0'
        z=np.load(directory/'physics_200hz.npz');c=np.load(directory/'rollout.npz');initial=np.load(directory/'initial.npz')
        geometry=np.load(directory/'contact_geometry.npz')['contacts'];dt=float(z['physics_dt']);n=len(z['qpos'])
        control_dt=n/len(c['actions'])*dt
        report=json.loads((directory/'report.json').read_text())
        phase_frames=int(round(checkpoint['canonical_repeats'][0].get('music_start_frame',20)))
        joints=[];large_changes=[]
        for row in meta['joints']:
            aid,qid,vid=row['actuator_id'],row['qpos_adr'],row['qvel_adr']
            qpos=z['qpos'][:,qid];target=z['ctrl'][:,aid];speed=z['qvel'][:,vid];force=z['actuator_force'][:,aid]
            diff=np.diff(np.r_[initial['ctrl'][aid],c['ctrl'][:,aid]])
            channel=contract[row['name']];width=channel['physical_max']-channel['physical_min']
            limited=row['force_limited'];limits=np.array(row['force_range']);den=np.max(np.abs(limits))
            saturated=((force<=limits[0]+max(1e-6,abs(limits[0])*.01)) |
                       (force>=limits[1]-max(1e-6,abs(limits[1])*.01))) if limited else np.zeros(n,bool)
            joint_limited=row['joint_limited'];lower,upper=row['joint_range'];tolerance=max(1e-6,(upper-lower)*.01)
            within_edge=((qpos<=lower+tolerance)|(qpos>=upper-tolerance)) if joint_limited else np.zeros(n,bool)
            outside=((qpos<lower-1e-5)|(qpos>upper+1e-5)) if joint_limited else np.zeros(n,bool)
            lag=np.abs(target-qpos)
            data=dict(**row,physical_policy_range=[channel['physical_min'],channel['physical_max']],
                      qpos=ranges(qpos),ctrl=ranges(target),absolute_lag=ranges(lag),absolute_velocity=ranges(np.abs(speed)),
                      absolute_ctrl_change=ranges(np.abs(diff)),signed_ctrl_change=ranges(diff),
                      ctrl_change_width_fraction_max=float(np.max(np.abs(diff))/width),
                      policy_abs_ge_099_fraction=float(np.mean(np.abs(c['actions'][:,row['action_index']])>=.99)),
                      ctrl_within_1percent_policy_edge_fraction=float(np.mean((target<=channel['physical_min']+width*.01)|(target>=channel['physical_max']-width*.01))),
                      actual_within_1percent_mechanical_edge_fraction=float(within_edge.mean()),
                      actual_outside_joint_range_substeps=int(outside.sum()),
                      actuator_force=ranges(force),force_saturation_fraction=float(saturated.mean()),
                      longest_force_saturation_seconds=max([v['duration'] for v in intervals(saturated,dt)],default=0.))
            joints.append(data)
            for i in np.argsort(np.abs(diff))[-3:]:
                large_changes.append(dict(joint=row['name'],policy_command_seconds=float(i*control_dt),
                                          music_seconds=float(i*control_dt-phase_frames*control_dt),
                                          signed_delta=float(diff[i]),width_fraction=float(abs(diff[i])/width),
                                          from_target=float(initial['ctrl'][aid] if i==0 else c['ctrl'][i-1,aid]),
                                          to_target=float(c['ctrl'][i,aid])))
        names=meta['geom_names'];body_ids=meta['geom_body_ids'];hand_case=np.zeros(n,bool)
        hand_key=np.zeros(n,bool);key_top=np.zeros(n,bool);key_side=np.zeros(n,bool);key_under=np.zeros(n,bool);non_distal_key=np.zeros(n,bool)
        same_hand=np.zeros(n,bool);cross_hand=np.zeros(n,bool)
        pair_rows={};point_rows=[]
        for point in geometry:
            tick,g0,g1=int(point[0]),int(point[1]),int(point[2]);key=int(point[10])
            def side(g):return 'right' if names[g].startswith('right_/') else 'left' if names[g].startswith('left_/') else None
            s0,s1=side(g0),side(g1)
            if s0 and s1:
                (same_hand if s0==s1 else cross_hand)[tick]=True
                continue
            hand=g0 if s0 else g1;other=g1 if s0 else g0
            hand_name=names[hand];other_name=names[other]
            is_case='piano_case' in other_name or (other_name.startswith('piano/') and key<0)
            if is_case:hand_case[tick]=True
            is_distal='_DIP_' in hand_name or '_thumb_PIP_' in hand_name
            # Positive key-local Z normal points toward the upper face. Edge
            # contacts are kept apart instead of automatically called side hits.
            surface='not_key'
            if key>=0:
                hand_key[tick]=True
                normal=point[14:17]
                surface='top' if normal[2]>=.7 else 'side_or_under'
                (key_top if surface=='top' else key_side)[tick]=True
                if normal[2]<=-.7:key_under[tick]=True
                if not is_distal:non_distal_key[tick]=True
            code=tuple(sorted((g0,g1)))
            if code not in pair_rows:
                pair_rows[code]=dict(geoms=[names[g] for g in code],hand_geom=hand_name,other_geom=other_name,
                                    type='piano_case' if is_case else 'key' if key>=0 else 'other',
                                    distal_link_geom=is_distal,key_midi=meta['key_geom_midi'].get(str(key)),
                                    ticks=set(),top_ticks=set(),side_ticks=set(),under_ticks=set(),penetration=[],under_penetration=[],normal_z=[],points=[])
            pair=pair_rows[code];pair['ticks'].add(tick);pair['penetration'].append(-point[3])
            if key>=0:
                pair['normal_z'].append(float(point[16]));pair['points'].append(point[11:14].tolist())
                pair['top_ticks' if surface=='top' else 'side_ticks'].add(tick)
                if point[16]<=-.7:
                    pair['under_ticks'].add(tick);pair['under_penetration'].append(-point[3])
        contacts=[]
        for pair in pair_rows.values():
            mask=np.zeros(n,bool);mask[list(pair.pop('ticks'))]=True
            top=pair.pop('top_ticks');sides=pair.pop('side_ticks');under=pair.pop('under_ticks');penetrations=pair.pop('penetration');under_penetration=pair.pop('under_penetration');normals=pair.pop('normal_z');points=pair.pop('points')
            side_mask=np.zeros(n,bool);side_mask[list(sides)]=True
            under_mask=np.zeros(n,bool);under_mask[list(under)]=True
            contacts.append(dict(**pair,contact_substeps=int(mask.sum()),contact_duration=float(mask.sum()*dt),
                                 intervals=intervals(mask,dt),max_penetration_mm=float(max(penetrations)*1000),
                                 top_substeps=len(top),side_or_under_substeps=len(sides),
                                 underside_substeps=len(under),side_or_under_intervals=intervals(side_mask,dt),underside_intervals=intervals(under_mask,dt),
                                 max_underside_penetration_mm=float(max(under_penetration,default=0)*1000),
                                 key_local_normal_z=ranges(normals) if normals else None,
                                 key_local_point_min=np.min(points,axis=0).tolist() if points else None,
                                 key_local_point_max=np.max(points,axis=0).tolist() if points else None))
        contacts.sort(key=lambda v:(v['type']!='piano_case',-v['contact_substeps']))
        categories={k:v for k,v in [('piano_case',hand_case),('any_key',hand_key),('key_top',key_top),('key_side_or_under',key_side),('key_underside',key_under),
                                    ('non_distal_link_key',non_distal_key),('same_hand',same_hand),('cross_hand',cross_hand)]}
        windows=[]
        for interval in intervals(hand_case,dt):
            if interval['duration']<.02:continue
            a,b=interval['first_tick'],interval['last_tick']+1
            motion=[]
            for row in meta['joints']:
                if '_Wrist_' not in row['name']:continue
                aid,qid,vid=row['actuator_id'],row['qpos_adr'],row['qvel_adr']
                motion.append(dict(joint=row['name'],position_start=float(z['qpos'][a,qid]),position_end=float(z['qpos'][b-1,qid]),
                                   target_min=float(z['ctrl'][a:b,aid].min()),target_max=float(z['ctrl'][a:b,aid].max()),
                                   abs_lag_mean=float(np.abs(z['ctrl'][a:b,aid]-z['qpos'][a:b,qid]).mean()),
                                   abs_speed_mean=float(np.abs(z['qvel'][a:b,vid]).mean()),
                                   abs_force_max=float(np.abs(z['actuator_force'][a:b,aid]).max())))
            windows.append(dict(**interval,wrists=motion))
        restricted_contact_windows=[]
        for pair in contacts:
            for segment in pair['side_or_under_intervals']:
                if segment['duration']<.075:continue
                a,b=segment['first_tick'],segment['last_tick']+1
                side_prefix='L_' if pair['hand_geom'].startswith('left_/') else 'R_'
                finger=next((part for part in ('thumb','index','mid','ring','pinky') if '_'+part+'_' in pair['hand_geom']),None)
                rows=[]
                for joint in meta['joints']:
                    if not joint['name'].startswith(side_prefix):continue
                    if '_Wrist_' not in joint['name'] and (finger is None or '_'+finger+'_' not in joint['name'].lower()):continue
                    aid,qid,vid=joint['actuator_id'],joint['qpos_adr'],joint['qvel_adr']
                    actual=z['qpos'][a:b,qid];target=z['ctrl'][a:b,aid];force=z['actuator_force'][a:b,aid]
                    rows.append(dict(joint=joint['name'],qpos_start=float(actual[0]),qpos_end=float(actual[-1]),
                                     target_min=float(target.min()),target_max=float(target.max()),
                                     abs_target_lag_mean=float(np.mean(abs(target-actual))),
                                     abs_speed_mean=float(np.mean(abs(z['qvel'][a:b,vid]))),
                                     force_min=float(force.min()),force_max=float(force.max()),
                                     constraint_min=float(z['qfrc_constraint'][a:b,vid].min()),constraint_max=float(z['qfrc_constraint'][a:b,vid].max())))
                restricted_contact_windows.append(dict(geoms=pair['geoms'],**segment,joints=rows))
        goal=c['goals'][phase_frames:,:88].astype(bool);actual=c['actual'][phase_frames:].astype(bool)
        def frame_f1(actual):
            tp=(goal&actual).sum(1);fp=(~goal&actual).sum(1);fn=(goal&~actual).sum(1);den=2*tp+fp+fn
            return np.divide(2*tp,den,out=np.ones(len(tp)),where=den>0)
        frame_scores=frame_f1(actual);loss=1-frame_scores
        error_frames=[]
        score=np.load(root/'score.npz')
        for i in np.flatnonzero(loss>0):
            error_frames.append(dict(music_command_seconds=float(i*control_dt),policy_command_seconds=float((i+phase_frames)*control_dt),
                                     frame_f1=float(frame_scores[i]),f1_loss_contribution=float(loss[i]/len(loss)),
                                     missed_midi=(np.flatnonzero(goal[i]&~actual[i])+21).tolist(),wrong_midi=(np.flatnonzero(actual[i]&~goal[i])+21).tolist(),
                                     fingering=[[int(f),int(k)+21] for f,k in zip(*np.nonzero(score['fingers'][i+phase_frames]))]))
        first_n=min(len(loss),round(.75/control_dt))
        error_summary=dict(music_frame_f1=float(frame_scores.mean()),
                           first_075_seconds_fraction_of_all_f1_loss=float(loss[:first_n].sum()/loss.sum()) if loss.sum() else 0.,
                           error_frames=error_frames)
        result=dict(checkpoint=checkpoint['checkpoint'],control_dt=control_dt,physics_dt=dt,
                    canonical_music_f1=[e['music_only_f1'] for e in checkpoint['canonical_repeats']],
                    joints=joints,large_target_changes=sorted(large_changes,key=lambda v:v['width_fraction'],reverse=True),
                    contact_categories={k:dict(substeps=int(v.sum()),duration=float(v.sum()*dt),intervals=intervals(v,dt)) for k,v in categories.items()},
                    hand_piano_contacts=contacts,piano_case_windows=windows,restricted_contact_windows=restricted_contact_windows,
                    error_summary=error_summary,
                    limitations=['Stored GPU contact geometry/forces come from the substep solve; qpos is its post-integration state (5ms later).',
                                 'Non-distal means contact on a non-distal link geom, not a measurement of anatomical fingertip contact.',
                                 'Key-side classification uses the key-local outward normal z<0.7; edge diagonals are included.',
                                 'Force near forcerange is actuator saturation; qfrc_constraint includes all constraints and is not a per-contact force.',
                                 'Large target lag can reflect inertia or blocked contact; timing association alone is not a causal experiment.'])
        (directory/'motion_contact_report.json').write_text(json.dumps(result,indent=2)+'\n')
        with (directory/'joint_ranges.csv').open('w') as handle:
            rows=[dict(joint=j['name'],qpos_min=j['qpos']['min'],qpos_max=j['qpos']['max'],ctrl_min=j['ctrl']['min'],ctrl_max=j['ctrl']['max'],
                       lag_p99=j['absolute_lag']['p99'],lag_max=j['absolute_lag']['max'],delta_p99=j['absolute_ctrl_change']['p99'],
                       delta_max=j['absolute_ctrl_change']['max'],speed_max=j['absolute_velocity']['max'],force_saturation=j['force_saturation_fraction']) for j in joints]
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summaries.append(dict(checkpoint=name,music_f1=result['canonical_music_f1'],contact_categories={k:{kk:vv for kk,vv in v.items() if kk!='intervals'} for k,v in result['contact_categories'].items()},
                              longest_case_window=max([v['duration'] for v in windows],default=0),piano_case_pairs=[v for v in contacts if v['type']=='piano_case']))
    (root/'motion_contact_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='piano_case_pairs'} for row in summaries],indent=2))


if __name__=='__main__':main()
