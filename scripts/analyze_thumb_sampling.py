#!/usr/bin/env python3
"""Audit scored thumb misses and control-endpoint sampling from recorded data.

Substep scores repeat each currently applied control goal. They are diagnostic
samples, not alternative official F1 or predictions of higher-frequency training.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit_episode(directory, score, meta):
    c=np.load(directory/'rollout.npz');p=np.load(directory/'physics_200hz.npz')
    d=np.load(directory/'control_diagnostics.npz')
    report=json.loads((directory/'report.json').read_text())
    dt=float(report['control_dt']);pdt=float(report['physics_dt'])
    if not np.isfinite([dt,pdt]).all() or min(dt,pdt)<=0:
        raise ValueError('Audit timesteps must be finite and positive: '+str(directory))
    frames=len(c['actions']);ratio=dt/pdt;decimation=round(ratio)
    if (frames<=0 or decimation<1 or not np.isclose(ratio,decimation,rtol=0,atol=1e-8)
            or len(p['qpos'])!=frames*decimation):
        raise ValueError('Control/physics timestamps and recorded sample counts disagree: '+str(directory))
    if 'physics_dt' in p and not np.isclose(float(p['physics_dt']),pdt,rtol=0,atol=1e-10):
        raise ValueError('Saved physics timestep disagrees with report: '+str(directory))
    goal=c['goals'][:,:88].astype(bool);actual=c['actual'].astype(bool)
    if goal.shape!=(frames,88) or actual.shape!=goal.shape:
        raise ValueError('Expected exactly 88 scored piano keys: '+str(directory))
    if (score['fingers'].shape!=(frames,10,88)
            or not np.array_equal(score['goal'],c['goals'])):
        raise ValueError('Saved score/fingering does not align with applied control goals: '+str(directory))
    limits=p['key_limits']
    if (limits.shape!=(88,2) or not np.array_equal(limits[:,0],np.zeros(88))
            or not np.isfinite(limits).all() or (limits[:,1]<=0).any()
            or p['qpos'].shape[1]<88 or c['qpos'].shape[1]<88):
        raise ValueError('Pressure audit requires the known 88-key zero-rest joint contract: '+str(directory))
    activation=np.abs(np.clip(p['qpos'][:,:88],limits[:,0],limits[:,1])-limits[:,1])<=float(p['key_threshold'])
    if not np.array_equal(activation,p['key_activation']):
        raise ValueError('qpos[0:88] does not reproduce saved key activation; supply a verified key mapping: '+str(directory))
    if (not np.array_equal(activation[decimation-1::decimation],actual)
            or not np.array_equal(p['qpos'][decimation-1::decimation,:88],c['qpos'][:,:88])):
        raise ValueError('Control endpoint and physics samples are misaligned: '+str(directory))
    for key,times in (('command_time_seconds',np.arange(frames)*dt),
                      ('returned_state_time_seconds',(np.arange(frames)+1)*dt)):
        if key in d and not np.allclose(d[key],times,rtol=0,atol=1e-8):
            raise ValueError('Control diagnostic timestamps are misaligned: '+str(directory))
    miss=goal&~actual;false=~goal&actual
    phys=activation.reshape(frames,decimation,88)
    wrong=phys&~goal[:,None,:]
    masked=wrong&~false[:,None,:]
    pressure=(p['qpos'][:,:88]/limits[:,1]).clip(0,1).reshape(frames,decimation,88)
    penalty=np.where(pressure>.001,-(.01+.14*pressure),0)*~goal[:,None,:]
    endpoint_pressure=(c['qpos'][:,:88]/limits[:,1]).clip(0,1)
    endpoint_penalty=np.where(endpoint_pressure>.001,-(.01+.14*endpoint_pressure),0)*~goal
    by_finger=[]
    for finger in range(10):
        assigned=score['fingers'][:,finger].astype(bool)
        indices=np.argwhere(miss&assigned)
        if len(indices):
            by_finger.append(dict(finger=('R' if finger<5 else 'L')+str(finger%5+1),
                missed_key_frames=len(indices),events=[dict(score_index=int(f),midi=int(k+21),
                command_seconds=float(f*dt),returned_seconds=float((f+1)*dt),
                projection=bool(d['finger_projection_mask'][f,finger]),
                post_step_permission=bool(d['finger_terminal_allowed'][f,finger]))for f,k in indices]))
    thumb_frames=np.flatnonzero((miss&score['fingers'][:,5].astype(bool)).any(-1))
    left_full=[x['full_index']for x in meta['action_contract']['channels']if x['name'].startswith('L_Wrist')]
    joints=[]
    if thumb_frames.size:
        ticks=(thumb_frames[:,None]*decimation+np.arange(decimation)).ravel()
        for joint in meta['joints']:
            if not joint['name'].startswith('L_Thumb'):continue
            a=joint['actuator_id'];q=joint['qpos_adr'];values=p['qpos'][ticks,q];force=p['actuator_force'][ticks,a]
            low,high=joint['force_range']
            saturated=((force<=.99*low)|(force>=.99*high)) if joint['force_limited'] else None
            joints.append(dict(name=joint['name'],physical_range=joint['joint_range'],
                actual_range=[float(values.min()),float(values.max())],
                target_range=[float(p['ctrl'][ticks,a].min()),float(p['ctrl'][ticks,a].max())],
                force_saturated_fraction=float(saturated.mean()) if saturated is not None else None,
                max_abs_force=float(np.abs(force).max())))
    result=dict(checkpoint=directory.parent.name,phase_scores=report['metrics']['music'],
        control_dt=dt,physics_dt=pdt,decimation=decimation,control_frames=frames,
        verified_key_qpos_indices=list(range(88)),
        missed_by_finger=by_finger,wrong_20hz_key_frames=int(false.sum()),
        wrong_200hz_key_substeps=int(wrong.sum()),
        endpoint_invisible_wrong_key_substeps=int(masked.sum()),
        endpoint_invisible_wrong_control_frames=int(masked.any((1,2)).sum()),
        left_thumb_miss_status='scored_misses_present' if thumb_frames.size else 'no_scored_misses',
        left_thumb_miss_frames=thumb_frames.tolist(),
        left_wrist_slew_during_thumb_miss_frames=thumb_frames[d['control_slew_mask'][thumb_frames][:,left_full].any(-1)].tolist(),
        left_thumb_projection_miss_frames=thumb_frames[d['finger_projection_mask'][thumb_frames,5]].tolist(),
        left_thumb_joints_during_misses=joints,
        false_press_sum_of_control_endpoint=float(endpoint_penalty.sum()),
        false_press_sum_of_substep_means=float(penalty.mean(1).sum()),
        sampling_note='Substeps use held current control score and include within-frame attack/release delay; diagnostic only, not replacement official F1. Legacy 20hz/200hz field names denote control/physics sampling; use saved timesteps for actual frequencies.')
    result['phase_scores']={k:v for k,v in result['phase_scores'].items()if k!='note_on_events'}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--key-midi',type=int,default=60,help='Piano MIDI note to plot (21..108), default C4=60')
    args=parser.parse_args();root=args.audit
    if not 21<=args.key_midi<=108:parser.error('--key-midi must be within the 88-key piano range 21..108')
    meta=json.loads((root/'model_indices.json').read_text())
    score=np.load(root/'score.npz')
    directories=sorted(root.glob('*/detail-0'))
    if not directories:parser.error('No checkpoint/detail-0 audit episodes found')
    results=[audit_episode(directory,score,meta)for directory in directories]
    (root/'thumb_sampling_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Summary written; matplotlib unavailable, figure omitted.')
        return
    fig,axes=plt.subplots(len(results),1,figsize=(10,3*len(results)),sharex=True,squeeze=False)
    note=('C','C#','D','D#','E','F','F#','G','G#','A','A#','B')[args.key_midi%12]+str(args.key_midi//12-1)
    key=args.key_midi-21;windows=[]
    for ax,result in zip(axes[:,0],results):
        directory=root/result['checkpoint']/'detail-0'
        p=np.load(directory/'physics_200hz.npz');c=np.load(directory/'rollout.npz')
        dt=result['control_dt'];pdt=result['physics_dt'];decimation=result['decimation']
        t=(np.arange(len(p['qpos']))+1)*pdt
        depth=p['qpos'][:,key]/p['key_limits'][key,1]
        ax.plot(t,depth,label=f'{note} travel at {1/pdt:g} Hz',lw=1.5)
        ct=(np.arange(len(c['qpos']))+1)*dt
        ax.scatter(ct,c['qpos'][:,key]/p['key_limits'][key,1],s=13,label=f'{1/dt:g} Hz sampled travel',zorder=3)
        goal=np.repeat(c['goals'][:,key],decimation)
        ax.fill_between(t,0,1,where=goal.astype(bool),alpha=.15,color='purple',label=f'{note} requested')
        threshold=1-float(p['key_threshold'])/p['key_limits'][key,1]
        ax.axhline(threshold,color='gray',ls='--',label='Activation threshold')
        assigned=np.flatnonzero(c['goals'][:,key])
        if assigned.size:
            end=assigned[0]
            while end+1<len(c['goals']) and c['goals'][end+1,key]:end+=1
            windows.append((max(0.,(assigned[0]-3)*dt),min(t[-1],(end+4)*dt)))
        else:windows.append((0.,t[-1]))
        ax.set(title=result['checkpoint'],ylabel='Normalized key travel',ylim=(-.05,1.1))
        ax.grid(alpha=.2)
    axes[-1,0].set(xlabel='Policy time (seconds)',xlim=(min(w[0]for w in windows),max(w[1]for w in windows)))
    axes[0,0].legend(loc='upper left',fontsize=8)
    fig.suptitle(f'{note} (MIDI {args.key_midi}): first requested segment, held control goals')
    fig.tight_layout()
    filename='thumb_c4_sampling.png' if args.key_midi==60 else f'thumb_midi{args.key_midi}_sampling.png'
    fig.savefig(root/filename,dpi=170)
    plt.close(fig)
    print(json.dumps([{k:r[k]for k in ('checkpoint','left_thumb_miss_frames','left_wrist_slew_during_thumb_miss_frames',
        'endpoint_invisible_wrong_key_substeps','endpoint_invisible_wrong_control_frames')}for r in results]))


if __name__=='__main__':main()
