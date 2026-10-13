#!/usr/bin/env python3
"""Audit whole-finger collision surfaces and score use from recorded trajectories.

CPU only; no simulation or policy modification. Global plane deficits are not
penetration estimates. XY overlap uses conservative primitive/key AABBs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FINGERS = ('thumb', 'index', 'mid', 'ring', 'pinky')


def extents(kinds, sizes, matrices):
    """Exact axis-aligned half extents for the supported collision primitives."""
    result = np.empty(matrices.shape[:-2] + (3,))
    for i, (kind, size) in enumerate(zip(kinds, sizes)):
        r = matrices[..., i, :, :]
        if kind == 2: e = np.broadcast_to(size[0], r.shape[:-1])
        elif kind == 3: e = size[0] + size[1] * np.abs(r[..., :, 2])
        elif kind == 4: e = np.sqrt(np.sum((r * size)**2, axis=-1))
        elif kind == 5: e = size[1]*np.abs(r[..., :, 2])+size[0]*np.sqrt(np.sum(r[..., :, :2]**2, axis=-1))
        elif kind == 6: e = np.sum(np.abs(r)*size, axis=-1)
        else: raise ValueError(f'Unsupported physical primitive type {kind}')
        result[..., i, :] = e
    return result


def mask_spans(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [dict(first_tick=int(s), last_tick=int(e-1), first_sample=(s+1)*dt,
                 last_sample=e*dt, sampled_duration=(e-s)*dt)
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--plot', action='store_true', help='Create a static opening diagnostic figure')
    a = p.parse_args(); root = a.audit
    meta = json.loads((root/'model_indices.json').read_text())
    audit = json.loads((root/'checkpoint_audit.json').read_text())
    names = meta['geom_names']; sites = meta['site_names']
    labels = [f'{side}_{finger}' for side in ('right', 'left') for finger in FINGERS]
    groups = [[i for i, name in enumerate(names) if name.startswith(side+'_/')
               and '_'+finger+'_' in name and '_collision_' in name]
              for side in ('right', 'left') for finger in FINGERS]
    if any(len(g) != 4 for g in groups): raise ValueError('Expected all four collision links per finger')
    gids = np.array(sum(groups, [])); types = np.array(meta['geom_type'])[gids]
    sizes = np.array(meta['geom_size'])[gids]
    tips = [sites.index(name) for side in ('right', 'left') for name in meta['fingertip_sites'][side]]
    kid = np.array([int(k) for k in meta['key_geom_midi']]); ksize = np.array(meta['geom_size'])[kid]
    origin = np.load(root/'origin_geometry.npz')
    kmat = origin['geom_xmat'][kid].reshape(-1,3,3); kpos = origin['geom_xpos'][kid]
    kext = np.sum(np.abs(kmat)*ksize[:,None,:],axis=-1)
    white = np.array(['white_key_' in names[i] for i in kid]); ktop = kpos[:,2]+kext[:,2]
    white_top = float(ktop[white].max()); black_top = float(ktop[~white].max())

    def geometry(z):
        pos = np.asarray(z['geom_xpos'])[...,gids,:]
        mat = np.asarray(z['geom_xmat']).reshape(pos.shape[:-2]+(len(names),3,3))[...,gids,:,:]
        ex = extents(types,sizes,mat); lo = pos-ex; hi=pos+ex
        lower = lo[...,2].reshape(pos.shape[:-2]+(10,4)).min(-1)
        overlap = ((hi[..., :,None,:2]>=kpos[:,:2]-kext[:,:2]) &
                   (lo[..., :,None,:2]<=kpos[:,:2]+kext[:,:2])).all(-1)
        local = np.min(np.where(overlap,lo[..., :,None,2]-ktop,np.inf),axis=-1)
        local = local.reshape(pos.shape[:-2]+(10,4)).min(-1)
        return dict(surface_min_z=lower,clearance_white=lower-white_top,
                    clearance_black=lower-black_top,local_aabb_clearance=local,
                    fingertip_z=np.asarray(z['site_xpos'])[...,tips,2])

    static = {}
    for name in ('origin','initial'):
        g=geometry(np.load(root/f'{name}_geometry.npz'))
        static[name]=[{ 'finger':label, **{k:(float(v[i]) if np.isfinite(v[i]) else None) for k,v in g.items()}}
                      for i,label in enumerate(labels)]
    score=np.load(root/'score.npz'); active=score['fingers'].astype(bool).any(-1)
    future025=active.copy();future050=active.copy()
    # Existing task has 20Hz controls and 0.25s gesture protection. Preserve its
    # inclusive current frame + five future frame convention explicitly.
    for offset in range(1,11):
        future050[:-offset] |= active[offset:]
        if offset<=5:future025[:-offset] |= active[offset:]
    summary=dict(key_rest_top_z=dict(white=white_top,black=black_top),static=static,
                 source_archive_sha256=audit['archive_sha256'],verified_source_files=audit['verified_source_files'],checkpoints=[])
    for entry in audit['checkpoints']:
        name=Path(entry['checkpoint']).stem; d=root/name/'detail-0'
        z=np.load(d/'physics_200hz.npz'); contact=np.load(d/'contact_geometry.npz')['contacts']
        dt=float(z['physics_dt']); n=len(z['qpos']); decimation=n//len(np.load(d/'rollout.npz')['actions'])
        control=np.arange(n)//decimation
        masks=dict(current=active[control],protected_025=future025[control],visible_050=future050[control])
        geo=geometry(z); under=np.zeros((n,10),bool); side=np.zeros((n,10),bool)
        for point in contact:
            if int(point[10])<0:continue
            gid=next((int(g) for g in point[1:3] if int(g) in gids),None)
            if gid is None:continue
            f=next(i for i,g in enumerate(groups) if gid in g); tick=int(point[0])
            if point[16]<.7:side[tick,f]=True
            if point[16]<=-.7:under[tick,f]=True
        rows=[]
        for f,label in enumerate(labels):
            idle=~masks['protected_025'][:,f]; bad=geo['clearance_white'][:,f]<0
            row=dict(finger=label,minimum_surface_z=float(geo['surface_min_z'][:,f].min()),
                     minimum_white_clearance=float(geo['clearance_white'][:,f].min()),
                     current_unused_below_white_substeps=int((bad&~masks['current'][:,f]).sum()),
                     protected025_unused_below_white_substeps=int((bad&idle).sum()),
                     protected025_unused_local_aabb_negative_substeps=int(((geo['local_aabb_clearance'][:,f]<0)&idle).sum()),
                     idle_tip_above_white_but_link_below_substeps=int((idle&bad&(geo['fingertip_z'][:,f]>=white_top)).sum()),
                     under_substeps=int(under[:,f].sum()),side_or_under_substeps=int(side[:,f].sum()),
                     under_spans=mask_spans(under[:,f],dt),side_or_under_spans=mask_spans(side[:,f],dt),
                     under_current_used_substeps=int((under[:,f]&masks['current'][:,f]).sum()),
                     under_future025_used_substeps=int((under[:,f]&masks['protected_025'][:,f]).sum()),
                     under_future050_used_substeps=int((under[:,f]&masks['visible_050'][:,f]).sum()))
            rows.append(row)
        np.savez_compressed(d/'finger_clearance.npz',**geo,**masks,under_contact=under,
                            side_or_under_contact=side,finger_names=np.array(labels),time=(np.arange(n)+1)*dt)
        summary['checkpoints'].append(dict(checkpoint=name,canonical_music_f1=entry['min_music_f1'],fingers=rows))
        if a.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            time=(np.arange(n)+1)*dt; selection=time<=1.8; f=5
            joint=next(j for j in meta['joints'] if j['name']=='L_Thumb_Rot2')
            fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True,layout='constrained')
            axes[0].plot(time[selection],1000*geo['surface_min_z'][selection,f],label='Whole collision finger: lowest surface')
            axes[0].plot(time[selection],1000*geo['fingertip_z'][selection,f],label='Tip site',alpha=.65)
            axes[0].axhline(1000*white_top,color='gray',label='White key resting top')
            axes[0].set_ylabel('World Z (mm)');axes[0].legend(fontsize=8)
            for y,(label,mask) in enumerate(masks.items()):
                axes[1].fill_between(time[selection],y,y+.8,where=mask[selection,f],step='pre',label=label)
            axes[1].set_yticks([.4,1.4,2.4],list(masks));axes[1].set_ylabel('Score use')
            axes[2].plot(time[selection],z['ctrl'][selection,joint['actuator_id']],label='Target')
            axes[2].plot(time[selection],z['qpos'][selection,joint['qpos_adr']],label='Actual')
            axes[2].set_ylabel('Thumb Rot2 (rad)');axes[2].legend(fontsize=8)
            axes[3].plot(time[selection],z['actuator_force'][selection,joint['actuator_id']],label='Actuator torque')
            axes[3].plot(time[selection],z['qfrc_constraint'][selection,joint['qvel_adr']],label='All constraint torque')
            axes[3].set_ylabel('Torque (N m)');axes[3].legend(fontsize=8)
            for ax in axes:
                ax.axvline(1,color='gray',ls='--');ax.grid(alpha=.15)
                for segment in mask_spans(under[:,f],dt):
                    ax.axvspan(segment['first_sample']-.5*dt,segment['last_sample']+.5*dt,color='red',alpha=.12)
            axes[-1].set_xlim(0,1.8);axes[-1].set_xlabel('Policy time (s); music begins at 1s; red = key underside contact')
            fig.suptitle(f'{name}: opening left-thumb clearance, use and contact')
            fig.savefig(d/'opening_finger_clearance.png',dpi=170);plt.close(fig)
    summary['limitations']=[
        'Whole-finger surface is the minimum over four collision primitives, not visual meshes.',
        'Global white/black plane deficits are height diagnostics, not contact or penetration.',
        'Local plane uses conservative XY AABB overlap and resting key tops, not exact projected shape overlap.',
        'Contact geometry comes from GPU solve; positions are the recorded post-integration state up to 5ms later.',
        'Current/0.25s/0.5s score masks label use, not whether a physical contact is appropriate.']
    (root/'finger_clearance_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    run=Path(audit['checkpoints'][0]['checkpoint']).parent
    evaluations=[]
    for line in (run/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line)
        if 'eval' not in row:continue
        e=row['eval'];evaluations.append(dict(vector_step=row['vector_step'],updates=row['updates'],
            music_f1_min=e['music_only_f1_min'],music_micro_f1_min=e['music_only_micro_f1_min'],
            preparation_wrong_frames=e['preparation_wrong_key_frames_max'],
            hand_collision_substeps=e['hand_collision_substeps_max']))
    history=dict(run=str(run),evaluations=evaluations,best=max(evaluations,key=lambda x:x['music_f1_min']),
                 final=evaluations[-1])
    (root/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(dict(key_rest_top_z=summary['key_rest_top_z'],static=static,
                         contact_fingers=[dict(checkpoint=c['checkpoint'],fingers=[r for r in c['fingers'] if r['under_substeps']]) for c in summary['checkpoints']]),indent=2))


if __name__=='__main__':main()
