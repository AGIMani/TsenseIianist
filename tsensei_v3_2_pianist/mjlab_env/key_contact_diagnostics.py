"""Read-only physical-substep hand/key contact diagnostics.

Classification matches the recorded GPU contact audit: key-local outward
normal z <= -0.7 is underside; z < +0.7 is side-or-underside. Counts represent
world/substep occurrence, independent of duplicate contact points or pairs.
This observer does not change collision lists, rewards, or contact solving.
"""
import mujoco
import numpy as np
import torch
import warp as wp


@wp.kernel
def mark_key_contacts(nacon: wp.array(dtype=wp.int32),
                      geom: wp.array(dtype=wp.vec2i),
                      distance: wp.array(dtype=wp.float32),
                      world: wp.array(dtype=wp.int32),
                      frame: wp.array(dtype=wp.mat33),
                      geom_xmat: wp.array2d(dtype=wp.mat33),
                      key_mask: wp.array(dtype=wp.int32),
                      hand_kind: wp.array(dtype=wp.int32),
                      present: wp.array2d(dtype=wp.int32)):
    i=wp.tid()
    if i<nacon[0] and distance[i]<=0.0:
        a=geom[i][0];b=geom[i][1];w=world[i]
        if a>=0 and b>=0 and a<key_mask.shape[0] and b<key_mask.shape[0] and w>=0 and w<present.shape[0]:
            key=int(-1);hand=int(-1);sign=float(1.0)
            if key_mask[a]>0 and hand_kind[b]>0:
                key=a;hand=b
            elif key_mask[b]>0 and hand_kind[a]>0:
                key=b;hand=a;sign=-1.0
            if key>=0:
                normal=frame[i]
                rotation=geom_xmat[w,key]
                local_z=sign*(normal[0,0]*rotation[0,2]+normal[0,1]*rotation[1,2]+normal[0,2]*rotation[2,2])
                if local_z<=-0.7:
                    wp.atomic_max(present,w,0,1)
                if local_z<0.7:
                    wp.atomic_max(present,w,1,1)
                if hand_kind[hand]==2:
                    wp.atomic_max(present,w,2,1)


@wp.kernel
def accumulate_key_substep(present: wp.array2d(dtype=wp.int32),
                           counts: wp.array2d(dtype=wp.int32)):
    world=wp.tid()
    for category in range(3):
        counts[world,category]=counts[world,category]+present[world,category]
    counts[world,3]=counts[world,3]+1


class KeyContactDiagnostics:
    """Observer of real main-world contacts; never runs another collision pass."""
    COLUMNS=('key_underside_substeps','key_side_or_under_substeps',
             'key_non_distal_substeps','key_contact_physics_substeps')

    def __init__(self,model,num_envs,device,*,key_geom_names=None):
        if num_envs<1:raise ValueError('num_envs must be positive')
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
        keys=([n for n in names if n.startswith('piano/') and ('white_key_geom_' in n or 'black_key_geom_' in n)]
              if key_geom_names is None else list(key_geom_names))
        if not keys or len(set(keys))!=len(keys) or any(n not in names for n in keys):
            raise ValueError('Key diagnostics require unique existing key geometry names')
        key_mask=np.array([int(n in keys) for n in names],dtype=np.int32)
        hand_kind=np.zeros(model.ngeom,dtype=np.int32)
        for i,name in enumerate(names):
            if not name.startswith(('right_/','left_/')) or '_collision_' not in name:continue
            # A3.2's ID palm housing is base_link, whereas the older model's
            # named palm is now the small mounting block. Include that explicit
            # physical housing, but not arbitrary wrist/mount/visual geoms.
            housing = name.split('/', 1)[1].startswith('base_link_collision_')
            if not housing and not any('_'+part+'_' in name for part in ('palm','thumb','index','mid','ring','pinky')):continue
            terminal=('_DIP_' in name or '_thumb_PIP_' in name)
            if terminal and model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError('Terminal contact-pad marking requires capsule collision geoms: '+name)
            hand_kind[i]=1 if terminal else 2
        if not hand_kind.any():raise ValueError('No explicit finger/palm collision geoms found')
        self.key_mask=wp.array(key_mask,dtype=wp.int32,device=str(device))
        self.hand_kind=wp.array(hand_kind,dtype=wp.int32,device=str(device))
        self.present=wp.zeros((num_envs,3),dtype=wp.int32,device=str(device))
        self.substep_counts=torch.zeros((num_envs,4),dtype=torch.int32,device=device)
        self.counts_wp=wp.from_torch(self.substep_counts)
        self.contract=dict(schema=1,mode='main_physics_key_contact_substeps_v1',
            distance_max=0.,key_local_outward_normal=dict(underside_z_le=-.7,side_or_under_z_lt=.7),
            key_geom_names=keys,terminal_capsule_names=[n for n,k in zip(names,hand_kind) if k==1],
            non_distal_geom_names=[n for n,k in zip(names,hand_kind) if k==2],
            columns=list(self.COLUMNS),deduplication='one occurrence per world/category/physics substep',
            scope='main physical contact buffer only; no reward, collision-mask or solver mutation')

    def clear(self,env_ids=None):
        if env_ids is None:self.substep_counts.zero_()
        else:self.substep_counts[env_ids]=0

    def sample(self,sim,*,data=None,count_substep=True):
        if not count_substep:return
        d=sim.wp_data if data is None else data
        self.present.zero_()
        wp.launch(mark_key_contacts,dim=d.naconmax,
                  inputs=[d.nacon,d.contact.geom,d.contact.dist,d.contact.worldid,d.contact.frame,
                          d.geom_xmat,self.key_mask,self.hand_kind,self.present],device=sim.wp_device)
        wp.launch(accumulate_key_substep,dim=self.substep_counts.shape[0],
                  inputs=[self.present,self.counts_wp],device=sim.wp_device)
