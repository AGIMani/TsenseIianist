"""Native CPU IK with the SceneArmFollower Torch/scene interface.

Single scenes use the serial compiled core; batches parallelize independent
worlds on CPU. Both retain the same IK and atomic constraint checks. No SDK
commands, file I/O or rendering occur in step().
"""
import numpy as np
import torch
from numba import njit, prange
from .arm_realtime import _advance, _rotation_error, _rot, _mm3, _mv3
from .arm_scene import SceneArmFollower


@njit(cache=True)
def _one(state, targets, constants, limits, settings):
    dt, iterations, weight, damping, j6, j7, coefficient, limit, soft = settings
    jp=constants[:21].reshape(7,3); jr=constants[21:84].reshape(7,3,3)
    axes=constants[84:105].reshape(7,3); bp=constants[105:108]; br=constants[108:117].reshape(3,3)
    toolp=constants[117:120]; toolr=constants[120:129].reshape(3,3)
    tp=targets[:3];tr=targets[3:].reshape(3,3)
    result=_advance(state[0],state[1],state[3],tp,tr,jp,jr,axes,bp,br,toolp,toolr,
        limits[0],limits[1],limits[2],limits[3],dt,int(iterations),weight,damping,
        int(j6),int(j7),coefficient,limit,soft)
    q,v,acc,ref,p,r,rv,ra,excess=result
    out=np.empty(63)
    out[:7]=q;out[7:14]=v;out[14:21]=acc;out[21:28]=ref
    out[28:31]=p;out[31:40]=r.reshape(9);out[40:47]=rv;out[47:54]=ra
    out[54]=excess;out[55:58]=_rotation_error(r,tr);out[58:61]=tp-p
    out[61]=np.linalg.norm(out[55:58]);out[62]=np.linalg.norm(out[58:61])
    return out


@njit(cache=True, nogil=True)
def _serial(state, targets, constants, limits, settings):
    out=np.empty((state.shape[0],2,63))
    for i in range(state.shape[0]):
        for side in range(2):out[i,side]=_one(state[i,side],targets[i,side],constants[side],limits[side],settings)
    return out


@njit(cache=True, inline='never')
def _safe_one(state, targets, constants, limits, settings):
    # Catch inside the worker, not in prange: multiple exits from a prange
    # body prevent Numba from parallelizing it. No result is committed here.
    try:
        return _one(state,targets,constants,limits,settings), 0
    except Exception:
        return np.zeros(63), 1


@njit(cache=True, parallel=True, nogil=True)
def _parallel(state, targets, constants, limits, settings):
    out=np.empty((state.shape[0],2,63));failed=np.zeros((state.shape[0],2),dtype=np.int64)
    for flat in prange(state.shape[0]*2):
        i=flat//2;side=flat%2
        row,status=_safe_one(state[i,side],targets[i,side],constants[side],limits[side],settings)
        out[i,side]=row;failed[i,side]=status
    return out,failed


@njit(cache=True)
def _quat(r):
    values=np.sqrt(np.maximum(np.array([1+r[0,0]+r[1,1]+r[2,2],
        1+r[0,0]-r[1,1]-r[2,2],1-r[0,0]+r[1,1]-r[2,2],1-r[0,0]-r[1,1]+r[2,2]]),0.))
    i=np.argmax(values)
    if i==0:q=np.array([values[0]**2,r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    elif i==1:q=np.array([r[2,1]-r[1,2],values[1]**2,r[1,0]+r[0,1],r[0,2]+r[2,0]])
    elif i==2:q=np.array([r[0,2]-r[2,0],r[1,0]+r[0,1],values[2]**2,r[2,1]+r[1,2]])
    else:q=np.array([r[1,0]-r[0,1],r[2,0]+r[0,2],r[2,1]+r[1,2],values[3]**2])
    q/=2*max(values[i],1e-8)
    return q/np.linalg.norm(q)


@njit(cache=True, nogil=True)
def _poses(q, constants):
    out=np.empty((q.shape[0],2,8,7))
    for world in range(q.shape[0]):
        for side in range(2):
            c=constants[side];jp=c[:21].reshape(7,3);jr=c[21:84].reshape(7,3,3)
            axes=c[84:105].reshape(7,3);p=c[105:108].copy();r=c[108:117].reshape(3,3).copy()
            out[world,side,0,:3]=p;out[world,side,0,3:]=_quat(r)
            for j in range(7):
                p=p+_mv3(r,jp[j]);r=_mm3(_mm3(r,jr[j]),_rot(axes[j],q[world,side,j]))
                out[world,side,j+1,:3]=p;out[world,side,j+1,3:]=_quat(r)
    return out


class NativeSceneArmFollower(SceneArmFollower):
    """Same scene API and observation/reward semantics, explicit native backend."""
    def __init__(self,bundle,*,num_envs,device):
        super().__init__(bundle,num_envs=num_envs,device=device)
        self.contract['execution_backend']='numba_cpu'
        self._native_constants=np.stack([np.concatenate([a[k].cpu().numpy().ravel() for k in
            ('joint_pos','joint_rot','axis','base_pos','base_rot','tool_pos','tool_rot')]) for a in self.sides.values()])
        self._native_limits=np.stack([np.stack([a[k].cpu().numpy() for k in ('low','high','vmax','amax')]) for a in self.sides.values()])
        self._native_state=np.stack([np.stack([a[k].cpu().numpy() for k in ('q','v','acceleration','reference')],axis=1)
                                    for a in self.sides.values()],axis=1)
        o=self.options;c=o['coupling']
        self._native_settings=np.array([self.dt,o['iterations'],o['orientation_weight_m'],o['damping'],
            c['j6_index'],c['j7_index'],c['coefficient'],c['sum_limit_rad'],o['soft_limit_ratio']])
        self._native_targets=np.empty((num_envs,2,12))
        # Compile both the selected execution path and scene transforms before
        # any caller starts its control clock. Warmup does not commit state.
        for j,a in enumerate(self.sides.values()):
            p,r,*_=self.fk(a,a['q'])
            self._native_targets[:,j,:3]=p.cpu().numpy();self._native_targets[:,j,3:]=r.cpu().numpy().reshape(num_envs,9)
        self._compute()
        _poses(self._native_state[:,:,0],self._native_constants)
        self._native_pose_cache=None

    def _compute(self):
        if self.num_envs==1:
            return _serial(self._native_state,self._native_targets,self._native_constants,self._native_limits,self._native_settings)
        result,failed=_parallel(self._native_state,self._native_targets,self._native_constants,self._native_limits,self._native_settings)
        if failed.any():raise RuntimeError('Native arm hard-limit/numerical failure; entire batch not advanced')
        return result

    def step(self,targets):
        for j,side in enumerate(self.sides):
            tp,tr=targets[side]
            if tuple(tp.shape)!=(self.num_envs,3) or tuple(tr.shape)!=(self.num_envs,3,3):
                raise ValueError('Invalid native palm pose shape: '+side)
            self._native_targets[:,j,:3]=tp.detach().cpu().numpy()
            self._native_targets[:,j,3:]=tr.detach().cpu().numpy().reshape(self.num_envs,9)
        if not np.isfinite(self._native_targets).all():raise ValueError('Invalid native palm pose request')
        result=self._compute()
        # All checks on all worlds/sides succeeded. Publish one packed transfer.
        tensor=torch.as_tensor(result,device=self.device,dtype=torch.float64)
        diagnostics={};poses={}
        for j,(side,a) in enumerate(self.sides.items()):
            row=tensor[:,j]
            for k,field in enumerate(('q','v','acceleration','reference')):
                a[field].copy_(row[:,k*7:(k+1)*7])
            tp,tr=targets[side]
            diagnostics[side]=dict(target_position=tp.double().clone(),target_rotation=tr.double().clone(),
                requested_q=row[:,21:28],requested_velocity=row[:,40:47],requested_acceleration=row[:,47:54],
                excess=row[:,54],rotation_error_vector=row[:,55:58],position_error_vector=row[:,58:61],
                orientation_error=row[:,61],position_error=row[:,62])
            self.excess=torch.maximum(self.excess,row[:,54])
            poses[side]=(row[:,28:31],row[:,31:40].reshape(self.num_envs,3,3))
        self._native_state[:]=result[:,:,:28].reshape(self.num_envs,2,4,7)
        self._native_pose_cache=None
        self.last=diagnostics
        return poses

    def reset(self,ids):
        super().reset(ids)
        rows=ids.detach().cpu().numpy() if isinstance(ids,torch.Tensor) else np.asarray(ids)
        for j,a in enumerate(self.sides.values()):
            self._native_state[rows,j,0]=a['q0'].cpu().numpy()
            self._native_state[rows,j,3]=a['q0'].cpu().numpy()
            self._native_state[rows,j,1:3]=0
        self._native_pose_cache=None

    def _poses_numpy(self):
        if self._native_pose_cache is None:
            self._native_pose_cache=_poses(self._native_state[:,:,0],self._native_constants)
        return self._native_pose_cache

    def transforms(self):
        values=torch.as_tensor(self._poses_numpy(),device=self.device,dtype=torch.float64)
        return {s:(values[:,j,:,:3],values[:,j,:,3:]) for j,s in enumerate(self.sides)}

    def publish_numpy(self,pos,quat):
        if self.num_envs!=1:raise ValueError('publish_numpy requires one world')
        values=self._poses_numpy()[0]
        for j,side in enumerate(self.sides):
            ids=self.wrists[side]['visual_ids']
            pos[ids]=values[j,:,:3];quat[ids]=values[j,:,3:]

    def load_recorded_q(self,q):
        """Rendering-only FK restore; never used to advance a control loop."""
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(self.num_envs,2,7):raise ValueError('Recorded q shape')
        self._native_state[:,:,0]=q
        for j,a in enumerate(self.sides.values()):a['q'].copy_(torch.as_tensor(q[:,j],device=self.device))
        self._native_pose_cache=None
