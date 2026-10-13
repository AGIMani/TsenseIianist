"""Batched Torch M6 differential IK with reachable five-DOF wrist targets.

All runtime tensors stay on the simulation device; no per-environment CPU IK.
The arms are kinematic mocap geometry and never add policy action dimensions.
"""
import torch


def skew(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3)


def rot(axis, angle):
    k=skew(axis);eye=torch.eye(3,device=angle.device,dtype=angle.dtype)
    return eye+angle.sin()[...,None,None]*k+(1-angle.cos())[...,None,None]*(k@k)


def quaternion(r):
    # Robust matrix conversion: select the largest of four quaternion candidates.
    values=torch.stack((1+r[...,0,0]+r[...,1,1]+r[...,2,2],
        1+r[...,0,0]-r[...,1,1]-r[...,2,2],1-r[...,0,0]+r[...,1,1]-r[...,2,2],
        1-r[...,0,0]-r[...,1,1]+r[...,2,2]),-1).clamp_min(0).sqrt()
    w,x,y,z=values.unbind(-1)
    candidates=torch.stack((torch.stack((w*w,r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1),
        torch.stack((r[...,2,1]-r[...,1,2],x*x,r[...,1,0]+r[...,0,1],r[...,0,2]+r[...,2,0]),-1),
        torch.stack((r[...,0,2]-r[...,2,0],r[...,1,0]+r[...,0,1],y*y,r[...,2,1]+r[...,1,2]),-1),
        torch.stack((r[...,1,0]-r[...,0,1],r[...,2,0]+r[...,0,2],r[...,2,1]+r[...,1,2],z*z),-1)),-2)
    candidates=candidates/(2*values.clamp_min(1e-8)[...,None])
    index=values.argmax(-1)[...,None,None].expand(*values.shape[:-1],1,4)
    q=candidates.gather(-2,index).squeeze(-2)
    return q/q.norm(dim=-1,keepdim=True)


class GPUArmIK:
    def __init__(self, scene):
        self.scene=scene;self.cfg=scene.bundle.manifest['m6'];self.sides={}
        self.last_error={}
        for side,c in self.cfg['arms'].items():
            a={k:scene.tensor(v) for k,v in c.items() if isinstance(v,list)}
            for key in ('hand_qadr','hand_dof','mocap_ids'):
                a[key]=a[key].long()
            a['hand_kind']=c['hand_kind'];a['handle_id']=c['handle_id']
            a['q0']=a['q'].clone();a['q']=a['q'].expand(scene.num_envs,7).clone()
            a['wrist']=scene.tensor(scene.bundle.reset_qpos[c['hand_qadr']]).expand(scene.num_envs,5).clone()
            a['wrist0']=a['wrist'][0].clone()
            self.sides[side]=a
        self.fractions=scene.tensor([1,.5,.25,.125,.0625,.03125,.015625,.0078125])

    def fk(self,a,q):
        p=a['base_pos'].expand(*q.shape[:-1],3)
        r=a['base_rot'].expand(*q.shape[:-1],3,3)
        poses=[p];rotations=[r];axes=[];anchors=[]
        for i in range(7):
            p=p+(r@a['joint_pos'][i,:,None]).squeeze(-1)
            r=r@a['joint_rot'][i]
            axes.append((r@a['axis'][i,:,None]).squeeze(-1));anchors.append(p)
            r=r@rot(a['axis'][i],q[...,i])
            poses.append(p);rotations.append(r)
        tip=p+(r@a['tool_pos'][:,None]).squeeze(-1);tr=r@a['tool_rot']
        axis=torch.stack(axes,-1);anchor=torch.stack(anchors,-1)
        jp=torch.cross(axis.transpose(-1,-2),(tip[...,None]-anchor).transpose(-1,-2),dim=-1).transpose(-1,-2)
        return tip,tr,jp,axis,torch.stack(poses,-2),torch.stack(rotations,-3)

    def hand_fk(self,a,q):
        p=a['hand_rest_pos'].expand(*q.shape[:-1],3)
        r=a['hand_rest_rot'].expand(*q.shape[:-1],3,3)
        for i,kind in enumerate(a['hand_kind']):
            if kind==2:
                p=p+(r@a['hand_axis'][i,:,None]).squeeze(-1)*q[...,i,None]
            else:
                r=r@rot(a['hand_axis'][i],q[...,i])
        return p,r

    def error(self,p,r,target_p,target_r):
        delta=target_r@r.transpose(-1,-2)
        q=quaternion(delta)
        q=torch.where(q[...,:1]<0,-q,q)
        v=q[...,1:];length=v.norm(dim=-1,keepdim=True)
        angular=v*(2*torch.atan2(length,q[...,:1].clamp_min(0))/length.clamp_min(1e-8))
        return torch.cat((target_p-p,self.cfg['config']['orientation_weight_m']*angular),-1)

    def reset(self,ids):
        for a in self.sides.values():
            a['q'][ids]=a['q0'];a['wrist'][ids]=a['wrist0']
        self.publish()

    def update(self):
        s=self.scene;dt=s.bundle.manifest['control_dt']
        for side,a in self.sides.items():
            requested=s.data.qpos[:,a['hand_qadr']].clone()
            candidates=a['wrist'][:,None]+self.fractions[None,:,None]*(requested-a['wrist'])[:,None]
            tp,tr=self.hand_fk(a,candidates)
            start=a['q'][:,None,:].expand(-1,8,-1)
            q=start.clone()
            low=torch.maximum(a['limits'][:,0],start-a['velocity']*dt)
            high=torch.minimum(a['limits'][:,1],start+a['velocity']*dt)
            # Solve all line-search candidates together, avoiding device-to-host decisions.
            for _ in range(2*self.cfg['config']['iterations_per_frame']):
                p,r,jp,jr,_,_=self.fk(a,q)
                error=self.error(p,r,tp,tr)
                jac=torch.cat((jp,self.cfg['config']['orientation_weight_m']*jr),-2)
                delta=(jac.transpose(-1,-2)@torch.linalg.solve(
                    jac@jac.transpose(-1,-2)+self.cfg['config']['damping']**2*torch.eye(6,device=s.device),error[...,None])).squeeze(-1)
                q=(q+delta).clamp(low,high)
            p,r,*_=self.fk(a,q);error=self.error(p,r,tp,tr).norm(dim=-1)
            valid=error<1e-5;anyvalid=valid.any(-1)
            index=valid.int().argmax(-1)
            rows=torch.arange(s.num_envs,device=s.device)
            chosen_q=q[rows,index];chosen_wrist=candidates[rows,index]
            a['q']=torch.where(anyvalid[:,None],chosen_q,a['q'])
            a['wrist']=torch.where(anyvalid[:,None],chosen_wrist,a['wrist'])
            s.data.qpos[:,a['hand_qadr']]=a['wrist']
            limited=(a['wrist']-requested).abs().amax(-1)>1e-8
            s.data.qvel[:,a['hand_dof']]=torch.where(limited[:,None],0,s.data.qvel[:,a['hand_dof']])
            self.last_error[side]=dict(limited=limited,accepted_residual=error[rows,index])
        self.publish()

    def publish(self):
        d=self.scene.data
        for a in self.sides.values():
            p,r,_,_,positions,rotations=self.fk(a,a['q'])
            d.mocap_pos[:,a['mocap_ids']]=positions
            d.mocap_quat[:,a['mocap_ids']]=quaternion(rotations)
            d.mocap_pos[:,a['handle_id']]=p
            d.mocap_quat[:,a['handle_id']]=quaternion(r)
