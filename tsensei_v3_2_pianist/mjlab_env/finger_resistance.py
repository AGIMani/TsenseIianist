"""Per-world dry-friction thresholds from the current physical joint pose."""
import numpy as np
import warp as wp
import mujoco_warp as mjwarp
from tsensei_v3_2_pianist.finger_resistance import Resistance

@wp.kernel
def update_threshold(joints: wp.array(dtype=wp.int32), dofs: wp.array(dtype=wp.int32),
                     masses: wp.array2d(dtype=wp.float32), params: wp.array2d(dtype=wp.float32),
                     gravity: wp.vec3, xipos: wp.array2d(dtype=wp.vec3),
                     anchor: wp.array2d(dtype=wp.vec3), axis: wp.array2d(dtype=wp.vec3),
                     friction: wp.array2d(dtype=wp.float32)):
    w,i=wp.tid()
    j=joints[i]
    g=float(0.0)
    for b in range(masses.shape[1]):
        g+=wp.dot(wp.cross(xipos[w,b]-anchor[w,j], masses[i,b]*gravity),axis[w,j])
    friction[w,dofs[i]]=params[i,0]+params[i,1]*wp.abs(g)

class BatchedResistance:
    def __init__(self, model, wm, wd, worlds, device):
        ref=Resistance(model)
        self.rows=ref.rows
        self.wm,self.wd,self.worlds=wm,wd,worlds
        if not self.rows:return
        masses=np.zeros((len(self.rows),model.nbody),np.float32)
        for i,(_,_,_,bodies) in enumerate(self.rows):masses[i,bodies]=model.body_mass[bodies]
        self.joints=wp.array([r[0] for r in self.rows],dtype=wp.int32,device=device)
        self.dofs=wp.array([r[1] for r in self.rows],dtype=wp.int32,device=device)
        self.masses=wp.array(masses,dtype=wp.float32,device=device)
        self.params=wp.array([[r[2]['T0'],r[2]['kg']] for r in self.rows],dtype=wp.float32,device=device)
        self.gravity=wp.vec3(*model.opt.gravity)
        wm.dof_frictionloss=wp.array(np.tile(model.dof_frictionloss,(worlds,1)).astype(np.float32),dtype=wp.float32,device=device)
    def update(self):
        if not self.rows:return
        mjwarp.kinematics(self.wm,self.wd)
        mjwarp.com_pos(self.wm,self.wd)
        wp.launch(update_threshold,dim=(self.worlds,len(self.rows)),inputs=[self.joints,self.dofs,self.masses,self.params,self.gravity,self.wd.xipos,self.wd.xanchor,self.wd.xaxis,self.wm.dof_frictionloss])
