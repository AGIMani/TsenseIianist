"""Joint-specific PD and posture-dependent resistance on current A3.2 assets."""
import json
from pathlib import Path
import numpy as np

PATH = Path(__file__).parent/'assets/finger_resistance.json'

def settings():
    doc = json.loads(PATH.read_text())
    if doc['contract'] != 'a3_resistance_compensated_v1' or len(doc['joints']) != 20:
        raise ValueError('Invalid fitted finger parameter contract')
    for name, row in doc['joints'].items():
        for field in ('kp','kd','effort_nm','armature_kg_m2','alpha','T0','b','kg'):
            value = row[field]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid fitted finger parameter: {name}.{field}')
        if min(row['kp'],row['effort_nm'],row['armature_kg_m2']) <= 0:
            raise ValueError(f'Invalid positive drive parameter: {name}')
    return doc

def short_name(name):
    return name.split('_hand_',1)[1]

def parameters(name):
    return settings()['joints'][short_name(name)]

class Resistance:
    def __init__(self, model):
        self.model=model
        rows=[]
        for j in range(model.njnt):
            name=model.joint(j).name or ''
            if '_hand_' not in name:
                continue
            p=parameters(name)
            # Activate only for the new compiled drive, never silently reinterpret old scenes.
            acts=np.flatnonzero(model.actuator_trnid[:,0]==j)
            if not len(acts) or not np.isclose(model.actuator_biasprm[acts[0],2],-p['kd']):
                continue
            children=[]
            for b in range(1,model.nbody):
                ancestor=b
                while ancestor and ancestor!=model.jnt_bodyid[j]:
                    ancestor=int(model.body_parentid[ancestor])
                if ancestor:children.append(b)
            if not np.allclose(model.body_gravcomp[children],1):
                raise ValueError('Current fitted finger resistance requires full gravity compensation')
            rows.append((j,int(model.jnt_dofadr[j]),p,children))
        self.rows=rows
    def update(self,data):
        import mujoco
        m=self.model
        mujoco.mj_kinematics(m,data)
        mujoco.mj_comPos(m,data)
        for j,v,p,bodies in self.rows:
            force=m.body_mass[bodies,None]*m.opt.gravity
            G=float(np.sum(np.cross(data.xipos[bodies]-data.xanchor[j],force)@data.xaxis[j]))
            m.dof_frictionloss[v]=p['T0']+p['kg']*abs(G)
        # Native body gravcomp cancels G outside the PD actuator torque cap.
