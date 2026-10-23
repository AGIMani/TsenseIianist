"""Mesh support points for IK clearance (no changes to collision geometry)."""
import mujoco
import numpy as np


def hand_collision_meshes(model):
    result={}
    for geom in range(model.ngeom):
        name=model.geom(geom).name or ''
        if not name.startswith(('right_','left_')) or '_collision_' not in name:continue
        mesh=int(model.geom_dataid[geom])
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH or mesh<0:
            raise ValueError(f'Expected full-link mesh: {name}')
        start=model.mesh_vertadr[mesh]
        result[geom]=model.mesh_vert[start:start+model.mesh_vertnum[mesh]].copy()
    return result


def thumb_cross_pairs(model, meshes):
    thumbs={g for g in meshes if '_thumb_' in model.geom(g).name}
    return [(g,h) for g in thumbs for h in meshes
            if model.geom(g).name.startswith('left_')!=model.geom(h).name.startswith('left_')
            and not (h in thumbs and h<g)]


def cross_clearance(model,data,meshes,pairs,margin=.004,dofs=None):
    """Conservative separating-axis gaps for thumb versus opposite-hand meshes."""
    world={g:vs@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g] for g,vs in meshes.items()}
    bounds={g:(vs.min(axis=0),vs.max(axis=0)) for g,vs in world.items()}
    if not pairs:return np.zeros(0 if dofs is None else (0,len(dofs)))
    first=np.array([bounds[g] for g,h in pairs]);second=np.array([bounds[h] for g,h in pairs])
    gaps=np.concatenate((first[:,0]-second[:,1],second[:,0]-first[:,1]),axis=1)
    chosen=np.argmax(gaps,axis=1)
    values=np.minimum(gaps[np.arange(len(pairs)),chosen]-margin,0.)
    if dofs is None:return values
    rows=np.zeros((len(pairs),len(dofs)))
    for i in np.flatnonzero(values<0):
        g,h=pairs[i];axis=int(chosen[i]%3)
        low,high=(g,h) if chosen[i]<3 else (h,g)
        lp=world[low][np.argmin(world[low][:,axis])]
        hp=world[high][np.argmax(world[high][:,axis])]
        jl=np.zeros((3,model.nv));jh=np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jl,None,lp,int(model.geom_bodyid[low]))
        mujoco.mj_jac(model,data,jh,None,hp,int(model.geom_bodyid[high]))
        rows[i]=(jl-jh)[axis,dofs]
    return rows


def thumb_meshes(model):
    result = {}
    for g in range(model.ngeom):
        name = model.geom(g).name or ''
        if '_thumb_' not in name or '_collision_' not in name:
            continue
        mid = int(model.geom_dataid[g])
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0:
            raise ValueError(f'Thumb clearance requires actual mesh geometry: {name}')
        start = model.mesh_vertadr[mid]
        result[g] = model.mesh_vert[start:start+model.mesh_vertnum[mid]].copy()
    if len(result) != 8:
        raise ValueError(f'Expected eight thumb collision links, found {len(result)}')
    return result


def lower_support(data, geom, vertices):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    heights = vertices @ rotation[2] + data.geom_xpos[geom, 2]
    vertex = vertices[int(np.argmin(heights))]
    return rotation @ vertex + data.geom_xpos[geom]


def clearance(model, data, meshes, height, dofs=None):
    values = []
    rows = []
    for geom, vertices in meshes.items():
        point = lower_support(data, geom, vertices)
        values.append(min(point[2]-height, 0.))
        if dofs is not None:
            jac = np.zeros((3, model.nv))
            if values[-1] < 0:
                mujoco.mj_jac(model, data, jac, None, point, int(model.geom_bodyid[geom]))
            rows.append(jac[2, dofs])
    return np.asarray(values if dofs is None else rows)


def keyboard_clearance(model, data, meshes, key_positions, key_sizes,
                       permitted_terminal_bodies=(), margin=.002, dofs=None):
    """Conservative whole-link clearance over overlapping keyboard footprints.

    Active terminal bodies may press; proximal links must remain above keys.
    This only supplies scratch-IK residuals, never changes physical geometry.
    """
    values=[];rows=[]
    for geom,vertices in meshes.items():
        body=int(model.geom_bodyid[geom])
        world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        lower=world.min(axis=0);upper=world.max(axis=0)
        overlap=np.all((key_positions[:,:2]-key_sizes[:,:2]<=upper[:2])&
                       (key_positions[:,:2]+key_sizes[:,:2]>=lower[:2]),axis=1)
        point=world[int(np.argmin(world[:,2]))]
        value=0.
        if body not in permitted_terminal_bodies and overlap.any():
            value=min(float(point[2]-np.max((key_positions+key_sizes)[overlap,2])-margin),0.)
        values.append(value)
        if dofs is not None:
            jac=np.zeros((3,model.nv))
            if value<0:mujoco.mj_jac(model,data,jac,None,point,body)
            rows.append(jac[2,dofs])
    return np.asarray(values if dofs is None else rows)


def landing_clearance(model, data, geom, vertices, key_position, key_size,
                      white_front_min=.022, margin=.001, dofs=None):
    """Keep low terminal vertices laterally inside the assigned key.

    A distal link may extend beyond the front edge while its pad presses the
    key top. Constraining that overhang in X prevents legitimate thumb presses.
    Only low vertices overlapping the key length constrain the lateral span.
    """
    world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
    near=((world[:,2]<key_position[2]+key_size[2]+.003)&
          (world[:,0]>=key_position[0]-key_size[0]-.001)&
          (world[:,0]<=key_position[0]+key_size[0]+.001))
    low=key_position[:2]-key_size[:2]+margin
    high=key_position[:2]+key_size[:2]-margin
    values=[];rows=[]
    for axis,bound,sign in [(1,low[1],-1),(1,high[1],1)]:
        errors=sign*(world[:,axis]-bound)
        errors[~near]=-np.inf
        vi=int(np.argmax(errors));value=max(float(errors[vi]),0.)
        values.append(value)
        if dofs is not None:
            jac=np.zeros((3,model.nv))
            if value>0:mujoco.mj_jac(model,data,jac,None,world[vi],int(model.geom_bodyid[geom]))
            rows.append(sign*jac[axis,dofs])
    return np.asarray(values if dofs is None else rows)
