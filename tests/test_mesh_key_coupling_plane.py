"""Regression: trapezoidal black mesh inertia axes are not key plane axes."""
import json
from pathlib import Path
import numpy as np
import torch
from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
from tsensei_v3_2_pianist.fingering_targets import key_reference

def test_mesh_key_coupling_plane():
    recipe=make_recipe(height=.09,ty=.04,yaw=.2617993877991494,initial_buffer_time=0.,preparation_time=0.,task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',exact_note_timing=True)
    scene,_,_=build(recipe)
    b=SceneBundle(scene,disable_wrist_tz=True,disable_wrist_yaw=True,preparation=recipe['preparation'])
    ranges=json.loads(Path('configs/finger_coupling/angle_ranges_10_80.json').read_text())
    c=FingerCoupler(b,device='cuda:0',options=dict(schema=1,mode='distal_angle_range_v1',angle_ranges_deg=ranges,reduce_observation=False))
    d=b.reset_cpu();normals=[];mesh_normals=[]
    for gid in range(b.model.ngeom):
        name=b.model.geom(gid).name or ''
        if name.startswith('piano/') and '_key_geom_' in name:
            sid,_=key_reference(b.model,gid);normals.append(d.site_xmat[sid].reshape(3,3)[:,2])
            mesh_normals.append(d.geom_xmat[gid].reshape(3,3)[:,2])
    assert len(normals)==88
    np.testing.assert_allclose(normals,np.tile(c.normal.cpu().numpy(),(88,1)),atol=1e-7)
    assert not np.allclose(mesh_normals,normals,atol=1e-7)
    assert torch.isfinite(c.normal).all()
