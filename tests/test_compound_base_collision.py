"""Regression for false base/swing contacts on the recorded exper028 start."""
import json
from pathlib import Path

from dm_control import mjcf
import mujoco
import numpy as np
import pytest

from tsensei_v3_2_pianist.models.hands import collision_proxies, urdf_model


@pytest.mark.parametrize('side', ['left','right'])
def test_recorded_start_has_no_base_swing_collision(side, monkeypatch):
    monkeypatch.setenv('TSENSEI_V32_COLLISION_PROFILE','convex_hull_v1')
    assets=collision_proxies.CONFIG_PATH.parent
    poses=json.loads((Path(__file__).parent/'fixtures/collision_mesh_repair_start.json').read_text())
    root,_=urdf_model.build(assets/'urdf'/f'TsenseI_{side}.urdf',side+'_')
    physics=mjcf.Physics.from_mjcf_model(root)
    try:
        m,d=physics.model.ptr,physics.data.ptr
        for i in range(m.njnt):
            name=m.joint(i).name.split('/')[-1]
            d.qpos[m.jnt_qposadr[i]]=poses[f'{side}_/{name}']
        mujoco.mj_forward(m,d)
        base=m.body('base_link').id
        collisions=[]
        for c in d.contact:
            ids=[int(m.geom_bodyid[g]) for g in c.geom]
            names=[m.body(b).name for b in ids]
            if c.dist<=0 and base in ids and any('swing_link' in n for n in names):
                collisions.append((names,float(c.dist)))
        assert not collisions,collisions
        geoms=np.flatnonzero((m.geom_bodyid==base)&(m.geom_group==3))
        assert len(geoms)==79
        assert np.all(m.geom_contype[geoms]==1)
        assert np.all(m.geom_conaffinity[geoms]==1)
    finally:
        physics.free()


def test_compound_paths_cannot_escape_assets():
    config={'hands':{'left':{'base_link':dict(type='compound',parts=[{'mesh_file':'../../../etc/passwd'}])}}}
    with pytest.raises(ValueError,match='escapes/missing'):
        collision_proxies.for_link(config,'left','base_link')
