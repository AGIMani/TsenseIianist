"""Physical bottom stop and visual activation regression, using real MuJoCo."""
import mujoco
import numpy as np
from tsensei_v3_2_pianist import suite
from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight, GREEN


def test_key_color_overlay_restores_and_preserves_physics():
    env = suite.load()
    try:
        env.reset()
        m, d = env.physics.model.ptr, env.physics.data.ptr
        h = ActiveKeyHighlight(m)
        scene = mujoco.MjvScene(m, maxgeom=1000)
        for k in (32, 33):
            d.qpos[h.qadr[k]] = h.maximum[k]
        mujoco.mj_forward(m, d)
        before, ctrl = d.qpos.copy(), d.ctrl.copy()
        mujoco.mjv_updateScene(m,d,mujoco.MjvOption(),None,mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_ALL,scene)
        original_count=scene.ngeom
        h.append(scene, d)
        # The white box gains one shell; the black mesh is colored directly.
        assert scene.ngeom == original_count+1
        black_geom=h.geom_ids[33]
        black_render=next(g for g in scene.geoms[:original_count] if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==black_geom)
        assert black_render.type==mujoco.mjtGeom.mjGEOM_MESH
        for g in (black_render,scene.geoms[original_count]):
            np.testing.assert_array_equal(g.rgba, GREEN)
        np.testing.assert_array_equal(d.qpos, before)
        np.testing.assert_array_equal(d.ctrl, ctrl)
        d.qpos[h.qadr] = 0
        mujoco.mj_forward(m, d)
        mujoco.mjv_updateScene(m,d,mujoco.MjvOption(),None,mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_ALL,scene)
        h.append(scene, d)
        assert scene.ngeom == original_count
    finally:
        env.close()


def test_legacy_hard_stop_under_continued_downward_target():
    from pathlib import Path
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    b = SceneBundle(Path(__file__).resolve().parents[1]/'training_mjlab/outputs/scene')
    m, d = b.model, b.reset_cpu()
    keyids = b.key_ids
    adrs = m.jnt_qposadr[keyids]
    levers = np.array([2*m.geom_size[np.flatnonzero(m.geom_bodyid == m.jnt_bodyid[j])[0], 0]
                       for j in keyids])
    for side in ('R', 'L'):
        wrist = m.actuator(side+'_Palm_Base_Tz').id
        np.testing.assert_array_equal(m.actuator_ctrlrange[wrist], [-.025, 0])
        d.ctrl[wrist] = -.025
        d.ctrl[m.actuator(side+'_Mid_MCP').id] += .1
    pen = over = 0.
    for _ in range(3000):
        mujoco.mj_step(m, d)
        over = max(over, float((np.maximum(d.qpos[adrs]-m.jnt_range[keyids, 1], 0)*levers).max()))
        for c in d.contact:
            a, b = m.geom(c.geom1).name, m.geom(c.geom2).name
            if 'key_geom_' in a and b.startswith(('right_/', 'left_/')):
                pen = max(pen, -c.dist)
    assert pen < .0005 and over < .0001, (pen, over)
    for side, k in [('R', 54), ('L', 33)]:
        assert abs(d.qpos[adrs[k]]-m.jnt_range[keyids[k], 1]) < .001
        act = m.actuator(side+'_Mid_MCP').id
        assert d.ctrl[act]-d.qpos[m.jnt_qposadr[m.actuator_trnid[act, 0]]] > .05
