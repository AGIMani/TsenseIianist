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
        scene = mujoco.MjvScene(m, maxgeom=100)
        for k in (33, 54):
            d.qpos[h.qadr[k]] = h.maximum[k]
        mujoco.mj_forward(m, d)
        before, ctrl = d.qpos.copy(), d.ctrl.copy()
        scene.ngeom = 0
        h.append(scene, d)
        assert scene.ngeom == 2
        for g in scene.geoms[:2]:
            np.testing.assert_array_equal(g.rgba, GREEN)
        np.testing.assert_array_equal(d.qpos, before)
        np.testing.assert_array_equal(d.ctrl, ctrl)
        d.qpos[h.qadr] = 0
        mujoco.mj_forward(m, d)
        scene.ngeom = 0
        h.append(scene, d)
        assert scene.ngeom == 0
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
        wrist = m.actuator(side+'_Wrist_TZ').id
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
