"""User-approved ideal layout and frozen-baseline dynamics (remote CPU only)."""
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from dm_control import mjcf

from tsensei_v3_2_pianist.models.piano import piano_mjcf


def _baseline_builder():
    directory = os.environ.get("PIANIST_LAYOUT_BASELINE")
    if not directory:
        pytest.skip("Set PIANIST_LAYOUT_BASELINE to the verified immediate parent source")
    modules = []
    for name in ("piano_constants", "piano_mjcf"):
        path = Path(directory) / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"layout_baseline_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    modules[1].consts = modules[0]
    return modules[1].build


def _layout(model):
    rows = []
    for i in range(88):
        black = (i + 21) % 12 in (1, 3, 6, 8, 10)
        color = "black" if black else "white"
        body = model.body(f"{color}_key_{i}")
        geom = model.geom(f"{color}_key_geom_{i}")
        rows.append(dict(index=i, midi=i+21, color=color,
                         center_mm=float(body.pos[1]*1000),
                         width_mm=float(model.site(f"{color}_key_site_{i}").size[1]*2000)))
    return rows


def test_compiled_layout():
    physics = mjcf.Physics.from_mjcf_model(piano_mjcf.build())
    rows = _layout(physics.model.ptr)
    centers = np.array([r["center_mm"] for r in rows])
    assert np.all(np.diff(centers) > 0)
    white = [r for r in rows if r["color"] == "white"]
    black = [r for r in rows if r["color"] == "black"]
    assert (len(white), len(black)) == (52, 36)
    np.testing.assert_allclose([r["width_mm"] for r in white], 22.5, atol=1e-12)
    np.testing.assert_allclose([r["width_mm"] for r in black], 10, atol=1e-12)
    assert white[0]["center_mm"]-11.25 == pytest.approx(-612)
    assert white[-1]["center_mm"]+11.25 == pytest.approx(612)
    np.testing.assert_allclose(np.diff([r["center_mm"] for r in white])-22.5,
                               1.058823529411765, atol=1e-10)
    for octave in range(7):
        k = 12*octave
        assert centers[k+6]-centers[k+4] == pytest.approx(28.2705882352941)
        for a,b in ((9,11),(11,13)):
            assert centers[k+b]-centers[k+a] == pytest.approx(26.9243697478992)
        assert centers[k+9]-centers[k+6] == pytest.approx(41.396218487395)
        assert centers[k+4]-centers[k+1] == pytest.approx(41.396218487395)
    output = os.environ.get("PIANIST_LAYOUT_OUTPUT")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/"compiled_key_layout.json").write_text(json.dumps({
            "units":"mm", "source":"compiled MuJoCo body positions and geom sizes",
            "layout":"user-approved ideal five/seven subdivision; not hardware measurement",
            "keys":rows}, indent=2)+"\n")


UNCHANGED = (
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_quat",
    "jnt_axis", "jnt_pos", "jnt_range", "jnt_stiffness", "jnt_solref", "jnt_solimp",
    "dof_armature", "dof_damping", "dof_frictionloss", "qpos_spring",
    "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity",
    "geom_condim", "geom_margin", "geom_gap",
    "actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm",
)


@pytest.mark.parametrize("actuated", [False, True])
def test_only_layout_changes_against_frozen_parent(actuated):
    old = mjcf.Physics.from_mjcf_model(_baseline_builder()(add_actuators=actuated))
    new = mjcf.Physics.from_mjcf_model(piano_mjcf.build(add_actuators=actuated))
    for field in UNCHANGED:
        np.testing.assert_allclose(getattr(new.model.ptr, field), getattr(old.model.ptr, field),
                                   rtol=1e-13, atol=1e-15, err_msg=field)
    np.testing.assert_array_equal(new.model.body_pos[:,[0,2]], old.model.body_pos[:,[0,2]])
    white_geoms = [g for g in range(new.model.ngeom) if 'black_key_geom_' not in (new.model.geom(g).name or '')]
    for field in ('geom_size', 'geom_pos', 'geom_quat'):
        np.testing.assert_array_equal(getattr(new.model, field)[white_geoms], getattr(old.model, field)[white_geoms])
    np.testing.assert_array_equal(new.model.site_size[:,[0,2]], old.model.site_size[:,[0,2]])
    # The same applied torques must produce the same isolated key motion.
    for step in range(200):
        force = np.sin(np.arange(88)*.3+step*.1)*.03
        for physics in (old,new):
            physics.data.qfrc_applied[:] = force
            physics.step()
        np.testing.assert_allclose(new.data.qpos, old.data.qpos, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(new.data.qvel, old.data.qvel, rtol=1e-11, atol=1e-12)


def test_complete_hand_scene_preserves_physics():
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.suite.tasks import base
    baseline = _baseline_builder()
    spec = importlib.util.spec_from_file_location("layout_baseline_base",
        Path(os.environ["PIANIST_LAYOUT_BASELINE"])/"base.py")
    old_base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_base)
    envs = []
    try:
        for builder,add_hand in ((baseline,old_base.PianoTask._add_hand),
                                 (piano_mjcf.build,base.PianoTask._add_hand)):
            with patch.object(piano_mjcf,"build",builder), \
                 patch.object(base.PianoTask,"_add_hand",add_hand), patch.dict(os.environ, {
                "TSENSEI_V32_COLLISION_PROFILE":"convex_hull_v1",
                "TSENSEI_V32_DRIVE_PROFILE":"shadow_v1"}):
                env = suite.load("Toy", task_kwargs={"control_timestep":.05,
                    "physics_timestep":.005,"initial_buffer_time":0.,
                    "gravity_compensation":True,"task_profile":"tsensei_collision_v1"})
                env.reset()
                envs.append(env)
        old,new = [e.physics.model.ptr for e in envs]
        for field in UNCHANGED:
            np.testing.assert_allclose(getattr(new,field),getattr(old,field),
                                       rtol=1e-12,atol=1e-14,err_msg=field)
        from tsensei_v3_2_pianist.key_activation import KEY_ACTIVATION_FRACTION
        assert KEY_ACTIVATION_FRACTION == 1.0
        # Short CPU physics smoke check, no policy inference or learning.
        env=envs[1]
        for _ in range(20):
            env.physics.step()
            assert np.isfinite(env.physics.data.qpos).all()
            assert np.isfinite(env.physics.data.qvel).all()
        assert not np.any(env.physics.data.warning.number)
    finally:
        for env in envs:
            env.close()
