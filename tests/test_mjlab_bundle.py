"""Portable physical scene contract; runs without installing MJLab or Torch."""
import shutil

import mujoco
import numpy as np
import pytest

from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, export_scene


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp('frozen_scene')
    export_scene(directory, control_timestep=.05, physics_timestep=.005)
    return directory


def test_exported_scene_has_frozen_pose_and_exact_control_mapping(exported):
    bundle = SceneBundle(exported)
    d = bundle.reset_cpu()
    assert bundle.manifest['gravity_compensation'] is True
    assert np.count_nonzero(bundle.model.body_gravcomp == 1) == 44
    assert (bundle.model.nq, bundle.model.nv, bundle.model.nu) == (138, 138, 50)
    assert len(bundle.manifest['action_names']) == 51
    assert len(set(bundle.action_ids)) == 50
    np.testing.assert_allclose(d.ctrl[bundle.action_ids], bundle.neutral_action[:-1])
    normalized = bundle.normalize(bundle.neutral_action)
    np.testing.assert_allclose(bundle.low+(normalized+1)*.5*(bundle.high-bundle.low), bundle.neutral_action)
    for hand in bundle.manifest['hands'].values():
        body = bundle.model.body(hand['body']).id
        np.testing.assert_allclose(d.xpos[body], hand['position'], atol=1e-10)
        sites = bundle.ids(mujoco.mjtObj.mjOBJ_SITE, hand['sites'])
        np.testing.assert_allclose(d.site_xpos[sites], hand['fingertips'], atol=1e-10)
    assert not bundle.key_activation(d.qpos).any()


def test_bundle_can_move_and_rejects_asset_tampering(exported, tmp_path):
    moved = tmp_path/'elsewhere'
    shutil.copytree(exported, moved)
    bundle = SceneBundle(moved)
    assert bundle.reset_cpu().qpos.shape == (138,)
    xml = moved/'scene.xml'
    xml.write_text(xml.read_text()+'\n')
    with pytest.raises(ValueError, match='modified'):
        SceneBundle(moved)


def test_export_refuses_overwrite_and_missing_names(exported):
    with pytest.raises(FileExistsError):
        export_scene(exported)
    bundle = SceneBundle(exported)
    with pytest.raises(ValueError, match='Missing'):
        bundle.ids(mujoco.mjtObj.mjOBJ_JOINT, ['missing_joint'])


def test_cpu_hand_press_and_release_with_frozen_model(exported):
    bundle = SceneBundle(exported)
    d = bundle.reset_cpu()
    d.ctrl[:] = bundle.reset_ctrl
    # Move the right index over the keybed from the frozen TY=0, Yaw=.25 pose.
    for name, value in [('R_Palm_Base_Ty', .04), ('R_Index_MCP', 1.2),
                        ('R_Index_PIP', .3), ('R_Index_DIP', 0.)]:
        d.ctrl[bundle.model.actuator(name).id] = value
    activated = set()
    touched = False
    for _ in range(300):
        mujoco.mj_step(bundle.model, d)
        activated.update(np.flatnonzero(bundle.key_activation(d.qpos)))
        for c in d.contact:
            names = [bundle.model.geom(g).name for g in (c.geom1, c.geom2)]
            a, b = names
            touched |= (('key_geom_' in a and b.startswith(('right_/', 'left_/')))
                        or ('key_geom_' in b and a.startswith(('right_/', 'left_/'))))
    assert touched and activated
    d.ctrl[:] = bundle.reset_ctrl
    for _ in range(500):
        mujoco.mj_step(bundle.model, d)
    assert not bundle.key_activation(d.qpos).any()
