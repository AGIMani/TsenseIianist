"""Remote-only actual root-position checks; no policy, training or rendering."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('PIANIST_REMOTE_MODEL_TESTS') != '1',
                                reason='Model construction is authorized only on remote volc-a32')


@pytest.fixture(scope='module', params=[(.16, .15, -.15, .09), (.20, .11, -.12, .12)])
def scene_and_pose(request):
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import build, make_recipe
    x, yr, yl, z = request.param
    recipe = make_recipe(world_x=x, right_world_y=yr, left_world_y=yl, world_z=z,
        palm_base_yaw=np.pi / 12, initial_buffer_time=0,
        task_profile='tsensei_collision_v1', drive_profile='shadow_v1')
    scene, _, _ = build(recipe)
    return scene, {'right': np.array([x, yr, z]), 'left': np.array([x, yl, z])}


def test_noarm_actual_palm_origins_and_names(scene_and_pose):
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    scene, expected = scene_and_pose
    bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True)
    data = bundle.reset_cpu()
    for side in ('right', 'left'):
        np.testing.assert_allclose(data.body(side + '_/base_link').xpos, expected[side], atol=2e-12, rtol=0)
        for axis in ('tx', 'ty', 'pitch'):
            assert bundle.model.joint(side + '_/palm_base_' + axis).id >= 0
    assert not any('forearm_' in bundle.model.joint(j).name for j in range(bundle.model.njnt))
    pose = json.loads((scene / 'manifest.json').read_text())['source_pose']
    for side in expected:
        np.testing.assert_allclose(pose[side + '_initial_world_xyz_m'], expected[side], atol=2e-12, rtol=0)


def test_arm_zero_request_keeps_same_world_origin(scene_and_pose):
    from tsensei_v3_2_pianist.mjlab_env.arm_options import add_arguments, from_args
    from tsensei_v3_2_pianist.mjlab_env.arm_scene import SceneArmFollower
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    # Existing test fixture, never a runtime default or a hardware calibration.
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    angles = ['40', '-30', '20', '-50', '40', '10', '-20']
    options = from_args(parser.parse_args(['--arms', '--arm-initial-right-deg', *angles,
                                           '--arm-initial-left-deg', *angles]))
    scene, expected = scene_and_pose
    bundle = SceneBundle(scene, disable_wrist_tz=False, disable_wrist_yaw=True, arm_follow=options)
    data = bundle.reset_cpu()
    follower = SceneArmFollower(bundle, num_envs=1, device='cpu')
    for side in expected:
        np.testing.assert_allclose(data.body(side + '_/base_link').xpos, expected[side], atol=2e-12, rtol=0)
        np.testing.assert_allclose(follower.wrists[side]['rest_pos'], expected[side], atol=2e-12, rtol=0)
