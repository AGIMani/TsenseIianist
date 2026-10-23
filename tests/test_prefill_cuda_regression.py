"""Regression for the first multi-collector prefill step (Slurm GPU only)."""
import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not os.environ.get('SLURM_JOB_ID'), reason='GPU validation requires Slurm')


def test_large_eager_collector_step_with_hard_limits_and_coupling():
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene

    recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1',
                         initial_buffer_time=0., preparation_time=0.)
    scene_dir, task, _ = build(recipe)
    scene_dir, _ = without_arms(scene_dir, task, scene_dir.parent / 'prefill_regression')
    scene = PianoScene(
        scene_dir, num_envs=171, physics_graph=False,
        action_profile='direct_joint_v1', disable_wrist_tz=True,
        disable_wrist_yaw=True, preparation=recipe['preparation'],
        control_limits=dict(schema=1, mode='target_clip_slew_v1', channels={},
                            joint_state_limits='hard_v1'),
        finger_coupling=dict(schema=1, mode='distal_60deg_v1',
                             target_angle_deg=60., reduce_observation=False),
        collision_diagnostics=True)
    action = scene.policy_action(scene.bundle.neutral_action).expand(scene.num_envs, -1)
    result = scene.step(action)
    assert torch.isfinite(scene.data.qpos[:]).all()
    assert torch.isfinite(scene.data.qvel[:]).all()
    assert torch.isfinite(result['qpos']).all()
    assert torch.isfinite(result['qvel']).all()
