"""Validate the repaired geometry on the same CUDA graph path as training."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='GPU validation requires Slurm')


def test_repaired_coupling_cpu_cuda_and_graph_agree():
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
    from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
    import mujoco

    root = Path(__file__).resolve().parents[1]
    recipe = make_recipe(task_profile='tsensei_collision_v1', drive_profile='shadow_v1',
        initial_buffer_time=0, preparation_time=1., preparation_lift=.025,
        preparation_mode='policy_v1', wrist_pitch_range_scale=.8)
    scene, task, _ = build(recipe)
    scene, task = without_arms(scene, task, scene.parent / 'hand_only')
    bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True, preparation=recipe['preparation'])
    assert bundle.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    low, high = physical_bounds('direct_joint_v1', bundle.manifest['action_names'],
        bundle.low, bundle.high, range_scales=bundle.manifest.get('action_range_scales'))
    options = dict(schema=1, mode='distal_angle_range_v1', reduce_observation=False,
        angle_ranges_deg=json.loads((root / 'configs/finger_coupling/angle_ranges_60_80.json').read_text()))
    cpu = FingerCoupler(bundle, device='cpu', options=options, physical_low=low, physical_high=high)
    gpu = FingerCoupler(bundle, device='cuda:0', options=options, physical_low=low, physical_high=high)
    data = bundle.reset_cpu(); data.qpos[:] = bundle.origin_qpos
    mujoco.mj_forward(bundle.model, data)
    qpos = torch.tensor(np.array(data.qpos), dtype=torch.float32)[None].expand(27, -1).clone()
    xmat = torch.tensor(np.array(data.xmat), dtype=torch.float32)[None].expand(27, -1, -1).clone()
    physical = torch.tensor(bundle.neutral_action, dtype=torch.float32)[None].expand(27, -1).clone()
    indices = cpu.indices[:, 0].numpy()
    physical[:, cpu.indices[:, 0]] = torch.tensor(np.array(low)[indices], dtype=torch.float32)[None] + (
        torch.linspace(0., 1., 27)[:, None] * torch.tensor((np.array(high) - low)[indices], dtype=torch.float32)[None])
    expected, info = cpu.apply(physical, actual_qpos=qpos, body_xmat=xmat)
    assert not info['coupling_infeasible'].any()
    physical, qpos, xmat = [v.cuda() for v in (physical, qpos, xmat)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gpu.apply(physical, actual_qpos=qpos, body_xmat=xmat)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, diagnostic = gpu.apply(physical, actual_qpos=qpos, body_xmat=xmat)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-4, rtol=0)
    torch.testing.assert_close(diagnostic['coupling_target_angle_deg'].cpu(),
                               info['coupling_target_angle_deg'], atol=2e-3, rtol=0)
    assert not diagnostic['coupling_infeasible'].any()
