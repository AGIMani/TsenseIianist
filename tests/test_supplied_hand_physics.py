"""Remote CPU compile/inertia/short physics acceptance of the current hand."""
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch


def test_current_entry_pose_and_one_second_physics(tmp_path):
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.action_mapping import physical_bounds
    from tsensei_v3_2_pianist.mjlab_env.finger_coupling import FingerCoupler
    from tests.test_current_model_contract import test_compiled_current_hands_preserve_all_explicit_urdf_inertias
    root = Path(__file__).resolve().parents[1]
    scene, task, _ = build(make_recipe(world_z=.06, world_x=.14, palm_base_yaw=np.pi/6,
        task_profile='tsensei_collision_v1', collision_profile='convex_hull_v1',
        drive_profile='shadow_v1', control_hz=20., physics_hz=200.,
        initial_buffer_time=0, exact_note_timing=True))
    bundle = SceneBundle(scene, disable_wrist_tz=True, disable_wrist_yaw=True)
    low, high = physical_bounds('direct_joint_v1', bundle.manifest['action_names'], bundle.low, bundle.high,
                               range_scales=bundle.manifest.get('action_range_scales'))
    options = dict(schema=1, mode='distal_angle_range_v1', reduce_observation=False,
        angle_ranges_deg=json.loads((root/'configs/finger_coupling/angle_ranges_60_80.json').read_text()))
    coupler = FingerCoupler(bundle, device='cpu', options=options, physical_low=low, physical_high=high)
    model = bundle.model
    data = bundle.reset_cpu()
    output = Path(os.environ.get('PIANIST_MODEL_ACCEPTANCE_OUTPUT', tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    mjb = output/'model.mjb'
    mujoco.mj_saveModel(model, str(mjb))
    old = os.environ.get('PIANIST_CURRENT_MODEL_MJB')
    try:
        os.environ['PIANIST_CURRENT_MODEL_MJB'] = str(mjb)
        test_compiled_current_hands_preserve_all_explicit_urdf_inertias()
    finally:
        if old is None: os.environ.pop('PIANIST_CURRENT_MODEL_MJB', None)
        else: os.environ['PIANIST_CURRENT_MODEL_MJB'] = old
    max_contacts=0;max_speed=0.;max_limit_excess=0.
    for control in range(20):
        targets,_ = coupler.apply(torch.tensor(bundle.neutral_action,dtype=torch.float32)[None],
            actual_qpos=torch.tensor(data.qpos.copy(),dtype=torch.float32)[None],
            body_xmat=torch.tensor(data.xmat.copy(),dtype=torch.float32)[None])
        full=targets[0].numpy()
        data.ctrl[bundle.action_ids] = full[np.asarray(bundle.active_action_indices)]
        for _ in range(10):
            mujoco.mj_step(model,data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            max_contacts=max(max_contacts,data.ncon)
            max_speed=max(max_speed,float(np.abs(data.qvel).max()))
            limited=np.flatnonzero(model.jnt_limited)
            q=data.qpos[model.jnt_qposadr[limited]]
            excess=np.maximum(model.jnt_range[limited,0]-q,q-model.jnt_range[limited,1])
            max_limit_excess=max(max_limit_excess,float(excess.max()))
    assert not any(w.number for w in data.warning)
    report=dict(control_hz=20,physics_hz=200,physics_steps=200,finite=True,
        mujoco_version=mujoco.__version__,warnings=[int(w.number) for w in data.warning],
        coupling_reference=coupler.reference_diagnostics,max_contacts=max_contacts,
        max_abs_generalized_velocity=max_speed,max_physical_limit_excess_m_or_rad=max_limit_excess,
        scope='CPU short fixed-target physics and compiled inertia check; no policy, F1 or hardware safety claim')
    (output/'physics.json').write_text(json.dumps(report,indent=2)+'\n')
