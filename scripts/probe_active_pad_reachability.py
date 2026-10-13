#!/usr/bin/env python3
"""CPU static IK audit of the real guard's white/black-key pad permission.

Uses the source-model test builder, not a learned policy; endpoint feasibility
does not demonstrate a safe approach or a successful dynamical key press.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    import mujoco
    import numpy as np
    import torch
    from scipy.optimize import least_squares
    from tests.test_finger_clearance import RealGeometryTests
    RealGeometryTests.setUpClass()
    b = RealGeometryTests.bundle;g = RealGeometryTests.guard;m = b.model
    cases = []
    for finger, key in ((0, 51), (0, 52), (1, 52), (4, 52), (5, 39), (5, 40), (6, 40), (9, 40)):
        group = g.groups[finger];joints = group['chain'];qadr = m.jnt_qposadr[joints]
        data = b.reset_cpu();data.qpos[:] = b.origin_qpos
        data.mocap_pos[:] = b.origin_mocap_pos;data.mocap_quat[:] = b.origin_mocap_quat
        start = data.qpos[qadr].copy()
        bounds = m.jnt_range[joints].copy()
        bounds[-4:, 0] = g.low[finger].numpy();bounds[-4:, 1] = g.high[finger].numpy()
        key_min = g.key_min_xy[key].numpy();key_max = g.key_max_xy[key].numpy()
        normal = g.pressed_normal[key].numpy();center = g.pressed_center[key].numpy()
        plane = float(center@normal+g.key_half_z[key])
        desired_xy = (key_min+key_max)*.5;desired_xy[0] = min(.02625, float(key_max[0])-.015)
        def geom(geom_id):
            axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
            radius, half = m.geom_size[geom_id, :2]
            ends = data.geom_xpos[geom_id]+np.asarray([-1., 1.])[:, None]*axis*half
            return ends, radius, ends.min(0)-radius, ends.max(0)+radius
        def residual(q):
            data.qpos[qadr] = q;mujoco.mj_kinematics(m, data)
            endpoints, radius, low, high = geom(group['geoms'][-1])
            result = [*(endpoints.mean(0)[:2]-desired_xy),
                      (endpoints@normal).min()-radius-plane]
            result.extend(10*np.maximum(key_min+.00001-low[:2], 0))
            result.extend(10*np.maximum(high[:2]-key_max+.00001, 0))
            for geom_id in group['geoms'][:-1]:
                _, _, low, high = geom(geom_id)
                overlap = ((high[:2] >= g.key_min_xy[g.black_ids].numpy())
                           & (low[:2] <= g.key_max_xy[g.black_ids].numpy())).all(-1).any()
                threshold = (g.highest_top if overlap else g.white_top)+g.options['clearance_m']
                result.append(10*max(threshold-low[2], 0))
            result.extend(1e-6*(q-start))
            return np.asarray(result)
        found = []
        for trial in range(2):
            initial = start.copy()
            if trial:
                initial[-4:] = (bounds[-4:, 0]+bounds[-4:, 1])*.5
                initial[1] = .09
            solved = least_squares(residual, initial, bounds=(bounds[:, 0], bounds[:, 1]),
                                   max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            data.qpos[qadr] = solved.x;mujoco.mj_forward(m, data)
            physical = g.physical_qpos(torch.tensor(data.qpos, dtype=torch.float32)[None])
            assigned = torch.zeros(1, 10, 88, dtype=torch.bool);assigned[0, finger, key] = True
            metrics = g.evaluate(physical, torch.zeros(1, 2), assigned)
            ends, radius, low, high = geom(group['geoms'][-1])
            found.append(dict(success=bool(solved.success), nfev=int(solved.nfev),
                pad_allowed=bool(metrics['finger_terminal_allowed'][0, finger]),
                finger_deficit_m=float(metrics['finger_clearance_deficit_m'][0, finger]),
                pressed_top_distance_m=float((ends@normal).min()-radius-plane),
                terminal_xy_min=low[:2].tolist(), terminal_xy_max=high[:2].tolist(),
                joints={m.joint(j).name:float(v) for j,v in zip(joints,solved.x)}))
        best = min(found, key=lambda row: abs(row['pressed_top_distance_m'])+row['finger_deficit_m']+(0 if row['pad_allowed'] else 1))
        row = dict(finger=group['name'], midi=key+21, key_name=g.contract['key_names'][key],
                   key_xy_min=key_min.tolist(), key_xy_max=key_max.tolist(), best=best, starts=found)
        cases.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('starts','best')}|dict(best={k:v for k,v in best.items() if k!='joints'})), flush=True)
    output = ROOT/'preview/outputs/diagnostics/finger_clearance_20260927/active_pad_reachability.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(cases=cases, contract=g.contract,
        limitation='Independent finger endpoints at fixed origin support; no dynamic approach, no simultaneous-hand/self collision proof'), indent=2)+'\n')


if __name__ == '__main__': main()
