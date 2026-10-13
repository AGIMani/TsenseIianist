#!/usr/bin/env python3
"""CPU counterexample search for endpoint-only geometric target constraints."""
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
    from tests.test_finger_clearance import RealGeometryTests
    RealGeometryTests.setUpClass()
    g = RealGeometryTests.guard;b = RealGeometryTests.bundle
    def inputs(n):
        q = torch.tensor(b.origin_qpos, dtype=torch.float32).expand(n, -1).clone()
        mocap = torch.tensor(b.origin_mocap_pos, dtype=torch.float32).expand(n, -1, -1).clone()
        return g.physical_qpos(q), q, mocap, torch.zeros(n, 10, 88, dtype=torch.bool)
    p, q, mocap, role = inputs(601)
    p[:, 46] = .07
    p[:, 45] = torch.linspace(float(b.low[45]), float(b.high[45]), 601)
    metrics = g.evaluate(p, torch.zeros(601, 2), role)
    deficit = metrics['finger_clearance_deficit_m'][:, 5]
    safe = deficit <= 1e-6
    zones = []
    for i in range(1, len(safe)-1):
        if safe[i-1] and not safe[i]:
            end = next((j for j in range(i+1, len(safe)) if safe[j]), None)
            if end is not None:
                zones.append((i-1, end))
    lateral = None
    if zones:
        start, end = min(zones, key=lambda pair:pair[1]-pair[0])
        middle = start+int(deficit[start:end+1].argmax())
        lateral = dict(start_tx=float(p[start, 45]), end_tx=float(p[end, 45]), ty=.07,
            start_deficit_m=float(deficit[start]), end_deficit_m=float(deficit[end]),
            intermediate_tx=float(p[middle, 45]), intermediate_deficit_m=float(deficit[middle]),
            note='Both endpoint finger targets clear, intermediate target FK violates local black-key envelope; not a simulated trajectory')
        data = b.reset_cpu();data.qpos[:] = b.origin_qpos
        data.mocap_pos[:] = b.origin_mocap_pos;data.mocap_quat[:] = b.origin_mocap_quat
        for act, full in zip(b.action_ids, b.active_action_indices):
            jid = int(b.model.actuator_trnid[act,0]);data.qpos[b.model.jnt_qposadr[jid]] = p[middle,full]
        mujoco.mj_forward(b.model, data)
        lateral['native_contacts'] = [dict(geoms=[b.model.geom(int(c.geom1)).name,b.model.geom(int(c.geom2)).name],
            dist=float(c.dist)) for c in data.contact[:data.ncon] if c.dist<=0 and
            any('left_hand_thumb' in (b.model.geom(int(x)).name or '') for x in (c.geom1,c.geom2))]
    n = 64
    p, q, mocap, role = inputs(n)
    gen = torch.Generator().manual_seed(599)
    endpoints = []
    for _ in range(2):
        requested = p.clone()
        requested[:, g.finger_indices] = g.low+(g.high-g.low)*torch.rand(n, 10, 4, generator=gen)
        applied, _ = g.filter(requested, actual_qpos=q, mocap_pos=mocap, assigned_keys=role)
        endpoints.append(applied)
    path_results = []
    for fraction in (.25, .5, .75):
        intermediate = endpoints[0]+fraction*(endpoints[1]-endpoints[0])
        mid = g.evaluate(intermediate, torch.zeros(n, 2), role)['finger_clearance_deficit_m']
        worst = int(mid.flatten().argmax());world, finger = divmod(worst, 10)
        path_results.append(dict(fraction=fraction, violating_finger_count=int((mid>1e-6).sum()),
            max_deficit_m=float(mid.max()), example_world=world, example_finger=g.groups[finger]['name'],
            start_angles=endpoints[0][world,g.finger_indices[finger]].tolist(),
            end_angles=endpoints[1][world,g.finger_indices[finger]].tolist()))
    endpoint_max = [float(g.evaluate(x,torch.zeros(n,2),role)['finger_clearance_deficit_m'].max()) for x in endpoints]
    result = dict(lateral_example=lateral, nonlinear_path_examples=path_results,
        endpoint_max_deficit_m=endpoint_max, random_paths=n, fingers_per_path=10,
        limitation='Finite CPU FK samples, not dynamical rollouts; examples disprove endpoint=>path implication but do not measure policy failure frequency')
    out = ROOT/'preview/outputs/diagnostics/finger_clearance_20260927/path_counterexamples.json'
    out.parent.mkdir(parents=True, exist_ok=True);out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
