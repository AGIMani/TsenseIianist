#!/usr/bin/env python3
"""CPU-only whole-keyboard translation audit of independent thumb endpoints.

This checks model-derived key geometry, original wrist ranges and active-contact
permission. It is not a simultaneous-finger IK or a dynamic playing test.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    import numpy as np
    import torch
    from tests.test_finger_clearance import RealGeometryTests, OPTIONS
    from tsensei_v3_2_pianist.mjlab_env.finger_clearance import FingerClearanceGuard

    torch.set_num_threads(1)
    source = ROOT/'preview/outputs/diagnostics/finger_clearance_20260927/active_pad_reachability.json'
    fixtures = json.loads(source.read_text())['cases']
    RealGeometryTests.setUpClass()
    bundle = RealGeometryTests.bundle
    guard = FingerClearanceGuard(bundle, device='cpu', options=dict(OPTIONS, active_command_mode='active_contact_v1'))
    physical = guard.physical_qpos(torch.tensor(bundle.origin_qpos, dtype=torch.float32)[None]).expand(176, -1).clone()
    assignment = torch.zeros((176, 10, 88), dtype=torch.bool)
    key_center = (guard.key_min_xy+guard.key_max_xy)*.5
    black = set(guard.black_ids.tolist())
    rows = []
    for hand, finger in enumerate((0, 5)):
        name = guard.groups[finger]['name']
        for key in range(88):
            row_index = hand*88+key
            fixture = next(row for row in fixtures if row['finger'] == name
                           and ((row['midi']-21 in black) == (key in black)))
            chain = guard.groups[finger]['chain']
            values = np.asarray([fixture['best']['joints'][bundle.model.joint(j).name] for j in chain])
            # Native TX is world +Y upstream of the wrist rotation. Other
            # wrist/finger angles are unchanged from the verified endpoint.
            values[0] += float(key_center[key, 1]-key_center[fixture['midi']-21, 1])
            bounds = bundle.model.jnt_range[chain].copy()
            bounds[-4:, 0] = guard.low[finger].numpy()
            bounds[-4:, 1] = guard.high[finger].numpy()
            within = bool(((values >= bounds[:, 0]-1e-7) & (values <= bounds[:, 1]+1e-7)).all())
            physical[row_index, guard.chain_indices[finger]] = torch.tensor(values, dtype=torch.float32)
            assignment[row_index, finger, key] = True
            rows.append(dict(finger=name, midi=key+21, black=key in black,
                             tx_m=float(values[0]), within_original_joint_ranges=within))
    result = guard.evaluate(physical, torch.zeros((176, 2)), assignment)
    wrong = torch.roll(assignment, 12, -1)
    rejected = guard.evaluate(physical, torch.zeros((176, 2)), wrong)
    for index, row in enumerate(rows):
        finger = 0 if index < 88 else 5
        row.update(force_permitted=bool(result['finger_active_force_permitted'][index, finger]),
                   deficit_m=float(result['finger_clearance_deficit_m'][index, finger]),
                   wrong_octave_permitted=bool(rejected['finger_active_force_permitted'][index, finger]))
    feasible = [row for row in rows if row['within_original_joint_ranges']]
    covered = sorted({row['midi'] for row in feasible if row['force_permitted']})
    summary = dict(poses=len(rows), within_original_ranges=len(feasible),
                   permitted_within_ranges=sum(row['force_permitted'] for row in feasible),
                   wrong_octave_permissions=sum(row['wrong_octave_permitted'] for row in feasible),
                   max_within_ranges_deficit_m=max(row['deficit_m'] for row in feasible),
                   distinct_keyboard_keys_covered=len(covered),
                   covered_midi=covered,
                   by_hand={name: dict(in_range=sum(row['within_original_joint_ranges'] for row in rows if row['finger'] == name),
                        permitted=sum(row['within_original_joint_ranges'] and row['force_permitted'] for row in rows if row['finger'] == name))
                            for name in ('right_thumb', 'left_thumb')})
    destination = source.with_name('keyboard_span.json')
    destination.write_text(json.dumps(dict(summary=summary, rows=rows,
        limitation='Independent translated static thumb endpoints only; no trajectory, physical contact, self/inter-hand collision or simultaneous reachability proof.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
