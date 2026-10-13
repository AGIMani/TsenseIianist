#!/usr/bin/env python3
"""Bake the approved central-aperture alignment into optional preview meshes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'preview/assets/arm_hand_connect'
    alignment = json.loads((ROOT / 'preview/outputs/connector_alignment/hole_aligned_mount.json').read_text())
    hand_from_assembly = np.linalg.inv(np.array(alignment['assembly_from_left_hand']))
    output = ROOT / 'preview/assets/hand_connector'
    output.mkdir(parents=True, exist_ok=True)
    config = {'schema': 1, 'alignment': alignment,
              'source_step_sha256': json.loads((source / 'assembly.json').read_text())['source_sha256'],
              'right_hand_connector': 'provisional mirror of left attachment; not a manufacturer right-hand design',
              'meshes': {}, 'mesh_sha256': {}, 'nominal_link7_to_hand': {},
              'offsets': {s: {'xyz_mm': [0., 0., 0.], 'rpy_deg': [0., 0., 0.]} for s in ('left', 'right')},
              'limitations': ['TJ-ARM-03: screw-hole fit and flange clocking need calibration',
                              'Connector visuals only: no assumed mass, inertia or collision forces']}
    rotation = Rotation.from_euler('xyz', [np.pi / 2, 0, 0]).as_matrix()
    for side in ('left', 'right'):
        mirror = np.diag([1., -1., 1.]) if side == 'right' else np.eye(3)
        flange_in_hand = mirror @ hand_from_assembly[:3, 3]
        tool = np.eye(4)
        tool[:3, :3] = rotation
        # The vendor Link7 terminal flange is the mesh plane y=-95 mm.
        tool[:3, 3] = np.array([0., -.095, 0.]) - rotation @ flange_in_hand
        config['nominal_link7_to_hand'][side] = tool.tolist()
        config['meshes'][side] = []
        transform = np.eye(4)
        transform[:3, :3] = mirror
        transform = transform @ hand_from_assembly
        for index in range(1, 7):
            mesh = trimesh.load(source / f'part_{index:02d}.stl')
            mesh.apply_transform(transform)
            path = output / f'{side}_{index:02d}.stl'
            mesh.export(path)
            name = path.relative_to(ROOT).as_posix()
            config['meshes'][side].append(name)
            config['mesh_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = ROOT / 'configs/preview/hand_connector_20260930.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing to overwrite interactive adjustments: ' + str(path))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    mount = {'shoulder_center_xyz_m': [.52, 0., .48], 'shoulder_spacing_m': .093,
             'base_rpy_rad': {'right': [-np.pi / 2, 0., 0.], 'left': [np.pi / 2, 0., 0.]},
             'seed_q_rad': {s: [0.] * 7 for s in ('right', 'left')},
             'link7_to_hand': {s: {'xyz_m': np.array(config['nominal_link7_to_hand'][s])[:3, 3].tolist(),
                                  'rpy_rad': [np.pi / 2, 0., 0.]} for s in ('right', 'left')},
             'orientation_weight_m': .15, 'damping': .008, 'iterations_per_frame': 8,
             'arm_rgba': [.9, .92, .95, 1.],
             'limitations': ['TJ-ARM-02/03: shoulder position, orientation and flange clocking are provisional']}
    arm_path = ROOT / 'preview/assets/tj_m6/preview_mount.json'
    if not arm_path.exists():
        arm_path.write_text(json.dumps(mount, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
