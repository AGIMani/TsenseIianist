#!/usr/bin/env python3
"""Install the verified 2026-09-30 Tianji adapter and its wide hand holder.

Uses already tessellated, named CAD components from import_hand_connector.py.
Only offline generation requires trimesh; normal preview uses packaged STLs.
"""
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from replace_hand_mount_from_step import mirror_mesh

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'tsensei_v3_2_pianist/assets'
SOURCE = ROOT / 'preview/assets/a32_tianji_verified_20260930'
CONFIG = ROOT / 'configs/preview/hand_connector_20260930.json'
REPORT = ASSETS / 'connector_mount_revision.json'
EXPECTED_SOURCE = 'A3.2_tianji.stp'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def plane_loops(mesh, z=-.022):
    mask = np.all(np.abs(mesh.triangles[:, :, 2] - z) < 1e-7, axis=1)
    if not mask.any():
        raise ValueError('Missing mating plane')
    plane = mesh.submesh([mask], append=True)
    return plane, plane.outline().discrete


def bolt_centres(mesh, diameter):
    _, loops = plane_loops(mesh)
    centres = []
    for loop in loops:
        bounds = np.array([loop.min(0), loop.max(0)])
        if np.allclose(np.diff(bounds, axis=0)[0, :2], diameter, atol=2e-7):
            centres.append(bounds.mean(0))
    if len(centres) != 4:
        raise ValueError('Expected exactly four mounting holes')
    centres = np.array(centres)
    return centres[np.lexsort((np.round(centres[:, 1], 7), np.round(centres[:, 0], 7)))]


def main():
    source = json.loads((SOURCE / 'assembly.json').read_text())
    if source['source_filename'] != EXPECTED_SOURCE:
        raise ValueError('Unexpected source CAD')
    parts = [p for p in source['instances'] if not p['is_reference_hand']]
    holders = source['extracted_components']
    if len(parts) != 6 or len(holders) != 1 or holders[0]['product'] != 'A3_2_HOLDER_WITH_WIDE_CONNECTOR':
        raise ValueError('Unexpected assembly contents')
    for entry in parts + holders:
        if digest(SOURCE / entry['mesh']) != entry['mesh_sha256']:
            raise ValueError('CAD component hash mismatch')
    holder = trimesh.load(SOURCE / holders[0]['mesh'])
    plate = trimesh.load(SOURCE / parts[5]['mesh'])
    if not holder.is_watertight or holder.volume <= 0:
        raise ValueError('Holder is not a closed positive-volume solid')
    plate_holes = bolt_centres(plate, .0043)
    holder_holes = bolt_centres(holder, .0037)
    np.testing.assert_allclose(plate_holes, holder_holes, atol=2e-7)
    # Preserve the URDF axes, source joint tree and fingers. The CAD holder's
    # four hole centres determine translation; finger direction fixes clocking.
    rotation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    urdf_holes = np.array([[-.0065, -.0055, -.004], [-.0065, .0135, -.004],
                           [.0335, -.0055, -.004], [.0335, .0135, -.004]])
    assembly_from_hand = np.eye(4)
    assembly_from_hand[:3, :3] = rotation
    assembly_from_hand[:3, 3] = holder_holes.mean(0) - rotation @ urdf_holes.mean(0)
    transformed = urdf_holes @ rotation.T + assembly_from_hand[:3, 3]
    errors = np.linalg.norm(transformed[:, None] - holder_holes[None], axis=2).min(1)
    if errors.max() > 2e-7:
        raise ValueError('URDF mounting holes do not fit the CAD holder')
    plane, _ = plane_loops(holder)
    alignment = {
        'status': 'A3.2_tianji CAD four-hole and mating-plane alignment',
        'assembly_from_left_hand': assembly_from_hand.tolist(),
        'method': 'match four mounting-hole centres and opposing normals; preserve existing hand axes',
        'source_alignment_point_m': urdf_holes.mean(0).tolist(),
        'target_alignment_point_m': holder_holes.mean(0).tolist(),
        'step_mount_face_size_mm': (np.ptp(plane.vertices, axis=0) * 1000).tolist(),
        'urdf_mount_face_size_mm': [24.6, 45.6, 0.],
        'mounting_hole_centres_assembly_m': holder_holes.tolist(),
        'plate_hole_diameter_mm': 4.3, 'holder_hole_diameter_mm': 3.7,
        'max_plate_holder_axis_error_mm': float(np.linalg.norm(plate_holes-holder_holes, axis=1).max()*1000),
        'max_original_urdf_hole_axis_error_mm': float(errors.max()*1000),
        'limitations': ['TODO(V32-MOUNT-01): hardware screw/thread fit and flange clocking remain uncalibrated'],
    }
    old_report = json.loads(REPORT.read_text())
    if old_report['source_step_sha256'] == source['source_sha256']:
        raise FileExistsError('This CAD revision is already installed')
    old_config = json.loads(CONFIG.read_text())
    # Refuse to discard interactive work or an unexpected URDF revision.
    for side in ('left', 'right'):
        if any(old_config['offsets'][side]['xyz_mm'] + old_config['offsets'][side]['rpy_deg']):
            raise ValueError('Existing interactive offsets must be reviewed before migration')
        if digest(ASSETS / f'urdf/TsenseI_{side}.urdf') != old_report['urdf_after_sha256'][side]:
            raise ValueError('Current hand differs from the saved mounting revision')
    backup = ROOT / 'model_backups/before_tianji_mount_20260930.tar.gz'
    backup.parent.mkdir(exist_ok=True)
    saved = [REPORT, CONFIG, ASSETS/'model_derivation.json', ROOT/'preview/assets/tj_m6/preview_mount.json']
    saved += [ASSETS/f'urdf/TsenseI_{s}.urdf' for s in ('left', 'right')]
    saved += [ASSETS/name for name in old_report['mesh_sha256']]
    saved += [ROOT/name for name in old_config['mesh_sha256']]
    with tarfile.open(backup, 'x:gz') as archive:
        for path in saved:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    report = copy.deepcopy(old_report)
    report.update(source_filename=EXPECTED_SOURCE, source_step_sha256=source['source_sha256'],
                  source_component=holders[0]['product'], source_bounds_assembly_m=holder.bounds.tolist(),
                  source_mount_volume_m3=float(holder.volume), alignment=alignment)
    report['previous_revision_backup'] = str(backup.relative_to(ROOT))
    report['mesh_sha256'] = {k: v for k, v in old_report['mesh_sha256'].items() if 'base_without_old_mount' in k}
    hand_from_assembly = np.linalg.inv(assembly_from_hand)
    mounted = holder.copy()
    mounted.apply_transform(hand_from_assembly)
    mounted.apply_translation(-np.array([.0237, .0065, .0441380069844]))
    for side in ('left', 'right'):
        mesh_name = f'meshes_{side}/wrist_mount_tianji_20260930.STL'
        path = ASSETS / mesh_name
        if side == 'left':
            mounted.export(path)
        else:
            mirror_mesh(ASSETS/'meshes_left/wrist_mount_tianji_20260930.STL', path)
        report['mesh_sha256'][mesh_name] = digest(path)
        prefix = 'Left' if side == 'left' else 'Right'
        report['mesh_replacements'][side][f'{prefix}_hand_palm_link'] = '../' + mesh_name
        root = ET.parse(ASSETS / f'source/TsenseI_{side}_mass952_before_mount.urdf').getroot()
        for link, name in report['mesh_replacements'][side].items():
            for node in root.find(f"link[@name='{link}']").findall('./*/geometry/mesh'):
                node.set('filename', name)
        urdf = ASSETS / f'urdf/TsenseI_{side}.urdf'
        urdf.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))
        report['urdf_after_sha256'][side] = digest(urdf)
    write_json(REPORT, report)
    manifest_path = ASSETS / 'model_derivation.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['mount_revision'] = {'report': REPORT.name, 'source_step_sha256': source['source_sha256'],
                                'inertia_status': report['inertia_status']}
    manifest['generated_sha256'].update(report['mesh_sha256'])
    for side in ('left', 'right'):
        manifest['generated_sha256'][f'urdf/TsenseI_{side}.urdf'] = report['urdf_after_sha256'][side]
    write_json(manifest_path, manifest)
    config = copy.deepcopy(old_config)
    config.update(alignment=alignment, source_filename=EXPECTED_SOURCE, source_step_sha256=source['source_sha256'])
    config['mesh_sha256'] = {}
    config['alignment_label'] = '新版安装件四孔已对齐'
    output = ROOT / 'preview/assets/hand_connector/tianji_20260930'
    output.mkdir(parents=True, exist_ok=False)
    tool_rotation = Rotation.from_euler('x', np.pi / 2).as_matrix()
    for side in ('left', 'right'):
        reflection = np.diag([1., -1., 1., 1.]) if side == 'right' else np.eye(4)
        transform = reflection @ hand_from_assembly
        config['meshes'][side] = []
        for i, part in enumerate(parts, 1):
            mesh = trimesh.load(SOURCE / part['mesh'])
            mesh.apply_transform(transform)
            path = output / f'{side}_{i:02d}.stl'
            mesh.export(path)
            name = str(path.relative_to(ROOT))
            config['meshes'][side].append(name)
            config['mesh_sha256'][name] = digest(path)
        tool = np.eye(4)
        tool[:3, :3] = tool_rotation
        tool[:3, 3] = np.array([0., -.095, 0.]) - tool_rotation @ transform[:3, 3]
        config['nominal_link7_to_hand'][side] = tool.tolist()
    write_json(CONFIG, config)
    mount_path = ROOT / 'preview/assets/tj_m6/preview_mount.json'
    mount = json.loads(mount_path.read_text())
    for side in ('left', 'right'):
        tool = np.array(config['nominal_link7_to_hand'][side])
        mount['link7_to_hand'][side] = {'xyz_m': tool[:3, 3].tolist(), 'rpy_rad': [np.pi/2, 0., 0.]}
    write_json(mount_path, mount)
    evidence = ROOT / 'preview/outputs/tianji_mount_20260930/alignment.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    write_json(evidence, alignment)
    print(json.dumps(alignment, indent=2))


if __name__ == '__main__':
    main()
