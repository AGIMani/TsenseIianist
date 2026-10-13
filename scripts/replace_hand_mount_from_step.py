#!/usr/bin/env python3
"""Replace only the wrist mounting block with the matching STEP hand part.

CAD dependencies are confined to the optional cad_tools environment. Generated
URDFs and meshes run independently of STEP, trimesh, and the CAD environment.
"""
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'tsensei_v3_2_pianist/assets'
REPORT = 'connector_mount_revision.json'


def mirror_mesh(source, target):
    """Use the same binary STL reflection as prepare_models (including winding)."""
    from prepare_models import read_triangles
    header, triangles = read_triangles(source)
    triangles['normal'][:, 1] *= -1
    triangles['vertices'][:, :, 1] *= -1
    triangles['vertices'] = triangles['vertices'][:, [0, 2, 1], :]
    target.write_bytes(header + struct.pack('<I', len(triangles)) + triangles.tobytes())


def apply_mount_revision(assets):
    """Reapply saved mesh references after regenerating the original models."""
    assets = Path(assets)
    report = json.loads((assets / REPORT).read_text())
    # Validate mesh inputs before replacing either URDF.
    for name, expected in report['mesh_sha256'].items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Mount mesh differs from approved generated geometry: ' + name)
    outputs = {}
    for side in ('left', 'right'):
        path = assets / f'urdf/TsenseI_{side}.urdf'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        if before == report['urdf_after_sha256'][side]:
            continue
        if before != report['urdf_before_sha256'][side]:
            raise ValueError('Mount revision requires the verified 952 g model: ' + side)
        root = ET.parse(path).getroot()
        for link, filename in report['mesh_replacements'][side].items():
            for node in root.find(f"link[@name='{link}']").findall('./*/geometry/mesh'):
                node.set('filename', filename)
        result = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        if hashlib.sha256(result).hexdigest() != report['urdf_after_sha256'][side]:
            raise ValueError('Unexpected mounted URDF output')
        outputs[path] = result
    for path, result in outputs.items():
        path.write_bytes(result)
    return report


def main():
    import numpy as np
    import trimesh
    if (ASSETS / REPORT).exists():
        raise FileExistsError('Mount revision already exists; use apply_mount_revision for regeneration')
    alignment = json.loads((ROOT / 'preview/outputs/connector_alignment/hole_aligned_mount.json').read_text())
    transform = np.linalg.inv(np.array(alignment['assembly_from_left_hand']))
    cad = trimesh.load(ROOT / 'preview/assets/arm_hand_connect/part_07.stl')
    candidates = [part for part in cad.split(only_watertight=False)
                  if np.allclose(part.bounds, [[-.0165, -.023, -.051], [.0085, .013, -.022]], atol=2e-7)]
    if len(candidates) != 1 or not candidates[0].is_watertight:
        raise ValueError('Cannot uniquely identify the closed STEP wrist mounting block')
    new_mount = candidates[0]
    mount_volume = float(new_mount.volume)
    new_mount.apply_transform(transform)
    palm_position = np.array([.0237, .0065, .0441380069844])
    new_mount.apply_translation(-palm_position)
    original_base = trimesh.load(ASSETS / 'meshes_left/base_link.STL')
    original_base.apply_scale(.001)
    original_base.apply_translation([.0147, -.0015, 0.])
    old_mount = trimesh.load(ASSETS / 'meshes_left/Left_hand_palm_link.STL')
    old_mount.apply_translation(palm_position)
    components = original_base.split(only_watertight=False)
    index = max(range(len(components)), key=lambda i: components[i].area)
    cut = trimesh.boolean.difference([components[index], old_mount], engine='manifold')
    # Coincident surfaces of the two vendor exports differ below mesh precision.
    # Keep the housing component and reject any unexpected remote fragment.
    pieces = cut.split(only_watertight=False)
    housing = max(pieces, key=lambda p: p.area)
    discarded = [p for p in pieces if p is not housing]
    for part in discarded:
        if not np.all(part.bounds[0] >= old_mount.bounds[0] - 2e-6) or not np.all(part.bounds[1] <= old_mount.bounds[1] + 2e-6):
            raise ValueError('Boolean subtraction produced a fragment outside the old mounting block')
    if not housing.is_watertight:
        raise ValueError('The remaining housing must be closed')
    components[index] = housing
    residual = trimesh.util.concatenate(components)
    residual.apply_translation([-.0147, .0015, 0.])
    residual.apply_scale(1000.)  # Keep the original base STL millimetre convention.
    report = {'schema': 1, 'source_step_sha256': json.loads((ROOT / 'preview/assets/arm_hand_connect/assembly.json').read_text())['source_sha256'],
              'source_component': 'A3 hand wrist mounting block; STEP hand shell 358 / tessellated connected component with recorded bounds',
              'source_bounds_assembly_m': [[-.0165, -.023, -.051], [.0085, .013, -.022]],
              'source_mount_volume_m3': mount_volume,
              'alignment': alignment, 'mesh_replacements': {}, 'mesh_sha256': {},
              'urdf_before_sha256': {}, 'urdf_after_sha256': {},
              'removed_duplicate_geometry': 'Old palm mounting block subtracted from base housing; other original base components retained',
              'inertia_status': 'Supplied 952 g per-link masses and inertia estimates retained. Geometry revision has not been CAD/dynamically recalibrated.',
              'limitations': ['V32-MOUNT-01: mounting transforms, masses and inertias after geometry replacement need hardware validation',
                              'Right hand and its replacement mounting block remain provisional mirrored geometry']}
    for side in ('left', 'right'):
        prefix = 'Left' if side == 'left' else 'Right'
        mount, base = new_mount.copy(), residual.copy()
        meshes = {f'meshes_{side}/wrist_mount_from_step.STL': mount,
                  f'meshes_{side}/base_without_old_mount.STL': base}
        for name, mesh in meshes.items():
            if side == 'left':
                mesh.export(ASSETS / name)
            else:
                mirror_mesh(ASSETS / name.replace('meshes_right/', 'meshes_left/'), ASSETS / name)
            report['mesh_sha256'][name] = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
        path = ASSETS / f'urdf/TsenseI_{side}.urdf'
        raw = path.read_bytes()
        backup = ASSETS / f'source/TsenseI_{side}_mass952_before_mount.urdf'
        if backup.exists():
            raise FileExistsError(backup)
        backup.write_bytes(raw)
        report['urdf_before_sha256'][side] = hashlib.sha256(raw).hexdigest()
        replacements = {'base_link': f'../meshes_{side}/base_without_old_mount.STL',
                        f'{prefix}_hand_palm_link': f'../meshes_{side}/wrist_mount_from_step.STL'}
        report['mesh_replacements'][side] = replacements
        root = ET.fromstring(raw)
        for link, filename in replacements.items():
            for node in root.find(f"link[@name='{link}']").findall('./*/geometry/mesh'):
                node.set('filename', filename)
        result = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        report['urdf_after_sha256'][side] = hashlib.sha256(result).hexdigest()
        path.write_bytes(result)
    (ASSETS / REPORT).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    manifest_path = ASSETS / 'model_derivation.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['mount_revision'] = {'report': REPORT, 'source_step_sha256': report['source_step_sha256'],
                                'inertia_status': report['inertia_status']}
    manifest['generated_sha256'].update(report['mesh_sha256'])
    for side, digest in report['urdf_after_sha256'].items():
        manifest['generated_sha256'][f'urdf/TsenseI_{side}.urdf'] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'mount_replaced': True, 'mount_plane_mm': [35.6, 24.6],
                      'source_mount_volume_mm3': mount_volume * 1e9,
                      'remaining_housing_closed': housing.is_watertight,
                      'report': str(ASSETS / REPORT)}))


if __name__ == '__main__':
    main()
