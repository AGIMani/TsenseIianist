#!/usr/bin/env python3
"""Rebuild the normalized left and provisional mirrored right from bundled V3.2.

Only reads this project's original left URDF and left STL files. No ROS/V2/V3
installation is required. The generated right is a hypothesis, not vendor data.
"""
from pathlib import Path
import copy
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[1] / 'tsensei_v3_2_pianist/assets'
REFLECTION = np.diag([1., -1., 1.])
# V3 palm anchor reused solely as a convenient canonical mounting frame.
# V32-01: this simulation mounting frame was accepted by the user on 2026-09-17.
# Real-hardware frame measurement remains a later deployment check.
CANONICAL_PALM_POSITION = np.array([.0237, .0065, .0441380069844])
# Apply simultaneously: sequential replacements would rename the same link twice.
PINKY_PART_NAMES = {'rotaback': 'MCP', 'MCP': 'swing', 'swing': 'PIP', 'PIP': 'DIP'}
MODEL_NAME = 'TsenseI_A3.2'
STL_DTYPE = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])


def fmt(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def origin_transform(origin):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return transform


def set_origin(origin, transform):
    origin.set('xyz', fmt(transform[:3, 3]))
    origin.set('rpy', fmt(Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')))


def read_triangles(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if not count or len(data) != 84 + count * 50:
        raise ValueError(f'Expected nonempty binary STL: {path}')
    return data[:80], np.frombuffer(data, dtype=STL_DTYPE, offset=84).copy()


def right_name(name):
    return name.replace('left_', 'right_').replace('Left_', 'Right_')


def write_xml(root, target):
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)


def convert_large_mesh(source):
    _, triangles = read_triangles(source)
    if len(triangles) <= 200000:
        return None
    # Exact float32 vertex values preserved in decimal; topology is unchanged.
    vertices, inverse = np.unique(triangles['vertices'].reshape(-1, 3), axis=0, return_inverse=True)
    target = source.with_suffix('.obj')
    with target.open('w') as out:
        out.write('# Same geometry as bundled STL; converted for MuJoCo STL face limit.\n')
        np.savetxt(out, vertices, fmt='v %.17g %.17g %.17g')
        np.savetxt(out, inverse.reshape(-1, 3) + 1, fmt='f %d %d %d')
    return {'file': str(target.relative_to(ASSETS)), 'triangles': len(triangles)}


def pinky_name_map(root):
    """Recognize vendor schemas before renaming; never shift the new chain."""
    names = {joint.get('name') for joint in root.findall('joint')}
    prefix = 'left_hand_pinky_'
    parts = {name.removeprefix(prefix).removesuffix('_joint')
             for name in names if name.startswith(prefix)}
    if parts == set(PINKY_PART_NAMES):
        schema, mapping = 'legacy_rotaback_chain', PINKY_PART_NAMES
    elif parts == {'MCP', 'SWING', 'PIP', 'DIP'}:
        schema, mapping = 'a32_with_id_semantic_chain', {'SWING': 'swing'}
    elif parts == {'MCP', 'swing', 'PIP', 'DIP'}:
        schema, mapping = 'semantic_chain', {}
    else:
        raise ValueError(f'Unrecognized pinky joint schema: {sorted(parts)}')
    return schema, {f'{prefix}{old}_{kind}': f'{prefix}{new}_{kind}'
                    for old, new in mapping.items() for kind in ('joint', 'link')}


def main():
    source = ASSETS / 'source/left_original.urdf'
    left = ET.parse(source).getroot()
    vendor_robot_name = left.get('name')
    left.set('name', MODEL_NAME + '_left_normalized')
    source_schema, source_name_map = pinky_name_map(left)
    for element in left.iter():
        for attribute in ('name', 'link'):
            value = element.get(attribute)
            if value in source_name_map:
                element.set(attribute, source_name_map[value])
    palm = left.find("joint[@name='Left_hand_palm_joint']/origin")
    target_palm = np.eye(4)
    target_palm[:3, 3] = CANONICAL_PALM_POSITION
    correction = target_palm @ np.linalg.inv(origin_transform(palm))
    # Change only the root coordinate frame. Other link frames remain unchanged.
    base = left.find("link[@name='base_link']")
    for origin in base.findall('./*/origin'):
        set_origin(origin, correction @ origin_transform(origin))
    for joint in left.findall('joint'):
        if joint.find('parent').get('link') == 'base_link':
            set_origin(joint.find('origin'), correction @ origin_transform(joint.find('origin')))
    for mesh in left.iter('mesh'):
        filename = Path(mesh.get('filename')).name
        if not (ASSETS / 'meshes_left' / filename).is_file():
            raise FileNotFoundError(filename)
        mesh.set('filename', '../meshes_left/' + filename)
    # This vendor export mixes one millimetre STL with 21 metre STLs. Keep
    # raw evidence intact; correct only the identified base visual/collider.
    mesh_unit_corrections = []
    if source_schema == 'a32_with_id_semantic_chain':
        base_mesh = ASSETS / 'meshes_left/base_link.STL'
        _, triangles = read_triangles(base_mesh)
        span = np.ptp(triangles['vertices'].reshape(-1, 3), axis=0)
        if not np.allclose(span, [50.91058349609375, 99.00718307495117, 105.30592346191406], atol=1e-5, rtol=0):
            raise ValueError('Unrecognized A3.2 base dimensions: re-audit vendor mesh units')
        for mesh in base.findall('./*/geometry/mesh'):
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            mesh.set('scale', fmt(scale * .001))
        mesh_unit_corrections.append({
            'link': 'base_link', 'mesh': 'base_link.STL',
            'scale_multiplier': [.001, .001, .001],
            'source_span': span.tolist(),
            'reason': 'This STL uses millimetres; the remaining vendor meshes and URDF use metres',
            'source_mesh_sha256': hashlib.sha256(base_mesh.read_bytes()).hexdigest(),
        })
    urdfs = ASSETS / 'urdf'
    urdfs.mkdir(exist_ok=True)
    write_xml(left, urdfs / 'TsenseI_left.urdf')

    # TODO(V32-10): Replace with the manufacturer's actual right-hand model.
    right = copy.deepcopy(left)
    right.set('name', MODEL_NAME + '_right_MIRRORED_PROVISIONAL')
    for element in right.iter():
        for key in ('name', 'link'):
            if key in element.attrib:
                element.set(key, right_name(element.get(key)))
        if element.tag == 'origin':
            transform = origin_transform(element)
            transform[:3, :3] = REFLECTION @ transform[:3, :3] @ REFLECTION
            transform[:3, 3] = REFLECTION @ transform[:3, 3]
            set_origin(element, transform)
        elif element.tag == 'axis':
            # Revolute axes are axial vectors: det(S) * S * a.
            element.set('xyz', fmt(-REFLECTION @ np.fromstring(element.get('xyz'), sep=' ')))
        elif element.tag == 'inertia':
            for cross_term in ('ixy', 'iyz'):
                element.set(cross_term, format(-float(element.get(cross_term)), '.17g'))
        elif element.tag == 'mesh':
            element.set('filename', '../meshes_right/' + right_name(Path(element.get('filename')).name))
    write_xml(right, urdfs / 'TsenseI_right.urdf')
    (ASSETS / 'meshes_right').mkdir(exist_ok=True)
    for path in sorted((ASSETS / 'meshes_left').glob('*.STL')):
        header, triangles = read_triangles(path)
        triangles['normal'][:, 1] *= -1
        triangles['vertices'][:, :, 1] *= -1
        # Reverse triangle winding to keep outward normals after reflection.
        triangles['vertices'] = triangles['vertices'][:, [0, 2, 1], :]
        target = ASSETS / 'meshes_right' / right_name(path.name)
        target.write_bytes(header + struct.pack('<I', len(triangles)) + triangles.tobytes())
    conversions = []
    for path in sorted(ASSETS.glob('meshes_*/*.STL')):
        entry = convert_large_mesh(path)
        if entry:
            conversions.append(entry)
    revision = None
    if (ASSETS / 'mass_update_report.json').exists():
        from apply_hand_mass_revision import apply_revision
        revision = apply_revision(ASSETS)
    mount_revision = None
    if (ASSETS / 'connector_mount_revision.json').exists():
        from replace_hand_mount_from_step import apply_mount_revision
        mount_revision = apply_mount_revision(ASSETS)
    generated = list(urdfs.glob('*.urdf')) + list((ASSETS / 'meshes_right').glob('*.STL')) + list(ASSETS.glob('meshes_*/*.obj'))
    if mount_revision:
        generated.extend(ASSETS / name for name in mount_revision['mesh_sha256'])
    report = {
        'model_name': MODEL_NAME,
        'vendor_robot_name': vendor_robot_name,
        'source_joint_schema': source_schema,
        'left_source': 'source/left_original.urdf',
        'left_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'left_source_name_map': source_name_map,
        'mesh_unit_corrections': mesh_unit_corrections,
        'canonical_from_source_root': correction.tolist(),
        'canonical_palm_position': CANONICAL_PALM_POSITION.tolist(),
        'right_hand_is_provisional_mirror': True,
        'reflection': REFLECTION.tolist(),
        'oversized_stl_conversions': conversions,
        'generated_sha256': {str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generated)},
    }
    if revision is not None:
        report['mass_revision'] = revision
    if mount_revision:
        report['mount_revision'] = {'report': 'connector_mount_revision.json',
                                   'source_step_sha256': mount_revision['source_step_sha256'],
                                   'inertia_status': mount_revision['inertia_status']}
    (ASSETS / 'model_derivation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'generated_sha256'}, indent=2))


if __name__ == '__main__':
    main()
