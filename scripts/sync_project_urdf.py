#!/usr/bin/env python3
"""Install the current supplied bilateral A3.2 multipart URDF on the remote host.

Preserve both joint trees and supplied inertias; restore the user-requested
Thumb_Rot2 coordinate sign; merge rigid visual parts and
build one convex hull per link from the source collision geometry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tsensei_v3_2_pianist" / "assets"
URDFS = ASSETS / "urdf"
REFLECTION = np.diag([1.0, -1.0, 1.0])
STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def fmt(values) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def right_name(name: str) -> str:
    return name.replace("left_", "right_").replace("Left_", "Right_")


def read_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError(f"Expected binary STL: {path}")
    return np.frombuffer(data, dtype=STL_DTYPE, offset=84).copy()


def write_binary_stl(path: Path, vertices: np.ndarray, header: bytes = b"TsenseI convex-hull collision mesh") -> None:
    triangles = np.zeros(len(vertices), dtype=STL_DTYPE)
    triangles["vertices"] = vertices.astype(np.float32)
    a = triangles["vertices"][:, 1] - triangles["vertices"][:, 0]
    b = triangles["vertices"][:, 2] - triangles["vertices"][:, 0]
    normal = np.cross(a, b)
    length = np.linalg.norm(normal, axis=1)
    valid = length > 0
    normal[valid] /= length[valid, None]
    triangles["normal"] = normal.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.ljust(80, b" ")[:80] + struct.pack("<I", len(triangles)) + triangles.tobytes())


def convex_hull(source: Path, target: Path) -> dict:
    triangles = read_triangles(source)
    unique = np.unique(np.round(triangles["vertices"].reshape(-1, 3), decimals=12), axis=0)
    if len(unique) < 4:
        raise ValueError(f"Not enough vertices for convex hull: {source}")
    hull = ConvexHull(unique)
    faces = unique[hull.simplices].copy()
    for index, equation in enumerate(hull.equations):
        normal = np.cross(faces[index, 1] - faces[index, 0], faces[index, 2] - faces[index, 0])
        if np.dot(normal, equation[:3]) < 0:
            faces[index, [1, 2]] = faces[index, [2, 1]]
    write_binary_stl(target, faces)
    return {
        "source_vertex_count": int(len(unique)),
        "hull_vertex_count": int(len(hull.vertices)),
        "hull_triangle_count": int(len(hull.simplices)),
        "hull_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def write_obj(path: Path, triangles: np.ndarray) -> None:
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# OBJ conversion for MuJoCo meshes over the STL face limit.\n")
        for vertex in vertices:
            stream.write("v " + fmt(vertex) + "\n")
        for face in inverse.reshape(-1, 3) + 1:
            stream.write(f"f {face[0]} {face[1]} {face[2]}\n")


def maybe_write_obj(stl: Path) -> None:
    triangles = read_triangles(stl)
    if len(triangles) > 200_000:
        write_obj(stl.with_suffix(".obj"), triangles["vertices"])


def write_xml(root: ET.Element, target: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)


def resolved_mesh(source_dir: Path, urdf: Path, element: ET.Element) -> Path:
    path = (urdf.parent / element.get('filename')).resolve()
    if not path.is_relative_to(source_dir.resolve()) or not path.is_file():
        raise ValueError(f'Mesh missing or outside supplied package: {path}')
    return path


def combined_geometry(source_dir: Path, urdf: Path, link: ET.Element, kind: str) -> np.ndarray:
    """Merge each rigid link in its own frame, honoring every mesh origin/scale."""
    pieces = []
    for geom in link.findall(kind):
        mesh = geom.find('geometry/mesh')
        if mesh is None:
            raise ValueError(f'Expected mesh: {link.get("name")}/{kind}')
        path = resolved_mesh(source_dir, urdf, mesh)
        points = read_triangles(path)['vertices'].astype(np.float64)
        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f'Invalid positive mesh scale: {path}')
        origin = geom.find('origin')
        xyz = np.zeros(3) if origin is None else np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.zeros(3) if origin is None else np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        rotation = Rotation.from_euler('xyz', rpy).as_matrix()
        points = (points * scale) @ rotation.T + xyz
        if not np.isfinite(points).all():
            raise ValueError(f'Nonfinite geometry: {path}')
        pieces.append(points)
    if not pieces:
        raise ValueError(f'Missing {kind}: {link.get("name")}')
    return np.concatenate(pieces)


def supplied_inertia(link: ET.Element) -> tuple[float, np.ndarray, np.ndarray]:
    inertial = link.find('inertial')
    if inertial is None:
        raise ValueError(f'Missing supplied inertia: {link.get("name")}')
    mass = float(inertial.find('mass').get('value'))
    origin = inertial.find('origin')
    com = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    values = inertial.find('inertia').attrib
    tensor = np.array([[float(values['i'+min(a,b)+max(a,b)]) for b in 'xyz'] for a in 'xyz'])
    eigenvalues = np.linalg.eigvalsh(tensor)
    if (not np.isfinite(mass) or mass <= 0 or not np.isfinite(com).all()
            or not np.isfinite(tensor).all() or eigenvalues[0] <= 0
            or eigenvalues[2] > eigenvalues[0]+eigenvalues[1]+1e-14):
        raise ValueError(f'Invalid supplied mass/inertia: {link.get("name")}')
    return mass, com, rotation @ tensor @ rotation.T


def install(source_dir: Path, source_zip_sha256: str) -> None:
    """Install both supplied hands; never infer the right hand or replace inertias."""
    import tempfile
    source_dir = source_dir.resolve()
    if len(source_zip_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_zip_sha256):
        raise ValueError('Expected source ZIP SHA256')
    sources = {}
    for side in ('left', 'right'):
        path = source_dir / 'urdf' / f'TsenseI_hand_{side}_v3_2_withid.urdf'
        root = ET.parse(path).getroot()
        links, joints = root.findall('link'), root.findall('joint')
        if len(links) != 22 or len(joints) != 21 or len(root.findall("joint[@type='revolute']")) != 20:
            raise ValueError(f'Expected A3.2 22 links / 20 moving joints: {side}')
        for link in links:
            supplied_inertia(link)
            for mesh in link.findall('.//mesh'):
                resolved_mesh(source_dir, path, mesh)
        sources[side] = (path, root)
    # Build in a staging directory. Caller preserves the prior revision before installation.
    with tempfile.TemporaryDirectory(prefix='hand-install-', dir=ASSETS.parent) as temporary:
        stage = Path(temporary)
        latest = stage / 'source/latest_project'
        (latest / 'urdf').mkdir(parents=True)
        (stage / 'urdf').mkdir()
        records = {'left': {}, 'right': {}}
        source_hashes = {}; generated = {}; totals = {}
        for side, (source_urdf, source_root) in sources.items():
            shutil.copy2(source_urdf, latest/'urdf'/source_urdf.name)
            for element in source_root.findall('.//mesh'):
                mesh_path = resolved_mesh(source_dir, source_urdf, element)
                target = latest / mesh_path.relative_to(source_dir)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mesh_path, target)
            source_hashes[side] = hashlib.sha256(source_urdf.read_bytes()).hexdigest()
            active = copy.deepcopy(source_root)
            active.set('name', f'TsenseI_A3.2_{side}_supplied_20261020')
            # User requested the previous Thumb_Rot2 coordinate direction.
            # Reverse axis AND angle limits: R(-a, -q) == R(a, q).
            thumb = active.find(f"joint[@name='{side}_hand_thumb_rotaback_joint2']")
            axis = thumb.find('axis')
            axis.set('xyz', fmt(-np.fromstring(axis.get('xyz'), sep=' ')))
            limit = thumb.find('limit')
            lower, upper = float(limit.get('lower')), float(limit.get('upper'))
            limit.set('lower', format(-upper, '.17g'))
            limit.set('upper', format(-lower, '.17g'))
            total = 0.
            for link, original in zip(active.findall('link'), source_root.findall('link')):
                name = link.get('name')
                visual_path = latest / 'merged' / side / f'{name}.STL'
                write_binary_stl(visual_path, combined_geometry(source_dir, source_urdf, original, 'visual'))
                maybe_write_obj(visual_path)
                collision_source = stage/'collision_sources'/side/f'{name}.stl'
                write_binary_stl(collision_source, combined_geometry(source_dir, source_urdf, original, 'collision'))
                hull = stage/'collision_hulls'/side/f'{name}_convex_hull.stl'
                stats = convex_hull(collision_source, hull)
                mass, com, inertia = supplied_inertia(original)
                total += mass
                for old in list(link.findall('visual')) + list(link.findall('collision')):
                    link.remove(old)
                for kind, mesh_path in [('visual', visual_path), ('collision', hull)]:
                    element = ET.SubElement(link, kind, name=f'{name}_{kind}')
                    ET.SubElement(element, 'origin', xyz='0 0 0', rpy='0 0 0')
                    geom = ET.SubElement(element, 'geometry')
                    ET.SubElement(geom, 'mesh', filename='../'+str(mesh_path.relative_to(stage)), scale='1 1 1')
                records[side][name] = dict(
                    type='mesh', mesh_file=str(hull.relative_to(stage)),
                    source_mesh=str(collision_source.relative_to(stage)),
                    source_sha256=hashlib.sha256(collision_source.read_bytes()).hexdigest(),
                    physical_role='rigid_mount' if name in {'base_link', f'{side.capitalize()}_hand_palm_link'} else 'finger_link',
                    hull_method='scipy_convex_hull_v1', mass_kg=mass,
                    inertial_origin_xyz_m=com.tolist(), inertia_kg_m2=inertia.tolist(),
                    inertia_method='supplied_urdf_inertial_preserved',
                    source_visual_count=len(original.findall('visual')),
                    source_collision_count=len(original.findall('collision')), **stats)
            totals[side] = total
            write_xml(active, stage/'urdf'/f'TsenseI_{side}.urdf')
        shutil.copy2(sources['left'][0], stage/'source/left_original.urdf')
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                generated[str(path.relative_to(stage))] = hashlib.sha256(path.read_bytes()).hexdigest()
        limitations = [
            'Convex hulls fill concave cavities; they are not exact shells.',
            'Both hands come from the user-supplied ZIP, not a newly generated mirror; manufacturer provenance is unverified.',
            'Supplied masses/inertias are preserved; hardware calibration is unverified (TODO(V32-MASS-SOURCE-CALIBRATION)).',
            'Visuals merge all rigid per-link parts after applying their original origins/scales; no original shape is trimmed.']
        manifest = dict(schema=1, profile='convex_hull_v1', units='metres',
                        reference='TsenseI_hand_A3.2_withid_upper_thumb5mm_noupper_mid',
                        source_urdf='source/latest_project/urdf/TsenseI_hand_left_v3_2_withid.urdf',
                        source_urdf_sha256=source_hashes['left'], source_urdf_sha256_by_side=source_hashes,
                        preserve_original_mesh_links=[], limitations=limitations, hands=records)
        report = dict(model_name='TsenseI_A3.2', source_joint_schema='a32_supplied_bilateral_parts_20261020',
                      left_source='source/left_original.urdf', left_source_sha256=source_hashes['left'],
                      source_package=source_dir.name, source_zip_sha256=source_zip_sha256,
                      source_package_urdf=manifest['source_urdf'], source_urdf_sha256=source_hashes['left'],
                      source_urdf_sha256_by_side=source_hashes,
                      right_hand_is_provisional_mirror=False, right_hand_source='supplied_package',
                      joint_coordinate_overrides={f'{side}_hand_thumb_rotaback_joint2': dict(axis_multiplier=-1, angle_multiplier=-1, reason='2026-10-20 user requested previous axis direction; equivalent physical rotation') for side in ('left','right')},
                      link_count=22, joint_count=21, revolute_joint_count=20, left_source_name_map={},
                      canonical_from_source_root=dict(xyz=[0.,0.,0.], rpy=[0.,0.,0.], meaning='Both supplied root frames preserved independently.'),
                      mass_source='supplied bilateral URDF inertial tags', mass_total_kg=totals['left'], mass_total_kg_by_side=totals,
                      inertia_estimation=dict(method='supplied_urdf_inertial_preserved', hardware_calibration_status='TODO(V32-MASS-SOURCE-CALIBRATION)', source_export_contains_inertial_tags=True),
                      collision_contract='convex_hull_v1 for every link; no raw mesh collision fallback',
                      generated_sha256=generated, limitations=limitations)
        (stage/'collision_proxies_convex_hull.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (stage/'collision_proxies.json').write_text(json.dumps(dict(manifest, profile='shadow_primitive_v1'), indent=2)+'\n')
        (stage/'model_derivation.json').write_text(json.dumps(report, indent=2)+'\n')
        for relative in ['source/latest_project', 'source/left_original.urdf', 'collision_hulls', 'collision_sources',
                         'urdf/TsenseI_left.urdf', 'urdf/TsenseI_right.urdf',
                         'collision_proxies_convex_hull.json', 'collision_proxies.json', 'model_derivation.json']:
            target = ASSETS/relative
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage/relative), str(target))
    print(json.dumps(dict(source_sha256=source_hashes, source_zip_sha256=source_zip_sha256,
                          links_per_hand=22, hulls=44, mass_kg=totals), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--source-zip-sha256', required=True)
    args = parser.parse_args()
    install(args.source_dir, args.source_zip_sha256)


if __name__ == '__main__':
    main()
