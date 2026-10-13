"""Check installed STEP geometry, mirror, and preserved hand joint/inertia data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_models', ROOT/'scripts/prepare_models.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
ASSETS = prepare.ASSETS


def test_step_mount_plane_and_four_hole_alignment():
    report = json.loads((ASSETS/'connector_mount_revision.json').read_text())
    filename = report['mesh_replacements']['left']['Left_hand_palm_link']
    _, triangles = prepare.read_triangles(ASSETS/'urdf'/filename)
    vertices = triangles['vertices'].astype(float) + prepare.CANONICAL_PALM_POSITION
    transform = np.asarray(report['alignment']['assembly_from_left_hand'])
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    plane = vertices[np.all(np.abs(vertices[:, :, 2]+.022) < 1e-7, axis=1)]
    assert len(plane) > 0
    np.testing.assert_allclose(np.ptp(plane.reshape(-1, 3), axis=0), [.0246, .0456, 0], atol=2e-7)
    np.testing.assert_allclose((plane.reshape(-1, 3).max(0)+plane.reshape(-1, 3).min(0))/2,
                               [0, 0, -.022], atol=2e-7)
    # New block is an exact rigidly transformed STEP part, not a global hand scale.
    assert abs(np.linalg.det(transform[:3, :3])-1) < 1e-12


def test_urdf_references_mirror_and_no_remaining_old_mount_surface():
    report = json.loads((ASSETS/'connector_mount_revision.json').read_text())
    for side in ('left', 'right'):
        root = ET.parse(ASSETS/f'urdf/TsenseI_{side}.urdf').getroot()
        before = ET.parse(ASSETS/f'source/TsenseI_{side}_mass952_before_mount.urdf').getroot()
        assert [ET.tostring(j) for j in root.findall('joint')] == [ET.tostring(j) for j in before.findall('joint')]
        for link in root.findall('link'):
            original = before.find(f"link[@name='{link.get('name')}']")
            assert ET.tostring(link.find('inertial')) == ET.tostring(original.find('inertial'))
        for link, filename in report['mesh_replacements'][side].items():
            for mesh in root.find(f"link[@name='{link}']").findall('./*/geometry/mesh'):
                assert mesh.get('filename') == filename
        _, mesh = prepare.read_triangles(ASSETS/f'meshes_{side}/base_without_old_mount.STL')
        assert not np.any(np.all(np.abs(mesh['vertices'][:, :, 2]*.001+.004) < 1e-7, axis=1))
    for name in ('wrist_mount_tianji_20260930.STL', 'base_without_old_mount.STL'):
        _, left = prepare.read_triangles(ASSETS/'meshes_left'/name)
        _, right = prepare.read_triangles(ASSETS/'meshes_right'/name)
        np.testing.assert_array_equal(right['vertices'], (left['vertices'] * [1, -1, 1])[:, [0, 2, 1]])
    for name, digest in report['mesh_sha256'].items():
        assert hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() == digest


def test_connector_flange_and_holder_agree_for_both_hands():
    config = json.loads((ROOT/'configs/preview/hand_connector_20260930.json').read_text())
    report = json.loads((ASSETS/'connector_mount_revision.json').read_text())
    assert config['source_step_sha256'] == report['source_step_sha256']
    transform = np.array(report['alignment']['assembly_from_left_hand'])
    holes = np.array([[-.0095, -.020, -.022], [-.0095, .020, -.022],
                      [.0095, -.020, -.022], [.0095, .020, -.022]])
    for side in ('left', 'right'):
        mirror = np.diag([1., -1., 1.]) if side == 'right' else np.eye(3)
        tool = np.array(config['nominal_link7_to_hand'][side])
        hand_flange = mirror @ np.linalg.inv(transform)[:3, 3]
        np.testing.assert_allclose(tool[:3, :3] @ hand_flange + tool[:3, 3], [0, -.095, 0], atol=1e-12)
        name = config['meshes'][side][5]
        _, triangles = prepare.read_triangles(ROOT/name)
        points = triangles['vertices'].reshape(-1, 3).astype(float) @ mirror
        points = points @ transform[:3, :3].T + transform[:3, 3]
        face = points[abs(points[:, 2]+.022) < 2e-7]
        # Every CAD plate hole retains its 4.3 mm circular opening about the
        # expected 19 x 40 mm pattern after baking and right-hand reflection.
        for centre in holes:
            radius = np.linalg.norm(face[:, :2]-centre[:2], axis=1)
            assert abs(radius.min()-.00215) < 2e-7
            assert np.count_nonzero(abs(radius-.00215)<2e-7) >= 40
