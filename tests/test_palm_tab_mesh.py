"""Active URDF visuals, collision hulls and proxy manifests share the trim."""
import json
import xml.etree.ElementTree as ET

import numpy as np

from scripts.sync_project_urdf import ASSETS, read_triangles, right_name


def test_tab_removed_from_visual_and_collision_both_hands():
    vertices = {}
    for side in ('left', 'right'):
        root = ET.parse(ASSETS / 'urdf' / f'TsenseI_{side}.urdf')
        link = root.find("link[@name='base_link']")
        for kind in ('visual', 'collision'):
            path = ASSETS / 'urdf' / link.find(f'{kind}/geometry/mesh').get('filename')
            points = read_triangles(path)['vertices'].reshape(-1, 3)
            vertices[side, kind] = points
            extent = points[:, 1] if side == 'left' else -points[:, 1]
            assert extent.max() < .05804  # Former rectangular tab reached 86.04 mm.
            assert extent.max() > .05802  # Do not cut the surrounding shell.
        # Large-mesh visual fallback must also be rebuilt.
        obj = ASSETS / 'source/latest_project/meshes' / side / 'base_link.obj'
        points = np.array([[float(v) for v in line.split()[1:]]
                           for line in obj.read_text().splitlines() if line.startswith('v ')])
        assert (points[:, 1] if side == 'left' else -points[:, 1]).max() < .05804
    for kind in ('visual', 'collision'):
        a = np.unique(vertices['left', kind] * [1, -1, 1], axis=0)
        b = np.unique(vertices['right', kind], axis=0)
        np.testing.assert_array_equal(a, b)


def test_both_proxy_manifests_use_same_new_hulls_and_joint_frames_unchanged():
    a, b = [json.loads((ASSETS / name).read_text()) for name in (
        'collision_proxies.json', 'collision_proxies_convex_hull.json')]
    assert a['hands'] == b['hands']
    source = ET.parse(ASSETS / 'source/latest_project/urdf/TsenseI_hand_left_v3_2_withid.urdf')
    active = ET.parse(ASSETS / 'urdf/TsenseI_left.urdf')
    for joint in source.findall('joint'):
        other = active.find(f"joint[@name='{joint.get('name')}']")
        assert joint.attrib == other.attrib
        for child in joint:
            assert child.attrib == other.find(child.tag).attrib
