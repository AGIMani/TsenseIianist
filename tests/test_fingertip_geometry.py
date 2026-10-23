"""Landmarks follow physical distal geometry across link-frame changes."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'tip_geometry', ROOT / 'tsensei_v3_2_pianist/models/hands/fingertip_geometry.py')
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def chain(rotation=np.eye(3), reflection=False):
    root = ET.fromstring('''<robot name="fixture">
      <joint name="previous" type="revolute"><parent link="base"/><child link="middle"/></joint>
      <joint name="terminal" type="revolute"><parent link="middle"/><child link="tip"/>
        <origin xyz="0 0.03 0.004" rpy="0 0 0"/><axis xyz="0 0 1"/>
      </joint></robot>''')
    joint = root.find("joint[@name='terminal']")
    if reflection:
        joint.find('origin').set('xyz', '0 -0.03 0.004')
        joint.find('axis').set('xyz', '0 0 -1')
    else:
        joint.find('origin').set('rpy', ' '.join(map(str, Rotation.from_matrix(rotation.T).as_euler('xyz'))))
        joint.find('axis').set('xyz', ' '.join(map(str, rotation @ [0, 0, 1])))
    return root


def box():
    return np.array([[x, y, z] for x in [-.005, .005] for y in [0., .025]
                     for z in [-.002, .004]])


def test_landmark_is_on_distal_face_not_the_hinge_axis_extreme():
    direction = geometry.distal_direction(chain(), 'tip')
    np.testing.assert_allclose(direction, [0, 1, 0], atol=1e-12)
    point = geometry.distal_support_center(box(), direction)
    np.testing.assert_allclose(point, [0, .025, .001], atol=1e-12)


@pytest.mark.parametrize('rotation', [Rotation.from_euler('xyz', v).as_matrix()
                                    for v in [[.8, -.4, 1.7], [1.2, 1.3, -.7]]])
def test_changing_local_coordinates_preserves_physical_landmark(rotation):
    direction = geometry.distal_direction(chain(rotation), 'tip')
    point = geometry.distal_support_center(box() @ rotation.T, direction)
    np.testing.assert_allclose(point, rotation @ [0, .025, .001], atol=1e-12)


def test_reflection_preserves_landmark_with_axial_vector_sign():
    reflection = np.diag([1., -1., 1.])
    direction = geometry.distal_direction(chain(reflection=True), 'tip')
    point = geometry.distal_support_center(box() @ reflection.T, direction)
    np.testing.assert_allclose(point, [0, -.025, .001], atol=1e-12)


def test_extra_interior_face_vertices_do_not_bias_center():
    vertices = np.concatenate([box(), [[.004, .025, .003], [.003, .025, .002]]])
    np.testing.assert_allclose(geometry.distal_support_center(vertices, [0, 1, 0]),
                               [0, .025, .001], atol=1e-12)


def test_missing_or_degenerate_joint_geometry_is_rejected():
    root = chain()
    root.find("joint[@name='terminal']/origin").set('xyz', '0 0 .004')
    with pytest.raises(ValueError, match='coincident hinge pivots'):
        geometry.distal_direction(root, 'tip')
    with pytest.raises(ValueError, match='incoming hinge'):
        geometry.distal_direction(root, 'unknown')


def test_all_current_hand_tips_are_on_their_supplied_distal_hulls():
    tips = {}
    for side in ('left', 'right'):
        path = ROOT / f'tsensei_v3_2_pianist/assets/urdf/TsenseI_{side}.urdf'
        root = ET.parse(path).getroot()
        for finger in ('thumb', 'index', 'mid', 'ring', 'pinky'):
            kind = 'PIP' if finger == 'thumb' else 'DIP'
            name = f'{side}_hand_{finger}_{kind}_link'
            point = np.asarray(geometry.measure_distal_tip(path, name))
            collision = root.find(f"link[@name='{name}']/collision")
            mesh = collision.find('geometry/mesh')
            vertices = geometry.read_stl_vertices(path.parent / mesh.get('filename'))
            vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            origin = collision.find('origin')
            vertices = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).apply(vertices)
            vertices += np.fromstring(origin.get('xyz'), sep=' ')
            direction = geometry.distal_direction(root, name)
            assert point @ direction == pytest.approx((vertices @ direction).max(), abs=1e-8)
            hull = ConvexHull(vertices)
            assert np.max(hull.equations[:, :3] @ point + hull.equations[:, 3]) < 1e-8
            # Regression for this CAD export, not a new runtime joint limit:
            # the selected point is at the finger end, away from the bearings.
            assert abs(point[1]) > .020
            assert abs(point[2] / np.linalg.norm(point)) < .20
            tips[side, finger] = point
    # The supplied bilateral geometry is authoritative; symmetry is not assumed.


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('finger', ['thumb', 'index', 'mid', 'ring', 'pinky'])
def test_task_point_setback_is_exactly_3mm_toward_palm(side, finger):
    path = ROOT / f'tsensei_v3_2_pianist/assets/urdf/TsenseI_{side}.urdf'
    name = f"{side}_hand_{finger}_{'PIP' if finger == 'thumb' else 'DIP'}_link"
    direction = geometry.distal_direction(ET.parse(path).getroot(), name)
    support = np.asarray(geometry.measure_distal_tip(path, name))
    target = np.asarray(geometry.measure_fingertip_target(path, name))
    delta = target - support
    np.testing.assert_allclose(delta, -.003 * direction, atol=1e-14)
    assert np.linalg.norm(delta) == pytest.approx(.003, abs=1e-14)
    assert delta @ direction < 0
    # Under arbitrary terminal-link rotation, it remains a 3 mm setback,
    # rather than a fixed world-axis offset.
    rotation = Rotation.from_euler('xyz', [.7, -.6, 1.2]).as_matrix()
    np.testing.assert_allclose(rotation @ delta, -.003 * (rotation @ direction), atol=1e-14)
