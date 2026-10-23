"""Check current URDF data and integrator reporting without old frozen bundles.

The optional compiled check consumes an existing MJB; it does not compile or
run a GPU model. Set PIANIST_CURRENT_MODEL_MJB for an independent CPU audit.
"""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tsensei_v3_2_pianist import physics_settings

ASSETS = Path(__file__).resolve().parents[1] / 'tsensei_v3_2_pianist/assets'
REFLECTION = np.diag([1., -1., 1.])


def origin(element):
    value = element.find('origin')
    if value is None:
        return np.zeros(3), np.eye(3)
    return (np.fromstring(value.get('xyz', '0 0 0'), sep=' '),
            Rotation.from_euler('xyz', np.fromstring(value.get('rpy', '0 0 0'), sep=' ')).as_matrix())


def link_inertia(link):
    element = link.find('inertial')
    assert element is not None, link.get('name')
    mass = float(element.find('mass').get('value'))
    com, rotation = origin(element)
    terms = element.find('inertia').attrib
    tensor = np.array([[float(terms['i' + a + b] if 'i' + a + b in terms else terms['i' + b + a])
                        for b in 'xyz'] for a in 'xyz'])
    return mass, com, rotation @ tensor @ rotation.T


def urdf(side):
    return ET.parse(ASSETS / 'urdf' / f'TsenseI_{side}.urdf').getroot()


@pytest.mark.parametrize('side', ['left', 'right'])
def test_all_current_links_have_physical_explicit_inertias(side):
    source = urdf(side)
    links = source.findall('link')
    assert len(links) == 22
    masses = []
    for link in links:
        mass, com, tensor = link_inertia(link)
        assert mass > 0 and np.isfinite(com).all() and np.isfinite(tensor).all()
        values = np.linalg.eigvalsh(tensor)
        assert values.min() > 0, link.get('name')
        assert values[-1] <= values[0] + values[1] + 1e-15, link.get('name')
        masses.append(mass)
    assert sum(masses) == pytest.approx(.952, abs=1e-12)
    joints = source.findall('joint')
    assert len(joints) == 21
    assert len({j.find('child').get('link') for j in joints}) == 21
    for joint in joints:
        if joint.get('type') == 'fixed':
            continue
        assert joint.get('type') == 'revolute'
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        assert np.linalg.norm(axis) == pytest.approx(1.)
        limit = joint.find('limit')
        assert float(limit.get('lower')) < float(limit.get('upper'))


@pytest.mark.parametrize('side', ['left', 'right'])
def test_active_urdf_preserves_its_own_supplied_joint_tree_and_inertias(side):
    source = ET.parse(ASSETS / 'source/latest_project/urdf' /
                      f'TsenseI_hand_{side}_v3_2_withid.urdf').getroot()
    active = urdf(side)
    for original in source.findall('link'):
        actual = active.find(f"link[@name='{original.get('name')}']")
        for a, b in zip(link_inertia(original), link_inertia(actual)):
            np.testing.assert_allclose(a, b, atol=1e-15, rtol=0)
    for original in source.findall('joint'):
        actual = active.find(f"joint[@name='{original.get('name')}']")
        assert actual.attrib == original.attrib
        for element in original:
            observed = actual.find(element.tag)
            if original.get('name') == f'{side}_hand_thumb_rotaback_joint2' and element.tag == 'axis':
                np.testing.assert_array_equal(np.fromstring(observed.get('xyz'), sep=' '),
                                              -np.fromstring(element.get('xyz'), sep=' '))
            elif original.get('name') == f'{side}_hand_thumb_rotaback_joint2' and element.tag == 'limit':
                assert float(observed.get('lower')) == -float(element.get('upper'))
                assert float(observed.get('upper')) == -float(element.get('lower'))
                for key in ('effort', 'velocity'): assert observed.get(key) == element.get(key)
            else:
                assert observed.attrib == element.attrib


def test_active_hulls_and_urdfs_match_recorded_derivation():
    derivation = json.loads((ASSETS / 'model_derivation.json').read_text())
    assert derivation['right_hand_is_provisional_mirror'] is False
    assert derivation['right_hand_source'] == 'supplied_package'
    assert 'TODO' in derivation['inertia_estimation']['hardware_calibration_status']
    profiles = [json.loads((ASSETS / name).read_text()) for name in
                ('collision_proxies.json', 'collision_proxies_convex_hull.json')]
    for side in ('left', 'right'):
        source = urdf(side)
        for link in source.findall('link'):
            name = link.get('name')
            collisions = link.findall('collision')
            assert len(collisions) == 1
            mesh = collisions[0].find('geometry/mesh')
            path = (ASSETS / 'urdf' / mesh.get('filename')).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = str(path.relative_to(ASSETS))
            assert digest == derivation['generated_sha256'][relative]
            for profile in profiles:
                entry = profile['hands'][side][name]
                assert entry['type'] == 'mesh' and not entry.get('preserve_original_mesh', False)
                assert (ASSETS / entry['mesh_file']).resolve() == path
                assert entry['hull_sha256'] == digest


def settings_model(integrator):
    """Tiny physical scene with the named groups used by from_model."""
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".005" integrator="{integrator}"/>
      <worldbody>
        <geom name="left_/hand" type="sphere" size=".01"/>
        <body name="piano/key"><joint name="piano/key" type="hinge"/>
          <geom name="piano/key" type="box" size=".01 .01 .01"/>
        </body>
      </worldbody></mujoco>''')


@pytest.mark.parametrize('integrator', ['Euler', 'implicitfast'])
def test_from_model_reports_actual_supported_integrator(integrator):
    model = settings_model(integrator)
    assert physics_settings.from_model(model, .05)['integrator'] == integrator


@pytest.mark.parametrize('integrator', ['RK4', 'implicit'])
def test_from_model_rejects_unrepresentable_integrator(integrator):
    with pytest.raises(ValueError, match='Unsupported compiled integrator'):
        physics_settings.from_model(settings_model(integrator), .05)


def test_current_default_and_impedance_restore_preserve_euler():
    assert physics_settings.load()['integrator'] == 'Euler'
    model = settings_model('Euler')
    controller = physics_settings.SolimpController(model, physics_settings.from_model(model, .05))
    controller.restore_shadow()
    controller.apply()
    assert controller.settings['integrator'] == 'Euler'
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER


def test_compiled_current_hands_preserve_all_explicit_urdf_inertias():
    path = os.environ.get('PIANIST_CURRENT_MODEL_MJB')
    if not path:
        pytest.skip('Set PIANIST_CURRENT_MODEL_MJB to audit an existing compiled model on CPU')
    model = mujoco.MjModel.from_binary_path(path)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    for side in ('left', 'right'):
        for link in urdf(side).findall('link'):
            body = model.body(f"{side}_/{link.get('name')}")
            mass, com, tensor = link_inertia(link)
            assert model.body_mass[body.id] == pytest.approx(mass, abs=1e-12)
            np.testing.assert_allclose(model.body_ipos[body.id], com, atol=1e-12, rtol=0)
            rotation = np.empty((3, 3))
            mujoco.mju_quat2Mat(rotation.ravel(), model.body_iquat[body.id])
            actual = rotation @ np.diag(model.body_inertia[body.id]) @ rotation.T
            # fullinertia diagonalization uses MuJoCo's finite eigensolver
            # tolerance; compare the actual converter to an independent tiny
            # explicit-inertial model, without visual/collision geometry.
            full = tensor[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
            reference = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body><inertial pos="0 0 0" mass="1" fullinertia="'
                + ' '.join(map(str, full)) + '"/></body></worldbody></mujoco>')
            np.testing.assert_allclose(model.body_inertia[body.id], reference.body_inertia[1],
                                       atol=1e-15, rtol=0)
            np.testing.assert_allclose(np.abs(model.body_iquat[body.id]), np.abs(reference.body_iquat[1]),
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual, tensor, atol=1e-10, rtol=0)
            assert np.min(model.body_inertia[body.id]) > 0
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20261015)
    for sample in range(4):
        if sample:
            for joint in range(model.njnt):
                if model.jnt_limited[joint] and model.jnt_type[joint] in (
                        mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    data.qpos[model.jnt_qposadr[joint]] = rng.uniform(*model.jnt_range[joint])
        # Only kinematics and inertia; no contact solve, policy, or simulation.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)
        matrix = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, matrix)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12, rtol=0)
        np.linalg.cholesky(matrix)
