"""Source-preserving bilateral replacement and restored thumb-coordinate regression."""
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'tsensei_v3_2_pianist/assets'
SPEC = importlib.util.spec_from_file_location('sync_hand', ROOT/'scripts/sync_project_urdf.py')
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def test_all_source_and_generated_hashes_match():
    metadata = json.loads((ASSETS/'model_derivation.json').read_text())
    assert metadata['source_zip_sha256'] == '0e260c21eaa83e0acf6ab3f8112c8ca7820382c56e45e1064ea5915ff697132f'
    for relative, expected in metadata['generated_sha256'].items():
        with (ASSETS/relative).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, relative


@pytest.mark.parametrize('side', ['left', 'right'])
def test_merged_geometry_matches_each_supplied_part_transform(side):
    package = ASSETS/'source/latest_project'
    path = package/'urdf'/f'TsenseI_hand_{side}_v3_2_withid.urdf'
    for link in ET.parse(path).getroot().findall('link'):
        name = link.get('name')
        for kind, target in [('visual', package/'merged'/side/f'{name}.STL'),
                             ('collision', ASSETS/'collision_sources'/side/f'{name}.stl')]:
            expected = IMPORTER.combined_geometry(package, path, link, kind).astype(np.float32)
            actual = IMPORTER.read_triangles(target)['vertices']
            np.testing.assert_array_equal(actual, expected, err_msg=f'{side}/{name}/{kind}')


@pytest.mark.parametrize('profile', ['direct_joint_v1', 'wuji_cartesian_v1'])
def test_thumb_restored_range_and_initial_target_fit_active_limits(profile):
    spec = importlib.util.spec_from_file_location('hand_constants',
        ROOT/'tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py')
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    rows = json.loads((ROOT/'configs/actions'/f'{profile}.json').read_text())['channels']
    for side in ('left', 'right'):
        root = ET.parse(ASSETS/'urdf'/f'TsenseI_{side}.urdf').getroot()
        limit = root.find(f"joint[@name='{side}_hand_thumb_rotaback_joint2']/limit")
        row = next(r for r in rows if r['name'] == side[0].upper()+'_Thumb_Rot2')
        assert float(limit.get('lower')) <= row['physical_min'] < row['physical_max'] <= float(limit.get('upper'))
        assert (row['physical_min'], row['physical_max']) == (.05, 1.57)
        angle = constants.INITIAL_ANGLES_BY_SIDE[side]['thumb_rotaback_joint2']
        assert row['physical_min'] < angle < row['physical_max']
        assert angle == .476648962


@pytest.mark.parametrize('side', ['left', 'right'])
def test_connector_four_holes_match_supplied_palm_in_hand_coordinates(side):
    from scipy.spatial.transform import Rotation
    config = json.loads((ROOT/'configs/preview/hand_connector_20260930.json').read_text())
    source = ET.parse(ASSETS/'urdf'/f'TsenseI_{side}.urdf').getroot()
    origin = source.find(f"joint[@name='{side.capitalize()}_hand_palm_joint']/origin")
    palm = IMPORTER.read_triangles(ASSETS/'source/latest_project/merged'/side/
        f'{side.capitalize()}_hand_palm_link.STL')['vertices'].reshape(-1,3).astype(float)
    palm = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'),sep=' ')).apply(palm)
    palm += np.fromstring(origin.get('xyz'),sep=' ')
    plate = IMPORTER.read_triangles(ROOT/config['meshes'][side][-1])['vertices'].reshape(-1,3).astype(float)
    offset = config['offsets'][side]
    plate = Rotation.from_euler('xyz',offset['rpy_deg'],degrees=True).apply(plate)
    plate += np.asarray(offset['xyz_mm'])/1000.
    for x in [-.0065,.0335]:
        for y in [-.0055,.0135]:
            for vertices,diameter in [(palm,.0037),(plate,.0043)]:
                plane = vertices[np.abs(vertices[:,2]+.004)<2e-7]
                ring = plane[np.abs(np.linalg.norm(plane[:,:2]-[x,y],axis=1)-diameter/2)<2e-7]
                ring=np.unique(np.round(ring,7),axis=0)
                assert len(ring)>=20, (side,x,y,diameter)
                np.testing.assert_allclose((ring.min(0)+ring.max(0))/2,[x,y,-.004],atol=2e-7,rtol=0)


@pytest.mark.parametrize('side,old_axis', [('left',[0,0,-1]),('right',[0,0,1])])
def test_restored_thumb_axis_has_identical_physical_rotation(side,old_axis):
    from scipy.spatial.transform import Rotation
    source = ET.parse(ASSETS/'source/latest_project/urdf'/f'TsenseI_hand_{side}_v3_2_withid.urdf').getroot()
    active = ET.parse(ASSETS/'urdf'/f'TsenseI_{side}.urdf').getroot()
    name=f'{side}_hand_thumb_rotaback_joint2'
    raw=source.find(f"joint[@name='{name}']")
    current=active.find(f"joint[@name='{name}']")
    source_axis=np.fromstring(raw.find('axis').get('xyz'),sep=' ')
    axis=np.fromstring(current.find('axis').get('xyz'),sep=' ')
    np.testing.assert_array_equal(axis,old_axis)
    assert current.find('origin').attrib == raw.find('origin').attrib
    for q in np.linspace(0,1.74,19):
        np.testing.assert_allclose(Rotation.from_rotvec(axis*q).as_matrix(),
                                   Rotation.from_rotvec(source_axis*(-q)).as_matrix(),atol=1e-14,rtol=0)
