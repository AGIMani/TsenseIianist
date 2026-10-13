"""Numerical and idempotence checks for the supplied 952 g hand revision."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'tsensei_v3_2_pianist/assets'
spec = importlib.util.spec_from_file_location('mass_revision', ROOT / 'scripts/apply_hand_mass_revision.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def revision_copy(tmp_path):
    (tmp_path / 'urdf').mkdir()
    for name in ('mass_update_report.json', 'inertia_update_report.json',
                 'urdf/TsenseI_left.urdf', 'urdf/TsenseI_right.urdf'):
        if name.startswith('urdf/') and (ASSETS / 'connector_mount_revision.json').exists():
            side = 'left' if 'left' in name else 'right'
            shutil.copy2(ASSETS / f'source/TsenseI_{side}_mass952_before_mount.urdf', tmp_path / name)
        else:
            shutil.copy2(ASSETS / name, tmp_path / name)
    return tmp_path


def test_44_link_revision_and_mirror():
    report = json.loads((ASSETS / 'inertia_update_report.json').read_text())
    matrices = {}
    for side in ('left', 'right'):
        tree = ET.parse(ASSETS / f'urdf/TsenseI_{side}.urdf').getroot()
        assert len(tree.findall('link')) == 22
        assert sum(float(n.get('value')) for n in tree.findall('link/inertial/mass')) == pytest.approx(.952)
        matrices[side] = []
        for row in report['hands'][side]['links']:
            node = tree.find(f"link[@name='{row['link']}']/inertial")
            assert node.find('origin').attrib == row['unchanged_inertial_origin']
            data = {k: float(v) for k, v in node.find('inertia').attrib.items()}
            for key in data:
                assert data[key] == pytest.approx(float(row['original_inertia'][key]) * float(row['new_mass_kg']) / float(row['old_mass_kg']), rel=2e-15, abs=1e-25)
            matrix = np.array([[data['ixx'], data['ixy'], data['ixz']],
                               [data['ixy'], data['iyy'], data['iyz']],
                               [data['ixz'], data['iyz'], data['izz']]])
            values = np.linalg.eigvalsh(matrix)
            assert values[0] > 0 and values[2] < values[:2].sum()
            matrices[side].append(matrix)
    reflection = np.diag([1., -1., 1.])
    np.testing.assert_allclose(matrices['right'], reflection @ matrices['left'] @ reflection, rtol=2e-15, atol=1e-25)


def test_original_to_exact_supplied_hash_and_idempotence(tmp_path):
    assets = revision_copy(tmp_path)
    mass = json.loads((assets / 'mass_update_report.json').read_text())
    inertia = json.loads((assets / 'inertia_update_report.json').read_text())
    for side in ('left', 'right'):
        path = assets / f'urdf/TsenseI_{side}.urdf'
        root = ET.parse(path).getroot()
        for row in inertia['hands'][side]['links']:
            node = root.find(f"link[@name='{row['link']}']/inertial")
            node.find('mass').set('value', row['old_mass_kg'])
            for key, value in row['original_inertia'].items():
                node.find('inertia').set(key, value)
        path.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == mass['hands'][side]['original_sha256']
    module.apply_revision(assets)
    before = {p: p.read_bytes() for p in (assets / 'urdf').glob('*.urdf')}
    module.apply_revision(assets)
    for p, raw in before.items():
        assert p.read_bytes() == raw


def test_unapproved_model_is_rejected_without_partial_write(tmp_path):
    assets = revision_copy(tmp_path)
    left = assets / 'urdf/TsenseI_left.urdf'
    right = assets / 'urdf/TsenseI_right.urdf'
    before = left.read_bytes()
    right.write_bytes(right.read_bytes().replace(b'0.15"', b'0.16"', 1))
    with pytest.raises(ValueError, match='approved'):
        module.apply_revision(assets)
    assert left.read_bytes() == before
