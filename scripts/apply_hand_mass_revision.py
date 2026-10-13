#!/usr/bin/env python3
"""Apply the supplied, audited per-link mass/inertia revision exactly once.

Original vendor URDFs remain untouched. Re-running prepare_models.py reapplies
the revision to newly derived originals, never to already scaled inertias.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def apply_revision(assets):
    assets = Path(assets)
    mass = json.loads((assets / 'mass_update_report.json').read_text())
    inertia = json.loads((assets / 'inertia_update_report.json').read_text())
    mount_path = assets / 'connector_mount_revision.json'
    mount = json.loads(mount_path.read_text()) if mount_path.exists() else None
    planned = {}
    for side in ('left', 'right'):
        path = assets / f'urdf/TsenseI_{side}.urdf'
        raw = path.read_bytes()
        expected = mass['hands'][side]
        digest = hashlib.sha256(raw).hexdigest()
        mounted_digest = mount['urdf_after_sha256'][side] if mount is not None else None
        if digest not in (expected['original_sha256'], expected['updated_sha256'], mounted_digest):
            raise ValueError(f'{side}: URDF is neither the approved original nor the approved revision')
        updated = digest in (expected['updated_sha256'], mounted_digest)
        tree = ET.fromstring(raw)
        changes = {row['link']: row for row in expected['changes']}
        tensors = {row['link']: row for row in inertia['hands'][side]['links']}
        if len(changes) != 22 or set(changes) != set(tensors) or set(changes) != {n.get('name') for n in tree.findall('link')}:
            raise ValueError(f'{side}: incomplete or inconsistent link mapping')
        total = 0.
        for link in tree.findall('link'):
            name = link.get('name')
            change, tensor = changes[name], tensors[name]
            old, new = float(change['old_kg']), float(change['new_kg'])
            if not np.isfinite([old, new]).all() or min(old, new) <= 0:
                raise ValueError(f'{name}: invalid mass')
            if (old != float(tensor['old_mass_kg']) or new != float(tensor['new_mass_kg'])):
                raise ValueError(f'{name}: report masses disagree')
            node = link.find('inertial')
            if node.find('origin').attrib != tensor['unchanged_inertial_origin']:
                raise ValueError(f'{name}: center of mass changed')
            if float(node.find('mass').get('value')) != (new if updated else old):
                raise ValueError(f'{name}: unexpected existing mass')
            target = tensor['updated_inertia']
            for component in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'):
                base = float(tensor['original_inertia'][component])
                value = float(target[component])
                if not np.isfinite(value) or not np.isclose(value, base * new / old, rtol=2e-15, atol=1e-25):
                    raise ValueError(f'{name}: invalid inertia scaling for {component}')
                current = float(node.find('inertia').get(component))
                if current != (value if updated else base):
                    raise ValueError(f'{name}: unexpected existing inertia')
                node.find('inertia').set(component, target[component])
            i = {k: float(v) for k, v in target.items()}
            values = np.linalg.eigvalsh([[i['ixx'], i['ixy'], i['ixz']],
                                        [i['ixy'], i['iyy'], i['iyz']],
                                        [i['ixz'], i['iyz'], i['izz']]])
            if values[0] <= 0 or values[2] > values[:2].sum() + 1e-15:
                raise ValueError(f'{name}: inertia is not physically consistent')
            node.find('mass').set('value', change['new_kg'])
            total += new
        if not np.isclose(total, .952, rtol=0, atol=1e-14):
            raise ValueError(f'{side}: expected 0.952 kg')
        result = raw if updated else ET.tostring(tree, encoding='utf-8', xml_declaration=True)
        output_digest = mounted_digest if digest == mounted_digest else expected['updated_sha256']
        if hashlib.sha256(result).hexdigest() != output_digest:
            raise ValueError(f'{side}: output differs from the supplied corrected URDF fingerprint')
        planned[path] = result
    # Validate both sides completely before changing either file.
    for path, content in planned.items():
        if path.read_bytes() != content:
            path.write_bytes(content)
    return {
        'source_workbook_sha256': mass['source_workbook_sha256'],
        'report': 'mass_update_report.json', 'single_hand_mass_kg': .952,
        'inertia_tensor_updated': True,
        'previous_urdf_sha256': {s: mass['hands'][s]['original_sha256'] for s in ('left', 'right')},
        'inertia_method': inertia['method'], 'inertia_report': 'inertia_update_report.json',
        'inertia_assumption': inertia['assumption'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[1] / 'tsensei_v3_2_pianist/assets')
    args = parser.parse_args()
    revision = apply_revision(args.assets)
    path = args.assets / 'model_derivation.json'
    derivation = json.loads(path.read_text())
    derivation['mass_revision'] = revision
    for side in ('left', 'right'):
        name = f'urdf/TsenseI_{side}.urdf'
        derivation['generated_sha256'][name] = hashlib.sha256((args.assets / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(derivation, indent=2) + '\n')
    print(json.dumps({'links_verified': 44, 'mass_per_hand_kg': .952, 'total_hand_mass_kg': 1.904,
                      'matches_supplied_mass_and_inertia': True, 'inertia_validation': 'passed',
                      'geometry_revision': (args.assets / 'connector_mount_revision.json').exists()}))


if __name__ == '__main__':
    main()
