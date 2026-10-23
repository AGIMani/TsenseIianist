"""Guard the vendor's changed pinky naming and mirrored revolute convention."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_models', ROOT / 'scripts/prepare_models.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ModelDerivationTest(unittest.TestCase):
    def schema(self, parts):
        root = ET.Element('robot')
        for part in parts:
            ET.SubElement(root, 'joint', name=f'left_hand_pinky_{part}_joint')
        return prepare.pinky_name_map(root)

    def test_new_semantic_chain_does_not_shift_flexion_joints(self):
        schema, mapping = self.schema(['MCP', 'SWING', 'PIP', 'DIP'])
        self.assertEqual(schema, 'a32_with_id_semantic_chain')
        self.assertEqual(mapping, {
            'left_hand_pinky_SWING_joint': 'left_hand_pinky_swing_joint',
            'left_hand_pinky_SWING_link': 'left_hand_pinky_swing_link'})

    def test_legacy_chain_and_ambiguous_schema(self):
        _, mapping = self.schema(['rotaback', 'MCP', 'swing', 'PIP'])
        self.assertEqual(mapping['left_hand_pinky_MCP_joint'], 'left_hand_pinky_swing_joint')
        for parts in (['MCP', 'PIP', 'DIP'], ['rotaback', 'MCP', 'SWING', 'PIP', 'DIP']):
            with self.assertRaises(ValueError):
                self.schema(parts)

    def test_native_joint_axes_ranges_and_semantic_links_are_preserved(self):
        source = ET.parse(prepare.ASSETS / 'source/left_original.urdf').getroot()
        normalized = ET.parse(prepare.ASSETS / 'urdf/TsenseI_left.urdf').getroot()
        _, mapping = prepare.pinky_name_map(source)
        for joint in source.findall('joint'):
            name = mapping.get(joint.get('name'), joint.get('name'))
            actual = normalized.find(f"joint[@name='{name}']")
            self.assertIsNotNone(actual, name)
            for relation in ('parent', 'child'):
                original = joint.find(relation).get('link')
                self.assertEqual(actual.find(relation).get('link'), mapping.get(original, original))
            if joint.get('type') == 'revolute':
                if name.endswith('thumb_rotaback_joint2'):
                    np.testing.assert_array_equal(np.fromstring(actual.find('axis').get('xyz'),sep=' '),
                                                  -np.fromstring(joint.find('axis').get('xyz'),sep=' '))
                    self.assertEqual(float(actual.find('limit').get('lower')), -float(joint.find('limit').get('upper')))
                    self.assertEqual(float(actual.find('limit').get('upper')), -float(joint.find('limit').get('lower')))
                else:
                    self.assertEqual(actual.find('axis').attrib, joint.find('axis').attrib)
                    self.assertEqual(actual.find('limit').attrib, joint.find('limit').attrib)
        self.assertEqual(len(normalized.findall("joint[@type='revolute']")), 20)

    def test_both_hands_preserve_supplied_axes_and_inertial_mass(self):
        for side in ('left', 'right'):
            active = ET.parse(prepare.ASSETS / f'urdf/TsenseI_{side}.urdf').getroot()
            source = ET.parse(prepare.ASSETS / 'source/latest_project/urdf' /
                              f'TsenseI_hand_{side}_v3_2_withid.urdf').getroot()
            for joint in source.findall("joint[@type='revolute']"):
                actual = active.find(f"joint[@name='{joint.get('name')}']")
                if joint.get('name').endswith('thumb_rotaback_joint2'):
                    np.testing.assert_array_equal(np.fromstring(actual.find('axis').get('xyz'),sep=' '),
                                                  -np.fromstring(joint.find('axis').get('xyz'),sep=' '))
                    self.assertEqual(float(actual.find('limit').get('upper')), 1.74)
                else:
                    self.assertEqual(actual.find('axis').attrib, joint.find('axis').attrib)
                    self.assertEqual(actual.find('limit').attrib, joint.find('limit').attrib)
            for link in source.findall('link'):
                actual = active.find(f"link[@name='{link.get('name')}']")
                self.assertEqual(actual.find('inertial/mass').attrib, link.find('inertial/mass').attrib)

    def test_compiled_mesh_units_are_hand_sized(self):
        for side in ('left', 'right'):
            path = prepare.ASSETS / f'urdf/TsenseI_{side}.urdf'
            for mesh in ET.parse(path).getroot().iter('mesh'):
                _, triangles = prepare.read_triangles(path.parent / mesh.get('filename'))
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                span = np.ptp(triangles['vertices'].reshape(-1, 3), axis=0) * scale
                self.assertTrue(np.all(span < .2), (side, mesh.get('filename'), span))
                self.assertTrue(np.all(span > .001), (side, mesh.get('filename'), span))


if __name__ == '__main__':
    unittest.main()
