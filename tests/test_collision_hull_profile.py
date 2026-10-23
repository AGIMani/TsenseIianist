import os
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from dm_control import mjcf

from tsensei_v3_2_pianist.models.hands import collision_proxies, urdf_model


class ConvexHullCollisionProfileTest(unittest.TestCase):
    def test_profile_records_all_22_links_without_raw_mesh_exceptions(self):
        config = collision_proxies.load('convex_hull_v1')
        self.assertEqual(config['profile'], 'convex_hull_v1')
        for side, entries in config['hands'].items():
            self.assertEqual(len(entries), 22)
            preserved = [name for name, entry in entries.items() if entry.get('preserve_original_mesh')]
            self.assertEqual(preserved, [])
            hulls = [part for entry in entries.values() for part in entry.get('parts', [entry])]
            self.assertGreaterEqual(len(hulls), 22)
            self.assertTrue(all(entry.get('hull_triangle_count', entry.get('triangles', 0)) > 4 for entry in hulls))
            for entry in hulls:
                self.assertTrue((collision_proxies.CONFIG_PATH.parent / entry['mesh_file']).is_file())

    def test_hull_profile_keeps_mass_and_joint_contract(self):
        assets = collision_proxies.CONFIG_PATH.parent
        models = []
        for profile in ('legacy_mesh_v1', 'convex_hull_v1'):
            with patch.dict(os.environ, {'TSENSEI_V32_COLLISION_PROFILE': profile}):
                root, _ = urdf_model.build(assets / 'urdf' / 'TsenseI_left.urdf', 'left_')
                models.append(mjcf.Physics.from_mjcf_model(root))
        legacy, hull = [physics.model.ptr for physics in models]
        for field in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'jnt_axis',
                      'jnt_range', 'actuator_ctrlrange'):
            np.testing.assert_array_equal(getattr(legacy, field), getattr(hull, field))
        self.assertEqual(hull.ngeom, legacy.ngeom)
        self.assertEqual(hull.nmesh, legacy.nmesh)
        self.assertTrue(all(hull.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in range(hull.ngeom)))
        for physics in models:
            physics.free()

class HullGeometryAcceptanceTest(unittest.TestCase):
    def test_all_source_vertices_enclosed_and_faces_outward(self):
        import hashlib
        import xml.etree.ElementTree as ET
        from scipy.spatial import ConvexHull
        from scripts.generate_collision_hulls import read_stl_vertices
        assets = collision_proxies.CONFIG_PATH.parent
        config = collision_proxies.load('convex_hull_v1')
        for side, entries in config['hands'].items():
            urdf = ET.parse(assets / 'urdf' / f'TsenseI_{side}.urdf').getroot()
            for name, entry in entries.items():
                if 'mesh_file' not in entry:
                    continue
                with self.subTest(side=side, link=name):
                    source = assets / entry['source_mesh']
                    mesh = urdf.find(f"link[@name='{name}']/collision/geometry/mesh")
                    vertices = read_stl_vertices(source) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                    target = assets / entry['mesh_file']
                    triangles = read_stl_vertices(target).reshape(-1, 3, 3)
                    hull = ConvexHull(triangles.reshape(-1, 3))
                    # Check source extremes: interior source vertices cannot
                    # violate a plane if its original hull vertices do not.
                    extremes = vertices[ConvexHull(vertices).vertices]
                    error = (extremes @ hull.equations[:, :3].T + hull.equations[:, 3]).max()
                    self.assertLessEqual(error, 1e-7)  # 0.1 micrometre float32 tolerance
                    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                    center = triangles.reshape(-1, 3).mean(0)
                    self.assertTrue(np.all(np.einsum('ij,ij->i', normals, triangles.mean(1)-center) > 0))
                    self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), entry['hull_sha256'])
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry['source_sha256'])

    def test_collision_display_preserves_physics_and_enables_actual_hulls(self):
        from tsensei_v3_2_pianist.preview_geometry import configure_options, color_hand_collision_geoms
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="hand">
            <geom name="left_/finger_visual_0" type="box" size=".01 .01 .01" group="1" contype="0" conaffinity="0"/>
            <geom name="left_/finger_collision_0" type="box" size=".01 .01 .01" group="3"/>
            </body></worldbody></mujoco>''')
        masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
        size = model.geom_size.copy()
        color_hand_collision_geoms(model)
        np.testing.assert_array_equal(model.geom_contype, masks[0])
        np.testing.assert_array_equal(model.geom_conaffinity, masks[1])
        np.testing.assert_array_equal(model.geom_size, size)
        self.assertLess(model.geom_rgba[0, 3], model.geom_rgba[1, 3])
        options = mujoco.MjvOption()
        configure_options(options, show_hand_collisions=True)
        self.assertEqual(options.geomgroup[3], 1)
        for flag in ('mjVIS_CONVEXHULL', 'mjVIS_CONTACTPOINT', 'mjVIS_CONTACTFORCE'):
            self.assertTrue(options.flags[getattr(mujoco.mjtVisFlag, flag)])


if __name__ == '__main__':
    unittest.main()
