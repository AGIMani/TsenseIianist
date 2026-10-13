import os
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from dm_control import mjcf

from tsensei_v3_2_pianist.models.hands import collision_proxies, urdf_model
from tsensei_v3_2_pianist.models.hands.fingertip_geometry import measure_distal_tip


class CollisionProxyTest(unittest.TestCase):
    def test_tip_surface_and_whole_link_thickness(self):
        config = collision_proxies.load("shadow_primitive_v1")
        assets = collision_proxies.CONFIG_PATH.parent
        for side, links in config["hands"].items():
            for name, entry in links.items():
                if "preserved_fingertip" not in entry:
                    continue
                tip = np.asarray(measure_distal_tip(assets / "urdf" / f"TsenseI_{side}.urdf", name))
                a, b = np.asarray(entry["fromto"]).reshape(2, 3)
                radius = entry["size"][0]
                np.testing.assert_allclose(b + [0, 0, radius], tip, atol=1e-12)
                # Rounded-cap chord widths (roughly 4 mm) must never be used
                # as whole-terminal diameters again. These bounds come from
                # the actual with-ID terminal meshes, not the 1 mm pole band.
                self.assertGreater(radius, .0058)
                self.assertLess(radius, .0065)
                self.assertEqual(entry['terminal_fit_method'], 'whole_mesh_inertia_tip_anchored_v1')
                self.assertNotIn('distal_measurement_band_m', entry)
                np.testing.assert_allclose([a[2]-radius, b[2]+radius],
                    np.asarray(entry['source_bounds'])[:, 2], atol=1e-12)
                self.assertGreater(b[2], a[2])

    def test_terminal_fit_uses_compiler_radius_and_axis_and_mirrors(self):
        import sys
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts'))
        self.addCleanup(sys.path.remove, str(root / 'scripts'))
        from generate_collision_proxies import inertial_capsule
        config = collision_proxies.load('shadow_primitive_v1')
        assets = collision_proxies.CONFIG_PATH.parent
        for side in ('left', 'right'):
            urdf = assets / 'urdf' / f'TsenseI_{side}.urdf'
            source = ET.parse(urdf).getroot()
            for name, entry in config['hands'][side].items():
                if 'preserved_fingertip' not in entry:
                    continue
                geom = source.find(f"link[@name='{name}']/collision")
                mesh = geom.find('geometry/mesh')
                fitted = inertial_capsule((urdf.parent/mesh.get('filename')).resolve(),
                    mesh.get('scale', '1 1 1'), geom.find('origin'))
                self.assertAlmostEqual(entry['size'][0], fitted['size'][0], places=10)
                a, b = np.asarray(entry['fromto']).reshape(2, 3)
                axis = Rotation.from_quat(np.roll(fitted['quat'], -1)).as_matrix()[:, 2]
                self.assertAlmostEqual(abs(float((b-a)@axis/np.linalg.norm(b-a))), 1., places=10)
                if side == 'left':
                    mirror = config['hands']['right'][name.replace('left_', 'right_', 1)]
                    np.testing.assert_allclose(np.asarray(entry['fromto']).reshape(2, 3)*[1,-1,1],
                        np.asarray(mirror['fromto']).reshape(2, 3), atol=1e-7)
                    self.assertAlmostEqual(entry['size'][0], mirror['size'][0], places=9)

    def test_compiled_physics_changes_only_colliders(self):
        assets = collision_proxies.CONFIG_PATH.parent
        config = collision_proxies.load("shadow_primitive_v1")
        for side in ("left", "right"):
            models = []
            for profile in collision_proxies.PROFILES:
                with patch.dict(os.environ, TSENSEI_V32_COLLISION_PROFILE=profile):
                    root, _ = urdf_model.build(assets / "urdf" / f"TsenseI_{side}.urdf", side + "_")
                    models.append(mjcf.Physics.from_mjcf_model(root))
            modern, legacy = [p.model.ptr for p in models]
            for attr in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_axis", "jnt_range", "actuator_ctrlrange"):
                np.testing.assert_array_equal(getattr(modern, attr), getattr(legacy, attr))
            primitives = 0
            for i in range(modern.ngeom):
                name = modern.geom(i).name
                if "_visual_" in name:
                    self.assertEqual(modern.geom_type[i], mujoco.mjtGeom.mjGEOM_MESH)
                    self.assertEqual(modern.geom_contype[i] | modern.geom_conaffinity[i], 0)
                else:
                    link_name = modern.body(modern.geom_bodyid[i]).name.removeprefix(side + "_/")
                    entry = config["hands"][side][link_name]
                    expected_type = getattr(mujoco.mjtGeom, "mjGEOM_" + entry["type"].upper())
                    self.assertEqual(modern.geom_type[i], expected_type)
                    primitives += entry["type"] != "mesh"
                    if entry.get("physical_role") == "palm_housing":
                        self.assertEqual(entry["type"], "mesh")
                    elif entry.get("physical_role") == "rigid_mount":
                        self.assertEqual(entry["type"], "box")
            self.assertEqual(primitives, 21)
            for p in models:
                p.free()

    def test_bad_profile_rejected(self):
        with self.assertRaises(ValueError):
            collision_proxies.load("typo")


if __name__ == "__main__":
    unittest.main()
