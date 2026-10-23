"""Recorded-replay key colors only: no fingertip markers or floating text."""
import mujoco
import numpy as np

FINGER_COLORS = np.array(((.8, .2, .8, 1), (.8, .2, .2, 1),
                         (.2, .8, .8, 1), (.2, .2, .8, 1),
                         (.8, .8, .2, 1)), dtype=np.float32)
DISPLAY_CONTRACT = 'piano_key_colors_only_v1'


class FingeringHighlight:
    def __init__(self, model, task_manifest, key_highlight):
        self.keys = key_highlight
        if not np.array_equal(self.keys.keys, np.arange(88)):
            raise ValueError('Replay key colors require 88 ordered piano keys')

    @staticmethod
    def assignments(fingers):
        fingers = np.asarray(fingers)
        if fingers.shape != (10, 88):
            raise ValueError('Expected a score frame with shape (10, 88)')
        return fingers > 0

    def append(self, scene, data, fingers):
        assigned = self.assignments(fingers)
        active = self.keys.active(data)
        for key in np.flatnonzero(assigned.any(axis=0)):
            fingers_here = np.flatnonzero(assigned[:, key])
            if active[key] or len(fingers_here) != 1:
                continue
            if hasattr(self.keys, 'color_mesh') and self.keys.color_mesh(scene, key, FINGER_COLORS[fingers_here[0] % 5]):
                continue
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('Viewer scene has no room for piano key colors')
            source = self.keys.geom_ids[key]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                self.keys.sizes[key] + .00002, data.geom_xpos[source],
                data.geom_xmat[source], FINGER_COLORS[fingers_here[0] % 5])
            # Clear reusable MuJoCo storage; no text/marker is constructed.
            geom.label = ''
            scene.ngeom += 1
