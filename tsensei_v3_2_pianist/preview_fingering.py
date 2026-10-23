"""Visual-only score annotations for the frozen MJLab scene.

The palette matches models/hands/tsensei_hand_constants.py. Keep this small
display module independent of the Composer hand package on the GPU runtime.
"""
import mujoco
import numpy as np

FINGER_COLORS = np.array(((.8, .2, .8, 1), (.8, .2, .2, 1),
                          (.2, .8, .8, 1), (.2, .2, .8, 1),
                          (.8, .8, .2, 1)), dtype=np.float32)
FINGER_NAMES = ('thumb', 'index', 'middle', 'ring', 'little')


def note_name(key):
    midi = int(key) + 21
    return ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')[midi % 12] + str(midi // 12 - 1)


def finger_label(finger):
    return ('R' if finger < 5 else 'L') + str(finger % 5 + 1)


class FingeringHighlight:
    def __init__(self, model, task_manifest, key_highlight):
        self.keys = key_highlight
        self.tip_ids = np.array([model.site(n).id for n in task_manifest['fingertip_sites']])
        if len(self.tip_ids) != 10 or not np.array_equal(self.keys.keys, np.arange(88)):
            raise ValueError('Fingering preview requires ten named fingertips and 88 ordered piano keys')

    @staticmethod
    def assignments(fingers):
        fingers = np.asarray(fingers)
        if fingers.shape != (10, 88):
            raise ValueError('Expected a single score frame with shape (10, 88)')
        return fingers > 0

    def text(self, fingers, index, length, completed=False):
        assigned = self.assignments(fingers)
        heading = 'Final target' if completed else 'Next target'
        lines = [f'{heading} frame {index}/{length-1} (0-based)',
                 'R = right, L = left; green = pressed',
                 '1 thumb: pink | 2 index: red | 3 middle: cyan',
                 '4 ring: blue | 5 little: yellow']
        for finger in range(10):
            keys = np.flatnonzero(assigned[finger])
            notes = ', '.join(f'{note_name(k)} [key {k}]' for k in keys) or '-'
            lines.append(f'{finger_label(finger)} {FINGER_NAMES[finger % 5]}: {notes}')
        return '\n'.join(lines)

    @staticmethod
    def _geom(scene, kind, size, pos, mat, color):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Viewer scene has no room for fingering annotations')
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, size, pos, mat, color)
        geom.label = ''
        scene.ngeom += 1

    def append(self, scene, data, fingers):
        assigned = self.assignments(fingers)
        active = self.keys.active(data)
        identity = np.eye(3).ravel()
        for key in np.flatnonzero(assigned.any(axis=0)):
            fingers_here = np.flatnonzero(assigned[:, key])
            geom = self.keys.geom_ids[key]
            size = self.keys.sizes[key]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            # One clear key color when unambiguous. Multiple assignments retain
            # individual markers instead of silently picking one finger.
            if not active[key] and len(fingers_here) == 1:
                if self.keys.color_mesh(scene, key, FINGER_COLORS[fingers_here[0] % 5]):
                    continue
                self._geom(scene, mujoco.mjtGeom.mjGEOM_BOX, size + .00002,
                           data.geom_xpos[geom], rotation.ravel(),
                           FINGER_COLORS[fingers_here[0] % 5])
