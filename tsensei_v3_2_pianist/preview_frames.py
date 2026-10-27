# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
"""World, palm_base, and optional M6 arm joint frames for the preview scene."""
import mujoco
import numpy as np


class CoordinateFrames:
    """RGB positive axes at world origin, hand roots, and optional M6 links.

    Palm_base means the normalized base_link carrying the virtual palm_base joints,
    not the original CAD root or the body's inertial center. M6 J1-J7 entries
    are the moving child-link frames from ArmPreview's kinematic model.
    """

    COLORS = ((1, .15, .15, 1), (.15, 1, .15, 1), (.2, .45, 1, 1))

    def __init__(self, physics, hands, axis_length=.09, arm_preview=None):
        if not np.isfinite(axis_length) or axis_length <= 0:
            raise ValueError('Axis length must be a finite positive number')
        self._physics = physics
        self.axis_length = axis_length
        # Entries use either the live piano physics data or the separate
        # kinematic M6 model maintained by ArmPreview.
        self._frames = [('World', 'physics', 0, 'world')]
        for hand in hands:
            name = hand.root_body.full_identifier
            body_id = physics.model.ptr.body(name).id
            label = hand.hand_side.name.capitalize() + ' palm_base'
            self._frames.append((label, 'physics', body_id, name))
        if arm_preview is not None:
            for side in ('left', 'right'):
                arm = arm_preview.sides[side]
                for index, joint_name in enumerate(arm['joints'], 1):
                    joint_id = arm_preview.model.joint(joint_name).id
                    body_id = int(arm_preview.model.jnt_bodyid[joint_id])
                    body_name = arm_preview.model.body(body_id).name
                    self._frames.append((f'{side.capitalize()} J{index}', 'arm', body_id, body_name))
        self._arm_preview = arm_preview
        self.geom_count = 6 * len(self._frames)  # Three arrows and endpoint labels.
        self._scene = None
        self._start = None

    def report(self):
        data = self._physics.data.ptr
        result = []
        for label, source, body_id, name in self._frames:
            source_data = self._arm_preview.data if source == 'arm' else data
            result.append({'label': label, 'body': name,
                 'origin_world_m': source_data.xpos[body_id].tolist(),
                 'rotation_world_from_local': source_data.xmat[body_id].reshape(3, 3).tolist(),
                 'axis_length_m': self.axis_length}
                )
        return result

    def _draw(self, scene, start, initialize):
        if start + self.geom_count > scene.maxgeom:
            raise ValueError(f'Coordinate frames need {self.geom_count} extra scene geoms')
        data = self._physics.data.ptr
        index = start
        for label, source, body_id, _ in self._frames:
            frame_data = self._arm_preview.data if source == 'arm' else data
            origin = frame_data.xpos[body_id]
            rotation = frame_data.xmat[body_id].reshape(3, 3)
            for axis, color in enumerate(self.COLORS):
                arrow, text = scene.geoms[index:index + 2]
                if initialize:
                    for geom, kind in ((arrow, mujoco.mjtGeom.mjGEOM_ARROW),
                                       (text, mujoco.mjtGeom.mjGEOM_LABEL)):
                        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3),
                                           np.eye(3).ravel(), np.asarray(color))
                        geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                        geom.objid = -1
                        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                        geom.segid = -1
                    arrow.emission = .5
                    text.label = f'{label} {"XYZ"[axis]}'
                direction = rotation[:, axis]
                end = origin + self.axis_length * direction
                mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .0025, origin, end)
                text.pos[:] = end + .008 * direction
                index += 2

    def append_to_scene(self, scene):
        """Append after the physical scene (and optional collision wireframe)."""
        self._draw(scene, scene.ngeom, initialize=True)
        scene.ngeom += self.geom_count

    def update_scene(self, scene, start=0):
        """Reuse a fixed slot range in viewer.user_scn, after the wireframe."""
        initialize = (self._scene is not scene or self._start != start
                      or scene.ngeom < start + self.geom_count)
        self._draw(scene, start, initialize)
        scene.ngeom = max(scene.ngeom, start + self.geom_count)
        self._scene, self._start = scene, start
