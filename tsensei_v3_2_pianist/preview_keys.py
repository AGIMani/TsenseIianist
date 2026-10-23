"""Green activation overlays without full-model copies or physics changes."""
import re
import mujoco
import numpy as np

GREEN = np.array((.2, .8, .2, 1.), dtype=np.float32)
if __package__:
    from .key_activation import KEY_ACTIVATION_FRACTION
else:  # Portable recorded replay, copied alongside this module.
    from key_activation import KEY_ACTIVATION_FRACTION


class ActiveKeyHighlight:
    def __init__(self, model):
        pairs = []
        for geom in range(model.ngeom):
            match = re.search(r'piano/(?:white|black)_key_geom_(\d+)$', model.geom(geom).name or '')
            if match:
                pairs.append((int(match[1]), geom))
        pairs.sort()
        self.geom_ids = np.array([g for _, g in pairs], dtype=int)
        self.keys = np.array([k for k, _ in pairs], dtype=int)
        joints = model.body_jntadr[model.geom_bodyid[self.geom_ids]]
        self.qadr = model.jnt_qposadr[joints]
        self.maximum = model.jnt_range[joints, 1]
        self.sizes = model.geom_size[self.geom_ids].copy()
        self.mesh_keys = model.geom_type[self.geom_ids] == mujoco.mjtGeom.mjGEOM_MESH

    def active(self, data):
        return np.clip(data.qpos[self.qadr], 0, self.maximum) >= self.maximum * KEY_ACTIVATION_FRACTION

    def append(self, scene, data):
        """Append tiny visual shells to user_scn; caller resets ngeom each frame.

        20-micrometre expansion prevents z-fighting. No collision geom is added.
        Materials are intentionally absent so both black and white keys turn green.
        """
        for i in np.flatnonzero(self.active(data)):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('Viewer scene has no room for key activation overlay')
            if self.color_mesh(scene, i, GREEN):
                continue
            g = self.geom_ids[i]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                               self.sizes[i]+.00002, data.geom_xpos[g], data.geom_xmat[g], GREEN)
            scene.ngeom += 1

    def color_mesh(self, scene, index, color):
        """Paint the original tapered surface; bounding boxes are not overlays."""
        if not self.mesh_keys[index]:
            return False
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == self.geom_ids[index]:
                geom.rgba[:] = color
                geom.matid = -1
                geom.label = ''
        return True

    def color_render_scene(self, scene, data):
        active = set(self.geom_ids[self.active(data)].tolist())
        for g in scene.geoms[:scene.ngeom]:
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in active:
                g.rgba[:] = GREEN
                g.matid = -1
