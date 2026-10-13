"""Display-only collision hull wireframes; no added physical geoms or contacts."""
import mujoco
import numpy as np

from tsensei_v3_2_pianist.models.hands.tsensei_hand import FINGERTIP_SITE_GROUP


def configure_options(options, show_fingertips=False, transparent_model=False, show_hand_collisions=False):
    options.geomgroup[3] = int(show_hand_collisions)
    if show_hand_collisions:
        # Native mesh contacts use the compiled convex hull, not CAD triangles.
        options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.sitegroup[FINGERTIP_SITE_GROUP] = int(show_fingertips)
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_model


def color_hand_collision_geoms(model, rgba=(1.0, 0.22, 0.08, 0.48)):
    """Color hand collision geoms for an inspection render only."""
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if model.geom_group[geom_id] == 3 and (name.startswith("left_/") or name.startswith("right_/")):
            model.geom_rgba[geom_id] = rgba
        elif (name.startswith("left_/") or name.startswith("right_/")) and "_visual_" in name:
            model.geom_rgba[geom_id, 3] = .12


class PadCollisionWireframe:
    """Convex hull edges of only the ten distal collision meshes.

    Uses MuJoCo's compiled hull faces and vertices, then the geom world transform. Full distal
    link hulls are shown: the pad is not a separate collision geom in the URDF.
    """

    def __init__(self, physics, hands):
        self._physics = physics
        model = physics.model.ptr
        self._meshes = []
        for hand in hands:
            for body in hand.fingertip_bodies:
                for geom in body.geom:
                    if geom.group != 3:
                        continue
                    geom_id = model.geom(geom.full_identifier).id
                    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                        continue
                    mesh_id = int(model.geom_dataid[geom_id])
                    start = model.mesh_vertadr[mesh_id]
                    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]].astype(float)
                    graph_start = int(model.mesh_graphadr[mesh_id])
                    if graph_start < 0:
                        # Newer MuJoCo may omit the optional collision graph.
                        # Reconstruct only the display hull from compiled vertices.
                        from scipy.spatial import ConvexHull
                        faces = ConvexHull(vertices).simplices
                        nface = len(faces)
                    else:
                        graph = model.mesh_graph
                        nvert, nface = map(int, graph[graph_start:graph_start + 2])
                        # MuJoCo convex-hull graph layout.
                        face_start = graph_start + 2 + 3 * nvert + 3 * nface
                        faces = graph[face_start:face_start + 3 * nface].reshape(-1, 3)
                    if len(faces) != nface or faces.min() < 0 or faces.max() >= len(vertices):
                        raise ValueError(f"Invalid compiled hull layout for {geom.full_identifier}")
                    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
                    edges = np.unique(np.sort(edges, axis=1), axis=0)
                    self._meshes.append((geom_id, vertices, edges))
        self.edge_count = sum(len(edges) for _, _, edges in self._meshes)
        self._local_lines = []
        for geom_id, vertices, edges in self._meshes:
            positions, matrices = [], []
            template = mujoco.MjvGeom()
            for a, b in edges:
                mujoco.mjv_connector(template, mujoco.mjtGeom.mjGEOM_LINE, 1.0, vertices[a], vertices[b])
                positions.append(template.pos.copy())
                matrices.append(template.mat.copy())
            self._local_lines.append((geom_id, np.asarray(positions), np.asarray(matrices)))
        self._cached_scene = None
        self._cached_slots = None
        self._last_pose = None

    @property
    def geom_names(self):
        return [self._physics.model.id2name(geom_id, "geom") for geom_id, _, _ in self._meshes]

    def append_to_scene(self, scene):
        """Append lines to an updated render scene or an empty viewer.user_scn."""
        if scene.ngeom + self.edge_count > scene.maxgeom:
            raise ValueError(f"Wireframe needs {self.edge_count} extra scene geoms; capacity {scene.maxgeom}")
        data = self._physics.data.ptr
        for geom_id, vertices, edges in self._meshes:
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            world = vertices @ rotation.T + data.geom_xpos[geom_id]
            for a, b in edges:
                line = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(line, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                                   np.zeros(3), np.eye(3).ravel(), np.array((.05, .95, .25, 1.)))
                line.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                line.objid = -1
                line.category = mujoco.mjtCatBit.mjCAT_DECOR
                line.segid = -1
                mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_LINE, 1.0, world[a], world[b])
                scene.ngeom += 1

    def update_scene(self, scene):
        """Update the leading user_scn slots, preserving overlays after the lines."""
        data = self._physics.data.ptr
        ids = [entry[0] for entry in self._meshes]
        pose = np.concatenate((data.geom_xpos[ids].ravel(), data.geom_xmat[ids].ravel()))
        if self._cached_scene is not scene or scene.ngeom < self.edge_count:
            scene.ngeom = 0
            self.append_to_scene(scene)
            self._cached_scene = scene
            self._cached_slots = tuple(scene.geoms[:self.edge_count])
        elif not np.array_equal(pose, self._last_pose):
            offset = 0
            for geom_id, positions, matrices in self._local_lines:
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                world_positions = positions @ rotation.T + data.geom_xpos[geom_id]
                world_matrices = rotation @ matrices
                for line, position, matrix in zip(self._cached_slots[offset:offset + len(positions)],
                                                  world_positions, world_matrices):
                    line.pos[:] = position
                    line.mat[:] = matrix
                offset += len(positions)
        self._last_pose = pose
