"""Measure distal endpoints from bundled URDF collision meshes, in link frames."""
from functools import lru_cache
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

LANDMARK_METHOD = "urdf_chain_distal_support_v1"
FINGERTIP_SETBACK_M = 0.003


def distal_direction(root, link_name):
    """Find the outward phalanx direction from the URDF, not a named local axis.

    A terminal hinge's origin connects the preceding joint pivot to this pivot.
    Express that displacement in the terminal link frame and remove its hinge
    axis component (the lateral offset between the two bearing planes). This
    preserves the direction under a rigid change of link coordinates and the
    project's right-hand reflection. It is not a measured contact-pad normal.
    """
    joints = [j for j in root.findall("joint")
              if j.find("child").get("link") == link_name]
    if len(joints) != 1 or joints[0].get("type") not in ("revolute", "continuous"):
        raise ValueError(f"Distal landmark requires one incoming hinge: {link_name}")
    joint = joints[0]
    parent = joint.find("parent").get("link")
    preceding = [j for j in root.findall("joint")
                 if j.find("child").get("link") == parent]
    if len(preceding) != 1 or preceding[0].get("type") not in ("revolute", "continuous"):
        raise ValueError(f"Distal landmark requires a preceding hinge pivot: {link_name}")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"Missing distal joint displacement: {link_name}")
    rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    displacement = rotation.inv().apply(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "))
    axis_element = joint.find("axis")
    axis = np.fromstring("1 0 0" if axis_element is None else axis_element.get("xyz"), sep=" ")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
        raise ValueError(f"Invalid distal hinge axis: {link_name}")
    axis /= np.linalg.norm(axis)
    direction = displacement - axis * np.dot(displacement, axis)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError(f"Cannot infer distal direction from coincident hinge pivots: {link_name}")
    return direction / length


def distal_support_center(vertices, direction):
    """Center the exposed distal face/edge, invariant to interior tessellation.

    All returned points are convex combinations of mesh support vertices. A
    change of link coordinates cannot select the lateral shell as the tip.
    """
    projection = vertices @ direction
    support = vertices[projection >= projection.max() - 1e-9]
    center = support.mean(axis=0)
    if len(support) == 1:
        return center
    _, values, basis = np.linalg.svd(support - center, full_matrices=False)
    if len(values) < 2 or values[1] < 1e-9:
        along = (support - center) @ basis[0]
        return center + basis[0] * ((along.min() + along.max()) / 2)
    plane = (support - center) @ basis[:2].T
    boundary = ConvexHull(plane).vertices
    polygon = plane[boundary]
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    centroid = ((polygon + following) * cross[:, None]).sum(axis=0) / (3 * cross.sum())
    return center + centroid @ basis[:2]


def read_stl_vertices(path):
    """Read the exact float32 coordinates in our binary STL assets, without rounding."""
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError(f"Truncated STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if not count or len(data) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(data, dtype=dtype, count=count, offset=84)
    return np.unique(triangles["vertices"].reshape(-1, 3), axis=0).astype(np.float64)


@lru_cache(maxsize=16)
def measure_distal_tip(urdf_path, link_name):
    """Return a provisional distal-face landmark in the URDF link frame, metres.

    Select the distal support surface along the joint-chain direction. The
    latest A3.2 export uses a different local frame: local +Z points sideways
    along the bearing axis and must not be interpreted as the finger's length.
    TODO(V32-07): Replace with the measured contact-pad center/geometry.
    """
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"Missing URDF link: {link_name}")
    clouds = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            raise ValueError(f"Expected collision mesh for {link_name}")
        source = (urdf_path.parent / mesh.get("filename")).resolve()
        if not source.is_relative_to(urdf_path.parent.parent):
            raise ValueError(f"Collision asset escapes package: {source}")
        vertices = read_stl_vertices(source)
        vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = collision.find("origin")
        if origin is not None:
            rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
            vertices = rotation.apply(vertices) + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        clouds.append(vertices)
    if not clouds:
        raise ValueError(f"No collision mesh for {link_name}")
    vertices = np.unique(np.concatenate(clouds), axis=0)
    point = distal_support_center(vertices, distal_direction(root, link_name))
    return tuple(float(value) for value in point)


@lru_cache(maxsize=16)
def measure_fingertip_target(urdf_path, link_name):
    """Reward/observation point, 3 mm toward the palm along the distal segment.

    The physical support landmark remains unchanged for collision construction
    and coupling orientation. This setback does not identify a pad normal or
    move the mesh; it is attached to the terminal link, not a world-space axis.
    """
    root = ET.parse(urdf_path).getroot()
    point = np.asarray(measure_distal_tip(urdf_path, link_name))
    point = point - FINGERTIP_SETBACK_M * distal_direction(root, link_name)
    return tuple(float(value) for value in point)
