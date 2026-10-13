"""Measure distal endpoints from bundled URDF collision meshes, in link frames."""
from functools import lru_cache
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


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

    V3.2 has flat/tied distal extrema rather than V2's unique farthest vertex.
    Use the bounding-box midpoint of the local +Z extreme vertices as an
    explicit, provisional reward landmark, independent of mesh tessellation.
    TODO(V32-07): Replace with the measured contact-pad center/geometry.
    """
    urdf_path = Path(urdf_path).resolve()
    link = ET.parse(urdf_path).getroot().find(f"link[@name='{link_name}']")
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
    distal = vertices[vertices[:, 2] == vertices[:, 2].max()]
    point = (distal.min(axis=0) + distal.max(axis=0)) / 2
    return tuple(float(value) for value in point)
