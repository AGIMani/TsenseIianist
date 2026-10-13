#!/usr/bin/env python3
"""Trim the identified base-link side tab in a copied URDF mesh package.

Only the 0.2 mm sheet extending to local Y=86.04 mm is cut, at Y=48 mm
inside the surrounding shell. Other components are retained. No joint or
frame is changed. Run sync_project_urdf.py on the resulting package.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from scripts.sync_project_urdf import read_triangles, write_binary_stl, maybe_write_obj


def trim(path):
    before = path.read_bytes()
    mesh = trimesh.load_mesh(path)
    groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    candidates = [ids for ids in groups if mesh.triangles[ids, :, 1].max() > .080]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one connected tab sheet; refusing a guessed cut')
    selected = candidates[0]
    sheet = mesh.submesh([selected], append=True)
    if not sheet.is_watertight or not np.allclose(sheet.bounds[:, 0], [.0308, .031], atol=1e-7):
        raise ValueError('Unexpected sheet geometry')
    faces = []
    for triangle in sheet.triangles:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            if a[1] <= .048:
                polygon.append(a)
            if (a[1] <= .048) != (b[1] <= .048):
                point = a + (.048-a[1])/(b[1]-a[1])*(b-a)
                point[1] = .048
                polygon.append(point)
        for i in range(1, len(polygon)-1):
            faces.append([polygon[0], polygon[i], polygon[i+1]])
    faces = np.asarray(faces)
    clipped = trimesh.Trimesh(vertices=faces.reshape(-1, 3),
                              faces=np.arange(faces.size//3).reshape(-1, 3))
    clipped.merge_vertices()
    # The cut is a single convex rectangular cross-section. Close its boundary
    # with a centre fan, preserving every boundary subdivision from the slice.
    edges, counts = np.unique(np.sort(clipped.edges, axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    points = clipped.vertices[np.unique(boundary)]
    if len(boundary) < 4 or not np.allclose(points[:, 1], .048):
        raise ValueError('Unexpected cut boundary')
    centre = points.mean(0)
    cap = []
    for a, b in boundary:
        triangle = np.array([clipped.vertices[a], clipped.vertices[b], centre])
        if np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])[1] < 0:
            triangle = triangle[[1, 0, 2]]
        cap.append(triangle)
    triangles = np.concatenate([clipped.triangles, cap])
    closed = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                             faces=np.arange(triangles.size//3).reshape(-1, 3))
    if not closed.is_watertight or not closed.is_winding_consistent:
        raise ValueError('Trim did not produce a closed, consistently oriented sheet')
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[selected] = False
    result = np.concatenate([mesh.triangles[keep], closed.triangles])
    write_binary_stl(path, result, b'TsenseI base link: side tab trimmed at local Y=48 mm')
    maybe_write_obj(path)  # URDF loader uses this for the large base STL.
    return dict(mesh=str(path), cut_plane_y_m=.048,
                source_sha256=hashlib.sha256(before).hexdigest(),
                result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_bounds_m=mesh.bounds.tolist(),
                result_bounds_m=[result.min(axis=(0, 1)).tolist(), result.max(axis=(0, 1)).tolist()],
                original_faces=len(mesh.faces), result_faces=len(result),
                untouched_faces=int(keep.sum()), trimmed_sheet_closed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = trim(args.mesh)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
