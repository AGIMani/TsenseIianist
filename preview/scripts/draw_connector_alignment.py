#!/usr/bin/env python3
"""CPU-only geometry comparison of the STEP hand and proposed URDF mounting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'preview/outputs/connector_alignment'
ASSETS = ROOT / 'preview/assets/arm_hand_connect'


def live_urdf_meshes():
    """Read the current URDF at zero joint angles, without a cached hand mesh."""
    path = ROOT / 'tsensei_v3_2_pianist/assets/urdf/TsenseI_left.urdf'
    root = ET.parse(path).getroot()
    def origin(node):
        t = np.eye(4)
        if node is not None:
            t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
            t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        return t
    transforms = {'base_link': np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        previous = len(pending)
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent in transforms:
                transforms[joint.find('child').get('link')] = transforms[parent] @ origin(joint.find('origin'))
                pending.remove(joint)
        if len(pending) == previous:
            raise ValueError('URDF joint tree is disconnected')
    meshes = {}
    for link in root.findall('link'):
        name = link.get('name')
        parts = []
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            part = trimesh.load(path.parent / mesh.get('filename'))
            part.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
            part.apply_transform(transforms[name] @ origin(visual.find('origin')))
            parts.append(part)
        meshes[name] = trimesh.util.concatenate(parts)
    return meshes


def simplified(mesh, step=.0008):
    """Vertex clustering for display only; original geometry is never changed."""
    _, first, inverse = np.unique(np.round(mesh.vertices / step).astype(np.int64),
                                 axis=0, return_index=True, return_inverse=True)
    faces = inverse[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    faces = np.unique(np.sort(faces, axis=1), axis=0)
    return trimesh.Trimesh(vertices=mesh.vertices[first], faces=faces, process=False)


def mesh_plot(ax, mesh, color, step=.0008):
    mesh = simplified(mesh, step)
    shade = .55 + .45 * np.abs(mesh.face_normals @ np.array([.36, -.48, .80]))
    rgb = matplotlib.colors.to_rgb(color)
    faces = Poly3DCollection(mesh.triangles * 1000, facecolors=shade[:, None] * rgb,
                             edgecolors='none', linewidths=0, rasterized=True)
    ax.add_collection3d(faces)


def mount_triangles(mesh, z):
    ids = (np.abs(mesh.face_normals[:, 2]) > .99999) & (np.abs(mesh.triangles_center[:, 2] - z) < 1e-6)
    return mesh.triangles[ids]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--align-hole', action='store_true', help='Align the central apertures instead of outside bounding boxes')
    parser.add_argument('--current-urdf', action='store_true', help='Draw the replaced mounting block from live URDF mesh references')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cad = trimesh.load(ASSETS / 'part_07.stl')
    if args.current_urdf:
        args.align_hole = True
        meshes = live_urdf_meshes()
        current = trimesh.util.concatenate(list(meshes.values()))
        base = meshes['Left_hand_palm_link']
    else:
        current = trimesh.load(OUT / 'urdf_zero_hand.stl')
        base = trimesh.load(ROOT / 'tsensei_v3_2_pianist/assets/meshes_left/base_link.STL')
        base.apply_scale(.001)
        base.apply_translation([.0147, -.0015, 0])
    original_face = mount_triangles(base, -.004)
    cad_face = mount_triangles(cad, -.022)
    bounds = lambda a: np.array([a.reshape(-1, 3).min(0), a.reshape(-1, 3).max(0)])
    source_center, target_center = bounds(original_face).mean(0), bounds(cad_face).mean(0)
    if args.align_hole:
        # Polygon-area centroids of the largest interior boundary of each
        # mounting face. Recorded from the STEP/URDF face-edge audit.
        source_center = np.array([.01675, .004, -.004])
        target_center = np.array([-.004, -.005, -.022])
    rotation = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    current.apply_transform(transform)
    proposed_face = original_face @ rotation.T + transform[:3, 3]
    candidate = {'status': 'central aperture alignment requested by user' if args.align_hole else 'candidate awaiting user inspection',
                 'assembly_from_left_hand': transform.tolist(),
                 'method': ('match central aperture area-centroids and opposing mounting normals' if args.align_hole else 'match mounting-plane bounding-box centres and opposing normals; clocking follows finger/thumbnail side'),
                 'source_alignment_point_m': source_center.tolist(),
                 'target_alignment_point_m': target_center.tolist(),
                 'step_mount_face_size_mm': np.ptp(cad_face.reshape(-1, 3), axis=0).tolist(),
                 'urdf_mount_face_size_mm': np.ptp(proposed_face.reshape(-1, 3), axis=0).tolist(),
                 'limitations': ['STEP reference hand differs from current URDF', 'bolt-hole fit and flange clocking not calibrated']}
    for key in ('step_mount_face_size_mm', 'urdf_mount_face_size_mm'):
        candidate[key] = (np.asarray(candidate[key]) * 1000).tolist()
    prefix = 'hole_aligned' if args.align_hole else 'candidate'
    if args.current_urdf:
        prefix = 'urdf_replaced'
        candidate['status'] = 'STEP mounting block installed in current URDF'
        candidate['limitations'] = ['hardware mounting and revised inertias remain uncalibrated']
    (OUT / f'{prefix}_mount.json').write_text(json.dumps(candidate, indent=2) + '\n')
    fig = plt.figure(figsize=(19.5, 10.8), facecolor='#f5f7fa')
    fig.suptitle('Connector installation — geometry review', fontsize=23, x=.035, ha='left', y=.96, fontweight='bold', color='#14263e')
    fig.text(.035, .917, 'Actual STEP and URDF meshes  |  ' + ('Central aperture centres aligned' if args.align_hole else 'Candidate: mounting-face centres and normals aligned'), fontsize=13, color='#52657b')
    a = fig.add_axes([.025, .18, .315, .70], projection='3d')
    b = fig.add_axes([.342, .18, .315, .70], projection='3d')
    c = fig.add_axes([.715, .34, .26, .42])
    for ax, hand, title, color in ((a, cad, 'A  STEP reference assembly', '#aeb7c1'),
                                   (b, current, 'B  Current URDF — ' + ('STEP block installed' if args.current_urdf else 'proposed mount'), '#4389c7')):
        ax.set_facecolor('#f5f7fa')
        mesh_plot(ax, hand, color)
        for index in range(1, 7):
            mesh_plot(ax, trimesh.load(ASSETS / f'part_{index:02d}.stl'), '#e5a942', step=.00035)
        ax.set(xlim=(-75, 75), ylim=(-65, 65), zlim=(-235, 15))
        ax.set_box_aspect((150, 130, 250))
        ax.view_init(elev=14, azim=-65)
        ax.set_axis_off()
        ax.set_title(title, fontsize=15, loc='center', pad=12, color='#14263e')
    c.add_collection(PolyCollection(cad_face[:, :, :2] * 1000, facecolors='#c3c9d0', edgecolors='none'))
    c.add_collection(PolyCollection(proposed_face[:, :, :2] * 1000, facecolors='#4389c755', edgecolors='none'))
    center = target_center[:2] * 1000
    for triangles, color, linestyle in ((cad_face, '#6b7280', '-'), (proposed_face, '#1967a3', '--')):
        low, high = bounds(triangles)[:, :2] * 1000
        c.plot([low[0], high[0], high[0], low[0], low[0]], [low[1], low[1], high[1], high[1], low[1]], color=color, ls=linestyle, lw=2)
    c.plot(*center, marker='+', color='#ba3434', markersize=15, markeredgewidth=2)
    c.set(xlim=(center[0]-32, center[0]+32), ylim=(center[1]-32, center[1]+32), xlabel='Assembly X (mm)', ylabel='Assembly Y (mm)')
    c.set_aspect('equal')
    c.grid(alpha=.16)
    c.set_title('C  Mounting plane, viewed from above', fontsize=14, pad=17, color='#14263e')
    c.legend(handles=[Patch(color='#c3c9d0', label='STEP hand: 35.6 × 24.6 mm'),
                      Patch(color='#4389c7', label='Current hand: ' + ('35.6' if args.current_urdf else '45.6') + ' × 24.6 mm')],
             loc='upper center', bbox_to_anchor=(.5, -.17), frameon=False, fontsize=11)
    fig.text(.04, .165, 'Gold: your six connector parts', color='#966312', fontsize=13)
    fig.text(.04, .12, ('Mounting face and central aperture now match the STEP reference; the finger mechanism retains its URDF geometry.' if args.current_urdf else 'Central aperture alignment shifts the URDF by 3.25 mm from the previous face-centre alignment.' if args.align_hole else 'The current hand is 10.0 mm wider at this mounting plane (5.0 mm per side after centring).'), fontsize=14, color='#14263e')
    fig.text(.04, .078, 'For review only: screw-hole fit, flange clocking, and hardware dimensions remain unverified.', fontsize=12, color='#6d7785')
    filename = 'connector_hole_aligned.png' if args.align_hole else 'connector_mount_comparison.png'
    if args.current_urdf:
        filename = 'connector_urdf_replaced.png'
    fig.savefig(OUT / filename, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT / filename)


if __name__ == '__main__':
    main()
