#!/usr/bin/env python3
"""CPU-only frame figures from current URDFs and the actual preview IK reset.

J1..J7 are moving child-link frames after the current joint rotation (not DH).
F inherits Link7 axes and is translated to its provisional y=-95 mm flange.
H is the normalized articulated hand base_link, not its inertial COM.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ['MUJOCO_GL'] = 'disable'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tsensei_v3_2_pianist import suite
from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview, ASSETS as ARM_ASSETS
from tsensei_v3_2_pianist.preview_connector import ConnectorPreview, tool_transforms, DEFAULT_CONFIG

COLORS = ['#db3535', '#159648', '#246ce0']
HALO = [effects.withStroke(linewidth=3, foreground='white')]


def matrix(position, rotation):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, position
    return t


def body_frame(data, body_id):
    return matrix(data.xpos[body_id], data.xmat[body_id].reshape(3, 3))


def origin(element):
    node = element.find('origin')
    if node is None:
        return np.eye(4)
    return matrix(np.fromstring(node.get('xyz', '0 0 0'), sep=' '),
                  Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix())


def mesh_geometry(model, data, side, hand_prefix):
    """Read compiled visual triangles; mesh principal-axis transforms are in geom_xmat."""
    meshes = []
    for gid in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[gid]).name
        if model.geom_group[gid] != 1 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (body_name.startswith(f'tj_{side}_') or body_name.startswith(hand_prefix)
                or body_name == f'hand_connector_{side}'):
            continue
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start+count].astype(float)
        start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        faces = model.mesh_face[start:start+count]
        # Cluster vertices for illustration only; the compiled model stays intact.
        step = .001 if body_name.startswith('tj_') else .0006
        _, first, inverse = np.unique(np.round(vertices/step).astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        faces = inverse[faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        vertices = vertices[first] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        triangles = vertices[faces]*1000
        color = '#d5a443' if body_name.startswith('hand_connector') else '#afb9c5'
        meshes.append((triangles, color, body_name))
    if not any(name.startswith(hand_prefix) for _, _, name in meshes):
        raise ValueError('Hand visual geometry missing from figure')
    return meshes


def draw_meshes(ax, meshes, centre, radius, alpha=.32):
    low, high = centre-radius, centre+radius
    for triangles, color, _ in meshes:
        keep = np.all(triangles.max(1) >= low, axis=1) & np.all(triangles.min(1) <= high, axis=1)
        tri = triangles[keep]
        if not len(tri):
            continue
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
        shade = .68 + .32*abs(normals @ np.array([.35, -.4, .847]))
        colors = np.zeros((len(tri), 4))
        colors[:, :3] = shade[:, None]*matplotlib.colors.to_rgb(color)
        colors[:, 3] = alpha
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, linewidths=0,
                                            rasterized=True, zorder=1))


def draw_frame(ax, transform, label, length, label_origin=True):
    point = transform[:3, 3]*1000
    rotation = transform[:3, :3]
    ax.scatter(*point, color='#233348', s=14, depthshade=False, zorder=20)
    for i, color in enumerate(COLORS):
        direction = rotation[:, i]*length
        ax.quiver(*point, *direction, color=color, linewidth=2.4, arrow_length_ratio=.19, zorder=30)
        end = point+1.16*direction
        ax.text(*end, 'XYZ'[i], color=color, fontsize=11, fontweight='bold', zorder=40, path_effects=HALO)
    if label_origin:
        ax.text(*point, label, fontsize=11, color='#192d43', zorder=50, path_effects=HALO)


def camera_for(rotations):
    # Keep every axis visible rather than looking straight down an axis.
    best = None
    for elevation in (20, 30, 40):
        for azimuth in range(-160, 180, 20):
            e, a = np.deg2rad([elevation, azimuth])
            view = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
            scores = [np.min(np.sqrt(np.maximum(0, 1-(view@r)**2))) for r in rotations]
            score = min(scores) - .0001*abs(azimuth+40)
            if best is None or score > best[0]:
                best = (score, elevation, azimuth)
    return best[1:]


def setup(ax, centre, radius, camera):
    ax.set(xlim=(centre[0]-radius, centre[0]+radius), ylim=(centre[1]-radius, centre[1]+radius),
           zlim=(centre[2]-radius, centre[2]+radius))
    ax.set_box_aspect([1, 1, 1], zoom=1.18)
    ax.set_proj_type('ortho')
    ax.view_init(elev=camera[0], azim=camera[1])
    ax.set_axis_off()


def page_header(fig, title, subtitle):
    fig.suptitle(title, fontsize=22, y=.975, color='#182e46', fontweight='bold')
    fig.text(.5, .93, subtitle, ha='center', fontsize=12, color='#4a5f74')
    for x, color, text in zip([.37, .47, .57], COLORS, ['X axis', 'Y axis', 'Z axis']):
        fig.text(x, .025, text, color=color, fontsize=13, fontweight='bold')


def joint_page(side, frames, meshes, q, output):
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    page_header(fig, f'{side.capitalize()} M6 arm | J1-J7 coordinate frames',
                'Current source reset + IK pose  |  Moving URDF child-link frames  |  +Z is the positive joint axis')
    axes = [fig.add_subplot(2, 4, i+1, projection='3d', computed_zorder=False) for i in range(8)]
    points = np.array([frames[f'J{i}'][:3, 3]*1000 for i in range(1, 8)])
    centre = (points.min(0)+points.max(0))/2
    radius = np.ptp(points, axis=0).max()/2+110
    normal = np.cross(points[2]-points[0], points[4]-points[0])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    camera = (np.rad2deg(np.arcsin(normal[2])), np.rad2deg(np.arctan2(normal[1], normal[0])))
    setup(axes[0], centre, radius, camera)
    draw_meshes(axes[0], meshes, centre, radius, alpha=.6)
    axes[0].plot(*points.T, color='#41556c', linewidth=1.2, zorder=20)
    for label, i, label_y in [('J1 / J2', 1, .87), ('J3 / J4', 3, .60), ('J5 / J6 / J7', 5, .22)]:
        p = frames[f'J{i}'][:3, 3]*1000
        axes[0].scatter(*p, color='#233348', s=18, zorder=25)
        x, y, _ = proj3d.proj_transform(*p, axes[0].get_proj())
        axes[0].annotate(label, xy=(x, y), xycoords='data', xytext=(.02, label_y),
                         textcoords='axes fraction', fontsize=10, zorder=40, path_effects=HALO,
                         arrowprops={'arrowstyle':'-', 'color':'#4b6074', 'lw':.8})
    axes[0].set_title('Joint origins (shared origins are intentional)', fontsize=11)
    cameras = {}
    for i, ax in enumerate(axes[1:], 1):
        t = frames[f'J{i}']
        p = t[:3, 3]*1000
        camera = camera_for([t[:3, :3]])
        cameras[f'J{i}'] = list(camera)
        setup(ax, p, 105, camera)
        draw_meshes(ax, meshes, p, 105)
        draw_frame(ax, t, f'J{i}', 64, label_origin=False)
        ax.set_title(f'J{i}   |   q = {np.rad2deg(q[i-1]):+.2f} deg', fontsize=14, color='#182e46')
        ax.text2D(.5, .04, '50 mm', transform=ax.transAxes, ha='center', color='#738293', fontsize=9)
        # Explicit scale line in the page plane, independent of axis orientation.
        ax.plot([p[0]-25, p[0]+25], [p[1], p[1]], [p[2]-85, p[2]-85], color='#738293', lw=1)
    fig.subplots_adjust(left=.015, right=.985, top=.875, bottom=.07, wspace=.02, hspace=.08)
    fig.savefig(output/f'joints_{side}.png', dpi=180)
    fig.savefig(output/f'joints_{side}.pdf', dpi=180)
    plt.close(fig)
    return cameras


def wrist_page(all_frames, meshes, output):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    page_header(fig, 'Flange F and hand H | installed A3.2_tianji adapter',
                'F: Link7 axes at local [0, -95, 0] mm  |  H: normalized hand base_link  |  Axes shown at true origins')
    cameras = {}
    for i, side in enumerate(('left', 'right')):
        f, h = all_frames[side]['F'], all_frames[side]['H']
        centre = (f[:3, 3]+h[:3, 3])*500
        ax = fig.add_subplot(1, 2, i+1, projection='3d', computed_zorder=False)
        cam = camera_for([f[:3, :3], h[:3, :3]])
        cameras[side] = list(cam)
        setup(ax, centre, 95, cam)
        draw_meshes(ax, meshes[side], centre, 95, alpha=.35)
        draw_frame(ax, f, 'F (flange)', 32)
        draw_frame(ax, h, 'H (hand)', 32)
        points = np.stack([f[:3, 3], h[:3, 3]])*1000
        ax.plot(*points.T, color='#596d82', linestyle='--', linewidth=1.1, zorder=20)
        relative = np.linalg.inv(f) @ h
        xyz = relative[:3, 3]*1000
        ax.set_title(f'{side.capitalize()} hand\nF to H: [{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm; Rx(+90 deg)',
                     fontsize=14, color='#182e46')
    fig.text(.5, .09, 'Gold: actual adapter meshes. F is a project flange convention; hardware frame calibration is still pending.',
             ha='center', fontsize=11, color='#53687b')
    fig.subplots_adjust(left=.015, right=.985, top=.82, bottom=.14, wspace=.04)
    fig.savefig(output/'flange_hand.png', dpi=180)
    fig.savefig(output/'flange_hand.pdf', dpi=180)
    plt.close(fig)
    return cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'preview/outputs/arm_frames_20260930')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = suite.load(task_kwargs={'gravity_compensation': True, 'control_timestep': .05, 'physics_timestep': .005})
    try:
        connector = ConnectorPreview(env.task.root_entity.mjcf_model)
        arms = ArmPreview(env.task.root_entity.mjcf_model, tool_overrides=tool_transforms(connector.config))
        env.reset()
        hands = (env.task.right_hand, env.task.left_hand)
        arms.bind(env.physics, hands)
        connector.bind(env.physics, hands, arms)
        frames, meshes, result = {}, {}, {}
        for side, suffix in [('left', 'L'), ('right', 'R')]:
            frames[side] = {}
            source = ET.parse(ARM_ASSETS/side/'arm.urdf').getroot()
            fk = body_frame(arms.data, arms.model.body(f'{side}_Base_{suffix}').id)
            errors = []
            q = arms.data.qpos[arms.sides[side]['qadr']].copy()
            for i, angle in enumerate(q, 1):
                name = f'Joint{i}_{suffix}'
                joint = source.find(f"joint[@name='{name}']")
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                np.testing.assert_allclose(axis, [0, 0, 1])
                fk = fk @ origin(joint) @ matrix(np.zeros(3), Rotation.from_rotvec(axis*angle).as_matrix())
                actual = body_frame(arms.data, arms.model.body(f'{side}_Link{i}_{suffix}').id)
                errors.append(float(np.max(abs(fk-actual))))
                np.testing.assert_allclose(fk, actual, atol=1e-10)
                frames[side][f'J{i}'] = actual
                jid = arms.model.joint(f'{side}_{name}').id
                np.testing.assert_allclose(actual[:3, :3][:, 2], arms.data.xaxis[jid], atol=1e-10)
            frames[side]['F'] = frames[side]['J7'] @ matrix(np.array([0., -.095, 0.]), np.eye(3))
            frames[side]['H'] = body_frame(env.physics.data.ptr, arms.hand_ids[side])
            expected_hand = frames[side]['J7'] @ tool_transforms(connector.config)[side]
            np.testing.assert_allclose(expected_hand, frames[side]['H'], atol=2e-8)
            hand_prefix = next(h.name for h in hands if h.hand_side.name.lower() == side) + '/'
            meshes[side] = mesh_geometry(env.physics.model.ptr, env.physics.data.ptr, side, hand_prefix)
            cameras = joint_page(side, frames[side], meshes[side], q, args.output)
            result[side] = {'q_rad': q.tolist(), 'frames_world_from_local': {k:v.tolist() for k,v in frames[side].items()},
                            'visual_body_names': sorted({name for _, _, name in meshes[side]}),
                            'independent_urdf_fk_max_error': max(errors), 'camera_elevation_azimuth_deg': cameras,
                            'flange_from_hand': (np.linalg.inv(frames[side]['F'])@frames[side]['H']).tolist()}
        wrist_cameras = wrist_page(frames, meshes, args.output)
        result['definitions'] = {'J': 'moving URDF child-link frame after q rotation; +Z is positive joint axis, not DH',
                                 'F': 'Link7 frame translated by local [0,-0.095,0] m, axes unchanged; provisional project flange',
                                 'H': 'current normalized hand base_link, not CAD root or inertial COM',
                                 'pose': 'current source reset; static bounded IK; no policy or dynamics rollout'}
        result['wrist_cameras'] = wrist_cameras
        inputs = [DEFAULT_CONFIG, ARM_ASSETS/'preview_mount.json', ARM_ASSETS/'left/arm.urdf', ARM_ASSETS/'right/arm.urdf',
                  ROOT/'tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py',
                  *[ROOT/f'tsensei_v3_2_pianist/assets/urdf/TsenseI_{s}.urdf' for s in ('left','right')]]
        result['input_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        (args.output/'frames.json').write_text(json.dumps(result, indent=2)+'\n')
        print(args.output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
