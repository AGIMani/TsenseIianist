#!/usr/bin/env python3
"""CPU-only figure of actual compiled forearm_reward collision geometry.

Uses mesh convex hulls and sampled primitive surfaces, not hand illustrations.
No OpenGL, CUDA, physics rollout, or modification of the source model.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def surface(model, data, gid):
    import mujoco
    import numpy as np
    from scipy.spatial import ConvexHull
    kind = model.geom_type[gid]
    size = model.geom_size[gid]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[gid]
        start = model.mesh_vertadr[mesh]
        points = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        points = np.asarray(list(itertools.product((-1, 1), repeat=3))) * size
    elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                  mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 24, endpoint=False),
                                 np.linspace(0, np.pi, 17))
        points = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta),
                           np.cos(phi)), axis=-1).reshape(-1, 3)
        points *= size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else size[0]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            points[:, 2] += np.where(points[:, 2] >= 0, size[1], -size[1])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
        ring = np.stack((size[0]*np.cos(theta), size[0]*np.sin(theta),
                         np.full_like(theta, size[1])), axis=-1)
        points = np.concatenate((ring, ring * [1, 1, -1]))
    else:
        raise ValueError(f'Unsupported collider {model.geom(gid).name}: {kind}')
    points = points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    return points[ConvexHull(points).simplices] * 1000


def draw(scene, task, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
    from tsensei_v3_2_pianist.forearm_collision import task_collision_groups

    bundle = SceneBundle(scene)
    manifest = TaskBundle(task, bundle).manifest
    if manifest.get('task_profile') != 'tsensei_collision_v1':
        raise ValueError('This figure describes tsensei_collision_v1 only')
    model, data = bundle.model, bundle.reset_cpu()
    groups = task_collision_groups(manifest)
    colors = ('#e36a52', '#318cc0')
    polygons = [[surface(model, data, model.geom(name).id) for name in group]
                for group in groups]
    vertices = np.concatenate([p.reshape(-1, 3) for group in polygons for p in group])
    lo, hi = vertices.min(0)-8, vertices.max(0)+8
    fig = plt.figure(figsize=(14, 9), facecolor='#f5f7fa')
    fig.suptitle('tsensei_collision_v1  |  Cross-hand collision reward',
                 fontsize=21, fontweight='bold', x=.5, y=.97)
    fig.text(.5, .91, 'All physical collision shapes: base shell + palm segment + every finger segment',
             ha='center', fontsize=13, color='#334155')
    for column, (elevation, azimuth, title) in enumerate(
            ((45, -40, 'Oblique view'), (90, 0, 'Top view (world XY)'))):
        ax = fig.add_subplot(1, 2, column+1, projection='3d', computed_zorder=True)
        ax.set_facecolor('#f5f7fa')
        for side, shapes in enumerate(polygons):
            for triangles in shapes:
                ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors[side],
                    linewidth=0, alpha=1, shade=True,
                    zsort='average'))
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
        ax.set_box_aspect(hi-lo)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
        ax.set_title(title, y=.97, fontsize=15, color='#334155')
    fig.legend(handles=[Patch(color=colors[0], label=f'Right hand: {len(groups[0])} colliders'),
                        Patch(color=colors[1], label=f'Left hand: {len(groups[1])} colliders')],
               loc='lower center', bbox_to_anchor=(.5, .22), ncol=2, frameon=False, fontsize=14)
    fig.text(.5, .18, 'Red touches blue anywhere: -0.5     |     No cross-hand contact: 0',
             ha='center', fontsize=17, fontweight='bold', color='#172b45')
    fig.text(.5, .125, '200 Hz contact union within each 20 Hz control step; one penalty per step.\n'
             'Includes palm-palm, finger-finger and finger-palm. Same-hand and hand-key contacts are excluded.',
             ha='center', va='center', fontsize=12, linespacing=1.7, color='#334155')
    fig.text(.5, .055, 'Compiled A3.2 initial pose. Actual collision geometry; primitives are engineering approximations.\n'
             'This is a scope diagram, not evidence of collision-free motion or hardware calibration.',
             ha='center', fontsize=10, color='#64748b')
    fig.subplots_adjust(left=.02, right=.98, bottom=.27, top=.85, wspace=.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(
        task_profile=manifest['task_profile'], scope=manifest['forearm_collision_scope'],
        scene_manifest_sha256=manifest['scene_manifest_sha256'],
        forearm_collision_geoms=groups, counts=list(map(len, groups)),
        geometry='Compiled MuJoCo convex mesh hulls and sampled primitive surfaces',
        pose='scene reset/origin; no rollout', no_contact_reward=0, contact_reward=-.5,
        control_hz=20, physics_hz=200), indent=2)+'\n')
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    draw(args.scene, args.task, args.output)
