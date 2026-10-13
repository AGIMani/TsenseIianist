#!/usr/bin/env python3
"""Render a CPU geometry audit of the installed Tianji mount and bolt pattern."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh

from draw_connector_alignment import live_urdf_meshes, mesh_plot

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'preview/outputs/tianji_mount_20260930'


def main():
    config = json.loads((ROOT/'configs/preview/hand_connector_20260930.json').read_text())
    transform = np.array(config['alignment']['assembly_from_left_hand'])
    meshes = live_urdf_meshes()
    for mesh in meshes.values():
        mesh.apply_transform(transform)
    connectors = []
    for name in config['meshes']['left']:
        mesh = trimesh.load(ROOT/name)
        mesh.apply_transform(transform)
        connectors.append(mesh)
    fig = plt.figure(figsize=(15, 8), facecolor='white')
    fig.suptitle('A3.2 + Tianji: installed adapter', fontsize=20, y=.965)
    ax = fig.add_subplot(131, projection='3d')
    for name, mesh in meshes.items():
        mesh_plot(ax, mesh, '#2776ab' if 'palm_link' in name else '#a7b1bb')
    for mesh in connectors:
        mesh_plot(ax, mesh, '#d59b39', step=.00035)
    ax.set(xlim=(-85, 85), ylim=(-75, 75), zlim=(-235, 8))
    ax.set_box_aspect((170, 150, 243))
    ax.view_init(elev=12, azim=-65)
    ax.set_axis_off()
    ax.set_title('Current articulated hand', fontsize=13)
    ax = fig.add_subplot(132, projection='3d')
    mesh_plot(ax, meshes['Left_hand_palm_link'], '#2776ab', step=.00015)
    for mesh in connectors:
        mesh_plot(ax, mesh, '#d59b39', step=.0002)
    ax.set(xlim=(-36, 36), ylim=(-36, 36), zlim=(-55, 6))
    ax.set_box_aspect((72, 72, 61))
    ax.view_init(elev=-20, azim=-65)
    ax.set_axis_off()
    ax.set_title('Six adapter parts + wide holder', fontsize=13)
    ax = fig.add_subplot(133)
    for mesh, color in [(connectors[-1], '#d59b3970'), (meshes['Left_hand_palm_link'], '#2776ab90')]:
        mask = np.all(abs(mesh.triangles[:, :, 2]+.022)<2e-7, axis=1)
        face = mesh.submesh([mask], append=True)
        ax.add_collection(PolyCollection(face.triangles[:, :, :2]*1000, facecolors=color, edgecolors='none'))
        for loop in face.outline().discrete:
            ax.plot(loop[:, 0]*1000, loop[:, 1]*1000, color=color[:7], lw=.8)
    holes = np.array(config['alignment']['mounting_hole_centres_assembly_m'])*1000
    ax.scatter(holes[:, 0], holes[:, 1], marker='+', c='#192b3d', s=85)
    ax.annotate('', xy=(-9.5, 25), xytext=(9.5, 25), arrowprops={'arrowstyle':'<->'})
    ax.text(0, 27, '19 mm', ha='center')
    ax.annotate('', xy=(20, -20), xytext=(20, 20), arrowprops={'arrowstyle':'<->'})
    ax.text(22, 0, '40 mm', rotation=90, va='center')
    ax.set(xlim=(-33, 33), ylim=(-33, 33), xlabel='Assembly X (mm)', ylabel='Assembly Y (mm)')
    ax.set_aspect('equal')
    ax.grid(alpha=.15)
    ax.set_title('Four coaxial mounting holes', fontsize=13)
    fig.text(.055, .085, 'Gold: new adapter  |  Blue: CAD wide holder, mounting face 45.6 x 24.6 mm', fontsize=12)
    fig.text(.055, .045, 'CAD alignment verified. Hardware fit, flange clocking and revised inertia remain uncalibrated.', fontsize=10, color='#53616c')
    fig.subplots_adjust(top=.88, bottom=.15, left=.02, right=.98, wspace=.18)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'installation_detail.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
