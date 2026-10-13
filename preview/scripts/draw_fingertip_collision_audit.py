#!/usr/bin/env python3
"""CPU-only A3.2 terminal mesh/capsule comparison with reproducible dimensions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.models.hands.fingertip_geometry import read_stl_vertices


def mesh_vertices(assets, side, name):
    urdf = assets / 'urdf' / f'TsenseI_{side}.urdf'
    collision = ET.parse(urdf).getroot().find(f"link[@name='{name}']/collision")
    mesh = collision.find('geometry/mesh')
    path = (urdf.parent / mesh.get('filename')).resolve()
    vertices = read_stl_vertices(path) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
    origin = collision.find('origin')
    if origin is not None:
        vertices = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).apply(vertices)
        vertices += np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return vertices, path


def outline(entry, axis):
    a, b = np.asarray(entry['fromto']).reshape(2, 3)[:, [axis, 2]]
    theta = np.linspace(0, 2*np.pi, 160, endpoint=False)
    circle = entry['size'][0] * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    points = np.concatenate((circle+a, circle+b)) * 1000
    hull = ConvexHull(points).vertices
    return points[np.r_[hull, hull[0]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    assets = ROOT / 'tsensei_v3_2_pianist/assets'
    before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
    fig, axes = plt.subplots(2, 5, figsize=(17, 9), facecolor='#f8fafc')
    rows = []
    for side in ('left', 'right'):
        for column, finger in enumerate(('thumb', 'index', 'mid', 'ring', 'pinky')):
            name = f'{side}_hand_{finger}_{"PIP" if finger == "thumb" else "DIP"}_link'
            old, new = [config['hands'][side][name] for config in (before, after)]
            vertices, path = mesh_vertices(assets, side, name)
            row = dict(side=side, link=name, source_mesh=str(path.relative_to(ROOT)),
                       mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       mesh_span_mm=(np.ptp(vertices, axis=0)*1000).tolist(),
                       old_diameter_mm=old['size'][0]*2000,
                       new_diameter_mm=new['size'][0]*2000,
                       tip_shift_m=(np.asarray(new['preserved_fingertip'])-
                                    old['preserved_fingertip']).tolist())
            for label, entry in (('old', old), ('new', new)):
                a, b = np.asarray(entry['fromto']).reshape(2, 3)
                direction = b-a
                phase = np.clip((vertices-a)@direction/(direction@direction), 0, 1)
                distance = np.linalg.norm(vertices-a-phase[:, None]*direction, axis=1)
                row[label+'_maximum_vertex_outside_mm'] = max(0., float(distance.max()-entry['size'][0]))*1000
            rows.append(row)
            if side != 'left':
                continue
            for r, axis in enumerate((0, 1)):
                ax = axes[r, column]
                ax.scatter(vertices[:, axis]*1000, vertices[:, 2]*1000,
                           s=1, c='#65758b', alpha=.35, rasterized=True)
                for entry, color, style in ((old, '#db584a', '--'), (new, '#008b74', '-')):
                    points = outline(entry, axis)
                    ax.plot(points[:, 0], points[:, 1], style, color=color, linewidth=1.8)
                ax.set_title(f'{finger.title()}  |  {"XZ" if axis == 0 else "YZ"}\n'
                             f'{row["old_diameter_mm"]:.2f} -> {row["new_diameter_mm"]:.2f} mm diameter',
                             fontsize=11)
                ax.set_aspect('equal'); ax.grid(alpha=.2)
                ax.set_xlabel(f'Local {"X" if axis == 0 else "Y"} (mm)'); ax.set_ylabel('Local Z (mm)')
    fig.suptitle('A3.2 with ID: actual terminal shell vs. collision capsules', fontsize=19, y=.985)
    fig.legend(handles=[Line2D([], [], marker='.', linestyle='', color='#65758b', label='Actual URDF mesh vertices'),
                        Line2D([], [], color='#db584a', linestyle='--', label='Before: radius from top 1 mm band'),
                        Line2D([], [], color='#008b74', label='Re-fit: whole-mesh thickness/axis, tip anchored')],
               loc='lower center', bbox_to_anchor=(.5, .04), ncol=3, frameon=False, fontsize=12)
    fig.text(.5, .015, 'Left hand shown; right hand checked separately. Reward fingertip landmarks are unchanged. '
             'Capsules remain approximations, not complete outer-shell envelopes.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .955))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=165)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        rows=rows, before_sha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
        after_sha256=hashlib.sha256(args.after.read_bytes()).hexdigest(),
        note='Vertex deviation is a geometric diagnostic, not a surface-area or collision-frequency measurement'),
        indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
