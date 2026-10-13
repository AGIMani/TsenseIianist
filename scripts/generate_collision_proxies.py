#!/usr/bin/env python3
"""Regenerate explicit V3.2 primitive proposals from bundled collision meshes.

TODO(V32-06): validate contact-pad geometry against measured hardware. The
chosen rounded proxies intentionally omit thin CAD flanges/gearbox corners.
The terminal capsule exactly preserves the existing distal-face landmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.models.hands.fingertip_geometry import read_stl_vertices, measure_distal_tip

ASSETS = ROOT / "tsensei_v3_2_pianist" / "assets"


def inertial_capsule(mesh_path, scale, origin):
    """Match Shadow's ``geom type=capsule mesh=...`` compiler fitting rule."""
    # Like urdf_model, prefer the bundled OBJ conversion for large CAD meshes;
    # native MuJoCo STL decoding rejects files with over 200,000 triangles.
    if mesh_path.with_suffix(".obj").is_file():
        mesh_path = mesh_path.with_suffix(".obj")
    xml = ET.Element("mujoco")
    ET.SubElement(xml, "compiler", angle="radian")
    asset = ET.SubElement(xml, "asset")
    ET.SubElement(asset, "mesh", name="source", file=str(mesh_path), scale=scale)
    world = ET.SubElement(xml, "worldbody")
    attributes = {"type": "capsule", "mesh": "source"}
    if origin is not None:
        attributes["pos"] = origin.get("xyz", "0 0 0")
        quat = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_quat()
        attributes["quat"] = " ".join(map(str, np.roll(quat, 1)))
    ET.SubElement(world, "geom", **attributes)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    return {"type": "capsule", "pos": model.geom_pos[0].tolist(),
            "quat": model.geom_quat[0].tolist(), "size": model.geom_size[0, :2].tolist()}


def terminal_capsule(mesh_path, scale, origin, vertices, tip):
    """Fit whole-link thickness/axis, retaining the measured +Z touch point.

    A rounded mesh's last 1 mm measures a chord near its pole, not the radius
    of a capsule representing the whole terminal link. Use the same MuJoCo
    equivalent-inertia fit as the other links, then anchor/extend its axis to
    preserve the URDF-local Z bounds and the existing fingertip landmark.
    This is an engineering approximation, not a conservative outer envelope.
    """
    fitted = inertial_capsule(mesh_path, scale, origin)
    radius = fitted['size'][0]
    axis = Rotation.from_quat(np.roll(fitted['quat'], -1)).as_matrix()[:, 2]
    if axis[2] < 0:
        axis = -axis
    if axis[2] <= 1e-6 or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Terminal fit must have positive radius and a distal-facing axis')
    b = np.asarray(tip, dtype=float) - [0., 0., radius]
    proximal_z = float(vertices[:, 2].min()) + radius
    length = (b[2] - proximal_z) / axis[2]
    if length <= 0:
        raise ValueError('Terminal capsule cannot preserve both local-Z bounds')
    a = b - length * axis
    return dict(type='capsule', fromto=np.concatenate((a, b)).tolist(), size=[radius],
                preserved_fingertip=np.asarray(tip).tolist(),
                terminal_fit_method='whole_mesh_inertia_tip_anchored_v1',
                unconstrained_inertial_fit=fitted,
                preserved_local_z_bounds_m=[float(vertices[:, 2].min()), float(vertices[:, 2].max())],
                reason='Whole terminal mesh equivalent-inertia radius and axis; '
                       'axis endpoints adjusted to retain measured fingertip and local-Z bounds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ASSETS / 'collision_proxies.json')
    parser.add_argument('--terminal-only', action='store_true',
                        help='Replace only ten terminal colliders; preserve every other entry verbatim')
    args = parser.parse_args()
    result = {
        "schema": 1, "profile": "shadow_primitive_v1",
        "units": "metres; body-local URDF frame; MuJoCo quaternion wxyz",
        "reference": "robopianist-rl/run.sh enables primitive fingertips; Shadow uses rounded finger primitives and separate visual meshes",
        "derivation": "scripts/generate_collision_proxies.py; MuJoCo " + mujoco.__version__ + "; V3.2 geometry only; does not copy Shadow dimensions or inertias",
        "limitations": [
            "TODO(V32-06): engineering collision approximation, not hardware calibration",
            "Palm housing retains original mesh because a single box fills the thumb recess; compound palm fitting remains TODO(V32-06)",
            "Rounded finger proxies omit some CAD flanges; joint tree, inertia and visual mesh unchanged",
            "Terminal capsules use whole-mesh equivalent-inertia thickness and axis, anchored to the measured tip; not a conservative envelope of the outer cover",
        ], "hands": {},
    }
    if args.terminal_only:
        previous = json.loads((ASSETS / 'collision_proxies.json').read_text())
        result['hands'] = previous['hands']
        result['derivation'] += '; --terminal-only; nonterminal entries retained from prior configuration'
    for side in ("left", "right"):
        path = ASSETS / "urdf" / f"TsenseI_{side}.urdf"
        entries = result['hands'].get(side, {})
        for link in ET.parse(path).getroot().findall("link"):
            name = link.get("name")
            terminal = name.endswith('_DIP_link') or name.endswith('thumb_PIP_link')
            if args.terminal_only and not terminal:
                continue
            collision = link.find("collision")
            mesh = collision.find("geometry/mesh")
            mesh_path = (path.parent / mesh.get("filename")).resolve()
            vertices = read_stl_vertices(mesh_path)
            vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = collision.find("origin")
            if origin is not None:
                vertices = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).apply(vertices)
                vertices += np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            lo, hi = vertices.min(0), vertices.max(0)
            center, span = (lo + hi) / 2, hi - lo
            entry = {
                "source_mesh": str(mesh_path.relative_to(ASSETS)),
                "source_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                "source_bounds": [lo.tolist(), hi.tolist()],
            }
            obj_path = mesh_path.with_suffix(".obj")
            if obj_path.is_file():
                entry["compiler_mesh"] = str(obj_path.relative_to(ASSETS))
                entry["compiler_mesh_sha256"] = hashlib.sha256(obj_path.read_bytes()).hexdigest()
            # The with-ID export swaps the physical roles of base_link and
            # the fixed palm link: base now holds the shell, palm the mount.
            with_id = json.loads((ASSETS / 'model_derivation.json').read_text()).get('source_joint_schema') == 'a32_with_id_semantic_chain'
            housing = name == 'base_link' if with_id else 'palm' in name.lower()
            mounting = 'palm' in name.lower() if with_id else name == 'base_link'
            if housing:
                entry.update(type="mesh", physical_role="palm_housing", reason="Retain thumb recess; compound box decomposition pending")
            elif mounting:
                entry.update(type="box", pos=center.tolist(), size=(span / 2).tolist(),
                             physical_role="rigid_mount",
                             reason="URDF-frame bounding box for rigid mounting base")
            else:
                if terminal:
                    tip = np.asarray(measure_distal_tip(path, name))
                    entry.update(terminal_capsule(mesh_path, mesh.get('scale', '1 1 1'),
                                                  origin, vertices, tip))
                else:
                    entry.update(inertial_capsule(mesh_path, mesh.get("scale", "1 1 1"), origin),
                                 reason="Shadow-compatible MuJoCo mesh-to-capsule equivalent-inertia fitting; compiled dimensions stored explicitly")
            entries[name] = entry
        result["hands"][side] = entries
    output = args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
