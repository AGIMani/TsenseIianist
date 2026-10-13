#!/usr/bin/env python3
"""Install the latest 20-output hand export and convex-hull collisions.

The input package is a visual-only URDF with one merged STL per link. This
script copies that package into the standalone project, mirrors it into the
provisional right hand, computes a convex hull for every link, and writes the
two URDFs plus both collision-proxy manifests. The active URDFs always point
at the hull meshes, so a legacy mesh profile cannot silently re-enable raw CAD
collision geometry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tsensei_v3_2_pianist" / "assets"
URDFS = ASSETS / "urdf"
REFLECTION = np.diag([1.0, -1.0, 1.0])
STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def fmt(values) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def right_name(name: str) -> str:
    return name.replace("left_", "right_").replace("Left_", "Right_")


def read_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL is too short: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError(f"Expected binary STL: {path}")
    return np.frombuffer(data, dtype=STL_DTYPE, offset=84).copy()


def write_binary_stl(path: Path, vertices: np.ndarray, header: bytes = b"TsenseI convex-hull collision mesh") -> None:
    triangles = np.zeros(len(vertices), dtype=STL_DTYPE)
    triangles["vertices"] = vertices.astype(np.float32)
    a = triangles["vertices"][:, 1] - triangles["vertices"][:, 0]
    b = triangles["vertices"][:, 2] - triangles["vertices"][:, 0]
    normal = np.cross(a, b)
    length = np.linalg.norm(normal, axis=1)
    valid = length > 0
    normal[valid] /= length[valid, None]
    triangles["normal"] = normal.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.ljust(80, b" ")[:80] + struct.pack("<I", len(triangles)) + triangles.tobytes())


def convex_hull(source: Path, target: Path) -> dict:
    triangles = read_triangles(source)
    unique = np.unique(np.round(triangles["vertices"].reshape(-1, 3), decimals=12), axis=0)
    if len(unique) < 4:
        raise ValueError(f"Not enough vertices for convex hull: {source}")
    hull = ConvexHull(unique)
    faces = unique[hull.simplices].copy()
    for index, equation in enumerate(hull.equations):
        normal = np.cross(faces[index, 1] - faces[index, 0], faces[index, 2] - faces[index, 0])
        if np.dot(normal, equation[:3]) < 0:
            faces[index, [1, 2]] = faces[index, [2, 1]]
    write_binary_stl(target, faces)
    return {
        "source_vertex_count": int(len(unique)),
        "hull_vertex_count": int(len(hull.vertices)),
        "hull_triangle_count": int(len(hull.simplices)),
        "hull_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def mirror_stl(source: Path, target: Path) -> None:
    triangles = read_triangles(source)
    triangles["vertices"][:, :, 1] *= -1
    triangles["vertices"] = triangles["vertices"][:, [0, 2, 1], :]
    triangles["normal"][:, 1] *= -1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes()[:80] + struct.pack("<I", len(triangles)) + triangles.tobytes())


def write_obj(path: Path, triangles: np.ndarray) -> None:
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# OBJ conversion for MuJoCo meshes over the STL face limit.\n")
        for vertex in vertices:
            stream.write("v " + fmt(vertex) + "\n")
        for face in inverse.reshape(-1, 3) + 1:
            stream.write(f"f {face[0]} {face[1]} {face[2]}\n")


def maybe_write_obj(stl: Path) -> None:
    triangles = read_triangles(stl)
    if len(triangles) > 200_000:
        write_obj(stl.with_suffix(".obj"), triangles["vertices"])


def mirror_obj(source: Path, target: Path) -> None:
    vertices = []
    faces = []
    for line in source.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(parts[1]), -float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            faces.append([parts[1], parts[3], parts[2]])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as stream:
        stream.write("# Mirrored OBJ conversion for the provisional right hand.\n")
        for vertex in vertices:
            stream.write("v " + fmt(vertex) + "\n")
        for face in faces:
            stream.write("f " + " ".join(face) + "\n")


def mirror_origin(origin: ET.Element) -> None:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    transform[:3, :3] = REFLECTION @ transform[:3, :3] @ REFLECTION
    transform[:3, 3] = REFLECTION @ transform[:3, 3]
    origin.set("xyz", fmt(transform[:3, 3]))
    origin.set("rpy", fmt(Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")))


def mirror_inertia(element: ET.Element) -> None:
    for cross_term in ("ixy", "iyz"):
        if element.get(cross_term) is not None:
            element.set(cross_term, format(-float(element.get(cross_term)), ".17g"))


def hull_inertial(path: Path, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate COM and full body-frame inertia from a uniform convex hull."""
    xml = (f'<mujoco><compiler angle="radian"/><asset><mesh name="h" file="{path.resolve()}"/>'
           '</asset><worldbody><body name="link"><geom type="mesh" mesh="h" density="1"/>'
           '</body></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    volume = float(model.body_mass[1])
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"Convex hull has no positive volume: {path}")
    com = model.body_ipos[1].copy()
    matrix = np.zeros((3, 3))
    mujoco.mju_quat2Mat(matrix.reshape(-1), model.body_iquat[1])
    inertia = matrix @ np.diag(model.body_inertia[1]) @ matrix.T
    return com, inertia * (float(mass) / volume)


def add_inertial(link: ET.Element, com: np.ndarray, inertia: np.ndarray, mass: float) -> None:
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=fmt(com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=format(float(mass), ".17g"))
    ET.SubElement(inertial, "inertia", {
        "ixx": format(float(inertia[0, 0]), ".17g"),
        "ixy": format(float(inertia[0, 1]), ".17g"),
        "ixz": format(float(inertia[0, 2]), ".17g"),
        "iyy": format(float(inertia[1, 1]), ".17g"),
        "iyz": format(float(inertia[1, 2]), ".17g"),
        "izz": format(float(inertia[2, 2]), ".17g"),
    })
    link.insert(0, inertial)


def write_xml(root: ET.Element, target: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)


def install(source_dir: Path) -> None:
    source_dir = source_dir.resolve()
    source_urdf = source_dir / "urdf" / "TsenseI_hand_left_v3_2_withid.urdf"
    source_mesh_dir = source_dir / "meshes" / "left"
    if not source_urdf.is_file() or not source_mesh_dir.is_dir():
        raise FileNotFoundError(f"Expected latest package URDF and meshes under {source_dir}")
    source_root = ET.parse(source_urdf).getroot()
    links = source_root.findall("link")
    joints = source_root.findall("joint")
    if len(links) != 22 or len(joints) != 21:
        raise ValueError(f"Expected latest export to have 22 links/21 joints, got {len(links)}/{len(joints)}")
    if any(link.find("inertial") is not None for link in links):
        raise ValueError("This importer expects the latest visual-only export; inertials must be reviewed separately")

    latest_root = ASSETS / "source" / "latest_project"
    if latest_root.exists():
        shutil.rmtree(latest_root)
    (latest_root / "urdf").mkdir(parents=True)
    shutil.copy2(source_urdf, latest_root / "urdf" / source_urdf.name)
    shutil.copytree(source_mesh_dir, latest_root / "meshes" / "left")
    for mesh in (latest_root / "meshes" / "left").glob("*.STL"):
        maybe_write_obj(mesh)
    (latest_root / "source_directory.txt").write_text(str(source_dir) + "\n")

    hull_root = ASSETS / "collision_hulls"
    if hull_root.exists():
        shutil.rmtree(hull_root)
    stale_collision_sources = ASSETS / "collision_sources"
    if stale_collision_sources.exists():
        shutil.rmtree(stale_collision_sources)
    records = {"left": {}, "right": {}}
    left_hulls = {}
    mass_source = ASSETS / "source" / "TsenseI_left_mass952_before_mount.urdf"
    if not mass_source.is_file():
        raise FileNotFoundError(f"Missing approved 0.952 kg mass source: {mass_source}")
    mass_root = ET.parse(mass_source).getroot()
    mass_by_link = {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in mass_root.findall("link")
    }
    for link in links:
        name = link.get("name")
        source_name = f"{name}.STL"
        source_mesh = source_mesh_dir / source_name
        if not source_mesh.is_file():
            raise FileNotFoundError(f"Missing merged link mesh: {source_mesh}")
        hull_target = hull_root / "left" / f"{name}_convex_hull.stl"
        hull_stats = convex_hull(source_mesh, hull_target)
        left_hulls[name] = hull_target
        mass = mass_by_link.get(name)
        if mass is None:
            raise ValueError(f"Approved mass source has no link {name}")
        com, inertia = hull_inertial(hull_target, mass)
        records["left"][name] = {
            "source_mesh": str((latest_root / "meshes" / "left" / source_name).relative_to(ASSETS)),
            "source_sha256": hashlib.sha256(source_mesh.read_bytes()).hexdigest(),
            "type": "mesh",
            "mesh_file": str(hull_target.relative_to(ASSETS)),
            "physical_role": "rigid_mount" if name in {"base_link", "Left_hand_palm_link"} else "finger_link",
            "hull_method": "scipy_convex_hull_v1",
            "mass_kg": mass,
            "inertial_origin_xyz_m": com.tolist(),
            "inertia_kg_m2": inertia.tolist(),
            "inertia_method": "uniform_convex_hull_scaled_to_0.952kg_link_mass",
            **hull_stats,
        }

    left = copy.deepcopy(source_root)
    left.set("name", "TsenseI_A3.2_left_latest_project")
    for link in left.findall("link"):
        name = link.get("name")
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            mesh.set("filename", f"../source/latest_project/meshes/left/{name}.STL")
        collision = ET.SubElement(link, "collision", {"name": "convex_hull_collision"})
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"../collision_hulls/left/{name}_convex_hull.stl", "scale": "1 1 1"})
        row = records["left"][name]
        add_inertial(link, np.asarray(row["inertial_origin_xyz_m"]),
                     np.asarray(row["inertia_kg_m2"]), row["mass_kg"])
    write_xml(left, URDFS / "TsenseI_left.urdf")

    right = copy.deepcopy(left)
    right.set("name", "TsenseI_A3.2_right_MIRRORED_PROVISIONAL")
    right_mesh_dir = latest_root / "meshes" / "right"
    right_mesh_dir.mkdir(parents=True)
    right_hull_dir = hull_root / "right"
    for element in right.iter():
        for key in ("name", "link"):
            if key in element.attrib:
                element.set(key, right_name(element.get(key)))
        if element.tag == "origin":
            mirror_origin(element)
        elif element.tag == "axis":
            element.set("xyz", fmt(-REFLECTION @ np.fromstring(element.get("xyz"), sep=" ")))
        elif element.tag == "inertia":
            mirror_inertia(element)
        elif element.tag == "mesh":
            filename = Path(element.get("filename"))
            if "source/latest_project/meshes/left" in element.get("filename"):
                source_mesh = latest_root / "meshes" / "left" / filename.name
                target = right_mesh_dir / right_name(filename.name)
                if not target.exists():
                    mirror_stl(source_mesh, target)
                    if source_mesh.with_suffix(".obj").is_file():
                        mirror_obj(source_mesh.with_suffix(".obj"), target.with_suffix(".obj"))
                element.set("filename", f"../source/latest_project/meshes/right/{target.name}")
            elif "collision_hulls/left" in element.get("filename"):
                left_name = filename.name.removesuffix("_convex_hull.stl")
                source_hull = left_hulls[left_name]
                target = right_hull_dir / f"{right_name(left_name)}_convex_hull.stl"
                if not target.exists():
                    mirror_stl(source_hull, target)
                element.set("filename", f"../collision_hulls/right/{target.name}")
    for name, left_entry in records["left"].items():
        source_mesh = latest_root / "meshes" / "left" / f"{name}.STL"
        raw_target = right_mesh_dir / f"{right_name(name)}.STL"
        if not raw_target.exists():
            mirror_stl(source_mesh, raw_target)
        hull_target = right_hull_dir / f"{right_name(name)}_convex_hull.stl"
        records["right"][right_name(name)] = {
            **left_entry,
            "source_mesh": str(raw_target.relative_to(ASSETS)),
            "source_sha256": hashlib.sha256(raw_target.read_bytes()).hexdigest(),
            "mesh_file": str(hull_target.relative_to(ASSETS)),
            "hull_sha256": hashlib.sha256(hull_target.read_bytes()).hexdigest(),
            "inertial_origin_xyz_m": (REFLECTION @ np.asarray(left_entry["inertial_origin_xyz_m"])).tolist(),
            "inertia_kg_m2": (REFLECTION @ np.asarray(left_entry["inertia_kg_m2"]) @ REFLECTION).tolist(),
        }
    write_xml(right, URDFS / "TsenseI_right.urdf")

    source_slot = ASSETS / "source" / "left_original.urdf"
    previous = source_slot.read_bytes() if source_slot.exists() else None
    if previous is not None and previous != source_urdf.read_bytes():
        backup = ROOT / "deprecated" / "model_revisions" / "updated_urdf_20261012" / "left_original_before_latest_project.urdf"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(previous)
    shutil.copy2(source_urdf, source_slot)

    manifest = {
        "schema": 1,
        "profile": "convex_hull_v1",
        "units": "metres; all active collision meshes are scipy convex hulls of the supplied merged per-link STL",
        "reference": "TsenseI_hand_20_outputs_description",
        "source_urdf": "source/latest_project/urdf/TsenseI_hand_left_v3_2_withid.urdf",
        "source_urdf_sha256": hashlib.sha256(source_urdf.read_bytes()).hexdigest(),
        "preserve_original_mesh_links": [],
        "limitations": [
            "Convex hulls fill concave cavities and are an outer collision approximation.",
            "The right hand is a geometric mirror candidate, not a manufacturer right-hand model.",
            "The supplied export has no inertial tags; dynamics must be reviewed before hardware claims.",
        ],
        "hands": records,
    }
    (ASSETS / "collision_proxies_convex_hull.json").write_text(json.dumps(manifest, indent=2) + "\n")
    primitive_alias = dict(manifest)
    primitive_alias["profile"] = "shadow_primitive_v1"
    primitive_alias["reference"] = "Deprecated profile name retained only as a hull-only alias"
    (ASSETS / "collision_proxies.json").write_text(json.dumps(primitive_alias, indent=2) + "\n")
    report = {
        "model_name": "TsenseI_A3.2",
        "source_joint_schema": "latest_project_20_outputs_withid_v1",
        "left_source": "source/left_original.urdf",
        "left_source_sha256": hashlib.sha256(source_urdf.read_bytes()).hexdigest(),
        "source_package": str(source_dir),
        "source_package_urdf": "source/latest_project/urdf/TsenseI_hand_left_v3_2_withid.urdf",
        "right_hand_is_provisional_mirror": True,
        "reflection": REFLECTION.tolist(),
        "link_count": 22,
        "joint_count": 21,
        "revolute_joint_count": 20,
        "left_source_name_map": {},
        "canonical_from_source_root": {
            "xyz": [0.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0],
            "meaning": "The supplied latest export is already in the project hand frame.",
        },
        "mass_source": "source/TsenseI_left_mass952_before_mount.urdf",
        "mass_total_kg": float(sum(mass_by_link.values())),
        "inertia_estimation": {
            "method": "uniform convex hull volume at density one, scaled to each retained link mass",
            "hardware_calibration_status": "TODO(V32-MASS-HULL-CALIBRATION)",
            "source_export_contains_inertial_tags": False,
        },
        "collision_contract": "convex_hull_v1 for every link; no raw mesh collision fallback",
        "source_urdf_sha256": hashlib.sha256(source_urdf.read_bytes()).hexdigest(),
        "generated_sha256": {
            str(path.relative_to(ASSETS)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted([URDFS / "TsenseI_left.urdf", URDFS / "TsenseI_right.urdf"] +
                               list(hull_root.rglob("*.stl")))
        },
        "limitations": manifest["limitations"],
    }
    (ASSETS / "model_derivation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source": str(source_urdf), "source_sha256": report["source_urdf_sha256"],
                      "links": 22, "joints": 21, "hulls": 44}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/home/hj/TsenseI_hand_20_outputs_description"))
    args = parser.parse_args()
    install(args.source_dir)


if __name__ == "__main__":
    main()
