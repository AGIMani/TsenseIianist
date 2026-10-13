"""Local URDF scene manifest and CPU FK, independent of simulation/GPU/drivers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """URDF fixed-axis RPY: T_A_B maps B coordinates to A."""
    x, y, z = np.asarray(rpy, dtype=float)
    cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
    result = np.eye(4)
    result[:3, :3] = [[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]]
    result[:3, 3] = xyz
    return result


def _origin(element):
    item = element.find("origin")
    return (np.eye(4) if item is None else
            transform(np.fromstring(item.get("xyz", "0 0 0"), sep=" "),
                      np.fromstring(item.get("rpy", "0 0 0"), sep=" ")))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _constants(relative, module_name):
    # These existing modules contain stdlib-only constants. Avoid importing the
    # package's simulation __init__, which would make the web UI require MuJoCo.
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _urdf(path, component, assets, warnings):
    source = ET.parse(path).getroot()
    prefix = component.removeprefix("arm_") + "_" if component.startswith("arm_") else ""
    links, joints = [], []
    for link in source.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename")
            mesh_path = ((path.parent / "meshes" / Path(filename).name) if filename.startswith("package://")
                         else path.parent / filename).resolve()
            if not mesh_path.is_relative_to(ROOT) or not mesh_path.is_file():
                warnings.append(f"缺少本地模型资源: {component}/{Path(filename).name}")
                continue
            # Stable opaque IDs; HTTP handler may serve only this explicit map.
            asset_id = hashlib.sha256(str(mesh_path.relative_to(ROOT)).encode()).hexdigest()[:20]
            assets[asset_id] = str(mesh_path)
            color = visual.find("material/color")
            visuals.append({"mesh_url": "/api/assets/" + asset_id, "asset_id": asset_id,
                            "transform": _origin(visual).tolist(),
                            "scale": np.fromstring(mesh.get("scale", "1 1 1"), sep=" ").tolist(),
                            "color": np.fromstring(color.get("rgba"), sep=" ").tolist() if color is not None else [.8,.83,.87,1]})
        links.append({"name": link.get("name"), "visuals": visuals})
    for joint in source.findall("joint"):
        limit, axis = joint.find("limit"), joint.find("axis")
        bounds = {k: float(v) for k, v in limit.attrib.items()} if limit is not None else {}
        # Vendor effort/velocity=0 are unknown, not measured immobilization.
        for key in ("effort", "velocity"):
            if bounds.get(key) == 0:
                bounds[key] = None
        joints.append({"name": prefix + joint.get("name"), "urdf_name": joint.get("name"),
                       "parent": joint.find("parent").get("link"), "child": joint.find("child").get("link"),
                       "type": joint.get("type"), "origin": _origin(joint).tolist(),
                       "axis": np.fromstring(axis.get("xyz"), sep=" ").tolist() if axis is not None else [1,0,0],
                       "limit": bounds})
    roots = {item["name"] for item in links} - {item["child"] for item in joints}
    if len(roots) != 1:
        raise ValueError("URDF 必须是单根运动学树: " + str(path))
    return {"id": component, "root_link": roots.pop(), "links": links, "joints": joints,
            "root_transform": np.eye(4).tolist(), "parent": None, "initial_positions": {},
            "source": {"urdf": str(path.relative_to(ROOT)), "sha256": _sha(path),
                       "hardware_calibrated": False}}


def build_scene_manifest() -> dict:
    """Read existing CAD/installation transforms. Hand roots attach to Link7.

    The zero/seed arm posture is a labelled preview initial pose, not a hardware
    pose or IK solution. Missing models remain missing; no substituted robot.
    """
    assets, warnings, models = {}, [], []
    arm_path = ROOT / "preview/assets/tj_m6"
    mount_path = arm_path / "preview_mount.json"
    mount = json.loads(mount_path.read_text()) if mount_path.is_file() else None
    connector_path = ROOT / "configs/preview/hand_connector_20260930.json"
    connector = json.loads(connector_path.read_text()) if connector_path.is_file() else None
    constants = _constants("tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py", "_online_hand_constants")
    piano = _constants("tsensei_v3_2_pianist/models/piano/piano_constants.py", "_online_piano_constants")
    pose_path = ROOT / "online_control/config/preview_pose.json"
    pose = json.loads(pose_path.read_text()) if pose_path.is_file() else None
    if pose and any(not (ROOT / p).is_file() or _sha(ROOT / p) != value for p, value in pose["sources"].items()):
        warnings.append("预览初始 IK 姿态的模型来源已变化；回退到明确标注的 seed 姿态")
        pose = None
    for side, sign in (("right", 1), ("left", -1)):
        arm_urdf = arm_path / side / "arm.urdf"
        if mount is None or not arm_urdf.is_file():
            warnings.append(f"{side} 机械臂模型或安装配置缺失；仅显示已有手模型")
            continue
        arm = _urdf(arm_urdf, "arm_" + side, assets, warnings)
        center = np.asarray(mount["shoulder_center_xyz_m"]) + [0, sign*mount["shoulder_spacing_m"]/2, 0]
        # A calibrated side may have its own base location. Unspecified sides
        # retain the legacy shoulder-center/spacing placement.
        base_xyz = mount.get("base_xyz_m", {})
        if not isinstance(base_xyz, dict) or set(base_xyz) - {"left", "right"}:
            raise ValueError("base_xyz_m must map left/right to piano-frame positions")
        center = np.asarray(base_xyz.get(side, center), dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Invalid arm base position: " + side)
        arm["root_transform"] = transform(center, mount["base_rpy_rad"][side]).tolist()
        arm["initial_positions"] = dict(zip([j["name"] for j in arm["joints"] if j["type"] != "fixed"], mount["seed_q_rad"][side]))
        if pose and "arm_"+side in pose["positions"]:
            arm["initial_positions"] = dict(pose["positions"]["arm_"+side])
            arm["source"]["preview_pose"] = str(pose_path.relative_to(ROOT))
            arm["source"]["preview_pose_sha256"] = _sha(pose_path)
        arm["source"].update(mount=str(mount_path.relative_to(ROOT)), mount_sha256=_sha(mount_path))
        models.append(arm)
    for side in ("right", "left"):
        path = ROOT / f"tsensei_v3_2_pianist/assets/urdf/TsenseI_{side}.urdf"
        if not path.is_file():
            warnings.append(side + " 手模型缺失")
            continue
        hand = _urdf(path, "hand_" + side, assets, warnings)
        angles = {**constants.INITIAL_ANGLES, **constants.INITIAL_ANGLES_BY_SIDE.get(side, {})}
        hand["initial_positions"] = {
            j["name"]: angles.get(j["name"].removeprefix(side + "_hand_").removesuffix("_joint"), 0.)
            for j in hand["joints"] if j["type"] != "fixed"}
        if any(m["id"] == "arm_"+side for m in models):
            if connector:
                offset = connector["offsets"][side]
                tool = np.asarray(connector["nominal_link7_to_hand"][side]) @ np.linalg.inv(
                    transform(np.asarray(offset["xyz_mm"])/1000, np.deg2rad(offset["rpy_deg"])))
            else:
                tool = transform(**{ "xyz": mount["link7_to_hand"][side]["xyz_m"],
                                    "rpy": mount["link7_to_hand"][side]["rpy_rad"]})
            hand["parent"] = {"model": "arm_"+side, "link": "Link7_"+("R" if side == "right" else "L"),
                              "transform": tool.tolist()}
        else:
            # Standalone hand mounting from existing preview constants.
            rotation = np.array([[-0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
            root = np.eye(4); root[:3, :3] = rotation
            root[:3, 3] = constants.RIGHT_POSITION if side == "right" else constants.LEFT_POSITION
            hand["root_transform"] = root.tolist()
        hand["source"].update(right_hand_provisional_mirror=side == "right",
                             connector=str(connector_path.relative_to(ROOT)) if connector else None,
                             connector_sha256=_sha(connector_path) if connector else None)
        if connector:
            # Connector meshes are expressed in the hand offset frame in the
            # existing preview; preserve that exact composition here.
            offset = connector["offsets"][side]
            matrix = transform(np.asarray(offset["xyz_mm"])/1000, np.deg2rad(offset["rpy_deg"]))
            base = next(link for link in hand["links"] if link["name"] == hand["root_link"])
            for name in connector["meshes"][side]:
                mesh = (ROOT / name).resolve()
                if not mesh.is_relative_to(ROOT) or not mesh.is_file():
                    warnings.append("安装件网格缺失: " + name); continue
                if _sha(mesh) != connector["mesh_sha256"][name]:
                    warnings.append("安装件网格散列不匹配，已跳过: " + name); continue
                asset_id = hashlib.sha256(name.encode()).hexdigest()[:20]; assets[asset_id] = str(mesh)
                base["visuals"].append({"mesh_url": "/api/assets/" + asset_id, "asset_id": asset_id,
                                        "transform": matrix.tolist(), "scale": [1,1,1], "color": [.85,.6,.22,1]})
        models.append(hand)
    warnings += ["TJ-ARM-02/03、TODO(V32-MOUNT-01)：肩座与法兰安装变换仍待真机标定",
                 "右手为左手镜像推导；手 URDF velocity/effort=0 为未知限制，不可用作实机限值",
                 ("初始臂姿态来自现有 CPU ArmPreview IK；仅作预览，非回传" if pose else
                  "初始臂姿态使用配置 seed，仅作预览；不得视为已到位回传")]
    return {"schema": "online_control.scene.v1", "frame": "base", "models": models, "assets": assets,
            "warnings": warnings, "piano": {"key_count": piano.NUM_KEYS, "first_midi": 21,
                "white_width": piano.WHITE_KEY_WIDTH, "white_length": piano.WHITE_KEY_LENGTH,
                "white_height": piano.WHITE_KEY_HEIGHT, "gap": piano.SPACING_BETWEEN_WHITE_KEYS,
                "black_width": piano.BLACK_KEY_WIDTH, "black_length": piano.BLACK_KEY_LENGTH,
                "black_height": piano.BLACK_KEY_HEIGHT, "black_raise": piano.BLACK_OFFSET_FROM_WHITE,
                "white_travel": piano.WHITE_KEY_TRAVEL_DISTANCE, "black_travel": piano.BLACK_KEY_TRAVEL_DISTANCE,
                "xyz": [0,0,0], "rpy": [0,0,0], "source": "models/piano/piano_constants.py",
                "hardware_calibrated": False}}


def _joint_motion(joint, value):
    axis = np.asarray(joint["axis"], dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("非固定关节轴为零: " + joint["name"])
    axis = axis / norm
    result = np.eye(4)
    if joint["type"] == "prismatic":
        result[:3, 3] = axis * value
    elif joint["type"] in ("revolute", "continuous"):
        x, y, z = axis
        cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        result[:3, :3] = np.eye(3) + np.sin(value)*cross + (1-np.cos(value))*(cross @ cross)
    else:
        raise ValueError("不支持的关节类型: " + joint["type"])
    return result


def forward_kinematics(manifest: dict, positions_by_component: dict, use_defaults: bool = False) -> dict:
    """Return component→link→world 4×4; omit descendants of missing joints.

    Defaults are an explicit preview aid only. Recorded feedback reconstruction
    must keep use_defaults=False so missing data never masquerade as a pose.
    """
    result = {}
    for model in manifest["models"]:
        component = model["id"]
        positions = positions_by_component.get(component, {})
        if "joint_names" in positions:
            positions = dict(zip(positions["joint_names"], positions.get("positions", [])))
        positions = {name.split("/")[-1]: value for name, value in positions.items()}
        if use_defaults:
            positions = {**model["initial_positions"], **positions}
        parent = model["parent"]
        if parent:
            frame = result.get(parent["model"], {}).get(parent["link"])
            if frame is None:
                result[component] = {}; continue
            root = np.asarray(frame) @ np.asarray(parent["transform"])
        else:
            root = np.asarray(model["root_transform"])
        frames = {model["root_link"]: root}
        pending = list(model["joints"])
        for _ in range(len(pending)+1):
            next_pending = []
            for joint in pending:
                if joint["parent"] not in frames:
                    next_pending.append(joint); continue
                matrix = frames[joint["parent"]] @ np.asarray(joint["origin"])
                if joint["type"] != "fixed":
                    value = positions.get(joint["name"], positions.get(joint["urdf_name"]))
                    if value is None or not np.isfinite(value):
                        continue
                    matrix = matrix @ _joint_motion(joint, value)
                frames[joint["child"]] = matrix
            if len(next_pending) == len(pending):
                break
            pending = next_pending
        result[component] = {name: value.tolist() for name, value in frames.items()}
    return result


def model_limits(manifest=None) -> dict:
    """Read URDF limits; acceleration remains absent and must be commissioned."""
    return {model["id"]: {joint["name"]: dict(joint["limit"]) for joint in model["joints"]
                          if joint["type"] != "fixed"} for model in (manifest or build_scene_manifest())["models"]}
