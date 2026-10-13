"""Convert existing offline traces; checkpoint inference reuses the Slurm renderer.

No policy action is interpreted as a hardware angle. Compiled MuJoCo actuator
transmission metadata is the only authority for converting ctrl to joint targets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from .models import ROOT, build_scene_manifest
from .trajectory import SCHEMA, load_trajectory


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def import_arm_npz(path, metadata=None, *, limited_time=False):
    """Import this repository's time/q arm traces without inventing joint order.

    Existing report.arm_mount/connector identifies the right-then-left URDF
    ordering used by preview_trajectory.differential_track. Other files require
    metadata.tracks with explicit columns/joint_names/units; units may be unknown
    for preview. IK traces always remain unverified for hardware.
    """
    path = Path(path)
    if metadata is None:
        candidates = (path.with_name("arm_report.json"), path.with_name("report.json"))
        metadata = next((_json(p) for p in candidates if p.is_file()), {})
    if isinstance(metadata, (str, Path)):
        metadata = _json(metadata)
    with np.load(path, allow_pickle=False) as archive:
        if not {"time", "q"}.issubset(archive.files):
            raise ValueError("现有臂轨迹必须具有 time/q 数组")
        times, q = np.asarray(archive["time"], float), np.asarray(archive["q"], float)
        scale = float(archive["time_scale"]) if "time_scale" in archive.files else float(metadata.get("time_scale", 1))
    if q.ndim != 2 or times.ndim != 1 or len(q) != len(times):
        raise ValueError("臂轨迹 time/q 维度错误")
    tracks = copy.deepcopy(metadata.get("tracks", {}))
    if not tracks and "arm_mount" in metadata and "connector" in metadata:
        models = [m for m in build_scene_manifest()["models"] if m["id"].startswith("arm_")]
        if q.shape[1] != sum(len(m["initial_positions"]) for m in models) or len(models) != 2:
            raise ValueError("已有臂轨迹与本地双臂 URDF 关节数不一致")
        offset = 0
        for model in models:  # Existing differential_track SIDES = right,left.
            names = list(model["initial_positions"])
            tracks[model["id"]] = {"columns": list(range(offset, offset+len(names))), "joint_names": names,
                                   "units": ["rad"]*len(names), "frame": "base",
                                   "model_id": model["source"]["sha256"]}
            offset += len(names)
    if not tracks:
        raise ValueError("NPZ 未携带关节顺序；请提供原 report.json 或显式 columns/joint_names 元数据。禁止猜测通道映射")
    if limited_time:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("时间缩放系数必须有限且为正")
        times = times * scale
    for name, track in tracks.items():
        columns = track.pop("columns", None)
        if (not isinstance(columns, list) or len(columns) != len(track.get("joint_names", [])) or
                any(type(c) is not int or c < 0 or c >= q.shape[1] for c in columns) or len(set(columns)) != len(columns)):
            raise ValueError(name + ": columns 元数据无效")
        track.update(times=times.tolist(), positions=q[:, columns].tolist(), semantic="control_target",
                     mapping_verified=False, offset=0.)
        track.setdefault("units", ["unknown"] * len(columns))
        track.setdefault("frame", "unknown")
        track.setdefault("model_id", "unknown")
    return load_trajectory({"schema": SCHEMA, "name": path.stem, "tracks": tracks,
                            "source": {"format": "existing_arm_time_q_npz", "path": str(path.resolve()), "sha256": _sha(path)},
                            "metadata": {"import_report": metadata, "time_scale_applied": scale if limited_time else 1.,
                            "hardware_status": "预览/候选轨迹；安装标定、设备映射与动态限制未验证",
                            "interpolation": "linear; not the historical C2-limited renderer"}})


def model_export_metadata(model):
    """Read actual scalar-joint indices and position-servo transmissions."""
    import mujoco
    joints, actuators = [], []
    for index in range(model.njnt):
        kind = int(model.jnt_type[index])
        if kind not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        name = model.joint(index).name
        joints.append({"id": index, "name": name, "qpos_index": int(model.jnt_qposadr[index]),
                       "units": "rad" if kind == int(mujoco.mjtJoint.mjJNT_HINGE) else "m",
                       "range": model.jnt_range[index].tolist()})
    by_id = {j["id"]: j for j in joints}
    for index in range(model.nu):
        joint_id = int(model.actuator_trnid[index, 0]); gain = float(model.actuator_gainprm[index, 0])
        gear = float(model.actuator_gear[index, 0])
        position_servo = (
            int(model.actuator_trntype[index]) == int(mujoco.mjtTrn.mjTRN_JOINT) and joint_id in by_id and
            int(model.actuator_dyntype[index]) == int(mujoco.mjtDyn.mjDYN_NONE) and
            int(model.actuator_gaintype[index]) == int(mujoco.mjtGain.mjGAIN_FIXED) and
            int(model.actuator_biastype[index]) == int(mujoco.mjtBias.mjBIAS_AFFINE) and
            gain > 0 and abs(gear) > 1e-12 and abs(float(model.actuator_biasprm[index, 0])) < 1e-10 and
            np.isclose(float(model.actuator_biasprm[index, 1]), -gain))
        actuators.append({"index": index, "name": model.actuator(index).name, "joint_id": joint_id,
                          "position_servo_verified": bool(position_servo), "gear": gear,
                          "ctrl_limited": bool(model.actuator_ctrllimited[index]) and not bool(
                              int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)),
                          "ctrl_range": model.actuator_ctrlrange[index].tolist()})
    return {"schema": "online_control.compiled_model.v1", "nq": model.nq, "nu": model.nu,
            "joints": joints, "actuators": actuators}


def _component(joint_name):
    short = joint_name.split("/")[-1]
    for side in ("left", "right"):
        if short.startswith(side + "_hand_"):
            return "hand_" + side, short
        if short.startswith(side + "_Joint"):
            return "arm_" + side, short
    return None, short


def reference_notes_from_goals(goals, times, control_hz):
    """Join exact binary goal intervals from goal_applied, excluding sustain.

    The score was already quantized by the original task. Rearticulations with
    no intervening empty score frame cannot be recovered and are not invented.
    """
    values, times = np.asarray(goals, dtype=float), np.asarray(times, dtype=float)
    if (values.ndim != 2 or values.shape[0] != len(times) or values.shape[1] not in (88, 89)
            or not np.isfinite(values).all() or not np.isin(values[:, :88], (0., 1.)).all()
            or not isinstance(control_hz, (float, int)) or not np.isfinite(control_hz) or control_hz <= 0):
        raise ValueError("缺少明确的88键二值 goal_applied/控制频率，不能推测谱面参考")
    starts = np.r_[times[0] - 1/control_hz, times[:-1]]
    if starts[0] < -1e-9 or np.any(times <= starts):
        raise ValueError("谱面目标时间不符合控制区间")
    starts[0] = max(0., starts[0])
    notes = []
    active = {}
    for i, row in enumerate(values[:, :88].astype(bool)):
        for key in list(active):
            if not row[key]:
                notes.append({"midi": key+21, "start": active.pop(key), "end": float(starts[i])})
        for key in np.flatnonzero(row):
            active.setdefault(int(key), float(starts[i]))
    notes.extend({"midi": key+21, "start": start, "end": float(times[-1])} for key, start in active.items())
    return sorted(notes, key=lambda item: (item["start"], item["midi"]))


def convert_rollout_arrays(arrays, model_info, report=None, *, source=None, components=None):
    """Pure conversion core for tests and JSON compiled-model sidecars."""
    report = report or {}; source = source or {}; components = set(components or ("hand_left", "hand_right", "arm_left", "arm_right"))
    if not {"qpos", "ctrl"}.issubset(arrays):
        raise ValueError("此旧 rollout 缺少实际 qpos/ctrl；请使用匹配原源码重新离线回放采集，不能由 action 猜关节目标")
    qpos = np.asarray(arrays["qpos"], dtype=float)
    ctrl = np.asarray(arrays["ctrl"], dtype=float)
    if (qpos.ndim != 2 or ctrl.ndim != 2 or len(qpos) != len(ctrl) or
            qpos.shape[1] != model_info["nq"] or ctrl.shape[1] != model_info["nu"]):
        raise ValueError("rollout 与实际编译模型的 qpos/ctrl 数量不一致")
    if "policy_time" in arrays:
        times = np.asarray(arrays["policy_time"], dtype=float)
    elif "time" in arrays:
        times = np.asarray(arrays["time"], dtype=float)
    elif report.get("fps"):
        times = (np.arange(len(qpos)) + 1) / float(report["fps"])
    else:
        raise ValueError("缺少保存的时间戳或控制频率；不能推测采样率")
    actual_tracks, target_tracks, omitted = {}, {}, []
    effective_ctrl = ctrl.copy()
    for actuator in model_info["actuators"]:
        if actuator.get("ctrl_limited"):
            effective_ctrl[:, actuator["index"]] = np.clip(effective_ctrl[:, actuator["index"]], *actuator["ctrl_range"])
    mapping = {}
    for joint in model_info["joints"]:
        component, short = _component(joint["name"])
        if component not in components:
            omitted.append(joint); continue
        mapping[joint["id"]] = (component, short, joint)
        track = actual_tracks.setdefault(component, {"joint_names": [], "units": [], "columns": []})
        track["joint_names"].append(short); track["units"].append(joint["units"]); track["columns"].append(joint["qpos_index"])
    for actuator in model_info["actuators"]:
        if actuator["joint_id"] not in mapping or not actuator["position_servo_verified"]:
            continue
        component, short, joint = mapping[actuator["joint_id"]]
        track = target_tracks.setdefault(component, {"joint_names": [], "units": [], "columns": [], "gear": []})
        if short in track["joint_names"]:
            raise ValueError("同一关节有多个位置驱动器，无法无歧义导出: " + short)
        track["joint_names"].append(short); track["units"].append(joint["units"])
        track["columns"].append(actuator["index"]); track["gear"].append(actuator["gear"])
    if not actual_tracks:
        raise ValueError("没有可识别的臂/手关节；请检查实际编译模型来源")
    model_id = source.get("model_sha256", report.get("scene_sha256", "unknown"))
    for tracks, values, semantic in ((actual_tracks, qpos, "simulation_actual"), (target_tracks, effective_ctrl, "control_target")):
        for track in tracks.values():
            selected = values[:, track.pop("columns")]
            if "gear" in track:
                selected = selected / np.asarray(track.pop("gear"))
            sample_times = times
            if semantic == "control_target" and report.get("fps"):
                # Existing renderer captures ctrl after stepping: that target
                # acted during the preceding interval, unlike endpoint qpos.
                sample_times = np.r_[times[0] - 1/float(report["fps"]), times]
                if sample_times[0] < -1e-9:
                    raise ValueError("记录的控制目标时间与保存的控制频率冲突")
                sample_times[0] = max(0., sample_times[0])
                selected = np.concatenate([selected, selected[-1:]], axis=0)
            track.update(positions=selected.tolist(), times=sample_times.tolist(), offset=0., frame="joint_local",
                         semantic=semantic, mapping_verified=False, model_id=model_id,
                         interpolation="previous" if semantic == "control_target" else "linear",
                         metadata={"simulation_mapping_verified": True,
                                   "timestamps": "control interval start (final hold endpoint appended)" if semantic == "control_target" and report.get("fps") else "recorded state endpoint",
                                   "hardware_mapping": "未验证；模拟关节与 SDK 不可直接等同"})
    events = []
    key_states = None
    if "actual" in arrays:
        activation = np.asarray(arrays["actual"], dtype=bool)
        if activation.ndim == 2 and activation.shape == (len(times), 88):
            previous = np.zeros(88, dtype=bool)
            for time, row in zip(times, activation):
                for key in np.flatnonzero(row != previous):
                    events.append({"time": float(time), "kind": "note_on" if row[key] else "note_off",
                                   "midi": int(key)+21, "source": "simulation_key_state", "velocity": None})
                previous = row
            key_states = {"times": times.tolist(), "activation": activation.tolist(), "source": "simulation_actual",
                          "first_midi": 21, "key_count": 88}
            key_joints = []
            for joint in model_info["joints"]:
                match = re.search(r"(?:white|black)_joint_(\d+)$", joint["name"])
                if match:
                    key_joints.append((int(match.group(1)), joint))
            if len(key_joints) == 88 and sorted(i for i, _ in key_joints) == list(range(88)):
                travel = np.column_stack([(qpos[:, j["qpos_index"]]-j["range"][0]) /
                                          (j["range"][1]-j["range"][0]) for _, j in sorted(key_joints)])
                key_states["travel_normalized"] = travel.tolist()
    common = {"schema": SCHEMA, "source": {**source, "report": report}, "events": events,
              "metadata": {"control_hz": report.get("fps"), "time_origin": "policy_reset; state saved after each step",
                           "sampling": "actual: linear preview; control: preceding interval start + original zero order hold",
                           "virtual_wrist_warning": "训练虚拟腕轴不等于机械臂角度；臂轨迹须通过已有记录轨迹 IK 独立导出",
                           "omitted_non_robot_joint_names": [j["name"] for j in omitted], "key_states": key_states,
                           "hardware_status": "仅仿真/预览；未完成 SDK 映射、限制、安装与碰撞验收"}}
    if "goals" in arrays:
        try:
            common["metadata"]["reference_notes"] = reference_notes_from_goals(arrays["goals"], times, report.get("fps"))
            common["metadata"]["reference_notes_source"] = "recorded_score_goals"
            common["metadata"]["reference_notes_timing"] = "goal_applied over preceding control interval; quantized original score; same-pitch adjacent active frames merged"
        except ValueError as exc:
            common["metadata"]["reference_notes_unavailable"] = str(exc)
    actual = load_trajectory({**copy.deepcopy(common), "name": "仿真实际关节状态", "tracks": actual_tracks})
    targets = load_trajectory({**copy.deepcopy(common), "name": "仿真控制器目标（已映射）", "tracks": target_tracks}) if target_tracks else None
    actions = {"schema": "online_control.policy_actions.v1", "semantic": "policy_action_not_joint_angle",
               "times": times.tolist(), "source": source, "action_profile": report.get("action_profile"),
               "values": np.asarray(arrays["actions"]).tolist() if "actions" in arrays else None,
               "raw_ctrl": ctrl.tolist(), "actuator_names": [a["name"] for a in model_info["actuators"]],
               "raw_qpos": qpos.tolist(), "compiled_model": model_info}
    return {"control_targets": targets, "simulation_actual": actual, "policy_actions": actions}


def export_rollout(rollout, output, *, scene=None, compiled_metadata=None, report=None, components=None):
    rollout, output = Path(rollout), Path(output)
    report = _json(report or rollout.with_name("report.json"))
    if compiled_metadata:
        info = _json(compiled_metadata)
        model_path = Path(compiled_metadata)
    else:
        try:
            import mujoco
        except ImportError as exc:
            raise ValueError("读取保存的 scene.xml 需要可选 mujoco；可在训练 CPU 环境转换或提供 compiled_metadata JSON") from exc
        model_path = Path(scene or rollout.parent / "kinematic_scene/scene.xml")
        if not model_path.is_file():
            raise ValueError("缺少原始编译 scene.xml；请通过 --capture-physics 重放。禁止用当前 URDF 猜测旧 checkpoint 顺序")
        info = model_export_metadata(mujoco.MjModel.from_xml_path(str(model_path)))
    with np.load(rollout, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("qpos", "ctrl", "actions", "actual", "goals", "policy_time", "time") if key in archive.files}
    result = convert_rollout_arrays(arrays, info, report,
        source={"format": "mjlab_rollout_npz", "path": str(rollout.resolve()), "sha256": _sha(rollout),
                "model_source": str(model_path.resolve()), "model_sha256": _sha(model_path)}, components=components)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in result.items():
        if value is not None:
            _save(output / (name + ".json"), value)
    _save(output / "export_manifest.json", {"source": result["simulation_actual"]["source"],
          "files": {p.name: _sha(p) for p in output.glob("*.json")}, "hardware_executable": False})
    return result


def checkpoint_export(args):
    """Run the existing strict renderer/observer only inside a Slurm allocation."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("checkpoint 推理必须通过 bash scripts/hpc/submit.sh online-export ...；普通会话仅可转换离线 NPZ")
    checkpoint = args.checkpoint.resolve(); config = (args.config or checkpoint.parent / "config.json").resolve()
    output = args.output.resolve()
    if not checkpoint.is_file() or not config.is_file():
        raise ValueError("checkpoint 与对应 config.json 必须存在")
    if output.exists():
        raise ValueError("输出目录已经存在，请选择新实验目录")
    output.mkdir(parents=True)
    replay = output / "recorded_rollout"
    if args.source_archive:
        command = [sys.executable, str(ROOT / "scripts/eval_from_source_archive.py"),
                   "--source-archive", str(args.source_archive.resolve()), "--capture-physics"]
    else:
        command = [sys.executable, str(ROOT / "scripts/capture_archive_render.py"), "--archive-root", str(ROOT)]
    command += ["--checkpoint", str(checkpoint), "--config", str(config), "--mode", "render",
                "--output", str(replay), "--no-audio"]
    _save(output / "export_request.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
          "config": _json(config), "config_sha256": _sha(config), "command": command,
          "slurm_job_id": os.environ["SLURM_JOB_ID"], "retraining": False})
    subprocess.run(command, cwd=ROOT, check=True)
    result = export_rollout(replay / "rollout.npz", output / "hand", components=["hand_left", "hand_right"])
    if args.with_arms:
        arm_output = output / "arm_ik"
        python = ROOT / "training/.venv/bin/python"
        if not python.is_file():
            raise ValueError("臂 IK 需要现有 training/.venv 环境；已完成的手导出保留")
        subprocess.run([str(python), str(ROOT / "preview/scripts/replay_arm_ik.py"), "--replay", str(replay),
                        "--output", str(arm_output), "--analysis-only", "--original-samples",
                        "--render-timing", "original"], cwd=ROOT, check=True)
        _save(output / "arm_control_targets.json", import_arm_npz(arm_output / "arm_trajectory.npz"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rollout = commands.add_parser("rollout", help="离线转换已有 rollout.npz，不运行 GPU")
    rollout.add_argument("--rollout", type=Path, required=True); rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--scene", type=Path); rollout.add_argument("--compiled-metadata", type=Path)
    rollout.add_argument("--report", type=Path); rollout.add_argument("--component", action="append", dest="components")
    arm = commands.add_parser("arm", help="导入既有 time/q 臂轨迹")
    arm.add_argument("--input", type=Path, required=True); arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--metadata", type=Path); arm.add_argument("--limited-time", action="store_true")
    checkpoint = commands.add_parser("checkpoint", help="Slurm 内使用原评估器导出，无训练")
    checkpoint.add_argument("--checkpoint", type=Path, required=True); checkpoint.add_argument("--config", type=Path)
    checkpoint.add_argument("--source-archive", type=Path); checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.add_argument("--with-arms", action="store_true", help="额外使用现有差分 IK 导出臂轨迹")
    args = parser.parse_args(argv)
    try:
        if args.command == "arm":
            if args.output.exists():
                raise ValueError("输出文件已存在")
            _save(args.output, import_arm_npz(args.input, args.metadata, limited_time=args.limited_time))
        elif args.command == "rollout":
            export_rollout(args.rollout, args.output, scene=args.scene, compiled_metadata=args.compiled_metadata,
                           report=args.report, components=args.components)
        else:
            checkpoint_export(args)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"导出失败: {exc}\n")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
