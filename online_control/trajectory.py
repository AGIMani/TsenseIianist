"""Pure trajectory import, alignment and validation. No driver imports or I/O sends."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "online_control.trajectory.v1"
COMPONENTS = frozenset(("arm_left", "arm_right", "hand_left", "hand_right"))
SEMANTICS = frozenset(("control_target", "simulation_actual", "feedback"))


def _array(value, name, ndim=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: 必须为数值数组") from exc
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError(f"{name}: 维数错误或包含 NaN/Inf")
    return result


def load_trajectory(document: dict | str | Path) -> dict:
    """Normalize a detached document. Unknown units/mapping remain preview-only.

    A JSON file is accepted for the CLI. HTTP callers should pass parsed JSON;
    this function is not an unrestricted server file-upload endpoint.
    """
    if isinstance(document, (str, Path)):
        document = json.loads(Path(document).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("不支持的轨迹格式；需要 online_control.trajectory.v1")
    result = copy.deepcopy(document)
    tracks = result.get("tracks")
    if not isinstance(tracks, dict) or not tracks or set(tracks) - COMPONENTS:
        raise ValueError("tracks 必须包含已知臂/手分组，且不能为空")
    for component, track in tracks.items():
        if not isinstance(track, dict):
            raise ValueError(f"{component}: 轨迹必须是对象")
        names = track.get("joint_names")
        if (not isinstance(names, list) or not names or
                any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
            raise ValueError(f"{component}: 关节名必须非空且唯一")
        times = _array(track.get("times"), component + ".times", 1)
        positions = _array(track.get("positions"), component + ".positions", 2)
        if not len(times) or positions.shape != (len(times), len(names)):
            raise ValueError(f"{component}: 时间、位置和关节数量不一致")
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f"{component}: 时间必须非负且严格递增")
        offset = float(track.get("offset", 0))
        if not np.isfinite(offset):
            raise ValueError(f"{component}: 时间偏移必须有限")
        units = track.get("units", ["unknown"] * len(names))
        if isinstance(units, str):
            units = [units] * len(names)
        if not isinstance(units, list) or len(units) != len(names) or not all(isinstance(u, str) for u in units):
            raise ValueError(f"{component}: 每个关节必须具有明确单位字段")
        semantic = track.get("semantic", "simulation_actual")
        if semantic not in SEMANTICS:
            raise ValueError(f"{component}: 策略 action 不能作为关节轨迹")
        interpolation = track.get("interpolation", "linear")
        if interpolation not in ("linear", "previous"):
            raise ValueError(f"{component}: 不支持的插值方式")
        track.update(times=times.tolist(), positions=positions.tolist(), offset=offset,
                     units=units, semantic=semantic, mapping_verified=track.get("mapping_verified") is True,
                     frame=str(track.get("frame", "unknown")), model_id=str(track.get("model_id", "unknown")),
                     interpolation=interpolation)
        if "velocities" in track:
            velocities = _array(track["velocities"], component + ".velocities", 2)
            if velocities.shape != positions.shape:
                raise ValueError(f"{component}: 速度与位置数组尺寸不一致")
            track["velocities"] = velocities.tolist()
        track.setdefault("metadata", {})
    result.setdefault("name", "未命名轨迹")
    result.setdefault("source", {})
    result.setdefault("metadata", {})
    result.setdefault("events", [])
    if not isinstance(result["events"], list):
        raise ValueError("events 必须是数组")
    for event in result["events"]:
        if not isinstance(event, dict) or not np.isfinite(float(event.get("time", float("nan")))):
            raise ValueError("事件必须带有限 time 秒时间戳")
    result["duration"] = max(t["times"][-1] + t["offset"] for t in tracks.values())
    result["start_time"] = min(t["times"][0] + t["offset"] for t in tracks.values())
    return result


def sample_trajectory(trajectory: dict, time: float) -> dict:
    """Linear alignment on one timeline; omitted components mean *no command*.

    No extrapolation, invented zero, or silent hold is performed. Execution must
    explicitly select its own missing-component hold policy.
    """
    if not np.isfinite(time):
        raise ValueError("采样时间必须有限")
    result = {}
    for component, track in trajectory["tracks"].items():
        local = time - track.get("offset", 0)
        times = np.asarray(track["times"])
        if local < times[0] - 1e-10 or local > times[-1] + 1e-10:
            continue
        local = np.clip(local, times[0], times[-1])
        i = int(np.searchsorted(times, local, side="right"))
        if i == 0 or i == len(times):
            values = np.asarray(track["positions"][min(i, len(times) - 1)])
            fraction, lower = 0., min(i, len(times) - 1)
        else:
            lower = i - 1
            fraction = float((local - times[lower]) / (times[i] - times[lower]))
            if track.get("interpolation") == "previous":
                fraction = 0.
            values = ((1-fraction) * np.asarray(track["positions"][lower]) +
                      fraction * np.asarray(track["positions"][i]))
        result[component] = {"joint_names": list(track["joint_names"]), "positions": values.tolist(),
                             "units": list(track["units"]), "frame": track["frame"],
                             "semantic": track["semantic"], "source_time": float(local),
                             "interpolated": bool(fraction > 1e-10), "sample_index": lower}
    return result


def validate_trajectory(trajectory: dict, limits: dict | None = None, hardware: bool = False) -> list[str]:
    """Reject bad numbers/times and measured-limit violations; do not invent limits.

    limits[component][joint_name] = {lower, upper, velocity, acceleration} in SI.
    Velocity/acceleration are conservative finite-difference path checks. They
    do not certify controller interpolation, spatial feasibility or collision.
    """
    try:
        trajectory = load_trajectory(trajectory)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return [str(exc)]
    errors = []
    limits = limits or {}
    for component, track in trajectory["tracks"].items():
        if hardware:
            if track["semantic"] != "control_target":
                errors.append(f"{component}: {track['semantic']} 只能预览，不能直接下发")
            if not track["mapping_verified"]:
                errors.append(f"{component}: 实机关节映射未验证")
            if any(u not in ("rad", "deg", "m") for u in track["units"]):
                errors.append(f"{component}: 关节单位不明确")
            if track["frame"] in ("", "unknown") or track["model_id"] in ("", "unknown"):
                errors.append(f"{component}: 缺少坐标系或模型来源")
        positions = np.asarray(track["positions"])
        times = np.asarray(track["times"])
        dt = np.diff(times)
        velocity = np.diff(positions, axis=0) / dt[:, None]
        acceleration = (np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:])[:, None] / 2)
                        if len(dt) > 1 else np.empty((0, positions.shape[1])))
        for j, name in enumerate(track["joint_names"]):
            bound = limits.get(component, {}).get(name, {})
            lower = bound.get("lower", bound.get("min"))
            upper = bound.get("upper", bound.get("max"))
            speed = bound.get("velocity")
            accel = bound.get("acceleration")
            def finite(value):
                return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
            complete = (all(finite(v) for v in (lower, upper, speed, accel))
                        and lower < upper and speed > 0 and accel > 0)
            if hardware and not complete:
                errors.append(f"{component}/{name}: 位置/速度/加速度限制未确认")
            if any(v is not None and not finite(v) for v in (lower, upper, speed, accel)):
                errors.append(f"{component}/{name}: 限制必须为有限数值")
                continue
            if lower is not None and np.any(positions[:, j] < lower - 1e-9):
                errors.append(f"{component}/{name}: 低于位置下限")
            if upper is not None and np.any(positions[:, j] > upper + 1e-9):
                errors.append(f"{component}/{name}: 超过位置上限")
            if speed is not None and (speed <= 0 or np.any(np.abs(velocity[:, j]) > speed + 1e-9)):
                errors.append(f"{component}/{name}: 超过速度上限或限制未确认")
            if accel is not None and (accel <= 0 or np.any(np.abs(acceleration[:, j]) > accel + 1e-9)):
                errors.append(f"{component}/{name}: 超过加速度上限或限制未确认")
    return errors


def trajectory_digest(trajectory: dict) -> str:
    return hashlib.sha256(json.dumps(trajectory, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def combine_trajectories(documents: list[dict], offsets: dict | None = None) -> dict:
    """Merge independent arm/hand tracks; reject duplicate ownership."""
    combined = {"schema": SCHEMA, "name": "联合时间轴", "tracks": {}, "source": {"inputs": []}, "events": []}
    normalized = []
    for document in documents:
        current = load_trajectory(document)
        normalized.append(current)
        if set(current["tracks"]) & set(combined["tracks"]):
            raise ValueError("联合轨迹中部件重复，请先选择唯一轨迹来源")
        combined["tracks"].update(copy.deepcopy(current["tracks"]))
        combined["source"]["inputs"].append({"name": current["name"], "sha256": trajectory_digest(current)})
        combined["events"].extend(copy.deepcopy(current["events"]))
    if len(normalized) == 1:
        combined["metadata"] = copy.deepcopy(normalized[0]["metadata"])
    else:
        references = [d["metadata"] for d in normalized if "reference_notes" in d["metadata"]]
        if references and all(r["reference_notes"] == references[0]["reference_notes"] for r in references):
            combined["metadata"] = {k: copy.deepcopy(v) for k,v in references[0].items() if k.startswith("reference_notes")}
        elif references:
            combined["metadata"] = {"reference_notes_unavailable": "合并来源的谱面参考不同，未猜测统一音符基准"}
    changed_timing = False
    for component, offset in (offsets or {}).items():
        if component not in combined["tracks"]:
            raise ValueError("偏移部件不在轨迹中: " + component)
        changed_timing |= float(offset) != combined["tracks"][component]["offset"]
        combined["tracks"][component]["offset"] = float(offset)
    if changed_timing and "reference_notes" in combined.get("metadata", {}):
        combined["metadata"].pop("reference_notes")
        combined["metadata"]["reference_notes_unavailable"] = "轨道时间偏移已更改，必须显式提供对齐后的谱面参考"
    return load_trajectory(combined)
