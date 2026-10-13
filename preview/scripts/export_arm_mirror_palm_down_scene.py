#!/usr/bin/env python3
"""Export an interactive, audited snapshot of the current arm training scene.

The export is deliberately built through ``build_bundle_from_config``.  It
does not change the default training configuration or run a policy.  The
current left seed is compared with the recorded right-to-left mirror audit,
the fixed Link7 -> flange -> connector -> hand chain is checked, and the five
distal-tip sites are used to report the palm-down (horizontal fingertip plane)
criterion.  The HTML uses Plotly for orbit/zoom and side/feature toggles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsensei_v3_2_pianist.mjlab_env.arm_scene import FLANGE_IN_LINK7, SceneArmFollower
from tsensei_v3_2_pianist.mjlab_env.preview_contract import (
    build_bundle_from_config,
    load_training_config,
)

SIDES = ("left", "right")
SUFFIX = {"left": "L", "right": "R"}
MIRROR = np.diag([1.0, -1.0, 1.0])
TIP_NAMES = ("thumb_tip", "index_tip", "mid_tip", "ring_tip", "pinky_tip")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(data, body_id: int) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = data.xmat[body_id].reshape(3, 3)
    out[:3, 3] = data.xpos[body_id]
    return out


def body(model, data, name: str) -> np.ndarray:
    return pose(data, int(model.body(name).id))


def chain_names(side: str):
    s = SUFFIX[side]
    return [f"arm_follow_{side}_Base_{s}"] + [
        f"arm_follow_{side}_Link{i}_{s}" for i in range(1, 8)
    ]


def mount_names(side: str):
    hand = "Right_hand_palm_link" if side == "right" else "Left_hand_palm_link"
    return [*chain_names(side), f"arm_flange_{side}", f"arm_connector_{side}", f"{side}_/{hand}"]


def hand_root_name(side: str) -> str:
    return f"{side}_/base_link"


def site_points(model, data, side: str):
    return np.asarray([
        data.site_xpos[model.site(f"{side}_/{name}").id].copy() for name in TIP_NAMES
    ])


def plane_report(points: np.ndarray):
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    tilt = math.degrees(math.acos(float(np.clip(normal @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))))
    return {
        "center_m": center.tolist(),
        "normal_toward_world_plus_z": normal.tolist(),
        "tilt_from_horizontal_deg": tilt,
        "z_range_m": [float(points[:, 2].min()), float(points[:, 2].max())],
        "criterion": "five distal-tip sites coplanar with the piano/world horizontal plane",
    }


def mirror_report(model, data):
    rows = []
    position_errors = []
    for name_l, name_r in zip(chain_names("left"), chain_names("right")):
        tl, tr = body(model, data, name_l), body(model, data, name_r)
        expected_p = MIRROR @ tr[:3, 3]
        pe = float(np.linalg.norm(tl[:3, 3] - expected_p))
        position_errors.append(pe)
        rows.append({"left_body": name_l, "right_body": name_r,
                     "position_error_m": pe})
    left_tips, right_tips = site_points(model, data, "left"), site_points(model, data, "right")
    tip_errors = np.linalg.norm(left_tips - right_tips @ MIRROR.T, axis=1)
    left_hand = body(model, data, hand_root_name("left"))
    right_hand = body(model, data, hand_root_name("right"))
    hand_orientation_error = float(np.max(np.abs(left_hand[:3, :3] - MIRROR @ right_hand[:3, :3] @ MIRROR)))
    return {
        "reflection_matrix": MIRROR.tolist(),
        "plane": "world Y=0; p_left = diag(1,-1,1) p_right",
        "body_rows": rows,
        "max_body_position_error_m": max(position_errors),
        "hand_orientation_max_abs_error": hand_orientation_error,
        "tip_site_errors_m": tip_errors.tolist(),
        "max_tip_error_m": float(tip_errors.max()),
    }


def rigid_chain_report(model, data, options):
    errors = {}
    for side in SIDES:
        tf = body(model, data, f"arm_flange_{side}")
        th = body(model, data, hand_root_name(side))
        actual = np.linalg.inv(tf) @ th
        tl7 = body(model, data, chain_names(side)[-1])
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler("xyz", options["link7_to_hand"][side]["rpy_rad"]).as_matrix()
        expected[:3, 3] = np.asarray(options["link7_to_hand"][side]["xyz_m"], dtype=float)
        err = np.max(np.abs(np.linalg.inv(tl7) @ th - expected))
        errors[side] = {
            "link7_to_hand_max_abs_error": float(err),
            "actual_flange_to_hand_m": actual[:3, 3].tolist(),
            "configured_link7_to_hand": options["link7_to_hand"][side],
            "chain": "Link7 > fixed_flange > fixed_four_hole_connector > fixed_hand_base",
        }
    return errors


def figure_data(model, data):
    traces = []
    trace_meta = []

    def add(trace, side=None, feature=None):
        traces.append(trace)
        trace_meta.append({"side": side, "feature": feature})

    for side, color in (("left", "#2563eb"), ("right", "#dc2626")):
        names = chain_names(side)
        points = np.asarray([body(model, data, name)[:3, 3] for name in names]) * 1000.0
        add({"type": "scatter3d", "mode": "lines+markers", "x": points[:, 0].tolist(),
             "y": points[:, 1].tolist(), "z": points[:, 2].tolist(),
             "name": f"{side} arm B→J7", "line": {"color": color, "width": 8},
             "marker": {"color": color, "size": 4},
             "text": ["B"] + [f"J{i}" for i in range(1, 8)], "hovertemplate": "%{text}<br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>"}, side, "arm")
        mount = np.asarray([body(model, data, name)[:3, 3] for name in [chain_names(side)[-1], f"arm_flange_{side}", f"arm_connector_{side}", hand_root_name(side)]]) * 1000.0
        add({"type": "scatter3d", "mode": "lines+markers", "x": mount[:, 0].tolist(),
             "y": mount[:, 1].tolist(), "z": mount[:, 2].tolist(),
             "name": f"{side} rigid F→C→H", "line": {"color": "#d97706", "width": 7, "dash": "dash"},
             "marker": {"color": "#d97706", "size": 5},
             "text": ["J7", "F", "C", "H"], "hovertemplate": "%{text}<br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>"}, side, "mount")
        tips = site_points(model, data, side) * 1000.0
        add({"type": "scatter3d", "mode": "lines+markers", "x": tips[:, 0].tolist(),
             "y": tips[:, 1].tolist(), "z": tips[:, 2].tolist(), "name": f"{side} fingertip plane",
             "line": {"color": color, "width": 4}, "marker": {"color": color, "size": 7},
             "text": list(TIP_NAMES), "hovertemplate": "%{text}<br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>"}, side, "tips")

    white, black = [], []
    for i in range(model.nbody):
        name = model.body(i).name or ""
        if name.startswith("piano/white_key_"):
            white.append(data.xpos[i] * 1000.0)
        elif name.startswith("piano/black_key_"):
            black.append(data.xpos[i] * 1000.0)
    for label, points, color in (("white keys", white, "#e5e7eb"), ("black keys", black, "#111827")):
        p = np.asarray(points)
        add({"type": "scatter3d", "mode": "markers", "x": p[:, 0].tolist(), "y": p[:, 1].tolist(),
             "z": p[:, 2].tolist(), "name": label, "marker": {"color": color, "size": 3}}, None, "piano")
    # The keyboard top outline is a visual guide; key centers above are from the compiled model.
    outline = np.asarray([[-65, -620, 22.5], [45, -620, 22.5], [45, 620, 22.5], [-65, 620, 22.5], [-65, -620, 22.5]])
    add({"type": "scatter3d", "mode": "lines", "x": outline[:, 0].tolist(), "y": outline[:, 1].tolist(),
         "z": outline[:, 2].tolist(), "name": "piano keybed outline", "line": {"color": "#64748b", "width": 3}}, None, "piano")
    return traces, trace_meta


def static_overview(model, data, path: Path):
    """Small static companion image using the same reset data as the HTML."""
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    for side, color in (("left", "#2563eb"), ("right", "#dc2626")):
        names = chain_names(side)
        points = np.asarray([body(model, data, name)[:3, 3] for name in names]) * 1000.0
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.2, label=f"{side} arm")
        mount_names_local = [names[-1], f"arm_flange_{side}", f"arm_connector_{side}", hand_root_name(side)]
        mount = np.asarray([body(model, data, name)[:3, 3] for name in mount_names_local]) * 1000.0
        ax.plot(mount[:, 0], mount[:, 1], mount[:, 2], color="#d97706", linestyle="--", linewidth=1.6)
        tips = site_points(model, data, side) * 1000.0
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color=color, linewidth=1.4, alpha=.8)
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], color=color, s=20)
    white, black = [], []
    for i in range(model.nbody):
        name = model.body(i).name or ""
        if name.startswith("piano/white_key_"):
            white.append(data.xpos[i] * 1000.0)
        elif name.startswith("piano/black_key_"):
            black.append(data.xpos[i] * 1000.0)
    for pts, color, label in ((white, "#cbd5e1", "white keys"), (black, "#111827", "black keys")):
        p = np.asarray(pts)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=color, s=8, label=label)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    ax.set_title("Current MJLab training scene — mirror arms, rigid flange/hand, palm-down tips")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def html_page(report, traces, trace_meta):
    vis = [True] * len(traces)
    payload = json.dumps({"report": report, "traces": trace_meta}, ensure_ascii=False)
    trace_json = json.dumps(traces, ensure_ascii=False, separators=(",", ":"))
    buttons = []
    for label, predicate in (("全部", lambda m: True), ("左手", lambda m: m["side"] in (None, "left") and m["feature"] == "piano" or m["side"] == "left"), ("右手", lambda m: m["side"] in (None, "right") and m["feature"] == "piano" or m["side"] == "right")):
        state = [bool(predicate(m)) for m in trace_meta]
        buttons.append({"label": label, "method": "update", "args": [{"visible": state}, {"title": f"训练场景：{label}"}]})
    for label, predicate in (("只看臂链", lambda m: m["feature"] == "arm"), ("只看安装", lambda m: m["feature"] == "mount"), ("只看指尖", lambda m: m["feature"] == "tips")):
        buttons.append({"label": label, "method": "update", "args": [{"visible": [predicate(m) for m in trace_meta]}, {"title": f"训练场景：{label}"}]})
    buttons_json = json.dumps(buttons, ensure_ascii=False, separators=(",", ":"))
    mirror = report["validation"]["mirror"]
    palm = report["validation"]["palm_down"]["left"]
    chain = report["validation"]["rigid_chain"]
    summary = (f"左臂镜像最大位置误差 {mirror['max_body_position_error_m']*1000:.4f} mm；"
               f"左手指尖平面倾角 {palm['tilt_from_horizontal_deg']:.6f}°；"
               f"左侧 Link7→手最大矩阵误差 {chain['left']['link7_to_hand_max_abs_error']:.3e}")
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MJLab 双机械臂镜像与掌面朝下训练场景</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#182e46;background:#f8fafc}}#plot{{width:100vw;height:82vh}}.bar{{padding:12px 18px;background:white;border-bottom:1px solid #d5dce5}}.small{{color:#53687b;font-size:13px}}code{{font-size:12px}}</style></head>
<body><div class="bar"><b>当前 MJLab 训练场景：左臂镜像、刚性法兰跟随、左手掌面朝下</b><br><span class="small">{summary}</span><br><span class="small">数据来自当前保存配置与同一 CPU 编译场景；拖动旋转、滚轮缩放、点击图例隐藏对象。Plotly 首次打开需要网络加载库。</span></div><div id="plot"></div>
<script>
const traces={trace_json};
const buttons={buttons_json};
const layout={{title:'训练场景：全部',margin:{{l:0,r:0,t:42,b:0}},scene:{{aspectmode:'data',xaxis:{{title:'X (mm)'}},yaxis:{{title:'Y (mm)'}},zaxis:{{title:'Z (mm)'}},camera:{{eye:{{x:1.45,y:1.45,z:1.05}}}}}},legend:{{orientation:'h',y:1.02}} ,updatemenus:[{{type:'buttons',direction:'right',x:0.01,y:1.09,buttons:buttons}}]}};
Plotly.newPlot('plot',traces,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});
</script>
<script id="audit-report" type="application/json">{payload}</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/preview/current_arm_training_20261011.json")
    parser.add_argument("--output", type=Path, default=ROOT / "preview/outputs/arm_mirror_palm_down_scene_20261011")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = args.training_config.expanduser().resolve()
    config = load_training_config(config_path)
    bundle, scene_directory, _ = build_bundle_from_config(config)
    data = bundle.reset_cpu()
    follower = SceneArmFollower(bundle, num_envs=1, device="cpu")
    options = follower.options
    model = bundle.model

    mirror = mirror_report(model, data)
    palm = {side: plane_report(site_points(model, data, side)) for side in SIDES}
    chain = rigid_chain_report(model, data, options)
    q_deg = {side: np.degrees(options["initial_q_rad"][side]).tolist() for side in SIDES}
    report = {
        "schema": "arm_mirror_palm_down_scene.v1",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "temporary_scene": str(scene_directory),
        "pose": {"initial_q_deg": q_deg, "source": "arm_follow.initial_q_rad"},
        "interpretation": {
            "mirror": "The current left seven-angle seed is already the recorded right-to-left FK mirror; no unverified sign-flip was substituted.",
            "flange_follow": "Hand is a fixed descendant of Link7 through flange and four-hole connector; wrist physical DOFs remain removed.",
            "palm_down": "Palm-down is audited as a horizontal five-distal-tip plane parallel to the piano keybed; this is a geometric scene criterion, not hardware calibration.",
        },
        "validation": {"mirror": mirror, "palm_down": palm, "rigid_chain": chain,
                       "physics_steps": 0, "hardware_commands": 0, "policy_evaluations": 0},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            config_path, ROOT / "preview/scripts/export_arm_mirror_palm_down_scene.py",
            ROOT / "tsensei_v3_2_pianist/mjlab_env/arm_scene.py",
            ROOT / "tsensei_v3_2_pianist/mjlab_env/arm_follow.py",
            ROOT / "tsensei_v3_2_pianist/mjlab_env/preview_contract.py",
        ]},
    }
    traces, trace_meta = figure_data(model, data)
    (output / "training_scene_mirror_palm_down.html").write_text(html_page(report, traces, trace_meta), encoding="utf-8")
    static_overview(model, data, output / "training_scene_overview.png")
    (output / "scene_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text("""# 双机械臂镜像与左手掌面朝下场景导出\n\n这是从 `configs/preview/current_arm_training_20261011.json` 通过 `build_bundle_from_config` 重建的 CPU 场景快照。当前配置中的左七轴角已经满足记录的右到左 FK 镜像关系，因此没有把七轴简单全部取反；导出报告给出镜像误差。手根是 Link7 的固定后代，法兰、四孔连接件和手根一起随 Link7 旋转。左手五个 distal-tip 站点与钢琴键面平行，用作掌面朝下的几何验收。\n\n打开 `training_scene_mirror_palm_down.html` 可旋转、缩放并用按钮切换左右手、臂链、安装链和指尖。HTML 使用 Plotly CDN，首次打开需要网络加载 Plotly；机器可读验收在 `scene_report.json`。未运行策略、未推进物理、未发送硬件命令，也未改写默认训练配置。\n""", encoding="utf-8")
    files = [p for p in output.iterdir() if p.is_file()]
    manifest = {"schema": "arm_mirror_palm_down_scene_manifest.v1", "files": {p.name: sha256(p) for p in files}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output / "arm_mirror_palm_down_scene_bundle.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name != "arm_mirror_palm_down_scene_bundle.zip":
                zf.write(p, p.name)
    print(json.dumps({"output": str(output), "html": str(output / "training_scene_mirror_palm_down.html"),
                      "mirror_position_error_mm": mirror["max_body_position_error_m"] * 1000,
                      "left_palm_tilt_deg": palm["left"]["tilt_from_horizontal_deg"],
                      "left_link7_hand_error": chain["left"]["link7_to_hand_max_abs_error"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
