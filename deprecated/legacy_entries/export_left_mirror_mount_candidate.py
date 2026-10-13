#!/usr/bin/env python3
"""Build and export the explicit left mirrored-angle installation candidate.

This is a preview candidate, kept separate from the pinned training archive.
It copies the absolute magnitudes of the right J1..J7 angles to the left,
keeps the existing left sign convention (+,-,-,-,+,+,-), and solves the new
left Link7-to-hand transform against the existing calibrated base.  The
resulting connector offset is stored in a separate connector JSON so the
approved four-hole file is not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tsensei_v3_2_pianist.mjlab_env import arm_options, arm_scene
from tsensei_v3_2_pianist.mjlab_env.arm_scene import SceneArmFollower
from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
from tsensei_v3_2_pianist.mjlab_env.preview_contract import (
    build_source_from_config,
    load_training_config,
)

from preview.scripts.export_arm_mirror_palm_down_scene import (
    figure_data,
    html_page,
    mirror_report,
    plane_report,
    rigid_chain_report,
    site_points,
    static_overview,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh_color(body_name: str):
    if body_name.startswith("arm_follow_"):
        return "#64748b"
    if body_name.startswith("arm_connector"):
        return "#d08a24"
    if body_name.startswith("left_/"):
        return "#2563eb"
    return "#dc2626"


def compiled_mesh_traces(model, data):
    """Return reduced Plotly mesh3d traces from the actual compiled model."""
    traces = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or int(model.geom_group[gid]) != 1:
            continue
        body_name = model.body(int(model.geom_bodyid[gid])).name or ""
        if not (body_name.startswith("arm_follow_left_") or body_name.startswith("arm_follow_right_")
                or body_name.startswith("left_/") or body_name.startswith("right_/")
                or body_name.startswith("arm_connector_visual_")):
            continue
        mesh_id = int(model.geom_dataid[gid])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[va:va + vn].astype(float)
        faces = model.mesh_face[fa:fa + fn].astype(np.int64)
        # Keep the scene responsive while retaining the real visual surfaces.
        step = 0.0022 if body_name.startswith(("left_/", "right_/")) else 0.0028
        _, first, inverse = np.unique(np.round(vertices / step).astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        faces = inverse[faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                      & (faces[:, 0] != faces[:, 2])]
        if not len(faces):
            continue
        vertices = vertices[first]
        R = data.geom_xmat[gid].reshape(3, 3)
        vertices = vertices @ R.T + data.geom_xpos[gid]
        side = "left" if (body_name.startswith(("arm_follow_left_", "left_/", "arm_connector_visual_left"))) else "right"
        traces.append({
            "type": "mesh3d", "x": (vertices[:, 0] * 1000).tolist(),
            "y": (vertices[:, 1] * 1000).tolist(), "z": (vertices[:, 2] * 1000).tolist(),
            "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist(),
            "name": body_name, "legendgroup": side, "showlegend": False,
            "color": _mesh_color(body_name), "opacity": 0.86,
            "flatshading": True, "lighting": {"ambient": 0.45, "diffuse": 0.7, "specular": 0.15},
            "hovertemplate": body_name + "<br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>",
            "_side": side, "_feature": "mesh",
        })
    return traces


def mesh_html(report, mesh_traces, line_traces):
    traces = []
    meta = []
    for t in mesh_traces:
        traces.append({k: v for k, v in t.items() if not k.startswith("_")})
        meta.append({"side": t["_side"], "feature": "mesh"})
    for t, m in zip(line_traces[0], line_traces[1]):
        traces.append(t)
        meta.append(m)
    q = report["pose"]["initial_q_deg"]
    left = q["left"]
    install = report["installation_adjustment"]
    palm = report["validation"]["palm_down"]["left"]
    summary = (f"左 J5/J6/J7 = {left[4]:+.5f}°, {left[5]:+.5f}°, {left[6]:+.5f}°；"
               f"指尖平面倾角 {palm['tilt_from_horizontal_deg']:.6f}°；"
               f"主导局部 Z 时钟 {install['dominant_local_clocking_deg']:+.5f}°")
    buttons = []
    for label, sides in (("全部", {"left", "right", None}), ("左臂/左手", {"left", None}), ("右臂/右手", {"right", None})):
        vis = [m["side"] in sides for m in meta]
        buttons.append({"label": label, "method": "update", "args": [{"visible": vis}, {"title": "真实编译网格：" + label}]})
    for label, feature in (("只看真实网格", "mesh"), ("只看安装链", "mount"), ("只看指尖", "tips")):
        vis = [m["feature"] == feature for m in meta]
        buttons.append({"label": label, "method": "update", "args": [{"visible": vis}, {"title": "真实编译网格：" + label}]})
    rows = "".join(f"<tr><td>J{i}</td><td>{q['right'][i-1]:+.5f}°</td><td>{q['left'][i-1]:+.5f}°</td><td>{abs(q['right'][i-1]):.5f}°</td></tr>" for i in range(1, 8))
    traces_json = json.dumps(traces, ensure_ascii=False, separators=(",", ":"))
    buttons_json = json.dumps(buttons, ensure_ascii=False, separators=(",", ":"))
    report_json = json.dumps(report, ensure_ascii=False)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实编译网格：左臂镜像安装候选</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;background:#f8fafc;color:#172b4d;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}.bar{{background:white;padding:12px 18px;border-bottom:1px solid #d7dee8}}.small{{font-size:13px;color:#52677f}}.grid{{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:0}}#plot{{height:86vh;min-height:620px}}.panel{{background:white;padding:16px;border-left:1px solid #d7dee8;line-height:1.45}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #e5eaf0;padding:4px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.tag{{display:inline-block;padding:2px 6px;border-radius:4px;background:#e8eef7;margin:2px 0}}</style></head>
<body><div class="bar"><b>真实 MJLab 编译场景：左臂镜像角度 + 掌面朝下安装</b><br><span class="small">{summary}</span><br><span class="small">显示的是实际机械臂、连接件、A3.2 手部可视网格；可拖动旋转、滚轮缩放、点击按钮切换对象。Plotly 首次打开需要联网加载。</span></div>
<div class="grid"><div id="plot"></div><aside class="panel"><h3>镜像关节角</h3><table><thead><tr><th>关节</th><th>右臂</th><th>左臂</th><th>绝对值</th></tr></thead><tbody>{rows}</tbody></table><h3>掌面朝下安装</h3><div class="tag">左指尖平面倾角：{palm['tilt_from_horizontal_deg']:.6f}°</div><br><div class="tag">局部 Z 时钟：{install['dominant_local_clocking_deg']:+.5f}°</div><p class="small">Link7→手：<br>xyz = [{install['left_link7_to_hand']['xyz_m'][0]*1000:+.2f}, {install['left_link7_to_hand']['xyz_m'][1]*1000:+.2f}, {install['left_link7_to_hand']['xyz_m'][2]*1000:+.2f}] mm</p><p class="small">页面使用固定当前左基座的候选装配；四孔紧固、法兰时钟和实机零位仍需硬件复核。</p></aside></div>
<script>const traces={traces_json};const buttons={buttons_json};const layout={{title:'真实编译网格：全部',margin:{{l:0,r:0,t:44,b:0}},scene:{{aspectmode:'data',xaxis:{{title:'X (mm)'}},yaxis:{{title:'Y (mm)'}},zaxis:{{title:'Z (mm)'}},camera:{{eye:{{x:1.45,y:1.45,z:1.05}}}}}},legend:{{orientation:'h',y:1.02}},updatemenus:[{{type:'buttons',direction:'right',x:0.01,y:1.08,buttons:buttons}}]}};Plotly.newPlot('plot',traces,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});</script><script id="audit-report" type="application/json">{report_json}</script></body></html>'''


def mesh_static_overview(model, data, path: Path, q_deg, report):
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    meshes = compiled_mesh_traces(model, data)
    points = []
    for mesh in meshes:
        v = np.column_stack([mesh["x"], mesh["y"], mesh["z"]])
        faces = np.column_stack([mesh["i"], mesh["j"], mesh["k"]])
        tri = v[faces]
        points.append(v)
        ax.add_collection3d(Poly3DCollection(tri, facecolor=mesh["color"], edgecolor="none", alpha=.72))
    for side, color in (("left", "#1d4ed8"), ("right", "#b91c1c")):
        from preview.scripts.export_arm_mirror_palm_down_scene import chain_names, body, hand_root_name, site_points
        names = chain_names(side)
        arm = np.asarray([body(model, data, n)[:3, 3] for n in names]) * 1000
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], color=color, linewidth=2.5)
        tips = site_points(model, data, side) * 1000
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], color=color, linewidth=1.5)
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], color=color, s=18)
    allp = np.concatenate(points)
    low, high = allp.min(0), allp.max(0); center=(low+high)/2; radius=max(np.ptp(allp,axis=0).max()/2, 220)
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.set_xlabel("X (mm)");ax.set_ylabel("Y (mm)");ax.set_zlabel("Z (mm)");ax.view_init(elev=18,azim=-62)
    ax.set_title(f"Compiled meshes | left J5/J6/J7 = {q_deg['left'][4]:+.2f}, {q_deg['left'][5]:+.2f}, {q_deg['left'][6]:+.2f} deg | palm tilt {report['validation']['palm_down']['left']['tilt_from_horizontal_deg']:.3f} deg")
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path,
                        default=ROOT / "configs/preview/current_arm_training_left_mirror_palm_down_20261011.json")
    parser.add_argument("--connector-config", type=Path,
                        default=ROOT / "configs/preview/hand_connector_left_mirror_palm_down_20261011.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "preview/outputs/arm_left_mirror_mount_candidate_20261011")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {output}")

    connector = args.connector_config.expanduser().resolve()
    # The existing scene source is compiled without arms, then the candidate
    # arm contract is attached in-process.  This avoids changing the pinned
    # source recipe while still using the production arm_scene conversion.
    arm_options.CONNECTOR = connector
    arm_scene.CONNECTOR = connector
    config = load_training_config(args.training_config.expanduser().resolve())
    source_scene, _, _ = build_source_from_config(config)
    bundle = SceneBundle(
        source_scene,
        disable_wrist_tz=bool(config.get("disable_wrist_tz", False)),
        disable_wrist_yaw=bool(config.get("disable_wrist_yaw", False)),
        preparation=config.get("preparation"),
        arm_follow=None,
    )
    arm_scene.augment_bundle(bundle, config["arm_follow"])
    data = bundle.reset_cpu()
    follower = SceneArmFollower(bundle, num_envs=1, device="cpu")
    model = bundle.model
    options = follower.options

    mirror = mirror_report(model, data)
    palm = {side: plane_report(site_points(model, data, side)) for side in ("left", "right")}
    chain = rigid_chain_report(model, data, options)
    q_deg = {side: np.degrees(options["initial_q_rad"][side]).tolist() for side in ("left", "right")}
    report = {
        "schema": "left_mirror_mount_candidate.v1",
        "status": "preview_candidate_not_promoted_to_pinned_training_archive",
        "config": str(args.training_config.expanduser().resolve()),
        "config_sha256": sha256(args.training_config.expanduser().resolve()),
        "connector_config": str(connector),
        "connector_sha256": sha256(connector),
        "temporary_scene": str(source_scene),
        "pose": {"initial_q_deg": q_deg, "source": "candidate arm_follow.initial_q_rad"},
        "mirror_rule": {
            "left_abs_equals_right_abs": True,
            "left_signs": [1, -1, -1, -1, 1, 1, -1],
            "right_to_left": "abs(J1..J7) copied from right; left signs retained/selected as [+,-,-,-,+,+,-]",
        },
        "installation_adjustment": {
            "method": "fixed current calibrated base; solve Link7-to-hand so mirrored q retains the current verified palm-down hand pose",
            "left_link7_to_hand": options["link7_to_hand"]["left"],
            "left_connector_offset": json.loads(connector.read_text())["offsets"]["left"],
            "dominant_local_clocking_deg": -89.99910939151206,
            "hardware_status": "unverified flange clocking and four-hole fastening; candidate only",
        },
        "validation": {
            "mirror": mirror,
            "palm_down": palm,
            "rigid_chain": chain,
            "physics_steps": 0,
            "policy_evaluations": 0,
            "hardware_commands": 0,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                args.training_config.expanduser().resolve(), connector,
                ROOT / "preview/scripts/export_left_mirror_mount_candidate.py",
                ROOT / "preview/scripts/export_arm_mirror_palm_down_scene.py",
                ROOT / "tsensei_v3_2_pianist/mjlab_env/arm_scene.py",
                ROOT / "tsensei_v3_2_pianist/mjlab_env/arm_follow.py",
            )
        },
    }
    line_traces = figure_data(model, data)
    mesh_traces = compiled_mesh_traces(model, data)
    output.mkdir(parents=True, exist_ok=True)
    (output / "left_mirror_mount_candidate.html").write_text(
        mesh_html(report, mesh_traces, line_traces), encoding="utf-8")
    mesh_static_overview(model, data, output / "left_mirror_mount_candidate.png", q_deg, report)
    # Keep the line-only diagnostic alongside the real-mesh deliverable for comparison.
    static_overview(model, data, output / "left_mirror_mount_candidate_lines.png")
    (output / "candidate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(args.training_config.expanduser().resolve(), output / "candidate_training_config.json")
    shutil.copy2(connector, output / "candidate_connector_config.json")
    (output / "README.md").write_text("""# 左臂镜像角度与真实网格安装候选\n\n本候选把右臂 J1–J7 的角度绝对值复制到左臂，左侧符号取 `[+,-,-,-,+,+,-]`，因此 J5/J6/J7 的绝对值严格相同。保持当前已标定的左基座，通过改变左侧 Link7→手的安装变换补偿后三轴姿态，使五个 distal-tip 站点保持与琴键面平行。主导安装时钟调整约为绕工具局部 Z 的 `-90°`，同时包含由 URDF 链带来的小角度与平移补偿。\n\nHTML 中的 `mesh3d` 来自同一次 MuJoCo 编译场景的实际可视网格，包含双侧 M6 连杆、四孔连接件和 A3.2 手部；页面右侧固定显示 J1–J7 的右/左角度、J5/J6/J7 绝对值、Link7→手安装参数和掌面倾角。\n\n这是独立于固定训练归档的预览候选；四孔紧固、法兰时钟和实机编码器零位尚未硬件复核，不能直接称为已生效训练默认。\n\n- `left_mirror_mount_candidate.html`：真实编译网格的 Plotly 3D 场景。\n- `left_mirror_mount_candidate.png`：真实网格静态总览。\n- `candidate_report.json`：镜像、刚性链、掌面平面和来源散列。\n- `candidate_training_config.json` / `candidate_connector_config.json`：候选输入。\n""", encoding="utf-8")
    files = [p for p in output.iterdir() if p.is_file()]
    (output / "manifest.json").write_text(json.dumps({
        "schema": "left_mirror_mount_candidate_manifest.v1",
        "files": {p.name: sha256(p) for p in files},
    }, indent=2) + "\n")
    with zipfile.ZipFile(output / "left_mirror_mount_candidate_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "left_mirror_mount_candidate_bundle.zip":
                archive.write(path, path.name)
    print(json.dumps({
        "output": str(output),
        "html": str(output / "left_mirror_mount_candidate.html"),
        "left_q_deg": q_deg["left"],
        "mirror_position_error_mm": mirror["max_body_position_error_m"] * 1000,
        "left_palm_tilt_deg": palm["left"]["tilt_from_horizontal_deg"],
        "left_link7_hand_error": chain["left"]["link7_to_hand_max_abs_error"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
