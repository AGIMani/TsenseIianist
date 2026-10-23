#!/usr/bin/env python3
"""Render a pinned hand's reset pose and single-joint FK sweep inside Slurm.

This is a kinematic inspection, not a policy/physics replay. No mj_step, training,
pose saving, actuator control, or model-asset edits are performed.
Run through the external pytest launcher with scripts/hpc/submit.sh validate.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path


SOURCE_SHA = "9299ab01d69563092b65cd4d91b09429ae99aa501cedbb797b9956bd098fece8"


def render_review(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model compilation and GPU rendering require Slurm.")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.mjlab_env import source_runtime
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.preview_backend import current_gl_info

    root = Path.cwd()
    assert source_runtime.ROOT.resolve() == root.resolve(), "Use the extracted pinned source"
    assert json.loads((root.parent / "source_audit.json").read_text())["sha256"] == SOURCE_SHA
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recipe = source_runtime.make_recipe(
        height=.06, ty=.04, yaw=np.pi / 6, control_hz=20, physics_hz=200,
        initial_buffer_time=0, exact_note_timing=True, drive_profile="shadow_v1",
        task_profile="tsensei_collision_v1", collision_profile="convex_hull_v1",
    )
    scene_path, _, _ = source_runtime.build(recipe)
    bundle = SceneBundle(scene_path, disable_wrist_tz=True, disable_wrist_yaw=True)
    model, data = bundle.model, bundle.reset_cpu()
    initial = data.qpos.copy()
    model.vis.global_.offwidth = 1000
    model.vis.global_.offheight = 700
    model.vis.headlight.ambient[:] = [.35, .35, .35]
    model.vis.headlight.diffuse[:] = [.50, .50, .50]
    model.vis.headlight.specular[:] = [.08, .08, .08]
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    option.sitegroup[:] = 0
    for flag in (mujoco.mjtVisFlag.mjVIS_TENDON, mujoco.mjtVisFlag.mjVIS_LIGHT,
                 mujoco.mjtVisFlag.mjVIS_CAMERA):
        option.flags[flag] = False
    body_names = [model.body(i).name for i in range(model.nbody)]
    geom_side = {i: next((s for s in ("left", "right")
                          if body_names[model.geom_bodyid[i]].startswith(s + "_/")), None)
                 for i in range(model.ngeom)}
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 16)
    sides = {}
    for side, expected_axis in (("left", [0, 0, -1]), ("right", [0, 0, 1])):
        joint = model.joint(f"{side}_/{side}_hand_thumb_rotaback_joint2")
        np.testing.assert_allclose(model.jnt_axis[joint.id], expected_axis, atol=1e-12)
        np.testing.assert_allclose(model.jnt_range[joint.id], [0, 1.74], atol=1e-12)
        qadr = int(model.jnt_qposadr[joint.id])
        np.testing.assert_allclose(initial[qadr], .476648962, atol=1e-8)
        body_ids = [i for i, n in enumerate(body_names) if n.startswith(side + "_/")]
        center = (data.xpos[body_ids].min(axis=0) + data.xpos[body_ids].max(axis=0)) / 2
        endpoint = model.body(f"{side}_/{side}_hand_thumb_PIP_link").id
        sides[side] = dict(joint=joint.id, qadr=qadr, center=center, endpoint=endpoint,
                           local_axis=expected_axis)
    angles = np.unique(np.r_[np.linspace(.05, 1.57, 41), initial[sides["left"]["qadr"]]])
    initial_index = int(np.argmin(abs(angles - initial[sides["left"]["qadr"]])))
    report = dict(source_sha256=SOURCE_SHA, job_id=os.environ["SLURM_JOB_ID"],
                  renderer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Kinematic FK inspection of pinned default training reset; no policy or physics steps",
                  physics_steps=0, pose=recipe["pose"], preparation=None,
                  comparison_score=None, notes=[
                      "Only Thumb_Rot2 changes; all other qpos remain identical to reset.",
                      "Single-hand views hide the piano and the other hand for visibility only.",
                      "Gold highlights the thumb chain; blue arrow is the true projected positive axis, drawn through the housing.",
                      "Oblique cameras face the thumb side independently (left azimuth 215, right 145 degrees).",
                      "0.05..1.57 rad is the configured control interval; mechanical range is 0..1.74 rad.",
                      "This reset uses constants (Thumb_Rot2=0.476648962), not the older named initial JSON preset (0.4).",
                      "Frame spacing is for inspection; playback speed is not a commanded hardware velocity.",
                  ], sides={}, angle_rad=angles.tolist(), initial_index=initial_index)
    for side, s in sides.items():
        # Independent derivative check: a positive dq follows axis x radius.
        data.qpos[:] = initial
        mujoco.mj_forward(model, data)
        anchor, axis = data.xanchor[s["joint"]].copy(), data.xaxis[s["joint"]].copy()
        point = data.xpos[s["endpoint"]].copy()
        eps = 1e-6
        data.qpos[s["qadr"]] += eps
        mujoco.mj_forward(model, data)
        plus = data.xpos[s["endpoint"]].copy()
        data.qpos[s["qadr"]] -= 2 * eps
        mujoco.mj_forward(model, data)
        minus = data.xpos[s["endpoint"]].copy()
        actual = (plus - minus) / (2 * eps)
        expected = np.cross(axis, point - anchor)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        report["sides"][side] = dict(joint=model.joint(s["joint"]).name,
                                    local_axis=s["local_axis"], world_axis=axis.tolist(),
                                    initial_rad=float(initial[s["qadr"]]),
                                    initial_deg=float(np.degrees(initial[s["qadr"]])),
                                    limit_rad=[0, 1.74], sweep_rad=[.05, 1.57],
                                    derivative_error=float(np.max(abs(actual - expected))))
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)

    def draw_frame(renderer, side=None, view="oblique", q=None):
        data.qpos[:] = initial
        if side is not None and q is not None:
            data.qpos[sides[side]["qadr"]] = q
        mujoco.mj_forward(model, data)
        unchanged = np.ones(model.nq, dtype=bool)
        if side is not None:
            unchanged[sides[side]["qadr"]] = False
        np.testing.assert_array_equal(data.qpos[unchanged], initial[unchanged])
        cam = mujoco.MjvCamera()
        model.vis.global_.orthographic = 0
        model.vis.global_.fovy = 45
        if side:
            cam.lookat[:] = sides[side]["center"]
            cam.distance = .40
            cam.azimuth, cam.elevation = ((215 if side == "left" else 145), -35) if view == "oblique" else (180, -88)
        else:
            cam.lookat[:] = np.mean([s["center"] for s in sides.values()], axis=0)
            cam.distance = .90
            cam.azimuth, cam.elevation = (180, -40) if view == "oblique" else (180, -88)
        renderer.update_scene(data, camera=cam, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        if side:
            for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                gid = geom.objid
                if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or gid < 0:
                    continue
                if geom_side.get(gid) != side:
                    geom.rgba[3] = 0
                elif "thumb" in body_names[model.geom_bodyid[gid]]:
                    geom.rgba[:] = [.95, .66, .22, 1]
            anchor = data.xanchor[sides[side]["joint"]].copy()
            axis = data.xaxis[sides[side]["joint"]].copy()
        image = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(image)
        if side:
            # Engineering overlay uses the renderer's actual perspective camera.
            # Draw through the shell, since the revolute joint lies inside it.
            cameras = renderer.scene.camera
            eye = (cameras[0].pos + cameras[1].pos) / 2
            forward = np.asarray(cameras[0].forward)
            up = np.asarray(cameras[0].up)
            right = np.cross(forward, up)
            c = cameras[0]
            focal = image.height * c.frustum_near / (c.frustum_top - c.frustum_bottom)
            def project(point):
                v = point - eye
                depth = np.dot(v, forward)
                return np.array([image.width / 2 + focal * np.dot(v, right) / depth,
                                 image.height / 2 - focal * np.dot(v, up) / depth])
            a, b = project(anchor), project(anchor + .045 * axis)
            direction = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            perp = np.array([-direction[1], direction[0]])
            d.line([tuple(a), tuple(b)], fill=(12, 25, 40), width=8)
            d.line([tuple(a), tuple(b)], fill=(65, 180, 255), width=4)
            d.polygon([tuple(b), tuple(b - 13 * direction + 5 * perp),
                       tuple(b - 13 * direction - 5 * perp)], fill=(65, 180, 255))
            d.ellipse((a[0]-4, a[1]-4, a[0]+4, a[1]+4), fill="white", outline="black")
            d.text(tuple(b + direction * 12), "+axis", font=small, fill=(65, 180, 255),
                   stroke_width=2, stroke_fill=(12, 25, 40))
        d.rectangle((0, 0, image.width, 70), fill=(20, 28, 42))
        if side:
            name = "LEFT" if side == "left" else "RIGHT"
            value = initial[sides[side]["qadr"]] if q is None else q
            d.text((16, 10), f"{name} | Thumb_Rot2 = {value:.4f} rad / {np.degrees(value):.2f} deg",
                   fill="white", font=font)
            direction = "-Z" if side == "left" else "+Z"
            d.text((16, 41), f"Local positive axis: {direction} | other joints fixed | {view}",
                   fill=(166, 206, 245), font=small)
        else:
            d.text((16, 10), f"New A3.2 | current training reset | {view}", fill="white", font=font)
            d.text((16, 41), "Thumb_Rot2: 0.47665 rad (27.31 deg) | no physics settling",
                   fill=(166, 206, 245), font=small)
        return image

    def combine(images):
        result = Image.new("RGB", (sum(i.width for i in images), max(i.height for i in images)))
        x = 0
        for img in images:
            result.paste(img, (x, 0))
            x += img.width
        return result

    frames = {"oblique": [], "top": []}
    samples = {}
    animation = []
    with mujoco.Renderer(model, height=600, width=850, max_geom=5000) as renderer:
        report["graphics"] = current_gl_info()
        assert not report["graphics"].get("software_renderer", False)
        init_overview = combine([draw_frame(renderer, view=v) for v in ("oblique", "top")])
        init_detail = combine([draw_frame(renderer, side=s) for s in ("left", "right")])
        sheet = Image.new("RGB", (1700, 1200))
        sheet.paste(init_overview, (0, 0))
        sheet.paste(init_detail, (0, 600))
        sheet.save(output / "initial_pose.png")
        for view in frames:
            for i, q in enumerate(angles):
                pair = combine([draw_frame(renderer, side=s, view=view, q=float(q))
                                for s in ("left", "right")])
                buffer = io.BytesIO()
                pair.save(buffer, format="JPEG", quality=85)
                frames[view].append(base64.b64encode(buffer.getvalue()).decode())
                if view == "oblique":
                    animation.append(pair.resize((1020, 360)))
                    if i in (0, initial_index, len(angles) - 1):
                        samples[i] = pair
        sweep_sheet = Image.new("RGB", (1700, 1800))
        for row, i in enumerate((0, initial_index, len(angles) - 1)):
            sweep_sheet.paste(samples[i], (0, row * 600))
        sweep_sheet.save(output / "thumb_direction.png")
    animation[0].save(output / "thumb_direction.gif", save_all=True,
                      append_images=animation[1:] + animation[-2:0:-1], duration=90, loop=0)
    np.savez_compressed(output / "pose_arrays.npz", initial_qpos=initial,
                        sweep_angles_rad=angles,
                        thumb_qpos_indices=[sides[s]["qadr"] for s in ("left", "right")])
    (output / "source_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_html(output, frames, angles.tolist(), initial_index)
    print(json.dumps(dict(output=str(output), frames_per_view=len(angles),
                          job=report["job_id"], derivative_checks=report["sides"]), indent=2))


def write_html(output, frames, angles, initial_index):
    initial = base64.b64encode((output / "initial_pose.png").read_bytes()).decode()
    payload = json.dumps(dict(frames=frames, angles=angles, initial=initial_index))
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>新 A3.2 · 初始姿态与拇指方向</title>
<style>
body{margin:0;background:#101824;color:#e9eef8;font:16px system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 10px}
p{line-height:1.7;color:#b9c8dc}section{margin:22px 0;background:#172335;padding:20px;border-radius:14px}
img{width:100%;display:block;border-radius:8px}button,select{padding:10px 14px;background:#293d57;color:white;border:1px solid #58718f;border-radius:7px;font-size:15px;cursor:pointer}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}
input{width:min(600px,100%)}output{font-variant-numeric:tabular-nums;font-weight:600}
details{margin-top:24px}summary{cursor:pointer;color:#a8d8ff}small{color:#9eb1c9}
</style><main><h1>新 A3.2：初始姿态与拇指第二关节</h1>
<p>基于当前固定 v6 模型。左手第二旋转轴为局部 <b>−Z</b>，右手为局部 <b>+Z</b>。
角度增大按各自正轴的右手定则旋转。金色是拇指链，蓝色箭头按真实投影透视标出正轴。
左右特写分别从拇指侧观察，不能仅凭画面上的顺／逆时针比较。</p>
<section><h2>单关节方向检查</h2>
<p>左右两栏分别仅改变该手的 Thumb_Rot2，其余关节保持当前训练初始角。
特写隐藏琴键及另一手；不改变模型几何。初始角为 <b>0.476649 rad（27.31°）</b>。</p>
<div class="controls"><button id="play">播放角度变化</button><button id="reset">回到初始角</button>
<select id="view"><option value="oblique">斜视特写</option><option value="top">俯视特写</option></select>
<output id="angle"></output></div><input id="slider" type="range" min="0" step="1">
<p><small>左端 0.05 rad（2.86°） → 右端 1.57 rad（89.95°）；机械限位为 0～1.74 rad。</small></p>
<img id="frame" alt="左手与右手的拇指第二关节角度预览"></section>
<section><h2>当前训练初始姿态</h2><p>上排为双手斜视与俯视，下排为左右手特写。
对应统一入口 height=0.06 m、ty=0.04 m、yaw=30°，无臂、无准备段，尚未执行物理步。</p>
<img src="data:image/png;base64,__INITIAL__" alt="新模型的当前训练初始姿态"></section>
<details><summary>来源与预览范围</summary><p>归档 SHA256：__SHA__</p>
<p>本页是远端 Slurm 中 MuJoCo 正向运动学渲染，滑块重放已渲染角度。没有运行策略、物理仿真或控制硬件；
播放速度不代表执行器速度。该页不评估整曲演奏，因此没有 MIDI/按键轨迹。
初始姿态取当前训练常量，不是旧的 initial JSON 命名预设（该预设第二关节为 0.4 rad）。</p></details>
</main><script>const data=__PAYLOAD__;
const slider=document.querySelector('#slider'),view=document.querySelector('#view'),frame=document.querySelector('#frame');
slider.max=data.angles.length-1;slider.value=data.initial;let timer=null,dir=1;
function show(){let i=+slider.value,q=data.angles[i];frame.src='data:image/jpeg;base64,'+data.frames[view.value][i];
document.querySelector('#angle').textContent=q.toFixed(4)+' rad / '+(q*180/Math.PI).toFixed(2)+'°';}
function stop(){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放角度变化';}
slider.addEventListener('input',()=>{stop();show()});view.addEventListener('change',show);
document.querySelector('#reset').onclick=()=>{stop();slider.value=data.initial;show()};
document.querySelector('#play').onclick=()=>{if(timer){stop();return}document.querySelector('#play').textContent='暂停';
timer=setInterval(()=>{let i=+slider.value+dir;if(i>=data.angles.length){dir=-1;i=data.angles.length-2}
if(i<0){dir=1;i=1}slider.value=i;show()},100)};show();</script></html>"""
    (output / "index.html").write_text(
        html.replace("__INITIAL__", initial).replace("__SHA__", SOURCE_SHA).replace("__PAYLOAD__", payload))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    render_review(p.parse_args().output)
