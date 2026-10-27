#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
"""Slurm-only static frames of the current rigid assembly and actual IK updates.

This is a controller-component demonstration, not a policy or physics replay.
Finger qpos is held at reset. A prescribed post-control-shaping palm request is
fed into the production requests/step/publish path at its configured timestep.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tsensei_v3_2_pianist.initial_world_pose import (
        add_arguments as add_pose_arguments, reject_legacy_pose_arguments, WORLD_POSE_CONTRACT)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True)
    p.add_argument('--ik-backend', choices=('torch', 'numba_cpu'), default='torch')
    add_pose_arguments(p)
    p.add_argument('--scene-realtime', action='store_true', help='Pace four XYZ cycles including scene update before rendering')
    p.add_argument('--realtime-clock', choices=('sleep','pinned_busy'), default='sleep')
    p.add_argument('--realtime-seconds', type=int, default=56)
    p.add_argument('--benchmark-only', action='store_true')
    p.add_argument('--latency-profile', action='store_true')
    p.add_argument('--pin-cpu-index', type=int, default=0)
    p.add_argument('--single-thread-pools', action='store_true')
    p.add_argument('--video', action='store_true', help='Render an XYZ controller demonstration instead of static frames')
    reject_legacy_pose_arguments(sys.argv[1:] if argv is None else argv, p)
    args = p.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Render through scripts/hpc/submit.sh in a Slurm allocation')
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    os.environ['MUJOCO_GL'] = 'egl'
    if args.single_thread_pools:
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
            os.environ[key]='1'
    import mujoco
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    if args.single_thread_pools:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    from tsensei_v3_2_pianist.mjlab_env import source_runtime
    from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_bundle_from_config
    from tsensei_v3_2_pianist.mjlab_env.arm_scene import SceneArmFollower
    from tsensei_v3_2_pianist.preview_backend import current_gl_info

    original = json.loads((root/'configs/preview/current_arm_training_20261011.json').read_text())
    config = copy.deepcopy(original)
    # This fresh Cartesian diagnostic has fixed Pitch; the shared hand-only
    # preset retains its active Pitch limits. Never alter a checkpoint config.
    config["control_limits"]["channels"] = {k: v for k, v in config["control_limits"]["channels"].items()
        if not k.endswith("_Pitch")}
    recipe = config['source_runtime']
    # This is explicitly a new workspace diagnostic config, never a migrated
    # checkpoint. Re-express the template's old height=.10/Ty=.04 position
    # as world X=.14/Z=.10, then apply explicitly requested world coordinates.
    recipe['pose'] = dict(world_x=args.world_x, world_z=args.world_z,
                          right_world_y=args.right_world_y, left_world_y=args.left_world_y,
                          palm_base_yaw=args.palm_base_yaw)
    recipe['pose_contract'] = WORLD_POSE_CONTRACT
    recipe['model']['wrist_pitch_range_scale'] = 1.0
    recipe['source_sha256'] = source_runtime.source_files()
    recipe['versions'] = source_runtime.dependency_versions()
    config['diagnostic_provenance'] = {
        'purpose': 'new current-workspace geometry/IK demonstration; no checkpoint',
        'initial_world_pose': recipe['pose'],
        'pose_origin': 'fresh diagnostic from dated template; old height/Ty re-expressed as world coordinates; no checkpoint migration',
        'overrides': {'wrist_pitch_range_scale': 1.0},
        'fingerprints': 'new workspace source/dependencies, not a historical replay',
    }
    config['arm_follow']['execution_backend'] = args.ik_backend
    (out/'render_config.json').write_text(json.dumps(config, indent=2)+'\n')
    print('Building current SceneBundle with wuji_cartesian_v1', flush=True)
    bundle, _, _ = build_bundle_from_config(config)
    model, data = bundle.model, bundle.reset_cpu()
    follower_type = SceneArmFollower
    follower_device = 'cuda:0'
    if args.ik_backend == 'numba_cpu':
        from tsensei_v3_2_pianist.mjlab_env.arm_native_scene import NativeSceneArmFollower
        follower_type, follower_device = NativeSceneArmFollower, 'cpu'
    follower = follower_type(bundle, num_envs=1, device=follower_device)
    model.vis.global_.offwidth = 1200
    model.vis.global_.offheight = 820
    model.vis.headlight.ambient[:] = (.35, .35, .35)
    model.vis.headlight.diffuse[:] = (.35, .35, .35)
    model.vis.headlight.specular[:] = (.05, .05, .05)
    model.light_diffuse[:] *= .65
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    option.sitegroup[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    fonts = {n: ImageFont.truetype(str(args.font), n) for n in (19, 23, 26, 30, 38)}
    rgb = [(1., .18, .16, 1.), (.18, .9, .32, 1.), (.2, .5, 1., 1.)]
    report = {'scope': __doc__, 'job_id': os.environ['SLURM_JOB_ID'],
              'controller': follower.contract, 'frames': [], 'request_delta_m': [0., .020, .010],
              'physics_steps': 0, 'held_fingers': True,
              'input_stage': 'prescribed displacement after upstream policy mapping/control shaping',
              'simulation_dt_s': follower.dt, 'scene_realtime': args.scene_realtime, 'realtime_clock': args.realtime_clock,
              'realtime_seconds': args.realtime_seconds,
              'benchmark_only': args.benchmark_only, 'latency_profile':args.latency_profile,
              'pin_cpu_index':args.pin_cpu_index, 'single_thread_pools':args.single_thread_pools,
              'execution_backend': args.ik_backend, 'palm_base_yaw_rad': args.palm_base_yaw}

    def publish():
        for side, (positions, rotations) in follower.transforms().items():
            ids = follower.wrists[side]['visual_ids']
            data.mocap_pos[ids] = positions[0].cpu().numpy()
            data.mocap_quat[ids] = rotations[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        for side, a in follower.sides.items():
            palm, rot, *_ = follower.fk(a, a['q'])
            bid = follower.wrists[side]['body_id']
            np.testing.assert_allclose(data.xpos[bid], palm[0].cpu().numpy(), atol=1e-8)
            np.testing.assert_allclose(data.xmat[bid].reshape(3, 3), rot[0].cpu().numpy(), atol=1e-8)

    def geom(scene, typ, a, b=None, color=(1,1,1,1), width=.001):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, typ, np.array([width]*3), np.array(a), np.eye(3).ravel(), np.array(color,dtype=np.float32))
        if b is not None:
            mujoco.mjv_connector(g, typ, width, np.array(a), np.array(b))
        scene.ngeom += 1

    overlay_axes = []
    def axes(scene, origin, rotation, length, prefix):
        # Engineering overlay: retain the exact projected origin/directions,
        # but draw through the shell so internal frame origins stay readable.
        overlay_axes.append((np.array(origin).copy(), np.array(rotation).copy(), length, prefix))

    def render(renderer, name, title, detail, side='right', target=None, elapsed=0., fixed_camera=False):
        publish()
        overlay_axes.clear()
        cam = mujoco.MjvCamera()
        bid = follower.wrists[side]['body_id']
        palm = data.xpos[bid].copy()
        flange_id = model.body('arm_flange_'+side).id
        flange = data.xpos[flange_id].copy()
        center = follower.wrists[side]['rest_pos'].cpu().numpy() if fixed_camera else palm
        cam.lookat[:] = center + np.array([-.018, 0., .015]) if detail else [.34, 0., .24]
        cam.distance, cam.azimuth, cam.elevation = (((.42 if fixed_camera else .33), 125., -28.) if detail else (1.40 if fixed_camera else 1.65, 155., -28.))
        renderer.update_scene(data, camera=cam, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        labels = []
        if detail:
            axes(renderer.scene, flange, data.xmat[flange_id].reshape(3,3), .027, 'F')
            axes(renderer.scene, palm, data.xmat[bid].reshape(3,3), .032, 'P')
            labels += [(flange, 'F  法兰原点', (255,219,135)), (palm, 'P  实际 palm_base', (255,255,255))]
            if target is not None:
                tp,tr = target
                axes(renderer.scene, tp, tr, .022, 'T')
                geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, tp, color=(1,.1,1,1), width=.0028)
                geom(renderer.scene, mujoco.mjtGeom.mjGEOM_LINE, palm, tp, (1,.15,1,1), 3.)
                labels += [(tp, 'T  请求的 palm_base', (255,125,255))]
        else:
            axes(renderer.scene, np.zeros(3), np.eye(3), .12, 'W')
            labels += [(np.zeros(3), 'W  世界原点', (255,255,255))]
            for s in ('right','left'):
                b=follower.wrists[s]['body_id']; f=model.body('arm_flange_'+s).id
                axes(renderer.scene, data.xpos[b], data.xmat[b].reshape(3,3), .055, s[0].upper()+'P')
                axes(renderer.scene, data.xpos[f], data.xmat[f].reshape(3,3), .045, s[0].upper()+'F')
                labels += [(data.xpos[b].copy(), ('R' if s=='right' else 'L')+'  palm_base', (255,255,255))]
            if target is not None:
                axes(renderer.scene, target[0], target[1], .055, 'T')
                labels += [(target[0], 'T  右手掌根目标', (255,125,255))]
        # Derive 2-D annotation anchors from MuJoCo's actual render camera.
        cameras=renderer.scene.camera
        eye=(cameras[0].pos+cameras[1].pos)/2
        forward=np.asarray(cameras[0].forward); up=np.asarray(cameras[0].up)
        right=np.cross(forward,up)
        c=cameras[0]; near=c.frustum_near; top=c.frustum_top; bottom=c.frustum_bottom
        def project(point):
            v=np.asarray(point)-eye; depth=np.dot(v,forward)
            focal=820*near/(top-bottom)
            return (600+focal*np.dot(v,right)/depth, 410-focal*np.dot(v,up)/depth)
        picture=Image.fromarray(renderer.render())
        draw=ImageDraw.Draw(picture)
        for origin,rotation,length,prefix in overlay_axes:
            a=np.array(project(origin))
            for j in range(3):
                b=np.array(project(origin+rotation[:,j]*length))
                direction=b-a; direction/=max(np.linalg.norm(direction),1e-9)
                perp=np.array([-direction[1],direction[0]])
                color=tuple(round(x*255) for x in rgb[j][:3])
                draw.line([tuple(a),tuple(b)],fill=(15,22,30),width=8)
                draw.line([tuple(a),tuple(b)],fill=color,width=4)
                draw.polygon([tuple(b),tuple(b-direction*13+perp*5),tuple(b-direction*13-perp*5)],fill=color)
                if detail or prefix=='W':
                    textpos=b+direction*8
                    draw.text(tuple(textpos),prefix+'.'+'XYZ'[j],font=fonts[19],fill=color,stroke_width=2,stroke_fill=(15,22,30))
            draw.ellipse((a[0]-5,a[1]-5,a[0]+5,a[1]+5),fill=(255,125,255) if prefix=='T' else 'white',outline='black',width=1)
        draw.rectangle((0,0,1200,88), fill=(17,27,41))
        draw.text((24,10), title, font=fonts[30], fill='white')
        draw.text((24,54), 'X 红 / Y 绿 / Z 蓝 · 金色：四孔连接件 · 轴按真实投影穿透显示', font=fonts[19], fill=(195,210,230))
        boxes=[(28,110),(28,185),(650,110)] if detail else [(30,710),(30,120),(780,120),(780,185)]
        for (point,label,color),(x,y) in zip(labels,boxes):
            px,py=project(point)
            draw.line((x+20,y+30,px,py), fill=color,width=2)
            draw.ellipse((px-4,py-4,px+4,py+4),fill=color)
            draw.rectangle((x-6,y-3,x+350,y+36),fill=(17,27,41))
            draw.text((x,y),label,font=fonts[23],fill=color)
        error = None if target is None else float(np.linalg.norm(target[0]-palm)*1000)
        footer = '刚性链：Link7 → F 法兰 → 四孔连接件 → P 掌根' if error is None else f't = {elapsed:.3f} s   |   请求→实际误差 = {error:.3f} mm   |   200 Hz 同源 IK'
        draw.rectangle((0,765,1200,820),fill=(17,27,41))
        draw.text((24,778),footer,font=fonts[23],fill='white')
        if name is not None:
            picture.save(out/(name+'.png'))
        Tfp=np.eye(4); Tfp[:3,:3]=data.xmat[flange_id].reshape(3,3).T@data.xmat[bid].reshape(3,3)
        Tfp[:3,3]=data.xmat[flange_id].reshape(3,3).T@(palm-flange)
        if name is not None:
            report['frames'].append(dict(name=name,time_s=elapsed,palm_world_m=palm.tolist(),flange_world_m=flange.tolist(),
                flange_to_palm=Tfp.tolist(),target_world_m=None if target is None else target[0].tolist(),error_mm=error))
        return picture

    if args.video:
        # Resolve the sibling explicitly: runpy wrappers and external pytest
        # drivers must never pick another deployment's module on sys.path.
        import importlib.util
        video_path=Path(__file__).resolve().with_name('render_cartesian_video.py')
        spec=importlib.util.spec_from_file_location('current_cartesian_video',video_path)
        video_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(video_module)
        report['video_driver_path']=str(video_path)
        report['video_driver_sha256']=hashlib.sha256(video_path.read_bytes()).hexdigest()
        export_video=video_module.export_video
        export_video(root, out, args.font, bundle, model, data, follower, render, report)
        return

    with mujoco.Renderer(model,width=1200,height=820,max_geom=5000) as renderer:
        report['graphics']=current_gl_info()
        tiles=[render(renderer,'overview','① 当前刚性训练装配：双 M6 / 手掌 / 琴键',False),
               render(renderer,'mount_detail','② 放大安装关系：F 与 P 相隔 29.57 mm',True)]
        physical=torch.as_tensor(bundle.neutral_action[None],dtype=torch.float64,device=follower.device).clone()
        indices=follower.wrists['right']['indices']
        physical[:,indices]=torch.tensor([[0.,.020,.010]],dtype=torch.float64,device=follower.device)
        trace=[]
        preparation_duration=(follower.preparation or {}).get('duration',0.)
        for i in range(1,201):
            elapsed=i*follower.dt
            targets=follower.requests(physical,torch.tensor([preparation_duration+elapsed],device=follower.device))
            follower.step(targets)
            trace.append(dict(time_s=elapsed,q_right=follower.sides['right']['q'][0].tolist(),
                error_m=float(follower.last['right']['position_error'][0])))
            if i in (10,200):
                target=tuple(x[0].cpu().numpy() for x in targets['right'])
                tiles.append(render(renderer,'tracking_'+str(i),
                    '③ 目标已移动，实际装配受速度/加速度限制' if i==10 else '④ 连续 IK 跟随后的实际安装位置',True,target=target,elapsed=elapsed))
                print('frame',i,'error mm',trace[-1]['error_m']*1000,flush=True)
        report['trace']=trace
    # Four physical renders in one explanatory sheet; no artificial CAD geometry.
    sheet=Image.new('RGB',(2400,1850),(240,244,249)); d=ImageDraw.Draw(sheet)
    d.text((35,18),'wuji_cartesian_v1：目标 palm_base → 连续 IK → 刚性装配',font=fonts[38],fill=(15,35,58))
    d.text((35,76),'生产 IK 组件诊断：右手世界 ΔY=+20 mm、ΔZ=+10 mm；固定掌朝向；手指保持 reset；非策略演奏回放',font=fonts[26],fill=(45,65,85))
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*1200,125+(i//2)*820))
    d.text((35,1780),'T 是请求的掌根位姿；P 是执行结果。误差允许存在，法兰—连接件—手掌始终固定连接。',font=fonts[26],fill=(15,35,58))
    sheet.save(out/'wuji_cartesian_installation.png')
    report['source_hashes']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [root/'tsensei_v3_2_pianist/mjlab_env/arm_scene.py',root/'tsensei_v3_2_pianist/mjlab_env/arm_follow.py',root/'configs/actions/wuji_cartesian_v1.json']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',out,flush=True)


if __name__=='__main__':
    main()
