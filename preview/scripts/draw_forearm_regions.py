#!/usr/bin/env python3
"""Draw the exact frozen task collision groups used by forearm_reward."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--task', type=Path, help='Visualize a new task on the same recorded physical scene')
    parser.add_argument('--time', type=float, default=21.575)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    from tsensei_v3_2_pianist.preview_backend import configure_backend
    configure_backend(headless=True)
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.mjlab_env.paths import for_recording
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
    from tsensei_v3_2_pianist.forearm_collision import task_collision_groups
    from tsensei_v3_2_pianist.preview_camera import configure_camera

    report = json.loads((args.recording / 'report.json').read_text())
    scene_path, task_path = for_recording(report)
    task_path = args.task or task_path
    task = json.loads((task_path / 'task.json').read_text())
    if task['scene_manifest_sha256'] != sha256(scene_path / 'manifest.json'):
        parser.error('Task must use the same frozen physical scene as the recording')
    bundle = SceneBundle(scene_path)
    model, data = bundle.model, bundle.reset_cpu()
    names = task_collision_groups(task)
    expanded = 'forearm_collision_geoms' in task
    ids = [[model.geom(name).id for name in side] for side in names]
    selected = set(ids[0] + ids[1])
    rollout = np.load(args.recording / 'rollout.npz')
    dt = bundle.manifest['control_dt']
    index = round(args.time / dt) - 1
    if not 0 <= index < len(rollout['qpos']):
        parser.error('--time must identify a saved post-step frame')
    args.output.mkdir(parents=True)
    # Change appearance only in this diagnostic model. Use actual convex collision
    # meshes, not a guessed box around the visible palm or a new contact mask.
    model.vis.global_.offwidth = 1000
    model.vis.global_.offheight = 650
    original_rgba = model.geom_rgba.copy()
    original_group = model.geom_group.copy()
    camera = mujoco.MjvCamera()
    configure_camera(camera, model, view='front', framing='hands', distance=.50, elevation=-55)
    camera.lookat[:] = (.065, 0, .11)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    option.sitegroup[:] = 0
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 24)
    small = ImageFont.truetype(font_path, 20)
    title_font = ImageFont.truetype(font_path, 32)
    canvas = Image.new('RGB', (2000, 1540), (19, 24, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 12), 'forearm_reward 检测区域：' + ('左右掌部＋全部手指碰撞凸包' if expanded else '仅左右掌部碰撞凸包'), font=title_font, fill='white')
    draw.text((25, 62), '红色＝右手检测区域     蓝色＝左手检测区域     仅红蓝之间接触触发；同手接触与琴键接触不计入', font=font, fill='white')
    measurements = []
    with mujoco.Renderer(model, height=650, width=1000) as renderer:
        for row, (state_name, qpos) in enumerate((('初始姿态 t=0', bundle.reset_qpos),
                                               (f'视频姿态 t={(index+1)*dt:.3f}s', rollout['qpos'][index]))):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            contacts = [float(c.dist) for c in data.contact
                        if ((int(c.geom[0]) in ids[0] and int(c.geom[1]) in ids[1]) or
                            (int(c.geom[1]) in ids[0] and int(c.geom[0]) in ids[1]))
                        and c.dist <= task['collision_margin']]
            measurements.append(dict(state=state_name, forearm_contacts=contacts))
            for col in range(2):
                model.geom_rgba[:] = original_rgba
                model.geom_group[:] = original_group
                option.geomgroup[:] = 0
                option.geomgroup[5] = 1
                if col == 0:
                    option.geomgroup[1] = 1  # Visual hand meshes as context.
                for gid in range(model.ngeom):
                    if gid in selected:
                        model.geom_group[gid] = 5
                        model.geom_rgba[gid] = (1, .20, .16, 1) if gid in ids[0] else (.12, .50, 1, 1)
                    elif original_group[gid] == 1:
                        model.geom_rgba[gid] = (.72, .75, .80, .30)
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
                picture = Image.fromarray(renderer.render())
                filename = f'{"initial" if row == 0 else "recorded"}_{"context" if col == 0 else "regions_only"}.png'
                picture.save(args.output / filename)
                x, y = col*1000, 110+row*700
                canvas.paste(picture, (x, y+45))
                draw.text((x+20, y), state_name + ('｜叠加在半透明手模型上' if col == 0 else '｜仅显示奖励检测凸包'), font=font, fill='white')
    draw.text((25, 1510), '用保存姿态展示所选任务的检测范围；绘图不修改原视频。无跨手接触 +0.5，出现接触 0。', font=small, fill='white')
    canvas.save(args.output / 'forearm_regions.png')
    (args.output / 'regions.json').write_text(json.dumps(dict(
        source_recording=str(args.recording.resolve()), scene=str(scene_path), task=str(task_path),
        forearm_collision_geoms=names, scope=task.get('forearm_collision_scope', 'palms_only_legacy'),
        geom_ids=ids, collision_margin=task['collision_margin'],
        note='Native MuJoCo convex-hull display at saved GPU qpos; geometry only, not GPU force.',
        states=measurements), ensure_ascii=False, indent=2)+'\n')
    print(args.output / 'forearm_regions.png')


if __name__ == '__main__':
    main()
