#!/usr/bin/env python3
"""Export a wrist translation/rotation demo of preview differential IK, not a policy."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output directory')
    from tsensei_v3_2_pianist.preview_backend import configure_backend
    configure_backend(headless=True)
    import numpy as np
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview
    from tsensei_v3_2_pianist.preview_geometry import configure_options
    env = suite.load()
    arms = ArmPreview(env.task.root_entity.mjcf_model)
    for hand in (env.task.right_hand,env.task.left_hand):
        for geom in hand.mjcf_model.find_all('geom'):
            if geom.group == 1: geom.rgba = (.94,.95,.97,1)
    env.reset();physics = env.physics
    arms.bind(physics,(env.task.right_hand,env.task.left_hand))
    model = physics.model.ptr
    model.vis.headlight.ambient[:] = (.45,.45,.45)
    camera = mujoco.MjvCamera();arms.camera(camera)
    option = mujoco.MjvOption();configure_options(option)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    a.output.mkdir(parents=True)
    encoder = subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo',
        '-pixel_format','rgb24','-video_size','1280x720','-framerate','30','-i','pipe:0',
        '-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',
        str(a.output/'arm_follow.mp4')],stdin=subprocess.PIPE)
    records=[]
    try:
        with mujoco.Renderer(model,height=720,width=1280) as renderer:
            for frame in range(150):
                phase=2*np.pi*frame/149
                for side,sign in [('R',1),('L',-1)]:
                    values={'TX':sign*.055*np.sin(phase),'TY':.045*(1-np.cos(phase))/2,
                            'TZ':-.012*(np.sin(phase)**2),'Pitch':.08*np.sin(phase),
                            'Yaw':.25-.15*(1-np.cos(phase))/2}
                    for channel,value in values.items():
                        aid=next(i for i in range(model.nu) if model.actuator(i).name.endswith(f'{side}_Wrist_{channel}'))
                        jid=model.actuator_trnid[aid,0]
                        physics.data.qpos[model.jnt_qposadr[jid]]=value
                physics.forward();arms.update(1/30)
                record=arms.report()['arms'];records.append(record)
                renderer.update_scene(physics.data.ptr,camera=camera,scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]=False
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,1280,76),fill=(22,27,34))
                draw.text((15,8),'M6＋V3.2：手腕平移／旋转 → 双臂差分IK跟随（预览演示，非训练策略）',font=font,fill='white')
                draw.text((15,40),f'右腕误差 {record["right"]["position_error_mm"]:.3f} mm  |  左腕误差 {record["left"]["position_error_mm"]:.3f} mm  |  肩座中心距 {arms.config["shoulder_spacing_m"]*1000:.0f} mm',font=font,fill='white')
                encoder.stdin.write(np.asarray(image).tobytes())
                if frame==37:image.save(a.output/'motion.png')
    finally:
        encoder.stdin.close();code=encoder.wait();env.close()
    if code:raise RuntimeError(f'ffmpeg exit {code}')
    summary=dict(frames=150,fps=30,seconds=5,mode='kinematic wrist target demo; not training',
                 max_position_error_mm={s:max(r[s]['position_error_mm'] for r in records) for s in ('right','left')},
                 max_orientation_error_deg={s:max(r[s]['orientation_error_deg'] for r in records) for s in ('right','left')})
    (a.output/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'tracking.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
