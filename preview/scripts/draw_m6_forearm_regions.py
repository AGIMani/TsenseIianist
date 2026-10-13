#!/usr/bin/env python3
"""Render the actual M6 forearm reward convex hulls and save explicit pair list."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Choose new output directory')
    from tsensei_v3_2_pianist.preview_backend import configure_backend
    configure_backend(headless=True)
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview
    from tsensei_v3_2_pianist.preview_arm_collision import color_regions
    env=suite.load();arms=ArmPreview(env.task.root_entity.mjcf_model);env.reset()
    arms.bind(env.physics,(env.task.right_hand,env.task.left_hand))
    m=env.physics.model.ptr;d=env.physics.data.ptr
    color_regions(m,arms.collision)
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=800
    cam=mujoco.MjvCamera();arms.camera(cam)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]=True
    opt.sitegroup[:]=0
    a.output.mkdir(parents=True)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    with mujoco.Renderer(m,height=800,width=1280) as renderer:
        renderer.update_scene(d,camera=cam,scene_option=opt)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]=False
        img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
        draw.rectangle((0,0,1280,102),fill=(20,25,32))
        draw.text((14,8),'forearm 奖励区域：右侧红色／左侧蓝色；肩座安装中心距 93 mm',font=font,fill='white')
        draw.text((14,38),'检测：跨手掌指＋左右机械臂＋机械臂与任一手；排除自身 Link7—掌部安装接触',font=font,fill='white')
        draw.text((14,68),'无碰撞 +0.5／有碰撞 0；不含同臂自碰撞或机械臂—钢琴碰撞；仅检测，不施加接触力',font=font,fill='white')
        img.save(a.output/'forearm_regions.png')
    (a.output/'regions.json').write_text(json.dumps(arms.collision.manifest(),indent=2)+'\n')
    env.close();print(a.output)
if __name__=='__main__':main()
