#!/usr/bin/env python3
"""Render saved native qpos/mocap states; never act, step physics, or solve IK."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera-view',choices=('front','performer','overview'),default='front')
    p.add_argument('--camera-framing',choices=('full','hands'),default='full')
    p.add_argument('--camera-distance',type=float,default=1.35)
    p.add_argument('--camera-elevation',type=float,default=-25)
    p.add_argument('--audio',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    from tsensei_v3_2_pianist.preview_backend import configure_backend,current_gl_info
    configure_backend(headless=True)
    import mujoco,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from tsensei_v3_2_pianist.mjlab_env.paths import for_checkpoint
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.rl_torch.checkpoint import reward_options,load_agent
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    report=json.loads((a.recording/'report.json').read_text())
    assert sha256(a.checkpoint)==report['checkpoint_sha256']
    scene,task=for_checkpoint(a.checkpoint,None,None);options=reward_options(a.checkpoint)
    env=PianoTaskEnv(scene,task,device='cuda:0',**options)
    load_agent(a.checkpoint,scene_sha256=sha256(scene/'manifest.json'),task_sha256=sha256(task/'task.json'),
        obs_size=env.observation_size,action_size=env.action_size,device='cuda:0',action_contract=env.scene.action_contract,
        control_limits_contract=env.scene.control_limits,finger_clearance_contract=env.scene.finger_clearance,
        finger_coupling_contract=env.scene.finger_coupling,coupling_observation_contract=env.coupling_observation_contract,
        arm_follow_contract=env.scene.arm_follower.contract if env.scene.arm_follower is not None else None,
        preparation_contract=env.scene.bundle.preparation,**options)
    env.reset();model=env.scene.model;data=env.scene.cpu_snapshot()
    with np.load(a.recording/'rollout.npz',allow_pickle=False) as z:trace={k:z[k] for k in z.files}
    assert trace['qpos'].shape==(env.logic.length,model.nq)
    assert trace['mocap_pos'].shape[1:]==data.mocap_pos.shape
    assert trace['mocap_quat'].shape[1:]==data.mocap_quat.shape
    assert report['preparation_prefix_seconds']==0, 'This renderer requires an already recorded policy preparation'
    camera=mujoco.MjvCamera();settings=configure_camera(camera,model,view=a.camera_view,framing=a.camera_framing,distance=a.camera_distance,elevation=a.camera_elevation)
    camera.lookat[:]=(.30,0,.30);settings['lookat']=camera.lookat.tolist()
    keys=ActiveKeyHighlight(model)
    model.vis.global_.offwidth=max(1280,model.vis.global_.offwidth);model.vis.global_.offheight=max(720,model.vis.global_.offheight)
    a.output.mkdir(parents=True,exist_ok=False);silent=a.output/'silent.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate',str(report['fps']),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)],stdin=subprocess.PIPE)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    frames=0
    with mujoco.Renderer(model,height=720,width=1280,max_geom=4000) as renderer:
        graphics=current_gl_info()
        def draw(t,initial=False):
            renderer.update_scene(data,camera=camera);keys.color_render_scene(renderer.scene,data)
            img=Image.fromarray(renderer.render());d=ImageDraw.Draw(img)
            d.rectangle((0,0,1280,62),fill=(18,23,32));d.text((18,8),'expe00001 | step 500 | 256,000 samples | dual M6 + A3.2',font=font,fill='white')
            d.text((18,36),f't = {t:.2f} s | '+('initial pose' if initial else 'same recorded rollout; green = actual active keys'),font=small,fill=(180,230,195))
            return img,np.asarray(img).tobytes()
        img,frame=draw(0,True);encoder.stdin.write(frame);frames+=1;img.save(a.output/'initial_frame.png')
        for i in range(len(trace['qpos'])):
            data.qpos[:]=trace['qpos'][i];data.ctrl[:]=trace['ctrl'][i];data.mocap_pos[:]=trace['mocap_pos'][i];data.mocap_quat[:]=trace['mocap_quat'][i];data.time=float(trace['policy_time'][i]);mujoco.mj_forward(model,data)
            img,frame=draw(data.time);encoder.stdin.write(frame);frames+=1
            if i==len(trace['qpos'])//3:img.save(a.output/'preview.png')
        for _ in range(round(report['final_hold_seconds']*report['fps'])):encoder.stdin.write(frame);frames+=1
    encoder.stdin.close();assert encoder.wait()==0
    assert frames==report['video_frames']
    video=a.output/'replay.mp4'
    if a.audio:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(silent),'-i',str(a.recording/'played.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(video)],check=True)
        silent.unlink()
    else:silent.rename(video)
    result=dict(source_recording=str(a.recording),checkpoint_sha256=report['checkpoint_sha256'],rollout_sha256=sha256(a.recording/'rollout.npz'),audio_source_sha256=sha256(a.recording/'played.wav') if a.audio else None,frames=frames,camera=settings,graphics=graphics,scope='Read-only native qpos/mocap rendering and mj_forward; no policy inference, no mj_step and no IK. Same original audio.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
