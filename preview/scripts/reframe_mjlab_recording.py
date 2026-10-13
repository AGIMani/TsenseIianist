#!/usr/bin/env python3
"""Change the camera of a saved physical trajectory, preserving its exact audio."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',type=Path,default=None)
    p.add_argument('--task',type=Path,default=None)
    p.add_argument('--camera-view',choices=('front','performer','overview'),default='front')
    p.add_argument('--camera-framing',choices=('full','hands'),default='full')
    p.add_argument('--camera-elevation',type=float,help='Camera elevation in degrees; front defaults to -30')
    p.add_argument('--camera-distance',type=float,help='Orthographic framing scale; full defaults to 1.05')
    a=p.parse_args()
    from tsensei_v3_2_pianist.preview_backend import configure_backend
    configure_backend(headless=True)
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle,sha256
    from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight,FINGER_COLORS,FINGER_NAMES,finger_label,note_name
    report=json.loads((a.recording/'report.json').read_text())
    from tsensei_v3_2_pianist.mjlab_env.paths import for_recording
    a.scene,a.task=for_recording(report,a.scene,a.task)
    if report['scene_sha256']!=sha256(a.scene/'manifest.json') or report['task_sha256']!=sha256(a.task/'task.json'):
        raise ValueError('Recorded trajectory scene/task mismatch')
    if report.get('initial_frame_seconds')!=1/report['fps']:
        raise ValueError('This reframer requires recordings with an initial frame')
    if a.output.exists():p.error('Output already exists')
    bundle=SceneBundle(a.scene);task=TaskBundle(a.task,bundle);m=bundle.model;d=bundle.reset_cpu()
    rollout=np.load(a.recording/'rollout.npz');n=len(rollout['qpos'])
    if n!=report['metrics']['length']:raise ValueError('Incomplete saved trajectory')
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
    camera=mujoco.MjvCamera();camera_settings=configure_camera(camera,m,view=a.camera_view,framing=a.camera_framing,
                                                            elevation=a.camera_elevation,distance=a.camera_distance)
    keys=ActiveKeyHighlight(m);fingers=FingeringHighlight(m,task.manifest,keys)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',25)
    a.output.mkdir(parents=True)
    has_audio=bool(report.get('audio'));video=a.output/('silent.mp4' if has_audio else 'replay.mp4')
    fps=report['fps'];dt=1/fps;count=0
    ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    f1=[]
    try:
        with mujoco.Renderer(m,width=1280,height=720,max_geom=4000) as renderer:
            for index in range(-1,n):
                if index>=0:
                    d.qpos[:]=rollout['qpos'][index];d.ctrl[:]=rollout['ctrl'][index]
                    if m.nmocap:
                        d.mocap_pos[:]=rollout['mocap_pos'][index]
                        d.mocap_quat[:]=rollout['mocap_quat'][index]
                    mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=camera)
                keys.color_render_scene(renderer.scene,d)
                assigned=task.fingers[max(index,0)]
                start=renderer.scene.ngeom;fingers.append(renderer.scene,d,assigned)
                for geom in renderer.scene.geoms[start:renderer.scene.ngeom]:geom.label=''
                img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,1280,100),fill=(18,23,32))
                view_label = ('full piano' if a.camera_framing=='full' else 'hand detail')+' / '+a.camera_view+(' orthographic' if camera_settings['orthographic'] else ' perspective')
                draw.text((20,10),f'V3.2 | {Path(report["checkpoint"]).name} | {view_label}',font=bold,fill='white')
                if index>=0:
                    active=rollout['actual'][index].astype(bool);goal=rollout['goals'][index,:88].astype(bool)
                    tp=(active&goal).sum();den=2*tp+(active&~goal).sum()+(~active&goal).sum();f1.append(2*tp/den if den else 1.)
                else:active=np.zeros(88,dtype=bool)
                draw.text((20,48),f't={max(0,index+1)*dt:.2f}s   mean key F1={np.mean(f1) if f1 else 0:.4f}   green = physically pressed',font=font,fill='white')
                draw.text((20,75),'Piano audio from actual key strikes and pedal | no score backing track',font=font,fill=(180,230,195))
                draw.rectangle((0,614,1280,720),fill=(18,23,32))
                for f in range(5):draw.text((20+245*f,622),f'{f+1}: {FINGER_NAMES[f]}',font=font,fill=tuple(int(v*255) for v in FINGER_COLORS[f,:3]))
                target='  '.join(f'{finger_label(f)}:{note_name(k)}' for f,k in zip(*np.nonzero(assigned))) or 'rest'
                draw.text((20,651),'Target / fingering: '+target,font=font,fill='white')
                draw.text((20,679),'Pressed: '+(', '.join(note_name(k) for k in np.flatnonzero(active)) or 'none'),font=font,fill=(90,235,100))
                frame=np.asarray(img).tobytes();ff.stdin.write(frame);count+=1
                if index==n//3:img.save(a.output/'preview.png')
            for _ in range(round(report['final_hold_seconds']*fps)):ff.stdin.write(frame);count+=1
    finally:
        ff.stdin.close();code=ff.wait()
    if code or count!=report['video_frames']:raise RuntimeError('Frame count / encoder failure')
    if has_audio:
        shutil.copy2(a.recording/'played.wav',a.output/'played.wav')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(video),'-i',str(a.output/'played.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(a.output/'replay.mp4')],check=True)
        video.unlink()
    for name in ('rollout.npz','audio_substeps.npz','audio_events.json','played.mid'):
        if (a.recording/name).exists():shutil.copy2(a.recording/name,a.output/name)
    report.update(camera=camera_settings,source_recording=str(a.recording.resolve()),reused_physical_trajectory=True,video='replay.mp4')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(a.output/'replay.mp4')


if __name__=='__main__':main()
