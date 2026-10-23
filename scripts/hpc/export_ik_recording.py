#!/usr/bin/env python3
"""Slurm-only video from recorded IK physics; never reruns the controller."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(recording,output):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Submit GPU rendering through scripts/hpc/submit.sh')
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from tsensei_v3_2_pianist.preview_fingering import FingeringHighlight
    from tsensei_v3_2_pianist.preview_audio import PianoSynth,write_midi
    from preview.scripts.export_video_job import draw_fingering
    recording=Path(recording);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    trace_path=recording/'physical_trace.npz';z=np.load(trace_path,allow_pickle=False)
    report=json.loads((recording/'replay_report.json').read_text());times=z['time'];events=z['original_events']
    b=SceneBundle(Path(report['source']['build'])/'scene',disable_wrist_tz=True,disable_wrist_yaw=True)
    m=b.model;d=b.reset_cpu();task=json.loads((Path(report['source']['build'])/'task/task.json').read_text())
    actual=b.key_activation(z['qpos'])
    if not np.array_equal(actual,z['actual_keys']):raise ValueError('Recorded key activation differs from physical threshold')
    if z['qpos'].shape[1]!=m.nq:raise ValueError('Model/recording shape mismatch')
    for n in ('physical_trace.npz','replay_report.json','arguments.json','thumb_contact_audit.json',
              'independent_reproduction_check.json','parking_pose.json','parking_provenance.json',
              'optimization_runner.py','geometry_helper.py','selective_thumb_summary.json'):
        if (recording/n).exists():shutil.copy2(recording/n,output/n)
    if report['source'].get('geometry_helper_sha256'):
        if not (output/'geometry_helper.py').exists() or sha(output/'geometry_helper.py')!=report['source']['geometry_helper_sha256']:
            raise ValueError('Geometry helper snapshot missing or mismatched')
    if report['source'].get('thumb_idle_reference'):
        (output/'thumb_idle_reference.json').write_text(json.dumps(dict(
            joints_rad=report['source']['thumb_idle_reference'],
            reference_file_sha256=report['source']['thumb_idle_pose_sha256'],
            use='Inactive-thumb IK reference; actual motion remains recorded physics'),indent=2))
    score=Path(report['source']['build'])/'score.proto'
    if sha(score)!=report['source']['score_sha256']:raise ValueError('Recorded fingering score hash mismatch')
    shutil.copy2(score,output/'recorded_fingering.proto')
    revision_dir=os.environ.get('IK_EXPORT_FINGERING_DIRECTORY')
    if revision_dir:
        revision_dir=Path(revision_dir)
        if sha(revision_dir/'015-5_fingered.proto')!=sha(score):raise ValueError('Fingering revision does not match replay')
        shutil.copy2(revision_dir/'015-5_fingering.txt',output/'revised_fingering.txt')
        shutil.copy2(revision_dir/'revision_report.json',output/'fingering_revision_report.json')
    original=Path(__file__).resolve().parents[2]/'tsensei_v3_2_pianist/music/data/015-5.mid'
    if sha(original)!=report['source']['original_midi_sha256']:raise ValueError('Original MIDI hash mismatch')
    shutil.copy2(original,output/'original_015-5.mid')
    generator_hash=report['source'].get('refinement_runner_sha256',report['source']['source_runner_sha256'])
    generator_candidates=[recording.parent/(recording.name+'.runner.py'),
                          recording.parent.parent/'scripts/hpc'/(recording.name+'.runner.py')]
    generator=next((p for p in generator_candidates if p.is_file() and sha(p)==generator_hash),None)
    if generator is not None:shutil.copy2(generator,output/'generation_runner.py')
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'plot_ik_piano_comparison.py'),str(output)],check=True)
    width,height,fps=1600,900,30
    count=int(np.ceil(times[-1]*fps))+1;duration=count/fps
    audio_events=[];previous=np.zeros(88,bool)
    for stamp,active in zip(times,actual):
        for k in np.flatnonzero(previous&~active):audio_events.append(dict(time=float(stamp),type='note_off',note=int(k)+21))
        for k in np.flatnonzero(active&~previous):audio_events.append(dict(time=float(stamp),type='note_on',note=int(k)+21,velocity=127))
        previous=active
    for k in np.flatnonzero(previous):audio_events.append(dict(time=float(times[-1]),type='note_off',note=int(k)+21,end_of_replay=True))
    (output/'actual_key_audio_events.json').write_text(json.dumps(audio_events,indent=2));write_midi(output/'actual_played.mid',audio_events)
    synth=PianoSynth()
    try:audio=synth.render(audio_events,duration,output/'played.wav')
    finally:synth.close()
    if audio['rms']<=1e-6:raise ValueError('Silent audio')
    m.vis.global_.offwidth=max(m.vis.global_.offwidth,width);m.vis.global_.offheight=max(m.vis.global_.offheight,height)
    keys=ActiveKeyHighlight(m);labels=FingeringHighlight(m,task,keys)
    # Frame both moving hands and the score's keyboard region in one fixed view.
    from itertools import product
    signs=np.array(list(product((-1.,1.),repeat=3)))
    kmin=max(int(events[:,2].min())-4,0);kmax=min(int(events[:,2].max())+4,87)
    key_geoms={m.geom(task['key_geoms'][k]).id for k in range(kmin,kmax+1)}
    points=[]
    for i in np.linspace(0,len(times)-1,25).astype(int):
        d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d)
        for g in range(m.ngeom):
            body=m.body(m.geom_bodyid[g]).name
            if m.geom_group[g]>=2 or not (g in key_geoms or body.startswith(('right_/','left_/'))):continue
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE:continue
            local=m.geom_aabb[g,:3]+signs*m.geom_aabb[g,3:]
            points.append(local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    points=np.concatenate(points);camera=mujoco.MjvCamera();camera.azimuth=0;camera.elevation=-50;camera.distance=2.5;camera.orthographic=1
    el=np.deg2rad(camera.elevation);right=np.array([0.,-1.,0.]);up=np.array([-np.sin(el),0.,np.cos(el)])
    hp=points@right;vp=points@up;caption,footer=68,100
    ortho=1.12*max(float(np.ptp(hp))*height/width,float(np.ptp(vp))*height/(height-caption-footer))
    center=(points.min(0)+points.max(0))/2
    center+=right*((hp.min()+hp.max())/2-center@right)
    center+=up*((vp.min()+vp.max())/2+ortho*(caption-footer)/(2*height)-center@up)
    camera.lookat[:]=center;m.vis.global_.orthographic=1;m.vis.global_.fovy=ortho
    option=mujoco.MjvOption();option.geomgroup[2:5]=0;option.sitegroup[:]=0
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise FileNotFoundError('FFmpeg not available in allocated job')
    silent=output/'replay.silent.mp4';video=output/'replay.mp4'
    encoder=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
    font=ImageFont.load_default(size=24);small=ImageFont.load_default(size=19)
    try:
        with mujoco.Renderer(m,height=height,width=width,max_geom=4000) as renderer:
            for frame in range(count):
                at=min(frame/fps,float(times[-1]));i=int(np.clip(np.searchsorted(times,at,side='right')-1,0,len(times)-1))
                d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i];d.time=float(times[i]);mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=camera,scene_option=option);keys.color_render_scene(renderer.scene,d)
                assigned=z['assignment'][i];labels.append(renderer.scene,d,assigned)
                picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
                draw.rectangle((0,0,width,caption),fill=(18,25,36))
                draw.text((14,7),f'015-5 | physical IK | {recording.name} | t = {at:.3f} s | 1x',font=font,fill='white')
                status='target 0.95 reached' if report.get('acceptance_passed') else 'target 0.95 not reached'
                draw.text((14,38),f'Original MIDI: micro-F1 {report["original_midi_micro_f1"]:.4f}, mean frame F1 {report["original_midi_mean_frame_f1"]:.4f} | actual-key audio | {status}',font=small,fill='#c9d4e3')
                draw_fingering(picture,assigned,footer,True)
                if frame in (0,round(fps),round(12*fps)):picture.save(output/f'preview_{frame:04d}.png')
                encoder.stdin.write(np.asarray(picture).tobytes())
                if frame%150==0:print(json.dumps(dict(rendered_frames=frame,total_frames=count)),flush=True)
    finally:encoder.stdin.close();code=encoder.wait()
    if code:raise RuntimeError(f'FFmpeg encoder exit {code}')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-n','-i',str(silent),'-i',str(output/'played.wav'),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t',str(duration),'-movflags','+faststart',str(video)],check=True)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-xerror','-i',str(video),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
    silent.unlink()
    result=dict(recording=str(recording),trace_sha256=sha(trace_path),run=recording.name,slurm_job_id=os.environ['SLURM_JOB_ID'],
        source_runner_sha256=sha(__file__),video_file=video.name,score_sha256=sha(score),
        generation_runner_sha256=generator_hash,generation_runner_included=generator is not None,
        physical_hz=report['physics_hz'],video_fps=fps,frames=count,duration_s=duration,
        recorded_time_range=[float(times[0]),float(times[-1])],preparation='negative-time preparation excluded; t=0 is original MIDI origin',
        micro_f1=report['original_midi_micro_f1'],mean_frame_f1=report['original_midi_mean_frame_f1'],
        audio=audio,score_key_color_hints=True,
        actual_key_highlight=True,decoded_video_and_audio=True,physics_rerun=False,
        rendering='recorded physical samples; no state interpolation or controller rerun',camera=dict(lookat=center.tolist(),azimuth=0,elevation=-50,ortho_height=ortho),
        scene_manifest_sha256=sha(Path(report['source']['build'])/'scene/manifest.json'))
    (output/'export_report.json').write_text(json.dumps(result,indent=2))
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()}
    (output/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    return result
