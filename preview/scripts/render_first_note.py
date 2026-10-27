#!/usr/bin/env python3
"""Render one saved preparation trace; no inference, model compilation or physics stepping."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import os
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Render through scripts/hpc/submit.sh in Slurm')
    from tsensei_v3_2_pianist.preview_backend import configure_backend,current_gl_info
    configure_backend(headless=True)
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from tsensei_v3_2_pianist.preview_camera import configure_camera
    from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
    from tsensei_v3_2_pianist.preview_audio import events_from_states,write_midi,PianoSynth
    report=json.loads((a.recording/'report.json').read_text())
    if mujoco.__version__!=report['mujoco_version']:raise ValueError('Recorded MJB MuJoCo version mismatch')
    if digest(a.recording/'model.mjb')!=report['model_sha256'] or digest(a.recording/'preparation.npz')!=report['trace_sha256']:
        raise ValueError('Recording checksum mismatch')
    m=mujoco.MjModel.from_binary_path(str(a.recording/'model.mjb'))
    d=mujoco.MjData(m)
    with np.load(a.recording/'preparation.npz',allow_pickle=False) as z:trace=dict(z)
    manifest=json.loads((a.recording/'manifest.json').read_text())
    times=trace['time'];dt=float(m.opt.timestep);duration=float(times[-1]);fps=20.
    if not np.allclose(times,np.arange(len(times))*dt,rtol=0,atol=1e-10):raise ValueError('Nonuniform physics recording')
    if not np.isclose(duration*fps,round(duration*fps)):raise ValueError('Duration is not whole control frames')
    a.output.mkdir(parents=True,exist_ok=False)
    camera=mujoco.MjvCamera();camera_info=configure_camera(camera,m,view='front',framing='hands')
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
    highlights=ActiveKeyHighlight(m)
    first_geoms={m.geom(report['preparation']['source']['key_geoms'][int(row[2])]).id for row in report['preparation']['source']['notes']}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    silent=a.output/'silent.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','1280x720','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(silent)],stdin=subprocess.PIPE)
    nframes=round(duration*fps)
    try:
        with mujoco.Renderer(m,height=720,width=1280,max_geom=4000) as renderer:
            graphics=current_gl_info()
            for frame in range(nframes):
                # Frame-end sample: includes the actual terminal state in the final frame.
                index=round((frame+1)/fps/dt)
                for key in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):getattr(d,key)[:]=trace[key][index]
                d.time=times[index];mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=camera)
                for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                    g.label=''
                    if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid in first_geoms:
                        g.rgba[:]=(.35,.65,1.,1.);g.matid=-1
                highlights.color_render_scene(renderer.scene,d)
                img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
                draw.rectangle((0,0,1280,95),fill=(18,23,32))
                draw.text((20,12),f'First-note hover preparation | {d.time:.2f} / {duration:.2f} s | {1/dt:.0f} Hz physics',font=font,fill='white')
                draw.text((20,48),'Blue: upcoming keys | Green: actual activation | Original score starts after preparation',font=font,fill=(185,215,255))
                encoder.stdin.write(np.asarray(img).tobytes())
                if frame in (0,nframes-1):img.save(a.output/('start.png' if frame==0 else 'end.png'))
    finally:
        encoder.stdin.close();code=encoder.wait()
    if code:raise RuntimeError('ffmpeg failed')
    keyj=np.array([m.joint(name).id for name in manifest['piano_joints']]);qadr=m.jnt_qposadr[keyj]
    q=trace['qpos'][1:,qadr];limits=m.jnt_range[keyj];fraction=manifest['key_activation_fraction']
    actual=q>=limits[:,1]*fraction
    if not np.array_equal(actual,trace['key_activation'][1:]):raise ValueError('Saved actual-key record differs from real qpos')
    events=events_from_states(q,limits,np.zeros(len(q)),dt,fraction)
    (a.output/'audio_events.json').write_text(json.dumps(events,indent=2)+'\n')
    write_midi(a.output/'played.mid',events)
    synth=PianoSynth()
    try:audio=synth.render(events,duration,a.output/'played.wav')
    finally:synth.close()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(silent),'-i',str(a.output/'played.wav'),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-shortest',str(a.output/'preparation.mp4')],check=True)
    # Keep the silent intermediate as provenance; no rewrite of recorded state.
    with np.load(a.recording/'original_score.npz',allow_pickle=False) as z:original=z['events']
    reference=original.copy();reference[:,:2]+=duration
    np.savez_compressed(a.output/'comparison_arrays.npz',time=times[1:],actual=actual,key_qpos=q,key_limits=limits,
        original_events=original,reference_events_on_video_axis=reference,preparation_offset_seconds=duration,physics_dt=dt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True,sharey=True)
    window=duration+.5
    visible=reference[reference[:,0]<=window]
    patches=[[(s,k+20.6),(e,k+20.6),(e,k+21.4),(s,k+21.4)] for s,e,k,_ in visible]
    axes[0].add_collection(PolyCollection(patches,facecolors='#397bb7'))
    for k in range(88):
        transitions=np.diff(np.r_[False,actual[:,k],False].astype(int))
        for start,end in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
            axes[1].broken_barh([(times[start+1],(end-start)*dt)],(k+20.6,.8),facecolors='#dc643f')
    pitches=np.r_[visible[:,2]+21,np.flatnonzero(actual.any(axis=0))+21]
    if not len(pitches):pitches=np.array([21,108])
    for ax,title in zip(axes,('Original score (starts after preparation)','Actual keys from the SAME physical recording')):
        ax.set_title(title,loc='left');ax.set_ylabel('MIDI pitch');ax.grid(alpha=.2)
        ax.axvline(duration,color='#444',linestyle='--');ax.axvspan(duration,window,color='#ddd',alpha=.5)
        ax.set_ylim(max(20,pitches.min()-4),min(109,pitches.max()+4));ax.set_xlim(0,window)
    axes[1].set_xlabel('Video time (s); grey region: reference only, beyond the preparation video')
    fig.suptitle(f'First-note preparation | {1/dt:.0f} Hz actual keys | offset {duration:.2f} s | no checkpoint / no policy',fontsize=12)
    fig.tight_layout();fig.savefig(a.output/'piano_roll_comparison.png',dpi=160);plt.close(fig)
    result=dict(recording=str(a.recording),trace_sha256=report['trace_sha256'],model_sha256=report['model_sha256'],
        frames=nframes,fps=fps,duration=duration,physics_hz=1/dt,graphics=graphics,camera=camera_info,audio=audio,
        actual_key_activation_samples=int(actual.any(axis=1).sum()),scope=report['scope'],
        score_offset_seconds=duration,sampling='original physics samples; no interpolation or clipping',
        physical_validation=report['preparation']['report'])
    (a.output/'render_report.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'sha256.json').write_text(json.dumps({p.name:digest(p) for p in a.output.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
