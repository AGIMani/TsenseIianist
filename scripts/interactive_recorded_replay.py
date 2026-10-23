#!/usr/bin/env python3
"""Portable MuJoCo viewer for recorded physics, with seek/pause/speed and actual-key audio.

No policy or mj_step runs here. --validate checks data on CPU without a window.
"""
import argparse, json, os, shutil, subprocess, time, wave
from pathlib import Path
import numpy as np


def seek_time(x,width,end):
    return float(np.clip(x/max(width,1),0,1))*end


def load(folder):
    import mujoco
    meta=json.loads((folder/'replay.json').read_text())
    scene=folder if (folder/'model.mjb').is_file() else folder/'kinematic_scene'
    expected=meta['dependency_versions']['mujoco']
    if mujoco.__version__ != expected:
        raise RuntimeError(f'Replay requires MuJoCo {expected}; installed {mujoco.__version__}')
    binary=scene/'model.mjb'
    if not binary.is_file():
        raise RuntimeError('Missing compiled model.mjb; regenerate the replay package remotely. XML recompilation is disabled.')
    model=mujoco.MjModel.from_binary_path(str(binary))
    with np.load(folder/'physics_trace.npz',allow_pickle=False) as z:
        trace={k:np.concatenate((z['initial_'+k][None],z[k])) for k in ('qpos','mocap_pos','mocap_quat')}
        dt=float(z['physics_dt'])
    prefix=meta['preparation_prefix_seconds']
    if prefix:
        with np.load(folder/'preparation.npz',allow_pickle=False) as z:
            for k in trace:
                if not np.allclose(z[k][-1],trace[k][0],atol=1e-5):raise ValueError('Preparation discontinuity')
                trace[k]=np.concatenate((z[k][:-1],trace[k]))
    trace['time']=np.arange(len(trace['qpos']))*dt
    with np.load(folder/'score_annotations.npz',allow_pickle=False) as z:score={k:z[k] for k in z.files}
    if not np.isclose(trace['time'][-1],score['time'][-1],atol=1e-7):raise ValueError('Score/physics duration mismatch')
    if trace['qpos'].shape[1]!=model.nq or trace['mocap_pos'].shape[1]!=model.nmocap:raise ValueError('Model/trace shape mismatch')
    if not all(np.isfinite(v).all() for v in trace.values()):raise ValueError('Nonfinite state')
    with wave.open(str(folder/'played.wav'),'rb') as w:
        if w.getnframes()/w.getframerate()<trace['time'][-1]-.01:raise ValueError('Audio is incomplete')
    return model,trace,score,meta


def sample(model,data,trace,score,meta,at):
    import mujoco
    # Choose a recorded physical sample, never infer key states from actions.
    i=int(np.clip(np.searchsorted(trace['time'],at,side='right')-1,0,len(trace['time'])-1))
    data.qpos[:]=trace['qpos'][i];data.mocap_pos[:]=trace['mocap_pos'][i];data.mocap_quat[:]=trace['mocap_quat'][i];data.time=float(trace['time'][i])
    # Recorded poses need transforms and camera tracking, not collision/constraint solving.
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_camlight(model,data)
    if at<=meta['preparation_prefix_seconds']+1e-9:return np.zeros((10,88))
    j=min(int(np.searchsorted(score['time'],at-1e-9)),len(score['time'])-1)
    return score['fingers'][j]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--validate',action='store_true');p.add_argument('--render-check',action='store_true');p.add_argument('--viewer-check-seconds','--check-seconds',dest='viewer_check_seconds',type=float,default=0);p.add_argument('--external-audio',action='store_true');a=p.parse_args()
    import mujoco
    from preview_keys import ActiveKeyHighlight
    from preview_fingering import FingeringHighlight
    model,trace,score,meta=load(a.folder);data=mujoco.MjData(model);keys=ActiveKeyHighlight(model);key_colors=FingeringHighlight(model,meta,keys)
    if a.validate or a.render_check:
        scn=mujoco.MjvScene(model,maxgeom=4000)
        display_samples=np.unique(np.r_[0.,score['time'],trace['time'][-1]])
        for at in display_samples:
            assigned=sample(model,data,trace,score,meta,at);scn.ngeom=0;keys.append(scn,data);key_colors.append(scn,data,assigned)
            for geom in scn.geoms[:scn.ngeom]:
                if geom.label or geom.type != mujoco.mjtGeom.mjGEOM_BOX:
                    raise RuntimeError('Replay display must contain only unlabeled piano key boxes')
        renderer=None
        if a.render_check:
            import glfw,ctypes
            if not glfw.init():raise RuntimeError('GLFW could not connect to the local display')
            window=None;context=None
            try:
                glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
                window=glfw.create_window(640,360,'MuJoCo replay validation',None,None)
                if not window:raise RuntimeError('Cannot create local OpenGL context')
                glfw.make_context_current(window)
                gl=ctypes.CDLL('libGL.so.1');gl.glGetString.restype=ctypes.c_char_p
                renderer=gl.glGetString(0x1F01).decode()
                if 'NVIDIA' not in renderer and 'RTX' not in renderer:
                    raise RuntimeError('NVIDIA offload was not selected: '+renderer)
                context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_100)
                camera=mujoco.MjvCamera();camera.lookat[:]=(.2,0,.2);camera.distance=1.6
                for at in (0.,float(trace['time'][-1])/2,float(trace['time'][-1])):
                    assigned=sample(model,data,trace,score,meta,at)
                    mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),None,camera,mujoco.mjtCatBit.mjCAT_ALL,scn)
                    keys.append(scn,data);key_colors.append(scn,data,assigned)
                    mujoco.mjr_render(mujoco.MjrRect(0,0,640,360),scn,context)
                    if at == float(trace['time'][-1])/2:
                        from PIL import Image
                        pixels=np.empty((360,640,3),dtype=np.uint8)
                        mujoco.mjr_readPixels(pixels,None,mujoco.MjrRect(0,0,640,360),context)
                        Image.fromarray(pixels[::-1]).save(a.folder/'render_check.png')
                    glfw.swap_buffers(window);glfw.poll_events()
            finally:
                if context is not None:context.free()
                if window:glfw.destroy_window(window)
                glfw.terminate()
        print(json.dumps(dict(state='validated',duration_s=float(trace['time'][-1]),physical_hz=1/float(np.diff(trace['time'])[0]),frames=len(trace['time']),native_arms=meta['native_arms'],fingering=True,key_highlight=True,audio=True,renderer=renderer,display_contract='piano_key_colors_only_v1',floating_label_count=0,annotation_sphere_count=0,display_samples_checked=len(display_samples),validation='Compiled model/state/annotation/audio-data checks; optional local OpenGL rendering check')));return
    if not a.external_audio and not shutil.which('ffplay'):raise RuntimeError('ffplay is required for audible replay; install FFmpeg in the user environment')
    import glfw
    pending=[]
    if not glfw.init():raise RuntimeError('Cannot connect to local display')
    # Upload large mesh buffers before exposing a window to the desktop watchdog.
    glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
    window=glfw.create_window(1100,760,'Pianist recorded replay | Space: play/pause | 1-4: views',None,None)
    if not window:
        glfw.terminate();raise RuntimeError('Cannot create local OpenGL window')
    glfw.make_context_current(window);glfw.swap_interval(1)
    scene=mujoco.MjvScene(model,maxgeom=4000)
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
    context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_100)
    camera=mujoco.MjvCamera();camera.lookat[:]=(.2,0,.2);camera.distance=1.6;camera.azimuth=90;camera.elevation=-35
    option=mujoco.MjvOption();last_mouse=[0.,0.];dragging=[False];axes=[0]
    option.geomgroup[:]=[1,1,0,0,0,0];option.sitegroup[:]=0
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=False
    hand_ids=[i for i in range(model.nbody) if mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) in ('right_/base_link','left_/base_link')]
    flange_ids=[i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').lower().endswith(('link7','link_7','flange'))]
    sample(model,data,trace,score,meta,0.)
    def view(number):
        focus=np.mean(data.xpos[hand_ids],axis=0) if hand_ids else np.array([.2,0,.2])
        camera.lookat[:]=[.30,.05,.18] if number==1 else focus
        camera.distance=1.65 if number==1 else .48
        camera.azimuth=90 if number in (3,4) else 135
        camera.elevation=-88 if number==4 else -4 if number==3 else -15 if number==2 else -22
    view(1)
    def key(win,k,scancode,action,mods):
        if action==glfw.PRESS:
            if k==glfw.KEY_ESCAPE:glfw.set_window_should_close(win,True)
            else:pending.append(k)
    def cursor(win,x,y):
        nonlocal at
        if dragging[0]:
            at=seek_time(x,glfw.get_window_size(win)[0],end);return
        dx=x-last_mouse[0];dy=y-last_mouse[1];last_mouse[:]=[x,y]
        left=glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
        right=glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
        middle=glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_MIDDLE)==glfw.PRESS
        if not (left or right or middle):return
        shift=glfw.get_key(win,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(win,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS
        action=(mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V) if right else ((mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V) if left else mujoco.mjtMouse.mjMOUSE_ZOOM)
        height=max(glfw.get_window_size(win)[1],1)
        mujoco.mjv_moveCamera(model,action,dx/height,dy/height,camera)
    def scroll(win,x,y):mujoco.mjv_moveCamera(model,mujoco.mjtMouse.mjMOUSE_ZOOM,0.,-.05*y,camera)
    def button(win,button,action,mods):
        nonlocal at
        if button!=glfw.MOUSE_BUTTON_LEFT:return
        x,y=glfw.get_cursor_pos(win);width,height=glfw.get_window_size(win)
        if action==glfw.PRESS and y>=height-50:
            dragging[0]=True;at=seek_time(x,width,end);stop_audio()
        elif action==glfw.RELEASE and dragging[0]:
            dragging[0]=False
            if playing and at<end:start_audio()
    glfw.set_mouse_button_callback(window,button)
    glfw.set_key_callback(window,key);glfw.set_cursor_pos_callback(window,cursor);glfw.set_scroll_callback(window,scroll)
    if not a.viewer_check_seconds:glfw.show_window(window)
    glfw.poll_events()
    at=0.;speed=1.;playing=False;audio=None;end=float(trace['time'][-1]);clock=time.monotonic()
    def stop_audio():
        nonlocal audio
        if a.external_audio:
            print('REPLAY_AUDIO '+json.dumps(dict(state='pause')),flush=True)
            return
        if audio is not None:
            if audio.poll() is None:audio.terminate()
            audio.wait(timeout=5);audio=None
    def start_audio():
        nonlocal audio
        stop_audio()
        if a.external_audio:
            print('REPLAY_AUDIO '+json.dumps(dict(state='play',at=at,speed=speed)),flush=True)
            return
        audio=subprocess.Popen(['ffplay','-nodisp','-autoexit','-loglevel','error','-ss',str(at),'-af',f'atempo={speed}',str(a.folder/'played.wav')],stdin=subprocess.DEVNULL)
    print('Space: play/pause | R: reset and replay | Left/Right: seek 1 s | 1-4: views | A: axes | drag bottom bar: seek | Home: beginning | -/=: speed .5/1/2x | mouse: rotate/pan/zoom')
    if a.viewer_check_seconds:
        playing=True;start_audio();clock=time.monotonic()
    started=time.monotonic();frames=0;last_at=None;assigned=None;timing=dict(pose=0.,scene=0.,render=0.,swap=0.)
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now=time.monotonic()
            if a.viewer_check_seconds and now-started>=a.viewer_check_seconds:break
            if playing and not dragging[0]:at=min(end,at+(now-clock)*speed)
            clock=now;events=pending[:];pending.clear();changed=False
            if a.viewer_check_seconds:
                view(1+(frames//15)%4);axes[0]=(frames//15)%4
            for k in events:
                if k==32:
                    if at>=end:at=0.
                    playing=not playing;changed=True
                elif glfw.KEY_1<=k<=glfw.KEY_4:view(k-glfw.KEY_0)
                elif k==glfw.KEY_A:axes[0]=(axes[0]+1)%4
                elif k==glfw.KEY_R:at=0.;playing=True;last_at=None;changed=True
                elif k in (262,263):at=float(np.clip(at+(1 if k==262 else -1),0,end));changed=True
                elif k==268:at=0.;changed=True
                elif k in (45,61):speed=float(np.clip(speed*(.5 if k==45 else 2),.5,2));changed=True
            if at>=end and playing:playing=False;changed=True
            if changed:
                start_audio() if playing else stop_audio();clock=time.monotonic()
            if playing and audio is not None and audio.poll() is not None and at<end-.2:
                raise RuntimeError('Audio playback stopped unexpectedly; inspect ffplay output')
            tick=time.monotonic()
            if at!=last_at:
                assigned=sample(model,data,trace,score,meta,at);last_at=at
            timing["pose"]+=time.monotonic()-tick;tick=time.monotonic()
            mujoco.mjv_updateScene(model,data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            keys.append(scene,data);key_colors.append(scene,data,assigned)
            selected=(hand_ids if axes[0] in (1,3) else [])+(flange_ids if axes[0] in (2,3) else [])
            for body_id in selected:
                rot=data.xmat[body_id].reshape(3,3);pos=data.xpos[body_id]
                for axis,rgba in enumerate(((1,.1,.1,1),(.1,1,.2,1),(.2,.45,1,1))):
                    if scene.ngeom>=scene.maxgeom:break
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(rgba))
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.0015,pos,pos+.065*rot[:,axis]);scene.ngeom+=1
            timing["scene"]+=time.monotonic()-tick;tick=time.monotonic()
            width,height=glfw.get_framebuffer_size(window)
            rect=mujoco.MjrRect(0,0,width,height)
            mujoco.mjr_render(rect,scene,context)
            phase='PREPARATION' if at<meta['preparation_prefix_seconds'] else 'PERFORMANCE'
            hud=f'{at:6.2f} / {end:.2f} s | {phase} | {speed:g}x | {"PLAY" if playing else "PAUSE"}\nSpace: play/pause | R: restart | arrows: seek | -/=: speed\n1 overview | 2 close | 3 side | 4 top | A: actual body axes\nSame recorded physics, actual keys and audio'
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,hud,'',context)
            mujoco.mjr_rectangle(mujoco.MjrRect(0,0,width,35),.12,.16,.22,1.)
            mujoco.mjr_rectangle(mujoco.MjrRect(0,0,max(1,int(width*at/max(end,1e-12))),35),.15,.55,.85,1.)
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,rect,'Click / drag this bar to seek','',context)
            timing["render"]+=time.monotonic()-tick;tick=time.monotonic()
            glfw.swap_buffers(window);timing["swap"]+=time.monotonic()-tick;frames+=1;time.sleep(.01)
        if a.viewer_check_seconds:
            cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
            print(json.dumps(dict(viewer_check='passed',frames=frames,seconds=time.monotonic()-started,timing=timing,mesh_faces=model.nmeshface,memory_events=(cg/'memory.events').read_text())),flush=True)
    finally:
        stop_audio();context.free();glfw.destroy_window(window);glfw.terminate()

if __name__=='__main__':main()
