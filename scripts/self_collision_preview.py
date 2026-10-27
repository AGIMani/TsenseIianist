#!/usr/bin/env python3
"""Display remote-recorded poses and collision reward diagnostics, without local physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def slider_value(x, start, width, low, high):
    if width <= 0 or high < low:
        raise ValueError('Invalid slider range')
    return float(low + np.clip((x-start)/width, 0., 1.)*(high-low))


def selected_controls(controls, side, finger):
    prefix = side + '_' + ('Palm_Base' if finger == 'Wrist' else finger) + '_'
    return [c for c in controls if c['name'].startswith(prefix)]


def check_pose(response, nq):
    qpos = np.asarray(response['qpos'], dtype=float)
    if qpos.shape != (nq,) or not np.isfinite(qpos).all():
        raise ValueError('Remote recorded pose is invalid')
    score = response['score']
    for field in ('instant_substep_penalty', 'held_control_step_penalty'):
        value = float(score[field])
        if not np.isfinite(value) or value > 1e-12:
            raise ValueError('Invalid remote collision penalty')
    return qpos


class RemoteProbe:
    """One in-flight and one replaceable request; only complete confirmed poses are shown."""
    def __init__(self, config, session):
        self.config, self.session = config, session
        self.requests = queue.Queue(maxsize=1)
        self.responses = queue.Queue(maxsize=4)
        self.process = None
        self.closed = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, request):
        while True:
            try:
                self.requests.put_nowait(request)
                return
            except queue.Full:
                try:
                    self.requests.get_nowait()
                except queue.Empty:
                    pass

    def run(self):
        c = self.config
        root = c['remote_root']
        remote_log = root + '/task_outputs/collision_preview_sessions/' + self.session.name + '/requests.jsonl'
        argv = ['scripts/hpc/self_collision_probe.py', '--recording', c['recording'],
                '--expected-model-sha256', c['model_sha256'],
                '--expected-archive-sha256', c['archive_sha256'], '--log', remote_log]
        bootstrap = ('import hashlib, pathlib, runpy, sys; '
                     f'checks={c["worker_files_sha256"]!r}; '
                     'assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in checks.items()), "Remote worker source changed"; '
                     f'sys.argv={argv!r}; runpy.run_path(sys.argv[0],run_name="__main__")')
        command = ('cd ' + shlex.quote(root) + ' && ' + shlex.join([
            'env', 'PYTHONPATH=.', 'MUJOCO_GL=disable', 'CUDA_VISIBLE_DEVICES=',
            'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            c['python'], '-u', '-c', bootstrap]))
        try:
            with (self.session/'remote_stderr.log').open('a') as errors, (self.session/'responses.jsonl').open('a') as records:
                (self.session/'command.json').write_text(json.dumps(dict(host=c['host'], command=command, remote_log=remote_log), indent=2))
                self.process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', c['host'], command],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
                (self.session/'ssh.pid').write_text(str(self.process.pid)+'\n')

                def read():
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError('Remote collision worker disconnected; see remote_stderr.log')
                    row = json.loads(line)
                    records.write(json.dumps(row, allow_nan=False)+'\n')
                    records.flush()
                    self.responses.put(row)

                read()
                while not self.closed:
                    request = self.requests.get()
                    if request is None:
                        break
                    self.process.stdin.write(json.dumps(request, allow_nan=False)+'\n')
                    self.process.stdin.flush()
                    read()
        except Exception as exc:
            if not self.closed:
                self.responses.put(dict(type='error', error=str(exc)))
        finally:
            if self.process:
                try:
                    self.process.stdin.close()
                    code = self.process.wait(timeout=3)
                except (subprocess.TimeoutExpired, BrokenPipeError):
                    self.process.terminate()
                    code = self.process.wait(timeout=3)
                (self.session/'remote_exit.json').write_text(json.dumps(dict(exit_status=code)))

    def close(self):
        self.closed = True
        self.submit(None)
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.thread.join(timeout=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/self_collision_preview/remote.json')
    parser.add_argument('--check-seconds', type=float, default=0)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    session = ROOT/'logs/self_collision_preview'/str(time.time_ns())
    session.mkdir(parents=True, exist_ok=False)
    remote = RemoteProbe(config, session)
    context = window = None
    try:
        ready = remote.responses.get(timeout=90)
        if ready.get('type') != 'ready':
            raise RuntimeError(ready.get('error', 'Invalid remote ready response'))
        if ready.get('protocol') != 'static_self_collision_v1' or ready['model_sha256'] != config['model_sha256']:
            raise RuntimeError('Remote protocol/model does not match the display model')
        binary = Path(config['local_model'])
        with binary.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != config['model_sha256']:
                raise RuntimeError('Local MJB differs from verified remote model')
        # The local model is only a display of already recorded remote responses.
        import mujoco
        import glfw
        if mujoco.__version__ != ready['mujoco_version']:
            raise RuntimeError('Local MuJoCo version differs from recorded model version')
        model = mujoco.MjModel.from_binary_path(str(binary))
        data = mujoco.MjData(model)
        controls = ready['controls']
        defaults = {c['name']: float(c['value']) for c in controls}
        values = dict(defaults)
        confirmed = ready
        data.qpos[:] = check_pose(ready, model.nq)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        if not glfw.init():
            raise RuntimeError('Cannot connect to local desktop display')
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(1360, 850, 'Self-collision inspector | remote CPU | no local physics', None, None)
        if not window:
            raise RuntimeError('Cannot create MuJoCo display')
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        scene = mujoco.MjvScene(model, maxgeom=5000)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
        option.sitegroup[:] = 0
        option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        hand_geoms = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(('right_/', 'left_/'))]
        for i in hand_geoms:
            model.geom_group[i] = 2 if model.geom_contype[i] or model.geom_conaffinity[i] else 1
        initial_colors = model.geom_rgba.copy()
        side, finger = 'R', 'Index'
        mode, show_ignored = 0, True
        last_mouse, drag = [0., 0.], None
        sliders, buttons = [], []
        request_id, shown_id, last_sent = 0, 0, 0.
        dirty, error = False, ''

        def focus(top=False):
            body = model.body(('right_' if side == 'R' else 'left_')+'/base_link').id
            camera.lookat[:] = data.xpos[body]
            camera.distance, camera.azimuth, camera.elevation = .37, 140, -88 if top else -25

        def select(kind, value):
            nonlocal side, finger, mode, show_ignored, dirty, values
            if kind == 'side':
                side = value
                focus()
            elif kind == 'finger':
                finger = value
            elif kind == 'mode':
                mode = (mode+1) % 3
            elif kind == 'ignored':
                show_ignored = not show_ignored
            elif kind == 'reset':
                values = dict(defaults)
                dirty = True

        def button(win, which, action, mods):
            nonlocal drag, dirty
            x, y = glfw.get_cursor_pos(win)
            last_mouse[:] = [x, y]
            if which != glfw.MOUSE_BUTTON_LEFT:
                return
            if action == glfw.RELEASE:
                drag = None
                return
            for rect, kind, value in buttons:
                left, top, width, height = rect
                if left <= x <= left+width and top <= y <= top+height:
                    select(kind, value)
                    return
            for rect, control in sliders:
                left, top, width, height = rect
                if left <= x <= left+width and top <= y <= top+height:
                    drag = (rect, control)
                    values[control['name']] = slider_value(x, left, width, control['min'], control['max'])
                    dirty = True
                    return

        def cursor(win, x, y):
            nonlocal dirty
            dx, dy = x-last_mouse[0], y-last_mouse[1]
            last_mouse[:] = [x, y]
            if drag:
                rect, c = drag
                values[c['name']] = slider_value(x, rect[0], rect[2], c['min'], c['max'])
                dirty = True
                return
            width, height = glfw.get_window_size(win)
            if x >= width-370:
                return
            left = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            right = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            if left or right:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V if right else mujoco.mjtMouse.mjMOUSE_ROTATE_V
                mujoco.mjv_moveCamera(model, action, dx/max(height,1), dy/max(height,1), camera)

        def key(win, code, scan, action, mods):
            if action != glfw.PRESS:
                return
            if code == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif code == glfw.KEY_TAB:
                select('side', 'L' if side == 'R' else 'R')
            elif glfw.KEY_1 <= code <= glfw.KEY_6:
                select('finger', ['Thumb','Index','Mid','Ring','Pinky','Wrist'][code-glfw.KEY_1])
            elif code == glfw.KEY_C:
                select('mode', None)
            elif code == glfw.KEY_E:
                select('ignored', None)
            elif code == glfw.KEY_N:
                select('reset', None)
            elif code == glfw.KEY_T:
                focus(True)

        glfw.set_mouse_button_callback(window, button)
        glfw.set_cursor_pos_callback(window, cursor)
        glfw.set_key_callback(window, key)
        glfw.set_scroll_callback(window, lambda w,x,y: mujoco.mjv_moveCamera(model,mujoco.mjtMouse.mjMOUSE_ZOOM,0.,-.05*y,camera))
        focus()
        if not args.check_seconds:
            glfw.show_window(window)
            glfw.focus_window(window)
        started, frames, check_stage = time.monotonic(), 0, -1
        saved_screenshot = False
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = time.monotonic()
            if args.check_seconds and now-started >= args.check_seconds:
                break
            if args.check_seconds:
                stage = min(int((now-started)/max(args.check_seconds,1)*4),3)
                if stage != check_stage:
                    check_stage = stage
                    values = dict(defaults)
                    if stage < 2:
                        finger = 'Index'
                        values['R_Index_MCP'] = [.25,1.4][stage]
                    elif stage == 2:
                        finger = 'Mid'
                        values['R_Mid_MCP'], values['R_Mid_SW'] = .7, .35
                    mode = stage % 3
                    dirty = True
            if dirty and now-last_sent >= .10:
                request_id += 1
                remote.submit(dict(id=request_id, values=dict(values)))
                dirty, last_sent = False, now
            while not remote.responses.empty():
                response = remote.responses.get_nowait()
                if response.get('type') == 'error':
                    error = response.get('error', 'Remote error')
                    continue
                if response.get('type') == 'pose' and response['id'] >= shown_id:
                    data.qpos[:] = check_pose(response, model.nq)
                    confirmed, shown_id, error = response, response['id'], ''
                    mujoco.mj_kinematics(model,data)
                    mujoco.mj_comPos(model,data)
                    mujoco.mj_camlight(model,data)
            option.geomgroup[:] = [1,mode in (0,2),mode in (1,2),0,0,0]
            model.geom_rgba[:] = initial_colors
            for contact in confirmed['contacts']:
                if contact['category'] in ('hand_key','hand_piano_case','other'):
                    continue
                if not contact['reward_counted'] and not show_ignored:
                    continue
                for name in (contact['body1'],contact['body2']):
                    body_id = model.body(name).id
                    color = [1.,.16,.1,1.] if contact['reward_counted'] else [1.,.65,.12,.65]
                    mask = np.flatnonzero(model.geom_bodyid == body_id)
                    if contact['reward_counted'] or not np.any(model.geom_rgba[mask,1] == .16):
                        model.geom_rgba[mask] = color
            mujoco.mjv_updateScene(model,data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            width,height = glfw.get_framebuffer_size(window)
            ww,hh = glfw.get_window_size(window)
            sx,sy = width/ww,height/hh
            panel_width = 370
            view = mujoco.MjrRect(0,0,int((ww-panel_width)*sx),height)
            mujoco.mjr_render(view,scene,context)
            panel = mujoco.MjrRect(view.width,0,width-view.width,height)
            mujoco.mjr_rectangle(panel,.075,.09,.12,1.)

            def rectangle(x,y,w,h,color):
                mujoco.mjr_rectangle(mujoco.MjrRect(int(x*sx),int((hh-y-h)*sy),int(w*sx),int(h*sy)),*color)

            def label(x,y,w,h,text):
                rect = mujoco.MjrRect(int(x*sx),int((hh-y-h)*sy),int(w*sx),int(h*sy))
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,text,'',context)

            left = ww-panel_width+14
            label(left,10,344,48,'SELF-COLLISION REWARD\nRemote recorded pose | no local physics')
            buttons.clear()
            for i,hand in enumerate(('R','L')):
                rect=(left+i*170,64,160,30)
                rectangle(*rect, (.16,.38,.48,1.) if side==hand else (.16,.19,.24,1.))
                label(*rect, 'Right hand' if hand=='R' else 'Left hand')
                buttons.append((rect,'side',hand))
            for i,name in enumerate(('Thumb','Index','Mid','Ring','Pinky','Wrist')):
                rect=(left+(i%3)*113,103+(i//3)*36,106,30)
                rectangle(*rect,(.17,.4,.48,1.) if finger==name else (.16,.19,.24,1.))
                label(*rect,f'{i+1} {name}')
                buttons.append((rect,'finger',name))
            sliders.clear()
            for i,c in enumerate(selected_controls(controls,side,finger)):
                y=184+i*62
                name=c['name']
                label(left,y,344,26,f'{name}: {values[name]:.4f} {c["unit"]}')
                rect=(left+8,y+30,320,17)
                rectangle(*rect,(.18,.22,.28,1.))
                ratio=(values[name]-c['min'])/max(c['max']-c['min'],1e-12)
                rectangle(rect[0],rect[1],max(2,rect[2]*ratio),rect[3],(.25,.66,.82,1.))
                sliders.append(((rect[0],y+25,rect[2],29),c))
            if finger in ('Index','Mid','Ring','Pinky'):
                follow = [confirmed['applied_actions'][ready['action_names'].index(f'{side}_{finger}_{axis}')]
                          for axis in ('PIP','DIP')]
                label(left,340,344,55,f'Coupled PIP / DIP (read-only)\n{follow[0]:.4f} / {follow[1]:.4f} rad')
            converged = confirmed['coupling']['converged']
            label(left,405,344,30,'Coupling: '+('converged' if converged else 'NOT CONVERGED'))
            score=confirmed['score']
            pending = dirty or shown_id < request_id
            text=(f'{"UPDATING" if pending else "CONFIRMED"} pose {shown_id}\n'
                  f'Counted body pairs: {score["eligible_body_pair_count"]}\n'
                  f'Per physical substep: {score["instant_substep_penalty"]:.3f}\n'
                  f'Held for one control step: {score["held_control_step_penalty"]:.3f}\n'
                  f'Static estimate: {score["substeps_per_control_step"]} substeps; cap {score["cap"]:g}\n'
                  'Red: counted | amber: excluded here\nNot measured force or a simulated control step')
            label(left,450,344,150,text)
            mode_text=['Visual mesh','Collision hulls','Visual + hulls'][mode]
            for y,kind,caption in ((611,'mode','C: '+mode_text),(647,'ignored','E: excluded '+('ON' if show_ignored else 'OFF')),(683,'reset','N: reset all joints')):
                rect=(left,y,338,30)
                rectangle(*rect,(.16,.19,.24,1.))
                label(*rect,caption)
                buttons.append((rect,kind,None))
            contact_lines=[]
            for contact in sorted(confirmed['contacts'],key=lambda c:not c['reward_counted']):
                if contact['category'] in ('hand_key','hand_piano_case','other'):
                    continue
                if not contact['reward_counted'] and not show_ignored:
                    continue
                def short_body(name):
                    return name.replace('right_/','R/').replace('left_/','L/').replace('right_hand_','').replace('left_hand_','').replace('_link','')
                a,b=map(short_body,(contact['body1'],contact['body2']))
                line=f'{"PENALTY" if contact["reward_counted"] else "excluded"}: {a} - {b} ({contact["depth_m"]*1000:.2f} mm)'
                if line not in contact_lines:
                    contact_lines.append(line)
            hud='Drag joint bars | Tab hand | 1-6 finger/wrist | T top view\nMouse in scene: rotate/pan/zoom | N reset | Esc close\n'+'\n'.join(contact_lines[:7])
            if error:
                hud='REMOTE ERROR (display is stale): '+error+'\n'+hud
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,view,hud,'',context)
            label(left,733,344,78,'MCP/SW controls use current coupling.\nFollower angles are computed remotely.\nLogs: logs/self_collision_preview/')
            if args.screenshot and not saved_screenshot and shown_id > 0 and now-started > max(1,args.check_seconds*.55):
                from PIL import Image
                rgb=np.empty((height,width,3),dtype=np.uint8)
                mujoco.mjr_readPixels(rgb,None,mujoco.MjrRect(0,0,width,height),context)
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb[::-1]).save(args.screenshot)
                saved_screenshot=True
            glfw.swap_buffers(window)
            frames+=1
            glfw.wait_events_timeout(.02)
        if args.check_seconds and shown_id < 2:
            raise RuntimeError('Interactive check did not receive enough changed remote poses')
        if args.check_seconds and error:
            raise RuntimeError(error)
        cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
        report=dict(status='passed' if not error else 'remote_error',frames=frames,confirmed_id=shown_id,
                    last_score=confirmed['score'],local_physics_steps=0,local_collision_solves=0,
                    model_sha256=config['model_sha256'],session=str(session),
                    memory_events=(cg/'memory.events').read_text(),memory_peak_bytes=int((cg/'memory.peak').read_text()))
        (session/'validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:
        remote.close()
        if context is not None:
            context.free()
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()


if __name__ == '__main__':
    main()
