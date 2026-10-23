"""Local camera-only preview of a verified MJB; never compile XML or step physics."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import glfw
import mujoco
import numpy as np


def draw_piano_frame(scene, context, viewport, origin, rotation):
    """Project the true piano origin as a visible overlay, even below the keys."""
    cams = scene.camera
    eye = (cams[0].pos + cams[1].pos) / 2
    forward = (cams[0].forward + cams[1].forward) / 2
    forward /= np.linalg.norm(forward)
    up = (cams[0].up + cams[1].up) / 2
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    near = (cams[0].frustum_near + cams[1].frustum_near) / 2
    bottom = (cams[0].frustum_bottom + cams[1].frustum_bottom) / 2
    top = (cams[0].frustum_top + cams[1].frustum_top) / 2
    center = (cams[0].frustum_center + cams[1].frustum_center) / 2
    width = (top-bottom)*viewport.width/viewport.height

    def project(point):
        delta = point-eye
        depth = np.dot(delta, forward)
        if depth <= 0:
            return None
        x = (.5+(np.dot(delta, right)*near/depth-center)/width)*viewport.width
        y = (np.dot(delta, up)*near/depth-bottom)/(top-bottom)*viewport.height
        return np.array([x, y])

    def line(a, b, color):
        count = min(2000, max(2, int(np.linalg.norm(b-a))))
        for x, y in np.linspace(a, b, count):
            if 0 <= x < viewport.width and 0 <= y < viewport.height:
                mujoco.mjr_rectangle(mujoco.MjrRect(int(x)-1, int(y)-1, 3, 3), *color, 1.)

    def label(point, text, color, width=35):
        x, y = point
        if 0 <= x < viewport.width and 0 <= y < viewport.height:
            mujoco.mjr_rectangle(mujoco.MjrRect(int(x)-3, int(y)-4, width, 26), .05, .05, .05, .8)
            mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, text, context,
                .03, .25, *color)

    start = project(origin)
    if start is None:
        return
    for i, (name, color) in enumerate((('+X', (1., .25, .2)), ('+Y', (.2, 1., .35)), ('+Z', (.3, .65, 1.)))):
        end = project(origin + .08*rotation[:, i])
        if end is None:
            continue
        line(start, end, color)
        direction = end-start
        norm = np.linalg.norm(direction)
        if norm > 10:
            direction /= norm
            side = np.array([-direction[1], direction[0]])
            for sign in (-1, 1):
                line(end, end-10*direction+sign*5*side, color)
        label(end+5, name, color)
    line(start+[-5, 0], start+[5, 0], (1., 1., .1))
    line(start+[0, -5], start+[0, 5], (1., 1., .1))
    label(start+[10, -30], 'O (0, 0, 0)', (1., 1., .1), 150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--check-seconds', type=float, default=0)
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--view', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if not np.isfinite(args.check_seconds) or args.check_seconds < 0:
        parser.error('--check-seconds must be finite and nonnegative')
    folder = args.folder.resolve()
    metadata = json.loads((folder/'preview.json').read_text())
    if mujoco.__version__ != metadata['mujoco']:
        raise RuntimeError(f"Model requires MuJoCo {metadata['mujoco']}, installed {mujoco.__version__}")
    for name, expected in metadata['files'].items():
        path = folder/name
        if path.parent != folder:
            raise ValueError('Invalid model file path')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise RuntimeError(f'Preview file changed: {name}')
    root = Path(__file__).resolve().parents[1]
    for name, expected in metadata.get('layout_source_sha256', {}).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Piano source changed; regenerate this preview from the new model.')
    model = mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    data = mujoco.MjData(model)
    with np.load(folder/'state.npz', allow_pickle=False) as state:
        for name in ('qpos', 'mocap_pos', 'mocap_quat'):
            getattr(data, name)[:] = state[name]
    initial_qpos = data.qpos.copy()
    # Update only transforms and lights. No contacts, policy or time integration.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    white, black = [], []
    for key in metadata['keys']:
        index, color = key['index'], key['color']
        body = model.body(f'piano/{color}_key_{index}')
        geom = model.geom(f'piano/{color}_key_geom_{index}')
        if not np.isclose(body.pos[1]*1000, key['center_mm'], atol=1e-9, rtol=0):
            raise ValueError('Compiled key coordinate mismatch')
        reference = model.site(f'piano/{color}_key_site_{index}')
        if not np.isclose(reference.size[1]*2000, key['width_mm'], atol=1e-9, rtol=0):
            raise ValueError('Compiled key width mismatch')
        (black if color == 'black' else white).append(body.pos[1])
    assert (len(white), len(black)) == (52, 36)
    assert np.isclose(white[-1]-white[0]+.0225, 1.224, atol=1e-12, rtol=0)
    original_groups = model.geom_group.copy()
    hand_geoms = [i for i in range(model.ngeom)
                  if (model.geom(i).name or '').startswith(('left_/', 'right_/'))]
    # Group changes affect only this display copy, never the saved physical model.
    model.geom_group[hand_geoms] = 5
    if not glfw.init():
        raise RuntimeError('Cannot connect to the local desktop display')
    window = context = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(1280, 800, 'Piano 1224 mm | Static preview | 1/2/3 views | H hands', None, None)
        if not window:
            raise RuntimeError('Cannot create local OpenGL window')
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        gl = ctypes.CDLL('libGL.so.1')
        gl.glGetString.restype = ctypes.c_char_p
        renderer = gl.glGetString(0x1F01).decode()
        scene = mujoco.MjvScene(model, maxgeom=4000)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
        option.sitegroup[:] = 0
        option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        hands = False
        show_origin = True
        piano_parent = int(model.body('piano/base').parentid[0])
        piano_origin = data.xpos[piano_parent].copy()
        piano_rotation = data.xmat[piano_parent].reshape(3, 3).copy()

        def view(number):
            camera.lookat[:] = [0, 0, .04]
            camera.azimuth = 180  # Player side: low notes left, key fronts nearest.
            camera.elevation = -89 if number == 2 else -55
            camera.distance = .43 if number == 3 else 1.25

        def key(win, code, scan, action, mods):
            nonlocal hands, show_origin
            if action != glfw.PRESS:
                return
            if code == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif glfw.KEY_1 <= code <= glfw.KEY_3:
                view(code-glfw.KEY_0)
                print(f'View: {code-glfw.KEY_0}', flush=True)
            elif code == glfw.KEY_O:
                show_origin = not show_origin
            elif code == glfw.KEY_H:
                hands = not hands
                model.geom_group[hand_geoms] = original_groups[hand_geoms] if hands else 5
                print(f'Hands visible: {hands}', flush=True)

        mouse = [0., 0.]

        def cursor(win, x, y):
            dx, dy = x-mouse[0], y-mouse[1]
            mouse[:] = [x, y]
            left = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            right = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            if not (left or right):
                return
            shift = any(glfw.get_key(win, k) == glfw.PRESS for k in (glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT))
            mode = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V) if right else (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
            height = max(glfw.get_window_size(win)[1], 1)
            mujoco.mjv_moveCamera(model, mode, dx/height, dy/height, camera)

        glfw.set_key_callback(window, key)
        glfw.set_cursor_pos_callback(window, cursor)
        glfw.set_mouse_button_callback(window, lambda w, b, a, m: mouse.__setitem__(slice(None), glfw.get_cursor_pos(w)))
        glfw.set_scroll_callback(window, lambda w, x, y: mujoco.mjv_moveCamera(
            model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0., -.05*y, camera))
        view(args.view or 1)
        glfw.show_window(window)
        glfw.focus_window(window)
        started = time.monotonic()
        frames = 0
        report = dict(status='visible', renderer=renderer, mujoco=mujoco.__version__,
                      total_width_mm=1224, white_keys=52, black_keys=36,
                      physics_steps=0, training=False, piano_origin_world_m=piano_origin.tolist(),
                      origin_marker='projected actual piano origin; visible through keys', source_archive_sha256=metadata['source_archive_sha256'])
        print(json.dumps(report), flush=True)
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if args.check_seconds and args.view is None:
                phase = min(int((time.monotonic()-started)/args.check_seconds*3), 2)
                view(phase+1)
                model.geom_group[hand_geoms] = original_groups[hand_geoms] if phase == 2 else 5
            mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
            width, height = glfw.get_framebuffer_size(window)
            rect = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(rect, scene, context)
            if show_origin:
                draw_piano_frame(scene, context, rect, piano_origin, piano_rotation)
            hud = ('PIANO 1224 mm | black top 10 x 90 / bottom 12 x 95 / height 18 mm\n'
                   'Black centers: pair 28.271 mm | triple 26.924 mm\n'
                   'O origin/axes (overlay through keys) | +X player | +Y high notes | +Z up\n'
                   '1 overview | 2 top | 3 close | H show/hide hands | Esc close\n'
                   'Mouse: left rotate | right pan | wheel zoom | STATIC VIEW')
            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, rect, hud, '', context)
            if frames == 10:
                if args.screenshot:
                    from PIL import Image
                    rgb = np.empty((height, width, 3), dtype=np.uint8)
                    mujoco.mjr_readPixels(rgb, None, rect, context)
                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb[::-1]).save(args.screenshot)
                (folder/'viewer_status.json').write_text(json.dumps(report, indent=2)+'\n')
            glfw.swap_buffers(window)
            frames += 1
            if args.check_seconds and time.monotonic()-started >= args.check_seconds:
                break
            glfw.wait_events_timeout(.02)
        np.testing.assert_array_equal(data.qpos, initial_qpos)
        assert data.time == 0
        cg = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[-1].lstrip('/')
        report.update(status='closed', frames=frames, seconds=time.monotonic()-started,
                      state_unchanged=True, memory_events=(cg/'memory.events').read_text(),
                      memory_peak_bytes=int((cg/'memory.peak').read_text()))
        (folder/'last_session.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        if context is not None:
            context.free()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == '__main__':
    main()
