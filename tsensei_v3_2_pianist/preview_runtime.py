"""Interactive physics preview driven by the MuJoCo viewer's Control panel."""
from contextlib import contextmanager
from queue import Empty, SimpleQueue
import threading
from collections import deque
import inspect
import numpy as np
import time

import mujoco

from .preview_geometry import configure_options
from .preview_keys import ActiveKeyHighlight


@contextmanager
def _managed_viewer(model, data, **kwargs):
    """Wait for our viewer's render thread before Python/GLFW tears down.

    MuJoCo Handle.close() only signals its daemon thread. The public handle has
    no join method; isolate the dependency on _launch_internal here, and never
    wait for unrelated viewers or application threads.
    """
    import mujoco.viewer
    existing = set(threading.enumerate())
    handle = mujoco.viewer.launch_passive(model, data, **kwargs)
    target = getattr(mujoco.viewer, '_launch_internal', None)
    owned = [thread for thread in threading.enumerate()
             if thread not in existing and target is not None
             and getattr(thread, '_target', None) is target]
    try:
        with handle:
            yield handle
    finally:
        for thread in owned:
            thread.join()


def run_interactive(physics, *, wireframe=None, coordinate_frames=None, show_fingertips=False, start_paused=False,
                    duration=None, full_sync=False, transparent_model=False, show_hand_collisions=False, on_save=None, contact_controller=None,
                    arm_preview=None, connector_preview=None, connector_controls=True):
    """Consume GUI controls and advance physics without task actions overriding ctrl.

    Space toggles pause, R restores the entry state (including controls/time).
    The passive viewer syncs GUI inputs; it does not step the model itself.
    """
    import mujoco.viewer

    model, data = physics.model.ptr, physics.data.ptr
    key_highlight = ActiveKeyHighlight(model)
    initial = mujoco.MjData(model)
    mujoco.mj_copyData(initial, model, data)
    from contextlib import ExitStack
    from .preview_contacts import ContactPanel
    keys = SimpleQueue()
    paused = start_paused
    print("Control panel: drag finger or forearm sliders. Space: pause/resume. R: reset.")
    print("R/L_Palm_Base_Tx: world +Y; TY: world -X; TZ: world +Z.")
    if arm_preview is not None:
        print('M6 arms: differential IK follows actual hand wrists. F: fit full arms+piano camera.')
        print('Paused: Wrist Control sliders position the hand kinematically; arms keep following.')
        print('Mouse: double-click a green wrist handle; Ctrl+right-drag translates, Ctrl+left-drag rotates.')
    wrist_actuators = [i for i in range(model.nu) if 'Palm_Base_' in (model.actuator(i).name or '')]
    if on_save is not None:
        print("P: save both hands as next startup's initial pose (with a backup).")
    timings = {key: deque(maxlen=3600) for key in ('sync_ms', 'physics_ms', 'wire_ms', 'axes_ms', 'work_ms')}
    load_start = time.perf_counter()
    frames = 0
    with ExitStack() as cleanup:
        panel = ContactPanel(contact_controller) if contact_controller is not None else None
        if panel is not None:
            cleanup.callback(panel.close)
        connector_panel = None
        if connector_preview is not None and connector_controls:
            from .preview_connector import ConnectorPanel
            connector_panel = ConnectorPanel(connector_preview)
            cleanup.callback(connector_panel.close)
        viewer = cleanup.enter_context(_managed_viewer(
            model, data, key_callback=keys.put, show_left_ui=True, show_right_ui=True))
        viewer_load_s = time.perf_counter() - load_start
        state_only = not full_sync and 'state_only' in inspect.signature(viewer.sync).parameters
        print('Viewer sync: ' + ('state only' if state_only else 'full (compatibility mode)'))
        loop_start = time.perf_counter()
        with viewer.lock():
            viewer.cam.lookat[:] = (0.03, 0, .06)
            viewer.cam.distance = 1.05
            viewer.cam.azimuth = 155
            viewer.cam.elevation = -50
            if arm_preview is not None:
                arm_preview.camera(viewer.cam)
            configure_options(viewer.opt, show_fingertips, transparent_model, show_hand_collisions)
            if arm_preview is not None and getattr(arm_preview, 'show_collision_regions', False):
                viewer.opt.geomgroup[4] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
        while viewer.is_running() and (duration is None or time.perf_counter() - loop_start < duration):
            if panel is not None:
                panel.poll()
            if connector_panel is not None:
                connector_panel.poll()
            frame_start = time.perf_counter()
            # sync locks internally: never call it while holding viewer.lock().
            # Imports Control slider values into data.ctrl and displays last frame.
            if state_only:
                viewer.sync(state_only=True)
            else:
                viewer.sync()
            after_sync = time.perf_counter()
            timings['sync_ms'].append((after_sync - frame_start) * 1000)
            with viewer.lock():
                if connector_preview is not None:
                    connector_preview.apply()
                if contact_controller is not None:
                    contact_controller.apply()
                reset = False
                save_requested = False
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key == ord(' '):
                        paused = not paused
                    elif key in (ord('R'), ord('r')):
                        reset = True
                    elif key in (ord('P'), ord('p')):
                        save_requested = True
                    elif key in (ord('F'), ord('f')) and arm_preview is not None:
                        arm_preview.camera(viewer.cam)
                steps = max(1, round((1 / 60) / model.opt.timestep))
                frame_duration = steps * model.opt.timestep
                if arm_preview is not None and not reset:
                    arm_preview.apply_drag_handles()
                if reset:
                    mujoco.mj_copyData(data, model, initial)
                elif not paused:
                    # Position actuators consume the user's targets already in ctrl.
                    physics.step(nstep=steps)
                elif arm_preview is not None:
                    # Allow wrist sliders while paused, without stepping piano physics.
                    for actuator in wrist_actuators:
                        joint = model.actuator_trnid[actuator, 0]
                        data.qpos[model.jnt_qposadr[joint]] = data.ctrl[actuator]
                        data.qvel[model.jnt_dofadr[joint]] = 0
                physics.forward()
                if arm_preview is not None:
                    arm_preview.update(frame_duration, reset=reset)
                    if frames % 120 == 0:
                        arm_report=arm_preview.report()
                        collision=arm_report['forearm_collision']
                        print(f"forearm_reward={collision['forearm_reward']:.1f}, collision pairs={len(collision['contacts'])}",flush=True)
                        for side, status in arm_report['arms'].items():
                            request=status['requested_target_error']
                            if request and request['limited']:
                                print(f'M6 {side}: wrist target limited to reachable motion; requested gap {request["position_mm"]:.1f} mm.',flush=True)
                            if status['position_error_mm'] > 5 or status['orientation_error_deg'] > 3:
                                print(f'M6 {side} IK tracking error: {status["position_error_mm"]:.1f} mm, '
                                      f'{status["orientation_error_deg"]:.1f} deg; target may be unreachable.', flush=True)
                after_physics = time.perf_counter()
                if connector_preview is not None:
                    connector_preview.publish()
                if wireframe is not None:
                    wireframe.update_scene(viewer.user_scn)
                after_wire = time.perf_counter()
                if coordinate_frames is not None:
                    coordinate_frames.update_scene(
                        viewer.user_scn, start=wireframe.edge_count if wireframe else 0)
                viewer.user_scn.ngeom = ((wireframe.edge_count if wireframe else 0)
                                         + (coordinate_frames.geom_count if coordinate_frames else 0))
                key_highlight.append(viewer.user_scn, data)
                after_axes = time.perf_counter()
            timings['physics_ms'].append((after_physics - after_sync) * 1000)
            timings['wire_ms'].append((after_wire - after_physics) * 1000)
            timings['axes_ms'].append((after_axes - after_wire) * 1000)
            timings['work_ms'].append((after_axes - frame_start) * 1000)
            if save_requested and on_save is not None:
                # File IO only on an explicit save event, outside the viewer lock.
                try:
                    on_save()
                except (OSError, ValueError, SyntaxError) as error:
                    print(f'Initial pose was not saved: {error}', flush=True)
            frames += 1
            remaining = frame_duration - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
        elapsed = time.perf_counter() - loop_start
    return {'viewer_load_s': viewer_load_s, 'state_only_sync': state_only,
            'frames': frames, 'elapsed_s': elapsed, 'update_hz': frames / max(elapsed, 1e-9),
            'timings': {key: {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95))}
                        for key, values in timings.items() if values}}
