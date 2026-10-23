"""Exercise the real physics preview loop with GUI events supplied without a display."""
from contextlib import contextmanager
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tsensei_v3_2_pianist import preview_runtime, suite
from tsensei_v3_2_pianist.preview_geometry import PadCollisionWireframe
from tsensei_v3_2_pianist.preview_frames import CoordinateFrames


@pytest.fixture(scope='module')
def env():
    environment = suite.load()
    # Isolate actuator response from piano contact; contact behavior is covered
    # by the existing environment tests at the real default mounting height.
    for hand in (environment.task.right_hand, environment.task.left_hand):
        hand.root_body.pos = np.asarray(hand.root_body.pos) + (0, 0, .1)
    yield environment
    environment.close()


class FakeViewer:
    """Only GUI event transport is mocked; controls and MuJoCo physics are real."""
    def __init__(self, physics, frames, on_sync=None, maxgeom=10):
        self.frames, self.frame, self.on_sync = frames, 0, on_sync
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.opt = mujoco.MjvOption()
        self.user_scn = mujoco.MjvScene(physics.model.ptr, maxgeom=maxgeom)
        self.locked = False

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def is_running(self): return self.frame < self.frames

    @contextmanager
    def lock(self):
        assert not self.locked
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    def sync(self, state_only=False):
        assert state_only, "Normal preview must avoid full model copies"
        assert not self.locked, 'sync locks internally and must run outside viewer.lock'
        if self.on_sync:
            self.on_sync(self.frame, self)
        self.frame += 1


def run_gui(monkeypatch, env, frames=60, on_sync=None, wireframe=None, paused=False,
            coordinate_frames=None, on_save=None, arm_preview=None):
    import mujoco.viewer
    viewer = FakeViewer(env.physics, frames, on_sync,
                        maxgeom=10 + (wireframe.edge_count if wireframe else 0)
                        + (coordinate_frames.geom_count if coordinate_frames else 0))
    def launch(model, data, **kwargs):
        assert kwargs['show_right_ui']
        viewer.key_callback = kwargs['key_callback']
        return viewer
    monkeypatch.setattr(mujoco.viewer, 'launch_passive', launch)
    monkeypatch.setattr(preview_runtime.time, 'sleep', lambda seconds: None)
    preview_runtime.run_interactive(env.physics, wireframe=wireframe, start_paused=paused,
                                    coordinate_frames=coordinate_frames, on_save=on_save,
                                    arm_preview=arm_preview)
    return viewer


def test_save_key_consumes_latest_paused_controls_and_survives_rejection(env, monkeypatch, capsys):
    env.reset()
    before = env.physics.data.qpos.copy()
    saved = []
    def save():
        saved.append(env.physics.data.ctrl.copy())
        if len(saved) == 1:
            raise ValueError('simulated config conflict')
    def events(frame, viewer):
        env.physics.data.ctrl[0] = .1 + .01 * frame
        if frame in (0, 2):
            viewer.key_callback(ord('P'))
    run_gui(monkeypatch, env, frames=3, paused=True, on_sync=events, on_save=save)
    assert len(saved) == 2
    assert saved[0][0] == .1
    assert saved[1][0] == pytest.approx(.12)
    np.testing.assert_array_equal(env.physics.data.qpos, before)
    assert 'simulated config conflict' in capsys.readouterr().out


def test_paused_arm_preview_consumes_slider_and_mouse_target(monkeypatch):
    from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview
    environment = suite.load()
    try:
        arms = ArmPreview(environment.task.root_entity.mjcf_model)
        environment.reset()
        physics = environment.physics
        arms.bind(physics, (environment.task.right_hand, environment.task.left_hand))
        aid = next(i for i in range(physics.model.nu)
                   if physics.model.ptr.actuator(i).name.endswith('R_Palm_Base_Tx'))
        jid = physics.model.ptr.actuator_trnid[aid, 0]
        qadr = physics.model.ptr.jnt_qposadr[jid]
        initial_time = physics.data.time
        def events(frame, viewer):
            if frame == 0:
                physics.data.ctrl[aid] = .025
            elif frame == 1:
                assert physics.data.qpos[qadr] == pytest.approx(.025)
                physics.data.mocap_pos[arms.handle_ids['right']] += [0, .01, 0]
        run_gui(monkeypatch, environment, frames=60, paused=True,
                on_sync=events, arm_preview=arms)
        assert physics.data.qpos[qadr] == pytest.approx(.035, abs=2e-5)
        assert physics.data.time == initial_time
        assert arms.report()['arms']['right']['position_error_mm'] < 1
    finally:
        environment.close()


@pytest.mark.parametrize('with_wire', [False, True])
def test_world_and_wrist_axes_follow_controls_and_preserve_wire_cache(env, monkeypatch, with_wire):
    env.reset()
    hands = (env.task.right_hand, env.task.left_hand)
    axes = CoordinateFrames(env.physics, hands)
    wire = PadCollisionWireframe(env.physics, hands) if with_wire else None
    count = 0
    if wire:
        append = wire.append_to_scene
        def counted_append(scene):
            nonlocal count
            count += 1
            append(scene)
        monkeypatch.setattr(wire, 'append_to_scene', counted_append)
    def move(frame, viewer):
        if frame == 0:
            for hand in hands:
                for name, target in [('forearm_tx', .02), ('forearm_pitch', .12), ('forearm_yaw', .2)]:
                    actuator = hand.actuator_for_joint(name)
                    env.physics.bind(actuator).ctrl = target
    viewer = run_gui(monkeypatch, env, frames=12, on_sync=move,
                     wireframe=wire, coordinate_frames=axes)
    start = wire.edge_count if wire else 0
    assert viewer.user_scn.ngeom == start + axes.geom_count
    assert count == int(with_wire), 'Axes must not cause wireframe reconstruction every frame'
    origins = [np.zeros(3)] + [env.physics.bind(hand.root_body).xpos for hand in hands]
    rotations = [np.eye(3)] + [env.physics.bind(hand.root_body).xmat.reshape(3, 3) for hand in hands]
    for frame, (origin, rotation) in enumerate(zip(origins, rotations)):
        for axis in range(3):
            arrow = viewer.user_scn.geoms[start + 6 * frame + 2 * axis]
            np.testing.assert_allclose(arrow.pos, origin, atol=1e-7)
            np.testing.assert_allclose(arrow.pos + arrow.mat[:, 2] * arrow.size[2],
                                       origin + axes.axis_length * rotation[:, axis], atol=1e-7)
    assert not np.allclose(rotations[1], rotations[0])
    qpos, ctrl = env.physics.data.qpos.copy(), env.physics.data.ctrl.copy()
    axes.update_scene(viewer.user_scn, start)
    np.testing.assert_array_equal(env.physics.data.qpos, qpos)
    np.testing.assert_array_equal(env.physics.data.ctrl, ctrl)


def test_control_sliders_drive_both_fingers_and_three_position_axes(env, monkeypatch):
    env.reset()
    hands = (env.task.right_hand, env.task.left_hand)
    joints = [hand.mjcf_model.find('joint', hand.hand_side.name.lower() + '_hand_index_MCP_joint')
              for hand in hands]
    run_gui(monkeypatch, env)
    baseline_roots = [env.physics.bind(hand.root_body).xpos.copy() for hand in hands]
    baseline_angles = [float(env.physics.bind(joint).qpos[0]) for joint in joints]
    env.reset()
    targets = {}
    for hand in hands:
        for actuator in hand.actuators:
            name = actuator.joint.name
            if name in ('forearm_tx', 'forearm_ty', 'forearm_tz') or name.endswith('index_MCP_joint'):
                current = float(env.physics.bind(actuator).ctrl)
                targets[actuator] = current + {'forearm_tx': .012, 'forearm_ty': .008,
                                               'forearm_tz': -.005}.get(name, .15)
    def sliders(frame, viewer):
        if frame == 0:
            for actuator, target in targets.items():
                env.physics.bind(actuator).ctrl = target
    run_gui(monkeypatch, env, on_sync=sliders)
    for i, hand in enumerate(hands):
        delta = env.physics.bind(hand.root_body).xpos - baseline_roots[i]
        assert delta[0] < -.003, delta
        assert delta[1] > .005, delta
        assert delta[2] < -.002, delta
        assert float(env.physics.bind(joints[i]).qpos[0]) > baseline_angles[i] + .03
    for actuator, target in targets.items():
        np.testing.assert_allclose(env.physics.bind(actuator).ctrl, [target])
    assert np.isfinite(env.physics.data.qpos).all()
    assert env.physics.data.time > .5


def test_pause_resume_and_reset_keep_control_targets(env, monkeypatch):
    env.reset()
    initial_qpos = env.physics.data.qpos.copy()
    def set_slider(frame, viewer):
        env.physics.bind(env.task.right_hand.actuators[0]).ctrl = .2
    run_gui(monkeypatch, env, frames=3, paused=True, on_sync=set_slider)
    np.testing.assert_array_equal(env.physics.data.qpos, initial_qpos)
    assert env.physics.data.time == 0
    np.testing.assert_allclose(env.physics.bind(env.task.right_hand.actuators[0]).ctrl, [.2])
    def resume(frame, viewer):
        if frame == 0:
            viewer.key_callback(ord(' '))
    run_gui(monkeypatch, env, frames=3, paused=True, on_sync=resume)
    assert env.physics.data.time > 0
    env.reset()
    initial_ctrl = env.physics.data.ctrl.copy()
    def move_then_reset(frame, viewer):
        if frame == 0:
            env.physics.data.ctrl[0] += .2
        if frame == 4:
            viewer.key_callback(ord('R'))
    run_gui(monkeypatch, env, frames=5, on_sync=move_then_reset)
    np.testing.assert_array_equal(env.physics.data.qpos, initial_qpos)
    np.testing.assert_array_equal(env.physics.data.ctrl, initial_ctrl)
    assert env.physics.data.time == 0


@pytest.fixture
def mesh_env(monkeypatch):
    # Hull wireframe tests need actual mesh geoms; capsule dataid is -1.
    monkeypatch.setenv('TSENSEI_V32_COLLISION_PROFILE', 'convex_hull_v1')
    environment = suite.load()
    for hand in (environment.task.right_hand, environment.task.left_hand):
        hand.root_body.pos = np.asarray(hand.root_body.pos) + (0, 0, .1)
    yield environment
    environment.close()


def test_wireframe_tracks_moving_hands_without_accumulating(mesh_env, monkeypatch):
    env = mesh_env
    env.reset()
    wire = PadCollisionWireframe(env.physics, (env.task.right_hand, env.task.left_hand))
    initial_scene = mujoco.MjvScene(env.physics.model.ptr, maxgeom=wire.edge_count + 10)
    wire.append_to_scene(initial_scene)
    initial_positions = np.array([g.pos.copy() for g in initial_scene.geoms[:initial_scene.ngeom]])
    def slider(frame, viewer):
        if frame == 0:
            actuator = env.task.right_hand.actuator_for_joint('forearm_tx')
            env.physics.bind(actuator).ctrl = .02
    viewer = run_gui(monkeypatch, env, frames=10, on_sync=slider, wireframe=wire)
    assert viewer.user_scn.ngeom == wire.edge_count
    expected_scene = mujoco.MjvScene(env.physics.model.ptr, maxgeom=wire.edge_count + 10)
    wire.append_to_scene(expected_scene)
    actual_positions = np.array([g.pos.copy() for g in viewer.user_scn.geoms[:viewer.user_scn.ngeom]])
    expected_positions = np.array([g.pos.copy() for g in expected_scene.geoms[:expected_scene.ngeom]])
    np.testing.assert_allclose(actual_positions, expected_positions)
    assert np.max(np.abs(actual_positions - initial_positions)) > .001

    # A line's far endpoint uses its local +Z axis and length. Cached transforms
    # must preserve both endpoints when the hand rotates, not only line origins.
    def endpoints(scene):
        return np.array([g.pos + g.mat[:, 2] * g.size[2]
                         for g in scene.geoms[:scene.ngeom]])
    np.testing.assert_allclose(endpoints(viewer.user_scn), endpoints(expected_scene), atol=1e-7)
    # Paused frames must reuse lines without invoking either initialization path.
    monkeypatch.setattr(wire, 'append_to_scene', lambda scene: pytest.fail('rebuilt static lines'))
    previous = endpoints(viewer.user_scn)
    wire.update_scene(viewer.user_scn)
    np.testing.assert_array_equal(endpoints(viewer.user_scn), previous)


def test_window_close_waits_for_its_render_thread(monkeypatch):
    import threading
    import mujoco.viewer
    stop, cleaned = threading.Event(), threading.Event()
    def render():
        assert stop.wait(5)
        cleaned.set()
    worker = threading.Thread(target=render, daemon=True)
    class Handle:
        def __enter__(self): return self
        def __exit__(self, *args): stop.set()
    def launch(*args, **kwargs):
        worker.start()
        return Handle()
    monkeypatch.setattr(mujoco.viewer, '_launch_internal', render)
    monkeypatch.setattr(mujoco.viewer, 'launch_passive', launch)
    with preview_runtime._managed_viewer(None, None):
        assert worker.is_alive()
    assert cleaned.is_set()
    assert not worker.is_alive()
