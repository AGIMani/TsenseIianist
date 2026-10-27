"""Lightweight offline-planner contracts; no MuJoCo model loading or physics."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1]/'scripts/hpc/plan_piano_ik.py'
spec = importlib.util.spec_from_file_location('planned_fingering', PATH)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def test_current_planner_defaults_and_explicit_physics_override():
    parser = planner.argument_parser()
    required = ['--score', 'score.proto', '--original-midi', 'original.mid', '--output', 'new']
    args = parser.parse_args(required)
    assert args.physics_hz == 200
    assert args.control_hz == 20
    assert args.coupling and args.coupled_ik
    assert not args.free_key_x
    assert args.thumb_idle_pose is None
    assert parser.parse_args(required + ['--physics-hz', '1000']).physics_hz == 1000
    options = planner.recipe_options(Path('score.proto'), control_hz=20, physics_hz=1000)
    assert options['world_z'] == .09
    assert options['world_x'] == .14
    assert options['palm_base_yaw'] == pytest.approx(np.pi/12)
    assert options['exact_note_timing'] is True
    assert options['score_validation'] == 'strict_v1'
    assert options['gravity_compensation'] is True
    assert options['physics_hz'] == 1000
    assert planner.coupling_options()['angle_ranges_deg']['left_index'] == [10., 80.]


def test_parking_rejects_non_coupled_solver_before_model_work():
    parser=planner.argument_parser()
    required=['--score','x','--original-midi','x','--output','x']
    with pytest.raises(ValueError,match='coupled'):
        planner.validate_planner_options(parser.parse_args(required+['--idle-pinky-parking','--no-coupling']))
    planner.validate_planner_options(parser.parse_args(required+['--idle-pinky-parking']))


def test_non_grid_song_end_reports_actual_recorded_duration_for_packaging():
    song_end = 222.47395833333334
    requested = song_end + .25
    times = np.arange(round(requested/.001)+1)*.001
    timing = planner.recording_timing(times, requested, song_end)
    assert timing['duration_s'] == times[-1]
    assert timing['duration_s'] == pytest.approx(222.724, abs=1e-12)
    assert timing['requested_duration_s'] == requested
    assert timing['full_piece']
    assert timing['duration_s'] != requested
    assert not planner.recording_timing(times[:3001], 3., song_end)['full_piece']


def test_reused_build_rejects_changed_pose_source_and_score():
    expected = dict(pose=dict(world_z=.09), source_sha256={'model.py': 'abc'},
                    external_score={'sha256': 'score'}, environment={'physics_hz': 1000})
    planner.validate_build_recipe(copy.deepcopy(expected), expected)
    for section, field, replacement in [('pose', 'world_z', .06),
                                       ('source_sha256', 'model.py', 'old'),
                                       ('external_score', 'sha256', 'different'),
                                       ('environment', 'physics_hz', 200)]:
        recorded = copy.deepcopy(expected)
        recorded[section][field] = replacement
        with pytest.raises(ValueError, match='build recipe'):
            planner.validate_build_recipe(recorded, expected)


def test_black_mesh_principal_geometry_does_not_move_key_targets():
    # The black mesh centroid/dimensions deliberately do not describe its playing face.
    geoms = [SimpleNamespace(name='piano/white_key_geom_0', bodyid=1),
             SimpleNamespace(name='piano/black_key_geom_1', bodyid=2)]
    sites = {'piano/white_key_site_0': SimpleNamespace(id=0, bodyid=1, size=np.array([.075,.011,.006])),
             'piano/black_key_site_1': SimpleNamespace(id=1, bodyid=2, size=np.array([.045,.005,.009]))}
    model = SimpleNamespace(geom=lambda i: geoms[int(i)], site=lambda n: sites[n],
                            geom_size=np.full((2,3), 999.))
    data = SimpleNamespace(site_xpos=np.array([[.1,.2,.3],[.4,.5,.6]]),
                           geom_xpos=np.full((2,3), -999.))
    positions, sizes, targets = planner.key_geometry(model, data, [0,1])
    np.testing.assert_array_equal(positions, data.site_xpos)
    np.testing.assert_allclose(sizes[1], [.045,.005,.009])
    assert targets.shape == (10,2,3)
    np.testing.assert_allclose(targets[0,0], [.135,.2,.303])
    np.testing.assert_allclose(targets[5,1], [.425,.5,.6045])
    np.testing.assert_array_equal(targets[0], targets[1])


def test_resistance_is_refreshed_before_each_physics_step(monkeypatch):
    # Only the external integrator is replaced; this checks the runtime adapter order.
    data = SimpleNamespace(q=2., friction=0.)
    resistance = SimpleNamespace(update=lambda d: setattr(d, 'friction', .1+.3*abs(d.q)))
    def integrate(model, state):
        state.q -= state.friction
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(mj_step=integrate))
    planner.step_physics(object(), data, resistance)
    assert data.q == pytest.approx(1.3)
    planner.step_physics(object(), data, resistance)
    assert data.q == pytest.approx(.81)


def test_command_audit_includes_reset_velocity_and_acceleration():
    names = ['right_Palm_Base_Tx', 'right_Palm_Base_Ty', 'right_Palm_Base_Pitch']
    contract = planner.wrist_motion_contract(names, [-1,-1,0], [1,1,.5], .05)
    assert contract['channels'][names[0]]['max_velocity'] == 1.99
    assert contract['channels'][names[1]]['max_acceleration'] == 1.99
    assert contract['channels'][names[2]]['max_velocity'] == 3.1416
    accepted = planner.command_motion_diagnostics(np.array([[.001,0,0],[.003,0,0]]),
                                                  contract, np.zeros(3))
    assert accepted['passed']
    assert accepted['channels'][names[0]]['max_abs_velocity'] == pytest.approx(.04)
    assert accepted['channels'][names[0]]['max_abs_acceleration'] == pytest.approx(.4)
    rejected = planner.command_motion_diagnostics(np.array([[.1,0,0]]), contract, np.zeros(3))
    assert not rejected['passed']
    assert rejected['channels'][names[0]]['acceleration_violation_samples'] == 1


def test_note_metrics_keep_misses_and_partial_notes_separate():
    times = np.array([0.,.1,.2,.3,.4])
    events = np.array([[.1,.3,0,1],[.1,.3,1,6],[.35,.6,2,2]])
    actual = np.zeros((5,88),bool)
    actual[2,0] = True
    actual[4,2] = True
    result = planner.recording_metrics(times, actual, events)
    assert result['original_midi_micro_f1'] == pytest.approx(4/7)
    assert result['notes'][0]['activation_recall'] == .5
    assert result['notes'][0]['onset_delay_s'] == pytest.approx(.1)
    assert result['notes'][1]['status'] == 'missed'
    assert result['notes'][2]['status'] == 'partial_window'
    assert result['missed_notes'] == 1


def test_external_wrist_bounds_are_fixed_in_coupled_solve():
    # Existing coupled IK used global qlow/qhigh, ignoring these per-step bounds.
    q = np.array([.9, -.5, .2])
    low, high = np.array([.1-1e-8, -.2, 0.]), np.array([.1+1e-8, .3, 1.])
    projected, active = planner.bounded_ik_variables(q, low, high, np.arange(3))
    assert projected[0] == pytest.approx(.1, abs=1e-8)
    assert projected[1] == -.2
    np.testing.assert_array_equal(active, [1,2])
    assert q[0] == .9  # Only scratch projection output is changed.


def test_coupled_jacobian_perturbations_stay_in_per_step_bounds():
    action = np.zeros(51)
    action[[4,20,23]] = [1.4, .1, .5]
    batch, delta = planner.coupled_finite_difference_batch(action, np.array([4,20,23]),
        np.array([0,2]), np.array([0., .1-1e-8, 0.]), np.array([1.4,.1+1e-8,.5]))
    np.testing.assert_array_equal(batch[:,20], [.1,.1,.1])
    assert batch[1,4] == pytest.approx(1.399)
    assert batch[2,23] == pytest.approx(.499)
    np.testing.assert_allclose(delta, [-.001,-.001])
    assert np.max(batch[:,4]) <= 1.4


def test_joint_seed_selects_preceding_exact_event_without_interpolation(tmp_path):
    path = tmp_path/'seeds.npz'
    actions = np.zeros((3,51)); actions[:,4] = [.1,.5,.9]
    np.savez(path, time=[0.,.125,.25], actions=actions)
    reference = planner.load_joint_seed_reference(path)
    assert planner.joint_seed_at(reference, .10, lead=.05)[4] == .5
    assert planner.joint_seed_at(reference, .124, lead=0)[4] == .1
    seed = planner.joint_seed_at(reference, .25, lead=0)
    seed[4] = 0
    assert reference['actions'][-1,4] == .9
    np.savez(path, time=[0.,.125,.1], actions=actions)
    with pytest.raises(ValueError, match='increasing'):
        planner.load_joint_seed_reference(path)


def test_joint_seed_grid_arithmetic_does_not_select_previous_50ms_frame():
    # A .001s physical clock and .05s reference grid differ by several ULPs.
    times = np.arange(4456) * .05
    actions = np.zeros((len(times), 51))
    actions[:, 4] = np.arange(len(times))
    reference = dict(time=times, actions=actions)
    for control in range(4454):
        physical_time = (control * 50) * .001
        assert planner.joint_seed_at(reference, physical_time, lead=.05)[4] == control + 1
    # Tolerance only repairs floating arithmetic, not actual early transitions.
    assert planner.joint_seed_at(reference, times[100] - 1e-9)[4] == 99


