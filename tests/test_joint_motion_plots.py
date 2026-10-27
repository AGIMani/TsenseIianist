"""Analytic finite differences and actual compiled joint-unit selection."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'preview/scripts'))
from joint_motion_plots import derivatives, export_joint_motion
from export_video_job import hand_motion_columns
from arm_motion_limits import load_m6_joint_limits, limits_table_html, M6_LIMITS


@pytest.mark.parametrize('time', [np.linspace(0, 2, 81), np.array([0., .03, .17, .4, .41, 1.3, 2.])])
def test_derivatives_of_quadratic_and_constant_velocity(time):
    q = np.column_stack((2*time**2-3*time+1, -.4*time+7))
    vt, v, at, a = derivatives(time, q)
    np.testing.assert_allclose(v[:, 0], 4*vt-3, atol=1e-12)
    np.testing.assert_allclose(v[:, 1], -.4, atol=1e-12)
    np.testing.assert_allclose(a[:, 0], 4, atol=1e-10)
    np.testing.assert_allclose(a[:, 1], 0, atol=1e-10)
    assert len(vt) == len(time)-1 and len(at) == len(time)-2
    assert time[0] < vt[0] < at[0] < at[-1] < vt[-1] < time[-1]


@pytest.mark.parametrize('time,q', [([0., 0., 1.], [[1], [2], [3]]),
                                     ([1., 0.], [[1], [2]]),
                                     ([0., 1.], [[1], [np.nan]]),
                                     ([0.], [[1]])])
def test_invalid_positions_or_times_are_rejected(time, q):
    with pytest.raises(ValueError):
        derivatives(time, q)


def test_short_clip_does_not_invent_acceleration(tmp_path):
    t = np.array([3., 3.001]);q = np.array([[1.], [1.002]])
    report = export_joint_motion(tmp_path/'curve', t, q, [dict(name='right_J1', side='right', unit='rad')], 'test')
    assert report['velocity_samples'] == 1 and report['acceleration_samples'] == 0
    assert report['joints'][0]['acceleration'] is None
    assert report['reference_limits'] is None
    np.testing.assert_allclose(report['joints'][0]['velocity']['maximum_absolute'], 2.)
    assert (tmp_path/'curve/right_J1_acceleration.png').is_file()
    assert (tmp_path/'curve/right_J1_velocity.pdf').is_file()


def test_physical_slide_and_hinge_units_follow_compiled_joint_definitions():
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="piano/key"/><geom size=".1"/></body>
      <body><joint name="right_/palm_base_tx" type="slide"/><geom size=".1"/></body>
      <body><joint name="right_/finger" type="hinge"/><geom size=".1"/></body>
      <body><joint name="left_/finger"/><geom size=".1"/></body>
      </worldbody></mujoco>''')
    poses = np.arange(12).reshape(3, 4)
    selected, joints = hand_motion_columns(model, poses)
    np.testing.assert_array_equal(selected, poses[:, 1:])
    assert [j['unit'] for j in joints] == ['m', 'rad', 'rad']
    assert [j['name'] for j in joints] == ['right_/palm_base_tx', 'right_/finger', 'left_/finger']


def test_bundled_sdk_limits_match_vendor_file_and_convert_degree_units():
    limits = load_m6_joint_limits()
    assert len(limits['joints']) == 14
    assert limits['controller_limits_verified'] is False
    for side in ('right', 'left'):
        for j in range(1, 8):
            assert limits['joints'][f'{side}_J{j}']['velocity'] == pytest.approx(np.pi)
            assert limits['joints'][f'{side}_J{j}']['acceleration'] == pytest.approx(
                2.5*np.pi if j <= 2 else 5*np.pi)
    source = ROOT/limits['source']['path']
    if source.exists():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == limits['source']['sha256']
        rows = source.read_text().splitlines()
        for block, side in enumerate(('left', 'right')):
            for j in range(1, 8):
                values = [float(x) for x in rows[block*13+j].split(',') if x]
                assert limits['sdk_joints'][f'{side}_J{j}'] == {
                    'velocity': values[6], 'acceleration': values[7]}
    page = limits_table_html(limits, 'arm_joints')
    assert '7.853982' in page and '15.707963' in page and '3.141593' in page
    assert 'arm_joints/joint_limits.json' in page


def test_signed_limits_and_exceedances_remain_visible_without_clipping(tmp_path, monkeypatch):
    import matplotlib.figure
    plots = {}
    original = matplotlib.figure.Figure.savefig

    def capture(self, filename, *args, **kwargs):
        # Inspect what is actually drawn; positive and negative spikes must survive.
        if Path(filename).name in ('right_J1_velocity.png', 'right_J1_acceleration.png'):
            axis = self.axes[0]
            plots[Path(filename).name] = {
                'limits': [float(line.get_ydata()[0]) for line in axis.lines if line.get_linestyle() == '--'],
                'curve': np.asarray(axis.lines[0].get_ydata()), 'ylim': axis.get_ylim(),
                'red_points': np.asarray(axis.collections[0].get_offsets()),
            }
        return original(self, filename, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    limits = {'units': {'velocity': 'rad/s', 'acceleration': 'rad/s^2'},
              'joints': {'right_J1': {'velocity': 3., 'acceleration': 5.}}}
    result = export_joint_motion(tmp_path/'curve', [0, 1, 2, 3], [[0], [2], [-2], [0]],
                                 [dict(name='right_J1', side='right', unit='rad')], 'test', limits=limits)
    v, a = result['joints'][0]['velocity'], result['joints'][0]['acceleration']
    assert v['exceeded_samples'] == 1 and v['first_exceedance_s'] == 1.5
    assert v['peak_to_limit_ratio'] == pytest.approx(4/3)
    assert a['exceeded_samples'] == 2 and a['exceeded_sample_fraction'] == 1.
    assert a['first_exceedance_s'] == 1. and a['last_exceedance_s'] == 2.
    for name, expected, bounds in [('velocity', [2, -4, 2], [3., -3.]),
                                    ('acceleration', [-6, 6], [5., -5.])]:
        drawn = plots[f'right_J1_{name}.png']
        assert drawn['limits'] == bounds
        np.testing.assert_array_equal(drawn['curve'], expected)
        assert drawn['ylim'][0] < min(expected + bounds)
        assert drawn['ylim'][1] > max(expected + bounds)
    np.testing.assert_array_equal(plots['right_J1_velocity.png']['red_points'], [[1.5, -4.]])
    np.testing.assert_array_equal(plots['right_J1_acceleration.png']['red_points'], [[1., -6.], [2., 6.]])
    assert json.loads((tmp_path/'curve/joint_limits.json').read_text()) == limits


@pytest.mark.parametrize('name,unit,value', [('left_J1', 'rad/s', 3.),
                                           ('right_J1', 'deg/s', 180.),
                                           ('right_J1', 'rad/s', 0.),
                                           ('right_J1', 'rad/s', float('inf'))])
def test_invalid_or_mismatched_limits_are_rejected_before_export(tmp_path, name, unit, value):
    limits = {'units': {'velocity': unit, 'acceleration': 'rad/s^2'},
              'joints': {name: {'velocity': value, 'acceleration': 5.}}}
    with pytest.raises(ValueError, match='Reference limits'):
        export_joint_motion(tmp_path/'curve', [0, 1, 2], [[0], [1], [2]],
                            [dict(name='right_J1', side='right', unit='rad')], 'test', limits=limits)
    assert not (tmp_path/'curve').exists()


def test_out_of_range_positions_cannot_be_exported_as_valid_joint_csv(tmp_path):
    joints = [dict(name='left_pip', side='left', unit='rad', position_min=0., position_max=1.57)]
    with pytest.raises(ValueError, match='outside limits'):
        export_joint_motion(tmp_path/'invalid', [0., .01], [[0.], [-.001]], joints, 'test')
    assert not (tmp_path/'invalid').exists()


def test_recorded_physics_preserves_both_bound_exceedances_and_derivatives(tmp_path, monkeypatch):
    import matplotlib.figure
    drawn = {}
    original = matplotlib.figure.Figure.savefig
    def capture(self, path, *args, **kwargs):
        if Path(path).name == 'right_yaw_position.png':
            ax = self.axes[0]
            drawn.update(curve=np.asarray(ax.lines[0].get_ydata()),
                         points=np.asarray(ax.collections[0].get_offsets()), ylim=ax.get_ylim())
        return original(self, path, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    time = np.array([0., .005, .01, .015])
    q = np.array([[0., 2.], [-.001, 3.], [.27, 4.], [.1, 5.]])
    original_q = q.copy()
    joints = [dict(name='right_yaw', side='right', unit='rad', position_min=0., position_max=.26),
              dict(name='left_unlimited', side='left', unit='m')]
    folder = tmp_path/'measured'
    report = export_joint_motion(folder, time, q, joints, 'test', position_source='recorded_physics')
    assert report['position_bounds_status'] == 'exceeded'
    assert report['positions_modified'] is False
    audit = report['joints'][0]['position']
    assert audit['below_limit_samples'] == audit['above_limit_samples'] == 1
    assert audit['maximum_undershoot'] == pytest.approx(.001)
    assert audit['maximum_overshoot'] == pytest.approx(.01)
    assert audit['first_exceedance_s'] == .005 and audit['last_exceedance_s'] == .01
    assert report['joints'][1]['position'] is None
    np.testing.assert_array_equal(q, original_q)
    np.testing.assert_array_equal(np.loadtxt(folder/'joint_positions.csv', delimiter=',', skiprows=1),
                                  np.column_stack((time, original_q)))
    with np.load(folder/'joint_motion_samples.npz') as saved:
        np.testing.assert_array_equal(saved['positions'], original_q)
        np.testing.assert_array_equal(saved['velocity'], derivatives(time, original_q)[1])
        np.testing.assert_array_equal(saved['acceleration'], derivatives(time, original_q)[3])
    np.testing.assert_array_equal(drawn['curve'], original_q[:, 0])
    np.testing.assert_array_equal(drawn['points'], [[.005, -.001], [.01, .27]])
    assert drawn['ylim'][0] < -.001 and drawn['ylim'][1] > .27


def test_recorded_physics_rejects_synthetic_path(tmp_path):
    with pytest.raises(ValueError, match='cannot use a planned'):
        export_joint_motion(tmp_path/'invalid', [0., .01], [[0.], [0.]],
                            [dict(name='yaw', side='right', unit='rad')], 'test',
                            position_source='recorded_physics', continuous_path=lambda t: t)
