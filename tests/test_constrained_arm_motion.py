"""Continuous bounds, exact derivative plots, and original-tempo CLI behavior."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'preview/scripts')]
from constrained_arm_motion import spline_basis, certify_coefficients, load_path, save_path
from joint_motion_plots import export_joint_motion
from export_videos import parse_args


def test_continuous_bounds_cover_dense_derivatives_and_stationary_endpoints(tmp_path):
    t = np.linspace(1, 3.005, 403)
    knots, basis, d1, d2 = spline_basis(t, .03)
    c = np.sin(np.linspace(0, .2, basis.shape[1]))[:, None]
    c[:3] = c[0]; c[-3:] = c[-1]
    path = BSpline(knots, c, 3, extrapolate=False)
    velocity = np.abs(d1 @ c).max(axis=0)
    acceleration = np.abs(d2 @ c).max(axis=0)
    cert = certify_coefficients(c, d1, d2, np.array([[-1, 1]]), velocity*1.01, acceleration*1.01)
    assert cert['passed']
    dense = np.linspace(t[0], t[-1], 100003)
    assert np.max(np.abs(path(dense, 1))) <= velocity[0]+1e-12
    assert np.max(np.abs(path(dense, 2))) <= acceleration[0]+1e-12
    np.testing.assert_allclose(basis @ c, path(t), atol=1e-13)
    for derivative in (1, 2):
        np.testing.assert_allclose(path([t[0], t[-1]], derivative), 0, atol=1e-10)
    assert not certify_coefficients(c, d1, d2, np.array([[-1, 1]]), velocity*.9, acceleration)['passed']
    assert not certify_coefficients(c, d1, d2, np.array([[-1, 1]]), velocity, acceleration*.9)['passed']
    assert not certify_coefficients(c, d1, d2, np.array([[-1, .01]]), velocity, acceleration)['passed']
    save_path(tmp_path/'path.npz', path, cert)
    np.testing.assert_array_equal(load_path(tmp_path/'path.npz')(dense), path(dense))


@pytest.mark.parametrize('time', [[0, 0, 1], [0, np.nan, 1], [1, 0, 2]])
def test_invalid_time_does_not_produce_a_path(time):
    with pytest.raises(ValueError):
        spline_basis(time)


def test_plots_use_analytic_rendered_path_and_include_acceleration_extrema(tmp_path):
    # A cubic Bezier has varying velocity and acceleration. Coarse finite
    # differences miss endpoint acceleration; exact plotted derivatives must not.
    path = BSpline([0, 0, 0, 0, 1, 1, 1, 1], np.array([[0.], [0.], [.1], [.1]]), 3)
    t = np.array([0., .5, 1.])
    report = export_joint_motion(tmp_path/'curves', t, path(t),
        [dict(name='right_J1', side='right', unit='rad')], 'test', continuous_path=path)
    data = np.load(tmp_path/'curves/joint_motion_samples.npz')
    np.testing.assert_allclose(data['velocity'][:, 0], .6*data['velocity_time']*(1-data['velocity_time']))
    np.testing.assert_allclose(data['acceleration'][:, 0], .6-1.2*data['acceleration_time'])
    assert report['joints'][0]['acceleration']['maximum_absolute'] == pytest.approx(.6)
    assert 'Analytic' in report['acceleration_definition']
    with pytest.raises(AssertionError, match='rendered continuous path'):
        export_joint_motion(tmp_path/'wrong', t, path(t)+.01,
                            [dict(name='right_J1', side='right', unit='rad')], 'test', continuous_path=path)


def test_constrained_default_keeps_original_tempo_and_explicit_tolerances():
    a = parse_args(['--pt', 'a.pt', '--video1', 'arms', 'front'])
    assert a.arm_ik == 'constrained' and a.speed == 1
    assert a.arm_position_tolerance_mm == .1 and a.arm_rotation_tolerance_deg == .5
    with pytest.raises(SystemExit):
        parse_args(['--pt', 'a.pt', '--video1', 'arms', 'front', '--speed', '2'])
    a = parse_args(['--pt', 'a.pt', '--video1', 'arms', 'front', '--speed', '2', '--arm-ik', 'raw'])
    assert a.arm_ik == 'raw'
    for option in ('--arm-position-tolerance-mm', '--arm-rotation-tolerance-deg'):
        with pytest.raises(SystemExit):
            parse_args(['--pt', 'a.pt', option, '0'])
