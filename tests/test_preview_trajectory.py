"""Independent circular-motion checks for TCP derivatives and time scaling."""
from types import SimpleNamespace
import mujoco
import numpy as np

from tsensei_v3_2_pianist.preview_trajectory import path_dynamics, scale_path


def circular_arms():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="r" axis="0 0 1"/><geom size=".1" mass="1"/>
        <site name="right" pos="1 0 0"/></body>
      <body pos="0 3 0"><joint name="l" axis="0 0 1"/><geom size=".1" mass="1"/>
        <site name="left" pos="1 0 0"/></body>
    </worldbody></mujoco>''')
    return SimpleNamespace(model=model, data=mujoco.MjData(model), sides={
        side: dict(sid=model.site(side).id, qadr=np.array([i]), dof=np.array([i]),
                   limits=np.array([[-3., 3.]]), velocity=np.array([.8]))
        for i, side in enumerate(('right', 'left'))})


def test_tcp_velocity_and_acceleration_match_analytic_circle():
    arms = circular_arms()
    q = np.array([[.2, -.3], [.8, -.7]])
    qd = np.array([[1.2, -.6], [.5, 1.1]])
    qdd = np.array([[-.4, .3], [.7, -.8]])
    _, values = path_dynamics(arms, q, qd, qdd)
    for k in range(2):
        for i in range(2):
            c, s, v, a = np.cos(q[k, i]), np.sin(q[k, i]), qd[k, i], qdd[k, i]
            expected = [[-s*v, c*v, 0], [-c*v*v-s*a, -s*v*v+c*a, 0], [0, 0, v], [0, 0, a]]
            np.testing.assert_allclose(values[k, i], expected, atol=1e-9)


def test_time_scaling_satisfies_all_tcp_and_vendor_speed_limits():
    arms = circular_arms()
    time = np.linspace(0, 1., 21)
    q = np.column_stack((.8*np.sin(np.pi*time), .6*np.sin(np.pi*time)))
    limits = dict(linear_speed_m_s=.2, linear_acceleration_m_s2=.4,
                  angular_speed_rad_s=.3, angular_acceleration_rad_s2=.5)
    spline, report, audit = scale_path(time, q, arms, limits, audit_hz=200)
    scale = report['time_scale']
    assert scale > 1.
    np.testing.assert_allclose(spline([0., 1.], 1), 0, atol=1e-12)
    actual = np.linalg.norm(audit['limited_tcp_vectors'], axis=-1).max(axis=(0, 1))
    assert np.all(actual <= np.array(list(limits.values())))
    assert np.max(np.abs(audit['qd'])/scale) <= .8
    # Independently re-evaluate at the scaled velocities and accelerations.
    _, physical = path_dynamics(arms, audit['q'][::10], audit['qd'][::10]/scale, audit['qdd'][::10]/scale**2)
    np.testing.assert_allclose(physical, audit['limited_tcp_vectors'][::10], atol=1e-9)
