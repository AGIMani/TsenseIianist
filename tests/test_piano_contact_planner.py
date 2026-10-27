"""Lightweight mathematical tests; never build or load a MuJoCo scene here."""
import numpy as np

from scripts.hpc.piano_contact_planner import fingertip_residual, solve_bounded_pose, terminal_top_error, idle_pinky_coordinates


def test_idle_clearance_does_not_pull_a_safe_finger_down():
    actual = np.array([[0., 0., .07], [0., 0., .02]])
    wanted = np.array([[0., 0., .045], [0., 0., .045]])
    errors = fingertip_residual(actual, wanted, np.ones((2, 3)), active=set())
    np.testing.assert_allclose(errors[:, 2], [0., -.025])


def test_active_press_still_requires_exact_height_and_free_x_stays_in_top_domain():
    actual = np.array([[.02, .01, .03], [.06, .0, .01]])
    wanted = np.array([[.035, .0, .01], [.035, .0, .01]])
    result = fingertip_residual(actual, wanted, np.ones((2, 3)), active={0, 1},
                               x_bounds={0: (-.01, .04), 1: (-.01, .04)})
    np.testing.assert_allclose(result, [[0., .01, .02], [.02, 0., 0.]])


def test_bounded_pose_solves_nonlinear_contact_without_changing_fixed_wrist():
    # An exact fixed coordinate is deliberately part of the requested set.
    visited = []
    def residual(q):
        visited.append(q.copy())
        return np.array([q[0] ** 2 - .25, q[1] - .2])
    result = solve_bounded_pose(np.array([.1, .2]), np.array([0., .2]),
                                np.array([1., .2]), [0, 1], residual, max_nfev=60)
    assert result.success
    np.testing.assert_allclose(result.pose, [.5, .2], atol=1e-6)
    assert all(q[1] == .2 and 0 <= q[0] <= 1 for q in visited)


def test_bounded_pose_keeps_best_feasible_boundary_instead_of_overshooting():
    result = solve_bounded_pose(np.array([.01]), np.array([0.]), np.array([.4]),
                                [0], lambda q: q - 1., max_nfev=30)
    np.testing.assert_allclose(result.pose, [.4], atol=1e-6)
    assert result.max_abs_residual >= .6 - 1e-8


def test_fully_fixed_pose_is_evaluated_without_a_solver():
    result = solve_bounded_pose(np.array([.2]), np.array([.2]), np.array([.2]),
                                [0], lambda q: q - .5)
    np.testing.assert_allclose(result.pose, [.2])
    assert result.nfev == 1


def test_contact_point_uses_lowest_mesh_vertex_and_stays_inside_finite_key():
    vertices=np.array([[.046,0.,.012],[.02,0.,.018]])
    error,point=terminal_top_error(vertices,np.zeros(3),np.eye(3),np.array([.045,.005,.01]),
                                  edge_margin=.005,vertical_offset=0.)
    np.testing.assert_allclose(point,[.046,0.,.012])
    np.testing.assert_allclose(error,[.006,0.,.002])


def test_contact_error_rotates_with_depressed_key_top_plane():
    angle=.1
    rotation=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
    center=np.array([.2,.3,.4]);local=np.array([[0.,.003,.01],[0.,0.,.015]])
    error,point=terminal_top_error(local@rotation.T+center,center,rotation,
                                  np.array([.045,.005,.01]),vertical_offset=-.001)
    np.testing.assert_allclose(error,[0.,.003,.001],atol=1e-10)
    np.testing.assert_allclose(point,local[0],atol=1e-10)


def test_idle_pinky_names_freeze_only_mcp_and_swing_never_coupled_followers():
    names=['R_Pinky_DIP','R_Pinky_SW','R_Pinky_MCP','R_Pinky_PIP',
           'L_Pinky_DIP','L_Pinky_MCP','L_Pinky_SW','L_Pinky_PIP']
    full=[2,7,1,4,0,5,6,3]
    coordinates=idle_pinky_coordinates(names,full,set())
    assert set(coordinates)=={0,2,5,6}
    assert coordinates[2]==.01000039
    assert coordinates[0]==.490541428
    assert set(idle_pinky_coordinates(names,full,{4}))=={5,6}
