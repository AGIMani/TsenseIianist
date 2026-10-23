"""Raw palm request vs rigid actual-body tracking; pure CPU arithmetic tests."""
import math
from types import SimpleNamespace

import pytest
import torch

from tests.test_paper_rewards import scene_fixture, only
from tsensei_v3_2_pianist.mjlab_env.reward_config import validate
from tsensei_v3_2_pianist.mjlab_env.rewards import (
    ArmTrackingErrorAccumulator, ArmTrackingMetrics, RewardManager)
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config

TERM = 'arm_tracking_error_penalty'


def evaluate(errors, *, config=None, dt=.05):
    errors = torch.as_tensor(errors, dtype=torch.float64)
    config = only(TERM) if config is None else config
    manager = RewardManager(config, dict(scene_fixture(), control_dt=dt), {},
                            num_envs=len(errors), device='cpu')
    sample = dict(arm_tracking_requested_error_m=errors,
                  arm_tracking_ik_target_error_m=torch.zeros_like(errors))
    result = manager({}, torch.zeros(len(errors), 51), sample,
                     torch.zeros(len(errors), 89), torch.zeros(len(errors), 10, 88))
    return result[TERM], manager


def positions(x_by_side):
    x = torch.as_tensor(x_by_side, dtype=torch.float64)
    result = torch.zeros((*x.shape, 3), dtype=torch.float64)
    result[..., 0] = x
    return result


def test_dense_penalty_targets_zero_and_keeps_distinguishing_large_errors():
    value, manager = evaluate([0., .0001, .0005, .001, .0015, .002, .003, .02, .2])
    expected = torch.tensor([0., -.01, -.25, -1.,
                             *[-(1. + 2. * math.log(r)) for r in (1.5, 2., 3., 20., 200.)]])
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(manager.raw_terms[TERM], expected)
    assert value.dtype == torch.float32
    assert (value[1:] < value[:-1]).all()
    assert value[-1].item() == pytest.approx(-11.596634733)


def test_quadratic_and_log_tail_have_continuous_value_and_slope_at_1mm():
    errors = torch.tensor([.001 * (1. - 1e-6), .001, .001 * (1. + 1e-6)],
                          dtype=torch.float64, requires_grad=True)
    values, _ = evaluate(errors)
    gradient, = torch.autograd.grad(values.sum(), errors)
    torch.testing.assert_close(values, -torch.ones(3), atol=3e-6, rtol=0)
    torch.testing.assert_close(gradient, torch.full_like(errors, -2000.), atol=.003, rtol=0)


@pytest.mark.parametrize('dt', [.025, .05, .1])
def test_control_step_penalty_not_scaled_by_dt(dt):
    config = only(TERM)
    config.update(scale_by_dt=True, reference_dt=.05)
    config['terms'][TERM]['weight'] = .5
    value, _ = evaluate([.002, .004], config=config, dt=dt)
    torch.testing.assert_close(value, torch.tensor([-.5 * (1. + 2. * math.log(r)) for r in (2., 4.)]))


def test_nonfinite_input_is_rejected_by_the_task_finite_state_gate():
    value, _ = evaluate([float('inf'), float('nan')])
    assert torch.isnan(value).all()  # Task finite-state gate rejects transition.


def test_peak_uses_world_euclidean_error_and_retains_earlier_transients():
    accumulator = ArmTrackingErrorAccumulator(num_envs=2, device='cpu')
    requested = positions([[.0008, 0], [0, .003]])
    requested[0, 0, 1] = .0008  # Each axis <1mm, Euclidean distance >1mm.
    actual = torch.zeros_like(requested)
    accumulator.sample(requested, actual, actual)
    accumulator.sample(actual, actual, actual)  # Returning to zero cannot hide peak.
    result = accumulator.snapshot()
    torch.testing.assert_close(result['arm_tracking_requested_error_m'],
        torch.tensor([2 ** .5 * .0008, .003], dtype=torch.float64))
    torch.testing.assert_close(result['arm_tracking_ik_target_error_m'], torch.zeros(2, dtype=torch.float64))
    assert result['arm_tracking_over_1mm_substeps'].tolist() == [1, 1]
    assert result['arm_tracking_over_1mm_side_substeps'].tolist() == [[1, 0], [0, 1]]
    assert result['arm_tracking_sampled_substeps'].tolist() == [2, 2]
    assert result['arm_tracking_within_1mm'].tolist() == [False, False]


def test_events_cross_control_boundaries_and_partial_reset():
    accumulator = ArmTrackingErrorAccumulator(num_envs=2, device='cpu')
    above = positions([[.002, .003], [.002, 0]])
    zero = torch.zeros_like(above)
    accumulator.sample(above, zero, zero)
    accumulator.sample(above, zero, zero)
    assert accumulator.over_1mm_events.tolist() == [1, 1]
    assert accumulator.over_1mm_substeps.tolist() == [2, 2]  # Both hands count once.
    accumulator.begin_control()
    accumulator.sample(above, zero, zero)
    assert accumulator.over_1mm_events.tolist() == [0, 0]
    accumulator.reset(torch.tensor([0]))
    accumulator.sample(above, zero, zero)
    assert accumulator.over_1mm_events.tolist() == [1, 0]
    assert accumulator.sampled_substeps.tolist() == [1, 2]
    accumulator.sample(zero, zero, zero)
    accumulator.sample(above, zero, zero)
    assert accumulator.over_1mm_events.tolist() == [2, 1]


def test_metrics_count_world_substeps_and_ignore_unused_evaluation_worlds():
    accumulator = ArmTrackingErrorAccumulator(num_envs=2, device='cpu')
    metric = ArmTrackingMetrics()
    zero = positions([[0, 0], [0, 0]])
    accumulator.sample(positions([[.002, .003], [.1, .1]]), zero, zero)
    accumulator.sample(zero, zero, zero)
    metric.sample(accumulator.snapshot(), active_worlds=1)
    accumulator.begin_control()
    accumulator.sample(zero, zero, zero)
    metric.sample(accumulator.snapshot(), active_worlds=1)
    result = metric.result()
    assert result['requested_peak_m'] == .003
    assert result['ik_target_peak_m'] == 0
    assert result['sampled_substeps'] == 3
    assert result['over_1mm_substeps'] == 1
    assert result['within_1mm_substep_fraction'] == pytest.approx(2 / 3)
    assert result['over_1mm_entry_events'] == 1
    assert result['sampled_control_steps'] == 2
    assert result['over_1mm_control_steps'] == 1
    assert result['over_1mm_side_substeps'] == {'right': 1, 'left': 1}
    assert ArmTrackingMetrics().result() is None


def test_task_callback_uses_raw_request_actual_body_and_next_substep_time():
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    received = {}
    def requests(raw, times):
        received.update(raw=raw.clone(), times=times.clone())
        return {side: (raw[:, i:i+1].expand(1, 3), None)
                for i, side in enumerate(('right', 'left'))}
    actual = torch.tensor([[[.001, .001, .001], [.002, .002, .002]]], dtype=torch.float64)
    raw = torch.tensor([[.004, .006]], dtype=torch.float64)
    follower = SimpleNamespace(dt=.005, sides={'right': {}, 'left': {}}, requests=requests,
        wrists={'right': {'body_id': 0}, 'left': {'body_id': 1}},
        last={side: {'target_position': actual[:, i]} for i, side in enumerate(('right', 'left'))})
    scene = SimpleNamespace(arm_follower=follower, control_steps=torch.tensor([2]),
        decimation=10, last_requested_physical_action=raw, data=SimpleNamespace(xpos=actual))
    task = PianoTaskEnv.__new__(PianoTaskEnv)
    task.arm_tracking = ArmTrackingErrorAccumulator(num_envs=1, device='cpu')
    task._sample_arm_tracking(scene)
    torch.testing.assert_close(received['raw'], raw)
    torch.testing.assert_close(received['times'], torch.tensor([.105]))
    assert task.arm_tracking.snapshot()['arm_tracking_requested_error_m'].item() == pytest.approx(.004 * 3 ** .5)
    assert task.arm_tracking.snapshot()['arm_tracking_ik_target_error_m'].item() == 0


@pytest.mark.parametrize(('field', 'value'), [
    ('error_scale_m', -.001), ('error_scale_m', True), ('error_scale_m', 0),
    ('error_scale_m', float('inf')), ('weight', 1.1)])
def test_invalid_parameters_rejected(field, value):
    config = only(TERM)
    target = config['terms'][TERM] if field == 'weight' else config['terms'][TERM]['params']
    target[field] = value
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize(('field', 'value'), [('threshold_m', .001),
    ('maximum_penalty_per_step', 5.), ('mode', 'capped_quadratic')])
def test_removed_parameters_are_rejected_instead_of_migrated(field, value):
    config = only(TERM)
    config['terms'][TERM]['params'][field] = value
    with pytest.raises(ValueError, match='params must contain'):
        validate(config)


def test_cli_arm_only_default_explicit_disable_and_old_term_set_rejected():
    baseline = effective_config(parser().parse_args(['--no-arms']))
    assert not baseline['reward_config']['terms'][TERM]['enabled']
    flags = ['--arms', '--disable-wrist-yaw', '--action-profile', 'wuji_cartesian_v1',
             '--arm-initial-right-deg', *map(str, [40, -30, 20, -50, 40, 10, -20]),
             '--arm-initial-left-deg', *map(str, [40, -30, 20, -50, 40, 10, -20])]
    enabled = effective_config(parser().parse_args(flags))
    assert enabled['reward_config']['terms'][TERM]['enabled']
    disabled = effective_config(parser().parse_args(flags + ['--disable-reward', TERM]))
    assert not disabled['reward_config']['terms'][TERM]['enabled']
    with pytest.raises(ValueError, match='requires --arms'):
        effective_config(parser().parse_args(['--no-arms', '--enable-reward', TERM]))
    old = only()
    old['terms'].pop(TERM)
    with pytest.raises(ValueError, match='Reward terms must match'):
        validate(old)
