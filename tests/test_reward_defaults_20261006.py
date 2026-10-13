"""User-selected endpoint rewards, signed wrong-key cost and idle target L1."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from tests.test_paper_rewards import scene_fixture, only
from tests.test_shadow_reward_only import inputs as shadow_inputs
from tsensei_v3_2_pianist.mjlab_env.reward_config import validate, contact_options
from tsensei_v3_2_pianist.mjlab_env.rewards import RewardManager, gesture_protection_mask
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'configs/rewards/default_20261006.json'
GESTURE = 'humanoid_finger_gesture_reward'


def evaluate_posture(action, fingers=None, *, scene=None, cfg=None, snapshot=None):
    cfg = only(GESTURE) if cfg is None else cfg
    scene = scene_fixture() if scene is None else scene
    n = len(action)
    manager = RewardManager(cfg, scene, {}, num_envs=n, device='cpu')
    fingers = torch.zeros(n, 10, 88) if fingers is None else fingers
    snapshot = {} if snapshot is None else dict(snapshot)
    snapshot.setdefault('gesture_protected_fingers', fingers.bool().any(-1))
    result = manager({}, action, snapshot,
                     torch.zeros(n, 89), fingers)
    return result[GESTURE], manager


def test_current_default_contract_and_no_deleted_terms():
    c = validate(json.loads(DEFAULT.read_text()))
    assert len(c['terms']) == 22
    assert c['terms']['arm_motion_limit_penalty']['enabled'] is False
    assert c['terms']['self_collision_reward'] == dict(
        enabled=True, weight=.1, params=dict(aggregation='substep_sum', maximum_penalty_per_step=5.))
    assert c['terms']['false_press_penalty'] == dict(enabled=False, weight=-.05, params={})
    assert c['terms'][GESTURE] == dict(enabled=True, weight=1., params=dict(
        coefficient=-.05, reference='scene_reset', lookahead_steps=20))
    assert c['terms']['key_press_reward']['params'] == dict(mode='paper', chord_bonus=False, chord_scope=None)
    assert c['terms']['joint_motion_limit_penalty']['params']['full_penalty_excess'] == .1
    assert contact_options(c) == dict(collision_categories=('self',), ignored_body_pairs=[],
                                    self_collision_substep_sum=True)
    for name in ('collision_reward', 'natural_posture_reward'):
        assert name not in c['terms']
        old = deepcopy(c)
        old['terms'][name] = dict(enabled=False, weight=1., params={})
        with pytest.raises(ValueError, match='Reward terms must match'):
            validate(old)
        with pytest.raises(SystemExit):
            parser().parse_args(['--enable-reward', name])


def test_zero_at_initial_target_and_physical_radian_sum_includes_followers():
    scene = scene_fixture()
    scene['neutral_action'] = [.3] * 50 + [0.]
    action = torch.tensor(scene['neutral_action']).repeat(3, 1)
    action[1, 4:8] += torch.tensor([.1, -.2, .3, -.4])  # MCP/SW/PIP/DIP of one idle finger.
    action[2, 0] += 4.  # Demonstrates no old -.1 saturation or ten-finger average.
    action[:, 20:25] = 2.  # Wrist and translation are excluded.
    action[:, 45:] = 2.  # Other wrist and pedal are excluded.
    result, _ = evaluate_posture(action, scene=scene)
    torch.testing.assert_close(result, torch.tensor([0., -.1, -.4]), atol=1e-7, rtol=1e-6)


def test_current_used_fingers_are_exempt_and_measured_pose_is_irrelevant():
    action = torch.ones(3, 51)
    fingers = torch.zeros(3, 10, 88)
    fingers[1, :2, 10] = 1
    fingers[2, :, 10] = 1
    cfg = only(GESTURE)
    cfg['performance'] = dict(anticipatory_fingering_time=.15, anticipatory_fingering_reward=False)
    sample = dict(right_qpos=torch.full((3, 25), 100.), left_qpos=torch.full((3, 25), -100.),
                  anticipatory_mask=torch.ones(3, 10, dtype=torch.bool))
    result, _ = evaluate_posture(action, fingers, cfg=cfg, snapshot=sample)
    torch.testing.assert_close(result, torch.tensor([-4., -3.2, 0.]))


@pytest.mark.parametrize('dt', [.01, .025, .05])
def test_posture_final_control_step_units_and_weight_apply_once(dt):
    cfg = only(GESTURE)
    cfg.update(action_units='normalized', scale_by_dt=True, reference_dt=.05)
    cfg['terms'][GESTURE]['weight'] = .5
    scene = scene_fixture(); scene['control_dt'] = dt
    result, manager = evaluate_posture(torch.ones(1, 51), cfg=cfg, scene=scene)
    torch.testing.assert_close(manager.raw_terms[GESTURE], torch.tensor([-4.]))
    torch.testing.assert_close(result, torch.tensor([-2.]))


def test_idle_reference_uses_updated_real_reset_target_and_survives_partial_reset():
    cfg = only(GESTURE)
    manager = RewardManager(cfg, scene_fixture(), {}, num_envs=2, device='cpu')
    manager.neutral = torch.full((50,), .2)  # Same override as the physical scene reset target.
    action = torch.full((2, 51), .2)
    fingers = torch.zeros(2, 10, 88)
    target = torch.zeros(2, 89)
    sample = dict(gesture_protected_fingers=torch.zeros(2, 10, dtype=torch.bool))
    torch.testing.assert_close(manager({}, action, sample, target, fingers)[GESTURE], torch.zeros(2))
    action[1, 7] += .5
    manager.reset(torch.tensor([0]))
    torch.testing.assert_close(manager({}, action, sample, target, fingers)[GESTURE], torch.tensor([0., -.05]))


def test_signed_false_press_weight_is_negative_cost_and_self_pairs_are_additive():
    cfg = only('false_press_penalty', 'self_collision_reward')
    cfg['terms']['false_press_penalty']['weight'] = -.05
    manager = RewardManager(cfg, scene_fixture(), {}, num_envs=3, device='cpu')
    state = torch.zeros(3, 88); state[:, 2] = torch.tensor([0., .5, 1.])
    sample = dict(key_state=state, self_collision_pair_substeps=torch.tensor([0, 1, 3]))
    result = manager({}, torch.zeros(3, 51), sample, torch.zeros(3, 89), torch.zeros(3, 10, 88))
    torch.testing.assert_close(manager.raw_terms['false_press_penalty'], torch.tensor([0., .08, .15]))
    torch.testing.assert_close(result['false_press_penalty'], torch.tensor([0., -.004, -.0075]))
    torch.testing.assert_close(result['self_collision_reward'], torch.tensor([0., -.2, -.6]))


def test_key_scoring_includes_earlier_transients():
    bundle, logic, sample, base = shadow_inputs()
    cfg = json.loads(DEFAULT.read_text())
    sample['requested_physical_action'] = torch.zeros(1, 51)
    sample['self_collision_pairs'] = torch.zeros(1)
    sample['self_collision_pair_substeps'] = torch.zeros(1)
    sample['palm_collision'] = torch.tensor([False])
    # The generic fixture uses legacy +.5 when clear; current task uses zero.
    from tsensei_v3_2_pianist.forearm_collision import forearm_contact_reward
    base['forearm_reward'] = forearm_contact_reward(sample['palm_collision'].float(), 'tsensei_collision_v1')
    # Earlier substeps have a wrong key/other pedal goal/cross-hand contact.
    manager = RewardManager(cfg, scene_fixture(), bundle.manifest, num_envs=1, device='cpu')
    sample['gesture_protected_fingers'] = gesture_protection_mask(
        logic.fingers, manager.gesture_lookahead_steps)[logic.index]
    result = manager(base, torch.zeros(1, 51), sample,
                     logic.goal[logic.index], logic.fingers[logic.index])
    assert 'false_press_penalty' not in result
    assert result['key_press_reward'].item() == pytest.approx(.1+.9*(.3*10**(-3.61)+.4*__import__('math').exp(-1.25)))
    assert result['sustain_reward'].item() == pytest.approx(1.)
    assert result['forearm_reward'].item() == 0.
    assert result['energy_reward'].item() == base['energy_reward'].item()
    p = manager.performance
    assert not {'reward_sampling', 'substep_rewards', 'target_key_substep_reward',
                'target_key_activation_weight'}.intersection(p)
    assert p['exact_note_timing'] and not cfg['scale_by_dt']


def test_cli_final_configuration_and_coupling_switch():
    for flag, enabled in [('--finger-coupling', True), ('--no-finger-coupling', False)]:
        args = parser().parse_args(['--inspect-only', '--reward-config', str(DEFAULT), flag,
            '--wrist-pitch-max-velocity', '3.1416', '--wrist-pitch-max-acceleration', '15.708'])
        config = effective_config(args)
        assert bool(config['finger_coupling']) == enabled
        assert config['reward_config']['terms']['self_collision_reward']['enabled']
        assert config['reward_config']['terms']['false_press_penalty']['weight'] == -.05


@pytest.mark.parametrize('extra', [(), ('--no-finger-coupling',)])
def test_unified_entry_uses_physical_key_rewards(tmp_path, extra):
    entry = ROOT / 'all_use_this_train.sh'
    if not entry.exists():
        pytest.skip('External unified launcher is tested separately in the workspace')
    for name in ('sbatch', 'python3'):
        path = tmp_path / name
        path.write_text('#!/bin/sh\nprintf "%s\\0" "$@"\n')
        path.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH']}
    raw = subprocess.check_output(['bash', str(entry), *extra], cwd=ROOT, env=env)
    submitted = raw.decode().rstrip('\0').split('\0')
    assert '--gres=gpu:2' in submitted
    args = submitted[submitted.index('train')+1:]
    # job.sbatch now invokes train_from_source directly; the archived coupled
    # wrapper is no longer an active entry. Unified arguments are complete.
    parsed, _ = parser().parse_known_args(args)
    config = effective_config(parsed)
    expected = validate(json.loads(DEFAULT.read_text()))
    assert config['reward_config'] == expected
    assert bool(config['finger_coupling']) == (not extra)
    assert config['requested_transitions'] == 10000384
    assert config['execution_layout']['collector_pipeline']
    assert config['control_limits']['joint_state_limits'] == 'hard_v1'
    for side in ('R', 'L'):
        row = config['control_limits']['channels'][side+'_Wrist_Pitch']
        assert row['max_velocity'] == 3.1416 and row['max_acceleration'] == 15.708
