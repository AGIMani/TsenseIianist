"""CPU checks for motion-penalty CLI precedence and checkpoint contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
from tests.test_torch_sac import agent


ROOT = Path(__file__).resolve().parents[1]
TERM = 'joint_motion_limit_penalty'
BASELINE = ROOT / 'configs/rewards/paper_chord_off_20261005.json'


def resolve(*arguments):
    return effective_config(parser().parse_args(['--inspect-only', *arguments]))['reward_config']


def test_default_and_old_json_keep_motion_disabled_without_changing_other_rewards():
    assert resolve()['terms'][TERM] == {
        'enabled': False, 'weight': 1., 'params': {
            'finger_velocity_limit': 10., 'finger_acceleration_limit': 200.,
            'wrist_velocity_limit': None, 'wrist_acceleration_limit': None,
            'full_penalty_excess': .2, 'maximum_penalty_per_step': 5.,
        }}
    original = BASELINE.read_bytes()
    baseline = json.loads(original)
    effective = resolve('--reward-config', str(BASELINE), '--joint-motion-limit-penalty')
    assert effective['terms'][TERM]['enabled']
    del effective['terms'][TERM]
    assert effective == baseline
    assert BASELINE.read_bytes() == original


@pytest.mark.parametrize(('arguments', 'enabled'), [
    (('--joint-motion-limit-penalty', '--no-joint-motion-limit-penalty'), False),
    (('--no-joint-motion-limit-penalty', '--joint-motion-limit-penalty'), True),
    (('--joint-motion-limit-penalty', '--disable-reward', TERM), False),
    (('--disable-reward', TERM, '--joint-motion-limit-penalty'), False),
    (('--no-joint-motion-limit-penalty', '--enable-reward', TERM), True),
])
def test_dedicated_boolean_last_value_and_generic_term_precedence(arguments, enabled):
    assert resolve(*arguments)['terms'][TERM]['enabled'] is enabled


def test_generic_enable_disable_conflict_is_still_rejected():
    with pytest.raises(ValueError, match='both enabled and disabled'):
        resolve('--enable-reward', TERM, '--disable-reward', TERM)


def test_scalar_overrides_preserve_json_and_do_not_implicitly_enable(tmp_path):
    baseline = resolve('--reward-config', str(BASELINE))
    baseline['terms'][TERM]['params'].update(
        finger_velocity_limit=9., finger_acceleration_limit=180.,
        wrist_velocity_limit=2., wrist_acceleration_limit=12.,
        full_penalty_excess=.3, maximum_penalty_per_step=4.)
    path = tmp_path / 'rewards.json'
    path.write_text(json.dumps(baseline))
    assert resolve('--reward-config', str(path)) == baseline
    expected = deepcopy(baseline)
    expected['terms'][TERM]['params'].update(finger_velocity_limit=8., full_penalty_excess=.25)
    actual = resolve('--reward-config', str(path), '--finger-target-velocity-limit', '7',
                     '--finger-target-velocity-limit', '8', '--joint-motion-full-penalty-excess', '.25')
    assert actual == expected
    assert not actual['terms'][TERM]['enabled']
    baseline['terms'][TERM]['enabled'] = True
    path.write_text(json.dumps(baseline))
    assert resolve('--reward-config', str(path)) == baseline


def test_explicit_finger_and_user_approved_wrist_thresholds_are_serialized():
    actual = resolve('--reward-config', str(BASELINE), '--joint-motion-limit-penalty',
        '--finger-target-velocity-limit', '10', '--finger-target-acceleration-limit', '200',
        '--wrist-target-velocity-limit', '3', '--wrist-target-acceleration-limit', '15',
        '--joint-motion-full-penalty-excess', '.2', '--joint-motion-max-penalty', '5')
    assert actual['terms'][TERM] == {'enabled': True, 'weight': 1., 'params': {
        'finger_velocity_limit': 10., 'finger_acceleration_limit': 200.,
        'wrist_velocity_limit': 3., 'wrist_acceleration_limit': 15.,
        'full_penalty_excess': .2, 'maximum_penalty_per_step': 5.,
    }}
    # These thresholds use physical targets without changing other action-unit semantics.
    assert actual['action_units'] is None


@pytest.mark.parametrize('flag', [
    '--finger-target-velocity-limit', '--finger-target-acceleration-limit',
    '--wrist-target-velocity-limit', '--wrist-target-acceleration-limit',
    '--joint-motion-full-penalty-excess',
])
@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_nonpositive_or_nonfinite_thresholds_fail_even_when_disabled(flag, value):
    with pytest.raises(ValueError):
        resolve('--wrist-target-velocity-limit', '3',
                '--wrist-target-acceleration-limit', '15', flag, value)


@pytest.mark.parametrize('flag', ['--wrist-target-velocity-limit', '--wrist-target-acceleration-limit'])
def test_wrist_thresholds_require_a_complete_explicit_pair(flag):
    with pytest.raises(ValueError):
        resolve('--joint-motion-limit-penalty', flag, '3')


@pytest.mark.parametrize(('extra', 'enabled', 'maximum'), [
    ((), True, 5.),
    (('--no-joint-motion-limit-penalty',), False, 5.),
    (('--joint-motion-max-penalty', '4'), True, 4.),
])
def test_unified_wrapper_effective_rewards_and_hard_wrist_controls(tmp_path, extra, enabled, maximum):
    entry = ROOT / 'all_use_this_train.sh'
    if not entry.is_file():
        pytest.skip('Pinned source bundles intentionally omit the external all_use_this_train.sh launcher; '
                    'the real wrapper chain is tested in the workspace')
    # The real scripts run, but both external scheduling and Python execution
    # end at argument-capture stubs. This never submits a job or creates physics.
    for name in ('sbatch', 'python3'):
        stub = tmp_path / name
        stub.write_text('#!/bin/sh\nprintf "%s\\0" "$@"\n')
        stub.chmod(0o755)
    environment = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}
    raw = subprocess.check_output(['bash', str(entry), *extra], cwd=ROOT, env=environment)
    submitted = raw.decode().rstrip('\0').split('\0')
    assert '--gres=gpu:2' in submitted
    arguments = submitted[submitted.index('train') + 1:]
    raw = subprocess.check_output(
        ['bash', str(ROOT / 'scripts/hpc/train_coupled_20hz.sh'), *arguments],
        cwd=ROOT, env=environment)
    source_arguments = raw.decode().rstrip('\0').split('\0')
    assert source_arguments[0] == 'scripts/train_from_source.py'
    args, _ = parser().parse_known_args(source_arguments[1:])
    actual = effective_config(args)
    reward = actual['reward_config']
    assert reward['terms'][TERM] == {'enabled': enabled, 'weight': 1., 'params': {
        'finger_velocity_limit': 10., 'finger_acceleration_limit': 200.,
        'wrist_velocity_limit': 3., 'wrist_acceleration_limit': 15.,
        'full_penalty_excess': .2, 'maximum_penalty_per_step': maximum,
    }}
    original = json.loads(BASELINE.read_text())
    assert {name: term for name, term in reward['terms'].items() if name != TERM} == original['terms']
    assert reward['terms']['key_press_reward']['params'] == {
        'mode': 'paper', 'chord_bonus': False, 'chord_scope': None}
    assert not reward['terms']['false_press_penalty']['enabled']
    assert reward['scale_by_dt'] is False
    controls = actual['control_limits']
    assert controls['joint_state_limits'] == 'hard_v1'
    for side in ('R', 'L'):
        assert controls['channels'][f'{side}_Wrist_Pitch']['max_velocity'] == 3.1416
        assert controls['channels'][f'{side}_Wrist_Pitch']['max_acceleration'] == 15.708
    layout = actual['execution_layout']
    assert (layout['collector_device'], layout['learner_device']) == ('cuda:0', 'cuda:1')
    assert layout['collector_pipeline'] is True
    # The extra cap flag changes only the penalty budget, not the training budget.
    assert actual['requested_transitions'] == 10000384


def test_historical_checkpoint_missing_term_matches_disabled_but_rejects_new_penalty(tmp_path):
    legacy = json.loads(BASELINE.read_text())
    assert TERM not in legacy['terms']
    metadata = dict(scene_sha256='scene', task_sha256='task', stage=1, reward_config=legacy)
    path = tmp_path / 'legacy.pt'
    checkpoint.save(path, agent(), metadata, torch.Generator().manual_seed(7))
    disabled = resolve('--reward-config', str(BASELINE))
    expected = {**metadata, 'reward_config': disabled}
    checkpoint.load(path, agent(), expected)
    assert checkpoint.reward_options(path)['reward_config'] == disabled
    assert TERM not in metadata['reward_config']['terms']

    enabled = resolve('--reward-config', str(BASELINE), '--joint-motion-limit-penalty',
        '--wrist-target-velocity-limit', '3', '--wrist-target-acceleration-limit', '15')
    changed = {**metadata, 'reward_config': enabled}
    with pytest.raises(ValueError, match='reward contract mismatch'):
        checkpoint.load(path, agent(), changed)
    with pytest.raises(ValueError, match='networks-only'):
        checkpoint.load(path, agent(), changed, allow_reward_change=True)
    fresh = agent()
    checkpoint.load(path, fresh, changed, weights_only=True,
                    allow_reward_change=True, transfer_mode='finetune')
    assert fresh.updates == 0
    assert not fresh.actor_optimizer.state

    new_path = tmp_path / 'new.pt'
    checkpoint.save(new_path, agent(), changed, torch.Generator().manual_seed(8))
    assert checkpoint.reward_options(new_path)['reward_config'] == enabled
    checkpoint.load_agent(new_path, scene_sha256='scene', task_sha256='task',
                          obs_size=7, action_size=3, device='cpu', reward_config=enabled)
    for field, value in (('finger_velocity_limit', 11.), ('finger_acceleration_limit', 210.),
                         ('wrist_velocity_limit', 2.9), ('wrist_acceleration_limit', 14.9),
                         ('full_penalty_excess', .3), ('maximum_penalty_per_step', 4.)):
        altered = deepcopy(changed)
        altered['reward_config']['terms'][TERM]['params'][field] = value
        with pytest.raises(ValueError, match='reward contract mismatch'):
            checkpoint.load(new_path, agent(), altered)
