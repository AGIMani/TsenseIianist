"""CPU launch-contract tests; do not build a scene or start a training process."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('clearance_launcher', ROOT/'scripts/launch_clearance_training.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def fixture():
    args = SimpleNamespace(parent=Path('/trusted/parent.pt'), actor_lr=3e-5,
        learning_start_transitions=None, steps=512, eval_interval=128,
        eval_episodes=3, save_interval=0, output_dir=Path('/tmp/probe'))
    parent = dict(observation_size=1181, action_size=47, learned_alpha=.00169, actor_lr=3e-5,
        num_envs=512, action_profile='direct_joint_v1', learning_start_transition_1_based=91137,
        source_runtime=dict(pose=dict(height=.1,ty=.04,yaw=.15), model=dict(drive_profile='shadow_v1'),
            environment=dict(midi='/trusted/other-score.proto',control_hz=20,physics_hz=200,
                action_reward_observation=True,gravity_compensation=True,disable_forearm_reward=False),
            preparation=dict(duration=1,lift=.02,mode='policy_v1')))
    return args, parent


def test_parent_score_and_learned_temperature_preserved():
    args, parent = fixture(); command = launcher.build_command(args,parent,[])
    assert command[command.index('--midi')+1] == '/trusted/other-score.proto'
    assert float(command[command.index('--init-temperature')+1]) == .00169
    assert command[command.index('--learning-start-transitions')+1] == '91137'
    assert command[command.index('--finetune-from')+1] == '/trusted/parent.pt'
    assert '--allow-reward-change' in command and '--allow-control-change' in command
    assert '--target-music-f1' not in command  # The invoked run.sh owns thresholds.


def test_forwarded_options_keep_order_without_hardcoded_song():
    args, parent = fixture(); rest=['--midi','explicit.proto','--batch-size','1024']
    assert launcher.build_command(args,parent,rest)[-len(rest):] == rest


def test_thumb_source_options_are_forwarded_as_cli_and_legacy_profile_is_explicit():
    args, parent = fixture()
    command = launcher.build_command(args, parent, [])
    assert command[command.index('--thumb-target-mode')+1] == 'profile'
    assert '--thumb-white-target-distance' not in command
    parent['source_runtime']['environment']['thumb_target'] = {
        'mode': 'profile', 'white_distance': .02, 'black_distance': .015}
    command = launcher.build_command(args, parent, [])
    assert command[command.index('--thumb-white-target-distance')+1] == '0.02'
    assert command[command.index('--thumb-black-target-distance')+1] == '0.015'
    assert '--thumb-target' not in command
    parent['source_runtime']['environment']['thumb_target'] = {
        'mode': 'uniform', 'white_distance': None, 'black_distance': None}
    command = launcher.build_command(args, parent, [])
    assert command[command.index('--thumb-target-mode')+1] == 'uniform'
    assert '--thumb-white-target-distance' not in command


def test_unknown_source_recipe_fields_fail_instead_of_becoming_untyped_cli():
    args, parent = fixture()
    parent['source_runtime']['environment']['future_unhandled_mode'] = {'enabled': True}
    with pytest.raises(ValueError, match='Unsupported source recipe fields'):
        launcher.build_command(args, parent, [])


@pytest.mark.parametrize('flag', ['--warm-start','--stage2-from','--finetune-from','--init-temperature=1'])
def test_cannot_silently_replace_transfer_or_temperature(flag):
    args,parent = fixture()
    with pytest.raises(ValueError,match='owned by this wrapper'):
        launcher.build_command(args,parent,[flag])


def test_missing_refill_requires_explicit_value_and_other_shapes_rejected():
    args,parent = fixture(); parent.pop('learning_start_transition_1_based')
    with pytest.raises(ValueError,match='refill'): launcher.build_command(args,parent,[])
    args.learning_start_transitions=120001
    assert launcher.build_command(args,parent,[])
    parent['action_size']=49
    with pytest.raises(ValueError,match='47-action'): launcher.build_command(args,parent,[])


def test_embedded_score_rejected_instead_of_using_missing_basename():
    args,parent = fixture(); parent['source_runtime']['external_score']={'filename':'piece.proto','bytes':'fixture'}
    with pytest.raises(ValueError,match='Embedded external score'):
        launcher.build_command(args,parent,[])


def test_invalid_stop_toggle_rejected_and_help_available_without_checkpoint():
    args,parent = fixture()
    with pytest.raises(ValueError,match='no target-stop toggle'):
        launcher.build_command(args,parent,['--no-target-stop'])
    result = subprocess.run([sys.executable,str(ROOT/'scripts/launch_clearance_training.py'),'--help'],
        text=True,capture_output=True,check=True)
    assert '--parent' in result.stdout and '--actor-lr' in result.stdout
    assert 'no disable switch' in result.stdout
    assert '--no-target-stop' not in result.stdout
