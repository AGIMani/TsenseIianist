"""CPU checks for isolated, pinned, equal-budget F1 refinement orchestration."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/hpc'))
import run_twinkle_f1_finetune as runner


@pytest.fixture
def contracts(tmp_path):
    parent = tmp_path/'best.pt'
    parent.write_bytes(b'parent')
    baseline = dict(sac=dict(init_temperature=.5, gamma=.75, actor_lr=.0003),
        requested_vector_steps=19532, requested_transitions=10000384,
        checkpoint_retention=dict(forward_vector_step=1000, periodic_interval=0),
        evaluation_selection=dict(best_metric='music_only_f1_min', consecutive_evaluations=3),
        source_sha256={}, source_runtime=dict(source_sha256={}, environment=dict(frame_stack=1)),
        reward_config=dict(terms=dict(false_press_penalty=dict(enabled=True, weight=1.),
            key_press_reward=dict(enabled=True, weight=1.)),
            performance=dict(target_key_activation_weight=.5, anticipatory_fingering_time=.15)),
        learning_start_transition_1_based=186881, warmup_transitions=9999,
        observation_size=2297, action_size=31, replay_bytes=37020000000, target_entropy=-15.5,
        replay_capacity=2000000, num_envs=512, scene_sha256='unchanged', cli={})
    template = deepcopy(baseline)
    template['sac']['init_temperature'] = .0006
    template.update(requested_vector_steps=400, requested_transitions=400*512,
        origin=dict(checkpoint=str(parent), checkpoint_sha256=runner.digest(parent),
            transfer_mode='finetune_networks_only', fresh_optimizer_alpha_rng=True, fresh_replay=True))
    template['checkpoint_retention']['forward_vector_step'] = 200
    template['evaluation_selection'].update(best_metric=runner.METRIC,
        target_exact_micro_f1=.99, consecutive_evaluations=4098)
    template['cli'].update(seed=42, num_envs=512, eval_interval=200, eval_episodes=3,
        eval_num_envs=3, save_interval=0, save_forward=True, allow_reward_change=True,
        allow_control_change=False, warm_start=None, stage2_from=None,
        finetune_from=str(parent), init_temperature=.0006, target_consecutive_evals=4098,
        best_f1_metric=runner.METRIC, target_exact_micro_f1=.99,
        vector_steps=400, output_dir=str(tmp_path/'smoke'), reward_config=str(tmp_path/'reward.json'),
        finger_coupling_angle_ranges=str(runner.ROOT/'logs/hpc/1364/steps/34/source/configs/finger_coupling/angle_ranges_60_80.json'))
    baseline['cli'] = deepcopy(template['cli'])
    baseline['cli'].update(vector_steps=19532, finetune_from=None, init_temperature=.5,
        eval_interval=1000, allow_reward_change=False, target_consecutive_evals=3)
    baseline['cli'].pop('best_f1_metric')
    baseline['cli'].pop('target_exact_micro_f1')
    return baseline, template, parent


def stages():
    return [dict(label=l, false_press_weight=f, activation_weight=a, gamma=g, status='pending')
            for l, f, a, g in runner.STAGES]


def test_smoke_is_full_baseline_contract_with_only_shared_changes(contracts):
    baseline, template, parent = contracts
    runner.validate_smoke(template, baseline, parent, runner.digest(parent))
    for path, replacement in [(('sac', 'actor_lr'), .0001), (('source_runtime', 'environment'), {}),
            (('reward_config', 'terms'), {}), (('origin', 'fresh_optimizer_alpha_rng'), False)]:
        changed = deepcopy(template)
        changed[path[0]][path[1]] = replacement
        with pytest.raises(ValueError):
            runner.validate_smoke(changed, baseline, parent, runner.digest(parent))


@pytest.mark.parametrize('stage', stages(), ids=lambda s:s['label'])
def test_stage_runtime_all_fields_strict_and_initial_write_deferred(contracts, tmp_path, stage):
    _, template, parent = contracts
    run, reward = tmp_path/'run', tmp_path/'reward.json'
    actual = runner.expected_config(template, stage, run, reward, parent)
    runner.validate_config(actual, template, stage, run, reward, parent)
    assert runner.validate_ready_config(actual, template, stage, run, reward, parent)
    partial = deepcopy(actual)
    partial.pop('replay_bytes')
    assert not runner.validate_ready_config(partial, template, stage, run, reward, parent)
    for mutate in [lambda c:c['cli'].update(seed=43),
            lambda c:c['sac'].update(actor_lr=.0001),
            lambda c:c.update(scene_sha256='changed'),
            lambda c:c['reward_config']['terms']['key_press_reward'].update(enabled=False),
            lambda c:c.update(replay_capacity=1000000),
            lambda c:c['cli'].update(finetune_from='/different-parent.pt')]:
        changed = deepcopy(actual)
        mutate(changed)
        with pytest.raises(ValueError):
            runner.validate_config(changed, template, stage, run, reward, parent)


def test_commands_preserve_allocation_and_last_cli_activation(contracts, tmp_path):
    _, _, parent = contracts
    args = SimpleNamespace(parent=parent, output=tmp_path, source_archive=tmp_path/'source.tar.gz',
        reward_config=tmp_path/'baseline.json', false025_reward_config=tmp_path/'false025.json')
    for stage in stages():
        argv = runner.command(args, stage, 1364)
        value = lambda option: argv[argv.index(option)+1]
        assert argv[:2] == ['bash', str(runner.ROOT/'scripts/hpc/submit.sh')]
        assert value('--allocation') == '1364'
        assert value('--source-archive') == str(args.source_archive)
        assert value('--finetune-from') == str(parent)
        assert value('--vector-steps') == '4096'
        assert value('--eval-interval') == '200'
        assert value('--init-temperature') == '.0006'
        assert float(value('--target-key-activation-weight')) == stage['activation_weight']
        assert float(value('--discount')) == stage['gamma']
        assert value('--reward-config') == str(args.false025_reward_config if stage['false_press_weight']==.25 else args.reward_config)
        assert '--warmup-transitions' not in argv and '--warm-start' not in argv
        assert int(value('--target-consecutive-evals')) > runner.STEPS//200+1
    assert (runner.STEPS*runner.ENVS-186880)//4 == runner.UPDATES


def test_dispatch_two_pairs_first_free_and_stop_after_failure():
    rows = stages()
    first = runner.dispatchable(rows, [1364, 1457])
    assert [(s['label'], a) for s,a in first] == [('R0_baseline',1364),('R1_false025',1457)]
    rows[0].update(status='running', allocation=1364)
    rows[1].update(status='completed', allocation=1457)
    assert [(s['label'], a) for s,a in runner.dispatchable(rows,[1364,1457])] == [('R2_activation08',1457)]
    rows[1]['status'] = 'failed'
    assert runner.dispatchable(rows,[1364,1457]) == []
    assert rows[0]['status'] == 'running'


def test_source_hashes_are_enumerated_and_verified_in_archive(contracts, tmp_path):
    baseline, template, _ = contracts
    name = 'tsensei_v3_2_pianist/rl_torch/checkpoint.py'
    old, new = hashlib.sha256(b'old').hexdigest(), hashlib.sha256(b'new').hexdigest()
    baseline['source_sha256'][name] = old
    template['source_sha256'][name] = new
    archive = tmp_path/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as packed:
        info=tarfile.TarInfo('./'+name);info.size=3
        packed.addfile(info,io.BytesIO(b'new'))
    runner.source_contract(template,baseline,archive)
    template['source_sha256'][name] = old
    with pytest.raises(ValueError,match='Archive source'):
        runner.source_contract(template,baseline,archive)
    unexpected='tsensei_v3_2_pianist/mjlab_env/rewards.py'
    baseline['source_sha256'][unexpected]=old
    template['source_sha256'][unexpected]=new
    with pytest.raises(ValueError,match='Unexpected source'):
        runner.source_contract(template,baseline,archive)


def test_step_ownership_and_cleanup_never_cancel_reservation(monkeypatch, tmp_path):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    stage=dict(allocation=1364,command=['--output-dir','/expected/output'])
    path=tmp_path/'logs/hpc/1364/steps/36/command.txt'
    assert runner.owned_steps(stage,['1364.36']) == []
    path.parent.mkdir(parents=True)
    path.write_text('bash train --output-dir /another/output\n')
    with pytest.raises(ValueError,match='Unrelated step'):
        runner.owned_steps(stage,['1364.36'])
    path.write_text('bash train --output-dir /expected/output\n')
    assert runner.owned_steps(stage,['1364.36']) == ['1364.36']
    stage['owned_step_ids']=['1364','1364.batch','1364.extern','1457.3','1364.36']
    called=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda args,**kwargs:called.append(args))
    child=Mock();child.pid=4321;child.poll.return_value=0
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:4321)
    monkeypatch.setattr(runner.os,'getpgrp',lambda:9999)
    signaled=[]
    monkeypatch.setattr(runner.os,'killpg',lambda group,sig:signaled.append((group,sig)))
    runner.cancel_stage(stage,child)
    assert called==[['scancel','1364.36']]
    assert signaled==[(4321,runner.signal.SIGTERM)]
    child.send_signal.assert_not_called()


def test_early_cancel_kills_own_flock_srun_group_and_never_shared_group(monkeypatch):
    child=Mock();child.pid=4321;child.poll.return_value=None
    child.wait.side_effect=[runner.subprocess.TimeoutExpired('flock',10),0]
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:4321)
    monkeypatch.setattr(runner.os,'getpgrp',lambda:9999)
    signaled=[]
    monkeypatch.setattr(runner.os,'killpg',lambda group,sig:signaled.append((group,sig)))
    runner.cancel_stage(dict(allocation=1364,owned_step_ids=[]),child)
    assert signaled==[(4321,runner.signal.SIGTERM),(4321,runner.signal.SIGKILL)]
    signaled.clear();child.wait.side_effect=None
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:9999)
    runner.cancel_stage(dict(allocation=1364,owned_step_ids=[]),child)
    assert not signaled
    child.terminate.assert_called_once()


def test_pinned_input_mutation_rejected(tmp_path):
    path=tmp_path/'config.json';path.write_text('{}')
    sha=runner.digest(path);runner.pinned(path,sha)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        runner.pinned(path,sha)
