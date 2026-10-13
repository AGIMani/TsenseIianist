"""CPU invariants for R6/R7 isolated reward/actor-LR refinements."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/hpc'))
import run_twinkle_reward_refine as runner
from test_twinkle_f1_finetune import contracts
from test_twinkle_learner_finetune import learner_files,optimizer


@pytest.mark.parametrize('stage',runner.STAGES,ids=lambda s:s['label'])
def test_full_configs_allow_only_reward_and_declared_actor_lr(contracts,tmp_path,stage):
    _,template,parent=contracts
    template['sac'].update(critic_lr=.0003,temp_lr=.0003)
    template['cli']['actor_lr']=.0003
    run,reward=tmp_path/'run',tmp_path/'reward.json'
    config=runner.expected_config(template,stage,run,reward,parent)
    runner.validate_config(config,template,stage,run,reward,parent)
    assert config['sac']['actor_lr']==config['cli']['actor_lr']==stage['actor_lr']
    assert config['reward_config']['terms']['false_press_penalty']['weight']==stage['false_press_weight']
    assert config['sac']['init_temperature']==.0006 and config['sac']['gamma']==.75
    assert config['origin']['fresh_optimizer_alpha_rng'] is True
    partial=deepcopy(config);partial.pop('replay_bytes')
    assert not runner.validate_ready_config(partial,template,stage,run,reward,parent)
    for change in [lambda c:c['sac'].update(actor_lr=.0001),lambda c:c['sac'].update(critic_lr=.00003),
        lambda c:c['sac'].update(gamma=.9),lambda c:c['sac'].update(init_temperature=.5),
        lambda c:c['cli'].update(actor_lr=.0001),lambda c:c['cli'].update(finetune_from='/R1/best.pt'),
        lambda c:c['cli'].update(warm_start=str(parent)),lambda c:c['cli'].update(allow_reward_change=False),
        lambda c:c['reward_config']['terms']['false_press_penalty'].update(weight=.9),
        lambda c:c['reward_config']['performance'].update(target_key_activation_weight=.8),
        lambda c:c.update(warmup_transitions=0),lambda c:c.update(replay_capacity=1000000),
        lambda c:c['origin'].update(fresh_optimizer_alpha_rng=False)]:
        bad=deepcopy(config);change(bad)
        with pytest.raises(ValueError,match='effective-config'):
            runner.validate_config(bad,template,stage,run,reward,parent)


def test_reward_files_only_change_false_press_weight(contracts,tmp_path):
    baseline,_,_=contracts
    args=SimpleNamespace(r6_reward_config=tmp_path/'r6.json',r7_reward_config=tmp_path/'r7.json')
    for stage in runner.STAGES:
        reward=deepcopy(baseline['reward_config']);reward['terms']['false_press_penalty']['weight']=stage['false_press_weight']
        runner.reward_path(args,stage).write_text(json.dumps(reward))
    runner.validate_reward_jsons(args,baseline)
    changed=json.loads(args.r7_reward_config.read_text());changed['performance']['target_key_activation_weight']=.8
    args.r7_reward_config.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='additional reward change'):runner.validate_reward_jsons(args,baseline)


def test_commands_use_original_parent_and_correct_reward_file(contracts,tmp_path):
    _,_,parent=contracts
    args=SimpleNamespace(parent=parent,output=tmp_path,source_archive=tmp_path/'source.tar.gz',
        r6_reward_config=tmp_path/'r6.json',r7_reward_config=tmp_path/'r7.json')
    for stage in runner.STAGES:
        command=runner.command(args,stage,1457);get=lambda key:command[command.index(key)+1]
        assert get('--source-archive')==str(args.source_archive) and get('--allocation')=='1457'
        assert get('--finetune-from')==str(parent) and '--warm-start' not in command
        assert '--allow-reward-change' in command and get('--init-temperature')=='.0006'
        assert get('--reward-config')==str(runner.reward_path(args,stage))
        assert get('--vector-steps')=='4096' and get('--eval-interval')=='200'
        assert float(get('--target-key-activation-weight'))==.5 and float(get('--discount'))==.75
        if stage['label']=='R6_false010':assert '--actor-lr' not in command
        else:assert float(get('--actor-lr'))==.00003


@pytest.mark.parametrize('stage',runner.STAGES,ids=lambda s:s['label'])
def test_forward_parent_networks_empty_adam_alpha_and_group_lr(learner_files,stage):
    template,parent,previous,tmp_path=learner_files
    run=tmp_path/stage['label'];run.mkdir()
    config=runner.expected_config(template,stage,run,tmp_path/'reward.json',parent)
    current=deepcopy(previous);current.update(updates=0,log_temperature=torch.tensor(math.log(.0006)),config=deepcopy(config['sac']))
    for name,lr in [('actor_optimizer',stage['actor_lr']),('critic_optimizer',.0003),('temp_optimizer',.0003)]:
        current[name]=optimizer(lr,False)
    metadata=deepcopy(config);metadata.update(vector_step=200,updates=0)
    payload=dict(schema=1,kind='learner_warm_restart',agent=current,metadata=metadata)
    torch.save(payload,run/'forward.pt')
    acceptance=runner.verify_forward(run,config,stage,parent)
    assert acceptance['new_updates']==0 and acceptance['parent_networks_exact']
    for mutate in [lambda a:a['actor'].update(weight=torch.tensor([99.])),
        lambda a:a.update(updates=2257280),lambda a:a.update(log_temperature=torch.tensor(math.log(.5))),
        lambda a:a['actor_optimizer']['param_groups'][0].update(lr=.0001),
        lambda a:a['critic_optimizer']['param_groups'][0].update(lr=.00003),
        lambda a:a['temp_optimizer']['param_groups'][0].update(lr=.00003),
        lambda a:a['actor_optimizer']['state'].update({0:{'step':torch.tensor(1.)}})]:
        bad=deepcopy(payload);mutate(bad['agent']);torch.save(bad,run/'forward.pt')
        with pytest.raises(ValueError):runner.verify_forward(run,config,stage,parent)


@pytest.mark.parametrize('stage',runner.STAGES,ids=lambda s:s['label'])
def test_fixed_new_budget_never_includes_historical_updates(stage):
    assert runner.learner.initial_updates(stage)==0
    assert runner.learner.updates_at(stage,200)==0
    assert runner.learner.updates_at(stage,4096)==477568
    summary=dict(status='completed',vector_steps=4096,transitions=2097152,replay_insertions=2097152,
        updates_this_run=477568,updates_total=477568,stop_reason='requested_budget_exhausted',
        target_reached=False,random_action_transitions=9999,
        collector_pipeline_state=dict(pending_vector_step=None,learner_updated_through_vector_step=4096,collected_vector_steps=4096))
    runner.learner.validate_budget(summary,stage)
    for key,value in [('updates_total',2734848),('updates_this_run',2734848),('random_action_transitions',0)]:
        bad=deepcopy(summary);bad[key]=value
        with pytest.raises(ValueError):runner.learner.validate_budget(bad,stage)


def test_two_pairs_and_failure_isolation_use_existing_owned_step_helpers():
    stages=[dict(s,status='pending') for s in runner.STAGES]
    assert [(s['label'],a) for s,a in runner.base.dispatchable(stages,[1364,1457])]==[
        ('R6_false010',1364),('R7_false025_actor3e5',1457)]
    stages[0].update(status='failed',allocation=1364);stages[1].update(status='running',allocation=1457)
    assert runner.base.dispatchable(stages,[1364,1457])==[] and stages[1]['status']=='running'
