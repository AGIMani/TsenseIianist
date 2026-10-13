"""CPU contract and pre-learning learner-state checks for supplemental R4/R5."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/hpc'))
import run_twinkle_learner_finetune as runner
from test_twinkle_f1_finetune import contracts


@pytest.mark.parametrize('stage',runner.STAGES,ids=lambda s:s['label'])
def test_full_config_rejects_unrelated_or_wrong_transfer_changes(contracts,tmp_path,stage):
    _,template,parent=contracts
    template['sac'].update(critic_lr=.0003,temp_lr=.0003)
    template['cli']['actor_lr']=.0003
    run,reward=tmp_path/'run',tmp_path/'reward.json'
    config=runner.expected_config(template,stage,run,reward,parent)
    runner.validate_config(config,template,stage,run,reward,parent)
    assert runner.validate_ready_config(config,template,stage,run,reward,parent)
    partial=deepcopy(config);partial.pop('replay_bytes')
    assert not runner.validate_ready_config(partial,template,stage,run,reward,parent)
    changes=[lambda c:c['sac'].update(gamma=.9),lambda c:c['sac'].update(init_temperature=.1),
        lambda c:c['sac'].update(actor_lr=.0001),lambda c:c['sac'].update(critic_lr=.00003),
        lambda c:c['cli'].update(seed=43),lambda c:c['cli'].update(actor_lr=.0001),
        lambda c:c['cli'].update(warm_start='/wrong.pt'),
        lambda c:c['cli'].update(allow_reward_change=not c['cli']['allow_reward_change']),
        lambda c:c['reward_config']['terms']['false_press_penalty'].update(weight=.25),
        lambda c:c.update(learning_start_transition_1_based=10000),
        lambda c:c.update(scene_sha256='different'),
        lambda c:c['origin'].update(fresh_optimizer_alpha_rng=not c['origin']['fresh_optimizer_alpha_rng'])]
    for change in changes:
        mutated=deepcopy(config);change(mutated)
        with pytest.raises(ValueError,match='effective-config'):
            runner.validate_config(mutated,template,stage,run,reward,parent)
    if stage['mode']=='warm_start':
        assert config['sac']['init_temperature']==.5
        assert config['cli']['warm_start']==str(parent) and config['cli']['finetune_from'] is None
        assert config['origin']['transfer_mode']=='learner_warm_restart'
    else:
        assert config['sac']['actor_lr']==config['cli']['actor_lr']==.00003


def test_transfer_commands_are_mutually_exclusive_and_preserve_reward(contracts,tmp_path):
    _,_,parent=contracts
    args=SimpleNamespace(parent=parent,output=tmp_path,source_archive=tmp_path/'source.tar.gz',reward_config=tmp_path/'reward.json')
    for stage in runner.STAGES:
        argv=runner.command(args,stage,1364)
        get=lambda key:argv[argv.index(key)+1]
        assert get('--allocation')=='1364' and get('--reward-config')==str(args.reward_config)
        assert get('--vector-steps')=='4096' and get('--eval-interval')=='200'
        assert get('--discount')=='0.75' and get('--target-key-activation-weight')=='0.5'
        assert '--warmup-transitions' not in argv
        if stage['mode']=='warm_start':
            assert get('--warm-start')==str(parent) and get('--init-temperature')=='.5'
            assert '--finetune-from' not in argv and '--allow-reward-change' not in argv and '--actor-lr' not in argv
        else:
            assert get('--finetune-from')==str(parent) and get('--init-temperature')=='.0006'
            assert '--allow-reward-change' in argv and '--warm-start' not in argv
            assert float(get('--actor-lr'))==.00003


@pytest.mark.parametrize('stage',runner.STAGES,ids=lambda s:s['label'])
def test_new_and_cumulative_update_budgets_are_distinct(stage):
    offset=2257280 if stage['mode']=='warm_start' else 0
    assert runner.updates_at(stage,0)==runner.updates_at(stage,200)==runner.updates_at(stage,365)==offset
    assert runner.updates_at(stage,366)==offset+128
    assert runner.updates_at(stage,4096)==offset+477568
    summary=dict(status='completed',vector_steps=4096,transitions=2097152,replay_insertions=2097152,
        updates_this_run=477568,updates_total=offset+477568,stop_reason='requested_budget_exhausted',
        target_reached=False,random_action_transitions=9999,
        collector_pipeline_state=dict(pending_vector_step=None,learner_updated_through_vector_step=4096,collected_vector_steps=4096))
    runner.validate_budget(summary,stage)
    for field,value in [('updates_this_run',offset+1),('updates_total',477569),('vector_steps',4095)]:
        changed=deepcopy(summary);changed[field]=value
        with pytest.raises(ValueError):runner.validate_budget(changed,stage)
    changed=deepcopy(summary);changed['collector_pipeline_state']['pending_vector_step']=4096
    with pytest.raises(ValueError,match='drained'):runner.validate_budget(changed,stage)


def optimizer(lr,filled):
    return dict(state={0:dict(step=torch.tensor(2257280.),exp_avg=torch.tensor([.1,.2]),exp_avg_sq=torch.tensor([.3,.4]))} if filled else {},
        param_groups=[dict(lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0,params=[0],fused=True,foreach=False,capturable=True)])


@pytest.fixture
def learner_files(contracts,tmp_path):
    _,template,parent=contracts
    template['cli']['actor_lr']=.0003
    template['sac'].update(critic_lr=.0003,temp_lr=.0003)
    previous=dict(actor={'weight':torch.tensor([1.,2.])},critic={'weight':torch.tensor([3.,4.])},
        target_critic={'weight':torch.tensor([5.,6.])},log_temperature=torch.tensor(math.log(.000593800854)),
        actor_optimizer=optimizer(.0003,True),critic_optimizer=optimizer(.0003,True),temp_optimizer=optimizer(.0003,True),
        updates=2257280,update_rng=torch.tensor([1,2,3],dtype=torch.uint8),action_rng=torch.tensor([4,5],dtype=torch.uint8))
    torch.save(dict(agent=previous),parent)
    return template,parent,previous,tmp_path


def forward_file(files,stage):
    template,parent,previous,tmp_path=files
    run=tmp_path/stage['label'];run.mkdir(exist_ok=True)
    config=runner.expected_config(template,stage,run,tmp_path/'reward.json',parent)
    current=deepcopy(previous)
    current['action_rng']=torch.tensor([77,88],dtype=torch.uint8)
    if stage['mode']=='warm_start':
        for name in ('actor_optimizer','critic_optimizer','temp_optimizer'):
            current[name]['param_groups'][0].update(fused=False,foreach=True,capturable=False)
    else:
        current.update(updates=0,log_temperature=torch.tensor(math.log(.0006)))
        for name,lr in [('actor_optimizer',.00003),('critic_optimizer',.0003),('temp_optimizer',.0003)]:
            current[name]=optimizer(lr,False)
    metadata=deepcopy(config);metadata.update(vector_step=200,updates=runner.initial_updates(stage))
    current['config']=deepcopy(config['sac'])
    payload=dict(schema=1,kind='learner_warm_restart',agent=current,metadata=metadata,exploration_rng=torch.tensor([99]))
    torch.save(payload,run/'forward.pt')
    return run,config,payload


def test_warm_forward_exact_state_except_runtime_flags_and_collection_rng(learner_files):
    stage=runner.STAGES[0];run,config,payload=forward_file(learner_files,stage);parent=learner_files[1]
    result=runner.verify_forward(run,config,stage,parent)
    assert result['initial_updates']==2257280 and result['new_updates']==0 and result['update_rng_parent_exact']
    mutations=[lambda a:a.update(log_temperature=torch.tensor(math.log(.5))),
        lambda a:a.update(update_rng=torch.tensor([8,9],dtype=torch.uint8)),
        lambda a:a['actor_optimizer']['state'][0].update(exp_avg=torch.tensor([.1,.25])),
        lambda a:a['critic_optimizer']['state'][0].update(step=torch.tensor(2257281.)),
        lambda a:a['temp_optimizer']['param_groups'][0].update(betas=(.8,.99)),
        lambda a:a['actor_optimizer']['param_groups'][0].update(lr=.00003),
        lambda a:a['actor'].update(weight=torch.tensor([1.,2.1]))]
    for mutate in mutations:
        changed=deepcopy(payload);mutate(changed['agent']);torch.save(changed,run/'forward.pt')
        with pytest.raises(ValueError):runner.verify_forward(run,config,stage,parent)


def test_network_forward_fresh_adam_temperature_and_all_learning_rates(learner_files):
    stage=runner.STAGES[1];run,config,payload=forward_file(learner_files,stage);parent=learner_files[1]
    result=runner.verify_forward(run,config,stage,parent)
    assert result['initial_updates']==0 and result['new_updates']==0
    for name in ('actor_optimizer','critic_optimizer','temp_optimizer'):
        changed=deepcopy(payload);changed['agent'][name]['param_groups'][0]['lr']=.1
        torch.save(changed,run/'forward.pt')
        with pytest.raises(ValueError,match='optimizer LR'):runner.verify_forward(run,config,stage,parent)
    changed=deepcopy(payload);changed['agent']['log_temperature']=torch.tensor(math.log(.5));torch.save(changed,run/'forward.pt')
    with pytest.raises(ValueError,match='alpha'):runner.verify_forward(run,config,stage,parent)
    changed=deepcopy(payload);changed['agent']['actor_optimizer']['state'][0]={};torch.save(changed,run/'forward.pt')
    with pytest.raises(ValueError,match='not fresh'):runner.verify_forward(run,config,stage,parent)


def test_dispatch_and_cleanup_reuse_independent_queue_helpers():
    stages=[dict(s,status='pending') for s in runner.STAGES]
    assert [(s['label'],a) for s,a in runner.base.dispatchable(stages,[1364,1457])]==[
        ('R4_learner_state',1364),('R5_actor3e5',1457)]
    stages[0].update(status='failed',allocation=1364)
    stages[1].update(status='running',allocation=1457)
    assert runner.base.dispatchable(stages,[1364,1457])==[]
    assert stages[1]['status']=='running'
