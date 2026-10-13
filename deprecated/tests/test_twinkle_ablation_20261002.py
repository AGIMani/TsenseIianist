"""Real entrypoint resolution and strict experiment isolation for Twinkle ablations."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/hpc'))
from run_twinkle_ablation import STAGES, overrides, validate_config, validate_ready_config
from test_hpc_40hz_entry import arguments
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config


def resolve(extra):
    args,_=parser().parse_known_args(arguments('--seed','42','--vector-steps','19532',*extra))
    return effective_config(args)


@pytest.mark.parametrize('label,anticipation,false_press',STAGES)
def test_only_requested_effective_reward_fields_change(label,anticipation,false_press):
    baseline=resolve(['--no-eval-metrics-graph'])
    expected=deepcopy(baseline)
    expected['reward_config']['performance']['anticipatory_fingering_time']=anticipation
    expected['reward_config']['terms']['false_press_penalty']['enabled']=false_press
    actual=resolve(overrides(anticipation,false_press))
    assert actual==expected
    assert actual['reward_config']['terms']['key_press_reward']==baseline['reward_config']['terms']['key_press_reward']
    assert actual['sac']['gamma']==.75
    assert actual['checkpoint_retention']['forward_vector_step']==1000


@pytest.mark.parametrize('label,anticipation,false_press',STAGES)
def test_runtime_validator_allows_only_switch_effects(label,anticipation,false_press):
    baseline=resolve(['--no-eval-metrics-graph'])
    baseline.update(cli={'seed':42},observation_size=2297,replay_bytes=37020000000)
    actual=resolve(overrides(anticipation,false_press))
    actual.update(cli={'seed':42},observation_size=2297 if anticipation else 2247,
                  replay_bytes=37020000000-(0 if anticipation else 800000000))
    validate_config(actual,baseline,anticipation,false_press)
    initial=deepcopy(actual);initial.pop('observation_size');initial.pop('replay_bytes')
    assert not validate_ready_config(initial,baseline,anticipation,false_press)
    assert validate_ready_config(actual,baseline,anticipation,false_press)
    changed=deepcopy(actual);changed['sac']['gamma']=.8
    with pytest.raises(ValueError,match='sac'):validate_config(changed,baseline,anticipation,false_press)
    changed=deepcopy(actual);changed['cli']['seed']=43
    with pytest.raises(ValueError,match='seed'):validate_config(changed,baseline,anticipation,false_press)
    changed=deepcopy(actual);changed['reward_config']['terms']['key_press_reward']['enabled']=False
    with pytest.raises(ValueError,match='reward_config'):validate_config(changed,baseline,anticipation,false_press)
