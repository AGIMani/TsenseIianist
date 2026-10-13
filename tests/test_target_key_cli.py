"""Removed reward-clock switches must fail explicitly, including disabled values."""
import json
from pathlib import Path

import pytest

from tsensei_v3_2_pianist.mjlab_env.reward_config import template
from tsensei_v3_2_pianist.rl_torch.train import parser, effective_config


@pytest.mark.parametrize('name,value', [
    ('substep_rewards', False), ('substep_rewards', True),
    ('target_key_substep_reward', False), ('target_key_substep_reward', True),
    ('target_key_activation_weight', 0.), ('target_key_activation_weight', .5),
    ('reward_sampling', 'auto'), ('reward_sampling', 'control_endpoint')])
def test_old_json_is_rejected_without_silent_migration(tmp_path, name, value):
    cfg = template()
    cfg['performance'] = {name: value}
    path = tmp_path/'reward.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='matching archived source'):
        effective_config(parser().parse_args(['--reward-config', str(path)]))


def test_default_wrapper_contains_no_removed_reward_switches():
    root = Path(__file__).resolve().parents[1]
    for name in ('all_use_this_train.sh', 'scripts/hpc/train_coupled_20hz.sh'):
        path = root/name
        if not path.exists():
            continue  # Root launcher is intentionally outside the frozen source.
        text = path.read_text()
        for flag in ('substep-rewards', 'target-key-substep-reward',
                     'target-key-activation-weight', 'reward-sampling'):
            assert flag not in text
