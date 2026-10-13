"""Historical Shadow ablation JSON must use its immutable source archive."""
import json
from pathlib import Path
import pytest
from tsensei_v3_2_pianist.mjlab_env.reward_config import validate


def test_historical_shadow_ablation_requires_archived_source():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/experiments/twinkle_shadow_only_20261003.json').read_text())
    with pytest.raises(ValueError, match='matching archived source'):
        validate(cfg)
