"""Recorded preparation proposals accept current names without model loading."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('transition', Path(__file__).resolve().parents[1] /
    'scripts/prepare_ik_wrist_transition.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('prefix', ['palm_base_', 'forearm_'])
def test_transition_recognizes_current_or_explicit_archived_joint_names(prefix):
    names = [side + '_/' + prefix + axis for side in ('right', 'left') for axis in ('tx', 'ty', 'pitch')]
    actions = [side + '_Palm_Base_' + axis for side in ('R', 'L') for axis in ('Tx', 'Ty', 'Pitch')]
    columns, initial, naming = module.palm_base_columns(
        dict(joints=names, initial_physical_controlled_qpos=[0, .02, 0, 0, .02, 0]), actions)
    assert columns == list(range(6))
    assert initial == [0, .02, 0, 0, .02, 0]
    assert naming == ('palm_base_v1' if prefix == 'palm_base_' else 'archived_forearm')


def test_mixed_naming_is_not_silently_merged():
    with pytest.raises(ValueError, match='Mixed'):
        module.palm_base_columns(dict(joints=['right_/forearm_tx', 'left_/palm_base_ty']), [])
