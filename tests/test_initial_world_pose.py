"""Pure coordinate/CLI tests: never construct or load a MuJoCo model."""
import argparse
import importlib.util
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


def contract():
    spec = importlib.util.spec_from_file_location('initial_world_pose', ROOT / 'tsensei_v3_2_pianist/initial_world_pose.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_entry_world_coordinates_convert_to_accepted_slides():
    pose = contract().validate(dict(world_x=.16, world_z=.09, right_world_y=.15,
                                    left_world_y=-.15, palm_base_yaw=.2617993877991494))
    for side, y in (('right', .15), ('left', -.15)):
        slides = contract().slides_for_world(pose, side, (.18, y, .10))
        assert slides == pytest.approx(dict(palm_base_tx=0., palm_base_ty=.02, palm_base_tz=0.))


def test_nondefault_world_y_and_x_are_positions_not_joint_offsets():
    pose = contract().validate(dict(world_x=.20, world_z=.12, right_world_y=.11,
                                    left_world_y=-.12, palm_base_yaw=0.))
    right = contract().slides_for_world(pose, 'right', (.18, .15, .10))
    left = contract().slides_for_world(pose, 'left', (.18, -.15, .10))
    assert right == pytest.approx(dict(palm_base_tx=-.04, palm_base_ty=-.02, palm_base_tz=0.))
    assert left == pytest.approx(dict(palm_base_tx=.03, palm_base_ty=-.02, palm_base_tz=0.))


@pytest.mark.parametrize('updates', [dict(world_x=.24), dict(world_z=0), dict(world_x=True),
    dict(left_world_y=float('nan')), dict(palm_base_yaw=1), dict(ty=.02), dict(height=.09)])
def test_invalid_or_ambiguous_pose_rejected(updates):
    with pytest.raises(ValueError):
        contract().validate({**contract().DEFAULT_WORLD_POSE, **updates})


@pytest.mark.parametrize('argument', ['--height', '--ty=.02', '--yaw=.1'])
def test_old_options_fail_before_build(argument):
    with pytest.raises(SystemExit) as error:
        contract().reject_legacy_pose_arguments([argument], argparse.ArgumentParser())
    assert error.value.code == 2


def test_constants_world_override_preserves_mount_and_computes_slides(monkeypatch):
    monkeypatch.setenv('TSENSEI_V32_EXPORT_POSE',
        '{"world_x":0.16,"world_z":0.09,"right_world_y":0.11,"left_world_y":-0.12,"palm_base_yaw":0.2}')
    values = runpy.run_path(str(ROOT / 'tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py'))
    assert values['RIGHT_POSITION'] == (.18, .15, .09)
    assert values['LEFT_POSITION'] == (.18, -.15, .09)
    for side, tx in (('right', -.04), ('left', .03)):
        q = values['INITIAL_ANGLES_BY_SIDE'][side]
        assert q['palm_base_tx'] == pytest.approx(tx)
        assert q['palm_base_ty'] == pytest.approx(.02)
        assert q['palm_base_tz'] == 0
        assert q['palm_base_yaw'] == .2
    assert not any(name.startswith('forearm_') for name in values['PALM_BASE_DOFS'])
