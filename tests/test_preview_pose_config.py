"""Persist asymmetric poses and reconstruct them in a fresh physics process."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tsensei_v3_2_pianist import suite
from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as consts
from tsensei_v3_2_pianist.preview_pose_config import InitialPoseSaver, _replace_values


@pytest.fixture(scope='module')
def env():
    environment = suite.load()
    for hand in (environment.task.right_hand, environment.task.left_hand):
        hand.root_body.pos = np.asarray(hand.root_body.pos) + (.01, 0, .1)
    yield environment
    environment.close()


def saver_for(env, tmp_path, source='ctrl'):
    path = tmp_path / 'tsensei_hand_constants.py'
    path.write_bytes(Path(consts.__file__).read_bytes())
    return InitialPoseSaver(env.physics, (env.task.right_hand, env.task.left_hand),
                            path, tmp_path / 'backups', source=source)


@pytest.mark.parametrize('source', ['ctrl', 'qpos'])
def test_save_and_reload_asymmetric_pose_with_wrist_rotation(env, tmp_path, source):
    env.reset()
    before = Path(consts.__file__).read_bytes()
    hands = (env.task.right_hand, env.task.left_hand)
    for side, hand in enumerate(hands):
        for short, actual, target in [
                ('palm_base_tx', .01 - .03 * side, .03 - .05 * side),
                ('palm_base_ty', .01, .02), ('palm_base_tz', -.002, -.004),
                ('palm_base_pitch', .03 + .02 * side, .08 - .1 * side),
                ('palm_base_yaw', -.05 + .1 * side, .1 - .2 * side),
                (hand.hand_side.name.lower() + '_hand_index_MCP_joint', .3 + .1 * side, .5 + .1 * side)]:
            actuator = hand.actuator_for_joint(short)
            env.physics.bind(actuator.joint).qpos = actual
            env.physics.bind(actuator).ctrl = target
    env.physics.forward()
    saver = saver_for(env, tmp_path, source)
    initial_qpos, initial_ctrl = env.physics.data.qpos.copy(), env.physics.data.ctrl.copy()
    record = saver.save()
    np.testing.assert_array_equal(env.physics.data.qpos, initial_qpos)
    np.testing.assert_array_equal(env.physics.data.ctrl, initial_ctrl)
    assert Path(record['backup_file']).read_bytes() == before
    assert Path(consts.__file__).read_bytes() == before, 'Tests must not write real defaults'
    # Expected reset uses chosen values for BOTH qpos and ctrl. Installation
    # offsets and wrist slide joints must be composed once, without double counting.
    expected = {}
    for hand in hands:
        side = hand.hand_side.name.lower()
        values = record['saved_values']['INITIAL_ANGLES_BY_SIDE'][side]
        for joint in hand.joints:
            short = joint.name.removeprefix(side + '_hand_').removesuffix('_joint')
            env.physics.bind(joint).qpos = values[short]
        env.physics.forward()
        expected[side] = {'root': env.physics.bind(hand.root_body).xpos.copy(),
                          'rotation': env.physics.bind(hand.root_body).xmat.copy(),
                          'tips': env.physics.bind(hand.fingertip_sites).xpos.copy()}
    # Load saved Python source into a fresh constants module, retaining __file__
    # only so bundled assets resolve. No source repository files are overwritten.
    child = '''
import json,sys
from pathlib import Path
from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as consts
exec(compile(Path(sys.argv[1]).read_text(), sys.argv[1], 'exec'), vars(consts))
from tsensei_v3_2_pianist import suite
env=suite.load();env.reset();out={}
for hand in (env.task.right_hand,env.task.left_hand):
 side=hand.hand_side.name.lower()
 out[side]={'root':env.physics.bind(hand.root_body).xpos.tolist(),
            'rotation':env.physics.bind(hand.root_body).xmat.tolist(),
            'tips':env.physics.bind(hand.fingertip_sites).xpos.tolist(),
            'joints':{j.name:float(env.physics.bind(j).qpos[0]) for j in hand.joints},
            'ctrl':env.physics.bind(hand.actuators).ctrl.tolist(),
            'neutral':hand.neutral_control.tolist()}
env.close();Path(sys.argv[2]).write_text(json.dumps(out))
'''
    output = tmp_path / 'reloaded.json'
    subprocess.run([sys.executable, '-c', child, str(saver.path), str(output)],
                   check=True, capture_output=True, text=True)
    loaded = json.loads(output.read_text())
    for side in expected:
        for key in ('root', 'rotation', 'tips'):
            np.testing.assert_allclose(loaded[side][key], expected[side][key], atol=1e-9)
        np.testing.assert_array_equal(loaded[side]['ctrl'], loaded[side]['neutral'])
        for name, value in loaded[side]['joints'].items():
            short = name.removeprefix(side + '_hand_').removesuffix('_joint')
            assert value == record['saved_values']['INITIAL_ANGLES_BY_SIDE'][side][short]
    # Repeated saves keep separate backups and do not treat our own write as a conflict.
    assert saver.save()['backup_file'] != record['backup_file']


@pytest.mark.parametrize('invalid', [float('nan'), 999.])
def test_invalid_values_do_not_write_configuration(env, tmp_path, invalid):
    env.reset()
    saver = saver_for(env, tmp_path)
    before = saver.path.read_bytes()
    env.physics.bind(env.task.right_hand.actuators[0]).ctrl = invalid
    with pytest.raises(ValueError, match='outside'):
        saver.save()
    assert saver.path.read_bytes() == before
    assert not saver.backup_dir.exists()


def test_concurrent_edit_and_failed_replace_preserve_file(env, tmp_path, monkeypatch):
    env.reset()
    saver = saver_for(env, tmp_path)
    external = saver.path.read_bytes() + b'\n# User edit during preview\n'
    saver.path.write_bytes(external)
    with pytest.raises(ValueError, match='changed since'):
        saver.save()
    assert saver.path.read_bytes() == external
    saver = saver_for(env, tmp_path)
    before = saver.path.read_bytes()
    def fail(*args):
        raise PermissionError('simulated write failure')
    monkeypatch.setattr('tsensei_v3_2_pianist.preview_pose_config.os.replace', fail)
    with pytest.raises(PermissionError, match='simulated'):
        saver.save()
    assert saver.path.read_bytes() == before
    assert next(saver.backup_dir.glob('*.py')).read_bytes() == before
    assert json.loads(next(saver.backup_dir.glob('*.json')).read_text())['status'] == 'failed'


def test_source_edit_preserves_comments_and_unrelated_settings():
    source = '# 坐标\nRIGHT_POSITION = (1, 2, 3)  # 米\nOTHER = {"keep": 2}\n'
    updated = _replace_values(source, {'RIGHT_POSITION': (.1, .2, .3)})
    assert updated == '# 坐标\nRIGHT_POSITION = (0.1, 0.2, 0.3)  # 米\nOTHER = {"keep": 2}\n'
    with pytest.raises(ValueError, match='Expected one'):
        _replace_values(source, {'MISSING': 0})
