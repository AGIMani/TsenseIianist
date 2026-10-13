"""Exercise mounting changes against real compiled hand/arm FK, without a GUI."""
import json
import shutil
import numpy as np
import pytest

from tsensei_v3_2_pianist import suite
from tsensei_v3_2_pianist.preview_arm_ik import ArmPreview
from tsensei_v3_2_pianist.preview_connector import ConnectorPreview, DEFAULT_CONFIG, offset_transform, tool_transforms


def test_adjust_reset_save_and_reject_concurrent_configuration_change(tmp_path):
    config_path = tmp_path/'mount.json'
    shutil.copy2(DEFAULT_CONFIG, config_path)
    env = suite.load()
    try:
        connector = ConnectorPreview(env.task.root_entity.mjcf_model, config_path)
        arms = ArmPreview(env.task.root_entity.mjcf_model, tool_overrides=tool_transforms(connector.config))
        env.reset()
        hands = (env.task.right_hand, env.task.left_hand)
        arms.bind(env.physics, hands)
        connector.bind(env.physics, hands, arms)
        before = env.physics.data.qpos.copy()
        original_tool = tool_transforms(connector.config)['left']
        connector.request('left', [1., 2., -.5], [0., 0., 1.])
        connector.apply()
        connector.publish()
        assert not connector.status.startswith('未应用')
        new_tool = tool_transforms(connector.config)['left']
        np.testing.assert_allclose(new_tool @ offset_transform(connector.config['offsets']['left']), original_tool, atol=1e-12)
        np.testing.assert_array_equal(before, env.physics.data.qpos)
        assert arms.report()['arms']['left']['position_error_mm'] < .01
        connector.save()
        saved = json.loads(config_path.read_text())
        assert saved['offsets']['left']['xyz_mm'] == [1., 2., -.5]
        assert len(list(tmp_path.glob('mount.json.bak-*'))) == 1
        connector.request('left', [0.]*3, [0.]*3)
        connector.apply()
        connector.publish()
        np.testing.assert_allclose(tool_transforms(connector.config)['left'], original_tool, atol=1e-12)
        config_path.write_text(config_path.read_text()+'\n')
        with pytest.raises(ValueError, match='其他进程'):
            connector.save()
    finally:
        env.close()
