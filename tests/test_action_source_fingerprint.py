"""Action JSON edits invalidate source replay even when package code is unchanged."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tsensei_v3_2_pianist.mjlab_env import source_runtime


class ActionFingerprintChecks(unittest.TestCase):
    def test_physical_range_edit_rejects_old_source_recipe_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tsensei_v3_2_pianist').mkdir()
            (root/'tsensei_v3_2_pianist/module.py').write_text('# unchanged\n')
            actions = root/'configs/actions'
            actions.mkdir(parents=True)
            target = actions/'direct_joint_v1.json'
            target.write_text('{"physical_max":1.2}\n')
            with patch.object(source_runtime, 'ROOT', root):
                original = source_runtime.source_files()
                recipe = dict(schema=1, method='source_config_v1', source_sha256=original)
                target.write_text('{"physical_max":1.3}\n')
                current = source_runtime.source_files()
                self.assertEqual(current['tsensei_v3_2_pianist/module.py'],
                                 original['tsensei_v3_2_pianist/module.py'])
                self.assertNotEqual(current['configs/actions/direct_joint_v1.json'],
                                    original['configs/actions/direct_joint_v1.json'])
                with self.assertRaisesRegex(ValueError, 'configs/actions/direct_joint_v1.json'):
                    source_runtime.build(recipe)


if __name__=='__main__':
    unittest.main()
