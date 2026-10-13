"""The MJLab preview reconstructs the current dual-arm training contract."""
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MUJOCO_GL", "disable")

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - composer-only environments
    torch = None


@unittest.skipUnless(torch is not None, "training_mjlab environment is required")
class MjlabPreviewSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import (
            build_bundle_from_config, load_training_config,
        )

        cls.config_path = Path(__file__).parents[1] / "configs/preview/current_arm_training_20261011.json"
        cls.config = load_training_config(cls.config_path)
        cls.bundle, cls.scene_directory, _ = build_bundle_from_config(cls.config)

    def test_source_and_arm_contract_are_current(self):
        self.assertEqual(self.config["arm_follow"]["controller"], "rigid_wrist_command")
        self.assertEqual(
            len(self.bundle.arm_follow_spec["parent_collision_filter"]["restored_body_pairs"]), 20
        )
        self.assertTrue(self.config["reward_config"]["terms"]["self_collision_reward"]["enabled"])
        self.assertEqual(
            self.config["reward_config"]["terms"]["self_collision_reward"]["params"]["maximum_penalty_per_step"],
            5.0,
        )

    def test_small_reachability_scan_writes_classified_rows(self):
        from tsensei_v3_2_pianist.mjlab_env.arm_reachability import scan_reachability

        with tempfile.TemporaryDirectory(prefix="tsensei-preview-test-") as directory:
            summary = scan_reachability(
                self.bundle, directory, tx_points=2, ty_points=2, pitch_points=2,
                preparation_points=1, extra_attempts=1, batch_size=8,
            )
            self.assertEqual(summary["sides"]["left"]["total"], 8)
            self.assertEqual(summary["sides"]["right"]["total"], 8)
            for side in ("left", "right"):
                rows = summary["sides"][side]
                self.assertEqual(rows["solved"] + rows["proven_outside_bound"] + rows["unresolved"], rows["total"])
                self.assertTrue((Path(directory) / f"{side}.csv").is_file())
                self.assertTrue((Path(directory) / "summary.json").is_file())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "Slurm CUDA allocation required")
    def test_gpu_task_uses_arm_contract_for_one_step(self):
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_task_from_config

        env, _, _, _ = build_task_from_config(self.config, device="cuda:0", num_envs=1)
        observation = env.reset()
        self.assertEqual(env.scene.arm_follower.contract["controller"], "rigid_wrist_command")
        self.assertEqual(len(env.scene.arm_follower.contract["parent_collision_filter"]["restored_body_pairs"]), 20)
        action = torch.zeros((1, env.action_size), device="cuda:0")
        result = env.step(action)
        self.assertTrue(torch.isfinite(observation).all())
        self.assertTrue(torch.isfinite(result.observation).all())
        self.assertTrue(torch.isfinite(result.reward).all())


if __name__ == "__main__":
    unittest.main()
