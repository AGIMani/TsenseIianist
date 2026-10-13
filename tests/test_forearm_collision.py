"""Cross-hand coverage, old frozen-task compatibility, and real GPU reward use."""
import unittest
from pathlib import Path
import numpy as np
import torch
import warp as wp

from tsensei_v3_2_pianist.forearm_collision import is_forearm_collision_geom, task_collision_groups
from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'training_mjlab/outputs'


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ForearmCoverageTest(unittest.TestCase):
    def test_cross_hand_contacts_and_legacy_masks(self):
        new_task = OUTPUTS/'task_clair_mixed_forearm_full_20260921'
        if not new_task.exists():
            self.skipTest('Export the new task first')
        env = PianoTaskEnv(OUTPUTS/'scene_clair_mixed_pitch02_20260920', new_task,
                           num_envs=8, action_profile='mixed_v1')
        model = env.scene.model
        groups = task_collision_groups(env.bundle.manifest)
        self.assertEqual(list(map(len, groups)), [21, 21])
        self.assertTrue(all(is_forearm_collision_geom(n) for group in groups for n in group))
        self.assertFalse(is_forearm_collision_geom('right_/base_link_collision_0'))
        self.assertFalse(is_forearm_collision_geom('right_/right_hand_index_DIP_link_visual_0'))
        def geom(name): return model.geom(name).id
        rp, lp = [geom(g[0]) for g in env.bundle.manifest['palm_geoms']]
        rf = geom('right_/right_hand_index_DIP_link_collision_0')
        lf = geom('left_/left_hand_thumb_PIP_link_collision_0')
        key = geom(env.bundle.manifest['key_geoms'][0])
        # Eight independent worlds: palm/palm, finger/finger, finger/palm,
        # same-hand, key, separated pair, reversed order, unused buffer slot.
        pairs = [(rp,lp),(rf,lf),(rf,lp),(rf,rp),(rf,key),(rf,lf),(lf,rp),(rf,lf)]
        raw = env.scene.sim.wp_data
        wp.to_torch(raw.contact.geom)[:8] = torch.tensor(pairs, device=env.device, dtype=torch.int32)
        wp.to_torch(raw.contact.worldid)[:8] = torch.arange(8, device=env.device, dtype=torch.int32)
        wp.to_torch(raw.contact.dist)[:8] = -.001
        wp.to_torch(raw.contact.dist)[5] = .01
        wp.to_torch(raw.nacon)[0] = 7
        snap = env.snapshot()
        expected = [True,True,True,False,False,False,True,False]
        self.assertEqual(snap['palm_collision'].tolist(), expected)
        terms, _ = env.logic.reward_terms(snap, env.logic.index)
        self.assertEqual(terms['forearm_reward'].tolist(), [0,0,0,.5,.5,.5,0,.5])
        # Current code no longer silently falls back to a palm-only mask.
        with self.assertRaisesRegex(ValueError, 'archived source'):
            task_collision_groups({'palm_geoms': env.bundle.manifest['palm_geoms']})
        env.reset()
        for _ in range(4):
            out = env.step(torch.zeros((8,51), device=env.device) + env.scene.action_shift)
            values = out.info['reward_terms']['forearm_reward']
            self.assertTrue(torch.all((values == 0) | (values == .5)))
            self.assertTrue(torch.isfinite(out.reward).all())
        env.reset([1,3])
        self.assertEqual(env.logic.index.tolist(), [4,0,4,0,4,4,4,4])


if __name__ == '__main__':
    unittest.main()
