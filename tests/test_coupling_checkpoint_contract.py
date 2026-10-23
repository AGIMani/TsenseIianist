"""Coupled policies cannot silently change follower geometry or hidden state."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tsensei_v3_2_pianist.rl_torch import checkpoint
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
from tsensei_v3_2_pianist.mjlab_env.finger_coupling import canonical, RANGE_MODE


OPTIONS = dict(schema=1, mode='distal_60deg_v1', target_angle_deg=60., reduce_observation=True)
GEOMETRY = dict(**OPTIONS, plane_normal=[0., 0., 1.], follower_full_indices=[6, 7])
OBSERVATION = dict(retained_base_indices=[0, 1], removed_base_indices=[2, 3])


def run_config(*args):
    return effective_config(parser().parse_args(list(args)))


class CouplingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'best.pt'
        self.config = SACConfig(actor_hidden=(8,), critic_hidden=(8,), critic_dropout=0.)
        self.agent = SAC(7, 3, self.config, device='cpu', seed=19)
        self.metadata = dict(scene_sha256='scene', task_sha256='task', stage=1)

    def save(self, metadata):
        checkpoint.save(self.path, self.agent, metadata, torch.Generator().manual_seed(11))

    def coupled(self):
        return dict(self.metadata, finger_coupling=deepcopy(OPTIONS),
                    finger_coupling_contract=deepcopy(GEOMETRY),
                    coupling_observation_contract=deepcopy(OBSERVATION))

    def test_disabled_legacy_checkpoint_reloads_unchanged(self):
        self.save(self.metadata)
        self.assertEqual(checkpoint.coupling_contract(self.metadata), (None, None, None))
        self.assertNotIn('finger_coupling', checkpoint.reward_options(self.path))
        checkpoint.load(self.path, self.agent, dict(self.metadata, finger_coupling=None,
            finger_coupling_contract=None, coupling_observation_contract=None))

    def test_matching_coupling_reloads_and_incomplete_contracts_fail(self):
        metadata = self.coupled()
        self.save(metadata)
        options = checkpoint.reward_options(self.path)
        self.assertEqual(options['finger_coupling'], OPTIONS)
        restored = checkpoint.load_agent(self.path, scene_sha256='scene', task_sha256='task',
            obs_size=7, action_size=3, device='cpu', finger_coupling_contract=deepcopy(GEOMETRY),
            coupling_observation_contract=deepcopy(OBSERVATION), **options)
        observation = torch.arange(7, dtype=torch.float32)[None]/10
        torch.testing.assert_close(restored.act(observation, deterministic=True),
                                   self.agent.act(observation, deterministic=True), rtol=0, atol=0)
        for key in ('finger_coupling', 'finger_coupling_contract', 'coupling_observation_contract'):
            broken = deepcopy(metadata); broken.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'requires'):
                checkpoint.coupling_contract(broken)

    def test_geometry_and_observation_changes_rejected_even_for_finetune(self):
        self.save(self.coupled())
        alternatives = [self.metadata, self.coupled(), self.coupled(), self.coupled()]
        alternatives[1]['finger_coupling_contract']['plane_normal'] = [0., 1., 0.]
        alternatives[2]['coupling_observation_contract']['removed_base_indices'] = [3, 2]
        alternatives[3]['finger_coupling']['target_angle_deg'] = 55.
        for changed in alternatives:
            for flags in ({}, dict(weights_only=True, transfer_mode='finetune', allow_control_change=True)):
                fresh = SAC(7, 3, self.config, device='cpu', seed=31)
                before = deepcopy(fresh.actor.state_dict())
                with self.subTest(changed=changed, flags=flags), self.assertRaisesRegex(ValueError, 'coupling'):
                    checkpoint.load(self.path, fresh, changed, **flags)
                for name, value in fresh.actor.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_cli_is_optional_and_full_observation_is_independent(self):
        self.assertIsNone(run_config()['finger_coupling'])
        ranges = canonical(dict(schema=1, mode=RANGE_MODE))
        self.assertEqual(run_config('--finger-coupling')['finger_coupling'], ranges)
        self.assertEqual(run_config('--finger-coupling', '--finger-coupling-angle', '60')['finger_coupling'], OPTIONS)
        full = run_config('--finger-coupling', '--no-coupling-reduce-observation')
        self.assertEqual(full['finger_coupling'], dict(ranges, reduce_observation=False))
        self.assertEqual(full['reward_config'], run_config()['reward_config'])
        self.assertEqual(full['control_limits'], run_config()['control_limits'])
        self.assertIsNone(run_config('--finger-coupling', '--no-finger-coupling')['finger_coupling'])
        for angle in ('0', '90', 'nan', '-1'):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                run_config('--finger-coupling', '--finger-coupling-angle', angle)

    def test_range_file_cli_and_checkpoint_reject_changed_finger_or_fixed_angle(self):
        options = canonical(dict(schema=1, mode=RANGE_MODE))
        options['angle_ranges_deg']['left_pinky'] = [55., 75.]
        path = Path(self.directory.name)/'ranges.json'
        path.write_text(json.dumps(options['angle_ranges_deg']))
        args = ('--finger-coupling', '--finger-coupling-angle-ranges', str(path))
        self.assertEqual(run_config(*args)['finger_coupling'], options)
        self.assertEqual(run_config(*args, '--finger-coupling-angle', '60')['finger_coupling'], OPTIONS)
        self.assertIsNone(run_config(*args, '--no-finger-coupling')['finger_coupling'])
        metadata = self.coupled()
        metadata.update(finger_coupling=options,
                        finger_coupling_contract=dict(options, plane_normal=[0., 0., 1.]))
        self.save(metadata)
        checkpoint.load(self.path, self.agent, metadata)
        self.assertEqual(checkpoint.reward_options(self.path)['finger_coupling'], options)
        changed = deepcopy(metadata)
        changed['finger_coupling']['angle_ranges_deg']['left_pinky'] = [50., 80.]
        for incompatible in (changed, self.coupled()):
            with self.assertRaisesRegex(ValueError, 'coupling'):
                checkpoint.load(self.path, self.agent, incompatible)


if __name__ == '__main__':
    unittest.main()
