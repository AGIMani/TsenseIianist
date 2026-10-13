"""Two-GPU source range scaling and execution-only comparison contracts."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from tsensei_v3_2_pianist.mjlab_env import action_mapping, source_runtime


def recipe(**kwargs):
    with patch.object(source_runtime, 'source_files', return_value={}), patch.object(
            source_runtime, 'dependency_versions', return_value={}):
        return source_runtime.make_recipe(**kwargs)


def test_pitch_recipe_and_explicit_range_validation():
    assert 'wrist_pitch_range_scale' not in recipe()['model']
    assert recipe(wrist_pitch_range_scale=.8)['model']['wrist_pitch_range_scale'] == .8
    for value in (0, -1, 1.01, float('inf'), float('nan'), True):
        with pytest.raises(ValueError, match='range_scale'):
            recipe(wrist_pitch_range_scale=value)
    rows = action_mapping._direct_contract()['channels']
    names = [r['name'] for r in rows]
    low, high = ([r['physical_'+side] for r in rows] for side in ('min', 'max'))
    scales = {'R_Wrist_Pitch': .8, 'L_Wrist_Pitch': .8}
    for name in scales:
        i = names.index(name);low[i] *= .8;high[i] *= .8
    with pytest.raises(ValueError, match='mechanical range'):
        action_mapping.physical_bounds('direct_joint_v1', names, low, high)
    bounds = action_mapping.physical_bounds('direct_joint_v1', names, low, high, range_scales=scales)
    np.testing.assert_allclose(bounds, (low, high), atol=0, rtol=0)
    contract = action_mapping.resolved_contract('direct_joint_v1', 31, names, low, high, range_scales=scales)
    for row in contract['channels']:
        if row['name'] in scales:
            assert row['physical_min'] == pytest.approx(-.16)
            assert row['physical_max'] == pytest.approx(.16)
            assert (row['policy_min'], row['policy_max']) == (-1, 1)


def test_wrapper_comparison_only_execution_differs():
    from test_hpc_40hz_entry import arguments
    from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
    import argparse
    configs = []
    for flag in ('--no-eval-metrics-graph', '--eval-metrics-graph'):
        argv = arguments('--vector-steps', '19532', '--seed', '42', flag)
        args, _ = parser().parse_known_args(argv)
        source = argparse.ArgumentParser();source.add_argument('--wrist-pitch-range-scale', type=float)
        assert source.parse_known_args(argv)[0].wrist_pitch_range_scale == .8
        config = effective_config(args)
        assert config['sac']['gamma'] == .75
        configs.append(config)
    assert not configs[0]['acceleration']['eval_metrics_graph']
    assert configs[1]['acceleration']['eval_metrics_graph']
    configs[1]['acceleration']['eval_metrics_graph'] = False
    assert configs[0] == configs[1]


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='Slurm GPU source integration')
def test_actual_pitch_model_policy_endpoints_and_twinkle(tmp_path):
    import torch
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser
    from test_hpc_40hz_entry import arguments
    r = source_runtime.make_recipe(wrist_pitch_range_scale=.8, control_hz=40, physics_hz=400,
        lookahead=20, initial_buffer_time=0, release_tail_time=.25, score_validation='strict_v1',
        task_profile='tsensei_collision_v1',drive_profile='shadow_v1',preparation_time=1,
        preparation_lift=.025,preparation_mode='policy_v1',exact_note_timing=True)
    scene, task, config = source_runtime.build(r)
    args, _ = parser().parse_known_args(arguments())
    c = effective_config(args)
    env = PianoTaskEnv(scene,task,num_envs=3,device='cuda:0',action_profile='direct_joint_v1',
        disable_wrist_tz=True,disable_wrist_yaw=True,preparation=c['preparation'],
        reward_config=c['reward_config'],finger_coupling=c['finger_coupling'],
        control_limits=c['control_limits'],finger_clearance=c['finger_clearance'],collision_diagnostics=True)
    assert env.action_size == 31 and env.observation_size == 2297
    assert len(env.bundle.exact_note_events) == 34
    assert env.logic.length == 365
    for row in env.scene.action_contract['channels']:
        if row['name'].endswith('_Wrist_Pitch'):
            full = row['full_index'];act = env.scene.bundle.action_ids[list(env.scene.bundle.active_action_indices).index(full)]
            jid = env.scene.model.actuator_trnid[act,0]
            np.testing.assert_allclose(env.scene.model.jnt_range[jid], [-.16,.16], rtol=0, atol=1e-12)
            np.testing.assert_allclose(env.scene.model.actuator_ctrlrange[act], [-.16,.16], rtol=0, atol=1e-12)
            for action, target in ((-1,-.16),(1,.16)):
                policy=torch.zeros((3,31),device='cuda:0');policy[:,row['index']]=action
                assert env.scene.physical_action(policy)[0,full].item() == pytest.approx(target)
    # A complete diagnostic episode also proves the new graph accepts real fields.
    from tsensei_v3_2_pianist.rl_torch.train import evaluate
    class Neutral:
        device='cuda:0'
        def act(self, obs, deterministic=False):
            return torch.zeros((len(obs),31),device=self.device)
    result=evaluate(Neutral(),env,3,metrics_graph=True)
    assert result['completed_episodes']==3
    assert result['collision_diagnostics_valid'] and result['key_contact_diagnostics_valid']
    assert result['substep_key_diagnostics_valid']
    assert result['note_timing_reference_count']==102
    # Checkpoint metadata must not contain unsafe numpy scalar globals.
    out=tmp_path/'metrics.pt';torch.save(result,out)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(out)==[]
    assert torch.load(out,weights_only=True)==result
