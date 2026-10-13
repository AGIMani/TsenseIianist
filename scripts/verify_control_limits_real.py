#!/usr/bin/env python3
"""Scheduled GPU acceptance of software targets in the real V3.2 scene.

No learner, gradients, model edits or checkpoint writes. Rebuild the parent's
model/score parameters from current source and require identical exported
scene/task hashes. Exercise both ordinary and captured physics with per-world
requests; verify actual reset ctrl, applied-action observation and subset reset.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--limits', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--steps', type=int, default=24)
    a = p.parse_args()
    if a.num_envs < 4 or a.steps < 12:
        p.error('Use at least 4 environments and 12 control steps per path')
    if a.output.exists():
        p.error('Output already exists; preserve previous acceptance reports')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import torch
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch import checkpoint
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    old = payload['metadata']
    recipe = old['source_runtime']
    if recipe['external_score'] is not None:
        raise ValueError('Registered scores only in this acceptance script')
    prep = recipe['preparation'] or {}
    current = make_recipe(**recipe['pose'], **recipe['model'], **recipe['environment'],
        preparation_time=prep.get('duration', 0.), preparation_lift=prep.get('lift', .02),
        preparation_mode=prep.get('mode', 'fixed_v1'))
    scene, task, _ = build(current)
    if old['cli'].get('no_arms'):
        scene, task = without_arms(scene, task, scene.parent/'hand_only_bundle')
    scene_hash, task_hash = sha256(scene/'manifest.json'), sha256(task/'task.json')
    assert scene_hash == old['scene_sha256'], 'Parent physical scene changed'
    assert task_hash == old['task_sha256'], 'Parent music task changed'
    controls = canonical(json.loads(a.limits.read_text()))
    assert controls is not None
    report = dict(parent=str(a.checkpoint.resolve()), parent_sha256=sha256(a.checkpoint),
        source_recipe=current, scene_sha256=scene_hash, task_sha256=task_hash,
        control_limits=controls, num_envs=a.num_envs, steps_per_path=a.steps, paths=[],
        note='Interface acceptance only; extreme synthetic target requests, no policy quality claim')

    for graph in (False, True):
        options = checkpoint.reward_options(a.checkpoint)
        options.update(control_limits=controls, collision_diagnostics=True)
        env = PianoTaskEnv(scene, task, num_envs=a.num_envs, physics_graph=graph, **options)
        s = env.scene
        obs = env.reset()
        assert (env.observation_size, env.action_size, s.model.nq, s.model.nu) == (1181, 47, 134, 46)
        expected_reset = s.tensor(s.bundle.neutral_action).clone()
        expected_reset[s.policy_ids[:-1]] = s.tensor(s.bundle.reset_ctrl)[s.action_ids]
        expected_reset[-1] = 0
        torch.testing.assert_close(s.control_reset_action, expected_reset, atol=0, rtol=0)
        torch.testing.assert_close(env.logic.last_action,
            expected_reset.expand(a.num_envs, -1), atol=0, rtol=0)
        fields = env.bundle.manifest['contract']['observation_fields_before_stack']
        offset = 0
        action_slice = None
        for field in fields:
            if field['path'] == 'action':
                assert field['size'] == 51
                action_slice = slice(offset, offset + 51)
            offset += field['size']
        assert action_slice is not None

        def check_action_observation(observation, actual):
            # Base features use feature-major frame stacking, preparation phase
            # is appended afterward. Inspect the publicly returned observation.
            base = observation[:, :env.logic.observation_size].reshape_as(env.logic.history)
            torch.testing.assert_close(base[:, action_slice, -1], actual, atol=0, rtol=0)
            torch.testing.assert_close(env.logic.last_action, actual, atol=0, rtol=0)

        check_action_observation(obs, expected_reset.expand(a.num_envs, -1))
        reset_ids = torch.arange(1, a.num_envs, 3, device=s.device)
        continuing = torch.ones(a.num_envs, device=s.device, dtype=torch.bool)
        continuing[reset_ids] = False
        all_states = lambda: dict(qpos=s.data.qpos, qvel=s.data.qvel, ctrl=s.data.ctrl,
            qacc_warmstart=s.data.qacc_warmstart, mocap_pos=s.data.mocap_pos,
            mocap_quat=s.data.mocap_quat, sustain=s.sustain, control_steps=s.control_steps,
            applied=s.last_applied_physical_action, requested=s.last_requested_physical_action,
            clip=s.last_control_clip_mask, slew=s.last_control_slew_mask,
            last_action=env.logic.last_action, history=env.logic.history,
            index=env.logic.index, last_reward=env.logic.last_reward,
            metrics=env.logic.metrics_sum, key_counts=env.logic.key_counts, returns=env.logic.returns)
        max_delta = {name: 0. for name in s.control_limits['channels']}
        clipped = slewed = 0
        saved_info = None
        for step in range(a.steps):
            if step == 7:
                preserved = {key: value[continuing].clone() for key, value in all_states().items()}
                obs = env.reset(reset_ids)
                for key, value in all_states().items():
                    assert torch.equal(value[continuing], preserved[key]), 'Subset reset polluted '+key
                torch.testing.assert_close(s.last_applied_physical_action[reset_ids],
                    expected_reset.expand(len(reset_ids), -1), atol=0, rtol=0)
                torch.testing.assert_close(s.data.ctrl[reset_ids],
                    s.tensor(s.bundle.reset_ctrl).expand(len(reset_ids), -1), atol=0, rtol=0)
                check_action_observation(obs, s.last_applied_physical_action)
            previous = s.last_applied_physical_action.clone()
            requested = expected_reset.expand(a.num_envs, -1).clone()
            for column, row in enumerate(s.control_limits['channels'].values()):
                index = row['index']
                high = (torch.arange(a.num_envs, device=s.device) + step + column) % 2 == 0
                requested[:, index] = torch.where(high, s.physical_high[index], s.physical_low[index])
            raw = s.policy_action(requested)
            requested = s.physical_action(raw)
            expected = requested.clone()
            for name, row in s.control_limits['channels'].items():
                index = row['index']
                bounded = requested[:, index].clamp(row['min'], row['max'])
                if row['max_velocity'] is not None:
                    delta = row['max_velocity'] * s.control_limits['control_dt']
                    bounded = torch.maximum(previous[:, index]-delta,
                                            torch.minimum(bounded, previous[:, index]+delta))
                expected[:, index] = bounded
            result = env.step(raw)
            actual = result.info['applied_physical_action']
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
            torch.testing.assert_close(result.info['raw_policy_action'], raw, atol=0, rtol=0)
            torch.testing.assert_close(result.info['requested_physical_action'], requested, atol=0, rtol=0)
            torch.testing.assert_close(s.data.ctrl[:, s.action_ids], actual[:, s.policy_ids[:-1]], atol=0, rtol=0)
            check_action_observation(result.observation, actual)
            assert torch.isfinite(s.data.qpos).all() and torch.isfinite(s.data.qvel).all()
            for name, row in s.control_limits['channels'].items():
                index = row['index']
                values = actual[:, index]
                assert (values >= row['min']-1e-7).all() and (values <= row['max']+1e-7).all()
                observed_delta = float((values-previous[:, index]).abs().max())
                max_delta[name] = max(max_delta[name], observed_delta)
                if row['max_velocity'] is not None:
                    assert observed_delta <= row['max_velocity']*s.control_limits['control_dt']+1e-7
            clipped += int(result.info['control_clip_mask'].sum())
            slewed += int(result.info['control_slew_mask'].sum())
            if saved_info is None:
                saved_info = result.info['applied_physical_action']
                saved_copy = saved_info.clone()
            else:
                assert torch.equal(saved_info, saved_copy), 'Returned info mutated after later step/reset'
        assert clipped > 0, 'Static clip path was not exercised'
        if any(row['max_velocity'] is not None for row in s.control_limits['channels'].values()):
            assert slewed > 0, 'Slew path was not exercised'
        if graph:
            assert s.control_graph is not None, 'Captured physical graph was not used'
        row = dict(physics_graph=graph, graph_captured=s.control_graph is not None,
            passed=True, subset_reset_ids=reset_ids.cpu().tolist(),
            unchanged_other_world_fields=list(all_states()), reset_ctrl_matches_model=True,
            applied_action_observation_matches=True, max_target_delta=max_delta,
            clip_count=clipped, slew_count=slewed, finite_state=True,
            control_limits_contract=s.control_limits)
        report['paths'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'control_limits_contract'}), flush=True)
        del env, s, result, all_states, saved_info, saved_copy
        gc.collect()
        torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
