#!/usr/bin/env python3
"""Explicitly compare approved software control limits on the same saved actor.

Rebuild current source with the parent's exact model/score recipe parameters,
then retain strict scene/task/preparation/network checks. Changed controller
contracts use the explicit networks-only migration gate. No learner updates.
"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--limits', type=Path, nargs='*', default=[])
    p.add_argument('--include-baseline', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.episodes < 1:
        p.error('episodes must be positive')
    a.checkpoint = a.checkpoint.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.mjlab_env.control_limits import canonical
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch import checkpoint
    from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
    from tsensei_v3_2_pianist.rl_torch.train import evaluate
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    old = payload['metadata']
    recipe = old['source_runtime']
    if recipe['external_score'] is not None:
        raise ValueError('This probe currently supports registered scores only')
    prep = recipe['preparation'] or {}
    current = make_recipe(**recipe['pose'], **recipe['model'], **recipe['environment'],
        preparation_time=prep.get('duration', 0.), preparation_lift=prep.get('lift', .02),
        preparation_mode=prep.get('mode', 'fixed_v1'))
    scene, task, _ = build(current)
    if old['cli'].get('no_arms'):
        scene, task = without_arms(scene, task, scene.parent/'hand_only_bundle')
    assert sha256(scene/'manifest.json') == old['scene_sha256']
    assert sha256(task/'task.json') == old['task_sha256']
    (a.output/'source_recipe.json').write_text(json.dumps(current, indent=2)+'\n')
    variants = [('baseline', None)] if a.include_baseline else []
    variants += [(path.stem, canonical(json.loads(path.read_text()))) for path in a.limits]
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError('Variant filenames must be unique')
    results = []
    for name, limits in variants:
        options = checkpoint.reward_options(a.checkpoint)
        options['control_limits'] = limits
        options['collision_diagnostics'] = True
        env = PianoTaskEnv(scene, task, device='cuda:0', **options)
        expected = deepcopy(old)
        expected.update(control_limits=limits, control_limits_contract=env.scene.control_limits,
                        preparation_contract=env.scene.bundle.preparation)
        sac_config = deepcopy(payload['agent']['config'])
        sac_config['init_temperature'] = float(payload['agent']['log_temperature'].exp())
        agent = SAC(env.observation_size, env.action_size, SACConfig(**sac_config), device='cuda:0',
                    compile_learner=False, fused_optimizer=False, action_profile=old['action_profile'])
        checkpoint.load(a.checkpoint, agent, expected, weights_only=True,
                        transfer_mode='finetune', allow_control_change=limits != old.get('control_limits'))
        start = time.perf_counter()
        metrics = evaluate(agent, env, a.episodes)
        result = dict(name=name, control_limits=limits, resolved=env.scene.control_limits,
                      metrics=metrics, elapsed_s=time.perf_counter()-start,
                      parent=str(a.checkpoint), parent_sha256=sha256(a.checkpoint),
                      scene_sha256=old['scene_sha256'], task_sha256=old['task_sha256'],
                      note='No gradient updates; repeated canonical CUDA-graph evaluation')
        (a.output/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
        results.append(result)
        (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(dict(name=name, music_f1_min=metrics['music_only_f1_min'],
            micro_f1_min=metrics['music_only_micro_f1_min'],
            prep_wrong_max=metrics['preparation_wrong_key_frames_max'],
            hand_substeps_max=metrics['hand_collision_substeps_max'],
            tx_jump=metrics['motion_max_wrist_tx_target_jump_m'],
            finger_jump=metrics['motion_max_finger_target_jump_rad'])), flush=True)
        del env, agent
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
