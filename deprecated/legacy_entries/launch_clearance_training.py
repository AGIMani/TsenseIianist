#!/usr/bin/env python3
"""Run explicit geometry/reward network fine-tuning through the current run.sh.

Arguments (also available via --help):
  --parent PATH                 Required trusted local checkpoint, 1181/47.
  --steps INT                   Vector steps, default 4096 (not transitions).
  --actor-lr FLOAT              Actor LR; default parent saved Actor LR.
  --learning-start-transitions N First learning sample (one-based); default
                                parent value. Check full-song refill after any
                                intentional environment-count/task override.
  --eval-interval INT           Vector steps between evaluations, default 100.
  --eval-episodes INT           Complete episodes/evaluation, default 3.
  --save-interval INT           Periodic checkpoint interval, default 500.
  --output-dir PATH             Output base name; source entry adds timestamp.
  --plan-output PATH            Optionally write exact launch JSON/provenance.
  --dry-run                     Read parent on CPU, print command, do not train.
  -- [run.sh arguments ...]     Forward overrides unchanged, e.g. explicit
                                --finger-clearance-config, --reward-config,
                                --control-limits, --midi, --batch-size, --seed.

Source pose/score/timing default to the parent's recipe; no song is hardcoded.
Other new controller/reward/execution defaults come from run.sh. Learned alpha
is read from the parent, while Adam/replay are fresh. This is an explicit
networks-only transfer, not exact learner/world/replay restoration. Scene,
score, action, observation and preparation hashes remain strictly checked by
the trainer. A different score is not made compatible by forwarding --midi.
This wrapper starts in the foreground; a supervising caller owns background
processes/log files. Use --dry-run to review before launching an experiment.
Early-stop thresholds are owned by run.sh; this wrapper has no disable switch.
Embedded external scores are rejected explicitly; use the source reconstruction
entry directly until verified temporary-score export is supported here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_parent(path):
    code = '''import json,sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
m=x['metadata'];s=x['agent']
print(json.dumps(dict(learned_alpha=float(s['log_temperature'].exp()),
 actor_lr=s['config']['actor_lr'],sac=s['config'],observation_size=s['obs_size'],
 action_size=s['action_size'],source_runtime=m['source_runtime'],
 learning_start_transition_1_based=m.get('learning_start_transition_1_based'),
 num_envs=m['num_envs'],action_profile=m['action_profile'],
 scene_sha256=m['scene_sha256'],task_sha256=m['task_sha256'],
 control_limits=m.get('control_limits'),finger_clearance=m.get('finger_clearance'),
 reward_config=m.get('reward_config'),parent_updates=s['updates'])))
'''
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return json.loads(subprocess.check_output([str(ROOT/'training_mjlab/.venv/bin/python'),
        '-c', code, str(path)], text=True, env=env))


def build_command(args, parent, rest):
    forbidden = {'--warm-start','--stage2-from','--finetune-from','--init-temperature',
                 '--allow-control-change','--allow-reward-change'}
    if any(value.split('=', 1)[0] in forbidden for value in rest):
        raise ValueError('Transfer and inherited learned-alpha arguments are owned by this wrapper')
    if any(value.split('=', 1)[0] in {'--target-stop', '--no-target-stop'} for value in rest):
        raise ValueError('Early-stop thresholds come from run.sh; this wrapper has no target-stop toggle')
    if (parent['observation_size'], parent['action_size']) != (1181, 47):
        raise ValueError('Current clearance controller requires the 1181-observation / 47-action parent contract')
    alpha = parent['learned_alpha']
    lr = args.actor_lr if args.actor_lr is not None else parent['actor_lr']
    start = args.learning_start_transitions
    if start is None:
        start = parent.get('learning_start_transition_1_based')
    if type(start) is not int or start <= 0:
        raise ValueError('Parent has no valid refill schedule; pass --learning-start-transitions explicitly')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Parent learned temperature must be finite and positive')
    recipe = parent['source_runtime']
    if recipe.get('external_score') is not None:
        raise ValueError('Embedded external score is not supported by this wrapper: its basename is not a portable file; use a verified source reconstruction entry')
    source = {**recipe['pose'], **recipe['model'], **recipe['environment']}
    accepted_source_fields = {'height', 'ty', 'yaw', 'collision_profile', 'drive_profile',
        'midi', 'control_hz', 'physics_hz', 'lookahead', 'frame_stack', 'initial_buffer_time',
        'gravity_compensation', 'action_reward_observation', 'disable_forearm_reward',
        'task_profile', 'thumb_target'}
    unknown = set(source) - accepted_source_fields
    if unknown:
        raise ValueError('Unsupported source recipe fields: ' + ', '.join(sorted(unknown)))
    from tsensei_v3_2_pianist.fingering_targets import thumb_target_options, validate_thumb_target
    thumb_target = validate_thumb_target(source.pop('thumb_target', None)) or thumb_target_options()
    command = ['bash', str(ROOT/'run.sh')]
    for name, value in source.items():
        flag = '--'+name.replace('_','-')
        if name == 'disable_forearm_reward':
            if value: command.append(flag)
        elif isinstance(value, bool):
            command.append(flag if value else '--no-'+name.replace('_','-'))
        elif value is not None:
            command.extend((flag, str(value)))
    # Always select the parent mode explicitly, so newer run.sh thumb presets
    # cannot silently replace the target recorded by an older source recipe.
    command += ['--thumb-target-mode', thumb_target['mode']]
    for colour in ('white', 'black'):
        distance = thumb_target[colour+'_distance']
        if distance is not None:
            command += [f'--thumb-{colour}-target-distance', str(distance)]
    prep = recipe.get('preparation') or {}
    command += ['--preparation-time', str(prep.get('duration', 0)),
        '--preparation-lift', str(prep.get('lift', .02)),
        '--preparation-mode', prep.get('mode', 'fixed_v1'),
        '--num-envs', str(parent['num_envs']), '--vector-steps', str(args.steps),
        '--action-profile', parent['action_profile'],
        '--actor-lr', repr(lr), '--init-temperature', repr(alpha),
        '--warmup-transitions', '0', '--learning-start-transitions', str(start),
        '--eval-at-start', '--eval-interval', str(args.eval_interval),
        '--eval-episodes', str(args.eval_episodes), '--save-best',
        '--save-interval', str(args.save_interval), '--collision-diagnostics',
        '--output-dir', str(args.output_dir),
        '--finetune-from', str(args.parent), '--allow-control-change', '--allow-reward-change']
    return command+list(rest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4096)
    p.add_argument('--actor-lr', type=float)
    p.add_argument('--learning-start-transitions', type=int)
    p.add_argument('--eval-interval', type=int, default=100)
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--save-interval', type=int, default=500)
    p.add_argument('--output-dir', type=Path, default=ROOT/'training_mjlab/outputs/clearance_finetune')
    p.add_argument('--plan-output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    args, rest = p.parse_known_args(argv)
    if rest and rest[0] == '--': rest = rest[1:]
    if min(args.steps, args.eval_interval, args.eval_episodes) <= 0 or args.save_interval < 0:
        p.error('Steps/eval interval/episodes must be positive; save interval must be nonnegative')
    if args.actor_lr is not None and (not math.isfinite(args.actor_lr) or args.actor_lr <= 0):
        p.error('Actor learning rate must be finite and positive')
    args.parent = args.parent.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.parent.is_file(): p.error('Parent checkpoint does not exist')
    parent = read_parent(args.parent)
    try: command = build_command(args, parent, rest)
    except ValueError as exc: p.error(str(exc))
    with args.parent.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    record = dict(parent=str(args.parent), parent_sha256=digest, parent_metadata=parent,
        transfer='finetune_networks_only_fresh_adam_replay_inherited_learned_alpha',
        run_sh_sha256=hashlib.sha256((ROOT/'run.sh').read_bytes()).hexdigest(),
        command=command, cwd=str(ROOT), dry_run=args.dry_run,
        note='Final complete source/controller/reward/SAC config is recorded by trainer; no convergence guarantee')
    print(json.dumps(record, indent=2), flush=True)
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(record, indent=2)+'\n')
    if not args.dry_run:
        env = os.environ.copy(); env['MUJOCO_GL'] = 'egl'
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
