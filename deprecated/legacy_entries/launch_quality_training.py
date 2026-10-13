#!/usr/bin/env python3
"""Explicit, reproducible quality trials and network fine-tuning from step8500.

Run from either directory. Uses source + full config.json, never frozen scenes.
All options are listed by --help; parent physical/task contracts are checked by
the learner. Variant changes require the explicit networks-only reward transfer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / ('training_mjlab/outputs/'
    'twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894/step-8500.pt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT,
                   help='Same-model 1181/47 checkpoint; defaults to audited step8500')
    p.add_argument('--resume-learner', action='store_true',
                   help='Restore same-reward learner/Adam/alpha/RNG; replay and physics restart after full-song refill')
    p.add_argument('--variant', choices=('baseline', 'safe', 'humanoid'), default='humanoid',
                   help='baseline=5 rewards; safe=+false1/self.2; humanoid=+bounded conditioned posture')
    p.add_argument('--budget', choices=('original', 'accelerated'), default='accelerated',
                   help='original=batch256*1 update/transition; accelerated=1024*.25')
    p.add_argument('--actor-lr', type=float,
                   help='Actor learning rate; default 3e-4 for network transfer, saved value for learner resume')
    p.add_argument('--control-limits', type=Path,
                   help='Explicit software target bounds/rates JSON; mechanical model remains unchanged')
    p.add_argument('--allow-control-change', action='store_true',
                   help='Explicitly permit a changed control contract for networks-only transfer')
    p.add_argument('--learning-start-transitions', type=int, default=91137,
                   help='First learning sample; default fills one 178-frame song across 512 environments')
    p.add_argument('--vector-steps', type=int, default=4096,
                   help='512 transitions per vector step; default 2097152 samples')
    p.add_argument('--eval-interval', type=int, default=100,
                   help='Vector steps between evaluations, default 51200 samples')
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--save-interval', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path,
                   help='Base output name; source entry appends a fresh Beijing timestamp')
    p.add_argument('--target-stop', action=argparse.BooleanOptionalAction, default=True,
                   help='Stop only after 3 evaluations with every episode music/frame micro F1>=.99, no prep wrong frames or hand collisions')
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--dry-run', action='store_true', help='Print exact argument vector without training')
    a = p.parse_args()
    if min(a.vector_steps, a.eval_interval, a.eval_episodes) <= 0 or a.save_interval < 0:
        p.error('Steps, eval interval and episodes must be positive; save interval nonnegative')
    if a.learning_start_transitions <= 0:
        p.error('learning-start-transitions must be positive')
    if a.actor_lr is not None and (not math.isfinite(a.actor_lr) or a.actor_lr <= 0):
        p.error('actor-lr must be positive and finite')
    if a.resume_learner and a.allow_control_change:
        p.error('Changed control requires network transfer, not --resume-learner')
    if a.control_limits is not None:
        a.control_limits = a.control_limits.expanduser().resolve()
        if not a.control_limits.is_file():
            p.error(f'Control limits missing: {a.control_limits}')
    a.parent = a.parent.expanduser().resolve()
    if not a.parent.is_file():
        p.error(f'Parent checkpoint missing: {a.parent}')
    # Read learned entropy temperature on CPU. Fresh Adam/replay are intentional;
    # resetting a skilled policy to alpha=1 would change exploration dramatically.
    code = ('import torch,sys,json; x=torch.load(sys.argv[1],map_location="cpu",weights_only=True); '
            'print(json.dumps(dict(learned=float(x["agent"]["log_temperature"].exp()), '
            'configured=x["agent"]["config"]["init_temperature"], '
            'actor_lr=x["agent"]["config"]["actor_lr"], '
            'action_profile=x["metadata"].get("action_profile","legacy"))))')
    temperature = json.loads(subprocess.check_output(
        [str(ROOT/'training_mjlab/.venv/bin/python'), '-c', code, str(a.parent)], text=True))
    if temperature['action_profile'] != 'direct_joint_v1':
        p.error('Parent checkpoint must use direct_joint_v1 action mapping')
    alpha = temperature['configured'] if a.resume_learner else temperature['learned']
    actor_lr = a.actor_lr if a.actor_lr is not None else (
        temperature['actor_lr'] if a.resume_learner else .0003)
    reward = ROOT/'configs/rewards'/f'quality_{a.variant}_20260926.json'
    batch, ratio = ('256', '1') if a.budget == 'original' else ('1024', '.25')
    output = a.output_dir or ROOT/'training_mjlab/outputs'/f'twinkle_quality_{a.variant}'
    command = [sys.executable, '-u', str(ROOT/'scripts/train_from_source.py'),
        '--height', '.10', '--ty', '.04', '--yaw', '.15',
        '--control-hz', '20', '--physics-hz', '200', '--lookahead', '10', '--frame-stack', '1',
        '--midi', 'TwinkleTwinkleRousseau', '--initial-buffer-time', '0',
        '--preparation-mode', 'policy_v1', '--preparation-time', '1', '--preparation-lift', '.02',
        '--gravity-compensation', '--action-reward-observation', '--task-profile', 'shadow_v1',
        '--collision-profile', 'shadow_primitive_v1', '--drive-profile', 'shadow_v1',
        '--no-arms', '--disable-wrist-tz', '--disable-wrist-yaw', '--action-profile', 'direct_joint_v1',
        '--self-collision-weight', '0', '--reward-config', str(reward),
        '--stage', '1', '--seed', str(a.seed), '--init-temperature', repr(alpha),
        '--actor-hidden', '256', '256', '256', '--discount', '.8',
        '--actor-lr', repr(actor_lr), '--critic-lr', '.0003', '--temp-lr', '.0003',
        '--num-envs', '512', '--vector-steps', str(a.vector_steps),
        '--replay-capacity', '1000000', '--batch-size', batch,
        '--warmup-transitions', '0', '--learning-start-transitions', str(a.learning_start_transitions),
        '--updates-per-transition', ratio,
        '--eval-at-start', '--eval-interval', str(a.eval_interval), '--eval-episodes', str(a.eval_episodes),
        '--save-best', '--save-interval', str(a.save_interval), '--log-interval', '10',
        '--collision-diagnostics', '--physics-graph', '--compile-learner', '--fused-optimizer',
        '--learner-graph', '--optimizer-layout', '--output-dir', str(output),
        '--progress' if a.progress else '--no-progress']
    command += (['--warm-start', str(a.parent)] if a.resume_learner else
                ['--finetune-from', str(a.parent), '--allow-reward-change'])
    if a.control_limits is not None:
        command += ['--control-limits', str(a.control_limits)]
    if a.allow_control_change:
        command += ['--allow-control-change']
    if a.target_stop:
        command += ['--target-music-f1', '.99', '--target-micro-f1', '.99',
                    '--target-consecutive-evals', '3', '--target-max-preparation-wrong-frames', '0',
                    '--target-max-hand-collision-frames', '0']
    print(json.dumps(dict(parent=str(a.parent), parent_learned_alpha=temperature['learned'],
                          transfer='learner_warm_restart' if a.resume_learner else 'networks_only',
                          variant=a.variant, budget=a.budget, command=command), indent=2), flush=True)
    if not a.dry_run:
        env = os.environ.copy()
        env['MUJOCO_GL'] = 'egl'
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
