#!/usr/bin/env python3
"""Recover the exact archived task's fingering for an existing physics recording.

Only the temporary source/task compiler runs; no policy or physics rollout is
repeated. The reconstructed task hash and every recorded goal must match.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from eval_from_source_archive import extract_archive, validate_sources
from export_videos import sha256

ROOT = Path(__file__).resolve().parents[1]


def align_score(goal, fingers, policy_time, recorded_goals, control_dt, prefix):
    import numpy as np
    goal, fingers = np.asarray(goal), np.asarray(fingers)
    time = np.asarray(policy_time, dtype=float)
    if (goal.ndim != 2 or goal.shape[1] != 89 or
            fingers.shape != (len(goal), 10, 88) or
            not np.isfinite(fingers).all() or (fingers < 0).any() or
            not np.isin(goal, (0, 1)).all() or
            not np.array_equal(fingers.sum(axis=1) > 0, goal[:, :88] > 0)):
        raise ValueError('Invalid or inconsistent archived score/fingering')
    if (not np.isfinite(control_dt) or control_dt <= 0 or
            not np.isfinite(prefix) or prefix < 0 or time.ndim != 1 or
            not len(time) or not np.isfinite(time).all() or
            not np.allclose(time, np.arange(1, len(time)+1)*control_dt, rtol=0, atol=1e-7)):
        raise ValueError('Recorded control times are incomplete or inconsistent')
    if len(time) > len(goal) or not np.array_equal(recorded_goals, goal[:len(time)]):
        raise ValueError('Archived fingering targets differ from recorded goals')
    return dict(time=time+prefix, goal=goal[:len(time)], fingers=fingers[:len(time)])


def recorded_score_targets(score, policy_time):
    """Return action-frame Shadow targets, including policy preparation.

    policy_time labels the end of each recorded interval. Exact events supply
    observation/diagnostics only; they must not replace the applied grid goal
    or its fingering. align_score still verifies every recorded applied goal.
    """
    return score['goal'], score['fingers']


def recover(args):
    # This branch runs in a fresh interpreter with only verified archived code.
    sys.path.insert(0, str(args.archive_root))
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import build
    config = json.loads(args.config.read_text())
    report = json.loads((args.recording/'report.json').read_text())
    if report['source_runtime'] != config['source_runtime']:
        raise ValueError('Recording and source recipe differ')
    _, task, _ = build(config['source_runtime'])
    task_hash = sha256(task/'task.json')
    if task_hash != report['task_sha256']:
        raise ValueError('Reconstructed task hash differs from the recorded task')
    manifest = json.loads((task/'task.json').read_text())
    if sha256(task/'score.npz') != manifest['files']['score.npz']:
        raise ValueError('Archived score checksum mismatch')
    dt = float(manifest['control_dt'])
    if not np.isclose(dt*report['fps'], 1., rtol=0, atol=1e-8):
        raise ValueError('Recorded and archived control frequencies differ')
    prefix = float(report.get('preparation_prefix_seconds', 0.))
    with np.load(task/'score.npz', allow_pickle=False) as score, \
            np.load(args.recording/'rollout.npz', allow_pickle=False) as rollout:
        policy_time = rollout['policy_time'] if 'policy_time' in rollout else np.arange(1, len(rollout['goals'])+1)*dt
        goal, fingers = recorded_score_targets(score, policy_time)
        arrays = align_score(goal, fingers, policy_time, rollout['goals'], dt, prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'score_annotations.npz', **arrays)
    metadata = dict(schema=1, fingertip_sites=manifest['fingertip_sites'], control_dt=dt,
                    preparation_prefix_seconds=prefix, frames=len(arrays['time']),
                    checkpoint_sha256=report['checkpoint_sha256'], task_sha256=task_hash,
                    source_score_sha256=manifest['files']['score.npz'],
                    annotations_sha256=sha256(args.output/'score_annotations.npz'),
                    target_basis='shadow_action_control_frame',
                    source='exact archived task; all applied goals verified against the original recording',
                    time_definition='end of each scored control interval in original physics time')
    (args.output/'score_annotations.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Recovered {len(arrays["time"])} verified fingering frames without rerunning the policy.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive', type=Path)
    parser.add_argument('--archive-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.archive_root:
        recover(args)
        return
    if not args.source_archive:
        parser.error('--source-archive is required')
    recipe = json.loads(args.config.read_text())['source_runtime']
    with tempfile.TemporaryDirectory(prefix='tsensei-fingering-archive-') as temporary:
        archive_root = Path(temporary)
        extract_archive(args.source_archive, archive_root)
        validate_sources(archive_root, recipe['source_sha256'])
        for name in ('training', 'training_mjlab'):
            link = archive_root/name/'.venv'
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((ROOT/name/'.venv').resolve(), target_is_directory=True)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(archive_root), PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([str(ROOT/'training_mjlab/.venv/bin/python'), str(Path(__file__).resolve()),
                        '--archive-root', str(archive_root), '--config', str(args.config.resolve()),
                        '--recording', str(args.recording.resolve()), '--output', str(args.output.resolve())],
                       cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
