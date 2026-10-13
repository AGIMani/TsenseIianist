#!/usr/bin/env python3
"""Rescore one recorded trajectory without changing actions, physics or F1."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from tsensei_v3_2_pianist.mjlab_env.rewards import target_key_positive_reward

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--activation-weight', type=float, default=.5)
    parser.add_argument('--key-tolerance', type=float, default=.05,
                        help='Use the matching task manifest tolerance (current tasks: .05)')
    args = parser.parse_args()
    if not math.isfinite(args.key_tolerance) or args.key_tolerance <= 0:
        parser.error('--key-tolerance must be positive and finite')
    video = np.load(args.recording/'rollout.npz')
    audio = np.load(args.recording/'audio_substeps.npz')
    target = torch.as_tensor(video['goals'], dtype=torch.float32)
    frames = len(target)
    # The exported recording has float64 storage for GPU float32 states.
    qpos = torch.as_tensor(audio['key_qpos'], dtype=torch.float32)
    upper = torch.as_tensor(audio['key_limits'][:, 1], dtype=torch.float32)
    if qpos.shape[0] % frames or qpos.shape[1] != 88:
        raise ValueError('Expected one complete physics trajectory without a separate preparation prefix')
    if int(audio['preparation_prefix_substeps']):
        raise ValueError('Remove/handle fixed preparation prefix explicitly before auditing')
    steps = qpos.shape[0]//frames
    clipped = torch.maximum(qpos, torch.zeros_like(qpos)).minimum(upper)
    activation = ((clipped-upper).abs() <= float(audio['key_threshold'])).reshape(frames, steps, 88)
    travel = (clipped/upper).reshape(frames, steps, 88)
    assert torch.equal(activation[:, -1], torch.as_tensor(video['actual']))
    snapshot = dict(key_state=travel[:, -1], key_activation=activation[:, -1],
                    key_state_history=travel, key_activation_history=activation)
    tolerance = args.key_tolerance
    modes = dict(endpoint=dict(), substep_continuous=dict(substeps=True),
                 substep_activation=dict(substeps=True, activation_weight=args.activation_weight))
    scores = {name: target_key_positive_reward(snapshot, target, tolerance, **options).numpy()
              for name, options in modes.items()}
    active = target[:, :88].bool()
    occupancy = ((activation & active[:, None]).float().sum(-1)
                 / active.sum(-1).clamp_min(1)[:, None]).mean(-1).numpy()
    music = np.asarray(video['phase']) == 'music'
    # Renderer uses 'playing' in some historical recordings.
    if not music.any():
        music = np.asarray(video['music_index']) >= 0
    selected = music & active.any(-1).numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ['frame', 'policy_time', 'music_index', 'target_keys', 'activation_occupancy', *scores]
    with (args.output_dir/'target_reward_frames.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(frames):
            writer.writerow(dict(frame=i, policy_time=float(video['policy_time'][i]),
                music_index=int(video['music_index'][i]), target_keys=int(active[i].sum()),
                activation_occupancy=float(occupancy[i]), **{name:float(x[i]) for name,x in scores.items()}))
    result = dict(scope='same recorded states; no policy run or F1 improvement claim',
                  activation_weight=args.activation_weight, key_tolerance=tolerance,
                  frames=frames, substeps=steps, active_music_frames=int(selected.sum()),
                  endpoint_activation_exact_match=True,
                  mean_positive_component_active_music={name:float(x[selected].mean()) for name,x in scores.items()},
                  mean_activation_occupancy_active_music=float(occupancy[selected].mean()),
                  source_sha256={name:hashlib.sha256((args.recording/name).read_bytes()).hexdigest()
                                 for name in ('rollout.npz','audio_substeps.npz')})
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
