#!/usr/bin/env python3
"""Validate new reward using saved physical rollout actions; does not train."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=60)
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists')
    with np.load(a.rollout, allow_pickle=False) as data:
        actions = data['actions'][:a.steps]
    if not len(actions) or a.steps < 1: p.error('Need at least one action')
    env = PianoTaskEnv(ROOT/'training_mjlab/outputs/scene', ROOT/'training_mjlab/outputs/task_twinkle',
                       self_collision_weight=.2)
    tracker = env.scene.self_collision
    model = env.scene.model
    lookup = tracker.lookup.numpy()
    rows = []
    all_pairs = set()
    # Independently count the native contact records on CPU during one callback
    # control interval, including the final forward sample. No CPU physics step.
    reference = set()
    def collect(scene):
        d = scene.data
        n = int(d.nacon[0].item())
        geoms = d.contact.geom[:n].cpu().numpy()
        dist = d.contact.dist[:n].cpu().numpy()
        for (g1,g2), distance in zip(geoms, dist):
            if distance <= 0 and g1 >= 0 and g2 >= 0 and lookup[g1,g2]:
                reference.add(int(lookup[g1,g2]))
    original_step = env.scene.step
    for i, action in enumerate(actions):
        # Check callback compatibility on the third action (already colliding).
        if i == 2:
            env.scene.step = lambda x: original_step(x, substep_callback=collect)
        result = env.step(torch.as_tensor(action, device=env.device)[None])
        if i == 2:
            collect(env.scene)
            assert len(reference) == result.info['self_collision_pairs'][0].item()
            env.scene.step = original_step
        count = int(result.info['self_collision_pairs'][0].item())
        terms = result.info['reward_terms']
        penalty = terms['self_collision_reward'][0].item()
        torch.testing.assert_close(result.reward, torch.stack(list(terms.values())).sum(0))
        assert abs(penalty + .2*count) < 1e-5
        pairs = tracker.seen[0,1:].nonzero().flatten().cpu().tolist()
        all_pairs.update(pairs)
        rows.append(dict(time=(i+1)*.05,count=count,penalty=penalty,
                         pairs=[tracker.pair_names[j] for j in pairs]))
        if result.done.any(): break
    assert any(row['count'] > 0 for row in rows), 'Acceptance rollout must contain known self collisions'
    env.reset()
    assert tracker.count().item() == 0
    report = dict(status='passed',source_rollout=str(a.rollout.resolve()),steps=len(rows),
        collision_control_steps=sum(r['count']>0 for r in rows),
        unique_pairs=len(all_pairs),max_pairs_per_control=max(r['count'] for r in rows),
        candidate_total_penalty=sum(r['penalty'] for r in rows),
        callback_reference_pairs=len(reference),reset_count=tracker.count().item(),
        note='Replayed saved actions; this is not a policy trained with the new reward or an exact historical GPU trajectory.',rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__ == '__main__': main()
