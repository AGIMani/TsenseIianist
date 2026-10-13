#!/usr/bin/env python3
"""Export annotated music, observation contract and actual CPU reference transitions."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=ROOT/'training_mjlab/outputs/scene_shadow_40hz400hz')
    p.add_argument('--output',type=Path,default=ROOT/'training_mjlab/outputs/task_twinkle_shadow_40hz400hz')
    p.add_argument('--midi',default='TwinkleTwinkleRousseau')
    p.add_argument('--lookahead',type=int,default=20)
    p.add_argument('--frame-stack',type=int,default=1)
    p.add_argument('--reference-steps',type=int,default=64)
    p.add_argument('--release-tail-time',type=float,default=0.)
    p.add_argument('--score-validation',choices=('legacy','strict_v1'),default='legacy')
    p.add_argument('--exact-note-timing',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--action-reward-observation',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--disable-forearm-reward',action='store_true')
    p.add_argument('--task-profile',choices=('tsensei_v1','tsensei_collision_v1'),default='tsensei_v1')
    a=p.parse_args()
    if a.reference_steps<1:p.error('--reference-steps must be positive')
    from tsensei_v3_2_pianist.mjlab_env.task_bundle import export_task
    m=export_task(a.output,a.scene,midi=a.midi,lookahead=a.lookahead,frame_stack=a.frame_stack,
        reference_steps=a.reference_steps,action_reward_observation=a.action_reward_observation,
        disable_forearm_reward=a.disable_forearm_reward,task_profile=a.task_profile,
        release_tail_time=a.release_tail_time,score_validation=a.score_validation,
        exact_note_timing=a.exact_note_timing)
    print(f"Exported music task: {a.output}; frames={m['length']}, observation={m['contract']['observation_size']}")
if __name__=='__main__':main()
