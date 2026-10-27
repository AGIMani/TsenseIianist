#!/usr/bin/env python3
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
"""Build and record the actual first-note preparation; no policy or training."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--midi',required=True)
    p.add_argument('--physics-hz',type=float,default=1000.)
    p.add_argument('--duration',type=float,default=2.)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    import numpy as np
    import mujoco
    import torch
    torch.set_num_threads(1)
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe,build,digest
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.first_note_preparation import PreparationFailure
    from tsensei_v3_2_pianist.rl_torch.train import parser,effective_config
    args=parser().parse_args(['--no-arms','--disable-wrist-tz','--disable-wrist-yaw',
        '--action-profile','direct_joint_v1','--finger-coupling','--no-coupling-reduce-observation',
        '--finger-coupling-angle-ranges',str(ROOT/'configs/finger_coupling/angle_ranges_10_80.json'),
        '--preparation-mode','first_note_v1','--preparation-time',str(a.duration),'--preparation-lift','0',
        '--control-limits',str(ROOT/'configs/control_limits/generic_wrist_rate_20260927.json'),
        '--wrist-tx-max-velocity','1.99','--wrist-ty-max-velocity','1.99',
        '--wrist-tx-max-acceleration','1.99','--wrist-ty-max-acceleration','1.99',
        '--wrist-pitch-max-velocity','3.1416','--wrist-pitch-max-acceleration','15.708','--hard-joint-limits'])
    config=effective_config(args)
    context={key:config[key] for key in ('action_profile','finger_coupling','control_limits')}
    recipe=make_recipe(world_x=.14,world_z=.09,palm_base_yaw=.2617993877991494,control_hz=20.,physics_hz=a.physics_hz,
        midi=a.midi,lookahead=10,frame_stack=1,initial_buffer_time=0.,release_tail_time=.25,
        score_validation='strict_v1',task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',
        drive_profile='shadow_v1',preparation_time=a.duration,preparation_lift=0.,preparation_mode='first_note_v1')
    scene,task,_=build(recipe)
    scene,task=without_arms(scene,task,scene.parent/'hand_only_bundle')
    a.output.mkdir(parents=True)
    (a.output/'source_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    failed=None
    try:
        bundle=SceneBundle(scene,disable_wrist_tz=True,disable_wrist_yaw=True,
            preparation=config['preparation'],preparation_context=context)
        contract,trace=bundle.preparation,bundle.preparation_trace
    except PreparationFailure as error:
        failed=error
        contract,trace=error.contract,error.trace
        bundle=SceneBundle(scene,disable_wrist_tz=True,disable_wrist_yaw=True)
    # Export the same model topology and original physics trace, never re-simulate for video.
    mujoco.mj_saveModel(bundle.model,str(a.output/'model.mjb'))
    bundle.model.vis.global_.offwidth=1280;bundle.model.vis.global_.offheight=720
    np.savez_compressed(a.output/'preparation.npz',**trace)
    with np.load(task/'score.npz',allow_pickle=False) as z:
        np.savez_compressed(a.output/'original_score.npz',events=z['exact_note_events'])
    (a.output/'manifest.json').write_text(json.dumps(bundle.manifest,indent=2)+'\n')
    (a.output/'report.json').write_text(json.dumps(dict(preparation=contract,
        scope='deterministic preparation only; no checkpoint or policy inference',
        mujoco_version=mujoco.__version__,model_sha256=digest(a.output/'model.mjb'),
        trace_sha256=digest(a.output/'preparation.npz')),indent=2)+'\n')
    print(json.dumps(contract['report'],indent=2),flush=True)
    if failed:raise failed

if __name__=='__main__':main()
