"""Slurm acceptance for the requested fresh Prelude mounting and actuator ranges."""
import math
from pathlib import Path
import numpy as np
import torch
from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle


def test_yaw30_revised_score_and_current_physical_mapping():
    root=Path(__file__).resolve().parents[1]
    recipe=make_recipe(yaw=math.pi/6, midi=str(root/'training_mjlab/inputs/prelude_bwv847/candidate_fingering/prelude-bwv-847_fingered.proto'),
        initial_buffer_time=0,release_tail_time=.25,score_validation='strict_v1',exact_note_timing=True,
        task_profile='tsensei_collision_v1',collision_profile='convex_hull_v1',drive_profile='shadow_v1',
        preparation_time=1,preparation_lift=.025,preparation_mode='policy_v1')
    assert recipe['external_score']['sha256']=='ad58e67de1998c218e380b7370c95d64ba23fe80d6dfbe26f415184e1a80b33a'
    scene_path,task_path,_=build(recipe)
    scene_path,task_path=without_arms(scene_path,task_path,scene_path.parent/'hand_only')
    scene=PianoScene(scene_path,action_profile='direct_joint_v1',disable_wrist_tz=True,disable_wrist_yaw=True)
    TaskBundle(task_path,scene.bundle)
    for side in ('right','left'):
        assert abs(scene.bundle.fixed_joint_values[side+'_/forearm_yaw']-math.pi/6)<1e-10
        assert side+'_/forearm_yaw' not in [scene.model.joint(i).name for i in range(scene.model.njnt)]
    p0=scene.physical_action(scene.action_low[None]);p1=scene.physical_action(scene.action_high[None])
    for index in (4,8,12,16,29,33,37,41):
        assert abs(p0[0,index].item())<1e-6
        assert abs(p1[0,index].item()-1.3)<1e-6
    for index in (23,48):
        assert abs(p0[0,index].item())<1e-6
        assert abs(p1[0,index].item()-.3)<1e-6
    scene.step((scene.action_low[None]+scene.action_high[None])*.5)
    assert torch.isfinite(scene.data.qpos).all()
    print('PRELUDE_CONTRACT yaw=pi/6 fixed structurally; revised_score_sha256='+recipe['external_score']['sha256']+'; pitch=[0,.3]; eight MCP=[0,1.3]; GPU step finite')
