"""Deterministically evaluate a Torch checkpoint in an independent GPU scene."""
import argparse
import json
from pathlib import Path

from .checkpoint import load_agent, reward_options
from .train import ROOT, evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--scene',type=Path,default=None)
    p.add_argument('--task',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--eval-num-envs',type=int,default=1,
                   help='Independent worlds per evaluation batch; default preserves serial evaluation')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    from ..mjlab_env.paths import for_checkpoint
    a.scene,a.task=for_checkpoint(a.checkpoint,a.scene,a.task)
    if a.episodes<1:p.error('--episodes must be positive')
    if a.eval_num_envs<1:p.error('--eval-num-envs must be positive')
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    options=reward_options(a.checkpoint)
    env=PianoTaskEnv(a.scene,a.task,device=a.device,num_envs=min(a.eval_num_envs,a.episodes),**options)
    agent=load_agent(a.checkpoint,scene_sha256=sha256(a.scene/'manifest.json'),task_sha256=sha256(a.task/'task.json'),
                     obs_size=env.observation_size,action_size=env.action_size,device=a.device,
                     action_contract=env.scene.action_contract,
                     control_limits_contract=env.scene.control_limits,
                     finger_clearance_contract=env.scene.finger_clearance,
                     finger_coupling_contract=env.scene.finger_coupling,
                     coupling_observation_contract=env.coupling_observation_contract,
                     arm_follow_contract=(env.scene.arm_follower.contract if env.scene.arm_follower is not None else None),
                     preparation_contract=env.scene.bundle.preparation,**options)
    result=dict(checkpoint=str(a.checkpoint),episodes=a.episodes,metrics=evaluate(agent,env,a.episodes),
                reward_options=options, control_limits_contract=env.scene.control_limits,
                finger_clearance_contract=env.scene.finger_clearance,
                note='Deterministic full-score GPU evaluation; no training or replay insertion')
    text=json.dumps(result,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text)
    print(text)

if __name__=='__main__':main()
