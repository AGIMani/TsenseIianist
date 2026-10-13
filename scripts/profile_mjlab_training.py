#!/usr/bin/env python3
"""Bounded training component timing in an isolated learner; never saves weights."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
from tsensei_v3_2_pianist.rl_torch.checkpoint import load_agent
from tsensei_v3_2_pianist.rl_torch.replay import Replay,Transition


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=12)
 p.add_argument('--physics-graph',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--compile-learner',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--fused-optimizer',action=argparse.BooleanOptionalAction,default=True)
 a=p.parse_args()
 if a.output.exists():p.error('Output exists')
 if not 1<=a.iterations<=120:p.error('iterations must be in [1,120] for this bounded benchmark')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from tsensei_v3_2_pianist.mjlab_env.paths import for_checkpoint
 scene,task=for_checkpoint(a.checkpoint)
 from tsensei_v3_2_pianist.rl_torch.checkpoint import reward_options
 options=reward_options(a.checkpoint)
 env=PianoTaskEnv(scene,task,num_envs=256,physics_graph=a.physics_graph,**options);agent=load_agent(a.checkpoint,scene_sha256=sha256(scene/'manifest.json'),task_sha256=sha256(task/'task.json'),obs_size=env.observation_size,action_size=env.action_size,action_contract=env.scene.action_contract,compile_learner=a.compile_learner,fused_optimizer=a.fused_optimizer,preparation_contract=env.scene.bundle.preparation,control_limits_contract=env.scene.control_limits,finger_clearance_contract=env.scene.finger_clearance,finger_coupling_contract=env.scene.finger_coupling,coupling_observation_contract=env.coupling_observation_contract,**options)
 replay=Replay(env.observation_size,env.action_size,262144,4096,action_profile=options.get("action_profile","legacy"));obs=env.reset();batches=[]
 for _ in range(16):
  action=agent.act(obs);result=env.step(action);batch=Transition(obs,action,result.reward,result.discount,result.observation,result.terminated,result.truncated);batches.append(batch);replay.insert(batch);obs=result.observation
 # Full-size replay, duplicated valid warmup transitions. This is a timing workload,
 # not a learning-quality experiment or reconstruction of the old replay contents.
 for i in range(1024-16):replay.insert(batches[i%16])
 torch.cuda.synchronize();compile_start=time.perf_counter()
 for _ in range(32):agent.update(replay.sample())
 torch.cuda.synchronize();learner_warmup_s=time.perf_counter()-compile_start
 def timed(fn):
  torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize();return out,time.perf_counter()-start
 rows=[];physics_times=[]
 original_step=env.scene.step
 def measured_physics(action, **kwargs):
  out,elapsed=timed(lambda:original_step(action, **kwargs));physics_times.append(elapsed);return out
 env.scene.step=measured_physics
 for i in range(a.iterations):
  action,actor=timed(lambda:agent.act(obs));result,environment=timed(lambda:env.step(action))
  _,insert=timed(lambda:replay.insert(Transition(obs,action,result.reward,result.discount,result.observation,result.terminated,result.truncated)))
  def learn():
   for _ in range(32):agent.update(replay.sample())
  _,updates=timed(learn);obs=result.observation
  rows.append(dict(actor=actor,physics=physics_times[-1],task_other=environment-physics_times[-1],replay_insert=insert,updates32_including_replay_sampling=updates))
  print('iteration',i+1,rows[-1],flush=True)
 avg={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]};total=sum(avg.values())
 report=dict(device=torch.cuda.get_device_name(),num_envs=256,batch=4096,updates_per_vector_step=32,substeps_per_action=env.scene.decimation,iterations=a.iterations,physics_single_step_cuda_graph=env.scene.sim.use_cuda_graph,physics_graph=a.physics_graph,compile_learner=a.compile_learner,fused_optimizer=a.fused_optimizer,learner_warmup_s=learner_warmup_s,mean_seconds=avg,fraction={k:v/total for k,v in avg.items()},mean_vector_seconds=total,estimated_transitions_per_second=256/total,rows=rows,
             note='Separate isolated learner; TF32 off; full-size replay filled with 16 repeated valid warmup batches. Timings synchronize GPU between phases and exclude initialization, evaluation and file saves. Not an exact attribution of historical runtime.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
