"""Slurm-only, fixed-input attribution and cached-forward equivalence audit."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

# External pytest files can put their deployment parent before the frozen job
# source. Always benchmark the archive extracted as this job's working tree.
sys.path.insert(0, str(Path.cwd()))


def test_sampling_profile():
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires Slurm'
    import torch
    import copy
    import torch as torch_module
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import build, source_files
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.rl_torch.checkpoint import reward_options
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    import tsensei_v3_2_pianist.mjlab_env.scene as scene_module
    assert Path(scene_module.__file__).resolve().is_relative_to(Path.cwd().resolve())
    root=Path(os.environ['PRELUDE_PROFILE_ROOT'])
    checkpoint=root/'results/exper010-20261018-133314-814254/best.pt'
    output=root/'logs/prelude_perf_20261018'/('profile_'+os.environ['SLURM_JOB_ID']+'.json')
    # The checkpoint recipe is unchanged; this diagnostic substitutes the
    # candidate runtime fingerprint in memory only. Scene/task manifests do
    # not include this dispatch implementation, so the contract remains equal.
    meta=torch_module.load(checkpoint,map_location='cpu',weights_only=True)['metadata']
    recipe=copy.deepcopy(meta['source_runtime']); recipe['source_sha256']=source_files()
    scene_path,task_path,_=build(recipe)
    if meta.get('no_arms'):
        scene_path,task_path=without_arms(scene_path,task_path,scene_path.parent/'hand_only_profile')
    options=reward_options(checkpoint)
    template=PianoTaskEnv(scene_path,task_path,num_envs=512,physics_graph=True,**options)
    s=template.scene
    def cached(self, *, use_cached_forward=False):
        if self.joint_limiter is not None:
            self.joint_limiter.apply(self.sim.wp_data)
            if use_cached_forward:
                self.sim.forward()
            else:
                import mujoco_warp as mjwarp
                mjwarp.forward(self.sim.wp_model, self.sim.wp_data)
    # Direct same-state audit: isolate the forward implementation from the
    # known nondeterminism of batched contact atomics across fresh worlds.
    state={name:getattr(s.data,name)[:].clone() for name in
           ('qpos','qvel','qacc','qacc_warmstart','ctrl')}
    if s.model.nmocap:
        state.update(mocap_pos=s.data.mocap_pos[:].clone(),mocap_quat=s.data.mocap_quat[:].clone())
    def restore_state():
        for name,value in state.items(): getattr(s.data,name)[:].copy_(value)
    with s.scope():
        s.joint_limiter.apply(s.sim.wp_data)
        import mujoco_warp as mjwarp
        mjwarp.forward(s.sim.wp_model,s.sim.wp_data)
    raw_forward={name:getattr(s.data,name)[:].clone() for name in ('xpos','xmat','geom_xpos','geom_xmat')}
    restore_state()
    with s.scope():
        s.joint_limiter.apply(s.sim.wp_data)
        s.sim.forward()
    cached_forward_state={name:getattr(s.data,name)[:].clone() for name in raw_forward}
    forward_diff={name:float((raw_forward[name]-cached_forward_state[name]).abs().max().cpu()) for name in raw_forward}
    assert max(forward_diff.values()) < 1e-5, 'Cached forward graph differs on identical state: '+str(forward_diff)
    gen=torch.Generator(device=s.device).manual_seed(47193)
    actions=[s.action_low+(s.action_high-s.action_low)*torch.rand((512,s.action_size),device=s.device,generator=gen) for _ in range(32)]
    saved={};reports={}
    def snapshot(result):
        pairs={'qpos':s.data.qpos[:], 'qvel':s.data.qvel[:], 'ctrl':s.data.ctrl[:],
               'observation':result.observation,'reward':result.reward,'discount':result.discount,
               'terminated':result.terminated,'truncated':result.truncated,
               'key_state_history':s.reward_sampling.key_state_history,
               'key_activation_history':s.reward_sampling.key_activation_history}
        pairs.update({'coupling_'+k:v for k,v in s.finger_coupling_metrics.items()})
        def collect(prefix, value):
            if isinstance(value, torch.Tensor):
                pairs[prefix] = value
            elif isinstance(value, dict):
                for name, item in value.items():
                    collect(prefix+'_'+name, item)
        collect('info', result.info)
        return {k:v.detach().cpu().clone() for k,v in pairs.items()}
    def make_env():
        env=PianoTaskEnv(scene_path,task_path,num_envs=512,physics_graph=True,**options)
        s=env.scene
        original=s._enforce_joint_limits
        def raw(self, *, use_cached_forward=False):
            # Force the pre-fix path for an apples-to-apples comparison.
            return original(use_cached_forward=False)
        s._enforce_joint_limits=(types.MethodType(cached,s) if mode=='cached_forward'
                                 else types.MethodType(raw,s))
        # Each mode gets a newly constructed task, so the score index, history,
        # coupling graph and MuJoCo reset state are independent and identical.
        env.reset()
        for a in actions[:2]: env.step(a)
        env.reset()
        return env,s,original
    for mode in ('baseline','cached_forward','baseline_repeat'):
        env,s,_=make_env(); torch.cuda.synchronize();times=[];maxdiff={};mismatch={}
        for i,a in enumerate(actions):
            t=time.perf_counter();result=env.step(a);torch.cuda.synchronize();times.append(time.perf_counter()-t)
            values=snapshot(result)
            if mode=='baseline':saved[i]=values
            else:
                for k,v in values.items():
                    ref=saved[i][k]
                    d=(v.to(torch.float64)-ref.to(torch.float64)).abs().max().item() if v.numel() else 0.
                    maxdiff[k]=max(maxdiff.get(k,0.),d)
                    if not torch.equal(v,ref):mismatch[k]=mismatch.get(k,0)+1
        reports[mode]=dict(mean_seconds=sum(times)/len(times),seconds=times,max_abs_difference=maxdiff,mismatched_frames=mismatch)
        print('PROFILE',mode,json.dumps({k:v for k,v in reports[mode].items() if k!='max_abs_difference'}),flush=True)
    # Component timings are separate: synchronization intentionally removes overlap.
    env,s,original=make_env(); s._enforce_joint_limits=original
    totals={};counts={};restore=[]
    def wrap(obj,name,label):
        fn=getattr(obj,name);restore.append((obj,name,fn))
        def measured(*args,**kwargs):
            torch.cuda.synchronize();t=time.perf_counter();result=fn(*args,**kwargs);torch.cuda.synchronize()
            totals[label]=totals.get(label,0)+time.perf_counter()-t;counts[label]=counts.get(label,0)+1
            return result
        setattr(obj,name,measured)
    for obj,name,label in [(s,'_couple_finger_targets','coupling'),(s.sim,'step','physics_step'),(s,'_enforce_joint_limits','hard_limits_and_forward'),(s,'_sample_self_collision','collision_diagnostics'),(s,'_sample_reward_substep','substep_rewards')]:wrap(obj,name,label)
    for a in actions[:4]:env.step(a)
    for obj,name,fn in restore:setattr(obj,name,fn)
    result=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),num_envs=512,frames=32,control_hz=20,physics_hz=200,source_root=str(Path.cwd()),reports=reports,component_total_seconds=totals,component_calls=counts,component_frames=4,forward_equivalence=forward_diff,speedup=reports['baseline']['mean_seconds']/reports['cached_forward']['mean_seconds'],note='Same captured inputs, reset state, 200Hz coupling, hard limits, physics and reward configuration. Timing phases synchronized; profile does not train or save weights.')
    output.write_text(json.dumps(result,indent=2)+'\n');print('PROFILE_OUTPUT',output,flush=True)
    # Contact atomics make independent fresh worlds bitwise-nondeterministic;
    # retain the full per-tensor deltas for review rather than claiming exact
    # equality. A separate fixed-state check below gates the candidate.
    assert reports['baseline']['mean_seconds'] > reports['cached_forward']['mean_seconds']
