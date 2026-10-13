#!/usr/bin/env python3
"""CPU-only draft: local target trust region plus sampled connected prefix.

The production guard is not modified. This tests a counterexample set, not
physics safety, learned performance or a permission to deploy this candidate.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    import numpy as np
    import torch
    from tests.test_finger_clearance import RealGeometryTests
    RealGeometryTests.setUpClass();g=RealGeometryTests.guard;b=RealGeometryTests.bundle
    n=64
    q=torch.tensor(b.origin_qpos,dtype=torch.float32).expand(n,-1).clone()
    mocap=torch.tensor(b.origin_mocap_pos,dtype=torch.float32).expand(n,-1,-1).clone()
    physical=g.physical_qpos(q);role=torch.zeros(n,10,88,dtype=torch.bool)
    generator=torch.Generator().manual_seed(599);ends=[]
    for _ in range(2):
        requested=physical.clone()
        requested[:,g.finger_indices]=g.low+(g.high-g.low)*torch.rand(n,10,4,generator=generator)
        applied,_=g.filter(requested,actual_qpos=q,mocap_pos=mocap,assigned_keys=role)
        ends.append(applied)
    start,desired=ends
    trust=.2
    candidate=desired.clone()
    candidate[:,g.finger_indices]=start[:,g.finger_indices]+(desired[:,g.finger_indices]-start[:,g.finger_indices]).clamp(-trust,trust)
    def deficits(values):
        shape=values.shape[:-1]
        flat=values.reshape(-1,51)
        result=g.evaluate(flat,torch.zeros(len(flat),2),torch.zeros(len(flat),10,88,dtype=torch.bool))['finger_clearance_deficit_m']
        return result.reshape(*shape,10)
    def sampled(a,z,fractions):
        values=a[None]+fractions[:,None,None]*(z-a)[None]
        return deficits(values)
    alphas=torch.tensor([1.,.5,.25,.125,.0625,0.])
    fractions=torch.arange(1,9)/8
    all_candidates=start[None]+alphas[:,None,None]*(candidate-start)[None]
    paths=start[None,None]+fractions[:,None,None,None]*(all_candidates-start[None])[None]
    violations=deficits(paths).amax(0)
    valid=violations<=1.01e-6
    choice=valid.to(torch.int64).argmax(0)
    chosen_alpha=alphas[choice]
    applied=candidate.clone()
    applied[:,g.finger_indices]=start[:,g.finger_indices]+chosen_alpha[...,None]*(candidate[:,g.finger_indices]-start[:,g.finger_indices])
    dense_fractions=torch.arange(1,65)/64
    old_dense=sampled(start,desired,dense_fractions)
    new_dense=sampled(start,applied,dense_fractions)
    # Local continuous recovery from a currently deficient actual pose. Accept
    # only steps whose sampled maximum deficit does not worsen the initial one.
    unsafe=physical.clone()
    unsafe[:,g.finger_indices]=g.low+(g.high-g.low)*torch.rand(n,10,4,generator=generator)
    before=deficits(unsafe)
    recovery=unsafe.clone()
    for _ in range(6):
        metrics=g._evaluate(recovery,torch.zeros(n,2),role,jacobian=True)
        grad=metrics['gradient'];depth=metrics['finger_clearance_deficit_m']
        increment=depth[...,None]*grad/(grad.square().sum(-1,keepdim=True)+1e-10)
        angles=(recovery[:,g.finger_indices]+increment.clamp(-.1,.1)).clamp(g.low,g.high)
        angles=torch.maximum(unsafe[:,g.finger_indices]-trust,torch.minimum(angles,unsafe[:,g.finger_indices]+trust))
        recovery[:,g.finger_indices]=angles
    recovery_candidates=unsafe[None]+alphas[:,None,None]*(recovery-unsafe)[None]
    recovery_paths=unsafe[None,None]+fractions[:,None,None,None]*(recovery_candidates-unsafe[None])[None]
    recovery_bad=deficits(recovery_paths)
    recovery_valid=(recovery_bad<=before[None,None]+1e-6).all(0)
    rchoice=recovery_valid.to(torch.int64).argmax(0)
    ralpha=alphas[rchoice]
    recovered=recovery.clone()
    recovered[:,g.finger_indices]=unsafe[:,g.finger_indices]+ralpha[...,None]*(recovery[:,g.finger_indices]-unsafe[:,g.finger_indices])
    after=deficits(recovered);mask=before>1e-6
    recovered_path=sampled(unsafe,recovered,dense_fractions)
    report=dict(trust_region_rad=trust, sampled_fractions=fractions.tolist(), prefix_choices=alphas.tolist(),
        verification_fractions=64, paths=n, fingers=10,
        old_violating_finger_paths=int((old_dense.amax(0)>1e-6).sum()),old_max_deficit_m=float(old_dense.max()),
        new_violating_finger_paths=int((new_dense.amax(0)>1.1e-6).sum()),new_max_deficit_m=float(new_dense.max()),
        idle_fingers_held=int((chosen_alpha==0).sum()),mean_prefix=float(chosen_alpha.mean()),
        mean_max_joint_movement_rad=float((applied[:,g.finger_indices]-start[:,g.finger_indices]).abs().amax(-1).mean()),
        recovery_initial_bad_fingers=int(mask.sum()),recovery_improved_fingers=int(((after<before-1e-6)&mask).sum()),
        recovery_fully_clear_fingers=int(((after<=1e-6)&mask).sum()),
        recovery_new_dense_worsen_fingers=int(((recovered_path.amax(0)>before+1.1e-6)&mask).sum()),
        recovery_mean_before_mm=float(before[mask].mean()*1000),recovery_mean_after_mm=float(after[mask].mean()*1000),
        limitation='Idle-finger CPU FK draft; finite samples can miss geometry, held fingers can stall, actual dynamics and moving wrists are not certified')
    out=ROOT/'preview/outputs/diagnostics/finger_clearance_20260927/local_path_draft.json'
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
