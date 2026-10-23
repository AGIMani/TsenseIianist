"""Compare the CPU realtime follower with a saved current-production IK trace.

Run in the remote Slurm allocation. No hardware commands or rendering.
Requires the video_report.json and ik_trace_200hz.npz from the same replay.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from tsensei_v3_2_pianist.mjlab_env.arm_follow import ArmFollower
from tsensei_v3_2_pianist.mjlab_env import arm_realtime as m

def stats(values):
    x=np.array(values)*1000
    return {k:float(v) for k,v in [('mean_ms',x.mean()),('p50_ms',np.percentile(x,50)),('p95_ms',np.percentile(x,95)),('p99_ms',np.percentile(x,99)),('max_ms',x.max()),('over_5ms',np.sum(x>5))]}

def benchmark(reference_directory, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    reference_directory = Path(reference_directory)
    torch.set_num_threads(1)
    report=json.loads((reference_directory/'video_report.json').read_text())
    x=np.load(reference_directory/'ik_trace_200hz.npz')
    sides=('right','left');o=report['controller']['options']
    poses={s:(x['palm_actual_m'][0,j],x['palm_actual_rotation'][0,j]) for j,s in enumerate(sides)}
    f=m.RealtimeArmFollower(o,poses)
    start=time.perf_counter();f.warmup();warmup=time.perf_counter()-start
    elapsed=[];q=[];v=[];acc=[];p=[]
    for k in range(1,len(x['time_s'])):
        targets={s:(x['palm_target_m'][k,j],x['palm_target_rotation'][k,j]) for j,s in enumerate(sides)}
        start=time.perf_counter();res=f.step(targets);elapsed.append(time.perf_counter()-start)
        q.append([f.sides[s]['q'].copy() for s in sides]);v.append([f.sides[s]['v'].copy() for s in sides]);acc.append([f.sides[s]['acceleration'].copy() for s in sides]);p.append([res[s][0] for s in sides])
    q,v,acc,p=map(np.array,(q,v,acc,p))
    delta={'q_rad':float(np.max(np.abs(q-x['q'][1:]))),'velocity_rad_s':float(np.max(np.abs(v-x['velocity'][1:]))),'acceleration_rad_s2':float(np.max(np.abs(acc-x['acceleration'][1:]))),'palm_m':float(np.max(np.abs(p-x['palm_actual_m'][1:])))}
    print('TIMING',stats(elapsed),'REFERENCE DELTA',delta,flush=True)
    checks={}
    for j,s in enumerate(sides):
        a=f.sides[s];c=o['coupling']
        assert np.all(q[:,j]>=a['low']-1e-12) and np.all(q[:,j]<=a['high']+1e-12)
        assert np.max(np.abs(v[:,j])-a['vmax'])<1e-8
        assert np.max(np.abs(acc[:,j])-a['amax'])<1e-7
        assert np.max(c['coefficient']*abs(q[:,j,5])+abs(q[:,j,6]))<c['sum_limit_rad']+1e-12
    checks['all_trace_hard_limits']=True
    # Accuracy does not sacrifice rate-limited execution to force exact IK.
    assert delta['q_rad']<1e-7 and delta['palm_m']<1e-8,delta
    assert delta['velocity_rad_s']<1e-5 and delta['acceleration_rad_s2']<.002,delta
    checks['full_trajectory_reference_agreement']=True
    # Independent random same-state comparisons include unreachable targets,
    # near-URDF-limit seeds, and near-coupling-boundary seeds at zero velocity.
    ref=ArmFollower(o,poses,num_envs=1,dt=.005,device='cpu')
    rng=np.random.default_rng(4700);worst=0.;accepted=0;rejected=0
    for case in range(36):
        f.reset();ref.reset(torch.tensor([0]))
        for s in sides:
            a=f.sides[s]
            seed=a['q0'].copy()
            if case%3==1:
                seed[case%5]=a['high'][case%5]-1e-5
            if case%3==2:
                seed[5]=.9;seed[6]=o['coupling']['sum_limit_rad']-1.025*.9-1e-8
            for key in ('q','reference'):
                a[key][:]=seed;ref.sides[s][key].copy_(torch.tensor(seed[None]))
        targets={s:(poses[s][0]+rng.uniform(-1,1,3)*(2 if case%4==0 else .03),poses[s][1]) for s in sides}
        tt={s:tuple(torch.tensor(t[None]) for t in pair) for s,pair in targets.items()}
        exceptions=[]
        for obj,tar in [(ref,tt),(f,targets)]:
            try:obj.step(tar);exceptions.append(None)
            except (RuntimeError,FloatingPointError) as e:exceptions.append(type(e).__name__)
        assert exceptions[0]==exceptions[1],(case,exceptions)
        if exceptions[0]:rejected+=1;continue
        accepted+=1
        for s in sides:
            for key,tol in [('q',1e-8),('v',1e-6),('acceleration',.0002),('reference',1e-7)]:
                d=float(np.max(abs(f.sides[s][key]-ref.sides[s][key][0].numpy())))
                assert d<tol,(case,s,key,d)
                if key=='q':worst=max(worst,d)
    checks['random_reference_cases']={'accepted':accepted,'both_rejected':rejected,'worst_q_rad':worst}
    # Direct projection test exercises the active coupling branch.
    for _ in range(100):
        a=rng.uniform(-.5,.5,(1,7));b=rng.uniform(-2,2,(1,7))
        c=o['coupling'];expected=ref._coupled_segment(torch.tensor(a),torch.tensor(b))[0].numpy()
        actual=m._segment(a[0],b[0],5,6,c['coefficient'],c['sum_limit_rad'])
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    checks['coupling_projection_100_cases']=True
    # NaN in second arm must not partially commit the first arm.
    before={s:f.sides[s]['q'].copy() for s in sides}
    bad={s:(poses[s][0].copy(),poses[s][1]) for s in sides};bad['left'][0][0]=np.nan
    try:f.step(bad);raise AssertionError('accepted NaN')
    except ValueError:pass
    for s in sides:np.testing.assert_array_equal(f.sides[s]['q'],before[s])
    checks['atomic_failure']=True
    # Paced 200Hz loop including target assembly and a bounded memory logger.
    f.reset();dt=.005;compute=[];finish_lateness=[];starts=[]
    start=time.perf_counter()+.1
    for k in range(1,len(x['time_s'])):
        deadline=start+(k-1)*dt
        remaining=deadline-time.perf_counter()
        if remaining>0:time.sleep(remaining)
        begin=time.perf_counter();starts.append(begin-deadline)
        targets={s:(x['palm_target_m'][k,j],x['palm_target_rotation'][k,j]) for j,s in enumerate(sides)}
        f.step(targets)
        q[k-1]=[f.sides[s]['q'] for s in sides]
        finish=time.perf_counter();compute.append(finish-begin);finish_lateness.append(max(0.,finish-(deadline+dt)))
    results={'job_id':os.environ.get('SLURM_JOB_ID'),'warmup_seconds':warmup,'unpaced':stats(elapsed),'paced_compute':stats(compute),'paced_start_lateness':stats(starts),'paced_finish_deadline_misses':int(np.count_nonzero(finish_lateness)),'paced_worst_finish_late_ms':float(max(finish_lateness)*1000),'reference_delta':delta,'checks':checks,'numpy':np.__version__,'numba':__import__('numba').__version__,'scope':'single dual arm CPU, no SDK/network/render; 12 iterations, 5 candidates, all hard checks'}
    np.savez_compressed(out/'optimized_trace.npz',q=q,velocity=v,acceleration=acc,palm_actual_m=p,compute_seconds=elapsed,paced_compute_seconds=compute,paced_start_lateness_seconds=starts)
    (out/'optimized.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',json.dumps(results),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.reference_directory, args.output)


if __name__ == '__main__':
    main()
