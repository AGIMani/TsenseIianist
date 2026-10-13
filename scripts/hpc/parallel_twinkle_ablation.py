#!/usr/bin/env python3
"""Adopt running C without restarting it; launch pending ablations on free pairs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_speed_comparison import now, write
from run_twinkle_ablation import ROOT, overrides, summarize, validate_ready_config


def free_allocations(stages, allocations):
    busy={s['allocation'] for s in stages if s['status']=='running'}
    return [a for a in allocations if a not in busy]


def dispatchable(stages):
    return [] if any(s['status']=='failed' for s in stages) else [s for s in stages if s['status']=='pending']


def numeric_steps(allocation):
    rows=subprocess.check_output(['squeue','-h','--steps','-j',str(allocation),'-o','%i'],text=True).split()
    return [r for r in rows if r.startswith(str(allocation)+'.') and r.split('.')[-1].isdigit()]


def verify_allocation(allocation):
    value=subprocess.check_output(['squeue','-h','-j',str(allocation),'-o','%u %T %j'],text=True).strip()
    if value!=os.environ['USER']+' RUNNING TsenseI-reserve2':
        raise RuntimeError('Allocation is not an owned running reserve: '+value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--extra-allocation',required=True,type=int)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();a.runtime=a.runtime.resolve();output=a.runtime.parent
    if a.detach:
        with (output/'parallel_coordinator.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                *[x for x in sys.argv[1:] if x!='--detach']],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,runtime=str(a.runtime))));return
    lock=(output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(a.runtime.read_text())
    assert state['status']=='running' and 'parallel_adopted_at' not in state
    stages=state['stages'];assert [s['status'] for s in stages]==['running','pending','pending']
    old_pid=state['pid'];assert not Path(f'/proc/{old_pid}').exists(),'Old coordinator still alive'
    allocations=[state['allocation'],a.extra_allocation];assert len(set(allocations))==2
    for allocation in allocations:verify_allocation(allocation)
    assert not numeric_steps(a.extra_allocation)
    c=stages[0];c['allocation']=state['allocation'];c['adopted']=True
    c['step_ids']=[x for x in c['slurm_steps'] if x.split('.')[-1].isdigit()]
    assert len(c['step_ids'])==1 and numeric_steps(c['allocation'])==c['step_ids']
    archive=Path(state['archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==state['archive_sha256']
    bp=Path(state['baseline_run'])/'config.json'
    assert hashlib.sha256(bp.read_bytes()).hexdigest()==state['baseline_config_sha256']
    baseline=json.loads(bp.read_text())
    state.update(previous_coordinator_pid=old_pid,pid=os.getpid(),parallel_adopted_at=now(),
        allocations=allocations,scheduling='C continues; D on extra pair; E on first available completed pair')
    children={};logs={}
    write(a.runtime,state)
    while True:
        # Complete or fail existing work before dispatching E. Never cancel an
        # adopted/running experiment because an independent experiment fails.
        for st in [s for s in stages if s['status']=='running']:
            label=st['label'];steps=numeric_steps(st['allocation'])
            if steps:st['step_ids']=steps
            st['slurm_steps']=steps
            runs=[r for r in output.glob(label+'-*') if (r/'config.json').exists()]
            st['output_dirs']=[str(r) for r in runs]
            if len(runs)>1:raise RuntimeError('Duplicate experiment output: '+label)
            if runs and not st.get('config_verified'):
                try:config=json.loads((runs[0]/'config.json').read_text())
                except json.JSONDecodeError:pass
                else:
                    try:st['config_verified']=validate_ready_config(config,baseline,st['anticipation'],st['false_press'])
                    except Exception as error:
                        for step in steps:subprocess.run(['scancel',step],check=False)
                        st.update(status='failed',error=repr(error),finished_at=now())
                        continue
            child=children.get(label)
            done=(child.poll() is not None) if child else not steps
            if not done:continue
            if child:
                st['exit_code']=child.returncode
                logs.pop(label).close()
            try:
                if child and child.returncode:raise RuntimeError(f'Slurm child exited {child.returncode}')
                if not runs or not (runs[0]/'summary.json').exists():raise RuntimeError('Step ended without a complete summary')
                st['result']=summarize(runs[0],baseline,st['anticipation'],st['false_press'])
                st.update(status='completed',finished_at=now())
            except Exception as error:st.update(status='failed',error=repr(error),finished_at=now())
        for st,allocation in zip(dispatchable(stages),free_allocations(stages,allocations)):
            verify_allocation(allocation)
            assert not numeric_steps(allocation)
            assert hashlib.sha256(archive.read_bytes()).hexdigest()==state['archive_sha256']
            command=['bash',str(ROOT/'scripts/hpc/submit.sh'),'--allocation',str(allocation),
                '--source-archive',str(archive),'train','--seed','42','--vector-steps','19532',
                '--output-dir',str(output/st['label']),*overrides(st['anticipation'],st['false_press'])]
            st.update(status='running',allocation=allocation,started_at=now(),command=command)
            write(a.runtime,state)
            log=(output/(st['label']+'.submit.log')).open('w');logs[st['label']]=log
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children[st['label']]=child;st['submit_pid']=child.pid
        running=[s['label'] for s in stages if s['status']=='running']
        state.update(updated_at=now(),active_stages=running,active_stage=running[0] if len(running)==1 else None)
        if not running:
            state['status']='failed' if any(s['status']=='failed' for s in stages) else 'completed'
            write(a.runtime,state);break
        write(a.runtime,state);time.sleep(5)


if __name__=='__main__':main()
