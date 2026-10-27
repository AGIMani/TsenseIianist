#!/usr/bin/env python3
"""Explicit user-authorized node4 single-GPU jobs; Slurm remains the default."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'}

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def authorize(project):
    cfg = json.loads((project/'process_backend.json').read_text())
    if (cfg.get('authorized') is not True or cfg.get('hostname') != socket.gethostname()
            or str(project) not in cfg.get('projects', []) or cfg.get('gpu') != 0):
        raise RuntimeError('No matching explicit process-backend authorization')
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Process backend must not nest inside Slurm')
    return cfg

def status(path):
    d=json.loads(path.read_text())
    if d['state'] not in TERMINAL and d.get('pid'):
        try:
            os.kill(d['pid'],0)
            if Path(f'/proc/{d["pid"]}/stat').read_text().split()[2]=='Z': raise ProcessLookupError()
        except (ProcessLookupError,FileNotFoundError):
            d.update(state='FAILED',exit_code='1:0',error='Background worker exited without terminal status')
    return d

def work(project, job):
    authorize(project)
    folder=project/'logs/hpc'/job; request=json.loads((folder/'request.json').read_text())
    state=folder/'process_state.json'
    d=dict(job_id=job,backend='process',state='PENDING',pid=os.getpid(),exit_code=None,reason='dependency',created_at=request['created_at'])
    save(state,d)
    registry=Path.home()/'.local/state/pianist-process-jobs';registry.mkdir(parents=True,exist_ok=True)
    child=None
    def stopped(sig,frame):
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
        d.update(state='CANCELLED',exit_code=f'0:{sig}',ended_at=time.time());save(state,d)
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        for kind,dep in request['dependencies']:
            dep_path=Path(json.loads((registry/f'{dep}.json').read_text())['state_path'])
            while True:
                other=status(dep_path)
                if other['state'] in TERMINAL: break
                time.sleep(5)
            if kind=='afterok' and other['state']!='COMPLETED':
                d.update(state='CANCELLED',exit_code='1:0',reason='dependency_failed');save(state,d);return
        d['reason']='gpu_lock';save(state,d)
        with (registry/'gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            # Respect unrelated compute workloads; no killing or oversubscription.
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                d['reason']='external_gpu_process';save(state,d);time.sleep(30)
            d.update(state='RUNNING',reason=None,started_at=time.time());save(state,d)
            env=os.environ.copy();env.update(TSENSEI_JOB_BACKEND='process',PIANIST_PROCESS_JOB_ID=job,CUDA_VISIBLE_DEVICES='0',TSENSEI_EXPECTED_GPUS='1')
            cmd=['bash',str(project/'scripts/hpc/job.sbatch'),str(project),request['mode'],*request['arguments']]
            child=subprocess.Popen(cmd,cwd=project,env=env,start_new_session=True)
            try:
                rc=child.wait(timeout=1200 if request['mode']=='validate' else 86400)
                d.update(state='COMPLETED' if rc==0 else 'FAILED',exit_code=f'{rc}:0')
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                d.update(state='TIMEOUT',exit_code='1:0')
            d['ended_at']=time.time();save(state,d)
    except Exception as exc:
        d.update(state='FAILED',exit_code='1:0',error=repr(exc),ended_at=time.time());save(state,d);raise

def main():
    if sys.argv[1]=='worker':
        work(Path(sys.argv[2]).resolve(),sys.argv[3]);return
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--gpus',type=int,required=True);p.add_argument('--dependency',action='append',default=[]);p.add_argument('--kill-on-invalid-dep',action='store_true');p.add_argument('mode');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    project=a.project.resolve();authorize(project)
    if a.gpus!=1 or a.mode not in {'validate','train','export','replay','mjlab-preview'}:p.error('Authorized backend supports single-GPU jobs only')
    dependencies=[]
    for expr in a.dependency:
        kind,*ids=expr.split(':')
        if kind not in {'afterok','afterany'} or not ids or not all(i.isdigit() for i in ids):p.error('Unsupported dependency')
        dependencies.extend((kind,i) for i in ids)
    job=str(time.time_ns());folder=project/'logs/hpc'/job;folder.mkdir(parents=True)
    save(folder/'request.json',dict(mode=a.mode,arguments=a.arguments,dependencies=dependencies,created_at=time.time()))
    save(folder/'process_state.json',dict(job_id=job,state='PENDING',pid=None,exit_code=None,backend='process'))
    registry=Path.home()/'.local/state/pianist-process-jobs';registry.mkdir(parents=True,exist_ok=True)
    save(registry/f'{job}.json',dict(state_path=str(folder/'process_state.json')))
    with (folder.parent/f'{job}.out').open('a') as out,(folder.parent/f'{job}.err').open('a') as err:
        child=subprocess.Popen([sys.executable,__file__,'worker',str(project),job],stdout=out,stderr=err,start_new_session=True)
    # The worker will replace this state; save PID separately to avoid races.
    (folder/'worker.pid').write_text(str(child.pid)+'\n')
    print(job,flush=True)

if __name__=='__main__':main()
