#!/usr/bin/env python3
"""Run a pinned, gated experiment manifest inside an existing Slurm allocation.

This foreground controller only queries Slurm and invokes submit.sh --allocation.
A caller may detach it with Popen(start_new_session=True), redirecting all streams.
It never cancels the allocation, submits a new reservation, or runs GPU code itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
WAITING_STATES = {'PENDING', 'CONFIGURING', 'REQUEUED', 'RESIZING', 'SUSPENDED'}
MODES = {'validate', 'strike-diagnostic', 'train', 'replay', 'probe', 'benchmark', 'performance'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_manifest(path, plan):
    """Same-directory atomic replacement keeps readers away from partial JSON."""
    plan['updated_at'] = now()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(plan, handle, indent=2, ensure_ascii=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def query_allocation(allocation_id):
    """Return owner/state/name, falling back to scontrol if absent from squeue."""
    result = subprocess.run(['squeue', '-h', '-j', str(allocation_id), '-o', '%u|%T|%j'],
                            text=True, capture_output=True, timeout=30)
    text = result.stdout.strip()
    if result.returncode == 0 and text:
        lines = text.splitlines()
        if len(lines) != 1 or len(lines[0].split('|')) != 3:
            raise RuntimeError('Unexpected squeue allocation response: '+text[:1000])
        owner,state,name = lines[0].split('|')
        return dict(owner=owner.strip(), state=state.strip(), name=name.strip(), source='squeue')
    fallback = subprocess.run(['scontrol', 'show', 'job', str(allocation_id), '--oneliner'],
                              text=True, capture_output=True, timeout=30)
    values = dict(re.findall(r'(\w+)=(\S+)', fallback.stdout))
    if fallback.returncode == 0 and {'UserId','JobState','JobName'} <= values.keys():
        return dict(owner=values['UserId'].split('(')[0], state=values['JobState'],
                    name=values['JobName'], source='scontrol')
    return dict(owner=None, state='MISSING', name=None, source='scontrol',
                query_error=(result.stderr+'\n'+fallback.stderr).strip()[-2000:])


def wait_for_allocation(allocation_id, *, username, poll_seconds, missing_polls, on_state,
                        sleep=time.sleep):
    missing = 0
    while True:
        state = query_allocation(allocation_id)
        on_state(state)
        if state['state'] == 'MISSING':
            missing += 1
            if missing >= missing_polls:
                raise RuntimeError(f'Allocation {allocation_id} is missing after {missing} scheduler checks')
        else:
            missing = 0
            if state['owner'] != username or state['name'] != 'TsenseI-reserve2':
                raise RuntimeError(f'Allocation {allocation_id} must belong to {username} and be named TsenseI-reserve2')
            if state['state'] == 'RUNNING':
                return state
            if state['state'] not in WAITING_STATES:
                raise RuntimeError(f'Allocation {allocation_id} is {state["state"]}; no further steps will run')
        sleep(poll_seconds)


def validate_plan(plan, project_root):
    allocation = str(plan.get('allocation_id', ''))
    if not allocation.isdigit() or int(allocation) <= 0:
        raise ValueError('Manifest requires a positive allocation_id')
    archive = Path(plan['source_archive'])
    if not archive.is_absolute() or not archive.is_file():
        raise ValueError('Manifest source_archive must be an existing absolute path')
    expected = plan.get('source', {}).get('sha256')
    if not expected or not re.fullmatch('[a-fA-F0-9]{64}', expected):
        raise ValueError('Manifest requires source.sha256 from the immutable package')
    if digest(archive) != expected.lower():
        raise ValueError('Pinned source archive checksum changed')
    if not (project_root/'scripts/hpc/submit.sh').is_file():
        raise ValueError('Project root has no scripts/hpc/submit.sh')
    jobs = plan.get('jobs')
    if not isinstance(jobs,list) or not jobs:
        raise ValueError('Manifest requires a nonempty jobs list')
    labels = []
    for job in jobs:
        label = job.get('label')
        if not isinstance(label,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',label):
            raise ValueError('Job label must be a safe unique filename')
        labels.append(label)
        if job.get('mode') not in MODES:
            raise ValueError('Unsupported allocation job mode: '+str(job.get('mode')))
        if not isinstance(job.get('arguments'),list) or not all(isinstance(x,str) for x in job['arguments']):
            raise ValueError('Job arguments must be a list of strings')
        if job.get('status') not in (None,'planned','queued','waiting'):
            raise ValueError('Refusing to rerun a previously started/completed/failed job: '+label)
        if job.get('started_at') or job.get('returncode') is not None:
            raise ValueError('Refusing to rerun a job with an execution record: '+label)
    if len(labels) != len(set(labels)):
        raise ValueError('Job labels must be unique')
    return allocation,archive,expected.lower()


def run_queue(manifest, *, project_root, poll_seconds=30., missing_polls=3):
    """Hold a lifetime lock; every following step depends on prior exit code zero."""
    manifest = Path(manifest).resolve()
    project_root = Path(project_root).resolve()
    lock_path = manifest.with_suffix(manifest.suffix+'.controller.lock')
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another controller already holds '+str(lock_path)) from exc
        plan = json.loads(manifest.read_text())
        allocation,archive,expected = validate_plan(plan,project_root)
        username = pwd.getpwuid(os.getuid()).pw_name
        controller = plan['controller'] = dict(pid=os.getpid(),status='waiting_for_allocation',
            started_at=now(),project_root=str(project_root),allocation_id=allocation,
            source_sha256=expected,poll_seconds=poll_seconds,lock=str(lock_path),
            allocation_release='Preserved; this controller never cancels the reservation')
        logs = manifest.parent/'controller_steps'
        logs.mkdir(exist_ok=True)
        for job in plan['jobs']:
            job['status']='waiting'
        write_manifest(manifest,plan)
        previous_state = None

        def record_state(state):
            nonlocal previous_state
            controller['allocation_state']=state
            controller['last_scheduler_check_at']=now()
            write_manifest(manifest,plan)
            if state != previous_state:
                print(json.dumps(dict(time=now(),allocation_id=allocation,**state),ensure_ascii=False),flush=True)
                previous_state=state.copy()

        active_job = None
        try:
            for job in plan['jobs']:
                wait_for_allocation(allocation,username=username,poll_seconds=poll_seconds,
                                    missing_polls=missing_polls,on_state=record_state)
                # A file changed while waiting cannot silently become the new source.
                if digest(archive) != expected:
                    raise RuntimeError('Pinned source archive changed while waiting')
                active_job = job
                command=['bash',str(project_root/'scripts/hpc/submit.sh'),'--allocation',allocation,
                         '--source-archive',str(archive),job['mode'],*job['arguments']]
                stdout=logs/(job['label']+'.out')
                stderr=logs/(job['label']+'.err')
                job.update(status='running',started_at=now(),command=command,
                           controller_stdout=str(stdout),controller_stderr=str(stderr))
                controller.update(status='running',active_label=job['label'])
                write_manifest(manifest,plan)
                print('Starting '+job['label'],flush=True)
                with stdout.open('w') as out,stderr.open('w') as err:
                    # foreground srun completion is the dependency gate. Do not
                    # infer a numeric job ID from stdout in allocation mode.
                    process=subprocess.Popen(command,cwd=project_root,stdin=subprocess.DEVNULL,
                                             stdout=out,stderr=err)
                    job['submit_process_pid']=process.pid
                    write_manifest(manifest,plan)
                    returncode=process.wait()
                job.update(returncode=returncode,finished_at=now(),
                           status='completed' if returncode==0 else 'failed')
                write_manifest(manifest,plan)
                if returncode:
                    job['stderr_tail']=stderr.read_text(errors='replace')[-4000:]
                    job['stdout_tail']=stdout.read_text(errors='replace')[-4000:]
                    raise RuntimeError(f'{job["label"]} failed with exit code {returncode}; later steps blocked')
                active_job=None
                print('Completed '+job['label'],flush=True)
            controller.update(status='completed',active_label=None,finished_at=now())
            write_manifest(manifest,plan)
            print('Sequence completed; allocation '+allocation+' remains reserved until its Slurm limit or explicit user cancellation.',flush=True)
            return 0
        except BaseException as exc:
            if active_job is not None and active_job.get('status')=='running':
                active_job.update(status='interrupted_unknown',failure=str(exc),finished_at=now())
                # In particular, never assume a controller interrupt cancelled
                # an already started srun step; resubmission requires inspection.
            for job in plan['jobs']:
                if job['status']=='waiting':
                    job.update(status='blocked',blocked_reason=str(exc))
            controller.update(status='failed',failure=str(exc),finished_at=now())
            write_manifest(manifest,plan)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--missing-polls',type=int,default=3)
    args=parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not 0<args.poll_seconds<=60:
        parser.error('--poll-seconds must be finite and within (0,60]')
    if args.missing_polls<1:
        parser.error('--missing-polls must be positive')
    try:
        return run_queue(args.manifest,project_root=args.project_root,
                         poll_seconds=args.poll_seconds,missing_polls=args.missing_polls)
    except Exception as exc:
        print('Allocation queue stopped: '+str(exc),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
