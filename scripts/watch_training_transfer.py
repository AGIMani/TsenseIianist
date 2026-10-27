#!/usr/bin/env python3
"""Persistently export best.pt and verify full delivery after Slurm; no model calls."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


REMOTE = r'''
import hashlib,json,re,subprocess,sys
from pathlib import Path
project,prefix,job,hashes,backend=sys.argv[1:]
project=Path(project); prefix=Path(prefix)
def run(args):
    result=subprocess.run(args,text=True,capture_output=True,timeout=20)
    if result.returncode: raise RuntimeError(result.stderr.strip())
    return result.stdout
if backend=='process':
    sys.path.insert(0,str(project/'scripts/hpc'))
    from process_backend import status
    current=status(project/'logs/hpc'/job/'process_state.json')
    state=current['state'];exit_code=current.get('exit_code')
else:
    rows=run(['sacct','-n','-X','-j',job,'--parsable2','--format=JobIDRaw,State,ExitCode']).splitlines()
    state=exit_code=None
    for row in rows:
        values=row.split('|')
        if values[0]==job:
            state=values[1].split()[0].rstrip('+');exit_code=values[2];break
    if state is None:
        pending=run(['squeue','-h','-j',job,'-o','%T']).strip()
        state=pending or 'UNKNOWN'
error_path=project/'logs/hpc'/f'{job}.err'
output=None
if error_path.is_file():
    matches=re.findall(r'本次输出目录（北京时间）：([^\n]+)',error_path.read_text(errors='replace'))
    if matches: output=Path(matches[-1].strip())
if output is not None and (output.parent!=prefix.parent or not output.name.startswith(prefix.name+'-')):
    raise RuntimeError('Job output does not match the registered experiment')
if output is not None and not output.exists(): output=None
files=[]
if output and output.exists():
    for path in sorted(output.rglob('*')):
        if path.is_symlink(): raise RuntimeError('Result symlinks are not supported')
        if path.is_file(): files.append(dict(path=str(path),local=str(path.relative_to(output))))
job_dir=project/'logs/hpc'/job
for name in ('command.txt','source.tar.gz.sha256','source_audit.json','gpu_probe.json','gpu_uuids.txt','process_state.json','request.json','worker.pid'):
    path=job_dir/name
    if path.is_file(): files.append(dict(path=str(path),local='slurm_job_'+job+'/'+name))
for suffix in ('out','err'):
    path=project/'logs/hpc'/f'{job}.{suffix}'
    if path.is_file(): files.append(dict(path=str(path),local='slurm_job_'+job+'/'+path.name))
deployment=project/'logs/deployment'
if deployment.exists():
    for path in sorted(deployment.rglob('*')):
        if path.is_file() and not path.is_symlink():
            files.append(dict(path=str(path),local='deployment/'+str(path.relative_to(deployment))))
if hashes=='yes':
    for item in files:
        digest=hashlib.sha256()
        with open(item['path'],'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        item['sha256']=digest.hexdigest()
print(json.dumps(dict(state=state,exit_code=exit_code,output=str(output) if output else None,files=files)))
'''

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def query(submission, hashes=False):
    args = [submission['remote_project'], submission['remote_output_prefix'],
            str(submission['job_id']), 'yes' if hashes else 'no', submission.get('backend','slurm')]
    command = 'python3 -c '+shlex.quote(REMOTE)+' '+shlex.join(args)
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                   submission['remote_host'], command], text=True, timeout=120)
    return json.loads(raw)


def copy(host, remote, local, directory=False):
    local.parent.mkdir(parents=True, exist_ok=True)
    source = host+':'+remote+('/' if directory else '')
    subprocess.run(['rsync', '-a', '--protect-args', '--timeout=60', '-e',
                    'ssh -o BatchMode=yes -o ConnectTimeout=15', '--', source,
                    str(local)+('/' if directory else '')], check=True, timeout=1800)


def transfer(submission, root, remote):
    job = str(submission['job_id'])
    status = dict(job_id=job, experiment=submission['experiment'], updated_at=now(),
                  slurm_state=remote['state'] if submission.get('backend','slurm')=='slurm' else None,
                  backend=submission.get('backend','slurm'), job_state=remote['state'], slurm_exit_code=remote['exit_code'],
                  remote_output=remote['output'], state='transferring', watcher='background_script')
    save(root/'sync_status.json', status)
    previous = root/'transfer_manifest.json'
    if previous.exists() and str(json.loads(previous.read_text())['job_id']) != job:
        raise RuntimeError('Local result identity differs; refusing overwrite')
    if remote['output']:
        if (root/'config.json').exists() and not previous.exists():
            expected = next((x['sha256'] for x in remote['files'] if x['local']=='config.json'), None)
            if digest(root/'config.json') != expected:
                raise RuntimeError('Existing local config differs; refusing overwrite')
        # Save identity before the first copy so retries belong to this same job.
        save(previous, dict(job_id=job, remote_output=remote['output'], state='copying'))
        copy(submission['remote_host'], remote['output'], root, directory=True)
    for item in remote['files']:
        relative = Path(item['local'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe local destination')
        if relative.parts[0] in ('slurm_job_'+job, 'deployment'):
            copy(submission['remote_host'], item['path'], root/relative)
    bad = [item['local'] for item in remote['files']
           if not (root/item['local']).is_file() or digest(root/item['local']) != item['sha256']]
    if bad:
        raise RuntimeError('Checksum mismatch: '+', '.join(bad[:8]))
    success = remote['state']=='COMPLETED' and remote['exit_code']=='0:0'
    summary = None
    if success:
        for name in ('final.pt', 'config.json', 'summary.json', 'metrics.jsonl'):
            if not (root/name).is_file():
                raise RuntimeError('Completed job is missing '+name)
        summary = json.loads((root/'summary.json').read_text())
        budget = int(summary.get('vector_steps', -1)) >= submission['requested_vector_steps']
        success = summary.get('status')=='completed' and budget
    save(previous, dict(job_id=job, remote_output=remote['output'], verified_at=now(),
                        files={item['local']:item['sha256'] for item in remote['files']}))
    delivery = submission.get('export_required', False)
    validation = None
    if delivery and submission.get('export_relative'):
        folder = root/submission['export_relative']
        proof = folder/'delivery_validation.json'
        if proof.is_file():
            validation = json.loads(proof.read_text())
            if validation.get('state') != 'validated':
                raise RuntimeError('Export has not passed delivery validation')
            required = ('midi_comparison/piano_roll_comparison.png',
                        'midi_comparison/key_comparison.npz', 'replay_data/audio_substeps.npz')
            if not all((folder/name).is_file() for name in required):
                raise RuntimeError('Missing comparison/replay delivery files')
            if submission.get('export_requirements', {}).get('interactive_replay'):
                if validation.get('interactive_replay', {}).get('state') != 'validated':
                    raise RuntimeError('Interactive replay has not passed validation')
                for name in ('replay.py','physics_trace.npz','played.wav','replay.json','score_annotations.npz','launch_cpu.sh'):
                    if not (folder/'interactive_replay'/name).is_file():
                        raise RuntimeError('Missing interactive replay file: '+name)
            plan = json.loads((folder/'export_plan.json').read_text())
            if digest(root/'best.pt') != validation['checkpoint_sha256'] or plan['checkpoint_sha256'] != validation['checkpoint_sha256']:
                raise RuntimeError('Delivered video does not match this round best.pt')
            report = json.loads((folder/'report.json').read_text())
            for video in report['videos']:
                path = folder/video['file']
                probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)], text=True))
                if not {'audio','video'} <= {s['codec_type'] for s in probe['streams']}:
                    raise RuntimeError('Local video is missing an audio/video stream')
                subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'], check=True, timeout=300)
    complete = not delivery or validation is not None
    status.update(state='completed' if complete else 'training_files_transferred_video_pending', updated_at=now(), checksum_verified=True,
                  training_success=success, files_verified=len(remote['files']),
                  export_state='validated' if validation else ('pending' if delivery else 'not_requested'),
                  export_job_id=submission.get('export_job_id'), video_validation=validation,
                  training_summary={k:summary.get(k) for k in ('status','vector_steps','transitions','stop_reason')}
                  if summary else None)
    save(root/'sync_status.json', status)
    if complete:
        (root/'TRANSFER_COMPLETE').write_text(now()+'\n')
    print('transfer_verified '+json.dumps(status, ensure_ascii=False), flush=True)
    return complete


EXPORT_REMOTE = r'''
import fcntl,json,os,subprocess,sys
from pathlib import Path
s=json.loads(sys.argv[1]); output=Path(sys.argv[2]); project=Path(s.get('export_project',s['remote_project']))
if s.get('backend')=='process':
    os.environ['TSENSEI_JOB_BACKEND']='process'
    os.environ['TSENSEI_JOB_SOURCE_ARCHIVE']=s['remote_source_archive']
    os.environ['TSENSEI_JOB_SOURCE_SHA256']=s['source_archive_sha256']
lock=(output/'auto_export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
best=output/'best.pt'
if not best.is_file():
    print(json.dumps(dict(state='missing_best',error='No best.pt available for export')));sys.exit(0)
folder=output/s.get('export_relative','video_best')
registration=folder/'submission.json'
if not registration.exists():
    if folder.exists():
        # Never duplicate an ambiguous submission; recover from the durable job command.
        matches=[]
        for p in (project/'logs/hpc').glob('*/command.txt'):
            if str(folder/'export_plan.json') in p.read_text():matches.append(p.parent.name)
        if len(matches)==1:
            registration.write_text(json.dumps(dict(job_id=matches[0],plan=str(folder/'export_plan.json'))))
        else:
            print(json.dumps(dict(state='submission_needs_recovery',error='Export directory exists without a unique registered job; not resubmitting')));sys.exit(0)
    else:
        args=['bash',str(project/'export_video.sh'),'--pt',str(best),'--source-archive',s['remote_source_archive'],
              '--output',str(folder),'--gpus','1','--video1',('native' if s.get('arms') else 'noarms'),'front','name=best_fingering',
              '--plots','hand-joints','--audio','--fingering','--key-highlight']
        result=subprocess.run(args,text=True,capture_output=True)
        (output/'auto_export_submit.log').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:raise RuntimeError('Export submission failed: '+result.stderr)
job=str(json.loads(registration.read_text())['job_id'])
if s.get('backend')=='process':
    sys.path.insert(0,str(project/'scripts/hpc'))
    from process_backend import status
    current=status(project/'logs/hpc'/job/'process_state.json')
    state=current['state'];code=current.get('exit_code')
else:
    rows=subprocess.check_output(['sacct','-n','-X','-j',job,'--parsable2','--format=JobIDRaw,State,ExitCode'],text=True)
    state='UNKNOWN';code=None
    for row in rows.splitlines():
        v=row.split('|')
        if v[0]==job:state=v[1].split()[0].rstrip('+');code=v[2]
result=dict(job_id=job,state=state,exit_code=code,output=str(folder))
if state=='COMPLETED' and code=='0:0':
    proof=folder/'delivery_validation.json'
    needs_interactive=s.get('export_requirements',{}).get('interactive_replay')
    if not proof.is_file() or (needs_interactive and json.loads(proof.read_text()).get('interactive_replay',{}).get('state')!='validated'):
        env=os.environ.copy()
        env['PATH']=str(project/'training_mjlab/hpc/tools')+':'+env.get('PATH','')
        with (output/'auto_export_finalize.log').open('a') as log:
            args=[str(project/'training/.venv/bin/python'),str(project/'scripts/finalize_training_video.py'),
                '--export',str(folder)]
            if s.get('original_midi'):
                args += ['--original-midi',s['original_midi'],
                         '--original-midi-sha256',s['original_midi_sha256']]
            if s.get('registered_score'):
                args += ['--registered-score',s['registered_score'],
                         '--registered-score-sha256',s['registered_score_sha256']]
            done=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError('CPU export validation failed; see auto_export_finalize.log')
    result['state']='validated'
print(json.dumps(result))
'''


def advance_export(submission, root, remote):
    if not remote['output']:
        save(root/'export_status.json', dict(state='missing_output', training_state=remote['state'], updated_at=now()))
        return None
    command = 'python3 -c '+shlex.quote(EXPORT_REMOTE)+' '+shlex.join([json.dumps(submission), remote['output']])
    result = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        submission['remote_host'], command], text=True, timeout=1200)
    status = json.loads(result)
    status['updated_at'] = now()
    status['training_state'] = remote['state']
    status['checkpoint_label'] = 'best.pt' if remote['state']=='COMPLETED' else 'failed_training_best.pt'
    save(root/'export_status.json', status)
    if status.get('job_id'):
        submission.update(export_job_id=status['job_id'], remote_export_output=status['output'],
                          remote_output=remote['output'], export_state=status['state'])
        save(root/'submission.json', submission)
        # Include exporter diagnostics alongside the training diagnostics.
        export_job = str(status['job_id'])
        for suffix in ('out','err'):
            try:
                copy(submission['remote_host'], submission.get('export_project',submission['remote_project'])+'/logs/hpc/'+export_job+'.'+suffix,
                     root/'slurm_export'/ (export_job+'.'+suffix))
            except subprocess.SubprocessError:
                pass
    if status['state']=='missing_best':
        return None
    return status['state']=='validated'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission', required=True, type=Path)
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()
    if args.interval < 10:
        p.error('interval must be at least 10 seconds')
    submission = json.loads(args.submission.read_text())
    root = Path(submission['local_output'])
    if args.submission.resolve().parent != root.resolve():
        p.error('Submission must reside in its registered local output')
    lock = (root/'transfer.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A transfer watcher is already running for this experiment')
    (root/'transfer.pid').write_text(str(os.getpid())+'\n')
    last_state = None
    print(f'watcher_started job={submission["job_id"]} utc={now()}', flush=True)
    while True:
        try:
            remote = query(submission)
            submission['training_state'] = remote['state']
            submission['remote_output'] = remote['output']
            save(root/'submission.json', submission)
            if remote['state'] != last_state:
                print(f'state={remote["state"]} utc={now()}', flush=True)
                last_state = remote['state']
            if remote['state'] in TERMINAL:
                if not submission.get('export_required'):
                    transfer(submission, root, query(submission, hashes=True))
                    return
                manifest = root/'transfer_manifest.json'
                if not manifest.exists() or not json.loads(manifest.read_text()).get('verified_at'):
                    transfer(submission, root, query(submission, hashes=True))
                exported = advance_export(submission, root, remote)
                if exported is None:
                    save(root/'sync_status.json', dict(state='failed_training_diagnostics_transferred_replay_missing',
                        training_success=False, slurm_state=remote['state'] if submission.get('backend','slurm')=='slurm' else None,
                  backend=submission.get('backend','slurm'), job_state=remote['state'], updated_at=now(),
                        reason='No output or no best.pt; required replay cannot be produced. See export_status.json.'))
                    return
                if exported:
                    if transfer(submission, root, query(submission, hashes=True)):
                        return
                else:
                    export_status = json.loads((root/'export_status.json').read_text())
                    save(root/'sync_status.json', dict(state='training_files_transferred_video_pending',
                        training_success=remote['state']=='COMPLETED' and remote['exit_code']=='0:0',
                        job_id=submission['job_id'], experiment=submission['experiment'],
                        training_state=remote['state'], export_state=export_status['state'],
                        export_job_id=export_status.get('job_id'), updated_at=now(),
                        watcher='background_script', checksum_verified=True))
                time.sleep(args.interval)
                continue
            save(root/'sync_status.json', dict(state='waiting', slurm_state=remote['state'] if submission.get('backend','slurm')=='slurm' else None,
                  backend=submission.get('backend','slurm'), job_state=remote['state'],
                 job_id=submission['job_id'], experiment=submission['experiment'],
                 remote_output=remote['output'], watcher='background_script', updated_at=now()))
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            print(f'retry utc={now()} error={error}', flush=True)
            save(root/'sync_status.json', dict(state='retrying', job_id=submission['job_id'],
                 watcher='background_script', updated_at=now(), error=str(error)))
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
