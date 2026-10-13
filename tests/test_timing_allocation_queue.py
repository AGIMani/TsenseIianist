"""CPU scheduler/entry stubs validate gates without Slurm submissions or GPUs."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/hpc/run_timing_allocation_queue.py'
SPEC=importlib.util.spec_from_file_location('timing_allocation_queue',SCRIPT)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class AllocationQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='timing queue ');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.bin=self.root/'bin';self.bin.mkdir()
        self.project=self.root/'project';(self.project/'scripts/hpc').mkdir(parents=True)
        (self.project/'scripts/hpc/submit.sh').write_text('''#!/bin/bash
set -eu
python3 - "$@" <<'STUB'
import json,os,sys
with open(os.environ['QUEUE_CALLS'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAIL_MODE')==sys.argv[5]:
 print('deliberate gate failure',file=sys.stderr);sys.exit(7)
print('step finished')
STUB
''')
        self.counter=self.root/'polls';self.calls=self.root/'calls'
        (self.bin/'squeue').write_text(f'''#!{sys.executable}
import os
from pathlib import Path
p=Path(os.environ['QUEUE_COUNTER']);n=int(p.read_text()) if p.exists() else 0;p.write_text(str(n+1))
states=os.environ['QUEUE_STATES'].split(',');state=states[min(n,len(states)-1)]
if state!='MISSING':print(os.environ['QUEUE_OWNER']+'|'+state+'|TsenseI-reserve2')
''')
        (self.bin/'scontrol').write_text(f'''#!{sys.executable}
import os,sys
state=os.environ.get('CONTROL_STATE','MISSING')
if state=='MISSING':sys.exit(1)
print('UserId='+os.environ['QUEUE_OWNER']+'(123) JobState='+state+' JobName=TsenseI-reserve2')
''')
        for name in ('squeue','scontrol'):(self.bin/name).chmod(0o755)
        archive=self.root/'source.tar.gz';archive.write_bytes(b'pinned test source')
        self.manifest=self.root/'manifest.json'
        self.plan=dict(allocation_id=1364,source_archive=str(archive),
            source=dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),jobs=[
                dict(label='validation',mode='validate',arguments=['tests/example.py']),
                dict(label='training',mode='train',arguments=['--output-dir','/space in path/result'])])
        self.manifest.write_text(json.dumps(self.plan))
        self.env=dict(os.environ,PATH=str(self.bin)+os.pathsep+os.environ['PATH'],
            QUEUE_COUNTER=str(self.counter),QUEUE_CALLS=str(self.calls),QUEUE_STATES='RUNNING',
            QUEUE_OWNER=pwd.getpwuid(os.getuid()).pw_name)

    def run_controller(self,success=True):
        result=subprocess.run([sys.executable,str(SCRIPT),'--manifest',str(self.manifest),
            '--project-root',str(self.project),'--poll-seconds','.001','--missing-polls','2'],
            text=True,capture_output=True,env=self.env,timeout=20)
        self.assertEqual(result.returncode==0,success,result.stderr)
        return json.loads(self.manifest.read_text()),result

    def test_pending_wait_then_gated_steps_preserve_allocation_and_argv(self):
        self.env['QUEUE_STATES']='PENDING,RUNNING'
        plan,_=self.run_controller()
        self.assertEqual(plan['controller']['status'],'completed')
        self.assertEqual([j['status'] for j in plan['jobs']],['completed','completed'])
        calls=[json.loads(row) for row in self.calls.read_text().splitlines()]
        self.assertEqual(calls[0][:4],['--allocation','1364','--source-archive',self.plan['source_archive']])
        self.assertEqual(calls[1][-1],'/space in path/result')
        self.assertGreaterEqual(int(self.counter.read_text()),3)
        for job in plan['jobs']:
            self.assertIn('submit_process_pid',job);self.assertIn('finished_at',job)
        # The only workload entry in the stubs is submit.sh --allocation;
        # neither sbatch nor scancel is installed or invoked by the controller.

    def test_failed_gate_blocks_training_and_keeps_error_evidence(self):
        self.env['FAIL_MODE']='validate'
        plan,_=self.run_controller(success=False)
        self.assertEqual(plan['controller']['status'],'failed')
        self.assertEqual([j['status'] for j in plan['jobs']],['failed','blocked'])
        self.assertEqual(plan['jobs'][0]['returncode'],7)
        self.assertIn('deliberate gate failure',plan['jobs'][0]['stderr_tail'])
        self.assertEqual(len(self.calls.read_text().splitlines()),1)

    def test_terminal_or_foreign_allocation_never_starts_steps(self):
        for owner,state in ((self.env['QUEUE_OWNER'],'CANCELLED'),('other','RUNNING')):
            with self.subTest(owner=owner,state=state):
                self.manifest.write_text(json.dumps(self.plan));self.counter.unlink(missing_ok=True)
                self.env.update(QUEUE_OWNER=owner,QUEUE_STATES=state)
                plan,_=self.run_controller(success=False)
                self.assertTrue(all(j['status']=='blocked' for j in plan['jobs']))
                self.assertFalse(self.calls.exists())

    def test_missing_squeue_falls_back_and_missing_job_has_bounded_grace(self):
        self.env.update(QUEUE_STATES='MISSING',CONTROL_STATE='RUNNING')
        plan,_=self.run_controller();self.assertEqual(plan['controller']['status'],'completed')
        self.manifest.write_text(json.dumps(self.plan));self.calls.unlink();self.counter.unlink()
        self.env['CONTROL_STATE']='MISSING'
        plan,_=self.run_controller(success=False)
        self.assertEqual(int(self.counter.read_text()),2)
        self.assertFalse(self.calls.exists())

    def test_duplicate_lock_or_completed_manifest_is_rejected(self):
        lock=self.manifest.with_suffix('.json.controller.lock')
        with lock.open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            _,result=self.run_controller(success=False)
            self.assertIn('already holds',result.stderr)
        self.run_controller()
        calls=self.calls.read_text()
        _,result=self.run_controller(success=False)
        self.assertIn('Refusing to rerun',result.stderr)
        self.assertEqual(self.calls.read_text(),calls)

    def test_archive_modification_rejected_before_step(self):
        Path(self.plan['source_archive']).write_bytes(b'changed')
        _,result=self.run_controller(success=False)
        self.assertIn('checksum changed',result.stderr)
        self.assertFalse(self.calls.exists())


if __name__=='__main__':unittest.main()
