"""Recovery export deployment must not replace the original training identity."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_transfer_recovery', ROOT/'scripts/watch_training_transfer.py')
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def test_export_registration_and_logs_use_recovery_project(tmp_path, monkeypatch):
    submission = dict(remote_host='host', remote_project='/original', export_project='/recovery',
                      job_id=9266, experiment='exper042')
    remote = dict(output='/original/results/run', state='COMPLETED')
    calls = []
    monkeypatch.setattr(watcher.subprocess, 'check_output', lambda *a, **kw: json.dumps(
        dict(job_id='9500', state='RUNNING', output='/original/results/run/recovered')))
    monkeypatch.setattr(watcher, 'copy', lambda *a, **kw: calls.append(a))
    assert watcher.advance_export(submission, tmp_path, remote) is False
    saved = json.loads((tmp_path/'submission.json').read_text())
    assert saved['remote_project'] == '/original' and saved['job_id'] == 9266
    assert saved['export_job_id'] == '9500' and saved['export_state'] == 'RUNNING'
    assert [a[1] for a in calls] == ['/recovery/logs/hpc/9500.out', '/recovery/logs/hpc/9500.err']


def test_remote_finalizer_uses_recovery_deployment(tmp_path, monkeypatch, capsys):
    output = tmp_path/'original/results/run'
    folder = output/'recovered'
    folder.mkdir(parents=True)
    (output/'best.pt').write_bytes(b'checkpoint')
    (folder/'submission.json').write_text(json.dumps(dict(job_id='9500')))
    recovery = tmp_path/'recovery'
    submission = dict(remote_project=str(tmp_path/'original'), export_project=str(recovery),
                      export_relative='recovered', export_requirements={'interactive_replay': True})
    monkeypatch.setattr(sys, 'argv', ['remote', json.dumps(submission), str(output)])
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '9500|COMPLETED|0:0|\n')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', run)
    exec(watcher.EXPORT_REMOTE, {})
    assert calls[0][:2] == [str(recovery/'training/.venv/bin/python'),
                            str(recovery/'scripts/finalize_training_video.py')]
    assert json.loads(capsys.readouterr().out)['state'] == 'validated'
