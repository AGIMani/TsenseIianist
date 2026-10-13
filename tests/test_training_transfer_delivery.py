"""Delivery completion must include the same best checkpoint, video and plot."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('watch_training_transfer', Path(__file__).parents[1]/'scripts/watch_training_transfer.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'copy', lambda *args, **kwargs: None)
    (tmp_path/'best.pt').write_bytes(b'best')
    (tmp_path/'final.pt').write_bytes(b'final')
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'metrics.jsonl').write_text('{}\n')
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='completed',vector_steps=19532)))
    files = [dict(local=p.name, path='/remote/'+p.name, sha256=watch.digest(p)) for p in tmp_path.iterdir()]
    submission = dict(job_id=1,experiment='exper004',remote_host='fixture',requested_vector_steps=19532,
                      export_required=True,export_relative='video_best')
    remote = dict(state='COMPLETED',exit_code='0:0',output='/remote/run',files=files)
    return submission, remote


def test_training_only_never_marks_complete(tmp_path, monkeypatch):
    submission, remote = setup_run(tmp_path, monkeypatch)
    assert not watch.transfer(submission, tmp_path, remote)
    assert not (tmp_path/'TRANSFER_COMPLETE').exists()
    assert json.loads((tmp_path/'sync_status.json').read_text())['training_success']


def test_different_checkpoint_is_rejected(tmp_path, monkeypatch):
    submission, remote = setup_run(tmp_path, monkeypatch)
    folder = tmp_path/'video_best'
    for name in ('midi_comparison/piano_roll_comparison.png','midi_comparison/key_comparison.npz','replay_data/audio_substeps.npz'):
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    (folder/'delivery_validation.json').write_text(json.dumps(dict(state='validated',checkpoint_sha256='wrong')))
    (folder/'export_plan.json').write_text(json.dumps(dict(checkpoint_sha256='wrong')))
    with pytest.raises(RuntimeError, match='best.pt'):
        watch.transfer(submission, tmp_path, remote)
    assert not (tmp_path/'TRANSFER_COMPLETE').exists()


def test_checksum_failure_never_marks_complete(tmp_path, monkeypatch):
    submission, remote = setup_run(tmp_path, monkeypatch)
    (tmp_path/'best.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='Checksum mismatch'):
        watch.transfer(submission, tmp_path, remote)
    assert not (tmp_path/'TRANSFER_COMPLETE').exists()
