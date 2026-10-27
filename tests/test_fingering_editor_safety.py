"""Lightweight camera and budget tests; no model construction or simulation."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_above_keyboard_performer_camera_has_correct_hands_and_occlusion():
    view = module('fingering_editor_view')
    basis = view.camera_basis(0, .8)
    # Same piano convention as the frozen source: +Y higher pitch/right hand,
    # +X performer, +Z above the keyboard. Screen Y increases downwards.
    assert basis[0] @ [0, .15, 0] > basis[0] @ [0, -.15, 0]
    assert basis[1] @ [0, 0, .1] < 0
    assert basis[2] @ [0, 0, .1] > 0
    assert basis[1] @ [.18, 0, 0] > basis[1] @ [-.03, 0, 0]
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-15)
    opposite = view.camera_basis(np.pi, .8)
    assert opposite[0] @ [0, .15, 0] < opposite[0] @ [0, -.15, 0]


def fake_group(tmp_path, **updates):
    values = {'memory.max': str(1536*1024**2), 'memory.high': str(1024*1024**2),
              'memory.swap.max': '0', 'memory.oom.group': '0'}
    values.update(updates)
    for key, value in values.items():
        (tmp_path/key).write_text(value)
    return tmp_path


def test_effective_limits_reject_unbounded_or_swapping_group(tmp_path, monkeypatch):
    limits = module('fingering_editor_limits')
    group = fake_group(tmp_path)
    monkeypatch.setattr(limits, 'cgroup_directory', lambda: group)
    monkeypatch.setattr(limits, 'available_memory', lambda: 5*1024**3)
    result = limits.verify_limits()
    assert result['memory.oom.group'] == '1'
    (group/'memory.max').write_text('max')
    (group/'memory.oom.group').write_text('0')
    with pytest.raises(RuntimeError, match='missing fixed'):
        limits.verify_limits()
    assert (group/'memory.oom.group').read_text() == '0'
    fake_group(tmp_path, **{'memory.swap.max': '1048576'})
    with pytest.raises(RuntimeError, match='missing fixed'):
        limits.verify_limits()
    fake_group(tmp_path)
    monkeypatch.setattr(limits, 'available_memory', lambda: 3*1024**3)
    with pytest.raises(RuntimeError, match='4 GiB'):
        limits.verify_limits()


def test_scrubbing_keeps_only_latest_pending_request(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import queue
    from fingering_editor import RemotePreview
    preview = object.__new__(RemotePreview)
    preview.requests = queue.Queue(maxsize=1)
    for i in range(10000):
        preview.submit({'request_id': i})
    assert preview.requests.qsize() == 1
    assert preview.requests.get_nowait() == {'request_id': 9999}


def test_missing_remote_config_initializes_off_gui_thread(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import fingering_editor as ui
    from types import SimpleNamespace
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    monkeypatch.setattr(ui.threading.Thread, 'start', lambda self: None)
    config = tmp_path/'missing/remote.json'
    preview = ui.RemotePreview(config)
    assert not config.exists()
    commands = []

    def failed_preparation(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(ui.subprocess, 'run', failed_preparation)
    preview.run()
    assert preview.responses.get_nowait()['type'] == 'status'
    failure = preview.responses.get_nowait()
    assert failure['type'] == 'error'
    assert '初始化失败' in failure['error']
    assert commands[0][-2:] == ['--output', str(config)]
    assert preview.process is None


def test_connection_config_migrates_out_of_logs_without_overwriting(tmp_path, monkeypatch):
    paths = module('fingering_editor_config')
    legacy = tmp_path/'logs/fingering_editor/remote.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"host":"volc-a32"}\n')
    destination = paths.connection_config(tmp_path, 'remote')
    assert destination == tmp_path/'configs/fingering_editor/remote.json'
    assert destination.read_text() == '{"host":"volc-a32"}\n'
    assert not legacy.exists()
    legacy.write_text('{"host":"old"}')
    assert paths.connection_config(tmp_path, 'remote').read_text() == '{"host":"volc-a32"}\n'
    assert legacy.exists()


def test_diagonal_hatching_stays_inside_convex_key_face():
    view = module('fingering_editor_view')
    points = np.array([[0, 0], [20, 0], [20, 40], [0, 40]])
    stripes = list(view.diagonal_hatching(points))
    assert len(stripes) > 2
    for a, b in stripes:
        for p in (a, b):
            assert 0 <= p[0] <= 20 and 0 <= p[1] <= 40
        assert a[0]+a[1] == pytest.approx(b[0]+b[1])


def ssh_preview(tmp_path, monkeypatch, failures, diagnostic, code=255):
    """Exercise real pipes/process cleanup without connecting to an SSH server."""
    import json
    import os
    import queue
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import fingering_editor as ui
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    monkeypatch.setattr(ui.threading.Thread, 'start', lambda self: None)
    counter = tmp_path/'attempts'
    executable = tmp_path/'ssh'
    executable.write_text(f'''#!{sys.executable}
import json, pathlib, sys
counter = pathlib.Path({str(counter)!r})
n = int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
if n <= {failures}:
    print({diagnostic!r}, file=sys.stderr, flush=True)
    sys.exit({code})
print(json.dumps(dict(type='ready', meshes=[], frame={{}})), flush=True)
sys.stdin.read()
''')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path)+os.pathsep+os.environ['PATH'])
    config = tmp_path/'remote.json'
    config.write_text(json.dumps(dict(host='test', remote_root='/test',
                                     worker_sha256='test', source_sha256='test', python='python')))
    preview = ui.RemotePreview(config)
    preview.responses = queue.Queue()
    preview.submit(None)
    return preview, counter


def test_preview_recovers_from_ssh_handshake_resets(tmp_path, monkeypatch):
    preview, _ = ssh_preview(tmp_path, monkeypatch, 2,
                            'kex_exchange_identification: read: Connection reset by peer')
    preview.run()
    results = list(preview.responses.queue)
    assert any(item['type'] == 'ready' for item in results)
    assert not any(item['type'] == 'error' for item in results)
    assert preview.process.poll() == 0


@pytest.mark.parametrize('diagnostic,code,attempts', [
    ('Connection reset by peer', 255, 3),
    ('Permission denied (publickey).', 255, 1),
    ('Permission denied (publickey).\nConnection closed by remote host', 255, 1),
    ('REMOTE HOST IDENTIFICATION HAS CHANGED!', 255, 1),
    ('ValueError: invalid model configuration', 1, 1),
])
def test_preview_bounds_retries_and_reports_actual_failure(tmp_path, monkeypatch,
                                                         diagnostic, code, attempts):
    preview, counter = ssh_preview(tmp_path, monkeypatch, 10, diagnostic, code)
    preview.run()
    failure = list(preview.responses.queue)[-1]
    assert failure['type'] == 'error'
    assert diagnostic in failure['error']
    assert int(counter.read_text()) == attempts
    assert preview.process.poll() == code


def test_closing_preview_cancels_connection_retry(tmp_path, monkeypatch):
    import threading
    start = threading.Thread.start
    preview, counter = ssh_preview(tmp_path, monkeypatch, 10, 'Connection reset by peer')
    worker = threading.Thread(target=preview.run, daemon=True)
    start(worker)
    try:
        assert preview.responses.get(timeout=5)['type'] == 'status'
    finally:
        preview.close()
        worker.join(timeout=3)
    assert not worker.is_alive()
    assert int(counter.read_text()) == 1


def test_failed_connection_disables_pose_playback(monkeypatch):
    import queue
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from fingering_editor import Editor
    editor = object.__new__(Editor)
    editor.ready = True
    editor.request_id = 7
    editor.playing = True
    editor.play_button = SimpleNamespace(configure=lambda **kw: None)
    editor.pose_status = SimpleNamespace(configure=lambda **kw: None)
    editor.root = SimpleNamespace(after=lambda *a: None)
    editor.remote = SimpleNamespace(responses=queue.Queue())
    editor.remote.responses.put(dict(type='error', error='SSH transport failed'))
    editor.poll()
    assert not editor.ready
    assert not editor.playing
