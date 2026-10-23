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
