"""API boundary tests: uploads and export provenance without any simulation."""
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trajectory_compare.app import create_app


CONTENT = b"a known original native CSV"
SHA = hashlib.sha256(CONTENT).hexdigest()
CURVE = {"time_s": [0, 1], "positions": {"index_tip": [[1, 2, 3], [4, 5, 6]]},
         "frame": "piano", "unit": "mm"}


def fixture_catalog(tmp_path):
    video = tmp_path / "cached.mp4"
    video.write_bytes(b"0123456789")
    raw = {"metadata": {"sha256": SHA, "filename": "original.csv", "duration_s": 1,
                        "samples": 2, "channels": 27}, "curve": CURVE,
           "video_url": "/media/video", "comparison_url": "/media/compare",
           "f1": {"value": 0.25, "scope": "same run", "sampling_hz": 400},
           "provenance": {"record": "verified"}}
    return {"manifest": {"test": True}, "media": {"video": str(video)},
            "landmarks": [{"id": "index_tip", "label": "食指"}],
            "slots": [{"id": "targets", "mode": "piano_only", "result": raw}],
            "cached_exports": {SHA: {"piano_only": raw}}}


def parser(path, filename, manifest):
    assert path.is_file() and manifest == {"test": True}
    if path.read_bytes() == b"invalid":
        raise ValueError("时间必须递增")
    return {"metadata": {"filename": filename, "duration_s": 1, "samples": 2, "channels": 27},
            "curve": CURVE}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "runtime", fixture_catalog(tmp_path), parser)) as app:
        yield app


def upload(client, content=CONTENT, filename="joint_positions.csv"):
    response = client.post("/api/trajectories", params={"filename": filename}, content=content)
    assert response.status_code == 201, response.text
    return response.json()


def test_bootstrap_and_range_video(client):
    state = client.get("/api/bootstrap").json()
    assert [slot["mode"] for slot in state["slots"]] == ["robot_dynamics", "prescribed", "prescribed"]
    assert state["slots"][1]["result"]["f1"] == .25
    assert state["slots"][1]["curve"] == CURVE
    assert "manifest" not in state and "media" not in state
    response = client.get("/media/video", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206 and response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"


def test_cached_export_requires_exact_bytes_and_mode(client):
    identity = upload(client)["trajectory"]["id"]
    matching = client.post("/api/jobs", json={"slot_id": 2, "mode": "prescribed", "trajectory_id": identity})
    assert matching.status_code == 202
    assert matching.json()["status"] == "completed" and matching.json()["result"]["f1"] == .25
    wrong_mode = client.post("/api/jobs", json={"slot_id": 1, "mode": "robot_dynamics", "trajectory_id": identity}).json()
    assert wrong_mode["status"] == "pending_backend" and wrong_mode["result"] is None
    changed = upload(client, CONTENT + b"changed")["trajectory"]["id"]
    unknown = client.post("/api/jobs", json={"slot_id": 3, "mode": "prescribed", "trajectory_id": changed}).json()
    assert unknown["status"] == "pending_backend" and unknown["result"] is None
    assert unknown["contract"]["robot_dynamic_dofs"] == 0
    assert client.get("/api/jobs/" + unknown["id"]).json() == unknown
    state = client.get("/api/bootstrap").json()
    assert state["slots"][2]["result"] is None and state["slots"][2]["curve"] == CURVE


def test_upload_validation_and_cleanup(client, monkeypatch):
    for name in ("../a.csv", "foo\\a.csv", "a.exe", "a\x00.csv"):
        assert client.post("/api/trajectories", params={"filename": name}, content=CONTENT).status_code == 422
    assert client.post("/api/trajectories", params={"filename": "bad.csv"}, content=b"invalid").status_code == 422
    monkeypatch.setattr("trajectory_compare.app.MAX_UPLOAD_BYTES", 3)
    assert client.post("/api/trajectories", params={"filename": "a.csv"}, content=CONTENT).status_code == 413
    assert not list(client.app.state.store.runtime.joinpath("trajectories").glob("*.csv"))
    assert client.get("/media/missing").status_code == 404


def test_job_validation(client):
    identity = upload(client)["trajectory"]["id"]
    assert client.post("/api/jobs", json={"slot_id": 2, "mode": "robot_dynamics", "trajectory_id": identity}).status_code == 422
    assert client.post("/api/jobs", json={"slot_id": 3, "mode": "prescribed", "trajectory_id": "unknown"}).status_code == 404
    assert client.post("/api/jobs", json={"slot_id": 3, "mode": "prescribed", "trajectory_id": identity, "command": "run"}).status_code == 422


def test_restore_upload_pending_job_and_registered_media(tmp_path):
    runtime = tmp_path / "isolated"
    catalog = fixture_catalog(tmp_path)
    with TestClient(create_app(runtime, catalog, parser)) as client:
        info = upload(client, b"new")
        frozen = client.get(info["trajectory"]["scene_manifest_url"])
        assert frozen.json() == {"test": True}
        assert hashlib.sha256(frozen.content).hexdigest() == info["trajectory"]["scene_manifest_sha256"]
        job = client.post("/api/jobs", json={"slot_id": 3, "mode": "prescribed", "trajectory_id": info["trajectory"]["id"]}).json()
        assert job["contract"]["scene_manifest_sha256"] == info["trajectory"]["scene_manifest_sha256"]
    with TestClient(create_app(runtime, catalog, parser)) as restored:
        state = restored.get("/api/bootstrap").json()
        assert state["slots"][2]["trajectory"]["id"] == info["trajectory"]["id"]
        assert state["slots"][2]["job"]["id"] == job["id"] and state["slots"][2]["result"] is None
        assert restored.get(info["trajectory"]["download_url"]).content == b"new"
        assert restored.get("/api/trajectories/" + info["trajectory"]["id"] + "/curve").json() == CURVE


def test_cached_dynamic_returns_executed_curve_and_restores_it(tmp_path):
    """Input targets may differ from simulated execution; plots must follow video."""
    runtime = tmp_path / 'runtime'
    catalog = fixture_catalog(tmp_path)
    raw = dict(catalog['cached_exports'][SHA]['piano_only'])
    executed = {'time_s': [0, 1], 'positions': {'index_tip': [[10, 20, 30], [40, 50, 60]]},
                'frame': 'piano', 'unit': 'mm'}
    raw.update(curve=executed, f1=None, metric_status='缺少原始MIDI',
               actual_keys_url='/media/video', source_note='20 finger DOFs, arm prescribed')
    catalog['cached_exports'][SHA]['robot_dynamics'] = raw
    with TestClient(create_app(runtime, catalog, parser)) as client:
        identity = upload(client)['trajectory']['id']
        job = client.post('/api/jobs', json={'slot_id': 1, 'mode': 'robot_dynamics',
                          'trajectory_id': identity}).json()
        assert job['curve'] == executed and job['result']['f1'] is None
        assert job['result']['metric_status'] == '缺少原始MIDI'
        slot = client.get('/api/bootstrap').json()['slots'][0]
        assert slot['curve'] == executed
    with TestClient(create_app(runtime, catalog, parser)) as client:
        assert client.get('/api/bootstrap').json()['slots'][0]['curve'] == executed


def test_server_session_changes_on_restart_without_stale_page_cache(tmp_path):
    catalog = fixture_catalog(tmp_path)
    runtime = tmp_path / 'runtime'
    with TestClient(create_app(runtime, catalog, parser)) as client:
        first = client.get('/api/bootstrap').json()['session_id']
        session = client.get('/api/session')
        assert session.json() == {'session_id': first}
        assert session.headers['cache-control'] == 'no-store'
        assert client.get('/').headers['cache-control'] == 'no-store'
    with TestClient(create_app(runtime, catalog, parser)) as client:
        assert client.get('/api/session').json()['session_id'] != first


def test_export_cache_rejects_another_base_manifest(client):
    info = upload(client)['trajectory']
    cached = client.app.state.store.catalog['cached_exports'][SHA]['piano_only']
    cached['provenance']['input_manifest_sha256'] = 'another-base'
    job = client.post('/api/jobs', json={'slot_id': 2, 'mode': 'prescribed', 'trajectory_id': info['id']}).json()
    assert job['status'] == 'pending_backend' and job['result'] is None


def test_completed_worker_media_and_cache_restore_with_integrity_check(tmp_path, monkeypatch):
    """A completed worker must remain usable after restarting the web process."""
    import json
    from trajectory_compare.app import Store
    from trajectory_compare.data import fingerprint
    from trajectory_compare.exporter import ExportManager
    runtime = tmp_path / 'runtime'
    store = Store(runtime, fixture_catalog(tmp_path))
    output = runtime / 'exports' / 'worker-test'; output.mkdir(parents=True)
    media = output / 'video.mp4'; media.write_bytes(b'unit-test-media')
    raw = dict(store.catalog['cached_exports'][SHA]['piano_only'])
    raw.update(video_url='/media/new-video', comparison_url=None,
               provenance={'input_sha256': SHA, 'input_manifest_sha256': store.manifest_sha})
    dataset = output / 'dataset.json'
    dataset.write_text(json.dumps({'schema': 'trajectory_compare.catalog.v1', 'manifest': {'models': []},
        'media': {'new-video': 'video.mp4'}, 'slots': [{'mode': 'prescribed', 'result': raw}]}))
    (output / 'manifest.json').write_text(json.dumps({'files': {'dataset.json': fingerprint(dataset), 'video.mp4': fingerprint(media)}}))
    store.jobs['unit-job'] = {'id': 'unit-job', 'status': 'completed', 'output_dataset': str(dataset)}
    monkeypatch.setattr('trajectory_compare.exporter.threading.Thread.start', lambda self: None)
    manager = ExportManager(store)
    assert store.media['new-video'] == media
    assert store.catalog['cached_exports'][SHA]['piano_only']['video_url'] == '/media/new-video'
    media.write_bytes(b'changed')
    with pytest.raises(ValueError, match='散列'):
        manager._register_output(dataset)
