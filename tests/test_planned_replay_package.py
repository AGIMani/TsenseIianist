"""CPU-only contracts for packaging existing physics, never loading MuJoCo."""
import hashlib
import importlib
import importlib.util
import json
import wave

import numpy as np
import pytest


def implementation():
    assert importlib.util.find_spec("scripts.package_planned_replay") is not None, "planned replay packager is missing"
    return importlib.import_module("scripts.package_planned_replay")


def trace_fixture():
    time = np.array([0., .01, .02, .03])
    keys = np.zeros((4, 88))
    keys[1:3, 39] = 1.
    keys[2, 40] = 1.
    target = np.zeros_like(keys, dtype=bool)
    target[1:3, 39] = True
    assignment = np.zeros((4, 10, 88), dtype=bool)
    assignment[1:3, 1, 39] = True
    return dict(time=time, qpos=keys.copy(), qvel=np.zeros((4, 88)), ctrl=np.zeros((4, 2)),
                key_qpos=keys, key_limits=np.tile([0., 1.], (88, 1)),
                key_activation_fraction=np.array(1.), actual_keys=keys.astype(bool),
                target_keys=target, assignment=assignment,
                original_events=np.array([[.01, .03, 39, 1]]),
                mocap_pos=np.zeros((4, 0, 3)), mocap_quat=np.zeros((4, 0, 4)))


def test_same_recording_positions_determine_activation_and_metrics():
    result = implementation().validate_trace(trace_fixture())
    assert result["metrics"] == dict(tp=2, fp=1, fn=0, physical_sample_micro_f1=.8)
    assert result["duration_s"] == .03
    assert result["physics_dt"] == .01


def test_long_uniform_timestamps_do_not_accumulate_a_median_roundoff_error():
    # Only a 1.8 MB timestamp array; no full state arrays or MuJoCo model.
    time=np.arange(222475)*.001
    assert np.max(np.abs(time-np.arange(len(time))*np.median(np.diff(time))))>1e-9
    assert implementation().recorded_physics_dt(time,.001)==.001
    assert implementation().recorded_physics_dt(time)==.001
    time[100000]+=.00001
    with pytest.raises(ValueError,match='uniform'):
        implementation().recorded_physics_dt(time,.001)


def test_original_note_boundaries_allow_only_planner_floating_point_tolerance():
    trace = trace_fixture()
    trace["original_events"][0, :2] += 5e-11
    assert implementation().validate_trace(trace)["metrics"]["physical_sample_micro_f1"] == .8


@pytest.mark.parametrize("corruption,match", [
    ("actual", "actual_keys"), ("target", "target_keys"),
    ("assignment", "assignment"), ("time", "uniform"),
])
def test_inconsistent_recorded_claims_are_rejected(corruption, match):
    trace = trace_fixture()
    if corruption == "actual":
        trace["actual_keys"][0, 0] = True
    elif corruption == "target":
        trace["target_keys"][0, 0] = True
    elif corruption == "assignment":
        trace["assignment"][1, 1, 39] = False
    else:
        trace["time"][2] = .021
    with pytest.raises(ValueError, match=match):
        implementation().validate_trace(trace)


def test_audio_and_intervals_use_t0_samples_without_extra_final_frame():
    module = implementation()
    time = np.array([0., .01, .02, .03])
    actual = np.zeros((4, 88), bool)
    actual[0:2, 39] = True
    actual[2:, 40] = True
    assert module.actual_intervals(actual, time) == [[0., .02, 60], [.02, .03, 61]]
    assert module.audio_events(actual, time) == [
        dict(time=0., type="note_on", note=60, velocity=127),
        dict(time=.02, type="note_off", note=60),
        dict(time=.02, type="note_on", note=61, velocity=127),
        dict(time=.03, type="note_off", note=61, end_of_replay=True),
    ]


def test_annotations_match_viewer_next_frame_lookup():
    trace = trace_fixture()
    score = implementation().viewer_annotations(trace["time"], trace["assignment"])
    for at, active in ((.005, False), (.01, True), (.015, True), (.029, True), (.03, False)):
        j = min(int(np.searchsorted(score["time"], at-1e-9)), len(score["time"])-1)
        assert bool(score["fingers"][j, 1, 39]) is active
    assert score["time"][-1] == .03


def write_score_and_midi(folder):
    pretty_midi = pytest.importorskip("pretty_midi")
    from scripts.fingering_editor_data import Sequence
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=1000)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=.01, end=.03))
    midi.instruments.append(instrument)
    midi_path = folder / "original.mid"
    midi.write(str(midi_path))
    seq = Sequence()
    seq.notes.add(start_time=.01, end_time=.03, pitch=60, velocity=90, instrument=0, part=1)
    seq.total_time = .03
    score_path = folder / "score.proto"
    score_path.write_bytes(seq.SerializeToString())
    return score_path, midi_path, seq


def test_score_verification_preserves_original_note_attributes(tmp_path):
    module = implementation()
    score, midi, seq = write_score_and_midi(tmp_path)
    verified = module.verify_score(score, midi)
    assert verified["events"] == [[.01, .03, 39, 1]]
    seq.notes[0].velocity = 89
    score.write_bytes(seq.SerializeToString())
    with pytest.raises(ValueError, match="MIDI"):
        module.verify_score(score, midi)


def test_package_is_self_contained_and_preserves_poor_physical_result(tmp_path):
    module = implementation()
    score, midi, _ = write_score_and_midi(tmp_path)
    recording = tmp_path / "recording"
    recording.mkdir()
    trace = trace_fixture()
    np.savez_compressed(recording / "physical_trace.npz", **trace)
    # Deliberately not a model: packaging must copy bytes without loading MuJoCo.
    (recording / "model.mjb").write_bytes(b"opaque recorded model fixture")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(kind="actuator_only_ik_physical_v1", original_midi_micro_f1=.8,
                  tp=2, fp=1, fn=0, full_piece=True, acceptance_passed=False,
                  physics_hz=100., duration_s=.03, same_hand_contact_samples=3,
                  dependency_versions={"mujoco": "fixture-version"},
                  source=dict(score_sha256=sha(score), original_midi_sha256=sha(midi),
                              archive_sha256="a"*64, archive_path="source_current.tar.gz"),
                  model_sha256=sha(recording / "model.mjb"))
    report['robot_state_projection']='current training hard_v1 stops only; keys excluded'
    report['physical_control']='unchanged position actuators; mj_step; hard_v1 robot stops'
    (recording / "replay_report.json").write_text(json.dumps(report))

    output = tmp_path / "output"
    result = module.package(recording, midi, score, output)
    assert result["metrics"]["physical_sample_micro_f1"] == .8
    assert result["source_report"]["same_hand_contact_samples"] == 3
    assert result["source_report"]["acceptance_passed"] is False
    assert result["checkpoint"] is None
    assert result['robot_state_projection']==report['robot_state_projection']
    assert 'actuator-only' not in result['controller']
    assert 'hard_v1' in (output/'README.txt').read_text()
    assert 'keys excluded' in (output/'README.txt').read_text()
    assert result["audio"]["rms"] > 0
    with wave.open(str(output / "played.wav"), "rb") as wav:
        assert wav.getnframes() / wav.getframerate() == .03
    assert result["model_validation"] == "pending protected viewer --validate"
    assert (output / "model.mjb").read_bytes() == (recording / "model.mjb").read_bytes()
    with np.load(output / "physics_trace.npz") as packaged:
        np.testing.assert_array_equal(packaged["initial_qpos"], trace["qpos"][0])
        np.testing.assert_array_equal(packaged["qpos"], trace["qpos"][1:])
        assert packaged["mocap_pos"].shape == (3, 0, 3)
    comparison = json.loads((output / "comparison_events.json").read_text())
    assert comparison["actual_events"] == [[.01, .03, 60], [.02, .03, 61]]
    assert comparison["original_events"] == [[.01, .03, 60, 1]]
    for filename in ("launch_local.sh", "viewer.py", "preview_keys.py", "preview_fingering.py",
                     "played.wav", "actual_vs_original_midi.png", "key_comparison.npz"):
        assert (output / filename).is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        assert sha(output / filename) == digest
    with pytest.raises(FileExistsError):
        module.package(recording, midi, score, output)


@pytest.mark.parametrize("field,value,match", [
    ("model_sha256", "0"*64, "model_sha256"),
    ("duration_s", .04, "duration"),
    ("physics_hz", 200., "physics_hz"),
    ("original_midi_micro_f1", 1., "F1"),
])
def test_package_rejects_stale_model_timing_or_metrics_before_writing(tmp_path, field, value, match):
    module = implementation()
    score, midi, _ = write_score_and_midi(tmp_path)
    recording = tmp_path / "recording"
    recording.mkdir()
    np.savez_compressed(recording / "physical_trace.npz", **trace_fixture())
    (recording / "model.mjb").write_bytes(b"opaque fixture")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(kind="actuator_only_ik_physical_v1", original_midi_micro_f1=.8,
                  tp=2, fp=1, fn=0, duration_s=.03, physics_hz=100.,
                  dependency_versions={"mujoco": "fixture-version"},
                  source=dict(score_sha256=sha(score), original_midi_sha256=sha(midi), archive_sha256="a"*64),
                  model_sha256=sha(recording / "model.mjb"))
    report[field] = value
    (recording / "replay_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        module.package(recording, midi, score, tmp_path / "output")
    assert not (tmp_path / "output").exists()
