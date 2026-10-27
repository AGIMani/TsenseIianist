"""Small-array preparation contracts; never import or load MuJoCo locally."""
import importlib
import importlib.util
import hashlib
import io
import tarfile
import wave

import numpy as np
import pytest


def implementation():
    assert importlib.util.find_spec('scripts.package_preparation_replay') is not None, 'preparation adapter missing'
    return importlib.import_module('scripts.package_preparation_replay')


def fixture():
    rows = 4
    return dict(time=np.arange(rows)*.01, qpos=np.zeros((rows, 90)),
                qvel=np.zeros((rows, 90)), ctrl=np.zeros((rows, 2)),
                mocap_pos=np.zeros((rows, 0, 3)), mocap_quat=np.zeros((rows, 0, 4)),
                key_activation=np.zeros((rows, 88), bool))


def checked(trace):
    return implementation().validate_preparation_trace(
        trace, np.arange(2, 90), np.tile([0., 1.], (88, 1)), .01, 1.,
        dict(nq=90, nv=90, nu=2, nmocap=0))


def test_actual_model_addresses_determine_preparation_activation():
    trace = fixture()
    trace['qpos'][1, 0] = 5.  # Robot coordinate must not be mistaken for a key.
    trace['qpos'][2, 41] = 1.
    trace['key_activation'][2, 39] = True
    result = checked(trace)
    assert result['activated_key_samples'] == 1
    assert result['actual'][2, 39]
    assert not result['actual'][1].any()
    trace['key_activation'][2, 39] = False
    with pytest.raises(ValueError, match='key_activation'):
        checked(trace)


def test_model_shape_and_nonuniform_time_are_rejected():
    trace = fixture(); trace['qvel'] = trace['qvel'][:, :-1]
    with pytest.raises(ValueError, match='qvel'):
        checked(trace)
    trace = fixture(); trace['time'][1] += .001
    with pytest.raises(ValueError, match='uniform'):
        checked(trace)


def test_activation_frame_count_retains_multiple_key_cell_count_separately():
    trace = fixture()
    trace['qpos'][1, 41:43] = 1.
    trace['key_activation'][1, 39:41] = True
    result = checked(trace)
    assert result['activated_key_samples'] == 1
    assert result['activated_key_cell_count'] == 2


def test_preparation_offset_does_not_score_unrecorded_performance():
    module = implementation()
    original = [[0., .02, 39., 1.], [.5, .6, 40., 2.]]
    result = module.make_preparation_comparison(checked(fixture()), original)
    assert result['original_events'] == [[0., .02, 60, 1], [.5, .6, 61, 2]]
    assert result['reference_events_on_playback_axis'][0][:2] == [.03, .05]
    assert result['original_midi_start_at_playback_s'] == .03
    assert result['recorded_interval_s'] == [0., .03]
    assert result['actual_events'] == []
    assert result['performance_recorded'] is False
    assert result['physical_sample_micro_f1'] is None
    assert result['metrics'] is None


def test_zero_activation_audio_is_exact_silence_with_recorded_duration(tmp_path):
    module = implementation(); trace = fixture(); path = tmp_path/'played.wav'
    report = module.write_preparation_audio(checked(trace), path)
    assert report['silent'] is True
    assert report['event_count'] == 0
    with wave.open(str(path), 'rb') as stream:
        assert stream.getnframes()/stream.getframerate() == .03
        assert not any(stream.readframes(stream.getnframes()))


def test_claimed_archive_must_contain_the_recorded_source_hashes(tmp_path):
    path = tmp_path/'source.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        info = tarfile.TarInfo('source/module.py'); info.size = 4
        archive.addfile(info, io.BytesIO(b'code'))
    module = implementation()
    good = {'source/module.py': hashlib.sha256(b'code').hexdigest()}
    assert module.verify_source_archive(path, good)['matched_source_files'] == 1
    with pytest.raises(ValueError, match='source'):
        module.verify_source_archive(path, {'source/module.py': '0'*64})
    with pytest.raises(ValueError, match='missing'):
        module.verify_source_archive(path, {'missing.py': '0'*64})
    with pytest.raises(ValueError, match='archive SHA'):
        module.verify_source_archive(path, good, expected_sha256='0'*64)
