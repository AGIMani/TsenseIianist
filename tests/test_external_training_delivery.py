"""An external-score delivery must retain the original MIDI provenance."""
import base64
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import finalize_training_video as finalize
from plot_recorded_midi_comparison import verify_embedded_midi
from note_seq.protobuf import music_pb2


@pytest.fixture
def score():
    root = Path(__file__).parents[1] / 'training_mjlab/inputs/nocturne_op9_no2_beginner'
    proto = root / 'generated_v1/nocturne-op-9-no-2-beginner_fingered.proto'
    raw = proto.read_bytes()
    config = {'source_runtime': {'external_score': {
        'base64': base64.b64encode(raw).decode(), 'sha256': finalize.sha256(proto)}}}
    return config, root / 'original.mid'


def test_external_requires_registered_original(score, tmp_path):
    config, midi = score
    with pytest.raises(ValueError, match='requires registered'):
        finalize.comparison_reference_arguments(config, tmp_path)
    with pytest.raises(ValueError, match='SHA256 differs'):
        finalize.comparison_reference_arguments(config, tmp_path, midi, 'bad')
    assert finalize.comparison_reference_arguments(config, tmp_path, midi, finalize.sha256(midi)) == [
        '--original-midi', str(midi)]
    events, _ = verify_embedded_midi(config, midi, tmp_path)
    assert len(events) == 171


def test_changed_note_timing_is_rejected(score, tmp_path):
    config, midi = score
    embedded = config['source_runtime']['external_score']
    seq = music_pb2.NoteSequence.FromString(base64.b64decode(embedded['base64']))
    seq.notes[0].end_time += .01
    raw = seq.SerializeToString()
    import hashlib
    embedded.update(base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(AssertionError, match='Original MIDI differs'):
        verify_embedded_midi(config, midi, tmp_path)
