"""Registered original events must exactly match the frozen score, including releases."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from note_seq.protobuf import music_pb2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from finalize_training_video import comparison_reference_arguments


def test_reference_retains_exact_registered_times_and_rejects_drift(tmp_path):
    seq = music_pb2.NoteSequence()
    seq.notes.add(pitch=60,start_time=.123456789,end_time=.995,velocity=90,part=1)
    raw=seq.SerializeToString(); digest=hashlib.sha256(raw).hexdigest()
    score=tmp_path/'score.proto';score.write_bytes(raw)
    config=dict(source_runtime=dict(external_score=dict(base64=base64.b64encode(raw).decode(),sha256=digest,suffix='.proto')))
    (tmp_path/'inputs').mkdir();(tmp_path/'inputs/config.json').write_text(json.dumps(config))
    args=comparison_reference_arguments(config,tmp_path,registered_score=score,registered_score_sha256=digest)
    ref=json.loads(Path(args[1]).read_text())
    assert ref['events']==[[.123456789,.995,60]]
    assert ref['reference_kind']=='registered_note_sequence'
    assert Path(ref['midi_path']).read_bytes()==raw
    seq.notes[0].end_time=1.;score.write_bytes(seq.SerializeToString())
    with pytest.raises(ValueError,match='differs'):
        comparison_reference_arguments(config,tmp_path,registered_score=score,registered_score_sha256=digest)


def test_nonzero_midi_origin_matches_trimmed_training_timeline():
    from plot_recorded_midi_comparison import align_external_events
    seq=music_pb2.NoteSequence()
    seq.notes.add(pitch=71,start_time=.0049,end_time=.1231,part=3)
    rows,origin=align_external_events([[.0049,.1231,71]],seq.SerializeToString(),.25)
    assert origin==.0049
    assert rows[0][0]==.25
    assert rows[0][1]==pytest.approx(.3682)
    assert rows[0][2]==71


def test_delivery_requires_audio_fingering_and_highlights():
    from finalize_training_video import validate_display_requirements
    plan=dict(audio=True,fingering=True,key_highlight=True)
    validate_display_requirements(plan)
    for name in ('audio','fingering','key_highlight'):
        with pytest.raises(ValueError,match='missing'):
            validate_display_requirements(dict(plan,**{name:False}))
