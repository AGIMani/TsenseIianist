"""CPU-only score integrity tests; never import the simulation package."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/fingering_editor_data.py"
SPEC = importlib.util.spec_from_file_location("editor_data", PATH)
editor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(editor)


def source(tmp_path):
    seq = editor.Sequence()
    seq.total_time = 3
    seq.tempos.add(time=0, qpm=120)
    seq.control_changes.add(time=2.5, control_number=64, control_value=127)
    seq.notes.add(pitch=60, start_time=.1, end_time=.5, part=0, velocity=80, instrument=3)
    seq.notes.add(pitch=64, start_time=.5, end_time=1., part=1, velocity=70)
    path = tmp_path / "original.proto"
    # Unknown field 1000, varint 7 must survive edits too.
    path.write_bytes(seq.SerializeToString() + b"\xc0\x3e\x07")
    return path


def test_roundtrip_and_explicit_edits_preserve_every_other_field(tmp_path):
    path = source(tmp_path)
    score = editor.Score(path)
    assert score.output_bytes() == path.read_bytes()
    score.edit([0], part=9, start=.2, end=4)
    result = editor.Sequence.FromString(score.output_bytes())
    original = editor.Sequence.FromString(path.read_bytes())
    assert result.total_time == 4
    assert (result.notes[0].part, result.notes[0].start_time, result.notes[0].end_time) == (9, .2, 4)
    result.notes[0].CopyFrom(original.notes[0])
    result.total_time = original.total_time
    assert result == original
    score.undo()
    assert score.output_bytes() == path.read_bytes()
    score.redo()
    assert score.seq.notes[0].part == 9
    output = tmp_path / "edited.proto"
    score.save(output)
    assert output.exists() and not score.dirty
    assert path.read_bytes().endswith(b"\xc0\x3e\x07")
    with pytest.raises(FileExistsError):
        score.save(output)
    with pytest.raises(ValueError):
        score.save(path)


def test_validation_atomicity_conflicts_and_half_open_times(tmp_path):
    score = editor.Score(source(tmp_path))
    assert score.conflicts() == set()
    assert [i for i, n in score.active(.5)] == [1]
    score.edit([1], part=0, start=.4)
    assert score.conflicts() == {0, 1}
    before = score.output_bytes()
    with pytest.raises(ValueError):
        score.edit([0, 1], start=.75)
    assert score.output_bytes() == before
    for options in (dict(part=10), dict(start=float("nan")), dict(end=0), dict(start=-1)):
        with pytest.raises(ValueError):
            score.edit([0], **options)
    assert score.output_bytes() == before


def test_quantized_timing_is_not_silently_corrupted(tmp_path):
    path = source(tmp_path)
    seq = editor.Sequence.FromString(path.read_bytes())
    seq.quantization_info.steps_per_quarter = 4
    path.write_bytes(seq.SerializeToString())
    score = editor.Score(path)
    score.edit([0], part=5)
    with pytest.raises(ValueError, match="量化"):
        score.edit([0], start=.2)
    score.seq.notes[0].pitch = 61
    with pytest.raises(ValueError, match="范围以外"):
        score.output_bytes()


def test_supplied_example_is_lossless_and_has_no_conflicting_fingers():
    path = PATH.parents[1] / "results/exper024/input_sources/015-5_fingered.proto"
    if not path.exists():
        pytest.skip("User example is local and intentionally untracked")
    score = editor.Score(path)
    assert len(score.seq.notes) == 269
    assert score.duration == pytest.approx(24.0521)
    assert not score.conflicts()
    assert score.output_bytes() == path.read_bytes()


def test_previous_fingering_uses_score_time_and_follows_edit_undo(tmp_path):
    score = editor.Score(source(tmp_path))
    assert score.previous(0) == []
    assert score.previous(.2) == []
    assert [i for i, _ in score.previous(.5)] == [0]
    assert [i for i, _ in score.previous(.9)] == [0]
    assert [i for i, _ in score.previous(1.)] == [1]
    # Seeking backwards does not retain the most recently viewed frame.
    assert score.previous(.2) == []
    score.edit([0], part=9)
    assert score.previous(.6)[0][1].part == 9
    score.undo()
    assert score.previous(.6)[0][1].part == 0
    assert score.output_bytes() == score.original


def test_previous_chord_keeps_only_released_notes(tmp_path):
    seq = editor.Sequence()
    seq.notes.add(pitch=60, part=0, start_time=.1, end_time=2.)
    seq.notes.add(pitch=64, part=1, start_time=.1, end_time=.3)
    seq.notes.add(pitch=67, part=2, start_time=.5, end_time=.9)
    path = tmp_path/'chord.proto'
    path.write_bytes(seq.SerializeToString())
    score = editor.Score(path)
    assert [i for i, _ in score.previous(.6)] == [1]
    assert [i for i, _ in score.previous(.95)] == [2]
    assert score.previous(.2) == []
    assert score.output_bytes() == path.read_bytes()


def test_upcoming_group_uses_exact_onsets_and_updated_fingering(tmp_path):
    score = editor.Score(source(tmp_path))
    assert [i for i, _ in score.upcoming(0)] == [0]
    assert [i for i, _ in score.upcoming(.1)] == [1]
    assert [i for i, _ in score.upcoming(.49)] == [1]
    assert score.upcoming(.5) == []
    assert score.upcoming(3) == []
    score.edit([1], part=9)
    assert score.upcoming(.2)[0][1].part == 9
    score.undo()
    assert score.upcoming(.2)[0][1].part == 1
    assert score.output_bytes() == score.original
