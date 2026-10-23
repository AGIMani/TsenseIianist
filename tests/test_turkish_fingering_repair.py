"""Score-level repair invariants; these tests do not certify physical clearance."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location(
    'repair_turkish', Path(__file__).resolve().parents[1]/'scripts/repair_turkish_fingering.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def note(pitch, start, end, part):
    return SimpleNamespace(pitch=pitch, start_time=start, end_time=end, part=part)


def test_short_gap_reverse_is_detected_without_changing_note_times():
    notes = [note(81, 64.75, 64.86770833333333, 1), note(80, 64.875, 64.99, 2)]
    before = [(n.start_time, n.end_time) for n in notes]
    edges = repair.conflict_edges(notes, .05, .05)
    issues = repair.violations(notes, [1, 2], edges)
    assert len(issues) == 1
    assert issues[0]['index_middle_reverse']
    assert 0 < issues[0]['gap_s'] < .008
    assert [(n.start_time, n.end_time) for n in notes] == before


def test_left_hand_order_is_mirrored():
    notes = [note(50, 0, .5, 6), note(48, 0, .5, 7)]
    edges = repair.conflict_edges(notes, .05, .05)
    assert repair.violations(notes, [6, 7], edges) == []
    assert repair.violations(notes, [7, 6], edges)[0]['index_middle_reverse']


def test_one_finger_cannot_hold_two_control_grid_keys():
    notes = [note(69, 0, .1177, 1), note(71, .125, .24, 1)]
    edges = repair.conflict_edges(notes, .05, .05)
    assert repair.violations(notes, [1, 1], edges)[0]['same_finger_conflict']


def test_hands_do_not_share_symbolic_constraints():
    notes = [note(60, 0, .5, 1), note(61, 0, .5, 7)]
    assert repair.conflict_edges(notes, .05, .05) == []


def test_sufficient_release_time_has_no_transition_edge():
    notes = [note(81, 0, .1, 1), note(80, .25, .4, 2)]
    assert repair.conflict_edges(notes, .05, .05) == []


def test_changing_middle_to_ring_does_not_hide_crossing():
    notes = [note(81, 0, .1177, 1), note(80, .125, .24, 3)]
    edges = repair.conflict_edges(notes, .05, .05)
    issues = repair.violations(notes, [1, 3], edges)
    assert len(issues) == 1
    assert issues[0]['nonthumb_reverse']
    assert not issues[0]['index_middle_reverse']
