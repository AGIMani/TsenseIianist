"""Small synthetic event/geometry checks; no robot model or physics is loaded."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def get_planner():
    path = Path(__file__).resolve().parents[1] / "scripts/plan_model_fingering.py"
    assert path.exists(), "The exact-event geometry planner has not been implemented"
    spec = importlib.util.spec_from_file_location("model_fingering_plan", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def note(pitch, start, end, instrument=0):
    return SimpleNamespace(pitch=pitch, start_time=start, end_time=end,
                           instrument=instrument, velocity=73, part=0,
                           is_drum=False, program=0)


def geometry(z=0.0, bounds=None):
    tips = [[x, 0.0, z] for x in [0, .02, .04, .06, .08, .08, .06, .04, .02, 0]]
    targets = [[[key * .02, 0., 0.] for key in range(88)] for _ in range(10)]
    data = {"tip_xyz": tips, "key_target_xyz": targets,
            "reachable_samples": [[{"tip_xyz": tip, "action": [0.] * 51,
                                    "wrist_pitch": 0.}] for tip in tips]}
    bounds = bounds if bounds is not None else [-10., 10.]
    data["wrist_translation_bounds"] = {
        hand: {"tx": bounds, "ty": bounds} for hand in ("right", "left")}
    data["wrist_translation_axes"] = {
        hand: [[1., 0., 0.], [0., 1., 0.]] for hand in ("right", "left")}
    return data


def test_subframe_gap_uses_raw_events_and_allows_finger_reuse():
    planner = get_planner()
    # A floor/ceil control grid incorrectly makes these two five-note chords overlap.
    notes = [note(p, .001, .011) for p in range(60, 65)]
    notes += [note(p, .012, .022) for p in range(60, 65)]
    result = planner.plan_hand(notes, list(range(10)), planner.Geometry(geometry()), left=False)
    assert {a["time_seconds"] for a in result["anchors"]} == {.001, .011, .012, .022}
    assert len(set(result["assignments"][i] for i in range(5))) == 5
    assert len(set(result["assignments"][i] for i in range(5, 10))) == 5
    assert all(result["assignments"][i] == result["assignments"][i + 5] for i in range(5))


@pytest.mark.parametrize("exchange_weight", [0., 10.])
def test_held_note_keeps_finger_across_other_notes_on_and_off(exchange_weight):
    planner = get_planner()
    notes = [note(60, 0., .040), note(62, .005, .015), note(64, .020, .035)]
    result = planner.plan_hand(notes, list(range(3)), planner.Geometry(geometry()),
                               left=False, exchange_weight=exchange_weight)
    held = [a["note_fingers"]["0"] for a in result["anchors"] if "0" in a["note_fingers"]]
    assert len(held) > 2 and len(set(held)) == 1
    assert result["assignments"][0] != result["assignments"][1]
    assert result["assignments"][0] != result["assignments"][2]


def test_exact_overlap_over_five_notes_is_rejected():
    planner = get_planner()
    notes = [note(p, 0., .02) for p in range(60, 66)]
    with pytest.raises(ValueError, match="five|5"):
        planner.plan_hand(notes, list(range(6)), planner.Geometry(geometry()), left=False)


@pytest.mark.parametrize("exchange_weight", [0., 10.])
def test_thumb_is_softly_penalized_but_retained_in_each_hand(exchange_weight):
    planner = get_planner()
    for left in (False, True):
        notes = [note(60, t, t + .08, int(left)) for t in (0., .1, .2, .3)]
        result = planner.plan_hand(notes, list(range(4)), planner.Geometry(geometry()),
                                   left=left, thumb_penalty=100., exchange_weight=exchange_weight)
        thumb = 5 if left else 0
        assert list(result["assignments"].values()).count(thumb) == 1
        assert result["used_thumb"]


def test_nonthumb_order_mirrors_and_thumb_can_be_at_either_end():
    planner = get_planner()
    right = set(planner.finger_options(3, left=False))
    left = set(planner.finger_options(3, left=True))
    assert (0, 1, 2) in right and (1, 2, 0) in right
    assert (9, 8, 5) in left and (5, 9, 8) in left
    assert (2, 1, 3) not in right and (6, 7, 8) not in left


def test_geometry_reports_vertical_mismatch_and_obeys_wrist_bounds():
    planner = get_planner()
    model = planner.Geometry(geometry(z=.010, bounds=[-.01, .01]))
    poses = model.candidates((1,), (60,), left=False)
    assert poses
    for pose in poses:
        assert np.max(np.abs(pose.wrist_xy)) <= .010000001
        assert pose.max_z_error_m == pytest.approx(.010)
        assert pose.z_fallback_count == 1
        assert pose.cost > 0


def test_no_sample_cloud_is_not_silently_replaced_by_neutral_pose():
    planner = get_planner()
    data = geometry()
    data["reachable_samples"][0] = []
    with pytest.raises(ValueError, match="sample"):
        planner.Geometry(data)


def test_joint_translation_limits_use_actual_slide_axes():
    planner = get_planner()
    data = geometry()
    data["wrist_translation_axes"] = {
        hand: [[0., 1., 0.], [-1., 0., 0.]] for hand in ("right", "left")}
    data["wrist_translation_bounds"] = {
        hand: {"tx": [-.1, .1], "ty": [-.8, .8]} for hand in ("right", "left")}
    pose = planner.Geometry(data).candidates((1,), (60,), left=False)[0]
    # World shift +.76 X is joint Ty=-.76, not joint Tx=+.76.
    assert pose.residuals_m[0] < 1e-9
    assert pose.wrist_xy == pytest.approx([.76, 0.])
    assert pose.wrist_tx_ty == pytest.approx([0., -.76])


@pytest.mark.parametrize("missing", ["wrist_translation_axes", "wrist_translation_bounds"])
def test_geometry_requires_recorded_translation_contract(missing):
    planner = get_planner()
    data = geometry()
    del data[missing]
    with pytest.raises(ValueError, match="translation"):
        planner.Geometry(data)


def test_geometry_rejects_unbounded_joint_translation():
    planner = get_planner()
    data = geometry(bounds=[-float("inf"), float("inf")])
    with pytest.raises(ValueError, match="bounds"):
        planner.Geometry(data)


def test_note_release_does_not_force_a_new_wrist_anchor():
    planner = get_planner()
    data = geometry()
    for targets in data["key_target_xyz"]:
        targets[40][0] = .81  # Three cm separation, between sampled two/four cm spans.
    result = planner.plan_hand([note(60, 0., .3), note(61, 0., .1)], [0, 1],
                               planner.Geometry(data), left=False, require_thumb=False,
                               thumb_penalty=100.)
    assert result["anchors"][1]["wrist_tx_ty_m"] == result["anchors"][0]["wrist_tx_ty_m"]


def test_excess_reach_cost_is_not_erased_by_short_note_duration():
    planner = get_planner()
    notes = [note(60, 0., .1), note(60, .101, .102), note(64, .101, .102)]
    loose = planner.plan_hand(notes, [0, 1, 2], planner.Geometry(geometry()), left=False,
                              require_thumb=False, thumb_penalty=10., movement_weight=0., reach_weight=0.)
    tight = planner.plan_hand(notes, [0, 1, 2], planner.Geometry(geometry()), left=False,
                              require_thumb=False, thumb_penalty=10., movement_weight=0., reach_weight=100.)
    assert loose["statistics"]["max_target_residual_m"] > .009
    assert tight["statistics"]["max_target_residual_m"] < 1e-8
    assert tight["assignments"][1] == 0  # Thumb is useful for the wide chord.


def test_onset_can_keep_prior_anchor_as_a_continuity_candidate():
    planner = get_planner()
    data = geometry()
    for targets in data["key_target_xyz"]:
        targets[40][0] = .79
    result = planner.plan_hand([note(60, 0., .1), note(61, .1, .2)], [0, 1],
                               planner.Geometry(data), left=False, require_thumb=False,
                               thumb_penalty=100., movement_weight=10000., reach_weight=0.)
    assert result["anchors"][1]["wrist_tx_ty_m"] == result["anchors"][0]["wrist_tx_ty_m"]


def test_unresolved_sampled_chord_exposes_exact_time_and_best_residual():
    planner = get_planner()
    notes = [note(p, .12345, .2) for p in (60, 68, 76, 84)]
    result = planner.plan_hand(notes, [0, 1, 2, 3], planner.Geometry(geometry()), left=False)
    diagnosis = result["reach_diagnostics"][0]
    assert diagnosis["time_seconds"] == .12345
    assert diagnosis["pitches"] == [60, 68, 76, 84]
    assert diagnosis["best_candidate_max_residual_m"] > .1
    assert diagnosis["no_candidate_within_tolerance"]
    assert diagnosis["selected_max_residual_m"] >= diagnosis["best_candidate_max_residual_m"]


def identical_finger_geometry():
    """Remove reach preferences so recovery history determines tied choices."""
    data = geometry()
    data["tip_xyz"] = [[0., 0., 0.] for _ in range(10)]
    data["reachable_samples"] = [[{"tip_xyz": [0., 0., 0.], "action": [0.] * 51,
                                  "wrist_pitch": 0.}] for _ in range(10)]
    return data


@pytest.mark.parametrize("pitches", [(60, 61), (60, 60)])
def test_short_gap_exchange_cost_prefers_idle_finger_including_repeated_key(pitches):
    planner = get_planner()
    notes = [note(pitches[0], 0., .1), note(pitches[1], .105, .2)]
    common = dict(left=False, require_thumb=False, thumb_penalty=200., movement_weight=0.)
    baseline = planner.plan_hand(notes, [0, 1], planner.Geometry(identical_finger_geometry()),
                                 exchange_weight=0., recovery_time=.05, **common)
    recovered = planner.plan_hand(notes, [0, 1], planner.Geometry(identical_finger_geometry()),
                                  exchange_weight=10., recovery_time=.05, **common)
    assert baseline["assignments"][0] == baseline["assignments"][1]
    assert recovered["assignments"][0] != recovered["assignments"][1]
    assert recovered["costs"]["finger_exchange"] == pytest.approx(0.)
    assert recovered["statistics"]["same_finger_gaps_below_50ms"] == 0
    assert [(n.start_time, n.end_time) for n in notes] == [(0., .1), (.105, .2)]


def test_recent_finger_history_survives_empty_hand_and_equivalent_pose_merging():
    planner = get_planner()
    notes = [note(60, t, t + .01) for t in (0., .02, .04)]
    result = planner.plan_hand(notes, [0, 1, 2], planner.Geometry(identical_finger_geometry()),
                               left=False, require_thumb=False, thumb_penalty=200.,
                               movement_weight=0., exchange_weight=10., recovery_time=.05)
    assert len(set(result["assignments"].values())) == 3
    assert result["costs"]["finger_exchange"] == pytest.approx(0.)


def test_finger_reuse_is_free_after_preferred_recovery_time():
    planner = get_planner()
    notes = [note(60, 0., .1), note(60, .16, .2)]
    result = planner.plan_hand(notes, [0, 1], planner.Geometry(identical_finger_geometry()),
                               left=False, require_thumb=False, thumb_penalty=200.,
                               movement_weight=0., exchange_weight=10., recovery_time=.05)
    assert result["assignments"][0] == result["assignments"][1]
    assert result["costs"]["finger_exchange"] == pytest.approx(0.)


def test_exchange_cost_is_soft_when_all_fingers_need_to_be_reused():
    planner = get_planner()
    notes = [note(p, 0., .1) for p in range(60, 65)]
    notes += [note(p, .105, .2) for p in range(60, 65)]
    result = planner.plan_hand(notes, list(range(10)), planner.Geometry(geometry()),
                               left=False, exchange_weight=10., recovery_time=.05)
    assert len(result["assignments"]) == 10
    assert result["costs"]["finger_exchange"] == pytest.approx(5 * 10 * .9 ** 2)
    assert result["statistics"]["same_finger_gaps_below_50ms"] == 5


def test_fixed_note_constraint_preserves_hold_and_replans_short_gap_neighbor():
    planner = get_planner()
    notes = [note(60, 0., .1), note(60, .105, .3), note(64, .15, .2)]
    result = planner.plan_hand(notes, [0, 1, 2], planner.Geometry(identical_finger_geometry()),
                               left=False, require_thumb=False, thumb_penalty=200.,
                               movement_weight=0., exchange_weight=500., recovery_time=.05,
                               fixed_assignments={1: 1})
    assert result["assignments"][1] == 1
    assert result["assignments"][0] != 1
    assert result["assignments"][2] != 1
    assert all(a["note_fingers"]["1"] == 1 for a in result["anchors"] if "1" in a["note_fingers"])


def test_conflicting_fixed_assignments_fail_instead_of_silently_relaxing():
    planner = get_planner()
    notes = [note(60, 0., .3), note(64, .1, .2)]
    with pytest.raises(ValueError, match="assignment"):
        planner.plan_hand(notes, [0, 1], planner.Geometry(identical_finger_geometry()),
                           left=False, require_thumb=False, fixed_assignments={0: 1, 1: 1})


@pytest.mark.parametrize("kwargs", [{"exchange_weight": -1.}, {"exchange_weight": float("nan")},
                                     {"recovery_time": 0.}, {"recovery_time": float("inf")}])
def test_invalid_exchange_contract_rejected(kwargs):
    planner = get_planner()
    with pytest.raises(ValueError, match="finite|recovery"):
        planner.plan_hand([note(60, 0., .1)], [0], planner.Geometry(geometry()),
                           left=False, **kwargs)


@pytest.mark.parametrize("syntax", ["proto2", "proto3"])
def test_proto_copy_preserves_all_fields_except_fingering(syntax):
    planner = get_planner()
    # A tiny real protobuf exercises unknown-to-the-planner sequence metadata.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    file = descriptor_pb2.FileDescriptorProto(name="planning_test.proto", package="plan_test", syntax=syntax)
    n = file.message_type.add(name="Note")
    for number, name in enumerate(("part", "pitch", "velocity"), 1):
        n.field.add(name=name, number=number, type=5, label=1)
    n.field.add(name="start_time", number=4, type=1, label=1)
    seq = file.message_type.add(name="Sequence")
    seq.field.add(name="notes", number=1, type=11, label=3, type_name=".plan_test.Note")
    seq.field.add(name="title", number=2, type=9, label=1)
    seq.field.add(name="tempo", number=3, type=1, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("plan_test.Sequence"))
    source = cls(title="original metadata", tempo=137.25)
    source.notes.add(part=9, pitch=62, velocity=73, start_time=.00123)
    before = source.SerializeToString(deterministic=True)
    updated = planner.with_fingering(source, {0: 2})
    assert updated.notes[0].part == 2
    updated.notes[0].part = 9
    assert updated.SerializeToString(deterministic=True) == before
    assert source.SerializeToString(deterministic=True) == before
