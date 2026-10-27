#!/usr/bin/env python3
"""Plan fingering from exact MIDI events and remotely sampled A3.2 geometry.

This module never constructs a MuJoCo model.  Search is exact over the retained
finite candidate graph; sampled reach, its fitted wrist anchors and soft motion
costs are approximations, not a physical feasibility certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class Pose:
    wrist_xy: np.ndarray
    wrist_tx_ty: np.ndarray
    wrist_pitch: float
    cost: float
    residuals_m: tuple[float, ...]
    max_z_error_m: float
    z_fallback_count: int
    sample_indices: tuple[int, ...]


def finger_options(count: int, *, left: bool):
    """Natural nonthumb order; the thumb remains allowed on either side."""
    if not 1 <= count <= 5:
        raise ValueError("Each hand needs one to five active notes")
    nonthumb = (9, 8, 7, 6) if left else (1, 2, 3, 4)
    thumb = 5 if left else 0
    result = list(itertools.combinations(nonthumb, count))
    for fingers in itertools.combinations(nonthumb, count - 1):
        result.append((thumb,) + fingers)
        if fingers:
            result.append(fingers + (thumb,))
    return tuple(result)


class Geometry:
    """Fit shared Tx/Ty against samples having one common wrist pitch."""

    def __init__(self, data, *, z_tolerance_m=.003, max_poses=3):
        if not math.isfinite(z_tolerance_m) or z_tolerance_m <= 0 or max_poses < 1:
            raise ValueError("Positive geometry tolerance and candidate count required")
        self.data = data
        self.tips = np.asarray(data["tip_xyz"], dtype=float)
        self.targets = np.asarray(data["key_target_xyz"], dtype=float)
        if self.tips.shape != (10, 3) or self.targets.shape != (10, 88, 3):
            raise ValueError("Geometry needs tip_xyz[10,3] and key_target_xyz[10,88,3]")
        if not np.isfinite(self.tips).all() or not np.isfinite(self.targets).all():
            raise ValueError("Geometry coordinates must be finite")
        raw = data["reachable_samples"]
        if len(raw) != 10 or any(not samples for samples in raw):
            raise ValueError("Every finger needs a nonempty reachable sample cloud")
        self.samples = []
        self.pitch_samples = []
        for samples in raw:
            xyz = np.asarray([s["tip_xyz"] for s in samples], dtype=float)
            pitch = np.asarray([s["wrist_pitch"] for s in samples], dtype=float)
            actions = np.asarray([s["action"] for s in samples], dtype=float)
            if xyz.shape != (len(samples), 3) or actions.shape != (len(samples), 51):
                raise ValueError("Each sample needs tip_xyz[3] and action[51]")
            if not all(np.isfinite(a).all() for a in (xyz, pitch, actions)):
                raise ValueError("Reachable samples must be finite")
            self.samples.append(xyz)
            self.pitch_samples.append({float(p): np.flatnonzero(pitch == p)
                                       for p in np.unique(pitch)})
        self.bounds, self.axes, self.inverse_axes = {}, {}, {}
        for hand in ("right", "left"):
            if any(hand not in data.get(field, {}) for field in
                   ("wrist_translation_axes", "wrist_translation_bounds")):
                raise ValueError(f"Recorded wrist translation axes and bounds required for {hand}")
            axes = np.asarray(data["wrist_translation_axes"][hand], dtype=float).T
            if (axes.shape != (3, 2) or not np.isfinite(axes).all()
                    or np.linalg.matrix_rank(axes) != 2 or np.max(np.abs(axes[2])) > 1e-9):
                raise ValueError("Planner requires two independent horizontal slide axes")
            self.axes[hand] = axes
            self.inverse_axes[hand] = np.linalg.pinv(axes)
            bounds = data["wrist_translation_bounds"][hand]
            if not isinstance(bounds, dict) or not {"tx", "ty"} <= bounds.keys():
                raise ValueError("Invalid wrist translation bounds")
            b = np.asarray([bounds["tx"], bounds["ty"]], dtype=float)
            if b.shape != (2, 2) or not np.isfinite(b).all() or np.any(b[:, 0] > b[:, 1]):
                raise ValueError("Invalid wrist translation bounds")
            self.bounds[hand] = b
        self.z_tolerance = z_tolerance_m
        self.max_poses = max_poses
        self._clouds = {}
        self._poses = {}

    def _cloud(self, finger, pitch, wrist_pitch):
        key = (finger, pitch, wrist_pitch)
        if key not in self._clouds:
            target = self.targets[finger, pitch - 21]
            indices = self.pitch_samples[finger][wrist_pitch]
            xyz = self.samples[finger][indices]
            dz = np.abs(xyz[:, 2] - target[2])
            keep = dz <= self.z_tolerance
            fallback = not bool(keep.any())
            if fallback:
                # Keep only the nearest height band, explicitly penalizing its error.
                keep = dz <= float(dz.min()) + .0001
            indices, xyz = indices[keep], xyz[keep]
            hand = "left" if finger >= 5 else "right"
            anchors = (target - xyz) @ self.inverse_axes[hand].T
            posture = np.sum(((xyz - self.tips[finger]) / .05) ** 2, axis=1)
            self._clouds[key] = (indices, xyz, anchors, posture, target, fallback)
        return self._clouds[key]

    def pose_for_samples(self, fingers, pitches, wrist_tx_ty, wrist_pitch, sample_indices, *, left):
        """Evaluate known coupled samples at a fixed common wrist translation."""
        anchor = np.asarray(wrist_tx_ty, dtype=float)
        axes = self.axes["left" if left else "right"]
        errors, z_errors, posture = [], [], []
        for finger, pitch, index in zip(fingers, pitches, sample_indices):
            xyz = self.samples[finger][index]
            error = self.targets[finger, pitch - 21] - xyz - axes @ anchor
            errors.append(float(np.linalg.norm(error)))
            z_errors.append(float(abs(error[2])))
            posture.append(float(np.sum(((xyz - self.tips[finger]) / .05) ** 2)))
        cost = (float(np.mean(np.square(np.asarray(errors) / self.z_tolerance)))
                + .05 * float(np.mean(posture)) + .1 * (wrist_pitch / .5) ** 2)
        return Pose((axes @ anchor)[:2], anchor, wrist_pitch, cost, tuple(errors), max(z_errors),
                    sum(z > self.z_tolerance for z in z_errors), tuple(sample_indices))

    def at_anchor(self, fingers, pitches, previous_pose, *, left):
        """Nearest sampled fingers at a predecessor wrist pose (no forced jump)."""
        indices = []
        for finger, pitch in zip(fingers, pitches):
            if previous_pose.wrist_pitch not in self.pitch_samples[finger]:
                return None
            sample_indices, xyz, xy, posture, target, _ = self._cloud(
                finger, pitch, previous_pose.wrist_pitch)
            objective = (np.sum((xy - previous_pose.wrist_tx_ty) ** 2, axis=1)
                         + (xyz[:, 2] - target[2]) ** 2
                         + .05 * self.z_tolerance ** 2 * posture)
            indices.append(int(sample_indices[int(np.argmin(objective))]))
        return self.pose_for_samples(fingers, pitches, previous_pose.wrist_tx_ty,
                                     previous_pose.wrist_pitch, indices, left=left)

    def candidates(self, fingers, pitches, *, left):
        fingers, pitches = tuple(fingers), tuple(pitches)
        key = (fingers, pitches, left)
        if key in self._poses:
            return self._poses[key]
        if len(fingers) != len(pitches) or not fingers or any(not 21 <= p <= 108 for p in pitches):
            raise ValueError("Invalid finger/key candidate")
        common_pitch = set(self.pitch_samples[fingers[0]])
        for finger in fingers[1:]:
            common_pitch.intersection_update(self.pitch_samples[finger])
        if not common_pitch:
            raise ValueError("Active fingers have no samples at a common wrist pitch")
        bounds = self.bounds["left" if left else "right"]
        axes = self.axes["left" if left else "right"]
        poses = []
        for wrist_pitch in sorted(common_pitch):
            clouds = [self._cloud(f, p, wrist_pitch) for f, p in zip(fingers, pitches)]
            medians = [np.median(c[2], axis=0) for c in clouds]
            preferred = [c[2][int(np.argmin(c[3]))] for c in clouds]
            seeds = [np.mean(medians, axis=0), np.mean(preferred, axis=0), *medians]
            # Alternate anchors matter for future transitions, especially single notes.
            for quantile in (.15, .85):
                seeds.append(np.mean([c[2][np.argsort(c[2][:, 0])[
                    int(quantile * (len(c[2]) - 1))]] for c in clouds], axis=0))
            for seed in seeds:
                anchor = np.clip(seed, bounds[:, 0], bounds[:, 1])
                selected = []
                for _ in range(6):
                    selected = []
                    for _, xyz, xy, posture, target, _ in clouds:
                        objective = (np.sum((xy - anchor) ** 2, axis=1)
                                     + (xyz[:, 2] - target[2]) ** 2
                                     + .05 * self.z_tolerance ** 2 * posture)
                        selected.append(int(np.argmin(objective)))
                    new_anchor = np.clip(np.mean([c[2][j] for c, j in zip(clouds, selected)], axis=0),
                                         bounds[:, 0], bounds[:, 1])
                    if np.linalg.norm(new_anchor - anchor) < 1e-8:
                        anchor = new_anchor
                        break
                    anchor = new_anchor
                errors, z_errors, indices, posture_cost = [], [], [], []
                for c, j in zip(clouds, selected):
                    sample_indices, xyz, _, posture, target, _ = c
                    error = target - xyz[j] - axes @ anchor
                    errors.append(float(np.linalg.norm(error)))
                    z_errors.append(float(abs(error[2])))
                    indices.append(int(sample_indices[j]))
                    posture_cost.append(float(posture[j]))
                cost = (float(np.mean(np.square(np.asarray(errors) / self.z_tolerance)))
                        + .05 * float(np.mean(posture_cost)) + .1 * (wrist_pitch / .5) ** 2)
                poses.append(Pose((axes @ anchor)[:2], anchor, wrist_pitch, cost, tuple(errors), max(z_errors),
                                  sum(c[5] for c in clouds), tuple(indices)))
        # Retain the best geometrically distinct alternatives, no stochastic search.
        chosen = []
        for pose in sorted(poses, key=lambda p: (p.cost, p.wrist_pitch, *p.wrist_xy)):
            if any(np.linalg.norm(pose.wrist_xy - old.wrist_xy) < .006
                   and abs(pose.wrist_pitch - old.wrist_pitch) < .2 for old in chosen):
                continue
            chosen.append(pose)
            if len(chosen) >= self.max_poses:
                break
        self._poses[key] = tuple(chosen)
        return self._poses[key]


@dataclass
class State:
    time: float
    mapping: dict[int, int]
    ordered_ids: tuple[int, ...]
    pose: Pose
    used_thumb: bool
    geometry_cost: float
    reach_cost: float
    thumb_cost: float
    movement_cost: float
    parent: State | None
    exchange_cost: float = 0.
    finger_history: tuple[float, ...] = ()

    @property
    def cost(self):
        return (self.geometry_cost + self.reach_cost + self.thumb_cost
                + self.movement_cost + self.exchange_cost)


def _events(notes, indices):
    events = {}
    for i in indices:
        note = notes[i]
        start, end = float(note.start_time), float(note.end_time)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid note interval at index {i}")
        if not 21 <= note.pitch <= 108 or getattr(note, "is_drum", False):
            raise ValueError(f"Expected piano note at index {i}")
        events.setdefault(start, [[], []])[0].append(i)
        events.setdefault(end, [[], []])[1].append(i)
    return events


def plan_hand(notes, indices, geometry, *, left, thumb_penalty=6., movement_weight=500.,
              pitch_movement_weight=.05, reach_weight=100., require_thumb=True,
              exchange_weight=0., recovery_time=.05, fixed_assignments=None):
    """Exact event occupancy and optimal path over the retained geometric candidates.

    Baseline geometry cost is integrated over each active state. Excess beyond
    the geometry tolerance has an additional squared per-event cost, so tiny
    note durations cannot erase bad reach. Movement cost is weight * distance^2
    / actual elapsed seconds. Thumb cost is paid once per thumb note onset.
    Optional exchange cost is weight * max(0, 1 - gap / recovery_time)^2
    per reused-finger onset, including same-key repetitions. This is a soft
    scheduling preference, not a hardware recovery time or physical guarantee.
    """
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("A hand needs distinct nonempty note indices")
    fixed_assignments = dict(fixed_assignments or {})
    if any(type(i) is not int or i not in indices or type(f) is not int
           or not (5 if left else 0) <= f < (10 if left else 5)
           for i, f in fixed_assignments.items()):
        raise ValueError("Fixed assignments must map this hand's note indices to its finger ids")
    for value in (thumb_penalty, movement_weight, pitch_movement_weight, reach_weight, exchange_weight):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Planning costs must be finite and nonnegative")
    if not math.isfinite(recovery_time) or recovery_time <= 0:
        raise ValueError("A finite positive recovery_time is required")
    events = _events(notes, indices)
    times = sorted(events)
    thumb = 5 if left else 0
    def reach_cost(pose):
        excess = np.maximum(0., np.asarray(pose.residuals_m) - geometry.z_tolerance)
        return reach_weight * float(np.mean((excess / geometry.z_tolerance) ** 2))
    def recent_history(history, at):
        # After recovery_time a released note cannot affect any future cost.
        return tuple(end if end > at - recovery_time else -1. for end in history)
    def exchange_transition(history, mapping, starts, at):
        if not exchange_weight:
            return (), 0.
        updated = list(history or (-1.,) * 5)
        cost = 0.
        for i in starts:
            finger = mapping[i] % 5
            if updated[finger] >= 0:
                gap = at - updated[finger]
                cost += exchange_weight * max(0., 1. - gap / recovery_time) ** 2
            updated[finger] = float(notes[i].end_time)
        return tuple(updated), cost
    active, previous, diagnostics = set(), [], {}
    max_active, max_states = 0, 0
    for event_index, at in enumerate(times):
        starts, ends = events[at]
        active.difference_update(ends)  # [onset, offset): release before simultaneous onset.
        active.update(starts)
        if len(active) > 5:
            raise ValueError(f"More than five overlapping notes in {'left' if left else 'right'} hand at {at:.12g}s")
        max_active = max(max_active, len(active))
        duration = times[event_index + 1] - at if event_index + 1 < len(times) else 0.
        if not active:
            # A free hand retains its last pose; do not invent a return to the origin.
            best = {}
            for prev in previous:
                history = recent_history(prev.finger_history, at)
                state_key = (tuple(prev.pose.wrist_xy), prev.pose.wrist_pitch, prev.used_thumb, history)
                state = State(at, {}, (), prev.pose, prev.used_thumb, prev.geometry_cost,
                              prev.reach_cost, prev.thumb_cost, prev.movement_cost, prev,
                              prev.exchange_cost, history)
                if state_key not in best or state.cost < best[state_key].cost:
                    best[state_key] = state
            previous = list(best.values())
            continue
        ordered = tuple(sorted(active, key=lambda i: (notes[i].pitch, i)))
        pitches = tuple(int(notes[i].pitch) for i in ordered)
        if not starts:
            # Releasing a finger does not require wrist repositioning. Keep the
            # actual predecessor pose and its remaining samples, not a new Top-K
            # chord fit which can jump within a submillisecond release interval.
            current = []
            for prev in previous:
                sample_by_note = dict(zip(prev.ordered_ids, prev.pose.sample_indices))
                mapping = {i: prev.mapping[i] for i in ordered}
                pose = geometry.pose_for_samples(tuple(mapping[i] for i in ordered), pitches,
                    prev.pose.wrist_tx_ty, prev.pose.wrist_pitch,
                    tuple(sample_by_note[i] for i in ordered), left=left)
                current.append(State(at, mapping, ordered, pose, prev.used_thumb,
                    prev.geometry_cost + duration * pose.cost, prev.reach_cost + reach_cost(pose),
                    prev.thumb_cost, prev.movement_cost, prev, prev.exchange_cost,
                    recent_history(prev.finger_history, at)))
            previous = current
            continue
        retained = tuple(i for i in ordered if i not in starts)
        predecessor_groups = {}
        for prev in previous:
            group = (tuple(prev.mapping[i] for i in retained), prev.used_thumb,
                     recent_history(prev.finger_history, at))
            predecessor_groups.setdefault(group, []).append(prev)
        group_arrays = {}
        for group, states in predecessor_groups.items():
            group_arrays[group] = (np.asarray([s.cost for s in states]),
                                   np.asarray([s.pose.wrist_xy for s in states]),
                                   np.asarray([s.pose.wrist_pitch for s in states]))
        current = []
        event_pose_residuals = []
        elapsed = at - times[event_index - 1] if event_index else None
        for fingers in finger_options(len(ordered), left=left):
            mapping = dict(zip(ordered, fingers))
            if any(i in fixed_assignments and fixed_assignments[i] != mapping[i] for i in ordered):
                continue
            new_thumb = sum(mapping[i] == thumb for i in starts)
            kept = tuple(mapping[i] for i in retained)
            poses = list(geometry.candidates(fingers, pitches, left=left))
            # Bounded continuity alternatives: the cheapest compatible predecessor
            # for each thumb-use state. They preserve a good existing hand position
            # that a local geometry-only Top-K list would otherwise discard.
            for already_used in (False, True):
                compatible = [state for group, states in predecessor_groups.items()
                              if group[:2] == (kept, already_used) for state in states]
                if compatible:
                    prev = min(compatible, key=lambda s: s.cost)
                    continuity = geometry.at_anchor(fingers, pitches, prev.pose, left=left)
                    if continuity is not None and not any(
                        np.linalg.norm(continuity.wrist_xy - p.wrist_xy) < 1e-8
                        and abs(continuity.wrist_pitch - p.wrist_pitch) < 1e-8 for p in poses):
                        poses.append(continuity)
            for pose in poses:
                event_pose_residuals.append(max(pose.residuals_m))
                local_geometry = duration * pose.cost
                local_reach = reach_cost(pose)
                local_thumb = thumb_penalty * new_thumb
                if not previous:
                    history, exchange = exchange_transition((), mapping, starts, at)
                    current.append(State(at, mapping, ordered, pose, bool(new_thumb),
                                         local_geometry, local_reach, local_thumb, 0., None,
                                         exchange, history))
                    continue
                winners = {}
                for already_used in (False, True):
                    for group, states in predecessor_groups.items():
                        if group[:2] != (kept, already_used):
                            continue
                        history, exchange = exchange_transition(group[2], mapping, starts, at)
                        scores, xy, pitch = group_arrays[group]
                        motion = (movement_weight * np.sum((xy - pose.wrist_xy) ** 2, axis=1)
                                  + pitch_movement_weight * (pitch - pose.wrist_pitch) ** 2) / elapsed
                        j = int(np.argmin(scores + motion))
                        prev = states[j]
                        used = already_used or bool(new_thumb)
                        state = State(at, mapping, ordered, pose, used,
                                      prev.geometry_cost + local_geometry,
                                      prev.reach_cost + local_reach,
                                      prev.thumb_cost + local_thumb,
                                      prev.movement_cost + float(motion[j]), prev,
                                      prev.exchange_cost + exchange, history)
                        winner_key = (used, history)
                        if winner_key not in winners or state.cost < winners[winner_key].cost:
                            winners[winner_key] = state
                current.extend(winners.values())
        if not current:
            raise ValueError(f"No retained-note-compatible assignment satisfying fixed constraints at {at:.12g}s")
        diagnostics[at] = {"time_seconds": at, "hand": "left" if left else "right",
                           "pitches": list(pitches), "note_indices": list(ordered),
                           "best_candidate_max_residual_m": min(event_pose_residuals),
                           "no_candidate_within_tolerance": min(event_pose_residuals) > geometry.z_tolerance,
                           "candidate_count": len(event_pose_residuals)}
        previous = current
        max_states = max(max_states, len(current))
    finals = [s for s in previous if s.used_thumb or not require_thumb]
    if not finals:
        raise ValueError("No path includes a useful thumb note; no post-hoc reassignment was made")
    final = min(finals, key=lambda s: s.cost)
    path = []
    node = final
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    assignments, anchors = {}, []
    for state in path:
        if state.time in diagnostics:
            diagnostics[state.time]["selected_max_residual_m"] = max(state.pose.residuals_m)
            diagnostics[state.time]["selected_fingers"] = [state.mapping[i] for i in state.ordered_ids]
        for i, finger in state.mapping.items():
            if i in assignments and assignments[i] != finger:
                raise AssertionError("A sustained note changed finger")
            assignments[i] = finger
        active_pose = bool(state.mapping)
        anchors.append({"time_seconds": state.time, "hand": "left" if left else "right",
                        "wrist_xy_m": state.pose.wrist_xy.tolist(),
                        "wrist_tx_ty_m": state.pose.wrist_tx_ty.tolist(),
                        "wrist_pitch_rad": state.pose.wrist_pitch,
                        "note_fingers": {str(i): f for i, f in state.mapping.items()},
                        "sample_indices": {str(i): j for i, j in zip(state.ordered_ids, state.pose.sample_indices)},
                        "residuals_m": {str(i): r for i, r in zip(state.ordered_ids, state.pose.residuals_m)},
                        "max_z_error_m": state.pose.max_z_error_m if active_pose else 0.,
                        "z_fallback_count": state.pose.z_fallback_count if active_pose else 0})
    if set(assignments) != set(indices):
        raise AssertionError("Missing note assignments")
    validate_assignments(notes, assignments)
    residuals = [r for a in anchors for r in a["residuals_m"].values()]
    velocities = [float(np.linalg.norm(np.asarray(b["wrist_xy_m"]) - a["wrist_xy_m"])) /
                  (b["time_seconds"] - a["time_seconds"]) for a, b in zip(anchors, anchors[1:])]
    last_note, gaps = {}, []
    for i in sorted(indices, key=lambda i: (notes[i].start_time, i)):
        finger = assignments[i]
        if finger in last_note:
            gaps.append(float(notes[i].start_time - notes[last_note[finger]].end_time))
        last_note[finger] = i
    return {"assignments": assignments, "anchors": anchors,
            "reach_diagnostics": list(diagnostics.values()), "used_thumb": final.used_thumb,
            "costs": {"geometry_time_integral": final.geometry_cost,
                      "reach_excess_per_event": final.reach_cost,
                      "thumb_onsets": final.thumb_cost, "movement": final.movement_cost,
                      "finger_exchange": final.exchange_cost,
                      "total": final.cost},
            "statistics": {"notes": len(indices), "exact_events": len(times),
                           "maximum_active_notes": max_active, "maximum_dp_states": max_states,
                           "thumb_notes": sum(f == thumb for f in assignments.values()),
                           "same_finger_gaps_below_50ms": sum(gap < .05 - 1e-10 for gap in gaps),
                           "same_finger_gaps_below_recovery_time": sum(gap < recovery_time - 1e-10 for gap in gaps),
                           "max_target_residual_m": max(residuals, default=0.),
                           "mean_event_target_residual_m": float(np.mean(residuals)) if residuals else 0.,
                           "max_vertical_residual_m": max(a["max_z_error_m"] for a in anchors),
                           "z_fallback_finger_events": sum(a["z_fallback_count"] for a in anchors),
                           "onsets_without_candidate_within_tolerance": sum(
                               d["no_candidate_within_tolerance"] for d in diagnostics.values()),
                           "onsets_without_candidate_within_10mm": sum(
                               d["best_candidate_max_residual_m"] > .01 for d in diagnostics.values()),
                           "max_anchor_transition_speed_m_s": max(velocities, default=0.)}}


def validate_assignments(notes, assignments):
    occupied = {}
    events = _events(notes, assignments)
    for at in sorted(events):
        starts, ends = events[at]
        for i in ends:
            occupied.pop(assignments[i], None)
        for i in starts:
            finger = assignments[i]
            if not 0 <= finger < 10:
                raise ValueError(f"Invalid finger {finger}")
            if finger in occupied:
                raise ValueError(f"Finger {finger} assigned overlapping notes {occupied[finger]} and {i} at {at}")
            occupied[finger] = i


def with_fingering(sequence, assignments):
    if set(assignments) != set(range(len(sequence.notes))):
        raise ValueError("Fingering must cover every source note")
    result = type(sequence)()
    result.CopyFrom(sequence)
    for i, finger in assignments.items():
        if not 0 <= finger < 10:
            raise ValueError("Finger ids must be in 0..9")
        result.notes[i].part = finger
    check = type(sequence)()
    check.CopyFrom(result)
    for old, restored in zip(sequence.notes, check.notes):
        restored.CopyFrom(old)  # Restore notes, then compare all sequence metadata.
    if check != sequence:
        raise AssertionError("Non-note sequence fields changed")
    for old, new in zip(sequence.notes, result.notes):
        restored = type(old)()
        restored.CopyFrom(new)
        if not old.DESCRIPTOR.fields_by_name["part"].has_presence or old.HasField("part"):
            restored.part = old.part
        else:
            restored.ClearField("part")
        if restored != old:
            raise AssertionError("Non-fingering note fields changed")
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, nonexisting directory")
    parser.add_argument("--thumb-penalty", type=float, default=6.)
    parser.add_argument("--movement-weight", type=float, default=500.)
    parser.add_argument("--pitch-movement-weight", type=float, default=.05)
    parser.add_argument("--reach-weight", type=float, default=100.,
                        help="Per-event squared penalty for reach residual beyond z-tolerance-m")
    parser.add_argument("--exchange-weight", type=float, default=0.,
                        help="Soft cost for reusing a finger before recovery-time; zero preserves the prior objective")
    parser.add_argument("--recovery-time", type=float, default=.05,
                        help="Preferred same-finger release gap in seconds; scheduling preference, not hardware calibration")
    parser.add_argument("--z-tolerance-m", type=float, default=.003)
    parser.add_argument("--geometry-candidates", type=int, default=3)
    parser.add_argument("--fixed-fingering", type=Path,
                        help="Optional JSON with source_midi_sha256 and note-index to finger assignments")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; refusing to overwrite")
    if args.midi.suffix.lower() not in (".mid", ".midi"):
        parser.error("Input must be the original raw MIDI")
    # Import music parsers only for the CLI; synthetic tests remain small and pure.
    import pretty_midi
    from note_seq import midi_io
    sequence = midi_io.midi_file_to_note_sequence(str(args.midi))
    parsed = pretty_midi.PrettyMIDI(str(args.midi))
    expected = sorted((n.pitch, n.start, n.end, n.velocity, i, inst.program, inst.is_drum)
                      for i, inst in enumerate(parsed.instruments) for n in inst.notes)
    actual = sorted((n.pitch, n.start_time, n.end_time, n.velocity, n.instrument, n.program, n.is_drum)
                    for n in sequence.notes)
    if actual != expected:
        raise ValueError("note_seq and pretty_midi disagree on original note attributes")
    if {n.instrument for n in sequence.notes} != {0, 1}:
        raise ValueError("This score requires exactly verified instrument 0=right and 1=left")
    source_sequence_hash = hashlib.sha256(sequence.SerializeToString(deterministic=True)).hexdigest()
    fixed_manifest, fixed_assignments = None, {}
    if args.fixed_fingering:
        fixed_manifest = json.loads(args.fixed_fingering.read_text())
        if fixed_manifest.get("source_midi_sha256") != sha256(args.midi):
            raise ValueError("Fixed fingering source MIDI SHA does not match")
        fixed_assignments = {int(i): f for i, f in fixed_manifest["assignments"].items()}
        if any(not 0 <= i < len(sequence.notes) or type(f) is not int
               or not sequence.notes[i].instrument * 5 <= f < sequence.notes[i].instrument * 5 + 5
               for i, f in fixed_assignments.items()):
            raise ValueError("Invalid fixed fingering note index, hand, or finger")
    data = json.loads(args.geometry.read_text())
    model = Geometry(data, z_tolerance_m=args.z_tolerance_m, max_poses=args.geometry_candidates)
    begin = time.monotonic()
    assignments, hands, anchors, diagnostics = {}, {}, [], []
    for left, instrument in ((False, 0), (True, 1)):
        indices = [i for i, n in enumerate(sequence.notes) if n.instrument == instrument]
        result = plan_hand(sequence.notes, indices, model, left=left,
                           thumb_penalty=args.thumb_penalty, movement_weight=args.movement_weight,
                           pitch_movement_weight=args.pitch_movement_weight, reach_weight=args.reach_weight,
                           exchange_weight=args.exchange_weight, recovery_time=args.recovery_time,
                           fixed_assignments={i: fixed_assignments[i] for i in indices if i in fixed_assignments})
        assignments.update(result.pop("assignments"))
        anchors.extend(result.pop("anchors"))
        diagnostics.extend(result.pop("reach_diagnostics"))
        hands["left" if left else "right"] = result
        print(json.dumps({"hand": "left" if left else "right", **result}), flush=True)
    validate_assignments(sequence.notes, assignments)
    fingered = with_fingering(sequence, assignments)
    args.output.mkdir(parents=True, exist_ok=False)
    if fixed_manifest is not None:
        (args.output / "fixed_fingering.json").write_text(json.dumps(fixed_manifest, indent=2) + "\n")
    proto = args.output / "turkish-march-advanced_fingered.proto"
    proto.write_bytes(fingered.SerializeToString())
    reloaded = type(sequence)()
    reloaded.ParseFromString(proto.read_bytes())
    if reloaded != fingered:
        raise AssertionError("Written proto does not round-trip")
    csv_path = args.output / "fingering.csv"
    with csv_path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["note_index", "instrument", "pitch", "start_seconds", "end_seconds",
                         "velocity", "finger_id", "hand", "finger_number"])
        for i, note in enumerate(sequence.notes):
            finger = assignments[i]
            writer.writerow([i, note.instrument, note.pitch, note.start_time, note.end_time,
                             note.velocity, finger, "left" if finger >= 5 else "right", finger % 5 + 1])
    anchor_path = args.output / "candidate_wrist_anchors.json"
    anchor_path.write_text(json.dumps({"schema": "sampled_geometry_exact_event_anchors_v1",
                                      "time_origin": "original MIDI seconds; no trimming or preparation offset",
                                      "coordinate_convention": "wrist_tx_ty_m is joint Tx/Ty; wrist_xy_m is its world XY displacement using extracted slide axes",
                                      "geometry_sha256": sha256(args.geometry),
                                      "anchors": sorted(anchors, key=lambda a: (a["time_seconds"], a["hand"]))},
                                     indent=2) + "\n")
    diagnosis_path = args.output / "reach_diagnostics.json"
    diagnosis_path.write_text(json.dumps({
        "scope": "Locally fitted finite geometry candidates, including continuity anchors; geometric best ignores sustained-finger compatibility and prior motion. No claim of continuous IK infeasibility.",
        "tolerance_m": args.z_tolerance_m,
        "onsets": sorted(diagnostics, key=lambda d: (d["time_seconds"], d["hand"]))}, indent=2) + "\n")
    report = {"schema": "a32_exact_event_fingering_plan_v1",
              "method": "Exact-event dynamic programming over natural nonthumb orders, optional thumb at either end, and shared-wrist sampled geometry candidates",
              "source_midi": str(args.midi.resolve()), "source_midi_sha256": sha256(args.midi),
              "source_sequence_sha256": source_sequence_hash,
              "geometry_path": str(args.geometry.resolve()), "geometry_sha256": sha256(args.geometry),
              "geometry_metadata": {k: v for k, v in data.items() if k not in
                                    ("tip_xyz", "key_target_xyz", "reachable_samples")},
              "planner_sha256": sha256(__file__), "note_count": len(sequence.notes),
              "fixed_fingering": {"path": str(args.fixed_fingering.resolve()),
                                   "sha256": sha256(args.fixed_fingering),
                                   "count": len(fixed_assignments)} if args.fixed_fingering else None,
              "all_non_fingering_sequence_fields_preserved": True,
              "instrument_to_hand": {"0": "right", "1": "left"},
              "counts_by_finger": {str(f): sum(v == f for v in assignments.values()) for f in range(10)},
              "simultaneous_finger_conflicts": 0, "required_thumb_per_hand": True,
              "parameters": {"thumb_penalty_per_onset": args.thumb_penalty,
                             "movement_weight_m2_per_second": args.movement_weight,
                             "pitch_movement_weight_rad2_per_second": args.pitch_movement_weight,
                             "reach_excess_weight_per_event": args.reach_weight,
                             "exchange_weight_per_reused_finger_onset": args.exchange_weight,
                             "recovery_time_seconds": args.recovery_time,
                             "exchange_formula": "weight * max(0, 1 - exact_same_finger_release_gap / recovery_time)^2; same-pitch repetitions included",
                             "exchange_history": "Per-finger most recent release, retained until recovery_time has elapsed; included in DP merge identity when enabled",
                             "z_tolerance_m": args.z_tolerance_m,
                             "maximum_geometry_candidates_per_assignment": args.geometry_candidates},
              "hands": hands, "planning_seconds": time.monotonic() - begin,
              "physical_certification": False,
              "limitations": ["Finite sampled geometry and locally fitted wrist anchors approximate continuous reachability; nonzero residuals are not certified reachable targets.",
                              "All active fingers share sampled wrist pitch and bounded XY translation, but collisions, key travel, forces and dynamical tracking are not simulated.",
                              "Motion is a soft time-aware objective, not a velocity/acceleration guarantee; first placement is free and no preparation motion is modeled.",
                              "Independent hand searches do not enforce cross-hand collision clearance.",
                              "Thumb usage is positively penalized and constrained to at least one actual note per hand; it is not post-edited.",
                              "Finger recovery is a soft scheduling preference; alternating fingers does not certify same-key release, collision clearance or physical timing.",
                              "Baseline geometric cost is integrated over exact durations; excess reach has a separate per-event penalty and thumb cost is per onset.",
                              "Wrist pose is retained at pure release events; onset candidates include up to two compatible predecessor anchors. Bounded candidate pruning can exclude a globally preferable continuous trajectory."],
              "outputs": {p.name: sha256(p) for p in (proto, csv_path, anchor_path, diagnosis_path)}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
