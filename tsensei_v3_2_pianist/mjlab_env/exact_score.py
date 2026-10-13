"""Exact MIDI event targets sampled at physical state timestamps.

The control clock never rounds a note boundary. Targets use [start, end), and
sample k describes the state after physics step k: time (k + 1) * physics_dt.
Only lookup tables are used during rollouts, including CUDA graph capture.
"""
import math

import numpy as np

SCHEMA = 'exact_note_events_v1'
OBSERVATION_SIZE = 176


def validate_events(events, sustain_events, duration):
    events = np.asarray(events, dtype=np.float64)
    sustain_events = np.asarray(sustain_events, dtype=np.float64)
    if events.ndim != 2 or events.shape[1] != 4 or not len(events):
        raise ValueError('Exact note events must be nonempty [start, end, key, finger] rows')
    if not np.isfinite(events).all() or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Exact note times must be finite')
    if ((events[:, 0] < 0).any() or (events[:, 1] <= events[:, 0]).any()
            or (events[:, 1] > duration + 1e-9).any()):
        raise ValueError('Exact note intervals must be positive and within the episode')
    for col, size in ((2, 88), (3, 10)):
        if ((events[:, col] != events[:, col].astype(int)).any()
                or (events[:, col] < 0).any() or (events[:, col] >= size).any()):
            raise ValueError('Invalid exact note key/finger assignment')
    if (sustain_events.ndim != 2 or sustain_events.shape[1] != 2
            or not np.isfinite(sustain_events).all()
            or (sustain_events[:, 0] < 0).any()
            or (sustain_events[:, 0] > duration + 1e-9).any()
            or not np.isin(sustain_events[:, 1], [0., 1.]).all()):
        raise ValueError('Exact sustain events must contain time and binary state')
    for label, column in (('key', 2), ('finger', 3)):
        for assigned in np.unique(events[:, column]):
            selected = events[events[:, column] == assigned]
            selected = selected[np.argsort(selected[:, 0], kind='stable')]
            previous_end = -1.
            for start, end, key, finger in selected:
                if start < previous_end - 1e-10:
                    raise ValueError(f'Overlapping real MIDI events for {label} {int(assigned)}')
                previous_end = max(previous_end, end)
    return events.copy(), sustain_events.copy()


def physical_coverage(events, *, length, control_dt, physics_dt):
    times = (np.arange(round(length * control_dt / physics_dt), dtype=np.float64) + 1) * physics_dt
    counts = []
    for start, end, key, finger in events:
        count = int(((times >= start - 1e-10) & (times < end - 1e-10)).sum())
        if count == 0:
            raise ValueError(f'Note key {int(key)} at {start:g}s has no physical target sample')
        counts.append(count)
    for key in np.unique(events[:, 2]):
        selected = events[events[:, 2] == key]
        selected = selected[np.argsort(selected[:, 0], kind='stable')]
        for previous, following in zip(selected[:-1], selected[1:]):
            released = (times >= previous[1] - 1e-10) & (times < following[0] - 1e-10)
            if not released.any():
                raise ValueError(f'Repeated key {int(key)} has no release sample at physics resolution')
    return counts


def sequence_events(sequence, initial_buffer_time, control_dt, music_end_time):
    """Use the already trimmed/annotated source sequence, before grid expansion."""
    offset = round(initial_buffer_time / control_dt) * control_dt
    events = np.asarray([(n.start_time + offset, n.end_time + offset,
                          n.pitch - 21, n.part) for n in sequence.notes], dtype=np.float64)
    sustain = [(c.time + offset, float(c.control_value >= 64))
               for c in sequence.control_changes if c.control_number == 64]
    # The explicit release tail turns the pedal off, even if the source leaves it on.
    sustain.append((music_end_time, 0.))
    return events.reshape(-1, 4), np.asarray(sustain, dtype=np.float64).reshape(-1, 2)


def sample_targets(events, sustain_events, times, *, fingers=True):
    """CPU export-time sampling; 1e-10 absorbs arithmetic at exact grid boundaries."""
    times = np.asarray(times, dtype=np.float64)
    goal = np.zeros((len(times), 89), dtype=np.float32)
    assignments = np.zeros((len(times), 10, 88), dtype=np.float32) if fingers else None
    for start, end, key, finger in events:
        active = (times >= start - 1e-10) & (times < end - 1e-10)
        goal[active, int(key)] = 1.
        if fingers:
            assignments[active, int(finger), int(key)] += 1.
    for time, state in sorted(sustain_events.tolist(), key=lambda row: row[0]):
        goal[times >= time - 1e-10, -1] = state
    return goal, assignments


def event_countdowns(events, times, horizon):
    """Per-key seconds to next on/off, -1 if no event is within lookahead.

    Layout: 88 onset countdowns, then 88 release countdowns. An event exactly
    at the observation time has countdown zero; both short notes and subframe
    transitions are visible even if no control-grid target falls inside them.
    """
    result = np.full((len(times), 176), -1., dtype=np.float32)
    for kind, col in ((0, 0), (1, 1)):
        for key in range(88):
            boundaries = np.sort(events[events[:, 2] == key, col])
            if not len(boundaries):
                continue
            ids = np.searchsorted(boundaries, times - 1e-10)
            delta = boundaries[np.minimum(ids, len(boundaries) - 1)] - times
            valid = (ids < len(boundaries)) & (delta <= horizon + 1e-10)
            result[valid, kind * 88 + key] = np.maximum(delta[valid], 0.)
    return result


def future_assignments(events, times, current, horizon):
    """Nearest exact future onset for currently idle fingers, without grid rounding."""
    result = np.zeros_like(current)
    seconds = np.zeros(current.shape[:2], dtype=np.float32)
    idle = ~current.astype(bool).any(-1)
    for row, time in enumerate(times):
        for finger in range(10):
            candidates = events[(events[:, 3] == finger) & (events[:, 0] > time + 1e-10)
                                & (events[:, 0] <= time + horizon + 1e-10)]
            if not idle[row, finger] or not len(candidates):
                continue
            start = candidates[:, 0].min()
            nearest = candidates[np.abs(candidates[:, 0] - start) <= 1e-10]
            for note in nearest:
                result[row, finger, int(note[2])] += 1.
            seconds[row, finger] = start - time
    return result, seconds


class ExactScore:
    """Immutable device tables; all dynamic access is graph-safe tensor indexing."""
    def __init__(self, events, sustain_events, *, length, control_dt, physics_dt,
                 lookahead_steps, anticipation_horizon=0., device='cpu'):
        import torch
        decimation = round(control_dt / physics_dt)
        if (decimation < 1 or not math.isclose(decimation * physics_dt, control_dt,
                                               rel_tol=0, abs_tol=1e-10)):
            raise ValueError('Exact timing requires integer physics steps per control frame')
        events, sustain_events = validate_events(events, sustain_events, length * control_dt)
        physical_coverage(events, length=length, control_dt=control_dt, physics_dt=physics_dt)
        self.events, self.sustain_events = events.copy(), sustain_events.copy()
        self.decimation = decimation
        start_times = np.arange(length, dtype=np.float64) * control_dt
        end_times = (np.arange(length, dtype=np.float64) + 1) * control_dt
        sample_times = (np.arange(length * decimation, dtype=np.float64) + 1) * physics_dt
        goal, fingers = sample_targets(events, sustain_events, start_times)
        end_goal, end_fingers = sample_targets(events, sustain_events, end_times)
        history, _ = sample_targets(events, sustain_events, sample_times, fingers=False)
        def tensor(value):
            return torch.as_tensor(value, device=device, dtype=torch.float32)
        self.goal, self.fingers = tensor(goal), tensor(fingers)
        self.end_goal, self.end_fingers = tensor(end_goal), tensor(end_fingers)
        self.goal_history = tensor(history.reshape(length, decimation, 89))
        self.observation = tensor(event_countdowns(events, start_times,
                                                   max(lookahead_steps * control_dt, control_dt)))
        self.future, self.future_seconds = map(tensor, future_assignments(
            events, start_times, fingers, anticipation_horizon))
        self.end_future, self.end_future_seconds = map(tensor, future_assignments(
            events, end_times, end_fingers, anticipation_horizon))
        self.contract = dict(schema=SCHEMA, interval='start_inclusive_end_exclusive',
            physical_sample_time='post_step', observation='next_on_off_seconds_v1',
            observation_size=OBSERVATION_SIZE, absent_event=-1.,
            observation_horizon=max(lookahead_steps * control_dt, control_dt))
