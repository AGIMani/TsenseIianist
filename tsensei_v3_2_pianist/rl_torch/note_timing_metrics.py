"""Physical note timing on the original MIDI clock, independent of training goals.

The 150 ms same-pitch onset matching is a diagnostic association rule, not a
standard note F1. Key-frame F1 uses exact [start, end) targets at post-step times.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


_SCALARS = (
    'exact_original_music_micro_f1', 'exact_original_music_true_positive_count',
    'exact_original_music_false_positive_count', 'exact_original_music_false_negative_count',
    'note_timing_reference_count', 'note_timing_played_count', 'note_timing_matched_count',
    'note_timing_missed_count', 'note_timing_extra_count', 'note_timing_short_count',
    'note_timing_short_matched_count', 'note_timing_short_recall',
    'note_timing_onset_mean_ms', 'note_timing_onset_mae_ms', 'note_timing_onset_abs_p95_ms',
    'note_timing_offset_mean_ms', 'note_timing_offset_mae_ms', 'note_timing_offset_abs_p95_ms',
    'note_timing_censored_release_count',
)


def _unavailable(reason):
    return dict.fromkeys(_SCALARS, None) | dict(note_timing_available=False,
        note_timing_unavailable_reason=reason, note_timing_records=None,
        note_timing_extra_records=None, note_timing_source=None)


def _summary(records, extras, tp, fp, fn):
    matched = [r for r in records if r['status'] == 'matched']
    short = [r for r in records if r['target_end_s'] - r['target_start_s'] < .1 - 1e-10]
    short_matched = sum(r['status'] == 'matched' for r in short)
    out = dict(exact_original_music_micro_f1=(2 * tp / (2 * tp + fp + fn)
                                             if 2 * tp + fp + fn else 1.),
        exact_original_music_true_positive_count=int(tp),
        exact_original_music_false_positive_count=int(fp),
        exact_original_music_false_negative_count=int(fn),
        note_timing_reference_count=len(records), note_timing_played_count=len(matched) + len(extras),
        note_timing_matched_count=len(matched), note_timing_missed_count=len(records) - len(matched),
        note_timing_extra_count=len(extras), note_timing_short_count=len(short),
        note_timing_short_matched_count=short_matched,
        note_timing_short_recall=short_matched / len(short) if short else None,
        note_timing_censored_release_count=sum(r['release_censored'] for r in matched))
    for event in ('onset', 'offset'):
        errors = np.asarray([r[event + '_error_ms'] for r in matched
                             if r[event + '_error_ms'] is not None], dtype=float)
        for name, value in (('mean', np.mean(errors) if len(errors) else None),
                            ('mae', np.mean(np.abs(errors)) if len(errors) else None),
                            ('abs_p95', np.percentile(np.abs(errors), 95) if len(errors) else None)):
            out['note_timing_' + event + '_' + name + '_ms'] = None if value is None else float(value)
    return out


class NoteTimingMetrics:
    """One synchronized evaluation batch; sample() performs no host transfer.

    Construct with an environment, then sample [world, substep, 88] activation
    histories. result(world) lazily copies the complete batch once, so multiple
    world results do not cause repeated device synchronizations.
    """
    def __init__(self, env=None, *, events=None, physics_dt=None, music_start=None,
                 num_envs=1, onset_tolerance=.15):
        self.histories = []
        self._history_cpu = None
        self._results = {}
        self.reason = None
        self.source = 'explicit_exact_events'
        self.num_envs = num_envs
        if env is not None:
            bundle = getattr(env, 'bundle', None)
            manifest = getattr(bundle, 'manifest', {})
            self.num_envs = getattr(env, 'num_envs', 1)
            physics_dt = manifest.get('physics_dt')
            environment = manifest.get('contract', {}).get('environment', {})
            music_start = environment.get('initial_buffer_time', 0.)
            events = getattr(bundle, 'exact_note_events', None)
            if events is not None:
                self.source = 'exact_note_events'
            else:
                directory = getattr(bundle, 'directory', None)
                path = None if directory is None else Path(directory) / 'score_coverage.json'
                if path is None or not path.is_file():
                    self.reason = 'Original note events and score_coverage.json are unavailable'
                else:
                    try:
                        report = json.loads(path.read_text())
                        events = [(n['start_time'] + music_start, n['end_time'] + music_start,
                                   n['pitch'] - 21, n['fingering']) for n in report['notes']]
                        self.source = 'score_coverage_original_times'
                    except (KeyError, TypeError, ValueError) as exc:
                        self.reason = 'Invalid original score coverage: ' + str(exc)
        if self.reason is None and events is None:
            self.reason = 'Original note events are unavailable'
        if self.reason is None:
            try:
                self.events = np.asarray(events, dtype=np.float64)
                if (self.events.ndim != 2 or self.events.shape[1] != 4 or not len(self.events)
                        or not np.isfinite(self.events).all()
                        or (self.events[:, 0] < 0).any()
                        or (self.events[:, 1] <= self.events[:, 0]).any()
                        or (self.events[:, 2:] != self.events[:, 2:].astype(int)).any()
                        or (self.events[:, 2] < 0).any() or (self.events[:, 2] >= 88).any()
                        or (self.events[:, 3] < 0).any() or (self.events[:, 3] >= 10).any()):
                    raise ValueError('Expected nonempty finite [start,end,key,finger] events')
                self.physics_dt = float(physics_dt)
                self.music_start = float(music_start if music_start is not None else self.events[:, 0].min())
                self.music_end = float(self.events[:, 1].max())
                if (not math.isfinite(self.physics_dt) or self.physics_dt <= 0
                        or not math.isfinite(self.music_start) or self.music_start < 0
                        or self.music_start > self.events[:, 0].min() + 1e-10):
                    raise ValueError('Invalid physical clock or preparation offset')
            except (ValueError, TypeError) as exc:
                self.reason = str(exc)
        if (not math.isfinite(onset_tolerance) or onset_tolerance < 0
                or type(self.num_envs) is not int or self.num_envs < 1):
            raise ValueError('Invalid matching tolerance or number of worlds')
        self.onset_tolerance = float(onset_tolerance)

    def sample(self, history):
        if self.reason is not None:
            return
        if self._history_cpu is not None:
            raise RuntimeError('Cannot add samples after note timing results were materialized')
        if history is None:
            self.reason = 'Physical key activation history is unavailable'
            self.histories.clear()
            return
        if (not isinstance(history, torch.Tensor) or history.ndim != 3
                or history.shape[0] != self.num_envs or history.shape[-1] != 88
                or history.shape[1] < 1 or history.dtype != torch.bool):
            raise ValueError('Expected boolean key activation history [world,substep,88]')
        self.histories.append(history.detach().clone())

    def _played(self, active):
        result = []
        for key in range(88):
            column = active[:, key]
            changed = np.diff(np.r_[False, column].astype(np.int8))
            starts = np.flatnonzero(changed == 1)
            ends = np.flatnonzero(changed == -1)
            for start in starts:
                after = ends[ends > start]
                end = (int(after[0]) + 1) * self.physics_dt if len(after) else None
                result.append(dict(key=key, start_s=(int(start) + 1) * self.physics_dt,
                                   end_s=end, release_censored=end is None))
        return sorted(result, key=lambda note: (note['start_s'], note['key']))

    def _match(self, played):
        matched = {}
        # Maximum cardinality, then minimum total onset distance, independently
        # for each pitch. Dummy columns preserve unmatched notes. This avoids a
        # greedy nearest pair stealing the only feasible match of a short note.
        for key in np.unique(self.events[:, 2].astype(int)):
            originals = np.flatnonzero(self.events[:, 2] == key)
            candidates = [i for i, note in enumerate(played) if note['key'] == key]
            if not candidates:
                continue
            differences = np.abs(self.events[originals, 0, None]
                                 - np.asarray([played[i]['start_s'] for i in candidates])[None])
            cost = np.full((len(originals), len(candidates) + len(originals)), 1e6)
            cost[:, :len(candidates)] = np.where(differences <= self.onset_tolerance + 1e-10,
                                                  differences, 1e9)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if column < len(candidates) and cost[row, column] < 1e6:
                    matched[int(originals[row])] = candidates[column]
        return matched

    def result(self, world=0):
        if not 0 <= world < self.num_envs:
            raise ValueError('Invalid world index')
        if self.reason is not None:
            return _unavailable(self.reason)
        if not self.histories and self._history_cpu is None:
            return _unavailable('No physical key samples were recorded')
        if self._history_cpu is None:
            self._history_cpu = torch.cat(self.histories, dim=1).cpu().numpy()
            self.histories.clear()
        if world in self._results:
            return self._results[world]
        active = self._history_cpu[world]
        times = (np.arange(len(active), dtype=np.float64) + 1) * self.physics_dt
        if times[-1] < self.music_end - 1e-10:
            return _unavailable('Physical recording ended before the original score completed')
        goal = np.zeros_like(active)
        for start, end, key, _ in self.events:
            goal[(times >= start - 1e-10) & (times < end - 1e-10), int(key)] = True
        music = (times >= self.music_start - 1e-10) & (times < self.music_end - 1e-10)
        tp = int((active[music] & goal[music]).sum())
        fp = int((active[music] & ~goal[music]).sum())
        fn = int((~active[music] & goal[music]).sum())
        played = self._played(active)
        matched = self._match(played)
        used = set(matched.values())
        records = []
        for index, (start, end, key, finger) in enumerate(self.events):
            note = played[matched[index]] if index in matched else None
            records.append(dict(index=index, pitch=int(key) + 21, finger=int(finger),
                target_start_s=float(start), target_end_s=float(end),
                status='matched' if note else 'missed',
                played_start_s=note['start_s'] if note else None,
                played_end_s=note['end_s'] if note else None,
                onset_error_ms=float((note['start_s'] - start) * 1000) if note else None,
                offset_error_ms=(float((note['end_s'] - end) * 1000)
                                 if note and note['end_s'] is not None else None),
                release_censored=note['release_censored'] if note else False))
        extras = [dict(pitch=n['key'] + 21, played_start_s=n['start_s'], played_end_s=n['end_s'],
                       release_censored=n['release_censored']) for i, n in enumerate(played) if i not in used]
        result = _summary(records, extras, tp, fp, fn)
        result.update(note_timing_available=True, note_timing_unavailable_reason=None,
            note_timing_source=self.source, note_timing_onset_matching_tolerance_ms=self.onset_tolerance * 1000,
            note_timing_interval='start_inclusive_end_exclusive', note_timing_sample_time='post_step',
            note_timing_music_start_s=self.music_start, note_timing_music_end_s=self.music_end,
            note_timing_records=records, note_timing_extra_records=extras)
        self._results[world] = result
        return result

    @staticmethod
    def aggregate(rows):
        """Pool event errors/counts across episodes; do not average per-world F1."""
        if not rows or not all(row.get('note_timing_available', False) for row in rows):
            return _unavailable('Original-clock note diagnostics are incomplete for one or more episodes')
        records = [record for row in rows for record in row['note_timing_records']]
        extras = [record for row in rows for record in row['note_timing_extra_records']]
        tp, fp, fn = (sum(row['exact_original_music_' + kind + '_count'] for row in rows)
                      for kind in ('true_positive', 'false_positive', 'false_negative'))
        result = _summary(records, extras, tp, fp, fn)
        result.update(note_timing_available=True, note_timing_unavailable_reason=None,
                      note_timing_source=sorted({row['note_timing_source'] for row in rows}),
                      note_timing_episodes=len(rows))
        return result
