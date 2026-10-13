"""Check that control-frame conversion preserves every fingered MIDI event.

The strict mode validates the historical score without moving its targets. A
score that cannot represent a repeat at the requested frequency must be fixed
explicitly, rather than silently losing a note or merging it into a long hold.
This module deliberately has no Composer or note-seq dependency.
"""
import math


MODES = ('legacy', 'strict_v1')


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown score validation mode: {mode!r}')
    return mode


class ScoreCoverageError(ValueError):
    def __init__(self, report):
        self.report = report
        details = '; '.join(item['message'] for item in report['errors'][:8])
        super().__init__('Score coverage validation failed: ' + details +
                         '. Use a finer control timestep or correct the MIDI/fingering; '
                         'notes are never silently deleted in strict_v1.')


def audit_note_coverage(sequence, frames, dt, *, min_pitch=21, max_pitch=108):
    """Audit source ownership and independent activation for each MIDI note.

    Floor/ceil and stable start-time order exactly match sequence_to_pianoroll.
    Ownership prevents an overlapping later note from falsely "covering" an
    erased earlier event. Every repeated note also needs an actual release frame.
    The returned frame indices precede any artificial preparation/tail frames.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Score timestep must be positive and finite')
    report = dict(schema='strict_note_coverage_v1', control_dt=dt,
                  source_notes=len(sequence.notes), score_frames=len(frames),
                  covered_notes=0, notes=[], errors=[])
    errors = report['errors']
    sources = []
    owners = {}
    fps = 1 / dt
    for index, note in sorted(enumerate(sequence.notes), key=lambda pair: pair[1].start_time):
        item = dict(source_index=index, pitch=int(note.pitch), fingering=int(note.part),
                    start_time=float(note.start_time), end_time=float(note.end_time))
        sources.append(item)
        if (not math.isfinite(note.start_time) or not math.isfinite(note.end_time)
                or note.start_time < 0 or note.end_time <= note.start_time):
            errors.append(dict(kind='invalid_note_time', message=f'note {index}, pitch {note.pitch}: invalid interval'))
            continue
        if not min_pitch <= note.pitch <= max_pitch or not 1 <= note.velocity <= 127:
            errors.append(dict(kind='unsupported_note', message=f'note {index}, pitch {note.pitch}: outside piano range or zero/invalid velocity'))
            continue
        if not 0 <= note.part < 10:
            errors.append(dict(kind='invalid_fingering', message=f'note {index}, pitch {note.pitch}: fingering {note.part} outside 0..9'))
            continue
        start = int(note.start_time * fps)
        end = max(start + 1, int(math.ceil(note.end_time * fps)))
        item.update(quantized_start_frame=start, quantized_end_frame=end)
        if end > len(frames):
            errors.append(dict(kind='truncated_note', source_index=index,
                message=f'note {index}, pitch {note.pitch}: sequence ends before its target interval'))
        for frame in range(start, min(end, len(frames))):
            owners[frame, note.pitch] = index
    retained = {index: [] for index in range(len(sequence.notes))}
    for frame_index, notes in enumerate(frames):
        by_finger = {}
        by_pitch = {}
        for note in notes:
            pitch = note.key + min_pitch
            by_finger.setdefault(note.fingering, set()).add(pitch)
            by_pitch.setdefault(pitch, []).append(note.fingering)
            owner = owners.get((frame_index, pitch))
            if owner is not None and sequence.notes[owner].part == note.fingering:
                retained[owner].append(frame_index)
        for finger, pitches in by_finger.items():
            if len(pitches) > 1:
                errors.append(dict(kind='finger_conflict', frame=frame_index,
                    message=f'frame {frame_index}: finger {finger} assigned keys {sorted(pitches)}'))
        for pitch, fingers in by_pitch.items():
            if len(fingers) > 1:
                errors.append(dict(kind='key_conflict', frame=frame_index,
                    message=f'frame {frame_index}: pitch {pitch} assigned multiple fingers {fingers}'))
    previous = {}
    for item in sources:
        active = retained[item['source_index']]
        item['retained_frames'] = active
        report['notes'].append(item)
        if 'quantized_start_frame' not in item:
            continue
        if not active:
            errors.append(dict(kind='missing_note', source_index=item['source_index'],
                message=f"note {item['source_index']}, pitch {item['pitch']}, "
                        f"[{item['start_time']:.6f},{item['end_time']:.6f})s has no independent target frame"))
        else:
            report['covered_notes'] += 1
            item['target_start_time'] = active[0] * dt
            item['target_end_time'] = (active[-1] + 1) * dt
            item['onset_error_seconds'] = item['target_start_time'] - item['start_time']
            item['offset_error_seconds'] = item['target_end_time'] - item['end_time']
            prior = previous.get(item['pitch'])
            if prior is not None and active[0] <= prior[-1] + 1:
                errors.append(dict(kind='merged_repeat', source_index=item['source_index'],
                    message=f"note {item['source_index']}, pitch {item['pitch']}: repeated onset has no release frame"))
            previous[item['pitch']] = active
    report['valid'] = not errors
    return report


def require_note_coverage(sequence, frames, dt):
    report = audit_note_coverage(sequence, frames, dt)
    if not report['valid']:
        raise ScoreCoverageError(report)
    return report


def release_tail_steps(seconds, dt):
    if (not math.isfinite(seconds) or seconds < 0 or not math.isfinite(dt)
            or dt <= 0 or not math.isclose(seconds / dt, round(seconds / dt), abs_tol=1e-9)):
        raise ValueError('Release tail must be finite, nonnegative and contain whole control frames')
    return round(seconds / dt)
