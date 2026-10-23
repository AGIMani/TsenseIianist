"""Lossless NoteSequence editing without importing simulation or audio packages."""
from __future__ import annotations

import hashlib
import importlib.util
from importlib.metadata import distribution
import math
from pathlib import Path


def sequence_type():
    # Importing note_seq.__init__ also imports audio/ML libraries. Only its
    # installed generated protobuf module is needed for this local editor.
    path = distribution("note-seq").locate_file("note_seq/protobuf/music_pb2.py")
    spec = importlib.util.spec_from_file_location("fingering_editor_music_pb2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NoteSequence


Sequence = sequence_type()
FINGERS = tuple(
    f"{side}{name} · {offset + i}"
    for side, offset in (("右手", 0), ("左手", 5))
    for i, name in enumerate(("拇指", "食指", "中指", "无名指", "小指"))
)


def note_name(pitch):
    return ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")[pitch % 12] + str(pitch // 12 - 1)


class Score:
    """Stable original note indices; only explicit part/start/end edits are made."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("文件超过 16 MiB，请导入较小的 NoteSequence proto")
        self.original = self.path.read_bytes()
        self.seq = Sequence.FromString(self.original)
        if not 0 < len(self.seq.notes) <= 50000:
            raise ValueError("需要含 1–50000 个音符的 NoteSequence 二进制 .proto")
        if not math.isfinite(self.seq.total_time) or self.seq.total_time < 0:
            raise ValueError("文件总时长必须为非负有限秒数")
        for note in self.seq.notes:
            self.validate(note.part, note.start_time, note.end_time)
            if not 0 <= note.pitch <= 127:
                raise ValueError("音高超出 MIDI 0–127")
        self.undo_stack = []
        self.redo_stack = []
        self.saved = self.seq.SerializeToString()

    @staticmethod
    def validate(part, start, end):
        if type(part) is not int or not 0 <= part <= 9:
            raise ValueError("指法编号必须为 0–9（右手 0–4，左手 5–9）")
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("时间必须为有限秒数，且 0 ≤ 起音 < 松键")

    @property
    def duration(self):
        return max(self.seq.total_time, max(n.end_time for n in self.seq.notes))

    @property
    def dirty(self):
        return self.seq.SerializeToString() != self.saved

    def edit(self, indices, *, part=None, start=None, end=None):
        changes = []
        for i in sorted(set(indices)):
            n = self.seq.notes[i]
            old = (n.part, n.start_time, n.end_time)
            new = (n.part if part is None else part,
                   n.start_time if start is None else start,
                   n.end_time if end is None else end)
            self.validate(*new)
            # Quantized timing is a second timing contract. Do not silently
            # leave it inconsistent with new seconds.
            if new[1:] != old[1:] and (
                self.seq.quantization_info.steps_per_quarter
                or self.seq.quantization_info.steps_per_second
                or n.quantized_start_step or n.quantized_end_step
            ):
                raise ValueError("此文件同时含量化步数；请先明确转换为秒级谱，当前仅允许改指法")
            if old != new:
                changes.append((i, old, new))
        if changes:
            self._apply(changes, 2)
            self.undo_stack.append(changes)
            self.redo_stack.clear()
        return bool(changes)

    def _apply(self, changes, which):
        for change in changes:
            n = self.seq.notes[change[0]]
            n.part, n.start_time, n.end_time = change[which]
        baseline = Sequence.FromString(self.original)
        changed_timing = any(
            (a.start_time, a.end_time) != (b.start_time, b.end_time)
            for a, b in zip(self.seq.notes, baseline.notes)
        )
        # Keep an intentional source tail and other events intact; extend it
        # only when an edited note moves beyond the original total_time.
        self.seq.total_time = (
            max(baseline.total_time, max(n.end_time for n in self.seq.notes))
            if changed_timing else baseline.total_time
        )

    def undo(self):
        if self.undo_stack:
            changes = self.undo_stack.pop()
            self._apply(changes, 1)
            self.redo_stack.append(changes)

    def redo(self):
        if self.redo_stack:
            changes = self.redo_stack.pop()
            self._apply(changes, 2)
            self.undo_stack.append(changes)

    def active(self, time):
        return [(i, n) for i, n in enumerate(self.seq.notes) if n.start_time <= time < n.end_time]

    def previous(self, time):
        """Released notes of the previous score-onset group, independent of seeks.

        During a gap the most recently released group remains visible. A held
        note remains current, never becomes a pale duplicate of itself.
        """
        starts = sorted({n.start_time for n in self.seq.notes if n.start_time <= time})
        if not starts:
            return []
        latest = starts[-1]
        if any(n.start_time == latest and n.end_time > time for n in self.seq.notes):
            if len(starts) < 2:
                return []
            latest = starts[-2]
        return [(i, n) for i, n in enumerate(self.seq.notes)
                if n.start_time == latest and n.end_time <= time]

    def upcoming(self, time):
        """Next exact onset group, including repeated keys with a new finger."""
        next_start = min((n.start_time for n in self.seq.notes if n.start_time > time), default=None)
        return [(i, n) for i, n in enumerate(self.seq.notes) if n.start_time == next_start]

    def conflicts(self):
        """Every note sharing a finger with another temporally overlapping note."""
        result = set()
        for finger in range(10):
            latest_end, latest_id = -1.0, None
            for i in sorted((i for i, n in enumerate(self.seq.notes) if n.part == finger),
                            key=lambda i: self.seq.notes[i].start_time):
                n = self.seq.notes[i]
                if n.start_time < latest_end:
                    result.update((i, latest_id))
                if n.end_time > latest_end:
                    latest_end, latest_id = n.end_time, i
        return result

    def output_bytes(self):
        baseline = Sequence.FromString(self.original)
        check = Sequence()
        check.CopyFrom(self.seq)
        for current, original in zip(check.notes, baseline.notes):
            for field in ("part", "start_time", "end_time"):
                setattr(current, field, getattr(original, field))
        check.total_time = baseline.total_time
        if check != baseline:
            raise ValueError("检测到编辑范围以外的谱面修改，拒绝导出")
        return self.original if self.seq == baseline else self.seq.SerializeToString()

    def save(self, path):
        path = Path(path).resolve()
        if path == self.path:
            raise ValueError("请另存新文件，保留导入的原始谱")
        raw = self.output_bytes()
        with path.open("xb") as stream:
            stream.write(raw)
        if path.read_bytes() != raw:
            raise OSError("导出后的文件校验失败")
        self.saved = self.seq.SerializeToString()
        return hashlib.sha256(raw).hexdigest()
