#!/usr/bin/env python3
"""Make audited, symbolic 8-measure practice scores; never load a robot model.

The five reviewed ornament runs are specific to the pinned exper044 score.
Ordinary sixteenth notes and broken chords are deliberately excluded.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import mido

from audit_fingering_repeats import assignment_issue, parse_midi
from fingering_editor_data import Sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'training_mjlab/inputs/turkish_march_advanced'
FINGERED = SOURCE / 'a32_repeat_static_guarded_20261025'
EXPECTED = '66563e8b5e338c1ef3e426233647da2a008ac931939775b1ee87c36857339af0'
# Original NoteSequence indices, inspected with raw MIDI ticks and finger CSV.
GROUPS = [([187, 188], 189), ([283, 284], 285),
          ([1067, 1068], 1069), ([1163, 1164], 1165),
          ([1367, 1368, 1369], 1370)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone(message):
    result = type(message)()
    result.CopyFrom(message)
    return result


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def as_notes(original_notes, records):
    # This source is verified constant 120 BPM / 480 ticks per quarter.
    return [replace(original_notes[i], start=round(n.start_time * 960),
                    end=round(n.end_time * 960), start_seconds=n.start_time,
                    end_seconds=n.end_time, finger=n.part)
            for i, n in records.items()]


def transform(sequence, notes):
    records = {i: clone(n) for i, n in enumerate(sequence.notes)}
    provenance = {i: [i] for i in records}
    changes = []
    for grace_ids, principal in GROUPS:
        main = records[principal]
        assert all(records[i].instrument == main.instrument == 0 for i in grace_ids)
        assert all(0 < records[i].end_time - records[i].start_time < .06 for i in grace_ids)
        assert abs(main.start_time - records[grace_ids[0]].start_time - .125) < 1e-9
        before = [dict(index=i, pitch=records[i].pitch, finger=records[i].part,
                       start=records[i].start_time, end=records[i].end_time)
                  for i in grace_ids + [principal]]
        by_pitch = {main.pitch: principal}
        for i in grace_ids:
            n = records[i]
            if n.pitch in by_pitch:
                provenance[by_pitch[n.pitch]].extend(provenance.pop(i))
                del records[i]
            else:
                n.start_time, n.end_time = main.start_time, main.end_time
                by_pitch[n.pitch] = i
        chord = sorted(by_pitch.values(), key=lambda i: records[i].pitch)
        old = {i: records[i].part for i in chord}
        candidates = []
        for fingers in itertools.permutations(range(5), len(chord)):
            mapping = dict(zip(chord, fingers))
            if mapping[principal] != old[principal]:
                continue  # Keep the principal note's confirmed finger.
            nonthumb = [f for f in fingers if f != 0]
            if nonthumb != sorted(nonthumb) or (0 in fingers and fingers.index(0) not in (0, len(fingers)-1)):
                continue
            assignments = {i: n.part for i, n in records.items()} | mapping
            if assignment_issue(as_notes(notes, records), assignments, {}):
                continue
            cost = (sum(mapping[i] != old[i] for i in chord),
                    sum(abs(mapping[i]-old[i]) for i in chord), fingers)
            candidates.append((cost, mapping))
        if not candidates:
            raise ValueError(f'No symbolically valid fingering for ornament at {main.start_time}')
        mapping = min(candidates, key=lambda item: item[0])[1]
        for i in chord:
            records[i].part = mapping[i]
        changes.append(dict(principal_original_index=principal, before=before,
                            chord_start_seconds=main.start_time, chord_end_seconds=main.end_time,
                            after=[dict(pitch=records[i].pitch, finger=records[i].part,
                                        source_indices=sorted(provenance[i])) for i in chord],
                            finger_changes=[dict(source_index=i, old=old[i], new=mapping[i])
                                            for i in chord if old[i] != mapping[i]]))
    assert assignment_issue(as_notes(notes, records), {i:n.part for i,n in records.items()}, {}) is None
    edited = {i for grace, main in GROUPS for i in grace + [main]}
    assert all(records[i] == sequence.notes[i] for i in records if i not in edited)
    assert sorted(i for ids in provenance.values() for i in ids) == list(range(len(sequence.notes)))
    return records, provenance, changes


def crop(sequence, records, start, end):
    result = clone(sequence)
    result.ClearField('notes')
    ids = [i for i, n in records.items() if n.start_time < end and n.end_time > start]
    for i in ids:
        n = records[i]
        # This edition has no notes crossing the selected 8-bar boundaries.
        if n.start_time < start - 1e-9 or n.end_time > end + 1e-9:
            raise ValueError('Cross-boundary note: do not silently truncate or duplicate')
        out = result.notes.add()
        out.CopyFrom(n)
        out.start_time -= start
        out.end_time -= start
    for field in ('tempos', 'time_signatures', 'key_signatures'):
        events = list(getattr(sequence, field))
        past = [x for x in events if x.time <= start]
        kept = ([max(past, key=lambda x:x.time)] if past else []) + [x for x in events if start < x.time < end]
        result.ClearField(field)
        for item in kept:
            out = getattr(result, field).add()
            out.CopyFrom(item)
            out.time = max(0., out.time-start)
    # Original controller initialization contains an ordered RPN transaction.
    # Preserve the complete transaction, not a last-value-per-controller map.
    assert all(x.time == 0 for x in sequence.control_changes)
    result.total_time = end-start
    return result, ids


def write_midi(path, seq):
    midi = mido.MidiFile(ticks_per_beat=480)
    for hand in (0, 1):
        events = [(0, -5, mido.MetaMessage('track_name', name=('Right hand' if hand == 0 else 'Left hand')))]
        if hand == 0:
            for t in seq.tempos:
                events.append((round(t.time*960), -4, mido.MetaMessage('set_tempo', tempo=round(60e6/t.qpm))))
            for t in seq.time_signatures:
                events.append((round(t.time*960), -3, mido.MetaMessage('time_signature', numerator=t.numerator, denominator=t.denominator)))
            keys = ('C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B')
            for t in seq.key_signatures:
                if t.mode != 0:
                    raise ValueError('Unexpected key mode in pinned source')
                events.append((round(t.time*960), -2, mido.MetaMessage('key_signature', key=keys[t.key])))
        events.append((0, -1, mido.Message('program_change', channel=hand, program=0)))
        for c in seq.control_changes:
            if c.instrument == hand:
                events.append((round(c.time*960), 0, mido.Message('control_change', channel=hand, control=c.control_number, value=c.control_value)))
        for i, n in enumerate(seq.notes):
            if n.instrument != hand:
                continue
            events.extend([(round(n.start_time*960), 2, mido.MetaMessage('text', text=f'note_index={i};finger_id={n.part}')),
                           (round(n.start_time*960), 3, mido.Message('note_on', channel=hand, note=n.pitch, velocity=n.velocity)),
                           (round(n.end_time*960), 1, mido.Message('note_off', channel=hand, note=n.pitch, velocity=0))])
        events.append((round(seq.total_time*960), 9, mido.MetaMessage('end_of_track')))
        track = mido.MidiTrack()
        at = 0
        for tick, _, msg in sorted(events, key=lambda x:(x[0],x[1])):
            track.append(msg.copy(time=tick-at))
            at = tick
        midi.tracks.append(track)
    midi.save(path)


def write_score(folder, stem, seq, ids, provenance, offset):
    proto, midi, csv_path = [folder / (stem+suffix) for suffix in ('.proto','.mid','.csv')]
    proto.write_bytes(seq.SerializeToString())
    assert Sequence.FromString(proto.read_bytes()) == seq
    fields = ['note_index','instrument','pitch','start_seconds','end_seconds','velocity',
              'finger_id','hand','finger_number','source_indices','original_timeline_start_seconds','original_timeline_end_seconds']
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (n, original_id) in enumerate(zip(seq.notes, ids)):
            writer.writerow(dict(note_index=i, instrument=n.instrument, pitch=n.pitch,
                start_seconds=n.start_time, end_seconds=n.end_time, velocity=n.velocity,
                finger_id=n.part, hand='right' if n.part < 5 else 'left', finger_number=n.part%5+1,
                source_indices=';'.join(map(str, sorted(provenance[original_id]))),
                original_timeline_start_seconds=n.start_time+offset,
                original_timeline_end_seconds=n.end_time+offset))
    write_midi(midi, seq)
    # Independent raw MIDI note parser verifies pitch/ticks/velocity and CSV fingers.
    parsed, _, _, _ = parse_midi(midi, csv_path)
    assert len(parsed) == len(seq.notes)
    assert assignment_issue(parsed, {n.index:n.finger for n in parsed}, {}) is None
    return dict(proto=proto.name, midi=midi.name, fingering_csv=csv_path.name,
                notes=len(ids), duration_seconds=seq.total_time, source_record_ids=ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tsensei_v3_2_pianist/music/data/turkish_march_8bars_chords_20261026')
    args = parser.parse_args()
    source = FINGERED/'turkish-march-advanced_fingered.proto'
    assert sha(source) == EXPECTED
    sequence = Sequence.FromString(source.read_bytes())
    notes, bars, metadata, _ = parse_midi(SOURCE/'original.mid', FINGERED/'fingering.csv')
    assert [n.index for n in notes] == list(range(len(sequence.notes)))
    assert len(bars) == 223 and metadata['tempo_events'] == [dict(tick=0,tempo_microseconds_per_quarter=500000,quarter_notes_per_minute=120.)]
    assert all((n.pitch,n.finger,n.start_seconds,n.end_seconds) == (s.pitch,s.part,s.start_time,s.end_time) for n,s in zip(notes,sequence.notes))
    records, provenance, changes = transform(sequence, notes)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'sources').mkdir()
    for src, name in [(source,'original_fingered.proto'),(SOURCE/'original.mid','original.mid'),(FINGERED/'fingering.csv','original_fingering.csv')]:
        shutil.copy2(src, out/'sources'/name)
    full, ids = crop(sequence, records, 0, sequence.total_time)
    full_entry = write_score(out, 'full_chords_fingered', full, ids, provenance, 0)
    segments = []
    for start_bar in range(0, len(bars), 8):
        end_bar = min(start_bar+8,len(bars))
        start = bars[start_bar]['start_seconds']
        end = min(bars[end_bar-1]['end_seconds'],sequence.total_time)
        score, ids = crop(sequence,records,start,end)
        stem = f'part_{len(segments)+1:02d}_bars_{start_bar+1:03d}-{end_bar:03d}'
        entry = write_score(out,stem,score,ids,provenance,start)
        segments.append(dict(part=len(segments)+1,first_bar=start_bar+1,last_bar=end_bar,
                             original_start_seconds=start,original_end_seconds=end,**entry))
    assert sorted(i for entry in segments for i in entry['source_record_ids']) == sorted(records)
    assert sum(x['notes'] for x in segments) == len(records) == 2539
    assert len(segments) == 28
    dump(out/'ornament_changes.json',changes)
    manifest = dict(schema=1, source_experiment='exper044', source_fingered_sha256=EXPECTED,
        source_original_midi_sha256=sha(SOURCE/'original.mid'), generator_sha256=sha(__file__),
        bars_per_part=8,bar_numbering='1-based including initial 1/4 pickup; bar 2 onward 2/4',
        total_bars=len(bars), source_notes=len(sequence.notes), result_notes=len(records),
        conversion='Five reviewed short ornament runs; distinct pitches join following principal at its unchanged onset and release; repeated pitches coalesce',
        fingering='Principal fingers fixed; minimum changes to existing distinct-finger/natural-order symbolic rules',
        full_score=full_entry,segments=segments,bars=bars,
        validation=dict(proto_roundtrip=True,midi_csv_note_roundtrip=True,same_finger_overlap=False,
                        natural_order_conflict=False,unmodified_notes_exact=True,all_source_notes_accounted_for=True,
                        segment_coverage_exact=True,boundary_notes_clipped=0,physics_tested=False),
        default_training_input_changed=False)
    dump(out/'manifest.json',manifest)
    lines = ['# 土耳其进行曲：8小节分段／装饰音和弦版','',
        '来源：exper044训练同一指法谱。含开头1/4弱起共223小节，拆为28段；首段7.5秒，中间每段8秒，末段约6.974秒。小节编号包含弱起。', '',
        '只改5处已审查的短装饰音串：31.375、47.375、159.375、175.375、205.375秒起。把各串不同音高移到后接主音的原始起音，松键统一为主音原始松键；重复音高合成一个音。空出的装饰音时隙保留为空，不压缩曲长。普通旋律、分解和弦、速度和其余指法不变。', '',
        '前4处为A4+B4双音；最后一处为E5+A5+C#6。最后一处A5原右小指改右中指（part 4→2），主音C#6仍用右无名指，以避免非拇指顺序交叉。其余保留音的指法不变。2544个原事件合成2539个事件，全部来源索引可追溯。', '',
        '`.proto`是带完整指法的训练/编辑器输入；`.mid`供通用播放器查看（指法附在text元事件，普通播放器通常不显示）；`.csv`含十指编号和原谱全局时间。每段从0秒开始，不含机器人准备/曲尾回程。段间无跨界音，没有裁剪持续音。', '',
        '这是符号层的练习改编版；已验指法冲突、音符覆盖和MIDI/proto读回，未做机械手可达性或物理演奏验收。未替换默认训练曲、未启动训练。sources保留原件，manifest.json记录来源和分段，ornament_changes.json记录逐项改动，SHA256SUMS校验文件。', '',
        '|分段|原曲小节（含弱起）|原曲时间/秒|音符数|文件|', '|---|---|---|---|---|']
    for e in segments:
        lines.append(f"|{e['part']:02d}|{e['first_bar']}–{e['last_bar']}|{e['original_start_seconds']:.3f}–{e['original_end_seconds']:.3f}|{e['notes']}|[{e['proto']}]({e['proto']})|")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file()))
    print(json.dumps(dict(output=str(out),parts=len(segments),notes=len(records),changes=changes),ensure_ascii=False))


if __name__ == '__main__':
    main()
