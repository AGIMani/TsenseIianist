#!/usr/bin/env python3
"""CPU-only delivery checks and original-score plot for one completed export."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_videos import sha256
from eval_from_source_archive import validate_sources, extract_archive


def comparison_reference_arguments(config, output, original_midi=None, original_midi_sha256=None, registered_score=None, registered_score_sha256=None):
    recipe = config['source_runtime']
    if registered_score is not None or registered_score_sha256 is not None:
        if original_midi is not None or original_midi_sha256 is not None:
            raise ValueError('Select one reference source')
        if registered_score is None or registered_score_sha256 is None:
            raise ValueError('Registered score requires path and SHA256')
        score = recipe['external_score']
        if not score or score['suffix'] != '.proto':
            raise ValueError('Registered score reference requires an embedded proto')
        raw = base64.b64decode(score['base64'], validate=True)
        if (hashlib.sha256(raw).hexdigest() != registered_score_sha256
                or score['sha256'] != registered_score_sha256
                or registered_score.read_bytes() != raw):
            raise ValueError('Registered score differs from checkpoint score')
        from note_seq.protobuf import music_pb2
        seq = music_pb2.NoteSequence.FromString(raw)
        frozen = output/'registered_reference.proto'
        frozen.write_bytes(raw)
        reference = output/'original_reference.json'
        reference.write_text(json.dumps(dict(config_sha256=sha256(output/'inputs/config.json'),
            midi_path=str(frozen), midi_sha256=registered_score_sha256,
            reference_kind='registered_note_sequence',
            source='Exact registered score events, before control-grid quantization',
            events=[[n.start_time,n.end_time,n.pitch] for n in seq.notes]), indent=2)+'\n')
        return ['--reference-events', str(reference)]
    if recipe['external_score']:
        if original_midi is None or original_midi_sha256 is None:
            raise ValueError('External score requires registered original MIDI and SHA256')
        if sha256(original_midi) != original_midi_sha256:
            raise ValueError('Original MIDI SHA256 differs from submission')
        # The plotter compares all embedded note attributes with this original
        # MIDI before plotting; it never uses control-grid goals as reference.
        return ['--original-midi', str(original_midi)]
    if original_midi is not None or original_midi_sha256 is not None:
        raise ValueError('External MIDI arguments require an external checkpoint score')
    if recipe['environment']['midi'] != 'TwinkleTwinkleRousseau':
        raise ValueError('Automatic original-score provenance is not implemented for this score')
    from note_seq.midi_io import midi_file_to_note_sequence
    from note_seq import sequences_lib
    midi = ROOT/'tsensei_v3_2_pianist/music/data/rousseau/twinkle-twinkle-trimmed.mid'
    seq = midi_file_to_note_sequence(str(midi))
    origin = seq.notes[0].start_time
    seq = sequences_lib.extract_subsequence(seq, origin, seq.notes[-1].end_time)
    reference = output/'original_reference.json'
    reference.write_text(json.dumps(dict(config_sha256=sha256(output/'inputs/config.json'), midi_path=str(midi),
        midi_sha256=sha256(midi), trim_origin_seconds=origin,
        source='Original registered MIDI from verified checkpoint source; same trim_silence as task',
        events=[[n.start_time,n.end_time,n.pitch] for n in seq.notes]), indent=2)+'\n')
    return ['--reference-events', str(reference)]


def validate_display_requirements(plan):
    if not all(plan.get(name) for name in ('audio', 'fingering', 'key_highlight')):
        raise ValueError('Required audio/fingering/key highlights are missing')


def finalize(output, original_midi=None, original_midi_sha256=None, registered_score=None, registered_score_sha256=None):
    plan = json.loads((output/'export_plan.json').read_text())
    report = json.loads((output/'report.json').read_text())
    config_path = output/'inputs/config.json'
    config = json.loads(config_path.read_text())
    if json.loads((output/'status.json').read_text())['state'] != 'completed':
        raise ValueError('Video export is not completed')
    validate_display_requirements(plan)
    if plan['start'] != 0 or plan['end'] is not None or plan['speed'] != 1:
        raise ValueError('Training delivery requires the full original-tempo replay')
    if sha256(config_path) != plan['config_sha256'] or sha256(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('Frozen config/checkpoint changed')
    # Verify frozen physics sources, independently of current display-only edits.
    archive = Path(plan['source_archive'])
    if sha256(archive) != plan['archive_sha256']:
        raise ValueError('Frozen source archive changed')
    with tempfile.TemporaryDirectory(prefix='verify-export-source-') as temporary:
        frozen = Path(temporary)
        extract_archive(archive, frozen)
        validate_sources(frozen, config['source_runtime']['source_sha256'])
    if bool(config.get('arm_follow')) != all(v.get('native_arms', False) for v in report['videos']):
        raise ValueError('Video does not preserve the checkpoint native arm configuration')
    recording = Path(report['recording'])
    reference_args = comparison_reference_arguments(config, output, original_midi, original_midi_sha256, registered_score, registered_score_sha256)
    comparison = output/'midi_comparison'
    if not (comparison/'comparison.json').is_file():
        # An interrupted CPU plot can be rebuilt from the same immutable replay.
        if comparison.exists():
            shutil.rmtree(comparison)
        subprocess.run([sys.executable, str(ROOT/'scripts/plot_recorded_midi_comparison.py'),
            '--export', str(output), '--recording', str(recording),
            *reference_args, '--label', Path(plan['checkpoint_origin']).parent.name+' best.pt'], check=True)
    metric = json.loads((comparison/'comparison.json').read_text())
    if original_midi_sha256 is not None and metric['original_midi_sha256'] != original_midi_sha256:
        raise ValueError('Existing comparison uses a different original MIDI')
    if registered_score_sha256 is not None and (metric['original_midi_sha256'] != registered_score_sha256
            or metric['embedded_proto_sha256'] != registered_score_sha256):
        raise ValueError('Existing comparison uses a different registered score')
    if metric['checkpoint_sha256'] != plan['checkpoint_sha256'] or metric['audio_substeps_sha256'] != sha256(recording/'audio_substeps.npz'):
        raise ValueError('Comparison belongs to a different replay')
    for video in report['videos']:
        path = output/video['file']
        if sha256(path) != video['sha256']:
            raise ValueError('Video hash mismatch')
        # Required stream maps fail if either stream is absent, and decode both.
        # The remote private runtime ships static ffmpeg but not ffprobe.
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),
                        '-map','0:v:0','-map','0:a:0','-f','null','-'], check=True)
    replay = output/'replay_data'; replay.mkdir(exist_ok=True)
    for name in ('report.json','audio_substeps.npz'):
        shutil.copy2(recording/name, replay/name)
    from package_interactive_replay import package
    interactive = package(output, recording, plan)
    index = output/'index.html'
    html = index.read_text()
    if 'midi_comparison/piano_roll_comparison.png' not in html:
        index.write_text(html+'\n<h2>Actual keys vs original MIDI</h2><a href="midi_comparison/comparison.json">Single replay metrics</a><img src="midi_comparison/piano_roll_comparison.png">\n')
    html = index.read_text()
    if 'interactive_replay/README.txt' not in html:
        index.write_text(html+'\n<h2>Interactive MuJoCo replay</h2><a href="interactive_replay/README.txt">Launch instructions (same physics recording, with audio)</a>\n')
    files = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name not in ('manifest.json','delivery_validation.json')}
    (output/'manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    result = dict(state='validated', checkpoint_sha256=plan['checkpoint_sha256'],
        physical_sample_micro_f1=metric['physical_sample_micro_f1'], sample_hz=metric['sample_hz'],
        audio_video_decoded=True, same_replay_comparison=True, interactive_replay=interactive)
    (output/'delivery_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--original-midi', type=Path)
    parser.add_argument('--original-midi-sha256')
    parser.add_argument('--registered-score', type=Path)
    parser.add_argument('--registered-score-sha256')
    args = parser.parse_args()
    finalize(args.export.resolve(), args.original_midi, args.original_midi_sha256, args.registered_score, args.registered_score_sha256)
