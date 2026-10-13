#!/usr/bin/env python3
"""CPU-only delivery checks and original-score plot for one completed export."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_videos import sha256
from eval_from_source_archive import validate_sources


def finalize(output):
    plan = json.loads((output/'export_plan.json').read_text())
    report = json.loads((output/'report.json').read_text())
    config_path = output/'inputs/config.json'
    config = json.loads(config_path.read_text())
    if json.loads((output/'status.json').read_text())['state'] != 'completed':
        raise ValueError('Video export is not completed')
    if not (plan['audio'] and plan['fingering'] and plan['finger_labels'] and plan['key_highlight']):
        raise ValueError('Required audio/fingering/labels/key highlights are missing')
    if plan['start'] != 0 or plan['end'] is not None or plan['speed'] != 1:
        raise ValueError('Training delivery requires the full original-tempo replay')
    if sha256(config_path) != plan['config_sha256'] or sha256(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('Frozen config/checkpoint changed')
    validate_sources(ROOT, config['source_runtime']['source_sha256'])
    recording = Path(report['recording'])
    reference = output/'original_reference.json'
    # This automatic path supports the registered Twinkle score. Never guess an
    # external score's provenance or turn quantized goals into original events.
    recipe = config['source_runtime']
    if recipe['external_score'] or recipe['environment']['midi'] != 'TwinkleTwinkleRousseau':
        raise ValueError('Automatic original-score provenance is not implemented for this score')
    from note_seq.midi_io import midi_file_to_note_sequence
    from note_seq import sequences_lib
    midi = ROOT/'tsensei_v3_2_pianist/music/data/rousseau/twinkle-twinkle-trimmed.mid'
    seq = midi_file_to_note_sequence(str(midi))
    origin = seq.notes[0].start_time
    seq = sequences_lib.extract_subsequence(seq, origin, seq.notes[-1].end_time)
    reference.write_text(json.dumps(dict(config_sha256=sha256(config_path), midi_path=str(midi),
        midi_sha256=sha256(midi), trim_origin_seconds=origin,
        source='Original registered MIDI from verified checkpoint source; same trim_silence as task',
        events=[[n.start_time,n.end_time,n.pitch] for n in seq.notes]), indent=2)+'\n')
    comparison = output/'midi_comparison'
    if not (comparison/'comparison.json').is_file():
        # An interrupted CPU plot can be rebuilt from the same immutable replay.
        if comparison.exists():
            shutil.rmtree(comparison)
        subprocess.run([sys.executable, str(ROOT/'scripts/plot_recorded_midi_comparison.py'),
            '--export', str(output), '--recording', str(recording),
            '--reference-events', str(reference), '--label', Path(plan['checkpoint_origin']).parent.name+' best.pt'], check=True)
    metric = json.loads((comparison/'comparison.json').read_text())
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
    index = output/'index.html'
    html = index.read_text()
    if 'midi_comparison/piano_roll_comparison.png' not in html:
        index.write_text(html+'\n<h2>Actual keys vs original MIDI</h2><a href="midi_comparison/comparison.json">Single replay metrics</a><img src="midi_comparison/piano_roll_comparison.png">\n')
    files = {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name not in ('manifest.json','delivery_validation.json')}
    (output/'manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    result = dict(state='validated', checkpoint_sha256=plan['checkpoint_sha256'],
        physical_sample_micro_f1=metric['physical_sample_micro_f1'], sample_hz=metric['sample_hz'],
        audio_video_decoded=True, same_replay_comparison=True)
    (output/'delivery_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    finalize(parser.parse_args().export.resolve())
