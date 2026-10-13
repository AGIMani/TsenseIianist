#!/usr/bin/env python3
"""Same-checkpoint zero-update control ablations using its archived model/score.

Only the explicitly selected current finger-clearance implementation is overlaid
on a temporary source archive. Each child probe verifies original scene/task
hashes and records networks-only migration; this is not historical replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((checkpoint.parent/'config.json').read_text())
    base = config['finger_clearance']
    limits = config['control_limits']
    overlay = ROOT/'tsensei_v3_2_pianist/mjlab_env/finger_clearance.py'
    shutil.copy2(overlay, output/'finger_clearance_overlay.py')
    (output/'provenance.json').write_text(json.dumps(dict(
        archive=str(args.archive.resolve()), checkpoint=str(checkpoint),
        overlay_sha256=hashlib.sha256(overlay.read_bytes()).hexdigest(),
        scope='Zero learner updates, same saved physical model/score; one canonical and one detailed episode per candidate.'), indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='v32_guard_ablation_') as temporary:
        unpacked = Path(temporary)
        with tarfile.open(args.archive) as archive:
            archive.extractall(unpacked, filter='data')
        candidates = list(unpacked.rglob('scripts/probe_clearance_contacts.py'))
        if len(candidates) != 1:
            raise ValueError('Archive must contain exactly one probe entry')
        source = candidates[0].parents[1]
        for folder in ('training', 'training_mjlab'):
            (source/folder).mkdir(exist_ok=True)
            (source/folder/'.venv').symlink_to(ROOT/folder/'.venv', target_is_directory=True)
        shutil.copy2(overlay, source/'tsensei_v3_2_pianist/mjlab_env/finger_clearance.py')
        limits_path = output/'saved_wrist_limits.json'
        limits_path.write_text(json.dumps(limits, indent=2)+'\n')
        for name, idle, active, wrist in (
                ('no_wrist_rate', True, True, False),
                ('no_idle_guard', False, True, True),
                ('no_active_permission', True, False, True),
                ('both_guards_off', False, False, True)):
            options = dict(base, idle_finger_protection=idle, active_press_permission=active)
            path = output/(name+'.json')
            path.write_text(json.dumps(options, indent=2)+'\n')
            command = [sys.executable, str(candidates[0]), '--checkpoint', str(checkpoint),
                       '--finger-clearance-config', str(path), '--no-include-baseline',
                       '--candidate-name', name, '--repeats', '1', '--detail-repeats', '1',
                       '--output', str(output/name)]
            if wrist:
                command += ['--control-limits', str(limits_path)]
            with (output/(name+'.log')).open('w') as log:
                subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output/name/'summary.json').read_text())[0]
            print(json.dumps(dict(candidate=name, music_f1=summary['music_f1_min'],
                key_underside_substeps=summary['contact_categories']['key_underside']['substeps'])), flush=True)


if __name__ == '__main__':
    main()
