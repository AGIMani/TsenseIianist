#!/usr/bin/env python3
"""Evaluate/render a checkpoint using its trusted local source archive.

Example (from the project directory):
  training_mjlab/.venv/bin/python scripts/eval_from_source_archive.py \
    --source-archive training_mjlab/source_versions/before_preparation_20260926.tar.gz \
    --checkpoint training_mjlab/outputs/RUN/final.pt \
    --mode evaluate --output preview/outputs/archive_evaluation.json

This executes archived Python source. Use an archive you trust. Dependencies are
referenced from the existing local environments; source and compiled scenes are
temporary. Source/dependency/checkpoint consistency checks remain mandatory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'tsensei_v3_2_pianist'


def extract_archive(archive_path, destination):
    """Accept only ordinary files/directories inside the temporary directory."""
    with tarfile.open(archive_path, 'r:*') as archive:
        members = archive.getmembers()
        # Validate the entire list before writing anything. Reject links even when
        # their current target is inside the archive: no link traversal is needed.
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '.venv' in name.parts
                    or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe/unsupported source archive member: '+member.name)
        archive.extractall(destination, members=members, filter='data')


def validate_sources(directory, expected):
    source = directory/PACKAGE
    if not source.is_dir():
        raise ValueError(f'Archive is missing {PACKAGE}/ at its root')
    # Lock the action contract read by action_mapping.py outside the package.
    action_config = directory/'configs/actions'
    actual = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob('*')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')
    }
    actual.update({str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in action_config.glob('*.json') if path.is_file()})
    if actual != expected:
        different = sorted(key for key in actual.keys() | expected.keys()
                           if actual.get(key) != expected.get(key))
        raise ValueError('Archive source differs from checkpoint; no source substitution allowed: '
                         +', '.join(different[:8]))


def validate_scripts(directory, mode):
    required = [directory/'scripts/eval_from_source.py']
    if mode != 'evaluate':
        required.append(directory/'preview/scripts'/(
            'preview_mjlab.py' if mode == 'preview' else 'render_mjlab_checkpoint.py'))
    missing = [str(path.relative_to(directory)) for path in required if not path.is_file()]
    if missing:
        raise ValueError('Archive is incomplete for this replay mode; missing '+', '.join(missing)
                         +'. Use a complete matching archive; current-source scripts are not substituted.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Defaults to config.json beside checkpoint')
    parser.add_argument('--mode', choices=('evaluate', 'preview', 'render'), default='evaluate')
    parser.add_argument('--capture-physics', action='store_true',
                        help='For render: additionally record full 200 Hz state for offline arm IK')
    args, rest = parser.parse_known_args(argv)
    if args.capture_physics and args.mode != 'render':
        parser.error('--capture-physics requires --mode render')
    if any(value.split('=')[0] in ('--scene', '--task') for value in rest):
        parser.error('Archived-source replay reconstructs scene/task; explicit frozen bundles are not accepted')
    archive_path = args.source_archive.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = (args.config or checkpoint_path.parent/'config.json').resolve()
    for path in (archive_path, checkpoint_path, config_path):
        if not path.is_file():
            parser.error('Required file does not exist: '+str(path))
    environments = {name: ROOT/name/'.venv' for name in ('training', 'training_mjlab')}
    for path in environments.values():
        if not (path/'bin/python').is_file():
            parser.error('Existing project environment is required: '+str(path))
    import torch
    metadata = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['metadata']
    recipe = metadata.get('source_runtime')
    if not recipe or not isinstance(recipe.get('source_sha256'), dict):
        parser.error('Checkpoint has no complete source_runtime fingerprint')
    config = json.loads(config_path.read_text())
    if config.get('source_runtime') != recipe:
        parser.error('config.json source_runtime differs from checkpoint metadata')
    with tempfile.TemporaryDirectory(prefix='tsensei-v32-source-archive-') as temporary:
        directory = Path(temporary)
        try:
            extract_archive(archive_path, directory)
            validate_scripts(directory, args.mode)
            validate_sources(directory, recipe['source_sha256'])
        except (ValueError, tarfile.TarError) as error:
            parser.error(str(error))
        for name, target in environments.items():
            link = directory/name/'.venv'
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target.resolve(), target_is_directory=True)
        # Audio libraries/SoundFont are installed runtime dependencies, outside
        # the fingerprinted model package. Reuse them just like the Python envs.
        # If an archive supplies its own audio runtime, retain that exact copy.
        audio_runtime = ROOT/'training_mjlab/audio_runtime'
        archived_audio = directory/'training_mjlab/audio_runtime'
        if audio_runtime.is_dir() and not archived_audio.exists():
            archived_audio.symlink_to(audio_runtime.resolve(), target_is_directory=True)
        env = os.environ.copy()
        # Replace rather than append: do not fall through to the current project.
        env['PYTHONPATH'] = str(directory)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [str(directory/'training_mjlab/.venv/bin/python'),
                   str(directory/'scripts/eval_from_source.py'),
                   '--checkpoint', str(checkpoint_path), '--config', str(config_path),
                   '--mode', args.mode, *rest]
        if args.capture_physics:
            command = [str(directory/'training_mjlab/.venv/bin/python'),
                       str(ROOT/'scripts/capture_archive_render.py'), '--archive-root', str(directory),
                       '--checkpoint', str(checkpoint_path), '--config', str(config_path),
                       '--mode', args.mode, *rest]
        print(f'Archive source verified ({len(recipe["source_sha256"])} files): {archive_path}', flush=True)
        # Relative output locations keep the ordinary project's meaning. The
        # archived entrypoint explicitly inserts its own root before imports.
        return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    sys.exit(main())
