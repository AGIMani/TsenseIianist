#!/usr/bin/env python3
"""Create a local frozen preview runtime, only inside its bounded cgroup."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from fingering_editor_limits import verify_limits
from fingering_editor_config import connection_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '9299ab01d69563092b65cd4d91b09429ae99aa501cedbb797b9956bd098fece8'


def prepare():
    limits = verify_limits()
    archive = ROOT / 'logs/model_replace_20261020/source_supplied_hand_v6.tar.gz'
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SOURCE_SHA:
            raise ValueError('Pinned hand archive SHA mismatch')
    worker = ROOT / 'scripts/fingering_pose_worker.py'
    helper = ROOT / 'scripts/fingering_editor_limits.py'
    identity = hashlib.sha256(worker.read_bytes() + helper.read_bytes()).hexdigest()
    runtime = ROOT / 'logs/fingering_editor/local_runtimes' / f'{SOURCE_SHA[:8]}_{identity[:12]}'
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError('Need at least 3 GiB disk space')
    if not runtime.is_dir():
        temporary = runtime.with_name(runtime.name + '.preparing')
        temporary.mkdir(parents=True, exist_ok=False)
        # Trusted fixed digest, data filter and an atomic directory rename.
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter='data')
        for role in ('training', 'training_mjlab'):
            dependency = (ROOT / role / '.venv').resolve()
            if not (dependency / 'bin/python').is_file():
                raise FileNotFoundError(dependency)
            (temporary / role).mkdir(exist_ok=True)
            (temporary / role / '.venv').symlink_to(dependency, target_is_directory=True)
        for source in (worker, helper):
            shutil.copy2(source, temporary / 'scripts' / source.name)
        (temporary / 'EDITOR_SOURCE_SHA256').write_text(SOURCE_SHA + '\n')
        temporary.rename(runtime)
    for source in (worker, helper):
        if source.read_bytes() != (runtime / 'scripts' / source.name).read_bytes():
            raise ValueError('Local runtime worker/helper differs from requested source')
    config = dict(backend='local', runtime_root=str(runtime), source_sha256=SOURCE_SHA,
                  worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                  python=str(ROOT / 'training_mjlab/.venv/bin/python'), limits=limits)
    path = connection_config(ROOT, 'local')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + '\n')
    print(path, flush=True)
    return config


if __name__ == '__main__':
    prepare()
