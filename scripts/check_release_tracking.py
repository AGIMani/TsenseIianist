#!/usr/bin/env python3
"""Register the pinned release, or verify its committed/index LFS dependency."""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.PIPE).decode()


def release(entry):
    paths = re.findall(r'^training_archive="\$ROOT/([^"\n]+)"$', entry, re.M)
    hashes = re.findall(r'^training_archive_sha=([0-9a-f]{64})$', entry, re.M)
    if len(paths) != 1 or len(hashes) != 1:
        raise ValueError('Expected one pinned archive and SHA in the unified entry')
    path = Path(paths[0])
    if path.is_absolute() or '..' in path.parts or path.parts[0] != 'logs':
        raise ValueError('Pinned release must be a safe project-relative logs path')
    return path.as_posix(), hashes[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true', help='Whitelist current archive and sidecar; then git add and verify')
    parser.add_argument('--ref', default='HEAD', help='Git revision to verify, or INDEX for staged contents')
    args = parser.parse_args()
    if args.register:
        path, _ = release((ROOT / 'all_use_this_train.sh').read_text())
        for name in (path, path + '.sha256'):
            if not (ROOT / name).is_file():
                raise ValueError(f'Missing release dependency: {name}')
        parent = Path(path).parent.as_posix()
        ignore = ROOT / '.gitignore'
        existing = ignore.read_text()
        additions = []
        if f'!/{parent}/' not in existing.splitlines():
            additions.extend([f'!/{parent}/', f'/{parent}/*'])
        for name in (path, path + '.sha256'):
            rule = f'!/{name}'
            if rule not in existing.splitlines():
                additions.append(rule)
        if additions:
            with ignore.open('a') as f:
                f.write('\n# Pinned release registered from all_use_this_train.sh\n' + '\n'.join(additions) + '\n')
        # General source-archive LFS rule also covers future release directories.
        print('Registered archive and sidecar. Run git add -A, then this script --ref INDEX.')
        return
    prefix = ':' if args.ref == 'INDEX' else args.ref + ':'
    path, expected = release(git('show', prefix + 'all_use_this_train.sh'))
    pointer = git('show', prefix + path)
    if not pointer.startswith('version https://git-lfs.github.com/spec/v1\n'):
        raise ValueError(f'{path} is not a Git LFS pointer')
    if f'oid sha256:{expected}\n' not in pointer:
        raise ValueError('Entry SHA differs from committed LFS object SHA')
    sidecar = git('show', prefix + path + '.sha256').split()
    if not sidecar or sidecar[0] != expected:
        raise ValueError('SHA sidecar differs from entry')
    print(f'PASS: {args.ref} tracks pinned archive, LFS SHA and sidecar: {path}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'Release tracking FAILED: {exc}. Register with --register, stage files and recheck --ref INDEX.')
