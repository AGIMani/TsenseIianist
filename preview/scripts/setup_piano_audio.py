#!/usr/bin/env python3
"""Ubuntu 22.04: project-local piano synthesis libraries, without sudo.

--mirror https://archive.ubuntu.com/ubuntu/ avoids a broken configured mirror.
Downloads are verified against apt-cache SHA-256 metadata before extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
from urllib.parse import urljoin
from urllib.request import urlopen

PACKAGES = ('libfluidsynth3', 'libinstpatch-1.0-2', 'timgm6mb-soundfont',
            'libjack-jackd2-0', 'libasound2', 'libpulse0', 'libsndfile1', 'libsdl2-2.0-0',
            'libflac8', 'libvorbis0a', 'libvorbisenc2', 'libopus0', 'libogg0', 'libgbm1',
            'libwayland-egl1', 'libwayland-client0', 'libwayland-cursor0', 'libwayland-server0',
            'libxkbcommon0', 'libdecor-0-0', 'libasyncns0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', help='Optional HTTPS Ubuntu archive mirror')
    args = parser.parse_args()
    if args.mirror and not args.mirror.startswith('https://'):
        parser.error('--mirror requires HTTPS')
    root = Path(__file__).resolve().parents[2]/'training_mjlab/audio_runtime'
    packages = root/'packages'
    packages.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in PACKAGES:
        text = subprocess.check_output(['apt-cache', 'show', name], text=True)
        records = [dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)
                   for block in text.split('\n\n')]
        metadata = next((r for r in records if 'Filename' in r and 'SHA256' in r), None)
        if metadata is None:
            raise ValueError('No apt package metadata for '+name)
        target = packages/Path(metadata['Filename']).name
        valid = lambda: target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == metadata['SHA256']
        if not valid():
            if args.mirror:
                with urlopen(urljoin(args.mirror.rstrip('/')+'/', metadata['Filename']), timeout=90) as response:
                    target.write_bytes(response.read())
            else:
                subprocess.run(['apt-get', 'download', name+'='+metadata['Version']], cwd=packages, check=True)
        if not valid():
            raise ValueError('Package checksum mismatch: '+str(target))
        subprocess.run(['dpkg-deb', '-x', str(target), str(root/'root')], check=True)
        manifest.append({key: metadata[key] for key in ('Package', 'Version', 'Filename', 'SHA256')})
    licenses = root/'root/usr/share/common-licenses'
    licenses.mkdir(parents=True, exist_ok=True)
    for name in ('GPL-2', 'GPL-3', 'LGPL-2', 'LGPL-2.1'):
        source = Path('/usr/share/common-licenses')/name
        if source.exists():
            shutil.copy2(source, licenses/name)
    (root/'packages.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Local runtime installed: {root}; package copyright files retained under root/usr/share/doc')


if __name__ == '__main__':
    main()
