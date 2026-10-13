#!/usr/bin/env bash
# Install a private CUDA 12.8 compiler. No apt, driver install, or system writes.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$PROJECT_DIR/training_mjlab/.venv/bin/python"
[[ -x "$PYTHON" ]] || { printf 'Install the learner environment with setup_envs.sh first.\n' >&2; exit 1; }
exec "$PYTHON" - "$PROJECT_DIR" "$@" <<'PY'
import argparse
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

root = Path(sys.argv[1]).resolve()
parser = argparse.ArgumentParser(description="Install NVIDIA's CUDA 12.8.1 compiler components in user storage.")
parser.add_argument('--with-compat', action='store_true', help='Also unpack cuda-compat-12-8; activation remains a separate explicit source command.')
parser.add_argument('--print-plan', action='store_true', help='Show URLs, fixed SHA256 and paths without downloads or writes.')
args = parser.parse_args(sys.argv[2:])
hpc = root / 'training_mjlab/hpc'
prefix = hpc / 'cuda-12.8.1'
compat_prefix = hpc / 'cuda-compat-12-8-570.211.01'
redist = 'https://developer.download.nvidia.com/compute/cuda/redist/'
sources = [
    {'component': 'cuda_nvcc', 'version': '12.8.93',
     'url': redist + 'cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz',
     'sha256': '9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3', 'bytes': 79015464},
    {'component': 'cuda_cudart', 'version': '12.8.90',
     'url': redist + 'cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz',
     'sha256': '8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68', 'bytes': 1354240},
    {'component': 'cuda_cccl', 'version': '12.8.90',
     'url': redist + 'cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz',
     'sha256': '0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b', 'bytes': 928524},
]
compat_source = {
    'component': 'cuda-compat-12-8', 'version': '570.211.01-0ubuntu1',
    'url': 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-12-8_570.211.01-0ubuntu1_amd64.deb',
    'sha256': '4363d39d2fcec488e03d5aaac13ba7e775e60ccfb2b4bc22bf9a407510681493', 'bytes': 37946088,
}
header_packages = ('nvidia-cublas-cu12', 'nvidia-cusolver-cu12', 'nvidia-cusparse-cu12', 'nvidia-curand-cu12')
plan = {'schema': 1, 'prefix': str(prefix), 'sources': sources,
        'header_packages': list(header_packages),
        'compat': {'prefix': str(compat_prefix), 'source': compat_source} if args.with_compat else None,
        'manifest_source': redist + 'redistrib_12.8.1.json',
        'compat_checksum_source': 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz'}
if args.print_plan:
    print(json.dumps(plan, indent=2))
    raise SystemExit(0)
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('This pinned toolchain is only for Linux x86_64.')
if sys.version_info < (3, 12):
    raise SystemExit('Use the Python 3.12 learner environment.')
for command in ('c++', 'gcc'):
    if not shutil.which(command):
        raise SystemExit(f'Missing system host compiler {command}; ask the cluster administrator for the supported toolchain.')
if args.with_compat and not shutil.which('dpkg-deb'):
    raise SystemExit('dpkg-deb is needed to unpack the optional compat package without installing it.')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def guard_directory(path, marker):
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir() or (path / '.tsensei-cuda-owner').read_text().strip() != marker:
            raise SystemExit(f'Refusing an unmanaged or incompatible existing directory: {path}')


def download(source):
    target = cache / source['url'].rsplit('/', 1)[-1]
    if target.exists():
        if target.stat().st_size == source['bytes'] and digest(target) == source['sha256']:
            return target
        raise RuntimeError(f'Cached archive hash mismatch; inspect/remove this archive before retrying: {target}')
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            print(f"Downloading {source['component']} ({source['bytes'] / 1e6:.1f} MB)", flush=True)
            with urllib.request.urlopen(source['url'], timeout=120) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            if partial.stat().st_size != source['bytes'] or digest(partial) != source['sha256']:
                raise RuntimeError(f"SHA256/size verification failed for {source['url']}")
            partial.replace(target)
            return target
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


compiler_marker = 'tsensei-cuda-12.8.1-v1'
compat_marker = 'tsensei-cuda-compat-12-8-570.211.01-v1'
installations = [(prefix, compiler_marker)]
if args.with_compat:
    installations.append((compat_prefix, compat_marker))
for path, marker in installations:
    try:
        guard_directory(path, marker)
    except FileNotFoundError:
        raise SystemExit(f'Refusing an existing directory without our ownership marker: {path}')

# Only read distribution metadata and files; importing Torch could initialize GPUs.
headers = []
for package in header_packages:
    dist = metadata.distribution(package)
    files = [item for item in dist.files or () if '/include/' in str(item) and not str(item).endswith(('.py', '.pyc'))]
    if not files:
        raise SystemExit(f'{package} has no development headers; finish the locked environment installation first.')
    headers.append((package, dist.version, dist, files))
plan['header_packages'] = {name: version for name, version, _, _ in headers}

hpc.mkdir(parents=True, exist_ok=True)
lock = (hpc / 'cuda-setup.lock').open('w')
try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('Another setup_cuda.sh is already running for this project.')
cache = hpc / 'cuda-downloads'
cache.mkdir(exist_ok=True)

if not prefix.exists():
    with tempfile.TemporaryDirectory(prefix='.cuda-stage-', dir=hpc) as temporary:
        staging = Path(temporary) / 'toolkit'
        staging.mkdir()
        licenses = staging / 'component-licenses'
        licenses.mkdir()
        for source in sources:
            archive = download(source)
            unpack = Path(temporary) / source['component']
            unpack.mkdir()
            with tarfile.open(archive, 'r:xz') as handle:
                handle.extractall(unpack, filter='data')
            children = list(unpack.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise RuntimeError(f'Unexpected archive layout: {archive}')
            component_root = children[0]
            for candidate in component_root.glob('*LICENSE*'):
                if candidate.is_file():
                    shutil.copy2(candidate, licenses / f"{source['component']}-{candidate.name}")
            shutil.copytree(component_root, staging, dirs_exist_ok=True, symlinks=True)
        # CUDAContextLight.h includes these math-library headers even for Adam.
        # Reuse the already-pinned wheel headers instead of downloading whole libraries.
        header_record = {}
        for package, version, dist, files in headers:
            for item in files:
                relative = str(item).split('/include/', 1)[1]
                source = Path(dist.locate_file(item))
                target = staging / 'include' / relative
                if target.exists() and digest(target) != digest(source):
                    raise RuntimeError(f'Conflicting development header: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                header_record[relative] = {'package': package, 'version': version, 'sha256': digest(target)}
        if not (staging / 'lib64').exists():
            (staging / 'lib64').symlink_to('lib', target_is_directory=True)
        for relative in ('bin/nvcc', 'include/cuda_runtime.h', 'include/cublas_v2.h',
                         'include/cusolverDn.h', 'include/cusparse.h', 'include/cub/cub.cuh', 'lib/libcudart.so'):
            if not (staging / relative).exists():
                raise RuntimeError(f'Missing required toolchain file: {relative}')
        (staging / '.tsensei-cuda-owner').write_text(compiler_marker + '\n')
        (staging / 'installed.json').write_text(json.dumps({**plan, 'copied_headers': header_record}, indent=2) + '\n')
        staging.rename(prefix)
else:
    old = json.loads((prefix / 'installed.json').read_text())
    if old['sources'] != sources or old['header_packages'] != plan['header_packages']:
        raise SystemExit('Existing private CUDA installation differs from pinned components/wheel headers; use a fresh project copy.')

if args.with_compat and not compat_prefix.exists():
    archive = download(compat_source)
    with tempfile.TemporaryDirectory(prefix='.cuda-compat-stage-', dir=hpc) as temporary:
        staging = Path(temporary) / 'compat'
        staging.mkdir()
        subprocess.run(['dpkg-deb', '--extract', str(archive), str(staging)], check=True)
        library_dir = staging / 'usr/local/cuda-12.8/compat'
        if not (library_dir / 'libcuda.so.1').exists():
            raise RuntimeError('The unpacked compat package does not provide libcuda.so.1.')
        (staging / '.tsensei-cuda-owner').write_text(compat_marker + '\n')
        (staging / 'installed.json').write_text(json.dumps(compat_source, indent=2) + '\n')
        staging.rename(compat_prefix)

activation = hpc / 'cuda_env.sh'
activation.write_text('# Generated user-only CUDA 12.8 compiler environment.\n'
                      f'export CUDA_HOME={shlex.quote(str(prefix))}\n'
                      'export PATH="$CUDA_HOME/bin:$PATH"\n'
                      'export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n')
if args.with_compat:
    compat_activation = hpc / 'cuda_compat_env.sh'
    compat_libraries = compat_prefix / 'usr/local/cuda-12.8/compat'
    compat_activation.write_text(
        '# Optional CUDA compatibility libraries; source only for the intended older-driver GPU job.\n'
        '# Do not use these older compatibility libraries on a newer driver (580+).\n'
        f'export LD_LIBRARY_PATH={shlex.quote(str(compat_libraries))}' + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n')
    print(f'Optional compatibility libraries unpacked; activate explicitly: source {shlex.quote(str(compat_activation))}')
subprocess.run([str(prefix / 'bin/nvcc'), '--version'], check=True)
print(f'Compiler installed. Activate: source {shlex.quote(str(activation))}')
print('No GPU test, Adam compilation, driver compatibility, or training validation has run.')
PY
