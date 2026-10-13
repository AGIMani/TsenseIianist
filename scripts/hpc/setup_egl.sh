#!/usr/bin/env bash
# Private Ubuntu Jammy GLVND dispatcher; no apt install or system writes.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_DIR="$PROJECT_DIR/training_mjlab/hpc"
EGL_PREFIX="$HPC_DIR/egl-jammy-1.4.0-1"
LIB_DIR="$EGL_PREFIX/usr/lib/x86_64-linux-gnu"
OWNER=tsensei-egl-jammy-1.4.0-1-v1
BASE_URL=https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd
PACKAGES=(libegl1 libglvnd0)
HASHES=(4a35c0e925e15a076e7ce11d7c76f8ecb16615fc347c2c09afa4c205a7ca8ec4 6956d8be93b160b439a043141a061adfd54561d877423a956ace84d2fb3bfb4e)

usage() {
  cat <<'EOF'
Usage: bash scripts/hpc/setup_egl.sh [--print-plan]
Downloads fixed Ubuntu 22.04 libegl1/libglvnd0 1.4.0-1 packages, verifies
SHA256, and unpacks them into training_mjlab/hpc. No package manager install,
system changes, or GPU initialization. Generate egl_env.sh for explicit use.
EOF
}
if (( $# > 1 )); then usage >&2; exit 2; fi
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --print-plan)
    printf 'Prefix: %s\n' "$EGL_PREFIX"
    for index in 0 1; do
      printf '%s/%s_1.4.0-1_amd64.deb\nSHA256 %s\n' "$BASE_URL" "${PACKAGES[$index]}" "${HASHES[$index]}"
    done
    exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This dispatcher is pinned for Linux x86_64.'
for command in curl dpkg-deb sha256sum flock; do
  command -v "$command" >/dev/null || fail "Required command missing: $command"
done
if [[ -e "$EGL_PREFIX" || -L "$EGL_PREFIX" ]]; then
  [[ -d "$EGL_PREFIX" && ! -L "$EGL_PREFIX" && -f "$EGL_PREFIX/.tsensei-egl-owner" ]] || fail "Refusing unmanaged EGL directory: $EGL_PREFIX"
  [[ "$(cat -- "$EGL_PREFIX/.tsensei-egl-owner")" == "$OWNER" ]] || fail 'Existing EGL ownership marker differs.'
fi
PYTHON="$PROJECT_DIR/training_mjlab/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
mkdir -p -- "$HPC_DIR"
exec 9>"$HPC_DIR/egl-setup.lock"
flock -n 9 || fail 'Another setup_egl.sh is running for this project.'

if [[ ! -d "$EGL_PREFIX" ]]; then
  cache="$HPC_DIR/egl-downloads"
  mkdir -p -- "$cache"
  staging="$(mktemp -d "$HPC_DIR/.egl-stage.XXXXXX")"
  trap 'rm -rf -- "$staging"' EXIT
  mkdir -p -- "$staging/root"
  for index in 0 1; do
    package="${PACKAGES[$index]}"
    checksum="${HASHES[$index]}"
    archive="$cache/${package}_1.4.0-1_amd64.deb"
    if [[ ! -f "$archive" ]]; then
      curl --proto '=https' --tlsv1.2 --fail --location --retry 3 --silent --show-error \
        "$BASE_URL/${package}_1.4.0-1_amd64.deb" -o "$archive.part"
      printf '%s  %s\n' "$checksum" "$archive.part" | sha256sum --check --status || fail "Download SHA256 mismatch: $package"
      mv -- "$archive.part" "$archive"
    fi
    printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status || fail "Cached SHA256 mismatch: $archive"
    dpkg-deb --extract "$archive" "$staging/root"
  done
  printf '%s\n' "$OWNER" > "$staging/root/.tsensei-egl-owner"
  "$PYTHON" - "$staging/root/installed.json" "$BASE_URL" "${HASHES[0]}" "${HASHES[1]}" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({
    'schema': 1, 'version': '1.4.0-1', 'platform': 'ubuntu22.04-amd64',
    'packages': [{'name': name, 'sha256': digest,
                  'url': f'{sys.argv[2]}/{name}_1.4.0-1_amd64.deb'}
                 for name, digest in zip(('libegl1', 'libglvnd0'), sys.argv[3:5])],
    'gpu_validation': 'not_run; library load and symbol checks only',
}, indent=2) + '\n')
PY
  mv -- "$staging/root" "$EGL_PREFIX"
  rm -rf -- "$staging"
  trap - EXIT
fi

# Loading dispatcher symbols does not initialize an EGL display or a GPU.
env LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PYTHON" - "$LIB_DIR" <<'PY'
import ctypes
from pathlib import Path
import sys
root = Path(sys.argv[1])
ctypes.CDLL(str(root / 'libGLdispatch.so.0'))
egl = ctypes.CDLL(str(root / 'libEGL.so.1'))
assert getattr(egl, 'eglQueryString')
assert getattr(egl, 'eglGetProcAddress')
print('GLVND dispatcher load and EGL symbols passed; no EGL display initialized.')
PY
{
  printf '# Generated private EGL dispatcher environment; system NVIDIA vendor configuration is preserved.\n'
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$LIB_DIR"
  printf 'export MUJOCO_GL=egl\nexport PYOPENGL_PLATFORM=egl\n'
} > "$HPC_DIR/egl_env.sh"
printf 'Activate for the Slurm job: source %s/egl_env.sh\nRendering still requires the installed NVIDIA vendor and a GPU allocation.\n' "$HPC_DIR"
