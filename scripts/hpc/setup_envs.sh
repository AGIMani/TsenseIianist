#!/usr/bin/env bash
# User-owned Python environments; never install a driver or a system CUDA toolkit.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_DIR="$PROJECT_DIR/training_mjlab/hpc"
UV_VERSION=0.12.15
UV_BIN="$HPC_DIR/tools/uv"
COMPOSER_ENV="$PROJECT_DIR/training/.venv"
LEARNER_ENV="$PROJECT_DIR/training_mjlab/.venv"
MARKER_NAME=.tsensei-a32-hpc-environment

usage() {
  cat <<'EOF'
Usage: bash scripts/hpc/setup_envs.sh [--print-plan]

Creates training/.venv (Composer) and training_mjlab/.venv (Torch/MJLab).
Uses user-owned uv 0.12.15 and Python 3.12; leaves system CUDA and drivers intact.
Refuses existing environments not created by this script. Reruns repair only
script-owned environments. Requirements stay pinned; failures are not worked
around by silently changing versions. SwanLab is not installed.

TSENSEI_MJLAB_WHEEL may point to the verified offline wheel from the original
fixed-commit installation; its pinned SHA is checked before use.

--print-plan  Show intended locations without downloads or filesystem changes.
EOF
}

if (( $# > 1 )); then usage >&2; exit 2; fi
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --print-plan)
    printf 'Project: %s\nComposer: %s\nLearner: %s\nuv: %s\nPython: 3.12\nReports: %s/environment\n' \
      "$PROJECT_DIR" "$COMPOSER_ENV" "$LEARNER_ENV" "$UV_BIN" "$HPC_DIR"
    exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 0 )); then usage >&2; exit 2; fi

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
command -v flock >/dev/null || fail 'flock is required (normally included in Ubuntu util-linux).'

# Check both paths before any installation; never adopt an unrelated environment.
for role in composer learner; do
  if [[ "$role" == composer ]]; then env_dir="$COMPOSER_ENV"; else env_dir="$LEARNER_ENV"; fi
  if [[ -e "$env_dir" || -L "$env_dir" ]]; then
    [[ -d "$env_dir" && ! -L "$env_dir" && -O "$env_dir" ]] || fail "Unsafe environment path: $env_dir"
    [[ -f "$env_dir/$MARKER_NAME" ]] || fail "Existing unmanaged environment: $env_dir. Use a fresh project copy."
    [[ "$(cat -- "$env_dir/$MARKER_NAME")" == "tsensei-a32-hpc-v1:$role" ]] || fail "Environment marker differs: $env_dir"
  fi
done
[[ -f "$PROJECT_DIR/requirements-tested.txt" ]] || fail 'Missing Composer requirements-tested.txt.'
[[ -f "$PROJECT_DIR/training_mjlab/requirements-tested.txt" ]] || fail 'Missing MJLab requirements-tested.txt.'

mkdir -p -- "$HPC_DIR/tools" "$HPC_DIR/environment"
exec 9>"$HPC_DIR/setup.lock"
flock -n 9 || fail 'Another setup_envs.sh is already running for this project.'

if [[ ! -x "$UV_BIN" ]]; then
  command -v curl >/dev/null || fail 'curl is required to download the official uv installer.'
  installer="$(mktemp)"
  trap 'rm -f -- "$installer"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
    "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$installer"
  UV_INSTALL_DIR="$HPC_DIR/tools" UV_NO_MODIFY_PATH=1 sh "$installer"
  rm -f -- "$installer"
  trap - EXIT
fi
[[ "$("$UV_BIN" --version)" == "uv $UV_VERSION"* ]] || fail "Expected uv $UV_VERSION at $UV_BIN."

# Keep managed Python and downloaded wheels inside this project/user storage.
export UV_PYTHON_INSTALL_DIR="$HPC_DIR/python"
export UV_CACHE_DIR="$HPC_DIR/uv-cache"
export UV_LINK_MODE=copy
"$UV_BIN" --system-certs python install 3.12

create_owned_env() {
  local env_dir="$1" role="$2"
  if [[ ! -d "$env_dir" ]]; then
    mkdir -p -- "$env_dir"
    printf 'tsensei-a32-hpc-v1:%s\n' "$role" > "$env_dir/$MARKER_NAME"
  fi
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$UV_BIN" --system-certs venv --allow-existing --managed-python --python 3.12 "$env_dir"
  fi
  "$env_dir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
}

create_owned_env "$COMPOSER_ENV" composer
create_owned_env "$LEARNER_ENV" learner

"$UV_BIN" --system-certs pip install --python "$COMPOSER_ENV/bin/python" \
  -r "$PROJECT_DIR/requirements-tested.txt" 'dm-env-wrappers==0.0.13'
"$UV_BIN" pip check --python "$COMPOSER_ENV/bin/python"
learner_requirements="$PROJECT_DIR/training_mjlab/requirements-tested.txt"
if [[ -n "${TSENSEI_MJLAB_WHEEL:-}" ]]; then
  learner_requirements="$HPC_DIR/learner-offline-requirements.txt"
  "$LEARNER_ENV/bin/python" - "$PROJECT_DIR" "$TSENSEI_MJLAB_WHEEL" "$learner_requirements" <<'PY'
import hashlib
from pathlib import Path
import sys
root, wheel, output = map(Path, sys.argv[1:])
expected = 'ccb22b92162e67991f104c6499626f681b43afb91b1cb98b2bae7ce73f420f32'
if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
    raise SystemExit('Offline MJLab wheel does not match the verified fixed-commit artifact')
lines = (root / 'training_mjlab/requirements-tested.txt').read_text().splitlines()
matches = [index for index, line in enumerate(lines) if line.startswith('mjlab @ ')]
if len(matches) != 1 or '8ee51fbcf806a7419189f706d9e394cbeb7790fa' not in lines[matches[0]]:
    raise SystemExit('MJLab requirement no longer matches this offline artifact')
lines[matches[0]] = 'mjlab @ ' + wheel.resolve().as_uri()
output.write_text('\n'.join(lines) + '\n')
print('Using verified MJLab 1.6.0 wheel, fixed source commit 8ee51fbcf806a7419189f706d9e394cbeb7790fa')
PY
fi
"$UV_BIN" --system-certs pip install --python "$LEARNER_ENV/bin/python" \
  -r "$learner_requirements"
"$UV_BIN" pip check --python "$LEARNER_ENV/bin/python"

# No GPU imports here: environment installation is safe on a login node.
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_dir="$(mktemp -d "$HPC_DIR/environment/$stamp.XXXXXX")"
"$UV_BIN" pip freeze --python "$COMPOSER_ENV/bin/python" > "$report_dir/composer.lock.txt"
"$UV_BIN" pip freeze --python "$LEARNER_ENV/bin/python" > "$report_dir/learner.lock.txt"
"$COMPOSER_ENV/bin/python" - "$PROJECT_DIR" "$report_dir" "$UV_BIN" <<'PY'
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

root, report, uv = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
versions = {}
for role, relative in (("composer", "training"), ("learner", "training_mjlab")):
    python = root / relative / ".venv/bin/python"
    code = 'import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'
    versions[role] = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
data = {
    "schema": 1, "platform": platform.platform(),
    "uv": subprocess.check_output([uv, "--version"], text=True).strip(),
    "requirements_sha256": {str(p.relative_to(root)): sha(p) for p in (
        root / "requirements-tested.txt", root / "training_mjlab/requirements-tested.txt")},
    "composer_extra": "dm-env-wrappers==0.0.13", "environments": versions,
    "gpu_validation": "not_run; submit scripts/hpc/probe_gpus.py through Slurm",
}
(report / "installed.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
printf 'Environment installation complete. Actual locks: %s\nGPU/driver/Warp/Adam training compatibility has not been validated.\n' "$report_dir"
