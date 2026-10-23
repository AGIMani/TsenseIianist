#!/usr/bin/env bash
set -euo pipefail
scene_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PIANIST_REPLAY_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  ancestor="$scene_dir"
  while [[ "$ancestor" != / ]]; do
    if [[ -x "$ancestor/training_mjlab/.venv/bin/python" ]]; then python_bin="$ancestor/training_mjlab/.venv/bin/python"; break; fi
    ancestor="$(dirname -- "$ancestor")"
  done
fi
[[ -x "$python_bin" ]] || { echo 'Set PIANIST_REPLAY_PYTHON to matching MuJoCo Python.' >&2; exit 1; }
[[ -f "$scene_dir/model.mjb" ]] || { echo 'Missing saved model.mjb; local XML compilation is disabled.' >&2; exit 1; }
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "$available_kib" -ge 4194304 ]] || { echo 'Need 4 GiB available memory. Close other apps and retry.' >&2; exit 1; }
exec systemd-run --user --scope --quiet --unit="pianist-full-$$" -p MemoryHigh=2300M -p MemoryMax=3G -p MemorySwapMax=0 -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=glfw __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia "$python_bin" "$scene_dir/viewer.py" "$@"
