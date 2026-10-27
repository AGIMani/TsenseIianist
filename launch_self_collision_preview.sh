#!/usr/bin/env bash
# Remote CPU collision probing; local display of verified, recorded MJB poses.
set -euo pipefail
preview_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
preview_python="$preview_root/training_mjlab/.venv/bin/python"
[[ -x "$preview_python" ]] || { echo 'Missing training_mjlab/.venv/bin/python.' >&2; exit 1; }
[[ -f "$preview_root/configs/self_collision_preview/remote.json" ]] || { echo 'Missing verified self-collision preview configuration.' >&2; exit 1; }
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "$available_kib" -ge 4194304 ]] || { echo 'Need 4 GiB available memory. Close other apps and retry.' >&2; exit 1; }
exec systemd-run --user --scope --quiet --unit="pianist-collision-preview-$$" \
  -p MemoryHigh=2300M -p MemoryMax=3G -p MemorySwapMax=0 \
  -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  MUJOCO_GL=glfw __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  "$preview_python" "$preview_root/scripts/self_collision_preview.py" "$@"
