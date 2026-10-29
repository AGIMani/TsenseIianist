#!/usr/bin/env bash
# Local static viewer of the verified 1224 mm piano. No compilation or mj_step.
set -euo pipefail
preview_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --self-collision ]]; then
  shift
  exec "$preview_root/launch_self_collision_preview.sh" "$@"
fi
preview_python="$preview_root/training_mjlab/.venv/bin/python"
[[ -x "$preview_python" ]] || { echo 'Missing training_mjlab/.venv/bin/python.' >&2; exit 1; }
[[ -f "$preview_root/preview/outputs/piano_tcp_soft_20261029/model.mjb" ]] || {
  echo 'Missing prepared piano model; restore preview/outputs/piano_tcp_soft_20261029.' >&2; exit 1;
}
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "$available_kib" -ge 4194304 ]] || { echo 'Need at least 4 GiB available memory. Preview was not started.' >&2; exit 1; }
exec systemd-run --user --scope --quiet --unit="pianist-static-preview-$$" \
  -p MemoryHigh=2300M -p MemoryMax=3G -p MemorySwapMax=0 \
  -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  MUJOCO_GL=glfw __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  "$preview_python" "$preview_root/scripts/view_static_piano.py" \
  --folder "$preview_root/preview/outputs/piano_tcp_soft_20261029" "$@"
