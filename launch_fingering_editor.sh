#!/usr/bin/env bash
# Separate bounded local-render scope; opt-in local construction/IK uses its
# own smaller worker scope and resource monitor.
set -euo pipefail
editor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
editor_python="$editor_root/.venv/bin/python"
if [[ ! -x "$editor_python" ]]; then
  echo "Missing $editor_python; install project dependencies in a user virtual environment." >&2
  exit 1
fi
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ "$available_kib" -ge 4194304 ]] || { echo 'Need at least 4 GiB MemAvailable. Preview was not started.' >&2; exit 1; }
exec systemd-run --user --scope --quiet --unit="pianist-fingering-ui-$$" \
  -p MemoryHigh=384M -p MemoryMax=512M -p MemorySwapMax=0 \
  -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$editor_python" "$editor_root/scripts/fingering_editor.py" "$@"
