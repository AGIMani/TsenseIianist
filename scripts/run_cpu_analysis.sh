#!/usr/bin/env bash
# Isolate local CPU analysis from the desktop's memory budget.
set -euo pipefail
if (($# == 0)); then
  echo 'Usage: scripts/run_cpu_analysis.sh COMMAND [ARGS...]' >&2
  exit 2
fi
# Refuse a new heavy job when the desktop cannot absorb its bounded scope.
# These checks are intentionally conservative: a previous local MuJoCo build
# triggered global OOM and killed the Codex/Chromium scope.
min_free_gib=${PIANIST_CPU_MIN_FREE_GIB:-8}
min_available_gib=${PIANIST_CPU_MIN_AVAILABLE_GIB:-4}
free_kib=$(df -Pk . | awk 'NR==2 {print $4}')
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
min_free_kib=$((min_free_gib * 1024 * 1024))
min_available_kib=$((min_available_gib * 1024 * 1024))
if [[ -z "$free_kib" || "$free_kib" -lt "$min_free_kib" ]]; then
  echo "Refusing local CPU analysis: need ${min_free_gib}GiB free on $(pwd | cut -d/ -f1-3), have $(( ${free_kib:-0} / 1024 / 1024 ))GiB. Use Slurm or free space first." >&2
  exit 1
fi
if [[ -z "$available_kib" || "$available_kib" -lt "$min_available_kib" ]]; then
  echo "Refusing local CPU analysis: need ${min_available_gib}GiB MemAvailable, have $(( ${available_kib:-0} / 1024 / 1024 ))GiB." >&2
  exit 1
fi
exec systemd-run --user --scope --quiet \
  -p MemoryHigh="${PIANIST_CPU_MEMORY_HIGH:-2G}" \
  -p MemoryMax="${PIANIST_CPU_MEMORY_MAX:-3G}" \
  -p MemorySwapMax="${PIANIST_CPU_SWAP_MAX:-1G}" -- "$@"
