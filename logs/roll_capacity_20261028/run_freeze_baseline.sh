#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_roll_capacity_20261028
parent=/home/houj/projects/pianist_score_coverage_20261028/source_score_coverage_v1.tar.gz
if systemd-run --user --scope --quiet --unit=pianist-capacity-freeze-20261028 -p MemoryHigh=8G -p MemoryMax=10G -p MemorySwapMax=0 -p CPUQuota=200% \
 python3 "$root/overlay/logs/roll_capacity_20261028/freeze_baseline.py" "$root/overlay" "$parent" "$root/baseline_source.tar.gz" > "$root/freeze_baseline.log" 2>&1; then status=0; else status=$?; fi
printf '%s\n' "$status" > "$root/freeze_baseline.exit_status"
cat "$root/freeze_baseline.log"
exit "$status"
