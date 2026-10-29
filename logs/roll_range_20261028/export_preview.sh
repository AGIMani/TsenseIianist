#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_roll_range_20261028
source_root="$root/project/logs/hpc/10527/source"
cd "$source_root"
systemd-run --user --scope --quiet --unit=pianist-roll-range-preview \
 -p MemoryHigh=10G -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=300% \
 env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=disable \
 "$source_root/training_mjlab/.venv/bin/python" scripts/export_static_training_preview.py \
 --source-archive "$root/source_roll_range_v1.tar.gz" \
 --source-sha256 ec43ec5bfca879c947b11a3a04e12275dba94323cdcff42bf44448ecf251f3e2 \
 --launch-command "$root/acceptance/launch_command.json" \
 --score "$root/acceptance/original_score.proto" \
 --training-entry "$root/all_use_this_train.sh" --output "$root/static_preview"
