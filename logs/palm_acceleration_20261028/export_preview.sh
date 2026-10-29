#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_palm_acceleration_20261028
parent=/home/houj/projects/pianist_palm_ranges_20261028
source_root="$parent/project/logs/hpc/10530/source"
for i in {1..50}; do [[ -x "$source_root/training_mjlab/.venv/bin/python" ]] && break; sleep 1; done
cd "$source_root"
systemd-run --user --scope --quiet --unit=pianist-palm-acceleration-preview \
 -p MemoryHigh=10G -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=300% \
 env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=disable \
 "$source_root/training_mjlab/.venv/bin/python" scripts/export_static_training_preview.py \
 --source-archive "$parent/source_palm_ranges_v1.tar.gz" \
 --source-sha256 5573700d6cfdd855ecaaef7173c08246b88138bc06c1d31902b316c6f638cfd6 \
 --launch-command "$root/final_launch_command.json" \
 --score "$parent/acceptance/original_score.proto" \
 --training-entry "$root/all_use_this_train.sh" --output "$root/static_preview"
