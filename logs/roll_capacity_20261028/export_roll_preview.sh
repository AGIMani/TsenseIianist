#!/usr/bin/env bash
set -euo pipefail
root=/home/houj/projects/pianist_roll_capacity_20261028
source_root="$root/project/logs/hpc/10477/source"
cd "$source_root"
systemd-run --user --scope --quiet --unit=pianist-roll-preview-build-20261028 \
 -p MemoryHigh=10G -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=300% \
 env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=disable \
 "$source_root/training_mjlab/.venv/bin/python" scripts/export_static_training_preview.py \
 --source-archive "$root/source_roll_v1.tar.gz" \
 --source-sha256 afe79f92b803211c390d3a495dc83462ecb37d263a10b5139ada62f4c93f963f \
 --launch-command "$root/roll_acceptance_v1/launch_command.json" \
 --score "$root/roll_acceptance_v1/original_score.proto" \
 --training-entry "$root/all_use_this_train.sh" --output "$root/static_roll_final"
