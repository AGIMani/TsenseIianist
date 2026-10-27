#!/usr/bin/env bash
set -u
cd /home/houj/projects/pianist_turkish_hand_20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable
timeout 600s training_mjlab/.venv/bin/python scripts/hpc/piano_fullscore_contact_bank.py \
 --fingering task_outputs/fingering_repeat_refined_v6/fingering.csv \
 --geometry task_outputs/geometry/model_geometry.json \
 --recording task_outputs/refine_exchange500_thumb200_build \
 --expected-model-sha256 920475c61dc0469779d9f64769c75044140163151461201609ec181a7509fd4f \
 --expected-archive-sha256 1d4a67f15c796a849387ccc967c4960f4ee12d0b18bde2ff81f27a591c23d560 \
 --output task_outputs/fullscore_static_repeat_v6 \
 --workers 8 --node-cpu-budget 5 --max-nfev 60 \
 --cache task_outputs/fullscore_static_305_v2 \
 --cache task_outputs/fullscore_static_repeat_v5 \
 --cache task_outputs/repetition_paired_success_cache \
 --retry-failed \
 --seed-bank task_outputs/contact_bank_3s_v6_exchange500/report.json \
 --seed-bank task_outputs/hard_contact_banks_v1/right_one_thumb/report.json \
 --seed-bank task_outputs/hard_contact_banks_v1/left_one_thumb/report.json
status=$?
printf '%s\n' "$status" > task_outputs/fullscore_static_repeat_v6.exit
exit "$status"
