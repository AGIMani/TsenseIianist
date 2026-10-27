#!/usr/bin/env bash
set -u
cd /home/houj/projects/pianist_turkish_hand_20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= MUJOCO_GL=disable
timeout 480s training_mjlab/.venv/bin/python task_scripts/paired_contact_probe.py \
 --baseline-bank task_outputs/fullscore_static_305_v2 \
 --candidate-bank task_outputs/fullscore_static_repeat_v5 \
 --output task_outputs/repetition_paired_contact_v1
status=$?
printf '%s\n' "$status" > task_outputs/repetition_paired_contact_v1.exit
exit "$status"
