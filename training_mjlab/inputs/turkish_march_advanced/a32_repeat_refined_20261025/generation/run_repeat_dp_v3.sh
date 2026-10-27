#!/usr/bin/env bash
set -u
cd /home/houj/projects/pianist_turkish_hand_20261024
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 240s training/.venv/bin/python scripts/plan_model_fingering.py \
  --midi task_inputs/turkish-march-advanced.mid \
  --geometry task_outputs/geometry/model_geometry.json \
  --output task_outputs/fingering_repeat_refined_v3 \
  --thumb-penalty 200 --exchange-weight 1000 --recovery-time .05 \
  --fixed-fingering task_inputs/repeat_constraints_v2.json
status=$?
printf '%s\n' "$status" > task_outputs/fingering_repeat_refined_v3.exit
exit "$status"
