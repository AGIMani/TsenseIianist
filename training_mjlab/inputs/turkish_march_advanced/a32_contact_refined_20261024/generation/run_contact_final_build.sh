#!/usr/bin/env bash
set -u
cd /home/houj/projects/pianist_turkish_hand_20261024 || exit 2
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= PINNED_SOURCE_SHA256=1d4a67f15c796a849387ccc967c4960f4ee12d0b18bde2ff81f27a591c23d560
test ! -e task_outputs/refine_contact_final_build || exit 3
training_mjlab/.venv/bin/python -u scripts/hpc/plan_piano_ik.py --score training_mjlab/inputs/turkish_march_advanced/a32_contact_refined_20261024/turkish-march-advanced_fingered.proto --original-midi task_inputs/turkish-march-advanced.mid --output task_outputs/refine_contact_final_build --physics-hz 1000 --build-only > task_outputs/refine_contact_final_build.log 2>&1 &
pid=$!
echo "$pid" > task_outputs/refine_contact_final_build.pid
echo "PID=$pid"
wait "$pid"
result=$?
echo "$result" > task_outputs/refine_contact_final_build.exit
echo "EXIT_STATUS=$result"
tail -8 task_outputs/refine_contact_final_build.log
exit "$result"
