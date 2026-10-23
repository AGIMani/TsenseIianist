#!/usr/bin/env bash
#SBATCH --job-name=TsenseI-reserve2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
# Keep the allocation alive between workloads. Slurm owns the end time and
# GPU isolation; this process performs no GPU computation or utilization loop.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through scripts/hpc/submit.sh reserve}"
PROJECT_DIR="$1"
expected_gpus="${TSENSEI_EXPECTED_GPUS:-2}"
[[ "$expected_gpus" =~ ^(2|4)$ ]] || { echo 'Reserve requires two or four GPUs' >&2; exit 2; }
mkdir -p "$PROJECT_DIR/logs/hpc/$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" > "$PROJECT_DIR/logs/hpc/$SLURM_JOB_ID/allocation.txt"
printf '%s GPUs allocated for up to 48 hours. Job: %s; node: %s; started: %s\n' \
  "$expected_gpus" "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$(date --iso-8601=seconds)"
printf 'Run: bash scripts/hpc/submit.sh --gpus %s --allocation %s MODE [arguments...]\n' "$expected_gpus" "$SLURM_JOB_ID"
printf 'Release early: scancel %s\n' "$SLURM_JOB_ID"
trap 'exit 0' TERM INT
while :; do
  sleep 60 &
  wait "$!" || true
done
