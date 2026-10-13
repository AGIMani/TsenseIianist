#!/usr/bin/env bash
# Shadow five reward terms with substep key/energy means and collision union.
# Fingering stays at control endpoints; this differs from the historical
# endpoint-only 2026-10-03 run. GPU execution remains inside Slurm.
# Usage: bash scripts/hpc/submit_twinkle_shadow_rewards.sh [training arguments]
# Default: one available GPU, 10,000,384 transitions, seed42; no warm-start.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/scripts/hpc/submit.sh" --gpus 1 train \
  --reward-config "$ROOT/configs/experiments/twinkle_shadow_only_20261003.json" \
  --reward-sampling auto --no-anticipatory-fingering-reward \
  --anticipatory-fingering-time 0.15 --no-anticipatory-current-priority \
  --no-staged-fingering-reward --substep-rewards \
  --target-key-substep-reward --target-key-activation-weight 0 \
  --exact-note-timing --upcoming-press-permission-time 0 \
  --best-f1-metric exact_original_music_micro_f1_min --target-exact-micro-f1 .99 \
  --seed 42 --output-dir "$ROOT/training_mjlab/outputs/hpc/twinkle_shadow_substep_20261004" "$@"
