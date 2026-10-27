#!/usr/bin/env bash
# 2026-10-25: External initial positions use world-frame metres; yaw is the palm-base joint angle in radians.
# Historical compatibility entry; current training defaults live only in all_use_this_train.sh.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
exec python3 scripts/train_from_source.py \
  --world-z 0.10 --world-x 0.14 --palm-base-yaw 0.15 --control-hz 20 --physics-hz 200 \
  --no-arms --disable-wrist-tz --action-profile mixed_v1 \
  --self-collision-weight 0.2 --vector-steps 19532 \
  --reward-sampling auto --substep-rewards --target-key-substep-reward \
  --physics-graph --compile-learner --fused-optimizer --progress "$@"
