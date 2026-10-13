#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
exec python3 scripts/train_from_source.py \
  --height 0.12 --ty 0 --yaw 0.25 --control-hz 20 --physics-hz 200 \
  --no-arms --disable-wrist-tz --action-profile mixed_v1 \
  --self-collision-weight 0.2 --vector-steps 19532 \
  --reward-sampling auto --substep-rewards --target-key-substep-reward \
  --physics-graph --compile-learner --fused-optimizer --progress "$@"
