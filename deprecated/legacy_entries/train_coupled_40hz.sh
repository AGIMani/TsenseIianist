#!/usr/bin/env bash
# Compatibility name only: current training defaults are now 20/200 Hz.
# Historical jobs replay their archived source; explicit trailing overrides
# --control-hz 40 --physics-hz 400 --lookahead 20 still select that frequency.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/scripts/hpc/train_coupled_20hz.sh" "$@"
