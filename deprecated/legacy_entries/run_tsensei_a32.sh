#!/usr/bin/env bash
# Historical compatibility alias; it forwards to the archived-style lower-level defaults.
# New training uses all_use_this_train.sh, which pins source and submits through Slurm.
# This alias does not apply the current unified defaults or submit a Slurm job itself.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PROJECT_ROOT/scripts/hpc/train_coupled_20hz.sh" "$@"
