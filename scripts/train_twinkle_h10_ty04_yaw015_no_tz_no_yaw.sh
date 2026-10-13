#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_twinkle_h10_ty04_yaw015_no_tz.sh" --disable-wrist-yaw "$@"
