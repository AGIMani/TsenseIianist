#!/usr/bin/env bash
# config.json 默认为检查点同目录；--mode evaluate/preview/render。
# 示例：bash eval.sh --checkpoint training_mjlab/outputs/my-run/final.pt --mode preview
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROJECT_ROOT/training_mjlab/.venv/bin/python" scripts/eval_from_source.py "$@"
