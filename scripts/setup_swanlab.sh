#!/usr/bin/env bash
# Separate reporting environment; leave Composer / learner dependencies intact.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv "$ROOT/training_mjlab/monitoring/.venv"
"$ROOT/training_mjlab/monitoring/.venv/bin/python" -m pip install -r "$ROOT/training_mjlab/monitoring/requirements.txt"
