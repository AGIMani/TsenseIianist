#!/usr/bin/env bash
# Independent environment: Python 3.12, Linux x86_64, CUDA 12.8 Torch wheels.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3.12 -m venv "$PROJECT_DIR/training_mjlab/.venv"
"$PROJECT_DIR/training_mjlab/.venv/bin/python" -m pip install 'uv==0.12.15'
"$PROJECT_DIR/training_mjlab/.venv/bin/uv" pip install --system-certs \
  --python "$PROJECT_DIR/training_mjlab/.venv/bin/python" \
  -r "$PROJECT_DIR/training_mjlab/requirements-tested.txt"
"$PROJECT_DIR/training_mjlab/.venv/bin/python" -m pip check
