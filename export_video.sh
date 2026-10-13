#!/usr/bin/env bash
# Unified checkpoint video export. GPU work is submitted through Slurm.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$project_dir/scripts/export_videos.py" "$@"
