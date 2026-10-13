#!/usr/bin/env bash
# Export the newest complete checkpoint of the current Shadow-sync run.
# Usage: bash preview/scripts/export_latest_training_video.sh [RUN_DIRECTORY]
# This exact historical run uses its matching source archive. Other runs use
# eval.sh and must match the current source; no fingerprint checks are bypassed.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
run_dir="${1:-training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333}"

checkpoint="$(python3 - "$run_dir" <<'PY'
from pathlib import Path
import sys
import time

run = Path(sys.argv[1]).resolve()
if not run.is_dir():
    raise SystemExit(f'Training directory does not exist: {run}')
now = time.time()
final = run / 'final.pt'
steps = sorted(run.glob('step-*.pt'), key=lambda p: int(p.stem.split('-')[1]))
for path in ([final] if final.is_file() else []) + list(reversed(steps)):
    # The trainer writes a temporary file, fsyncs it, then atomically publishes
    # the final *.pt name with os.replace. Temporary save files are not selected.
    # Keep a nonempty/age guard for files externally copied into the directory;
    # the replay loader still validates the selected checkpoint and metadata.
    if path.stat().st_size > 0 and now - path.stat().st_mtime >= 5:
        print(path)
        break
else:
    raise SystemExit(f'No complete checkpoint found in {run}')
PY
)"

stamp="$(date +%Y%m%d-%H%M%S-%N)"
output="$project_root/preview/outputs/replays/shadow_sync_${checkpoint##*/}"
output="${output%.pt}_$stamp"
log="${output}_render.log"
historical_run="$(realpath -m -- "$project_root/training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333")"
if [[ "${checkpoint%/*}" == "$historical_run" ]]; then
    eval_command=("$project_root/training_mjlab/.venv/bin/python"
        "$project_root/scripts/eval_from_source_archive.py"
        --source-archive "$project_root/training_mjlab/source_versions/before_preparation_20260926.tar.gz")
else
    eval_command=(bash "$project_root/eval.sh")
fi
mkdir -p -- "$(dirname -- "$output")"
if ! "${eval_command[@]}" --checkpoint "$checkpoint" --mode render --output "$output" >"$log" 2>&1; then
    tail -n 40 "$log" >&2
    exit 1
fi
video="$output/replay.mp4"
test -s "$video"
printf 'Checkpoint: %s\nVideo: %s\nReport: %s\n' "$checkpoint" "$video" "$output/report.json"
