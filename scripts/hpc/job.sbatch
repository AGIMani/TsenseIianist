#!/usr/bin/env bash
#SBATCH --job-name=TsenseI-A32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
# The submit wrapper overrides this default with gpu:4 for multi-collector jobs.
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail
JOB_ID="${SLURM_JOB_ID:-${PIANIST_PROCESS_JOB_ID:-}}"
: "${JOB_ID:?Submit through scripts/hpc/submit.sh}"
PROJECT_DIR="$1"
MODE="$2"
shift 2
expected_gpus="${TSENSEI_EXPECTED_GPUS:-2}"
[[ "$expected_gpus" =~ ^(1|2|4)$ ]] || { echo 'TSENSEI_EXPECTED_GPUS must be 1, 2 or 4' >&2; exit 2; }
if [[ "$expected_gpus" == 1 ]]; then
  case "$MODE" in replay|train|validate|export|mjlab-preview|preparation-preview) ;; *) echo 'Unsupported single-GPU mode' >&2; exit 2;; esac
fi
JOB_DIR="$PROJECT_DIR/logs/hpc/$JOB_ID"
if [[ "${TSENSEI_ALLOCATION_STEP:-0}" == 1 ]]; then
  JOB_DIR="$JOB_DIR/steps/${SLURM_STEP_ID:?Missing allocated Slurm step ID}"
fi
mkdir -p "$JOB_DIR"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MAX_JOBS=4
# Optional native IK dependencies belong to this deployment, never the shared
# training environment. Propagate them to training and later Slurm exports.
if [[ -d "$PROJECT_DIR/training_mjlab/native_ik_deps" ]]; then
  export PYTHONPATH="$PROJECT_DIR/training_mjlab/native_ik_deps${PYTHONPATH:+:$PYTHONPATH}"
  export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-8}"
fi
# Use separate per-job JIT caches. Never alter Slurm's CUDA_VISIBLE_DEVICES.
export TORCH_EXTENSIONS_DIR="$JOB_DIR/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$JOB_DIR/torchinductor"
PYTHON_BIN="$PROJECT_DIR/training_mjlab/.venv/bin/python"
export PATH="$PROJECT_DIR/training_mjlab/.venv/bin:$PROJECT_DIR/training_mjlab/hpc/tools:$PATH"
if [[ "$MODE" == replay || "$MODE" == arm-replay || "$MODE" == export || "$MODE" == online-export || "$MODE" == preparation-preview ]] && [[ -d "$PROJECT_DIR/training_mjlab/audio_runtime/root/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$PROJECT_DIR/training_mjlab/audio_runtime/root/usr/lib/x86_64-linux-gnu:$PROJECT_DIR/training_mjlab/audio_runtime/root/usr/lib/x86_64-linux-gnu/pulseaudio${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -f "$PROJECT_DIR/training_mjlab/hpc/egl_env.sh" ]]; then
  source "$PROJECT_DIR/training_mjlab/hpc/egl_env.sh"
fi
if [[ -f "$PROJECT_DIR/training_mjlab/hpc/cuda_env.sh" ]]; then
  source "$PROJECT_DIR/training_mjlab/hpc/cuda_env.sh"
fi
if [[ "${TSENSEI_CUDA_COMPAT:-0}" == 1 ]]; then
  source "$PROJECT_DIR/training_mjlab/hpc/cuda_compat_env.sh"
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/hpc/probe_gpus.py" --expected-gpus "$expected_gpus" --output "$JOB_DIR/gpu_probe.json"
if [[ "$MODE" == probe ]]; then exit 0; fi
if ! command -v nvcc >/dev/null; then
  echo 'Training requires nvcc for the configured Adam extension; run scripts/hpc/setup_cuda.sh first.' >&2
  exit 1
fi

# Capture and run the exact source for this job. Subsequent SSH edits cannot
# change lazy imports or checkpoint reconstruction while training is running.
package_options=()
export_arm_assets=0
if [[ "$MODE" == export ]]; then
  export_arm_assets="$(python3 - "$@" <<'PY'
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--plan', required=True)
a = p.parse_args()
with open(a.plan) as f:
    plan = json.load(f)
print(int(any(v['arms'] for v in plan['videos']) or 'arm-joints' in plan['plots']))
PY
  )"
fi
if [[ "$MODE" == preview || "$MODE" == mjlab-preview || "$MODE" == arm-replay || "$MODE" == online-export || "$export_arm_assets" == 1 ]]; then
  package_options+=(--include-preview-assets)
fi
if [[ -n "${TSENSEI_JOB_SOURCE_ARCHIVE:-}" ]]; then
  [[ -n "${TSENSEI_JOB_SOURCE_SHA256:-}" ]] || { echo 'Pinned archive is missing its submitted SHA256' >&2; exit 1; }
  cp -- "$TSENSEI_JOB_SOURCE_ARCHIVE" "$JOB_DIR/source.tar.gz"
  actual_source_sha="$(sha256sum -- "$JOB_DIR/source.tar.gz" | cut -d ' ' -f 1)"
  [[ "$actual_source_sha" == "$TSENSEI_JOB_SOURCE_SHA256" ]] || { echo 'Submitted source archive changed while queued' >&2; exit 1; }
  python3 "$PROJECT_DIR/scripts/hpc/package_a32.py" --audit "$JOB_DIR/source.tar.gz" > "$JOB_DIR/source_audit.json"
  printf '%s  source.tar.gz\n' "$actual_source_sha" > "$JOB_DIR/source.tar.gz.sha256"
else
  python3 "$PROJECT_DIR/scripts/hpc/package_a32.py" --output "$JOB_DIR/source.tar.gz" "${package_options[@]}"
fi
mkdir -p "$JOB_DIR/source"
tar -xzf "$JOB_DIR/source.tar.gz" -C "$JOB_DIR/source"
RUN_ROOT="$JOB_DIR/source"
mkdir -p "$RUN_ROOT/training" "$RUN_ROOT/training_mjlab"
ln -s "$PROJECT_DIR/training/.venv" "$RUN_ROOT/training/.venv"
ln -s "$PROJECT_DIR/training_mjlab/.venv" "$RUN_ROOT/training_mjlab/.venv"
if [[ -d "$PROJECT_DIR/training_mjlab/native_ik_deps" ]]; then
  ln -s "$PROJECT_DIR/training_mjlab/native_ik_deps" "$RUN_ROOT/training_mjlab/native_ik_deps"
fi
if [[ -d "$PROJECT_DIR/training_mjlab/audio_runtime" ]]; then
  ln -s "$PROJECT_DIR/training_mjlab/audio_runtime" "$RUN_ROOT/training_mjlab/audio_runtime"
fi
cd "$RUN_ROOT"

# Query only the devices allocated to this job, identified by CUDA UUID.
GPU_UUIDS="$($PYTHON_BIN -c 'import torch; print(",".join("GPU-" + str(torch.cuda.get_device_properties(i).uuid).removeprefix("GPU-") for i in range(torch.cuda.device_count())))')"
printf '%s\n' "$GPU_UUIDS" > "$JOB_DIR/gpu_uuids.txt"
nvidia-smi --id="$GPU_UUIDS" \
  --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv -l 2 > "$JOB_DIR/gpu_utilization.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
sleep 1
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
  echo 'GPU telemetry failed to start; inspect gpu_utilization.csv and Slurm stderr.' >&2
  exit 1
fi
if [[ "$MODE" == benchmark || "$MODE" == performance ]]; then
  "$PYTHON_BIN" -m pytest -q tests/test_collector_pipeline.py \
    > "$JOB_DIR/collector_tests.log" 2>&1
  exec_command=("$PYTHON_BIN" scripts/hpc/benchmark.py --output "$JOB_DIR/benchmark" "$@")
  if [[ "$MODE" == performance ]]; then
    exec_command=("$PYTHON_BIN" scripts/hpc/benchmark.py --output "$JOB_DIR/performance" \
      --eval-interval 100 --repeats 1 \
      --modes single single_batched pipeline pipeline_batched paired_batched "$@")
  fi
elif [[ "$MODE" == validate ]]; then
  # Arguments are pytest arguments, never shell code. Source comes from this
  # job's immutable archive just as it does for training.
  exec_command=("$PYTHON_BIN" -m pytest -q "$@")
elif [[ "$MODE" == replay ]]; then
  exec_command=("$PYTHON_BIN" scripts/eval_from_source_archive.py "$@")
elif [[ "$MODE" == preview ]]; then
  exec_command=("$PROJECT_DIR/training/.venv/bin/python" preview/scripts/preview_pose.py "$@")
elif [[ "$MODE" == mjlab-preview ]]; then
  # The MJLab viewer and its headless reachability audit use the same learner
  # environment as training.  The source archive has already been extracted
  # into RUN_ROOT, so source_runtime verification cannot see live SSH edits.
  exec_command=("$PYTHON_BIN" preview/scripts/preview_mjlab.py "$@")
elif [[ "$MODE" == arm-replay ]]; then
  exec_command=("$PROJECT_DIR/training/.venv/bin/python" preview/scripts/replay_arm_ik.py "$@")
elif [[ "$MODE" == export ]]; then
  exec_command=("$PROJECT_DIR/training/.venv/bin/python" preview/scripts/export_video_job.py "$@")
elif [[ "$MODE" == online-export ]]; then
  exec_command=("$PYTHON_BIN" -m online_control.export checkpoint "$@")
elif [[ "$MODE" == strike-diagnostic ]]; then
  exec_command=("$PYTHON_BIN" scripts/hpc/strike_timing_diagnostic.py "$@")
elif [[ "$MODE" == preparation-preview ]]; then
  exec_command=("$PYTHON_BIN" preview/scripts/render_first_note.py "$@")
elif [[ "$MODE" == train ]]; then
  # Defaults are resolved by all_use_this_train.sh before submission.
  # Execute only the current archive entry; never recover deprecated defaults.
  exec_command=("$PYTHON_BIN" scripts/train_from_source.py \
    --output-dir "$PROJECT_DIR/training_mjlab/outputs/hpc/job-$JOB_ID" "$@")
  if [[ "$expected_gpus" == 1 ]]; then
    # Keep the training/reward budget; only place collection and learning on
    # the sole allocated device and disable the two-device execution pipeline.
    exec_command+=(--device cuda:0 --learner-device cuda:0 --no-collector-pipeline)
  fi
else
  echo "Unknown mode: $MODE" >&2; exit 2
fi
printf '%q ' "${exec_command[@]}" > "$JOB_DIR/command.txt"
printf '\n' >> "$JOB_DIR/command.txt"
"${exec_command[@]}"
