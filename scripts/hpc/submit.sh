#!/usr/bin/env bash
# A3.2 Slurm 基础设施入口；普通 SSH 只提交作业，GPU 工作在分配内执行。
# 新训练统一使用根目录 all_use_this_train.sh，由其维护当前参数并固定源码归档/SHA。
# 本脚本的 train 模式是底层兼容调用；直接调用不会自动采用统一入口的参数或固定归档。
# job.sbatch 在所选归档中调用历史 train_coupled 入口，再应用传入的末尾覆盖参数。
# 奖励以统一入口＋本轮末尾参数＋最终 config.json 为准，启动时打印完整奖励配置。
# 可选机械臂只由显式 --arms 及双侧各7个初角启用；使用统一入口可在申请GPU前检查。
# 申请多卡不意味着学习器梯度跨卡并行，也不等于已经实测加速；四卡仅在多采样器核心验收后使用。
# --gpus 1：支持新分配中的train/validate/reward-diagnostic/replay/export/mjlab-preview，不与--allocation合用。
#   单卡train在用户参数后强制cuda:0采样/学习及--no-collector-pipeline，其余训练预算不变。
# --gpus 4：为三路采样＋一张学习卡的多采样器训练预留四卡；训练参数必须显式指定
#   --collector-devices cuda:0,cuda:1,cuda:2 --learner-device cuda:3。
#   省略--gpus仍申请两卡；validate所选测试须适用于实际获配卡数。
# 示例：
#   bash scripts/hpc/submit.sh reserve
#   bash all_use_this_train.sh
#   bash all_use_this_train.sh --dependency afterok:123 --kill-on-invalid-dep --seed 43
#   bash scripts/hpc/submit.sh --gpus 1 validate tests/test_shadow_reward_only.py
#   bash scripts/hpc/submit.sh --gpus 4 probe
#   bash scripts/hpc/submit.sh performance --samples 262144 --repeats 1
#   bash scripts/hpc/submit.sh validate tests/test_collector_pipeline.py
#   bash scripts/hpc/submit.sh replay --source-archive /path/source.tar.gz \
#     --checkpoint /path/final.pt --mode render --output /path/new_replay
#   bash scripts/hpc/submit.sh --gpus 1 replay --source-archive /path/source.tar.gz \
#     --checkpoint /path/best.pt --mode evaluate --output /path/new_evaluation
# benchmark 保留旧单卡/流水、顺序评估基准；performance 增加批量评估与
# 两卡各跑一个独立实验的总吞吐对照。测试 F1 不用于判断长期收敛收益。
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
submission_options=()
allocation_id=""
gpu_count=2
while (($#)); do
  case "$1" in
    --gpus)
      [[ $# -ge 2 && "$2" =~ ^(1|2|4)$ ]] || { echo '--gpus requires 1, 2 or 4' >&2; exit 2; }
      gpu_count="$2"
      shift 2 ;;
    --allocation)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo '--allocation requires a numeric running reserve job ID' >&2; exit 2; }
      allocation_id="$2"
      shift 2 ;;
    --dependency)
      [[ $# -ge 2 ]] || { echo '--dependency requires a Slurm dependency expression' >&2; exit 2; }
      submission_options+=("--dependency=$2")
      shift 2 ;;
    --kill-on-invalid-dep)
      submission_options+=(--kill-on-invalid-dep=yes)
      shift ;;
    --source-archive)
      [[ $# -ge 2 && -f "$2" ]] || { echo '--source-archive requires an existing source archive' >&2; exit 2; }
      export TSENSEI_JOB_SOURCE_ARCHIVE="$(realpath -- "$2")"
      export TSENSEI_JOB_SOURCE_SHA256="$(sha256sum -- "$TSENSEI_JOB_SOURCE_ARCHIVE" | cut -d ' ' -f 1)"
      shift 2 ;;
    *) break ;;
  esac
done
MODE="${1:-probe}"
if (($#)); then shift; fi
case "$MODE" in reserve|probe|benchmark|performance|validate|train|replay|preview|mjlab-preview|arm-replay|export|online-export|strike-diagnostic|preparation-preview) ;; *) echo 'Usage: submit.sh [--gpus 1|2|4] [--allocation JOB_ID] [--dependency EXPR] [--kill-on-invalid-dep] [--source-archive FILE] {reserve|probe|benchmark|performance|validate|train|replay|preview|mjlab-preview|arm-replay|export|online-export|strike-diagnostic} [arguments...]' >&2; exit 2;; esac
if [[ "$gpu_count" == 1 ]]; then
  case "$MODE" in replay|train|validate|export|mjlab-preview|preparation-preview) ;; *) echo '--gpus 1 supports only replay, train, validate, export or mjlab-preview' >&2; exit 2;; esac
  [[ -z "$allocation_id" ]] || { echo '--gpus 1 cannot reuse --allocation: reserve jobs require two GPUs' >&2; exit 2; }
fi
if [[ "${TSENSEI_JOB_BACKEND:-slurm}" == process ]]; then
  [[ -z "$allocation_id" ]] || { echo 'Process backend cannot reuse Slurm allocations' >&2; exit 2; }
  process_options=()
  for option in "${submission_options[@]}"; do
    case "$option" in
      --dependency=*) process_options+=(--dependency "${option#--dependency=}") ;;
      --kill-on-invalid-dep=yes) process_options+=(--kill-on-invalid-dep) ;;
      *) echo "Unsupported process option: $option" >&2; exit 2 ;;
    esac
  done
  exec python3 "$PROJECT_DIR/scripts/hpc/process_backend.py" --project "$PROJECT_DIR" --gpus "$gpu_count" "${process_options[@]}" "$MODE" "$@"
fi
export TSENSEI_EXPECTED_GPUS="$gpu_count"
mkdir -p "$PROJECT_DIR/logs/hpc"
if [[ -n "$allocation_id" ]]; then
  [[ "$MODE" != reserve && ${#submission_options[@]} -eq 0 ]] || { echo '--allocation cannot be combined with reserve, --dependency or --kill-on-invalid-dep' >&2; exit 2; }
  allocation_state="$(squeue -h -j "$allocation_id" -o '%u %T %j')"
  reserve_job_name="TsenseI-reserve${gpu_count}"
  [[ "$allocation_state" == "$(id -un) RUNNING $reserve_job_name" ]] || { echo "Allocation must be your RUNNING $reserve_job_name job; inspect squeue first." >&2; exit 2; }
  allocation_cpus=16
  [[ "$gpu_count" == 4 ]] && allocation_cpus=32
  mkdir -p "$PROJECT_DIR/logs/hpc/$allocation_id/steps"
  export TSENSEI_ALLOCATION_STEP=1
  # Only one child workload at a time. --overlap shares resources with the
  # idle batch keeper, not with a second train/eval launched by this entry.
  exec flock -n "$PROJECT_DIR/logs/hpc/$allocation_id/workload.lock" \
    srun --jobid="$allocation_id" --overlap --exact --nodes=1 --ntasks=1 \
      --cpus-per-task="$allocation_cpus" --gres="gpu:$gpu_count" --chdir="$PROJECT_DIR" \
      --output="$PROJECT_DIR/logs/hpc/$allocation_id/steps/%s.out" \
      --error="$PROJECT_DIR/logs/hpc/$allocation_id/steps/%s.err" \
      bash "$PROJECT_DIR/scripts/hpc/job.sbatch" "$PROJECT_DIR" "$MODE" "$@"
fi
batch_script="$PROJECT_DIR/scripts/hpc/job.sbatch"
if [[ "$MODE" == reserve ]]; then
  [[ $# -eq 0 ]] || { echo 'reserve takes no arguments: requested GPUs are reserved for 48 hours' >&2; exit 2; }
  batch_script="$PROJECT_DIR/scripts/hpc/reserve.sbatch"
  submission_options+=(--time=2-00:00:00 --job-name="TsenseI-reserve${gpu_count}")
  if [[ "$gpu_count" == 4 ]]; then
    # Three collectors and one learner need more host scheduling headroom than
    # the historical two-GPU reserve.  Existing one/two-GPU defaults remain
    # unchanged.
    submission_options+=(--cpus-per-task=32)
  fi
elif [[ "$gpu_count" == 4 ]]; then
  # Keep the existing 16-CPU behavior for one/two-GPU jobs.  Four collectors
  # need independent host threads for environment stepping and IPC.
  submission_options+=(--cpus-per-task=32)
fi
exec sbatch --parsable --chdir="$PROJECT_DIR" --gres="gpu:$gpu_count" \
  "${submission_options[@]}" \
  --output="$PROJECT_DIR/logs/hpc/%j.out" \
  --error="$PROJECT_DIR/logs/hpc/%j.err" \
  "$batch_script" "$PROJECT_DIR" "$MODE" "$@"
