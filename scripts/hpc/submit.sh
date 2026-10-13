#!/usr/bin/env bash
# A3.2 Slurm 基础设施入口；普通 SSH 只提交作业，GPU 工作在分配内执行。
# 新训练统一使用根目录 all_use_this_train.sh，由其维护当前参数并固定源码归档/SHA。
# 本脚本的 train 模式是底层兼容调用；直接调用不会自动采用统一入口的参数或固定归档。
# job.sbatch 在所选归档中调用历史 train_coupled 入口，再应用传入的末尾覆盖参数。
# 奖励以统一入口＋本轮末尾参数＋最终 config.json 为准，启动时打印完整奖励配置。
# 可选机械臂只由显式 --arms 及双侧各7个初角启用；使用统一入口可在申请GPU前检查。
# 申请两卡不意味着学习器梯度跨卡并行，也不等于已经实测加速。
# --gpus 1：支持新分配中的train/validate/reward-diagnostic/replay/export/mjlab-preview，不与--allocation合用。
#   单卡train在用户参数后强制cuda:0采样/学习及--no-collector-pipeline，其余训练预算不变。
#   省略--gpus仍申请两卡；validate所选测试须适用于实际获配卡数。
# 示例：
#   bash scripts/hpc/submit.sh reserve
#   bash all_use_this_train.sh
#   bash all_use_this_train.sh --dependency afterok:123 --seed 43
#   bash scripts/hpc/submit.sh --gpus 1 validate tests/test_shadow_reward_only.py
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
      [[ $# -ge 2 && "$2" =~ ^[12]$ ]] || { echo '--gpus requires 1 or 2' >&2; exit 2; }
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
case "$MODE" in reserve|probe|benchmark|performance|validate|train|replay|preview|mjlab-preview|arm-replay|export|online-export|strike-diagnostic|reward-diagnostic) ;; *) echo 'Usage: submit.sh [--gpus 1|2] [--allocation JOB_ID] [--dependency EXPR] [--source-archive FILE] {reserve|probe|benchmark|performance|validate|train|replay|preview|mjlab-preview|arm-replay|export|online-export|strike-diagnostic|reward-diagnostic} [arguments...]' >&2; exit 2;; esac
if [[ "$gpu_count" == 1 ]]; then
  case "$MODE" in replay|train|validate|reward-diagnostic|export|mjlab-preview) ;; *) echo '--gpus 1 supports only replay, train, validate, reward-diagnostic, export or mjlab-preview' >&2; exit 2;; esac
  [[ -z "$allocation_id" ]] || { echo '--gpus 1 cannot reuse --allocation: reserve jobs require two GPUs' >&2; exit 2; }
fi
export TSENSEI_EXPECTED_GPUS="$gpu_count"
mkdir -p "$PROJECT_DIR/logs/hpc"
if [[ -n "$allocation_id" ]]; then
  [[ "$MODE" != reserve && ${#submission_options[@]} -eq 0 ]] || { echo '--allocation cannot be combined with reserve or --dependency' >&2; exit 2; }
  allocation_state="$(squeue -h -j "$allocation_id" -o '%u %T %j')"
  [[ "$allocation_state" == "$(id -un) RUNNING TsenseI-reserve2" ]] || { echo 'Allocation must be your RUNNING TsenseI-reserve2 job; inspect squeue first.' >&2; exit 2; }
  mkdir -p "$PROJECT_DIR/logs/hpc/$allocation_id/steps"
  export TSENSEI_ALLOCATION_STEP=1
  # Only one child workload at a time. --overlap shares resources with the
  # idle batch keeper, not with a second train/eval launched by this entry.
  exec flock -n "$PROJECT_DIR/logs/hpc/$allocation_id/workload.lock" \
    srun --jobid="$allocation_id" --overlap --exact --nodes=1 --ntasks=1 \
      --cpus-per-task=16 --gres=gpu:2 --chdir="$PROJECT_DIR" \
      --output="$PROJECT_DIR/logs/hpc/$allocation_id/steps/%s.out" \
      --error="$PROJECT_DIR/logs/hpc/$allocation_id/steps/%s.err" \
      bash "$PROJECT_DIR/scripts/hpc/job.sbatch" "$PROJECT_DIR" "$MODE" "$@"
fi
batch_script="$PROJECT_DIR/scripts/hpc/job.sbatch"
if [[ "$MODE" == reserve ]]; then
  [[ $# -eq 0 ]] || { echo 'reserve takes no arguments: two GPUs for 48 hours after allocation' >&2; exit 2; }
  batch_script="$PROJECT_DIR/scripts/hpc/reserve.sbatch"
  submission_options+=(--time=2-00:00:00 --job-name=TsenseI-reserve2)
fi
exec sbatch --parsable --chdir="$PROJECT_DIR" --gres="gpu:$gpu_count" \
  "${submission_options[@]}" \
  --output="$PROJECT_DIR/logs/hpc/%j.out" \
  --error="$PROJECT_DIR/logs/hpc/%j.err" \
  "$batch_script" "$PROJECT_DIR" "$MODE" "$@"
