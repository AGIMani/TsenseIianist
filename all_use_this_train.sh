#!/usr/bin/env bash
# 统一默认训练入口（更新2026-10-10）：后续训练参数只在本脚本维护。
# 用法：bash all_use_this_train.sh [--dependency afterok:JOB_ID] [--source-archive PATH --source-sha256 SHA] [训练参数...]
# 默认S4基础＋全手跨手碰撞；按键paper公式、和弦放大关闭，从头训练。
# 2026-10-06：目标超限惩罚，手指20/400、腕3/15；腕Pitch硬目标限制3.1416/15.708。
# 同手每对每物理子步-.1累加，整项每控制步最多-5；独立误触项关闭（保留未启用weight=-.05）、闲指（当前及未来20步均未用指）目标L1=-.05×Σ|q-q_reset|；按键按原谱物理子步评分，其余公共项按Shadow控制帧目标＋周期末实际状态评分。
# 例：bash all_use_this_train.sh --actor-lr 0.0001 --seed 43
# 自动通过 submit.sh 申请双卡 Slurm；cuda:0采样，cuda:1学习，启用流水。
# 新全手碰撞源码固定于下方归档；修改工作区Python后须更新归档及SHA。
# tsensei_collision_v1：控制周期末左右手任意部位接触一次-.5，无接触0。
# 修改算法源码时需另行验证并更新归档及SHA；当前奖励JSON用项目绝对路径传入。
# 自定义本地 MIDI/JSON 建议传绝对路径，避免误用归档内同名文件。
# 当前奖励与耦合开关详见 docs/reward_defaults_20261006.md。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# 可将续训排在父基线成功完成后；这是Slurm选项，不传入训练解析器。
submission_options=()
requested_archive=""
requested_archive_sha=""
while (($#)); do
  case "$1" in
    --dependency)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo '--dependency requires a Slurm dependency expression' >&2; exit 2; }
      submission_options+=(--dependency "$2")
      shift 2 ;;
    --source-archive)
      [[ $# -ge 2 && -f "$2" ]] || { echo '--source-archive requires an existing archive' >&2; exit 2; }
      requested_archive="$(realpath -- "$2")"
      shift 2 ;;
    --source-sha256)
      [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { echo '--source-sha256 requires 64 lowercase hex characters' >&2; exit 2; }
      requested_archive_sha="$2"
      shift 2 ;;
    *) break ;;
  esac
done
# Explicit arm inputs are checked before reserving GPUs; absent angles never
# fall back to the preview's historical seeds or an all-zero posture.
for argument in "$@"; do
  case "$argument" in
    --arms|--arm-initial-*-deg|--arm-mount|--arm-mount=*)
      python3 -m tsensei_v3_2_pianist.mjlab_env.arm_options "$@"
      break ;;
  esac
done
# 保留TCP/腕/手指及闲指20步配置；新增可选双M6，默认无臂。
# --arms须显式给双侧七角；腕目标→连续IK/两层制动→SDK双硬限臂轨迹，95%软阈值、独立额外最多-5。
# 法兰/四孔连接件/手根刚性相连，腕PD移除、保留腕请求与手指物理；验收见logs/arm_follow_repair_20261010/。
# 恢复刚性装配丢失的原生父子过滤，无臂/带臂非相邻自碰撞保留；验收见logs/arm_parent_filter_fix_20261010/。
# 同手奖励保持当前配置；按键恢复原谱子步/50%激活，验收见logs/key_substep_restore_20261013/。
# convex_hull_v1不能构造旧胶囊净空诊断；与exper002实际配置一致，不申请该诊断或其停止门槛。
training_archive="$ROOT/logs/mount_alignment_20261013/source_mount_aligned.tar.gz"
training_archive_sha=90a0a1a71a57d3ba9f9c78c7ea61564e42600f96c4eb67b05695d0c9b99da5bd
if [[ -n "$requested_archive" || -n "$requested_archive_sha" ]]; then
  [[ -n "$requested_archive" && -n "$requested_archive_sha" ]] || { echo 'Explicit source override requires both --source-archive and --source-sha256' >&2; exit 2; }
  training_archive="$requested_archive"
  training_archive_sha="$requested_archive_sha"
fi
if [[ ! -f "$training_archive" ]] || [[ "$(sha256sum -- "$training_archive" | cut -d ' ' -f 1)" != "$training_archive_sha" ]]; then
  echo "Training requires its verified pinned source archive: $training_archive" >&2
  exit 2
fi
# Current key reward requires original MIDI goals at every physical substep.
selection_options=(--best-f1-metric exact_original_music_micro_f1_min
  --target-exact-micro-f1 0.99)
for argument in "$@"; do
  if [[ "$argument" == "--no-exact-note-timing" ]]; then
    echo 'Current key reward requires --exact-note-timing; endpoint scoring was removed.' >&2
    exit 2
  fi
done
exec bash "$ROOT/scripts/hpc/submit.sh" --gpus 2 "${submission_options[@]}" --source-archive "$training_archive" train \
  --height 0.10 --ty 0.04 --yaw 0.15 \
  --control-hz 20 --physics-hz 200 --lookahead 10 --frame-stack 1 \
  --midi TwinkleTwinkleRousseau --initial-buffer-time 0 \
  --score-validation strict_v1 --release-tail-time 0.25 --exact-note-timing \
  --preparation-mode policy_v1 --preparation-time 1 --preparation-lift 0.025 \
  --gravity-compensation --action-reward-observation \
  --task-profile tsensei_collision_v1 --collision-profile convex_hull_v1 --drive-profile shadow_v1 \
  --no-arms --disable-wrist-tz --disable-wrist-yaw --action-profile direct_joint_v1 \
  --wrist-pitch-range-scale 0.8 --finger-coupling --no-coupling-reduce-observation \
  --finger-coupling-angle-ranges configs/finger_coupling/angle_ranges_60_80.json \
  --device cuda:0 --learner-device cuda:1 --collector-pipeline \
  --stage 1 --seed 42 --actor-hidden 256 256 256 --discount 0.80 --init-temperature 0.5 \
  --actor-lr 0.0003 --critic-lr 0.0003 --temp-lr 0.0003 \
  --num-envs 512 --vector-steps 19532 --replay-capacity 2000000 --batch-size 1024 \
  --warmup-transitions 9999 --learning-start-transitions 10000 --updates-per-transition 0.25 \
  --replay-prefill-episodes 1 \
  --eval-at-start --eval-interval 500 --eval-episodes 3 --eval-num-envs 3 \
  --save-best --save-interval 0 --save-forward --log-interval 20 \
  "${selection_options[@]}" \
  --physics-graph --compile-learner --fused-optimizer --learner-graph --optimizer-layout \
  --no-eval-metrics-graph \
  --hard-joint-limits \
  --control-limits configs/control_limits/generic_wrist_rate_20260927.json \
  --wrist-tx-max-velocity 0.99 --wrist-ty-max-velocity 0.99 \
  --wrist-tx-max-acceleration 1.99 --wrist-ty-max-acceleration 1.99 \
  --wrist-pitch-max-velocity 3.1416 --wrist-pitch-max-acceleration 15.708 \
  --reward-config "$ROOT/configs/rewards/default_20261006.json" \
  --joint-motion-limit-penalty \
  --finger-target-velocity-limit 20 --finger-target-acceleration-limit 400 \
  --wrist-target-velocity-limit 3 --wrist-target-acceleration-limit 15 \
  --tcp-target-velocity-limit 0.9 --tcp-target-acceleration-limit 1.90 \
  --joint-motion-full-penalty-excess 0.1 --joint-motion-max-penalty 5 \
  --anticipatory-fingering-time 0.15 --no-anticipatory-fingering-reward \
  --no-anticipatory-current-priority --no-staged-fingering-reward --upcoming-press-permission-time 0 \
  --collision-diagnostics --target-music-f1 0.99 --target-micro-f1 0.99 \
  --target-consecutive-evals 19534 --target-max-preparation-wrong-frames 0 \
  --target-max-hand-collision-frames 0 \
  --no-swanlab --no-progress \
  --output-dir "$ROOT/training_mjlab/outputs/hpc/reward_defaults_20261006" "$@"
