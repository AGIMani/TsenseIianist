#!/usr/bin/env bash
# 统一默认训练入口（更新2026-10-10）：后续训练参数只在本脚本维护。
# 用法：bash all_use_this_train.sh [--dependency afterok:JOB_ID] [--kill-on-invalid-dep] [--source-archive PATH --source-sha256 SHA] [训练参数...]
# 默认S4基础＋全手跨手碰撞；按键paper公式、和弦放大关闭，从头训练。
# 2026-10-06：目标超限惩罚，手指20/400、腕3/15；腕Pitch硬目标限制3.1416/15.708。
# 同手每对每物理子步-.1累加，整项每控制步最多-5；独立误触项关闭（保留未启用weight=-.05）、闲指（当前及未来20步均未用指）目标L1=-.02×Σ|q-q_reset|；按键与指法按原谱物理子步评分，当前目标与未来预告分开；其余项保留原合同。
# 例：bash all_use_this_train.sh --actor-lr 0.0001 --seed 43
# 自动通过 submit.sh 申请双卡 Slurm；cuda:0采样，cuda:1学习，单批流水重叠。
# 保持512环境、batch1024、每512条经验128次更新；不通过减少学习量提速。
# 当前训练源码固定于下方双侧新手模型归档；修改工作区Python后须更新归档及SHA。
# tsensei_collision_v1：每个200Hz物理子步采样左右手任意部位接触；20Hz控制步共10个子步，
# 每个接触子步-.05，整步最多-.5，周期末forward不重复计入。
# 修改算法源码时需另行验证并更新归档及SHA；当前奖励JSON用项目绝对路径传入。
# 自定义本地 MIDI/JSON 建议传绝对路径，避免误用归档内同名文件。
# 2026-10-22：无臂双侧Palm_Base_Pitch为0～0.5rad，归档模型/动作范围一致，range-scale=1。
# 当前奖励与耦合开关详见 docs/reward_defaults_20261006.md。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# 可将训练排在验收成功完成后；加 --kill-on-invalid-dep 在依赖失败时取消。
# 这些Slurm选项放在训练参数之前，不传入训练解析器。
submission_options=()
requested_archive=""
requested_archive_sha=""
while (($#)); do
  case "$1" in
    --dependency)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo '--dependency requires a Slurm dependency expression' >&2; exit 2; }
      submission_options+=(--dependency "$2")
      shift 2 ;;
    --kill-on-invalid-dep)
      submission_options+=(--kill-on-invalid-dep)
      shift ;;
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
      arm_python="$ROOT/training_mjlab/.venv/bin/python"
      [[ -x "$arm_python" ]] || arm_python=python3
      "$arm_python" -m tsensei_v3_2_pianist.mjlab_env.arm_options "$@"
      break ;;
  esac
done
# 保留TCP/腕/手指及闲指20步配置；新增可选双M6，默认无臂。
# --arms须显式给双侧七角；腕目标→连续IK→SDK双硬限臂轨迹，95%软阈值、独立额外最多-5；不做目标剩余距离提前制动。
# 法兰/四孔连接件/手根刚性相连，腕PD移除、保留腕请求与手指物理；验收见logs/arm_follow_repair_20261010/。
# 恢复刚性装配丢失的原生父子过滤，无臂/带臂非相邻自碰撞保留；验收见logs/arm_parent_filter_fix_20261010/。
# 同手奖励保持当前配置；按键连续键程＋原谱物理子步，验收见logs/training_consistency_20261015/。
# 2026-10-14：URDF关节链指尖定位＋八指参考姿态求解启动检查；积分器Euler。
# convex_hull_v1不能构造旧胶囊净空诊断；与exper002实际配置一致，不申请该诊断或其停止门槛。
# 2026-10-18：带臂追加稠密跟随罚，0mm为理想值；默认无臂仍关闭该项。
# 默认仍为无臂；传入 --arms 后切换到 wuji_cartesian_v1（世界系 XYZ 掌位移、固定掌朝向）。臂跟随只做当前步硬限投影，不做目标提前制动。
arm_scene_flags=(--no-arms)
arm_action_flags=(--disable-wrist-tz --disable-wrist-yaw --action-profile direct_joint_v1)
for argument in "$@"; do
  if [[ "$argument" == "--arms" ]]; then
    arm_scene_flags=()
    arm_action_flags=(--disable-wrist-yaw --action-profile wuji_cartesian_v1)
    break
  fi
done
# 2026-10-22恢复exper031腕驱动：Tx/Ty=300/34.15259872981849/±50N；Pitch=8/0.5/±5Nm。
training_archive="$ROOT/logs/black_key_model_20261023/source_black_key_shape_v1.tar.gz"
training_archive_sha=1951eb00657ad1a82a8bc7092f6274dcf4607f26746cc4dfa4d936f89761dd84
if [[ -n "$requested_archive" || -n "$requested_archive_sha" ]]; then
  [[ -n "$requested_archive" && -n "$requested_archive_sha" ]] || { echo 'Explicit source override requires both --source-archive and --source-sha256' >&2; exit 2; }
  training_archive="$requested_archive"
  training_archive_sha="$requested_archive_sha"
fi
if [[ ! -f "$training_archive" ]] || [[ "$(sha256sum -- "$training_archive" | cut -d ' ' -f 1)" != "$training_archive_sha" ]]; then
  echo "Training requires its verified pinned source archive: $training_archive" >&2
  exit 2
fi
# Current key/fingering rewards require original MIDI goals at every physical substep.
selection_options=(--best-f1-metric f1)
for argument in "$@"; do
  if [[ "$argument" == "--no-exact-note-timing" ]]; then
    echo 'Current key/fingering rewards require --exact-note-timing; endpoint scoring was removed.' >&2
    exit 2
  fi
done
exec bash "$ROOT/scripts/hpc/submit.sh" --gpus 2 "${submission_options[@]}" --source-archive "$training_archive" train \
  --height 0.09 --ty 0.04 --yaw 0.2617993877991494 \
  --control-hz 20 --physics-hz 200 --lookahead 10 --frame-stack 1 \
  --midi "$ROOT/training_mjlab/inputs/prelude_bwv847/candidate_fingering/prelude-bwv-847_fingered.proto" --initial-buffer-time 0 \
  --score-validation strict_v1 --release-tail-time 0.25 --exact-note-timing \
  --gravity-compensation --action-reward-observation \
  --task-profile tsensei_collision_v1 --collision-profile convex_hull_v1 --drive-profile shadow_v1 \
  "${arm_scene_flags[@]}" "${arm_action_flags[@]}" \
  --wrist-pitch-range-scale 1.0 --finger-coupling --no-coupling-reduce-observation \
  --finger-coupling-angle-ranges "$ROOT/configs/finger_coupling/angle_ranges_10_80.json" \
  --device cuda:0 --collector-devices cuda:0 --learner-device cuda:1 --collector-pipeline \
  --stage 1 --seed 42 --actor-hidden 256 256 256 --discount 0.80 --init-temperature 0.5 \
  --actor-lr 0.0003 --critic-lr 0.0003 --temp-lr 0.0003 \
  --num-envs 512 --vector-steps 19532 --replay-capacity 2000000 --batch-size 1024 \
  --warmup-transitions 9999 --learning-start-transitions 10000 --updates-per-transition 0.25 \
  --replay-prefill-episodes 1 \
  --eval-at-start --eval-interval 500 --eval-episodes 3 --eval-num-envs 3 \
  --save-best --save-interval 0 --save-forward --log-interval 20 \
  "${selection_options[@]}" \
  --no-physics-graph --compile-learner --fused-optimizer --learner-graph --optimizer-layout \
  --learner-matmul-precision tf32 \
  --no-eval-metrics-graph \
  --hard-joint-limits \
  --control-limits configs/control_limits/generic_wrist_rate_20260927.json \
  --wrist-tx-max-velocity 1.99 --wrist-ty-max-velocity 1.99 \
  --wrist-tx-max-acceleration 1.99 --wrist-ty-max-acceleration 1.99 \
  --wrist-pitch-max-velocity 3.1416 --wrist-pitch-max-acceleration 15.708 \
  --reward-config "$ROOT/configs/rewards/default_20261006.json" \
  --joint-motion-limit-penalty \
  --finger-target-velocity-limit 20 --finger-target-acceleration-limit 400 \
  --wrist-target-velocity-limit 3 --wrist-target-acceleration-limit 15 \
  --tcp-target-velocity-limit 1.90 --tcp-target-acceleration-limit 1.90 \
  --joint-motion-full-penalty-excess 0.1 --joint-motion-max-penalty 5 \
  --anticipatory-fingering-time 0.15 --upcoming-press-permission-time 0 \
  --collision-diagnostics \
  --target-consecutive-evals 19534 --target-max-preparation-wrong-frames 0 \
  --target-max-hand-collision-frames 0 \
  --no-swanlab --no-progress \
  --output-dir "$ROOT/training_mjlab/outputs/hpc/reward_defaults_20261006" "$@"
