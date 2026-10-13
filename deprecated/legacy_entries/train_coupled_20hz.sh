#!/usr/bin/env bash
# 历史双卡底层入口；新的统一默认参数维护于根目录 all_use_this_train.sh。
# 集群通过 bash scripts/hpc/submit.sh train [参数...] 提交，不在SSH直接运行GPU。
# 20/200 Hz、31动作耦合/1407完整观测、512环境、10,000,384样本。
# cuda:0采样，cuda:1学习；申请两卡不代表已经实测加速。
# tsensei_collision_v1：左右手全部实体碰撞几何，周期内任意接触一次-.5，无接触0。
# 同手每去重body对-1；闲指姿态最多-.1/步；其余奖励保留HPC配置。
# 评分连续项子步均值、碰撞周期并集；指法/姿态保持控制周期末。
# reward JSON保留dt/.05单位：当前20Hz为1；改频率须核对最终分值。
# 旧run.sh已归档，当前入口不再依赖它；40hz旧文件名仅兼容转发。
# 公共CLI说明暂存于归档run.sh头部；本历史入口末尾参数优先。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# 兼容入口；今后默认训练请使用项目根目录 all_use_this_train.sh。
if [[ "${1:-}" == --submit-s4 ]]; then
  shift
  exec bash "$ROOT/all_use_this_train.sh" "$@"
fi
cd "$ROOT"
export MUJOCO_GL=egl
exec python3 scripts/train_from_source.py \
  --height 0.10 --ty 0.04 --yaw 0.15 \
  --control-hz 20 --physics-hz 200 --lookahead 10 --frame-stack 1 \
  --midi TwinkleTwinkleRousseau --initial-buffer-time 0 \
  --score-validation strict_v1 --release-tail-time 0.25 --exact-note-timing \
  --preparation-mode policy_v1 --preparation-time 1 --preparation-lift 0.025 \
  --gravity-compensation --action-reward-observation \
  --task-profile tsensei_collision_v1 --collision-profile shadow_primitive_v1 --drive-profile shadow_v1 \
  --no-arms --disable-wrist-tz --disable-wrist-yaw --action-profile direct_joint_v1 \
  --wrist-pitch-range-scale 0.8 \
  --finger-coupling --no-coupling-reduce-observation \
  --finger-coupling-angle-ranges "$ROOT/configs/finger_coupling/angle_ranges_60_80.json" \
  --device cuda:0 --learner-device cuda:1 --collector-pipeline \
  --stage 1 --seed 42 --actor-hidden 256 256 256 --discount 0.75 --init-temperature 0.5 \
  --actor-lr 0.0003 --critic-lr 0.0003 --temp-lr 0.0003 \
  --num-envs 512 --vector-steps 19532 --replay-capacity 2000000 --batch-size 1024 \
  --warmup-transitions 9999 --learning-start-transitions 10000 --updates-per-transition 0.25 \
  --replay-prefill-episodes 1 \
  --eval-at-start --eval-interval 1000 --eval-episodes 3 --eval-num-envs 3 \
  --save-best --save-interval 0 --save-forward --log-interval 20 \
  --physics-graph --compile-learner --fused-optimizer --learner-graph --optimizer-layout \
  --control-limits configs/control_limits/generic_wrist_rate_20260927.json \
  --wrist-tx-max-velocity 1 --wrist-ty-max-velocity 1 \
  --finger-clearance-config configs/finger_clearance/control_disabled_20260927.json \
  --no-idle-finger-protection --no-active-press-permission \
  --reward-config "$ROOT/configs/rewards/coupled_40hz_idle_posture_20261001.json" \
  --anticipatory-fingering-time 0.15 --no-anticipatory-current-priority \
  --no-staged-fingering-reward --upcoming-press-permission-time 0 \
  --collision-diagnostics --target-music-f1 0.99 --target-micro-f1 0.99 \
  --target-consecutive-evals 3 --target-max-preparation-wrong-frames 0 \
  --target-max-hand-collision-frames 0 --target-max-key-underside-substeps 0 \
  --swanlab-project TsenseI_A3.2 --no-swanlab --no-progress \
  --output-dir training_mjlab/outputs/TsenseI_A3.2_anticipatory "$@"
