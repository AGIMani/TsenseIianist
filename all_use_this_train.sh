#!/usr/bin/env bash
# 统一默认训练入口（更新2026-10-10）：后续训练参数只在本脚本维护。
# 2026-10-25：初始位置追加参数统一为世界XYZ（米）；world_x=.16等价原Ty=.02。
# palm_base_*统一替代虚拟forearm_*；Palm_Base_Yaw仍是关节角rad，不是世界欧拉角。
# 用法：bash all_use_this_train.sh [--mem-gb 640] [--dependency afterok:JOB_ID] [--kill-on-invalid-dep] [--source-archive PATH --source-sha256 SHA] [训练参数...]
# 默认S4基础＋全手跨手碰撞；按键paper公式、和弦放大关闭，从头训练。
# 2026-10-06：目标超限惩罚，手指20/400、腕3/15；腕Pitch/Roll硬目标限制3.1416/15.708。
# 同手每对每物理子步-.1累加，整项每控制步最多-5；独立误触项关闭（保留未启用weight=-.05）、闲指（当前及未来20步均未用指）目标L1=-.008×Σw_j|q-q_reset|，八指SW权重.5；按键与指法按原谱物理子步评分，当前目标与未来预告分开；其余项保留原合同。
# 例：bash all_use_this_train.sh --actor-lr 0.0001 --seed 43
# 自动通过 submit.sh 申请双卡 Slurm；cuda:0采样，cuda:1学习，单批流水重叠。
# 2026-10-27：4096环境、batch1024、每4096条经验1024次更新；CPU replay不降低FP32存储精度。
# 当前训练源码固定于下方双侧新手模型归档；修改工作区Python后须更新归档及SHA。
# 2026-10-25：first_note_v1先真实物理悬停再学习；准备不占学习预算、不裁剪实际轨迹。
# 曲尾完整回放先松指再物理回原始手位；ending-time=2为最短回程，另加松指/稳定段，不进SAC经验。
# 2026-10-25：双侧初始Palm_Base_Ty=0.020m（向演奏者后退20mm），缩短首音准备路程；不改Ty行程，未消除起步瞬态。
# 2026-10-25：默认土耳其进行曲原始时值＋A3.2全曲重复/静态成功保护候选（拇指312音）；TZ仍关闭，未启动训练。
# 2026-10-28：用户授权关闭严格谱面采样覆盖拒绝；原谱/精确物理时值保持，漏采保存诊断。
# 2026-10-28：用户指定50Hz控制/500Hz物理、4096环境；联合验收范围/结果见logs/roll_capacity_20261028/roll/validation.json。
# replay-capacity=4800万，容纳50Hz整曲预填并保留约238万条余量；含Roll的FP32 CPU池510.60GiB，Slurm申请640GiB。
# 整曲预填11137步＋学习2244步=13381步；保留2297856次更新。松键尾段.25→.26s以对齐50Hz整数帧。
# 用户指定每1000向量步评估；保留完整3回合/物理诊断，评估之间经验量为原512×500的16倍。
# tsensei_collision_v1：500Hz物理、50Hz控制，每步10子步；每接触子步-.05，整步最多-.5。
# 修改算法源码时需另行验证并更新归档及SHA；当前奖励JSON用项目绝对路径传入。
# 自定义本地 MIDI/JSON 建议传绝对路径，避免误用归档内同名文件。
# 2026-10-28：无臂双侧Palm_Base_Pitch为−0.3～0.6rad，归档模型/动作范围一致，range-scale=1。
# 2026-10-26：按侧接触100%激活琴键时罚实际掌根Y横移；0.01m/s容忍、0.10m/s满罚、独立最多−0.5/控制步；默认JSON开启。
# 当前奖励与耦合开关详见 docs/reward_defaults_20261006.md。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# 可将训练排在验收成功完成后；加 --kill-on-invalid-dep 在依赖失败时取消。
# 这些Slurm选项放在训练参数之前，不传入训练解析器。
submission_options=()
requested_archive=""
requested_archive_sha=""
requested_memory_gb=640
while (($#)); do
  case "$1" in
    --mem-gb)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo '--mem-gb requires a positive integer GiB host-memory request' >&2; exit 2; }
      requested_memory_gb="$2"
      shift 2 ;;
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
# Fail unsupported first-note combinations before requesting GPUs. Later
# command-line options retain their normal last-value-wins semantics.
preparation_python="$ROOT/training_mjlab/.venv/bin/python"
[[ -x "$preparation_python" ]] || preparation_python=python3
"$preparation_python" - "$@" <<'PY_PREFLIGHT'
import argparse,sys,math
from tsensei_v3_2_pianist.initial_world_pose import add_arguments, reject_legacy_pose_arguments, validate, DEFAULT_WORLD_POSE
p=argparse.ArgumentParser(add_help=False)
add_arguments(p, dict(world_x=.16, world_z=.09, palm_base_yaw=.2617993877991494))
reject_legacy_pose_arguments(sys.argv[1:], p)
p.add_argument('--preparation-mode',default='first_note_v1')
p.add_argument('--preparation-time',type=float,default=2.)
p.add_argument('--ending-time',type=float,default=2.)
p.add_argument('--arms',action=argparse.BooleanOptionalAction,default=False)
p.add_argument('--disable-wrist-tz',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--disable-wrist-yaw',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--finger-coupling',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--action-profile',default='direct_joint_v1')
a,_=p.parse_known_args(sys.argv[1:])
try:
    validate({key: getattr(a, key) for key in DEFAULT_WORLD_POSE})
except ValueError as error:
    p.error(str(error))
if not math.isfinite(a.ending_time) or a.ending_time<0 or (a.ending_time and (a.preparation_mode!='first_note_v1' or a.preparation_time<=0)):
    p.error('Ending requires first_note_v1 preparation; explicitly use --ending-time 0 to disable')
if a.preparation_mode=='first_note_v1' and a.preparation_time>0:
    if a.arms or not a.disable_wrist_tz or not a.disable_wrist_yaw or not a.finger_coupling or a.action_profile!='direct_joint_v1':
        p.error('first_note_v1 requires hand-only direct_joint_v1, fixed TZ/Yaw and coupling; explicitly choose --preparation-time 0 for unsupported combinations')
PY_PREFLIGHT
# 2026-10-28：双侧Ty范围[-0.05,0.05]m、Tz范围[-0.6,0.46]m；默认TZ仍结构关闭。
# 2026-10-28：Pitch/Roll均为Kp=8、阻尼0.5、力矩限±18Nm；Roll右局部+Z/左−Z、[-0.3,0.5]rad。
training_archive="$ROOT/logs/palm_ranges_20261028/source_palm_ranges_v1.tar.gz"
training_archive_sha=5573700d6cfdd855ecaaef7173c08246b88138bc06c1d31902b316c6f638cfd6
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
exec bash "$ROOT/scripts/hpc/submit.sh" --gpus 2 --mem-gb "$requested_memory_gb" "${submission_options[@]}" --source-archive "$training_archive" train \
  --world-x 0.16 --right-world-y 0.15 --left-world-y -0.15 --world-z 0.09 \
  --palm-base-yaw 0.2617993877991494 \
  --control-hz 50 --physics-hz 500 --lookahead 10 --frame-stack 1 \
  --midi "$ROOT/training_mjlab/inputs/turkish_march_advanced/a32_repeat_static_guarded_20261025/turkish-march-advanced_fingered.proto" --initial-buffer-time 0 \
  --score-validation legacy --release-tail-time 0.26 --exact-note-timing \
  --preparation-mode first_note_v1 --preparation-time 2 --preparation-lift 0 --ending-time 2 \
  --gravity-compensation --action-reward-observation \
  --task-profile tsensei_collision_v1 --collision-profile convex_hull_v1 --drive-profile shadow_v1 \
  "${arm_scene_flags[@]}" "${arm_action_flags[@]}" \
  --wrist-pitch-range-scale 1.0 --finger-coupling --no-coupling-reduce-observation \
  --finger-coupling-angle-ranges "$ROOT/configs/finger_coupling/angle_ranges_10_80.json" \
  --device cuda:0 --collector-devices cuda:0 --learner-device cuda:1 --collector-pipeline \
  --stage 1 --seed 42 --actor-hidden 256 256 256 --discount 0.80 --init-temperature 0.5 \
  --actor-lr 0.0003 --critic-lr 0.0003 --temp-lr 0.0003 \
  --num-envs 4096 --vector-steps 13381 --replay-capacity 48000000 --replay-device cpu --batch-size 1024 \
  --warmup-transitions 9999 --learning-start-transitions 10000 --updates-per-transition 0.25 \
  --replay-prefill-episodes 1 \
  --eval-at-start --eval-interval 1000 --eval-episodes 3 --eval-num-envs 3 \
  --save-best --save-interval 0 --save-forward --log-interval 20 \
  "${selection_options[@]}" \
  --no-physics-graph --compile-learner --fused-optimizer --learner-graph --optimizer-layout \
  --learner-matmul-precision tf32 \
  --no-eval-metrics-graph \
  --hard-joint-limits \
  --control-limits configs/control_limits/generic_wrist_rate_20260927.json \
  --wrist-tx-max-velocity 1.99 --wrist-ty-max-velocity 1.99 \
  --wrist-tx-max-acceleration 5.0 --wrist-ty-max-acceleration 5.0 \
  --wrist-pitch-max-velocity 3.1416 --wrist-pitch-max-acceleration 15.708 \
  --wrist-roll-max-velocity 3.1416 --wrist-roll-max-acceleration 15.708 \
  --reward-config "$ROOT/configs/rewards/default_20261006.json" \
  --joint-motion-limit-penalty \
  --finger-target-velocity-limit 20 --finger-target-acceleration-limit 400 \
  --wrist-target-velocity-limit 3 --wrist-target-acceleration-limit 15 \
  --tcp-target-velocity-limit 1.90 --tcp-target-acceleration-limit 4.75 \
  --joint-motion-full-penalty-excess 0.1 --joint-motion-max-penalty 5 \
  --anticipatory-fingering-time 0.15 --upcoming-press-permission-time 0 \
  --collision-diagnostics \
  --target-consecutive-evals 19534 --target-max-preparation-wrong-frames 0 \
  --target-max-hand-collision-frames 0 \
  --no-swanlab --no-progress \
  --output-dir "$ROOT/training_mjlab/outputs/hpc/reward_defaults_20261006" "$@"
