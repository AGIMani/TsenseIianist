# 软件目标限制候选（2026-09-27）

## 2026-09-29：启动参数覆盖目标限速

`run.sh` 支持以下参数，由源码入口转交训练器，解析后写入现有 `control_limits` 和检查点合同：

- `--wrist-tx-max-velocity` / `--wrist-ty-max-velocity`：左右同时设置，单位m/s；`run.sh`显式各1，覆盖原JSON的1.2。
- `--wrist-pitch-max-velocity`：左右同时设置，单位rad/s；当前JSON默认4。
- `--wrist-tz-max-velocity` / `--wrist-yaw-max-velocity`：分别m/s、rad/s；当前入口移除了两轴，不能设置正限速。
- `--finger-max-velocity`：双手40个手指关节，单位rad/s；默认不新增限速。
- `--joint-max-velocity NAME=VALUE`：只设置指定关节，可重复，例如 `R_Wrist_TX=0.6`、`L_Pinky_PIP=8`。

底层CLI不传时沿用JSON；`run.sh`已显式传入TX/TY各1m/s，用户可在命令末尾覆盖。传0移除对应 `max_velocity`，保留该关节的min/max位置约束。优先级为JSON、组参数、单关节参数；单关节重复时最后一个生效。负数、NaN、无穷、未知关节及已移除轴的正限速会报错。全部限速移除且没有位置约束时，合同为None。

集群示例（会提交训练）：

```bash
bash scripts/hpc/submit.sh train \
  --wrist-tx-max-velocity 0.6 --wrist-ty-max-velocity 0.8 \
  --wrist-pitch-max-velocity 2 --finger-max-velocity 8 \
  --joint-max-velocity R_Pinky_PIP=6
```

这些值只是用法示例，未作训练效果或硬件标定验收。限速作用于执行器目标；历史检查点仍须匹配其源码和控制合同。

## 历史候选

完整定义及单位见 `docs/control_limits_20260927.md`，同best对照与最终选用结果见 `docs/best_range_finetuning_20260927.md`。这些文件不自动改变默认训练，须显式传 `--control-limits`；跨合同仅可显式 `--finetune-from ... --allow-control-change`。

2026-09-28整理：下表的envelope、thumb16、wrist_ty06/08、wrist_tighter、wrist_thumbpip12
已移至 `../archive/20260927_control_trials/`，旧内容与路径索引保留。
本目录仅保留当前generic限速、被历史训练引用的wrist_gentle及复测命令引用的combined。

- `envelope_20260927.json`：右TX[-0.2,0.2]m、左TX[-0.1,0.2]m，未改原策略映射或机械限位。
- `wrist_gentle_20260927.json`：上述范围，加TX/TY目标变化率1.2m/s、Pitch4rad/s；本轮优先短训候选。
- `thumb16_20260927.json`：上述TX范围，加左拇指四关节16rad/s。
- `combined_20260927.json`：腕部与拇指两种限制组合。
- `wrist_ty06_20260927.json`／`wrist_ty08_20260927.json`：gentle基础上将TY目标上限分别收至0.06／0.08m。
- `wrist_tighter_20260927.json`：TX/TY降至0.6m/s，Pitch2rad/s。
- `wrist_thumbpip12_20260927.json`：gentle基础上将左拇指PIP目标上限收至1.2rad。

限速是执行器目标的每帧变化率，不保证真实关节速度或无碰撞。全部候选保留物理接触；较紧并不必然更准或更少卡键，勿将测试模板视作已标定的硬件安全参数。

2026-09-27后续：`generic_wrist_rate_20260927.json` 仅保留TX/TY 1.2m/s与Pitch4rad/s，不含min/max位置覆盖；新run.sh显式选此通用候选，替换上述小星星位置窗口。整指几何规则见`../finger_clearance/README.md`，当前选择/实测以`docs/generic_finger_clearance_20260927.md`为准。
