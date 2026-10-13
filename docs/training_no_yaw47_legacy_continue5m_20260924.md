# 无重力补偿小星星 47 动作再续训 5M（2026-09-24）

用户选择 47 动作无重力补偿分支，以便与当前重力补偿 47 动作实验对照。本轮从已完成的 `sac-twinkle-no-yaw47-20260924-120155/final.pt` **热启动**，再采样 19,532 个向量步 × 256 环境＝5,000,192 条 transition。没有修改模型、乐谱、奖励或 SAC 参数。

## 冻结合同与续训语义

- scene/task：`training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/{scene,task}`，`gravity_compensation=false`。散列分别为 `27e349db570405f1f0a78fa135c06a18c0eb7234a6bafdb934de8341a71e1f7c`／`afb79ed600b2d22218017ea527c458056eaeaaf1556669f04bf37cd04b86444a`，与源 `final.pt` 完全匹配。当前 Torch 训练默认值已换成补偿场景，本轮在命令里显式传旧包，避免串场。
- 配置：无 M6 机械臂、手根 12 cm、20/200 Hz、TZ/Yaw 物理移除，47 动作、1180 维观测，`mixed_v1`、同手碰撞权重 0.2、三类实体接触，Stage1 SAC；新采样的 `warmup_vector_steps=0`，从第一步使用恢复策略；`learning_starts=1000`，新 replay 填充后才恢复更新。策略、Critic、目标网络、优化器、温度与探索 RNG 从旧 `final.pt` 恢复。仿真状态和 GPU replay 重新开始，**不是**逐步精确续跑；`updates` 计数从旧 593,056 继续。
- 源旧训练已经完成 5,000,192 条采样；本轮目标是**额外** 5,000,192 条。旧 49 动作分支未在本轮继续。

## 运行记录

2026-09-24 14:47:38（Asia/Shanghai）后台启动，初始 PID `336820`。运行目录：

`training_mjlab/outputs/sac-twinkle-no-yaw47-legacy-continue5m-20260924-144738/`

精确命令、工作目录、环境和源检查点保存在同级 `.launch.json`；进度见同级 `.log` 和运行目录 `metrics.jsonl`，最终结果以 `summary.json` 为准。核心启动命令：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
MUJOCO_GL=egl training_mjlab/.venv/bin/python -u -m tsensei_v3_2_pianist.rl_torch.train \
  --scene training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/scene \
  --task training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/task \
  --no-arms --disable-wrist-tz --disable-wrist-yaw --action-profile mixed_v1 \
  --self-collision-weight 0.2 \
  --warm-start training_mjlab/outputs/sac-twinkle-no-yaw47-20260924-120155/final.pt \
  --warmup-vector-steps 0 --learning-starts 1000 \
  --num-envs 256 --vector-steps 19532 --seed 42 \
  --physics-graph --compile-learner --fused-optimizer --progress \
  --output-dir training_mjlab/outputs/sac-twinkle-no-yaw47-legacy-continue5m-20260924-144738
```

启动验收：源检查点 schema、scene/task 散列、47 动作、1180 观测、TZ/Yaw 锁止、奖励权重和 Stage1 参数与旧训练 `config.json` 一致。新 `config.json` 记录源 `final.pt` 的 vector_step=19532／updates=593056，scene/task 仍为旧散列；GPU 已开始采样。核对时已达到向量步 1020／261,120 条采样，累计 updates=593,728；首轮完整评估 F1=0.762553，`step-1000.pt` 已保存。这只是早期观测值，不代表最终效果。与之并行的重力补偿 47 动作训练使用独立的 `contact_stable_h12_20hz200hz_gravcomp_20260924` scene/task，不共享 replay 或检查点。两轮共享同一张 GPU，吞吐可能下降，不据启动阶段 ETA 推断完成时间。

## 待验收

- **V32-LEGACY-47-CONT-01**：本轮完成后检查 `summary.json`、最终及最佳检查点、独立复播和与重力补偿实验的评价差异。当前只确认启动与采样，不能声称续训效果或 5M 已完成。
- **V32-SAC-03**：既有训练评价与独立复播差异仍需跟踪；旧 replay/物理状态无法由现有热启动机制恢复。
