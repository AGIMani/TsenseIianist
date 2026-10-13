# V3.2 通用净空配置：从头新训（2026-09-27）

用户要求“按照当前的配置新开一个训练”。北京时间13:25:20，通过当前项目的 `run.sh` 默认参数启动一轮**从头训练**。独立输出目录为 `training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927-20260927-132520-465250`，后台PID 2195168；本轮未加载best、final或其他检查点。旧训练目录、权重及源码均未改。

## 本轮实际合同

- 源码+完整 `config.json`；小星星原指法、20/200 Hz、1秒可训练准备、无机械臂、TZ/Yaw结构移除、47动作/1181观测，Shadow模型/任务/碰撞/驱动profile。
- 腕部通用限速 `configs/control_limits/generic_wrist_rate_20260927.json`，整指净空 `configs/finger_clearance/generic_20260927.json`，有界辅助奖励 `configs/rewards/generic_finger_clearance_20260927.json`；启用200 Hz碰撞诊断和键底0门槛。
- SAC：种子42、Actor学习率3e-5、Critic/温度3e-4、初始温度1、折扣0.8、512环境×9766向量步=5,000,192条新样本、batch1024、每新样本0.25次更新；前4999条随机，第5000条开始学习。每100向量步评估3回合、合格best保存。
- 与上一轮同名配置续训相比：源码配方、模型、奖励、净空、腕限速、评估选择、乐谱及姿态合同逐项相同；仅本轮**没有warm-start**，恢复默认初始温度1及前4999条预热/第5000条学习起点，另有新输出目录。上一轮是从短训best恢复学习器，所以两个运行不能当作只差随机种子的直接对照。

启动命令：

```bash
bash run.sh --output-dir training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927 --no-progress
```

入口会在基础目录后追加北京时间戳。`--no-progress`仅关闭终端进度条，不改采样、评估或训练配置。运行状态以[新目录的指标](../training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927-20260927-132520-465250/metrics.jsonl)及最终 `summary.json` 为准；[完整配置](../training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927-20260927-132520-465250/config.json)、[启动日志](../training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927.launch.log)、[启动快照](../training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927.launch.json)。

## 启动验证与未完成项

2026-09-27T13:27:11+08:00 快照：进程仍在运行，step 230、117760 条采样、28190 次更新；`best.pt/best.json` 已生成，当前best为step 0、音乐F1 0.158228，属于随机初始化初评，不代表已学会弹奏。`config.json` 的 `warm_start/finetune_from/stage2_from` 均为空；CLI种子42。与前一轮config比较，只有预热、学习起点、初始温度、输出目录、warm-start字段有差异，包内source/runtime及奖励/控制合同相同。后续须以长期评估与完整回放判断效果。

参考 `run.sh`、新旧两轮 `config.json`、`best.json`、`metrics.jsonl`，以及 `docs/references/biomimetics_2026_610_notes.md` 的既定决策。没有新论文参数选择或模型源码改动。

**TODO**：V32-FRESH-01 等本轮完成后验收最佳和最终音乐/micro-F1、键底/侧接触、自碰撞与视频；V32-QUALITY-01、V32-START-02的既有质量和短音盲区仍保留。启动成功不是收敛保证。

## 完成后的 best 导出与清理

本轮训练已完成。best 视频、F1 曲线及数据、清理范围与历史冻结场景限制见[导出与清理报告](generic_clearance_fresh_video_cleanup_20260927.md)。
