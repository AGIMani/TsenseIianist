# 2026-09-26：曲目最佳权重、达标停止与显式微调

## 问题和本次范围

旧训练只按固定间隔保存权重。评估峰值如果出现在两个保存点之间，日志虽然有分数，对应的策略却无法回放。带 1 秒空谱准备后，全回合平均 F1 也不能直接代表曲目质量；多回合取平均还可能掩盖一次明显失败。

本次在 `rl_torch/train.py`、`checkpoint.py` 新增最佳权重、可选达标停止及同任务网络微调。奖励和碰撞诊断实现由独立模块提供，本工程不改变它们的公式。用户本轮另行授权短测后使用 batch 1024、0.25 update/transition；它是显式学习配置变化，不再称作与 batch 256、1 update/transition 完全相同的纯执行优化。

## 最佳权重

- `--save-best` 默认开启，`--no-save-best` 关闭；只有完整、无截断的评估才参与选优。
- 主指标是 `music_only_f1_min`：每个评估回合先剔除准备和额外空谱前缀，再计算原逐帧 F1；以这批回合的**最小值**排序。`music_only_f1` 平均值和 micro F1 保留用于分析，不让一次好回合掩盖差回合。
- 每次严格改善立即通过临时文件、fsync、os.replace 原子写 `best.pt`，与定期 `step-N.pt` 的保存间隔无关。同分不覆盖较早权重。
- `best.pt` 的 `metadata.evaluation_selection_state.best` 保存确切 `vector_step`、`updates`、主指标和完整评估；`best.json` 是方便阅读的旁路索引，权重与内嵌 metadata 才是权威记录。
- 定期和最终检查点含 `last_evaluation_vector_step`，避免把较早评估结果误认为当前权重的实测分数；`summary.json` 含 `best_checkpoint`、最终评估和完整选优状态。
- 回放直接指定 `best.pt`。历史 `preview/scripts/export_best_mjlab.py` 的旧按日志匹配 step 文件策略不会自动识别新 best 文件，不能依赖其忽略 best 后的结果代表新方案的峰值。

## 达标停止：默认关闭

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--target-music-f1` | 未设置 | 设置后才启用目标停止 |
| `--target-micro-f1` | 同 music 目标 | 每回合曲目 micro F1 下限 |
| `--target-consecutive-evals` | 3 | 连续合格评估次数，必须至少 2 |
| `--target-max-preparation-wrong-frames` | 0 | 任一回合准备期错键控制帧上限 |
| `--target-max-hand-collision-frames` | 0 | 任一回合任意手—手接触控制帧上限 |
| `--collision-diagnostics` | 关闭 | 启用物理子步碰撞观测；目标停止必须开启 |
| `--eval-at-start` | 关闭 | 训练前复评已加载网络并保存初始 best，不计连续达标次数 |

例如显式目标 `--target-music-f1 .99 --target-micro-f1 .99` 表示每批评估的最差曲目逐帧 F1 和最差曲目 micro F1 均须 ≥0.99，准备错帧及碰撞数也必须同时满足上限。任一门槛失败，连续次数清零。

只有 `completed_episodes == evaluation_episodes`、无截断、完整有效的碰撞统计才能通过。缺失字段不会当成 0 碰撞：报告 `None`、覆盖率不足，停止门槛失败。碰撞还要求实际统计的物理子步覆盖率为 100%，并记录控制/物理采样率。输出 `checkpoint_selection.target_failed_checks` 可定位未达标原因。

目标达到后停止继续采样和学习，仍保存 `final.pt`，正常结束；`summary.status` 仍为 `completed`，`stop_reason=quality_target_reached` 区分预算用尽，另有 `target_reached=true`。预算用尽则不假称达标。F1≥0.99 是经验验收目标，不能保证任意训练或复评都会达到。

### 指标口径

评估保存 `episode_statistics` 全部单回合结果；顶层保留原全回合 `f1`、`key_frame_micro_f1`。新增曲目平均/min、准备错帧 max、self/inter-hand/union 的接触帧、body-pair 帧、物理子步数，以及覆盖率和采样率。

当前准备错键/F1 仍在 20 Hz 控制端点观测。碰撞诊断在 200 Hz 子步累计；二者采样口径不同。没有因此改变训练 reward，也没有解决 V32-START-02 中短于控制周期的错音盲区。碰撞诊断观察手—手接触，不把正常手指触琴自动当成手碰撞惩罚。

## 显式微调与旧检查点

原 `--stage2-from` 表示 Stage1→Stage2，必须是 Stage1 父检查点且目标 `--stage 2`；它不是任意模型通用迁移开关。新 `--finetune-from PATH` 支持任一 stage 的同任务网络初始化，与 warm-start、stage2-from 互斥。

1. 保持 scene/task SHA256、准备配置和完整准备合同、action profile、TZ/Yaw 结构与手碰撞物理合同严格匹配。1180→1181、动作映射或物理模型变更没有自动迁移捷径。
2. 先检查 actor、critic、target critic 全部权重名称和形状，再复制任何网络；形状错误不会出现只加载了 actor 的半完成状态。
3. 更换奖励必须显式加 `--allow-reward-change`，而且必须 networks-only；普通 warm-start 仍拒绝奖励变化。
4. 新 learner 的优化器、replay、随机数流和 update 计数均从新任务配置开始。温度也使用当前显式 `--init-temperature`，不会读取父 Adam/alpha 状态。可以明确传父策略学得的温度作为**新初始值**，避免无意改回 1；这仍是新优化器状态。
5. `config.json`/checkpoint 的 `origin` 保存父路径、文件 SHA256、原完整 metadata、转移方式和 fresh-state 说明；原文件不改写，不能把旧权重伪装成新奖励合同下训练出来的模型。

本轮父 5M 已是 policy preparation、1181 观测、47 动作，无需补输入维度。用户另授权新奖励时，last_reward 观测内容也可能改变，继承策略的输入分布会随之改变；`--eval-at-start` 在新奖励环境里先记录真实起点评估，不能直接沿用父分数。

微调新 replay 可显式设置 `--warmup-transitions 0 --learning-start-transitions 91137`：512 世界先采完整 178 控制步共 91,136 条继承策略数据，再开始更新。这个具体值是本曲配置，其他乐谱/缓冲长度不能机械沿用。5,000,192 样本、batch 1024、0.25 更新比下，从第 91,137 条开始实际更新预算为 **1,227,264**，稳态每条 transition 对应 256 个 sampled batch rows；它与旧 1 次更新 × batch 256 的更新节奏和训练结果仍可能不同。

## 验证和后续

CPU 实际命令：

```bash
CUDA_VISIBLE_DEVICES='' training_mjlab/.venv/bin/python -m pytest \
  tests/test_training_quality_control.py tests/test_preparation_training_entry.py \
  tests/test_preparation_checkpoint.py tests/test_shadow_training_budget.py \
  tests/test_torch_sac.py -q
```

结果：**26 passed、39 subtests passed**。覆盖多回合最差指标、前缀剔除、连续门槛重置、初始评估不计次数、缺失诊断 fail-closed、部分物理覆盖、完整回合要求、严格 scene/task/奖励/形状检查、fresh optimizer/温度/RNG，以及原预算回归。初轮未屏蔽 CUDA 的旧回归含自动枚举 CUDA 的小 replay 测试，发现后全部回归显式屏蔽 CUDA，未用该过程作性能结果。

父进程负责真实 GPU A/B/C/D 短测、源合同和回放验收，再按用户授权选择新训练参数；本文件单独的 CPU 测试不证明收敛、达标或实际 GPU 保存/停止流程已完整验收。V32-QUALITY-01：完整训练后用多回合复评验证曲目≥0.99、准备误触和碰撞目标，保留用户规定的帧率盲区说明。

参考：`rl_torch/train.py`、`checkpoint.py`、`sac.py`、`mjlab_env/task.py`、`tests/test_preparation_training_entry.py`、`tests/test_torch_sac.py`；论文决定见 `docs/references/biomimetics_2026_610_notes.md`。根及独立项目变更日志由本轮主记录统一合并。
