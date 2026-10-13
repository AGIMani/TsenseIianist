# 2026-09-26：微调短测汇总与初期退化核查

## 可重现汇总

`scripts/summarize_quality_trials.py` 只读取输出目录，不导入 Torch、不使用 GPU、不修改训练权重/配置。它保留 A/B/C/D 占位，并自动发现 E 及后续 A–Z 字母前缀组，支持尚未结束的 run，输出 JSON、逐次评估 CSV、每组宽表 CSV、PNG 和 SVG 图。

```bash
CUDA_VISIBLE_DEVICES='' training_mjlab/.venv/bin/python scripts/summarize_quality_trials.py
```

默认输入 `training_mjlab/outputs/quality_trials_20260926`，默认输出 [preview/outputs/diagnostics/quality_trials_20260926](../preview/outputs/diagnostics/quality_trials_20260926)。可用 `--root`、`--output` 改路径，`--variants A B E` 指定组，`--no-plot` 仅导出数据。遇多个同字母 run，选目录名排序最后一项，其他路径写入 JSON；不会把同目录下的 `A.log` 当成 run。运行中半写的最后 JSON 行会注明并跳过，完整坏行会报错。

“稳态采样速度”由训练日志反推：时间点 `t = transitions / loop_transitions_per_second`，相邻日志用 `Δtransitions / Δt`；只保留两个端点均已产生**本轮新增更新**的区间，并排除含评估、保存边界的区间，因此不纳入初次构图/编译。完整恢复的初始 update 大于0，必须先扣除；初版发现此边界后已修复并纳入测试。CSV 同时保留全局 updates 和 updates_this_run。本次各组保留 32 个区间，表中用中位数。它是整个训练更新期的实测日志推导速度，不是独立 kernel 计时，也不含每次评估开销。总耗时采用 runner `summary.elapsed_s`，不包含源码 launcher 的构建时间。

## 本次完整 512 向量步结果

每组 512 环境、262,144 transitions；前 178 向量步由继承策略收集完整曲目，不更新。所有分数取每批 3 回合的最差值。初始 step 0 是父策略在该组环境下的表现，**不是本次学习成果**。

| 组 | 设置 | 初始曲目 F1 | 256步 F1 | 512步 F1 / micro F1 | 末次准备错帧 max | 末次手碰撞帧 / 子步 max | 总秒 | 更新期采样/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | batch256 × 1，旧奖励 | .96745 | .72030 | .89916 / .89982 | 1 | 74 / 399 | 208.96 | 972 |
| B | batch1024 × .25，旧奖励 | .96745 | .88385 | .92743 / .92769 | 1 | 75 / 377 | 104.87 | 2631 |
| C | B + false/self/受任务条件限制的humanoid | .94973 | .75943 | .90190 / .91844 | 1 | 16 / 51 | 98.09 | 2642 |
| D | B + false/self，无humanoid | .96872 | .84720 | .86890 / .89170 | 1 | 18 / 66 | 98.02 | 2651 |
| E | B，恢复父Adam/alpha/RNG | .96998 | .89334 | .85341 / .87544 | 2 | 85 / 463 | 98.74 | 2618 |

A 更新 171,008 次，B/C/D/E 各新增 42,752 次；E 全局更新数为 4,389,753。A→B 短测总时间约减半，学习期采样吞吐约 2.71 倍；这是用户授权调整学习预算后的结果，不能当作更新轨迹不变的执行优化。

C/D 降低了真实手碰撞，却没有提升本轮音乐指标。B 的退化小于 A，但五组所有“学习后评估”的曲目 F1 均低于各自初始值；JSON 的 `learned_improved_over_initial` 全为 false。五组仍未达到 F1≥.99、准备零误触目标。不能因为 best.pt 留住初始策略，就声称新训练改善了效果。

![短测实测端点评估](../preview/outputs/diagnostics/quality_trials_20260926/quality_trials.png)

## A/B 退化的已核对事实

- A 初始 best.pt 的 actor、critic、target critic 与父 `step-8500.pt` 逐 bit 相同，update=0，三个 Adam state 为空；父 Adam step 均为 4,347,001。文件 SHA256 匹配，scene/task/准备/action 合同相同。
- 父与 A 初始实际温度均为 `0.0017778597539290786`，没有误重置成 1。A 的第 1024 次更新温度约 .001481，第 36,864 次约 .002245，没有温度数量级爆炸。
- A 与父 reward_config 的原始 JSON 存在新增 disabled 模板字段差异，但规范化后完全相同；没有误开 humanoid、false_press 或 self 奖励。
- 父原连续训练在 8500→8580 时曲目 F1 约 .96745→.95410；A 仅经历 78 个新控制步、39,936 次更新后已降到 .72030。A 到第 512 步恢复到 .89916，说明存在初期退化和部分恢复，不能据此断言永久遗忘。
- Replay 插入仍保存 reset 前终止观测，动态采样上限仍是实际有效行数。构图试跑有参数/Adam/RNG/计数回滚，此前 fresh/warm-Adam 数值测试保留。本次没有发现明确加载或计数错误，但只读检查不能代替所有运行路径的证明。

### 同一批父策略轨迹上的 CPU 对照

使用已有 `policy_training_5m_20260926/repeated/step-8500/detail-0/rollout.npz` 的真实 178 帧观测，单线程 CPU 前向，不重新仿真。父 CPU 动作与原 GPU 记录最大误差约 3.9e-6，确认观测和动作对齐。

| 对照 | 相对父动作平均绝对变化，归一化[-1,1] | 准备 / 曲目平均变化 | 最大变化 | 当前critic对父动作 / 新动作平均Q |
|---|---:|---:|---:|---:|
| A final | .27701 | .43308 / .25725 | 1.9761 | 15.8215 / 15.9888 |
| B final | .23504 | .18597 / .24126 | 1.9468 | 15.7900 / 15.9132 |

父 critic 对父动作的平均 Q 为 15.7891。A/B 在同样好轨迹状态上已经产生明显动作变化，同时各自 critic 更偏好这些新动作；实际闭环音乐指标却降低。这支持“策略明显偏离继承轨迹”的判断，不支持“只是评估随机波动”或“critic 数值整体爆炸”。Q 只是模型预测，以上不是对真实动作价值误差的直接测量。

## 机制推断与最小后续对照

下面是待验证解释，不是已确定的单一根因：

1. **Fresh Adam 不等于继续优化。** 父积累的动量、二阶矩和步数全部重置；即便学习率及权重相同，首轮参数更新的尺度和方向也可能改变。
2. **新 replay 更窄。** 初始仅一整轮 91,136 条相近的继承策略数据，首个评估时也只有 131,072 行；父稳态有 1M 条历史。原 critic 在新数据上的 TD loss 仍约 .01，只能说明能拟合当前采样分布，不能证明旧好轨迹附近的动作梯度可靠。文献也区分新数据拟合良好与保留初始化知识，并支持先用继承策略收集数据缓解分布变化；本文场景是在线 SAC 的重启，不可直接照搬其离线 RL 结论或超参。[WSRL 原论文 §3–4](https://arxiv.org/html/2412.07762v2)
3. **更新节奏改变策略漂移速度。** tau=.005 时，每个 A 向量步 512 次 Polyak 后旧 target 的系数约 .07681；B 每步 128 次后约 .52645。batch×更新比同为256，不代表 Adam/target 的时间尺度相同。B 初期退化较轻与此一致，但不能单凭一组种子证明因果。
4. **指标与学习目标不同。** SAC 优化加权 reward，不直接最大化曲目 F1；无行为约束时，多种动作可能取得近似 reward，却有不同误触、摆指和碰撞表现。新增奖励又会改变 last_reward 观测，C/D 初评应以各自 step0 为基准。

后续已执行 E：恢复完整 learner 状态 `--warm-start`，同时保持同样新 replay/prefill。它保留 Adam/alpha/RNG，但仍未恢复未保存的旧 replay；因为 RNG 也随之恢复，并非只改变 Adam 的严格单因素实验。严格配置匹配要求 CLI `--init-temperature 1` 与父原 SAC config 一致，load 后**实际温度恢复父 .00177786**，并不会把运行温度重设为1。此路径保持旧奖励，不加 `--allow-reward-change`。

E 并未改善退化，最终还低于 B。因此“只因 fresh Adam 导致”“完整恢复 Adam 就会更稳”均没有得到本轮支持；fresh optimizer 是候选因素，不是已证实主因。新增奖励仍要求显式 networks-only；已经在新奖励下学习过的 C final 若继续使用完全相同奖励，则可以严格完整恢复 C 自己的 Adam/alpha。后者合同合理，不代表已验证长训会更好。恢复时 CLI 初始温度要匹配 **C 保存的配置值**，实际运行温度由 C 学得状态恢复；完整长期达标仍属 V32-QUALITY-01。

其他候选包括先收集更多完整继承策略回合，或经用户确认再减小 Actor 学习率；没有在本审计中执行。随机重置 critic 会丢失当前仍有用的价值初始化，不是本轮首选。Actor LR 3e-5 的试验由主任务等待明确用户决定后处理，本脚本没有修改任何训练参数。

## 验证与日志

实际执行汇总并查看图表；三项 CPU 测试通过：

```bash
CUDA_VISIBLE_DEVICES='' training_mjlab/.venv/bin/python -m pytest tests/test_quality_trial_summary.py -q
```

覆盖排除评估/保存/首更新区间、初始 best 不冒充学习提升、运行中尾行读取。参考文件：各组 config/metrics/summary、父8500检查点及原 metrics、`rl_torch/{train,checkpoint,sac,learner_graph,replay}.py`、`scripts/audit_checkpoint_trajectories.py` 和已保存 rollout。V32-QUALITY-01 仍开放：没有候选证明长期训练可达到 .99；后续需按主任务验收和授权继续。主任务统一合并根/独立项目变更日志。
