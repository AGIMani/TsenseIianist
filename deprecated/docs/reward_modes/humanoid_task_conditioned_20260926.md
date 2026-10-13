# 限制姿态奖励预算，并保护近期演奏动作

## 审计结论

旧 `humanoid_finger_gesture_reward` 奖励实际角度接近 reset 姿态，**没有直接奖励零速度**。不过，忽略当前是否弹对、仅豁免当前指法，并给空闲十指合计最多 `+2.0/步`，会让姿态回报与音乐任务竞争。原五项和本轮旧 5M 均没有启用此项，不能把旧训练的错误归因于它。

- 准备空谱十指保持参考姿态可得 `+2.0/步`，20 帧合计 40。
- 一个下一音要用的手指，若四关节都提前偏离 0.5 rad，每步损失 `0.2×(1−exp(−2))≈0.17293`；提前 5 步合计损失约 0.8647。原按键项从完全漏音到按成，单步最多增加 0.5。
- 一个明确的代数反例：其他奖励相等，单音漏按但九个空闲指都在参考姿态时，`key+gesture=0.5+1.8=2.3`；弹对音但九个空闲指各四关节偏 0.5 rad 时，`1+1.8exp(−2)≈1.24360`。这是奖励公式允许的反例，不代表已测得一条同时实现这些角度和按键状态的物理轨迹。

用户已授权以 reset 的 40 个实际手指关节角为参考，本轮继续使用该映射。论文 pp.11–12 式20仍是非前臂动作命令的负绝对偏差，映射与单位缺口保持；以下为独立工程方案，没有替换 `natural_posture_reward` 或代填论文映射。

## 新可选模式 `task_conditioned_v1`

默认模板和旧参数对象保持不变。新模式必须显式写入奖励 JSON：

```json
"humanoid_finger_gesture_reward": {
  "enabled": true,
  "weight": 1.0,
  "params": {
    "mode": "task_conditioned_v1",
    "maximum_total": 0.05,
    "sigma": 0.25,
    "lookahead_seconds": 0.25
  }
}
```

其余奖励项仍须保留完整配置结构。

令 `P_f(t)` 表示当前帧至未来 0.25 秒内任何帧给手指 f 分配了目标。在 20 Hz 下保护当前帧及随后 5 帧，包含边界；曲末越界补空，不循环到曲首。预计算完整乐谱的保护表，按各环境实际 score index 索引，reset 不影响其他环境。范围不能超过任务现有乐谱前瞻。

令 `S(t)=1` 当且仅当当前至少有一个目标键，且当前 88 键激活向量与目标完全一致。漏键、错键、准备空谱和休止均为 0。踏板继续由独立原奖励处理。

`gesture = S(t) × maximum_total / 10 × Σ_f [¬P_f(t)] exp(−MSE(q_f,q⁰_f)/(2σ²))`

- 使用固定十指分母，不按剩余空闲指数重新放大。
- 当前以及近期要用的手指可自由离开参考姿态。
- `maximum_total≤0.1`、外部 `weight≤1` 强制保留小预算；本轮候选为 0.05。所有数字必须有限，`sigma>0`、时间和预算非负。
- 默认 `scale_by_dt=false` 时加权辅助项在 `[0,0.05]` 内；若用户显式选择总奖励乘 dt，沿用原缩放路径。
- 静止但漏音时新辅助项是 0；上述反例对应 `key+gesture=0.5`，弹对音一侧约 `1.00609`。这消除了该特定正奖励替代音乐的漏洞，不能保证任何总奖励配置下策略绝不静止或训练必然达到 F1≥0.99。

原按键、指法、踏板、能耗、掌部奖励和物理均不变。旧 `{maximum_per_finger:0.2,sigma:0.25}` 仍按原公式计算，历史配置规范化后没有新增参数。

## 碰撞计数与诊断

原 `self_collision_reward` 是控制周期内实际同手 body 对的并集数乘负权重，候选权重 0.2。它包含同手可碰撞的指—指及掌—指等，排除同 body、同 welded body、MuJoCo 未生成的装配排除对；只统计 `contact.dist≤0`，不按 manifold 多个点重复扣分。琴键、琴身及另一只手都不属于 self 类别。

Shadow 的原 `forearm_reward` 只看周期末跨手掌部代理，无碰撞 `+0.5`、碰撞 `0`。跨手指—指等接触仍有原物理反力，但这个原奖励不覆盖它们。本轮不自动扩充奖励名单或开启论文 `collision_reward`。

`PianoTaskEnv(..., collision_diagnostics=True)` 可额外记录同手及跨手；默认 `False` 不创建诊断计数数组，也不增加诊断 kernel。开启后仍支持完整物理 CUDA graph，每步 info 包含：

| 字段 | 定义 |
| --- | --- |
| `self_collision_pairs` | 本周期至少出现一次的同手 body 对数 |
| `inter_hand_collision_pairs` | 本周期至少出现一次的跨手 body 对数 |
| `self_collision_substeps` | 出现任意同手接触的物理子步数 |
| `inter_hand_collision_substeps` | 出现任意跨手接触的物理子步数 |
| `hand_collision_substeps` | 出现以上任一类接触的子步数，二者同时发生只计一次 |
| `collision_physics_substeps` | 本周期实际覆盖的物理子步数，20/200 Hz 应为 10 |

每项是 `[num_envs]` 张量。周期末额外 `forward` 可补充 body 对并集，但不多算第 11 个物理子步。单环境 reset 只清其行；普通 step 开始清上一控制周期的计数。计数代表接触出现，不代表接触力或严重程度。启用诊断不会把跨手数混入原 self 惩罚。

## 输入分布与恢复

观测保留原 last_reward 字段。启用 self、false_press 或新姿态项会按原逻辑改变这个字段，父策略即使暂不梯度更新，也可能因输入变化而产生不同动作。必须在新奖励环境中记录训练前评估；不能悄悄给旧奖励填回该槽。

代码变更属于新源码版本；旧 config 的严格 source hash 检查继续有效。新 recipe 须记录新散列，物理和乐谱的独立 scene/task 合同仍要与父 checkpoint 匹配。不能以“只是奖励”绕过源代码校验。

## 验证与限制

CPU 测试覆盖准备／休止、漏键／错键／和弦缺键、当前及近期指法豁免、前瞻边界／曲末补空、上限与无效配置、旧合同及反例。同手／跨手诊断测试覆盖 body 对去重、正常手—键排除、焊接排除、正距离排除、子步分母、末尾 forward 和局部清零隔离。

运行命令：

```bash
CUDA_VISIBLE_DEVICES='' training_mjlab/.venv/bin/python -m unittest \
  tests.test_task_conditioned_gesture tests.test_humanoid_finger_gesture \
  tests.test_collision_diagnostics tests.test_paper_rewards tests.test_mjlab_task
```

上述 CPU 30 项测试通过。GPU 两环境、24 个控制步的诊断开／关对照通过：原奖励最大差 0，qpos 最大差 `4.66e-10`、qvel 最大差 `2.98e-7`，保留 CUDA graph；每步覆盖 10 个物理子步，局部 reset 保留另一环境的状态及计数。另以 self 0.2＋false_press 1＋新姿态项组合运行 24 步，检查 1181/47 维度、准备期姿态奖励 0、奖励上限及分项、保护表索引均通过。

新 recipe 明确登记代码变更，导出的物理和乐谱仍与父 5M 相同：scene `5052ab74ad2a0c20baa94df1004ed7bef42be1e42cbaf217335e22e969fb8fb5`，task `ede90d1265b7d657262091490a7ea0ec6add0376d4d1a50f7a867c18410c7a08`。没有绕过旧源码校验。

GPU 集成验收脚本及实际结果保存在 `preview/outputs/diagnostics/humanoid_conditioned_20260926/`。24 步动作不是接触密集策略的完整回放，也不替代新策略的多次整曲评估。200 Hz 误触仍可能未被 20 Hz 原奖励采到；V32-START-02 保留。本轮没有改变原采样时序、错键终止、物理反力或动作插值。
