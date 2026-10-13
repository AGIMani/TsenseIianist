# 可选错键下压惩罚（2026-09-24）

当前 MJLab/Torch 奖励配置新增 `false_press_penalty`，默认 `enabled=false`、外部权重1。是否启用由用户决定；本轮未开启正式训练或修改已有训练进程。

## 来源与公式

直接核对旧项目 `DaxianPianist/robopianist/robopianist/suite/tasks/piano_with_shadow_hands.py`：第39–53行常量及 `_false_press_cost()`、第808–820行 `_compute_false_press_penalty()`、第741行 `_key_goal()`。旧 `run_daxian_unlock_pip_dip.sh` 的 `--false-press-penalty` 启用这一项。独立项目运行时不读取或导入旧项目。

令当前控制帧第 k 个琴键目标为 `g_k∈{0,1}`，归一化下压行程为 `x_k=clip(q_k/q_max,k,0,1)`，则：

```text
c(x) = 0                  当 x ≤ 0.001
       0.01 + 0.14 × x    当 x > 0.001

false_press_penalty = − Σ(k=1..88) (1−g_k) × c(x_k)
```

这与旧项目的最小代价0.01、最大代价0.15、行程阈值0.001一致。实现见包内 `mjlab_env/rewards.py` 的 `false_press_penalty()`。

| 当前非目标键的行程 | 每键每控制帧奖励 |
| --- | ---: |
| 未下压／负行程／≤0.001 | 0 |
| 0.1 | −0.024 |
| 0.5 | −0.08 |
| 1或超限后截到1 | −0.15 |
| 两个键均为0.5 | 合计−0.16 |

规则：

- 只检查当前帧不需要的88个琴键，踏板不参与；目标键不产生此项扣分。
- 读取控制周期末的实际琴键行程，不要求琴键已达到二值激活阈值。因此轻微误压也会扣分。
- 用本次动作对应的当前目标帧，不能误用动作后的下一帧或前瞻目标。1秒准备时间、曲中休止符期间目标为空，所有被误压的键都属于非目标键。
- 对多个错键求和，不取平均；持续误压则每控制帧持续计罚；这项奖励不会触发提前终止。

现有按键奖励仍保留：默认 `0.5 r_on + 0.5 I(无已激活错键)`。新增项开启后叠加到总奖励。旧项目另有 `scale_fingering_by_false_press` 缩放指法奖励的开关，本次没有联动新增或开启该机制，也没有修改指法表。

论文pp.10–11式13–14为 `0.6 r_on + 0.4 exp(−1.25 N_wrong)`，是按已激活错键数量计算的按键奖励。本次用户明确选择旧项目的下压深度惩罚，作为独立可选项；论文按键模式和其他奖励公式保持原设置。

## 用户选择开关

在本独立项目目录运行以下命令。它们是使用示例，本轮未代用户执行正式训练：

```bash
# 当前默认：关闭
bash run.sh --disable-reward false_press_penalty

# 用户决定启用时，加入这一项；保留run.sh已有的其他配置
bash run.sh --enable-reward false_press_penalty

# 只检查生效配置，不启动训练
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --enable-reward false_press_penalty --inspect-only
```

也可修改 `configs/rewards/current.json` 的这一段：

```json
"false_press_penalty": {
  "enabled": false,
  "weight": 1.0,
  "params": {}
}
```

改 `enabled` 为 `true` 即启用，`weight` 是额外倍率。公式内部0.01／0.15／0.001固定为旧项目值。所有四个奖励模板均默认关闭此项。

JSON方式和 `run.sh` 同用时，需覆盖其快捷自碰撞参数；要保留自碰撞0.2，请在自己的JSON中设置 `self_collision_reward.enabled=true`、`weight=0.2`：

```bash
bash run.sh --self-collision-weight 0 --reward-config configs/rewards/my_rewards.json
```

总奖励仍遵循原配置：`contribution = weight × raw × (scale_by_dt ? control_dt : 1)`。当前默认不乘dt，权重1时与旧公式的每帧量级一致；若用户选用乘dt的论文模板，该项也随全局设置乘dt，不会暗中绕过全局规则。

## 记录、兼容与验证

完整配置保存在训练 `config.json` 和checkpoint；回放恢复相同开关。启用时日志有 `reward_raw_terms.false_press_penalty`、`reward_weighted_terms.false_press_penalty` 和最终 `reward_terms.false_press_penalty`。关闭时无该分项，原默认奖励路径保持。

旧奖励JSON／检查点没有这个字段时，加载为默认关闭；未知项和缺失其他必需奖励项仍报错。热启动不能静默改变奖励合同。源码配置检查点既有的源码散列检查仍有效，这项奖励配置兼容不等于允许使用任意新源码回放旧版本。

已运行 `training_mjlab/.venv/bin/python -m unittest tests.test_false_press_penalty tests.test_paper_rewards tests.test_mjlab_task tests.test_torch_sac`：28项通过。新增5项覆盖CPU／CUDA边界、截断、轻压未激活、目标键排除、踏板排除、多键累加、休止帧、当前目标时序、局部reset、权重及dt缩放、历史配置、CLI和检查点开关恢复。

实际GPU检查使用新源码构建的20/200 Hz、47动作／1180观测环境，仅在临时测试环境内启用此项：2环境×30步，含局部reset，60个样本均有非零惩罚；逐步对照独立NumPy公式的最大误差为1.49e−7以内，总奖励等于各项之和、观测与奖励均有限。报告位于 `training_mjlab/outputs/false_press_penalty_validation_20260924.json`。最初检查脚本误用不存在的policy_high/low属性，改用场景实际action_high/low后通过；未改动训练控制映射。

验证只说明公式、配置和运行路径正确，不代表启用后一定提高F1或已经收敛。正式启用选择仍由用户决定。
