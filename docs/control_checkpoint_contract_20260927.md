# 软件目标限制：训练、迁移与回放合同（2026-09-27）

本次用户已批准软件目标范围裁剪、变化率限制及 Actor 学习率 `3e-5` 的短测择优。限制器本体和候选范围见 `mjlab_env/control_limits.py` 及本轮工作空间诊断；本文记录检查点工程，不将软件目标限制称为真实关节速度保证或物理子步插值。

## 参数与保存

- `--control-limits path.json`：读取完整 JSON，规范化后保存到 `config.json`／检查点 metadata 的 `control_limits`。默认 `None`，保留旧行为。
- `control_limits_contract`：保存重建环境得到的完整解析结果，包括各通道物理索引、单位、原范围、软件范围、目标变化率、控制时间、真实 reset 目标和观测规则。单有文件名或只有规范化输入不够。
- 限制启用时，`reset_contract.previous_action=actual_reset_actuator_ctrl_v1`；观测历史最新帧记录实际 reset 控制目标，完整 51 槽保留固定轴槽，踏板为 0。后续 previous action 使用实际执行目标，观测维数仍为 1181，策略和 replay 动作仍为原 47 维请求动作。
- `--actor-lr 3e-5` 沿用既有学习率选项；本次合同集成没有修改 SAC 数学、网络、batch、更新预算或 GPU 执行实现。

## 显式变更与旧检查点

`checkpoint.load` 和 `load_agent` 同时严格比较输入与解析后的控制合同。旧检查点两字段都缺失按 `None` 解释；一字段缺失、一字段有值直接报错。启用限制的新检查点必须使用同一限制回放，不能静默去掉限制或改变速度。

只有 `--finetune-from ... --allow-control-change` 允许改变控制合同；该路径只迁移 Actor、Critic 和目标 Critic，必须是新的优化器、replay、温度和 RNG。初始温度仍由调用者显式指定，建议本轮短测使用父检查点学得温度，以免无意恢复到 1。父 metadata、父新旧控制合同与授权标志保存在 `origin`。`--warm-start`、`--stage2-from` 不能用该标志绕过控制校验。

允许控制变更不放宽 scene/task SHA256、准备、动作映射、TZ/Yaw 结构、碰撞物理或网络形状合同。奖励变化仍需另外的 `--allow-reward-change`。源码重建散列校验仍保持；历史检查点使用对应源码归档，新代码不会自动替换历史源码。

## 回放与统计

`reward_options()` 在建环境前读取保存的控制选项；标准 evaluate、交互预览和视频导出均把 `env.scene.control_limits` 传给 `load_agent` 再校验。相同权重在不同软件控制合同中不被视为同一回放。

评估新增逐回合及总体字段：

- `control_limits_enabled`、`control_limited_channel_count`；
- `control_limits_diagnostics_coverage`；
- `control_clipped_frames`、`control_slew_limited_frames`；
- `control_clipped_fraction`、`control_slew_limited_fraction`，分母是实际帧数 × 配置了限制的通道数；
- 总体均值及相应 `_max`，逐回合保留在 `episode_statistics`。

缺少一帧 mask 时覆盖率不足，受限统计记 `None`，不报告为零。禁用限制时明确 disabled，比例为 `None`。限制诊断不更改已有 F1、奖励、早停门槛。

## 验证与范围

CPU 命令（使用训练虚拟环境，禁用 GPU、单线程）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
training_mjlab/.venv/bin/python -m pytest \
tests/test_control_checkpoint_contract.py tests/test_training_quality_control.py \
tests/test_preparation_training_entry.py tests/test_preparation_checkpoint.py \
tests/test_shadow_training_budget.py tests/test_torch_sac.py -q
```

结果：**31 passed，54 subtests passed（1.48 秒）**。覆盖旧缺项兼容、best 权重确定性回放一致、控制参数／解析结果不匹配拒绝、未授权迁移拒绝、其他合同不能绕过、fresh optimizer/alpha/RNG，以及限制统计分母和缺失覆盖。

这组测试不证明限制改善 F1 或不再卡琴。实际候选回放、GPU 环境和同父短训由本轮主报告单独记录；选择后才能开始新控制合同训练。`V32-START-01`、`V32-START-02` 仍保留。

参考及改动位置：`rl_torch/{train,checkpoint,evaluate}.py`、`mjlab_env/{control_limits,scene,task}.py`、`preview/scripts/{preview_mjlab,render_mjlab_checkpoint}.py`、`tests/test_control_checkpoint_contract.py`、`tests/test_training_quality_control.py`、`docs/references/biomimetics_2026_610_notes.md` 的 2026-09-27 用户决策。总日志由本轮主任务汇总到项目及仓库 `docs/change_log.md`。
