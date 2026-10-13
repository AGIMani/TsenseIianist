# 并行环境 reset 动作历史修复（2026-09-24）

## 结论

旧任务在物理 reset 时把手、琴键、速度和执行器目标恢复到冻结场景的中立状态，却把 51 维“上次动作”观测填成动作空间下限。策略看到的历史控制与机器人实际控制目标不一致。它会影响 reset 后的第一步，是回放起始大动作的诱因之一。

它不是旧回放大动作的唯一原因。对旧 47 动作 `final.pt` 做固定输入比较：修正 reset 观测会让确定性策略输出最大改变 1.2076（归一化动作）；旧观测下第一步物理目标相对中立姿态最大跳变 1.3256 rad，改成中立历史后仍有 1.2894 rad。旧策略已经在旧观测合同下训练，本身也会给出很大的首步位置目标。不能通过改一项观测保证旧检查点立即平滑。

## 主流并行环境的处理方式

- Gymnasium 1.1 明确区分 `NEXT_STEP`、`SAME_STEP` 和 `DISABLED` 三种向量环境自动 reset 模式。结束 transition 必须保留终止观测；同一步自动 reset 时另存 `final_obs`。当前项目采用等价于 `DISABLED` 的显式流程：先写入终止 transition，再只 reset `done_ids`。[Gymnasium 1.1 release notes](https://gymnasium.farama.org/main/gymnasium_release_notes/)、[SyncVectorEnv source](https://gymnasium.farama.org/_modules/gymnasium/vector/sync_vector_env/)
- Isaac Lab 的 RL 环境按 `env_ids` 重置 observation/action/reward/termination 等 manager；`ActionManager.reset(env_ids)` 专门清空选定环境的动作历史。位置控制 EMA 项在 reset 时把上一动作初始化到当前关节位置，避免历史状态与物理姿态脱节。[ManagerBasedRLEnv reset source](https://isaac-sim.github.io/IsaacLab/develop/_modules/isaaclab/envs/manager_based_rl_env.html)、[ActionManager API](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.managers.html)、[Isaac Lab MDP API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.mdp.html)

采用的规则是：**局部 reset、先保存终止帧、再恢复物理状态，并把动作历史恢复到实际中立控制目标**。没有加入动作插值、EMA 或限速，因为这些会改变整个控制器和训练 MDP，不属于 reset 状态修复。

## 实现

- 新导出 task 的 `reset_action` 取冻结 scene 的 `neutral_action`，并写入 `reset_action_semantics=scene_neutral_physical_v1`。
- 导出时只修正 CPU 参考数据中 reset 帧最新一层的 action 槽；后续动作、奖励、乐谱和历史顺序不变。
- task 加载时校验 reset action 为 51 个有限值，并在新语义下逐项等于 scene 中立动作。
- 训练 `config.json` 记录 `reset_contract`：显式局部 reset、终止 transition 先入 replay、物理恢复冻结场景、上一动作使用中立控制。
- 旧 task 没有新字段时继续按 `legacy_action_lower_bound_v1` 解释。新 task 散列不同，旧 checkpoint 不能静默 warm-start 到新合同。
- `train_twinkle_h12_no_tz.sh` 及其 47 动作包装入口改从当前源码缓存构建 scene/task，确保新训练使用修复合同。

## 验证

- 规则单测 7 项通过，含三帧历史只替换 reset 最新帧动作的检查。
- TZ/Yaw 结构锁止 2 项通过。
- 新缓存 task 的 `reset_action` 与 scene `neutral_action` 最大误差为 0；CPU reset 参考观测最大误差约 `1.31e-8`。
- 16 环境验收中，固定状态、局部 reset、未 reset 环境不变、终止观测所有权和时间截断均通过。已有 `press_physical_rollout` 差异仍为 `max observation error=0.13092`，超过历史阈值 0.1；它是既有 MuJoCo CPU/GPU 接触轨迹差异，不由本次 reset 改动引入。
- 47 动作、TZ/Yaw 物理移除的 2 环境×4 步 GPU smoke 完成，`config.json` 正确记录新 reset 合同。
- 从头训练的早期 `step-1000.pt` 已使用新合同；首步目标相对中立值最大为手指0.62994 rad、腕平移0.14644 m、腕转动0.00760 rad、踏板0.55615。它刚过学习起点，尤其腕平移仍大，不能作为最终平滑验收。

## 检查点与回放边界

旧检查点继续配旧 scene/task 回放，便于复现实验，但保留旧 reset 观测缺陷。修复后的正式训练必须从头开始；不能把旧网络称为已修复，也不能把旧 optimizer/replay 迁移到新 task。新训练完成后，应重新测量 reset 后首步目标相对中立姿态的每关节跳变量，并检查视频；如果仍有不可接受的大动作，再单独选择动作插值、目标变化率限制或平滑奖励，并把它作为新的控制合同训练。
