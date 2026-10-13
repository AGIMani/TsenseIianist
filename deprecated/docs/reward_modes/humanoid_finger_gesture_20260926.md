# 当前奖励与空闲手指标准姿态

后续已增加保留本页旧公式的可选 `task_conditioned_v1`，限制整个辅助项预算、保护近期将使用的手指，并在漏键／错键／空谱时关闭姿态正奖励。详见 [风险分析与新模式](humanoid_task_conditioned_20260926.md)。默认模板仍沿用本页关闭的旧模式，只有显式 JSON 才选择新模式。

当前入口为项目根目录 `run.sh`，默认 `shadow_v1`、20 Hz 控制／200 Hz 物理、`policy_v1` 准备，奖励配置为 `current`，`scale_by_dt=false`。下列五项默认启用、外部权重均为 1。单个控制周期结束时计算并直接求和；正在运行的旧进程不会被新源码热修改。

定义 `G(x;u,m)=exp[ln(0.1)·(max(−x,x−u,0)/m)^2]`，表示 `[0,u]` 内得 1、在上边界外距离 `m` 时得 0.1。`K` 为当前目标琴键集合，`A` 为当前激活琴键，`s_k∈[0,1]` 为键行程归一值，`F` 为乐谱当前帧所有指法指定的 `(手指,键)` 对。

| 当前启用项 | 每控制步原始公式 | 范围与含义 |
| --- | --- | --- |
| `key_press_reward` | `0.5·Σ_{k∈K} G(1−s_k;u_k,10u_k)/max(|K|,1) + 0.5·1[A\K=∅]` | `u_k` 为冻结 task 的 `key_tolerance`；空谱无错键为 0.5；任一错键失去后半项。 |
| `sustain_reward` | `G(target_pedal−1[sustain≥0.5];u_k,10u_k)` | 目标与实际踏板开闭一致时为 1。 |
| `energy_reward` | `−c_E·Σ_{左右手驱动器}|actuator_force·actuator_velocity|` | `c_E` 为 task 的 `energy_coefficient`，当前 0.005；踏板不在求和中。 |
| `fingering_reward` | `Σ_{(f,k)∈F} G(‖tip_f−target_{f,k}‖;u_f,10u_f)/max(|F|,1)` | `u_f` 为 task 的 `finger_tolerance`，当前 0.01 m；空谱为 0。Shadow profile 使用原统一键目标点。 |
| `forearm_reward` | `0.5·1[左右掌部无接触]` | 本入口无 M6，Shadow profile 只看跨手掌部代理；有接触得 0。 |

其他奖励项默认关闭，包括 `self_collision_reward`、`false_press_penalty`、论文的功率／速度／加速度／碰撞／自然姿态等奖励。全部可选项、系数及启用条件见 `reward_design_review.md`；本任务没有改变它们。

当前5M训练日志的最后一条 `vector_step=9766`（5,000,192条样本）在该**单个向量步、512环境平均**的五项实际值依次为：按键 `0.9822075`、踏板 `1.0`、能耗 `−0.0755012`、指法 `0.9839906`、掌部 `0.5`；合计约 `3.3906970`。来源为 `training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894/metrics.jsonl` 最后一条含 `reward_terms` 的记录。此数不是全训练平均，且该旧轮没有启用新奖励。

## `--humanoid-finger-gesture`

本次从与 `run.sh` 参数相同的源码模型实际导出 reset `qpos`，冻结为 [40 关节标准姿态](humanoid_finger_gesture_pose_20260926.json)，每手五指各四关节。基准是本次场景的初始角，不是论文作者未公开的自然姿态映射。运行时从当前 scene manifest 按物理关节名读取相同 40 值，写入新训练的 `config.json` 和 checkpoint 元数据；旧场景运行时使用自己的 reset 值，不能静默套用这份快照。

对手指 `f`，记四关节当前实际角为 `q_fj`、初始角为 `q⁰_fj`，`I_f=1` 当且仅当当前乐谱帧没有给此指任何目标键（准备期空谱也适用）。新增奖励：

`r_gesture = Σ_{f=1}^{10} I_f · 0.2 · exp{ − [Σ_{j=1}^{4}(q_fj−q⁰_fj)²/4] / [2·(0.25 rad)²] }`。

每个空闲指上限 0.2；十指均空闲且恰好为标准姿态时总上限 2.0。被分配指法的手指为 0，与它当前是否真的触键无关。角度用真实 `qpos`，不使用策略动作目标；偏离越大，奖励平滑趋近 0。权重、单指上限和 `sigma` 可在奖励 JSON 的 `humanoid_finger_gesture_reward` 中调整。默认关闭；新训练显式运行 `bash run.sh --humanoid-finger-gesture` 启用，也可用 `--enable-reward humanoid_finger_gesture_reward`。

论文 pp.11–12 式20为 `−0.004Σ|a−a_base|`，针对非前臂动作命令，且自然姿态映射未知。本奖励由本次用户指定，采用实际手指关节、只在空闲指计分、高斯正奖励，独立于论文 `natural_posture_reward`；不会启用或替换式20。

验证：实际源码构建得到左右各 20 个姿态值；`tests.test_humanoid_finger_gesture` 验证按指法屏蔽、关节误差公式、40 值映射和无效参数拒绝；`tests.test_paper_rewards`、`tests.test_mjlab_task` 回归通过。尚未执行新奖励的长期训练或学习效果对照。
