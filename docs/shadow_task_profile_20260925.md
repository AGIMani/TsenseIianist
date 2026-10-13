# Shadow 任务与动作合同同步（2026-09-25）

## 用户决定与范围

本轮用户要求按照本仓库完整 Shadow 训练同步，并授权新训练。参考是 `robopianist-rl/run.sh` 的有效配置及 `robopianist/robopianist/suite/tasks/piano_with_shadow_hands.py`，不是论文的另一套奖励或旧 Daxian 定制规则。新增显式 `task_profile=shadow_v1` 和 `action_profile=shadow_v1`；默认旧任务仍为 `tsensei_v1`，旧动作 profile 保留。

## 同步结果与保留的机械适配

| 项目 | 新 Shadow profile | 历史 V3.2 profile |
|---|---|---|
| 指法目标 | 所有十指均取琴键 geom 世界中心加 `(0.35*sizeX, 0, 0.5*sizeZ)` | 拇指用世界 +X 端向内白键 2cm／黑键 1.5cm，其他手指同 Shadow |
| forearm 奖励名单 | 仅左右掌部碰撞 geom；无接触 +0.5，有接触 0，控制周期末判定 | 左右掌部及五指，原公式不变 |
| 基本奖励 | 原按键、踏板、绝对功率、指法、forearm 五项 | 五项加用户所选可选奖励 |
| 策略动作 | 全部 `[-1,1]` 仿射到物理目标 | 旧 legacy／mixed_v1／direct_joint_v1 各自保留 |
| 四指 MCP 目标范围 | 保留每手食／中／无名／小指 `[0,1.2]rad`，策略 -1→0、0→0.6、1→1.2 | direct_joint_v1 为策略 0→0、1→0.6、2→1.2 |
| 控制结构 | V3.2 每关节独立位置执行器 | 不移植 Shadow 的 PIP/DIP 腱耦合 |
| 观测 | 原有类型、排序及物理动作历史；1180维 | 原合同保持 |

V3.2 没有 Shadow 的实体前臂，因此掌部是根部碰撞的形态适配，不能声称碰撞几何完全相同。手指仍具有真实碰撞反力；从 forearm 奖励名单排除手指不等于关闭其物理碰撞。新增 task profile 本身不锁死额外奖励开关；本轮启动脚本显式使用原五项，用户今后仍能选择附加项。

47维策略下，观测保留原有固定 TZ/Yaw 槽及51维物理动作历史，供旧物理合同稳定兼容；这与 Shadow 推导1164维输入有形态差异。手自身指尖 site、坐标、关节范围、惯量不由本任务 profile 改写。reset 的 neutral 动作历史与末帧所有权修复继续保留，不重新引入原 wrapper 的历史下限初值。

## 实现与兼容性

- `rl/environment.py` 的 `EnvironmentConfig.task_profile` 传入 CPU 任务构造器；`mjlab_env/task_bundle.py::export_task(..., task_profile='shadow_v1')` 记录在完整环境合同和 task manifest。
- `fingering_targets.py` 的目标算法与 manifest 按 profile 分派；GPU `mjlab_env/task.py` 使用同一算法。
- `forearm_collision.py` 提供版本化名单规则；旧缺少字段的冻结任务继续原 palm fallback，已有全手名单继续原语义。
- `mjlab_env/action_mapping.py` 为 `shadow_v1` 独立记录规范化策略坐标，物理目标复用已确认逐关节表；旧 checkpoint 跨动作 profile 恢复会明确拒绝。
- `TaskBundle` 校验 task profile、指法几何合同、Shadow palm-only 名单一致性。未知 profile 或混用旧拇指配置会报错，不能静默解释成新合同。
- `scripts/export_mjlab_task.py` 支持 `--task-profile`；源码训练入口由完整 recipe 传入该参数。

## 运动监测

`rl_torch/train.py::evaluate` 新增以下指标，整个回合在设备上累计最大值，结束后读取，不改变动作、奖励或物理：

- `motion_max_wrist_tx_speed_m_s`、`motion_max_wrist_ty_speed_m_s`。
- `motion_max_finger_speed_rad_s`、`motion_max_wrist_pitch_speed_rad_s`。
- 相应 `motion_max_*_target_jump_m`／`_rad`：相邻实际执行器 ctrl 目标之差，首帧比较 reset 后实际 ctrl。
- `motion_sample_hz` 标识实际控制采样率；本轮为20Hz。

**这些速度是每个控制周期末的值，不是200Hz物理子步峰值。** 接触后的短暂弹跳可能在两个采样点之间发生，因此这些指标用于长期训练筛查，不能替代逐物理子步接触力／速度诊断。

## 本次验证

18项测试全部通过，覆盖新旧拇指目标并存、掌部与手指名单分离、规范化动作与八个MCP目标上限、跨profile checkpoint拒绝、旧manifest加载、新manifest错配拒绝、首步ctrl目标突变以及reset不制造伪跳变，另包括原指法目标和训练预算／SAC测试。

实际 CPU Composer 构建、reset及一步控制通过：`task_profile=shadow_v1`、1180维观测、每手恰好1个掌部奖励geom、原五项奖励，拇指与其他手指的目标位置相同，奖励有限（该零规范化动作探针一步为1.7375292，不代表训练质量）。初次探针只在输出JSON时遇到NumPy标量序列化错误，转Python float后复验成功，无需修改训练源码。完整GPU导出与短训练结果由本轮主验收记录补充。

## 未解决项

- 形态适配参数与碰撞代理仍不是硬件标定结果。
- 输入维度、独立关节驱动、并行收集与更新时序不能与原 Shadow 单环境逐步轨迹完全等同。
- 平滑性改善须以新训练视频和子步诊断验证，不能仅由参数同步推出。
