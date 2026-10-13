# 整指几何保护配置

通过 `--finger-clearance-config PATH` 显式选择。通用 Python 默认 `None`；
新 `run.sh` 选 `control_disabled_20260927.json`：仅保留几何诊断/辅助奖励，
默认不改手指动作。`--idle-finger-protection`、`--active-press-permission` 可独立开启两项控制。
完整值和模型解析几何写入 config/checkpoint。
仅支持当前无臂、TZ/Yaw移除、47动作、Shadow primitive碰撞的V3.2模型。
旧检查点不自动启用；变更需要显式网络微调和 `--allow-control-change`。

| 参数 | 当前候选值 | 含义 |
|---|---:|---|
| `schema` / `mode` | `1` / `full_finger_clearance_v1` | 配置版本 |
| `idle_finger_protection` | false | 是否修正当前空闲整指目标；诊断/奖励不随此项关闭 |
| `active_press_permission` | false | 是否根据实际位置限制当前用指目标；可独立开启 |
| `clearance_m` | 0.002 | 闲指完整碰撞体相对白键平面/局部黑键的净空；单位米 |
| `contact_slack_m` | 0.001 | 严格目标投影的末端余量；不是实际按压力许可阈值 |
| `projection_iterations` | 6 | 固定次数几何投影 |
| `max_joint_correction_rad` | 0.25 | 每次投影更新上限，不代表每控制帧关节限速 |
| `active_command_mode` | `active_contact_v1` | 用实际位置判断用指传力/接近/恢复 |
| `active_contact_slack_m` | 0.005 | 相对完全按下顶面的实际软接触余量 |
| `active_support_xy_margin_m` | 0.002 | 实际末端与指定键支持区的XY余量 |
| `active_recovery_delta_rad` | 0.2 | 当前用指接近/危险恢复相对实际姿态的目标变化上限 |
| `wrist_clearance_interlock` | false | 可选低位闲指先抬起、暂缓该手腕目标移动；未作为更优默认 |

`strict_v1` 保留开发对照，但将正常按压目标误判为危险，原best回放F1降至0.083，
不作为生产选择。`idle_only_probe` 仅为消融，当前用指无保护，不能当正式全手方案。
旧 `generic_20260927.json` 缺少两个新开关时仍保持历史双控制开启，避免改变旧合同。
未采用的interlock/idle-only候选已移至 `../archive/20260927_control_trials/`；
腕部目标保持不等于瞬时刹停。诊断中的 `finger_clearance_enabled` 表示模块存在，
是否干预动作应查看上述两个配置开关与 `finger_projection_mask`，不能仅看该旧字段。

闲指成本使用请求/实际净空的最大缺口；正式模式的当前用指成本仅使用实际风险，
不会将用于产生按压力的PD目标穿透误当实际穿透。净空检查不裁剪曲目专属腕部位置，
不写物理qpos/qvel、不移除接触反力。有限的目标几何检查不是连续物理避障保证。
静态88键覆盖不等于任意曲目、和弦和速度下都已验证。

完整数据及后续短训选择见 `docs/generic_finger_clearance_20260927.md` 和
`docs/finger_clearance_geometry_20260927.md`（路径相对独立项目根目录）。
