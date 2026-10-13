# 2026-09-28：run.sh 参数、奖励与几何保护审计

本页对应用户要求：两种几何控制仅显式开启，查清自碰撞／humanoid／错键惩罚，解释腕限速，并减少含义重复的入口配置。日志由本轮主任务合并维护。

## 结论与实际改动

`run.sh` 之前确实每次传入启用两种保护的 `generic_20260927.json`。底层 Python 默认 `None` 不代表这个包装脚本默认关闭。本轮把**几何计算**和**动作控制**分开：

| 新 run.sh 行为 | 默认 | 权威配置 |
|---|---|---|
| 闲指整指目标净空控制 | 关 | `--no-idle-finger-protection` |
| 用指实际位置按压许可控制 | 关 | `--no-active-press-permission` |
| 几何缺口计算／200 Hz 键底接触诊断 | 开 | `configs/finger_clearance/control_disabled_20260927.json` |
| humanoid 几何／姿态有界负罚 | 开，mode=`geometry_clearance_v1` | `configs/rewards/generic_finger_clearance_20260927.json` |
| 同手自碰撞惩罚 | 开，权重 0.2 | 同上 JSON 的 `self_collision_reward` |
| 非目标键下压惩罚 | 开，权重 1 | 同上 JSON 的 `false_press_penalty` |
| 腕部目标限速 | 开 | `configs/control_limits/generic_wrist_rate_20260927.json` |

**关闭两种保护只取消它们的动作修正，不自动关闭几何奖励。** 这是为了满足“控制由训练参数开启”，同时保留此前已经批准的辅助奖励，没有把 humanoid 换回旧正奖励或重新设计其公式。

通用训练器两参数默认 `None`：无几何 JSON 且无显式参数时不构造模块；有 JSON 时按文件合同。旧 `generic_20260927.json` 缺少两个新字段，仍保留历史“两保护开启”语义和合同，旧检查点不得静默换为新行为。新默认 JSON 显式写入两个 `false`。

## 使用方式

```bash
# 默认：两种控制关闭，几何奖励与诊断保留。
bash run.sh

# 同时开启两种控制。
bash run.sh --idle-finger-protection --active-press-permission

# 只开启闲指整指净空。
bash run.sh --idle-finger-protection --no-active-press-permission

# 只开启用指按压许可。
bash run.sh --no-idle-finger-protection --active-press-permission

# 独立关闭 humanoid 辅助奖励（几何计算/键底诊断仍可保留）。
bash run.sh --disable-reward humanoid_finger_gesture_reward
```

成对布尔参数遵循最后出现者生效。因此用户在命令末尾传入开启参数，会覆盖脚本的显式关闭。直接使用底层训练器、没有 JSON 而只启用其中一项时，使用 `active_contact_v1` 的内置几何参数，另一项保持关闭。

新旧控制合同不同，不能直接 `--warm-start` 绕过校验；若确需改变保护设置后迁移网络，要显式 `--finetune-from ... --allow-control-change`，并保留旧回放所需源码归档。改变奖励则还须 `--allow-reward-change`。本次参数整理不热改已存在的运行与 checkpoint。

## 自碰撞为什么显示 0

原来命令中的 `--self-collision-weight 0` 是**未使用的快捷通道**，不是总奖励的最终权重。训练器遇到 `--reward-config` 时读取完整 JSON；该 JSON 明确把 `self_collision_reward` 设为 `enabled=true, weight=0.2`。非零快捷权重和 JSON 本来就互斥。

本轮移除脚本中冗余的 `--self-collision-weight 0`，保留 JSON 作为唯一奖励来源。底层兼容字段 `config.self_collision_weight` 仍可能为 0，这是旧快捷参数记录；查看实际奖励必须以 `config.reward_config.terms` 为准。

另发现 `scripts/plot_training_f1.py` 直接用旧快捷字段制作“自碰撞 0”图例，确实会误导。本轮修为优先读取完整奖励配置的 `enabled/weight`；只有没有完整配置的旧训练才回退到快捷字段。奖励本身未因此改变。

## geometry_clearance_v1 是什么

它是 `humanoid_finger_gesture_reward` 的一个工程模式名，不是第二份重复奖励，也不是 SAC 参数。

- 几何部分占 80%，姿态部分占 20%；当前用指与空闲手指采用已经批准的不同缺口定义。
- 准备、休止和弹错时均可扣分；不会因完全不动而发放正奖励。
- 总幅度受 `maximum_total=0.1` 和理想完整漏键损失的四分之一共同约束。
- 正常弹琴的 key/fingering 等原奖励继续独立计算。数值有界不是“任何情况音乐绝对优先”的证明。
- 它需要几何计算字段，但不要求启用动作保护。本轮训练器提前验证这个依赖，缺几何设置会在分配 GPU 前给出明确错误。
- 旧 `task_conditioned_v1` 仍可用，但本轮没有默认切回，因为用户只要求保护可选，没有要求回滚已批准的奖励。

论文 pp.11–12 式19/20与本工程项不同；本次只是控制开关／配置解释，不自行更换论文对应公式。未知自然姿态、方向或接触名单仍不代填。

## wrist 当前是否有限速

有。`--control-limits` 读取一个 JSON，在策略动作转换为物理目标后限制软件下发目标：

| 关节 | 目标变化率上限 | 20 Hz 时每控制帧上限 |
|---|---:|---:|
| 左右 Wrist_TX / TY | 1.2 m/s | 0.06 m |
| 左右 Wrist_Pitch | 4 rad/s | 0.2 rad |

它限制**目标变化率**，不是实际关节速度上限；惯性和接触仍能造成更快的瞬态实际运动。此 JSON 没有曲目专属位置 min/max，也没有手指速率限制。拇指 0.2 rad 的局部恢复参数属于用指保护算法，不能与腕部 Pitch 4 rad/s 混为一谈；关闭用指保护就不执行该恢复控制。

`run.sh` 可执行命令只有一个 `--control-limits`；头部注释和文档中的同名字段不是第二层限制。用户末尾再次传同名路径时按 argparse 的最后值覆盖。保留已经批准的腕限速，本轮没有因拇指漏键猜测自动撤掉它。

## 注释漂移与清理边界

检查时 `run.sh` 的实际 `--preparation-lift` 已是 **0.025 m**，头部仍写 2 cm。本轮保留现场的 0.025，修正文案为 2.5 cm；不能把这种未审计来源的本地修改擅自恢复成旧 0.02。

已清理入口冗余：移除 self-collision 0 快捷参数，清楚区分几何配置和控制开关。未增加重复奖励模板；原 `generic_finger_clearance_20260927.json` 奖励继续复用。

供主任务审核的归档候选（本页不执行删除）：

- 拒绝的 `configs/finger_clearance/idle_only_probe*`、`active_contact_interlock*` 候选及相应临时探针结果，可随失败实验一并归档，不应继续作为入口默认。
- 旧曲目专属 `configs/control_limits/wrist_gentle*` 应标记历史实验；旧续训合同／回放需要，不能直接删除。
- `configs/rewards/quality_baseline_*`、`quality_safe_*`、`quality_humanoid_*` 是已有对照实验可重建配置，不能仅因当前未引用而判定无用。
- source_versions、best/final 检查点、指标和完整 config 是旧训练的回放依据，必须保留；不能删除证据后声称旧回放仍可用。

## 参考与验证

参考文件：`run.sh`；`rl_torch/train.py::parser/effective_config`；`mjlab_env/reward_config.py`；`mjlab_env/rewards.py`；`mjlab_env/finger_clearance.py`；两个当前 JSON；`scripts/plot_training_f1.py`；`docs/references/biomimetics_2026_610_notes.md`。

实际验证：

- `bash -n run.sh` 通过；扫描两个 Python 入口的 AST，**100 个参数拼写（包含自动生成的 `--no-*`）全部出现在 run.sh 文件头**。
- `training_mjlab/.venv/bin/python -m pytest -q tests/test_run_entry_controls.py tests/test_training_quality_control.py tests/test_preparation_training_entry.py`：**21 passed，23 subtests passed**。实际执行 Bash，通过临时替代 Python 入口捕获最终 argv，没有构建 GPU 场景；验证默认两保护关闭、CLI 独立／最后值生效、旧文件合同、自碰撞真实权重及图例回退。
- 几何 reward 缺计算模块会在配置阶段拒绝；默认完整 run 配置通过，奖励保持 geometry 模式／self0.2／false1。

参数／CPU测试不等于新控制配置已经长训达到 F1 0.99。

### 配置引用清理审计（只读）

[机器可读清单](config_reference_inventory_20260928.json)记录了 24 个配置 JSON 的字节数、源码／文档字面文件名引用及已保存训练 config 引用。总共 **69,556 bytes（约 68 KiB）**；删除这些 JSON 对训练吞吐和显存没有实际帮助，整理的价值是降低入口歧义。

| 配置组 | 依赖证据／建议 |
|---|---|
| `generic_wrist_rate_20260927`、几何 `generic_20260927`、几何奖励 `generic_finger_clearance_20260927` | 新入口或旧回放／测试直接引用；各有 4 个已保存训练配置引用，保留 |
| 新 `control_disabled_20260927` | 本次新默认入口，保留 |
| `wrist_gentle_20260927` | 7 个已保存训练配置以及原续训说明引用，保留历史合同 |
| `combined_20260927` | 两份历史分析包含可执行复测命令，归档时必须更新这些命令，不能直接删除 |
| `quality_baseline/safe/humanoid_20260926` | `scripts/launch_quality_training.py` 按 `quality_{variant}_20260926.json` **动态拼文件名**；分别有 3/1/10 个已保存训练配置引用，保留 |
| 全部 `configs/rewards/*.json` | `tests/test_paper_rewards.py::test_templates_and_stage_weights`按 `glob('*.json')` 自动校验；目录新增任何 JSON 必须是合法奖励模板，不能放普通清单 JSON |
| `current`、`paper_stage1`、`paper_stage2_canon/other` | 参数文档／奖励验收引用，是用户已要求可选的配置，保留 |
| `actions/direct_joint_v1` | `test_shadow_task_profile.py` 直接读取，旧动作合同文档也引用，保留 |
| `poses/origin_20260926`、`initial_20260926` | 原准备姿态说明链接，属于留档证据；不进入新训练但应保留 |
| `envelope`、`thumb16`、`wrist_thumbpip12`、`wrist_tighter`、`wrist_ty06`、`wrist_ty08`、`active_contact_interlock`、`idle_only_probe` | 在本次限定源码／docs及训练 config.json 扫描中无完整文件名字面引用；一些存在候选简称的历史报告。可集中归档并给出索引，但“无字面引用”不等于未被 CLI／probe.json／旧归档使用 |

没有发现训练器扫描控制／几何目录自动启用其中全部配置；实际生效的是命令传入的单个 JSON。奖励目录的 `glob` 是验证测试，不是训练默认启用逻辑。候选归档不应放到训练输出搜索路径冒充新实验。

补充 CPU 回归：`tests/test_geometry_clearance_reward.py tests/test_clearance_checkpoint_contract.py tests/test_control_limits.py` **18 passed，43 subtests passed**；仅现有 Torch JIT 弃用警告。

主任务已完成清理：上述8个未采用候选移入`configs/archive/20260927_control_trials/`并保存index.json，未删历史依赖。新目标参数的manifest散列变化还要求任务合同兼容；`--allow-control-change`只放行控制，不能绕过task_sha验证。详见主报告。
