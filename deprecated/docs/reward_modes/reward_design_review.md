# rew. 总表：已实现的可选奖励与必填映射

| rew. 项／配置键 | 已实现公式（内部系数已含在公式内） | 默认训练 | 启用前的要求 |
| --- | --- | --- | --- |
| `rew.key` / `key_press_reward` | legacy：`0.5 r_on + 0.5 无错键`；paper：`0.6 r_on + 0.4 exp(−1.25 N_wrong)` | legacy 开，权重1 | `params.mode` 选 legacy／paper；可另开和弦倍率，作用范围必须填写 |
| `rew.sustain` / `sustain_reward` | 当前踏板目标／实际状态容差奖励 | 开，权重1 | 当前项目扩展，论文无此项，可独立关闭 |
| `rew.energy` / `energy_reward` | 当前 `−0.005 Σ abs(执行器力×速度)`（系数取冻结任务） | 开，权重1 | 与下面正功率三项互斥，避免重复计罚 |
| `rew.fingering` / `fingering_reward` | paper：活跃指到指定键目标的高斯容差平均，成功距离≤0.01m、margin0.1m；新task对R1/L1使用用户确认的白键底边向上2 cm、黑键1.5 cm（世界X） | legacy 开，权重1 | paper 模式固定论文容差；目标几何是TsenseI适配，旧冻结task仍用旧点 |
| `rew.forearm` / `forearm_reward` | 按冻结任务名单：无接触+0.5，否则0；新M6包扩展跨臂及臂—任一手，排除自身Link7—掌安装对；原掌指／掌掌包维持旧范围 | 开，权重1 | 2026-09-21/22用户授权扩展；需选择新M6 scene/task；与包含 inter_hand 的论文碰撞项互斥 |
| `rew.self_collision` / `self_collision_reward` | `−N_same_hand_pairs` | 关，候选外部权重0.2 | 延续既有同手部件对计数，可单独选择启用 |
| `rew.false_press` / `false_press_penalty` | `−Σ非目标键 I(x>0.001)×(0.01+0.14x)`，x为截到[0,1]的下压比例 | 关，权重1 | 用户指定旧Daxian公式；[启用方法与说明](false_press_penalty_20260924.md) |
| `rew.humanoid_finger_gesture` / `humanoid_finger_gesture_reward` | 当前无指法目标的每指 `0.2 exp(−MSE(q,q⁰)/(2·0.25²))`，四关节实际角；十指相加 | 关，权重1 | `--humanoid-finger-gesture` 显式启用；[标准姿态与公式](humanoid_finger_gesture_20260926.md) |
| `rew.forearm_power` / `forearm_power_reward` | `−0.001 Σ_T max(P,0)` | 关 | 先关闭 legacy energy |
| `rew.wrist_power` / `wrist_power_reward` | `−0.01 Σ_W max(P,0)` | 关 | 先关闭 legacy energy |
| `rew.finger_power` / `finger_power_reward` | `−0.005 Σ_F max(P,0)` | 关 | 先关闭 legacy energy |
| `rew.forearm_velocity` / `forearm_velocity_reward` | `−0.02 Σ_T abs((u_t−u_(t−1))/dt)` | 关 | 必填 `action_units` |
| `rew.wrist_velocity` / `wrist_velocity_reward` | `−0.005 Σ_W abs((u_t−u_(t−1))/dt)` | 关 | 必填 `action_units` |
| `rew.finger_velocity` / `finger_velocity_reward` | `−0.0002 Σ_F abs((u_t−u_(t−1))/dt)` | 关 | 必填 `action_units` |
| `rew.forearm_acceleration` / `forearm_acceleration_reward` | `−0.002 Σ_T abs((u_t−2u_(t−1)+u_(t−2))/dt²)` | 关 | 必填 `action_units` |
| `rew.wrist_acceleration` / `wrist_acceleration_reward` | `−0.0003 Σ_W abs((u_t−2u_(t−1)+u_(t−2))/dt²)` | 关 | 必填 `action_units` |
| `rew.finger_acceleration` / `finger_acceleration_reward` | `−0.00001 Σ_F abs((u_t−2u_(t−1)+u_(t−2))/dt²)` | 关 | 必填 `action_units` |
| `rew.finger_speed` / `finger_speed_reward` | `−0.05 Σ_F max(abs(Δu/dt)−6.5,0)` | 关 | 必须选择 physical，阈值单位rad/s；Stage1模板权重0，启用时还须自行选择权重 |
| `rew.collision` / `collision_reward` | `0.2 × (1−N_undesired_pairs)` | 关 | **必填 categories**；可显式填写忽略的body对 |
| `rew.natural_posture` / `natural_posture_reward` | `−0.004 Σ_(W∪F) abs(u−u_base)` | 关 | **必填动作单位及 reference**；不自动认定当前初始姿态就是自然姿态 |
| `rew.wrist_pitch` / `wrist_pitch_reward` | 每手 `0.05 × [direction × u_pitch > 0]`，双手相加 | 关 | **必填左右手 directions**，不自动填同号或反号 |
| `rew.forward_press` / `forward_press_reward` | 每指 `−0.1 × [φ_f > π/2]`，十指相加 | 关 | **必填十指 angle_map**；没有自动使用 MCP+PIP+DIP |

## 1. 本轮用户决定与实现范围

2026-09-18用户要求“按照论文实现当前所有奖励，但是由我自己来选择是否要启用”，并明确选择：**实现公式接口，但未确定的映射必须由用户填写才能启用**。

上表现有22项均已实现开关、外部权重和运行时计算，包含论文全部已公开的16类／分组奖励，以及当前项目的踏板、绝对能耗、左右掌部、纯同手碰撞兼容项。2026-09-24按用户要求新增旧Daxian的下压深度错键惩罚，默认关闭；用户启用后与按键项叠加。2026-09-26另增用户指定的空闲手指初始姿态奖励，默认关闭，独立于论文自然姿态项。和弦倍率仍作为按键项选项。

**原命令的默认奖励组合不变。** 不传新配置时，仍为原五项CPU规则，不乘dt；已有 `--self-collision-weight` 也保留。选择某个JSON、修改enabled或传CLI开关，才改变该次训练。论文模板不是“全部开启”的预设。

所有新增公式可在GPU上批量计算；本轮没有替你确认任何腕方向、十指角组合、自然姿态或论文不期望接触名单，也没有更改手模型、20Hz控制、腕TZ范围、1180维观测、终止规则或SAC参数。

## 2. 配置文件：由用户逐项选择

| 文件 | 选择此文件后的初始组合 | 其余项 |
| --- | --- | --- |
| [current.json](../configs/rewards/current.json) | 原五项，legacy按键／指法，总奖励不乘dt | 全关；其他外部权重按Stage1准备 |
| [paper_stage1.json](../configs/rewards/paper_stage1.json) | 只开paper按键＋指法，总奖励乘实际dt | 全关，待用户选择；使用表3 Stage1权重 |
| [paper_stage2_canon.json](../configs/rewards/paper_stage2_canon.json) | 只开paper按键＋指法，总奖励乘实际dt | 全关；表3 Stage2 Canon权重 |
| [paper_stage2_other.json](../configs/rewards/paper_stage2_other.json) | 只开paper按键＋指法，总奖励乘实际dt | 全关；表3 Für Elise／Beethoven权重；不自动将小星星归入此类 |

每项都有 `enabled`（开关）、`weight`（外部权重）、`params`（内部系数或映射）。下面只是字段片段，修改完整模板中的对应位置：

```json
"finger_power_reward": {
  "enabled": true,
  "weight": 0.2,
  "params": {"coefficient": -0.005}
}
```

权重不乘两次：`raw=内部系数×测量值`，`weighted=weight×raw`，`contribution=(scale_by_dt ? dt : 1)×weighted`，最后 `reward=Σcontribution`。

- `enabled=false`：不参与总奖励，不要求填写尚未确定的映射。
- `enabled=true`：相关必填项立即验证，即使weight=0也不能绕过映射检查。weight=0会使贡献为0；尤其Stage1手指超速的预设weight为0，不要只打开enabled就期待它产生扣分。
- 所有项都允许独立关闭，全部关闭时总reward为0；这仍可运行，但不再提供弹奏学习信号。
- 缺少字段、未知项、非有限值、负外部权重、同时启用重复能耗／碰撞项会报错，不静默忽略拼写错误。
- 编辑配置只影响下次启动。运行中修改JSON不会改变已经保存到内存和checkpoint的配置。

## 3. 使用命令

从独立项目根目录执行：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist

# 建立自己的配置副本，然后用编辑器修改enabled、weight和必填字段。
cp configs/rewards/current.json configs/rewards/my_rewards.json

# 查看最终配置及检查未填字段，不分配GPU物理场景。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --reward-config configs/rewards/my_rewards.json --inspect-only

# 按你的配置开始约5M样本训练；进度条及三项加速默认开启。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --reward-config configs/rewards/my_rewards.json --num-envs 256 --vector-steps 19532
```

也可用命令行覆盖单项开关，不覆盖数值或映射：

```bash
# 从当前规则改为论文分组正功率；其余保留当前配置。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --disable-reward energy_reward \
  --enable-reward forearm_power_reward \
  --enable-reward wrist_power_reward \
  --enable-reward finger_power_reward --inspect-only
```

`--enable-reward`／`--disable-reward`可以重复；先读JSON，再应用CLI。不得在同一命令中同时开关同一项。`--reward-config`与非零`--self-collision-weight`互斥，请在JSON内改self_collision_reward。

## 4. 必须由你填写的内容

### 4.1 动作单位：action_units

论文式16–18对归一化动作与rad/s阈值的对应位置没有完整披露（V32-PAPER-U02）。模板为null，不替你决定：

- `physical`：将策略动作映射为实际控制目标后作差；平移m，旋转rad。
- `normalized`：将目标转换回[-1,1]后作差；速度单位1/s，加速度1/s²，系数与物理单位版本的作用不同。
- 手指超速项的limit以rad/s定义，因此要求physical，不把6.5直接用于归一化差分。

按冻结scene的action_names映射：T=双手6个Wrist_TX/TY/TZ，W=双手4个Wrist_Pitch/Yaw，F=40个手指执行器。踏板不参与这些正则。

### 4.2 自然姿态：natural_posture_reward.params.reference

模板为null。你可显式填写字符串`scene_reset`，表示采用该冻结场景的初始控制目标；也可填写包含全部44个非前臂动作名的物理目标字典，每项需在限位内。自定义值始终以物理单位输入，选择normalized时程序再转换。

双手惩罚相加，未除以2；这些外部权重来自论文单手系统，对双手的相对强度需要另行比较，不能当作已标定参数。

### 4.3 腕pitch：wrist_pitch_reward.params.directions

模板为null。必须提供`R_Wrist_Pitch`和`L_Wrist_Pitch`两个键，每个值只能是+1或−1。+1奖励该手正的物理命令角，−1奖励负的物理命令角。当前模型同号pitch反向转动，因此不能把两侧符号自动当作同一物理姿态。

不填值时，设置enabled=true会报错；没有默认左右映射。两手符合时最多+0.1（尚未乘外部权重和dt）。

### 4.4 前向按压：forward_press_reward.params.angle_map

模板为null。必须填写十个键：`R_Thumb/Index/Mid/Ring/Pinky`及对应的`L_...`。

每个条目需要：

- `joints`：本指实际action_name到系数的字典；不能写不存在的关节，不能引用另一只手或另一根指。
- `offset`：等效角常量偏移，rad。

当前映射接口为显式线性组合：`φ_f = offset_f + Σ coefficient_(f,j) × physical_command_j`。然后严格按论文比较 `φ_f > π/2`（等于阈值不罚）。关节清单查冻结scene/manifest.json的action_names，或包内手模型。

**没有默认填入MCP+PIP+DIP，也没有默认排除或加入拇指旋转关节。** 这些由你填写。若确认后的定义需要非线性几何计算，则应扩展映射接口，而不是声称某个关节角和就是论文未披露的等效角（V32-PAPER-U05）。

### 4.5 碰撞：collision_reward.params.categories

模板为null。你必须选择非空类别列表：

| 可选值 | 当前模型中的明确含义 |
| --- | --- |
| `self` | 同一手内、不同且非同一焊接刚体的body对 |
| `inter_hand` | right_/与left_/之间的body对 |
| `piano_case` | 任一手的body与当前琴箱`piano/base` |

同一控制周期全部物理子步（当前40/400 Hz为10个，旧冻结包为100个）及末尾forward中，出现`contact.dist<=0`的body对取并集；多接触点、反向pair及多个子步不重复计数。正常手指—琴键不计入。物理模型本身的碰撞过滤／装配exclude继续生效，不由奖励改变。

你还可在`ignored_body_pairs`填两两body名称的列表作为显式忽略项；名称必须在当前模型中存在。当前记录的是接触发生，不是力、穿透深度或持续时间；此计数是本项目适配，论文未披露完整模型名单及时间归约，不能称为作者原配置。

选择self时必须关掉self_collision_reward；选择inter_hand时必须关掉forearm_reward。论文项保留式19的常数：无碰撞为+0.2，一对为0，两对为−0.2；与纯同手项`−0.2N`不同。短暂碰撞也会计入，但负奖励不能保证零碰撞，更不能替代硬限速。

### 4.6 和弦倍率：key_press_reward.params

`chord_bonus`默认false。启用时要求paper模式，并由你填写`chord_scope`：

- `on`：倍率只乘目标键深度项r_on。
- `whole_key`：倍率乘整个按键奖励。

目标为两键时倍率1.08，三键以上1.15；以当前双手目标键总数计数。原文p.11没有清楚指定作用位置，因此没有替你选scope。静音步r_on取0（沿用当前空目标约定），r_off仍按错键数计算。

### 4.7 Stage2 shaping：regularization_shaping

表3注释说Stage2还对功率／速度／加速度／超速应用shaping，但论文没有给出该函数。Stage2模板为null；启用上述正则时会要求显式填写。

目前可填`identity`，明确选择“只执行公开公式和表3权重，不使用未知的非线性shaping”。这不是论文shaping复现。若要严格一致，需要补充作者函数后再实现；程序不自行猜成tanh、指数或其他函数（V32-PAPER-U01）。Stage1没有这个未知变换，使用identity。

## 5. 论文外部权重（已写入模板）

内部系数见开头总表，外部权重来自p.12表3：

| 项目 | Stage1 | Stage2 Canon | Stage2 Für Elise／Beethoven |
| --- | ---: | ---: | ---: |
| 按键 | 1 | 2 | 2 |
| 指法 | 1 | 1.5 | 1.5 |
| 前臂／腕／指功率 | 0.2 / 0.176 / 0.2 | 4.6 / 0.1 / 1 | 4 / 0.1 / 1 |
| 前臂／腕／指速度 | 0.2 / 0.176 / 0.2 | 2.6 / 0.1 / 1 | 2.2 / 0.1 / 1 |
| 前臂／腕／指加速度 | 0.2 / 0.176 / 0.2 | 6.4 / 0.1 / 1 | 5.5 / 0.1 / 1 |
| 手指超速 | 0，默认关闭 | 0.4 | 0.3 |
| 碰撞／自然姿态／腕pitch／前向按压 | 各1 | 各1 | 各1 |

`--stage`仍选择SAC参数与权重转移流程；**不会自动修改奖励开关或奖励JSON**。weight_stage是所选权重模板的标识，选择哪套奖励由你决定。

## 6. dt、历史与数据合同

- 论文p.10式12：总奖励乘控制dt。`scale_by_dt=true`实现该规则，使用**当前场景实际dt=0.05s**；不硬写论文0.025s，也不借此将控制频率改为40Hz。scale_by_dt=false则直接求和。
- 每环境独立保存两步物理控制目标历史；reset将它们都设为冻结场景neutral_action（对应reset_ctrl），不能用旧观测reset_action的下界向量。首步跳变照常计罚。
- 局部reset只重置对应环境的历史。加速度项会使用第二步动作历史，当前1180维观测没有新增该历史或速度字段，仍可能存在部分可观测性；未擅自扩展SAC输入。
- 新reward进入replay、episode return及下一观测的上次reward。训练日志包含`reward_raw_terms`（内部公式值）、`reward_weighted_terms`（外部权重后）、`reward_terms`（dt缩放后实际贡献）。关闭项不出现在这三个逐步分项字典中，完整开关仍在config.json。
- 完整奖励配置保存在config.json与checkpoint内。独立评价、窗口回放、有声视频自动恢复配置，**不重新读取可能已经编辑过的外部JSON**。
- 旧checkpoint自动视为legacy五项，加上其原self_collision_weight。等价配置沿用原数值执行路径；正常warm-start要求奖励配置一致。

若你明确要Stage1网络转到不同的Stage2奖励，先填好另一份JSON，再显式选择仅网络转移：

```bash
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --stage 2 --stage2-from <Stage1检查点.pt> --allow-reward-change \
  --reward-config <你已填写的Stage2奖励.json>
```

此时新建optimizer／replay／alpha，记录旧奖励配置与来源；普通`--warm-start`不能通过此选项静默改变奖励。检查点的场景／乐谱合同仍须一致。

## 7. 验收与剩余资料缺口

实现入口：[reward_config.py](../tsensei_v3_2_pianist/mjlab_env/reward_config.py)、[rewards.py](../tsensei_v3_2_pianist/mjlab_env/rewards.py)、[task.py](../tsensei_v3_2_pianist/mjlab_env/task.py)、[self_collision.py](../tsensei_v3_2_pianist/mjlab_env/self_collision.py)。

本轮直接核对附件PDF的pp.10–12，而非仅使用先前摘要。公式与表3均按公开内容实现；详细实际验证见[奖励实现验收](reward_options_acceptance.md)。

```bash
training_mjlab/.venv/bin/python -m unittest \
  tests.test_paper_rewards tests.test_mjlab_task tests.test_torch_sac \
  tests.test_acceleration_and_collision -v
```

V32-PAPER-U01（Stage2 shaping）、U02（作者动作单位）、U05（作者碰撞／姿态／等效角映射）仍未由论文资料解决。你的配置选择是明确的工程适配，不会被记成查明作者实现。原手模型速度／力矩、接触与正式右手标定TODO继续保留。


## 2026-09-20：本轮启用同手自碰撞惩罚

用户要求增加手指自碰撞惩罚后再训练。本轮按既有定义启用`self_collision_reward`、采用现有候选权重0.2，原五项保持；全局默认模板仍关闭。不涉及需要填写名单的论文`collision_reward`或其他未知映射。范围包括同手手指／掌部，正常触键排除，10子步去重。具体run和对照条件见[训练记录](training_self_collision_20260920.md)。

2026-09-22 M6 扩展规则、1175 对名单、GPU 查询模型与排除项见 [93 mm M6 验收](m6_forearm_93mm_20260922.md)。
