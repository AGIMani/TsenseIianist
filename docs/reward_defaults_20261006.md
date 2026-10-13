# 当前奖励与惩罚总表（持续维护）

最后核对：2026-10-13。本文件是当前奖励的唯一维护总表，沿用已有路径；以后直接更新本文对应行，不另建互相矛盾的默认表。旧实验和旧源码文档仅作为隔离记录，当前训练不读取。

依据根目录 [`all_use_this_train.sh`](../all_use_this_train.sh)（本轮更新固定归档，未启动训练）、活动 [`configs/rewards/default_20261006.json`](../configs/rewards/default_20261006.json)，以及脚本固定的 `logs/key_substep_restore_20261013/source_key_substep_activation50.tar.gz`。SHA256以统一脚本的强校验值及同目录归档审计为准。表内是**不追加其他参数时的最终配置**；具体实验仍以末尾覆盖参数和保存的完整 `config.json` 为准。当前合同为 `physical_key_activation50_v1`：按用户要求恢复 Job1724 的按键子步/50%激活公式，删除按键周期末纯键程算法；不恢复历史配置开关，也不改其他奖励。

当前共 **22 项：默认无臂时 8 项开启、14 项关闭**。显式 `--arms` 且初始角齐全时，默认额外启用独立 `arm_motion_limit_penalty`，成为 9 开、13 关；显式 `--disable-reward arm_motion_limit_penalty` 可覆盖。下面的分值已考虑当前 weight；关闭项贡献为 0，表中的公式说明显式启用后的行为。通用模板值不是本表默认值。

## 计分单位与符号

- 控制频率 20Hz，`dt=0.05s`；物理频率 200Hz，每控制步 10 个物理子步。
- 按键奖励使用原始MIDI在每个物理子步结束时的目标和实际琴键状态，逐子步计算后取算术平均。其他公共奖励保留动作前控制帧目标与周期末状态；`scale_by_dt=false`，不乘 dt，也没有 `reference_dt`。
- `self_collision_reward` 是独立例外：只选同手不同手指的 body 对（排除掌壳及同指内部），先在每个物理子步内去重，再跨子步累计，最终交给控制步总奖励；不取均值、不除以 10。
- 可选 `arm_motion_limit_penalty` 每物理子步根据腕部目标请求求连续 IK 参考，计算限幅前 IK 需求的速度／加速度超限比例，双侧／关节／两类导数／子步取最大，每控制步独立最多扣 5；不与原 `joint_motion_limit_penalty` 合并，不逐子步叠加、不乘 dt。
- 按键奖励会计入周期中途的短误触；跨手奖励仍取周期末，可能漏掉中途接触，200Hz诊断独立记录。奖励计数保留实际采到的同手不同手指子步接触次数；完整同手接触另外保留在诊断中，罚分在乘权重后封顶，不截断诊断计数。
- 策略goal观察、踏板、当前/提前指法使用 `NoteTrajectory.from_midi(..., dt)` 的控制网格；原版Shadow将起音向下取整、结束向上取整，并为连续重复同音保留其原有的松键帧规则。这些控制帧项在第k步与第k帧比较，评分后观察才切到第k+1帧；按键奖励独立使用该步内原始事件。
- 当前按键奖励要求 `exact_note_timing=true`：原谱精确事件同时用于按键子步目标、176维倒计时观察和200Hz按键/F1诊断；音符区间为 `[开始,结束)`。控制帧观察及指法分配不变，原始MIDI不被改写。
- 刚性机械臂模式中，原腕/TCP目标运动项仍约束并计分策略腕部请求通道；仅带臂模式的目标轨迹额外按剩余距离提前制动（限制数值／罚分公式不变），臂SDK硬限作用于实际执行臂轨迹，手根由固定装配得到。实际按键和碰撞只按连接后的手评分。
- `q` 表示**最终下发物理目标**，包括限幅和耦合从动关节；`q_reset` 是实际 reset 下发目标（含硬位置限制修正）。`q` 不等于实际关节位置 `qpos`。
- `x` 是琴键实际归一化行程，0 为松开、1 为到底。`A` 是当前应按键集合；“实际触发”先将实际琴键角度裁剪到该键行程上下限，再判断距上限是否不超过 `0.00872665rad`。
- 容差函数 `T(e;b,m)=exp(ln(0.1) × [max(−e,e−b,0)/m]²)`：误差在 `[0,b]` 内得 1，超出后平滑衰减；超出 b 再走 m，分值为 0.1。

## 开启项：8 项完整表

| 奖励名称 | enabled / weight | 当前全部 params 与实际依赖 | 实际计分、范围与例子 |
|---|---|---|---|
| `key_press_reward` | true / 1 | `mode=paper`；`chord_bonus=false`；`chord_scope=null`。固定正确键混合比例0.5/0.5（不是可切换参数）；b=0.05、margin=0.5；需原谱事件及完整物理子步键程/真实激活记录。 | `mean_s[0.6×R_on(s)+0.4×exp(−1.25×N_wrong(s))]`；`R_on(s)=mean_{k∈A(s)}[0.5×T(1−x_k(s);0.05,0.5)+0.5×I_激活(k,s)]`。无目标键时R_on=0；N_wrong为该子步实际激活的非目标键数。默认范围0～1；每个子步目标键全到位且无错键得1，静音无误触得0.4。先逐子步评分后平均，不对键程/错键数先平均，不乘dt。 |
| `sustain_reward` | true / 1 | `params={}`。任务常数：踏板动作 ≥0.5 为打开；容差 b=0.05、margin=0.5。 | `T(目标踏板状态−实际踏板开关;0.05,0.5)`。正确得 1；错误仍有极小正值（漏踩约 0.000245，误踩约 0.0001），不是负分。即使原谱没有踏板，保持松开也得 1。 |
| `energy_reward` | true / 1 | `params={}`。任务常数 `energy_coefficient=0.005`。 | `−0.005×Σ abs(执行器力或力矩×实际执行器速度)`。对双手当前活动执行器求和：默认无臂含活动腕平移/旋转和手指；刚性臂模式只含40个手指执行器（腕PD已移除，运动学臂无虚构功率）；不含踏板；取周期末快照，非正值，无单独固定扣分上限。绝对功率和为 10 时扣 0.05。不是目标速度惩罚。 |
| `fingering_reward` | true / 1 | `mode=legacy`。任务常数：距离容差 b=0.01m、margin=0.1m；提前指法奖励和分阶段奖励均关闭。 | 按乐谱当前“手指—琴键”分配，对指尖与指定目标点的三维距离 d 求 `T(d;0.01,0.1)`，按分配权重平均。范围 0～1，无当前分配时 0；≤1cm 得 1，11cm 得 0.1。不直接判断键是否按响。 |
| `forearm_reward` | true / 1 | `params={}`。由 `task_profile=tsensei_collision_v1` 决定当前公式，任务接触距离容差 `1e−8m`。 | 左右手全部实体碰撞几何在周期末出现任意跨手接触，整项 −0.5；无跨手接触为 0。多接触点/多部位不累加。名称虽含 forearm，当前并非只检查前臂或掌部；不含同手及手—琴键接触。刚性臂模式对原手根／掌部几何补显式跨手接触对，避免MuJoCo过滤无自由度根导致漏罚；这些根间接触只检测，没有运动学根的动力学反作用。 |
| `self_collision_reward` | true / 0.1 | `aggregation=substep_sum`；`maximum_penalty_per_step=5`（最终奖励单位，正数表示最大扣分幅度）；无臂／带臂均开启。 | `−min(0.1×Σ子步 N(该子步内同手不同手指的body对),5)`。同一子步内同一body对的多个接触点/geom只算一次；一对连续10个子步扣1，两对各10个子步扣2，累计50对次及以上整项扣5。先跨子步求和并乘weight，再整项封顶一次，不乘dt、不除以子步数。只统计同手不同手指的body对；手指—掌壳、同一手指MCP/PIP/DIP/Swing以及同一拇指MCP/PIP/rota_block/rotaback_link2均不计罚。物理接触过滤和父子排除保持，跨手和手—琴键不计入，周期末额外forward不加一次。 |
| `humanoid_finger_gesture_reward` | true / 1 | **当前实现** 是闲置指相对 reset 目标的 L1；`coefficient=−0.05`；`reference=scene_reset`；`lookahead_steps=20`（控制步）。 | `−0.05×Σ闲指关节 abs(q−q_reset)`，单位rad。最多涉及双手40个手指通道，含拇指与耦合PIP/DIP；无腕、平移、踏板。仅当前帧k及未来k+1至k+20均无MIDI乐谱指法分配的手指算闲指；窗口内任一帧用到即豁免，包含第20步，不含第21步。20Hz对应未来1秒，曲尾零填充、不回绕。偏差合计1rad扣0.05；不除以指/关节数，**无旧−0.1封顶**，没有Gaussian或按键成功/几何门控。 |
| `joint_motion_limit_penalty` | true / 1 | `finger_velocity_limit=20rad/s`；`finger_acceleration_limit=400rad/s²`；`wrist_velocity_limit=3rad/s`；`wrist_acceleration_limit=15rad/s²`；`tcp_velocity_limit=0.9m/s`；`tcp_acceleration_limit=1.90m/s²`；`full_penalty_excess=0.1`；`maximum_penalty_per_step=5`。 | 手指/腕旋转对最终目标求 `v=(q−q_prev)/dt`、`a=(q−2q_prev+q_prev2)/dt²`；TCP为左右TX/TY四轴，按限幅前物理需求p求 `v_req=(p−q_prev)/dt`、`a_req=(v_req−(q_prev−q_prev2)/dt)/dt`。各自除对应单位阈值，取所有速度/加速度比例减1与0的最大值e；`−5×min(e/0.1,1)`，范围−5～0，超限5%扣2.5、≥10%扣5，不逐轴/双手/两类导数相加。每控制步一次、不乘dt；TZ、踏板不计。TCP启用须两个阈值均正、快照包含完整限幅前需求；null对关闭TCP组。 |

**总奖励是上述8项相加。** 同手碰撞和目标运动各自最多−5，不共享预算、不封顶总奖励；能耗和闲指罚仍会叠加；独立误触行程罚已关闭。`params={}`表示没有该项的JSON自有参数，不表示没有任务常数或采样依赖。

### key_press_reward 的参数逐项解释

| 字段或常数 | 当前值 | 作用与边界 |
|---|---|---|
| `enabled` | true | 是否将该项加入总奖励。 |
| `weight` | 1 | 对整项按键奖励乘1，不改变内部0.6/0.4比例。 |
| `params.mode` | `paper` | 唯一按键算法：原谱物理子步评分，正确键50%键程＋50%真实触发；删除legacy按键及端点纯键程分支。 |
| `params.chord_bonus` | false | 关闭额外和弦倍增。两个、三个以上目标键仍按正常目标键均值计分。 |
| `params.chord_scope` | null | 未选择和弦加成作用范围；当前不起作用。 |
| 目标时间（固定规则） | 原始MIDI物理子步时刻 `(k×decimation+s+1)×physics_dt` | 起音包含、松键不包含；同一控制步内可切换音符目标，不用下一控制帧代替。 |
| 状态采样（固定规则） | 每个物理子步积分后 | 从实际qpos读取键程及真实阈值激活；10次分别算奖励后取平均。缺少物理记录直接报错，不退回端点评分。同手累计独立。 |
| `key_tolerance` / margin | 0.05 / 0.5 | 写在任务源码/导出manifest的常数，不是JSON params。目标键行程达到95%后正确键连续分为1；55%时约0.2291，50%时约0.1549。 |
| 实际触发阈值 `key_threshold` | 0.00872665rad | 实际琴键角度先裁剪到该键上下限，再比较距最大行程的差值；这是绝对角度阈值，不等于上述归一化行程5%容差，也不是手指目标角度。 |

例如一个应按键：完全压到位且无错键，按键项=1；完全压到位但另有1个错触键，按键项约0.714602（2个约0.632834），不再额外叠加独立误触行程罚。应按键完全没压、但无错键时，连续容差有很小尾值，所以按键项约0.400147。这项不是F1，也不是“错一个键就扣1分”。

### 其余容易混淆的定义

- 指法目标不是随意设置的琴键中心：当前碰撞任务下拇指与其他指共用已保存目标，世界位置 `key_geom_position + (0.35×key_half_size_x,0,0.5×key_half_size_z)`。指法保持原谱分配；没有本次新增指法或改目标点。
- 同手项计的是引擎生成、通过现有碰撞过滤、接触距离≤0且属于同手不同手指的body对。无臂仍使用原生父子／同焊接体过滤；带臂转换在原／新编译模型间比较父子过滤关系，把丢失的关系写为物理body排除并记录`arm_follow_contract.parent_collision_filter`。当前20对自动推导，原显式geom接触对仍保留引擎优先级；物理接触未关闭；2026-10-13起奖励端另排除掌壳及同指内部接触，诊断仍记录它们。Slurm2140验证两模式各5对非相邻接触，对相同物理接触缓冲做10次子步采样当时按旧权重扣37.5（历史验收）；相同50对次在当前权重/封顶下为−5。带臂正常初态为0，修复前expe00002的−90/步只属于旧归档。跨手项用的接触容差另为1e−8m。详情见[同手子步累计](self_collision_substeps_20261008.md)。改物理频率会改变相同接触时长的累计扣分，不是按秒归一的罚率。
- 闲指窗口按训练对应MIDI的已保存指法与Shadow控制网格计算，不猜测用指、不改原谱；持续音在窗口任一帧仍有分配也属于用指。准备段首帧可豁免20步后即将弹首音的手指。奖励的`lookahead_steps=20`独立于策略goal观察的`--lookahead 10`（本次保持），也独立于0.15s提前指法观察；不扩大策略观察维数。旧 L1 配置缺少显式 `lookahead_steps` 时直接拒绝；当前训练只接受活动配置。
- 闲指项参考最终**目标**，当前只维护这一套 L1 公式。旧姿态、几何和任务条件分支已从活动源码删除，历史配置与说明位于 `deprecated/`。
- 目标运动项的首步与实际reset目标比较，初始目标速度按0；全量/局部reset同步清理相应历史。手指/腕旋转使用rad，TCP使用m；各组先除对应阈值再比较，当前 `action_units=null`不影响该项。TCP历史是已下发目标，不是连续两次原始需求，reset初始速度为0。
- 20Hz下，手指20rad/s对应相邻目标差1rad，400rad/s²对应两次目标差的变化1rad；罚满阈值分别22rad/s或440rad/s²（任一达到即可）。腕罚满阈值为3.3rad/s或16.5rad/s²，但最终目标硬限制更低，所以腕单独达到硬上限时约扣2.36，不会单靠腕触发−5。

### TCP限幅前需求的实际计算

与隔离机械臂实验 `arm_motion_limits.py:IKMotionLimiter.apply` 口径一致，仅参考其公式，不导入机械臂源码/资产。对每侧TX、TY逐轴，令p_k为本步策略映射后、任何目标限幅前的物理位置需求，q_{k−1}/q_{k−2}为前两步最终下发目标：

`v_req=(p_k−q_{k−1})/dt`，`v_prev=(q_{k−1}−q_{k−2})/dt`，`a_req=(v_req−v_prev)/dt`。

`e_tcp=max(0, max轴(abs(v_req)/0.9−1), max轴(abs(a_req)/1.90−1))`，与手指和腕旋转的e再取最大值，统一 `r=−5×clip(e/0.1,0,1)`。不额外叠加一个TCP −5预算，也不把米和弧度直接比较。例：其余分量未超限时，需求速度0.945m/s或需求加速度1.995m/s²扣2.5；0.99m/s或2.09m/s²扣满5。需求速度在阈值内但加速度超限仍扣分。速度硬上限0.99恰为软阈值0.9的110%；加速度硬上限1.99约为软阈值1.90的104.7368%，若需求加速度恰为1.99且其他分量未有更大超限，扣约2.3684。需求可以高于硬上限，因此仍可达到2.09并扣满5。

硬限制独立作用于最终下发q：`abs(q_k−q_{k−1})≤0.99dt`、`abs(q_k−2q_{k−1}+q_{k−2})≤1.99dt²`。20Hz下分别为0.0495m、0.004975m；第二个是相邻位移的变化，只有从静止起步时也等于首步最大位移4.975mm。首步需求20cm会扣满5，而实际下发最多4.975mm。投影考虑关节位置边界、刹停距离和float32向内舍入。以上均为目标差分，不是世界坐标实测TCP轨迹或动力学速度/加速度约束。

新活动奖励参数严格要求TCP两个字段；旧含运动项但缺字段的JSON不自动补值，使用对应旧源码归档。其他当前奖励、手指/腕阈值、训练预算、指法与模型不变。

## 关闭项：14 项完整表

令 `u` 来自最终下发目标q：physical时u=q，normalized时按场景保存的完整action_min/action_max计算 `u=2×(q−low)/(high−low)−1`，不是31维原始策略动作。`v=Δu/dt`，`a=Δ²u/dt²`；`P⁺=max(执行器力或力矩×实际执行器速度,0)`。以下每行都是 `enabled=false`，当前贡献为0。乘权重后的公式只在满足启用前提后成立。

| 奖励名称 | 当前weight | 当前全部params | 启用后的公式与意义 |
|---|---:|---|---|
| `false_press_penalty` | **−0.05** | `params={}`。固定行程阈值 0.001，固定成本 `0.01+0.14×x`，x截到[0,1]。 | 当前关闭、贡献0；显式开启后，对每个非目标键：x≤0.001时成本0，否则成本 `0.01+0.14x`；求和乘−0.05。一个误触键半程扣0.004、全程扣0.0075。静音时全部88键都属于非目标键；不必达到实际触发阈值即可扣分。开启时raw日志为非负成本，weighted/final才是负值。 |
| `arm_motion_limit_penalty` | 1 | `soft_limit_ratio=0.95`；`full_penalty_excess=0.1`；`maximum_penalty_per_step=5`。SDK：双侧J1–J7硬速度π rad/s；J1–J2硬加速度2.5π rad/s²，J3–J7为5π rad/s²。软阈值分别0.95π、2.375π、4.75π。 | 需 `--arms` 与双臂显式初始七角；启用时默认true。无参数入口仍为false。每子步h=.005s，IK目标为策略腕部目标经原目标硬限及准备段变换后的完整请求位姿，以前次IK参考为连续种子得到q_ref；以q_ref减前次下发q_cmd求v_req，再减前次下发v_cmd求a_req。`e=max(0,abs(v_req)/v_soft−1,abs(a_req)/a_soft−1)`，跨左右关节／导数／同控制步子步取最大；`−5×min(e/0.1,1)`，不乘dt。超过软阈值5%扣2.5、10%扣5；满罚原始需求为SDK硬限104.5%。整项独立范围[−5,0]，与原运动项合计可达−10；没有IK残差／失败额外罚。腕请求可能与实际装配位姿存在误差，法兰—连接件—手的刚性关系不脱开。 |
| `forearm_power_reward` | 0.2 | `coefficient=−0.001` | `−0.0002×Σ平移组 P⁺`；只罚正机械功率。刚性臂模式移除腕执行器，该组实际功率为0。 |
| `wrist_power_reward` | 0.176 | `coefficient=−0.01` | `−0.00176×Σ腕旋转组 P⁺`。刚性臂模式移除腕执行器，该组实际功率为0。 |
| `finger_power_reward` | 0.2 | `coefficient=−0.005` | `−0.001×Σ手指组 P⁺`。 |
| `forearm_velocity_reward` | 0.2 | `coefficient=−0.02` | `−0.004×Σ平移组 abs(v)`；无速度阈值，非零变化即罚。 |
| `wrist_velocity_reward` | 0.176 | `coefficient=−0.005` | `−0.00088×Σ腕旋转组 abs(v)`；无阈值。 |
| `finger_velocity_reward` | 0.2 | `coefficient=−0.0002` | `−0.00004×Σ手指组 abs(v)`；无阈值。 |
| `forearm_acceleration_reward` | 0.2 | `coefficient=−0.002` | `−0.0004×Σ平移组 abs(a)`；目标二阶差分，无阈值。 |
| `wrist_acceleration_reward` | 0.176 | `coefficient=−0.0003` | `−0.0000528×Σ腕旋转组 abs(a)`；无阈值。 |
| `finger_acceleration_reward` | 0.2 | `coefficient=−0.00001` | `−0.000002×Σ手指组 abs(a)`；无阈值。 |
| `finger_speed_reward` | **0** | `coefficient=−0.05`；`limit=6.5rad/s` | raw=`−0.05×Σ手指 max(abs(v)−6.5,0)`，再乘weight。仅开启enabled仍为0，必须显式设置非零weight；没有−5封顶。 |
| `wrist_pitch_reward` | 1 | `bonus=0.05`；`directions=null` | 每侧最终Pitch目标与指定方向同号时+0.05，两侧最多+0.1；目标为0不奖励。开启前必须显式提供R/L_Wrist_Pitch的方向±1，不能猜填。 |
| `forward_press_reward` | 1 | `coefficient=−0.1`；`threshold=1.5707963267948966rad`（π/2）；`angle_map=null` | 每指等效按压角 `θ=Σ映射系数×该指最终目标+offset`；θ严格大于阈值时每指−0.1，十指最多−1。开启前须为十指完整填写joints系数及offset(rad)，不假定等于某单关节角。 |

正则组名说明：`forearm`在上述三组传统正则中是双手Wrist_TX/TY/TZ平移通道，不是TJ_M6七轴机械臂；`wrist`是Pitch/Yaw旋转；`finger`是40个手指旋转目标。当前TZ/Yaw策略轴结构移除，固定通道差分为0。物理坐标下平移速度/加速度为m/s、m/s²，旋转为rad/s、rad/s²；归一化坐标下为归一化单位/s、/s²。新增机械臂项单独使用七关节SI量及SDK快照。

启用机械臂时，策略腕部请求仍主导IK；实际法兰—四孔连接件—手根是完整刚性链，竞争的腕PD已移除，手指／琴键保持物理。energy／正功率只统计40个手指执行器，不虚构运动学机械臂功率。原腕／TCP目标限制作用于腕请求通道，臂SDK硬限另约束实际执行臂轨迹，不能当作真实手指qvel/qacc保证。几何投影关闭，几何／键底诊断使用连接后的真实手根姿态；手指L1、同手／跨手接触、实际按键和原谱诊断保持各自合同。受限时允许腕请求误差，但刚性装配不脱离；没有新增残差罚。详见[当前实现](optional_arm_wrist_ik_requirements_20261008.md)。

启用前提：6个速度/加速度项需将 `action_units` 从null明确改为 `physical` 或 `normalized`；`finger_speed_reward`强制要求physical。所有上述正则要求 `regularization_shaping=identity`（当前已是）；3个正功率项与已开启的 `energy_reward` 互斥，启用前须关闭后者。禁止根据测试fixture猜腕方向或按压角映射。

## 影响奖励的全局与performance配置

| 字段 | 当前值 | 明确含义 |
|---|---|---|
| `reward_contract` | `physical_key_activation50_v1` | 绑定本次按键算法；旧endpoint奖励合同/配置/checkpoint直接拒绝，不静默迁移。 |
| `weight_stage` | `current` | 当前组合标识；本表以每项显式weight为准，不从其他stage恢复数值。 |
| `scale_by_dt` | false | 不对控制步奖励额外乘dt；没有reference_dt字段。 |
| `action_units` | null | 关闭的目标导数正则还没有选择坐标单位；不影响已开启的物理目标运动项和闲指L1项。 |
| `regularization_shaping` | `identity` | 直接使用已实现公式，不额外做未知整形。 |
| `exact_note_timing` | true（按键奖励必需） | 原谱事件用于子步按键目标，同时保留176维事件观察和物理子步F1；不改策略goal控制网格及指法。 |
| `anticipatory_fingering_time` | 0.15s | 按控制帧指法生成提前观察窗口；闲指豁免独立使用20步窗口，不依赖该开关。 |
| `anticipatory_fingering_reward` | false | 不把未来目标计入指法奖励。 |
| `staged_fingering_reward` | false | 不使用悬停/按压等分阶段指法评分。 |
| `anticipatory_current_priority` | false | 未开启当前用指优先加权；当前提前奖励本身也关闭。 |
| `upcoming_press_permission_time` | 0s | 未开启提前按键许可。 |

准备段为 `policy_v1` 的1s，从首步起正常产生奖励与训练transition。空谱目标时：按键无误触得0.4、踏板正确得1、当前指法0，只有未来20步也无分配的手指才是闲指，能耗/碰撞/闲指/运动罚照常计算，独立误触罚关闭；理想静止且无接触的空谱帧总分为1.4。策略仍有20个准备控制帧；但按键奖励按原谱子步时间判定：若首音起于t=1.00，区间[0.95,1.00]的最后一个物理采样已按首音评分，其余9个子步仍静音。该边界与原谱200Hz诊断一致，其他奖励仍遵循控制帧。曲目休止和0.25s释放尾段同样没有奖励豁免。reset本身不追加一个reward控制步。

没有错键终止或额外终止罚；错键影响上述奖励后继续演奏。曲目完成/显式时间截断的discount=1，SAC另外使用gamma=0.80。F1、碰撞诊断、键底接触、训练停止阈值和SAC熵项不是新的环境reward条目；诊断开启不代表存在对应罚分。

## 硬控制、耦合与奖励的区别

| 控制项 | 当前值 | 与奖励的关系 |
|---|---|---|
| 腕Pitch最终目标硬速度/加速度 | 3.1416rad/s / 15.708rad/s² | 先限制最终下发目标，再按软阈值3/15计算运动罚；在位置边界前减速。目标差分硬限制不保证实际物理qvel/qacc。 |
| 腕平移TX/TY最终目标硬速度/加速度 | 每侧每轴0.99m/s / 1.99m/s² | 最终投影在位置边界前提前刹停；限幅前需求按0.9m/s、1.90m/s²软阈值参与整项目标运动罚。逐轴限值，不是XY合速度上限；不是实际TCP物理速度保证。TZ/Yaw结构移除。 |
| 机器人硬位置限制 | `hard_v1` | 机器人关节位置约束，不是独立扣分项。 |
| 手指耦合 | `--finger-coupling`；范围文件 `angle_ranges_60_80.json` | 双手非拇指PIP/DIP由耦合生成；新运动罚、闲指L1都包含这些最终从动目标。`--no-coupling-reduce-observation`不代表关闭耦合。 |
| 闲指几何投影/主动按压许可 | false / false | 没有隐式开启几何保护；它们是控制开关，不是当前闲指L1的几何评分。 |

当前凸包模型不构造旧胶囊净空诊断，统一入口不设置对应琴键底面诊断停止门槛；与exper002实际部署一致，实体碰撞和运动硬限未因此关闭。

## 使用、维护与历史边界

本次指间碰撞独立快照使用 `shadow_control_frame_distinct_fingers_v1`；活动完整奖励合同以config为准，保留主会话的并行按键奖励变更。旧源码/模型检查点不能静默切换。指间筛选是固定规则，无旧评分模式开关；本轮证据见 `logs/finger_collision_mesh_20261013/`。

- 唯一默认入口是 `bash all_use_this_train.sh`；额外参数写末尾，启动时打印JSON和CLI合并后的完整奖励。
- 单项一般用 `--enable-reward NAME` / `--disable-reward NAME`；启用前需满足上述依赖。大部分weight和params修改对应JSON字段，不存在通用的 `--weight` 开关。
- 目标运动可用 `--finger-target-velocity-limit`、`--finger-target-acceleration-limit`、`--wrist-target-velocity-limit`、`--wrist-target-acceleration-limit`、`--tcp-target-velocity-limit`、`--tcp-target-acceleration-limit`、`--joint-motion-full-penalty-excess`、`--joint-motion-max-penalty` 覆盖；`--no-joint-motion-limit-penalty`只关闭罚分，保留腕硬控制。默认脚本已显式给出的CLI会覆盖JSON中同字段。
- `collision_reward`、`natural_posture_reward` 已从当前模板和实现删除，不属于22项；含它们的旧JSON即便enabled=false也会被新源码拒绝。
- 用户新约定：以后某奖励算法发生替换，在当前源码/配置/CLI/校验/测试和本表中删除被替代模式，直接维护新实现，不再为了兼容堆叠多个模式。仅改数值就更新参数，不新建mode。历史配置和产物集中在 `deprecated/`，当前入口不做旧 checkpoint 兼容转换。
- 本次同手项仅保留带显式`maximum_penalty_per_step`的子步累计公式；旧周期并集评分分支与不封顶配置不再支持，旧归档/实验不改。raw日志为负的合资格同手不同手指接触对—子步数，weighted/final日志为乘weight并封顶后的实际分值。`maximum_penalty_per_step`必须有限且非负；0表示该项罚分上限为0。
- 父版本直接替换L1项的“仅当前帧闲指”判定，显式未来20步、系数−0.1，不增加可切换的当前帧旧分支；其余奖励及训练参数保持。
- 2026-10-13按用户要求以Job1724按键公式替换端点算法。四个旧采样CLI/字段仍删除，50%激活及子步平均固定在唯一实现中；其余奖励保持当前值，不声称完整复现旧训练环境。验证记录见 `logs/key_substep_restore_20261013/`，本轮未启动训练。

本次同手封顶CPU／归档／入口验收证据：`logs/self_collision_cap_20261010/`，该历史验收权重0.2、整项最多−5；当前权重以本页总表为准。未启动训练；父接触GPU验收不冒充本次新评分GPU验证。

父版本参数CPU审计证据：`logs/tcp_thresholds_20261008/`；父算法证据：`logs/tcp_motion_limits_20261008/`（父闲指证据保留在`logs/idle_lookahead20_20261008/`）（父时序修复证据保留在`logs/shadow_score_sync_20261008/`）；包括归档白名单差异、独立解包测试、统一入口真实参数经sbatch/python stub捕获和新归档parser校验。CPU通过不代表GPU/CUDA graph或长期收敛验收；本次未启动训练。

原版依据：[Shadow任务](https://github.com/google-research/robopianist/blob/main/robopianist/suite/tasks/piano_with_shadow_hands.py)的 `_goal_current` 与 `_compute_key_press_reward`、[NoteTrajectory](https://github.com/google-research/robopianist/blob/main/robopianist/music/midi_file.py)及[piano_roll](https://github.com/google-research/robopianist/blob/main/robopianist/music/piano_roll.py)。同步的是目标与奖励时序，A3.2模型、额外奖励、硬限制与精确诊断仍是本项目配置。

实现依据：`mjlab_env/reward_config.py`（字段/启用约束）、`rewards.py`（权重/公式）、`task_logic.py`（五项基础奖励）、`task.py`（物理量端点快照/独立原谱诊断）、`self_collision.py`（接触计数）、`exact_score.py`（精确谱时）、`suite/tasks/piano_with_tsensei_hands.py`（任务常数）、`models/piano/piano.py`（触发阈值）、`fingering_targets.py`（指法目标点）、`rl_torch/train.py`（最终参数解析），均位于项目内 `tsensei_v3_2_pianist/`。
