# 当前奖励与惩罚总表（持续维护）

最后核对：2026-10-26。本文件是当前奖励的唯一维护总表，沿用已有路径；以后直接更新本文对应行，不另建互相矛盾的默认表。旧实验和旧源码文档仅作为隔离记录，当前训练不读取。

依据根目录 [`all_use_this_train.sh`](../all_use_this_train.sh)（当前源码归档与SHA由入口固定）、活动 [`configs/rewards/default_20261006.json`](../configs/rewards/default_20261006.json)，以及本轮显式选择的匹配源码归档。SHA256以统一脚本的强校验值、末尾覆盖和同目录归档审计为准；工作区新项须验证并进入本轮归档后才用于训练，不改写历史归档。表内是**不追加其他参数时的最终配置**；具体实验仍以末尾覆盖参数和保存的完整 `config.json` 为准。当前合同为 `physical_key_fingering_exact_activation100_v1`：按键保持exper001连续键程公式，指法改为原谱时间、实际指尖/琴键几何的逐物理子步Gaussian距离均值。当前默认土耳其进行曲，两项均按1000Hz采样，20Hz控制与学习预算不变；旧指法端点/提前/分阶段算法及其参数已移除，旧合同与checkpoint拒绝。

2026-10-20确定并沿用的入口覆盖：统一入口TX/TY每轴硬速度 **1.99m/s**、硬加速度 **1.99m/s²**；运动罚TCP软速度 **1.90m/s**、软加速度 **1.90m/s²**。仅速度阈值提高，加速度不变。学习器TF32，采样/评估IEEE FP32；历史exper025使用批准的015-5谱和指法、20/200Hz。历史配置不回写；2026-10-22仅修复代码并验收，未启动训练。

当前工作区共 **24 项：默认无臂时 9 项开启、15 项关闭**。显式 `--arms` 且初始角齐全时，默认额外启用独立 `arm_motion_limit_penalty` 与 `arm_tracking_error_penalty`，成为 11 开、13 关；显式 `--disable-reward NAME` 可分别覆盖。2026-10-18 用户最新选择以零误差为目标：1mm以内也连续扣分，超过1mm继续加重并记录超限次数；1mm仅是诊断计数阈值，不是免费区、硬停止或轨迹投影阈值。下面的分值已考虑当前 weight；关闭项贡献为 0，表中的公式说明显式启用后的行为。通用模板值不是本表默认值。新按键横移项在通用模板关闭、在唯一默认奖励JSON开启；缺少该项的旧JSON由完整项集合校验明确拒绝，不自动填入或迁移旧checkpoint。

历史实验 **exper025（用户015-5谱及原配指法）** 使用20Hz控制／200Hz物理，每步10子步；采用上述新TCP软速度阈值，其余奖励保持。沿用用户批准的47处提前松键版本，物理CUDA Graph关闭，学习CUDA Graph以TF32捕获。完整奖励与配置保存至 `results/exper025/`。此前exper024继续保持原硬速度0.99m/s、软速度0.90m/s及FP32，其历史config不回写。

## 计分单位与符号

- 积分器 Euler（2026-10-14用户要求）；2026-10-24统一默认改为当前A3.2少拇指版本土耳其进行曲，沿用2026-10-20用户批准的该曲1000Hz物理频率，以覆盖原谱短重复音松键间隙。控制20Hz、`dt=0.05s`，每控制步50子步。按键/指法取50子步均值；跨手每接触子步−0.01、整步最多−0.5；同手仍每对每子步−0.1累计且整步最多−5（相同时长接触更容易封顶）。其余开关/权重/参数和学习预算不变。历史exper023及其他实验保留各自冻结config；本次未提交训练。
- 按键和指法奖励使用原始MIDI在每个物理子步结束时的目标和实际物理状态，逐子步计算后取算术平均。踏板、能耗与其他未改动项保持各自原采样合同；`scale_by_dt=false`，不乘 dt，也没有 `reference_dt`。
- `self_collision_reward` 是独立例外：只选同手不同手指的 body 对（排除掌壳及同指内部），先在每个物理子步内去重，再跨子步累计，最终交给控制步总奖励；不取均值、不除以子步数（当前默认50）。
- 可选 `arm_motion_limit_penalty` 每物理子步根据腕部目标请求求连续 IK 参考，计算限幅前 IK 需求的速度／加速度超限比例，双侧／关节／两类导数／子步取最大，每控制步独立最多扣 5；不与原 `joint_motion_limit_penalty` 合并，不逐子步叠加、不乘 dt。
- 可选 `arm_tracking_error_penalty` 对每个物理子步的**原始策略掌根请求（含同一时刻准备段、TCP/IK限幅前）**与刚性装配实际 `palm_base` 世界位置求三维距离；双手／整个控制步子步取最大。以1mm为尺度，1mm内平方扣分，超过1mm连续改为对数扣分，只有零误差扣0；无硬封顶，不乘dt。大误差仍可区分且增长较缓，避免初期随机请求产生数百分惩罚。IK输入目标误差另外记录，不能用限幅后目标替代原始请求把滞后隐藏。
- 按键奖励会计入周期中途的短误触；跨手奖励现在也按每个1000Hz物理子步采样，50个子步内每个子步出现任意跨手接触计一次，控制步总分为 `−0.5×接触子步数/50`，因此单个接触子步扣−0.01，50个子步均接触时扣−0.5。周期末额外forward不重复计入。奖励计数保留实际采到的同手不同手指及跨手子步接触次数；完整接触另外保留在诊断中，罚分在各自公式中封顶。
- 策略goal窗口按精确事件在控制时刻 `k×0.05s` 点采样：第一块为当前，后续块为未来时刻；当前指法仅表示该时刻正在发声的用指。未来指法另以50维字段给出每指最近一次未来起音的相对悬停位置、秒数、有效标记；即使该指正在弹旧音，也能预告下一音，绝不并入当前指法奖励。另保留176维逐键起音/松键倒计时以显示控制帧之间的短音符。
- `exact_note_timing=true`为当前训练合同必需；音符区间为 `[开始,结束)`，物理采样时刻为 `(k×50+s+1)×0.001s`。原谱音高、指法、时值不改，仅沿用既有开头静音裁剪/准备段偏移。连续时间真实同指重叠仍拒绝；只由floor/ceil造成的同帧冲突不进入评分。原控制网格仍保留作踏板、传统控制帧F1诊断及闲指20步保护用途，不代表当前指法评分目标。
- 刚性机械臂模式中，原腕/TCP目标运动项仍约束并计分策略腕部请求通道；臂SDK硬限作用于实际执行臂轨迹，当前步只做速度、加速度、位置和耦合硬限，不按剩余距离提前制动，手根由固定装配得到。实际按键和碰撞只按连接后的手评分。
- `q` 表示**最终下发物理目标**，包括限幅和耦合从动关节；`q_reset` 是实际 reset 下发目标（含硬位置限制修正）。`q` 不等于实际关节位置 `qpos`。
- `x` 是琴键实际归一化行程，0 为松开、1 为到底。`A` 是当前应按键集合；“实际触发”先将实际琴键角度裁剪到该键行程上下限，再除以该键最大角行程，实际角度直接以 `qpos≥qmax` 判定激活（100%角行程）；白黑键统一。正确键连续奖励仍在95%行程满分，二者不同。
- 容差函数 `T(e;b,m)=exp(ln(0.1) × [max(−e,e−b,0)/m]²)`：误差在 `[0,b]` 内得 1，超出后平滑衰减；超出 b 再走 m，分值为 0.1。

## 开启项：8 项完整表

| 奖励名称 | enabled / weight | 当前全部 params 与实际依赖 | 实际计分、范围与例子 |
|---|---|---|---|
| `key_press_reward` | true / 1 | `mode=paper`；`chord_bonus=false`；`chord_scope=null`。正确键仅连续键程评分；b=0.05、margin=0.5；需原谱事件及完整物理子步键程/真实激活记录。 | `mean_s[0.6×R_on(s)+0.4×exp(−1.25×N_wrong(s))]`；`R_on(s)=mean_{k∈A(s)}T(1−x_k(s);0.05,0.5)`。无目标键时R_on=0；N_wrong为该子步实际激活的非目标键数。默认范围0～1；每个子步目标键全到位且无错键得1，静音无误触得0.4。先逐子步评分后平均，不对键程/错键数先平均，不乘dt。 |
| `sustain_reward` | true / 1 | `params={}`。任务常数：踏板动作 ≥0.5 为打开；容差 b=0.05、margin=0.5。 | `T(目标踏板状态−实际踏板开关;0.05,0.5)`。正确得 1；错误仍有极小正值（漏踩约 0.000245，误踩约 0.0001），不是负分。即使原谱没有踏板，保持松开也得 1。 |
| `energy_reward` | true / 1 | `params={}`。任务常数 `energy_coefficient=0.005`。 | `−0.005×Σ abs(执行器力或力矩×实际执行器速度)`。对双手当前活动执行器求和：默认无臂含活动腕平移/旋转和手指；刚性臂模式只含40个手指执行器（腕PD已移除，运动学臂无虚构功率）；不含踏板；取周期末快照，非正值，无单独固定扣分上限。绝对功率和为 10 时扣 0.05。不是目标速度惩罚。 |
| `fingering_reward` | true / 1 | `params={}`（无mode）。指尖任务点为URDF相邻铰链方向推导的末端凸包支撑点（`urdf_chain_distal_support_v1`）沿末节纵向朝掌根回退0.003m（方向A，`FINGERTIP_SETBACK_M=0.003`），两手十指一致；奖励和指尖位置观察均使用此点，随末节旋转，非世界固定轴偏移、非实测指腹中心。耦合仍用原支撑点，碰撞网格不变；当前几何来自2026-10-20用户双侧URDF，各自生成凸包与定位点，不假设左右镜像。目标点：十指统一白键距前端40mm、黑键距90mm上表面前端20mm（世界+X端往−X量取），Y居中，Z保持原键体中心上方半高的0.5倍。2026-10-23黑键实体为顶面10×90/底面12×95/高18mm、后端齐平；目标和逐物理子步采样读取原键体中心的reference site（白键半尺寸75×11.25×11.25mm、黑键45×5×9mm），不读取mesh主轴中心或包围盒；公式、目标数值与奖励参数保持。任务常数：距离容差 b=0.01m、margin=0.1m；需完整的原谱事件和每物理子步积分后指尖/键目标几何；旧提前/分阶段奖励及参数已删除。 | `mean_s[mean_{f∈F(s)} T(‖tip_f(s)−target_{key(f,s)}(s)‖;0.01,0.1)]`。F(s)仅含该物理时刻原谱正在用的手指，每指最多一个键；先对该子步用指平均，再对50子步平均，空子步贡献0。不先平均位置、不补未来奖励、不乘dt。范围0～1；≤1cm得1，11cm得0.1，不直接判断键是否按响。 |
| `forearm_reward` | true / 1 | `params={}`。由 `task_profile=tsensei_collision_v1` 决定当前公式，任务接触距离容差 `1e−8m`；当前控制/物理频率为20/1000Hz，即每控制步50个子步。 | 左右手全部实体碰撞几何在每个1000Hz物理子步采样；某子步出现任意跨手接触计1次，整项为 `−0.5×接触子步数/50`，所以每个接触子步扣−0.01，五十个子步均接触时扣−0.5，无接触为0。每个子步内不按接触点或多部位重复累加，周期末额外forward不重复计入。带刚性臂时，独立 `rigid_arm_detection_only_v1` 查询模型的`arm_arm`、`arm_hand`、`arm_piano`也按同一物理子步并入“该子步有冲突”计数；Link7/法兰/连接件到同侧手的刚性安装界面排除。主力求解器不启用臂网格碰撞，查询只检测、不产生运动学反作用。名称虽含 forearm，仍不含同手及手—琴键接触。 |
| `self_collision_reward` | true / 0.1 | `aggregation=substep_sum`；`maximum_penalty_per_step=5`（最终奖励单位，正数表示最大扣分幅度）；无臂／带臂均开启。 | `−min(0.1×Σ子步 N(该子步内同手不同手指的body对),5)`。同一子步内同一body对的多个接触点/geom只算一次；一对连续10个子步扣1，一对连续50个子步即达到整项−5封顶；累计50对次及以上整项扣5。先跨子步求和并乘weight，再整项封顶一次，不乘dt、不除以子步数。只统计同手不同手指的body对；手指—掌壳、同一手指MCP/PIP/DIP/Swing以及同一拇指MCP/PIP/rota_block/rotaback_link2均不计罚。物理接触过滤和父子排除保持，跨手和手—琴键不计入，周期末额外forward不加一次。 |
| `humanoid_finger_gesture_reward` | true / 1 | **当前实现** 是闲置指相对 reset 目标的 L1；`coefficient=−0.008`；`reference=scene_reset`；`swing_scale=0.5`（仅八指SW，其他关节1）；`lookahead_steps=20`（控制步）。 | `−0.008×Σ闲指关节 w_j×abs(q−q_reset)`，单位rad。最多涉及双手40个手指通道，含拇指与耦合PIP/DIP；无腕、平移、踏板。仅当前帧k及未来k+1至k+20均无MIDI乐谱指法分配的手指算闲指；窗口内任一帧用到即豁免，包含第20步，不含第21步。20Hz对应未来1秒，曲尾零填充、不回绕。八指SW偏差1rad扣0.004，其余关节偏差1rad扣0.008；不除以指/关节数，**无旧−0.1封顶**，没有Gaussian或按键成功/几何门控。 |
| `joint_motion_limit_penalty` | true / 1 | `finger_velocity_limit=20rad/s`；`finger_acceleration_limit=400rad/s²`；`wrist_velocity_limit=3rad/s`；`wrist_acceleration_limit=15rad/s²`；`tcp_velocity_limit=1.90m/s`；`tcp_acceleration_limit=1.90m/s²`；`full_penalty_excess=0.1`；`maximum_penalty_per_step=5`。 | 手指/腕旋转对最终目标求 `v=(q−q_prev)/dt`、`a=(q−2q_prev+q_prev2)/dt²`；TCP为左右TX/TY四轴，按限幅前物理需求p求 `v_req=(p−q_prev)/dt`、`a_req=(v_req−(q_prev−q_prev2)/dt)/dt`。各自除对应单位阈值，取所有速度/加速度比例减1与0的最大值e；`−5×min(e/0.1,1)`，范围−5～0，超限5%扣2.5、≥10%扣5，不逐轴/双手/两类导数相加。每控制步一次、不乘dt；TZ、踏板不计。TCP启用须两个阈值均正、快照包含完整限幅前需求；null对关闭TCP组。 |

| `key_hold_lateral_motion_penalty` | true / 1 | `velocity_tolerance_m_s=0.01m/s`；`full_penalty_velocity_m_s=0.10m/s`；`maximum_penalty_per_step=0.5`。当前20/1000Hz取50子步。门控为该侧实体碰撞几何与实际100%激活琴键接触（dist≤0）；正确/错误键均计入，主物理接触，不读乐谱/目标动作。速度为实际base_link原点沿固定钢琴/世界Y轴，非质心；无臂由祖先Jacobian×实际qvel计算，刚性臂mocap用同子步实际掌根位置差/物理dt，每控制步起点重新取实际位置，reset不产生虚假跨回合速度。 | 每侧每子步 `e=I×clip((abs(v)−0.01)/0.09,0,1)^2`；`r=−0.5×mean_s(max(e_R,e_L))`。无按住接触或速度≤0.01m/s为0；全步0.055m/s扣0.125，≥0.10m/s扣0.5；半数子步触发减半。多接触点/多键去重、双手不相加、不乘dt，200Hz或1000Hz同持续比例同分。weight后仍受整项cap限制；无额外限速/投影/接触力阈值；不罚世界X/Z、指尖局部滑动或手指旋转。无硬限位刷新时增加积分后主模型碰撞查询，不再求解/积分；期末forward不重复计入。初值非硬件标定。 |

**总奖励是上述9项相加。** 新增按键横移项独立最多−0.5，不与跨手碰撞或目标运动罚共享预算。 同手碰撞和目标运动各自最多−5，不共享预算、不封顶总奖励；能耗和闲指罚仍会叠加；独立误触行程罚已关闭。`params={}`表示没有该项的JSON自有参数，不表示没有任务常数或采样依赖。

### key_press_reward 的参数逐项解释

| 字段或常数 | 当前值 | 作用与边界 |
|---|---|---|
| `enabled` | true | 是否将该项加入总奖励。 |
| `weight` | 1 | 对整项按键奖励乘1，不改变内部0.6/0.4比例。 |
| `params.mode` | `paper` | 唯一按键算法：原谱物理子步评分，正确键连续Gaussian键程；实际激活用于错键计数/F1，删除50%激活混合分支。 |
| `params.chord_bonus` | false | 关闭额外和弦倍增。两个、三个以上目标键仍按正常目标键均值计分。 |
| `params.chord_scope` | null | 未选择和弦加成作用范围；当前不起作用。 |
| 目标时间（固定规则） | 原始MIDI物理子步时刻 `(k×decimation+s+1)×physics_dt` | 起音包含、松键不包含；同一控制步内可切换音符目标，不用下一控制帧代替。 |
| 状态采样（固定规则） | 每个物理子步积分后 | 从实际qpos读取键程及真实阈值激活；10次分别算奖励后取平均。缺少物理记录直接报错，不退回端点评分。同手累计独立。 |
| `key_tolerance` / margin | 0.05 / 0.5 | 写在任务源码/导出manifest的常数，不是JSON params。目标键行程达到95%后正确键连续分为1；55%时约0.2291，50%时约0.1549。 |
| 实际触发阈值 `key_activation_fraction` | 1.0（无量纲） | `clip(实际qpos,0,qmax)≥qmax`激活，直接比较物理角度，不引入额外容差。白键/黑键均100%角行程（默认标称12/10mm，2026-10-23更新；用户授权对照exper039单独为10/8mm，exper040仍12/10mm）；不使用指尖位置或目标角度。错键计数、F1、声音、高亮及对比图同步。`KEY_PRESS_TOLERANCE=0.05`保持，正确键连续分仍在95%满分；95%～不足100%时连续分可满分但未实际激活。旧95%与固定0.5°激活合同明确拒绝。 |

例如一个应按键：完全压到位且无错键，按键项=1；完全压到位但另有1个错触键，按键项约0.714602（2个约0.632834），不再额外叠加独立误触行程罚。应按键完全没压、但无错键时，连续容差有很小尾值，所以按键项约0.400147。这项不是F1，也不是“错一个键就扣1分”。

2026-10-23显式实验覆盖：exper041/042均沿用exper038最终完整奖励配置（闲指coefficient=−0.008、swing_scale=0.5、TCP软阈值1.90/1.90），只修订指法与height=.080。exper041固定Yaw15°；exper042双侧Yaw0°～15°、初值0°，原有腕旋转目标速度3rad/s、加速度15rad/s²及共享−5封顶同样作用于新增Yaw动作；未额外添加Yaw硬速度/加速度限制。其余启用、weight、params、公式和采样均不变，默认入口仍固定Yaw。

### 其余容易混淆的定义

- 指法目标采用实时琴键几何中心加世界轴偏移 `(key_half_size_x−distance, 0, 0.5×key_half_size_z)`，白键distance=0.040m、黑键=0.020m，十指一致。前端定义为中心的世界+X半长度端；保持现有世界轴偏移合同，不随琴键转动旋转偏移。当前未按下白键X=0.035m、黑键X=−0.005m；高度、距离容差、原谱指法保持。旧几何manifest和拇指专属选项明确拒绝；新训练须从头开始。
- 同手项计的是引擎生成、通过现有碰撞过滤、接触距离≤0且属于同手不同手指的body对。无臂仍使用原生父子／同焊接体过滤；带臂转换在原／新编译模型间比较父子过滤关系，把丢失的关系写为物理body排除并记录`arm_follow_contract.parent_collision_filter`。当前20对自动推导，原显式geom接触对仍保留引擎优先级；物理接触未关闭；2026-10-13起奖励端另排除掌壳及同指内部接触，诊断仍记录它们。Slurm2140验证两模式各5对非相邻接触，对相同物理接触缓冲做10次子步采样当时按旧权重扣37.5（历史验收）；相同50对次在当前权重/封顶下为−5。带臂正常初态为0，修复前expe00002的−90/步只属于旧归档。跨手项用的接触容差另为1e−8m。详情见[同手子步累计](self_collision_substeps_20261008.md)。改物理频率会改变相同接触时长的累计扣分，不是按秒归一的罚率。
- 闲指窗口按训练对应MIDI的已保存指法与Shadow控制网格计算，不猜测用指、不改原谱；持续音在窗口任一帧仍有分配也属于用指。准备段首帧可豁免20步后即将弹首音的手指。奖励的`lookahead_steps=20`独立于策略goal观察的`--lookahead 10`（本次保持），也独立于0.15s提前指法观察；不扩大策略观察维数。旧 L1 配置缺少显式 `lookahead_steps` 时直接拒绝；当前训练只接受活动配置。
- 闲指项参考最终**目标**，当前只维护这一套 L1 公式。旧姿态、几何和任务条件分支已从活动源码删除，历史配置与说明位于 `deprecated/`。
- 目标运动项的首步与实际reset目标比较，初始目标速度按0；全量/局部reset同步清理相应历史。手指/腕旋转使用rad，TCP使用m；各组先除对应阈值再比较，当前 `action_units=null`不影响该项。TCP历史是已下发目标，不是连续两次原始需求，reset初始速度为0。
- 20Hz下，手指20rad/s对应相邻目标差1rad，400rad/s²对应两次目标差的变化1rad；罚满阈值分别22rad/s或440rad/s²（任一达到即可）。腕罚满阈值为3.3rad/s或16.5rad/s²，但最终目标硬限制更低，所以腕单独达到硬上限时约扣2.36，不会单靠腕触发−5。

### TCP限幅前需求的实际计算

与隔离机械臂实验 `arm_motion_limits.py:IKMotionLimiter.apply` 口径一致，仅参考其公式，不导入机械臂源码/资产。对每侧TX、TY逐轴，令p_k为本步策略映射后、任何目标限幅前的物理位置需求，q_{k−1}/q_{k−2}为前两步最终下发目标：

`v_req=(p_k−q_{k−1})/dt`，`v_prev=(q_{k−1}−q_{k−2})/dt`，`a_req=(v_req−v_prev)/dt`。

`e_tcp=max(0, max轴(abs(v_req)/1.90−1), max轴(abs(a_req)/1.90−1))`，与手指和腕旋转的e再取最大值，统一 `r=−5×clip(e/0.1,0,1)`。不额外叠加一个TCP −5预算，也不把米和弧度直接比较。例：其余分量未超限时，需求速度1.995m/s或需求加速度1.995m/s²扣2.5；2.09m/s或2.09m/s²扣满5。需求速度在阈值内但加速度超限仍扣分。速度硬上限1.99与加速度硬上限1.99均约为各自软阈值1.90的104.7368%，若需求速度或加速度数值恰为1.99且其他分量未有更大超限，扣约2.3684。需求可以高于硬上限，因此仍可达到2.09并扣满5。

硬限制独立作用于最终下发q：`abs(q_k−q_{k−1})≤1.99dt`、`abs(q_k−2q_{k−1}+q_{k−2})≤1.99dt²`。20Hz下分别为0.0995m、0.004975m；第二个是相邻位移的变化，只有从静止起步时也等于首步最大位移4.975mm。首步需求20cm会扣满5，而实际下发最多4.975mm。投影考虑关节位置边界、刹停距离和float32向内舍入。以上均为目标差分，不是世界坐标实测TCP轨迹或动力学速度/加速度约束。

新活动奖励参数严格要求TCP两个字段；旧含运动项但缺字段的JSON不自动补值，使用对应旧源码归档。其他当前奖励、手指/腕阈值、训练预算、指法与模型不变。

## 默认无臂关闭项：15 项完整表

令 `u` 来自最终下发目标q：physical时u=q，normalized时按场景保存的完整action_min/action_max计算 `u=2×(q−low)/(high−low)−1`，不是31维原始策略动作。`v=Δu/dt`，`a=Δ²u/dt²`；`P⁺=max(执行器力或力矩×实际执行器速度,0)`。以下每行都是 `enabled=false`，当前贡献为0。乘权重后的公式只在满足启用前提后成立。

| 奖励名称 | 当前weight | 当前全部params | 启用后的公式与意义 |
|---|---:|---|---|
| `false_press_penalty` | **−0.05** | `params={}`。固定行程阈值 0.001，固定成本 `0.01+0.14×x`，x截到[0,1]。 | 当前关闭、贡献0；显式开启后，对每个非目标键：x≤0.001时成本0，否则成本 `0.01+0.14x`；求和乘−0.05。一个误触键半程扣0.004、全程扣0.0075。静音时全部88键都属于非目标键；不必达到实际触发阈值即可扣分。开启时raw日志为非负成本，weighted/final才是负值。 |
| `arm_motion_limit_penalty` | 1 | `soft_limit_ratio=0.95`；`full_penalty_excess=0.1`；`maximum_penalty_per_step=5`。SDK：双侧J1–J7硬速度π rad/s；J1–J2硬加速度2.5π rad/s²，J3–J7为5π rad/s²。软阈值分别0.95π、2.375π、4.75π。 | 需 `--arms` 与双臂显式初始七角；启用时默认true。无参数训练入口仍为false；MJLab无参手部预览也显式关闭该项，双臂配置预览沿用保存的enabled。每子步h=physics_dt（当前默认.001s），IK目标为策略腕部目标经原目标硬限及准备段变换后的完整请求位姿，以前次IK参考为连续种子得到q_ref；以q_ref减前次下发q_cmd求v_req，再减前次下发v_cmd求a_req。`e=max(0,abs(v_req)/v_soft−1,abs(a_req)/a_soft−1)`，跨左右关节／导数／同控制步子步取最大；`−5×min(e/0.1,1)`，不乘dt。超过软阈值5%扣2.5、10%扣5；满罚原始需求为SDK硬限104.5%。整项独立范围[−5,0]，与原运动项合计可达−10；本项不含IK残差，跟随误差由下行独立项计分。腕请求可能与实际装配位姿存在误差，法兰—连接件—手的刚性关系不脱开。 |
| `arm_tracking_error_penalty` | 1 | 唯一参数 `error_scale_m=0.001m`。默认无臂false，带臂默认true；显式关闭仅停罚分，保留误差诊断。 | `e=max_{双侧,子步} ‖p_raw(s)−p_actual(s)‖₂`，令 `u=e/0.001`；`u≤1` 时 `r=−u²`，`u>1` 时 `r=−(1+2ln(u))`，再乘weight。0mm扣0，0.1mm扣0.01，0.5mm扣0.25，1mm扣1，2mm约扣2.3863，20mm约扣6.9915，200mm约扣11.5966。1mm连接处值及一阶导数连续，误差增加罚分幅度严格增加；无免费区、无硬封顶。旧threshold_m、maximum_penalty_per_step及mode参数直接拒绝，不迁移旧checkpoint。每控制步一次，不对双手/子步求和或平均，不乘dt。raw请求是策略映射后的原始世界掌根位移加安装参考和同刻准备段，尚未经过TCP/IK目标限幅；actual读取物理子步结束forward后的刚性手根body世界坐标，不用IK参考或独立手部PD目标。关节硬限、加速度与耦合全部保持；奖励不会保证误差≤1mm。 |
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
| `wrist_pitch_reward` | 1 | `bonus=0.05`；`directions=null` | 每侧最终Pitch目标与指定方向同号时+0.05，两侧最多+0.1；目标为0不奖励。开启前必须显式提供R/L_Palm_Base_Pitch的方向±1，不能猜填。 |
| `forward_press_reward` | 1 | `coefficient=−0.1`；`threshold=1.5707963267948966rad`（π/2）；`angle_map=null` | 每指等效按压角 `θ=Σ映射系数×该指最终目标+offset`；θ严格大于阈值时每指−0.1，十指最多−1。开启前须为十指完整填写joints系数及offset(rad)，不假定等于某单关节角。 |

正则组名说明：`forearm`在上述三组传统正则中是双手Palm_Base_Tx/TY/TZ平移通道，不是TJ_M6七轴机械臂；`wrist`是Pitch/Yaw旋转；`finger`是40个手指旋转目标。默认无臂的TZ/Yaw策略轴结构移除；显式带臂 `wuji_cartesian_v1` 则保留XYZ位移、固定Pitch/Yaw，固定通道差分为0。物理坐标下平移速度/加速度为m/s、m/s²，旋转为rad/s、rad/s²；归一化坐标下为归一化单位/s、/s²。新增机械臂项单独使用七关节SI量及SDK快照。

启用机械臂时，策略腕部请求仍主导IK；实际法兰—四孔连接件—手根是完整刚性链，竞争的腕PD已移除，手指／琴键保持物理。energy／正功率只统计40个手指执行器，不虚构运动学机械臂功率。原腕／TCP目标限制作用于腕请求通道，臂SDK硬限另约束实际执行臂轨迹，不能当作真实手指qvel/qacc保证。几何投影关闭，几何／键底诊断使用连接后的真实手根姿态；手指L1、同手／跨手接触、实际按键和原谱诊断保持各自合同。受限时允许腕请求误差，但刚性装配不脱离；新增掌根跟随误差项只改变奖励及诊断，不改变目标、装配或任何关节硬约束。详见[当前实现](optional_arm_wrist_ik_requirements_20261008.md)。

掌根跟随诊断固定以 **1mm 三维欧氏距离** 判定：`arm_tracking_requested_error_m` 是每控制步双手/全部子步峰值，`arm_tracking_ik_target_error_m` 是同样采样的IK输入目标峰值，两者不可混报。`over_1mm_substeps` 为左右任一超限的物理子步数，每个环境每个子步最多1次；`over_1mm_side_substeps` 分别保存右/左超限次数。`over_1mm_entry_events` 只统计从双侧均≤1mm进入任一侧>1mm的事件，持续超限跨控制步不重复计事件，局部reset只清理被reset环境的连续状态。`over_1mm_control_steps` 计存在任何超限子步的环境控制步数。训练 `metrics.jsonl`、最终 `summary.json` 和评估 `arm_tracking` 对各自全部已采集transition聚合峰值、上述计数与真实样本数；`within_1mm_substep_fraction=1−超限世界子步数/已采样世界子步数`，不是控制帧通过比例或“至少一只手通过”的比例。无臂该诊断为null，评估最后一批未请求的多余环境不纳入分母。进入事件依赖相邻实际采样，只代表采样到的越界次数，不保证连续时间没有更短的越界。

启用前提：6个速度/加速度项需将 `action_units` 从null明确改为 `physical` 或 `normalized`；`finger_speed_reward`强制要求physical。所有上述正则要求 `regularization_shaping=identity`（当前已是）；3个正功率项与已开启的 `energy_reward` 互斥，启用前须关闭后者。禁止根据测试fixture猜腕方向或按压角映射。

## 影响奖励的全局与performance配置

| 字段 | 当前值 | 明确含义 |
|---|---|---|
| `reward_config.contract` | `physical_key_fingering_exact_activation100_v1` | 绑定连续键程＋原谱物理子步指法＋统一100%激活；旧激活及按键/指法endpoint合同和checkpoint直接拒绝，不静默迁移。 |
| `weight_stage` | `current` | 当前组合标识；本表以每项显式weight为准，不从其他stage恢复数值。 |
| `scale_by_dt` | false | 不对控制步奖励额外乘dt；没有reference_dt字段。 |
| `action_units` | null | 关闭的目标导数正则还没有选择坐标单位；不影响已开启的物理目标运动项和闲指L1项。 |
| `regularization_shaping` | `identity` | 直接使用已实现公式，不额外做未知整形。 |
| `exact_note_timing` | true（当前训练必需） | 原谱事件用于两项子步奖励、控制时刻当前goal/用指、176维事件观察和原谱按键诊断；不改原谱。 |
| `anticipatory_fingering_time` | 0.15s | 按原谱起音生成独立未来观察：含忙指下一音，延迟为原始秒数，不按控制帧取整，不影响当前奖励；闲指豁免独立使用20步窗口。 |
| `upcoming_press_permission_time` | 0s | 未开启提前按键许可。 |

旧 `anticipatory_fingering_reward`、`staged_fingering_reward`、`anticipatory_current_priority` 字段及CLI已删除，即使传false也拒绝；不再提供兼容评分分支。

当前统一入口的准备段为 `first_note_v1`：独立控制器在原谱时间开始前完成首音上方悬停，**整个准备前缀不生成奖励、不写入SAC replay、不占用学习/预填充控制步**。每次reset载入缓存的真实物理终态；动作观察和限速器从实际末次下发目标开始，闲指L1参考保持原始reset参考目标（不改为悬停姿态）。准备配置：`duration=2s`为主运动最短时长，`lift=0m`，`hover=.012m`，`settle_time=.5s`，`position_tolerance=.005m`；另有连续加速度起步段，保存实际总时长（本次2.55s）。每只手按原谱精确首音/和弦及原指法定位，原曲只在视频时间轴增加完整前缀偏移，不改时值。物理记录及超限诊断完整保留；悬停通过不等于全程无物理超限，详见准备报告。带臂/活动TZ或Yaw不支持此模式，明确拒绝。可选旧`policy_v1`仍按其显式配置产生准备期奖励/transition，不能把其20帧语义写成当前默认。正式演奏中的空谱帧按键无误触得0.4、踏板正确得1、当前指法0，其他项依各自公式计分；曲目休止和0.25s释放尾段无奖励豁免，reset本身不追加reward步。

没有错键终止或额外终止罚；错键影响上述奖励后继续演奏。曲目完成/显式时间截断的discount=1，SAC另外使用gamma=0.80。F1、碰撞诊断、键底接触、训练停止阈值和SAC熵项不是新的环境reward条目；诊断开启不代表存在对应罚分。

腕驱动当前恢复为exper031数值：TX/TY每轴armature=0.02kg、Kp=300N/m、阻尼34.15259872981849N·s/m、力限±50N，仍按轴显式配置。编译平移惯量须符合M0=.972kg；阻尼数值等于exper031在该惯量下的临界阻尼，但不改回旧动态计算分支。Pitch为Kp=8N·m/rad、阻尼0.5N·m·s/rad、力矩限±5N·m。手指驱动采用2026-10-23逐关节PD与拟合摩擦：Kp=2、逐轴Kd/力矩、armature=0.0001，关节阻尼b、干摩擦T0+kg|G|每物理子步更新，默认完整重力补偿；见[finger_resistance说明](finger_resistance_20261023.md)。目标运动限制与奖励公式不变；保留指尖任务点回退3mm、100%实际激活和1224mm琴键布局。非硬件标定或接触安全保证，见[默认驱动恢复与exper034](exper034_default_wrist031_20261022.md)。

## 硬控制、耦合与奖励的区别

| 控制项 | 当前值 | 与奖励的关系 |
|---|---|---|
| 腕Pitch位置范围 | −0.5～0.5rad（左右相同，2026-10-26） | 无臂direct_joint_v1动作映射与模型关节范围一致；初值0，统一入口range-scale=1；策略0/1/2分别映射−0.5/0/+0.5rad。wuji_cartesian_v1范围合同同步，但带臂Pitch仍结构固定。目标速度3.1416rad/s、加速度15.708rad/s²及驱动不变，实际轨迹不裁剪。 |
| 腕Pitch最终目标硬速度/加速度 | 3.1416rad/s / 15.708rad/s² | 默认无臂的可动Pitch先限制最终下发目标，再按软阈值3/15计算运动罚；在位置边界前减速。`wuji_cartesian_v1` 的Pitch结构固定，没有该目标运动通道，启动时明确清除通用配置的Pitch运动限值（末尾速度/加速度覆盖均为0），并非放宽可动关节约束。目标差分硬限制不保证实际物理qvel/qacc。 |
| 腕平移TX/TY最终目标硬速度/加速度 | 每侧每轴1.99m/s / 1.99m/s² | 最终投影在位置边界前提前刹停；限幅前需求按1.90m/s、1.90m/s²软阈值参与整项目标运动罚。逐轴限值，不是XY合速度上限；不是实际TCP物理速度保证。TZ/Yaw结构移除。 |
| 机器人硬位置限制 | `hard_v1` | 机器人关节位置约束，不是独立扣分项。 |
| MCP训练目标范围 | 左右十指（含拇指）统一0～1.4rad，约0°～80.21° | `direct_joint_v1`和`wuji_cartesian_v1`动作映射一致；URDF关节限位仍为0～1.57rad，末节耦合角范围仍为10°～80°。此项为目标位置范围，不新增奖励。 |
| 手指耦合 | `--finger-coupling`；范围文件 `angle_ranges_10_80.json`（八指10°～80°，2026-10-22用户覆盖） | 双手非拇指PIP/DIP由耦合生成；新运动罚、闲指L1都包含这些最终从动目标。`--no-coupling-reduce-observation`不代表关闭耦合。 |
| 闲指几何投影/主动按压许可 | false / false | 没有隐式开启几何保护；它们是控制开关，不是当前闲指L1的几何评分。 |

当前凸包模型不构造旧胶囊净空诊断，统一入口不设置对应琴键底面诊断停止门槛；与exper002实际部署一致，实体碰撞和运动硬限未因此关闭。

## 使用、维护与历史边界

本次指间碰撞独立快照使用 `shadow_control_frame_distinct_fingers_v1`；活动完整奖励合同以config为准，保留主会话的并行按键奖励变更。旧源码/模型检查点不能静默切换。指间筛选是固定规则，无旧评分模式开关；本轮证据见 `logs/finger_collision_mesh_20261013/`。

- 唯一默认入口是 `bash all_use_this_train.sh`；额外参数写末尾，启动时打印JSON和CLI合并后的完整奖励。
- 单项一般用 `--enable-reward NAME` / `--disable-reward NAME`；启用前需满足上述依赖。大部分weight和params修改对应JSON字段，不存在通用的 `--weight` 开关。
- 目标运动可用 `--finger-target-velocity-limit`、`--finger-target-acceleration-limit`、`--wrist-target-velocity-limit`、`--wrist-target-acceleration-limit`、`--tcp-target-velocity-limit`、`--tcp-target-acceleration-limit`、`--joint-motion-full-penalty-excess`、`--joint-motion-max-penalty` 覆盖；`--no-joint-motion-limit-penalty`只关闭罚分，保留腕硬控制。默认脚本已显式给出的CLI会覆盖JSON中同字段。
- `collision_reward`、`natural_posture_reward` 已从当前模板和实现删除，不属于23项；含它们的旧JSON即便enabled=false也会被新源码拒绝。
- 用户新约定：以后某奖励算法发生替换，在当前源码/配置/CLI/校验/测试和本表中删除被替代模式，直接维护新实现，不再为了兼容堆叠多个模式。仅改数值就更新参数，不新建mode。历史配置和产物集中在 `deprecated/`，当前入口不做旧 checkpoint 兼容转换。
- 本次同手项仅保留带显式`maximum_penalty_per_step`的子步累计公式；旧周期并集评分分支与不封顶配置不再支持，旧归档/实验不改。raw日志为负的合资格同手不同手指接触对—子步数，weighted/final日志为乘weight并封顶后的实际分值。`maximum_penalty_per_step`必须有限且非负；0表示该项罚分上限为0。
- 父版本直接替换L1项的“仅当前帧闲指”判定，显式未来20步、系数−0.1，不增加可切换的当前帧旧分支；其余奖励及训练参数保持。
- 2026-10-15按用户选择，正确键改为连续键程，仍逐原谱物理子步评分后平均。四个旧采样CLI/字段仍删除；不保留50%激活混合算法。最终Slurm2764验收41项测试及4项子测试通过、1项可选外部MJB检查跳过；归档/SHA已由统一入口固定。验证记录见 `logs/training_consistency_20261015/` 和 [本轮修复](training_consistency_20261015.md)，未启动训练。

本次同手封顶CPU／归档／入口验收证据：`logs/self_collision_cap_20261010/`，该历史验收权重0.2、整项最多−5；当前权重以本页总表为准。未启动训练；父接触GPU验收不冒充本次新评分GPU验证。

父版本参数CPU审计证据：`logs/tcp_thresholds_20261008/`；父算法证据：`logs/tcp_motion_limits_20261008/`（父闲指证据保留在`logs/idle_lookahead20_20261008/`）（父时序修复证据保留在`logs/shadow_score_sync_20261008/`）；包括归档白名单差异、独立解包测试、统一入口真实参数经sbatch/python stub捕获和新归档parser校验。CPU通过不代表GPU/CUDA graph或长期收敛验收；本次未启动训练。

原版依据：[Shadow任务](https://github.com/google-research/robopianist/blob/main/robopianist/suite/tasks/piano_with_shadow_hands.py)的 `_goal_current` 与 `_compute_key_press_reward`、[NoteTrajectory](https://github.com/google-research/robopianist/blob/main/robopianist/music/midi_file.py)及[piano_roll](https://github.com/google-research/robopianist/blob/main/robopianist/music/piano_roll.py)。同步的是目标与奖励时序，A3.2模型、额外奖励、硬限制与精确诊断仍是本项目配置。

实现依据：`mjlab_env/reward_config.py`（字段/启用约束）、`rewards.py`（权重/公式）、`task_logic.py`（五项基础奖励）、`task.py`（当前/未来观察及子步原谱目标）、`substep_rewards.py`/`scene.py`（积分后指尖/键几何采样）、`self_collision.py`（接触计数）、`exact_score.py`（精确谱时）、`suite/tasks/piano_with_tsensei_hands.py`（任务常数）、`models/piano/piano.py`（触发阈值）、`fingering_targets.py`（指法目标点）、`rl_torch/train.py`（最终参数解析），均位于项目内 `tsensei_v3_2_pianist/`。

2026-10-22原谱子步指法及当前10°～80°耦合/MCP目标范围合并验收：CPU 113项、131个子检查及Composer物理回调通过；Slurm8174三项GPU物理测试全部通过。源码、SHA与验收记录见 `logs/exact_fingering_20261022/`。本轮未启动训练，未产生新策略F1。

### 当前闲指/reset参考姿态（2026-10-22抬高8mm）

两侧相同，单位rad；SW固定0，拇指和腕保持原值。按URDF静态FK最小化MCP/PIP/DIP角度平方变化，八个指尖均升高8mm，水平位移小于0.84mm。参考姿态本身不施加末节范围约束；运行时耦合按10°～80°生成最终目标，所以最终目标可以偏离reset参考。

| 手指 | MCP | SW | PIP | DIP |
|---|---:|---:|---:|---:|
| index | 0.215778546714 | 0 | 0.345984067635 | 0.177049848625 |
| mid | 0.166889245860 | 0 | 0.337244124360 | 0.179382869067 |
| ring | 0.174138781500 | 0 | 0.366567845739 | 0.193132156912 |
| pinky | 0.193021298123 | 0 | 0.389615614939 | 0.213344010776 |

求解记录：`logs/idle_lift8mm_20261022/solution.json`；旧配置缺少`swing_scale`明确拒绝。

本轮验收：远端CPU奖励35项、相关回归39项及18子检查通过；独立Composer重建1项通过，八指实际FK均+8mm；Slurm8510物理/奖励40项通过、2项因冻结归档无外层入口而跳过，包含physics graph开/关及局部reset。静态参考不约束运行时末节角；最终下发仍由10°～80°耦合决定。未启动训练。

2026-10-22 单轮对照exper033：奖励配置和全部奖励覆盖与exper032一致，默认值不变。仅通过统一入口的显式源码归档参数切换腕部驱动数值为exper031（Tx/Ty Kp300、阻尼34.15259872981849、力限±50N；Pitch Kp8、阻尼0.5、力矩限±5Nm）。该次提交时默认仍为强驱动；随后用户在exper034提交时将默认恢复为旧腕数值。本轮冻结来源见results/exper033/launch_provenance.json。


曲尾物理回程：统一入口 `--ending-time 2`，配置字段 `preparation.ending_duration=2s`；最少0.5s松指＋最少2s回程＋0.5s稳定，指令运动限可能增加前两段时长。此阶段没有奖励项、weight或学习transition，原曲评估仍仅使用原episode；对比图的完整物理micro-F1包含回程误触（静音TN不增加F1）。保持末帧qpos/qvel/ctrl/暖启动，原生物理每子步记录，状态不裁剪。验收按原始腕平移5mm/转角.01rad、末速度.1各轴原单位、无激活键及松指末无激活键；手指全部残差及>.05rad警告单独保存，不将原始reset参考等同物理静态平衡。通用CLI默认0关闭，当前统一入口显式2开启。
