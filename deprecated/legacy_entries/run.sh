#!/usr/bin/env bash
# ARCHIVED 2026-10-04: historical single-GPU configuration; do not edit for new training.
# Active defaults: all_use_this_train.sh; it submits through scripts/hpc/submit.sh.
# All remaining comments and commands in this archived file describe historical settings.
# 当前小星星训练：源码 + 完整 config.json；20/200 Hz，无机械臂、无 TZ/Yaw。
# 可训练准备1秒：initial=origin上方2.5cm，策略立即控制，整体平滑回origin高度。
# 前20帧为空谱并正常计奖励/存replay，之后开始原乐谱；额外曲首空白0秒，曲末松键0.25秒。
# 保留乐谱原指法、手部重力补偿；附加1维剩余准备进度，共1181维观测。
# TsenseI模型、Shadow任务/驱动：Actor/Critic 256×3，初始温度1，discount0.8；三个学习率均3e-4。
# 显式采用shadow_v1任务/驱动、direct_joint_v1动作及shadow_primitive_v1碰撞，保留V3.2尺寸与坐标。
# 示例：bash run.sh --output-dir training_mjlab/outputs/my-run
# 可在命令末尾覆盖参数；512 环境 × 9766 向量步 = 5000192 样本。
# 几何净空只用于奖励/诊断；闲指保护与用指按压许可均默认关闭，可分别用参数开启。
# 保留有界humanoid辅助项；腕部只限速，不裁剪曲目专属位置范围。
# 先采完整1回合再学习；本曲183帧×512=93696条预填，第93697条开始更新。
# 前4999条随机探索，之后用随机策略采集；预填不累积待补更新。稳态128次更新/向量步，batch1024。
# 这是并行采集后集中更新，仍不同于原Shadow单环境的逐条采集/更新。
# 默认显示启动阶段和训练进度条（采样数、速度、ETA、更新数及评估F1）；--no-progress可关闭。
#
# 参数说明（2026-09-28，默认值指本 run.sh 实际传入值）
# 用法：bash run.sh [参数...]
# 数值／路径／成对布尔开关重复传入时，最后一个生效；奖励开关可以重复指定不同项。
# 相对路径以本独立项目根目录为基准。向量步 = 所有环境各推进一步，非单条样本。
# 下列参数由 scripts/train_from_source.py 和 rl_torch/train.py 联合接收。
#
# 一、源码姿态、乐谱与时序
#   --height FLOAT                 手根安装高度，单位m；默认0.10，必须>0。
#   --ty FLOAT                     左右腕初始TY，单位m；默认0.04，范围[0,0.15]。
#   --yaw FLOAT                    左右腕初始Yaw，单位rad；默认0.15，范围[-0.25,0.25]。
#   --control-hz FLOAT             控制频率Hz；默认20，必须>0。
#   --physics-hz FLOAT             物理频率Hz；默认200，须为控制频率的正整数倍。
#   --midi NAME_OR_PATH            注册曲名或.mid/.proto文件；默认TwinkleTwinkleRousseau。
#                                  当前GPU任务要求完整指法；保留乐谱原指法，不覆盖为单指。
#   --lookahead INT                未来前瞻控制步数；默认10，必须>=0。
#                                  本脚本固定传10，改控制频率后若要保持0.5秒需同时调整；
#                                  仅直接调用源码入口且不传此项时按round(control_hz/2)计算。
#   --frame-stack INT              观测堆叠帧数；默认1，必须>=1，会改变观测维度。
#   --initial-buffer-time FLOAT    每回合曲首空白秒数；默认0，必须>=0且对应整数控制帧。
#                                  policy准备模式下，这是1秒准备之外的额外空谱。
#   --score-validation {legacy,strict_v1}
#                                  默认strict_v1：检查每个原音符均有独立目标帧、重音间有释放帧，
#                                  且同一手指不同时分配多个键；无法表达时报错，不悄悄删音/改时间。
#                                  legacy保留历史转换；本项通用源码入口默认legacy。
#   --release-tail-time FLOAT      曲末空谱秒数；默认0.25，须>=0且对应整数控制帧。
#                                  策略/奖励/replay正常运行并检查松键；music_only指标排除尾段。
#                                  另报release_tail误触/最终按键数；通用源码入口默认0。
#   --preparation-mode {fixed_v1,policy_v1}
#                                  本脚本默认policy_v1：准备期策略动作/奖励/transition正常参与SAC。
#                                  fixed_v1保留旧固定动作准备，独立计算终态，准备帧不入replay。
#   --preparation-time FLOAT       initial到origin高度的准备秒数；本脚本默认1，0关闭。
#                                  须对应整数控制帧（默认20Hz时为0.05秒的整数倍）。
#                                  全程五次平滑曲线，无停留；policy模式由策略控制手指及现有腕轴。
#                                  从抬高initial真正开始GPU仿真，各world独立计时和局部reset。
#                                  前20帧空谱照常计奖励/F1/样本；同时报告music_only_f1作曲目对照。
#                                  policy模式新增1维剩余准备进度，1180→1181；旧检查点不可直接混用。
#   --preparation-lift FLOAT       initial相对origin的世界Z抬升m；本脚本默认0.025，必须>=0。
#                                  使用运动学父节点，原TZ继续物理移除，47动作不变。
#                                  本功能须同时--no-arms --disable-wrist-tz --disable-wrist-yaw。
#                                  1秒末回origin高度，手指角度由策略决定，不强制回写origin关节角。
#   --gravity-compensation / --no-gravity-compensation
#                                  手部重力补偿；默认开启。
#   --action-reward-observation / --no-action-reward-observation
#                                  观测中包含上次动作／奖励；默认开启，切换会改变观测维度。
#   --disable-forearm-reward        从源码任务中移除forearm奖励；默认不传。
#                                  使用时还需 --disable-reward forearm_reward，使奖励配置一致。
#
#   --task-profile {tsensei_v1,shadow_v1}
#                                  默认shadow_v1：原Shadow指法目标及掌部forearm奖励范围。
#                                  tsensei_v1保留此前拇指单独目标、跨手掌指名单。
#   --thumb-target-mode {profile,uniform}
#                                  默认profile：采用task-profile原规则，可用下面距离覆盖拇指。
#                                  uniform：拇指与其余手指统一目标公式，不能同时传距离参数。
#   --thumb-white-target-distance FLOAT / --thumb-black-target-distance FLOAT
#                                  可选拇指目标距琴键世界+X端的距离m，朝−X；未传的颜色保留profile规则。
#                                  例如白键0.02、黑键0.015；只改变拇指指法距离奖励目标，不是机械限位。
#
# 二、模型／动作空间／接触
#   --collision-profile {legacy_mesh_v1,shadow_primitive_v1}
#                                  默认shadow_primitive_v1：按V3尺寸拟合的简单碰撞体。
#                                  掌部保留原mesh凹槽适配，外观CAD不变；legacy为原网格碰撞。
#   --drive-profile {legacy_v1,shadow_v1}
#                                  默认shadow_v1：TX/TY等效平移惯量、临界阻尼、指关节摩擦。
#                                  每平移轴armature3kg为仿真参照，非实测V3惯性；不增加Pitch质量。
#   --no-arms                      去掉M6手臂、IK及臂碰撞；本脚本已传入。
#   --disable-wrist-tz             编译前移除左右TZ关节／驱动器，固定0；本脚本已传入。
#   --disable-wrist-yaw            编译前移除左右Yaw并固定在初始角；本脚本已传入，须配合TZ关闭。
#                                  仅关闭TZ为49动作，同时关闭TZ/Yaw为47动作；结构锁止用于无臂模式。
#   上述三个为单向store_true开关，没有 --arms / --enable-wrist-tz / --enable-wrist-yaw。
#   若要恢复自由度／机械臂，需修改下方命令或直接调用源码入口，不能在末尾传false抵消。
#   --action-profile {legacy,mixed_v1,direct_joint_v1}
#                                  默认direct_joint_v1：四指MCP策略[0,2]映射目标[0,1.3]rad。
#   --finger-coupling / --no-finger-coupling
#                                  可选四指MCP驱动、PIP/DIP从动；默认关闭，拇指保持独立控制。
#                                  保留PIP/DIP实体关节，移除16个策略输出；当前47→31动作。
#                                  SW及腕部仍可控；从编译模型计算从动目标，不强行改写实际关节位置。
#   --finger-coupling-angle-ranges JSON
#                                  开启耦合后默认各指独立许可50–80度；JSON逐项覆盖左右index/mid/ring/pinky共8根指。
#                                  格式见configs/finger_coupling/angle_ranges_50_80.json，必须给全8项。
#                                  末节朝指尖轴线向琴面下倾的夹角，不是DIP弯曲角；区间内误差为0。
#   --finger-coupling-angle FLOAT  显式恢复旧单一目标角，须在(0,90)，优先于区间文件；如60。
#                                  从动关节仍受接触/驱动延迟影响，许可区间不是实际关节的硬约束。
#   --coupling-reduce-observation / --no-coupling-reduce-observation
#                                  耦合开启时默认压缩：删除16从动qpos及16上次动作槽/堆叠帧。
#                                  当前提前定位0.15秒时1231→1199维；关闭压缩用于单独比较控制效果。
#                                  从动实际状态仍受动力学影响，删观测可能损失信息，须以F1实验判断。
#                                  耦合需从头训练；禁止跨耦合/观测合同静默加载旧检查点。
#   --control-limits PATH          软件目标范围／变化率JSON；本脚本显式generic_wrist_rate_20260927.json。
#                                  本脚本用下方CLI覆盖TX/TY为1m/s，Pitch4rad/s；不收紧位置范围。
#                                  原动作映射后先裁剪范围，再按上次真实下发目标限速；机械限位不变。
#                                  有限速时必须保留action-reward-observation，观测记录实际下发目标。
#                                  完整解析合同写入config/checkpoint，回放自动使用；详见configs/control_limits。
#   --wrist-tx-max-velocity FLOAT / --wrist-ty-max-velocity FLOAT
#                                  双手TX/TY目标限速，单位m/s；本脚本显式各1，可在命令末尾覆盖。
#   --wrist-pitch-max-velocity FLOAT
#                                  双手Pitch目标限速，单位rad/s；不传沿用JSON，当前4。
#   --wrist-tz-max-velocity FLOAT / --wrist-yaw-max-velocity FLOAT
#                                  双手TZ(m/s)/Yaw(rad/s)目标限速；当前结构移除，拒绝设置正值。
#   --finger-max-velocity FLOAT    双手40个手指关节目标限速，单位rad/s；默认不新增限速。
#   --joint-max-velocity NAME=VALUE
#                                  单关节覆盖，可重复；如R_Wrist_TX=0.6、R_Pinky_PIP=8。
#                                  优先级：JSON < 双手轴/手指组参数 < 单关节参数；同名重复取最后值。
#                                  底层CLI未传时保留JSON；本脚本显式覆盖TX/TY。0移除限速，保留位置范围。
#                                  最终合同存入config/checkpoint；限制下发目标，不保证真实关节速度。
#   --finger-clearance-config PATH 几何计算JSON；本脚本用configs/finger_clearance/control_disabled_20260927.json。
#                                  用于geometry_clearance_v1奖励及键底诊断；加载它不代表动作保护已开启。
#                                  从模型40个指碰撞体/88键位置计算，不按曲名/音高/时刻写特例。
#                                  保存完整几何合同，回放自动重建；不修改物理qpos/机械限位/接触反力。
#   --idle-finger-protection / --no-idle-finger-protection
#                                  闲指整指目标净空保护；本脚本显式关闭。前者开启且覆盖JSON同名布尔值。
#   --active-press-permission / --no-active-press-permission
#                                  用指按实际位置获得按压许可；本脚本显式关闭。前者开启且覆盖JSON同名布尔值。
#                                  两项独立；关保护仍可计算几何奖励/诊断，不会停止物理接触。
#                                  底层两CLI未传为None：无JSON保持几何模块关闭，有JSON保留文件合同；
#                                  单独传开启参数而不传JSON时，只开启该项，使用active_contact_v1几何参数。
#                                  历史generic_20260927.json缺少两字段时保持原保护开启，旧回放不变。
#   --allow-control-change         仅配合--finetune-from显式迁移限速/几何保护合同，重建优化器和replay。
#                                  不允许用于--warm-start或Stage2迁移；旧检查点默认沿用原控制合同。
#   --self-collision-detection-only 同手碰撞只检测计罚、不产生反力；默认关闭。
#   --hand-collision-detection-only 同手及跨手碰撞只检测计罚、不产生反力；默认关闭。
#                                  两种模式均保留手—琴键真实接触；本脚本默认三类均有反力。
#
# 三、奖励（开关及数值最终保存到config.json和checkpoint）
#   --self-collision-weight FLOAT   无奖励JSON时的旧快捷权重，底层默认0；本脚本不传此参数。
#                                  本脚本实际自碰撞由下面JSON启用，权重0.2；不要把快捷值0当实际权重。
#   --reward-config PATH           完整奖励JSON；本脚本显式configs/rewards/generic_finger_clearance_20260927.json。
#                                  与非零self-collision-weight互斥；本脚本只在JSON内指定各奖励开关和权重。
#   --anticipatory-fingering-time FLOAT
#                                  提前定位秒数，0关闭；本脚本默认0，可用0.15进行单项对照。
#                                  开启后追加未来用指信息观测，不能直接加载旧维度检查点。
#   --anticipatory-current-priority / --no-anticipatory-current-priority
#                                  默认关闭：当前与未来用指共同求均值，未来项不再乘(1-current)。
#                                  显式开启才使用current+(1-current)×0.2×future的旧优先公式。
#                                  A3.2/HPC入口保持提前0.15秒；提前定位时间0时此开关无影响。
#                                  通用CLI不传保留JSON，旧JSON缺字段按关闭处理。
#   --staged-fingering-reward / --no-staged-fingering-reward
#                                  分阶段指法奖励：接近时对准，按压时强调真实琴键行程；本脚本默认关闭。
#   --exact-note-timing / --no-exact-note-timing
#                                  保留原MIDI起止，在物理采样时刻用[start,end)目标评分与评估；
#                                  追加88键各自下一起音/结束倒计时共176维。HPC默认开，普通run.sh不变。
#                                  旧检查点保留原时间合同；该项与新观测需从头训练。
#   --upcoming-press-permission-time FLOAT
#                                  可选提前请求目标几何许可，秒，默认0关闭，最大0.15；候选0.05。
#                                  须精确时序、提前定位窗口>=本值、active_contact_v1且两物理投影关闭。
#                                  只放宽即将用指在指定键内的末节请求深度，实际/近端/其他键风险继续扣分。
#   --substep-rewards / --no-substep-rewards
#                                  汇总物理子步奖励以补足瞬时误触；本脚本默认开启，单项对照可显式关闭。
#                                  20/200Hz和原20Hz F1保持；子步平均不会把惩罚简单放大10倍。
#   --reward-sampling auto|control_endpoint
#                                  默认auto，配合两个子步开关：按键/误触/能耗取均值，碰撞周期并集去重。
#                                  指法距离、闲指姿态和几何净空保持周期末评分。
#                                  control_endpoint用于历史对照，须同时关闭上述两个子步开关。
#   --target-key-substep-reward / --no-target-key-substep-reward
#                                  正确按键正奖励取整个控制周期的全部物理子步均值；本脚本默认开启。
#                                  在当前20/200Hz下取10个子步，迟按/早松减少得分，不修改物理或控制频率。
#   --target-key-activation-weight FLOAT
#                                  正确按键分中实际激活占比λ∈[0,1]；本脚本默认0.5。
#                                  每子步正项=(1-λ)×目标键连续行程分均值+λ×目标键实际激活比例，
#                                  再对全部子步求均值；没有目标键时正项为0，原正项总权重不变。
#                                  激活判定与F1一致；默认0.5防止未达触发阈值仍接近满分。
#                                  两项生效均要求key_press_reward开启且权重>0；关闭均值并设λ=0恢复旧正项。
#                                  上述底层CLI未传时保留奖励JSON；完整配置保存到reward_config.performance。
#   --enable-reward NAME            启用一项，可重复传；本脚本由JSON启用，不再叠加显式enable。
#   --humanoid-finger-gesture       启用当前JSON中选择的humanoid模式；不会把本脚本模式改成旧正奖。
#                                  本脚本为有界负罚；参考角取当前origin每手20关节，准备期也计罚。
#                                  旧JSON未选新模式时仍可使用原每闲指0.2的高斯正奖励。
#   --disable-reward NAME           关闭一项，可重复传；默认无额外关闭项。
#                                  先读JSON／模板，再应用CLI；同一项不能同时出现在两种开关中。
#   NAME全部可选值及条件（“默认开”以本脚本、不传额外参数为准）：
#     key_press_reward             按键奖励，默认开；legacy/paper模式及和弦倍率在JSON设置。
#     sustain_reward               踏板奖励，默认开。
#     energy_reward                原绝对功率惩罚，默认开。
#     fingering_reward             指法距离奖励，默认开。
#     forearm_reward               跨手碰撞奖励，默认开；shadow_v1无臂时仅采用两手掌部名单。
#     self_collision_reward        同手碰撞惩罚，本脚本JSON开启，权重0.2。
#     false_press_penalty           非目标键下压深度惩罚，本脚本JSON开启、权重1。
#     humanoid_finger_gesture_reward 默认开，geometry_clearance_v1是该奖励的模式，不是额外一项奖励。
#                                  准备/休止/弹错时也生效的有界负罚，与动作保护两开关独立。
#                                  几何80%/姿态20%，合计不超过0.1及单次漏键损失预算；可单独disable。
#                                  旧每指0.2及task_conditioned_v1仍可在JSON显式选择。
#     forearm_power_reward         腕部平移正功率惩罚，默认关；先关闭energy_reward。
#     wrist_power_reward           腕部旋转正功率惩罚，默认关；先关闭energy_reward。
#     finger_power_reward          手指正功率惩罚，默认关；先关闭energy_reward。
#     forearm_velocity_reward      腕部平移目标速度惩罚，默认关；须填action_units。
#     wrist_velocity_reward        腕部旋转目标速度惩罚，默认关；须填action_units。
#     finger_velocity_reward       手指目标速度惩罚，默认关；须填action_units。
#     forearm_acceleration_reward  腕部平移目标加速度惩罚，默认关；须填action_units。
#     wrist_acceleration_reward    腕部旋转目标加速度惩罚，默认关；须填action_units。
#     finger_acceleration_reward   手指目标加速度惩罚，默认关；须填action_units。
#     finger_speed_reward          手指目标超速惩罚，默认关；须physical单位及非零权重。
#                                  当前模板该项权重为0，仅开enabled不会产生扣分。
#     collision_reward             扩展碰撞计分，默认关；须填categories并排除重复计罚项。
#     natural_posture_reward       参考姿态偏差惩罚，默认关；须填action_units和reference。
#     wrist_pitch_reward           指定腕Pitch方向奖励，默认关；须填两侧directions。
#     forward_press_reward         等效按压角惩罚，默认关；须填十指angle_map。
#   action_units、weight、params、scale_by_dt等是JSON字段，不是本脚本CLI参数。
#   Stage2奖励模板开启功率／速度／加速度／超速项前还须显式填写regularization_shaping。
#   公式、映射及互斥关系详见docs/reward_design_review.md。
#
# 四、SAC与采样／更新预算
#   --stage {1,2}                  SAC阶段；默认1。阶段2需stage2-from、warm-start或finetune-from，inspect-only除外。
#                                  此项不自动切换奖励模板；本脚本其他显式参数也不会随阶段自动变化。
#   --init-temperature FLOAT       初始熵温度；默认1，必须>0，训练中继续自动学习。
#   --actor-hidden INT [INT ...]   Actor各隐层宽度；默认256 256 256，至少一层，各宽度>0。
#   --discount FLOAT               SAC gamma；默认0.8，范围[0,1]，与终止transition discount分开。
#   --actor-lr FLOAT               Actor学习率；本脚本默认0.0003，必须>0。
#   --critic-lr FLOAT              Critic学习率；默认0.0003，必须>0。
#   --temp-lr FLOAT                熵温度学习率；默认0.0003，必须>0。
#                                  显式覆盖stage预设；这三个CLI不自动随stage切换。
#   --replay-capacity INT          按transition计容量；默认1,000,000，覆盖replay-vector-slots。
#   --warmup-transitions INT       随机探索条数；默认4999，覆盖warmup-vector-steps，允许0。
#   --learning-start-transitions INT
#                                  请求首次学习的样本序号（从1计）；默认5000，覆盖learning-starts。
#                                  启用整曲预填时，实际取max(本项,预填条数+1)，保存请求值与实际值。
#   --replay-prefill-episodes INT   学习前完整采集的回合数；默认1，通用CLI默认0，须>=0。
#                                  按实际含准备/尾段的谱面长度计算；不延长随机探索、不追补预填更新。
#                                  要求replay至少容纳所有环境一轮整曲，且max-episode-steps不截断曲子。
#   --updates-per-transition FLOAT
#                                  新增样本的更新预算；默认0.25，覆盖gradient-steps，必须>0。
#                                  边界向量内按样本数划分随机/策略行与学习预算。
#   --seed INT                     随机种子；默认42。
#   --num-envs INT                 并行环境数；本脚本默认512，必须>0（底层默认256）。
#   --vector-steps INT             本次采样向量步；本脚本默认9766，必须>0（底层默认20000）。
#                                  总transition数=本项×num-envs；当前为5,000,192。
#   --replay-vector-slots INT       旧调度回放向量槽；底层默认1024，必须>0。
#                                  本脚本由replay-capacity覆盖，修改该项不会改变本轮容量。
#   --batch-size INT               每次SAC更新采样数；默认1024，必须>0。
#   --warmup-vector-steps INT       旧调度探索向量步；本脚本由warmup-transitions覆盖，允许0。
#                                  直接调用底层且未传此项时，Stage1为300、Stage2为0。
#   --learning-starts INT          旧调度学习门槛；底层默认1000，本脚本由learning-start-transitions覆盖。
#                                  同时须满足回放样本数足够等实际更新条件。
#   --gradient-steps INT           旧调度每向量步更新数；默认32，必须>0。
#                                  本脚本由updates-per-transition覆盖；稳态128次更新/向量步。
#
# 五、设备与加速
#   --device STRING                Torch设备；默认cuda:0。MJLab训练入口要求可用CUDA设备。
#   --physics-graph / --no-physics-graph
#                                  用CUDA graph执行物理子步；默认开启。
#   --compile-learner / --no-compile-learner
#                                  编译SAC更新计算；默认开启，首次更新有编译开销。
#   --fused-optimizer / --no-fused-optimizer
#                                  使用CUDA融合Adam；默认开启。
#   --learner-graph / --no-learner-graph
#                                  本脚本默认开启完整SAC CUDA图；须fused-optimizer开启。
#                                  每次更新的采样/噪声/优化顺序和次数不变，不启用TF32或混合精度。
#                                  图内逐次累计非有限值，向量更新块末检查，失败后不采集/保存。
#   --optimizer-layout / --no-optimizer-layout
#                                  本脚本默认开启Adam GPU执行布局优化，须learner-graph开启。
#                                  使用本机Torch原始adam_math，保留原公式/精度/优化器状态。
#                                  首次需nvcc/C++编译器/ninja，后续使用编译缓存。
#                                  关闭完整图时同时传--no-learner-graph --no-optimizer-layout。
#
# 六、评价、保存与显示
#   --eval-interval INT            每隔多少向量步做评价；默认100（51,200样本），0关闭周期评价（底层默认1000）。
#   --eval-episodes INT            每次评价回合数；默认3，必须>0。
#   --eval-num-envs INT            独立评估环境数；run.sh默认1，HPC双卡入口显式3。
#                                  最多同时运行eval-episodes个回合；逐回合指标及最差回合门槛保留。
#                                  只改变评估执行方式，不减少评估次数/物理采样或学习更新。
#   --save-interval INT            每隔多少向量步保存权重；默认500，0关闭周期保存。
#                                  底层默认1000；训练正常完成仍保存final.pt。
#   --log-interval INT             每隔多少向量步写训练指标；默认10（5,120样本），必须>0。
#   --progress / --no-progress     启动阶段信息及采样进度、速度、ETA等；默认开启。
#   --collision-diagnostics / --no-collision-diagnostics
#                                  本脚本开启；Python默认关闭。统计200Hz同手/跨手及手—键底接触，
#                                  保留CUDA图，不改碰撞反力；手—键诊断目前须配合整指几何模块。
#   --eval-at-start / --no-eval-at-start
#                                  本脚本默认开；学习前记录新环境中父策略真实评分，不计连续达标次数。
#   --save-best / --no-save-best    默认开；先满足显式键底上限及完整诊断，再按最差音乐F1保存best.pt。
#                                  卡底/缺诊断的更高分不能覆盖合格best；从未合格时不生成best。
#   --best-f1-metric NAME          默认music_only_f1_min（40Hz时为音乐逐帧最低值）；
#                                  可选exact_original_music_micro_f1_min，按各回合原谱精确物理子步F1最低值选best。
#                                  精确模式要求完整原谱计数/子步诊断，不改变奖励或原F1公式。
#   --target-exact-micro-f1 FLOAT   可选额外原谱精确F1最低值门槛；须同时启用target-music-f1，旧默认不加此门槛。
#   --target-music-f1 FLOAT         本脚本默认0.99；启用达标停止，范围[0,1]。
#   --target-micro-f1 FLOAT         本脚本默认0.99；音乐段micro F1最低值。
#   --target-consecutive-evals INT  默认3，必须>=2；所有回合连续达标才停止。
#   --target-max-preparation-wrong-frames FLOAT
#                                  默认0；任一评估回合准备误触控制帧上限，必须>=0。
#   --target-max-hand-collision-frames FLOAT
#                                  默认0；任一评估回合手碰撞控制帧上限，必须>=0。
#                                  目标停止须开启周期评估和collision-diagnostics；缺统计不算达标。
#                                  F1/准备误触采20Hz端点，碰撞覆盖200Hz，采样含义不同。
#   --target-max-key-underside-substeps INT
#                                  本脚本显式0；每回合200Hz真实键底接触子步上限，同时影响best与早停。
#                                  缺采样不能达标，也不覆盖已有合格best。
#                                  须加载finger-clearance-config几何模块、collision-diagnostics和target-music-f1；
#                                  两种动作保护可关闭，真实键底检测仍执行。
#                                  通用Python默认None，不改变历史目标门槛。
#   --max-episode-steps INT         显式回合步数上限；默认不设，若设须>0，可导致时间截断。
#   --output-dir PATH             运行目录的基础名称；入口自动追加北京时间时间戳。
#                                  默认training_mjlab/outputs/sac-YYYYMMDD-HHMMSS-微秒。
#                                  重复传同一名称会产生新目录，保留旧config、指标和权重。
#
# 七、恢复与阶段转移
#   --warm-start PATH             从checkpoint恢复学习器／优化器／随机状态；默认无。
#                                  回放缓冲区、物理、乐谱进度及调度重新开始，不是逐步精确续接。
#   --stage2-from PATH            仅从Stage1转移网络；默认无；须 --stage 2。
#                                  优化器、回放和温度重新初始化；与warm-start互斥。
#   --finetune-from PATH          同环境合同的网络微调；默认无，与warm-start/stage2-from互斥。
#                                  不复制优化器/replay/温度；温度用显式init-temperature。
#   --allow-reward-change         允许仅网络转移时改变奖励；默认关，配合stage2-from/finetune-from。
#   网络维度、动作定义及相关环境合同须满足加载检查；Actor层数不能靠热启动自动转换。
#
# 八、SwanLab 指标上传（独立进程读取本地日志，不改变采样／更新）
#   --swanlab / --no-swanlab       本脚本默认开启；直接Python入口默认关闭。
#   --swanlab-project NAME         默认 TsenseI_A3.2；未指定workspace时用个人空间。
#   --swanlab-workspace NAME       可选团队／用户空间。
#   --swanlab-run-name NAME        默认使用本轮输出目录名。
#   --swanlab-mode {online,offline,disabled} 默认online；offline只保存待上传日志。
#   --swanlab-key-file PATH        默认 ~/.config/tsensei-pianist/swanlab.key；SWANLAB_API_KEY优先。
#                                  不支持命令行明文API key，不写入训练config／checkpoint。
#   --swanlab-python PATH          默认 training_mjlab/monitoring/.venv/bin/python。
#                                  首次安装：bash scripts/setup_swanlab.sh；不改变训练Python依赖。
#   主图顺序：F1、recall、precision、启用奖励的字母序；数字前缀固定排序。
#   三主指标来自同一eval全段统计；音乐段F1、micro指标另列，避免混淆口径。
#   奖励主图上传加权实际值，原始值另列；横轴为累计样本数。
#   网络／认证失败不停止训练；本地metrics.jsonl保留，可用scripts/swanlab_upload.py补传。
#
# 九、帮助与检查
#   -h / --help                   显示源码构建入口帮助后退出，不启动训练。
#                                  转发到训练器的参数完整说明见本文件头；底层帮助命令：
#                                  training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train --help
#   --build-only                  仅构建并验证环境、打印配方后退出；默认关闭，不启动训练。
#   --inspect-only                构建环境后打印有效训练配置，不分配GPU训练场景；默认关闭。
#                                  经本脚本仍会先进行CPU模型构建；纯参数检查可直接调用底层模块。
#   本源码入口拒绝外部 --scene、--task、--source-config、--cache-dir；它们不能从run.sh覆盖。
#   Critic结构固定256×3/GELU/LayerNorm/dropout0.01；不能传Shadow的 --agent-config.* 参数。
#
# 示例（执行将按所选模式启动新进程，不会热改已有训练）：
#   bash run.sh --enable-reward false_press_penalty --output-dir training_mjlab/outputs/my-run
#   bash run.sh --num-envs 2 --vector-steps 10 --no-progress --inspect-only
#   bash run.sh --reward-config configs/rewards/my_rewards.json
#   bash run.sh --idle-finger-protection --active-press-permission
#   bash run.sh --idle-finger-protection --no-active-press-permission
#   bash run.sh --disable-reward humanoid_finger_gesture_reward
#   python3 scripts/run_reward_comparison.py --samples 1000000 --seed 42
# 修改本参数表时同步核对两个Python入口及实际下方命令，避免与解析器／默认值漂移。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export MUJOCO_GL=egl
exec python3 scripts/train_from_source.py \
  --swanlab --swanlab-project TsenseI_A3.2 \
  --height 0.10 --ty 0.04 --yaw 0.15 \
  --control-hz 20 --physics-hz 200 --lookahead 10 --frame-stack 1 \
  --midi TwinkleTwinkleRousseau --initial-buffer-time 0 \
  --score-validation strict_v1 --release-tail-time 0.25 \
  --preparation-mode policy_v1 --preparation-time 1 --preparation-lift 0.025 \
  --gravity-compensation --action-reward-observation \
  --task-profile shadow_v1 --collision-profile shadow_primitive_v1 --drive-profile shadow_v1 \
  --no-arms --disable-wrist-tz --disable-wrist-yaw --action-profile direct_joint_v1 \
  --stage 1 --seed 42 \
  --init-temperature 1 --actor-hidden 256 256 256 --discount 0.8 \
  --actor-lr 0.0003 --critic-lr 0.0003 --temp-lr 0.0003 \
  --num-envs 512 --vector-steps 9766 --replay-capacity 1000000 --batch-size 1024 \
  --warmup-transitions 4999 --learning-start-transitions 5000 --updates-per-transition 0.25 \
  --replay-prefill-episodes 1 \
  --eval-at-start --eval-interval 100 --eval-episodes 3 --save-best --save-interval 500 --log-interval 10 \
  --physics-graph --compile-learner --fused-optimizer --learner-graph --optimizer-layout --progress \
  --control-limits configs/control_limits/generic_wrist_rate_20260927.json \
  --wrist-tx-max-velocity 1 --wrist-ty-max-velocity 1 \
  --finger-clearance-config configs/finger_clearance/control_disabled_20260927.json \
  --no-idle-finger-protection --no-active-press-permission \
  --reward-config configs/rewards/generic_finger_clearance_20260927.json \
  --anticipatory-fingering-time 0 --no-staged-fingering-reward --reward-sampling auto --substep-rewards \
  --no-anticipatory-current-priority \
  --target-key-substep-reward --target-key-activation-weight 0.5 \
  --collision-diagnostics --target-music-f1 0.99 --target-micro-f1 0.99 \
  --target-consecutive-evals 3 --target-max-preparation-wrong-frames 0 \
  --target-max-hand-collision-frames 0 --target-max-key-underside-substeps 0 \
  "$@"
