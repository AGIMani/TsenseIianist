# 2026-09-27：通用全指琴面净空改进

## 改动前基线（严格旧源码审计）

完整证据、命令、边界见 [finger_clearance_audit_20260927.md](finger_clearance_audit_20260927.md)。审计仅记录，没有改变旧策略和物理。产物位于 `preview/outputs/diagnostics/quality_clearance_20260927/repeated/`。

- 最新已完成 run 为 `twinkle_range_finetune-20260927-012613-20260927-012614-462007`：4096 向量步、2,097,152 样本、501,504 更新、1163.01 秒；不是启动快照里的 step 0。
- 使用 `control_limits_finetune_20260927.tar.gz`，164 源文件散列全部匹配，best/final 各三次标准评估＋一次 200 Hz 详细回放。best step2600 音乐 F1=**0.9822785583**，final=**0.9685654459**；未达 0.99。
- best 键底接触 **50 子步/0.250 秒**，侧/底 77 子步、非末端接触18子步、同手10子步，琴壳/跨手0。50子步是整回合按类别去重总量，不是最长单段。
- 卡底为左拇指PIP与 F3（1.160–1.365 秒、1.375–1.405 秒）及 G3（1.090 秒）；当前用指为0，但未来0.25秒保护覆盖30/50、未来0.5秒可见覆盖50/50。它1.5秒需按C4；不能让“未来要用”豁免侧底进入。
- 0.9秒左拇指已在琴键XY外降到约−31.4mm，随后低位横向入键；1.25秒Rot2目标约.056、实际.843rad、力矩限幅−.6N·m并有相反约束力矩。原因不只是目标TX跳跃，限速未解决接近路线。
- origin全10指局部净空正；双拇指完整碰撞体对白键净空+11.819mm，initial+31.819mm。不是原始姿态埋琴。全球黑键高度35mm不应无视XY误报初始拇指穿透。
- 全指指标必须覆盖40个碰撞primitive，而非只有10个tip site；best右小指有19子步tip在白键上方但其他指节低于白键面。十指均有近期不用时局部包络净空为负的样本（保守包络不是实际穿透）。
- best准备20Hz误触0仍有200Hz E5在.965秒短响10ms；应保留原F1同时报告子步误触、净空和键底。不能用高20Hz分数证明没有卡键或杂音。

基线图：`preview/outputs/diagnostics/quality_clearance_20260927/repeated/best/detail-0/opening_finger_clearance.png`。

## 通用验收要求

应覆盖空谱、十指单独/组合用指、黑白键/不同位置、休止到按键、近期保护边界、tip安全而近端偏低、低位琴外向内移动。分别报告曲首/整曲任意手指键底子步、最大净空缺口、干预率及音乐F1，不能写入特定曲名/音高/时间特例。后续实现与实测由本任务补充；当前基线明确未通过“无键底进入”。

## 初版失败及固定轨迹根因诊断

用 [probe_clearance_contacts.py](../scripts/probe_clearance_contacts.py) 严格比较同一best的物理scene/乐谱散列，显式网络迁移后0次更新，baseline保留旧配置；候选采用新guard、新有界奖励和通用rate-only限制，无旧曲目X范围。各1次标准回放＋1次200Hz详细回放，作为集成筛查：

| 方案 | 音乐F1 | 键底子步 | 侧/底子步 | 非末端触键子步 | 同手子步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原best baseline | 0.98227856 | 50 | 77 | 18 | 10 |
| 初版strict投影 | 0.08333334 | 0 | 406 | 20 | 110 |
| idle-only主动指不投影消融 | 0.94002436 | 0 | 51 | 20 | 3 |

初版虽无键底，音乐严重失效且其他碰撞恶化，**未通过，不启动长训**。idle-only恢复大部分按键但仍低于baseline，且未处理主动指奖励输入的PD目标误罚；不能直接宣布正式方案通过。完整数据在 `clearance_contacts_probe_retry_20260927` 和 `clearance_idle_only_probe_20260927`，位于 `preview/outputs/diagnostics/`。

CPU固定baseline FK排除了“只是闭环偶然分叉”的解释：284成功用指帧中275会被strict改目标，四关节最大改变量中位0.485rad；265帧的PD目标末节比完全按下键面更低超过1mm，中位−15.71mm，而实际姿态中位仅−1.73mm。请求XY完整pad适配76/284，实际278/284。伺服目标与真实受接触约束姿态不是一回事；约束命令FK撤掉了正常按键过驱，并误判大量目标为不在目标键上方。实际平面包络与接触穿透亦须区分，不能按本数值直接放大“允许穿透”。

仅替换父Actor的previous-action观测带来动作通道差中位0.115（P95 0.385）；仅替换last_reward差中位0.0216（P95 0.0438）。两种输入改变确有分叉效应，但active命令误投影在固定轨迹上已经成立。详见 [奖励和PD目标审计](humanoid_clearance_reward_20260927.md#首轮守护器失败与pd目标诊断)。修正方向为闲置整指继续保护，主动指合法按压与真实风险分离；奖励也不能再拿合法active目标FK过驱直接扣分。

脚本验证：`tests/test_clearance_probe_tools.py` 4项、2子测试通过；包括baseline配置保持、候选不继承旧曲目范围、严格scene/task拒错、callback转发assigned_keys、实际/请求诊断与单位。首轮运行因辅助函数名写错在加载前退出，修为真实 `checkpoint.clearance_contract` 后在新输出目录成功；失败目录保留。工具不放宽source/scene/task合同、不持久化冻结scene。

## 网络微调入口

[launch_clearance_training.py](../scripts/launch_clearance_training.py) 复用当前 `run.sh`，不固定曲名；姿态、乐谱、时序默认读取父checkpoint的源码配方，额外 `--` 后参数原样转发。脚本头与 `--help` 列出全部包装参数，`--steps` 单位是向量步。父策略温度通过CPU读取得到 **0.0016916337190195918**，只继承网络并显式设置这份温度，Adam/replay重新开始；初次评估与best保存开启，基线不能伪称学习改善。已保存启动计划，当前只做dry-run，未开训。

```bash
training_mjlab/.venv/bin/python scripts/launch_clearance_training.py \
  --parent training_mjlab/outputs/twinkle_range_finetune-20260927-012613-20260927-012614-462007/best.pt \
  --steps 512 --actor-lr 0.00003 --dry-run \
  --plan-output preview/outputs/diagnostics/clearance_contacts_probe_retry_20260927/planned_launch.json
```

先继承父学习起点91137，512环境先采完整178帧再更新。若显式修改环境数或时序，必须重新核对完整曲目的填充量；本脚本不会把不同乐谱绕过严格task散列检查。控制/奖励可变更仅通过明确 `finetune-from + allow-control-change + allow-reward-change`，禁止转发另一个warm-start/stage2或偷偷重置alpha。达标门槛直接由run.sh指定，包装器不提供虚假的关闭开关。内嵌外部乐谱尚未实现临时字节导出，当前包装器明确拒绝，不能用原basename假装外部文件仍在；注册乐谱可正常使用。run.sh中的最终默认、完整source/config和新输出目录在真实启动时记录；当前启动计划不是长期训练已完成的证据。

验收命令 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 training_mjlab/.venv/bin/python -m pytest tests/test_clearance_launcher.py tests/test_clearance_probe_tools.py -q`：**13 passed、2 subtests passed，0.07秒**。补充验证父温度/乐谱/override保持、非法恢复覆盖拒绝、embedded明确失败、help无需父文件可读。包装器实际CPU读取best并dry-run成功；正式候选回放、短训和长期启动由主任务继续。

## 训练、检查点与回放合同

- `--finger-clearance-config PATH` 读取完整几何控制配置，默认 `None`。config/checkpoint 保存规范化的 `finger_clearance` 与重建几何得到的 `finger_clearance_contract`；只保存路径不够。
- 旧检查点缺少两个字段等价于关闭，旧物理/奖励/准备合同继续有效。新 guard 回放必须两个字段都完整并与重建结果一致。`eval_from_source.py` 原有全 config 双向校验也覆盖这些新增字段，不能删字段规避检查。
- 变更 guard 或软件限幅必须显式 `--finetune-from ... --allow-control-change`；是只迁移网络、使用新优化器/replay的微调，不是精确继续旧状态。场景/任务/准备/动作/实体碰撞仍严格检查，改变奖励另需 `--allow-reward-change`。
- train 与独立 eval 使用同一 guard 配置；任一动作变换启用时 reset 的 previous_action 来自实际驱动 ctrl。evaluate、视频、交互预览和计时脚本均传 resolved 合同；视频保存每帧干预/回退/请求、下发和实际缺口。
- 评估另报 `finger_clearance_projection_frames`、`finger_clearance_fallback_frames`、三类 `*_max_deficit_m` 与覆盖率。缺少一帧或非有限缺口不会被当作零违规。已有 200 Hz 碰撞覆盖和F1口径保持。
- CPU 验证：新几何检查点合同4项、训练质量/评估统计10项（含新增guard缺失/NaN覆盖）、旧软件限幅合同4项通过。新guard物理效果和速度以主任务后续GPU实测为准；接口测试不能替代卡键验收。

### 最佳视频必须使用训练器的合格best

发现旧 `preview/scripts/export_best_mjlab.py` 忽略 `best.pt/best.json`，从metrics中的step/final按整回合F1重新选择，可能把键底门槛排除的高分策略重新选回来。现凡config包含 `evaluation_selection`，只接受训练器权威best；best保存关闭、尚无合格best、文件缺失或配对不一致都明确失败，**不回退旧指标**。只有没有新selection合同的历史run保留原mean-frame F1、return、较晚step的排序规则。

脚本用 `torch.load(..., map_location='cpu', weights_only=True)` 读取受信本地检查点；当前Python缺Torch时调用独立训练venv CPU子进程。核对run配置、checkpoint selection配置及完整best记录、vector_step/updates、对应last_evaluation与best.json一致；门槛启用时另验证键底计数、完整有效物理采样及采样率。并发训练原子替换best.pt而best.json稍晚写入时，最多重读3次匹配稳定文件对，持续错配报错，不假定旧json适用于新权重。输出明确criterion与权威best记录。best仍可能随着正在运行的训练继续改善，输出是选择时的匹配记录。

实际CPU `--select-only` 验证原range run选到best.pt step2600、音乐F1 0.9822785583；新clearance lowLR短训也通过其键底门槛合同。`tests/test_best_checkpoint_selection.py` **8项、4子测试通过（0.65秒）**，覆盖更高F1但不合格step不能被选、新run缺best不回退、元数据/eligibility/coverage错配拒绝、并发json短暂落后重读、此前无键底门槛的music-best兼容、关闭保存明确失败及旧run原排序不变。没有为这项工具修复启动GPU或更改训练包。

## 200 Hz 键底诊断与可选达标门槛

新增 `mjlab_env/key_contact_diagnostics.py` 只读取主物理接触缓存，不启动额外碰撞求解，不改变接触掩码、力或奖励。每个物理子步按环境/类别去重：`key_underside_substeps`（键局部外法向Z≤−0.7）、`key_side_or_under_substeps`（Z<+0.7）、`key_non_distal_substeps`（接触不在DIP或拇指PIP末端胶囊）、`key_contact_physics_substeps`（实际采样数）。侧/底是包含关系，不能相加；阈值与基线详细审计相同。手掌和手指的显式碰撞几何参与观察，手—手碰撞另行保留原统计。

`--target-max-key-underside-substeps` 默认 `None`，保持历史达标与best选择规则；显式设为 `0` 时每个评估回合还必须无键底接触、完整物理采样、诊断有效，缺数据不能冒充0。这个条件同时约束**best候选资格和达标停止**：F1更高但重新卡底或缺少统计的评估不能覆盖已有best.pt；通过键底资格后才比较最差回合音乐F1。decision记录`best_eligible`/`best_ineligible_reasons`，best.json保存入选资格与键底限值；若从未出现合格候选则不创建best.pt，final仍按原规则保存。它要求 `--target-music-f1`、显式guard和 `--collision-diagnostics`。本轮新observer仅在guard与碰撞诊断同时开启时创建，旧generic/mesh场景不被新的末端胶囊识别约束意外阻断。新config存observer几何/阈值合同，评估与视频报告区分手—手与手—键底部。

CPU Warp四项验证覆盖法向与反序geom方向、旋转键的局部坐标、接触点重复、零/正距离、掌部/非末端、局部reset与非物理forward不重复计数，并验证输入缓存未被写入。训练质量测试扩为13项，新增“键底达标门槛缺失/无效覆盖不通过、更高F1但卡底不替换合格best、默认None保持历史、20Hz帧与200Hz子步计数分开”。首次测试fixture错误假设MuJoCo geom按XML书写顺序编号，已改为具名id映射后全部通过。实际GPU模型的接触统计仍由主任务集成验收，不将合成缓存测试当作物理策略验收。


## 最终采用的规则与未采用的候选

本轮选择 `full_finger_clearance_v1` + `active_contact_v1`，在新run.sh中显式加载，不修改通用Python默认。

1. **闲指**：根据编译模型的40个碰撞capsule，要求下发目标的整指高于无限白键平面2mm，并考虑局部黑键高度。琴外也不允许先把目标压到琴面下再横移钻底；不根据曲名、C4/F3、特定时刻或专属TX窗口判断。
2. **当前用指**：实际末端在指定键XY支持区内（允许2mm边缘余量）、未低于完全按下顶面5mm且整指未到相交键底面2mm保护区时，允许原PD目标产生按压力。未获许可时从当前实际姿态附近接近/抬起，每关节最多相对实际0.2rad；保留原机械限位和真实接触。5mm基于原策略成功触键中实际软接触最大4.770mm选择，是工程容差，不是论文参数或允许真实钻键底。
3. **辅助负罚**：沿用geometry_clearance_v1有界公式，80%几何/20%初始角姿态，总负罚≤0.1且≤一次理想漏键损失的1/4。当前用指只用实际风险，不能把正常PD目标的厘米级过驱计为危险；闲指用实际与请求缺口最大值。准备/休止/弹错都可有信号，不能靠不动获得额外正奖。其他七项与父保持。
4. **腕部**：仅保留TX/TY目标变化率1.2m/s和Pitch4rad/s，完全删除旧小星星TX区间裁剪。可选暂缓腕目标的interlock实测增加侧面接触，本轮关闭。未引入动作子步插值、物理qpos投影、TZ自由度或无接触穿模。
5. **评估/保存**：主F1仍按20Hz原公式；另外在200Hz真实主物理接触中记录键底、侧/底和非末端。每次3回合按最差音乐F1选best，显式键底阈值0先过滤候选，缺失诊断不能覆盖best。连续3次音乐/micro F1均≥.99、准备零错帧、手部零接触帧、键底零子步才早停。

静态许可只是必要条件。位置控制目标并非真实物体位置，指节惯性/伺服延迟/碰撞会使实际姿态越过目标包络；本轮没有证明任意动作序列的硬物理安全，也没有用瞬移掩盖违规。实际净空缺口、自碰撞、键侧摩擦仍持续记录，不能把“本回放键底0”写成“所有手指始终离琴/全部碰撞0”。

## 同父零更新对照

下表F1取标准CUDA图回放，键接触取逐5ms完整详细回放；候选各1标准+1详细回合用于选型，旧best另有3标准重复。发生强接触时两种执行记录方式可有浮点轨迹分叉，见原Shadow同步限制。全部保留同父网络/物理/乐谱/origin，候选同时迁移控制及humanoid，所以不能把每个分数变化归因于单一参数。

| 控制候选 | 音乐F1 | 详细键底子步 | 详细侧/底子步 | 结论 |
|---|---:|---:|---:|---|
| 父best/原曲目腕范围 | .982279 | 50 | 77 | 开头左拇指仍卡底 |
| 严格目标FK | .083333 | 0 | 406 | 错误截断正常按压力，淘汰 |
| 只保护闲指消融 | .940024 | 0 | 51 | 说明闲指保护有效，但不作为用指保护完备方案 |
| 实际位置许可 `active_contact_v1` | .915341 | 0 | 96 | 用于短训适配，仍有侧碰 |
| 上项＋低位停腕 | .912176 | 0 | 135 | 侧碰增加，本轮不启用 |

对应原始数据在preview/outputs/diagnostics/clearance_{contacts_probe_retry,idle_only_probe,active_contact_probe,interlock_probe}_20260927。该表不声称新策略已经优于父音乐效果。

## 同预算短训与正式续训

两组均从原best显式只迁移网络，场景/乐谱散列相同、1181观察/47动作不变。继承父已学习温度0.0016916337190195918；新Adam/replay先采完整178帧曲目（91,136样本），首学习样本91,137。512环境×512步=262,144样本，batch1024、0.25 update/transition，各42,752次更新；仅比较Actor LR，Critic/温度仍3e-4。每次3个完整回合，键底诊断覆盖率1。

| Actor LR | 末尾音乐F1 | best音乐F1 | 末尾键底/侧底子步 | 末尾手碰撞子步 | 准备错帧 | 训练耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 3e-5 | .921730 | .921730（step512） | 0 / 132 | 35 | 1 | 147.49s |
| 3e-4 | .867541 | .915341（step0） | 0 / 137 | 16 | 1 | 147.98s |

小LR相对新控制初评.915341略改善，大LR退化；两组均未达到原父.982279，不能说0.99已实现。小LR稳定循环末段约1,991 transition/s，总训练循环262,144/147.49≈1,777 transition/s；耗时口径不含外层源码构建，不宣称纯执行比以前更快。接触约束有计算代价，SAC仍使用已批准的batch1024×.25加速预算，FP32及原损失公式不变。

已于**2026-09-27 09:34:25北京时间**开启正式续训：

- 目录：`training_mjlab/outputs/twinkle_generic_clearance_finetune_20260927-20260927-093425-663203`。
- 父：`clearance_trial_lowlr_20260927-20260927-092532-822854/best.pt`，step512；完整恢复同合同Actor/Critic/目标网络、Adam、温度和RNG，replay/世界/调度重新开始。不是精确逐步恢复。
- 512环境×9766向量步=5,000,192新样本；同低LR与奖励/控制。先完整曲目预填，再0.25更新比；每100步3回合评估，保存合格best。
- 初评音乐F1 .921730、键底0，与父复现一致；已核实step356累计updates65,408，比父42,752新增22,656，进程确实进入学习。
- 完整argv、PID及启动快照在`training_mjlab/outputs/latest_clearance_launch.json`，日志为`twinkle_generic_clearance_finetune_20260927.launch.log`。PID/步骤均是时间快照，后续以进程/metrics/summary为准。
- 源码归档`training_mjlab/source_versions/generic_clearance_trials_20260927.tar.gz`，SHA256 `1caf48862dbfe1da1e699a521322155c2cf26c024df6bbba79ea578c6299da93`；432总文件/166包源文件。逐文件与两试验config/当前包完全匹配，不包含冻结scene/task。

短训best已通过严格eval.sh完整配置重建与有声回放：[视频](../preview/outputs/replays/clearance_lowlr_short_best_20260927/replay.mp4)。其200Hz报告键底0、侧底132、非末端93/1780子步，音乐评分与训练一致；仍有准备误触和手部接触，不能把演示视频当最终达标结果。

## 验收与下一步

- 本轮集中验收53 passed、76 subtests：真实MuJoCo FK、解析Jacobian、当前用指许可/压力语义、图与eager一致、应用动作历史、局部reset、奖励预算、旧模式、旧检查点、新控制合同、无效诊断拒绝和best/早停规则。另脚本13 passed、2subtests；没有因结果不好修改通过阈值。
- 独立全键盘静态平移：176候选中162处于各手原机械范围，162/162许可/零几何缺口；故意错配一八度162/162拒绝。两手覆盖并集88键。它不是和弦、手间避碰或多曲动态训练验收。
- V32-CLEARANCE-01：有限目标检查/实际接触许可并非连续动力安全证明；目前真实净空仍有暂态不足，目标是持续压低而非伪造为零。
- V32-QUALITY-01：新训练需验证音乐与micro≥.99、准备零误触及低自碰撞；新best必须先过键底0筛选。若长期不能恢复音乐效果，应检查用指许可切换时机/触键延迟，不直接加大humanoid把手压成静止。
- V32-CLEARANCE-02：再用其他带指法曲目、黑键/和弦/跨键移动做动态验收；保留完整机械范围，暂时不宣称所有曲子都已验证。
- V32-START-02：20Hz奖励/F1仍可能漏掉200Hz短错音；本轮保留原评分口径，音频照常记录，不静默更改论文/Shadow评价定义。

主要参考为项目实际模型/控制/奖励源码、严格归档best/final和新探针。外部技术参考：[MuJoCo仿真流程](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/simulation.rst)、[接触与动力计算](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst)；这里只使用模型几何和执行器目标语义，不声称实现了具有形式证明的控制屏障函数。

运行进度快照（2026-09-27T09:41:39.775442+08:00）：step1300、累计updates186368，当前合格best step1100音乐F1=0.941139511，键底0.0子步。该快照不替代最终summary；还未达.99。

- **最终归档与导出复核**：`training_mjlab/source_versions/generic_clearance_20260927.tar.gz`，SHA256 `bbce9ac3affecc0e0db6e59bfa719e31432cdd4ad17d4006fef2e8bcc2d3418a`；435个文件，166个包源码逐项匹配正式运行config。仅源码、配置、脚本、模型和文档，无冻结scene/task。权威best导出独立8项测试/4子测试通过，旧range与新短训实际CPU选择验证通过。
- **后续进度快照（2026-09-27T09:50:27+08:00）**：训练step2848，累计updates384384；合格best step2700音乐F1=0.961121281，三个回合键底子步[0.0, 0.0, 0.0]、手部碰撞子步[0.0, 0.0, 0.0]、侧/底子步[22.0, 22.0, 22.0]、20Hz准备错帧[0.0, 0.0, 0.0]。训练仍运行，当前音乐表现尚低于旧父.982279，侧接触与真实净空暂态仍有；不宣称达到.99或连续物理安全。
