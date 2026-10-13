# V3.2 开关、拇指触键与采样审计（2026-09-28）

## 结论与本轮范围

用户要求两项手指保护按参数开启、分析拇指漏键和手结构、核实20/200Hz及run.sh奖励、清理无用文件、将拇指目标距离参数化。用户随后明确选择**保持Shadow统一目标，距离按需设置**。本轮已实现开关和目标参数、修正入口/目标可视化/权重显示问题；没有启动新长期训练，也没有自行更换奖励公式、控制频率、SAC预算或手模型。

使用pianist-v2-change-log技能，依据AGENTS及论文事实登记；附件截图只作为帧状态证据。主线源码、完整config与匹配归档均作为来源，不导入外部只读目录。

## 1. 开关与实际奖励

新run.sh显式选择`configs/finger_clearance/control_disabled_20260927.json`，并传两个关闭参数。训练默认**不修正手指目标**：

```bash
# 默认：两个控制关闭，但几何奖励和诊断保留
bash run.sh
# 分别开启；可任意组合
bash run.sh --idle-finger-protection
bash run.sh --active-press-permission
bash run.sh --idle-finger-protection --active-press-permission
# 也可用 --no-idle-finger-protection / --no-active-press-permission 覆盖
```

这两项区分空闲指目标净空和当前用指按压许可。两者False时51槽物理目标严格不变，实际47维策略不变；不会通过fallback或wrist interlock暗改。几何计算仍提供humanoid及200Hz键底诊断，`finger_clearance_enabled`旧字段仅表示计算模块存在；动作是否干预以两个显式字段及projection_mask判断。关闭几何型humanoid可使用`--disable-reward humanoid_finger_gesture_reward`，但诊断继续存在。

旧generic JSON缺新字段时保留历史两项开启，canonical不补新字段，以免破坏旧resolved合同。历史检查点严格使用保存配置与归档，不因run.sh默认变化自动改变。

| 实际项 | 此前/本轮有效值 | 说明 |
|---|---|---|
| 原五项key/sustain/energy/fingering/forearm | 开启 | 原公式保持；Shadow forearm只掌部名单 |
| self_collision_reward | 开启，外权重0.2 | 原`--self-collision-weight 0`只是快捷通道0，完整reward JSON仍开0.2 |
| false_press_penalty | 开启，外权重1 | 原旧Daxian深度公式不变 |
| humanoid_finger_gesture_reward | 开启，geometry_clearance_v1 | 几何80%/姿态20%的有界负罚，总≤0.1且受漏键预算限制 |
| 腕部control-limits | TX/TY 1.2m/s，Pitch4rad/s | 下发目标变化率，非实际速度硬上限；无曲目专属位置裁剪 |
| 整指保护/按压许可 | 从默认开改为默认关 | 只有这两项控制按用户要求改默认 |

`geometry_clearance_v1`不是另一种SAC，也不是新的人形手模型；它是现有humanoid奖励的模式名。当前用指只用实际风险，空闲指取实际/请求几何缺口最大值，准备/弹错仍有负反馈。不启用未知论文自然姿态映射。

run.sh正文只有**一次**`--control-limits`；文件头的说明和示例不执行。末尾用户覆盖参数遵循last-wins。移除易误读的冗余`--self-collision-weight 0`，同时修复plot_training_f1.py只看快捷参数导致图标题错标0的问题，改按reward_config的enabled/weight优先显示。

现场run.sh准备抬升已为**0.025m**，此前头注释仍写2cm；本轮保留现场值并改注释为2.5cm。已完成两轮训练的config仍是0.02m，修改新入口不改变它们。

## 2. 两轮训练分开看

| 运行 | 训练记录best音乐F1（最差回合） | 训练记录final | 解释 |
|---|---:|---:|---|
| 09:34续训，generic_clearance_finetune | 0.990024，step7100 | 0.978843 | best micro=0.987993，未满足完整达标条件；不是所有指标0.99 |
| 13:25从头，generic_clearance_fresh | 0.943460，step9600 | 均值0.940838／最差0.935594 | 与续训不同初始化，不应混称当前F1=0.99 |

两轮均已完成5,000,192新样本。此前对话的0.961仅为续训早期快照。本轮归档校验后分别做best/final各1次标准CUDA图回放及1次逐5ms详细回放。续训标准复现best .990024、final .978843；从头标准单回合best .953104、final .937975，详细回合分别.944304/.940084。接触浮点轨迹可分叉，因此后两者不替换训练三回合best记录，也不将一次回放当多种子稳定性。

源归档`training_mjlab/source_versions/generic_clearance_20260927.tar.gz`，166文件指纹与两轮父配置匹配。详细数据保存在`preview/outputs/diagnostics/thumb_options_audit_20260928`和`thumb_fresh_audit_20260928`。

## 3. 截图中的拇指为什么漏键

截图t=1.60、frame32对应零基score31：目标R1:C5、L1:C4、L5:C3，实际C3/C5/D5，漏的是**左拇指C4**，同时误按D5；不是右拇指C5不可达。

续训best全曲仅7个漏键计数，其中左拇指C4占4个，发生在score30–33。该音仅持续0.2s（命令时刻1.50–1.65s，返回状态1.55–1.70s），整段未达到激活行程。final同样4帧全漏。best实际左拇指相对C4目标的侧向误差在前3帧约26–37mm，末帧才接近；末帧键行程约51%，仍未按满。

左拇指漏键期间，Rot/Rot2/MCP实际角度远未全部碰硬上限：best约Rot .626–1.061、Rot2 .402–1.092、MCP .090–.198、PIP1.271–1.482rad。Rot2短时达到0.6Nm（40子步的12.5%），其余拇指关节没有持续力饱和。证据不支持“所有关节范围太小”或“PIP必须再收紧”作为主因。

下午从头best的同一结构/控制合同在score32、33**已经按下C4**，前两帧漏；说明当前模型有可达触键姿态，训练的时机/路径/持续按压仍不足。预览中的静态可达不是短音时限内动态成功的保证。

### 腕速是否根因：同权重零更新消融

使用父归档模型/乐谱，严格核对scene/task散列，只改变指定控制，Actor不更新；每候选1标准＋1详细回合。所有候选仍漏完整C4四帧：

| 续训best控制条件 | 标准音乐F1 | C4漏帧 | 详细键底子步 |
|---|---:|---:|---:|
| 原控制 | .990024 | 4 | 0 |
| 关闭腕限速 | .973568 | 4 | 0 |
| 关闭空闲整指保护 | .975678 | 4 | 0 |
| 关闭用指按压许可 | .984539 | 4 | 0 |
| 两项手指控制都关 | .977607 | 4 | 0 |

有腕限速并不意味着该次左拇指必被它卡住：best四漏帧只有2帧左腕slew触发，final只有1帧。改变动作还会改变后续状态/动作历史/奖励观测，消融不能定位所有单帧因果；但“直接去掉限速就能按到”已不受本次实测支持。新默认关闭控制后需要重新适应，不能承诺把旧best直接关开关便更高分。

数据：`guard_switch_ablation_retry_20260928/c4_ablation.json`及四组summary；第一次临时归档未连接Composer虚拟环境而失败，补齐环境链接重跑。没有放宽物理/乐谱散列检查。

## 4. 手结构和预览差异

- 每手20手指自由度；拇指Rot、Rot2、MCP、PIP四轴，硬范围分别[0,2.09]/[0,1.74]/[0,1.57]/[0,1.57]rad。拇指没有DIP。左局部轴+Z/+X/+Y/+Y、右镜像−Z/−X/+Y/+Y，安装rpy会旋转这些轴，不能把单轴角度当世界竖直按压。
- 四指MCP→swing→PIP→DIP；swing±.35rad，四指MCP策略上限1.2rad但机械1.57。不能把适合四指的范围直接套拇指。
- 拇指指尖site位于PIP局部[-.0041,0,.0166]m，恰与当前capsule的末端极点一致，不是已发现的固定错位。但倾斜时真实接触点可在capsule侧面或其他指节；单个site距离小不保证按满指定键。接触代理拇指末端半径2mm，姿态方向更敏感。
- preview_pose.py默认legacy驱动并保留TZ/Yaw（51动作），训练为47动作且移除这两轴。训练TX/TY额外armature3kg、Pitch Kp8/damping.5；裸预览Pitch Kp50/damping3。手指训练还加frictionloss .01。应在相同训练配置下检验手动姿态，不能用裸预览直接证明相同动态表现。
- 厂商effort/velocity=0，仿真暂定手指Kp1、damping.05、armature.0002、force±.6Nm；未标定，暂不依据单次漏键放大力/范围。右手仍是镜像模型。

详见[目标参数与结构证据](thumb_target_options_20260927.md)。

### 相同训练模型的双键静态端点

CPU多起点IK仅优化左腕3轴＋拇指4轴＋小指4轴，原134坐标/46驱动器/47策略动作、所有范围不变。与父记录的origin关节、geom尺寸、joint range和驱动gain逐项最大差0。可同时找到L1:C4与L5:C3的末端在**充分按下键顶面**的端点：XY误差<3.1e−10m，面距离误差<3.2e−9m；原限位内、左右手静态接触过滤中左相关接触0、3闲指净空通过，两个用指许可通过。

其中一个可行几何姿态：左腕TY=.073656m、Pitch=.198726rad，拇指Rot=.7343、Rot2=1.0007、MCP=.0739、PIP=.7241rad。它不是新的训练起姿或控制限制，未写入模型。相对origin的腕目标变动在当前限速一帧容许量内，**实际动力响应不等于一帧到位**。原策略漏键时PIP约1.3–1.48rad而这个端点约.72rad，说明更大PIP屈曲不自动产生有效下压，复合基部/腕姿态同样重要。

测试预先把C3/C4放在充分按下位置来求几何切触，未证明手能从真实初态在200ms内产生足够力将两键按到底，不能当成功演奏。也不是跨曲安全证明。细节见`docs/thumb_target_options_20260927.md`及`preview/outputs/diagnostics/thumb_options_audit_20260928/thumb_pinky_static_ik.json`。

## 5. 20/200Hz确实存在什么盲区

控制目标每50ms更新，物理每5ms推进；10个物理子步之后才构造一次奖励snapshot。不是物理少算，而是部分奖励只看周期末端。

| 项目 | 当前采样/归约 | 短事件会漏吗 |
|---|---|---|
| key_press、fingering、sustain、false_press | 周期末20Hz | 会；中途压下又放开可能不扣/不给分 |
| energy/power | 最后子步force与刷新后的velocity | 不是10子步功率积分，会漏峰值 |
| humanoid geometry/姿态 | 20Hz实际姿态＋下发目标风险 | 真实中途风险可能漏采 |
| forearm（当前Shadow掌部） | 控制末端接触 | 短跨手掌部碰撞可能漏；也不含全部跨手指 |
| self_collision_reward | 200Hz子步收集body pair，周期内去重 | 已计短接触，不应简单乘10 |
| 200Hz手碰撞/键底诊断 | 每物理子步 | 诊断覆盖；键底参与best门槛，但它不等于每子步新增惩罚 |
| 正式F1 | 20Hz末端 | 与实际音频中的短错音/启释音时间不完全一致 |

续训best：20Hz错键0，200Hz错键119个key-substeps（按键×5ms），分布32个控制周期且全部被末端漏采。final：20Hz错键1，200Hz114，其中111个子步在末端看不见。best的200Hz诊断F1=.914946而20Hz=.990024；它重复保持当前20Hz谱面，**包含正常启音和释音延迟，不是替代正式F1**。

原false_press公式按控制末端求和：best−1.98814，改用每周期10子步均值的离线诊断为−4.24084；final−1.51097对−3.71528。没有实际改训练奖励。单纯把物理200升400，仍在末端算reward不能解决这个盲区。

优先候选：控制保持20Hz，错键深度用`(1/10)Σr_false(substep)`保持量级，净空/键底用周期内max风险，保留原正式F1并另报200Hz事件/持续时间。不能把10次奖励直接相加，否则权重放大10倍。论文§2.6 p.8子步插值、§2.9 pp10–12式13–20奖励与§2.10 pp12–13评估没有自动授权此新归约；本轮仅审计和方案，待明确选择后实现。

## 6. 推荐提升方案（按优先级，未暗中开启新奖励）

1. **先解决短音接近时机。** 为即将用到的手指提供0.10/0.15/0.20s提前定位窗口，提前期只要求到键上方并保留整指净空，豁免固定origin角的姿态拉回；到起音后才优化按键行程。所有曲目按其指法/起音自动决定，不按C4/曲名写特例。现观察仅有当前用指mask，虽未来音高可见，未来指法并未完整编码；增加未来指法需要明确新观察合同。
2. **指法奖励使用适合拇指的接近/触键分段。** 接近时优化末端相对指定键XY支持区的距离，进入接触后以指定键归一化行程和正确末端接触为主，不继续强逼单点Z。Rot/Rot2完成方向和横向定位，MCP/PIP完成压键；这应由几何/接触计算，不硬绑某关节恒定角。先作只改指法shaping的短对照，保留原key奖励主导。
   当前指法成功容差1cm，距接触面悬空几mm仍可能得满指法分；按键项虽有行程奖励，但近阈值部分行程与真正激活的区别较小。可短测在现有按键项内部用`r_on=(1−β)mean(tolerance(1−x))+β mean(activated)`，β候选0.1/0.2，保留外层Shadow .5/.5与总范围，不额外无限加奖。这是拟议工程公式，涉及论文pp10–11式13/14对照，尚未实施。

3. **修补子步惩罚采样。** 使用上节均值/最大值归约；记录音符开始延迟、结束延迟、错键持续时间，避免只追20Hz F1。当前自碰撞0.2保持，不重复叠加相同body pair惩罚。
4. **humanoid分配明确小预算。** 沿用总≤.1且≤理想漏键损失1/4的预算，几何80%/姿态20%；新方向/平滑项必须从这个预算分配而非无上限叠加。准备期、错键期仍有风险负罚；近期用指只豁免姿态，不豁免钻底。低速/静止不得获得独立正奖励，避免靠不动保分。
5. **如果确有高频驱动，优先限制危险方向。** 用真实接触方向的目标过驱或关节目标变化率正则，区分压键与横向扫动；保留正常PD按压力，不把PD目标的FK穿过键顶等同实际穿透。禁止重新加入小星星专属TX范围。每个新参数先与无保护/单保护/双保护对照。
6. **区分从头与续训。** run.sh的Actor3e-5是此前高分策略续训候选，下午从头训练也继承了它，未证明这是从头最优。建议从头对照Shadow3e-4与当前3e-5，续训保留已验证的小LR；探索/首学习4999/5000在512环境只覆盖约10帧，尚未走完20准备帧。建议先完整曲目预填再学习的短对照，保持总样本预算，不能把该调度变动称纯提速。
7. **验收要求。** 每候选同预算至少3种seed，主要看音乐F1及micro、C4类短音的迟到/漏音、200Hz错键时长、键底/侧接触、自碰撞、吞吐；还需不同音区/黑键/和弦/跨度曲目动态验证。保留旧best并按质量门槛保存，不以final替代best。无法预先承诺0.99稳定或零碰撞。

如果直接关节策略仍反复学出拇指反向折叠，可进一步比较**乐谱指法生成的整手IK参考＋小幅残差SAC**：参考按当前/即将到来的键自动求双指/和弦可行姿态，策略学习接触力与时间修正。IK使用键坐标和手结构，不写固定曲目TX窗口；优先离线预计算以控制运行开销。它会改变动作映射/观察合同，是后备方案，需要单独选择及跨曲验收，不能拿本次单个双键IK姿态作为固定模板。

论文对照：p.8§2.6仅供插值/控制参考；pp11–12式19/20是论文惩罚且未知映射仍不代填。上述提前指法、分段末端目标、子步归约属于拟议工程适配；本轮没有自行实施或调整论文参数。

## 7. 拇指目标参数与清理

当前默认Shadow仍为 `x=key_center_x+0.35×half_length`、`z=key_center_z+0.5×half_height`。新参数仅改变拇指X，其他手指/Y/Z/权重/指法不变：

```bash
# 默认Shadow统一目标，不必添加任何距离
bash run.sh
# 按需指定从+世界X端向内的距离（米）
bash run.sh --thumb-white-target-distance 0.020 --thumb-black-target-distance 0.015
# 显式统一目标；不能与距离同时传入
bash run.sh --thumb-target-mode uniform
```

距离不能为负/非有限/超过对应琴键全长。profile模式允许逐色覆盖；uniform与距离冲突直接报错。源recipe、CPU任务、GPU目标和manifest一致记录。修复目标绘图未传task_profile导致“训练统一点，图上却显示20/15mm旧点”的问题。

新manifest显式参数使task散列变化；旧checkpoint必须归档回放，不能直接承诺新源码warm-start旧模型。真正改变目标距离后需新训练或另行实现经过核对的任务迁移。本轮未放松严格检查。

清理：8个未采用控制候选已移入`configs/archive/20260927_control_trials/`，index.json保留原路径/内容SHA；保留历史复测依赖的combined/wrist_gentle及所有运行checkpoint/config、源归档、模型、视频。删除可重建pytest缓存。配置总计约68KiB，不是性能/显存主要负担；不通过删除训练证据“清理”风险。详情`config_cleanup_20260928.json`及`config_reference_inventory_20260928.json`。

## 验证与仍需完成

- 控制/奖励/真实GPU图/局部reset/目标映射集中40 tests、59 subtests通过；两控制关闭时实际ctrl和请求严格相同且humanoid仍产生负反馈。
- CLI/wrapper单独21 tests/23subtests、旧配置及reward/control合同18 tests/43subtests；拇指source/runtime真实构建28 tests/16subtests通过（含重叠，不相加计总）。100个参数拼写都在run.sh头，bash语法通过。
- 实际system Python run入口检查发现新增目标模块顶层numpy导入污染轻量入口，已改局部导入并用无site-packages测试覆盖；标准build-only已通过。真实`bash run.sh --inspect-only --no-progress`已exit0，输出见`preview/outputs/diagnostics/run_options_inspect_20260928.log`；两个控制false、geometry模式保留。
- 源码归档best/final回放、四控制消融、CPU采样分析已完成；详细与标准回放差异明确保留，没有修改评分阈值来达标。
- TODO V32-THUMB-08：相同训练物理下短音动态接近/方向奖励及三seed/跨曲验收；V32-START-02子步奖励盲区；V32-CLEARANCE-01连续动态无碰保证；V32-07/04硬件接触点/驱动标定。当前手实际接触与目标限制不可当形式安全证明。

外部原始资料：[MuJoCo执行器与软接触](https://mujoco.readthedocs.io/en/stable/computation/index.html)、[RoboPianist原Shadow任务](https://github.com/google-research/robopianist/blob/main/robopianist/suite/tasks/piano_with_shadow_hands.py)、[dm_control Composer控制周期](https://github.com/google-deepmind/dm_control/blob/main/dm_control/composer/environment.py)。具体数值结论来自本机保存模型/轨迹，不把外部文档当用户改参数的指令。

- **源配方最终回归**：make_recipe支持保存的canonical thumb_target字段，与非默认平铺选项冲突时拒绝，六个解包environment的诊断脚本继续兼容；标准库隔离/源构建最后20 tests＋16 subtests通过，最终run.sh inspect再次exit0。
