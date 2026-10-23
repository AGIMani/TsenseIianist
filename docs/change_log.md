# 当前变更日志

## 2026-10-23：指法连接配置移出logs并增加下一组斜线预告

- **需求与参考**：用户要求JSON配置不放logs，下一次指法按将使用手指的浅色加斜线显示。参考 `fingering_editor{,_data,_view}.py`、远端/本地准备脚本、本地worker与压力测试的配置读取位置，以及当前指键色表/上一组残留。
- **配置**：新增统一路径/迁移助手，连接设置使用 `configs/fingering_editor/{remote,local}.json`，旧文件校验原字节后迁移、已有新配置不覆盖；实际remote.json已迁移。本地后端准备、读取、压力测试和CLI默认同步，诊断日志仍留logs。
- **显示**：取原谱严格晚于当前时刻的最近起音组（含和弦），按即将使用手指的浅色显示斜线与“下＋编号”；当前同键保留原色并叠加斜线，下一组优先于同键上一组残留。可关闭预告，曲尾清空；编辑/撤销/倒退和异步旧响应按现有请求ID重新匹配。未来提示不改变当前求解/原谱或训练奖励。
- **验证**：13项数据/预算/相机/配置迁移/斜线裁剪测试通过，Python编译通过；受限Tk实际预览与远端连接使用新路径的证据见 `logs/fingering_editor/next_fingering_20261023.{png,json}`。保留原有内存上限和现有未保存编辑窗口，保存后重开生效；不修改当前训练源码归档/SHA或预览冻结模型。

## 2026-10-23：恢复指法编辑器丢失的远端配置

- **问题与来源**：用户截图显示“未配置远端三维”；检查发现本地整个 `logs/fingering_editor/` 和原v6归档已不存在，运行窗口因找不到remote.json直接停在未配置状态。SSH至volc-a32及原预览基目录、固定归档仍可用。参考编辑器构造/启动分支及 `prepare_fingering_editor.py`，不推断目录丢失原因。
- **修改**：远端配置延迟到后台线程加载，缺失时自动初始化，显示进度/错误及日志；准备脚本支持指定输出，本地固定归档不存在时允许核验远端完全匹配的固定SHA后独立恢复，禁止以远端工作区替代。保留原v6预览合同，不更改当前训练模型/奖励/归档，不静默升级预览。
- **验证**：配置恢复、远端进程ready及轻量状态请求验收见本轮 `logs/fingering_editor/`；增加缺失配置时后台初始化失败可见且不启动模型的测试，并进行Python编译与diff空白检查。已有窗口中可能存在未保存编辑，保持原窗口，保存后重开读取恢复配置。

## 2026-10-22：默认恢复exper031腕驱动并提交exper034

- 用户要求恢复默认腕驱动，并保留十指任务点回退3mm、100%实际激活和1224mm布局；以指尖冻结归档4f37d650…96f74为父本，只改 `assets/drive_settings.json` 的TX/TY（300/34.15259872981849/±50N）和Pitch（8/0.5/±5Nm）数值及来源文字，不恢复旧阻尼算法或旧物理。参照 `results/exper033/launch_provenance.json` 中exper031数值。更新统一入口归档/SHA、对应驱动测试、奖励总表当前驱动、README与AGENTS。
- 同步此前用户要求的无悬浮标签显示实现 `preview_fingering.py`、导出CLI/worker和finalize流程，补入冻结包遗漏的交互查看器与本地启动器；保留目标键颜色/触键高亮/声音及本地3GiB保护。此显示修复不改变训练数学。新增说明 `docs/exper034_default_wrist031_20261022.md`。
- 远端CPU验证：独立 `/home/houj/projects/pianist_exper034_20261022`；97项Composer/驱动/指尖/布局/导出合同测试及13项learner环境CPU测试通过（10个子检查）；3项历史布局父本对照因未设置专用旧父本跳过、1项要求Slurm的Composer激活边界测试跳过，当前布局/激活由本轮独立runtime断言验收；实际runtime与父本相比仅4组驱动数组改变，其余176组物理数组、十指位置和八指耦合合同/输出一致，十指9组姿态的3mm回退、100%激活和1224mm布局通过，20子步状态有限。完整日志及SHA见 `logs/wrist_default031_20261022`。
- exper034预算19532向量步、512环境、10,000,384经验；保留原0.25更新/经验、batch1024、双卡分工与学习器TF32，seed42，从头训练；乐谱沿用exper033已批准015-5修订v2（SHA b0ce7e52…a8222a）。只追加输入/输出参数，实际作业ID及状态由 `results/exper034/submission.json` 记录。训练终态由本地后台watcher自动best.pt→Slurm导出→同次回放/原谱对比→回传与SHA验收；未完成回放验收不标记完整交付。
- 已于2026-10-22 23:42提交Slurm **9007**，随后获得双A100进入RUNNING，基础GPU检查通过；后台回传PID **959502**已存活并记录首次真实Slurm查询。冻结源码SHA `06f469d71de00c516555b521fee5ea6939ae1af8c82ab310d71d8c56c2afe98f`，本地/远端一致。完整奖励已在CPU提交前检查打印；实际训练启动后仍打印并回传正式奖励启动日志。
- TODO：新策略F1/接触表现待本轮实测，本轮同时包含先前批准的几何/布局/激活更新，不称单因素对照；硬件能力仍未标定。既有exper033及其导出继续使用原冻结来源。

## 2026-10-22：exper033速度/加速度−5专项定位（实测排队）

- 核对exper033完整config、冻结rewards.py散列和976条训练记录；该项全部抽样均值−5。明确手指/腕旋转为最终目标导数，TCP为限幅前请求相对实际下发历史；20Hz下TCP加速度需求残差5.225mm可扣满，腕Pitch正常硬限不可能单独扣满。
- 新增 `results/exper033/motion_penalty_diagnostic_20261022` 的只读诊断、来源校验和后台回传脚本；通过原部署scripts/hpc/submit.sh单GPU提交8974，冻结归档714a9b5c…6528d1be，best.pt训练19000步。记录每通道和两类导数，要求复算逐样本等于原罚分；未改训练源码/奖励/目标或当前默认归档。
- 本地语法/散列检查通过；GPU实测受资源排队阻塞，不能声称已经定位具体责任关节。watch_diagnostic.py已启动且真实查询PENDING，完成后回传/校验并更新同目录report.md；实际结果以该报告、attribution.json与watch_status.json为准。未新增训练。

## 2026-10-22：方向A，十指任务点向掌根回退3mm

- 用户选择图示A。按 `models/hands/fingertip_geometry.py` 的URDF末节方向和原始支撑点，将左右十指任务点回退0.003m；`tsensei_hand.py` 的原 `*_tip` 同步奖励/观察/IK，新增透明 `*_distal_support` 保留原几何基准，`mjlab_env/finger_coupling.py` 改读该基准，防止位置调整意外改变耦合方向。网格、惯量、驱动、关节限位、reset角、琴键与学习预算不变。
- 参考当前100%激活归档（91b92107…5426a，含1224mm布局），未使用exper033弱腕归档作默认。新归档 `logs/fingertip_setback_20261022/source_fingertip_setback3mm_v1.tar.gz` 及SHA切入统一入口，仅更新归档路径/校验值；已运行exper033不变，本轮未提交训练。新增说明 `docs/fingertip_setback_20261022.md`，直接修改奖励总表当前指尖定义及README。
- 验证：volc-a32独立目录 `/home/houj/projects/pianist_fingertip_setback100_20261022` CPU验收退出0；28项测试、10个子检查通过；前后180组编译物理数组SHA相同；八指耦合合同及9组输出完全相同；十指9组姿态回退3mm，世界位移误差≤9.72e-17m；MuJoCo3.11.0加载和20物理子步有限状态通过。命令、退出码、日志、源码来源及完整性记录见上述logs目录。本地仅编辑/语法/文件散列检查。
- TODO(V32-07)：回退点不等于硬件标定的指腹中心，也不保证新策略只用指腹；未训练或复评F1。旧checkpoint继续用其冻结源码，不静默替换。

## 2026-10-22：1224mm理想键盘黑键布局图

- 用户选择总宽1224mm、白键宽22.5mm，并指定黑键宽12mm。按52白键/51缝求得均匀键缝1.058823529mm、白键节距23.558823529mm；沿用已说明的理想5/7等分中心定位，双黑组中心距28.270588235mm、三黑组26.924369748mm、跨组41.396218487mm；黑键边缘净距分别减12mm。
- 新增 `logs/piano_layout_1224_20261022` 的PNG/SVG、脚本与完整键位JSON。明确A#0周期延伸、前后尺寸未指定，非DIN/PX-S1100实测；未修改训练或源模型。
- 验证：52白键总边界1224mm、36黑键计数断言及PNG视觉检查。参考本轮用户尺寸和前轮理想构造公式；TODO实琴布局仍需测量确认。

## 2026-10-22：用户Piano.obj测量及22.5mm白键尺度标定

- 参考用户 `/home/hj/下载/Piano.obj`，分离52白键/36黑键连通组件，远端volc-a32 `/home/houj/projects/user_piano_obj_20261022` 执行标准库OBJ静态分析（退出0），本地/回传source_sha256一致；原始结果、日志与脚本见 `logs/user_piano_obj_20261022`。无物理构建/仿真。
- 用户补充单白键宽22.5mm，采用52个白键X包围盒宽度中位数对应22.5mm，按统一比例换算全部尺寸；保留模型内部键宽差异，未缩放源OBJ或替换训练模型。输出毫米PNG/SVG、标定JSON及88键/3828组横向中心距和投影边缘距JSON；保留未标定版本。
- 验证：连通组件计数、源码SHA、键对组合数及图像视觉检查。缺少Piano.mtl不影响几何；颜色依据零件名称重建。TODO：物理单位仅由一个用户尺寸标定，其余实物尺寸未独立复核；OBJ无关节，无法确认按键行程；投影距离非三维最短距离。

## 2026-10-22：NVlabs局部五黑键全部距离标注

- 参考用户截图及 `/home/hj/piano_reference_exports/nvlabs_20261022/geometry.json` 的已回传远端静态数据，为B0～B4全部10组键对标注外观包围盒横向中心距和投影边缘距离，新增 `logs/nvlabs_black_key_distances_20261022` 的PNG/SVG、脚本及来源SHA/距离JSON。跨键跨度不称可用间隙，投影距离不称三维最短距离；未使用当前训练模型数据。
- 验证：5键组合计数10组，PNG视觉检查通过，本机只对既有记录CPU绘图。未修改当前模型、训练参数或归档。
- 补记上一轮因/data满而未能写入的记录：NVlabs提交20b02f3380b7872bf262ac26859879d1c1385671的URDF/OBJ/MTL及LICENSE共78文件已保存于 `/home/hj/piano_reference_exports/nvlabs_20261022` 并打包；Git blob/SHA256校验，volc-a32独立目录 `/home/houj/projects/piano_reference_nvlabs_20261022` 执行静态分析退出0，回传geometry.json。39 links、36转动关节（21白/15黑）、活动键缺少独立惯量。未做物理仿真；TODO：实物尺度、完整动力学与DIN符合性均未验证。当前/data空间已恢复可写。

## 2026-10-22：DIN8996待测规格图与钢琴URDF检索

- 用户要求查找DIN8996完整尺寸图并按此前版式导出，缺项标待测、不用仿真数据；后续补充寻找钢琴URDF。核对DIN官方目录、Bechstein安装说明及Steinway历史信函英译，未取得完整标准原图。新增 `logs/din8996_reference_20261022` 的PNG/SVG、自绘脚本、来源记录和null尺寸JSON；已知厂家转引单独列参考框，不冒充标准值或卡西欧实测值。
- 找到NVlabs/sim-web-visualizer的piano URDF及配套网格，静态XML核对36个活动关节、39 links、活动键缺少独立惯量；不是已校验的88键或DIN/PX-S1100模型。链接和检查结果存sources.txt；未导入、替换或运行外部模型。
- 验证：本地轻量CPU绘图，PNG视觉检查，字体缺字修正；无MuJoCo构建、仿真、训练。TODO：取得授权标准全文后核对尺寸/公差，实测PX-S1100，候选URDF进一步物理与几何验收须遵循远端规则。

## 2026-10-22：当前钢琴尺寸与全部键对距离图

- 用户要求在一张图标注钢琴全部距离。参考当前 `models/piano/piano_constants.py`、`piano_mjcf.py` 与统一入口绑定的 `source_servo_gains_v1.tar.gz`，静态解析尺寸及键位，新增 `logs/piano_dimensions_20261022/draw.py`、PNG/SVG尺寸图、88键坐标CSV、3828组键对距离CSV及来源SHA记录。
- 图包含全键盘、八度俯视、侧视、黑白键横截面及任意键对公式，区分中心距、边缘净距、投影重叠和顶面高度；注明A#0例外以及简化盒体相交、键间碰撞禁用。单位mm，限定未按下qpos=0；非实物测量，不描述动态按下状态。
- 验证：冻结归档SHA符合统一入口，两个源文件与归档逐字节一致；88键/3828键对计数、关键尺寸断言和PNG视觉检查通过。仅轻量静态计算与CPU绘图，未加载或编译MuJoCo、未改训练源码或参数。TODO：实物尺寸及动态姿态距离需另有测量/状态输入。

## 2026-10-22：左手控制日志速度与加速度分析文档

- 用户要求将侧会话已完成的日志分析保存成文档。参考 `/home/hj/下载/hand_control_20261022_121139_828750.jsonl` 的成功downlink、按command_id关联的h7控制周期、telemetry设备时间和停止事件，新增 `docs/hand_control_velocity_acceleration_20261022.md`，并在README添加索引。
- 保存原始日志SHA256、6,587帧指令与8,224条反馈的统计范围、各手指及14个运动目标关节的速度/加速度峰值、主机时间抖动示例、60ms局部拟合反馈统计与小指PIP越界停止说明。明确指令rad/s、实测行程mm/s及各自加速度的区别。
- 验证：核对原始文件SHA及上一轮逐字段计算结果，回读文档与链接；本次仅文档变更。TODO：匹配本次SDK的行程—角度反算和已标定硬件限值尚未核验，不据本文判定实机超限。原始日志、训练和回放任务保持不变。

## 2026-10-22：所选exper027平滑轨迹手根高度75→85mm

- 用户在侧会话选中28.3秒yaw15/z075平滑轨迹并要求将75mm改为85mm。独立产物results/exper027/right_arm_yaw15_z085_smooth_20261022；不改原75mm版本、不改主会话的59.89秒CSV或物理触键仿真。0–2s原转向段逐元素保留，2–4s以原五次平滑进度移至手根[140,150,85]mm，4s起Z目标固定0.085m；全段XY/旋转目标、时长和手指轨迹保留。
- 参考logs/right_arm_yaw15_z075_20261021的DLS/硬约束样条及right_arm_prep_smooth_20261022的准备段jerk平滑和finalize；新增独立logs/right_arm_yaw15_z085_smooth_20261022。volc-a32单线程CPU重新解算2s后的机械臂，冻结0–2s支撑系数，保留SDK/URDF位置、连续速度/加速度和J6/J7耦合硬限，随后仅平滑2–4s。无策略、训练、物理仿真或硬件执行。
- 全区间硬限通过，峰值速度/加速度上界分别为限值69.135%/99.242%；准备段峰值加速度37.056°/s²、jerk189.590°/s³。移位段手根最大误差0.000345mm；演奏段位置最大误差1.062429mm、RMS0.072115mm，朝向最大误差0.624643°。第4秒实际手根约[140.000001,150,85]mm，法兰约[161.620310,145.994673,65.229745]mm，均为模型钢琴坐标。
- verify.py确认0–2s机械臂CSV q/v/a逐元素相同、整份500Hz手指CSV字节相同、目标XY/旋转及时间轴一致，200Hz臂5661行/500Hz指14151行；远端源码/输入及回传文件SHA核验。本次交付CSV、样条、跟踪图、报告与ZIP；未更新旧回放视频，也不宣称物理按键效果。命令、PID、来源及验收见本轮execution.json/verification.json。

## 2026-10-22：琴键激活与按键奖励95%满分行程对齐

- 用户要求激活行程与key_press_reward一致。参考当前pitch_range冻结归档、piano.py、task_logic.py、物理子步采样、CPU/Torch场景判定、preview_audio/preview_keys及录制对比图。将固定0.5°余量替换为白黑键统一 `clip(qpos,0,qmax)/qmax≥0.95`；共享正确键容差0.05常数，连续Gaussian/0.6与0.4比例、权重、琴键物理行程、指法目标及20/200Hz保持。
- 同步新字段key_activation_fraction至场景manifest、音频/分析记录及所有活动消费者；旧字段/旧奖励合同拒绝，合同physical_key_fingering_exact_activation95_v1。旧冻结归档/配置隔离保存并保留历史依赖，不重新计算历史结果。可交互包携带新共享模块；同步奖励唯一总表与README。
- 首次Slurm8417测试选择引用了归档未包含的旧导出测试，收集阶段退出；修正选择后Slurm8419其余117项/13子检查通过、2项条件跳过，但两个边界测试揭示除法舍入可把恰好95%判为未激活，已改用等价的实际角度≥0.95×最大角度直接比较，并保持阈值相邻浮点样本回归。
- 远端volc-a32独立目录 `/home/houj/projects/pianist_key_activation_20261022`，Slurm8429单A100复验119 passed、13 subtests passed、2 skipped（冻结包不含外层统一入口脚本，对应两种参数组合）；包括88键Composer边界、NumPy音频/高亮、Warp CPU/CUDA及相邻浮点样本、真实20/200Hz完整模型子步奖励、CUDA Graph和局部reset。测试冻结SHA `33834cc6d5c039976423ba09ff4a624f408059352c30f74915283b7107e74d5a`；最终归档仅同步文档，全部非文档成员逐文件与验收归档一致，SHA及验证记录见logs/key_activation_20261022。统一入口绑定新归档；旧归档在deprecated/reward_modes/key_activation_0p5deg_20261022保留硬链接及历史路径。未启动训练、未改历史实验。


## 2026-10-22：zx/1021输入CSV在pianist MuJoCo中回放

- 用户明确选择当前pianist MuJoCo右臂＋右手回放。读取/home/hj/TsenseIianist_platform/trajectory/zx/1021/right_arm_200Hz(10).csv与right_hand_500Hz(9).csv，分别11979/29946行，均0–59.89s；列名、有限值、200/500Hz时间间隔及两端一致性核验，原文件保持不变。来源SHA见logs/zx1021_csv_replay_20261022/input_audit.json。
- 新增同目录build_replay.py及独立viewer.py，参考此前已验收kinematics.py、右M6 SDK运动链、四孔连接件和装配MJB。基座与刚性工具关系固定，首帧法兰[201.802,157.225,100.007]mm；用1ms显示记录保留每一个原5ms臂、2ms手位置采样点，中间线性插值，不求IK、不改速、不重新规划。复用的是已编译视觉装配场景，不声称重新验证10月22日最新碰撞/奖励配置。没有原谱/实际琴键/音频输入，因此琴键静止、不配其他实验声音或F1。
- volc-a32独立目录单线程CPU进程3091868、87.506s、退出0，生成59891帧；每帧MuJoCo手根与SDK完整变换最大元素差2.67e−15，原CSV位置采样点保留、模型关节位置范围诊断无越界。首次执行因输入同步未结束而缺sdk.cfg，未加载模型即退出；待同步完成并逐项SHA核验后重跑成功。无物理积分、策略推理、GPU计算或硬件连接。
- 回传至results/csv_replays/zx1021_20261022，5项远端SHA含复用MJB核验通过。本机按已授权MJB回放例外，用2300M/3G/禁swap独立scope隐藏窗口完整检查31秒（2倍检查速度），1701帧、到达59.89秒，memory.events high/max/oom均0。随后启动可见交互窗口，按原速自动播放；启动PID、日志、命令见本轮execution.json/live_replay.pid。支持暂停、重播、时间跳转及视角切换；未对输入轨迹作实机、碰撞或动态硬限验收。

## 2026-10-22：Palm_Base_Pitch范围交互预览

- 用户要求打开预览观察当前0～0.5rad范围。基于Slurm8367验证的冻结源码，在volc-a32独立目录以CPU构建无臂/Yaw15°/Z0.090m模型，导出版本匹配MJB和101个双侧Pitch静态姿态，未做动力学积分。
- `results/pitch_range_preview_20261022/`新增只读姿态viewer及受限本地启动器：显示rad/deg、底部滑条、Home/End上下限、方向键0.01rad步进、空格往返扫描和三视角；本地仅读取预导出MJB/姿态，以独立3GiB/systemd scope和NVIDIA渲染。示意范围不代表物理演奏效果；不改变训练配置。

## 2026-10-22：后续训练Palm_Base_Pitch扩展至0～0.5rad

- 用户要求修改训练脚本范围。参考exper031实际config、统一入口、source_runtime范围缩放合同、手部常量及direct_joint_v1动作配置；range-scale只能缩小，故同步模型forearm_pitch与左右动作通道上限为0.5rad，入口scale保持1。带臂Pitch仍结构固定，速度/加速度与奖励公式、学习预算保持不变。运行中的exper031保持其冻结0～0.3rad配置，不重启或修改该实验。
- 同步奖励唯一总表和README；新归档与验证记录位于logs/pitch_range_20261022；Slurm8367 COMPLETED/0:0，1项真实物理验收通过（43.76秒），检查模型关节及最终动作范围、八指耦合/十指MCP、普通/CUDA Graph执行及子步奖励/局部reset。统一入口已切换source_pitch_range_v1.tar.gz，SHA256 `3242be4c205705d7709a9a3900403a4f13c32461fea1f62b03b354514bb97ee6`，回传散列一致；未提交新训练。

## 2026-10-22：exper031漏键复查与第二版候选指法（侧会话）

- 依据exper031 best.pt同次200Hz录制、config/源码SHA、原谱、score_coverage及实际琴键对比，核验当前合同physical_key_fingering_exact_v1，连续/控制网格/物理样本同指多键冲突均0；原exper027实际转换39帧冲突已不再作为当前指法奖励目标。此结论不等于实际漏键或重复音重触发已解决。
- 本回放micro-F1=0.9133676549，269音中16全漏、23覆盖不足50%；右食指11全漏、597/1581漏键样本。47组同键近邻间隔0.5～5ms中45组在5ms采样上未松键，含8组已换指；200Hz评分不增加20Hz动作频率。未将覆盖率当作指定手指接触证据，未声称帧间绝无短暂释放。
- 独立分析目录`results/exper031/fingering_revision_v2_20261022/`，远端`volc-a32:/home/houj/projects/pianist_exper031_fingering_v2_20261022`。9份本轮录制输入逐一SHA验证；只在远端CPU载入同次MJB/运动学并静态IK，未运行策略、物理积分或训练。使用本轮MCP0～1.4rad、耦合45～80°；更新旧分析工具副本的60°硬编码并显式恢复mocap。规划启发式和权衡记录在revision_report.json，非新策略性能证明。
- 选定`015-5_fingered_repaired_v2.proto`只改7个note.part（右5、左2；0起索引72/79/113/133/154/226/229），SHA256 `b0ce7e52c0eb211e009c463b168eb0cf6f7f14cac66fcadf87fbe7d790a8222a`。改指直接涉及2个全漏、3个低覆盖音，1个邻音配合、1个指定拇指明显偏离；其余14个全漏音缺少更好替代证据而保留。第80音（1起）由右中指改小指，旧覆盖100%，静态距离0.9→11.3mm，作为避免邻音占指的显式代价，需重点复验。
- 验证：269音符恢复part后protobuf字段逐字节一致，左右手/音高/速度/全部原谱时值不变；200Hz均有目标样本，同指多键及不足50ms不同键同指冲突均0；静态角度最大求解误差0.0175°（验收容差0.1°）。保存分析HTML/图、逐音CSV、输入/产物SHA、远端日志和执行记录。候选尚未动态评估，静态IK放松PIP/DIP耦合解，不能证明实际可达或无碰撞；保留未选中候选追溯。
- 本任务只新增分析/候选及本日志；未切换统一训练默认谱、未修改奖励算法/总表/当前源码归档、未启动训练。内部使用说明见该目录README.txt，未覆盖exper031训练输入或原回放产物。

## 2026-10-22：exper031新指法训练与回传交互模板

- 按用户本轮授权，以修正版015-5 proto（SHA256 `9f835c48cfc86702876ecdd6a7be06ab0a6cf5f73787dffab58f8c7bf435c3a3`）、Yaw15°/Z0.090m，经统一入口启动独立exper031；无臂、从头训练、双A100/512环境/20Hz控制/200Hz物理，统一入口19532向量步＝10000384条经验、每经验0.25次更新。原谱时值不改，八指耦合45～80°、十指MCP0～1.4rad。
- 冻结训练归档仍为`logs/control_ranges_20261022/source_exact_fingering_ranges_v1.tar.gz`（a6a88b22…4122d）；远端`volc-a32:/home/houj/projects/pianist_exper031_20261022`，Slurm8184，输出`results/exper031-20261022-015202-992900`。实际config、嵌入谱字节、范围、预算与频率已核对；完整奖励启动打印存在，采样已产生metrics。训练尚未完成，最终性能/完整预算/回放验收待后台流程。
- 提交时配置best.pt完整有声交互回放、同次实际琴键/原谱PNG及SHA回传；本地独立watch_training_transfer.py进程127787（transfer.pid）已确认存活且首次真实查询RUNNING。归档中的旧导出辅助工具不足，独立部署同步当前导出/交互工具并记录`results/exper031/delivery_tools_sha256.json`，不替换训练作业8184冻结源码。
- 用户随后指定`results/exper027/right_arm_prep_smooth_20261022/replay/launch_local.sh`生成方式。参考该launcher和viewer.py，修改`interactive_recorded_replay.py`、`package_interactive_replay.py`、`launch_local_recorded_replay.sh`：生成根目录model.mjb/viewer.py、本地受限NVIDIA启动、拖动时间条、1～4视角、A切换真实手根/法兰坐标轴、1秒跳转、播放/重播/倍速与HUD。保留replay.py兼容入口及旧目录加载；不复制旧28.3秒、4秒准备段、旧姿态或离线机械臂轨迹。新包仍完全使用本轮原始物理轨迹、真实按键/声音；无手指/琴键悬浮标签。
- 验证：远端导出/注册谱/回传66项通过；本地按已授权录制回放例外，用exper027既有MJB/轨迹运行新UI，2秒109帧及四视角/坐标轴循环110帧通过，NVIDIA RTX3050 render-check通过，oom/oom_kill均0，未编译XML或仿真；记录在`results/exper031/replay_ui_check.log`。新训练尚无最终录制，故本轮完整回放验收由训练终态后台流程执行。已同步README。

## 2026-10-22：指法奖励按原谱200Hz物理子步评分

- 用户明确授权修复训练代码：指法按原谱时间评分，当前与未来目标分离，保持20Hz控制、200Hz物理、原谱时值及学习预算，验证后冻结源码/SHA，先不训练。参考当前统一入口、活动奖励JSON、exper027原始015-5 proto（8bbf0c35…964db）、原谱269音符/39处实际控制网格同指多键证据，以及当前掌根碰撞修复归档70f4d255…64ccb8；保留该归档的模型/惯量/碰撞修复，没有改用上一轮候选指法。
- `mjlab_env/exact_score.py`按原谱半开区间构建每个物理样本的每指唯一键（闲指−1），原连续重叠仍拒绝。`substep_rewards.py`/`scene.py`保存积分后实际指尖和键目标几何；Euler无硬限/无臂路径仅刷新运动学，不额外积分或求接触。`task_logic.py`/`rewards.py`先对该子步当前用指Gaussian距离取均值，再对10个子步平均，空子步0；容差1cm、margin10cm、weight=1及原目标点保持，不先平均几何，不给未来目标分数。Composer参考指法回调也改为子步累积并清空reset历史；Composer按键评分仍不是完整GPU参考。
- `task.py`当前goal/用指按控制时刻精确点采样，未来50维按原谱每指最近起音独立预告，忙指也能看见下一音；原176维起落倒计时保留。原floor/ceil网格继续服务踏板、传统控制帧F1及闲指20步保护，不进入指法奖励。真实音高、指法、起止时间均不改；269音符完整覆盖，旧39处量化冲突不再成为同时指法奖励目标。
- 奖励合同替换为`physical_key_fingering_exact_v1`，事件合同`exact_note_fingering_substeps_v1`；删除旧指法mode、提前/分阶段评分模块和三个专属performance/CLI开关，即使false也拒绝。旧配置/算法/测试归入`deprecated/reward_modes/fingering_endpoint_20261022/`，旧完整归档另存硬链接并保留原历史依赖路径；新合同不迁移旧checkpoint。同步奖励唯一总表对应公式/字段、README、AGENTS、入口和监控元数据。
- 远端独立工作目录`volc-a32:/home/houj/projects/pianist_exact_fingering_20261022`；CPU 113项及131个子检查通过（2项GPU条件跳过），原生Composer物理回调1项通过。Slurm8166通过原谱完整目标生成、实际子步奖励、20/200Hz、硬限/耦合、局部重置，以及无硬限时积分后坐标的独立MuJoCo复算和CUDA Graph一致性；8169通过当前/未来观察与局部重置。先前8152/8166的失败均为旧验收夹具：漏拷mocap、误开胶囊净空、读取不存在的reset属性；修正夹具后对应三项全部通过，保留全部失败日志。8169使用已记录SHA的更新测试文件，运行时代码与8166冻结源码一致。
- 最后入口核对发现同一工作区侧会话已更新八指耦合45°～80°及十指MCP目标0～1.4rad。保留并合入三个配置文件；两份动作配置各10个MCP及新八指范围静态检查通过。合并后CPU仍113项/131子检查通过；Slurm8174（COMPLETED/0:0）一次通过全部三项真实物理测试，运行108.78s，覆盖新的动作范围/耦合、子步评分、积分后几何、图一致性和观察重置。最终归档运行时代码/配置与该成功作业逐文件一致，原URDF/网格/惯量保持；没有启动训练。
- 本任务验收归档`logs/exact_fingering_20261022/source_exact_fingering_v1.tar.gz`与同名`.sha256`保留。最终入口尊重侧会话随后发布的`logs/control_ranges_20261022/source_exact_fingering_ranges_v1.tar.gz`（a6a88b22…4122d），独立审计确认其1690个运行时/配置文件与Slurm8174逐一相同，并修复并发切换期间的路径/SHA配对；归档manifest/来源/执行记录/最终代码与已验收源码一致性检查同目录保存。启动参数拦截测试不提交作业，核对双卡0采样/1学习、512环境、batch1024、19532向量步、每条经验0.25次更新、20/200Hz保持；其他奖励数值保持原JSON及入口覆盖。
- 本轮只提交GPU验收，没有启动训练、分配新exper编号或修改历史实验。TODO：新合同的真实漏键率、重弹质量和收敛效果须日后从头训练测量；20Hz动作未改变，去掉矛盾奖励并不保证所有快速换键可达。旧回放中的floor/ceil目标标注仍是其冻结诊断，不能冒充新子步评分目标。

## 2026-10-22：十指MCP训练目标统一为0～1.4rad

- 用户明确要求MCP范围统一改为0～1.4。参考两个活动动作配置、`mjlab_env/action_mapping.py`与双侧活动URDF；修改`configs/actions/direct_joint_v1.json`和`wuji_cartesian_v1.json`的全部十个MCP通道（含拇指）。策略归一化范围与其他通道不变，URDF本体限位仍0～1.57rad，八指末节耦合仍45°～80°。
- 同步奖励总表硬控制表、README、AGENTS。静态核对两个配置各10个MCP为[0,1.4]，逐关节处于对应URDF限位内，所有非MCP通道逐字段不变；未启动训练或本地物理仿真。
- 用户随后要求更新归档：以主流程最终指法奖励归档8143323c…6d05c6为基底，只叠加两个动作配置、新耦合范围和独立验收夹具/说明，生成`logs/control_ranges_20261022/source_exact_fingering_ranges_v1.tar.gz`；统一入口切换到新归档及同目录SHA256。原归档与历史实验保留。
- 新范围在`volc-a32:/home/houj/projects/pianist_control_ranges_20261022`独立目录验证，Slurm8172单卡验收通过（1项，43.74秒）：原始269音符、20/200Hz、十指MCP实际解析范围、45～80°耦合启动、普通/CUDA Graph执行、子步奖励与局部reset。最终运行时代码/配置逐文件对照该作业快照；归档manifest及回传SHA检查记录在同目录validation.json/provenance.json。没有启动训练；带臂动作JSON同步且限位静态检查通过，本次物理验收采用默认无臂模式。


## 2026-10-22：八指耦合范围调整为45°～80°

- 用户在角度示意说明后明确要求将60°～80°改为45°～80°。参考 `mjlab_env/finger_coupling.py` 的末节方向相对静止键面定义，以及原范围JSON和统一入口。
- 新增 `configs/finger_coupling/angle_ranges_45_80.json`，左右八个非拇指均为[45,80]；统一入口以项目绝对路径传入，避免冻结归档内相对路径读取旧设置。保留旧范围文件供历史实验追溯；同步奖励总表、README、AGENTS。没有改动耦合算法、学习预算或其他数值。
- 验证：JSON八指名称与原配置完全一致，范围全部[45,80]，满足0<下限≤上限<90；统一入口bash语法检查通过。随后已按用户要求与MCP范围一并纳入新归档并通过Slurm8172验收，见上节及`logs/control_ranges_20261022/`；未启动训练。


## 2026-10-22：exper027 指法漏键分析与独立修正版

- 用户要求根据当前训练结果分析按不到及同指重复按键并生成新指法。参考 exper027 原始 `input_sources/015-5_fingered_edited_end.proto`、完整 config、best.pt、score_coverage、同次200Hz实际琴键/物理录制/指法注释，以及冻结归档内 midi_file、piano_roll、score_validation；不采用后续加臂/改姿展示作为新物理证据。
- 新增离线工具 `scripts/repair_recorded_fingering.py`。原谱269音符连续时间同指重叠为0，实际20Hz转换39处同指多键；原始floor/ceil区间41处，其中2处被原同音重弹释放处理遮蔽。逐音指标按原谱减起点0.0049s，匹配已有比较图的边界容差：原回放16个完全漏键（右14/左2）、23个覆盖不足50%，仅3个完全漏键直接参与实际同指多键。原回放200Hz micro-F1=0.915322，不能当新指法成绩。
- volc-a32独立目录 `/home/houj/projects/fingering_exper027_20261022` 以单线程CPU读取SHA一致的MuJoCo3.11录制MJB，提取原指尖并逐音/逐同手五指静态筛选；固定原回放中段手腕，使用原关节/驱动范围及60–80°角度带。原计算PID1987108，求解/验证退出0；命令、最终PID、散列和日志在本轮execution及logs目录。没有物理积分、策略推理或GPU工作，没有本地加载/编译大模型。
- 最终仅改48个note.part（右41/左7）；保留全部音高、精确时值、力度、顺序、左右手及其他protobuf字段。训练网格及50ms不同键释放余量内同指互斥；真实同时按键的非拇指顺序硬约束，前后换指反序软惩罚，避免过度约束迫使远距离拇指替代。最终保留14处前后换指反序风险（13处控制目标重叠），须实测；固定旧手腕静态残差均值4.923→4.223mm仅为筛选代理。输出 `results/exper027/fingering_revision_20261022/015-5_fingered_repaired.proto`，SHA256 `9f835c48cfc86702876ecdd6a7be06ab0a6cf5f73787dffab58f8c7bf435c3a3`。
- 验收使用匹配训练归档（cc692240…570abb0）中的真实转换器：旧谱重建指法数组与原录制逐元素一致，新谱269/269覆盖、0 errors/0 warnings；本地独立回读确认只改part及原谱/配置/checkpoint指纹，远端输出SHA全部通过。交付analysis.html、逐音CSV/JSON、候选IK数组、前后PNG、README.txt、执行/验证/散列记录；PNG已目视检查。原谱、训练入口、奖励、历史checkpoint和训练归档未替换。
- TODO：新谱尚未训练/动态触键复评；PIP/DIP角度带允许的静态解不保证等于活动确定性耦合解。静态距离不是接触距离，搜索失败不证明不可达；黑键位置偏差、开头无准备段及部分低音手腕偏差不能靠换指保证解决。需继续验证抬指衔接、自碰撞、手腕运动与实际漏键率。

## 2026-10-22：指法预览增加上一组浅色残留

- **用户需求与参考**：在当前指法中保留上一次指法位置，使用原手指对应颜色的浅色。参考 `scripts/fingering_editor.py` 的指键同色、当前帧请求/响应匹配，`fingering_editor_data.py` 的原始秒值乐谱与 `fingering_editor_view.py` 的显示色表。
- **修改**：新增按谱面起音组确定的上一组已松开音符；仅残留一组，间隙保留最新已松开组，延音继续作为当前目标。旧键以原指色30%＋白色70%显示并标“上＋编号”，当前同键优先，保留同键多指分区。默认开启，可用“上一组指法残留”关闭。只叠加琴键显示；拖动/回退、指法/时间编辑及撤销时重算，异步旧响应不混入新时刻。
- **验证**：数据层6项测试通过，覆盖起音/松键边界、和弦中持续按住的音符、向后定位、指法编辑及撤销、原始proto保留；实际受限Tk界面另验当前优先、开关和异步响应，截图/记录位于 `logs/fingering_editor/previous_fingering_20261022.{png,json}`。Python编译及diff空白检查通过。原谱、远端模型/求解、训练和奖励归档均未修改；已有窗口需保存当前编辑后重开更新。

## 2026-10-22：exper027 的15°/75mm轨迹前4秒关节抖动修正

- 用户询问先调15°后的移位终点，并要求改善准备段IK抖动。确认0–2s保持手根[175.005347,160.696584,112.016617]mm调整完整朝向，2–4s移动至手根[140,150,75]mm；对应第4秒SDK法兰约[161.620310,145.994673,55.229745]mm，均为钢琴坐标。继续保留2＋2秒与原速演奏。
- 参考上一版logs/right_arm_yaw15_z075_20261021的kinematics/plan_fixed_time/finalize、SDK/URDF与已验收trajectory_and_spline。旧目标只最小化末端误差和微弱参考偏差，缺少关节平滑项；第一至七轴中多个冗余方向存在明显往复，准备段峰值加速度变化率88706.773°/s³，峰值加速度600.248°/s²。新增logs/right_arm_prep_smooth_20261022/smooth_prep.py，在同一三次C2样条与笛卡尔阻尼IK拟合中加入解析三阶导数平方惩罚，固定0/2s支撑系数及所有影响t≥4s的原系数。候选逐次检查SDK/URDF位置、连续速度/加速度及J6/J7耦合硬限，越限候选缩步或拒绝，不放宽限值。
- volc-a32独立目录单线程CPU规划PID1904759、退出0。峰值加速度变化率降为189.590°/s³，RMS从10672.401降至29.610°/s³；准备段峰值速度56.522→23.976°/s，加速度600.248→37.056°/s²。模型内准备段最大手根位置误差0.000566mm、朝向误差0.000470°；全段连续硬限通过。CSV格式与全数字回读、t≥4机械臂q/v/a和实际手根/法兰逐元素一致，0/2/4秒边界一致，整份500Hz手指CSV字节一致。七轴前后对比图已查看。
- 新独立交付results/exper027/right_arm_prep_smooth_20261022，原版本保留；不修改训练、奖励、归档、演奏速度或指法。完整回放更新脚本位于logs/right_arm_prep_smooth_replay_20261022；远端CPU进程1914296生成全部5661帧，模型手根位姿元素差≤2.56e−15、原同次实际按键匹配，无物理积分；本机按MJB回放例外在2300M/3G/禁swap scope中完整播放30秒至28.3秒，1351帧、音频进程通过（测试静音）、memory.events high/max/oom均0。交互回放、原同次200Hz实际按键/未量化原谱对比PNG与事件、来源记录已回传并核对SHA。按键和声音仍是原exper027物理录制参考（micro-F1=0.915322），不声称新的带臂物理演奏结果。
- Slurm8085经统一入口提交完整MP4和俯视图导出，作业时限从默认24小时缩为10分钟以适应短导出；当前排队。独立本地finish_export.py后台进程每60秒查作业，完成后回传、核对远端SHA及与已验收交互包的motion/audio/keys一致性、检查音视频时长/完整解码并生成完整ZIP。状态见交付目录export_status.json及本轮transfer.log/PID；排队中的MP4不标记已完成。模型运动学误差不代表实机精度，未做实机、碰撞或扭矩验收。
- 同轮用户补充允许空闲H100训练，已写入AGENTS.md，未改动双卡/预算默认或新建训练。当前A100集群仅有TenseiNode1；另查SSH别名h100可连接至Tsensei2，Slurm登记8张H100。普通SSH的NVML查询返回Unknown Error，GPU资源以Slurm分配查询为准：CfgTRES=8、AllocTRES=8，目前无未分配H100；不把NVML查询失败称为设备故障或计算验收。

## 2026-10-21：exper027 演奏段改为局部Yaw15°、手根Z75mm

- 用户要求第4秒后的palm_base_yaw改15°、palm_base_z改0.075，并要求先告知原Z后继续。已答复原配置height=0.060m、录制初态约59.999976mm；新目标提高约15mm。保持先转向后移位：0–2s在原初始手根位置调整到新的15°完整朝向，2–4s保持朝向移至[140,150,75]mm，4–28.3s原速演奏。
- 参考冻结录制scene.xml、physics_trace/wrist_poses及当前models/hands/tsensei_hand_constants.py、mjlab_env/bundle.py的Yaw烘焙定义：右手Yaw是局部−Y轴上的末级腕旋转。将原完整旋转右乘Rot(局部−Y,−15°)，保留记录Pitch；独立重建原烘焙模型和新烘焙Pitch轴，逐帧验证两种算法一致。演奏段Z目标设为绝对0.075m，XY目标逐元素不变，全部500Hz手指CSV字节不变；不把局部Yaw当作钢琴世界Z轴旋转。
- 新独立目录logs/right_arm_yaw15_z075_20261021在volc-a32单线程CPU执行连续DLS与C2样条硬约束规划，固定基座、SDK法兰/四孔件/右手刚性关系和图片初始七角不变。连续速度/加速度/位置/耦合硬限通过，峰值速度/加速度占限68.487%/99.237%；演奏段最大位置误差1.05725mm、RMS0.07152mm、最大朝向误差0.62424°。两段准备最大位置误差分别0.000306/0.000670mm。
- CSV格式/首态/时间轴/有限值、原谱XY与Pitch保留、15°与75mm目标及文件SHA独立验证通过。交付results/exper027/right_arm_yaw15_z075_20261021，200Hz臂5661行、500Hz指14151行、28.3s不变。Slurm7969 COMPLETED/0:0导出完整有声MP4、交互回放及新第4秒俯视图；5661帧位姿最大元素差2.56e−15，源琴键逐帧匹配，音视频均28.3秒、解码/远端散列通过，俯视图已查看。本机受限scope完整播放验收至末帧，记录见local_validation.log；第4秒实际手根约[140.00000091,150,75.00000002]mm。琴键/声音/对比图来自原同次录制（200Hz F1=0.915322），不视为新姿态物理演奏。未修改训练参数、奖励、归档或原实验；不声称新的手指触键质量或物理安全。

## 2026-10-21：第4秒右手五指静态对齐Z=22.5mm

- 用户要求调整先转向后移位版轨迹t=4秒的手指角度并导出。读取right_arm_rotate_first_20261021的原手指CSV、同次回放MJB/motion及t4_top_view报告；全部来源SHA与第800帧手根变换核验。固定其他qpos和手根/机械臂，以每指四关节相对原姿态的平方改变量最小为目标，在模型位置限位内约束对应tip站点的钢琴坐标Z为22.5mm，XY自由。
- 独立产物results/exper027/right_fingertips_t4_z22p5_20261021含求解脚本、单行20关节弧度CSV、前后弧度/度数对照、指尖坐标、静态状态及来源报告。volc-a32单线程CPU进程1671691退出0，无GPU/物理积分；五指最大高度数值误差9.24e-14mm，20关节限位通过，其余qpos逐元素不变；5项远端输出回传SHA与CSV数字复核通过。原完整轨迹未修改，未重新规划衔接或验证碰撞/扭矩/动态硬限；旧轨迹验收不覆盖此次静态改姿。说明与执行记录见独立产物README.txt、execution.json。

## 2026-10-21：最新轨迹第4秒右手与右臂俯视图

- 按用户要求读取最新先转向后移位回放的motion.npz第800帧（t=4.000s），不重新求解关节或移动装配。新增logs/right_arm_t4_top_20261021/test_render.py，使用同包MJB与状态，逐项核验模型/轨迹/元数据SHA及手根/SDK法兰位姿。
- volc-a32经统一submit.sh单卡Slurm7938 COMPLETED/0:0，1 passed；相机沿钢琴+Z向−Z、正交投影、1800×1400，完整展示右手、连接件和右臂，无物理积分。PNG与报告回传至results/exper027/right_arm_rotate_first_20261021/t4_top_view，SHA和人工看图通过。旧轨迹、CSV、视频及完整包保持不变，本次新图单独交付。

## 2026-10-21：exper027 准备段改为先调整30°再移位

- 用户要求交换准备顺序。新建logs/right_arm_rotate_first_20261021与独立交付results/exper027/right_arm_rotate_first_20261021：0–2s保持初始手根位置[175.005347,160.696584,112.016617]mm并调整到原训练30°完整朝向，2–4s保持该朝向移至[140,150,60]mm；原始法兰初态、七轴角、刚性四孔安装和固定基座不变。参考前轮kinematics/plan_fixed_time/finalize与已验收trajectory_and_spline。
- 仅在远端volc-a32以单线程CPU重新求前4秒DLS参考及C2样条。固定所有影响t≥4s的原样条系数；移除这些不可变系数对应的重复99%优化缓冲约束（原轨迹已在实际硬限内、加速度上界99.241%），保留对全时段实际SDK/URDF位置、速度、加速度及J6/J7耦合的完整证书，不放宽硬限。新准备段最大位置误差0.001136mm，转向段最大朝向误差0.000540°；连续硬限通过。
- 本地轻量数值/散列核验确认t≥4s机械臂q/v/a、目标/实际手根与法兰逐元素完全相同；整个手指500Hz CSV字节相同。200Hz机械臂5661行、500Hz手指14151行，时间仍28.3s，列名/首态/时间轴/有限值和远端散列通过。原演奏段误差最大0.98833mm、朝向0.60807°不变；未重新训练或修改指法/音符时间。
- 全轨迹交互回放更新入口在logs/right_arm_rotate_first_replay_20261021；复用同一已编译刚性模型，按新数组显示先转动再移位。Slurm7928 COMPLETED/0:0、5661帧位姿核验最大元素差2.84e−15，15项导出散列通过；同源琴键/声音/未量化原谱对比图仍后移4秒，不是新带臂物理演奏。视频完整解码、音视频均28.3秒，已查看第2秒帧。交互回放在本机受限scope中完整播放至28.3秒（30秒检查、1059帧），音频进程正常（测试静音），high/max/oom均0。回放及README放在新交付目录replay/，保留原版本；完整包散列/CSV与播放证据分别保存于本轮verification、execution及local_validation.log。

## 2026-10-21：exper027 完整28.3秒右臂/右手可交互回放

- 用户纠正需要完整轨迹而非第4秒静态场景。新增logs/right_arm_full_replay_20261021的导出/回放工具，复用已交付CSV对应样条数组、SDK运动链、四孔安装及第4秒MJB；0–2s移位、2–4s调整至30°训练姿态、4–28.3s原速演奏。手指由500Hz q/qd/qdd作五次Hermite插值后在200Hz显示，机械臂读取200Hz已解算轨迹，不重新求IK、不改变硬限规划或原时值。
- volc-a32 Slurm7924完成/0:0，全部5661帧MuJoCo手根变换与已交付轨迹最大元素差2.84e−15。原录制琴键逐帧激活完全一致；声音、原谱/按键图及事件整体后移4秒。明确标注离线带臂运动学展示：原无臂物理录制的双手琴键/声音仅供同步参考，不冒充新右臂接触结果；原录制200Hz micro-F1=0.915322，本次无新物理F1。左手几何隐藏，不删除源双手按键。
- 完整1280×720/30fps有声MP4共849帧，音视频均28.300秒且完整解码通过。可交互播放器支持鼠标视角、时间条点击/拖动、播放暂停、重播、前后跳转、倍速及可选手根/SDK法兰轴；默认从0秒暂停。与静态包分目录交付于results/exper027/right_arm_full_replay_20261021。
- 本机沿用MJB录制加载例外，固定2300M/3G/swap0保护；30秒完整播放验收1062帧、到达28.3秒，音频解码/播放进程正常（测试音量0），memory.events high/max/oom均0。首次进度条调用参数不符已按实际MuJoCo API修正再验收。画面发现源包带旧悬浮标签，改用当前活动无标签preview_fingering.py并通过Slurm7925 COMPLETED/0:0单独重渲染；保留颜色/高亮，5秒显示复核172帧通过，最终演奏帧已查看。15项远端文件SHA全部验证，最终作业与包散列见submission/verification。未人工逐项拖动验证，不声称实机/碰撞/动力学安全；未修改训练代码、奖励或归档。

## 2026-10-21：exper027 第4秒安装图与可交互MuJoCo场景

- 用户要求查看完成30°姿态调整后的右手/法兰装配，并追加导出可交互MuJoCo。直接读取已交付trajectory_and_spline.npz的t=4.000s七轴角与20指角，保持原固定基座及刚性工具关系；无新IK、无策略推理、无物理积分、无硬件命令。SDK运动链使用q0固定标定的URDF网格坐标转换，不重置基座或独立移动手部。
- 新增logs/right_arm_t4_install_20261021/test_export.py及静态viewer.py/launch_local.sh；参考上一轮kinematics.py、assembly.mjb及活动本地回放的GLFW/内存保护实现。volc-a32 Slurm7920 COMPLETED/0:0，1 passed；手根及SDK法兰完整变换与第4秒交付数组一致（2e−12容差）。六视图及双坐标轴对照图与交互MJB/state来自同一状态；训练局部yaw=30°不重复叠加。
- 产物回传至results/exper027/right_arm_t4_install_20261021，10项远端文件SHA核验通过；安装整体及坐标轴图已目视检查。手根XYZ=[140.000001,150.000000,59.999976]mm；SDK法兰XYZ=[155.766697,145.994673,35.307628]mm。可交互场景支持鼠标旋转/平移/缩放、四预设视角和手根/法兰坐标轴切换；README记录来源、操作和静态局限。
- 按已授权本地录制加载例外，在2300M/3G/swap0独立scope中读取MuJoCo3.11 MJB，禁用本地XML编译；4.014秒187帧隐藏OpenGL检查通过，自动遍历四视角/轴模式，memory.events high/max/oom均0。尚未人工逐一拖动验收；不把静态场景当整曲回放，也不产生新演奏F1或伪造实际按键对比图。既有训练/奖励/轨迹CSV保持不变。

## 2026-10-21：exper027 右臂刚性装配、两段准备与原速约束轨迹

- 用户纠正上一版仅手根/手指插值的理解；已将本轮错误的 `results/exper027/right_hand_trajectory_500Hz`、对应临时脚本目录和 `scripts/export_recorded_right_hand.py` 移入回收站，保留原实验、输入和此前安装报告。
- 本轮使用图片法兰XYZ=[201.802,157.225,100.007]mm及 `right_arm_200Hz(7).csv` 首帧七角，手指格式/初角来自 `right_hand_500Hz(6).csv`。用户确认2s移位＋2s倾斜，随后最新明确要求原速度、允许些许误差，覆盖短暂同意减速的答复。
- 参考平台README与坐标轴文档、SDK配置的MDH/PNVA、当前项目右M6 URDF及四孔装配、exper027同次200Hz物理录制与手根位姿。外部平台只读取数据与算法定义，不导入设备驱动或发硬件命令。平台旧镜像右手安装不替换当前独立右手四孔修正。按已知法兰点与七角反算一次基座，保持刚性法兰/连接件/手根关系；训练30°参数转换为完整朝向，实际初态旋转差约33.37°。
- 工具/冻结输入/失败诊断保存在 `logs/right_arm_dls_20261021`。所有运动学求解和模型加载在volc-a32独立日志目录执行；静态装配构建/渲染通过Slurm7914，后续相机修正另记。先以连续阻尼IK生成参考，原逐步限幅候选最大偏差198mm，明确未作为交付；改用固定时间C2 B样条笛卡尔拟合，连续导数系数约束SDK速度/加速度、SDK与URDF位置交集及J6/J7耦合。
- 验证与交付状态：数值Jacobian检查误差1.3e−10；训练起点IK位置误差3.1e−8mm。原速28.3s（4s准备＋24.3s演奏）最终演奏段手根位置最大误差0.98833mm、RMS 0.06578mm、朝向最大误差0.60807°；连续样条速度/加速度证书通过，最大占硬限67.065%/99.241%。相机修正Slurm7916 COMPLETED/0:0，安装图已查看。交付目录 `results/exper027/right_arm_hand_original_speed_20261021`，200Hz机械臂5661行、500Hz手指14151行；列名/初态/时间轴/有限值/硬限、9项文件SHA及ZIP CRC全部核验，详见logs中的delivery_verification.json。ZIP SHA256为b0a748d2ffce2a863565162cf2e5c76acd4cae6c14b1165db1b4a573391dea1c。未改训练奖励、统一入口、源码归档或检查点；未上机执行，实体碰撞、扭矩与新轨迹演奏F1未验证。
- 针对用户再次确认palm_base_yaw=30°：核对exper027 config的source_runtime.pose.yaw=0.5235987755982988；第4秒目标完整旋转矩阵与同次录制初态完全一致，实际朝向误差约0.0000027°。局部30°已经包含在录制姿态中，没有再次叠加30°；交付安装图展示t=0调整前姿态。数值证据保存于logs/right_arm_dls_20261021/yaw_verification.json。

## 2026-10-21：指定右手位置与CSV七轴角静态安装图

- 用户确认[179.798,149.738,100.005]为右手根位置，朝向沿用文档；按毫米及当前yaw=30°合同，采用exper027同次录制初始手根旋转。右臂角使用本会话先前读取的right_arm_200Hz(5).csv t=0完整弧度值，原路径目前已不存在，未替换为新(6).csv。
- 参考当前统一入口/exper027配置、preview_connector.py、右M6 URDF、现行四孔连接配置及旧安装工具结构；固定右手位置/朝向与七角反算基座，保留Link7—法兰—四孔件—右手刚性变换。仅右侧静态图，左手隐藏，手指保持本轮录制初态；不修改训练/奖励/原始轨迹。
- 工具与输入保存在logs/right_install_20261021；GPU构建/渲染统一Slurm7860，远端使用已校验当前归档及额外资产逐文件SHA。安装验证7860 COMPLETED/0:0；位置误差6.1e−16m、旋转矩阵误差9.0e−16，URDF七轴限位及CCS696耦合校验通过。右臂基座反算XYZ=[684.633,17.575,59.881]mm，完整旋转保存于report.json。首版地面遮挡手臂后只调整显示：隐藏地面/左手、关闭反射阴影/天空盒、改透视相机；未移动装配。静态模型及四视图已回传，逐文件SHA验收，最终作业与相机检查记录见submission.json。未做轨迹或碰撞安全验收。

## 2026-10-21：右臂CSV初角通过基座反算对齐训练yaw30

- 用户提供right_arm_200Hz(5).csv并确认用于仿真重新确定基座。保持t=0原七角不变，使用exper027冻结源码的M6右臂URDF、hand_connector四孔变换、hand_constants及arm_scene/arm_follow零笛卡尔参考规则，反算基座以匹配训练初始手朝向。未导入或执行外部平台代码，未改原轨迹/训练默认/历史实验。
- 输出results/right_arm_yaw30_20261021：完整变换JSON、右初角CLI参数、复现脚本、执行/来源SHA记录和README。基座XYZ约(0.684835,0.017837,0.019876)m，RPY xyz约(91.289783,-33.333752,179.047832)度；法兰世界RPY为(-90,60,180)度，不能将训练局部腕yaw30直接当法兰欧拉yaw30。
- 在volc-a32独立目录进行CPU运动学计算，使用已核验SHA的7708冻结源；退出0，逐关节FK复核位置1.34e-13mm、朝向3.18e-15度，URDF位置限与J6/J7耦合31.141305≤110.5度通过；结果回传SHA一致。尚未进行整臂环境碰撞或动力学验收，未发硬件命令，未启动新训练。左臂初角未指定。

## 2026-10-21：保留原版同指多键目标，重新提交exper027

- 用户明确要求不改新指法及原始音符时间，取消同指冲突启动拒绝。参考原版RoboPianist music/piano_roll.py和midi_file.py、当前score_validation.py、tests/test_score_{validation,conversion}.py、exper026实际失败诊断。
- 只将finger_conflict从errors改为warnings；保留全部转换目标、floor/ceil及重复音规则，其他严格完整性检查不变。不改变奖励公式/采样/权重，原始proto字节保持SHA 8bbf0c35890361f544012286bfc886a3a69a132e0f6bf8af7a610bb3970564db。
- 以已验证40/20mm源码归档为父本，只覆盖本次校验器、测试与README/日志，重建manifest和SHA；默认入口同步到新归档，exper026失败记录与旧归档保留。新训练编号exper027，独立远端目录 /home/houj/projects/pianist_exper027_20261021，默认双卡/512环境/19532步，从头训练。自动best.pt回放、原谱对比图与回传继续由后台脚本处理。
- 验证：本机8项轻量单测、远端13项回归通过；实际谱269/269覆盖、39项warnings且errors为空，与原版转换目标逐项一致，原proto字节不变。新归档SHA cc692240a53e23f40bdf1dab66c1a64a00183cfa2a1b254010c15065c570abb0。已提交7708并分配两张A100-SXM4-80GB，后台PID82400存活并完成首次查询；已通过原失败的乐谱构建位置并生成config.json，核验嵌入谱SHA未变，完整奖励已在7708.out第15行开始打印，另存本地startup_reward_config.json；新归档本地回传SHA一致。日志见logs/score_conflict_20261021及results/exper027。训练/导出未完成时不宣称成功。

## 2026-10-21：按最新编辑指法启动 exper026

- **需求与参考**：用户要求根据最新指法训练；读取 `all_use_this_train.sh`、当前奖励JSON、已验证指法目标归档、`scripts/hpc/{submit.sh,job.sbatch}` 与自动回传/回放脚本。输入为 `results/exper024/input_sources/015-5_fingered_edited_end.proto`，冻结到exper026；269音符、24.0521秒，相对同目录原proto有91处指法变化，音高/起止/力度均不变。
- **配置与来源**：使用统一入口当前默认，无臂双卡、512环境、19532 vector steps（10000384经验）、TF32学习器，从头训练；只覆盖MIDI及输出目录，不修改奖励或算法。源码SHA为 `1a105ec4e447cd288327445474a2996df9f1e65b18e4715debfda7e1ece14ab5`，输入SHA为 `8bbf0c35890361f544012286bfc886a3a69a132e0f6bf8af7a610bb3970564db`。
- **部署与交付**：独立远端 `/home/houj/projects/pianist_exper026_20261021`，仅复用依赖环境；任务、命令、状态记录于 `results/exper026/submission.json` 和日志。后台脚本负责训练终态后Slurm导出best.pt、同次原始注册谱对比图、有声无悬浮标签交互回放、回传及SHA验收。
- **验证与待完成**：本机只作轻量proto验证和文件同步，远端核验归档；实际训练、导出、回传完成以该实验状态文件为准，提交不代表训练成功。完整奖励由训练启动日志打印，未修改当前奖励总表数值。已提交作业7580（双卡/16CPU/128GiB/24h），首次查询PENDING/Priority；后台PID66073存活且日志有真实状态查询，导出仍待训练终态。后续复查：7580于14:04:50启动、52秒后因strict_v1指法控制帧冲突失败，未产生训练输出/checkpoint；后台已回传7项诊断并校验。输入SHA未变化。轻量区间分析发现19对20Hz量化重叠、原始时间同指异键实际重叠为0（详见exper026/fingering_conflicts.json）；此为区间辅助分析，不等同完整pianoroll校验。进一步按7580冻结源码、实际trim_silence=true（原点0.0049秒）在远端CPU执行完整乐谱校验，复现39项同指冲突，详见exper026/training_score_coverage.json及training_conflict_locations.txt；此前19对仅未trim区间初筛，不是实际训练总数。未修改原谱、指法、算法或提交新作业。需用户决定是否授权修改时值或目标离散规则，未绕过严格校验、未重复提交。

## 2026-10-21：指法预览琴键与手指颜色同步

- **需求与参考**：用户截图要求琴键目标颜色跟随对应十指，不再统一黄色。参考 `scripts/fingering_editor.py` 的COLORS、帧内active指法映射和目标连线，以及现有本地CPU视图。
- **修改**：白/黑目标键按当前音符分配的finger索引使用同一色表；无目标键保留黑白，同键多个指法按对应颜色分区；目标虚线同步手指颜色，更新界面和编辑器说明。仅改变本地显示，不修改乐谱、远端模型、求解或训练/奖励归档。
- **验证**：在512MiB受限scope中检查1.2秒和弦，MIDI60/64/72分别使用手指7/5/2的绿/紫/青色；同时检查黑键及同键多指分区，原始proto字节不变。实际截图与记录为 `logs/fingering_editor/finger_key_colors_20261021.{png,json}`，峰值66.1MiB、OOM=0；Python编译和diff空白检查通过。已有窗口保留，保存当前编辑后重开应用显示更新。

## 2026-10-21：指法预览左右视角、本地内存隔离与压力测试

- **问题与参考**：用户怀疑左右手错误，要求本地渲染内存上限，随后明确要求评估本地模型构建/求解并设计压力测试。参考 `fingering_editor.py`、`fingering_pose_worker.py`、`launch_local_recorded_replay.sh`、实际ready中的44个手部凸包/10指站点/88键坐标、v6归档及现有数据测试。右手0–4位于世界+Y、左手5–9位于−Y，原编号没有交换；旧投影相机高度符号反向，且近远绘制排序错误。
- **修改**：新增独立显示相机函数，修正上方演奏者视角和遮挡，增加左右手身份文字及演奏者/对面/俯视按钮；保持原谱与指法。渲染原本就是本机CPU，本轮给UI增加384/512MiB软硬限及禁swap。请求队列改为只保留1个待处理请求。
- **本地例外与保护**：仅针对用户本次授权新增 `--backend local`、冻结源码本地准备和分阶段压力脚本；worker单独scope为1024/1536MiB软硬限、swap=0，启动须4GiB可用，250ms监控内存/PSI，超过80%硬预算、系统可用低于2GiB或PSI full avg10>1%主动终止。写入并核验实际cgroup `memory.oom.group`（systemd255不支持scope MemoryOOMGroup参数，首次拒绝诊断保留）。不扩大训练/仿真/GPU本地许可，不更改系统驱动/保护/交换空间。
- **实测与结果**：7项轻量测试通过；512MiB scope内20帧绘图峰值64.7MiB，中位0.280秒/帧、最大0.347秒/帧、OOM=0。两轮本地冷构建分别约1024.5/1039.3MiB时触发PSI主动停止，OOM/max事件均0，尚未进入持续IK。第二轮去掉无关训练任务，仅导出原完整场景，仍未通过；没有提高预算。结果见 `results/fingering_local_stress_20261021_120{532,602,727}/report.json`，绘图证据见 `logs/fingering_editor/camera_fixed_20261021.{png,json}`。
- **交付与TODO**：默认保持远端构建/求解＋本地受限CPU绘图；本地完整后端标为未通过压力验收的显式实验选项。详见 `docs/fingering_editor_local_stress_20261021.md`。持续IK、长期内存趋势和重启阶段未运行；不能宣称本机全流程通过、保证高帧率或物理安全。未改奖励/训练归档/SHA/历史checkpoint。

## 2026-10-21：指法奖励目标点示意图

- 按用户要求生成单张俯视＋白黑键侧视图，标出世界轴偏移、距前端距离及目标低于键面的高度。参考 `fingering_targets.py`、`mjlab_env/task.py`、`models/piano/piano_constants.py` 和 `piano_mjcf.py`。
- 产物及可复现绘图脚本：`outputs/fingering_targets_20261021/`（PNG/SVG/draw.py）。使用本机CPU二维绘图，没有加载MuJoCo模型或进行仿真；已核对尺寸公式并检查图像。未改训练参数或奖励算法。

## 2026-10-21：指法奖励目标点示意图

- 按用户要求用一张图展示白键和黑键的指法目标，含俯视投影和侧视剖面；依据 `fingering_targets.py`、`mjlab_env/task.py`、`models/piano/piano_constants.py` 与 `piano_mjcf.py` 的当前公式和尺寸。
- 产物：`outputs/fingering_target_diagram/fingering_targets.png`、同名SVG及可复现 `draw.py`。标注未按下水平键：前端距离白48.75mm/黑29.25mm，低于表面白5.625mm/黑4.5mm；本机仅CPU绘图，已检查图片排版。未改变训练、奖励或源码归档，无新增算法TODO。


## 2026-10-20：proto 指法、音符时间与三维姿态编辑器

- **需求与参考**：用户要求导入 proto、编辑十指编号及音符时间，并让三维手模型随音符移动；示例为 `results/exper024/input_sources/015-5_fingered.proto`。核对 `music/midi_file.py`、`register_supplied_fingering.py` 的 `part` 编号，当前 `all_use_this_train.sh`、`source_runtime.py`、`bundle.py`、`finger_coupling.py`、`fingering_targets.py` 和 v6 模型验收记录。
- **修改**：新增 `launch_fingering_editor.sh`、`scripts/fingering_editor{,_data}.py`、`prepare_fingering_editor.py`、`fingering_pose_worker.py`，以及编辑器说明和4项数据完整性测试。支持原始秒值时间轴、逐音起止编辑、0–9手指按钮、重叠提示、撤销/重做和只另存新文件；校验其他谱面字段及未知字段不变。没有将编辑结果自动注册或替换已有实验输入。
- **三维执行**：本机仅轻量Tk/CPU三角形绘制；`volc-a32`独立目录由本地/远端SHA一致的当前v6归档建立，worker另验SHA。远端单线程CPU构建模型及IK，禁用CUDA/OpenGL；保持动作/关节限位、结构性移除TZ/Yaw和八指60°–80°耦合。源码、命令、资源预检、进程及诊断记录在 `logs/fingering_editor/`。远端系统Python缺少 `hashlib.file_digest` 的首次预检失败后，改用已核验项目依赖解释器重试成功，没有本机模型加载。
- **验证**：`tests/test_fingering_editor_data.py` 4 passed；实际Tk导入/选择/指法修改/起止编辑/撤销及截图检查通过。示例269音符、24.0521秒、0个同指重叠音符；远端4个时刻通过限位、耦合和琴键qpos保持检查，单音偏差0.04–0.79mm，1.2秒和弦偏差24.95mm，界面如实显示。
- **局限/TODO**：按时刻独立运动学求解、按求解速度逐步播放，当前无声；简化凸包只用于绘制，不是精确外壳。没有动力学/速度/加速度/碰撞可行性保证，较大目标残差不证明不可达。未进行训练、物理回放或视频导出；没有修改奖励、训练归档/SHA或旧checkpoint。多指求解改进和有声播放留待后续需求。

## 2026-10-20：替换用户提供的双侧 multipart A3.2 手模型

- **问题与来源**：用户要求用 `TsenseI_hand_A3.2_withid_upper_thumb5mm_noupper_mid.zip` 替换当前模型。实际参考两侧包内URDF、全部引用STL、当前 `sync_project_urdf.py`、手实体/常量、动作JSON、指尖/耦合、碰撞分类、连接件配置、TF32 v5源码归档。
- **实现**：保留双侧原关节树及0.952kg源惯量；合并每link刚性零件并重建44个凸包，更新源和资产散列；按用户后续要求将拇指第二旋转轴恢复旧轴向（左−Z/右+Z）、正角限位0～1.74rad、目标0.05～1.57rad及初始角+0.476648962rad；轴与角度同时反号保持等效物理旋转；按源网格四孔将右连接件沿手部Y修正+8mm；更新源合同、模型说明、姿态预设和预览指纹。旧资产及旧已解析姿态留存deprecated，历史实验/归档不改。
- **验证**：只在volc-a32独立目录编译/分析/仿真，GPU验收经submit.sh；源几何逐三角一致、惯量、凸包、指尖/耦合、装配与物理记录见 `results/model_replace_20261020/`，最终状态见acceptance.json。初次失败诊断保留并针对原因修复，不放松源限位/碰撞/耦合容差。唯一训练入口绑定验证后的v6归档与SHA；未启动训练。
- **预览跟进**：按用户要求增加 `preview/scripts/render_supplied_hand_review.py`，参考固定v6归档、统一入口姿态、SceneBundle reset及MuJoCo实际相机；远端Slurm6533/6535生成初态PNG、单关节GIF和可独立打开的角度滑块HTML。第二版修正左拇指被掌壳遮挡的视角，轴标记按真实投影透视显示；局部轴/限位/初始角及中心差分方向均检查，其余关节保持reset。无物理步、策略或硬件命令，不改训练源码/模型/默认归档；产物与最终验收见 `results/model_preview_20261020_v2/` 及模型替换说明新增节。
- **TODO**：源惯量/驱动和实际螺纹装配仍未硬件标定；新几何初态不宣称共面，短测不代表F1或硬件安全。详情见 `docs/model_replacement_20261020.md`。

## 2026-10-20：跨手碰撞改为物理子步评分

- **用户变更**：`tsensei_collision_v1` 的跨手碰撞从控制周期末单点采样改为每个物理子步采样。当前20Hz控制、200Hz物理频率下，每控制步10个子步；每个有任意跨手接触的子步计一次，奖励为 `−0.5×接触子步数/10`，即每个子步−0.05，整步最多−0.5。
- **实现**：`mjlab_env/substep_rewards.py` 保存跨手接触子步历史；`task.py` 将该计数传入 `MusicTaskLogic`；`forearm_collision.py` 按物理子步数分摊当前公式。周期末 forward 只保留诊断，不重复计分；带臂检测冲突按同一物理子步并入当前步计数。
- **验证**：新增 `tests/test_reward_defaults_20261006.py::test_cross_hand_collision_is_scored_per_physics_substep`，验证0/1/5/10/超过10个子步分别得到0/−0.05/−0.25/−0.5/−0.5；相关 Python 文件通过编译。GPU长时验收待下一轮 Slurm smoke。
- **源码归档**：生成并审计 `logs/ik_no_target_braking_20261019/source_cross_hand_substep_v4.tar.gz`，SHA256=`2de5639f3fed9c872f54de3e00759e83d27ccf1a48b083387fd0f9c703cc9c65`；`all_use_this_train.sh` 已切换到该归档。
- **未解决 TODO**：尚未启动新训练；现有 checkpoint 不因奖励合同变化而兼容迁移。

## 2026-10-20：训练回传改为可交互有声 MuJoCo 回放

- **用户变更**：将每轮训练默认交付从有声视频改为使用该轮 `best.pt` 的可交互、有声 MuJoCo 回放；保留指法标注、实际触键高亮、同次实际按键与原始 MIDI 对比 PNG、绘图数据、回放报告、来源与 SHA 记录。
- **删除内容**：移除回传要求中的指尖编号；未修改训练、奖励、checkpoint 兼容或 GPU 导出约束。
- **修改位置**：`AGENTS.md` 的每轮回传规则；本次仅更新工作约定和变更日志，未声称现有导出脚本已经实现交互式回放。

## 2026-10-19：移除训练中的额外 F1 评估

- **用户变更**：删除 `music_only_f1`、`music_only_micro_f1`、`substep_key_micro_f1` 和 `exact_original_music_micro_f1` 四类额外评估及其最佳模型选择入口；统一入口改用原始逐控制帧 `f1` 选择 `best.pt`。
- **实现**：`rl_torch/train.py` 的新评估记录过滤上述字段（包括 episode 明细），`checkpoint.py` 仅接受 `f1` 作为最佳指标，SwanLab 映射同步移除这些曲线；带 MIDI/实际按键的同回放视频对比数据仍由导出流程保留。
- **兼容范围**：历史实验 JSON、checkpoint 和已完成视频不回写；旧命令若显式请求 music-only F1 停止目标会明确拒绝，避免静默使用已删除指标。
- **验证**：`py_compile` 通过；`tests/test_swanlab_metrics.py` 16 passed、9 subtests，`tests/test_batched_evaluation.py` 11 passed、2 skipped、5 subtests，`tests/test_training_quality_control.py` 7 passed、12 skipped、2 subtests；生成并审计源码包 `logs/eval_contract_20261019/source_frame_f1_only_v4.tar.gz`，SHA256=`dbe09c8a656bde16354434fc120a747c0fa6541a7855bcb8251dfeec68266e21`，入口已绑定该归档。

## 2026-10-19：Prelude exper017 手高 0.06m、默认关闭准备段

- **用户变更**：将统一入口 `all_use_this_train.sh` 的手根高度改为 `0.06m`；保留 Prelude 注册指法、yaw=`π/6`（30°）、现有关节范围、奖励和四卡采样／学习布局。移除默认 `--preparation-mode`、`--preparation-time`、`--preparation-lift` 参数，训练解析器默认 `preparation_time=0`，后续如需准备段再显式开启。
- **提交**：重新生成并审计可复现源码包 `logs/hand_height_20261019/source_hand_height_preparation_off_v3.tar.gz`，SHA256=`60d9bb5c6d84455e7950b5518caf75652bbb02d681504f32ce0075548d46457b`；Prelude proto SHA256=`ad58e67de1998c218e380b7370c95d64ba23fe80d6dfbe26f415184e1a80b33a`。远端 `volc-a32` Slurm 作业 `5346` 从头训练 `exper017`，默认预算 `19532×512=10000384` transitions；`5343` 因旧归档审计失败、`5344` 因多采样器首次物理图捕获的CUDA stream限制退出，均未计入训练，`5346` 仅关闭可选 `physics_graph` 优化重启。
- **自动回传**：本地后台 watcher PID `190397` 已启动并负责首次查询重试，持续负责 best.pt、完整有声视频、同回放200Hz实际按键／原始MIDI对比图、SHA校验和回传。

## 2026-10-19：移除机械臂目标剩余距离提前制动

- **用户变更**：删除带臂模式中从当前关节状态到 IK 参考目标的剩余距离提前制动，并删除上游 `TargetMotionLimiter` 的原始请求目标制动输入。机械臂仍保留当前步速度、加速度、位置和 CCS696 J6/J7 耦合硬限；位置边界的可行性限幅保留，用于避免越过 URDF 位置范围。
- **实现位置**：`mjlab_env/arm_follow.py` 与 `arm_realtime.py` 直接跟随当前步可行盒内的 IK 参考；`mjlab_env/scene.py` 不再传入 `braking_target`；`target_motion_limits.py` 删除 `brake_to_target`/`braking_target` API，活动合同改为 `wrist_target_rate_limits`。
- **测试**：`tests/test_target_motion_limits.py` 8 passed、12 subtests；`tests/test_optional_arm_follow.py` 通过；`tests/test_arm_realtime.py` 与 `tests/test_arm_native_scene.py` 2 skipped（需要显式环境）。`compileall` 通过。删除提前制动后，若当前状态已进入 CCS696 耦合边界且既有速度无法同时满足下一步加速度和耦合约束，执行会明确拒绝该步，而不是静默突破硬限；该行为需在 GPU 训练前重新验收。
- **文档**：同步 `AGENTS.md`、`all_use_this_train.sh`、`docs/training_environment.md`、`docs/optional_arm_wrist_ik_requirements_20261008.md`、`docs/reward_defaults_20261006.md` 和轨迹流程说明。
- **源码归档**：生成 `logs/ik_no_target_braking_20261019/source_no_target_braking_v1.tar.gz`，当前入口绑定 SHA256 `dfdd224541d13338fe03040902807bd84a75bbda00b96086e5e615bd386fac13`；归档含本次控制实现、测试及活动说明快照。

## 2026-10-18：长尾调度定位与掌根1mm超限惩罚

- **用户合同**：继续远端定位并优化长尾，保留所有关节位置/速度/加速度/耦合约束；严格200Hz长时间零截止违约通过后，才启动小星星约1000万样本带臂训练。用户后续明确允许跟随误差超过1mm，超过越多扣分越大并记录次数，因此1mm为奖励/诊断阈值，不是硬停止或训练准入门槛。
- **问题来源**：核对v6同轨迹数组、`arm_realtime.py`、`arm_native_scene.py`、`arm_scene.py`、视频计时循环、Slurm线程池设置及奖励/任务/训练日志入口。原峰值3.5453mm来自20Hz请求一次跳变3.6512mm，控制周期末最大误差0.1762mm；记录状态的保守关节可行域给出下一步误差至少2.1013mm，见 `preview/outputs/ik_strict_20261018/tracking_infeasibility.json`。不改变原请求定义或刚性安装来隐藏误差。
- **奖励实现**：按用户最后确认0mm为理想值、1mm内也扣分，新增独立 `arm_tracking_error_penalty`、weight=1；令 `u=e/0.001m`，当前唯一公式为 `u≤1: −u²; u>1: −(1+2ln u)`，无硬封顶。e取双侧/全部物理子步原始策略请求（含准备段、TCP/IK限幅前）到实际刚性掌根的最大欧氏距离，不乘dt。带臂默认开启，无臂关闭；1mm只用于超限计数。保存每200Hz子步超限次数/左右计数、跨控制步进入事件、控制步超限数、采样数、比例及原始/IK目标峰值；训练metrics、summary、eval和本地SwanLab映射同步。唯一奖励表直接更新，免费区与封顶中间方案的threshold_m/maximum_penalty_per_step字段删除、明确拒绝。
- **计算优化**：场景FK使用同一float64定长小矩阵运算；IK在加权位姿残差平方和≤1e-30时结束无意义迭代，最多12轮，仍完整执行目标制动和所有q/v/a/耦合/有限值检查；不修改目标、约束或现有等价测试容差。计时循环预分配时间Tensor并预触页完整记录数组，直接读取同源native状态减少包装分配。
- **进行中验证**：Slurm4266使用v6冻结运行时及SHA记录的诊断脚本，分别比较默认线程池、单线程池及另一颗获配CPU；增加线程CPU时间、线程上下文切换/缺页、线程亲和性和cgroup计数。诊断开销独立标注；未将单次短测或平均耗时当作长时间验收。4266诊断完成，三组112s的截止超时分别0/7/16；单线程慢周期wall≈thread CPU且ctxswitch/fault=0，未见cgroup节流，不能归因于普通抢占。进一步验证局部MPOL_BIND对NUMA自动扫描的影响，不改系统sysctl。4278测试配置误选父目录，收集阶段失败；4282显式指定冻结pyproject并核对导入SHA后重跑。

- **GPU接入实测**：4282的8环境260物理子步通过关节硬限、碰撞/手指耦合、局部reset、SAC更新和稠密跟随罚分/计数检查；显式跳变测试累计456个world-substeps超1mm，峰值24.928mm（压力输入，不是训练表现）。固定请求下512环境IK+发布约15.65ms/次，旧无早停约194ms；只说明该批量固定请求微基准，不冒充完整训练吞吐。长时实时门槛尚待完成。4282在短测后主动取消：外部runpy入口可能加载旧版同名视频helper，故该bound短测不作最终门槛；v8/v9修复为显式同目录加载并断言helper SHA，新增Slurm内MPOL_BIND进程启动器。v9预触页所有计时数组，并将profiling/阶段统计开销计入截止检查；两项最终计时标量写回在终点时间戳之后。

- **冻结部署**：v9归档SHA `9e1db3ae0270d39a06744751fa1a1f4f41862a69d2ec2b70de558b6ee44fb8ab`，1151文件本地/远端一致；656个运行时/配置成员与已GPU验收v7一致。统一入口更新该归档及SHA以匹配新23项奖励JSON，README/AGENTS源码路径同步，并将AGENTS残留的旧controller标识对齐当前wuji_cartesian_v1（仅文档纠正），无臂仍原8项开启。独立远端部署同步7个变更文件，完整641个源码/动作/奖励文件校验一致，旧文件备份保留，保障最终导出校验；不改变另一独立项目的exper010。4294外部测试相对路径问题在运行前取消，4295用绝对驱动路径正式执行112秒预检及1806秒长测。训练未提交。

- **最终长测**：Slurm4295 COMPLETED/0:0、2 passed；v9连续1806秒361200周期，平均1.317607ms、P99 1.718545ms、最大2.932029ms，计算超5ms和完成截止违约均0；最小完成余量2.067767ms。361201个全程q/v/a状态的位置、速度、加速度与CCS696耦合检查通过。包含规定目标/IK/mocap发布/完整mj_forward/内存记录及诊断统计，不含mj_step、策略或硬件通信；普通Linux单核忙等+同节点MPOL_BIND，不宣称硬实时保证。9个回传文件SHA通过，详情 `preview/outputs/ik_strict_20261018/acceptance_qa.json`。
- **条件满足后开训**：重新核对本地/远端/待运行编号后，exper011经唯一入口提交Slurm4318，小星星10,000,384样本、yaw30°、双M6、numba_cpu/4线程、原双GPU采样/学习流水及全部约束；使用稠密跟随罚，1mm不是训练硬门槛。本地后台watcher PID563639已存活并首次真实查询到RUNNING，已登记best.pt原生带臂有声完整曲目/指法编号/触键高亮/同次原谱对比/回传SHA要求；训练与导出尚未完成，不创建TRANSFER_COMPLETE。提交记录在 `results/exper011/submission.json`。

- **真实入口失败及修复**：exper011/4318在采样前失败，train_from_source重设PYTHONPATH丢失本部署Numba依赖；此前直接环境smoke未覆盖该外层入口。失败13项诊断已回传并验SHA，无best.pt，不生成视频/成功标记，旧watcher已停止并明确记录缺项。新增source_pythonpath仅允许所选源码+本部署native_ik_deps，训练/归档eval/指法恢复三个子进程共用；Slurm冻结目录映射独立依赖，不修改共享venv。3项隔离路径测试通过。IK/奖励/场景核心和656个配置运行时成员不变，30分钟验收仍对应同一计算实现；接下来用真实统一入口验证。

- **最终部署包**：v11 SHA `72aaa906daf75f25364abe806e8dc06df8ee1ae9522c7121bd5c2e0882b7ecf1`，1154文件本地/远端完整审计；与v9的656个运行时/配置及3个计时helper逐SHA一致，复用4295长测。唯一入口/README/AGENTS改为v11，外层训练、归档eval、指法恢复和renderer同步；3项路径隔离+4项音频组合测试通过。
- **导出钩子审查**：新task每子步显式传跟随误差callback，覆盖旧音频partial默认参数。renderer改为组合task/音频/归档trace三个钩子，同一子步各调用一次；新增录制数量必须等于实际控制步×decimation的检查。新测试及既有精确音符阶段回归5 passed。修复仅涉及导出脚本，不改控制、物理、奖励或长测驱动；训练依赖路径3项测试通过，真实统一入口短训4341继续验证。

- **真实入口复验与正式重启**：4341 COMPLETED/0:0，8环境32条采样、6次SAC更新、完整曲目评估及best/final保存通过。使用同一core/config的v11归档提交完整带臂音频/标注/同回放原谱对比导出smoke4350。再次检查本地/远端/待运行编号，exper012经all_use_this_train.sh提交4351，10,000,384条样本，原512环境/双GPU流水/yaw30和全部约束；后台watcher PID590216存活并首次查询PENDING，完整best.pt导出回传已登记，尚未写TRANSFER_COMPLETE。

- **完整导出验收发现的几何路径问题**：4350已从同一短训best.pt完整记录1830物理子步和音频；native后处理读取mj_saveLastXML产物时缺少right_Link1_R.STL，Slurm失败，保留录制和失败日志。完整几何已包含在同次model.mjb，后续使用其恢复导出，不重跑策略或换PT。exper012/4351仍因共享节点GPU资源排队，尚未宣称已采样。

- **原生导出修复**：小模型确认learner MuJoCo3.11保存的MJB不能在composer3.13读取。native worker在创建workspace/录制前按sys.prefix切换learner，加载同次完整model.mjb；read_targets接受并复用该模型，native绘图/渲染不再读缺网格XML。无臂XML和overlay路径保留；60项导出回归通过。只改外层export_video_job/replay_arm_ik及测试，不改训练源码/控制/奖励；下一导出作业自动冻结修复脚本，训练匹配归档仍v11。复用4350原录制恢复到新目录。

- **排队与后台后续**：节点8卡已全部占用，4351正式双卡训练仍待调度；4357单卡恢复已取得分配。只将诊断恢复4357缩为4CPU/64GiB/5分钟，不改正式训练预算/资源、不碰其他任务。新增一次性后台 `preview/outputs/ik_strict_20261018/watch_export_recovery.py`（PID608735，首次真实查询RUNNING）等待既有4357终态，成功后CPU finalize同次原谱对比/音视频验收/rsync/SHA，失败保留日志；绝不提交新GPU作业或重跑策略。正式exper012继续由原watcher PID590216管理，尚未开始采样或写TRANSFER_COMPLETE。

- **完整导出链通过**：4357 COMPLETED/0:0（3分28秒），原4350的1830物理子步逐SHA复用；native模型/指法编号/触键高亮/音轨/完整曲目导出成功。后台CPU finalize生成同回放原MIDI对比，200Hz音视频解码通过，199个文件回传SHA一致；短训仅32样本、6更新，未触键，单回放F1=0、音轨为真实静音，不能作为正式训练成绩。图像已目视检查，产物在 `preview/outputs/ik_strict_20261018/unified_export_smoke_v11_recovered/`。exper012/4351已获双卡RUNNING，完整config核对512环境/10,000,384样本/稠密跟随罚/全部约束及启动奖励打印通过，等待采样日志最终确认。

- **批量瓶颈与奖励修正**：exper012实际确认vector_step20/10240条样本、updates=0；20批采样313.479秒，约32.67条/秒，并非SAC首次编译。第20批跟随罚均值−5，累计全部world-substeps超1mm、raw峰值200.161mm；不能据单批奖励日志断言此前每批均−5。原方案在√5mm即封顶，无法满足大误差越大越扣，故活动奖励替换为上述二次/对数连续公式，0.1/0.5/1mm仍分别−0.01/−0.25/−1，20/200mm为−6.9915/−11.5966；58项CPU测试及14subtests通过。该轮在学习前主动取消，保留17项已SHA核对训练产物、原watcher继续导出已有best.pt（4389），不迁移checkpoint。新公式可能促使策略减少掌根移动，后续须同时评估音乐表现和误差，不能把罚分改善直接等同演奏提升。

- **批量优化候选**：代码核查发现512环境的独立臂query每子步按4194304碰撞容量重新分配scratch，硬限之后的完整forward也走eager；新增独立query图复用与带臂硬限后的已有forward图复用。保留1632显式pair、所有隐式mask、容量、逐200Hz采样、溢出检查、forward次数/顺序和全部约束；不删除重复forward或减少碰撞。新增eager/graph逐pair、重捕获、上下限状态/约束对照GPU测试。独立512环境基线阶段计时4387排队，候选尚待冻结包GPU等价/吞吐验收，工作区修改未自动进入旧训练。

- **分段计时与后续状态**：4387完成，512环境3批耗时14.691/15.649/15.634秒，其中独立query11.376–12.567秒，硬限后的eager forward2.191–2.345秒，Native IK0.519–0.850秒。v12 CPU回归125项及14subtests通过，独立只读审查未见阻断问题。4396因pytest按外部测试路径发现部署根pyproject的pythonpath而导入旧源码，3项失败，不能作为候选GPU结论；4405显式`-c /dev/null --rootdir=. --import-mode=importlib`并断言模块属于冻结目录，仍用相同v12归档重试。候选包含完整512环境物理/新奖励/全部硬限/碰撞容量与溢出/SAC更新/局部reset、每批均值小于2秒门槛；未通过前不启动替代训练。exper012已有best基线经4389成功导出，221文件及同回放200Hz原谱对比/音视频验收通过，单回放F1=0、训练未完成；原watcher已完成回传。

## 2026-10-18：原生IK接入wuji场景、含场景刷新的200Hz验收与带臂训练

- **范围/来源**：用户要求把已优化IK接入当前 `wuji_cartesian_v1`，远端真实几何XYZ视频、包含场景更新的200Hz实测，保留全部约束，通过后启动小星星约1000万样本。核对当前统一入口yaw=30°、源码包prelude_yaw30、`arm_follow.py`/`arm_scene.py`/`scene.py`、`control_limits.py`、预览配方及现有导出/后台回传脚本；不引用旧项目代码或旧checkpoint。
- **实现**：新增 `arm_native_scene.py` 和显式 `--arm-ik-backend numba_cpu`，CPU单世界串行/多世界并行计算后一次packed Torch发布；保留硬限、失败整批原子性、54维观察、奖励及reset语义。`arm_realtime.py` 定长3×3乘法/叉乘替换微小BLAS，未减少12迭代/5候选或放宽硬限。JIT核心释放GIL。修复prange内try导致串行退化：异常捕获放入独立worker并返回失败标志，整批通过后才提交。
- **配置**：新建带臂诊断配置清除结构固定的两个Pitch运动通道（共享手部预览配方保留原Pitch限值）；统一入口不改默认，以末尾Pitch速度/加速度=0显式覆盖通用JSON。XYZ目标/SDK关节速度与加速度/URDF限位/CCS696耦合/手指耦合/碰撞均保留；README和奖励唯一总表说明固定轴与可动轴区别，没有修改任何奖励公式。
- **导出**：`render_cartesian_installation.py`/`render_cartesian_video.py`新增本后端和56秒完整场景计时，所有周期相同内存记录，首14秒同轨迹渲染真实手/臂/法兰/四孔件与W/F/P/T。`export_videos.py`/`export_video_job.py`新增必须与checkpoint匹配的native选项，原生模型/实际mocap回放、不叠加另一条IK；watcher按arms选择native，finalizer核对原生臂配置。`job.sbatch`只为本部署加入独立native_ik_deps，不改共享依赖。新增场景批量/原子性/局部reset与导出防错测试。
- **阶段证据**：Slurm4228因外部测试路径错误未收集测试；4229三项接口测试通过、XYZ视频完成，但11200步平均3.577ms/P99 3.895ms/最大6.287ms/8次超时，未通过实时门槛；GPU场景发现上述固定Pitch配置问题。冻结v3完整1147文件SHA `b101542a309331b19c29a81ee76fa4dd5bf114b763fc9a03b78f798a3345fe29`，本地/远端一致；4239因资源不足未运行即取消，4240经统一入口提交后将该验收作业资源降为4CPU/64GiB/单GPU/30分钟、Numba4线程，正在复验。初测失败日志保留于 `preview/outputs/native_scene_20261018/baseline_failed`。
- **后续证据与修复**：4240缺少基包中的原生IK测试文件、4241把note_seq测试放在learner环境，均在收集阶段失败；改用冻结目录作为pytest root并按composer/learner拆分。4242原生一致性/原子性/局部reset及59项导出回传测试通过；同轨迹56秒测量平均2.2049ms/P99 2.6160ms/最大3.4777ms/0截止超时，视频完成。完整训练场景进一步发现源任务reset是绝对腕坐标、装配后是Cartesian零位移：`arm_scene`保留转换前参考供`TaskBundle`严格核对，运行时仍用转换后的reset，不放宽校验。`scene.physical_action`修复被删除物理滑块的TZ误覆盖为固定值，仅覆盖非策略固定轴，双侧Z动作保留。增加直接TZ映射回归及GPU源任务篡改拒绝检查；测试动作维度更正为保留手指耦合的31。补齐本部署音频/ffmpeg依赖链接，未改共享环境。v4仅为中间包，正式候选为后续v5。
- **v5训练验收**：Slurm4249完成，11 passed（内嵌composer环境另59 passed）。8个GPU物理环境260子步、全部臂硬限/手指耦合/碰撞、双侧TZ策略映射、局部reset、源任务篡改拒绝和SAC更新通过，动作31/观察1461；4线程512环境IK+发布5次约194ms/次。v5普通sleep场景复测11200步有2次超时、最晚0.401ms，未隐瞒或只选4242零超时结果。v6只增加诊断时钟/亲和性选项和全程q/v/a保存；运行时/动作/奖励文件与已通过GPU验收的v5逐字节相同，最终在Slurm绑定的CPU内单线程忙等112秒复测。
- **最终v6验收**：Slurm4257严格截止断言失败，112秒22400周期平均2.821621ms/P99 3.267524ms/最大6.612775ms；14次截止超时（其中12次计算本身超过5ms），最大晚1.612940ms。冻结源码SHA `b6c9675781233f3148afa35188aad5721ca47de847377e8a6cc68afbcaae371a`，运行时与v5逐字节一致。22401个全程q/v/a样本全部位置/速度/加速度/耦合硬限通过。视频独立导出成功：14秒350帧1080p25fps、同轨迹XYZ图、15项回传SHA一致、完整解码及视觉检查通过；峰值掌根误差3.545304mm，报告与日志保存在scene_video_v6。该组件演示没有checkpoint/MIDI，不冒充训练评估或生成F1。
- **训练状态与TODO**：用户明确“严格200Hz达标后再训练”，因此未提交小星星约1000万样本任务、未预留exper编号、未启动本轮watcher；统一入口默认源包未改。IK/场景/记录均有长尾，根因尚未确定，需进一步关联线程CPU时间、调度和缺页。计时不含mj_step/策略/通信，渲染在计时后执行；不将批量训练称为墙钟200Hz。绑核忙等仍是普通Linux调度，不能宣称硬实时。

## 2026-10-18：左法兰镜像当前右法兰并导出正面图

- 用户要求镜像效果；明确此前左目标为用户提供。本次以钢琴XZ面镜像，pL=S pR、RL=S RR S，S=diag(1,-1,1)，目标[194.947003,-157.591713,111.579731]mm。右角和基座保持，左保留完整标定基座，初值沿用用户七角。
- 新增preview/outputs/left_mirror_right_20261018，受限CPU求得[75.2425,-67.3624,-55.8938,-103.0766,72.4179,37.2110,-20.1547]度；当前固定归档Slurm4225渲染通过（47.61s）。正面PNG已回传查看，法兰位置/旋转镜像及右侧角/基座/法兰/手根不变核验通过。基座非镜像、七轴冗余，不保证整臂外形镜像；未验收路径、碰撞或实机，训练配置不变。

## 2026-10-18：左侧完整标定位姿正面图已完成

- 新增preview/outputs/left_full_calibration_front_20261018独立导出脚本、当前预览配方、左IK输入与状态；左角[91.0223,-49.4815,-74.0870,-106.4975,52.2377,24.5655,-3.2440]及完整标定基座，右角/基座沿用上一图。
- 上传当前training_consistent归档至/data独立预览项目，经submit.sh申请单GPU。4222因腕轴开关与带臂构建不兼容失败，导出脚本局部关闭disable_wrist_yaw/disable_wrist_tz，未改活动训练配置；重新提交4223。
- 恢复连接确认4223因必须结构性移除TZ/Yaw失败；独立导出配置两个disable选项改为True，4224通过（47.87s）。rendered_verified正面图已回传并视觉检查，4项SHA一致；右侧角度/基座/法兰/手根与旧图一致，左法兰位置误差<0.000662mm。静态数值验证，不代表实机或碰撞验收。

## 2026-10-18：左法兰恢复完整手眼与定位旋转重算

- 按用户要求，采用上传0903标定ZIP内T_base_camera及ff202a68同次定位完整pose，替换理想轴旋转；目标[216.791,-311.745,97.865]mm、初值与法兰朝向沿用上轮，无额外补偿。
- 核对pose原点与报告中心局部Z相差−22.5mm，由同次中心与完整标定反算并保存转换；复合平移精确恢复既有中心，相机序列号/旋转正交性验证通过。加入用户钢琴轴映射，完整变换、原始输入、SHA、脚本与说明保存在preview/outputs/left_full_calibration_20261018。
- 受限CPU求解得到[91.0223,-49.4815,-74.0870,-106.4975,52.2377,24.5655,-3.2440]度；FK、URDF位置限位、耦合28.4237≤110.5度及舍入限位通过。舍入位置误差<0.000662mm、旋转误差<0.000143度，仅为模型残差。原定位仍valid=false，未验收路径、碰撞与实机绝对精度；未改训练配置。

## 2026-10-18：双 M6 单场景 IK 计算瓶颈与200Hz CPU优化

- **问题与证据**：用户要求解释200Hz视频为何离线计算慢，并在保留速度/加速度/耦合硬约束下优化。实际参考当前 `arm_follow.py`、`arm_scene.py`、`arm_ik.py` 及Slurm4216同次200Hz位姿/q/v/a记录。Slurm4217独立计时：Torch CUDA双臂step169.92ms，solve两侧149.82ms、耦合9.67ms，额外记录4.49ms；CPU直接执行仍86.44ms。Profiler记录11652次cudaLaunchKernel调用，主要瓶颈为大量微小算子的调度、分配与同步，不能归因于视频渲染。
- **实现**：新增可选 `mjlab_env/arm_realtime.py` 的单场景 `RealtimeArmFollower`，以Numba编译CPU float64循环，候选只算末端姿态，已可行耦合候选直接返回；超界仍做原48轮二分。保留12次迭代、5候选、离散制动、全部位置/速度/加速度/CCS696耦合硬限、观察/计分内容与双侧原子检查。无硬件命令，未替换当前批量训练后端或修改奖励/安装/入口默认。
- **验收**：新增 `scripts/benchmark_arm_realtime.py` 与 `tests/test_arm_realtime.py`。volc-a32 Xeon8336C/A100分配中Slurm4220初测、冻结包Slurm4221复验均完成；最终4 passed。完整2800步与原GPU轨迹最大q差3.34e-14rad、掌根位置差4.19e-15m；36组边界/不可达同状态对照、100组耦合投影、第二侧NaN原子失败通过。最终CPU平均1.707ms、P99 2.053ms；真实时钟200Hz含目标组装/内存记录平均2.900ms、P99 3.089ms、最大3.532ms，2800周期截止时间违约0。
- **归档与说明**：独立源包 `logs/ik_realtime_20261018/source_ik_realtime_v1.tar.gz` SHA `f59746b453a730c1652d7f62bbef8fba7b8524c51cabad8bba66817813fcd072`，基于本日当前源包仅增加上述3个代码/测试文件，完整1119文件审计及远端/本地SHA一致。可选Numba0.67.0/llvmlite0.49.0仅装入远端独立deps目录；没有改共享训练依赖。报告、数组、profiler和作业日志在 `preview/outputs/ik_realtime_20261018/`，同步README及 `docs/arm_ik_realtime_20261018.md`。
- **未覆盖范围**：JIT冷启动约14.63秒，必须控制前预热；本次只证明单双臂计算组件的实测5ms预算，没有验证硬件SDK/网络、长时最坏调度、完整物理/碰撞查询或策略网络的端到端硬实时，也不推断批量GPU训练加速。后续接入须单独统计通信和截止时间，不能把本次候选归档称为统一训练入口已切换。

## 2026-10-18：远端 XYZ 掌根目标与生产 IK 跟随视频

- **需求与修改**：用户要求在远端生成带世界、法兰、实际掌根和目标坐标的连续视频。扩展 `preview/scripts/render_cartesian_installation.py --video`，新增 `preview/scripts/render_cartesian_video.py`；复用当前 `arm_scene.SceneArmFollower.requests/step`、动作 JSON、真实 A3.2/M6/四孔连接件网格。固定全景和掌根近景同步显示 W/F/P/T、实时数值及目标/实际 XYZ 曲线，所有帧直接重放同一次记录的关节 q。
- **轨迹与范围**：14秒，1–5秒右手世界X ±20mm，5–9秒Y ±20mm，9–13秒Z +20/+10mm及返回；左手固定、掌目标朝向固定。20Hz预设位移直接输入上游动作整形之后的生产请求接口，200Hz连续IK与关节轨迹制动；手指保持初态，不执行策略、琴键物理或原谱，因此不伪造MIDI对比或音乐F1，交付同轨迹XYZ对比PNG及200Hz完整数组。
- **来源**：沿用本日安装图的当前源码独立包，SHA `d03a2b1d56fecbd06b2961fcf0ba322a7ab9403fbb5117814000e28bf683ef87`；源包运行时/配置与当前工作区逐项一致。诊断配置保留双臂初角/刚性安装，yaw=15°、scale=1，未回读旧远端项目源码或迁移checkpoint。此次只增加演示工具，未改训练算法、入口默认或固定训练归档。
- **验证与交付**：经 `scripts/hpc/submit.sh` 提交volc-a32/A100 Slurm4216，COMPLETED/0:0，1 passed。2801个子步的关节位置/速度/加速度和J6/J7耦合约束通过，渲染掌根与记录FK一致至1e-8m。最大位置误差3.597256mm（t=4.405s），最终数值收敛；左手固定。输出1080p/25fps/350帧/14秒H.264，完整解码、三个轴画面/曲线目视和远端产物SHA校验通过；文件、配置、报告、脚本和日志位于 `preview/outputs/wuji_xyz_video_20261018/`。
- **实时性边界**：视频按模拟时间1倍速播放；轨迹计算墙钟472.65秒，渲染约150.44秒，200Hz是跟随时间步，不能称已达到实机实时计算吞吐。此组件演示不覆盖上游动作限速器、整曲策略、接触动力学或硬件验证。

## 2026-10-18：按 exper009 物理按键证据修订 Prelude 指法

- **需求与证据**：用户明确授权依据最新Prelude训练结果判断难按位置并修改指法。核对本地及远端已完成轮次为exper009、Slurm3670、25,000,448 transitions；使用step47000 best三次评估与Slurm4104同次200Hz回放。实际参考 `results/exper009/config.json`、`summary.json`、`video_best/midi_comparison/key_comparison.npz`、`video_best/replay_data/audio_substeps.npz` 及checkpoint内嵌原谱，校验SHA/真实qpos阈值/谱事件一致。
- **边界更正**：该训练pose yaw=.15rad，Pitch[-.16,.16]，MCP上限1.2；不能将其表现或八指60°耦合角测试作为yaw30°/新范围真实按键结论。
- **修改**：新增 `scripts/revise_prelude_fingering.py`，按回放及三次评估完整激活事件（含起音未匹配extras）分析保持率；以有样本支持的同手替代、冻结已按住音符与整句连续性约束修订61处。激活到 `training_mjlab/inputs/prelude_bwv847/candidate_fingering/` 的proto/CSV/report，SHA=ad58e67de1998c218e380b7370c95d64ba23fe80d6dfbe26f415184e1a80b33a。完整诊断/旧版备份/图/前后对照位于 `revision_20261018_v2/`，保留初稿不覆盖历史实验。
- **验证**：全部1090音符除part外protobuf逐字段相同；左右手不变；连续时间及20/40Hz同指多键冲突0；项目strict_v1解析20/40Hz通过；`tests/test_prelude_fingering_revision.py` 三项离线测试通过。仅CPU数组分析，无模型编译或GPU操作。
- **TODO与部署边界**：C2长音及无替代证据的高/低音仍列入未解决清单。候选替代覆盖率不是新策略实测提升；未启动训练/修改旧checkpoint或训练归档。后续训练需显式使用新谱绝对路径，不能从旧checkpoint回读指法。


## 2026-10-18：远端 A3.2 安装关系真实几何渲染

- **旧图更正**：Slurm 4213 的 PNG 实际调用 `preview_pose.py` / `ArmPreview`，读取 `preview_mount.json` 和连接件配置，使用预览专用 IK。它没有读取 `current_arm_training_20261011.json` 的生产 IK 初角/固定基座，也没有坐标轴，不能作为当前 `wuji_cartesian_v1` 跟随依据。保留原图于 `preview/outputs/training_scene_render_20261018/`，并更正来源记录。
- **修复**：新增 `preview/scripts/render_cartesian_installation.py`，经 `build_bundle_from_config` 构建当前刚性训练装配，用生产 `SceneArmFollower.requests/step` 在 CUDA 上计算 200 Hz 轨迹，将实际 FK 结果发布到模型。手指保持 reset，不运行策略或琴键物理；测试输入明确为上游动作映射/限速之后的右手位移请求 [0,20,10] mm。输出真实网格、W 世界/F 法兰/P 实际掌根/T 请求掌根坐标、安装局部图与两时刻跟随图，坐标按真实投影穿透显示。
- **配置来源**：当前工作区与本次源包的运行时/配置逐项一致，包 SHA `d03a2b1d56fecbd06b2961fcf0ba322a7ab9403fbb5117814000e28bf683ef87`。原预览配方仍保留过时 yaw=.15、Pitch scale=.8 和指纹；本次新建诊断 config，保留显式双臂初角和安装变换，按统一入口取 yaw=15°、scale=1，并记录实际源码/依赖指纹。未迁移 checkpoint、覆盖原预览配置或更新固定训练归档。
- **验证**：volc-a32/A100 Slurm 4214 实际执行通过；两侧 FK 掌根与编译模型 xpos/xmat 一致至 1e-8，在线关节速度/加速度和耦合检查通过。右手请求后 .05 s 误差14.518 mm，1 s 为约8.89e-14 mm（数值收敛，非硬件精度）；完整配置和逐子步 q/误差记录随图保存。此结果只验证该小位移组件演示，不证明整曲实时吞吐、策略质量或硬件安全。
- **最终交付**：Slurm 4215 使用穿透坐标标注重新渲染，COMPLETED/0:0，1 passed；复现相同误差。全景/安装细节/两帧跟随/四合一PNG、report、render_config 共7项远端/本地 SHA 一致，保存到 `preview/outputs/wuji_ik_frames_20261018/`，含生成脚本、说明和作业日志。发现左侧动作 JSON 的 XY 范围与 `position_bounds_m` 元数据不一致，已在交付说明中如实记录，未改算法或限位。

## 2026-10-18：完成 wrist 动作通道更名为 Palm_Base

- **修改**：将当前 `direct_joint_v1` 及 Cartesian 动作合同中的双侧 `R/L_Wrist_TX/TY/TZ/Pitch/Yaw` 统一更名为 `R/L_Palm_Base_Tx/Ty/Tz/Pitch/Yaw`；同步动作映射表、控制限速覆盖、奖励索引、预览/场景生成和分析脚本。CLI 的 `--wrist-*` 仍保留为用户接口兼容名，内部最终解析到 `Palm_Base` 通道。
- **修复**：翻译通道名称时统一使用大小写敏感的 `Tx/Ty/Tz`，避免动作合同为 `Tx` 而奖励和限速代码生成 `TX` 导致索引丢失；动作合同中的 Pitch/MCP 范围与当前 JSON 同步。
- **验证**：当前活动源码、配置、测试和脚本中不再存在 `R/L_Wrist_TX/TY/TZ/Pitch/Yaw` 动作通道名；源码构建生成 51 个唯一 canonical 通道，动作映射、控制限速、目标运动和奖励 CPU 测试通过。未重新打包固定训练归档，也未启动训练。

## 2026-10-17：更新当前 direct_joint_v1 的 Pitch、四指 MCP 与初始 yaw 合同

- **修改**：当前统一入口的初始 yaw 从 0.15rad 改为 `15° = π/12 = 0.2617993877991494rad`；同步 `source_runtime` 校验范围与 A3.2 前臂 yaw 机械范围至 ±0.3rad，并将左右手初始 yaw 更新为该值。当前仍启用 `--disable-wrist-yaw`，因此 yaw 是固定场景姿态，不是策略动作。
- **修改**：`direct_joint_v1` 的左右 `Palm_Base_Pitch` policy 范围改为 `[0,2]`、物理目标范围改为 `[0,0.3]`；统一入口取消原 `0.8` 缩放并显式使用 `1.0`，避免实际合同再次缩窄。左右食指、中指、无名指、小指 MCP 的物理目标上限从 `1.2rad` 改为 `1.3rad`，下限保持 `0rad`。
- **验证**：JSON 解析与 Python 字节码检查通过；仍需以本次修改后的源码重新打包并完成受限 CPU/GPU 合同验收后，才可把新范围用于训练。旧 checkpoint、旧冻结场景和旧源码归档不迁移。
- **远端验证准备**：新增 `tests/test_finger_reachability_report.py`，在同一编译场景和耦合求解器上逐指打印 `REACHABLE/UNREACHABLE`、目标角误差和 follower 限位状态；测试通过后才按逐指结果报告可达性。
- **远端验证完成**：volc-a32 Slurm 作业 4100 使用归档 `source_training_ranges_20261017_report_v2.tar.gz`（SHA `ba39f1a45e4282566a692384c7ec2b2a7a41aaaef2ad7596936d7ddef142e53a`）运行逐指报告；8/8 非拇指手指均 `REACHABLE`，目标约60°，误差≤0.000011°，follower 未触及限位。stdout/stderr 与逐指结果已回传 `results/finger_reachability_20261017/job_4100/`。

## 2026-10-16：修复闪退后的手指耦合与机械臂冲突检测

- 闪退根因：内核日志显示本机重型 MuJoCo/Python 任务触发全局 OOM，约 5.24GiB RSS 的进程被杀；未发现新的 segfault、GPU Xid 或 Codex coredump。磁盘空间也曾降至约 4.8GiB，增加了日志和回传失败风险。后续本机重型 CPU 检查必须经 `scripts/run_cpu_analysis.sh`，GPU/渲染继续经 Slurm；本轮未启动训练。
- 手指耦合：原来每个20Hz控制帧只求解一次PIP/DIP目标，200Hz物理子步内目标保持不变，有限刚度伺服在接触时产生实际角滞后。现在每个物理子步依据当前qpos/xmat重新求解，保留真实独立伺服、接触和力限，不做qpos投影；控制帧结果额外记录子步实际角误差最大值。耦合入口允许与刚性机械臂同时使用，仍要求完整51通道物理动作布局。
- 机械臂冲突：刚性装配原先生成了查询XML但未接入当前 `SceneBundle`，且曾把查询模型与主仿真数据混用，实际漏掉臂-手/臂-琴/跨臂瞬时接触。新增独立 `rigid_arm_detection_only_v1` 查询模型和Warp跟踪器，200Hz逐子步累计 `arm_arm`、`arm_hand`、`arm_piano`；Link7/法兰/连接件到同侧手的安装界面显式排除。主力求解器仍保持臂网格 `contype=0/conaffinity=0`，碰撞只进入跨手/前臂奖励与诊断，不伪造反作用力。
- 参考文件：`mjlab_env/{finger_coupling,scene,arm_scene,arm_collision,task}.py`、`tests/test_optional_arm_follow.py`、`docs/reward_defaults_20261006.md`、`docs/optional_arm_wrist_ik_requirements_20261008.md`。当前工作区改动尚未重新打包到固定训练归档，待受限CPU/GPU验收后更新归档SHA。

## 2026-10-15：exper007 使用修复配置从头训练 Nocturne

- 用户授权重新开启Nocturne并保持闲指−0.02。核对统一入口、默认完整奖励、Slurm2764验收归档、原171音符指法谱/原始MIDI，以及watch_training_transfer.py/finalize_training_video.py；检查本地和远端编号、排队任务后预留exper007，未改算法或默认参数。
- 通过all_use_this_train.sh仅追加原谱绝对路径和新输出前缀，提交双A100 Slurm2783。当前独立部署为/home/houj/projects/pianist_consistency_20261015，固定归档SHA d835dafd7ff85c778bb80819d266935f41190d5ed362f155c0ba04139739aba3；485个运行文件、交付脚本、原谱及完整奖励SHA预检通过，仅复用既有依赖环境和音频运行库。
- 实际config核对：无臂/fresh、512环境、19532向量步/10000384样本、cuda:0采样/cuda:1学习流水、Euler、腕TX/TY armature=.02kg/力限50N；连续键程physical_key_travel_exact_v1、20Hz控制/200Hz原谱物理子步、闲指−.02/未来20步。原谱/指法不变，八指启动目标校验通过。全部22项奖励已实际打印，保存于results/exper007/startup/reward_startup.log；配置/来源核验见startup_verification.json。首次确认100向量步/51200样本，处于预填充，508929条样本才开始学习，不把此阶段当作F1恢复证据。
- 输出为远端results/exper007-20261015-181236-933661。本地后台PID2052922已脱离终端、持有实验文件锁，首次实际查询RUNNING，60秒轮询；终态后用本轮best.pt经Slurm导出完整有声/指法/指尖标号/实际触键高亮视频，配同次200Hz真实按键与原始MIDI对比PNG/数组/报告，再回传并校验SHA和音视频。submission登记完整流程与原谱SHA，源归档及导出依赖已验证。未创建应用定时任务。
- 当前状态：训练运行中，导出等待训练终态，尚未完成预算或最终交付；失败仍保留诊断并尝试已有best，回传后台依赖本机/SSH保持可用。运行及交付状态以results/exper007中的登记和状态文件为准。

## 2026-10-15：训练模型、腕驱动与连续键程精确采样修复

- 用户要求检查并修复exper001/005差异，明确选择恢复连续键程并保留精准子步采样。参考两轮审计、006冻结MJB/录制、当前URDF/drive/contact、task_logic/exact_score/note_timing_metrics、统一入口和完整奖励配置；无旧源码导入或checkpoint迁移。
- 正确键改为100%Gaussian连续键程，真实激活仍用于错键；原谱post-step/[start,end)、子步平均保持。唯一合同physical_key_travel_exact_v1拒绝旧混合合同；训练和SwanLab元数据同步。准备段偏移计算与原谱诊断统一，默认整帧buffer不受该边界问题影响。旧混合审计/微调脚本及专属测试移入deprecated/training_consistency_20261015，旧reward-diagnostic入口移除，不再提供旧算法执行路径。
- TX/TY保持.02kg armature/Kp300，力限恢复50N；由compiled M0计算阻尼并拒绝质量声明漂移。12组Euler CPU对照显示无接触TX饱和40.56%→0%、移动RMSE44.40→39.45mm；回退3kg为73.74mm。接触压力测试力/穿深增加，实际加速度仍有20Hz阶梯目标尖峰，未宣称硬件安全或实际硬限。44link惯性/镜像/轴/限位与编译独立审计通过，保留当前模型及未标定TODO。
- Euler保持，修正from_model把不支持积分器误标implicitfast的问题。重建当前完整预览奖励及指纹，旧预览隔离保存；闲指−.02、同手不同指−.1/cap5与原谱指法保留。更新AGENTS、README、奖励唯一表、驱动说明和docs/training_consistency_20261015.md。更正training_environment.md现状，local_development_handoff_20261011.md与physical_parameters_technical_report_20261011.md明确旧参数/归档为历史快照。
- CPU已完成奖励/时序86项（32子测试）及监控34项（14子测试）、驱动5项、模型9项；独立44link编译审计通过。初次Slurm2760暴露观察断言与测试参数不一致、预览完整奖励与旧标量冲突；修复并统一当前入口测试配置、未启用诊断的None语义后，最终Slurm2764/A100为COMPLETED/0:0，41 passed、4 subtests passed、1 skipped（外部MJB可选项，等价检查已有独立审计），35条依赖弃用警告。完整日志与首次失败均保留。
- 唯一入口绑定logs/training_consistency_20261015/source_training_consistent.tar.gz，SHA256=d835dafd7ff85c778bb80819d266935f41190d5ed362f155c0ba04139739aba3；归档全量流式SHA校验、本地/远端/作业实际包一致，485个运行文件及预览指纹完全匹配。验收后补记文档不改动已验证归档。未启动训练，未宣称F1已恢复。
- TODO：接触中真实末节角跟踪、连续目标整形、惯性/驱动硬件标定与重训F1仍需独立验证，不能将目标求解0%失败当作物理角或音乐质量保证。

## 2026-10-14：左法兰目标调整为[216.791,-311.745,97.865]mm

- 沿用上一轮最新轴表、ff202a68基座变换、用户截图初值和左M6 URDF，不加补偿；在preview/outputs/left_target_216791_20261014保存脚本、文档快照、结果及README。
- 受限CPU求解结果[89.8897,-49.4249,-74.0730,-103.3551,51.3208,20.6731,-0.2187]度；四位小数FK位置残差<0.000328mm、朝向残差<0.000051度，限位及耦合21.4087≤110.5度通过。未修改训练配置；未验收路径、碰撞及实机。

## 2026-10-14：按更新左法兰轴向重算截图目标IK

- 读取最新T_definition文档，左法兰已改为+X→钢琴+Y、+Y→-Z、+Z→-X；按用户初值[87.127,-59.070,-62.546,-99.752,63.825,19.775,17.498]度及法兰目标[201.790929,-301.745126,99.864982]mm重新求解，未采用旧轴向。
- 新增preview/outputs/left_target_updated_axes_20261014的文档快照、脚本、结果与README；基座为ff202a68定位和理想轴表，使用当前左M6 URDF，零附加补偿。受限CPU scope求解得到[85.9210,-51.3065,-71.8357,-101.0621,52.6066,21.0231,-1.5295]度；FK、限位与耦合23.0783≤110.5度通过，舍入后位置误差<0.000227mm、旋转误差<0.000033度。
- 用户已提供法兰目标，未额外加CAD v4安装偏移，也未独立核验保持手根位置的条件。未改训练配置/归档；未验证碰撞、路径或实机精度。

## 2026-10-14：左法兰截图位置单点IK

- 读取用户左初角[87.127,-59.070,-62.546,-99.752,63.825,19.775,17.498]度与截图法兰目标[201.790929,-301.745126,99.864982]mm；参考T_definition现有左法兰轴表、ff202a68基座定位及当前左M6 URDF。没有应用右侧补偿；截图CAD v4手根条件不作为法兰IK输入。
- preview/outputs/left_target_20261014保存局部求解、16初值复核和最终位移优化记录。受限CPU scope找到符合位置/朝向/URDF/耦合的解；J2到下限且J5相对初值变化约-204.81度，明确不视为保持当前朝向或安全路径。展示J2取-120.0000避免四位小数取整越界，实际显示解FK误差存result.json。
- 未改训练配置、模型或源码归档；仍待确认该左法兰朝向是否为本次期望，未验证碰撞或硬件。来源、限制及手根条件区别见产物README.txt。

## 2026-10-14：按用户要求中断exper006并复核exper001实际角度

- Slurm2505于北京时间16:32:31确认CANCELLED，运行14分15秒；最后完整日志1500向量步／768000样本／64768更新，最后完成评估及best为1000步。保存cancellation.json，保留现有检查点、诊断及原后台导出回传流程；未重新训练或改动算法、奖励、模型和归档。
- 参考两轮完整config/metrics/best、001冻结源码、006匹配源码及各自保存视频的qpos/ctrl/MJB。新增docs/exper001_vs_exper006_20261014.md与results/exper006/comparison_exper001_20261014内审计脚本、SHA、源码差异、编译驱动参数、独立FK数组/图；更新上一轮修复文档的当前状态。旧代码只作为文本读取，不执行旧策略、不迁移检查点。
- 相同步数1000步原谱200Hz合并F1：001为1.1221%、006为5.5823%；用指实际角分别57.051°/53.247°。001首轮最佳F1为90.3493%，实际用指也仅53.516°；完整续训best为93.1626%，实际52.458°。此前目标求解失败已经修复，实际角度未达标是两轮共有问题，不能以高F1当作姿态达标。
- 对比AST确认耦合apply/solve/actual诊断等五方法相同；两轮编译模型均Euler、手指kp1／阻尼.05／frictionloss.01／力矩限±.6。控制器以指令MCP解目标，物理关节独立伺服；没有实际末节角误差补偿和best实际角度门槛。新模型碰撞、惯性、腕驱动与奖励仍有其他差异，未做单变量贡献归因。
- 独立远端CPU FK（不积分）核对原录制源合同和checkpoint SHA，每段7952个八指控制端点。001续训best／006早期best实际区间超容差占67.040%／76.019%，目标均0%。006平均目标－实际5.148°，对称几何分解MCP对应2.681°、PIP/DIP对应2.467°，上游变化均值.000434°；这不是动力学因果干预。另记录10/20/40秒接触几何，未伪称有原GPU力数据。图已查看；本地分析/绘图使用独立受限CPU scope。
- 自动回传已校验15个训练文件，训练成功标志为false；导出Slurm2512使用本轮第1000步best的完整有声／指法／指尖标签／触键高亮回放。导出与最终回传验收单独维护；完成状态见本次审计文档及exper006的sync_status/export_status，不能把已取消训练标为完成预算。
- TODO：受接触负载时的实际角度闭环与关节跟踪验收仍未完成；应先用接触力／力矩记录做短动力学验证，不能仅抬高目标、增大kp或长训来宣称修复。

## 2026-10-14：修复新A3.2指尖与耦合坐标匹配，恢复Euler训练

- 用户要求修复新模型匹配，并延续此前Euler重训要求。参考exper001/005审计、当前左右URDF/凸包、fingertip_geometry、FingerCoupler及用户确认的末节60°～80°定义。确认旧局部+Z定位误取当前末节侧壳。
- 改为URDF相邻铰链位移在末节转动平面内的朝外方向，取末端支撑面／边中心；同步十指site和指法距离输入，保留八指原角度容差／限位／求解器。新增八指参考姿态启动检查和具体失败诊断，耦合合同保存新定位点。模型、驱动、初角、谱面指法未修改。
- 依用户此前要求，将contact_settings中的integrator改为Euler；其余奖励参数沿用exper005（闲指−0.02）。更新奖励唯一总表的指尖依赖与积分器、README、AGENTS和修复说明。
- CPU对照仅替换定位点：旧2016/2016失败，新0/2016；初态八指均约60°且跟随不触限。新增坐标旋转／镜像／表面与独立FK／启动拒绝／CUDA graph回归。修复已有耦合测试的旧90°近似值及被移走的测试helper依赖，不加载deprecated测试或运行时代码。
- 本地场景编译触及受限scope预算（exit137），保留诊断转远端CPU，未提高本地上限。验收日志在results/model_matching_20261014；新归档以原固定归档为基仅覆盖列明文件，source与SHA更新后才用于训练。
- CPU验收：25项测试及39项子测试通过；当前完整命令范围1944个样本均满足原0.05°容差，独立原生FK吻合；原生碰撞表面另1项通过。新归档SHA为c3b88a23a6e2a8ae66f95dd182313eef6b4be449105fa69333b5a9a2870473cb，流式全量验收峰值26.3MiB。
- GPU验收：Slurm2504（A100）Euler编译选项与CPU/CUDA/CUDA graph一致，1 passed/41.15s、退出0:0。唯一入口启动exper006 Job2505，512环境／10000384样本预算，从头训练Nocturne，22项完整奖励及SAC与exper005相同；仅四个运行时源码文件有预期变化。后台PID1743698存活且首次真实查询RUNNING，登记best.pt全曲有声视频及同次对比图、SHA校验自动回传。
- 第0步三次整曲评估23856个手指样本：求解失败0%、触限0%，目标均值60.314669°，最大区间误差0.049946°；实际角均值56.420524°，仍有跟踪误差。训练尚在运行，未报告最终F1或视频交付。
- TODO(V32-07)：仍为几何指尖、非实测指腹；目标可行性不保证物理跟踪或F1。

## 2026-10-14：右臂新初值下法兰单点IK

- 用户图示初值[-69.474,-59.480,51.010,-100.262,-61.573,32.908,11.612]度；目标表达式化简为[202.210,157.265,99.921]mm，沿用[+.85,+4.14,-2]mm补偿及法兰+X→钢琴-Y、+Y→+Z、+Z→-X。
- 沿用最近安装图的理想基座轴表及ff202a68定位反算、当前右M6 URDF和Link7→SDK法兰定义。新增preview/outputs/right_target_new_seed_20261014独立脚本、结果和README；受限CPU scope内最小化初值附近关节位移，解[-80.1231,-59.6800,54.0340,-100.7533,-60.1235,36.3164,0.5762]度。四位小数解FK位置残差0.000630mm以内、朝向残差0.000105度以内；URDF限位、耦合37.8005≤110.5度通过。
- 未修改训练配置或模型；未验证路径、碰撞或硬件。基座采用理想轴关系而非此前含倾斜的相机标定旋转，数值残差不代表实机精度；原始定位仍未完成独立验收。

## 2026-10-14：按用户左臂七角更新安装正面图

- 用户截图左J1–J7为[79.140,-62.376,-78.708,-87.731,65.002,17.019,10.454]度，替换上一轮左侧展示IK解；右臂七角、基座位置/朝向和训练模型刚性安装保持。参考前轮mount_axes_20261014脚本、冻结定位与轴表快照。
- 新产物preview/outputs/mount_axes_left_measured_20261014包含脚本、输入快照、正面PNG、实际位姿report、state、SHA及README。采用固定训练归档经submit.sh单GPU Slurm2474渲染，1 passed/47.34s；双侧限位/耦合、模型手根与FK位姿核验通过，正面图视觉检查及4项回传SHA通过。
- 未修改训练配置、归档或用户文档；静态FK展示不代表碰撞或硬件验收。本轮左手姿态由用户角度决定，不再强制上一轮理想法兰朝向或51mm展示偏移。

## 2026-10-14：按用户轴表导出双臂安装静态图

- 参考用户T_definition坐标文档、冻结定位ff202a68、当前训练固定mount_aligned归档和右臂七角[-87.747,-78.514,73.192,-102.046,-82.404,17.665,14.777]度。训练归档与远端缓存953个文件SHA逐项一致，采用相同钢琴、A3.2手、M6及四孔连接件；法兰—手轴表与现有安装一致，保留约29.57mm偏移。
- 新增preview/outputs/mount_axes_20261014的输入快照、求解/渲染脚本、位姿报告、状态和五视图。基座旋转严格采用用户理想轴表，位置从冻结定位反算；右臂直接FK保持用户角度，法兰朝向与表中理想朝向差3.5084度。左臂固定原训练手根位置未找到满足条件的解，改求附近展示位置，偏移[+1.9734,-51.2700,+0.8450]mm；表中左法兰方向使左手侧立，与训练原手朝向差125.08度，J2/J7接近限位，不能称为原训练初姿。
- 统一submit.sh提交Slurm：2466系统盘满失败，仅清理本任务不完整源码副本；转/data独立目录，2468固定左位置求解失败，2469重复旧脚本主动取消；2470完成求解，2471最终渲染日志1 passed。实际模型手根与构造位姿核对通过，五图逐张检查，8项图像/配置/状态远端SHA本地验证通过。
- 未改训练源码、归档、配置或用户轴表，未执行策略、物理步进、碰撞验收或硬件控制。这是同训练资产的新静态安装预览；config是构建输入，最终姿态以report.json和state.npz为准。来源和复现说明见产物README.txt；TODO为左侧轴向与实际期望姿态的进一步核对。

## 2026-10-14：exper001 与 exper005 完整训练差异审计

- 用户要求查清结果差距。读取四段config/summary/best/metrics及匹配源码；exper001首轮173个指纹、exper005固定归档484个指纹全部核验，无缺失/不一致；旧产物仅文本/资产分析，不导入或运行。
- 新增 `docs/exper001_vs_exper005_20261014.md` 及 `results/exper005/comparison_exper001_20261014/` 的可复核审计脚本/JSON、30项已有源文件变更与311项新增记录、配置差异、XML/STL分析及同预算训练曲线。相关数值/绘图经独立受限CPU scope；未改变训练模型、奖励或源码归档，未提交训练。
- 同原谱/指法/seed/SAC/1000万预算，首轮原谱200Hz合并micro-F1为90.3493%与32.2228%；exper001完整续训best为93.1626%。此前将该跨3回合合并计数的micro-F1简称均值不够准确，现明确口径；不与20Hz逐帧F1混比。
- 最高优先异常：exper005全部41次评估耦合不可达/跟随触限均100%，exper001均0；新DIP局部轴与旧+Z指尖选择假设的相对方向显著改变，8个DIP网格重算与已保存合同一致。另确认碰撞由每手20胶囊+box/mesh变为22凸包、每link惯性改变、腕armature3→.02/力限50→15.32、Euler→implicitfast、按键算法/采样变化、闲指−.1→−.02、同手项筛选/权重变化。
- TODO：精确接触垫/指尖方向与耦合可达性仍需专项验收和修复后受控对照，不能从两轮同时多变量变化估计各项因果贡献。没有迁移旧checkpoint或恢复旧模式。

## 2026-10-14：MJLab预览残留胶囊净空配置

- **问题与实际参考**：用户日志在PianoScene创建FingerClearanceGuard时抛出“Whole-finger envelope requires explicit capsule collision geometry”。当前每link碰撞已为凸包，而专用预览JSON仍携带full_finger_clearance_v1；即使投影开关均false，非null配置仍初始化胶囊诊断。参考用户附件、mjlab_env/{scene,finger_clearance,preview_contract}.py、统一训练入口及exper005/startup/config.json（finger_clearance=null）。上轮CPU SceneBundle重建未覆盖该GPU初始化阶段。
- **修改**：仅将configs/preview/current_arm_training_20261011.json中的finger_clearance设为null，保留当前初角、刚性机械臂、凸包物理及奖励数值。原配置存logs/preview_clearance_fix_20261014/config_before.json；扩展已有test_mjlab_preview_sync.py，核对当前配置与真实GPU scene均不创建胶囊guard。更新README。包内算法与固定训练归档未修改。
- **验收部署**：复用远端本轮独立当前源码快照及已安装依赖，通过scripts/hpc/submit.sh申请单GPU验证。初次封装Job2407因打包会排除两份不用作运行输入的text/ZIP而与完整工作区指纹集合不同，已取消；新验证包的指纹明确由其实际文件集合生成，所有运行源码/模型文件SHA与本地一致，不放松source_runtime验证。仅验证快照省略这两份非运行文件；本地预览配置继续记录完整本地集合。
- **验证与TODO**：远端系统盘满导致第二次封装失败；仅删除已取消验证2407的源码解压副本（原源码归档SHA校验通过并保留）和不完整临时包，恢复1.4GiB空间供Slurm写入。新独立RAM工作目录及临时/JIT路径均位于/dev/shm，单GPU Job2409已RUNNING、GPU探针通过；Job2409已COMPLETED/0:0，双臂真实GPU初始化、reset及一步运行测试通过，原胶囊guard报错已消除。交互窗口仍要求Slurm分配及图形会话。

- **本轮追加无参入口修复**：用户无参启动另触发旧current_h10.../scene/manifest.json缺失；preview_mjlab.py改为从当前工作区预览recipe新建无M6场景，显式arm_follow=null并关闭arm_motion_limit_penalty（其余预览参数继承原preset），显式训练配置和checkpoint保持原合同。新增无臂GPU初始化/reset/step及双侧法兰不存在的验收，--task缺少--scene提前给出可操作错误。同步README和奖励总表的机械臂项启用前提。双臂Job2409已COMPLETED/0:0，GPU单步测试通过；无参无臂分支Slurm2413已COMPLETED/0:0，初始化/reset/step与双侧M6法兰不存在的检查通过。原始GPU日志与探针已回传logs/preview_clearance_fix_20261014/gpu_2409及gpu_2413；未启动交互窗口。

## 2026-10-14：Codex OOM与MJLab预览指纹修复

- **问题与实际参考**：内核日志明确记录10月13日12:07、22:15的ChatGPT界面OOM杀进程，以及多次5–8GiB Python、约7GiB Git进程OOM；10月14日09:41的CPU场景重建同样耗尽内存。本机约13GiB内存/原4GiB swap，根分区最初99%占用。参考用户提供的回溯、当前预览JSON、source_runtime/preview_contract、Codex桌面日志和官方排障页；OOM原始证据存于logs/codex_stability_20261014/oom_evidence.log。
- **实际修改**：备份本仓库及全局Git配置，限制pack.threads=1/windowMemory=128m/deltaCacheSize=64m/bigFileThreshold=32m，gc.auto=0以停止自动大规模打包；维护时须手动受限打包。将22份已停止写入的tmp_pack残留（6,789,630,552字节）逐文件SHA校验后移至/media/hj/Data/codex-repair-backups/20261014/stale-git-packs，清单存于本轮日志；有效pack、Git历史及实验文件保留。清理可重建pip/uv安装缓存；已有环境无指向这些缓存的符号链接，随后依赖版本查询正常。
- **本地保护**：新增scripts/run_cpu_analysis.sh，将大型CPU分析隔离至systemd用户scope，默认MemoryHigh=2GiB/MemoryMax=3GiB/MemorySwapMax=1GiB；读取cgroup实际限额验证通过。新增scripts/setup_local_swap.sh，检查磁盘余量后创建额外4GiB swap并登记fstab。用户在终端执行管理员阶段，实际swapon确认共8GiB且新swap在用、fstab条目存在；脚本语法检查通过。新swap启用后，本轮单次CPU重建scope改为4GiB RAM/3GiB swap预算（已读取实际属性），避免无限制重试。
- **预览修复**：更新configs/preview/current_arm_training_20261011.json的source_runtime.source_sha256及preview_provenance；原配置备份于logs/preview_config_fix_20261014/config_before.json。仅专用当前工作区预览指纹变化，配方、奖励数值、初始角、依赖版本均未改；源码/依赖/臂合同/奖励配置静态校验通过。本机全场景重建超过独立scope预算而终止（CONSTRAINT_MEMCG，Codex主进程保持存活）；随后转远端独立快照CPU验证通过：本地/远端归档SHA、全部源码指纹及依赖版本一致，重建nq=nv=128/ngeom=206及双侧法兰。报告位于logs/preview_config_fix_20261014/remote_cpu_validation.json；GPU窗口仍未验收；训练回放严格校验保持。更新README与AGENTS的本地内存约定。
- **边界与TODO**：未修改包内Python算法、固定训练归档、系统驱动或Codex安装文件；安装更新工具未检查远程Linux发行版本。额外swap、受限Git打包及分析隔离已缓解已证实的内存压力，不能保证所有未来应用崩溃均已根治；应以实际稳定运行和新的退出日志继续判断。GPU预览验收仍须Slurm分配及可用图形会话。

## 2026-10-14：exper005 无臂 Nocturne 与闲指系数−0.02

- 用户要求启动Nocturne训练并将闲指系数改为−0.02；沿用已训练的171音符beginner原谱及generated_v1指法，不重编指法/时值。参考统一入口、奖励JSON、曲谱指法报告、exper004自动回传脚本和原始MIDI。
- 更新活动奖励JSON的闲指coefficient为−0.02，同步本总日志、奖励唯一总表、AGENTS和入口注释；weight=1、未来20控制步窗口及其余训练参数保持。数值配置通过项目绝对路径传入，不修改固定归档中的训练算法。
- 预留exper005；以本地固定mount_aligned归档部署独立远端目录，使用原环境依赖。保存原始MIDI及SHA，扩展finalize_training_video.py和watch_training_transfer.py以支持显式原始MIDI/SHA，复用逐音符核验与同次物理回放绘图，拒绝原谱不匹配。
- 验证：原谱171音符属性与指法proto逐项一致，200Hz物理覆盖通过、每音符最少25采样；5项交付/原谱来源拒绝测试通过；远端固定归档SHA与本地一致，覆盖文件逐一SHA通过。已提交双GPU Job2405，Slurm首次RUNNING；后台PID3134663已脱离会话并记录首次真实RUNNING查询。启动config核验通过：无臂、原171音符proto SHA一致、闲指coefficient=−0.02/weight=1/lookahead_steps=20、512环境与10000384预算、cuda:0采样/cuda:1学习；完整奖励已打印并保存startup日志。Nocturne整回合994控制步，预填充508928样本后才学习，实际起学第508929条。结束后仍须视频、对比图、SHA及音视频验收。

## 2026-10-13：检查训练状态并恢复 exper004 自动交付

- **问题与参考**：核对 `results/exper00{2,3,4}/submission.json`、`sync_status.json`、回传日志、远端Slurm及exper004的summary/best。Job2339于北京时间13:11:39完成，19,532向量步/10,000,384 transitions；本地旧状态停在12:07，原回传进程不存活。
- **操作**：确认exper004文件锁空闲、未有完整交付标记，从原submission以独立后台进程恢复 `scripts/watch_training_transfer.py`，日志追加原transfer.log；未修改训练源码、奖励、配置或启动新训练。
- **验证**：恢复PID 4182226存活，日志首次真实查询为COMPLETED；15个训练文件已回传并通过SHA校验，自动导出Job2378已登记（首次状态PENDING）。best位于19,500步，训练内3回合原始MIDI/200Hz精确micro-F1均值0.9573889393、最低0.9550321199；未达0.99目标。exper002已完成，exper003因右臂硬限校验失败退出。
- **待完成**：后台继续既定best.pt有声视频、同次回放原始MIDI对比图、SHA及音视频验收；以实验内实时export_status/sync_status/transfer_manifest为准，尚不能称完整交付。原进程退出原因本次未单独判定。

## 2026-10-13：强化学习轨迹至实机执行流程文档

- **问题与范围**：整理RL物理回放、钢琴定位、轨迹补偿、刚性安装、首帧七角、连续IK、运动硬限及下发的衔接，防止把离线目标、历史记录或单点IK视为已完成实机执行。
- **实际参考**：`export_video.sh`、`scripts/export_videos.py`、`scripts/plot_recorded_midi_comparison.py`、双侧补偿脚本；`results/{left,right}_flange_ik_20261013/`、`left_wrist_compensation_20261013/`、`right_wrist_ik_20261013/`、`left_arm_joint_trajectory_20261013/`、`right_hand_arm_trajectory_20261013/`、`right_arm_qva_20261013/`中的脚本和报告；当前`arm_follow.py`、`online_control/trajectory.py`及README。
- **修改**：新增`docs/rl_to_hardware_trajectory_workflow_20261013.md`并在README添加入口。逐步列出输入、输出、脚本、命令、完成条件；明确外部标定采集及标准轨迹提取缺口、脚本写死路径/覆盖行为、左右最终参考来源差异、主轨迹/收停和真实硬件的边界。
- **验证**：按脚本静态核对参数与依赖；文档内只读CPU命令核查原记录，并对两侧主轨迹及含收停CSV重新差分验证q/v/a、位置、耦合和运动硬限；检查Markdown本地链接与命令块语法。未重跑训练、IK、GPU、标定或硬件命令；未修改奖励、运行时代码和源码归档。
- **TODO**：参数化录制提取与离线流水线、标定和多点补偿闭环、左侧跟踪误差/真实触键验收、全路径碰撞、进场、手指执行器映射、双臂同步及实机反馈下发，均在流程文档中明确标记未完成。

## 2026-10-13：左臂法兰目标离线逆解

- **请求与参考**：以图片左臂J1–J7初值 `[87.327,-61.546,-71.515,-103.016,-106.164,-20.337,3.729]°` 求钢琴系 `[200,0,100]mm` 法兰位姿；参考0903左臂手眼ZIP/30个SDK姿态、1010右臂30mm修正手眼、前轮匹配采集的 `results/right_flange_ik_20261013/report.json` 与活动左臂URDF。
- **产物与原因**：`results/left_flange_ik_20261013/` 保存CPU计算脚本、完整变换/初值/来源SHA/未舍入和舍入后验算报告、可复制角度及README。沿用已声明钢琴轴转换及法兰朝向；以用户给定中心作为原点，不使用仿真临时基座，不读取可能变化的latest.json。
- **验证**：30组记录核验固定Link7→SDK gripper，距平均变换最大平移差0.00255mm、旋转差0.000528°；有界多初值IK加局部关节距离优化，检查FK、URDF位置限位和CCS696耦合；J6/J7额外收紧到±60°/±90°以消除URDF小数近似越界。
- **限制/TODO**：源定位仍CALIBRATION_REQUIRED，数值残差不代表真实误差；J5大幅变化、J6靠近边界，未验证连续轨迹与碰撞，未下发硬件。未修改训练配置、奖励或源码归档。


## 2026-10-13：追溯左侧候选补偿并恢复镜像四孔安装

- 用户先要求追查偏置并导出图，随后明确要求左侧安装与右侧一致、四孔对齐。参考当前连接件配置、两侧活动URDF/STL、`preview_connector.tool_transforms`、`arm_scene.augment_bundle`及10月11日历史候选文档（只读追溯，未运行历史源码）。确认活动配置与固定基座/七角镜像候选字节相同；未找到单独晋升记录。旧左侧补偿使手—连接板四孔错开22.33–41.70mm、面倾斜14.6167°，旧法兰时钟使单独小孔偏离35.35534mm。
- 修改 `configs/preview/hand_connector_20260930.json`：左offset归零、左nominal按右侧的Link7-X/手根-Y镜像生成；同步 `preview/assets/tj_m6/preview_mount.json` 和 `configs/preview/current_arm_training_20261011.json` 臂合同/来源SHA。右侧安装、初始七角、奖励/物理参数不变。旧配置备份于 `preview/outputs/mount_alignment_20261013/before/`。
- 新增同目录CPU `audit.py`、实际网格及接口PNG、孔坐标/变换/来源SHA的report、README；同步机械臂要求文档。从当前网格提取孔边界，两侧手部四孔中心最大差2.31e-6mm，法兰四孔9.96e-7mm，单孔1.40e-6mm。不是实物精度。保持法兰/手根29.567719mm固定原点距离。
- 归档：以当前exper004归档为基，仅替换三个安装配置和三个维护文档，其余成员逐文件SHA保持不变；新归档 `logs/mount_alignment_20261013/source_mount_aligned.tar.gz`，SHA `90a0a1a71a57d3ba9f9c78c7ea61564e42600f96c4eb67b05695d0c9b99da5bd`。流式打包/完整回读SHA验收通过，峰值RSS约26MB；统一入口与AGENTS已同步，原实验归档保留。
- 闪退追查：用户要求排查Codex反复退出；内核/systemd在12:07、12:17、12:18、12:21明确记录全局OOM及桌面应用scope的oom-kill。重型本地装配测试Python约5–8GB RSS，本轮最后一次还与全内存归档并行，重复本地恢复加重压力。停止本地重型测试，迁移远端CPU验收，归档改为逐块流式，失败归档保留为incomplete_oom。诊断与过滤证据在 `logs/codex_crash_20261013/`。未清缓存/会话或改驱动、swap、应用设置；GPU子进程另有一次recoverable异常，不将它认定为上述全应用退出原因。
- 专项验收：远端同952成员源码CPU完成5项相关测试；连接件交互、初始角/源合同、基座反算、native固定链及不可达请求跟随均通过。首次宽选另有1项旧胶囊净空诊断因当前凸包模型不提供胶囊而失败，记录为已有不兼容，未把整套测试称为通过。验收日志保存在 `logs/mount_alignment_20261013/`。
- TODO：实机紧固、公差、单孔用途/时钟仍待标定；新安装同初角/手姿会改变反算左基座，旧固定基座/可达域不复用；历史训练/检查点不改、不迁移。本任务未启动训练或硬件操作。

## 2026-10-13：按右臂新截图姿态重算目标逆解

- 参考新截图七角、results/right_flange_ik_20261013/report.json及当前右臂URDF；沿用上轮坐标与朝向，未更新标定。
- 新增current_pose_recalculation计算脚本、结果与README.txt；在终点位姿/位置限位/耦合约束下局部最小化关节位移平方和。
- 四位小数解FK残差0.000148mm、0.0000527度，耦合71.40676度，小于110.5度。未运行GPU或硬件。TODO：定位有效性及路径碰撞仍未验收。


## 2026-10-13：右臂法兰目标离线逆解

- 参考用户0903/1010手眼ZIP、匹配中心坐标的latest.json、右臂活动URDF及图片轴向。采用右侧30mm修正版手眼旋转和用户给定中心原点。
- 在results/right_flange_ik_20261013保存CPU计算脚本、report.json与README.txt；未改训练参数或归档。
- 30组记录的Link7到SDK变换平移差最大0.00423mm、旋转差最大0.000760度。逆解检查URDF限位与J6/J7耦合。
- TODO：定位valid=false，坐标约定和实机误差/连续路径尚待验收；数值残差不代表硬件定位精度。


## 2026-10-13：trajectory_bundle 当前模型逐帧视频

- 用户要求将 `/home/hj/下载/trajectory_bundle.zip` 在当前环境导出视频，并明确选择原轨迹逐帧回放。补充统一 `export_video.sh` 的 `--trajectory-bundle` 路径，经现有submit.sh/export作业分发；新增 `scripts/export_trajectory_bundle.py`、`preview/scripts/render_trajectory_bundle.py`，不调整训练、奖励或默认归档。
- 输入参考为包内关节/手根矩阵，以及按SHA匹配恢复的原记录physics_trace、config、report、scene XML。旧XML仅解析关节索引和键程；不载入旧模型、checkpoint或旧源码。以当前A3.2/convex_hull_v1/裁剪掌部构建无臂场景，按名称填关节位置、通过mocap支撑恢复手根变换，不执行mj_step或策略。
- 原注册小星星MIDI、library、midi_file和琴键阈值源码须与原config散列相同；音符保留原始起止时刻及指法，记录去除曲首静音和准备段偏移。实际琴键、声音及F1均来自同一原始200Hz物理记录，明确不是当前几何重新演奏的成绩；保存对比PNG、事件、数组、来源和SHA。修正归一化验证按音符属性排序核对，不能假设protobuf序列化保留音符顺序。
- 验证：包内manifest、完整46关节1831帧位置/速度、原记录/config/XML来源链CPU核对，篡改补充记录拒绝测试通过，既有视频导出54项通过。独立当前源码部署的Slurm2338渲染407帧，2560×720、40fps、9.15秒轨迹＋末帧停留，全部1831帧手根矩阵最大误差6.66e−16；当前渲染快照SHA为881dd7a109cf2acd16f84bc5c6f41367f29101f464912d7aa9d95389b5ea9746。
- 恢复：Slurm2338在完成渲染、音频合成及FFmpeg解码后，因远端缺ffprobe于最终检查退出1:0；保留原视频，用本地CPU `recover_validation.py` 补验收，没有重新运行策略或重新渲染。27个回传文件SHA匹配；本地ffprobe确认407帧/H.264＋AAC/10.175秒，完整音视频解码通过，音轨RMS=0.04325。导出工具新增ffprobe缺失时的显式视频/音频双stream映射解码校验，不将缺工具当成缺流。
- 本次原物理单回放200Hz micro-F1=0.9925301666（TP2591/FP15/FN24），不是当前模型成绩。全部视频、对比PNG、事件/数组、原MIDI、报告及SHA位于 `preview/outputs/trajectory_bundle_current_20261013/rendered/`；训练未启动。TODO：当前通道同名不代表新旧手指末端几何相同；不提供实机映射或物理跟踪结论。

## 2026-10-13：exper004 降低闲指与同手碰撞惩罚，自动视频交付

- 用户要求小星星重新训练1000万 transitions：闲指 L1 系数改为−0.05，同手不同指子步对次系数改为−0.1（JSON weight=0.1，raw已有负号），同手整项每控制步−5封顶不变。其余统一入口参数、原谱物理子步按键算法、模型与指法保持，从头训练。
- 实际参考 all_use_this_train.sh、configs/rewards/default_20261006.json、scripts/watch_training_transfer.py 与现有 export_videos/plot_recorded_midi_comparison；同步唯一奖励总表、同手说明、README和AGENTS。仅改这两个奖励数值，不新增模式。
- 补齐后台 best.pt 导出：训练终态先回传诊断，经现有Slurm导出入口生成全曲有声/指法/指尖编号/触键高亮视频，随后从同次200Hz按键记录生成原始MIDI对比PNG及数据。新增 finalize_training_video.py 对源指纹、原谱来源、PT、录制及音视频解码核验；本轮自动谱源支持注册小星星，其他曲目不猜测。
- 回传状态独立记录训练/导出/验收；只有规定视频、对比图和本轮best.pt匹配并通过远端/本地散列与音视频验收才创建完成标记。恢复使用已登记导出作业，模糊提交拒绝重复GPU提交，失败保留远端与日志。后台每60秒查询，无模型轮询。
- 验证：奖励配置与交付状态CPU测试15项通过；合成CPU交付fixture验证同次按键PNG/事件、音视频解码与幂等重试（非模型F1）。34个原谱事件与已核验参考逐事件吻合。远端重建归档与本地SHA完全一致：3c024ff0f5ada93e19db50d71f4fd2ad0e0a9fadb5de7f8c9e7135f789a7c9d2。
- 已提交 exper004 / Slurm2339，预算19,532向量步×512=10,000,384 transitions；首次真实状态查询RUNNING。本地后台PID1104752，完整记录见 results/exper004/submission.json、transfer.log。长期F1与最终视频须等本轮训练后验收，不预报成功。该提交记录在固定归档生成后补记，不改变归档内运行代码。

## 2026-10-13：恢复 Job1724 按键子步评分，移除端点算法

- 用户要求恢复原0.99实验的按键算法并删除现算法。实际核对Job1724归档rewards.py及保存config：paper、和弦关闭、原谱exact_goal_history逐物理子步、正确键50%键程/50%真实激活、每步取子步均值。历史源码只读审计，不作为当前运行时导入。
- 当前task_logic/rewards统一调用唯一按键函数；移除legacy按键和周期末纯键程实现，强制完整物理键程/激活历史，原谱目标按[开始,结束)处理。task接入每步采样；统一入口拒绝关闭exact timing，config/checkpoint以physical_key_activation50_v1隔离。其他奖励（含当前同手不同手指范围）、模型/指法/硬限保持当前实现。
- 更新default奖励JSON、当前预览JSON、train保存时序元数据、AGENTS、README及奖励唯一总表对应当前行；没有恢复旧采样CLI或新增可切换旧模式。原endpoint配置证据隔离至deprecated/reward_experiments/key_endpoint_20261013。
- 同步统一入口中已不适用于convex_hull_v1的旧胶囊净空诊断及其停止门槛，与exper002实际部署一致；原先投影已禁用，不改变实体碰撞或运动硬限。启动测试直接核对当前train_from_source入口，不再调用已隔离的train_coupled脚本。
- CPU验证91项通过、134个子测试通过、3项GPU/Slurm测试跳过。Slurm2336以最终归档验收15项通过、2个子测试通过、COMPLETED/0:0；先前2334为EGL路径缺失、2335为测试初始化遗漏限幅器，均已在验收环境/测试中修复。最终SHA为`3df73c1d1ae081f8cc056da01623b2e95d457881a45cd04987e3fc15b4ccca86`。验证覆盖中途误触、仅末子步按下、激活与键程分离、原谱子步起止、静音、归一化、旧合同拒绝、局部reset和CUDA图。CPU及Slurm物理集成结果、Job ID、归档SHA收据集中于logs/key_substep_restore_20261013；短验收不证明F1恢复0.99，本次未启动正式训练。
- 修正相关测试对已删除endpoint公式和缺项静默兼容的旧断言。新训练归档与统一入口同步更新；旧实验/视频留作历史结果，不宣称已用新奖励重训。

## 2026-10-13：同手奖励限定不同手指，删除掌侧矩形凸起

- 按用户要求，`self_collision.py`按活动URDF的20个指节body/手映射五指，仅不同指计罚；掌壳、同指MCP/PIP/DIP/Swing及拇指rota_block/rotaback_link2/MCP/PIP均排除。`task.py`使用专用筛选计数，完整接触诊断/物理过滤保持；权重0.2、子步累计、封顶5与跨手项不变。独立验收快照奖励合同为`shadow_control_frame_distinct_fingers_v1`，拒绝旧JSON，不保留旧评分开关；活动代码保留主会话随后加入的`physical_key_activation50_v1`合同，本次未覆盖其按键奖励算法。
- 参考用户图片、活动URDF与`source/latest_project/meshes/left/base_link.STL`：识别0.2mm薄片延伸至Y=86.04mm，`scripts/remove_palm_tab.py`在壳内Y=48mm裁剪并封口。直接修改网格而非隐藏geom；同步大网格OBJ，`sync_project_urdf.py`重建镜像右手、44个凸包及两份代理清单。只有左右base_link视觉和凸包改变，359378个其他三角面保留；关节原点/轴/限位不变。总质量仍采用原暂定952g，base的均匀凸包惯性随几何更新，非硬件标定。
- 旧模型/奖励实现备份：`deprecated/model_revisions/palm_tab_20261013/`。当前总表、专文、README、AGENTS同步更新；独立验收归档`validation_snapshot.tar.gz`已打包并校验SHA；为避免覆盖主会话正在修改的按键奖励，本次不切换训练入口。已有训练和导出作业使用其原快照，不作迁移。
- 验证记录：`logs/finger_collision_mesh_20261013/`保存网格裁剪报告、CPU接触→奖励测试（独立版本39通过，含5个子测试；主会话更新后专项28通过）、几何测试（11通过，含44个子测试）、Slurm静态渲染及源码归档审计。Slurm2333完成并回传SHA校验：8组真实模型接触经CUDA计数均符合筛选，初态8对同手物理接触/0对计罚；7个变动姿态仍计出不同指接触。预览见`logs/finger_collision_mesh_20261013/rendered/comparison.png`。当前没有启动训练，长期收敛和硬件动力学仍未验证。

## 2026-10-13：2267 小星星低 F1 诊断

- 核对 exper002 的 config、best、summary、metrics 与 Job2327 同次回放；只读取历史 Job1724 保存配置/成绩作为对照，不运行历史代码、不改变当前训练参数。1724 的 best 原谱精确 micro-F1 三回合最小值为 0.9913378248；2267 同名最佳指标为 0.1999438360，导出完整单回放200Hz F1为0.1659854520，两种范围不混称同一分数。
- 两轮均小星星、512环境、约1000万 transitions，SAC学习率/gamma/网络等相同；2267换成更新URDF/关节坐标与 convex_hull_v1，增加腕加速度硬限、运动需求惩罚和闲指惩罚/自碰撞惩罚，公共按键奖励也从旧子步/激活混合变成当前控制周期末评分。不能宣称是复现原0.99环境。
- CPU复核183步录制：运动惩罚每步−5，仅TCP限幅前需求就足以使全部步饱和；这不是实测物理速度越界。按键奖励均值0.289、指法0.389、闲指−1.074、自碰撞−0.101。训练日志快照显示自碰撞罚显著改善而按键/指法改善有限；这些快照均值不是全训练逐步均值。
- 原谱对比TP1198/FP10622/FN1417均为键×物理采样点，表现主要是大量误触。高分Nocturne回放同样运动罚恒−5，故不能以负奖励总量认定唯一根因；模型、接触、奖励及控制合同的因果贡献仍需受控实验。
- 保存可重算脚本与数据：`results/exper002/diagnosis_20261013/analyze.py`、`diagnosis.json`、历史配置参考与本次录制scene XML。用 training/.venv/bin/python 在CPU执行通过；未训练、未推理、未修改奖励公式或归档。TODO：在当前模型上验证接触/耦合几何，并在保持硬限的条件下做单因素实验定位贡献。

## 2026-10-12：恢复最新模型坐标系并补回 Rot1 电机网格

- **问题**：前一版裁剪将 `hand_base` 直接当作 `base_link`，丢失源模型 `base_link -> hand_base` 的固定安装变换；同时删去固定 `thumb_mount` 时没有合并其电机视觉网格。
- **修复**：在 `tsensei_v3_2_pianist/assets/urdf/TsenseI_left_updated_project.urdf` 中恢复该固定变换对根部几何和 20 个输出根关节的复合变换；把各手指固定安装块合并到对应 MCP 输出 link，并补回 `thumb_mount` 的 Rot1 电机视觉网格（含 `NR_G0036.stl`）；根部惯性张量同步旋转到保留的源 `base_link` 坐标系。
- **验证**：裁剪模型 20 个保留 link 在零位的齐次位姿与完整 portable 源模型逐项误差小于 `1e-12`；电机网格引用存在；MuJoCo 构建成功（21 bodies、20 joints、20 actuators）。

## 2026-10-12：将最新模型裁剪到训练入口关节合同

- **用户要求／目的**：从 `TsenseI_left_updated_project.urdf` 去除最新模型的 14 个 prismatic 联动关节，并使关节集合与训练入口 `TsenseI_left.urdf` 的 20 个可动关节一致。
- **活动文件**：`tsensei_v3_2_pianist/assets/urdf/TsenseI_left_updated_project.urdf` 现在保留 21 个 project link、20 个 revolute joint 和 20 个 actuator；14 个 slider 关节、14 个 rod 分支及其断开的安装辅助分支已移除，20 个关节名称与训练入口逐项相同。
- **模型内容**：输出关节的轴、原点、限位和网格来自最新 portable 模型；由于源模型没有 inertial，沿用训练入口对应 link 的 inertial 以满足现有 MuJoCo 加载合同。
- **旧版本**：裁剪前的 55-link/54-joint 文件保存为 `deprecated/model_revisions/updated_urdf_20261012/TsenseI_left_updated_project_full54_before_reduction.urdf`。
- **验证**：XML、关节集合和网格路径检查通过；`urdf_model.build` 成功生成 21 bodies、20 joints、20 actuators；未启动训练或 GPU 作业。

## 2026-10-12：替换刚刚生成的项目 URDF 为最新中指 DIP 模型

- **用户要求／目的**：移除上一轮生成的 `TsenseI_left_updated_project.urdf` 简化模型，换用 `/home/hj/hand_middle_dip_standard_description/hand_middle_dip_standard_portable.urdf`，只同步项目需要的 20 个独立输出关节名称。
- **活动文件**：`tsensei_v3_2_pianist/assets/urdf/TsenseI_left_updated_project.urdf`；保留新模型的 55 个 link、54 个 joint、滑块／连杆联动、原点、轴、限位和网格内容；仅将 20 个独立输出关节改为 `left_hand_*` 项目命名。网格按 portable 文件的相对路径放在 `assets/urdf/meshes/visual/`。
- **旧版本**：上一轮生成的简化 URDF、转换脚本和说明移至 `deprecated/model_revisions/updated_urdf_20261012/`，不再作为活动输入。
- **运行边界**：当前训练入口仍读取 `assets/urdf/TsenseI_left.urdf`；新模型含 14 个 prismatic 联动关节和 34 个 revolute 关节，未将其冒充当前 20-DOF MuJoCo 训练合同。
- **验证**：XML 解析通过；新文件含 55 links、54 joints（6 fixed、34 revolute、14 prismatic）；307 个网格引用全部存在；20 个项目输出关节名称均已核对。未启动训练或 GPU 作业。

## 2026-10-12：预览坐标轴增加双侧 M6 关节坐标系

- **用户要求／目的**：在 `--show-coordinate-frames` 预览中同时显示左右机械臂 J1–J7 的移动 URDF 子链接坐标系。
- **修改位置**：`tsensei_v3_2_pianist/preview_frames.py` 从 `ArmPreview` 的独立运动学模型读取每个关节子链接的世界位姿；`preview/scripts/preview_pose.py` 传入 `arm_preview` 并更新坐标系说明；`docs/common_commands.md` 更新命令注释。
- **坐标轴口径**：保留 World 与左右 normalized wrist；新增 `Left/Right J1`–`J7`，RGB 对应局部 +X/+Y/+Z，轴随 ArmPreview 的 IK 姿态更新。
- **验证**：将执行现有预览坐标轴测试及带 M6 的 CPU headless smoke test；未启动训练或 GPU 作业。

## 2026-10-12：将活动 URDF 的 MID_PIP mesh 改为对应 INDEX_PIP mesh

- **用户要求／目的**：让左右手 `MID_PIP` 使用与对应 `INDEX_PIP` 相同的 mesh 文件，同时保持中指 PIP link 和关节连接关系不变。
- **修改位置**：`tsensei_v3_2_pianist/assets/urdf/TsenseI_left.urdf`、`tsensei_v3_2_pianist/assets/urdf/TsenseI_right.urdf`；各文件仅替换 `mid_PIP_link` 的 visual/collision mesh filename；同步更新 `configs/preview/current_arm_training_20261011.json` 中两份 URDF SHA-256，避免当前预览重建合同失配。
- **保持不变**：`mid_PIP_link` 的 link 名、惯性、`mid_PIP_joint` 的 parent/child、origin、axis、limits，以及后续 `mid_DIP` 连接。
- **碰撞口径**：当前活动 `shadow_primitive_v1` 使用独立的 `assets/collision_proxies.json` capsule 代理，本次没有擅自重算或修改该独立碰撞合同。
- **验证**：已完成 XML 解析、mesh 文件存在性检查和左右手中指 PIP parent/child 关系核对；未启动训练、未生成 GPU 场景、未发送硬件命令。

## 2026-10-12：修复 MJLab 预览 Control 面板不驱动手部

- **问题**：`preview/scripts/preview_mjlab.py` 无策略手动预览始终调用 `viewer.sync(state_only=True)`；该同步模式只同步显示状态，不把 MuJoCo Control 面板修改的 `data.ctrl` 写回控制循环，因此拖动手指／腕部滑块不会进入手部动作。
- **修改**：手动预览改用完整同步；确定性 checkpoint／策略回放继续使用 `state_only=True`，避免控制面板覆盖策略状态。手动目标随后照原有物理动作映射进入 MJLab `scene.step`。
- **验证范围**：完成源码静态核对；需在带桌面的 MJLab 窗口中拖动滑块确认可见手部运动。暂停状态下仍需按 Space 运行位置执行器，Control 面板只修改目标值，不绕过物理步进。

## 2026-10-11：整理当前手、钢琴、机械臂与控制器物理参数技术报告

- **用户要求／目的**：将当前工作区可核对的手、钢琴、机械臂、PD/IK、接触和未标定参数集中整理为一份带源码路径的 Markdown 技术报告。
- **新增文档**：`docs/physical_parameters_technical_report_20261011.md`。
- **实际参考**：`configs/preview/current_arm_training_20261011.json`、`configs/actions/direct_joint_v1.json`、双侧 TsenseI URDF、双侧 M6 URDF、`configs/preview/m6_sdk_joint_limits.json`、`tsensei_v3_2_pianist/models/{hands,piano}`、`mjlab_env/{source_runtime,bundle,arm_scene,arm_follow}`、`assets/{contact_settings,drive_settings,collision_proxies}.json`。
- **报告口径**：明确区分当前合同实际生效值、源码默认/历史兼容值、仅存在于 URDF/CAD 但当前动力学不使用的名义值，并记录工作区缺少 `all_use_this_train.sh` 与固定训练归档的复现边界。
- **验证**：只读核对源码路径、JSON 参数、左右 URDF 质量/惯量/关节限制、当前臂合同、接触配置和 PD/IK 构建逻辑；未启动训练、未修改模型算法、未发送硬件命令。

## 2026-10-11：清理失败方案并固定当前设计

- 将机械臂活动合同统一为 `controller=rigid_wrist_command`，保留原刚性装配、连续 IK、目标制动、SDK/URDF 硬限和 CCS696 J6/J7 耦合约束。
- 将当前双侧 reset 初始角设置为配置中确认的右侧
  `[-84.27055, -68.29417, 61.39675, -89.05803, -64.834334, 27.92905, -7.30834]°`
  与左侧
  `[84.27055, -68.29417, -61.39675, -89.05803, 64.834334, 27.92905, -7.30834]°`。
- 删除活动代码、配置和测试中的机械臂递增版本选择逻辑；旧臂方案与候选产物移到 `deprecated/arm_control_experiments/`。
- 删除活动 humanoid 奖励的多模式分支，只保留闲置手指相对 reset 目标的 L1 公式：`coefficient=-0.1`、`reference=scene_reset`、`lookahead_steps=20`。旧奖励 JSON、实验目录、模式测试和说明移到 `deprecated/`。
- 继续扫描后，将质量训练变体、几何/任务条件奖励说明、40Hz 对照、旧队列脚本及其测试全部移入 `deprecated/`；活动目录不再保留这些候选入口。
- `TargetMotionLimiter` 的活动合同改为 `controller=wrist_target_braking`，不再用递增 schema 表示算法版本。
- 活动入口不再承担历史 checkpoint 兼容转换；错误设计直接修改当前实现并重新开始训练。`deprecated/` 仅用于隔离和追溯，不会被当前训练入口读取。

## 验证

- 活动合同构建通过（128 nq、40 nu）；机械臂回归 16 项通过、奖励/目标运动 52 项通过、预览姿态 6 项通过，Python compileall 通过。
- 搜索活动目录确认没有机械臂递增版本、humanoid 旧模式或旧碰撞奖励分支残留；剩余 schema 字段是序列化/低层合同字段，不表示机械臂或 humanoid 的候选算法。
- 设计上训练仍应使用根目录 `all_use_this_train.sh` 并在启动前打印完整奖励配置；该文件当前不在工作区，本轮未启动训练。

## 当前初始角同步

- 活动配置 `configs/preview/current_arm_training_20261011.json` 使用右侧 `[-84.27055,-68.29417,61.39675,-89.05803,-64.834334,27.92905,-7.30834]°`、左侧 `[84.27055,-68.29417,-61.39675,-89.05803,64.834334,27.92905,-7.30834]°`；清理前的预览快照已隔离。
- 左侧镜像候选配置、连接件和导出脚本已移入 `deprecated/`；旧 `BUNDLE_MANIFEST.json` 也已隔离。
- 工作区当前没有根目录 `all_use_this_train.sh`，因此本轮只完成活动配置和代码清理，没有宣称已提交训练或更新固定训练归档。

## 2026-10-11：活动配置清理

- 清理前生成的重复预览快照 `current_arm_training_20261011_v8.json` 已移入 `deprecated/arm_control_experiments/`；活动 `configs/preview/` 只保留无版本后缀的当前配置。
- 活动配置的源文件散列、刚性机械臂合同、双侧初始角和当前奖励合同已重新核对；没有生成新的训练归档，也没有把旧 checkpoint 接入当前实现。
- 删除奖励实现中残留的 `collision_penalty_v1` 分支；当前同手子步聚合名称统一为 `substep_sum`，不再带递增版本后缀。
- 机械臂默认安装变换统一从当前四孔连接件配置计算，修复旧预览安装文件导致的左侧刚性装配校验失败。
- 清理前生成的预览场景、安装图和旧奖励比较测试移入 `deprecated/arm_control_experiments/outputs/` 与 `deprecated/tests/`；README 和环境说明改成当前合同的短入口，历史全文保存在 deprecated。

## 2026-10-11：左臂初始角镜像绝对值同步

- **用户要求／目的**：左臂 J1–J7 的初始角使用右臂对应角的绝对值，并采用当前镜像符号约定 `[+,-,-,-,+,+,-]`。
- **实际修改**：更新 `configs/preview/current_arm_training_20261011.json` 的 `arm_follow.initial_q_rad.left`；左侧角度变为 `[84.27055,-68.29417,-61.39675,-89.05803,64.834334,27.92905,-7.30834]°`。同步 README、训练环境说明和物理参数报告，避免活动文档继续引用旧 J5/J6/J7。
- **验证**：将新角度按当前左侧 M6 URDF 位置限位与 CCS696 `1.025×|J6|+|J7|≤110.5°` 约束校验；未启动训练或生成新的源码归档。

## 2026-10-12：Nocturne Op. 9 No. 2 候选指法

- **用户要求／目的**：为用户提供的 `/home/hj/下载/nocturne-op-9-no-2-beginner.mid` 生成带 `NoteSequence.part` 的候选指法。
- **实际修改**：新增 `training_mjlab/inputs/nocturne_op9_no2_beginner/generated_v1/`，轨道 0 映射右手、轨道 1 映射左手；生成 171 个音符的 `.proto`、逐音符 `fingering.csv`、指法报告和说明。候选以 20 Hz 时间网格生成，适配当前统一训练的控制频率。
- **验证**：原始音符属性和 171 个音符均保留；20 Hz 与 40 Hz `strict_v1` 覆盖检查通过、同一物理帧无同指多键冲突。算法是启发式候选，尚未经过专业演奏者、TsenseI IK、碰撞或实机验收。

## 2026-10-12：Nocturne 指法交互预览

- **用户要求／目的**：在交互式 MJLab 窗口中显示 Nocturne 每个目标音符对应的左右手指法。
- **实际产出**：新增 `preview/outputs/nocturne_op9_no2_fingering_interactive_20261012/scene`、`task` 和 `task_preview`；`task_preview` 使用同一带指法谱面、20 Hz/200 Hz 物理、`tsensei_collision_v1`，关闭仅影响训练评估配置匹配的 `exact_note_timing` 开关，未改变音符或指法。
- **验证**：任务导出成功（974 控制帧、1180 维观察）；交互预览已启动，指法颜色与 `R1–R5/L1–L5` 标记可见。预览窗口仅用于查看与手动控制，不代表已训练策略。

- 2026-10-12: Converted `/home/hj/hand_middle_dip_standard_description/urdf/hand_middle_dip_standard.urdf` to `tsensei_v3_2_pianist/assets/urdf/TsenseI_left_updated_project.urdf`; removed 33 auxiliary slider/rod joints, mapped the 20 output motions to the active `left_hand_*` contract, renamed `hand_base` to `base_link`, rewrote mesh paths, and restored project inertials for runtime loading. Verified MuJoCo parses the output (`njnt=20`, `nbody=21`). Absolute CAD base-frame transform remains TODO because the upload contains no explicit transform.

- 2026-10-12 follow-up: Applied the requested project frame to `assets/urdf/TsenseI_left_updated_project.urdf`: source +X (middle-finger) -> project +Z, source +Z -> project -X, origin at the four-hole mount `[0.0237, 0.0065, 0.0441380069844]`; changed `left_hand_thumb_rotaback_joint2` to positive `[0, 1.74]` rad and reversed its axis to preserve physical direction; retained and verified rota-block motor/housing meshes. MuJoCo parse remains valid (`njnt=20`, `nbody=21`).

- 2026-10-12 repair: Reverted the experimental global frame rotation after the viewer showed separated links. Restored the uploaded model's original internal origins/orientations in `assets/urdf/TsenseI_left_updated_project.urdf`; retained positive rota-back range/axis reversal and motor meshes. Moved the incorrect frame script to `deprecated/tools/`; MuJoCo validation remains `njnt=20`, `nbody=21`.

- 2026-10-12 motor-visual repair: Restored the source `thumb_mount` motor geometry into `left_hand_thumb_rota_block_link`, including `NR_G0036.stl`; the source thumb mount and zero-origin `thumb_rot1` share the same visual frame. XML and MuJoCo loading validated (`20` joints, `21` bodies).
## 2026-10-12：Bach Prelude BWV 847 候选指法

- **用户要求／目的**：为用户提供的 `/home/hj/下载/prelude-bwv-847.mid` 生成候选指法并加入当前项目。
- **实际产出**：保留并整理 `training_mjlab/inputs/prelude_bwv847/candidate_fingering/`，其中包含带 `NoteSequence.part` 的 `prelude-bwv-847_fingered.proto`、逐音符 `fingering.csv` 和 `fingering_report.json`；新增同目录 `README.md` 说明来源、轨道映射、生成口径和使用限制。
- **生成口径**：轨道 0 映射右手、轨道 1 映射左手；使用现有动态规划启发式和 `dt=0.05` 秒。项目内 `prelude-bwv-847_timing_normalized.mid` 保留原始 1090 个音符，移动 conductor tempo／调号／拍号事件以供导入器正确读取。
- **验证**：原始 MIDI SHA-256 为 `9e29d5b640d875bcfa93824cc963a4d9419ce2139cfd4e786148d45127e602d8`；候选 proto SHA-256 为 `bc8a027f4666af72fc09d2027b42e27c346ecb55c7ae811f43a8de91c15384b6`。重新生成结果与已加入项目的 proto SHA 一致；音符属性全部保留，20 Hz／40 Hz `strict_v1` 覆盖校验通过，同一轨迹帧同指多键冲突为 0。
- **未解决 TODO**：该指法仍是工程候选，尚未经过专业演奏者、TsenseI IK、碰撞或真实硬件验收；本轮未启动训练或 GPU 任务。

## 2026-10-12：Bach Prelude BWV 847 精确物理松键修复

- **问题**：首次提交 Job 2257 在 GPU 训练前的 exact 任务构建阶段被拒绝，键 36 的相邻重击没有 200Hz 物理松键采样点；完整扫描同时发现键 42 的同类相邻事件。
- **修改**：备份原 proto、CSV 和报告到 `deprecated/score_repairs/prelude_bwv847_20261012/`；活动 proto 中 5 个前一音符结束时间各提前 `0.005s`，保留起音、音高、轨道、指法和力度，加入一个物理采样周期的松键间隔。同步重写 `fingering.csv` 和 `fingering_report.json`，记录原/新 SHA 与调整音符。
- **验证**：活动 `exact_score.physical_coverage` 在 20Hz 控制、200Hz 物理下覆盖 1090/1090 音符通过，最少每音符 18 个物理目标样本；原候选 SHA `bc8a027f...15384b6`，修复后 proto SHA `be5b424d...24cf6c`。尚未重新提交训练。


## 2026-10-12：非精确物理时序谱面的训练入口兼容

- **问题**：带连续重复音符的谱面在 200 Hz 物理采样上可能没有释放采样点；显式 `--no-exact-note-timing` 时，统一入口仍保留 exact F1 选择和 stopping gate，导致训练在配置校验阶段拒绝。
- **修改**：`all_use_this_train.sh` 在检测到 `--no-exact-note-timing` 时自动选择 `music_only_f1_min`，并移除 exact micro-F1 stopping gate；默认精确时序路径保持不变。同步远端训练入口后提交 Bach Prelude Job 2262。
- **验证**：本地和远端入口 `bash -n` 通过；Job 2262 已通过启动阶段进入 RUNNING。Prelude 原始 proto SHA-256 保持 `bc8a027f4666af72fc09d2027b42e27c346ecb55c7ae811f43a8de91c15384b6`。
- **未解决 TODO**：该训练关闭精确物理时序诊断，训练目标仍使用原 proto 的控制网格；200 Hz exact-note coverage 仍因相邻重复键的释放采样缺失而不适用。

## 2026-10-12：Job 2256 best.pt 视频与原谱对比交付

- **用户要求**：导出 exper001 续训结果，最终明确使用该轮 `best.pt`。
- **执行**：通过当前 `export_video.sh`／`scripts/hpc/submit.sh` 提交单卡 Job 2264；匹配当前 v5 源码归档、173 文件来源指纹和完整 config。完整物理录制成功，但附加指法恢复因 `Archived fingering targets differ from recorded goals` 校验失败。Job 2265 复用完全相同的物理录制，关闭附加指法标注，完成正面有声视频；未重新运行策略、未修改检查点或训练算法。
- **产出**：`results/exper001/exper001-20261012-181407-524560/video_best_20261012_complete/`，含视频、HTML、原始 MIDI／内嵌 proto、同次 200 Hz 触键／物理轨迹、实际／目标事件 CSV、数组、PNG 对比和来源报告。新增 CPU 脚本 `scripts/plot_recorded_midi_comparison.py`，参考 `preview_audio.events_from_states`、录制 report 和训练内嵌谱面；按真实按键阈值重建实际状态，原 MIDI 171 个音符的音高／起止／力度／轨道与谱面逐项核对一致。
- **验证**：视频 1280×720、20fps、H.264＋AAC、50.75秒，完整解码通过、音轨非静音；完整物理轨迹 49.7秒、9940采样，准备段原谱偏移1秒，未裁剪／变速。回传视频及原始导出清单7文件散列一致，冻结PT与本轮best.pt一致；图表人工查看通过。本视频单回放相对原谱的200Hz micro-F1=0.924457（TP24371/FP1176/FN2807），不同于控制帧平均F1=0.966587。
- **未解决 TODO**：通用附加指法标注恢复入口的精确时序目标与当前公共评分目标兼容问题仍保留，未通过放松校验绕过；本次交付关闭该层，实际触键高亮、声音与原谱对比完整。


## 2026-10-12：Nocturne 带双 M6 新训练启动检查

- 用户要求 Nocturne 加机械臂从头训练。经远端统一 `all_use_this_train.sh` 提交 Job 2266，采用已确认双臂初始角、原 171 音符指法谱面、默认 19,532 向量步／10,000,384 transitions；未加载无臂 checkpoint。
- 双 GPU 和初始角检查通过，实际 config 为 31 动作／1461 观察，双 M6 刚性装配和独立机械臂运动罚已启用，完整奖励已打印。
- 启动核对发现指定归档 `source_arm_ccs696_coupling_v5.tar.gz`（SHA `5913682940f0e6a4a1dfacf3b9181f9910a625596dbd008e9ad49cb1070f82ed`）仍输出 `arm_follow.mode=rigid_wrist_command_v1`、`schema=3` 和同手 `substep_sum_v1`，与当前 AGENTS.md 要求的无旧合同活动实现不符；主动取消仅本次 Job 2266，Slurm 确认 CANCELLED。没有更改固定归档、其他训练或工作区源码。
- 实际参考：AGENTS.md、统一入口、`configs/preview/current_arm_training_20261011.json` 中的初始角、`docs/optional_arm_wrist_ik_requirements_20261008.md`、远端本轮 config／启动日志。审计副本见 `results/nocturne_arms/job_2266/`。
- 未解决：需明确按指定固定归档例外运行，还是先验收当前源码并生成新的匹配归档；未提交替代作业，也未宣称新训练正在运行。


## 2026-10-12：独立部署小星星 exper002／exper003 与自动回传

- 用户授权远端新目录仅部署本地当前源码、复用已装依赖环境，不以远端旧代码比较本地。新目录 `/home/houj/projects/pianist_local_twinkle_20261012_234400`，源码归档SHA `76fbd6fe9118c8e6710ffc07aaa5b85dec449906aed9476eb53866a79a6d9088`，946文件归档审计及传输SHA通过。
- 通过新目录统一入口提交Job2267无臂、Job2268双M6小星星，各19532向量步／10000384样本，从头训练，带臂初角取当前README已确认数值。部署入口仅关闭不兼容凸包的胶囊净空诊断和相应停止条件，保留当前物理模型、奖励与限幅；不修改共享活动训练源码。实际配置与完整奖励以作业config／启动日志为准。
- CPU验证61通过、4跳过、102子测试通过，4失败；臂核心单独10通过；凸包2通过、44子测试通过，2失败。失败原因及作用范围详见 deployments/twinkle_pair_20261012_234400/README.md，未称全部测试通过。后续GPU运行状态由独立作业记录。
- 用户要求自动回传并按experXXX顺序命名：更新AGENTS.md；本地预留results/exper002、exper003及submission.json。创建ACTIVE自动化exper002-exper003，每5分钟检查终态、同步完整结果和日志并校验散列，成功／失败时通知，两项同步完成后暂停。
- TODO：胶囊净空诊断不适用于凸包，未实现凸包诊断；遗留测试假设需另行维护。本次未修改运行中的Prelude作业。

## 2026-10-12：沿用 exper001 后台脚本自动回传

- 用户明确要求沿用exper001的回传方式，不使用Codex定时助手任务。参考exper001/transfer.log中的watcher_started／Slurm查询／transfer_complete记录、exper002／exper003的submission.json和现有自动化配置。
- 新增 scripts/watch_training_transfer.py：标准库后台进程、60秒Slurm查询、文件锁、终态后rsync全结果与日志／来源记录、流式SHA256校验、失败诊断回传和传输重试；不修改训练、不加载GPU或调用模型、不使用rsync --delete。结果目录身份不符时拒绝覆盖。
- 两个独立进程已启动，PID47183（Job2267，exper002，首次RUNNING）和PID47184（Job2268，exper003，首次PENDING）；transfer.log和sync_status已记录真实查询。Codex自动化exper002-exper003经工具确认PAUSED。
- CPU临时目录验证：完成结果校验、不同作业身份拒绝覆盖、失败作业标记training_success=false均通过；检查两个真实后台进程存活。更新AGENTS.md默认规则、根README和该部署README；实验完成前不宣称已回传。
- 运行条件：本地电脑／后台进程及SSH可用；关闭Codex不影响进程，电脑重启后需依据submission.json恢复。当前任务仅替换回传方式，未更改训练源码／奖励／Slurm作业。


## 2026-10-13：修复控制帧指法标注并复用 exper001 best 回放

- **问题与参考**：`scripts/export_score_annotations.py` 错把 exact_note_events 的周期末目标作为视频标注，再与录制的 goal_applied 核对。当前 `mjlab_env/task.py` 和 `task_bundle.py` 的公共目标使用动作前 Shadow 控制帧，exact 数据仅供观察／诊断，因此旧导出 Job 2264 在起止边界被严格核对拒绝。
- **修改**：标注恢复直接读取匹配归档任务的 goal/fingers，仍校验任务散列、全部录制目标和时间轴；元数据注明 shadow_action_control_frame。同步修正活动 `preview/scripts/render_mjlab_checkpoint.py` 的指法来源，更新 `tests/test_video_export.py`，覆盖 exact 事件与控制网格不同、错误目标仍拒绝的情况。README 补充导出说明。未修改训练奖励、物理或固定源码归档；活动 renderer 的改动不会自动进入旧归档，本次使用已修复的外部恢复工具。
- **验证与交付**：54项视频导出测试通过。Slurm Job 2325 通过当前 export_video.sh 提交，使用 Job 2256 的 best.pt 和 Job 2264 的同一物理轨迹，994个指法控制帧全部核对通过；没有重跑策略。导出目录 `results/exper001/exper001-20261012-181407-524560/video_best_fingering_20261013/`，含带指尖编号／目标指法／实际触键高亮的有声MP4及同轨迹原始MIDI对比PNG、CSV、NPZ和报告。1280×720、20fps、50.75秒，完整解码及画面人工检查通过，AAC音频均值−27.5dB、峰值−11.3dB；PT与轨迹SHA、裁剪范围、倍速与上次交付一致。原谱200Hz单回放micro-F1仍为0.924457，物理时长49.7秒，准备段1秒。
- **训练状态检查**：volc-a32训练队列为空。exper002／2267正常完成19532向量步、10000384 transitions，21个回传文件重新校验一致；exper003／2268以1:0失败，最后日志120向量步，错误为右臂Arm hard-limit verification failed，18个已有结果／诊断文件重新校验一致，不能把TRANSFER_COMPLETE解释为训练成功。Prelude 2261和2263均正常完成完整预算。此次只检查训练，没有重提或改变机械臂硬限。
- **未解决TODO**：exper003右臂硬限失败的更细原因仍需单独诊断；本次指法标注故障已由真实录制核对与成功导出验证修复。


## 2026-10-13：规定训练自动回传必须包含视频与原谱对比

- **用户要求／问题**：此前自动回传仅包含训练文件，exper001 视频由后续单独导出；用户要求每次回传都包含同类视频。
- **修改与参考**：根据现有 AGENTS.md 的视频验收／自动回传规则、README 和 exper001 实际交付，在根 AGENTS.md 增补每轮（含续训）默认 best.pt、有声／指法标注／实际触键高亮、同次原始MIDI对比PNG与数据、Slurm后续导出、分阶段状态、完整验收标记及失败训练处理要求；同步README。
- **验证**：检查新要求与原有GPU限制、源码匹配、同轨迹绘图和后台脚本规则一致。仅修改文档，未改训练算法、归档或回传脚本，未提交GPU任务。
- **未解决TODO**：现有 watch_training_transfer.py 的自动导出衔接仍需实现并实测；历史exper002／exper003／Prelude视频尚未因本次文档修改而补齐。


## 2026-10-13：补导出训练2267与2263的best视频并回传PC

- **用户要求**：为已成功训练的exper002／2267和Prelude／2263导出视频并回传本地。两轮均使用各自best.pt、完整config和匹配源码归档；导出Slurm分别为2327与2326，均完成。未修改训练或重提训练。
- **产出**：`results/exper002/video_best_20261013/` 和 `results/prelude_bwv847/job_2263/video_best_20261013/`，包含带指法／指尖编号的有声MP4、同次200Hz物理触键与原始MIDI对比PNG、CSV／NPZ、来源报告、冻结PT/config及SHA清单。183与1606个控制帧指法核对通过。视频分别10.2秒和81.35秒（物理9.15／80.3秒，准备段1秒，原速完整曲目，包含视频末帧停留）。
- **原谱验证／实际参考**：小星星源MIDI的SHA与2267 source_runtime清单一致，注册library／midi_file／environment源码匹配；保留34音原时序，按训练trim_silence去除14.692708333秒曲首静音。Prelude用mido全局tempo解析原始MIDI，验证其1090音与conductor-normalized文件一致；与训练内嵌proto的差异严格限于已记录的5处结束提前5ms，对比仍使用原始结束时间。参考脚本／证据在 `results/video_exports_2267_2263_20261013/`。
- **工具修改**：`scripts/plot_recorded_midi_comparison.py`增加已验证原谱事件／configSHA来源输入和检查点标题参数，移除固定Job2256标题，保留内嵌proto核对入口。README同步说明。
- **验收**：源码／PT／config及回传产物SHA核对通过，1280×720、20fps、H.264＋AAC，完整解码及画面／对比图检查通过；两段音轨非静音。单次完整物理回放相对原谱200Hz micro-F1：2267=0.165985（TP1198/FP10622/FN1417），2263=0.686782（TP19928/FP6282/FN11895）。这是本视频指标，非训练多回合选优指标；2267存在较多误触／长时间不松键。
- **范围／TODO**：仅补齐这两轮视频交付，未改失败的exper003，也未在本任务实现未来训练的自动导出调度。


## 2026-10-13：双臂初始安装姿态静态导出（侧对话）

- **用户要求**：展示当前双臂初始安装姿态，使用类似MJLab预览的完整场景。
- **实现与参考**：仅新增 `preview/outputs/arm_initial_20261013_side/` 内的独立脚本、当前本地源码快照和产物；使用 `configs/preview/current_arm_training_20261011.json` 中初始角、当前连接件合同，经 `preview_contract.build_bundle_from_config`／`SceneBundle.reset_cpu` 构建静态t=0状态。旧预览配置指纹落后于工作区，故生成单独的新预览配置，记录当前快照指纹及依赖；不修改活动配置、不作为2268回放、不推进仿真。
- **执行／验证**：经submit.sh单卡Slurm，2328／2329因临时脚本误用旧双臂命名／manifest字段而停止，改用arm_follow_spec与双侧法兰实体检查后2330完成、退出0:0。正面、斜视、俯视1600×1000 PNG均回传并人工查看，双臂、刚性连接件、双手及钢琴可见。保存初始关节角、固定基座、完整配置、状态NPZ与文件SHA。
- **输出与范围**：`preview/outputs/arm_initial_20261013_side/rendered_v3/`。这是静态安装图，无策略／音符演奏，故不生成演奏视频MIDI对比。没有修改训练、硬限或主对话的视频导出；安装参数仍为工程配置，不代表硬件标定。


## 2026-10-13：convex_hull_v1 碰撞模型对照图（侧对话）

- **用户要求**：画出凸包碰撞模型。新增独立产物 `preview/outputs/convex_hull_collision_20261013_side/`，复用前次静态预览的当前本地源码快照，显式选用 convex_hull_v1。参考 collision_proxies.py、collision_proxies_convex_hull.json 和 preview_geometry.py，保持同一初始姿态和相机，对照可见外观与group3实际编译碰撞凸包。
- **执行／验证**：所有渲染经submit.sh；2331因临时脚本线框枚举使用错误停止，改为mjtRndFlag.mjRND_WIREFRAME后2332完成。导出左右手外观／彩色凸包／线框及双手对比PNG，检查全部选中碰撞geom为mesh、完成画面查看并回传。保存完整静态配置、碰撞geom名单和SHA清单。
- **范围**：颜色仅为渲染显示，未修改活动模型／物理／训练配置或其他任务。凸包会填平凹陷，只代表碰撞外包近似；静态图不是接触检测或硬件安全验收。
## 2026-10-13：右手 wrist 轨迹的钢琴系平移补偿

- **问题与参考**：用户提供 `/home/hj/下载/trajectory_bundle.zip`，报告右手法兰在钢琴系相对目标偏移 `[-2,-14,+14] mm`。仅分析附件原始 CSV、NPZ、metadata 与 SHA 清单，不加载其历史训练源码。
- **修改**：新增 `scripts/compensate_right_wrist_20261013.py`，对全部1831帧右手 wrist 位姿左乘钢琴系平移 `[+2,+14,-14] mm`。输出 `results/right_wrist_compensation_20261013/` 和同名ZIP，含独立右手CSV、双手CSV、补偿NPZ、中文README、来源与SHA清单；附件原始文件逐字节保留在 original/。原始23维仿真关节记录不伪装成补偿后轨迹，输出NPZ仅另保留双侧20个手指关节数据。
- **安装关系**：保持固定安装矩阵 M 不变，`T_piano_flange = T_piano_wrist @ M`，因此 `(D @ T_piano_wrist) @ M = D @ T_piano_flange`；固定钢琴系平移无需再次叠加安装偏移，也没有把wrist原点当成法兰原点。附件未提供数值安装矩阵，本次不猜填法兰位姿。
- **验证**：输入成员SHA全部一致；200Hz、1831帧、0–9.15秒的右手XYZ逐帧补偿通过，左手位姿、右手朝向、时间及手指数据不变；NPZ往返一致，输出ZIP CRC与全部manifest散列通过。首帧右wrist约 `[140,150,125]` → `[142,164,111] mm`。
- **范围／TODO**：假定测得偏移为实际减目标、且在钢琴系中为全轨迹固定平移；单点测量不能证明该假定适用于全部姿态。下游仍需使用真实安装矩阵转换法兰目标并进行机械臂IK／路径验证。本任务未下发硬件命令、未修改训练源码／配置／归档，未运行策略或GPU渲染。

## 2026-10-13：补偿后右手轨迹 t=0 静态IK

- 用户追问补偿后轨迹首帧对应右臂角度。使用当前 `configs/preview/hand_connector_20260930.json`、右M6 URDF、此前 `results/right_flange_ik_20261013/report.json` 的基座—钢琴变换及补偿NPZ，保留完整wrist姿态，通过固定安装反算Link7目标。
- 产物 `results/right_wrist_ik_20261013/` 包含CPU求解脚本、日志、README和t0_report.json及来源SHA。多初值最小二乘＋约束局部优化，选择找到的距用户最近关节种子较近可行解，不声称唯一／全局最优。J1–J7为 `[-20.2185,-50.1405,-20.1171,-92.2856,21.6658,49.0973,-54.6847]` 度。
- 舍入后正解wrist误差0.000556mm以内、朝向0.000108度以内；URDF位置约束与J6/J7耦合105.009433≤110.5度通过。首帧wrist约[142,164,111]mm，不是此前[200,0,100]mm指定法兰目标。
- TODO：全轨迹IK尚未导出；当前至首帧的运动、碰撞和硬件精度未验证。沿用定位源仍CALIBRATION_REQUIRED，工程安装不宣称硬件标定。未改变训练、归档或发送硬件命令。

## 2026-10-13：补偿后右臂 t=0 IK 姿态仿真图

- **请求与参考**：用户指定右臂 `[-20.2185,-50.1405,-20.1171,-92.2856,21.6658,49.0973,-54.6847]` 度，要求在仿真环境中导出。参考 t0_report.json、补偿轨迹NPZ、当前 source_mount_aligned 归档及右手固定安装配置。
- **实现**：`preview/outputs/right_ik_t0_20261013/test_render.py` 使用当前MJLab SceneBundle／SceneArmFollower建立刚性机械臂—法兰—连接件—手场景，以保存的实测T_base_piano覆盖训练自动反算基座，再按指定角度正解；使用轨迹首帧手指姿态，保留既有约0.15rad手部倾斜。隐藏左侧几何，未改变右手装配或训练配置。
- **执行／验证**：经统一submit.sh单GPU Slurm 2344完成，退出0:0，静态导出检查通过。远端缓存重建包全部953文件SHA与本地当前归档一致。MuJoCo中wrist位置对目标误差0.00055505mm、朝向0.00010703度；四张1600×1000 PNG（斜视、正面、俯视、侧面）已查看并回传，7个渲染数据文件SHA与远端一致，交付ZIP CRC通过。
- **交付／限制**：同目录及同名ZIP包含图、配置、状态NPZ、报告、输入、脚本和SHA清单。没有物理步进／策略回放／硬件命令，静态图片不适用演奏视频MIDI对比。仍沿用未验收定位和工程安装，不构成路径规划、碰撞或实机精度验收；全轨迹IK仍未在本次导出。


## 2026-10-13：左手轨迹平移补偿、首帧逆解与装配视图

- **请求／来源**：附件`/home/hj/下载/trajectory_bundle.zip`，用户报告钢琴系左法兰实际相对期望偏移[-10,-9.6,+26.5]mm。使用附件记录数据、`results/left_flange_ik_20261013/report.json`中保存的左基座—钢琴及SDK法兰变换、当前`hand_connector_20260930.json`和左M6 URDF；未执行附件导出脚本或历史训练源码。
- **修改**：新增`scripts/compensate_left_wrist_20261013.py`、`results/left_wrist_compensation_20261013/`及同名ZIP。全部1831帧左wrist在钢琴系平移[+10,+9.6,-26.5]mm，保持原时间、朝向、右手及20手指轨迹。原始文件完整保留在original/，23维原TX/TY/Pitch不伪装成补偿后关节轨迹。按当前约29.568857mm刚性安装偏移导出逐帧SDK法兰位姿，保留原轨迹已包含的初始yaw=0.15rad，不重复旋转。
- **IK**：以用户此前左臂七角为种子，多初值约束优化求t=0。J1–J7=[37.6956,-51.3696,-2.7995,-92.0115,1.2034,54.0752,41.4737]度。wrist=[150.000001,-140.4,98.499971]mm；SDK法兰=[173.689187,-144.401391,81.262550]mm。URDF限位通过，J6/J7耦合96.90078≤110.5度。完整精度/四位小数角及源SHA在t0_report.json。
- **仿真／验收**：`preview/outputs/left_ik_t0_20261013/test_render.py`使用当前归档构建静态M6—法兰—连接件—A3.2—钢琴场景，以保存的实测基座变换及本次七角正解，不重新反算基座。远端缓存953文件SHA全部与本地当前归档一致。经submit.sh单GPU Slurm2347完成，退出0:0。MuJoCo wrist位置误差0.000223073mm、角误差0.000034749度；四幅1600×1000 PNG均已查看，7个远端文件SHA回传核对通过。输入清单、逐帧补偿、未修改字段、NPZ往返、ZIP CRC及38文件SHA通过。
- **交付／TODO**：ZIP包含wrist/法兰CSV与NPZ、首帧角、四视图、状态、配置、说明及原始数据。整条笛卡尔目标已补偿，仅求解首帧机械臂角度；全程IK／路径／碰撞和实机精度尚未验证。定位仍CALIBRATION_REQUIRED；假定测得偏差为全程固定钢琴系平移。没有训练、策略回放、物理步进、硬件命令或训练归档变更；静态图无演奏视频MIDI对比要求。

## 2026-10-13：补偿左wrist的全程左臂轨迹，按训练硬限保留原时序

- **用户要求／选择**：补齐左臂全程J1–J7，随后明确要求按当前奖励相关硬限制调整速度／加速度，并选择保留9.15秒原时序、允许跟踪滞后。输入为已完成[+10,+9.6,-26.5]mm补偿的1831帧左wrist；保留0.15rad既有姿态和当前刚性安装。使用此前t=0已验证逆解起点，不用后来实机读数替换起点，没有生成实机至起点的过渡运动。
- **参考／实现**：新增`results/left_arm_joint_trajectory_20261013/`离线脚本和`controller_limited/`最终产物。先用URDF完整6D等式、位置与CCS696耦合约束求1831帧连续几何IK作为对照；原时序有787个采样点超速度／加速度，故不作为下发文件。最终直接用当前`arm_follow.py:ArmFollower`在CPU执行`rigid_wrist_command`连续IK／目标制动，以保存T_base_piano固定基座；输入是补偿后的记录wrist位姿新请求，不冒充原策略目标／训练回放。对应8个源码、配置和URDF文件SHA与当前source_mount_aligned归档一致。
- **硬限与结果**：J1–J7硬速度180度/秒，J1–J2硬加速度450度/秒²、J3–J7为900度/秒²，位置用当前URDF，耦合≤110.5度；奖励95%软阈值保持原值。本次不改训练、奖励公式／参数或归档。最终CSV重新差分验证1831帧全部通过，最大速度[98.1247,59.7192,152.2942,63,180,78.8717,180]度/秒；耦合最大110.499836度。按当前URDF弧度表示的J6数值上限为60.0001403度，未把它报告成精确60度。
- **跟踪与终点**：腕位置最大误差36.132718mm（5.265秒）、平均5.868059mm、末帧0.277281mm；朝向最大14.165621度、末帧0.071302度。保留有限迭代IK及制动误差，法兰／连接件／手始终按固定刚性变换。主CSV保留0–9.15秒；末帧仍有速度，另附同控制器保持末请求的9.15–9.175秒可选收停段（共享首行），末速度／加速度均0、各硬限复核通过，不改主轨迹时序。
- **交付／验证**：`results/left_arm_joint_trajectory_20261013_hard_limited.zip`含度／弧度主CSV、NPZ、逐帧v/a／误差诊断、实际wrist／SDK法兰正解、参考轨迹、两幅已查看PNG、来源／代码及README。主CSV往返与差分、原时间戳、末段连续差分、ZIP CRC及34文件SHA通过。最终主文件为`controller_limited/left_arm_joints_200Hz_deg.csv`；reference_only中的精确IK明确不用于下发。
- **TODO／范围**：仅离线控制器离散关节目标硬限验证，不保证任意插值或电机实际动力学；未再次施加上游20Hz腕目标投影，不声称保证物理TCP限制。当前标定仍CALIBRATION_REQUIRED、实机坐标差异尚未闭环，未验碰撞或规划起始运动。无GPU、训练、策略回放、物理步进和硬件下发；未生成演奏视频。

## 2026-10-13：左臂硬限轨迹补齐弧度速度／加速度列

- 用户要求将左臂角度转弧度并补齐速度、加速度。以`left_arm_joint_trajectory_20261013/controller_limited/left_arm_trajectory_200Hz.npz`中已限幅q/v/a为源，新增`radians_with_derivatives/export.py`及22列CSV：time_s、J1–J7 rad、J1–J7 rad/s、J1–J7 rad/s²。主文件保持1831帧／200Hz／9.15秒，没有重算IK或改变路径。
- 同时转换此前可选25ms收停段，并导出删除共享首行后的1836帧／9.175秒合并版，最终v/a为0。交付`results/left_arm_joint_trajectory_20261013/left_arm_qva_rad.zip`，含CSV、说明、报告、脚本与源SHA。
- 验证：CSV双精度往返一致、角度/时间不变、逐帧一阶/二阶差分对应控制器v/a、速度π和加速度2.5π／5π rad单位硬限全部通过；ZIP CRC及成员SHA通过。初帧v/a沿用静止初态0。原跟踪误差和未验实机/碰撞/起始过渡限制不变；未修改训练、控制器、奖励或归档。
## 2026-10-13：按已确认起始角导出右手＋右臂完整轨迹

- **输入／范围**：用户要求沿用已展示的t=0七关节角导出右手＋臂轨迹。使用补偿后1831帧200Hz wrist完整位姿及20个手指关节、当前右M6 URDF／安装矩阵、先前T_base_piano和本地SDK限值快照；未改训练或归档。
- **实现**：`results/right_hand_arm_trajectory_20261013/export.py` 连续受约束IK，严格固定首帧 `[-20.2185,-50.1405,-20.1171,-92.2856,21.6658,49.0973,-54.6847]` 度。仅最近解在1.62秒触及J6/J7可行域，最终采用相邻角差平方＋0.01×(q6²+q7²)的冗余姿态偏好，以及精确6维位姿等式／四条线性耦合约束，提前调整臂姿；初始失败诊断另存attempt_diagnostics。
- **结果**：全部1831帧均有解，位置限位通过、最大耦合105.0094325度；最大wrist位置误差0.00055505mm、朝向0.00010703度，最大相邻关节变化1.765282度。手指和wrist目标逐数组与输入核对相同。导出臂角CSV（度）、20手指CSV（弧度）、合并CSV、完整NPZ及逐帧审计，不把仿真手指角当厂商执行器映射。
- **时间约束**：原9.15秒时序存在4个速度超限区间和538个加速度超限样本，峰值速度353.0563度/秒；明确标为不能直接按原速执行。`finalize.py`另附全手臂同步7.5倍慢速参考（68.625秒、26.666667Hz），通过离散速度／加速度及首末静止边界估算；不改变原时序数据、不宣称插值器或实机动态安全。
- **交付／验证**：同目录及同名ZIP含全部CSV/NPZ、曲线PNG、报告、README、复现脚本与SHA。CSV往返、原始手指精确保留、ZIP CRC和成员散列通过，曲线已查看。全部为CPU离线运动学，无GPU、训练、策略／物理回放或硬件命令。
- **TODO**：未做全路径碰撞、控制器插值／负载验收或其他姿态至起点的过渡。定位仍CALIBRATION_REQUIRED，安装仅工程配置；慢速参考不是硬件可直接执行保证。


## 2026-10-13：右手包导出22列弧度轨迹，沿用训练制动／硬限

- **请求与输入**：用户指定`results/right_hand_arm_trajectory_20261013.zip`，要求按左臂22列q/v/a格式导出右臂，并明确选择保留9.15秒、按训练硬限限幅、允许跟踪误差。输入ZIP CRC和全部成员SHA通过；只读取数据，未执行包内脚本。原腕补偿[+2,+14,-14]mm、完整朝向／安装／基座及原20手指角度保留。
- **实现与控制范围**：新增`results/right_arm_qva_20261013/export_recorded_reference.py`、`audit_recorded_reference.py`及最终`hard_limited/`。使用原包已验证连续关节IK作为q_ref，离线覆盖ArmFollower.solve参考来源，原训练step／bounds／_stopping_speed／_coupled_segment不变；保留200Hz时间戳、SDK硬速度π和加速度2.5π（J1–J2）／5π（J3–J7）rad单位限制、URDF位置与CCS696耦合。参考来源区别写入合同，不冒充完整训练在线IK回放。当前8个相关源码／配置／URDF与活动归档文件SHA一致。无训练源码／奖励／归档修改。
- **诊断与选择**：初次直接重跑在线IK在约1.2秒耦合边界处停止跟踪，产物只保留在同目录controller_limited/作诊断，不纳入交付ZIP；最终使用用户包已有的连续冗余解，避免重新选解丢失其避开边界的姿态分配。source_conversion/只作原包单位转换参考，未作为最终限幅轨迹。
- **结果**：主文件`hard_limited/right_arm_qva_200Hz_rad.csv`，1831帧／9.15秒，时间＋七角rad＋七速度rad/s＋七加速度rad/s²。另有42列“22列臂q/v/a＋原20手指位置”合并文件；手指CSV逐字节保留。最大角速度[44.6731,27.9783,70.9661,30.9144,85.9181,76.2625,69.1342]度/秒，最大加速度分别达到对应硬限；耦合最大105.009995度。全程CSV往返、从q重新差分、速度／加速度／位置／耦合全部通过。
- **误差／终点**：腕最大位置误差5.032020mm、平均1.040232mm、末帧0.041221mm；最大姿态误差2.295472度、末帧0.048298度；Z误差[-2.269852,+2.897350]mm。另附9.15–9.17秒20ms收停段，5行含共享首行；去重合并版1835帧，末v/a均0，硬限通过，主0–9.15秒不变。
- **交付／范围**：`results/right_arm_qva_20261013_hard_limited.zip`含主CSV、收停/合并CSV、手指、NPZ、逐帧审计、两幅已查看曲线、来源／代码／说明及29文件SHA；ZIP CRC与散列通过。未重做碰撞／实机／负载、初始过渡或实际按键验收；定位仍CALIBRATION_REQUIRED。仅CPU离线目标计算，无策略、物理步进、GPU或硬件下发，不保证任意插值器或实际电机响应。

## 2026-10-13：完整左右手关节与wrist同步导出

- **请求／来源**：用户要求完整左右手关节和wrist轨迹。读取左`left_arm_joint_trajectory_20261013/controller_limited/left_arm_trajectory_200Hz.npz`、右`right_arm_qva_20261013/hard_limited/right_arm_qva_200Hz.npz`和原补偿手指记录，来源manifest与SHA一致。保留左侧既有在线IK＋限幅和右侧既有原包IK参考＋限幅结果，没有暗中改用同一求解方法。
- **产出**：`results/both_hands_joints_wrist_20261013/`及同名ZIP，统一1831帧／200Hz／0–9.15秒。每侧7臂＋20手指共27关节，双侧54关节；导出54角55列、完整q/v/a163列、双wrist请求／执行各15列、177列总表和完整NPZ，以及分侧22列臂文件、手指／全关节文件与关节名称映射。wrist以钢琴系XYZ米＋wxyz四元数表示，执行位姿来自机械臂正解，不是法兰或硬件测量。
- **导数口径**：机械臂q/v/a逐值保留；手指主表导数从原角度后向差分，首帧速度／前两帧加速度因缺历史标NaN；原记录物理qvel另保存，不把两者混用或补成已测得的0。不新增手指限幅。左右原独立收停段仅单独附带，不改变9.15秒主表或伪造全手同步收停。
- **验收**：时间戳、两侧臂q/v/a、20手指角度与来源逐项一致；当前双URDF＋保存基座／安装独立FK全帧核对executed wrist，通过2e-12矩阵容差；四元数连续符号／矩阵回转、CSV数值往返（含NaN）、机械臂差分及硬限通过。ZIP CRC与33文件SHA通过。来源、控制合同、URDF、安装和说明可复核。
- **范围**：没有重新IK、滤波、插值、训练／奖励／归档修改、GPU渲染、策略或物理回放与硬件下发。原左最大位置误差36.132718mm、右5.032020mm仍保留；未验证双臂联合碰撞／实机精度／起始过渡，定位仍CALIBRATION_REQUIRED。
## 2026-10-13：新capture a08de8bc的右臂法兰单点IK

- 用户提供新钢琴基座位置，要求法兰钢琴系[200,0,100]mm。核对本地latest.json capture_id为a08de8bc-c5a9-4280-b97b-257ff33eb49d，冻结快照至 `results/right_flange_ik_a08de8bc_20261013/`；使用1010右臂手眼标定、此次完整旋转和中心，以及此前已采用的钢琴轴映射与Link7→SDK法兰固定变换。
- 沿用用户图示法兰朝向（+X=钢琴+Y，+Y=−Z，+Z=−X），不叠加上一轮轨迹补偿或wrist抬起角度。本条无新图片，已询问当前七角，同时以此前截图[-63.746,-83.933,34.844,-75.064,111.938,-44.655,2.640]度作为参考，未冒称当前实测。
- 新局部可行解J1–J7：[-64.2773,-77.7589,34.5895,-89.7133,110.2794,-52.5523,-12.2313]度。位置限位通过、耦合66.0974463度，四位小数FK误差0.000239023mm／0.000089078度以内。目录含prepare.py、calculate.py、输入、report及README和来源SHA。
- TODO：新当前角未收到，因此不是按本次实测姿态重新选出的最近解；定位仍CALIBRATION_REQUIRED，未做路径／碰撞或实机精度验收。纯CPU离线计算，未修改训练／归档或下发硬件。


## 2026-10-13：首次提交 GitHub 项目仓库

- **请求与参考**：用户要求使用 GitHub CLI 将当前项目提交至 `AGIMani/TsenseIianist`；核对 `AGENTS.md`、`README.md`、`all_use_this_train.sh` 和远端空仓库状态。
- **修改**：新增 `.gitignore` 排除本地环境、缓存、实验输出、部署副本与重复项目 ZIP；保留统一入口依赖的固定源码归档及其 SHA/manifest。新增 `.gitattributes`，模型网格、CAD 与固定归档使用 Git LFS；README 补充 CLI 克隆、LFS 获取、归档校验及本地实验链接说明。已有本地历史文件不删除。
- **验证范围**：发布前核对纳入文件中的凭据模式、Git 大文件、固定归档 SHA 和 LFS 对象；发布后核对远端分支提交。仅仓库初始化与文档变更，不修改训练算法、奖励、模型或固定归档，不运行 GPU 任务。
- **限制**：实验结果与生成输出不随 Git 克隆提供；训练依赖环境须独立安装。
## 2026-10-14：ff202a68右臂当前截图选解

- 用户将左臂请求更正为右臂，并提供当前角[-71.913,-93.127,56.649,-84.769,-92.534,30.082,14.530]度。使用1010右臂手眼、ff202a68完整定位快照和用户明确轴映射；目标法兰钢琴系[200,0,100]mm，朝向沿用+X→+Y、+Y→−Z、+Z→−X。没有叠加旧轨迹补偿／wrist倾斜。
- `results/right_flange_ik_ff202a68_20261014/`保存输入、脚本、报告、日志与README。在run_cpu_analysis.sh内完成多初值约束局部IK；可行角[-90.3694,-120.0002,80.3532,-89.5005,58.9899,-7.7631,-7.4780]度。J2四位小数按URDF向界内舍入，舍入后位置误差0.000799201mm、旋转0.000114905度，独立限位和耦合约束通过。
- 重要限制：J5相对当前约变化151.524度、J2接近下限；仅为找到的静态可行解，不声称全局最近或可直接移动。未做过渡／碰撞验证、未下发硬件，定位仍CALIBRATION_REQUIRED；无训练／归档变更。

## 2026-10-14：按用户修正的TCP轴向重算右臂

- 用户明确TCP+X→钢琴−Y、+Y→钢琴+Z、+Z→钢琴−X；位置保留[200,0,100]mm。使用ff202a68冻结定位、1010右手眼、此前SDK法兰固定变换以及最新截图初角，不叠加旧轨迹补偿。TCP假定与SDK法兰一致，未猜填工具偏置。
- 在 `results/right_tcp_reoriented_ff202a68_20261014/` 新建输入／复现脚本／报告／README，保留此前不同朝向计算。通过run_cpu_analysis.sh受限CPU作用域执行多初值局部约束优化。
- 新J1–J7为[-78.9120,-92.5852,55.4183,-89.4996,-93.4988,33.6370,15.9800]度。相对当前最大变化约6.999度；舍入后FK位置0.000526mm以内、朝向0.00005045度以内，位置限位及耦合约50.45799≤110.5度通过。
- 未修改训练或模型归档、未下发硬件。定位仍未验收；没有碰撞／移动路径验证，数值FK误差不代表实机精度。

## 2026-10-14：右臂实测偏差补偿与四点测试逆解

- 用户报告上次[200,0,100]目标实际为[199.15,4.14,102]mm，要求后续四点补偿。按固定钢琴系实际减命令误差[-0.85,+4.14,+2]，补偿[+0.85,-4.14,-2]mm；保留ff202a68基座变换、1010右手眼及最新TCP轴向（+X→−Y、+Y→+Z、+Z→−X），不再叠加旧补偿。
- `results/right_four_point_test_20261014/`保存脚本、输入、报告、日志及README。四个期望Y为215.5、177.5、144.5、120.5mm，命令X=242.85、Z=98mm，命令Y=211.36、173.36、140.36、116.36mm。首点以上次修正朝向解为参考，后续沿用前一点解，多初值局部约束优化。
- 经run_cpu_analysis.sh计算，四组舍入角全部通过URDF位置限位、耦合最大40.61661度；最大FK对补偿命令位置误差0.000522mm以内、角度0.000098度以内。详细七角在report.json／README。
- TODO：固定偏差假设仅基于单点，尚需四点实测确认；没有点间时序／动态约束／碰撞验收、未下发硬件，定位未验收。未改训练或归档。

## 2026-10-14：右臂法兰实测偏差补偿与四点测试IK

- 用户报告前次[200,0,100]mm目标实测[199.15,-4.14,102]mm，要求补偿并计算四点。按实际减目标偏差[-0.85,-4.14,+2]mm，在钢琴系施加反向命令补偿[+0.85,+4.14,-2]mm；这是单点固定平移假设，不冒称全空间标定。
- `results/right_flange_four_points_20261014/`保存受限CPU作用域执行的calculate.py、report.json、日志与README。使用ff202a68冻结变换、1010手眼与SDK法兰定义，朝向保持+X→钢琴−Y、+Y→+Z、+Z→−X。第一点以前次已计算目标关节角为参考，后续顺序选解；不声称全局最优。
- 期望XYZ依次[242,215.5,100]、[242,177.5,100]、[242,144.5,100]、[242,120.5,100]mm；补偿命令XYZ依次[242.85,219.64,98]、[242.85,181.64,98]、[242.85,148.64,98]、[242.85,124.64,98]mm。全部四位小数关节解经FK验证位置误差<0.001mm、旋转误差<0.000108度，独立限位和J6/J7耦合通过，最大耦合39.452535度。
- 未修改训练／归档或下发硬件。没有规划各端点之间路径、速度或加速度，也没有碰撞验收；四点实测补偿效果仍需用户验证。定位仍CALIBRATION_REQUIRED。

## 2026-10-14：右臂法兰[163,147.5,100]补偿单点

- 用户给出[185−22,155−7.5,100]mm，化简为[163,147.5,100]mm。沿用最新确认朝向、ff202a68定位／1010手眼及固定补偿[+0.85,+4.14,−2]mm，命令点[163.85,151.64,98]mm；以上组测试末点关节角为局部选解参考。
- `results/right_flange_163_147p5_20261014/` 保存脚本、报告、日志、README。经run_cpu_analysis.sh求得J1–J7[-80.1392,-80.2026,72.2402,-93.0069,-81.3248,17.1004,13.3726]度；四位小数FK位置误差0.000524661mm、旋转0.000051264度，位置限位及耦合30.90051度通过。
- 未规划运动／检查碰撞或下发硬件，未改训练；固定误差补偿效果仍待实测，定位未验收。

## 2026-10-14：右臂[207,162.5,100]mm补偿单点

- 用户请求[185+22,155+7.5,100]mm，化简为[207,162.5,100]；沿用固定误差补偿[+0.85,+4.14,-2]，命令[207.85,166.64,98]mm及已确认TCP朝向。选解参考为四点测试最后一组角度，非新硬件读数。
- 在run_cpu_analysis.sh内求解，输出 `results/right_flange_207_162p5_20261014/` 的脚本、报告及README。J1–J7为[-88.7672,-76.6327,73.7991,-103.3352,-80.7775,17.5920,14.5669]度；舍入后FK位置误差0.000159427mm以内，限位／耦合32.5987度通过。
- 未修改训练或下发硬件；固定偏差、定位精度与移动路径／碰撞仍未验收。

## 2026-10-14：将坐标与补偿约定保存到平台T_definition目录

- 按用户指定，在 `/home/hj/TsenseIianist_platform/calibration/T_definition/坐标系与右臂补偿约定_20261014.txt` 新建文档，记录T_A_B含义、用户／定位钢琴轴映射、ff202a68双侧定位中心与右T_base_piano、最新TCP轴向及[+0.85,+4.14,-2]mm实测补偿。
- 参考当前输入／报告及安装配置，明确preview中心是钢琴原点在基座系的位置，TCP暂同SDK法兰，wrist与Link7不能当作法兰；解释刚性安装、补偿左乘及不得与旧[+2,+14,-14]mm重复相加。列出测试点命令位置、来源及未验收范围。
- 核对文档矩阵与保存的ff202a68输入一致，补偿符号与用户实测一致。仅新增说明，没有更改控制器、标定、训练或已有轨迹。

## 2026-10-14：T_definition轴向标注错误及更正

- 此前将用户提供的左右臂基座轴向误标为TCP／法兰轴向，并错误推断此前末端目标朝向被替代；这些结论已撤回。用户随后明确这些关系属于基座，第4节旧末端说明不再有效。
- 已在T_definition的Markdown与TXT两份文档中清除误标和由此产生的旧IK失效推断，改为精简的基座轴向对照。左基座+X→钢琴−X、+Y→−Z、+Z→−Y；右基座+X→钢琴−X、+Y→+Z、+Z→+Y。本次基座定义本身没有指定新的TCP朝向，也未自动重算历史IK。

## 2026-10-14：坐标约定转为Markdown文档

- 按用户要求，将T_definition目录已有说明整理为同名 `.md` 文件，当时标题误标为左右臂TCP朝向（现已按基座定义更正），包含7节、轴向／测试位置表、矩阵与公式代码块以及来源文件链接。原TXT保留。
- 核对最新左右臂映射、补偿符号及旧IK尚未重算说明保留，检查标题与代码块结构；仅格式整理，未更改坐标定义、模型或计算结果。

## 2026-10-14：精简坐标朝向文档

- 按用户要求精简T_definition目录的 `坐标系与右臂补偿约定_20261014.md`，仅保留用户钢琴／定位程序轴向表、左右臂轴向表（当时误标为末端，现已更正为基座）。移除Markdown中的补偿、矩阵、推导与测试说明；最新轴向定义不变，未更改模型或已有计算结果。


## 2026-10-14：明确基座轴向并清除错误末端结论

- 按用户纠正，同时更新T_definition目录同名.md与.txt，仅保留定位／用户钢琴轴映射、左右基座轴映射；清除此前将基座关系解释为TCP目标、替代旧法兰朝向及据此认定旧关节解需重算的错误说明。核对两文件轴向一致。未修改模型、标定数据或已有IK结果。
## 2026-10-14：右臂[202.21,157.265,99.921]mm单点

- 用户请求[180.21+22,149.765+7.5,99.921]mm；沿用ff202a68实测基座变换、此前法兰朝向及[+0.85,+4.14,-2]mm补偿，命令位置[203.06,161.405,97.921]mm。文档的基座轴向不被误用为末端目标或新的实测旋转。
- 受限CPU作用域计算结果保存于 `results/right_flange_202p21_157p265_20261014/`，含脚本、报告及README。选解参考为最近[207,162.5,100]计算角；J1–J7为[-87.6912,-77.3404,73.1773,-102.1761,-81.2330,17.8820,14.5689]度。舍入后FK位置误差0.00006733mm以内，独立限位及耦合32.89795度通过。
- 无硬件命令、路径或碰撞验收；定位与固定补偿仍需实测验证，未改变训练或标定。

## 2026-10-15：提交当前工作区后整理非训练产物

- 按用户要求先创建提交 b31470187，保存当前源码、配置、文档、测试及固定源码归档；git diff --cached --check通过，本次没有重新执行训练验收。
- 参考results目录、实验submission/config、README回放引用、训练一致性与模型验收说明以及Git/LFS目录。将机械臂轨迹、IK、腕补偿等非训练产物及无活动Git LFS进程的临时文件移出仓库；完整路径、大小和SHA256见 docs/cleanup_nontraining_20261015.json。
- 保留全部exper实验、Prelude训练视频、原始MIDI回放依赖、当前模型及训练一致性验收；保留全部Git历史、有效对象和LFS对象，没有重写历史或删除检查点。
- 可恢复目录：/home/hj/pianist_cleanup_recovery/20261015_210432。此次为移出仓库整理，同一磁盘总空间不会因此释放；未做永久删除。清理后检查保留目录、Git状态和迁移文件SHA256。

## 2026-10-15：永久删除已移出的非训练产物

- 按用户明确要求，依据 docs/cleanup_nontraining_20261015.json 逐文件核对恢复目录清单和SHA256，确认无PT/PTH/CKPT等检查点且无清单外文件后，永久删除 /home/hj/pianist_cleanup_recovery/20261015_210432。
- 删除28项非训练产物及LFS临时文件，逻辑大小535446454字节；保留仓库内训练实验、检查点、视频及回放依赖、训练验收和Git有效对象。删除前后results全部文件大小及修改时间快照一致，恢复目录已不存在；实际剩余空间见清理JSON的permanent_deletion字段。未修改训练算法或归档，无需GPU验收。
# 2026-10-17 Cartesian arm contract migration

- **问题**：活动双 M6 训练仍以旧腕部请求合同解释策略动作，不能按 ROSELIA Wuji Cartesian 的掌位移接口使用。
- **修改**：活动机械臂合同改为 `controller=wuji_cartesian_v1`；策略动作固定为 47 维 `R XYZ + 20 指 + L XYZ + 20 指 + sustain`。Pitch/Yaw 在场景编译时固定为安装姿态，XYZ 按世界坐标米制掌位移解释；连续阻尼 IK、URDF/SDK 速度加速度硬限、CCS696 J6/J7 约束和刚性法兰/四孔连接件/手装配继续保留。旧 `rigid_wrist_command` 不再被活动代码或预览配置读取。
- **范围**：新增 `configs/actions/wuji_cartesian_v1.json`，更新动作映射、场景编译、跟随器请求、控制限速、可达域扫描、训练入口及当前预览配置；Cartesian 平移范围采用 ROSELIA 的 XYZ 轴顺序 `±[0.025, 0.10, 0.025] m`，实际绝对关节可达性由 M6 IK/URDF 限位验证。
- **验证**：`python3 -m compileall` 通过；动作映射静态检查确认 active 47 维及双侧 TX/TY/TZ 索引；标准库打包审计通过，归档 `logs/wuji_cartesian_20261017/source_wuji_cartesian_20261017_v6.tar.gz` SHA256=`26fefce65c4470821f48c3f89755f7d40595124f11e917e0f7a8beaa888db53e`，并同步当前 639 个源码指纹。预览源构建与 CPU IK/刚性场景单步通过（`nq=128, nu=40`）；训练 inspect-only 合同检查通过；本机 CUDA 不可用，GPU 单步验收需在 Slurm 分配内执行。

## 2026-10-18 Prelude exper010：修订指法、yaw30°从头25M与自动交付

- 用户要求：使用 revision_20261018_v2 的61处指法修改，从头训练；保留音高、起止时间、力度、踏板及手别。活动 proto SHA256 为 ad58e67de1998c218e380b7370c95d64ba23fe80d6dfbe26f415184e1a80b33a。
- 实际检查发现默认仍15°，旧校验只容许±0.3rad；修改 hand constants、source_runtime、train_from_source 和统一入口，使默认及本轮显式 yaw 为π/6=0.5235987755982988rad，初始化容许范围扩为±π/6；本轮 yaw 仍结构性移除为固定安装角，不新增策略yaw动作。Pitch策略[0,2]/目标[0,.3]、八个非拇指MCP目标[0,1.3]及其他当前关节设置不变。
- 注册原谱 proto 含既有五音松键时间修正，与更早 timing_normalized.mid 不完全相同。finalize_training_video、watch_training_transfer 与 plot_recorded_midi_comparison 新增显式注册谱路径/SHA校验：必须与checkpoint内嵌proto逐字节一致，使用原始未量化事件，并保留proto/来源；不再冒称更早MIDI逐音一致。
- 新增注册谱精确时间/篡改拒绝测试；Slurm验收测试构建新谱、检查实际固定yaw及关节映射并执行GPU物理步。完整作业号、归档SHA、验收结果见 results/exper010/submission.json 与 acceptance 日志。预算48829×512=25,000,448 transition，从随机初始化开始；有声完整 best.pt 视频及同回放200Hz实际按键对比PNG由本地后台脚本持续导出回传。
- 不把修改指法或单步有限值验收视为各指可达性/训练收敛证明。保留历史归档和结果。

- 实测验收：本地7项离线测试通过；在systemd受限scope内复用exper009原始recording报告和200Hz数组，注册谱对比图重建F1=0.7995604473511636，与已交付同回放完全相同（不是exper010训练结果）。Slurm4218验收通过（45.44s）：实际固定yaw=π/6、活动谱SHA、Pitch/MCP端点映射及GPU一步有限值。正式双A100作业4219依赖验收后已运行；本地watcher已首次真实查询。源码归档SHA256 bde046fd42e9874a7070218e9ba718b8a2f9b8b1763e429d8490f6728b36ed94；统一入口已指向该归档。训练与最终视频尚未完成，不宣称交付成功。

- 启动配置复核通过：实际 `warm_start/stage2_from/finetune_from=null`、512环境、cuda:0采样/cuda:1学习、yaw=π/6、注册谱SHA及请求预算全部匹配；完整奖励已在Slurm4219启动日志打印，副本在 `results/exper010/startup_verification/rewards.json`。新归档已回传本地并流式SHA256校验通过。网络短暂断连由后台脚本保留日志并重试；后台仍存活，不影响Slurm训练。

## 2026-10-18 exper010训练状态核验及作业时限风险

- 参考本轮submission、Slurm4219、远端metrics.jsonl/best.json和本地watcher状态；约17:41北京时间，6740向量步=3,450,880 transition（13.80%），最近6500步三回合200Hz原谱micro-F1合并0.1302347、best按最差回合0.1169154；20Hz控制帧micro-F1为0.1279625，未混用指标。
- PID275391存活且持有锁，60秒查询仍更新；历史SSH短暂失败已自动恢复。自动导出尚待训练终态，检查导出入口、注册谱SHA、匹配归档、音频运行时及finalizer支持。只核验现有流程，未重复提交训练/导出。详见results/exper010/progress_check_20261018.json。
- 发现原24小时分配可能不足：当前约233 transition/s，剩余约25.7小时（尚不计后续评估），分配只剩约19.8小时；尝试scontrol将现有4219延长到48小时，被Slurm以Access/permission denied拒绝。作业保持RUNNING且时限未变，截止北京时间2026-10-19 13:32:48；需有Slurm管理权限的账号延长。未修改训练预算、源码、奖励、checkpoint或默认训练入口。

## 2026-10-18 Prelude中断撤回与exper013恢复（累计20M）

- 用户先要求中断并导出，4219已于17:43左右取消，停在6820向量步/3,491,840 transition；保留best.pt（6500步/3,328,000 transition）。原导出4391随后提交。用户撤回中断要求时Slurm取消已完成，无法撤销；已取消尚未启动的4391并停掉原exper010回传进程，防止继续导出被替代轮次。已有训练文件已回传并校验，原结果不删除。
- 用户随后把目标改为累计2000万步。以相同源码归档、相同指法/30度yaw/关节范围从本轮best.pt做learner warm restart，恢复网络、优化器和RNG；经验池/物理状态重建，不能称无缝续跑。回滚6500之后未保存的320向量步，预算按已恢复检查点计数：新增32563×512=16,672,256，累计20,000,256 transition。使用唯一all_use_this_train.sh的显式源码/SHA及--warm-start末尾覆盖，不迁移到其他会话更新的新源码合同。
- 后续实验按编号登记为exper013；24小时新Slurm分配重新计时，独立submission和后台watcher继续在完成后导出该轮best.pt有声完整视频、指法/编号/实际键高亮及同回放200Hz原谱对比图，再回传验收。实际作业号及状态见results/exper013/submission.json和sync_status.json。
- 同阶段对比澄清：332.8万步时exper010精确200Hz合并F1=.1302347，exper009=.0899810；尚不能认定修改指法导致下降。当前主要表现是漏键/保持不足，三回合音符匹配379/3270。两轮相近阶段目标运动罚均到-5，不能把它单独归因成本轮差异。完整统计见results/exper010/interrupted_diagnostics.json。

## 2026-10-18 Prelude 采样性能回退定位（进行中）

- 用户要求定位约7.7倍采样回退，保持当前61处指法、yaw30°、Pitch策略[0,2]/物理[0,.3]、四指MCP[0,1.3]和全部奖励不变。参考exper009/010同为6820向量步的流水计时、各自冻结scene.py/finger_coupling.py、当前MJLab Simulation.forward和exper013恢复日志；不将曲长或显卡申请数当作归因。
- 发现耦合从控制帧改为每200Hz物理子步求解后，整控制步CUDA Graph条件排除了finger_coupler；普通子步循环每次硬关节限位后仍直接调用mjwarp.forward，失去图缓存。拟将该非捕获循环改用MJLab缓存forward图；保留完整子步求解/物理/奖励、保留外层控制图内部的原生kernel调用，避免嵌套捕获。
- 新增scripts/profile_prelude_sampling.py，以同checkpoint重建配置、512环境固定动作比较原路径/缓存路径/原路径复测，检查qpos/qvel/ctrl、观察/奖励/结束标志、200Hz真实键状态和耦合诊断逐元素一致；独立分段计时，不训练或保存权重。所有GPU测试经Slurm提交。4399因外部诊断文件用了相对路径收集失败（未执行物理测试）；4402改绝对路径排队。工作区补丁尚未GPU验收、归档或用于训练，exper013仍在运行。


## 2026-10-18 Prelude采样回退修复与H100迁移

- **定位**：exper009与exper010同为3491840 transitions时，采样分别1551.67s与12022.42s，回退7.748倍；学习器更新分别1619.76s与1648.05s，基本不变。因此瓶颈在采样物理流水，而不是曲长或SAC更新。
- **原因**：手指耦合改为每200Hz物理子步重解后，整控制步CUDA Graph条件因`finger_coupler`被排除；每个子步硬关节限位后又直接Python调度`mjwarp.forward`。
- **修复**：`scene.py`仅在普通物理子步循环使用已捕获的MJLab forward graph；控制图捕获路径仍使用原生kernel，200Hz耦合、硬限位、碰撞、按键物理采样与奖励公式均未改。工作区补丁通过候选归档SHA `87599884071e93a0859a22c9f01e98e80d1964236e4a0d2c83837faebe1b5285`，相对Prelude固定包仅改该文件。
- **验收**：H100 Slurm1840通过；同一物理状态下raw/cached `xpos/xmat/geom_xpos/geom_xmat`最大差均为0。512环境32帧采样均值由0.39016s降至0.18527s（2.106倍）；原始7.7倍回退中约2.1倍来自该forward路径，余下差异受GPU型号/流水调度影响。独立新环境接触原子操作存在非确定性，保留原始逐张量报告，不将其误报为算法变化。
- **训练迁移**：按用户“使用H100来训练”停止volc-a32 exper013（4394，停在1280向量步/655360 transitions，仍在预填充、updates未增加），H100 Tsensei2提交exper015/1841，20小时限时，以exper010 best.pt warm restart，累计目标仍20,000,256 transitions。启动日志已打印完整奖励配置，输出目录已生成best.pt/config/metrics，当前约1465 transitions/s，后台watcher PID784387运行。

## 2026-10-18 三路采样＋单卡学习多采样器接入

- **用户要求**：将当前训练资源布局改为 3 张 GPU 采样、`cuda:3` 学习；不改变奖励、关节速度/加速度、硬位置限位、CCS696 耦合或刚性安装合同。
- **入口修改**：`scripts/hpc/submit.sh` 接受 `--gpus 4`，四卡批作业/预留作业申请 32 个 CPU；`--allocation` 按 `TsenseI-reserve4` 与四卡 `srun` 校验；原 1/2 卡路径保留。`scripts/hpc/job.sbatch` 与 `reserve.sbatch` 接受四卡环境并记录实际卡数。H100 `gpu` 分区已只读核对为 8×H100，可申请 4 卡。
- **统一参数**：`all_use_this_train.sh` 现在请求 `--gpus 4`，显式传入 `--collector-devices cuda:0,cuda:1,cuda:2 --learner-device cuda:3`；训练核心新增多采样器分片、每卡独立 actor/RNG、中心 replay 聚合和一个待更新块的流水屏障，保持全局 transition 与 SAC 更新预算不变。三路 collector 的环境数按 512 分成 171/171/170，按固定顺序写入 learner 卡 replay；所有 collector 使用相同场景、奖励和关节/IK硬约束合同，不静默退化为单采样器。
- **源码接口**：`rl_torch/train.py` 新增 `--collector-devices`；`collector_pipeline.py` 新增 `MultiBatchPipeline`，每个向量步并发提交三路采样，更新前只提交已完整聚合的上一批。单卡及原双卡 `OneBatchPipeline` 保持兼容。检查点仍保存主 collector 的 canonical RNG；其余 collector 由固定 seed 派生，恢复时随机序列不承诺逐位复现，但不改变采样预算或约束。
- **验证**：新增 `tests/test_hpc_gpu_layout.py`，以 fake `sbatch` 做 CPU-only 资源命令检查（四卡、二卡兼容、四卡预留命名、非法三卡拒绝、统一入口设备布局），5 项通过；`tests/test_collector_pipeline.py` 11 passed、2 CUDA tests skipped；shell syntax 与 Python 编译检查通过。当前只完成 CPU/静态验证，未提交完整训练；四卡 GPU smoke 需在新归档经 Slurm 验收后再标记生效。
- **归档**：已生成并审计 `logs/multi_collector_20261018/source_multi_collector_v3.tar.gz`，文件数1112，SHA256=`99d1d284b15cacf6c62a36a9300f854374fd9493cfad2bf9835d044b62c6038c`；统一入口已切换到该归档。GPU smoke 作业4475曾提交但因节点仅剩3张空闲GPU而取消，未执行GPU验证；未启动正式训练。
# 2026-10-18 Nocturne exper016 四卡训练排队

- **用户要求／目的**：确认四张 GPU 与三路采样／单卡学习流水后，按统一四卡入口启动约 1500 万 transition 的 Nocturne 训练。
- **实际提交**：远端 `volc-a32:/home/houj/projects/pianist_native_ik_20261018` 提交四卡验收链：`4491`（四卡 GPU 探针＋collector/layout pytest）→ `4492`（3 collector + 1 learner 的最小真实训练 smoke）→ `4493`（Nocturne exper016，`vector_steps=29297`，有效 transition=15,000,064）。训练以 `afterok:4492` 且 `--kill-on-invalid-dep` 排队，验收失败不会启动训练。
- **当前布局**：`cuda:0,cuda:1,cuda:2` 独立采样，`cuda:3` replay/SAC learner，启用 `--collector-pipeline`；默认无机械臂、当前奖励／关节硬限合同保持不变。
- **谱面与来源**：使用项目内 `nocturne_op9_no2_beginner/generated_v1/nocturne-op-9-no-2-beginner_fingered.proto`（SHA256 `1b18576a4c3829d0d0a2e24db9832ef480daecb8a0009e70f7cdca65b5ae3db`）及匹配原始 MIDI（SHA256 `5c3a7046cda06fa0c28581716e10b72723b0180cd39cf30240980c1441d989d8`）。实际提交归档为远端 `source_multi_collector_remote.tar.gz`（SHA256 `a92573ee880c0ccc1cd99858606309f545de9c3d560e16fb56b53208855405be`），本地基线归档 SHA 为 `99d1d284b15cacf6c62a36a9300f854374fd9493cfad2bf9835d044b62c6038c`。
- **自动回传**：已创建 `results/exper016/submission.json` 并启动后台 `scripts/watch_training_transfer.py`（PID 879688）；训练结束后继续执行 best.pt 导出、同次回放实际按键／原始 MIDI 对比图、SHA 校验与回传。当前 `gpu_validation_state=PENDING`，作业等待四卡资源。
## 2026-10-19 exper015训练与导出回传完成

- H100作业1841按预算完成32563个vector steps（16672256 transitions），状态为COMPLETED，训练文件244项校验通过。
- 导出作业1842完成；首次最终验收因submission残留上一项目的注册谱面绝对路径而失败，改为H100项目内同SHA的candidate_fingering谱面后重新验收通过。
- 同次物理回放的200Hz实际按键/MIDI对比、音视频解码和checkpoint SHA均通过；本地结果目录已创建TRANSFER_COMPLETE。

## 2026-10-19 exper017 首个采样步 Warp 700 修复

- **问题**：exper017 作业5346在初始评估完成后、经验池预填充的第一个物理控制步失败；堆栈落在硬关节限位后的 `mjwarp.forward()`，Warp 记录 CUDA error 700。作业没有产生任何学习更新，`best.pt` 仅为初始评估基线。
- **原因**：外层 `physics_graph` 被关闭时，`scene.py` 仍调用 eager `mjwarp.forward()`；MJLab 的 `Simulation.step()` 已使用内部 step CUDA graph，随后 eager forward 会重复分配碰撞上下文并与批量 graph/allocator 交错。3环境初始评估未达到触发规模，三路采样的171/171/170环境首步触发。
- **修改**：硬限位后的几何刷新统一调用 `Simulation.forward()`，由 MJLab 复用已捕获的 forward graph；无 graph 的配置仍由该接口回退 eager 路径。移除按 `physics_graph` 选择 unsafe eager 分支，并在 GPU 硬限位回归中覆盖外层 graph 关闭、内部 forward graph 保留的路径。
- **验证**：项目 venv 中 `tests/test_collector_pipeline.py`、`tests/test_hpc_gpu_layout.py`、`tests/test_source_pythonpath.py` 共19项通过、2项跳过；专门的Slurm GPU回归 `tests/test_prefill_cuda_regression.py` 以171环境单步通过（工作区作业5590，固定归档作业5596，均1 passed；固定归档用时47.69s），没有再出现CUDA error 700。综合硬限位回归作业5585只在既有60步极限动作后的控制目标边界断言失败，未出现CUDA错误；本次首步验收以5596为准。当前固定源码包 `logs/ik_no_target_braking_20261019/source_no_target_braking_v2.tar.gz` 审计通过，SHA256=`dcc74fac1adfeedec992e3d4b1e517ffabdc74fce83eb9c3da05824c2b17601d`，AGENTS与统一入口已同步；此前用于固定归档回归的 frame_f1 v5/v6 包含相同代码。exper017旧失败回传保留为诊断，不把基线检查点称为训练结果。

## 2026-10-19 exper018 启动参数拒绝与统一入口修复

- exper018 作业5610在源码训练进程启动阶段失败，尚未进入预填充；当前 CLI 已删除 `--target-music-f1` 与 `--target-micro-f1`，但统一入口仍传入这两个旧参数。
- 从 `all_use_this_train.sh` 移除已删除参数，保留当前 `--best-f1-metric f1`、奖励配置与其余训练预算；本次不修改固定源码归档或奖励公式。
- 诊断保存在 `results/exper018`，不覆盖 exper017；下一次按编号使用 exper019。需重新核对入口脚本 SHA 后再提交。

## 2026-10-19 exper019 重试已越过首个预填充故障

- exper019 作业5617使用移除旧 CLI 参数后的统一入口，四卡三路采样／单卡学习、`--no-physics-graph` 重现条件、Prelude 注册谱和固定源码归档保持不变。
- 启动日志已打印完整奖励配置；预填充完成 1 个 episode、1586 个向量步（812032 transitions），当前仍在运行，未再出现 exper017 的 Warp CUDA error 700。后台回传 watcher PID53452 已启动并记录首次真实查询。

## 2026-10-19 Turkish March advanced 注册候选指法

- 用户提供 `/home/hj/下载/turkish-march-advanced.mid`；保留原始 MIDI 副本于 `training_mjlab/inputs/turkish_march_advanced/original.mid`，SHA256=`83707e4f40347d0a51402349a8ef8af7cf59c357656b50432591b3e8fceb88eb`。
- 使用项目现有 `scripts/generate_candidate_fingering.py` 生成 `generated_v1/turkish-march-advanced_fingered.proto` 与 `fingering.csv`：轨道0→右手、轨道1→左手，2544音符、40Hz网格、8899帧、同时按键无同指冲突；proto SHA256=`ebb921fb3909f69b5ce7e388cd7b65c610e021ce8be6d94097afcc6f7e6efb84`。
- 验证：注册 proto 可加载并含 `NoteSequence.part` 指法，音符数量与原始 MIDI 一致，原始音符序列属性保持一致。限制：启发式候选，尚未进行专业校订、TsenseI IK、碰撞或实机可达性验收。

## 2026-10-19 Turkish March 指法网格频率修正

- 复核统一入口确认当前合同为 20Hz 控制步、200Hz 物理子步；此前 `generated_v1` 使用脚本默认 40Hz，仅适用于离线检查，不能标作当前训练频率。
- 按 `--dt 0.05` 重新生成当前候选 `training_mjlab/inputs/turkish_march_advanced/generated_20hz_candidate/`：2544 音符、4450 个 20Hz 轨迹帧、无同时按键同指冲突；proto SHA256=`e4dbd3cb7179cb8cc1cce403b91a6bedc930a914eea8166f51197139182d4999`。
- `generated_v1/` 保留为诊断副本；README 已明确 20Hz 候选为当前注册文件。200Hz 物理按键评分仍由训练运行时按原始 MIDI 事件执行，不由该离线指法网格替代。

## 2026-10-19 Turkish March 前10秒 MuJoCo 指法运动学回放

- 在 `volc-a32` Slurm GPU 作业5661上，从当前20Hz注册 proto（SHA256=`e4dbd3cb7179cb8cc1cce403b91a6bedc930a914eea8166f51197139182d4999`）构建当前手部/钢琴 MuJoCo 场景，截取前10秒、200Hz轨迹并渲染左右手指法回放。
- 输出位于 `results/turkish_march_kinematic_first10s_v8/`，包含 `scene.xml`、`scene_manifest.json`、`task.json`、`score.npz`、`kinematic_trace.npz`、首帧 PNG、MP4 和 `replay_report.json`。
- 这是运动学 IK 诊断，不是策略或硬件轨迹。当前通用 IK 回放最大指尖误差约0.6634m、平均约0.6358m，说明该指法/固定手部姿态在此诊断器中未达到可达性验收；不能把视频称为已成功逐键按下。后续需单独改进掌根/腕部轨迹或使用正式控制器再测。

## 2026-10-19 Turkish March 回放器 IK 修正

- 首版回放因逐指更新腕部自由度，导致左右腕部被不同目标互相推到限位；图像中双手出现在琴键两端，不能作为触键结果。
- `scripts/hpc/run_turkish_march_kinematic.py` 已改为每只手五指联合阻尼 IK，并加入掌根 TX/TY/Pitch/Yaw（后续诊断含 TZ）整体求解。Slurm5662 的联合求解回放平均指尖误差降至约1.43mm，首帧和视频保存于 `results/turkish_march_kinematic_first10s_v9/`；该版本用于当前诊断引用。
- 仍存在少数跨度较大的不可达目标，最大误差约8.94cm；该回放用于发现可达性问题，不代表策略或硬件按键成功。

## 2026-10-19 Turkish March 回放音轨补齐

- 初版 `turkish_march_first10s.mp4` 只有 H.264 视频流，原因是 MuJoCo 帧编码器未加入音频。
- 使用注册原始 MIDI 与 `/usr/share/sounds/sf2/TimGM6mb.sf2` 合成同一前10秒 WAV，并复用同一回放视频封装为 `turkish_march_first10s_with_audio.mp4`；ffprobe确认包含 H.264 视频流和 AAC 音频流。

## 2026-10-19 Turkish March 平滑按键、绿色激活与实际激活音轨

- Slurm5670 回放改为 200Hz 物理步上的琴键目标插值，使用回放读出的实际琴键激活状态着色绿色；`replay_report.json` 记录 `smooth_key_interpolation=true`、`green_active_key_render=true`，同回放控制帧/物理键 F1=`1.0`（TP=5007、FP=0、FN=0）。
- 从 `kinematic_trace.npz` 的实际激活上升/下降沿重建 `actual_key_events.mid`，用 SoundFont 合成 `actual_key_sound.wav` 并封装为 `turkish_march_first10s_green_with_actual_key_audio.mp4`。视频音轨来自本回放实际激活事件，不直接复用原始 MIDI 音频；MuJoCo 本身不提供声学波形求解。
- 另以独立 oracle key actuator 在 Slurm5669 校验指法到琴键映射：10秒、2000物理帧、finger_conflicts=0、finger_missing=0、F1=1.0；该 oracle 是合同上界，不等于训练策略或硬件触键验收。

## 2026-10-19 Turkish March 正面机位回放

- **用户要求**：为同一段前10秒指法回放生成正面的钢琴视角；指法、20Hz控制/200Hz物理时序、平滑键程和绿色实际激活状态保持不变。
- **实现与验证**：Slurm GPU作业5672使用项目统一 `preview_camera.configure_camera(view='front', framing='full', elevation=-30°)` 渲染，输出 `results/turkish_march_kinematic_front_v1/`。回放trace SHA与v11相同；报告记录 `key_f1=1.0`（TP=5007、FP=0、FN=0）、平滑插值和绿色激活。视频为1280×720、40fps，包含AAC音轨，音频由同一回放的实际激活事件生成。
- **复核产物**：同目录保留正面首帧、MuJoCo scene/task、实际激活 MIDI/WAV、以及 `actual_vs_original_midi_front.png` 和事件 JSON；对比图使用原始 MIDI事件与200Hz实际激活状态，共享0–10秒/MIDI音高轴。
- **限制**：这是运动学 IK 诊断回放，不是训练策略或硬件轨迹；`key_f1` 是琴键激活时序指标，尚未作为指尖物理接触验收。

## 2026-10-19 Turkish March 指尖—琴键实际接触复测

- **用户目标**：每个手指先对准 MIDI 指法指定琴键，再按下；验收以实际指尖—琴键几何接触和按键 F1 为准，不把目标点距离单独当作成功。
- **求解修改**：`scripts/hpc/run_turkish_march_kinematic.py` 将每只手的阻尼 IK 从24次提高为最多200次、步长降为0.15，并在键程插值后读取指尖碰撞几何与琴键几何的有符号 MuJoCo 距离。琴键绿色状态仍来自插值后的实际琴键 qpos。
- **Slurm验收**：作业5740完成 `results/turkish_march_kinematic_contact_v1/`。200Hz、10秒共5007个当前指法配对，4864个有实际几何接触，指尖指定配对接触召回率 `0.97144`；琴键实际激活 TP=5007、FP=0、FN=0，按键 F1=`1.0`。平均指尖目标误差降至0.5896mm；目标点只作 IK warm start，接触距离才是新增验收证据。
- **交付**：同目录包含正面有声视频、实际激活 MIDI/WAV、MuJoCo scene/task、200Hz trace，以及原始 MIDI/实际按键对比 PNG 和事件 JSON。接触诊断仍是运动学 IK 回放，不宣称训练策略或硬件轨迹；未将同一指尖与相邻键的多接触事件混入指定配对召回率。

## 2026-10-20 Turkish March 原谱保留、A3.2 指法比较与训练

- 问题与纠正：exper020 /5754 的 200 Hz 原谱松键覆盖不通过；错误日志 key64 指琴键索引（MIDI85），不是 MIDI64。曾未经授权缩短六处时值的 strict_v1_candidate 已撤销活动地位，保留诊断。exper021 /6037 在整曲预填充校验失败；exper022 /6047 在三采样器 CUDA Graph 并发捕获阶段失败，没有训练更新，不能称 RUNNING 或成功。初始评估 best.pt 的自动导出6153仅属于失败训练诊断。
- 用户最新选择：保留原谱全部时值，物理1000 Hz、控制20 Hz。使用固定 v4 归档（SHA `2de5639f3fed9c872f54de3e00759e83d27ccf1a48b083387fd0f9c703cc9c65`）部署独立目录 `/home/houj/projects/pianist_turkish_20261020`，仅复用依赖环境，不读取旧项目算法。未修改训练算法或奖励公式，也未宣称工作区改动进入归档。
- 指法：新增 `scripts/fit_turkish_fingering.py`。实际参考 `source_runtime.py`、`bundle.py`、`finger_coupling.py`、`fingering_targets.py`、`generate_candidate_fingering.py`；编译与当前训练一致的无臂 A3.2 模型，读取60°～80°耦合中立姿态端点和琴键目标。比较基线与三种换位权重，按统一指标选择 model_xy_movement_0.35，综合代价降低25.36%，平均XY残差18.428→16.684mm。几何代理不等于受限IK、真实碰撞或演奏成功；原先运动学直接设置琴键的F1=1不能作为物理证据。输入与报告存于 `training_mjlab/inputs/turkish_march_advanced/model_xy_1000hz_candidate/`。
- 验证：原始 MIDI 2544 音符的非指法属性精确比对通过；20 Hz 无一指多键冲突；1000 Hz 严格原谱编译和八指耦合启动检查通过。Slurm6187四卡短测COMPLETED/0:0，384 transitions、92更新，3采样器＋1学习器，32控制批、无剩余pending。短测覆盖启动与学习，不代表完整训练收敛或加速证明。
- 正式训练：预留 exper023（本地、volc-a32、h100及队列检查无冲突），统一入口末尾覆盖本曲、1000 Hz、关闭物理CUDA Graph、3M replay（整曲预填充需2,280,960条），预算19,532×512=10,000,384 transitions。Slurm6188已RUNNING，输出 `results/exper023-20261020-110336-098112`；本地后台进程（PID见transfer.pid）已记录首次真实查询。完整训练config已回读，source_runtime及奖励与6187短测一致，3M回放池预填充合同通过。使用同轮submission登记作业、路径、SHA、完整原谱来源与自动导出要求。
- 回传修复：`scripts/package_interactive_replay.py` 和 `interactive_recorded_replay.py` 打包同一次物理录制、真实模型、指法、声音及原谱对比；支持旋转/缩放、暂停、跳转和倍速。`finalize_training_video.py` 增加CPU数据验收与清单；`watch_training_transfer.py` 拒绝只有视频的完整标记，记录训练状态，缺目录/缺best时先回传诊断并明确终止为不完整。已用6153的222.75秒完整录制通过CPU模型/时序/标注/音频数据检查；GUI和音频输出设备尚未实测，需桌面环境。训练、推理、视频渲染继续走Slurm；交互CPU启动器使用软件渲染和受限systemd scope。
- 测试：7项回传/原谱来源测试通过（含新增“视频不能满足interactive要求”）；原谱校验器对本轮MIDI/指法逐音符通过。后续自动导出与回传仍须实际完成并SHA验证才算交付，未生成本轮回放前不提供虚假F1或完成标记。
- 文档同步：更新本曲README、奖励唯一总表的1000 Hz实验覆盖说明；统一默认仍200 Hz。同手weight=.1/子步、封顶5不变；跨手每子步-.01、封顶.5；按键50子步均值。TODO：正式训练状态、best.pt回放与完整交付见exper023目录。


## 2026-10-20 物理仿真与重型分析默认自动使用 volc-a32

- 问题与请求：本机物理仿真触发持续内存压力，11:41:32 systemd-oomd终止用户会话管理器并导致桌面重启；用户要求将这类过程自动使用远端volc-a32写入AGENTS.md。
- 参考：现有AGENTS.md、scripts/run_cpu_analysis.sh、scripts/hpc/submit.sh及本机诊断报告 `/home/hj/文档/黑屏排查-2026-10-20-114132/诊断报告.txt`；SSH配置通过 `ssh -G volc-a32` 确认别名可解析，未进行实际连接测试。
- 修改：替换AGENTS.md中“先在本机受限scope运行、内存不足再转远端”的旧规则，明确包括单步／1秒短测的物理仿真、场景编译、模型加载和重型分析自动在volc-a32执行，无需再次确认；补充远端源码／输入同步校验、资源核对、作业记录、结果回传及远端失败不得回退本机。README同步执行说明。
- 验证与边界：核对文档差异及旧规则冲突；仅更新执行约定，未改启动脚本、驱动或系统服务，未提交远端任务，也未声称已有脚本具备自动SSH转发。TODO：下次实际任务按新规则核验远端连接、项目路径和依赖后执行。

- 2026-10-20 11:51（北京时间）用户要求中断：已取消exper023/Slurm6188，保留远端全部文件；已有checkpoint/config/metrics及作业诊断回传并SHA校验。自动后续处理暂停，本轮未完整训练，未启动导出，不标记TRANSFER_COMPLETE。

## 2026-10-20 用户提供015-5乐谱与指法：训练前校验

- 请求：使用 `tsensei_v3_2_pianist/music/data/015-5.mid` 和 `015-5_fingering.txt` 训练，保留用户指法。新增 `scripts/register_supplied_fingering.py`，将269项有符号指号精确匹配到MIDI，不修改原始音符时值；保存独立注册proto及SHA报告于 `training_mjlab/inputs/pig_015_5/`。
- 参考：`music/midi_file.py`、`score_validation.py`、`mjlab_env/exact_score.py`、`mjlab_env/task_bundle.py` 与统一入口。原谱24.0521秒，指法文本舍入最大20微秒，真实事件无同指重叠。20Hz网格出现6帧/7项一指多键，1000Hz物理仍无法覆盖所有重复音松键。
- 远端：已核对volc-a32连接、当前独立v4部署、双依赖环境、空队列及磁盘/内存；频率扫描在远端CPU、4GiB地址空间限制、单BLAS线程完成，退出0。已测2000Hz控制与4000Hz物理可通过谱面检查，但未执行场景编译或GPU训练。该频率改变控制步语义和计算预算，不静默当作原20Hz训练。
- 为最终选择准备具体时值提案：仅提前47个松键，最大49.100001ms，不改音高、起音、力度、指法；通过20Hz/200Hz纯谱面检查。目录明确标为 `timing_proposal_NOT_APPROVED`，未授权且未用于训练；原谱完整保留。首版提案受浮点ceil边界影响残留2项冲突，修正边界1ns余量后通过，未修改活动训练算法。
- 首次校验阶段未提交，随后用户明确批准47处提前松键、20Hz控制/200Hz物理、指法不变并要求开始。活动输入保存为 `approved_20hz_200hz/015-5_fingered.proto`，SHA `07e3c1c01a8a29066d98ffda68a2697d273f529c4c9177a0ba7a4a1fe24f337c`；远端逐音符核对确认只有批准结束时间改变，严格谱面校验通过，486步/24.3秒。

- 本次正式提交exper024/Slurm6251，统一入口使用固定v4归档，显式本谱20Hz/200Hz并关闭physics_graph避开既有并发捕获错误；保留其余默认预算/奖励，无臂，3路采样＋1路学习，2M回放池足够248832条整曲预填充。已核对本地、volc-a32、h100编号及队列无冲突；远端GPU探针、场景编译、耦合检查和完整奖励打印通过；实际config已回读验证。结果路径 `results/exper024-20261020-133136-100699`，后台PID记录于本地transfer.pid，首次SSH断连后60秒重试已得到RUNNING。
- 导出配套：修正 `plot_recorded_midi_comparison.py` 对外部谱开头静音的时间对齐（本曲源起点0.0049秒），保持原文件/原事件证据，仅应用与训练相同的去静音原点；避免把4.9ms原点差算作触键误差。参考批准注册谱，保留未改MIDI和批准清单。新增非零原点回归测试，远端6项来源/回传测试通过。
- 遵守新的远端模型规则：`package_interactive_replay.py` 默认生成远端启动器；`interactive_recorded_replay.py` 支持音频控制事件；新增 `launch_remote_recorded_replay.py` 通过SSH/X11在volc-a32加载/软件渲染模型（4GiB地址空间限制），本地只用ffplay播放同一次录制的声音，不在本机加载模型。脚本静态检查通过；真实桌面/X11/音频设备交互仍待实际打开验证，不称已实测。
- 文档同步：本曲README、根README及奖励唯一总表标明exper024当前配置；没有修改奖励公式或模型、也没有把辅助导出脚本改动称为进入训练归档。TODO：训练完成后的best.pt导出、同次对比图、SHA回传验收尚待执行，不能提前标记完整交付。

## 2026-10-20 015-5 无强化学习 IK 真实触键轨迹（进行中）

- 用户目标：保留 `music/data/015-5.mid`，通过IK和真实接触驱动琴键，整曲物理F1≥0.95。最初保留配套指法，用户观看第一轮后明确授权修订不可达指法并要求修复拇指键侧/键底接触；原始音高、时值与左右手分配仍保持。以未修改原始MIDI时序验收；不把训练用提前松键谱、运动学直接写琴键或历史F1=1作为本任务达标证据。
- 实际参考：`scripts/hpc/run_turkish_march_physical.py`、`source_runtime.py`、`bundle.py`、`finger_coupling.py`、`target_motion_limits.py`、`action_mapping.py`、统一入口及 `results/exper024/config.json`。旧物理诊断安装高度与腕轴未对齐当前训练，历史真正物理测试也未达到目标。
- 新增 `scripts/hpc/plan_piano_ik.py`：独立scratch IK，物理状态仅由原位置执行器和mj_step演化；当前模型安装高度0.06m、Shadow驱动、整link凸包、结构性移除TZ/Yaw，接入60°～80°八指耦合与当前腕目标限幅；原谱逐音符校验，记录200Hz实际激活、接触力、碰撞、qpos/qvel和控制目标。该求解器仍在调试，不宣称已经满足完整训练控制合同或达标。
- 远端独立部署：`volc-a32:/home/houj/projects/pianist_ik_015_5_20261020/source`。本地当前归档 `results/ik_015_5_20261020/source_initial.tar.gz` SHA256=`b790721c6ffb948f4bbe7245c3b99c8d830db3b021b0a6dd209a79d83b079b18`；为减少网络传输仅用远端副本作rsync种子，随后以本地清单逐个核验1154文件全部一致，无差异；新求解器另行同步与记录SHA。依赖仅复用独立venv，不读远端旧算法。
- 运行：远端CPU、无GPU、限制内存及BLAS线程；本机未构建/加载模型。首次OSMesa依赖导入失败，改为禁用渲染；首个2秒录制成功保存，但JSON中numpy布尔序列化失败已修正，原始诊断保留。此短测原谱micro-F1仅0.04671，绝不视为交付。正在分析耦合IK与物理跟随误差，TODO：优化整曲、独立复跑、源指纹、轨迹和真实触键对比验收。

- IK 后续进展：独立控制获用户确认；补齐训练 hard_v1 机器人子步硬限位。整曲 opt21_full_best 原谱 micro-F1=0.726605656，逐帧均值0.702465852，尚未达标。新增 refine_piano_controls.py，以真实动力学预测优化执行器目标；独立 replay_v28 的关节、控制、按键、接触序列与种子逐元素完全相同。新增纯数组绘图脚本与同次原谱/实际PNG及事件JSON。前2秒预测优化提高至0.926611，整曲预测优化仍待验收；未改训练源码/奖励，未使用RL。详情、来源和未解决项统一维护于 docs/ik_015_5_physical_20261020.md。

- 用户追加导出第一轮视频：第一轮整曲预测优化 mpc_full_v30 已完成，micro-F1=0.796738399，逐帧均值0.772217814，未达标。新增 Slurm-only 录制轨迹导出器和显式GPU导出验收（6317），只复用该次200Hz状态，真实触键音轨／指法／激活高亮和原谱对比PNG同源；不重跑策略、没有checkpoint。6317 COMPLETED/0:0，1项导出验收通过（175.12s）；24.3333秒有声视频与PNG等13项产物已回传并逐项SHA验证，首秒及12秒画面/同次对比图已检查。独立物理重放replay_full_v46的qpos/qvel/ctrl/激活/接触逐样本完全一致。旧生成器末尾哈希捕获缺陷已显式注明，新运行改为不可变runner快照及启动时哈希。第一轮导出完成，原整曲0.95目标仍继续。

- 指法/拇指修复：新增`revise_015_5_fingering.py`，依据A3.2指尖间距与时序生成独立修订谱，逐字段确认只改note.part。第一候选154处变化；第二候选155处变化，另加80ms异音同指释放间隔（规划参数，不修改谱时值）。`plan_piano_ik.py --park-thumbs`通过原执行器收拢抬起闲置拇指，拒绝含拇指分配的谱，保留碰撞/力限/硬限。`audit_ik_thumb_clearance.py`从完整200Hz录制按实际接触法线审计：旧mpc_full_v30有1716帧键侧、520帧键底接触；第一修订整曲revised_seed_v49的4861帧中拇指全部键接触为0。其F1只有0.49276，不能宣称完整修复交付；手腕平滑后revised_seed_v50为0.66883，继续动力学优化与第二候选验证。首次冻结runner位置导致依赖路径解析失败，保留v48日志/已正确构建的修订谱场景，改在scripts/hpc保存不可变runner后v49成功；未回退本机执行。
- 修复版v55完成：采用v2修订指法，65个拇指音符全部重分配，拇指收拢抬起。完整200Hz实际回放micro-F1=0.827396554、逐帧均值0.790694971，拇指键顶/键侧/键底接触均为0；v69独立物理重放所有关键数组逐元素一致、轨迹SHA同为40c28dedfbe3dae79c93317180dfa186990f10831362dfe256cacf931f805998。Slurm6364导出修复版视频及同次原谱对比图；导出器改为使用实际运行名称、验证并附带修订谱和原始MIDI、生成器快照及复现记录。静态编译通过，MIDI逐字段保留验证通过；整曲0.95目标仍未完成。门控/纵向落点/其他优化目标候选保留试验记录，不声称已作为v55生效，不修改训练奖励或训练指法。
- 修复版导出验收完成：6364 COMPLETED/0:0，Slurm导出测试1 passed（164.97s），20项产物已回传并逐项SHA核对，视频/音轨解码、1秒/12秒预览和原谱对比PNG检查通过。v2指法生成器按已记录SHA恢复完全相同快照（72e7f6f0…），源码增量归档`thumb_fix_source_v1.tar.gz`的SHA为adbfb04bcd88b217f5806fe0a3ed71a781c65a352f374abe5e8409a8bcf0953b，与既有初始完整归档配套。当前仍有漏键，继续原整曲F1目标，不把拇指接触通过等同整体目标完成。

- 用户复查开局双拇指相碰：旧v55只检查键接触，补录准备段的v78检出1048帧拇指之间接触。重新搜索八关节停放姿态，增加显式姿态JSON、episode初态选择与平滑准备段，禁止`--park-thumbs`静默复用原向内伸展姿态。`refine_piano_controls.py`保存准备段物理状态及全程四类拇指接触/力；音乐评分时段保持。v84全5261帧四类拇指接触均0，v86独立回放24数组逐元素一致；F1为0.75990/0.73426，低于旧版，继续按压优化。实际参考v55物理轨迹、模型凸包、v75/v76几何搜索、v77～v84实际回放；全部模型工作在volc-a32，未改训练奖励/模型/指法。当前代码静态编译通过，候选v84视频导出6377完成。最终v87仅优化其余手指，v88独立重放一致：F1=0.834950596/0.790913262，四类拇指接触仍全0，优于旧v55两项F1；最终视频/图回传待完成，整曲0.95目标仍未达到。

- 用户最新纠正：只减少拇指使用，不接受全部拇指指法清零；闲置参考当前initial。全禁用版本v84/v88/v89降为对照。读取initial精确双侧四角和SHA，新增初态参考JSON、`restore_reachable_thumb_fingering.py`生成17个拇指音符的v4候选（右5左12，原谱字段除part全部不变），加入闲置姿态参考、拇指专用深度/顶部接近/近端几何净空以及细分键侧/键底/非末端接触记录。v90直接使用initial角仍有少量碰撞，正在volc-a32验证v91新方案；不能把静态筛选当作物理验收。TODO：实际拇指击键、闲置回位与整曲F1验证、最终来源归档/视频回传。
- 逐音符物理检查发现v4只改拇指、未同步伴随和弦指法会破坏可达性；v5恢复16项伴随原指法，仍保留17个拇指音符，全部非part字段一致。v101加深拇指目标后micro-F1=0.721459659，已出现左右拇指真实触发，但仍有漏键及144帧全程双拇指接触。新增整link拇指—对侧全手分离约束，160对几何四姿态Jacobian中心差分最大误差2.72e-10；MPC启用避让时也允许调整闲置拇指，删除旧近端中心距离启发式。`prepare_ik_wrist_transition.py`生成负时间准备段五次过渡命令，不改物理状态；v106全5261帧跨手拇指接触均0，micro-F1=0.718732887，但仍有键侧接触，继续真实动力学优化。源码静态编译通过；模型/物理均在volc-a32，未修改训练归档/奖励。生成器开始保存几何helper快照及SHA，便于独立复现。
- 后续v108整曲micro-F1=0.816730038且跨手拇指接触0，但键侧/同手接触未通过。新增逐物理子步禁用接触优化代价和闲置initial姿态正则化，未改训练奖励；短窗口v115将键侧110帧降到8、近端38降到0，仍有左右拇指真实触发，不能冒充整曲验收。v6按已有真实触键召回筛选保留10个拇指音符（右1左9）；新轨迹未通过则不沿用旧证据。实际末端外壳检查发现邻键侧碰与前缘合法悬伸，修正落点约束为低顶点横向范围；6姿态Jacobian差分最大1.74e-10，通过。加入松键先抬离再回initial、准备段命令过渡与可复核命令组合工具。v113组合实验未成功保留右拇指触发并重现跨手接触，明确不采用。v117恢复右拇指真实触发（指定接触激活召回0.702），但整曲F1=0.68844且仍有122帧键侧/30帧双拇指接触；v118/v119继续完整物理优化，F1≥0.95和最终视频交付仍待完成。 选择性拇指源码增量归档`selective_thumb_source_v3.tar.gz`（61项）的SHA为`b0822cb6bbf77209268f5fb10834c6138dd24d0f3856b36939b40d8bc8f059ee`，远端校验通过；27个已完成候选/诊断的151项结果回传SHA全部通过。v118/v119后台wrapper PID为2768307/2768917，已观察真实优化日志；不将运行中状态或候选局部触键称为完成。

- 2026-10-21用户要求暂停并导出当前最佳折中候选：目标已paused，v130退出0且无存活优化进程，不启动新仿真。选择v129（独立复现v118全部25数组），保留右1左9拇指音符、initial闲置参考；全程跨手拇指接触0，但键侧12/同手30/近端碰键65帧和部分拇指漏音仍未修复。整曲micro-F1=0.7734483961、逐帧均值0.7494723145，未达0.95。Slurm6546导出成功，同次视频/声音/原始MIDI对比PNG及22项来源产物SHA与解码通过，位于`results/ik_015_5_20261020/runs/selective_verified_v129/video_export_v1/`。新增跟踪分解、非拇指目标补偿与有界采样搜索的来源/验证/局限已写入`docs/ik_015_5_physical_20261020.md`；v130尚未独立验收，不作为交付。当前瓶颈为计划与实际跟踪误差、拇指外壳避碰和足够键程之间的制约，以及局部优化的时序协调。

## 2026-10-20 exper024 单卡 TF32 隔离验收

- 用户要求在不中断当前训练的前提下，对照单卡 TF32 的速度与数值差异。参考 exper024/config.json、固定 v4 归档内 rl_torch/sac.py、learner_graph.py、replay.py，以及 PyTorch CUDA TF32 说明。
- 新增 scripts/hpc/test_tf32_learner_benchmark.py：独立 Slurm validate 作业，保护当前训练 GPU UUID；冻结 forward.pt，保持完整 SAC/Adam/RNG、网络维度、batch、replay 容量及学习加速配置，仅切换矩阵乘法 TF32。ABBA 四进程对照，固定合成 replay；报告单步与累计数值差异、同精度重复误差和稳态速度。
- 运行记录与限制：logs/tf32_learner_20261020/README.md、submission.json；当前训练 6251 不修改，验收作业 6270。脚本本机仅 py_compile，GPU 测试在远端执行；最终结果待回传后补记。没有修改活动训练算法、默认精度或源码归档。
- 用户随后明确允许IK独立控制手指各关节、保留物理限制；本轮求解器默认关闭八指耦合，仅影响独立IK实验，不修改训练默认、奖励或模型。腕TZ/Yaw仍移除，关节范围、原执行器/力限和腕目标限幅保留。早期几组3秒真实物理诊断最高micro-F1约0.543，仍不达标；正在改为带关节边界的最小二乘IK并检查动力学跟随。

- 验收完成：6270 COMPLETED/0:0，pytest 1 passed（122.18s），总作业2:29；FP32 2.34079 ms/update，TF32 1.45488 ms/update，1.60893×。1024次更新后Actor/Critic参数相对L2差异0.02612%/0.02950%，动作最大绝对差异0.0005544；同精度重复已检查量完全一致，随机状态一致，全程有限。结果及输入/数值快照已回传并逐文件SHA验证。原6251仍运行，学习约299.92ms/128updates，与测试前基本一致。TODO：固定合成数据不能证明真实演奏F1或长期收敛等价；本次不切换当前训练精度。

- 用户后续授权：保留 TF32 对照，在 exper024/6251 结束后启用学习器 TF32，并继续精度受控的提速对照。已登记 logs/tf32_learner_20261020/followup_plan.json 与本聊天跟进任务 tf32（每15分钟检查，运行中不修改，无变化静默）；既有回放/回传后台脚本保持独立。当前训练入口与归档尚未切换，不能把待执行计划说成已生效。
- 后续核验并补齐当前训练的 `hard_v1` 机器人子步硬限位（`joint_limits.py` 解析范围，`joint_limits_warp.project` 的CPU等价操作；琴键排除），报告明确区分机器人限位投影和绝不写真实琴键状态。全曲 `full_wrist_v19` F1=0.682914，已录制机器人状态越界为0；短段 `hard_wrist_v16` 3秒=0.819746、`smooth_v22` 6秒=0.840339，均未达标。对exper024保存的源哈希进行比对，已有文件差异0。
- 规划加入有边界最小二乘、有效键面纵向落点、非目标接触几何代价、腕部平滑/提前参考与可选测量反馈；参数和尝试均留独立记录。物理200Hz保持，明确试测20/100Hz指令且按同一秒制腕限幅，不修改训练默认。原谱时序/指法不变。新增本任务说明并链接README；完整算法与交付验收仍在进行，不能将局部成绩称为目标完成。

## 2026-10-20 exper025：TX/TY新阈值与学习器TF32

- 用户明确要求直接改统一脚本并新开训练，覆盖此前“等exper024结束再切换”的时序：TX/TY每轴硬速度1.99m/s、硬加速度1.99m/s²；软速度1.90m/s、软加速度1.90m/s²。速度阈值上调，加速度沿用原值。新实验从头开始，不继承exper024权重；保留015-5批准谱/原配指法、20Hz控制/200Hz物理、其余预算和奖励。exper024/6251不改动。
- 实际参考：exper024/config.json、统一入口、固定v4内rl_torch/train.py、learner_graph.py及已有图捕获测试；前一轮独立TF32微基准只用于性能/数值参考，不证明长期精度等价。
- 学习器新增显式 `--learner-matmul-precision ieee|tf32`。TF32要求compile和learner graph；首次学习前现有pipeline drain确保采样线程完成，再以TF32捕获图，finally恢复全局IEEE FP32。后续图重放保留已捕获GEMM精度，采样/评估仍IEEE FP32。完整config记录三种角色精度。仅直接改全局开关会影响采样/评估，因此不采用。
- 新增 tests/test_learner_precision.py：实际1406观察/31动作/256³网络/batch1024，验证捕获期间TF32、恢复全局开关、重放不依赖后续全局精度、异常也恢复。远端独立Slurm6316，10项新旧图测试通过（82.23s），14项仅弃用警告。
- 冻结来源：在已验证v4归档上仅替换train.py与learner_graph.py并新增上述测试，不把其他聊天未验证的工作区修改带入。新归档 logs/learner_tf32_20261020/source_learner_tf32_v5.tar.gz，SHA256 be9f6c441fade49453867baeb685fde90a986c282a026ab99e9ba61be126ec43；来源差异与构建脚本同目录保存。旧归档仍是正在运行exper024及其回放依赖，不删除或迁移。
- 直接更新all_use_this_train.sh的速度/软阈值、TF32参数与固定归档/SHA；同步奖励唯一总表的当前参数、公式示例、硬限表，以及README/AGENTS。不修改运动罚公式、模型、物理、时值或指法。后续自动跟进已更新为不重复提交、不覆盖新设置；两轮阈值不同，不能将全部质量变化归因于TF32。
- 提交结果：exper025/Slurm6318，四卡，10,000,384 transitions/19532步，独立输出前缀results/exper025；当前PENDING(Resources)，现有6251占4卡、独立IK任务6317占1卡，仅余3卡。不得把排队称为已开始学习。后台回传PID162162，已记录首次真实PENDING查询；完整best.pt有声交互回放、同次对比PNG与SHA回传要求写入submission.json。
- 从Slurm保存的真实SubmitLine重建有效参数（远端CPU纯配置解析，不构建模型），打印全部23项奖励，保存submitted_effective_config.json/submitted_reward_config.json/submitted_command.txt。对照exper024确认奖励仅tcp_velocity_limit由0.9改1.9，SAC/batch/环境数/replay容量一致；硬TX/TY速度1.99、加速度1.99、TF32学习图开关均正确。归档本地回传SHA验证通过。实际运行config及学习启动仍待调度；既有后台流程会保留并回传完整实际产物。

## 2026-10-20 恢复双卡默认与隔离性能验收（进行中）

- 用户要求四卡收益不明显时恢复双卡并尽量提高双卡性能。统一入口直接改为`--gpus 2`、`cuda:0`采样512环境、`cuda:1`学习、单批流水；保留TF32学习器、batch1024、每条经验0.25次更新、10M预算、20Hz控制/200Hz物理、新TX/TY阈值及全部奖励/诊断。README和AGENTS同步。不改变模型、指法或奖励公式，当前v5归档/SHA不变。
- 实际参考：exper025真实config（已回读）、rl_torch/train.py、collector_pipeline.py、learner_graph.py、mjlab_env/preview_contract.py与scene.py。exper024/6251与exper025/6318均已在A100运行，共占8卡，保持冻结配置；入口修改不会缩减运行中的Slurm分配。
- 用户随后查询H100空闲资源，确认8卡中5卡已分配、3卡可申请。为隔离对照建立`h100:/home/houj/projects/pianist_dual_perf_20261020`，只复用依赖环境，以本地v5归档及完整config为源码和参数依据；传输可用旧归档作rsync种子，但最终须核验本地v5完整SHA，绝不运行种子旧代码。
- 新增`scripts/hpc/test_dual_gpu_performance.py`：独立双卡Slurm validate、ABBA四进程；相同动作全曲比较512环境qpos/qvel/奖励/观察，真实物理经验重复填满2M replay，TF32学习图、128次更新/512条经验，测采样/学习同时运行的稳态时间；固定策略对照3环境完整评估的指标图开关。借用已冻结exper024 forward.pt仅作为数值基准起点，不作为新阈值续训或质量排名，短测不代表长期精度保证。
- 修正活动TF32微基准脚本在v5显式传`matmul_precision`，避免v5默认IEEE覆盖旧的全局TF32开关而误测；历史6270脚本和结果仍原样保存在logs/tf32_learner_20261020。
- 本机仅执行bash语法/Python编译等静态检查，不加载模型。测试命令、作业和结果统一记录于logs/dual_gpu_20261020；TODO：远端数值与吞吐验收、据结果采用或拒绝额外开关、结果回传核验。尚不能宣称双卡已测得加速或已达到硬件上限，H100和A100绝对速度不能直接归因于卡数变化。

- 双卡测试已提交H100 Slurm1942（RUNNING），独立分配2卡；确定性重建后的v5归档与本地完整SHA完全一致、1130文件逐项验证，Composer/Learner全部依赖版本一致。新增通用测试回传后台scripts/watch_benchmark_transfer.py，本地PID1618732，已取得首次真实RUNNING查询；终态回传结果/轨迹/诊断并核对远端SHA，失败也回传，不将失败标成功。测试仅控制帧qpos/qvel/奖励/观察逐样本对照，物理200Hz触键/碰撞通过完整评估指标核验，不冒充全部子步状态逐样本对照。
- 回读exper025真实config确认本轮阈值及仅学习器TF32正确，运行已观测3000控制批次/1536000 transitions/321792 updates；更新validation.json。原训练继续四卡，当前入口已双卡，二者明确区分。

- H1001942的A1/B1实测225.293/221.147ms每批、2272.60/2315.20条经验每秒，单次差异约1.9%；在同动作轨迹严格相等断言处FAILED/1:0，后两组未运行，不能宣称ABBA通过或稳定收益。A1/B1及配置/归档验证JSON已回传并校对远端SHA，较大原始trace后台仍在传输；回传器改为分块hash以控制本机内存，当前PID以transfer.pid为准。
- 新增独立诊断scripts/hpc/test_dual_gpu_diagnostic.py，Slurm1943 COMPLETED/0:0、1 passed（63.93s）。8个固定物理输入下耦合普通计算/图计算的目标和全部诊断量完全一致；关闭图的独立512环境回放也与原始记录不同（第0步qpos最大差0.031103）。因此严格全轨迹等值门槛未能隔离图开关影响，不能据此断言图算法错误或图导致精度下降；不同GPU/独立物理回放重复性仍需进一步定位。
- 最终额外物理图不采用：根入口显式--no-physics-graph、保持--no-eval-metrics-graph；TF32学习器/compile/fused/learner graph与单批流水保留。仅约1.9%且未完成重复对照的收益不足以支持启用，评估图也因独立回放差异不能单独验收。README/AGENTS同步；算法源码和v5归档不改。后续先建立可重复基准，再针对采样做有界优化，不能靠降频/减少更新/关闭诊断冒充加速。决策记录logs/dual_gpu_20261020/assessment.json。

- 最终回传完成：1942共10项（含原始轨迹）和1943共2项全部对照远端manifest逐文件SHA验证；1942仍标记benchmark_passed=false、1943=true。统一入口最终参数经拦截submit实参验证（未额外提交训练），确认双卡/TF32/流水、512/1024/0.25、20/200Hz、最新TX/TY阈值、物理图及评估图关闭。既有跟进任务已更新为先消除对照混杂、分析采样瓶颈，不盲目重跑失败的严格轨迹等值验收。

## 2026-10-20 可交互MuJoCo回放默认改为本地RTX

- 用户报告exper024远端启动`libc++abi: terminating`后明确要求本地RTX3050加载，后续默认本地读取回放。原远端4GiB地址空间限制下复现崩溃；仅远端放宽到16GiB后数据验证通过，XML编译峰值RSS约4.8GiB。`.Xauthority`首次创建提示不是本次模型加载失败的充分解释。
- 实际参考：scripts/interactive_recorded_replay.py、launch_remote_recorded_replay.py、package_interactive_replay.py、run_cpu_analysis.sh及exper024同次录制model.mjb。回放改为直接加载MuJoCo版本匹配的MJB；缺少二进制或版本不同明确拒绝，不回退XML编译。模型、训练源码归档、checkpoint、轨迹与评分不变。
- 新增launch_local_recorded_replay.sh，由后续导出打包为launch_local.sh/launch.sh；旧launch_remote.sh与launch_cpu.sh转为本地别名。自动查找项目依赖环境，设置NVIDIA PRIME/OpenGL和单线程BLAS；systemd独立用户scope限制MemoryHigh=2300M、MemoryMax=3G、MemorySwapMax=0，启动要求至少4GiB可用内存。未提高本机内存限额、修改swap或关闭OOM保护。系统不支持的MemoryOOMGroup属性已移除，实际scope启动成功。
- 本机已明确获用户授权后验收：MJB加载和首/中/末帧标注检查2.54秒，峰值RSS 2,303,944KiB；新增--render-check实际创建OpenGL上下文并渲染三个时点，确认NVIDIA GeForce RTX3050 Laptop GPU，9.00秒、峰值RSS 2,304,320KiB，均退出0。音频文件完整性验证通过；未宣称已人工听验扬声器或测试所有鼠标/按键交互。
- exper024现有本地回放已修复，保存local_render_validation.log和local_playback_upgrade.json，更新交付manifest；相同辅助脚本与修复包同步至volc-a32现有导出路径，以免未来回传恢复旧启动器。后续导出模板在本地与当前远端部署同步；训练和重新物理仿真仍远端Slurm，新增本地录制回放例外写入AGENTS/README。本次不是训练算法修改，不修改固定训练归档SHA。

### 本地回放窗口无响应修复（同日用户反馈）

- 真实桌面打开passive viewer后，scope memory.events high=28899，渲染线程停在mem_cgroup_handle_over_high；4GiB检查只能保证总内存余量，之前短时隐藏渲染检查未覆盖原生窗口的额外内存。已停止该卡住的回放scope，未影响训练。
- 改为直接单线程GLFW事件/鼠标/播放控制与MuJoCo渲染，避免passive viewer额外开销；保留2300M软阈值/3G硬限和swap=0。只播放既有物理轨迹，改用kinematics/comPos/camlight刷新显示变换，不重新求解碰撞/约束；暂停时缓存相同姿态。鼠标接口按本机MuJoCo3.11的五参数mjv_moveCamera修正。
- 6310880个网格三角面使阴影/反射渲染约0.74秒/帧。保留完整几何，关闭这两项光效后实测66帧/5.036秒，scope high/max/oom均0。再做有声动态播放检查，窗口关闭前15帧/2.223秒，无音频进程报错、scope节流/OOM均0；不把提前关闭说成完整5秒或整曲验收。
- 网格上传阶段保持窗口隐藏，上传完成后再显示和处理事件，避免桌面将初始化阶段判为无响应。旧启动命令兼容；后续导出模板和exper024修复包同步。新日志local_window_validation.log记录实际检查，更新交付散列。不提高内存上限、不改变训练模型/奖励/轨迹。

## 2026-10-20 exper024 初态双 M6 安装及琴键碰撞检查

- 用户要求在刚核对的 exper024 t=0 手姿上安装机械臂并检查琴键碰撞。参考 exper024 同次录制 MJB/initial_qpos、v4 冻结 arm_follow/arm_scene/arm_options、当前保存的双侧七角和四孔连接件。独立静态装配保持双手完整位姿，以 FK 反算基座；未修改运行实验、默认入口、奖励或训练归档。
- 新增 logs/exper024_arm_install_20261020/test_installation.py 与 test_distance_verification.py 及复现说明。本机仅静态检查；volc-a32 经 scripts/hpc/submit.sh 单卡 Slurm 执行。6367 缺 M6 资产失败；补充33个本地项目资产并逐文件SHA校验，6368/6369 均 COMPLETED/0:0。冻结v4算法保持不变，补充资产和独立分析脚本另有来源/SHA。
- 检查28个臂／连接件几何体与88键及1个琴体，共2492对。原距离接口4个零结果与最近点不一致；补充保守世界轴包围盒分离证明，拒绝异常零值。全部2492对严格分离，双侧连接件到琴键最小距离上下界均46.964102mm；臂身本体到琴键保守下界约64.2487mm。手根位置/旋转保持误差分别<2e-16m/<5e-16。
- 报告/距离数组/完整安装和近景PNG已回传并验证11项远端manifest散列；原始静态MJB保留远端，路径记入submission。相关说明见同目录README.md，最终结论以distance_verification.json为准。局限：仅t=0，不验证整曲运动、双臂互碰、实物基座或真实琴体完整外壳；未推理策略或积分物理。

### 2026-10-20 本地回放去除手指标签及重播

- 按用户要求，scripts/interactive_recorded_replay.py仅在录制回放中隐藏十个指尖悬浮标签及其标记，保留琴键指法和实际触键高亮；未修改共享preview_fingering、模型、轨迹或训练归档。
- 新增R键从0秒立即重播，清空姿态缓存并重启声音；保持当前速度和视角。窗口标题、终端提示及package_interactive_replay.py生成的README同步说明。
- exper024现有回放与后续导出模板已更新并同步当前volc-a32部署，四项回放修改文件已比对远端SHA，更新manifest及回传记录。Python语法检查通过；本次未重新加载模型或进行GUI按键实测，已打开窗口需重启后生效。

### 2026-10-20 全部活动视频／回放源码移除悬浮标签

- 用户要求直接修改所有源码：preview_fingering.py删除指尖编号球及琴键指法文字球的构造，公共绘制只保留目标琴键颜色；所有调用该公共组件的视频、JAX/MJLab预览、轨迹回放和IK导出均不再产生悬浮标签。保留底部指法说明、状态文字、实际触键高亮、声音及原始MIDI对比图；不改乐谱指法数据。
- 实际参考和修改：scripts/export_videos.py、watch_training_transfer.py、finalize_training_video.py、hpc/export_ik_recording.py、interactive_recorded_replay.py、package_interactive_replay.py，以及preview/scripts/export_video_job.py、render_mjlab_checkpoint.py。删除标签开启分支及自动提交参数；报告finger_labels固定false。旧后台任务的--finger-labels仅作为隐藏无效参数接收，不能恢复标签。
- finalizer改为核验plan指定归档的SHA及解包后完整源码指纹，避免用已更新的显示辅助源码冒充旧检查点物理源码；未放宽指纹校验、修改历史归档或固定训练归档SHA。README/AGENTS更新当前展示要求，既有历史视频不因源码修改而自动去字。
- exper024现有回放包同步公共无标签绘制并保留R重播，当前volc-a32部署同步9项活动显示/导出源码；5项交付修改更新manifest及远端SHA校验。60项CPU导出单测通过，Python静态编译通过；未重新进行GPU渲染或重导历史视频。训练算法、模型、物理频率与奖励未改。

## 2026-10-21 指法奖励目标距离

按用户要求，所有十指目标改为白键距世界+X前端40mm、黑键20mm；Y/Z及奖励容差保持。共享目标函数、Composer、批量环境、任务manifest及目标绘图同步，删除旧拇指专属模式/参数，拒绝旧几何合同，不迁移历史checkpoint。参考 fingering_targets.py、task.py、task_bundle.py、source_runtime.py 及奖励总表；同步README。验证：远端volc-a32独立目录完成8项目标/源码重建测试；本地语法、diff及归档SHA核验通过。新归档 source_key_targets_40_20_v1.tar.gz（SHA 1a105ec4e447cd288327445474a2996df9f1e65b18e4715debfda7e1ece14ab5）已回传并绑定统一入口。记录保存在 logs/fingering_targets_20261021；现有历史归档保留作为冻结回放依赖，本次不启动训练。

## 2026-10-21 exper024 安装图标注 base_link 与法兰 XYZ

- 用户要求直观看双手base_link与法兰轴向。参考上一轮logs/exper024_arm_install_20261020的静态MJB/初态、冻结安装报告；在独立logs/exper024_frame_axes_20261021/test_render_axes.py中读取实际body.xpos/xmat，逐侧验证相对旋转矩阵，按渲染相机投影真实原点及70mm正轴。红X绿Y蓝Z，四格同视角对照及单图，报告保存世界位姿与像素端点。
- volc-a32经submit.sh独立单卡作业7775初版正交相机造成轴过小；7777显式45°透视相机后成功0:0，轴投影长度断言及人工看图通过。6项产物回传SHA核验，相关说明和提交记录在同目录；无物理积分/策略推理/训练变更。图明确标记为此前exper024静态装配，不冒充后续新手模型或右连接件+8mm配置。

## 2026-10-21：exper027 原配置追加1000万经验（exper028）

- 用户在分支会话明确指定exper027/Slurm7708继续训练1000w步。按原实验transition口径追加19532×512=10000384条经验，原10000384条完成后累计20000768条；保留主会话轨迹文件，不引入离线装配机械臂。新编号exper028在本地、远端项目/排队记录查重后独立保留，原exper027不覆盖。
- 使用当前唯一all_use_this_train.sh，末尾仅原谱、--warm-start父final.pt、同预算和独立输出覆盖。父PT SHA bb884c38cca63738eabd1539d5733d40d237d7caa23d84858526f20f88c614cd，原谱SHA8bbf0c35890361f544012286bfc886a3a69a132e0f6bf8af7a610bb3970564db，固定归档cc692240a53e23f40bdf1dab66c1a64a00183cfa2a1b254010c15065c570abb0均核验。远端新部署/home/houj/projects/pianist_exper028_20261021来自匹配归档及当前本地入口/显示工具，仅复用既有依赖环境；13项部署文件逐项SHA一致。
- Slurm7934已获双A100并RUNNING；实际输出results/exper028-20261021-222133-103139。实际config.origin确认learner_warm_restart、父19532批/2437888更新、网络/优化器/温度/RNG恢复，fresh_optimizer_alpha_rng=false；经验池、环境与调度重建，非逐步精确断点。15项配置字段（含完整奖励、SAC、源码配方、动作/控制合同、采样/学习配置）与exper027完全一致；实际启动打印完整奖励，本地startup_reward_config.json及startup_validation.json保存证据。没有奖励/算法修改，不改统一默认或训练归档。
- scripts/watch_training_transfer.py后台PID2376322已存活并完成首次真实RUNNING查询；submission配置本轮best.pt的有声无标签交互回放、同次物理触键与原始谱对比PNG、Slurm导出、自动回传及散列验收，远端ffmpeg工具可用。README与启动命令在results/exper028；目前完成检查点加载，正在启动/经验预填充，尚无本轮最终F1或导出结果，不宣称训练完成。

## 2026-10-21：新增yaw15°训练，最终高度0.080m（exper030）

- 用户新增初始yaw15°、2000万经验实验，随后将高度最终明确为0.080m。初次0.060m的exper029/Slurm7954已取消，停止其后台回传进程，保存原输出/日志与取消原因；不将取消实验标记完成。使用新编号exper030，原exper028/7934续训未中断。
- 参考当前all_use_this_train.sh、exper027完整config、原始编辑谱及watch_training_transfer.py。仅在统一入口末尾覆盖--height 0.080、--yaw 0.2617993877991494、原谱路径、39063批预算和独立输出；从头训练、不加载旧朝向检查点。39063×512=20000256条经验。默认无臂，物理/控制频率、奖励和学习参数不变；没有修改默认入口或奖励总表。
- 独立远端部署/home/houj/projects/pianist_exper030_20261021，固定归档SHA cc692240a53e23f40bdf1dab66c1a64a00183cfa2a1b254010c15065c570abb0、原谱SHA 8bbf0c35890361f544012286bfc886a3a69a132e0f6bf8af7a610bb3970564db核验，13项本地入口/导出文件逐项SHA相同。全部模型构建与训练在volc-a32 Slurm7956双A100分配中执行，本机未运行仿真。
- 实际config已确认height=0.080、yaw=pi/12、39063批/20000256条、无warm_start；13项学习/奖励/控制字段与exper027完全相同，source_runtime仅pose改变。完整启动奖励保存results/exper030/startup_reward_config.json，启动证据和实际config位于startup_validation.json、deployment/running_config.json；实验README和命令同时保存。
- 本地后台watcher PID434424存活且已真实查询RUNNING；提交记录配置本轮best.pt终态Slurm导出完整有声无标签交互回放、同次物理触键/原谱对比图、自动回传和SHA验收。当前正在初始化/开始训练，未得到最终F1、回放或完整回传；只有规定产物全部验收才写TRANSFER_COMPLETE。自动回传依赖本机与SSH持续可用。


## 2026-10-21 exper027：仅按键手指加深，保持落点并按可达深度限幅

- 用户明确选择仅修改按键手指、抬手不变、t=4秒后开始，且保持水平落点，深度最多10mm并允许按可达结果限幅；原速、先旋转30°再移位的4秒准备段及机械臂均保留。原产物未覆盖，新交付目录为results/exper027/right_finger_deeper_up_to_10mm_20261021。
- 实际参考：right_arm_rotate_first_20261021的CSV/trajectory_and_spline.npz、同次MJB/motion.npz、exper027原始谱proto、原四孔刚性装配及A3.2手指链。重型模型加载/FK/IK/静态碰撞均在volc-a32的pianist_exper027_20261021独立日志目录通过submit.sh执行；Slurm7942/7944/7945/7949/7952全部COMPLETED/0:0。本地仅数组检查、插值修正、打包；未运行本地MuJoCo。
- 固定10mm中点DLS仅20/153次达到0.1mm误差要求，多起点优化仍不能全部满足；球形几何上界未排除任何事件，因此不宣称全局不可达证明。按用户选择，使用保留余量的数值可行深度，并在整段500Hz逐点求解失败时降低相应音符深度。原谱窗内采用两段五次脉冲，中点峰值，重叠同指平滑合成且不超过10mm；这一定义基于原谱时间窗，不冒充真实接触检测。
- 最终153次按键中19次规划10mm，134次限幅，规划峰值1.0988–10mm；500Hz指尖XY最大误差0.000249202mm，深度误差0.000210344mm。前4秒和非按键手指q/v/a逐元素相同；臂CSV逐字节相同，其余NPZ字段逐项相同。手CSV14151行/61列，臂5661行，0–28.3秒，原列名及rad/rad/s/rad/s²格式保留。
- 独立连续插值检查发现仅检查500Hz节点会漏掉最大0.00156398rad越界。logs/right_finger_deeper10_20261021/finalize_derivatives.py仅缩放14个活动关节时刻的速度/加速度，将邻接五次Bernstein控制点约束于关节范围；其余不能直接由包络证明的区间检查导数实根极值。最终连续位置越界0，位置采样和XY结果不变；不把未标定的手指速度/加速度当作硬件验收。
- 153个音符中点的静态接触检查仍存在指尖/琴键几何穿入，原/新全局最大均8.53127mm，部分中点穿入增加；沿用原琴键位置，无物理积分、接触力或新触键/F1验收。因此本次为离线运动学轨迹，不宣称额外真实琴键行程或可直接上机。采样间XY精度、实际接触和手指硬件能力仍为TODO。
- 同目录README说明上述方法、文件格式和限制；附逐音符深度CSV、深度/XY误差PNG、轨迹/指尖验证NPZ、报告、执行来源及SHA。原始远端产物回传核验后完成数组修正，再打包ZIP并验证CRC与逐文件SHA；源代码、作业日志及校验保存在logs/right_finger_deeper10_20261021。未改训练奖励/归档，未重新训练或将原回放F1冒充新轨迹结果。


## 2026-10-22 A3.2：按指定URDF来源修复掌根凸包假接触

- 用户授权定位、替换及修复碰撞网格，并指定TsenseI_hand_A3.2_withid_upper_thumb5mm_noupper_mid目录。左右URDF及全部引用网格SHA与当前保存源完全相同；保留来源、可视几何、动力学和关节合同。
- 实际参考：该目录的双侧URDF/CAD零件、exper028 best.pt同次200Hz录制、冻结归档cc692240…、collision_proxies及urdf_model。原整体base_link凸包跨越指根空隙；几何重建4810/4860采样存在接触。按CAD零件分组并对P0229掌壳进行CoACD分解，加入保守外扩/支撑面简化后仅剩6个真实候选指间接触采样，掌根接触为0。未删除碰撞对或缩小外壳。
- 改动：两侧base_link各79个凸组件，其他42个link不变；同步活动URDF、两份碰撞清单、模型来源、组件加载器和回归测试。新增生成/覆盖验证/录制接触审计脚本、来源清单及截面对照图；旧资产元数据备份于deprecated/model_revisions/whole_base_hull_20261022，旧归档和实验保留。
- 验证：所有重型工作在volc-a32独立目录执行。原CAD每侧564629个顶点/三角形中心无超过1um漏覆盖，全部质量/惯量/关节/驱动/排除不变量一致；CPU7项测试及42子测试通过。Slurm8134的512环境/110物理步验收通过（无容量溢出、qpos/qvel有限），100计时步17.459秒；非训练速度。Slurm8137旧几何同条件100步17.583秒，新几何17.459秒，单次短测无明显退化，不宣称训练加速。固定归档使用logs/collision_mesh_repair_20261022/source_cad_part_collision_v1.tar.gz，SHA与唯一入口强校验同步；冻结父归档仅叠加本任务修复，未混入其他活动算法修改。没有重训，不将原F1或接触重建当作新动力学结果。
- 说明与未解决项：docs/collision_mesh_repair_20261022.md。左拇指—食指约0.805至0.840秒仍有接触，分件检查后仍存在；需要动作/姿态优化，不能用删外壳来掩盖。实际新策略质量及硬件能力尚未验收。

## 2026-10-22 闲指参考抬高与Swing半权重

用户确认调整reset参考姿态，随后要求八指SW闲指贡献减半、末节范围改10°～80°。参考`tsensei_hand_constants.py`、URDF关节链、动作JSON、`reward_config.py`、`rewards.py`；远端静态FK在关节/动作限位内最小化角度改动，八指均+8mm，水平移动<0.84mm，拇指/腕/SW保持。新增显式`swing_scale=0.5`并拒绝缺失参数，当前加权L1替换原均匀L1。同步范围JSON、唯一入口、奖励总表和README。验证证据保存`logs/idle_lift8mm_20261022/`；不启动训练。TODO：静态解不代表硬件标定或学习收敛。


## 2026-10-22：Tx/Ty/Pitch伺服参数搜索（独立侧会话）

- 用户要求寻找kp、阻尼、力限，随后明确不要求保留旧F1、后续重新训练，以跟踪为优先。本任务只新增logs/servo_tuning_20261022和results/servo_tuning_20261022产物，未修改共享驱动、奖励、统一入口、模型或训练归档，未启动新训练。
- 实际参考exper031 best.pt同次24.3s录制的model.mjb/初态/ctrl、冻结config及源归档a6a88b22…、原始注册谱。所有模型加载和200Hz CPU MuJoCo积分在volc-a32独立部署经submit.sh进入Slurm执行；本机仅静态检查、结果统计、绘图和回传。初筛20Hz指目标保持，末轮恢复冻结源的200Hz动态45～80°手指耦合。历史激活阈值只用于旁路诊断，没有读取主会话正在修改的95%激活代码。
- 完成171组单轴候选、19组三轴组合及对应19组未参与单轴筛选的新轨迹（Tx±40mm、Ty±35mm、Pitch0～0.5rad）测试。综合准则为整曲/新轨迹三轴RMSE分别除以同条件基准后取平均，并筛除数值警告/保持段持续运动；F1不作验收条件。测试发现继续增加kp会回升误差，Pitch2048/阻尼2组合在保持段持续振动，被剔除。
- 已测最优pitch_k1024_d2：双侧Tx kp38400N/m、阻尼126N·s/m、力限±200N；Ty kp38400、阻尼250、力限±200N；Pitch kp1024N·m/rad、阻尼2N·m·s/rad、力限±20N·m。整曲RMSE从13.803mm/18.234mm/2.281°降至1.876mm/2.426mm/0.439°；新轨迹为0.707mm/0.519mm/0.228°。完整峰值/饱和/接触/硬位置投影/残余运动记录见report.json。
- 首次8436因历史录制字段key_threshold而失败；只在独立分析读取器显式支持该冻结格式。8443/8454/8467/8470/8484/8487/8495均COMPLETED/0:0。远端归档SHA与原实验一致，输入/脚本和回传清单保留；候选JSON、README、RMSE与误差时序PNG、全量原始结果和Slurm日志均保存。
- 局限/TODO：本结果是有限范围内的仿真候选，不是全局最优、实机标定或新策略成绩。实际瞬时速度/加速度不受目标限速保证，最优候选整曲Pitch速度峰值10.57rad/s；手部接触力采样峰值由166.1N增至2590.4N、最大穿入由8.23增至17.12mm，不能称物理安全验收。现有生产驱动共享Tx/Ty并计算临界阻尼；应用候选需后续明确逐轴实现、验证和新归档，candidate_parameters.json不是现有CLI直接可读配置。

本轮验收：远端CPU奖励35项、相关回归39项及18子检查通过；独立Composer重建1项通过，八指实际FK均+8mm；Slurm8510物理/奖励40项通过、2项因冻结归档无外层入口而跳过，包含physics graph开/关及局部reset。静态参考不约束运行时末节角；最终下发仍由10°～80°耦合决定。未启动训练。

## 2026-10-22：同步跟随筛选腕驱动增益到统一入口

- 用户要求将已筛选Tx/Ty/Pitch增益用于后续训练。参考 `results/servo_tuning_20261022/candidate_parameters.json`、`report.json` 和最新闲指8mm冻结源码；不恢复旧实验参数。
- `drive_settings.json` 改为平移逐轴Kp/阻尼/力限：Tx=38400/126/200，Ty=38400/250/200，Pitch=1024/2/20（SI单位）；`drive_settings.py` 严格校验逐轴配置，手模型构建保持M0=.972kg校验并应用显式阻尼。手指驱动、目标运动限幅、奖励及预算不变。
- 更新驱动契约/编译模型测试、归档中的物理步进集成测试；独立远端部署通过Slurm验收，具体作业、结果与归档SHA见 `logs/servo_release_20261022/validation.json`。首次作业8527因部署未携带CUDA路径在测试前失败，后补齐独立目录环境。
- 统一入口绑定本次新归档和SHA；同步README、奖励唯一总表依赖说明、AGENTS及 `docs/servo_gains_20261022.md`。保留父包的新闲指姿态，不更改已提交作业，不提交新训练。
- 未解决：参数以仿真跟随精度为目标，接触力与穿透峰值增大；没有硬件标定或长期新训练稳定性结论。

增益同步验收结果：Slurm 8532 COMPLETED（0:0），14项通过（含子进程5项Composer编译检查），仅35条TorchScript弃用提示；最终归档SHA256 `51a82184cefa5f897e678d555bf8fb01ffd98f0db623cbb822e633bfc153e352` 已回传校验并绑定统一入口。8529为学习器环境缺少dm_control导致的收集失败，后改用独立Composer环境；未提交训练。

## 2026-10-22：默认训练安装高度与Yaw

- 用户要求将yaw=15°、height=0.09m直接写入默认训练脚本。参考并修改 `all_use_this_train.sh` 的默认CLI参数，yaw以弧度0.2617993877991494传入；ty=0.04m及其余参数保持原值。同步README当前默认说明。
- 本次仅修改入口参数，沿用已验收的servo_gains源码归档及SHA，无运行时代码或奖励变更；已提交作业不受影响，未启动训练。
- 验证：`bash -n all_use_this_train.sh`通过，静态核对默认height/yaw及弧度换算。无新增TODO。

## 2026-10-22：exper032 使用015最新修复指法从头训练

- 用户要求开启015训练，使用 `results/exper031/fingering_revision_v2_20261022/015-5_fingered_repaired_v2.proto`。269音符、7处指法修复，原音高/时间不改，SHA256 b0ce7e52c0eb211e009c463b168eb0cf6f7f14cac66fcadf87fbe7d790a8222a；验收报告随本轮input_sources保存。
- 唯一入口 `all_use_this_train.sh`，冻结servo_gains_v1源码SHA 51a82184cefa5f897e678d555bf8fb01ffd98f0db623cbb822e633bfc153e352；yaw15度、height0.09m，双卡512环境、batch1024、19532向量步（10000384经验）、学习率0.0003，其他参数沿用当前入口。当前奖励预期完整JSON保存在本轮reward_config_expected.json，实际完整配置由训练启动打印并冻结。
- 独立部署 `/home/houj/projects/pianist_exper032_20261022`，校验源包/指法SHA与导出依赖后经统一入口提交Slurm8724，A100双卡；当时PENDING/Priority。H100仅余一张卡，不改变双卡分工。未运行本机物理仿真或GPU工作。
- 自动回传PID1151111，首次真实查询PENDING已确认；训练终态后导出best.pt的完整有声交互回放、同次实际触键/原谱对比图并校验回传。状态记录见results/exper032/submission.json、sync_status.json及transfer.log。尚未开始采样，无本轮F1；排队期间不声称已在训练或已有回放产物。

## 2026-10-22：按用户要求暂停exper032

- 用户要求先暂停实验。查询Slurm8724已RUNNING；`scontrol suspend`被调度权限拒绝，随后对该作业发送`scancel --signal=STOP --full 8724`。核验队列STOPPED，训练Python及批处理进程为T状态。未取消作业、未删检查点。
- 保留GPU分配和内存状态；作业墙钟时限继续计时。暂停记录及恢复命令见 `results/exper032/pause_status.json`。后台回传继续查询状态，不主动恢复训练。

## 2026-10-22：按用户指定目录复核exper032模型来源

- 用户认为上版有未提供内容，要求采用指定A3.2目录；随后授权若无差异可训练。核对736个引用源文件SHA相同，并在远端对实际Slurm8724冻结源码逐项验证；44个link可视三角形重新组合均相同，2个优先加载OBJ也一致，无额外可视部件，惯量/关节树一致。
- 差异仅已知训练适配：双侧拇指第二轴反向及正角范围、200个凸碰撞组件近似。由于存在适配差异，保持STOPPED并向用户确认是否保留后恢复；未改源目录或训练模型。证据与复现见logs/urdf_source_audit_20261022/。

## 2026-10-22：确认适配后恢复exper032

- 用户明确确认保留现有拇指轴向/正角范围及凸碰撞组件适配并恢复训练。向原作业8724发送CONT，核验Slurm RUNNING、训练进程退出T状态；保留原运行、模型、指法和学习预算，无重复提交。
- 自动回传进程1151111存活，原best.pt导出/同次触键对比图/校验回传流程保留。恢复记录见results/exper032/pause_status.json。

## 2026-10-22：当前训练模型可视/碰撞外壳图片对照

- 用户要求真实模型黑色、碰撞外壳浅色导出图片。使用exper032完整config的source_runtime和匹配servo_gains_v1冻结源码，在远端Slurm8755独立单卡构建并静态渲染；未运行策略或积分，未改变训练8724。
- 44个黑色可视geom、200个浅蓝碰撞geom；导出右俯视、左右斜视、右掌近景四张3000×1040 PNG，各三栏保持同姿态/相机。只调整导出副本显示颜色/透明度并隐藏琴键背景。
- 渲染测试1 passed；四张PNG解码及SHA回传校验通过，人工查看右手斜视/右掌近景。右掌矩形开口可见被凸碰撞近似填充。产物results/collision_overlay_20261022/，复现与作业记录logs/collision_overlay_20261022/。静态图片不提供新训练F1或接触力结论。

## 2026-10-22：exper031与exper032训练F1曲线导出

- 用户要求导出当前F1曲线；读取两轮results中的metrics.jsonl，每轮41个eval.f1原始点，以累计经验为横轴绘制，无平滑或重跑策略。
- 产物保存于results/f1_exper031_exper032_20261022/：PNG、SVG、82点CSV、来源SHA256及复现plot.py。图中标注20Hz/3回合及激活阈值不同，不将曲线差异作为单因素因果结论。
- 本地仅CPU离线绘图；验证每轮41点评估、最佳值031=0.8413/032=0.6770，检查PNG可视布局。训练配置和冻结归档未修改。

## 2026-10-22：用户授权磁盘清理

- 按用户要求删除 results/exper001 至 exper020（精确20个目录，约3.20GiB）；保留 exper001_continue_20261012 和 exper021 及之后实验。
- 清理 pip 下载缓存及项目 .pytest_cache；依据 git count-objects 的 garbage 报告，删除 110 个超过一小时且未被进程打开的 tmp_pack_ 临时文件，释放 15.98GiB；未删除正式Git对象、LFS、训练日志或源码归档。
- 验证指定20个目录已不存在，并检查磁盘可用空间。

## 2026-10-22：恢复exper032回放验收与自动回传

- 用户反馈未回传。核查submission/sync_status/transfer.log及远端Slurm8840：训练8724和导出8840均COMPLETED/0:0，训练文件此前已校验回传；远端旧finalize_training_video.py仍要求finger_labels，违背无悬浮标签要求，后台持续重试。
- 复用当前活动交付工具，同步finalize_training_video.py、package_interactive_replay.py、interactive_recorded_replay.py、launch_remote_recorded_replay.py、plot_recorded_midi_comparison.py及缺失launch_local_recorded_replay.sh至本实验远端独立部署；保留旧finalizer诊断副本。源码归档/PT/config及录制未修改，不重跑训练/策略。
- 原后台锁进程继续执行；远端音视频解码、冻结来源、同次原谱对比及交互数据验收通过，200Hz单回合micro-F1=0.7745311415。工具散列与恢复说明见results/exper032/delivery_recovery_20261022.json，最终回传状态以sync_status.json及TRANSFER_COMPLETE为准。
- 最终结果：2026-10-22 17:44（Asia/Singapore）原watcher完成525个文件SHA校验，sync_status=completed、export_state=validated，TRANSFER_COMPLETE已创建；完整回放约1.6GB已落地exper032/video_best。

## 2026-10-22：清理活动远端交付工具的悬浮标签旧功能

- 用户要求删除错误功能/逻辑，后续明确不保留专门针对旧标签参数的报错分支。核对当前本地及远端exper032、servo_release部署：旧远端watcher仍传--finger-labels，导出器及渲染器仍有开关分支，本地也有静默吞旧参数的兼容入口。
- 删除活动CLI标签参数及兼容别名，清除远端旧渲染分支、调用参数、输出计划/报告中无作用的finger_labels字段和相关测试断言；未保留针对旧标签的专门检测/报错。验收只核对必需的声音、指法颜色和琴键高亮。同步README。
- 两个当前部署同步11个交付工具并逐文件SHA核验，活动scripts/preview中的标签参数关键词检索零命中。旧部署工具备份隔离到各自deprecated/delivery_tools_before_cleanup_20261022；历史实验冻结归档、PT、录制及依赖未删除或改写。训练算法和冻结训练SHA未变，交付工具独立同步，不宣称历史归档已清理。
- 验证：远端仅CPU交付/CLI/原谱来源测试；执行及散列记录logs/delivery_cleanup_20261022/。本次范围是当前两个部署的交付链路，不代表已证明远端所有历史代码不存在其他缺陷。

## 2026-10-22：按用户指定exper027回放启动脚本同步导出模板

- 参考用户明确指定results/exper027/right_arm_prep_smooth_20261022/replay/launch_local.sh，将其逐字用于scripts/launch_local_recorded_replay.sh。后续package_interactive_replay生成的启动脚本采用同目录model.mjb/viewer.py、本地NVIDIA/OpenGL、至少4GiB可用内存及3GiB无swap独立scope。
- 核对参考viewer.py依赖exper027专属motion.npz和固定28.3秒轨迹，未将该数据读取器冒充通用训练回放；各轮仍使用本轮录制及匹配通用播放器。现有导出包和散列清单未改写，未重跑训练/仿真。
- 同步远端exper032与servo_release当前部署；bash语法检查通过，临时目录生成的4个启动文件逐字等于参考且可执行，本地及两远端SHA均a818eec52a94a08a47d839ad5be711e745e5cec80953b2eab810cb829566c773。记录logs/replay_launcher_sync_20261022/，README同步。

## 2026-10-22：更新exper032既有回放启动方式并本地验收

- 按用户要求将exper032/video_best/interactive_replay四个启动脚本更新为指定exper027启动模板，远端同一导出包同步；更新export manifest与本地transfer manifest，5个变更文件本地/远端SHA一致。
- 本机可用内存约6.8GiB，MuJoCo 3.11.0与录制匹配；经launch_local.sh独立systemd scope进行已录制MJB加载/渲染与26秒播放检查，无XML编译/策略/物理积分。实际渲染NVIDIA RTX 3050 Laptop，完整24.3秒轨迹、声音进程正常；1539帧/26.01秒，memory.events的high/max/oom/oom_kill均0。日志及结构化结果在exper032/local_*check.log及local_playback_validation.json。
- 未改变检查点、实际轨迹、原谱对比图或回放分数；声卡主观听感未独立测量，已验证ffplay播放进程无异常退出。

## 2026-10-22：exper031/032 F1下降离线定位

- 用户要求定位原因。复用两轮best同次回放audio_substeps/rollout、原谱comparison与冻结config，轻量CPU重评分及端点跟随分析，无模型加载或物理仿真。
- 统一95%后200Hz单回合micro-F1为0.903167/0.774531，评分口径仅使031该轨迹下降0.010200；FN1891→3508、FP249→1361。262个未改指法音符占1571/1617新增漏样本，双拇指约43%；15.495s和弦68/71出现明显按压不足。分离阈值训练效应与纯重评分效应，不声称因果归因已完成。
- 由保存XML关节名映射ctrl/qpos，032腕跟随误差明显减少；手指目标峰值速度/加速度没有增加。目标运动罚始终−5，但手指单独触发占比反而19.2%→13.0%，不能直接把满罚归咎手指。驱动、姿态范围、参考和指法变化仍混杂，未擅自启动消融训练。
- 结果/来源SHA/复现脚本/逐音CSV/图在results/f1_diagnosis_031_032_20261022/，conclusions.txt说明口径与局限；原始实验数据及训练配置未修改。

## 2026-10-22：exper033腕部驱动单因素对照训练

- 用户明确要求不修改当前训练脚本，以exper032为基准只恢复exper031腕部驱动，预算不变，结束自动导出回放并回传。all_use_this_train.sh哈希前后未变，且与exper032保存副本逐字一致。
- 从exper032冻结归档51a82184…e352派生专用归档，仅修改assets/drive_settings.json和对应bundle清单。Tx/Ty Kp300、临界阻尼2sqrt(300×.972)=34.15259872981849、力限±50N；Pitch Kp8/阻尼.5/力矩±5Nm。保留032逐轴配置结构及编译惯量检查，手指驱动、琴键、模型、奖励、指法v2、范围、学习算法均保持032，不将当前工作区其他修改带入。
- 专用归档logs/wrist_drive_control_20261022/source_exper033_wrist031_v1.tar.gz，SHA714a9b5c236a5f3af589adf019ceec1d7ca98ff32df952a99045a9ef6528d1be，已回传本地并校验。统一入口追加--source-archive/--source-sha256及原谱/输出路径，无新训练脚本、无修改默认归档。
- 独立远端部署/home/houj/projects/pianist_exper033_20261022，仅复用依赖环境。Slurm8938编译验收COMPLETED/0:0：6腕轴参数、40手指执行器及Pitch0～.5不变。训练8939在双A100运行，512环境、19532向量步/10000384经验、每经验.25更新；原脚本完整奖励启动打印已确认。
- 后台watch_training_transfer.py PID1095085独立运行，日志首次真实RUNNING查询13:27:11 UTC。submission配置best.pt、完整有声交互回放/高亮/同次原谱对比及SHA验收，使用已清理标签的交付工具和用户指定本地启动模板；完成前不创建TRANSFER_COMPLETE。配置/来源/日志存results/exper033/，训练及导出最终结果待完成。
- 启动完整config逐字段审计通过：仅3个实验路径、drive_settings文件SHA和派生scene/task SHA不同；奖励、动作、耦合、控制限位、原谱、全部学习CLI值相同。已保存config_comparison.json并同步远端deployment；训练日志已产生metrics.jsonl，统一脚本最终SHA复核未改动。

## 2026-10-22：钢琴1224mm分组黑键布局，保留其他物理参数

- 用户要求总宽1224mm、白键22.5mm、黑键12mm，沿用已确认的理想五/七等分布局；两黑键中心距28.270588mm，三黑键相邻中心距26.924370mm，跨组41.396218mm；白键间隙由总宽确定为1.058824mm。不冒充DIN/PX-S1100实测，保留矩形白键形状。
- 修改piano_constants.py/piano_mjcf.py横向布局及高亮尺寸，显式保留原黑键/固定基座质量和惯量。完整场景对照发现琴宽自动扩大腕范围，遂修改suite/tasks/base.py固定原0.6105m半工作区，避免关节/执行器范围及动作归一化改变；其余手模型、驱动、接触、键程、95%激活和奖励源码/JSON均未改动。
- 参考本轮原冻结归档51a82184cefa5f897e678d555bf8fb01ffd98f0db623cbb822e633bfc153e352和此前logs/piano_layout_1224_20261022/layout.json。缺失本地归档由远端同SHA副本恢复。新归档只叠加本次变更，统一入口仅换路径/SHA，所有训练参数逐字保持，不影响既有实验。
- volc-a32独立目录pianist_piano_layout_20261022远端CPU验收4项通过：88键尺寸/顺序/所有两三组间距；有/无执行器200步同力矩轨迹；完整双手物理数组和20步有限值检查。实际source_runtime重建后，在MuJoCo3.11.0运行时加载无臂场景成功（139body/334geom/134joint/46actuator，200Hz，95%门槛）。本次未启动训练或策略推理，未使用GPU。
- tests/test_piano_layout_1224.py及docs/piano_layout_1224_20261022.md新增，README同步；证据/坐标图/来源和SHA位于logs/piano_layout_model_20261022/。保留首次发现腕范围连带变化的失败日志及修复后通过日志。TODO：没有完整演奏/学习性能或硬件尺寸标定，旧checkpoint不迁移至新几何。

## 2026-10-22：本机直接查看新1224mm钢琴

- 用户明确要求本机直接打开新钢琴且不训练。新增根入口launch_piano_preview.sh与scripts/view_static_piano.py；采用远端已验证新归档20eb802e…7d6cc构建的静态MJB/初始姿态，本机加载MuJoCo3.11.0二进制，只更新坐标与相机，不执行XML编译、mj_step、策略或训练。默认显示钢琴，H切换双手；1/2/3总览/顶视/近景，鼠标旋转平移缩放。
- 参考scripts/launch_local_recorded_replay.sh的本机保护及scripts/interactive_recorded_replay.py的相机交互；保留独立systemd scope的MemoryHigh=2300M、MemoryMax=3G、MemorySwapMax=0，至少4GiB可用内存。加载前核验MuJoCo版本、MJB/姿态SHA及三份布局源码指纹，缺失/过期模型明确拒绝。
- 模型在volc-a32:/home/houj/projects/pianist_piano_layout_20261022/static_preview导出，全部88键坐标/宽度逐项匹配前轮编译验证。回传preview/outputs/piano_layout_1224并校验SHA；远端导出脚本与退出日志保存在logs/piano_layout_model_20261022/。
- 本机实际NVIDIA RTX3050 OpenGL窗口验收通过；修正默认视角为演奏者方向，4.04秒179帧覆盖总览/顶视/近景及双手显示，qpos保持、仿真时间0，内存峰值1434828800字节（约1.34GiB），high/max/oom/oom_kill均0。截图、日志、last_session.json位于预览包同目录。随后后台打开常驻窗口，README增加启动命令。
- 未修改训练入口、冻结源码归档、任何物理/奖励参数、现有实验或回放。此入口是静态模型查看，不是策略演奏或可按键物理模拟；预览模型文件为本地生成产物，新机器需要携带该目录。用户未要求演奏视频，本次未导出演奏视频或伪造MIDI对比。

## 2026-10-22 琴键实际激活改为100%

用户要求白/黑键激活均100%。参考key_activation.py、Composer琴键、Warp子步、音频/高亮与奖励总表；激活常数设1.0，保留正确键连续奖励容差0.05及95%满分。同步配置合同activation100并拒绝旧95%配置；子步错键/F1/音频/高亮共享新常数，对比图读同次录制阈值。修订边界、相邻浮点、音频及物理回归测试；物理测试的腕增益断言同步已发布父版本38400/126/250/±200，不改驱动参数。以当前1224mm布局归档20eb802e为父，仅覆盖本次文件，保留历史冻结归档。验证结果与SHA记录在logs/key_activation100_20261022/validation.json；未启动训练。

## 2026-10-22：导出新钢琴独立代码包

- 用户要求导出新钢琴代码包。新增exports/piano_1224_code_20261022.zip（229115字节，SHA256 1c852efe7b989a822bd68cf36df28fc2417f47b21e7410ec7aa17c78638e27d9）及解压目录、校验文件，README增加链接。
- 以新布局已验证归档20eb802e…7d6cc对应的piano_constants.py/piano_mjcf.py/suite/tasks/base.py为来源，三份原文件逐字保留；独立生成器仅改模块导入/类型注解，移除对原工程及mujoco-utils的依赖。按前轮编译scene_manifest提取实际20/200Hz/Euler/接触配置，附MJCF XML、88键坐标JSON、PNG/SVG标注图、许可证、依赖版本和来源清单。模型为box，不需要网格文件；不含训练、手模型或奖励/激活分类器。
- 远端CPU独立导出，与冻结原生成器的被动/自驱两种模型XML逐字相同，26项编译物理/几何数组逐项完全一致。清空PYTHONPATH后在独立解压目录重新生成，XML与包内文件逐字一致；MuJoCo3.11.0加载验证通过，全部键坐标/宽度/分组间距和200Hz/Euler正确。ZIP及包内文件SHA验证通过，未启动训练或GPU任务。
- 证据位于包内validation及logs/piano_layout_model_20261022/code_export_unzip*；本任务仅导出，没有修改活动模型、冻结训练入口或物理/奖励参数。项目历史说明中的任务级门槛不作为独立包接口，README已明确；独立模型接入后的任务行为由宿主项目负责。

## 2026-10-23：检查exper034并恢复后台回传

- 核对results/exper034/submission.json、sync_status.json、transfer.log及scripts/watch_training_transfer.py；远端volc-a32作业9007仍RUNNING，约01:01时14500/19532步，最佳20Hz评估F1=0.804347（14000步）。
- 本地原PID959502已退出且回传锁空闲，无TRANSFER_COMPLETE；依据原submission恢复脱离终端的后台监控PID26802，沿用60秒查询及原best.pt自动导出/回传要求。实际首次查询RUNNING并刷新状态，未改训练源码、预算或作业。
- TODO：训练、导出和完整回传仍待完成；后台进程退出原因未确定，日志有此前网络中断重试记录。

## 2026-10-23：彻底移除交互回放悬浮标签并同步远端

- 用户截图定位到exper033已导出的旧preview_fingering.py；旧打包器复制部署中的训练辅助模块，先前无标签修改及验收未覆盖该包。参照当前preview_fingering.py琴键着色和用户指定launch_local.sh，新增独立scripts/replay_key_highlight.py，打包器仅安装此显示模块，删除回放指尖文字/目标标签/悬浮球生成途径。
- scripts/interactive_recorded_replay.py增加全曲显示采样无文字/仅琴键盒验收和本地渲染截图。同步本地11包、远端8包和21个部署；保留各包激活代码、MJB、录制和音频，历史冻结训练来源不改写。README及docs/replay_display_cleanup_20261023.md同步。
- 远端test_replay_key_highlight.py两项通过；exper033远端加载和本机受保护NVIDIA OpenGL渲染检查通过，487显示采样标签/悬浮球均0，声音数据有效。更新回放交付清单；其余历史包仅静态检查，不冒称逐包播放验收。
- 默认源码以06f469d7父归档叠加本次显示脚本/测试/说明重新冻结；训练运行时文件完全不变，统一入口仅换归档及SHA，既有exper034训练不变，后续导出使用已修复打包器。证据logs/replay_labels_removed_20261023/，未重新训练或重新生成物理轨迹。

## 2026-10-23：exper031–034同预算指标核对

- 按用户要求比较14500向量步/7424000经验；参考各轮冻结config.json、metrics.jsonl及docs/exper034_default_wrist031_20261022.md。034远端日志/配置/best只读快照与comparison.json保存于results/exper034/status_comparison_20261023，报告记录日志SHA及三回合子步TP/FP/FN聚合micro-F1。
- 20Hz F1依次为.805041/.645494/.805629/.802270；200Hz聚合micro-F1依次为.901354/.752147/.902559/.886356。各轮实际激活合同不同，031指法/范围亦不同，不作为单因素因果比较；034相对033包含1224mm布局、任务点回退3mm和100%激活更新，腕驱动数值相同。未更改训练/奖励/冻结包。TODO：034尚未完成，需最终回放验证。

## 2026-10-23：按用户尺寸绘制黑键梯形示意

- 根据用户提供的参考图绘制黑键平面形状：中心线保持不变，顶面 10×90 mm、底面 12×95 mm，长度方向上下居中对齐，顶/底端各外伸 2.5 mm。
- 产物：`logs/black_key_shape_20261023/black_key_shape.svg`。本次仅生成尺寸示意图，未修改活动钢琴模型、训练配置或碰撞几何。
- 按用户后续要求将示意中的黑键整体倾斜约 18°，中心线与键体同步倾斜；尺寸数值不变。
- 根据用户反馈补充黑键立体透视表达：新增顶面、右侧面、前端面和高光边线；未指定的厚度仅作示意，不作为模型尺寸。

## 2026-10-23：黑键改为后端齐平的10×90/12×95mm梯形实体并标注钢琴原点

- 用户要求保持黑键中心位置与高度，上表面10×90mm、底面12×95mm，上下朝琴顶后端对齐。参考活动piano_constants.py/piano_mjcf.py、此前形状示意、1224mm布局、当前cc965e66父归档及静态预览。36个黑键改为共用8顶点/12三角面的真实凸mesh，碰撞/外观一致；18mm高度、原body/铰链/横向中心位置保持，后端X=−75mm齐平，前端上+15/下+20mm。底面中心与体积形心自然偏移，未谎称所有几何中心都能同时固定。
- MuJoCo会将mesh重定位/旋转到主轴；fingering_targets.py、Composer、task_bundle/task/substep采样和相关离线工具改读原中心reference site，保持指法公式和目标数值，更新几何manifest并拒绝旧几何合同。名义键程力臂、质量/显式惯量、关节与驱动参数不变；诊断读site尺寸，可选平面保护使用保守底面外包范围。黑键激活/指法着色直接着色mesh，避免盒覆盖斜面。奖励总表原行、布局说明、README及AGENTS同步，无新增奖励模式/参数。
- 静态预览入口改读独立新MJB目录；scripts/view_static_piano.py增加O原点/XYZ轴开关，按真实三维位置投影穿透显示原点(0,0,0)，不移动坐标系。轴为+X朝演奏者、+Y高音、+Z向上；原点位于白键底面平面、前后/横向中心。原预览和历史实验保留。尺寸/坐标图按远端编译顶点校验生成，保存PNG/SVG。
- 远端独立目录volc-a32:/home/houj/projects/pianist_black_key_20261023，仅复用用户依赖环境；parent SHA核验cc965e669d495bc59f5166411927c381c5330542e41890f342e8cf64170d0849。CPU模型/表面接触/原中心与参数对照/Composer子步9 passed（3项可选历史基线未启用）；CPU Warp逐子步reference采样及现有回归9 passed/1 CUDA skipped；实际mesh高亮1 passed；重建完整scene/task当前目标合同1 passed。合计20 passed，未运行CUDA验收/训练。首轮误选到依赖外部历史scene的legacy硬限测试因fixture缺失失败，保留诊断，随后仅执行本任务相关高亮节点通过。
- 静态模型经learner MuJoCo3.11.0重新构建并导出，兼容本机版本（Composer自身3.13.0的诊断MJB仅保留远端，不用于交付）。本机受保护NVIDIA静态查看确认qpos保持、仿真时间0，约1.34GiB峰值，high/max/oom均0；3GiB上限与4GiB可用检查保持。新训练归档由当前父包仅叠加本次文件构建并审计，统一入口同步新路径/SHA；没有启动训练、修改历史checkpoint或声称新演奏F1。证据、命令、来源和散列位于logs/black_key_model_20261023。TODO：未做新形状学习性能、GPU物理吞吐、完整演奏或硬件标定。

## 2026-10-23：打包当前梯形黑键钢琴并说明查看方式

- 用户要求查看并打包当前模型。参考launch_piano_preview.sh、scripts/view_static_piano.py、当前piano_constants.py/piano_mjcf.py、黑键形状/布局说明，以及preview/outputs/piano_black_key_20261023/preview.json。当前本机查看仍使用现有受保护的静态MJB入口；没有启动本机编译或物理仿真。
- 新包exports/piano_1224_black_key_20261023.zip（395265字节，SHA256 deda091dbfaf55ea62214ea73d567632c2671ed589027030bec73327e0652317），包括独立钢琴XML/MJB、当前源码及独立生成器、尺寸数据/图、来源和校验。旧20261022包保留。README当前下载链接改为新包；训练归档、奖励与模型源码未修改。
- 远端volc-a32 / TenseiNode1独立目录/home/houj/projects/piano_model_export_20261023_v1，使用明确依赖环境，CPU生成/编译和20步验证退出0：88键、总宽1224mm、逐键坐标/reference site宽度、36黑键全部8顶点与当前完整场景记录一致，状态有限、警告0、MJB保存加载一致。命令、退出和报告在包内validation；ZIP CRC及所有文件SHA复核通过。
- 包内MJB仅钢琴本体，版本3.11.0；本机原查看入口包含可选双手。未启动训练或策略回放；TODO：未做其他MuJoCo版本或其他操作系统的加载验收，包内不含音频/训练任务。

## 2026-10-23：提交当前项目到Git远程

- 用户要求提交当前项目。核对master、origin及工作区改动，沿用Git LFS与单线程/128MiB打包设置。参考all_use_this_train.sh、.gitignore及.gitattributes，补充当前黑键冻结源码、SHA、来源/构建与验收记录的跟踪规则；补齐已隔离旧归档的LFS规则，保留已有模型/代码/文档及导出包变更。
- 提交前检查差异空白、当前训练归档SHA与大文件LFS状态；本任务只做版本保存与推送，没有启动训练、物理仿真或GPU验证，已有验收结论以各专题记录为准。
- 用户后续明确deprecated不提交远程：停止首次推送，将deprecated/从Git索引移除并加入忽略规则，文件完整保留本机；既有Git历史不改写。当前活动训练归档继续通过LFS提交。
- GitHub首次完整推送被拒：未推送历史含186MiB非LFS历史归档。建立仅本地备份分支codex/local-backup-before-publish-20261023保留原提交，将未发布改动合并成当前快照再推送，避免deprecated进入新增远程历史；不强推、不改远程已有历史。
