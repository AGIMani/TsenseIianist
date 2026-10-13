# 指定PT、多视频、多视角统一导出

## 2026-10-10：expe00002 best视频与训练中断

腕范围补充：[同次20Hz腕部范围图](../logs/expe00002_result_analysis_20261010/wrist_range.png)及[数值与测量口径](../logs/expe00002_result_analysis_20261010/wrist_range.json)。TX/TY投影仍在请求区间，但实际高度平面／固定姿态偏离；排除规定准备抬高后，全程离面左25.53mm、右19.84mm，准备结束后左4.52mm、右12.12mm。原腕请求限位不等同实际FK工作空间硬约束；本轮仅分析，尚未修复。

后续离线分析：[F1／回报／惩罚曲线](../logs/expe00002_result_analysis_20261010/training_analysis.png)、[CPU端点接触定位](../logs/expe00002_result_analysis_20261010/analysis.json)。原视频183帧均存在相同12对同手接触，对应GPU录像中恒定−90/步；两项运动罚均每步−5。CPU20Hz接触不冒充200Hz完整GPU接触明细，根因与改法仍需受控验证；未改训练。

[正面双手／双机械臂有声视频](../preview/outputs/expe00002_best_20261010/full_scene/replay.mp4) · [同次实际按键／原始MIDI对比PNG](../preview/outputs/expe00002_best_20261010/midi_comparison/midi_comparison.png) · [查看页](../preview/outputs/expe00002_best_20261010/index.html) · [完整ZIP](../preview/outputs/expe00002_best_20261010/export_bundle.zip)。

用户授权导出best后中断：冻结第5000步（256万样本）best，Slurm2120用原刚性臂schema3源码录制，2121只重绘同次原生状态、完整显示双M6。1280×900/20fps/10.2秒，原速含1秒准备、9.15秒物理及1秒末帧停留；音轨、图、事件和数组同源1830个200Hz子步。训练三回合汇总原谱F1=1.13%，本视频单回合音乐micro F1=0%（TP0/FP130/FN2615）；这是通过接触筛选的best，不是未留档的历史9.00%模型。

源码/模型/PT/config合同、原谱时序/同次键阈值/音频一致、H264/AAC全片解码、manifest及ZIP校验通过；展示派生包只改取景、标题和零mocap兼容。导出验收后18:58中断2109，CANCELLED已确认，最后日志第6000步／307.2万样本。完整来源与停止证据在logs/expe00002_best_export_20261010/；原实验、算法、奖励和历史归档保持。

## 2026-10-10：刚性装配修复后的正面测试视频

[两组测试合并视频](../preview/outputs/rigid_arm_test_20261010/tests_combined.mp4) · [查看页](../preview/outputs/rigid_arm_test_20261010/index.html) · [同次实际按键／原MIDI图](../preview/outputs/rigid_arm_test_20261010/midi_comparison.png) · [运动曲线](../preview/outputs/rigid_arm_test_20261010/motion_overview.png) · [完整ZIP](../preview/outputs/rigid_arm_test_20261010/export_bundle.zip)。

按用户最新要求采用演奏者前方正面视角（front、azimuth=0、elevation=−28°），只显示左手／左臂，右下角为同帧四孔连接件近景。右侧仅隐藏，仍参与物理；声音与按键图记录完整88键，包含右手触键。物理左移为演奏者左侧，正面画面的左右相应镜像。

Slurm2103单A100按固定1c8b3147…265dea7f源码、当前统一配置和用户双侧初始角，重新记录两环境各13秒／2600个200Hz子步，无策略、无checkpoint、无学习器；使用scene.step执行预设指令，评分任务不推进、不伪造演奏奖励，臂运动罚仍按原限幅前需求计算。0–1秒准备，1–4秒三次方加速左移42cm：A随后保持，B于4秒请求48cm、6秒返回。停止／返回后分别0.355／0.945秒达到持续停稳标准；全部SDK v/a与URDF位置复核通过，固定安装残差峰值3.1229e−8m。保留的手指主动指令造成拖键及重复触键，不能把臂停稳解读为指键接触稳定或演奏成功。

Slurm2105仅读同次qpos／mocap重新取景，未重跑物理／IK；此前等待中的演奏者视角作业2104按用户纠正取消，没有执行。1280×900／20fps／原速，每段14.05秒（13秒物理＋1.05秒末帧静止），合并28.1秒，第二段起点14.05秒。每段独立0–13秒曲线、J1–J7速度／加速度PNG与200Hz CSV，含请求误差与安装残差独立标示。

原注册《小星星》34事件及原始MIDI与固定包逐字核对，参考仅移除注册前导静音并加1秒准备；图与声音均从同次真实琴键阈值重建。参考音乐micro F1为12.6181%／13.8665%，只描述两段预设测试与原谱的差异，不是训练成绩，也不作为仅左手F1。完整来源／数组／事件／配置／散列保存。独立NumPy URDF FK、200Hz关节差分、全部视频H264/AAC解码、曲线来源和ZIP CRC验收通过；已查看正面4秒、返回6.5秒画面及按键图。记录在logs/rigid_arm_video_20261010/。训练源码、默认配置、历史视频保持，未启动训练。

## 2026-10-10：左腕逐渐加速，机械臂跟随测试

[演奏者视角视频](../preview/outputs/left_wrist_ramp_20261010/performer_view/replay.mp4) · [速度／误差／罚分图](../preview/outputs/left_wrist_ramp_20261010/motion_overview.png) · [左臂J1–J7速度／加速度](../preview/outputs/left_wrist_ramp_20261010/left_arm_joints.png) · [同次实际按键／原MIDI图](../preview/outputs/left_wrist_ramp_20261010/midi_comparison/midi_comparison.png) · [页面与完整ZIP](../preview/outputs/left_wrist_ramp_20261010/index.html)。

这是一条预设腕动作，使用当前训练环境与用户指定初始七角，未加载策略／checkpoint或启动学习。0–1秒保留准备段，1–4秒向演奏者左侧低音方向加速请求42cm，随后保持；其余主动指令取reset值，手指耦合继续生效。实际最终移动42cm，速度峰值0.367m/s。加速段跟踪很近，停止后出现持续振荡：最大臂腕间隙119.36mm、最后46.10mm；下发关节SDK速度/加速度硬限仍通过。该现象如实保留，未在导出任务中更改算法。

Slurm2085同次200Hz物理记录，2091仅读qpos/mocap换成演奏者视角，原速20fps/1280×900/10.2秒，蓝点真实腕、橙点臂FK；右侧只在渲染隐藏。完整数据及双侧/左臂独立罚分可复核。未触键，音轨静音；对比图参考F1=0%不代表训练成绩。配置、源码、独立FK、同次按键/MIDI、视频解码/音轨/ZIP均通过，证据logs/left_wrist_ramp_20261010/。无需给该受控测试伪造检查点。

## 2026-10-10：expe00001中断后的最新双机械臂视频

Slurm2075在第500步原子forward.pt保存后按用户要求取消，保留256,000样本／40,576更新。[最终有声视频](../preview/outputs/expe00001_step500_20261010_v2/full_scene/replay.mp4)、[同次实际按键与原始MIDI图](../preview/outputs/expe00001_step500_20261010_v2/midi_comparison/midi_comparison.png)、[完整页面](../preview/outputs/expe00001_step500_20261010_v2/index.html)、[导出ZIP](../preview/outputs/expe00001_step500_20261010_v2/export_bundle.zip)。

视频直接回放checkpoint中可选M6方案A的原生机械臂运动，不走旧“手轨迹后附IK臂”流程。2078完成同次1830个200Hz物理键样本和音轨，2080仅对保存的qpos/mocap重新取景，不重跑策略/物理/IK。1280×720、20fps、10.2秒、原速含准备/曲尾/1秒末停；单片200Hz原谱音乐micro F1=5.2290%（TP141／FP2637／FN2474），原谱34音符。原MIDI、事件CSV、绘图数组、来源与22文件散列/ZIP CRC齐全。

机械臂模式的finger_clearance合同可非空而实际guard为None，旧渲染器因此KeyError；当前展示脚本按实际guard存在性读取诊断。--recording可使用render_recorded_native读取同次原生状态和音轨并完整校验模型/检查点合同；CUDA检查点需在分配内CUDA加载以匹配RNG状态。旧训练归档保持不变，本轮使用logs/expe00001_stop_export/下通过清单审计的展示派生包，全部173个模型源码散列保持；不是修改旧checkpoint或放松回放检查。训练仍中断，未处理best筛选与训练性能问题。

## 当前SDK四指swing约定：0对应最小值（2026-10-08用户纠正）

当前网页导入使用`hardware_sdk_v2`：`q_URDF(rad)=SDK角度(deg)×π/180−0.35`。SDK0→URDF−.35rad（最小值），20.0535deg→0，40.107deg→+.35rad（最大值）。CSV结构版本仍schema_version=1；原SDK关节角/时间不改，仅方向标记升级，旧hardware_sdk_v1新导入明确拒绝，旧视频/已归一化历史记录只读保留。urdf_v1输入不会再转SDK零位。

本轮`preview/outputs/left_sdk_zero_min_20261008`为新版标准轨迹/模板和重新回放目录，基座/法兰安装及其余关节沿用本轮已反算结果，不加0.15rad。SDK0静态图`swing_zero_pose`的四指均为模型最小值；此前最大值图中的SDK0标签属于已停用的旧解释。新版录像必须重新积分同次琴键，并配同次声音/按键图，不复用旧方向视频或分数。原谱MIDI/音乐起点仍缺，F1留空。

## 2026-10-08：统一CSV上传及按法兰首点一次反算基座

`trajectory_compare --enable-exports`已经接入真实Slurm回放/渲染。网页可下载v1模板与字段说明CSV：首行明确SDK/URDF单位约定、法兰钢琴XYZ及可选基座RPY/标定七角；完整27关节位置，首帧不空，后续空值沿同关节前值。基座朝向保持、反算平移一次并固定，不额外加0.15rad。机械臂已给关节轨迹时仅重算FK空间轨迹，不虚构新的IK目标。格式、API、启动及模型边界见`trajectory_compare/README.md`。

本次用户首点[207.090,−150.552,99.158]mm反算基座[609.880266,−19.281046,418.768079]mm；`preview/outputs/left_flange_recalibration_20261008`保存原臂200Hz文件、74s名义规划、88.092815933s实机时钟的标准目标/反馈、基座/坐标图、完整键录制/事件/声音/PNG和网页dataset。Slurm2017/2030/2026/2027完成四个有声视频，原文件/规定机器人每步不变及完整解码/AAC检查通过。前端三路为20指动力学（机械臂规定）、目标仅键动力学、反馈仅键动力学。

这次上传未提供匹配的原始MIDI和音乐起点；不拿其他实验乐谱替代。F1=null，PNG上层明确缺原曲，下层为同片同次400Hz实际按键。原谱对比尚未完整交付。通用CSV导出器当前尚无MIDI输入合同，因此将来新上传也明确显示此缺项，后续加入可靠原谱/时钟合同才能评分。声源仍为该片实际琴键激活。

## 2026-10-07：976最佳模型视频和F1

通过单H100 Slurm985录制976 best19000，保持原config/82199601…bda022归档、原谱指法、双硬限制；正面双手、有声、1280×720/20fps/10.2秒，1秒准备和末帧停留1秒。1830个200Hz实际物理按键样本用于同次音频和原始MIDI比较；单视频音乐micro F1=81.570043%，TP1808/FP10/FN807。训练best82.138845%/final81.856919%为三回合最差，41次评估曲线不平滑、不重跑策略。

交付目录preview/outputs/experiment_976_best_20261007/包含视频、原谱／实测比较PNG/PDF/事件/NPZ、training_f1的PNG/PDF/CSV/来源、HTML和ZIP；docs/figures/experiment_976_f1_20261007/保留独立曲线。HTTP预览仅监听本机127.0.0.1:18768，index.html已请求在Codex浏览器打开，返回queued不宣称客户端已渲染。视频全片含音轨解码、来源合同和原谱/音频/子步/控制端点一致、25项本地manifest及ZIP CRC/SHA通过；实际查看4.5秒帧、对比PNG和曲线。审计在logs/hpc/export_976_20261007/，服务器原归档和录制保留，未修改训练或硬件。

## 2026-10-07：635左臂差分IK曲线

用户给定七初角后，从635 best19500既有视频的200Hz完整手腕位姿求左臂差分IK，采用此前相同初角对应的冻结基座／安装，原0–9.15秒不变，无速度裁剪或滤波。[关节角度CSV](../preview/outputs/experiment_635_left_differential_ik_20261007/joint_angles_deg.csv)、[角度／速度／加速度图](../preview/outputs/experiment_635_left_differential_ik_20261007/plots/left_arm_motion.png)、[报告](../preview/outputs/experiment_635_left_differential_ik_20261007/report.json)、[完整ZIP](../preview/outputs/experiment_635_left_differential_ik_20261007/joint_curves.zip)。七关节速度均低于SDK规划180°/s，但七关节加速度均有超限，最大J7为6128.12°/s²，参考900°/s²。独立URDF FK1831点、来源散列及数据／图／ZIP验收通过；635原模型无臂，本输出仅为轨迹运动学分析，非重新训练或实机数据。审计在logs/ik_635_left_20261007/。

## 本地HTML查看入口（2026-10-07）

聊天中的index.html文件链接可能打开源码。683 A100导出目录当前由TenseiNode1上的127.0.0.1:18767提供HTTP服务，网页地址为 http://127.0.0.1:18767/index.html ，包括原视频和同次按键对比图。服务只监听服务器回环地址；若浏览器在另一台电脑，须在连接A100服务器的SSH会话配置本地18767到服务器127.0.0.1:18767的端口转发，再打开该地址。也可将完整export_bundle.zip下载、解压后在浏览器中打开index.html，保持相对目录完整。服务检查记录见logs/hpc/export_683_browser_20261007/；未重新生成历史导出或修改训练。

## 2026-10-06：用户附件左臂CSV有声回放

[视频与按键／原MIDI对比](../preview/outputs/left_csv_video_20261006/index.html)使用与`left_replan_smooth_20261006`完全一致的200Hz附件和配套优化手指轨迹，左臂＋左手、相机朝演奏者、固定基座、原速、有声。1600×1000／40fps／10.125秒，含原1秒准备与末帧1秒展示；同次新400Hz琴键积分记录生成声音和对比图，原录制音轨未复用。输入、实际琴键、原始MIDI、事件／绘图数组、来源和检查均保存在交付目录。

这是规定手指／手根运动的琴键动力学回放，每步恢复非琴键状态，机械臂按CSV运动学显示；不是重新执行checkpoint策略或验证PD／伺服／整臂动力学。右手移离琴键，踏板关闭。音乐区间单次400Hz micro F1=96.0778%（TP3013／FP44／FN202），不替代历史训练F1。CPU llvmpipe完成渲染，无GPU；完整视频解码、音轨、附件一致、实际阈值／音频／图同次记录及manifest／ZIP验收。方法与限制见目录README和report，任务脚本／证据在`logs/left_csv_video_20261006/`。

## 2026-10-06 19:41：683当前最佳视频已改用A100完成

用户要求改用A100导出。本机TenseiNode1的Slurm1791分配内双卡6/7通过CUDA/Graph/互传/Warp及设备查询，随后单卡Slurm1797实际物理2号A100通过基础检查，66秒COMPLETED/0:0，保持683 best19000、完整config及82199601…bda022归档。普通会话全卡NVML查询此前失败；分配内这些卡查询/计算通过，不据此宣称全部A100或历史ECC故障均修复。未改驱动/CUDA或其他训练。

[有声视频](../preview/outputs/experiment_683_best_a100_20261006/videos/experiment_683_best_19000.mp4)、[同次实际按键／原始MIDI PNG](../preview/outputs/experiment_683_best_a100_20261006/midi_comparison/midi_comparison.png)、[完整页面](../preview/outputs/experiment_683_best_a100_20261006/index.html)、[ZIP](../preview/outputs/experiment_683_best_a100_20261006/export_bundle.zip)。正面双手、1280×720、20fps、原速10.2秒，保留1秒准备和末帧停留1秒。视频和图来自同次1830个200Hz物理子步；图包括原曲、实际触键及逐采样正确/漏键/误触层，保留原始MIDI、事件CSV、NPZ及来源。

本视频音乐区间micro F1=.8976842105263158（89.77%，TP2132/FP3/FN483），是单回合指标，不替代训练best三回合最差89.23%。PT SHA`aef93caab214301f83dea08b92ff6d41afb84677cae2fbf721f447b82cda4f00`及config SHA`5fa26a5ed312d662c10e7ed7af902aceda261588c1bf99b41e7891e4fdaa6b16`复制后与H100原件完全一致。跨服务器传输较慢，采用独立用户目录内的SSH文件传输依赖并行复制，认证从终端读入，未保存密码或写入交付包。

源码／PT／config合同、原谱目标与200Hz实测数据、20Hz端点、真实按键音频事件一致；H264/AAC全片解码、PNG、21项manifest和ZIP SHA/CRC检查通过，已查看4.5秒帧及完整对比图。证据位于`logs/hpc/export_683_a100_20261006/`和1797 GPU探测／导出缓存。H100排队718已取消，其CPU等待器/本地收集器停止；旧本地查看页重定向到本次A100产物，不再自动提交H100重录。以下19:03排队状态为历史记录。

## 2026-10-06 19:03：683当前最佳视频导出718排队

用户要求导出当前视频，选择最近续训683的best19000（训练三回合最差F1=89.23%），checkpoint SHA`aef93caab214301f83dea08b92ff6d41afb84677cae2fbf721f447b82cda4f00`、config SHA`5fa26a5ed312d662c10e7ed7af902aceda261588c1bf99b41e7891e4fdaa6b16`、原82199601…bda022归档保持。经已有`export_video.sh --gpus 1`和submit.sh提交Slurm718，正面双手、原速、有声、不附额外关节曲线。

19:03检查718仍PENDING/Priority，H100八卡由其他作业占用，尚未录制视频或取得回放F1；不能宣称交付完成。已启动远端CPU后处理等待器与本地自动收集器，单次录制完成后生成200Hz真实按键/原始MIDI共享时间轴对比图及数据，校验原PT/config/源码、音轨、全片解码、manifest/ZIP，最后下载并再次校验。没有停止其他训练或在分配外使用GPU。

[本地待完成查看页](../preview/outputs/experiment_683_best_20261006/index.html)每30秒刷新，完成后自动替换为视频与对比图页面；当前为明确排队占位页。请求、CPU脚本与工作状态位于`logs/hpc/export_683_20261006/`。远端输出`/home/houj/projects/tsensei_video_export_20261005_2334/preview/outputs/experiment_683_best_20261006`。自动流程最长等待24小时，失败会记录状态，当前不保证GPU排队完成时间。

## 2026-10-06下午：本轮五组最佳模型视频汇总

用户要求检查当前训练并导出所有视频，按本轮580／581／582／585／620各自best导出，未改训练配置。[统一查看页](../preview/outputs/all_training_best_videos_20261006/index.html)、[全部MP4／对比图及数据ZIP](../preview/outputs/all_training_best_videos_20261006/all_videos_and_comparisons.zip)、[训练best／final与视频单回合F1表](../preview/outputs/all_training_best_videos_20261006/results.csv)。视频与图按训练作业号分目录。

585复用619同次已验收视频；580 best19532、581 best12500、582 best13500、620 best17500分别使用Slurm623／624／627／626单卡录制。四次均通过submit.sh执行，使用各自匹配训练归档、PT和完整config；保留hard_v1，未静默应用后来新奖励或目标运动规则。首次582导出625在物理5号卡无帧产出并停滞，未见明确ECC报错；627在物理0号卡通过探测后取消625，保持同一checkpoint重试成功，停滞日志保留。

每片正面双手、原速、1280×720、20fps、H264/AAC、10.2秒；每片同次录制1830个200Hz物理子步，原谱音乐偏移1秒，末帧停留1秒。PNG／PDF／CSV／NPZ、逐字原MIDI、实际MIDI及来源均随片提供；训练评估3回合最差值与视频单回合精确音乐micro F1分列。检查源码／PT／config、真实键程重建音频事件、原谱目标与物理子步及控制端点逐项一致，全片解码／音轨／PNG、SHA及ZIP校验；交付不包含模型网格或PT。来源、原始录制数组和校验在`logs/hpc/all_videos_20261006/`，585原证据仍位于export_585_20261006。

## 2026-10-06：585最佳模型有声视频与同次原谱对比

用户要求导出585，选择best19500（PT SHA `4f68d74192a4042069f9c7c28cfa29effdf3ec907e8a4a9e1c8e49db657a6fd4`），保留原config及82199601…bda022源码、hard_v1位置限位和Pitch目标运动限制，未应用最新默认奖励。H100单卡Slurm619经submit.sh执行，61秒COMPLETED 0:0；一次物理录制同时用于视频、声音和按键比较。

交付[有声视频](../preview/outputs/experiment_585_best_20261006/videos/experiment_585_best_19500.mp4)、[实际按键／原始MIDI对比PNG](../preview/outputs/experiment_585_best_20261006/midi_comparison/midi_comparison.png)、[完整索引](../preview/outputs/experiment_585_best_20261006/index.html)及[ZIP](../preview/outputs/experiment_585_best_20261006/export_bundle.zip)。正面双手、1280×720、20fps、原速10.2秒，保留1秒准备和末帧1秒停留；实际物理记录9.15秒、1830个200Hz子步，编码包含首帧展示。图采用原视频时间及1秒音乐偏移，不对末帧停留伪造物理样本。

本视频原谱音乐区间200Hz精确micro F1为`.8435576310762046`（TP1987／FP109／FN628），与训练best三回合最差`.8510908705782673`分别记录。原谱34音、实际28次起音，10个原谱音符在其目标区间内没有真实激活采样；该覆盖计数不是note匹配F1。完整原MIDI、实际／目标事件CSV、NPZ、来源和细节PNG／PDF包含在导出目录和ZIP中。

PT／config／源码合同、音频事件由真实琴键状态重建、原谱目标与录制子步逐项一致、20Hz端点与200Hz记录一致，全部通过；下载后21项manifest及ZIP散列／CRC、原录制输入散列、H264/AAC全片解码和PNG通过。已查看4.5秒视频帧及对比图。证据及同次原始数组位于`logs/hpc/export_585_20261006/`，远端录制保留在619/export_work/recording。后处理修正了验证脚本对省略fps时空值的处理，采用媒体实际20fps；没有重跑策略或修改物理记录。

## 2026-10-05：H100单卡导出关闭和弦final已完成

`export_video.sh --gpus 1`现支持一次完整检查点回放和多视角渲染；省略仍按原默认申请2卡。参数经`scripts/hpc/submit.sh`传给Slurm，并以实际申请卡数做启动探测。训练学习器使用第二张卡不意味着评估／导出也需要第二张；本次只改导出资源申请和入口许可，没有修改归档内物理、策略、奖励或检查点合同。

```bash
bash export_video.sh --gpus 1 --pt /absolute/path/final.pt \
  --source-archive /absolute/path/matching_source.tar.gz \
  --video1 noarms front --plots hand-joints --zip
```

关闭和弦1724最新`final.pt`（19532步，SHA `8984d4beffa465d424978969bc69a7550fdf11fbd6df818da48a84e16c1db5b9`）已通过H100单卡导出：[有声视频](../preview/outputs/paper_chord_off_final_h100_video_20261005/videos/chord_off_final_19532.mp4)、[同次原MIDI对比PNG](../preview/outputs/paper_chord_off_final_h100_video_20261005/midi_comparison/midi_comparison.png)、[完整索引](../preview/outputs/paper_chord_off_final_h100_video_20261005/index.html)、[ZIP](../preview/outputs/paper_chord_off_final_h100_video_20261005/export_bundle.zip)。1280×720、20fps、10.2秒，含1秒准备及末帧停留。1830个200Hz真实物理子步与原始MIDI比较，视频本次音乐区间micro F1=.9896392939370683；不是训练多回合分数。

H100 Slurm566完成双卡基础探测；568单卡完成物理录制，但附加手关节曲线触发真实状态越界校验。574单卡指定`--recording logs/hpc/568/export_work/recording --plots none`复用同次原始轨迹成功导出，不重新推理、不裁剪状态、不放宽曲线校验。曲线未交付，原录制保留。完整config与ae7e29f3训练归档严格匹配，未套用工作区后来新增的硬限位合同。原A100双卡1742和待运行单卡1743／后处理1744已取消，节点ECC故障未恢复。

独立远端目录为`/home/houj/projects/tsensei_video_export_20261005_2334`，脚本／请求／验证位于`logs/hpc/h100_video_export_20261005/`。同次PNG/PDF/CSV/NPZ与来源已纳入HTML／ZIP；源码、PT、目标／实际数组、音轨及全片解码、PNG、manifest／ZIP均通过。`delivery_status.json`记录后处理结果；仅GPU导出`status.json`完成不能替代此项。

CPU边界：已有物理录制可软件渲染并绘图；匹配训练归档的PianoScene要求CUDA，不能仅传`--device cpu`完成相同后端回放。本次物理录制与GPU渲染在Slurm内执行，对比绘图／交付校验用CPU完成；未实现或验证新的原生MuJoCo CPU后端。

## 最新：初始化反算基座，固定基座的左臂回放（2026-10-05）

交付[有声视频](../preview/outputs/left_base_from_start_20261005/left_arm_hand_front.mp4)、[左手WAV](../preview/outputs/left_base_from_start_20261005/left_keystrokes.wav)、[实际按键／原MIDI PNG](../preview/outputs/left_base_from_start_20261005/actual_keys_vs_original_midi.png)和[轨迹／完整索引](../preview/outputs/left_base_from_start_20261005/index.html)。用户确认原首帧与给定七角用于初始化反算一次左基座完整位姿；随后基座固定，只做关节IK。手／法兰和右侧不变，当前基座详见坐标与机械臂轨迹专题。

视频保留左臂／连接件／左手，相机面向演奏者，1600×1000、40FPS、405帧／10.125秒。首帧即用户七角，没有新增2秒衔接；原1秒准备段、9.125秒轨迹及1秒末帧展示保持。音频及键程取同次原400Hz录制15次左手真实激活，原MIDI取同一左声部原始起止时间，图／音频／高亮统一移除新增2秒偏移。每帧固定基座、首帧角度／位姿、源PT/config/归档合同、CSV、H264/AAC全片解码和图／音频事件一致性通过。原左声部F1=.966120566仍为原录制指标，不是新接触物理评价；CPU脚本及验收位于`logs/left_base_from_start_20261005/`。

## 2026-10-05：保留左机械臂，面向演奏者，独立左手击键音乐

用户对上一版指定七角重规划回放要求改视角和单独左声部，随后明确保留机械臂。交付[左臂与左手正面有声视频](../preview/outputs/left_only_facing_performer_20261005/left_arm_hand_front.mp4)、[独立击键WAV](../preview/outputs/left_only_facing_performer_20261005/left_keystrokes.wav)、[实际左手按键与原始MIDI对比PNG](../preview/outputs/left_only_facing_performer_20261005/actual_keys_vs_original_midi.png)和同目录HTML／数据／来源报告。视频1600×1000、40FPS、12.125秒；相机在钢琴−X侧朝＋X演奏者方向，azimuth=0°、elevation=−38°，左臂基座至指尖完整入镜。保留原2秒衔接＋9.125秒轨迹及1秒末帧停留，音乐准备边界为视频3秒；本次仅改变呈现，臂／手NPZ和安装SHA保持。

左声部按同次1359原400Hz录制重建：冻结原场景对3650个已保存状态仅执行CPU前向接触检测，31次激活均有唯一手归属且左右接触键集合不相交。取15次左手触发，排除16次右手触发；起音／松键来自真实键程阈值，接触只判定手的归属，未达阈值的接触不发音。实际声音采用原归档音频代码、原SoundFont及原共用踏板，48kHz立体声；原谱参考用原始MIDI中指法5..9的左手音符，未量化时值。图、音频、高亮共享事件与时间轴，保存接触归属、左右事件、左键完整数组和原谱事件。

H264/AAC全片解码、15次音符起止与按键数组一致、WAV／AAC音频相关系数0.9999858、源归档内全部合同SHA、PNG和交付manifest检查通过。原录制左声部400Hz音乐段micro F1=.966120566，只描述原物理录制；画面沿用新臂轨迹的运动学回放，未重新进行接触物理验收。实现与日志：`logs/left_only_facing_performer_20261005/`，全程CPU OSMesa llvmpipe，原训练及历史产物保持。

## 2026-10-05：左手四指向左摆动的编辑预览

以best19000的同次t=0录制为基底，仅将左食指、中指、无名指、小指swing设为+15°，方向为俯视图左侧／钢琴−Y。输出位于`preview/outputs/idle_only_best_left_swing_left_20261005`，包含真实模型俯视图、青色投影图、`swing_joint_angles.csv`、`edited_pose.npz`和来源报告。该目录是手动编辑姿态，不是原始策略回放；原始录制和此前IK不变。生成及验证脚本／证据位于`logs/idle_only_first_frame_swing_left_20261005`。

## 2026-10-05：同一左手第一帧的正交俯视与琴面投影

用户指定已交付best19000左手第一帧斜视图，要求俯视并直接观察琴面投影。复用Slurm1720同一物理录制的initial_qpos/mocap及匹配归档模型，未重新推理或换轨迹。相机沿钢琴＋Z向下、正交投影，屏幕向右为＋Y、向下为＋X；左手根仍为原[140,−150,125]mm和原朝向，t=0准备起点。

交付[左手俯视PNG](../preview/outputs/idle_only_best_left_first_frame_top_20261005/left_hand_top_view.png)、[半透明琴面投影PNG](../preview/outputs/idle_only_best_left_first_frame_top_20261005/left_hand_keyboard_projection.png)、[无标注PNG](../preview/outputs/idle_only_best_left_first_frame_top_20261005/top_view_clean.png)，另有分割mask、来源/指尖坐标JSON、HTML及ZIP。

投影取原模型左手可见几何的MuJoCo分割轮廓，以同一相机投到`Z=.0225m`白键面参考平面：`(x,y,z)→(x,y,.0225)`。背景只显示原琴键，青色40%叠加／轮廓保留下面琴键可见，黄色五点使用原归档模型的实际fingertip site。黑键保留真实外观，手掌超出键盘区域的投影不截去，不把几何投影当真实触键。阴影及反射关闭，避免将光照阴影混作正投影；P_xy/H_xy分别是钢琴原点与手根的XY投影。

四张PNG均1600×1200，mask172754像素，轮廓无裁剪；同一XY沿Z移动前后的投影像素误差0，手根矩阵与上一帧数据一致。CPU OSMesa llvmpipe重渲染，无GPU、物理推进、IK或姿态修改。PNG实际查看、输入SHA、6项交付SHA、ZIP内容/CRC通过，证据与复现脚本位于logs/idle_only_first_frame_top_20261005。原斜视、IK、视频及训练均保留。

## 2026-10-05：从best19000左手第一帧求左臂IK

用户进一步要求使用已交付第一帧的完整左手位姿，求当前机械臂每关节角度。目标直接读取`idle_only_best_left_first_frame_20261005/frames.json`的`T_world_left_hand`，保持t=0的[140,−150,125]mm及原朝向。采用当前本地左M6 URDF、preview_mount与现有法兰到手安装变换，未应用上轮实机CSV的J1／法兰90°候选。

CPU SciPy有界6维IK，以位置及0.15m/rad的SO(3)旋转误差求解。当前预览seed=0及固定种子随机初值共7次后得到首个可达解；首解J7离限位仅.033°，随后以SLSQP保持完整6维目标等式，沿同一分支增大最小归一化关节限位余量，最终最小绝对余量9.635949°。这是七自由度的一组局部可达解，不声称唯一姿态或全局最优，也未进行碰撞／动力学验证。

| 关节 | rad | deg |
|---|---:|---:|
| J1 | .719112880 | 41.202133 |
| J2 | .983340641 | 56.341269 |
| J3 | −2.774078527 | −158.942992 |
| J4 | −1.461788227 | −83.754296 |
| J5 | 2.701710572 | 154.796613 |
| J6 | −.792145257 | −45.386580 |
| J7 | −1.402620965 | −80.364262 |

交付[姿态PNG](../preview/outputs/idle_only_best_left_first_frame_ik_20261005/left_arm_ik_pose.png)、[CSV](../preview/outputs/idle_only_best_left_first_frame_ik_20261005/left_arm_joint_angles.csv)、[完整JSON](../preview/outputs/idle_only_best_left_first_frame_ik_20261005/ik_result.json)及同目录HTML／ZIP。手部URDF SHA匹配训练归档，手指20个关节使用第一帧记录角度；手根由机械臂FK达到原目标。独立MuJoCo位置误差2.5172e-16m、朝向误差4.4067e-16rad；将关节角保留6位小数rad再回代，误差2.1913e-7m／7.8960e-7rad。所有运算与单帧软件渲染在CPU完成。

来源散列、7关节限位与CSV一致性、PNG、交付SHA／ZIP验收见`logs/idle_only_first_frame_ik_20261005/validation.json`，同目录保留求解、冗余余量调整和渲染验证脚本。数值精度仅表示该模型下FK一致，不代表实机精度；角度为URDF约定，SDK编码器映射与肩座／工具安装仍待标定。未修改训练、模型或安装默认。

## 2026-10-05：仅姿态惩罚最佳检查点的左手第一帧坐标系

用户要求第一帧静态图，后明确只导出左手。实验1681 `paper_chord_finetune_idle_only_20261005-20261005-075829-425286`已完成，best19000按三回合最差`exact_original_music_micro_f1_min=.9942528735632183`选出；该分数是训练选模指标，不是静态图评分。仅姿态指在原基础奖励上开启idle_posture_penalty_v1而self_collision_reward关闭。

通过submit.sh单卡replay提交1720，严格使用匹配`source_v2_fingertips.tar.gz`归档与冻结best/config，167文件源码校验及checkpoint合同通过，34秒COMPLETED 0:0。被动录制1830个200Hz物理子步及独立initial状态。取initial_qpos/mocap的t=0准备段起点，手根已抬升25mm、策略尚未动作；不是准备结束或第一个按键时刻。本次交付仅静态第一帧，不交付内部完整录制视频；没有对该帧编造演奏／MIDI指标。

交付[左手坐标图](../preview/outputs/idle_only_best_left_first_frame_20261005/left_hand_piano_frames_oblique.png)、[侧视图](../preview/outputs/idle_only_best_left_first_frame_20261005/left_hand_piano_frames_side.png)、[无标注第一帧](../preview/outputs/idle_only_best_left_first_frame_20261005/first_frame_left_oblique.png)、[矩阵JSON](../preview/outputs/idle_only_best_left_first_frame_20261005/frames.json)、[HTML](../preview/outputs/idle_only_best_left_first_frame_20261005/index.html)及同目录ZIP。只有左手可见；右手仅在原物理录制中保留，离线绘图隐藏其可视几何，不改变首帧状态。没有叠加M6或此前实机CSV的J1／法兰候选修正。

P取实际归档模型`piano/`根系，H取`left_/base_link`，不将掌心、指尖、惯性质心或法兰混作手根。P在世界系为单位矩阵，白键面在P的Z=22.5mm；红X／绿Y／蓝Z，黄色虚线连接真实原点。坐标轴按真实三维位置投影覆盖画面，可遮盖模型表面；不是为了可见性移动坐标原点。手根在P系为[.1400000008940807,−.15,.12499997062657948] m。

`T_P_H`将手系齐次列向量映射到钢琴系，旋转矩阵列为H的X/Y/Z在P中表达：

```text
[-0.1494372792   0  -0.9887712069   0.1400000009]
[ 0            -1   0             -0.1500000000]
[-0.9887712069   0   0.1494372792   0.1249999706]
[ 0             0   0              1           ]
```

完整精度、逆变换、来源SHA与左手23个实际关节位置均保存。CPU按归档模型body/joint树独立FK与实际MuJoCo xmat/xpos误差1.3877787807814457e-17；坐标图用既有OSMesa llvmpipe，GPU推理／首次录制仅在1720分配内完成。验证矩阵合成／逆、PT与config/源码合同、PNG、交付SHA／ZIP通过，见logs/idle_only_first_frame_20261005/export_validation.json。首帧属于初始化合同，不能据此评价姿态惩罚训练效果。

## 每次导出必须附带原始MIDI对比图（2026-10-04用户要求）

2026-10-05本轮“基线＋三组续训”的best视频已汇总：[四组查看页](../preview/outputs/paper_chord_all_best_20261005/index.html)、[全部MP4／对比PNG与数据ZIP](../preview/outputs/paper_chord_all_best_20261005/all_videos_and_comparisons.zip)。新增Slurm1722（仅self，1682 best18500）与1723（idle＋self，1683 best18500）分别79／80秒COMPLETED 0:0；复用1678基线与1721仅idle原视频及各自原按键图，无新策略回放替换旧图。四段均1280×720、20fps、原速有声10.2秒；每组1830个200Hz物理子步各自对原始MIDI。本视频单次精确F1依次为基线.990439771、idle.993098160、self.992905081、idle＋self.992720307，不能代替训练三回合排名。独立完整导出仍在各paper_chord_*_best_20261005目录，汇总包只收视频、对比图及原始对比数据和来源，不重复打包手关节曲线或PT。新增两段的源码／PT／奖励／录制目标核验、视频解码、AAC音轨、各组及汇总manifest/ZIP验收通过，证据logs/hpc/{1722,1723}/video_export_validation.json及logs/hpc/paper_chord_all_best_export_20261005/validation.json。

2026-10-05仅姿态惩罚续训1681的best19000已通过Slurm1721导出：[有声视频](../preview/outputs/paper_chord_idle_only_best_20261005/videos/idle_only_best.mp4)、[同次原MIDI对比PNG](../preview/outputs/paper_chord_idle_only_best_20261005/midi_comparison/midi_comparison.png)、[HTML](../preview/outputs/paper_chord_idle_only_best_20261005/index.html)。正面无臂，1280×720、20fps、原速，含准备／曲尾及末停，共10.2秒。实际1830个200Hz子步重建原始按键，音乐区间精确F1=.993098160（TP/FP/FN=2590/11/25）；该值仅属于本视频，训练best的三回合最差F1=.994252874另存data/training_selection.json。使用原匹配归档及完整奖励配置，humanoid_finger_gesture_reward=idle_posture_penalty_v1开启、self_collision_reward关闭。全片解码、AAC音轨、源码／PT／目标／实际记录一致性、manifest及ZIP验收通过，证据logs/hpc/1721/video_export_validation.json。

2026-10-05最新paper/chord训练1620已完成；best15500经Slurm1678导出至`preview/outputs/paper_chord_on_best_20261005/`，入口[index.html](../preview/outputs/paper_chord_on_best_20261005/index.html)，[有声视频](../preview/outputs/paper_chord_on_best_20261005/videos/latest_best.mp4)、[同次原MIDI对比PNG](../preview/outputs/paper_chord_on_best_20261005/midi_comparison/midi_comparison.png)。200Hz原谱音乐区间精确F1=.990439771，34原音符均匹配起音，额外一次A3重触发；视频保留1秒准备、原速完整9.15秒物理轨迹及1秒末帧停留。HTML/ZIP包含对比图和数据，训练三回合指标与视频单次指标分别保存。同手自碰撞仍见训练评估；本视频未录制独立同手碰撞计数，不宣称整体无碰撞。

2026-10-05已为最新S4最佳视频补齐同次回放对比：[全曲PNG](../preview/outputs/twinkle_20hz_best_S4_front_20261004/midi_comparison/midi_comparison.png)、[局部PNG](../preview/outputs/twinkle_20hz_best_S4_front_20261004/midi_comparison/midi_comparison_detail.png)。原曲目标与1830个200Hz录制目标逐项一致，实际键事件与视频音频同源；单视频精确音乐F1=.983130、两次短E5未触发。目录含CSV/NPZ/MIDI/PDF/来源及CPU重建脚本，已纳入原HTML/ZIP；未重新推理策略，不将多回合复评分数套到本图。

每次视频交付同时提供“实际按键与原始MIDI时序对比图”（至少PNG），无需再次询问。实际按键必须来自视频同一checkpoint的同一次物理回放，以真实琴键激活状态重建，优先使用完整物理子步；参考为匹配归档/config的原始MIDI或注册原曲起止事件，不能以量化训练目标代替。共享时间/音高轴展示原曲与实际演奏，并能看出漏键、误触、起音和松键偏差；注明采样频率和准备段偏移，裁剪/倍速与视频时间说明对齐。

图与绘图数据/来源记录保存在视频导出目录；交付HTML/ZIP时一起包含，最终答复同时链接视频与图。核验图与该回放的按键数据和时间范围一致，不能借用另一次评估轨迹或将训练多回合F1标成本视频分数。缺失数据先从原归档/同次录制恢复，无法恢复则明确缺项，不能宣称完整交付。

这是导出工作流的强制交付规则；本次仅更新约定，未将生成器改为自动出图。使用现有CLI后如没有对比图，执行导出的助手须基于同次录制补充CPU绘图后再交付；下文旧导出记录不代表已追补此项。完整规则见根目录 `AGENTS.md`。

2026-10-02修复：精确音符时间训练的录制`goal_applied`来自控制周期末的原始事件采样，不能与旧网格`score.goal`直接比较。指法恢复现从经过归档散列核验的exact_note_events/exact_sustain_events，调用该归档的sample_targets按记录policy_time生成目标和原指法，再逐帧严格匹配录制goals；旧网格模式保持。只改变导出标注读取，不更改训练、谱面、PT或已录物理轨迹。

2026-10-01更新：默认全部视角使用**正交投影**，消除近大远小的透视变形；按编译钢琴几何自动计算取景，钢琴宽度铺满每个画面的宽度。渲染所需STL/XML只存作业缓存，交付目录和ZIP不含模型文件。此前用户导出目录中的120个STL已连同中间场景移到原作业缓存，视频和分析数据保留。

指法显示修复：多视角重渲染曾遗漏原回放的指法层，现默认显示原乐谱的目标指法、指尖颜色和目标琴键颜色。每个画面底部标明`Score Right: R1:C5`、`Score Left: L3:C3`等，R/L分别为右/左手，1–5为拇指、食指、中指、无名指、小指；绿色仍表示实际按下的琴键。可加`--finger-labels`在三维指尖/目标键上显示悬浮编号，或`--no-fingering`关闭整层指法显示。`--no-overlay`只关闭顶部视角/时间信息，不关闭指法。

指法直接恢复自该PT精确匹配的源码归档和完整config；同时检查重建task的SHA与原回放一致、每个控制帧目标与`rollout.npz`一致，不按音高重新编配指法。已有录制也兼容，无需重新推理策略。标注采用原物理时间的控制区间，精确端点显示刚评分的目标；准备空谱/释放段显示rest，裁剪、倍速、末帧停留和多视角共用该时间轴。底部说明预留取景空间，琴宽铺满和正交投影保持。

旧产物清理：2026-10-01按用户要求删除`preview/outputs/`下1070个历史导出文件，共2,072,455,555字节；旧视频/曲线/MIDI对比目录需重新导出。只保留两个安装工具仍读取的`connector_alignment/hole_aligned_mount.json`（888字节）。训练PT、模型资产和`logs/hpc`作业缓存保留，清理清单位于`logs/preview_outputs_cleanup_20261001.json`。下方历史验收记录描述当时结果，不表示旧交付路径仍存在。

入口是项目根目录的 `export_video.sh`，内部先检查参数并固定PT副本，再经
`scripts/hpc/submit.sh export` 提交Slurm；普通SSH不会初始化GPU。
脚本返回作业号、输出目录和日志路径。导出与图表都在作业内完成。

## 直接使用

在项目根目录执行；将PT路径替换成想导出的best/step/final文件。
不写`--plots`时，默认导出机械臂和双手全部实际关节的速度、加速度；每关节独立PNG/PDF，另有左右汇总图及CSV/NPZ。
新导出默认`--arm-ik constrained`：保留原节奏，联合优化整段机械臂轨迹，约束关节角、SDK速度和加速度。默认位置误差上限0.1 mm、姿态误差上限0.5°；任何限值或跟踪验收未通过则报错，不自动减速或放宽。原差分IK仅作为初值；`--arm-ik raw`可显式保留旧诊断行为。详见[限速算法与实测](arm_motion_constraints_20261001.md)。

```bash
bash export_video.sh \
  --pt training_mjlab/outputs/hpc/gesture10-comparison-20260930-140658-445247/best.pt \
  --videonums 2 \
  --video1 arms backend \
  --video2 noarms forhead \
  --zip
```

上例生成：视频1显示机械臂、后视；视频2不显示机械臂、前视；
另外导出机械臂14关节及双手/腕部实际关节的速度、加速度图。需要两腕TCP线速度/角速度时加`--plots all`。
`backend`等于`back`，`forhead`/`forehead`等于`front`；`performer`也等于`back`。
默认带真实琴键音频、原速、完整轨迹、末尾停留1秒。

一段视频内同步拼接多个视角：

```bash
bash export_video.sh \
  --pt /absolute/path/best.pt \
  --video1 arms front top overview layout=grid name=arms_three_views \
  --video2 noarms front hands layout=horizontal name=hands_two_views \
  --plots all --size 1280x720 --fps 40 --zip
```

`--videonums`可以省略，按连续的`--video1 ... --videoN`推导。
每个`--videoN`第一个词是`arms`或`noarms`，后面写1–4个不同视角。
也可写`front,top`。最多16段视频；编号缺失、重复或数量不符会在提交前报错。
不写任何`--videoN`默认导出一段`noarms front`。

只检查命令，不提交/不创建目录：在任意命令末尾加`--dry-run`。
使用`bash export_video.sh --help`查看全部参数。

## 视角与布局

| 视角 | 含义 |
|---|---|
| `front` | 琴前向手部看；世界方位角0° |
| `back` | 演奏者背后向键盘看；世界方位角180° |
| `left` | 侧视，世界方位角90° |
| `right` | 侧视，世界方位角270° |
| `top` | 顶视，俯仰−90° |
| `overview` | 斜侧整体视角，适合双臂 |
| `hands` | 较高俯视角，适合观察触键；同样按整琴宽度取景 |

每段视频可加`layout=grid`、`layout=horizontal`或`layout=vertical`。
默认grid：1视角单格、2视角左右并列、3–4视角2×2；缺少的第四格留空。
`--layout`设全局默认，每段的`layout=`优先。
`--size`是**每个视角**的尺寸：1280×720双视角左右拼接输出2560×720。
所有视角始终是同一物理时刻，不是轮流切镜头。

默认`--piano-fill 1`：每个子画面左右边缘与该视角钢琴的投影宽度对齐。
`--piano-fill 0.9`可让钢琴占90%宽度、两侧各留约5%空白。
取景宽度不因增加机械臂自动拉远；竖直居中参考所选片段的21个姿态采样，并优先保留键盘。
如果完整手臂超出画面高度，可使用更高画布（例如`--size 1280x960`），琴宽仍铺满。
侧视或斜视的钢琴投影形状受视角影响，不通过非等比拉伸改变形状。

单视角还能自定义相机，例如：

```bash
bash export_video.sh --pt /absolute/path/best.pt \
  --video1 arms overview azimuth=135 elevation=-35 name=my_camera --piano-fill 0.9
```

方位/俯仰单位度。`distance=`仍可指定相机的空间距离（米），在正交投影下不控制缩放；
缩放改用`--piano-fill`。相机中心由钢琴和所选片段的显示几何计算。
自定义角度只允许单视角，避免一个配置含糊地覆盖多个镜头。
`name=`只改变视频文件名，使用英文字母/数字/下划线/连字符。

## 图表与数据

| 参数 | 输出 |
|---|---|
| 不传`--plots` | 默认`arm-joints hand-joints`，机械臂及双手每关节速度、加速度全部导出 |
| `--plots none` | 不画图表 |
| `--plots arm-joints` | M6左右各7关节：每关节独立速度/加速度PNG/PDF，左右两类汇总图、CSV/NPZ及峰值报告 |
| `--plots tcp` | 左右wrist/TCP世界坐标线速度m/s、角速度rad/s，XYZ与模长，PNG/PDF/CSV |
| `--plots hand-joints` | 原物理模型内手指/腕部每关节独立速度/加速度PNG/PDF、两手汇总图、CSV/NPZ及峰值报告 |
| `--plots arm-joints tcp` | 可组合任意单项 |
| `--plots all` | 三类全部导出 |

`arm-joints`指机械臂J1–J7，`hand-joints`指手指和腕部实际物理关节。
当前31动作模型仍有46个手/腕物理关节；PIP/DIP从动关节也包含，结构移除的TZ/Yaw不虚构曲线。
平移关节速度/加速度单位m/s、m/s²，转动关节rad/s、rad/s²。TCP是手URDF `base_link` 原点。
显式`--plots`会替换默认组。例如旧命令`--plots arm-joints tcp`只选机械臂和TCP，不包含手；要保留全部可改`--plots all`，或去掉该项使用臂+手默认。
即使所有视频为noarms，默认的arm-joints仍会独立求解IK和画图；只需手曲线可用`--plots hand-joints`。

手关节、旧导出和`--arm-ik raw`采用`v[i]=(q[i+1]-q[i])/(t[i+1]-t[i])`，记录在区间中点；加速度再对相邻速度差分。新约束机械臂轨迹采用与视频完全一致的三次B样条及其解析一阶/二阶导数，采样覆盖原物理时刻、中点和全部样条节点；连续速度/加速度另以导数系数界验证，不靠采样点推断。曲线不是策略目标命令或真机测量。机械臂与下述SDK配置限值比较，未读取实物控制器限值。TCP角速度采用相邻旋转的旋转向量差分。
绘图阶段不滤波、不裁剪数值，不将播放倍速乘进曲线；图表横坐标始终是原录制时间。约束臂的运动限制在规划阶段执行。
图表只包含`--start/--end`片段，精确插入裁剪端点；`data/`原始CSV/NPZ保留完整轨迹。
末帧显示停留不追加样本。差分数据不足三个位置采样时不伪造加速度；解析样条可直接求极短片段的导数。完整约束臂轨迹本身具有静止端点。
`plots/arm_joints`和`plots/hand_joints`均包含`joint_positions.csv`、`joint_velocities.csv`、`joint_accelerations.csv`、`joint_motion_samples.npz`、`joint_motion_report.json`。
若计算了IK，无论是否画图都会输出关节角/速度CSV及IK报告。
腕部位姿CSV/NPZ始终保存，便于后续重新分析。

### 机械臂 SDK 限值标注（默认开启）

统一视频导出与已有视频补导均自动在机械臂的**每关节及左右汇总**速度/加速度PNG、PDF上绘制正负限值红虚线，图例给出数值/单位，红点标记`abs(sample) > limit`的采样，显示峰值与限值之比。纵轴同时容纳完整曲线和限值，不裁剪、不平滑、不自动减速。无需新增命令参数。

采用SDK `CommonConfig/ccs_m6_40.MvKDCfg` 的M6配置，左右相同：

| 关节 | 速度绝对值上限 | 加速度绝对值上限 |
|---|---|---|
| J1、J2 | 180°/s = π rad/s ≈ 3.141593 rad/s | 450°/s² = 2.5π rad/s² ≈ 7.853982 rad/s² |
| J3–J7 | 180°/s = π rad/s ≈ 3.141593 rad/s | 900°/s² = 5π rad/s² ≈ 15.707963 rad/s² |

来源：[SDK固定提交配置](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/blob/a0f8c1134895a1b3ab2a5621bb93ae4ba0634990/CommonConfig/ccs_m6_40.MvKDCfg)，文件SHA-256 `ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52`。`kinematicsSDK/FxRobot.cpp` 的`LOADMvCfg`读取每轴第7/8列为`PNVA[][2/3]`，`FX_Init_Robot_Lmt_CCS`写入`m_JLmtVel/m_JLmtAcc`；`AxisPln.cpp`用加速度计算`v²/(2a)`和`sqrt(位移/a)`。这也澄清了SDK部分概述中把速度/加速度误写为加速度/加加速度，以及未给加速度时间单位平方的问题。`SDK_PYTHON/fx_kine.py`规定配置块0=左臂、1=右臂。

限值及来源快照保存在`configs/preview/m6_sdk_joint_limits.json`，纳入独立源码包；画图只读该快照，不导入/连接硬件SDK。导出的`arm_joints/joint_limits.json`包含度制原值、SI值、来源commit/SHA、快照SHA；`joint_motion_report.json`记录每关节两类峰值比例、超限采样数/比例、首末超限时刻。HTML自动使用同一份数值生成限值表。

这是SDK默认规划配置参考，未按实物负载降额或运行时速度/加速度百分比缩放；也不表示PVT/位置直发一定强制执行这些限制。URDF的速度3.1416rad/s是近似值，标注按SDK的180°/s精确换算。新约束导出会修改离线机械臂轨迹；训练轨迹、手部状态、硬件SDK设置和历史导出保持。手/腕没有本次已确认的限值，图上注明`Reference limit unavailable`，不借用机械臂数值。

### 给已有视频补导关节曲线

```bash
training/.venv/bin/python preview/scripts/export_joint_curves.py \
  --video /absolute/path/export/videos/video1_arms_front.mp4
```

纯CPU读取该视频的原导出报告、机械臂NPZ和缓存中的手部物理记录，检查视频/轨迹SHA及关节列顺序；不重新推理、求IK或渲染视频。默认同时补导机械臂及双手，可用`--plots arm-joints`或`--plots hand-joints`选择。原导出必须保留`data/`及其`logs/hpc/.../recording`缓存。
输出到原导出目录下`视频名_joint_curves/`，含单关节/汇总PNG/PDF、CSV/NPZ、独立报告、HTML索引及`joint_curves.zip`，不改变原视频/原manifest/原ZIP；目录已存在会拒绝覆盖，可用`--output 新目录`重新导出。

### 在 Codex 浏览器或远程 PC 打开报告

`/home/.../index.html` 是服务器文件路径，直接粘贴到浏览器地址栏可能被当成搜索词；本地 PC 也不能直接读取服务器文件。无需重新导出曲线。在服务器启动只服务该报告目录的静态网页：

```bash
python3 -m http.server 18766 --bind 127.0.0.1 \
  --directory /absolute/path/video1_arms_front_joint_curves
```

若 Codex/浏览器位于另一台电脑，在**本地 PC 的 PowerShell/终端**运行：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18766:127.0.0.1:18766 houj@服务器IP
```

将 SSH 地址替换为平时登录服务器的地址/别名，非默认端口沿用 `-p 端口`。保持服务与转发运行，再在 Codex 浏览器访问 <http://127.0.0.1:18766/index.html>。`127.0.0.1` 指浏览器所在电脑；仅在远程启动服务不代表本地端口已经连通。也可把完整 `joint_curves.zip` 下载到 PC 后解压并打开其中的 `index.html`；只复制 HTML 会丢失图片和数据链接。

## 播放与其他选项

| 参数 | 默认与说明 |
|---|---|
| `--pt PATH` | 必填；也可用`--checkpoint` |
| `--config PATH` | 默认PT同目录`config.json` |
| `--fps 40` | 默认检查点控制频率；支持1–240；只改渲染采样，不改策略频率 |
| `--size 1280x720` | 每视角尺寸，两边为240–4096内偶数 |
| `--piano-fill 1` | 琴宽占子画面宽度比例，0.1–1，默认铺满；始终正交投影 |
| `--start 1 --end 5` | 原时间轴秒数，0为initial；包含策略准备或历史固定准备前缀 |
| `--speed 0.5` | 半速；2为双速；支持0.05–20；约束机械臂视频要求1，避免改变已验收运动时间；音频同步变速且保持音高 |
| `--hold 1` | 最后画面额外停留秒数；音频保留完整录制末尾衰减，片段裁剪不播放后续音符 |
| `--no-audio` | 静音；默认有声 |
| `--no-overlay` | 隐藏视角、时间、PT标签；默认显示 |
| `--no-key-highlight` | 关闭实际按键绿色高亮；默认开启 |
| `--no-fingering` | 关闭目标指法/指尖/目标琴键颜色及底部指法说明；默认开启 |
| `--finger-labels` | 在指法显示开启时额外显示三维悬浮编号；默认关闭以减少重叠 |
| `--output DIR` | 必须新目录；默认`preview/outputs/export_PT名_北京时间` |
| `--zip` | 额外打包视频、图、CSV/NPZ、报告，不含PT/完整模型 |
| `--source-archive PATH` | 手动指定匹配训练源码归档；默认按SHA自动查找 |
| `--recording DIR` | 可传已完成导出目录，自动读取其缓存位置；也兼容旧完整录制目录 |
| `--arm-config PATH` | 可选M6肩座/种子等安装JSON；默认当前预览配置 |
| `--connector-config PATH` | 可选手腕连接件JSON；默认当前天机连接件配置 |
| `--arm-ik constrained/raw` | 默认constrained：原节奏、连续关节角/速度/加速度约束；raw只供旧差分IK诊断 |
| `--arm-position-tolerance-mm 0.1` | 默认手腕位置范数误差上限0.1 mm，不自动放宽 |
| `--arm-rotation-tolerance-deg 0.5` | 默认手腕旋转向量范数误差上限0.5°，与位置误差分别检查 |
| `--dry-run` | 只做文件/参数/源码归档匹配检查并打印计划，严格PT元数据校验仍在作业内执行 |

输出帧数包含起始帧；例如1秒片段、20FPS、hold0为21帧/1.05秒。
FPS/倍速改变仅用于重采样显示；不会重新计算较低物理频率的策略或修改原物理碰撞。

复用上次完整记录，重新换视角：

```bash
bash export_video.sh \
  --pt training_mjlab/outputs/hpc/gesture10-comparison-20260930-140658-445247/best.pt \
  --recording /absolute/path/previous_export \
  --video1 arms overview top layout=horizontal \
  --start 1 --end 5 --speed 1 --plots arm-joints tcp
```

新导出把完整记录/模型放在`logs/hpc/JOB_ID/export_work/`，交付目录不出现STL/XML。
后续把完整**导出目录**传给`--recording`即可，脚本按报告自动定位作业缓存；
也可直接传缓存里的`recording/`。复用仍需要保留原作业缓存，视频/图表的播放查看不依赖缓存。
开启`--recording`时不再做策略推理，提交前匹配PT/源码合同并记录轨迹、场景、音频等输入SHA，
作业启动后再次核对。所有视频共享一条真实轨迹；不因换视角再次采样。
开启指法显示时，复用录制仍需精确匹配的源码归档，额外恢复/校验原任务指法；不重复策略或物理回放。缺少原`rollout.npz`或匹配归档会明确报错，可补齐输入或使用`--no-fingering`。

## 产物与查询

- `videos/`：各段MP4和PNG预览。
- `plots/`：所选速度图和相应片段CSV。
- `data/`：完整wrist位姿、可选完整机械臂关节角/速度/IK误差和数值诊断；启用指法时含`score_annotations.npz/json`（原时间、逐帧目标/指法、指尖名称、任务和数据SHA）。
- 约束机械臂额外保存`data/arm_spline.npz/json`、原始初值`arm_trajectory_raw.npz`、跟踪误差CSV/PNG/PDF及帧间审计。视频和补导图表都读取同一条保存的样条。
- `index.html`：打开本地浏览器查看所有视频和图表链接。
- `export_plan.json`：完整调用参数和输入SHA；`report.json`：视角、帧数、尺寸、IK及完整回合指标。
- `inputs/`：提交时固定的PT/config和自定义安装配置副本；训练中的best更新不会替换本次副本。
- `status.json`：running/completed/failed；失败时记录原因，已有文件保留。
- `manifest.json`：交付文件SHA256；启用zip后有`export_bundle.zip`。

提交后终端会打印作业号，例如：

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -f logs/hpc/JOB_ID.out
```

如果sbatch提交失败，参数文件和输入副本保留，不覆盖已有结果；修正后使用新输出目录重试。
如果IK不可达或合同不匹配，任务会报错并保留原始录制，不会静默放宽限位或更换训练模型。

## 合同与验收

2026-10-01指法修复验收：58项CPU回归通过；1103原归档重建任务SHA匹配1353录制，365帧目标逐项一致。使用OSMesa/llvmpipe软件渲染实际物理记录，检查机械臂后视+顶视、无臂前视、悬浮编号开/关、顶部文字关闭但指法保留、整层指法关闭及裁剪/倍速，三个短视频全片解码通过，帧图人工检查通过。报告在`logs/fingering_cpu_validation_20261001/cpu_render_validation.json`。Slurm1356因全部GPU被占用持续等待，完成软件渲染检查后取消；本轮未重新进行GPU推理或GPU渲染验收。

支持当前项目具有完整source_config合同的无臂手部检查点；通过匹配的训练源码归档进行严格回放。
自动匹配范围为本项目`logs/hpc/*/source.tar.gz`及`training_mjlab/source_versions/*.tar.gz`。
找不到时明确要求`--source-archive`，不回退到不匹配的新源码。旧含机械臂物理模型的PT拒绝叠加双臂。

arms视频使用**当前装配的运动学M6覆盖层**，保留已记录手部物理/F1；肩座/工具变换仍待实机标定。
14关节角限和数值跟踪误差检查，速度超限次数/触限次数记入`data/arm_report.json`，不自动改变轨迹。
裁剪视频报告中F1仍是原完整回合指标，明确标注，不冒充裁剪段重算F1。
历史fixed_v1准备前缀按原始状态接入时间轴并校验连续性，policy_v1不额外添加准备。

2026-09-30验证：CPU参数/归档/插值/相机/关节枚举及打包测试通过。
Slurm1214从PT和自动匹配1103归档重新记录3650个400Hz子步，静音/无文字/无高亮视频成功。
Slurm1215复用记录，导出后视+顶视双画面机械臂视频及独立前视无臂视频，
0.3–2.3秒/2倍速/20FPS/末停.2秒，两片同为25帧/1.25秒；AAC音频和全片解码通过。
三组共65张PNG、5份PDF，检查14关节CSV导数、46手关节覆盖、精确裁剪端点及ZIP完整性通过。
首次1213因NumPy标量和MuJoCo枚举比较导致手关节筛选为空，已显式转int并增加回归测试。
Slurm1216复核仅noarms视频也可选择arm-joints图、近景/自定义文件名及角限/超速报告，全部成功。
CPU最终31项通过。Slurm1217进一步验证无机械臂的前视+顶视竖向拼接，480×720、静音、0退出；作业归档确认不含M6。
无机械臂视频且不选arm-joints时，不打包/加载M6资产，保持A3.2轻量安装可用。
2026-10-01完成参数文档与修改日志整理。

TODO(TJ-ARM-02/03/04)、TODO(V32-MOUNT-01)继续保留；以上导出不是机械臂动力学、避障或真机验收。


### 2026-10-01：无模型交付与正交铺满验收

- `camera_for`使用编译场景钢琴可见几何的世界边界，按实际相机右向量投影求宽；
  正交垂直视野=钢琴投影宽/占比×画面高/宽。MuJoCo相机及model globals均启用正交。
  每个子画面渲染前恢复对应视野，避免拼接视角互相覆盖全局参数。
- 51项CPU测试通过，含15组实际MuJoCo GL camera的边界投影验收：
  五种视角×三种宽高比，钢琴左右投影恰为视口边界；改变distance不再缩放。
- Slurm1294验证前/后及顶视+斜视拼接，结果目录0 STL。
  Slurm1299完整重跑1103 best并导出原命令的两段视频，40FPS/1280×720/406帧/10.15秒、有声，0退出。
  当前前/后视钢琴投影宽1.221m，正交垂直视野.6868125m，横向占比1.0。
  21个取景样本中两段显示内容均无竖直裁剪；此采样不作为全连续轨迹边界保证。
- 新结果在`preview/outputs/export_best_orthographic_20261001`；视频全片解码、ZIP CRC、全部交付SHA通过。
  该目录/ZIP中STL、XML、MJB均为0，内部录制/合并模型位于`logs/hpc/1299/export_work`。
- 用户此前的`export_best_20261001_003535_801190`中recording和scene移到
  `logs/hpc/1253/export_work`，120个STL从交付目录移出；视频/分析不变。
  报告缓存路径、相关预览引用、manifest和ZIP同步。旧/新交付目录都可直接用于`--recording`，只读dry-run核对通过。


## 2026-10-01：指定视频的 MIDI 导出与原曲对比

`preview/outputs/export_best_20261001_101143_985592/midi_comparison/` 保存本次用户请求的离线交付，入口为 `report.html`，完整下载为 `midi_comparison.zip`。未修改视频导出默认行为或原交付 manifest/ZIP；新增目录有独立 SHA-256 manifest。

- `played_video_time.mid`：从作业1353原始录制复制，与视频音频同源，保留1秒准备；不是从压缩音频识别生成。
- `played_aligned.mid` / `original_aligned.mid`：分别扣除1秒准备和原文件14.692708333秒曲首空白，可直接导入DAW比较。原始归零版保留绝对秒数、力度和控制器，以固定tempo重新编码；原tempo map/元数据的逐字节版本为 `original_unchanged.mid`。
- `target_40hz_aligned.mid`：从实际rollout目标帧导出的量化音符，无踏板，示意力度80；不能当作未量化原曲。
- `note_comparison.csv` 同时给出原曲、量化目标和实际的起止时间；正误差表示晚，负误差表示早。音符时长不合并踏板余响。
- 原曲34音，实际31音；同音高/起音差≤150ms且唯一匹配下31匹配、3漏音、0多余。匹配计数不是标准note-F1或训练frame-F1。相对原曲起音延迟中位29.792ms、松键晚13.125ms；相对量化目标起音晚45ms、松键早5ms。前一轮聊天的早松毫秒数采用最后激活采样时刻，本次MIDI采用真正note_off边沿，二者相差一个2.5ms物理采样周期。

可用 `training/.venv/bin/python preview/outputs/export_best_20261001_101143_985592/midi_comparison/compare.py` 重建；需保留原录制数据与项目Composer环境。该脚本只处理已保存数组/MIDI，不运行仿真或GPU。已验证原曲哈希匹配训练config、400Hz事件重建一致、实际MIDI重建字节一致、归零文件回读误差小于0.1ms。物理力度仍未标定，实际velocity统一127，原曲17～61，不应用音量主观比较代替时序比较。

## 外部200Hz左臂CSV＋固定左手（2026-10-05）

用户提供`left_arm_200Hz.csv`并确认`marvin_0_arm_joint_0…6`依次对应当前预览Marvin M6左臂J1…J7，零位和正方向一致。新增`preview/scripts/replay_arm_csv.py`，读取time_s、七个q_rad、qd_rad_s及qdd_rad_s2；本次6201帧／0–31秒／5ms，数据全部有限，原始位置不插值、不裁剪、不重定时。40fps视频每5个源样本取一帧，保留最终样本，最后停留1秒。

视频使用本项目`preview/assets/tj_m6/left/arm.urdf`、现有shoulder/base变换、`configs/preview/hand_connector_20260930.json`及A3.2左手URDF/网格。仅左臂、左手和现有安装件；背景为本环境原Piano模型。左手20个实际指关节固定为当前场景initial_positions，已明确标注为原始预览姿态，并在data/left_hand_fixed_pose.json逐项保存。CSV没有手指轨迹，未附加训练策略或推断手指动作。

整臂与腕端跟随近景同步拼接，1440×720、40fps、原速、无音轨。使用已有用户目录OSMesa/llvmpipe软件渲染，强制CUDA不可见并检查software_renderer；不初始化GPU，不需要为此占用Slurm卡。GPU训练、推理或GPU渲染仍必须走submit.sh。源q/qd/qdd均按CSV保存，角速度/角加速度曲线不以位置差分替代；图上的红虚线采用现有SDK快照，只作规划限值参考。

```bash
LD_LIBRARY_PATH=/home/houj/.cache/tsensei-osmesa/runtime/usr/lib/x86_64-linux-gnu \
  LP_NUM_THREADS=4 training/.venv/bin/python preview/scripts/replay_arm_csv.py \
  --csv /absolute/path/left_arm_200Hz.csv --confirmed-j1-j7 \
  --output /absolute/path/new_output --fps 40
```

`--confirmed-j1-j7`仅用于已确认的同模型/顺序/零位/方向合同，不可用来跳过未知CSV映射。输入角度超URDF范围直接报错；原位置不投影到限位。导出会检查全部6201源角度精确保留，并在65个姿态对所有30个臂/手连杆独立比较URDF矩阵FK与MuJoCo实际渲染变换。

本次交付`preview/outputs/colleague_left_arm_200hz_20261005/`含MP4、三幅源角度/角速度/角加速度PNG、原始CSV、NPZ、逐视频帧源时间、固定左手姿态、场景/模型来源/映射记录、截图、index.html及replay_bundle.zip。模型/关节图均使用原CSV时间秒；视频结束停留不追加轨迹样本。

这是外部关节轨迹的运动学展示，不是训练／评估回放：没有checkpoint、原始MIDI或真实按键记录，原MIDI对比和F1不适用，不捏造演奏指标。肩座及法兰仍采用现有暂定安装变换，未宣称实机标定或动力学验证。本次不改变训练默认、训练归档或正在运行的训练。


## 2026-10-06：635最佳模型导出

635 best19500经单卡Slurm685录制，原控制/奖励合同保持。[有声视频](../preview/outputs/experiment_635_best_20261006/videos/experiment_635_best_19500.mp4)、[同次实际按键与原始MIDI对比](../preview/outputs/experiment_635_best_20261006/midi_comparison/midi_comparison.png)、[完整ZIP](../preview/outputs/experiment_635_best_20261006/export_bundle.zip)。本视频200Hz原谱音乐micro F1=92.89%，训练三回合最差93.23%另属训练指标。1280×720、20fps、10.2秒；1830真实物理子步，同次录制用于声音及对比，完整校验及来源记录在`logs/hpc/export_635_20261006/`。

## 2026-10-07：原网络目标与实际执行导出

`preview/outputs/network_targets_execution_20261007/`保存1103 gesture10-comparison best step19400在Slurm1359同次回放的原网络目标与执行结果，入口`index.html`、说明`README.md`，完整包为同名ZIP。此次只分析／绘制已有记录，不重新运行策略或物理。

- `network_actions_40Hz.csv`保存31维原网络action，`actuator_targets_40Hz.csv`保存46维最终ctrl，`left_execution_targets_40Hz.csv`为左20指关节及3虚拟腕通道。action不是全部关节物理单位的目标，映射／耦合／单位见`data/channel_mapping.json`。
- 原rollout的policy_time是周期末；首行目标作用区间[0,0.025)，对应实际端点t=0.025。ctrl每控制周期赋值一次，10个400Hz物理子步间保持。CSV给出interval_start/end；`left_targets_vs_actual_40Hz.csv`并列同周期目标与端点实测。
- `left_actual_joint_positions_400Hz.csv`为同次物理后实际位置，含t=0；`left_actual_joint_velocities_400Hz.csv`仅导出录制的3650子步速度，不伪造初始qvel。实际手根位姿另存`left_actual_hand_base_piano_400Hz.csv`；完整原始NPZ随包交付。
- `left_arm_hand_front.mp4`仅显示左臂／左手、朝演奏者、有声，使用原实际指关节／琴键和原左键声音。原训练no_arms=True，臂仅作固定基座IK显示；不是网络输出七臂关节或实际臂动力学。原回放双手运行，左键按同次接触分析归属，不称新的独立左手仿真。
- `actual_keys_vs_original_midi.png`、事件JSON及数组来自同次400Hz物理键程与匹配原始MIDI。音乐区间[1,8.875)，1秒准备、0.25秒释放，视频额外1秒末帧展示。本视频左键原谱micro F1=96.6121%；原双手400Hz原谱micro F1=94.2072%；双手40Hz端点与量化目标micro F1=92.6033%，三者分开保存于metrics.json。

未再次平滑原网络目标或实际手指轨迹；与此前`left_csv_video_20261006`规定运动下新积分琴键的实验不同。源PT/config/归档/MIDI SHA、CSV精确读回、F1独立重算、视频全片解码／音轨及manifest／ZIP验证见交付validation.json与`logs/network_targets_execution_20261007/delivery_validation.json`。

## 2026-10-07：实机下发参考／反馈的无动力学回放

用户附件`command_aligned_20261006_210206_8add5170_angle_all.csv`包含60102行、左臂7关节及左手20关节。交付在`preview/outputs/hardware_kinematic_replay_20261007/`，同步并排视频`command_vs_feedback.mp4`及独立`command.mp4`／`feedback.mp4`；相机面对演奏者，下方附手部俯视。当前基座、钢琴及手—法兰安装保持，仅调用mj_kinematics／mj_camlight，无IK重规划、物理步进、网络推理或实物发送；CPU OSMesa软件渲染。

按用户**最新**轴向说明，实机四指swing零度对应原URDF最大值，`q模型=0.35−角度×π/180`；这取代此前临时确认的“减20°”映射，其余关节直接deg转rad。同一映射用于下发、驱动参考及反馈。真实负swing反馈映射后略超上限，最大约0.00914°，保留原值，不静默裁剪；逐项证据在sampling_summary。

指令按指令时间零阶保持，实测按独立反馈时间去重，仅在≤30ms空段内线性显示插值；更大间隙、范围外或起始缺失的指节及其后代隐藏并标注，禁止以指令填补。原手反馈约1174个独立快照，不能把2226发送行当作200Hz手实测。视频t=0对应原1.268903879秒，源末帧12.995417858秒，额外约1秒末帧展示。

“参考”视频使用完整的实际下发位置；“驱动参考位置”另存CSV和三色曲线，原左臂及拇指两个旋转关节全缺，未拼接下发值冒充完整驱动参考。原CSV、去重原时间采样、模型映射、固定场景、40Hz显示数据、逐帧缺失状态及关节误差指标随包保存；误差为显示采样口径。文件没有真实琴键状态／MIDI来源或声音，钢琴只是静态背景，本次静音；实际按键对比及音乐F1缺少依据，未生成或挪用其他回放结果。

### 同日追加：按授权以前帧反馈补空，再从完整组导出

用户后来要求实测位置为空时沿用上一帧。最新版angle_all与上一份SHA一致；追加self_9的2226行是相同中指PIP副本，没有补入新反馈。新交付独立保存于`preview/outputs/hardware_ffill_replay_20261007/`，旧视频不改。

逐关节只填`实测位置`，补111项；开头15项无前值不猜填。跳过前11组，从第12组的完整指令时间1.326636539秒开始，到12.995417858秒；原格式`filled_selected_angle_all.csv`保留2215组／59805行，27关节实测位置字段均非空，新增填充方式／原有效反馈时间两列。9个原本全空的驱动参考通道保持未知。全部原非空实测、指令／时间等其他字段不改。

真实快照间保持已有≤30ms显示插值规则；出现授权补值的区间保持上一帧有效实测，不提前插值到恢复后的新值。映射仍是四指swing `0.35−实机角度×π/180`，当前安装固定。新参考／反馈并排与两个独立视频不再隐藏手指，画面标注Held和逐帧补值状态；完整原文件、合并／填充来源、完整填充表、裁切表及40Hz绘制数据可复核。没有新按键记录或琴键接触仿真，本次仍静音，不产生音乐F1；关节角差指标含补值，不能当作全频实测误差上界。


### 同日追加：严格固定机器人轨迹，仅琴键接触动力学

用户选择“方式一”，要求实测臂／手指不得被动力学改变。新交付 `preview/outputs/hardware_key_contact_20261007/`（同名ZIP）在Slurm1877一次分别回放参考与反馈：22个左手连杆由原位置FK规定为独立mocap，物理状态只含88个琴键自由度，机器人自由度为0；没有机器人PD、动力学更新后复位或接触修正。每一步mocap逐值不变，输入27关节与规定FK文件SHA不变。当前钢琴／基座／法兰安装、最新swing映射及上一轮前值补空合同保持。

- `command_vs_feedback.mp4`左右并列参考／反馈，声音来自反馈列；`command.mp4`及`feedback.mp4`各有对应本次琴键声音。只有左臂与左手，前视面对演奏者，下方手俯视；40fps507帧12.675秒，原CSV1.326636539–12.995417858秒，其后约1秒仅显示末帧。
- 原生时间直接重建400Hz，参考零阶保持、真反馈短段插值、补值区间前值保持。每个回放4668物理步／4669键状态，末步0.001281319秒；400Hz是仿真，不是硬件反馈采样率。原填充CSV、27关节400Hz CSV、来源／不变证据随包。
- `command_actual_keys_vs_original_midi.png`、`feedback_actual_keys_vs_original_midi.png`与对应视频／声音同一次新琴键记录，原谱、真实模拟按键、叠加三层及事件／数组齐全。按键由真实key_qpos距键底0.00872665rad阈值重建，不以动作／指尖投影代替；声音取这些边沿，固定力度127，无踏板。
- 全段400Hz原谱micro F1：参考27.030539%、反馈24.927085%；音乐区间另为27.975790%／25.446606%。原始MIDI率1.0及时值不变，归档指法5–9左手。CSV缺谱播放时钟，按下发臂轨迹与原规划匹配估计音乐起点视频3.104971321秒，两条使用同一估计；不能当作硬件同步评估。
- 固定轨迹在部分黑键限位处穿入约7.2–7.4mm，原软限位有超程，峰值求解力约15万N；没有改机器人位置或调参来修正。这些力不可信为实物受力，F1是该受约束模型诊断，实际击键／声音与实物标定仍需另验证。接触形状来自匹配原1359几何，与当前训练默认代理不同；PT/config/归档来源只用于旧规划／谱／几何追溯，没有重新跑checkpoint。

执行记录在 `logs/hardware_key_contact_20261007/`：prepare生成不可积分的机器人＋琴键only XML；test_record经submit.sh validate分配录制；postprocess只读同次键记录生成声音／图／F1，render强制CPU软件绘制、读取相应原键程；finish独立重建原生角／填充、激活／F1／音频事件及原MIDI／归档指法，核对三个音视频全片507帧、manifest／ZIP。验收 `validation.json`、`media.json` 与 `data/trajectory_invariance.json` 可复核。结果ZIP含数据与资产SHA／项目路径依赖，不是无依赖模型安装包；当前训练和之前各版视频保持。


### 同日追加：原rollout下发ctrl与原实际qpos，固定机器人对比

用户确认两条都导出，交付`preview/outputs/rollout_prescribed_comparison_20261007/`及同名ZIP。来源1103 best19400在1359的原rollout/匹配400Hztrace；没有重新运行网络。ctrl按原40Hz周期保持，actual是原400Hz物理后的手指／虚拟腕状态；原rollout40Hz端点另存，全部原模型弧度，不套实机swing零位映射。原1秒准备／lift、0.15rad安装倾角、原谱及时值保留。

当前基座和手／法兰安装固定。原no_arms训练无七臂关节，分别用各自腕根位姿IK得到J1–J7，最大位置误差约1e−8m、旋转1e−7rad，指关节原值不变；ctrl首目标不等于原实际初始位置，未增加平滑连接。物理模型只有88琴键关节，机器人自由度0、22手连杆mocap规定FK，无PD／力反馈或动力学复位。经submit.sh单卡validate Slurm1884分别新录键程，逐步mocap/输入角度/SHA不变。

- `targets_vs_actual.mp4`左下发ctrl／右原实际qpos，对比声音来自右列；`targets.mp4`与`actual.mp4`各自有本次声音。405帧40fps10.125秒，原轨迹9.125秒及1秒末帧展示；仅左臂左手，前视面对演奏者，下方手俯视。
- `command_joint_positions_400Hz.csv`／`feedback_joint_positions_400Hz.csv`为七臂IK＋20原指；`left_native_ctrl_40Hz.csv`、`left_native_rollout_qpos_40Hz.csv`、`left_original_actual_400Hz.csv`为原23虚拟腕／指通道，单位与采样语义见channel_mapping。
- 两张`*_actual_keys_vs_original_midi.png`及事件／数组／原MIDI、音频、原NPZ同目录。按键取本次相应400Hz真实键程阈值，声、图、绿色键及F1同源。音乐窗口[1,8.875)原谱micro F1：ctrl22.096850%、actual95.578340%，全段分数相同，不能沿用旧训练／录制的分数。
- ctrl规定位置压过琴键软限位，偏差最高0.238271rad／求解力206394N；actual也有0.0288071rad／2466.72N的软限位偏差／求解力。未裁剪机器人或调参提升F1，这些力非硬件力估计，F1是该规定运动琴键模型的诊断。声音固定力度127，无踏板或力—响度标定。

验收原PT/config/源码/原MIDI/指法和录制来源、当前安装和模型资产SHA，原ctrl与actual角度/CSV逐值、IK误差、0机器人自由度与各步不变、同次键激活/F1/音频事件/MIDI一致；三个H264/AAC视频全片405帧/40fps、原谱时序PNG、64交付文件manifest与ZIP CRC/SHA通过，日志`logs/rollout_prescribed_comparison_20261007/`。包保留项目绝对资产依赖，不是无依赖模型安装包；旧视频、当前安装和训练配置不变。


### 2026-10-08：上传ZIP实机轨迹接入三路工作台

交付 `preview/outputs/wzx_import_20261008_1412/`，启动命令和合同见 `trajectory_compare/README.md` 上传数据测试一节。新 `--dataset` 只加载校验过的离线导出目录，不自动把未知上传轨迹送到GPU。第一路仅20手指原物理驱动、机械臂规定；第二下发与第三反馈机器人0动态DOF，只有88键动态，保持当前冻结安装。源日志各关节自身时钟ZOH，缺值保留同关节上个实值；初始未知跳过，源1.907642593–90.000458526s无时钟缩放；400Hz物理录制Slurm1957/25fps有声EGL视频Slurm1960。每路2228帧89.12s（尾部展示hold），同次真实键激活产生音轨与实际按键PNG；视频/曲线/数据按SHA绑定。

**原始MIDI与音乐同步起点未提供、项目候选无法核实身份，故本次F1为空，没有完整原谱对比交付。** 页面与PNG明确缺项，不套用旧小星星MIDI或分数；完整400Hz键程和按键事件已留存，可随后用同次数据补图／F1。规定机器人仍可能压过琴键软限位，求解力非硬件估计。小指规划旧名映射和四指swing转换经全序列核验，保留证据；当前训练源码/归档/checkpoint及旧导出未改。

## 左手四指swing最大值静态图（2026-10-08）

`preview/outputs/left_swing_max_20261008`提供正面/俯视PNG及关节角对照：以74s规划首帧为基准，仅四swing设为URDF上限+.35rad，对应当前SDK输入0deg；其余关节和安装不变。Slurm2031通过统一入口渲染，未积分动力学或修改轨迹。复核来源/角度及图片SHA见该目录report/manifest；静态图不作为按键视频或评分。
