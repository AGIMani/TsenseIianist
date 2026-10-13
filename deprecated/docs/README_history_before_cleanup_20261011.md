# TsenseI A3.2 钢琴环境

## 当前活动设计（2026-10-11）

当前只维护一个机械臂控制合同：`controller=rigid_wrist_command`。策略腕部目标经过连续 IK、目标制动和 SDK/URDF 硬限，沿 Link7—法兰—四孔连接件—手的刚性链执行。带臂模式的 reset 初始角使用 `configs/preview/current_arm_training_20261011.json` 中的双侧七角；默认训练仍为无臂。

当前 humanoid 手指奖励只有一套公式：闲置手指相对 reset 目标的 L1 惩罚，参数为 `coefficient=-0.1`、`reference=scene_reset`、`lookahead_steps=20`。旧机械臂方案、旧奖励模式、旧入口和候选产物已移入 [`deprecated/`](deprecated/)，当前入口不会读取它们，也不做旧 checkpoint 转换。

新的设计直接修改活动源码、配置和文档；不要再新增递增的 schema 或 mode 名称。 活动目录中仍存在少量 `schema` 字段用于 JSON/检查点序列化格式（例如控制限幅、指尖耦合和导出报告）；它们不是机械臂或 humanoid 奖励的候选算法。训练设计要求使用 [`all_use_this_train.sh`](all_use_this_train.sh)，但当前工作区未包含该文件；本轮不要把历史入口当作替代入口。恢复统一入口后，每次开始训练必须打印完整奖励配置。当前规则和本轮变更见 [`AGENTS.md`](AGENTS.md) 与 [`docs/change_log.md`](docs/change_log.md)。

---

以下内容是历史记录，仅供追溯；其中的旧配置、旧实验和旧数值不代表当前环境。

# TsenseI_A3.2 独立弹钢琴环境

## 当前训练入口与机械臂状态

**MJLab预览已与带臂训练合同同步**：使用 [当前双臂预览配置](configs/preview/current_arm_training_20261011.json) 和 `preview/scripts/preview_mjlab.py`，预览会从同一 `source_runtime` 重建场景／乐谱，并传入相同的奖励、控制限幅、准备段、手指耦合和 当前刚性臂控制 刚性双M6合同。它不再调用旧的独立腕部IK预览。桌面查看（需图形会话）运行：

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --training-config configs/preview/current_arm_training_20261011.json
```

要判断不可达位置，用固定归档通过Slurm运行无窗口扫描：

```bash
bash scripts/hpc/submit.sh --gpus 1 --source-archive \
  logs/mjlab_preview_sync_20261011/source_mjlab_preview_sync_v7.tar.gz mjlab-preview \
  --training-config configs/preview/current_arm_training_20261011.json \
  --reachability-audit "$PWD/logs/arm_workspace_audit_20261011/mjlab_preview_scan"
```

输出目录含左右侧CSV、PNG和 `summary.json`。`solved` 是当前IK的数值见证，`proven_outside_bound` 是保守Link7外界之外的几何证明，`unresolved` 仍需路径、速度／加速度和碰撞检查；不能把跟踪暂态误差当作几何不可达。预览窗口还会显示腕部请求到实际FK的毫米／毫弧度误差。

**当前双臂安装图与角度余量**：按同一配置的 reset `q₀` 导出了 [左右安装图与余量包](preview/outputs/arm_installation_limits_20261011/README.md)。图中标出 `Base → J1…J7 → 固定法兰 → 四孔连接件 → 手根`，并按局部 +Z 右手定则标注每个关节的顺时针／逆时针剩余角度；精确弧度、限位、J6/J7 耦合余量和来源散列在 [joint_margins.csv](preview/outputs/arm_installation_limits_20261011/joint_margins.csv) 与 [installation_report.json](preview/outputs/arm_installation_limits_20261011/installation_report.json)。复现命令：

```bash
MUJOCO_GL=disable training_mjlab/.venv/bin/python \
  preview/scripts/export_arm_installation_limits.py
```

**当前默认左臂镜像角度与掌面朝下安装（2026-10-11）**：按最新要求，左侧 J1–J7 的角度绝对值复制右侧，左侧符号取 `[+,-,-,-,+,+,-]`，所以 J5/J6/J7 的绝对值分别与右侧完全相同。固定当前左基座，求解新的左侧 Link7→手安装变换；主导法兰时钟约为工具局部 Z `−90°`，并包含 URDF 链造成的平移／小角度补偿。[真实网格交互 HTML](preview/outputs/arm_left_mirror_mount_candidate_20261011/left_mirror_mount_candidate.html) · [真实网格静态图](preview/outputs/arm_left_mirror_mount_candidate_20261011/left_mirror_mount_candidate.png) · [候选报告](preview/outputs/arm_left_mirror_mount_candidate_20261011/candidate_report.json) · [候选输入包](preview/outputs/arm_left_mirror_mount_candidate_20261011/left_mirror_mount_candidate_bundle.zip)。页面包含实际 M6 连杆、四孔连接件和 A3.2 手部网格，以及 J5/J6/J7 数值面板。候选左指尖平面倾角 `0°`，Link7→手误差约 `1.16e−15`。当前双臂预览配置已经采用这组角度与安装变换；固定训练源码归档仍保持不变，四孔紧固、法兰时钟和实机编码器零位需要硬件复核后再生成新的训练归档。旧基线导出保留在 `arm_mirror_palm_down_scene_20261011`。

```bash
MUJOCO_GL=disable training_mjlab/.venv/bin/python \
  preview/scripts/export_arm_mirror_palm_down_scene.py
```

**同手碰撞惩罚更新**：`self_collision_reward`权重0.2，最终每控制步最多扣5：`−min(0.2×同手接触对—子步数,5)`，25对次起封顶。无臂／带臂共用；计数和物理过滤保持，旧不封顶模式仅由历史归档回放。源码归档及唯一奖励总表已同步，本次未启动训练。

新训练唯一入口为根目录 [all_use_this_train.sh](all_use_this_train.sh)，通过Slurm双卡采样／学习，默认无臂；显式`--arms`并提供双侧七角后启用当前当前刚性双M6。带臂在线IK同时执行CCS696的`1.025×|J6|+|J7|≤110.5°`耦合约束，并在每个200Hz下发步保留速度／加速度／位置硬限。训练固定源码仍是`logs/arm_ccs696_coupling_20261011/source_arm_ccs696_coupling_v5.tar.gz`，路径与SHA由根入口校验，工作区修改不会自动进入训练；MJLab预览使用上方单独的 `mjlab_preview_sync_v7` 归档。

**本地开发包（2026-10-11）**：可用 `scripts/hpc/package_a32.py --include-preview-assets` 生成独立源包；包内包含代码、配置、测试、文档、预览脚本和 M6/连接件资产，不含虚拟环境、训练输出、checkpoint、缓存或历史归档。解包和“本地开发、远程 Slurm 训练”的流程见 [本地开发与远程训练交接](docs/local_development_handoff_20261011.md)。

**2026-10-10晚间核对：expe00001和expe00002均已取消。** expe00002最后记录到6000步／307.2万样本；保存best为5000步，200Hz原谱音乐micro F1三回合合并1.13%、最差0%，属于原归档结果。

**刚性臂父子接触过滤已修复，无臂自碰撞保留。** 从原模型自动恢复因手根改挂mocap而丢失的20对父子排除，初态异常12对接触消除；非相邻手指、跨手及手—琴键接触继续检测。38项CPU回归及Slurm2140的5项GPU验收通过：无臂／带臂测试均检测5对非相邻自碰撞，对相同物理接触缓冲做10次子步采样按原规则扣37.5分；正常带臂初态同手罚为0。新合同拒绝直接混用修复前PT；证据见`logs/arm_parent_filter_fix_20261010/`。运动罚饱和及长期演奏效果仍需后续验证，本次未开启正式训练。

[当前环境、入口和机械臂实验总览](docs/training_environment.md)说明两个Python环境的用途、当前无版本机械臂合同、已保存模型及待解决问题。奖励仍只维护[唯一奖励总表](docs/reward_defaults_20261006.md)，本轮实际配置读取对应`config.json`。旧底层入口不带统一覆盖时仍引用旧schema奖励，不能当作新训练备用入口。

## 历史更新与验收快照

下方按日期保留原实验、启动阶段与验收记录。“已启动”“运行中”“未开启长期训练”和各轮默认值均指记录当时；实时状态、未来入口及当前机械臂方案以本页上方和环境总览为准。

**2026-10-10 13:58：expe00002刚性双M6训练已启动（Slurm2109）**。双A100，从随机初始化训练10,000,384样本，沿用用户双侧初角和当前奖励／双硬限；腕目标主导连续IK，手—四孔连接件—法兰刚性固定。运行config已核对31动作／1461观察、512环境／20Hz控制／200Hz物理、双卡采样学习流水、9项奖励开启；当前启动阶段先初始评估及93,696样本经验池预填。[本轮配置](training_mjlab/outputs/hpc/expe00002-20261010-135845-564168/config.json) · [完整奖励](logs/expe00002/final_reward_config.json) · [启动记录](docs/optional_arm_wrist_ik_requirements_20261008.md#2026-10-10expe00002-首轮刚性装配训练已启动)。

**2026-10-10 刚性装配修复后正面测试视频**：[28.1秒合并视频](preview/outputs/rigid_arm_test_20261010/tests_combined.mp4) · [查看页](preview/outputs/rigid_arm_test_20261010/index.html) · [同次实际按键／原MIDI图](preview/outputs/rigid_arm_test_20261010/midi_comparison.png)。按用户指定演奏者前方正面，左手／左臂及连接件近景；42cm左移停止、48cm远请求返回。两组200Hz预设动作测试，停止／返回后0.355／0.945秒停稳，SDK双硬限与刚性连接验收通过；保留拖键和重复触键，未运行策略／训练。[导出范围与验证](deprecated/docs/reward_modes/video_export_cli.md#2026-10-10刚性装配修复后的正面测试视频)。

**2026-10-10 左腕加速跟随测试**：[视频](preview/outputs/left_wrist_ramp_20261010/performer_view/replay.mp4) · [运动曲线](preview/outputs/left_wrist_ramp_20261010/motion_overview.png) · [同次按键／原MIDI图](preview/outputs/left_wrist_ramp_20261010/midi_comparison/midi_comparison.png)。当前环境预设动作、未训练：实际左移42cm，腕速峰值0.367m/s；停止后机械臂振荡，最大间隙119.36mm、末态46.10mm，SDK双硬限通过。[原因定位与隔离对照](deprecated/arm_control_experiments/schema2_oscillation.md)：停止超调触发近伸展／短暂不可达；仅优化下一步位姿、缺少目标制动使臂反复过冲。固定目标延长CPU对照后会收敛，原记录与诊断保留；当前修复见下方刚性装配合同；[测试范围与数据](deprecated/docs/reward_modes/video_export_cli.md#2026-10-10左腕逐渐加速机械臂跟随测试)。

**2026-10-10 expe00001最新视频已导出**：训练已中断，保留第500步／25.6万样本模型。[双M6有声视频](preview/outputs/expe00001_step500_20261010_v2/full_scene/replay.mp4) · [同次200Hz实际按键／原MIDI图](preview/outputs/expe00001_step500_20261010_v2/midi_comparison/midi_comparison.png)。单视频原谱音乐F1为5.23%，完整原始数据、HTML和ZIP已验收；[流程与边界](deprecated/docs/reward_modes/video_export_cli.md#2026-10-10expe00001中断后的最新双机械臂视频)。

**2026-10-10 expe00001 已按用户要求中断（Slurm2075）**：已保留第500步forward.pt（256,000样本／40,576更新）。原计划：用户指定左臂七角、右臂按完整安装镜像，从头10,000,384样本；双A100、paper／无和弦放大／独立误触罚关闭，开启机械臂SDK双硬限与独立最多−5/步运动罚。实际config、28维新增臂观察及镜像初态核验通过；[初始角、基座与记录](docs/optional_arm_wrist_ik_requirements_20261008.md#2026-10-09expe00001-首轮带臂训练)。

**2026-10-08 当前按键默认更新**：统一入口读取外部奖励JSON，按键切换`paper`（0.6正确键行程＋0.4指数错键分），和弦放大关闭，独立`false_press_penalty`关闭（保留未启用weight=−.05）；指法仍为legacy，其余训练设置保持。本次仅修改未来默认，未启动训练；完整配置见[唯一奖励总表](docs/reward_defaults_20261006.md)。

**2026-10-10 刚性装配与机械臂制动修复**：[当前控制合同与验收](docs/optional_arm_wrist_ik_requirements_20261008.md)。策略腕目标主导连续IK，目标制动与SDK双硬限生成臂轨迹；法兰—四孔连接件—手根结构固定，保留安装偏移，手指／琴键继续物理。移除竞争的腕PD，动作仍31，新增54维实际臂状态／参考／误差观察；独立臂运动罚仍最多−5/步。默认无臂，缺双侧七角拒绝；当前刚性臂控制不直接接续旧PT。CPU及A100验收通过：42cm左移停止后0.355秒停稳，48cm远请求返回后0.945秒停稳；固定装配数值残差最大约0.000031mm。两层目标制动、掌部互碰诊断补漏完成，源码归档与统一入口已同步；[验证曲线](preview/outputs/arm_follow_repair_20261010/validation.png)。这些是预设请求验收，未开启长期训练。

**当前奖励唯一总表**：[奖励与惩罚完整参数表（持续维护）](docs/reward_defaults_20261006.md)，2026-10-08核对统一入口＋固定归档：22项，默认无臂8开14关；显式启用机械臂时默认额外开启独立机械臂运动罚。逐项列出权重、全部参数、公式、单位、采样和启用条件。历史归档保留用于旧模型回放。当前目标/奖励时序沿用Shadow合同，未启动生产训练。

**2026-10-08 TCP目标运动限制**：当前统一入口已改为双侧TX/TY逐轴硬速度0.99m/s、硬加速度1.99m/s²；惩罚阈值0.9m/s、1.90m/s²。按机械臂同样的“限幅前需求对前次下发目标/速度”计算TCP超限，与手指/腕旋转共享10%超限扣满−5/步的预算。固定新归档`source_tcp_motion_limits_v2.tar.gz`；公式见[唯一奖励总表](docs/reward_defaults_20261006.md)。本次数值核验在`logs/tcp_thresholds_20261008/`，父算法CPU验收在`logs/tcp_motion_limits_20261008/`，GPU待Slurm验收，未启动训练。

**2026-10-08 闲指窗口与系数**：当前MIDI指法及未来20控制步（20Hz下1秒）均未用到才算闲指；L1系数改为−0.1、weight=1，无整项−0.1封顶。第20步用指即豁免，曲尾不回绕；策略观察长度及其他训练参数保持。该轮归档为`source_idle_lookahead20_v1.tar.gz`，旧归档/作业保持；CPU验证通过，GPU待Slurm验收，本次未启动训练。

**2026-10-08 Shadow目标/奖励时序修复**：训练目标和指法统一采用控制频率量化谱，第k步动作始终对动作前观察的第k帧目标评分，实际状态取周期末；原始MIDI与200Hz诊断独立保留。已删除精确端点换目标、公共子步平均及正确键激活率混合开关，奖励schema升级2并拒绝旧合同。当轮归档为`source_shadow_score_sync_v1.tar.gz`；同手子步累计、奖励权重和腕硬限制保持。CPU回归及真实乐谱导出通过，GPU集成待Slurm验收，未开训。细节见[唯一奖励总表](docs/reward_defaults_20261006.md)。

**2026-10-08 同手碰撞按物理子步累计**：后续统一默认启用`substep_sum`，同一body对在每个物理子步内去重、跨子步累加；weight=.2时连续10个子步扣2，当前默认weight=.75则扣7.5。周期末刷新不重复扣分，闲指/其他奖励保持；旧config及归档保留周期并集合同。[实现、兼容与验证](docs/self_collision_substeps_20261008.md)。新默认归档已固定，GPU集成待Slurm验收，未启动训练。

**2026-10-08 新增训练1931**：paper按键、旧闲指整项最多−.5、手指目标惩罚阈值20/400，从头10,000,384样本；双A100、20/200Hz、无臂、旧子步奖励合同。匹配归档与实际配置核对通过，完整奖励已打印；[参数、提交和验收](configs/experiments/paper_idle05_motion20_400_20261008/README.md)。

**2026-10-08 实验与F1总对照**：[完整分组表及结论](docs/experiment_comparison_20261008.md)、[50项实验CSV](logs/experiment_comparison_20261008/experiments.csv)。覆盖奖励、环境/频率、初始手型、机械臂、运动限制、初始化和预算，区分训练最差/合并F1及独立复评；同预算单因素结果支持paper按键公式、减轻旧闲指罚和放宽手指目标惩罚阈值。729/988最终结果本地缺失，当前L1/端点新默认尚无完整长期实测；未修改训练默认或启动训练。

**2026-10-07 单左手和弦接触测试**：[网页视频与力曲线](http://127.0.0.1:18771/index.html)、[本地导出](tactile_benchmark/outputs/single_left_chord_20261007/index.html)。按用户确认的小指C4／中指E4／拇指G4摆位，右手从物理场景移除；蓝色指定键、橙色实际激活键。5档下压仅触发指定三音，保留0.75秒无接触预备段、同次200Hz逐指—逐键力及原谱对比图；109项CPU测试和数据／视频／音轨核验通过。固定诊断轨迹，无学习训练；[复现与边界](tactile_benchmark/README.md#当前实验单左手摆位激活琴键与接触力)。

**2026-10-07：三路轨迹对比网页框架**：运行 `online_control/.venv/bin/python -m trajectory_compare` 后打开 <http://127.0.0.1:8787>。支持文件选择、三视频／F1、同步播放、指尖／手关节点空间曲线及横纵轴独立缩放；已载入三份验证过的历史导出，新文件可解析并查看曲线，新的 Slurm 仿真视频导出器待接入。[使用与接口说明](trajectory_compare/README.md)。

**2026-10-07 和弦 velocity／接触力测试**：[五组有声视频与旁边的逐键接触力曲线](tactile_benchmark/outputs/chord_velocity_force_20261007/index.html)、[完整 ZIP](tactile_benchmark/outputs/chord_velocity_force_20261007/export_bundle.zip)。共同琴键 MIDI 56 实测 velocity 为 11/26/48/111/127；附 200Hz 手—键求解力、同次实际／原始 MIDI 对比图和 CSV。固定同步下压仍无法准确演奏 C 大三和弦（五组 F1=0）；页面另含指定和弦六档音源试听及混合 velocity 声部测试，不为合成音频配造力。106项CPU回归、原始键程重建事件、视频音轨及文件／ZIP校验通过。[复现说明](tactile_benchmark/README.md#和弦-velocity-与同次接触力2026-10-07)。

**2026-10-07：986最佳模型追加1500万样本，Slurm988已开始学习**。从best19532（F1 91.58%）恢复完整学习状态，保持原奖励/20与400惩罚阈值/双硬限制，追加15,000,064样本，经验池重建；已新增32,768次更新，数值正常。[来源与验证](docs/h100_training_20261001.md)。

**2026-10-07 13:58：986／987正常完成**。原谱音乐三回合最差：986从头best/final91.58%，987从976微调best96.69%/final96.36%。两组各1000万样本，数值正常；仍有同手接触子步，跨手/键底评估0。[核查结果](docs/h100_training_20261001.md)。以下为历史快照。

**2026-10-07 benchmark音频接入**：[六档velocity试听、真实物理测速单键试听及有声双组视频](tactile_benchmark/outputs/velocity_audio_20261007_listen/index.html)。独立FluidSynth＋TimGM6mb、固定增益0.8；同音高同音长的velocity32→127对应数字RMS提升约24dB，原物理记录音高56的11/26/48/111/127也随力度递增。原force/none视频按同次事件配音，保留原静音文件及按键图，无新物理回放；原快速动作均velocity127，两组声音相同。新测试可加`--audio --audio-gain 0.8`；[使用说明](tactile_benchmark/README.md#velocity控制钢琴声音2026-10-07)。

**2026-10-07 触觉benchmark双组测试导出**：[连续接触力／无触觉视频及同次按键对比图](tactile_benchmark/outputs/contact_comparison_20261007/index.html)、[完整ZIP](tactile_benchmark/outputs/contact_comparison_20261007/export_bundle.zip)。各13.25秒、200Hz物理、20fps静音视频，附原始接触力CSV和实际MIDI。相同固定诊断动作下两组物理轨迹一致，单末节法向力峰值19.9818N，无触觉观测全零；两组F1=0，未加载学习策略，不能用作触觉收益结论。CPU软件渲染与同次记录校验说明见[tactile benchmark](tactile_benchmark/README.md#接触力视频与无触觉对照2026-10-07)。

**2026-10-07：手指目标运动惩罚默认阈值更新为20rad/s、400rad/s²，986／987双实验已启动**。986随机初始化，987从976 best19000继承网络；两组均保留976闲指−.1和其余配置，各1000万样本。奖励改变，987重建优化器/温度/随机状态及经验池。[配置与验收](configs/experiments/finger20_400_20261007/README.md)。

**2026-10-07：976视频与F1曲线已导出**。[有声视频](preview/outputs/experiment_976_best_20261007/videos/experiment_976_best_19000.mp4)、[同次实际按键／原谱对比](preview/outputs/experiment_976_best_20261007/midi_comparison/midi_comparison.png)、[训练F1](docs/figures/experiment_976_f1_20261007/f1.png)、[完整ZIP](preview/outputs/experiment_976_best_20261007/export_bundle.zip)。best19000原合同回放，单视频200Hz音乐F1 81.57%；训练best82.14%、final81.86%另属三回合最差。

**2026-10-07 12:25：632闲指−0.1对比976正常完成**。1000万样本，best原谱音乐F1三回合最差82.14%、final81.86%，比632原闲指−1最佳提高4.36个百分点；best/final接触检查为0，仍有漏音。[结果与核查](docs/h100_training_20261001.md)。以下为历史快照。

**2026-10-07 08:45：632闲指−0.1对比实验976已在双H100启动**。从随机初始化开始，只将632闲指整项每步最大惩罚−1改为−0.1；其余奖励、双硬限制、源码、seed42及10,000,384样本预算保持。GPU和实际完整配置核验通过，08:47已19,968次学习更新，数值正常。[配置与验证](configs/experiments/legacy_idle01_compare_632_20261007/README.md)。以下为历史快照。

**2026-10-06 20:32：683续训729已开始学习**。从683 best19000（F1 89.23%）恢复完整学习状态，保持原奖励/硬限制/训练配置，双H100再训练10,000,384样本，经验池重建。已新增40,576次学习更新，数值正常。[配置与恢复验证](docs/h100_training_20261001.md)。以下为历史快照。

**2026-10-06 19:41：683最佳模型视频已通过A100导出**。[有声视频](preview/outputs/experiment_683_best_a100_20261006/videos/experiment_683_best_19000.mp4)、[同次实际按键／原始MIDI对比图](preview/outputs/experiment_683_best_a100_20261006/midi_comparison/midi_comparison.png)、[完整下载](preview/outputs/experiment_683_best_a100_20261006/export_bundle.zip)。Slurm1797完成，保持原PT/config/源码及硬限制，单视频200Hz原谱音乐F1=89.77%。H100排队718已取消，原查看页转到此次A100成果；不是训练三回合89.23%的替代指标。

**2026-10-06 18:46：632续训683已正常完成**。追加10M样本，best19000原谱音乐F1三回合最差89.23%，final89.18%，较父632最佳提高11.45个百分点。best评估接触均0，final仍有7个同手接触物理子步。[结果与F1曲线](docs/h100_training_20261001.md)。以下运行状态为历史快照。

**2026-10-06 17:25：632续训Slurm683已开始学习**。从632 best19500恢复网络、优化器、温度及随机状态，再训练10,000,384样本，经验池重建；legacy、闲指−1、目标运动惩罚、hard_v1及Pitch双硬限制均保持。双H100已新增17,408次更新，数值正常。[配置、来源与恢复验证](docs/h100_training_20261001.md)。以下为历史快照。

**2026-10-06 17时检查：632／635／638全部正常完成10M训练**。统一200Hz原谱音乐三回合最差F1：最佳分别77.78%／93.23%／87.77%，最终77.45%／92.49%／86.90%；638适应新限制后明显恢复，当前仍低于同设置随机初始化的635。各best／final评估同手、跨手及键底接触均0。[结果、曲线及比较边界](docs/h100_training_20261001.md)。以下运行状态为历史快照。

**2026-10-06 14:49：582模型续训Slurm638已开始学习**。从582 best13500加载网络，保留paper／闲指整项−.1，新增目标速度/加速度超限惩罚，按用户确认保留Pitch目标3.1416rad/s／15.708rad/s²双硬限制；优化器、温度及经验池重新初始化。已产生19,968次更新。新限制下初始F1约6.99%，适应效果待评估。[完整奖励、来源与启动验证](configs/experiments/continue_582_motion_20261006/README.md)。

**2026-10-06 14:41：新增对照Slurm635已开始学习**。沿用582的paper按键和闲指整项−.1/步，奖励仅新增目标速度/加速度超限惩罚；与632相同随机初始化、seed42、双H100预算及腕Pitch目标3.1416rad/s/15.708rad/s²硬限制。已产生2,048次学习更新，632继续运行。[两组差异、完整奖励及验证](configs/experiments/paper_idle01_motion_20261006/README.md)。

**2026-10-06 14:34：新实验Slurm632已开始学习**。以582配置从随机初始化训练：按键legacy、闲指整项最多−1/控制步、开启目标速度/加速度超限惩罚；双侧腕Pitch最终目标硬限制同时开启3.1416rad/s和15.708rad/s²，位置hard_v1保持。其余582设置逐项核对保持，双H100已产生12,288次学习更新。[本轮完整奖励、配置和验收](configs/experiments/legacy_idle1_motion_20261006/README.md)。以下记录为历史快照。

**2026-10-06 独立触觉 benchmark**：新增 [tactile_benchmark](tactile_benchmark/README.md)，包含独立Python环境、固定A3.2源码副本、CPU MuJoCo/Gymnasium环境、三种触觉观测和逐音符MIDI velocity评测。真实键程/接触力、同次MIDI对比PNG及来源SHA可导出；94项CPU测试、目录独立复制与三组物理链路验收通过。它采用独立46维接口，velocity为待标定代理；不兼容现有31维训练checkpoint，未启动学习对照。现有统一训练入口和归档保持。[验收记录](tactile_benchmark/validation/summary.json)。

**2026-10-06下午：620续训已完成，本轮五组视频汇总**。620于10:08:25正常结束，新增10,000,384样本，best精确F1 **94.7410%**、final **92.0825%**（200Hz原谱音乐区间3回合最差值），best较父585提高9.6319个百分点。580／581／582／585／620全部完成；[五组最佳模型有声视频及同次实际按键／原MIDI图](preview/outputs/all_training_best_videos_20261006/index.html)、[完整ZIP](preview/outputs/all_training_best_videos_20261006/all_videos_and_comparisons.zip)。[核查与导出记录](docs/h100_training_20261001.md)。以下运行／空闲状态为历史快照。

**2026-10-06 585续训已启动：Slurm620**。从585 best19500恢复网络、优化器、温度和随机状态，双H100再训练10,000,384样本；经验池及物理场景重新初始化。保持585原奖励、82199601…bda022归档、hard_v1及Pitch目标运动限制，未应用后来本地默认奖励修改。配置及恢复来源已核验，实际更新见[H100续训记录](docs/h100_training_20261001.md)与`logs/hpc/continue_585_20261006/startup_validation.json`；下方08:46空闲状态为此前快照。

**2026-10-06 08:46 训练检查**：H100的580／581／582／585均已正常完成各10,000,384样本，当前队列为空、8卡idle。best三回合最差200Hz原谱音乐micro F1依次99.2328%／99.2111%／99.3870%／85.1091%；585从头训练，前三组由父模型迁移，不能直接归因比较。四组实际配置与启动快照一致；下列最新默认奖励尚未提交训练，不能把585视为新奖励实验。[完整结果与实际／默认配置对照](docs/h100_training_20261001.md)。以下“运行中”均为历史快照。

**2026-10-06 最新默认奖励**：统一入口使用 `configs/rewards/default_20261006.json`。同手碰撞默认开启，每去重body接触对−.75；误触开启，负权重−.05乘非负行程代价；闲指姿态改为weight=1的`−.04×Σabs(最终下发目标−初始目标)`。删除通用`collision_reward`和`natural_posture_reward`，新源码拒绝含删除项的旧JSON。公共采样改为Shadow控制周期末计分、无dt缩放、关闭子步奖励及触发占比混合，200Hz诊断保留。`--finger-coupling`仍开启，末尾`--no-finger-coupling`可关闭。本次未启动新训练、运行中实验保持；完整公式、采样范围与归档见[当前奖励说明](docs/reward_defaults_20261006.md)。以下为历史实施记录。

**2026-10-06 当前统一入口新增目标运动限制**：`all_use_this_train.sh`显式开启`joint_motion_limit_penalty`。手指下发目标速度／加速度超过10rad/s／200rad/s²开始扣分；腕旋转惩罚阈值3rad/s／15rad/s²，当前双侧Pitch最终目标另设3.1416rad/s／15.708rad/s²硬限制。所有关节、两类超限比例取最大值，当前统一脚本超限10%时整项每控制步封顶−5（通用默认20%）。包含耦合PIP/DIP，初帧和准备期计罚；原paper无和弦／false_press关闭、硬位置限位及学习预算保持。最终奖励启动时完整打印，新归档路径／SHA以根脚本为准；旧归档／运行中训练不变。CPU验收记录在`logs/joint_motion_limits_20261006/`，H100 Slurm584集成验收57项／111子测试通过（3项外置入口案例跳过，本次真实入口另核对），已用当前脚本从头启动双卡训练585，物理卡6／7，启动快照已产生37,888次学习更新；新策略效果待实测。硬限制针对下发目标离散差分，不能当作真实关节qvel/qacc保证。[参数、公式、兼容与验证](deprecated/docs/reward_modes/hpc_training_options_20260930.md#2026-10-06下发目标超限惩罚与腕部硬速度加速度限制)。

**2026-10-06 H100三组对照运行中**：按用户最新要求全部保留`hard_v1`硬关节限位；580仅闲指姿态、581仅同手碰撞、582两者组合，每组双卡与10,000,384样本，从1724 best19500仅迁移网络并重置学习状态。H100验收36项／28子测试通过，完整奖励已打印，保存配置确认hard_v1，三组均已跨过预填并产生58,368次学习更新（640步快照）。详见[H100执行记录](docs/h100_training_20261001.md)和[硬限位说明](docs/hard_joint_limits_20261005.md)。以下10月5日记录属于旧动力学；原A100 ECC尚未恢复。

**2026-10-05 当前训练奖励**：统一入口以续训前Slurm1620为基线，关闭和弦加成：`key_press_reward` 保持 `mode="paper"`，改为 `chord_bonus=false`、`chord_scope=null`，使用 [完整奖励配置](deprecated/reward_experiments/paper_chord_off_20261005.json)。按键分为 `0.6×正确键分＋0.4×exp(−1.25×错键数)`，正确键内部实际激活占比仍为0.5。false_press、闲指姿态及同手碰撞惩罚保持关闭；跨手接触每控制步−0.5保持。Slurm **1724** 于14:47提交，从头训练10,000,384样本；其余参数逐项对齐1620，仍为已验收v2指尖归档、20/200Hz及双卡流水。实际结果以新config和指标为准。

**关闭和弦基线的三组对照（10月5日18:33故障快照）**：1724已完成，best19500三回合最差原谱精确F1为.9913378248。**1725 仅闲指姿态、1726 仅同手碰撞、1727 两者组合**均从同一父best计划追加10,000,384样本；1726报GPU不可纠正ECC后失败，1725卡在预填后、1727卡在GPU探测。按用户修复要求已取消两项卡死作业，但进程残留、节点DRAIN且NVML初始化失败，仍待管理员恢复。父模型和冻结奖励已校验保存，尚未重新提交；三组没有可比较的续训收益。[故障现场与恢复步骤](docs/gpu_ecc_recovery_20261005.md)；[原配置与命令](docs/twinkle_20hz_comparison_20261004.md#2026-10-05关闭和弦基线的三组奖励续训对照)。

**10月5日H100实测与视频**：101.47.9.106的物理0／2号H100通过Slurm566基础CUDA／P2P／Warp测试。关闭和弦最新final19532已完成单卡回放和导出：[有声视频](preview/outputs/paper_chord_off_final_h100_video_20261005/videos/chord_off_final_19532.mp4)、[同次200Hz实际按键／原始MIDI对比](preview/outputs/paper_chord_off_final_h100_video_20261005/midi_comparison/midi_comparison.png)，本视频音乐区间micro F1=.989639294。附加关节曲线因旧录制状态越界未导出，物理轨迹保持；没有启动新训练或判定A100已恢复。[验收详情](deprecated/docs/reward_modes/video_export_cli.md)。

原和弦开启基线Slurm **1620** 已正常完成10,000,384样本，训练耗时6200.725秒；best15500三回合最差原谱精确F1=.9908081195。跨手/琴键底部接触为0，同手自碰撞仍有记录；输出目录为 `training_mjlab/outputs/hpc/paper_chord_on_fullhand_20261005-20261005-004515-261777`。

**2026-10-05 继续训练三组奖励对照**：已通过根统一脚本末尾追加参数提交 **1681 idle_only、1682 self_only、1683 idle_and_self**，同一best15500网络起点、每组新增约10M。闲指姿态采用`已弃用的闲置姿态方案`，weight=1、全手最多−.1/控制步；同手碰撞继承当前weight=.2，每去重body对−.2/控制步。更改奖励须采用显式网络微调，优化器/温度/回放池重建；其他父配置保持，结果待实测。[完整配置、三组命令与证据](docs/twinkle_20hz_comparison_20261004.md#2026-10-05paperchord最佳模型继续训练三组奖励对照)。

**2026-10-04 指尖碰撞修正**：核对with ID真实网格，修正顶端1mm薄带导致整根末节胶囊过窄的问题。十个末节直径由4.02–4.84mm改为完整网格拟合的11.72–12.82mm，保持原指尖目标点/局部Z范围，其余34个碰撞体保持。根统一训练入口已更新为验收过的新v2指尖归档，双卡及学习参数保持；本次未开长期训练。[实测、前后叠加图与限制](docs/fingertip_collision_audit_20261004.md)。

**2026-10-04后续指定的统一默认入口：[all_use_this_train.sh](all_use_this_train.sh)。** 运行 `bash all_use_this_train.sh` 默认按第⑤组S4参数＋新全手跨手碰撞从头训练，自动提交双卡Slurm（cuda:0采样、cuda:1学习，开启流水）；以后修改此脚本或在末尾追加参数，例如 `bash all_use_this_train.sh --actor-lr 0.0001 --seed 43`。固定新的tsensei_collision_v1源码归档；左右手任意部位接触−0.5/步，无接触0，其余S4参数保持。源码改动须同步更新归档；本次未开训。[位置、参数与使用说明](docs/twinkle_20hz_comparison_20261004.md#统一默认训练脚本)。此决定覆盖下方此前双卡入口的默认训练说明，工作区全手碰撞代码保留。

2026-10-05：统一入口新增机器人关节硬位置限位，保存完整控制合同；每个物理子步后将状态限制在模型与显式范围内。CPU验证已通过，GPU验收因Slurm节点draining待运行；未启动训练。[实现、策略审计与限位轨迹](docs/hard_joint_limits_20261005.md)。

**2026-10-04 此前双卡入口整合（最新默认以上方根脚本为准）**：本轮先整合为 [双卡训练脚本](scripts/hpc/train_coupled_20hz.sh)，经 `bash scripts/hpc/submit.sh train [本轮追加参数...]` 提交；默认参数集中在该脚本，不再新建每轮训练脚本。`run.sh` 已 **ARCHIVED**，A3.2根入口/旧40Hz文件名仅转发。新 `tsensei_collision_v1` 直接替换源码旧掌部任务模式：左右手全部22×22碰撞几何组合，周期内任意跨手接触一次−0.5，无接触0；统一指法目标、实体接触和其他HPC奖励保持。[范围图、合同与验收](deprecated/docs/reward_modes/tsensei_collision_v1_20261004.md)。下方日期记录中的run.sh/旧掌部说明为历史，不再决定新训练默认。

2026-10-04训练评分默认统一：普通/A3.2/HPC及h10/h12训练脚本均开启按键、误触、能耗子步均值和碰撞周期并集去重，20/200Hz每周期10个物理子步；**指法、闲指姿态、几何净空仍按周期末评分**。Shadow五项专用入口也改为子步采样，实际激活占比仍为0，不新增奖励。历史端点对照及正在运行的源码归档保持原配置；本次未启动训练。[评分范围与覆盖参数](deprecated/docs/reward_modes/hpc_training_options_20260930.md#2026-10-04统一子步评分入口)。

2026-10-04 15:41检查：[20/200Hz五组小星星实验](docs/twinkle_20hz_comparison_20261004.md)。S0–S3各10M已正常完成，保存best精确F1依次.964542/.980321/.979322/**.984377**；S3同手惩罚.2暂领先，短音覆盖92.59%、同手接触0/跨手14子步、键底0，尚未.99。S4已自动接续1587，约12%，待五组结束后统一独立10回合复评。有效进度在`training_mjlab/outputs/hpc/twinkle_20hz_comparison_20261004_v2/`，日志核验已修复、旧失败尝试独立保留，当前尚无最终排名。

2026-10-04检查：[仅Shadow原五项奖励训练与提前松键诊断](docs/twinkle_shadow_rewards_20261003.md)：1517从头10M已于10月3日20:29正常完成。保存时best精确F1=.980483、final=.967118；1557独立10回合复评best **.977806**，低于旧R6 **.984785**。早松漏采样减少68.9%、三个短音均能匹配，但错采样增至7.56倍，手接触覆盖86.41%物理子步；键底0。40Hz F1>.99不等于400Hz精确达标。完整合同/计数验收通过，当前无续训，结果及曲线已记录。

2026-10-03：[小星星F1奖励与微调实测](docs/twinkle_f1_finetune_20261003.md)：八组各2.1M均已完成。最佳为**false_press=.1、Actor LR3e-4**，10回合独立复评400Hz精确F1 **.984785**（父模型.973818、.25惩罚组.981096），最低回合.984016；两个短E5恢复，中间D5仍漏，未达.99。手碰撞/键底计数0，但侧面接触增多；完整学习器恢复和低Actor LR均未超过该候选。新增可选`submit.sh --gpus 1 replay`已在Slurm1499验收，训练默认两卡保持。

2026-10-03：[当前小星星 F1 0.99／1 候选分析](docs/twinkle_f1_candidates_20261003.md)：按 A/C/D/E 实际记录拆分失分；A best=.975153，即使理想补齐三个短音也仅 .982429，还需改善长音起止。建议分别测试 false_press 降权、真实激活占比、gamma 与速度观测，并对齐精确 F1 选模。此次仅离线审计，没有修改训练默认值或启动新训练。

2026-10-02晚：[小星星三组消融对照](docs/twinkle_ablation_20261002.md)：相对已完成A基线，依次关闭提前指法定位、仅关闭false_press_penalty、两者都关闭。三组各约10M均于22:59完成，final精确F1 C=.94363、D=.97142、E=.96344（A基线=.97391）。D恢复三个短音匹配，但侧面接触增多，未达.99；两对GPU继续预留至10月3日11:50。

2026-10-02：[双卡小星星视频与F1导出](preview/outputs/twinkle_comparison_20261002/index.html)：A/B各best、final，正面+斜俯视原速有声；含原训练F1总览/后期放大图、PDF/CSV及ZIP。

2026-10-01晚间双卡配置更新：原小星星及原指法，Pitch范围为原范围80%（**±0.16rad**），**gamma=0.75**。新速度对照保持同一源码/初始姿态、奖励、512环境、40/400Hz和学习预算；定期评估改为每1000步/3回合，仅保存第1000步forward.pt、最佳best.pt和最终final.pt。评估统计CUDA graph通过GPU验证，A100交错短测吞吐提高约1.05%，两组各约10M样本已于10月2日03:37全部完成：final精确F1 A=.97391/B=.97301，均漏三个短音、未达.99；B正式总耗时仅减少0.20%，两卡预留保持。[配置、速度验证及新实验](docs/twinkle_pitch_speed_20261001.md)。此前月光训练和R1–R4等待队列已于18:16按用户要求停止，GPU预留保留；下列下午运行记录为历史。

2026-10-01追加实验：用户指定`R4_gamma08`，在H100原R0–R3队列之后运行，仅将R0的gamma改为**0.8**；同144音谱、5.12M样本和seed42，奖励保持R0。使用原固定源码，17:34独立等待器已启动并确认存活，当前等待前四组完成，训练效果待实测。[执行记录](docs/h100_training_20261001.md)。

2026-10-01最新交接：R1/R2/R3对照配置已整理进Git，转交H100会话执行。本地协调器未存活、三组未在A100提交，A100基线1364.10继续；H100已通过实际144音谱GPU验收（36项/35子测试），17:19:57在396.6启动同机R0，R1/R2/R3串行等待，各组5.12M/seed42；独立协调器已确认存活，尚无四组结论。[完整方案、同步和启动命令](docs/h100_training_20261001.md)。

2026-10-01当前双卡默认按用户最新要求采用**仅闲指姿态gesture**：每指最多−0.01、全手最多−0.1/步；跨掌0/−0.25、同手weight=1/每去重body对−0.5。当前指法优先关闭，当前与未来用指共同平均，提前定位0.15秒。保留《月光》、当前范围、40/400Hz、60–80°耦合、腕速1m/s及约10M预算。已按用户要求直接替换原训练谱：合并B4/C5两处同指连音，146→144音，400Hz精确覆盖和2637帧CPU构建通过；已在保留分配1364的步骤1364.10启动双卡训练。[完整双卡配置](deprecated/docs/reward_modes/hpc_training_options_20260930.md)。

2026-10-01 15:05，1364四组3M对照全部完成：原MIDI精确F1 A/B/C/D为 **0.6563 / 0.6780 / 0.2479 / 0.5939**。B精确时序最好，C分阶段和D请求许可本轮均未改善，三个短音各组均未匹配。队列14:41完成，1364两卡继续保留；本批冻结配置不包含后来新奖励与范围修改。[最终比较与曲线记录](docs/note_timing_repair_tests_20261001.md)。

2026-10-01 14:11，1364原对照A/B/C均完成3M：原MIDI精确F1分别 **0.6563 / 0.6780 / 0.2479**。B精确时序目前较好，C分阶段仅匹配4/34音，三个短音各组均未匹配。D提前请求许可组已训练约89万样本，继续运行。本批使用冻结的十倍geometry配置，后续新奖励与范围修改另行验证。[完整结果与曲线记录](docs/note_timing_repair_tests_20261001.md)。

2026-10-01 13:55，1364原四组对照：A/B均完成3M，原MIDI精确F1 **0.6563→0.6780**，手碰撞240.3→17.7子步/回合；三个短音均未匹配。C分阶段组在276.48万样本时F1仅0.2482，明显低于同预算B0.6494；D尚待运行。[最终A/B与当前C结果](docs/note_timing_repair_tests_20261001.md)。

2026-10-01早期奖励方案（最新决定已重新采用）：当时`submit.sh train`采用闲指姿态最终每步最多扣0.1，移除几何评分/十倍倍率；掌部碰撞0/-0.25，同手每去重碰撞对-0.5，当前指法优先关闭。60°–80°耦合、TX/TY 1m/s保持；已冻结实验继续原配置。[最终分值与开关说明](deprecated/docs/reward_modes/hpc_training_options_20260930.md#新奖励的分值与开关2026-10-01)。

2026-10-01：[H100两卡预留、Git同步与训练部署](docs/h100_training_20261001.md)。预留396持有两张H100 80GB至10月3日13:04；环境、35项GPU验收、512环境短训及检查点重建通过。13:23按用户新要求停止正式396.4，保留两卡，待确认奖励后再开对比实验。

2026-10-01 13:14效果检查：A已完成3M，B训练约279万，C/D等待。同为276.48万样本时，原MIDI精确F1 **0.5866→0.6494**，手碰撞 **316.3→37.3子步/回合**；B匹配音仍平均晚起68.9ms、早松20.5ms，三个短音未匹配。已完成下压干预支持25ms提前候选，短音还需足够屈曲命令。[统计口径、曲线及限制](docs/note_timing_repair_tests_20261001.md)。

源码仓库：[ONKIALL26/tsensei_v3_2_pianist](https://github.com/ONKIALL26/tsensei_v3_2_pianist)。首次克隆、日常推送/拉取、环境安装及排队源码固定见 [Git 同步说明](docs/git_sync.md)。Git 版采用 A3.2 轻量训练范围；本地 M6 预览资产、硬件 SDK、完整在线应用和训练产物不随本次提交上传。

2026-10-01 12:15状态：1364已分配两卡；83项含GPU验收、下压干预及功能短训完成，A组基线正在训练（约156万/300万样本），B/C/D自动依次运行。详情见[本轮队列记录](docs/note_timing_repair_tests_20261001.md)。

2026-10-01 最新：[精确音符时间修复与分项训练](docs/note_timing_repair_tests_20261001.md)。双卡入口显式使用 **60°–80°四指耦合、gesture得分×10（weight=1）、TX/TY目标限速1m/s**，开启原MIDI起止时间的400Hz评分及176维事件倒计时（31动作/2297观测）。在1364保留分配内排队执行验收、0/25/50ms下压干预、功能短训和A/B/C/D四组各约3M样本；分阶段指法与50ms请求目标许可独立对照，默认关闭。GPU结果及F1收益待实际运行验证。

2026-10-01：按用户要求提交双GPU **48小时保留申请1364**，时长从实际分配开始计算。进入RUNNING后用`bash scripts/hpc/submit.sh --allocation 1364 train`复用这两张卡；任务之间继续保留，到期释放。[使用、状态与提前释放](docs/gpu_allocation_48h_20261001.md)。

2026-10-01：[晚弹、松键时序与短音漏弹审计](docs/note_timing_diagnosis_20261001.md)：1103 best的31个成功音相对原MIDI平均晚起32.11ms；相对训练目标晚45.08ms。松键相对原MIDI晚11.21ms，主要来自目标结束向上取整，实际相对训练目标平均早2.90ms。主要MCP下压命令在起音当帧才跃增；三个短音的末节距目标键仍有3.3～3.8mm。附400Hz记录、命令／关节／键程图、奖励复算和分项修复方案，本次未改训练行为。

2026-10-01早期：四指耦合曾采用50°–80°许可，随后用户明确改为 **60°–80°**，当前双卡入口使用[新范围文件](configs/finger_coupling/angle_ranges_60_80.json)。八根四指独立范围、拇指独立控制；旧固定60°和50°–80°配置保留。[参数与求解说明](docs/finger_coupling_angle_ranges_20261001.md)。

2026-10-01：新增 [统一视频导出脚本及参数说明](deprecated/docs/reward_modes/video_export_cli.md)。`bash export_video.sh --pt 路径 --video1 arms backend --video2 noarms forhead`，默认导出左右机械臂及双手每个实际关节的**速度和加速度**PNG/PDF、CSV/NPZ；`--plots none`关闭图表，`--plots all`另含TCP速度，显式`--plots`替换默认选择。支持多视频/多视角、指定PT、裁剪/倍速、音频、复用录制及自动Slurm提交。新默认正交投影、琴宽铺满子画面；STL/XML仅保存在作业缓存，交付目录不含模型。默认显示原乐谱指法（R1–R5/L1–L5、目标琴键与指尖颜色）；`--no-fingering`关闭，`--finger-labels`额外显示悬浮编号。已有视频可用`training/.venv/bin/python preview/scripts/export_joint_curves.py --video 视频绝对路径`从原记录离线补导图表。 机械臂图默认标注SDK正负速度/加速度限值红虚线和超限红点：两臂J1–J7速度180°/s，J1–J2加速度450°/s²、J3–J7为900°/s²；图中换算为rad单位，来源与比较结果随报告导出。

新视频默认启用[原节奏机械臂限速规划](docs/arm_motion_constraints_20261001.md)：`--arm-ik constrained`同时约束连续关节角、SDK速度/加速度；手腕位置误差默认≤0.1 mm、姿态≤0.5°，不合格则停止。视频与关节图使用同一条样条，保存帧间跟踪误差和连续限值证书。`--arm-ik raw`保留旧诊断行为；手部原轨迹保持，尚无已确认手关节限值。

2026-09-30：[当前1103 best原速有声视频、双臂差分IK和14关节速度曲线](docs/best_wrist_ik_export_20260930.md)已导出；400Hz腕部位姿跟踪，单回放音乐F1 .92550。保留当前天机安装件，关节速度为原时间差分；左J7触及角限，不代表真机动力学/避障验收。

**两组10M已完成（2026-09-30）**：1099原强度与1103十倍gesture均正常结束。十倍组final逐帧音乐F1 .90942→.92381，但400Hz音乐micro .94332→.93209、漏键采样增加18.5%；手碰撞164.33→1.33子步/回合，减少99.2%。两组均未达.99联合目标。[三种F1、接触、末段趋势与候选检查点](docs/gesture_score_comparison_20260930.md)。当前默认仍为10倍，后续3/5倍仅为对照建议。

**当前humanoid对照（2026-09-30）**：双卡默认在原40/400Hz配置上，将`已弃用的几何净空方案`得分乘10，外部`weight=1`且严格保留`weight≤1`校验。该项为负罚，实际每步最大扣分由0.05变为0.5；1099保留原强度，新实验从头使用相同种子与预算。[公式、对照和启动记录](docs/gesture_score_comparison_20260930.md)。

2026-09-30 最新装配：已安装用户提供的 **A3.2_tianji.stp**，采用 **45.6×24.6 mm宽安装座、19×40 mm四孔对齐**，替换同日先前的窄安装座方案。左右连接件和M6工具变换同步，质量仍为 **0.952 kg/手**，预览保留平移/旋转/保存面板。[新安装件、装配图与验收](docs/tianji_mount_20260930.md)。此前1078回放及速度图保留在[历史装配与回放说明](docs/hand_mount_mass_arm_replay_20260930.md)；旧检查点仍使用旧归档，新几何下F1未重测。

**双卡基础配置（2026-09-30，10月1日更新时序）**：`bash scripts/hpc/submit.sh train` 使用四指耦合、完整观测、40/400 Hz，精确时序新增176维后为31动作/2297观测；提前定位0.15秒，当前指法优先关闭，TX/TY各1m/s。配置集中于 `scripts/hpc/train_coupled_40hz.sh`。奖励按dt/0.05换算、gamma=√0.8、初温0.5；通用入口约10M样本，本轮对照显式覆盖为各3M。普通单卡run.sh保持20/200Hz。[训练证据与时间换算](docs/coupled_40hz_training_20260930.md)。F1与碰撞效果须新一轮实测。

2026-10-01曲目更新：`bash scripts/hpc/submit.sh train` 现默认使用《月光》初学者版的既有 `generated_v1/clair-de-lune-beginner_fingered.proto`（原146音，后续合并连音为144音，候选指法），其余双卡参数保持。末尾加 `--midi TwinkleTwinkleRousseau` 可切回小星星。此修改未提交训练；当前模型的候选指法可达性和演奏效果仍待实测。详见[双卡入口说明](deprecated/docs/reward_modes/hpc_training_options_20260930.md)。

**机械臂软件（2026-09-30）**

已完整克隆 [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) 到项目根目录同名文件夹，检出 `master`，提交 `a0f8c1134895a1b3ab2a5621bb93ae4ba0634990`，保留 Git 历史；该版本没有子模块或 Git LFS 配置。

- 控制 SDK：`TJ_FX_ROBOT_CONTRL_SDK/contrlSDK100343/`，另保留上游旧版 `contrlSDK/`。
- 运动学与干涉检测：`kinematicsSDK/`、`interferenceCheck/`（均位于上述 SDK 根目录）。
- Python 接口、动态库和示例：`SDK_PYTHON/`、`DEMO_PYTHON/`；C++ 示例：`DEMO_C++/`。
- 上位机：`MarvinPlatform_EN/`，含 GUI 源码及 Windows/Linux 可执行文件；使用手册、接口文档和配置随仓库保留，入口见 [SDK README](TJ_FX_ROBOT_CONTRL_SDK/README.md)。

本次完成源码及仓库内软件获取和 Git 完整性核验，未安装运行依赖、编译、连接机械臂或接入训练。上游说明 Python ≥3.10，Linux 编译使用 g++/make；GUI 源码还使用 Tkinter/Pillow，部分示例使用 NumPy。`contrlSDK100343` 要求匹配 100343 及以上控制器，控制器固件与机器人专用参数另由上游 Releases 提供，须按实际硬件版本选择，未在本次下载或刷写。现有 A3.2 轻量迁移包不因此自动包含此 SDK。

2026-09-30：`run.sh` 默认关闭当前指法优先，A3.2／HPC入口继续提前 **0.15秒**定位。恢复当前与未来用指共同平均，未来奖励不再受 `(1-current)` 抑制；分阶段指法仍关闭。通用 `run.sh` 的提前时间仍为0，A3.2／HPC包装入口显式覆盖为0.15。此次只调整后续启动默认值，F1收益待训练验证。

2026-09-30：三组3M耦合对照已完成。31动作/完整1231观测的best音乐F1为0.93115，优于基线0.89212和压缩观测0.92024；提升来自漏键减少，但final手碰撞由基线10.67增至79子步/回合。当前建议保留耦合、关闭观测压缩，并先解决跨手碰撞。[完整F1、200Hz、末段、碰撞和角度分析](docs/coupling_f1_analysis_20260930.md)。

2026-09-30早期性能与耦合对照：Slurm1074在20/200Hz实测双卡批量评估稳定1391样本/秒，比双卡顺序评估提高33.7%；双独立策略合计2316样本/秒，只代表实验总吞吐。见[双卡配置与测试](deprecated/docs/reward_modes/hpc_training_options_20260930.md)。四指MCP驱动、PIP/DIP从动，拇指保持，目标末节向琴面下倾60°；20Hz下动作47→31，观测可选1231→1199。1077/1078/1079三组3M均已完成；[耦合实验记录](docs/finger_coupling_experiment_20260930.md)。新40/400Hz吞吐需重新实测。

2026-09-30：[双卡速度审计](docs/dual_gpu_speed_audit_20260930.md)：同A100基准稳定吞吐提升28.15%；学习约291ms／批、采样约89ms／批已重叠。正式5M的99次串行评估及边界管理约占26分钟，是整体加速受限的另一主因。本次只分析已有数据，训练参数保持。

2026-09-29：`run.sh`直接指定左右TX/TY目标限速各 **1m/s**（20Hz下每帧最多变化5cm），Pitch仍为4rad/s；末尾CLI可覆盖，实际值保存到config/checkpoint。原限速JSON保留1.2m/s历史值。[限速参数说明](configs/control_limits/README.md)。

2026-09-28：新增 [火山云 A3.2 双卡迁移入口](docs/volcengine_a32_migration_20260928.md)，含独立轻量包、双环境安装、Slurm 两卡探针／基准和可选采样学习流水。A100 完整四次短训/检查点验证通过，双卡稳态平均加速28.15%，Slurm935已从头开训；普通本机 run.sh 默认保持单卡。

2026-09-28：[整音漏弹与提前松键修复方案](docs/f1_regression_repair_plan_20260928.md)已制定：以当前A3.2配置重跑基线，分别测试关闭当前指法优先、开启分阶段指法，每组约3M样本，再按逐音／F1／碰撞结果进行组合和完整预算验证。方案尚未执行，默认参数保持当前值。Slurm935已完成5,000,192样本，其best三回合音乐F1均值.926381、最低.925467，尚未达到.99目标；与下方本地历史运行分别记录。


## 当前 F1 修复与参数核验（2026-09-28）

17:39 已按此配置从头启动一轮 A3.2 小星星正式训练，512环境／约5M样本、提前定位.15秒，
SwanLab在线上传开启。[本轮目录、配置与启动核验](docs/training_f1fix_launch_20260928.md)。

9/28 当时动作 `direct_joint_v1`：八个四指 MCP `[0,2]→[0,1.3] rad`（10/1新目标范围见下方完整表）；任务／驱动的
`shadow_v1` 是独立设置。评估、视频和预览现严格校验完整动作范围，防止同名 JSON 改值后误加载。
`run.sh` 显式增加一轮整曲预填、当前指法优先、严格音符覆盖检查和 .25 秒真实松键尾段；
音乐 F1 排除尾段，单列松键结果。三 LR 仍 `3e-4`，20/200 Hz，几何动作保护仍关闭。
提前定位在通用 `run.sh` 仍为 0，在 `run_tsensei_a32.sh` 保持 .15 秒。
参数与取舍见 [F1 修复与验收报告](docs/f1_evidence_repairs_20260928.md)。
本节及下方较早分析与训练结果是历史记录；新训练默认以根目录all_use_this_train.sh、末尾覆盖和本轮完整config.json为准，run.sh已归档。

## 当前入口与拇指审计（2026-09-28）

`run.sh` 默认关闭两项手指动作修正，用 `--idle-finger-protection` 和
`--active-press-permission` 分别开启。几何诊断及 `已弃用的几何净空方案` humanoid负罚仍开启，
自碰撞实际权重0.2、错键惩罚1；当前run.sh腕目标限TX/TY 1m/s（9月29日CLI覆盖）、Pitch4rad/s，无曲目位置裁剪。
拇指默认保留Shadow统一目标；可按需传 `--thumb-white-target-distance` /
`--thumb-black-target-distance`（米，从+世界X端向内）。所有参数均列在run.sh头。

[本轮完整分析与改动](docs/training_controls_thumb_review_20260928.md) ·
[入口/奖励核对](docs/run_options_review_20260928.md) ·
[拇指参数与手结构](docs/thumb_target_options_20260927.md)。
20Hz末端奖励确有短事件盲区，自碰撞已按200Hz子步汇总；拟议奖励优化尚未自行启用。
上午续训best音乐F1=.990024（micro=.987993），下午从头best=.943460，均已结束且未通过全部目标。
本轮不启动新训练。新目标manifest改变任务散列，旧检查点使用匹配归档；不能绕过合同直接用新源码续训。
8个未采用控制候选已移入 `configs/archive/20260927_control_trials/`，保留历史索引。

## 历史整指净空方案（2026-09-27，启动时记录）

上一轮续训已经完成，best 音乐F1为0.98227856，但仍有250ms左拇指键底接触。本轮区分实际手指姿态与PD施力目标：闲指按整根碰撞几何保护，当前用指符合实际琴键上方支持条件时保留按压力；准备期也有有界负辅助项。没有新增曲目专属TX位置范围。新run.sh显式使用几何/奖励/rate-only JSON，Python通用模块默认关闭；下方较早日期的配置是历史记录。

[完整诊断与试验](docs/generic_finger_clearance_20260927.md) · [几何规则与泛化检查](docs/finger_clearance_geometry_20260927.md) · [humanoid负罚公式](docs/humanoid_clearance_reward_20260927.md)。新best候选与早停都要求完整200Hz键底采样及零键底接触，不能以高F1覆盖一个无卡底的best。两组短训后选择Actor 3e-5，09:34已从小LR短训best恢复学习器继续约5M样本；初期合格best已到0.941，键底0。F1 0.99、自碰撞和跨曲动态安全仍未达成，完整记录在报告及latest_clearance_launch.json。

从同一乐谱/模型的旧权重迁移使用 `python3 scripts/launch_clearance_training.py --parent /absolute/path/best.pt --steps 4096`；该脚本继承父温度与源码姿态，明确重建优化器/replay，并通过run.sh选择新控制/奖励。训练参数全部列在run.sh文件头；旧检查点使用匹配源码归档，不会静默加上新保护。


2026-09-27：已完整审计上一轮best（音乐F1=.97356849），确认主要卡键为左拇指键底／键侧受阻，曲首0.75秒占66.25%失分。新增可选软件目标范围／变化率限制，完成9组零更新复评、三组同预算短训和真实8环境局部reset验收。本轮显式选择腕部限制＋Actor LR=3e-5，原best权重在新控制下F1=.97525623；所有短训仍有回退，未宣称0.99或无卡键。**01:26已从保留的高分初始best开启约2.1M样本续训，首批更新已验证**。[完整诊断、方案、试验及新训练](docs/best_range_finetuning_20260927.md) · [接触证据](docs/motion_contact_audit_20260927.md) · [短训比较](docs/range_trial_summary_20260927.md) · [可选控制参数](docs/control_limits_20260927.md)。通用入口不会自动开启限制，本轮完整启动参数保存在 `training_mjlab/outputs/latest_range_launch.json`；启动状态是时间快照，运行结束以summary为准。

2026-09-26 效果优化：已完成四检查点共16回合审计，确认step8500优于final、20Hz短错音盲区和同手碰撞。新增有条件的小姿态奖励、200Hz碰撞诊断、每次评估保存best、严格网络微调与质量目标停止；用户已允许短测后选用batch1024×0.25预算。[详细优化计划与短测／新训练](docs/training_quality_optimization_20260926.md) · [完整旧训练审计](docs/training_run_review_20260926.md)。新续训已于19:45启动，采用false1+self.2+有条件humanoid.05、batch1024×.25，完整恢复C短测learner。独立入口 `bash run_quality.sh --help`；全部参数写在脚本文件头，尚未达到F1 0.99。

2026-09-26 最终小星星5M训练已完成；[有声最终视频与489点评估F1曲线](docs/training_final_video_f1_20260926.md)。最终整回合F1=0.9368、乐曲段F1=0.9351；源码变化后使用匹配训练归档完成严格校验回放。

2026-09-26 02:01：按用户要求，从头启动当前默认512环境的小星星5M训练，1秒准备帧参与SAC，GPU执行加速开启。该运行现已完成；此链接保留当时启动记录。[运行目录、日志和完整配置](docs/training_policy_preparation_5m_20260926.md)。

2026-09-26 GPU容量实测：当前512环境稳定约968条/秒，含评估等总吞吐约768；2048环境同采样/更新/评估预算仅提高到1010稳定、783总体。满1M replay学习器约1050更新/秒，4096环境纯采集约3.80万条/秒（不含学习）。正式默认保持512，未因容量测试改SAC或精度。[实测报告、图表和复测入口](docs/gpu_training_ceiling_20260926.md)。

2026-09-26 后续更新：用户要求准备帧进入训练。`run.sh` 现采用 **policy_v1**：从抬高2cm的initial开始，策略立即控制，1秒内支撑平滑回origin高度；前20个空谱帧照常计奖励、写replay。新增1维准备进度，1181观测／47动作。执行加速保持Shadow学习参数和更新预算，详见[准备学习与加速说明](docs/policy_preparation_acceleration_20260926.md)。下方同日固定准备记录属于此前模式，不是当前默认。

同配置512环境实测稳定吞吐260→962条/秒（3.70倍）；当前5M预算短测外推约102分钟，**尚未恢复旧约46分钟**，没有降低更新次数。本轮仅短训验收，未新开5M。[真实准备回放与验收](preview/outputs/preparation/policy_active_final_acceptance_20260926/README.md)保留未充分训练策略的真实误触。

2026-09-26：`run.sh` 新增固定准备：保存当前姿态为 **origin**，世界 Z 抬高 2 cm 为 **initial**，用 1 秒平滑回到 origin 后开始策略和乐谱。准备期固定关节目标、踏板关闭；TZ/Yaw 仍删除、47 动作。GPU reset 复用独立物理计算的准备终态，不向 replay 填入固定动作。[准备视频](preview/outputs/preparation/lift_2cm_1s_shadow_20260926/preparation.mp4)；[参数、曲首问题与验收](docs/startup_preparation_20260926.md)。回到 origin 不保证策略下一条大目标消失。

2026-09-26：Shadow 同步 5M 已完成，耗时 5 小时 34 分钟；前一轮约 46 分钟。SAC 更新由约 59 万增加到约 500 万是主因，物理 CUDA graph 和编译学习器已开启。[速度实测](docs/training_speed_audit_20260926.md)。本轮保留 Shadow 学习预算，只完成准备功能短训验证，没有新开 5M。

2026-09-26 最终训练视频：[Shadow同步 final 整曲回放](preview/outputs/replays/shadow_sync_final_20260926-000853-788815687/replay.mp4)，带真实琴键音频。运行 `bash preview/scripts/export_latest_training_video.sh` 可重新导出该已完成训练的最新检查点，自动创建带时间戳的输出；可选第一个参数传其他训练目录。该旧训练按其原源码归档回放，没有新准备段，见[归档回放](docs/archive_replay_20260926.md)。

2026-09-25：`run.sh` 已切换为 Shadow 同步配置：原五项奖励、统一[-1,1]动作、batch256/replay1M/每样本1次更新，以及V3尺寸碰撞代理和TX/TY等效平移惯量。保留47动作、20/200Hz、0秒曲谱空白；固定准备阶段后续由上方9月26日决定覆盖。参数对照、被拒候选和实际验收见[Shadow同步说明](docs/shadow_sync_20260925.md)。


2026-09-25：已对最新策略进行逐5ms爆冲诊断及同模型CPU/GPU对照。确认两类问题：大幅腕部目标阶跃，以及双拇指碰撞反弹／小指被琴键侧面夹挤；同状态5ms诊断分支已区分接触与命令因果。详见[完整训练链路、旧Daxian／Shadow差异与实测](docs/training_motion_audit_20260925.md)。本次未改训练参数。

2026-09-25：最新512环境／direct_joint_v1小星星5M训练已完成，最终F1=0.904307、本轮最佳F1=0.919101；两段有声视频已导出，回放按键指标与日志一致。[视频、曲线和剩余问题分析](docs/training_twinkle_latest_20260925.md)。

`bash scripts/hpc/submit.sh train --output-dir training_mjlab/outputs/my-run` 会自动在目录名末尾追加北京时间时间戳。再次使用同一基础名称会写入新目录，回放时使用启动输出打印的实际目录。详见[源码训练说明](docs/source_config_training_20260924.md)。

当前默认参数以 [根统一入口](all_use_this_train.sh) 为准；底层双卡脚本与[归档run.sh注释](run.sh)保留历史CLI，不决定新默认。原五项公式及当时的 `--humanoid-finger-gesture` 见[空闲手指标准姿态奖励](docs/humanoid_finger_gesture_20260926.md)；当前公式读取唯一奖励总表。

2026-09-24：新增旧Daxian公式的可选错键下压惩罚 `false_press_penalty`，默认关闭。用户决定启用时使用 `bash run.sh --enable-reward false_press_penalty`；公式、JSON配置和验收见[错键惩罚说明](docs/false_press_penalty_20260924.md)。

2026-09-24 当前源码训练进行到约step-10110时，已保存检查点中`step-5000.pt`的日志逐帧按键F1最高（0.360121）；独立有声回放F1为0.334762，精确率0.246119、召回率0.818820，显示误按仍明显，不能称为学会演奏。[阶段效果、视频和指标边界](docs/source_config_training_20260924.md#当前训练阶段回放2026-09-24)。

2026-09-24 历史入口（参数已被上方9月25日Shadow配置覆盖）：`bash run.sh` 从源码构建环境，保存完整 `config.json`；`bash eval.sh --checkpoint RUN/final.pt --mode preview` 从源码和配置重建回放，不再依赖持久化 scene/task 缓存。后续默认使用512环境×9766向量步，保持5,000,192条总样本、262,144条replay容量和0.125 update/transition；已启动的256环境训练不受影响。默认曲首缓冲1秒、手部重力补偿开启、Actor256×3、初始温度1、discount0.8。**指法按用户最新要求恢复原乐谱，未全部设成7。** 使用方法和验证见[源码配置训练说明](docs/source_config_training_20260924.md)；[可获得的聊天回答记录](docs/chat_answers_20260924.md)明确标注历史缺口。

2026-09-24（已被上条替代）：曾使用[脚本从源码构建并缓存 scene/task](docs/source_bundle_cache_20260924.md)。当前源码手根 0.10 m、TY=0.04 m、两侧 Yaw=0.15 rad，手部重力补偿开启。直接源码构建的新 task 含拇指目标点规则，与旧派生 task 有奖励目标差异。

2026-09-24：已导出当前无补偿续训与重力补偿新训各一段小星星视频，并生成含旧47动作首轮的[F1训练曲线与阶段结果](docs/training_video_f1_snapshot_20260924.md)。两轮导出时仍在训练，视频是已保存检查点的独立完整回放。

2026-09-24 14:47：按用户选择，已从无重力补偿的 47 动作 `final.pt` 再热启动小星星 5M 训练，显式使用旧 scene/task；与当前重力补偿 47 动作新训并行。[续训合同、进度位置与限制](docs/training_no_yaw47_legacy_continue5m_20260924.md)。

2026-09-24（历史设置）：源姿态预览和新场景导出曾按当时小星星训练冻结场景同步：手根 0.12 m、TY=0、左右 Yaw=0.25 rad、20/200 Hz、手部重力补偿开启。通用 `suite.load()` 的历史 40/400 Hz 配置仍保留，训练时的无臂及 TZ/Yaw 结构锁止仍由训练入口处理。[当时的同步对照](docs/gravity_compensation_20260924.md#源码与当前冻结场景同步)。

2026-09-24：源姿态预览默认开启双手重力补偿；无检查点的 MJLab 预览与 Torch SAC 默认读取新的 20/200 Hz 重力补偿冻结场景。49／47 动作小星星脚本也已切换，47 动作新一轮训练从头启动。旧检查点仍按散列读取原场景，不会混用。[配置、验证及运行记录](docs/gravity_compensation_20260924.md)。

2026-09-24 08:14：已启动修复接触振荡后的新小星星 5M 训练，20/200 Hz、12 cm、无臂、49 动作、同手／跨手／手—琴键真实接触。[本轮配置与运行日志](docs/training_twinkle_contact_stable_20260924.md)。

2026-09-24：已定位并修复复现案例中的接触振荡：手指 armature=0.0002、Kp=1，预览及新 12 cm／无 TZ 训练包恢复同手／跨手／手—琴键真实接触。需重启预览。[Shadow 对照、实测曲线与验收命令](docs/contact_stability_shadow_20260924.md)。

2026-09-23更新：M6小星星5M已完成，最佳step-18000日志F1=0.204550，有声回放F1=0.193968。[结果、曲线及视频](docs/training_m6_twinkle_20260923.md)。

2026-09-23：已启动含93 mm M6机械臂的小星星5M训练，mixed_v1、forearm扩展＋同手自碰撞0.2。[本轮命令与运行记录](docs/training_m6_twinkle_20260923.md)。

2026-09-22：肩座安装中心距已改为 **93 mm**，预览加入可达腕部约束以避免手臂脱开；新 M6 GPU 场景扩展 forearm 的跨臂／臂—手检测，策略仍 51 动作。[区域图、启动与验收](docs/m6_forearm_93mm_20260922.md)。

2026-09-21：源姿态预览已接入白色 Marvin M6 双臂，18 cm 安装中心距、可调临时安装，拖动 Wrist 滑块或绿色目标球可用差分 IK 跟随。[启动、演示与验收](docs/tj_m6_preview_ik.md)。仅运动学预览，未接入机械臂动力学／碰撞或训练策略。

按照 V2／V3 的方式，复制必要的 RoboPianist 钢琴、舞台、音乐、任务和奖励代码，替换为用户最新 A3.2 手模型。当前模型命名为 `TsenseI_A3.2`，采用用户提供的 `TsenseI_A3.2__hand_withid.zip` 中的 `A3.1V6 26.9.27(6).urdf`，包含新增 ID 部分。目录和 Python 包名均为 `tsensei_v3_2_pianist`，可与旧版本共存；运行时不需要相邻原项目。

**当前提供的模型只有左手。右手是本项目生成的临时镜像模型，并非厂商正式右手。** 所有原始关节的 effort/velocity 都为 0；为了运行仿真，手指暂用 ±0.6 N·m 力矩上限。正式右手、真实控制参数和接触标定均列入 [TODO 与修改记录](docs/change_log.md)。已验证环境 reset/step 和 SAC 短程更新，未验证训练收敛或演奏能力。

## 当前物理与时序（2026-09-18）

此节及链接记录2026-09-18至09-24的物理／源码构建历史，含旧40/400Hz及1180维观察，不作为现在的训练配置。当前统一入口为20/200Hz、无臂31动作／1407观察，带臂31动作／1461观察；以本轮config为准。历史构建过程见[源码配置记录](docs/source_config_training_20260924.md)。源姿态预览默认显示solimp滑块面板；GPU预览用`--contact-controls`打开，仍须经Slurm。新检查点回放核对源码和依赖版本，旧检查点匹配原冻结包。

[历史40/400 Hz参数、滑块用法与验收限制](docs/shadow_contact_40hz_acceptance.md)：当时34 项单元测试、2 项回归及短训练／有声回放通过；严格 CPU／GPU 物理对照为269/270，按压激活差异仍待定位。该页数据属于历史40/400 Hz包，不应替代当前20/200 Hz重力补偿场景的验收结果。

## 动作映射（2026-09-25更新）

当前双卡脚本的**动作映射**使用 `direct_joint_v1`（2026-10-01更新）：拇指Rot/Rot2目标`[0.05,1.57]`、拇指MCP/PIP`[0.05,1.2]`、四指MCP`[0,1.2]`、PIP`[0,0.9]`、DIP`[0,0.8] rad`，独立策略和耦合从动采用相同PIP/DIP范围；双腕TY目标及模型范围扩展为`[-0.05,0.15] m`。策略坐标不变，四指MCP中点`1→0.6 rad`，手指模型机械限位不变。原始姿态保持，本次未开训。`tsensei_collision_v1` 用于任务，`shadow_v1` 用于驱动，物理碰撞几何使用 `shadow_primitive_v1`；这些参数不决定策略动作坐标。已移除 `shadow_v1` **动作**选项。完整数值读取 `configs/actions/direct_joint_v1.json`，新训练的实际合同写入 `config.json`。[47维完整范围表与公式](docs/action_mapping_direct_joint_v1_20260924.md)。

## 常用命令

2026-09-21：[所有训练结果与F1曲线](docs/training_results_20260921.md)。5轮正式MJLab已完成，另1轮主动停止；两轮《月光》最终F1明显低于初期，仍需定位原因。报告含可复用绘图命令与各轮最佳检查点。

训练、最佳视频导出、策略回放、姿态编辑与重新导出场景的命令见 [V3.2 常用命令速查](docs/common_commands.md)。2026-09-18已完成本轮小星星5M采样，日志最佳step-18000的有声视频与实测指标也在该文档中记录。

## 预览目录

预览默认自动选择 NVIDIA GPU，并在终端打印实际渲染器；也可用 `--gpu nvidia` 显式选择。已优化每帧状态同步与碰撞线框缓存，控制方式保持一致。GPU 选项、性能报告及兼容模式见 [性能诊断](preview/README.md#gpu-与性能诊断)。

- `preview/scripts/preview_pose.py`：预览主脚本。
- `preview/outputs/`：默认生成 `pose.png` 和 `pose.json`，重复运行会覆盖同名输出。
- `preview/README.md`：窗口、离屏及碰撞显示用法。

当前碰撞几何另提供一个只用于预览审计的 `convex_hull_v1`：除掌壳和两个拇指根部链接 保留原始 collision mesh 外，其余 link 使用完整 STL 顶点生成的简化凸包。`--real-collision-mesh` 可切换到原始 URDF collision mesh，并自动以橙色显示求解器凸包和接触点（原生mesh接触仍是凸包）；两种观察都会记录到输出 JSON。当前训练默认仍为 `shadow_primitive_v1`。[碰撞 profile 说明](docs/collision_geometry_profiles_20261011.md)

脚本可从任意工作目录启动，默认输出始终位于本项目 `preview/outputs/`；显式传入的相对 `--output`／`--export-xml` 路径以当前工作目录为准。根目录 `preview_pose.py` 保留为兼容入口。`docs/` 下既有预览保留为历史文档快照。生成输出不纳入 Git，保留 `.gitkeep` 以建立空目录。

## 查看当前姿态

2026-10-01：仅调整双手手指初始角，使含拇指的五个指尖标记点与琴面平行；保留轻度弯曲和原腕部/安装。initial高于白键面约56.27mm，origin约31.27mm，静态均无接触。[角度、预览与检查边界](docs/initial_five_tip_plane_20261001.md)。

坐标轴默认隐藏，添加 `--show-coordinate-frames` 可绘制 World、Left wrist、Right wrist 三组坐标轴（红 X、绿 Y、蓝 Z）。wrist 对应当前规范化 `base_link` 并随手运动；被模型遮挡时可再加 `--transparent-model`。详见 [坐标轴说明](preview/README.md#world-与-wrist-坐标轴)。

本机已安装本项目独立 `.venv`：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
.venv/bin/python preview/scripts/preview_pose.py
```

交互窗口默认运行物理仿真，右侧 Control 面板可驱动手指和手部位置；鼠标可旋转和缩放，空格暂停／继续，R 恢复打开时的状态。使用 `--paused` 可从冻结姿态开始。已在本机 X11 桌面验证 GPU 窗口启动／运行／关闭，控制输入通过真实物理循环回归测试，未自动拖动桌面鼠标滑块。

![V3.2 左手与临时镜像右手](docs/pose_preview.png)

```bash
# 无窗口：生成 PNG 与同名 JSON
.venv/bin/python preview/scripts/preview_pose.py --headless --output preview/outputs/pose.png
# 指垫末节碰撞凸包线框，以及黑色临时指尖定位球
.venv/bin/python preview/scripts/preview_pose.py --show-pad-collisions --show-fingertips
# 取消两对装配碰撞排除，检查原始接触
.venv/bin/python preview/scripts/preview_pose.py --headless --include-assembly-collisions
# 临时调整安装高度 Z／深度 X，单位米
.venv/bin/python preview/scripts/preview_pose.py --headless --height-offset 0.01 --depth-offset -0.02
# 推进物理或导出全部场景资源
.venv/bin/python preview/scripts/preview_pose.py --headless --settle-steps 250
.venv/bin/python preview/scripts/preview_pose.py --headless --export-xml preview/outputs/scene
```

坐标轴、半透明、碰撞线框与指尖球显示开关不改变物理；`--include-assembly-collisions` 会恢复被排除的碰撞。JSON 记录镜像右手标识、暂定力矩、坐标转换、关节动作顺序、定位点和穿透。导出 XML 不保存本次初始 qpos，精确姿态以脚本及 JSON 为准。

Control 面板直接使用短执行器名，例如右手食指侧摆为 `R_Index_SW`，腕部为 `R/L_Wrist_TX/TY/TZ/Pitch/Yaw`；界面不再显示组合模型命名空间 `right_/` 或 `left_/`。同符号 yaw 为左右同向，同符号 pitch 为左右反向（以对齐腕部为基准）。完整缩写、操作与范围见 [预览说明](preview/README.md#交互控制)。预览 JSON 记录打开窗口前的状态，不自动保存之后的滑块操作。

## 保存初始姿态

初始安装位置、固定旋转和关节值在 [`tsensei_hand_constants.py`](tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py)。预览窗口调节 Control 后按 **P**，会备份并将左右手各自的目标、安装位置、腕部平移／pitch／yaw 写回该文件，下次启动生效。也可用 `--save-initial-pose` 在关闭窗口时保存；`--save-pose-source qpos` 选择保存实际角度。完整初值、单位、优先级及恢复方法见 [初始姿态保存说明](preview/README.md#初始关节角度位置与保存)。

## 独立复制、安装与验证

复制本目录即可，可排除 `.venv`、`build`、缓存、`*.egg-info` 和 `outputs`。无需原始 `robopianist/`、V2／V3 目录、ROS、Menagerie 或音色资源。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
MUJOCO_GL=egl .venv/bin/python smoke_env.py --steps 100
MUJOCO_GL=egl .venv/bin/python -m pytest tests -q
```

`requirements-tested.txt` 为本次 Linux / Python 3.12 实际依赖版本，复现时可先安装它；不是跨平台锁文件。模型、转换后的网格、MIDI 都包含在 wheel 安装包中。原始适配验证见 [validation_results.json](docs/validation_results.json)，最新模型替换验证见 [a32_model_replacement.json](docs/a32_model_replacement.json)。

## 训练接口

**2026-09-28：已取消论文比对要求。** 后续训练按当前固定源码、`all_use_this_train.sh`显式参数和F1／碰撞／速度实测优化，不自动采用论文配置，不要求事先阅读或列出论文差异，也不因与论文内容重合而增加确认流程。[论文资料与旧对照记录](docs/references/biomimetics_2026_610_notes.md)仅保留作历史参考。

下方JAX、51维动作与早期Torch接口属于对应日期的实现记录；当前主线与机械臂接口见上方环境总览，run.sh不再决定未来训练默认。

本目录已包含独立的单环境 JAX/SAC 基线，安装、短测命令以及相对 Shadow 的差异和待确认配置见 [SAC 实现与确认表](docs/sac_shadow_review.md)。原预览 `.venv` 保持不变；本机 SAC 验证环境在 `training/.venv`。当前验证为 CPU MuJoCo + CPU JAX 的短程功能测试，未验证训练收敛。

2026-09-17 按用户要求运行旧版单环境小星星训练：双方 `Wrist_TZ` 关节／控制范围已实测为 `[-0.025, 0]` m，初值 0；CPU MuJoCo + JAX SAC，从头训练 100000 步、seed=42、关闭错键终止，沿用旧 JAX 算法参数。运行目录 `runs/legacy-twinkle-tz025-20260917-153647/`，指标在 `training/metrics.jsonl`，每 10000 步评价并保存检查点；当前进度以日志为准。启动及核验记录见 [修改日志](docs/change_log.md)。

**后续主线已调整为 MJLab + MuJoCo Warp + PyTorch SAC + GPU replay**，详见 [MJLab + SAC 方案与实施顺序](docs/dual_a100_sac_plan.md)。现有 JAX SAC 保留为正确性参考。MJLab 第一阶段已提供完整场景导出、单 GPU 物理接口、Control 预览与自动验收，见 [场景迁移与验收步骤](docs/mjlab_scene_acceptance.md)。16／64 环境、独立 reset 和接触止挡已通过本机验收，见 [批量物理与绿色琴键验收](docs/mjlab_batch_acceptance.md)。**Torch 有指法任务现已完成**：乐谱、1180维观测、分项奖励、结束／部分 reset 和指标均已接通，见 [任务接口与验收](docs/mjlab_task_acceptance.md)。按用户决定，错键只扣分、继续演奏，本轮不验收无指法分支。2026-09-24 新导出任务已将 reset 后的上一动作观测改为与物理中立控制一致；旧任务／检查点保留旧合同，详见[并行环境 reset 修复](docs/reset_recovery_20260924.md)。MJLab任务预览默认显示各指颜色、左右手／音名标签和十指分配表，拇指为粉红／品红，实际按下仍为绿色；可用 `--no-show-fingering` 隐藏。**单卡 Torch SAC 与 CUDA replay 已实现并通过数值对照和短训练**，早期曾采用论文Stage1/2预设，现训练配置以`run.sh`及本轮`config.json`为准；训练、评价、保存／恢复和策略播放见 [Torch SAC验收](docs/torch_sac_acceptance.md)。下一步是正式曲目学习验收与双卡基准；短测不证明演奏能力。

修复合同的47动作小星星已从头训练，目录为 `training_mjlab/outputs/sac-twinkle-reset-neutral-no-yaw47-20260924-154950/`；目标5,000,192采样。它没有加载旧网络，因为旧策略是在动作下限 reset 历史上训练的。完成前不能据早期指标判断首步运动是否已达到要求。

```python
import numpy as np
from tsensei_v3_2_pianist import suite

env = suite.load("Toy", seed=0)  # 也接受 MIDI/proto 文件路径或内置曲名
step = env.reset()
spec = env.action_spec()
action = np.r_[env.task.right_hand.neutral_control,
               env.task.left_hand.neutral_control, 0.0]
step = env.step(action)
print(step.reward, step.observation)
env.close()
```

环境沿用 Composer／dm_env 接口。上述裸环境使用物理动作上下限；新增 SAC 适配器接受 [-1,1] 动作并自动映射。SAC 位于 `tsensei_v3_2_pianist/rl/`，通过可选 `train` 依赖安装。虽然动作维数相同，V2、V3 与 V3.2 模型几何、惯量和初始目标不同，不能将 checkpoint 当成已验证的 V3.2 策略。

| 项目 | 当前设置 |
| --- | --- |
| 手指关节 | 每手 20 个，保留原 URDF 关节限位和大小写，小指为 `MCP/swing/PIP/DIP` |
| 腕部 | 每手 5 个虚拟自由度 tx、ty、tz、pitch、yaw |
| 动作 | 51：右手 25、左手 25、延音踏板 1 |
| 模型 | nq=nv=138、nu=50，含 88 个琴键关节 |
| 默认 Toy 观测 | 327，随前瞻和指法选项改变 |
| 仿真／控制周期 | 0.0005 s／0.05 s（100 个子步） |
| 手指驱动 | 弧度位置目标，暂定 kp=1、armature=0.0002、damping=0.05、±0.6 N·m |
| 手指速度 | 源文件全为 0，视为未提供有效速度规格；尚未实现速度限制 |
| 奖励 | 沿用 V3 按键、踏板、能耗、指法／OT 距离及掌部碰撞项 |

每手前 20 个动作依次是拇指 rota_block、rotaback_joint2、MCP、PIP；食指／中指／无名指各为 MCP、swing、PIP、DIP；小指为 MCP、swing、PIP、DIP；最后是 5 个腕部动作。小指 swing 为 X 轴侧摆（±0.35 rad），MCP／PIP／DIP 为屈曲。Control 短名为 `Pinky_MCP/SW/PIP/DIP`。运行 URDF 的关节和连杆采用此命名；原始输入副本保留来源名称，映射记录于 `assets/model_derivation.json`。

动作总数仍为 51，但小指四个槽位的关节定义已改变，旧策略不能仅凭动作维度相同就直接复用，需重新核对映射并训练／验证。Shadow 专用的 reduced_action_space／primitive_fingertip_collisions 设为 True 会明确报错。

## 训练加速与奖励审查（2026-09-18）

MJLab训练已默认开启100子步整体CUDA graph、Torch学习器编译和融合Adam；当前256环境实测吞吐约519→680 transition/s，物理段未明显提速。开关、数值对照和测量边界见[加速验收](docs/acceleration_acceptance.md)，操作见[常用命令第11节](docs/common_commands.md#11-加速开关与待确认奖励2026-09-18)。

论文公开的全部奖励公式连同项目扩展项现为22个可配置项，可用[奖励配置模板](deprecated/reward_experiments/current_schema1_reward.json)逐项设置enabled／weight，训练传`--reward-config`。默认仍保留原五项规则。[rew.总表与必填字段](docs/reward_design_review.md)说明单位、权重、dt和各项开关；按用户决定，腕方向、等效角、自然姿态、接触类别等未知映射必须显式填写，否则不能启用。完整配置随checkpoint保存，回放自动恢复。奖励不能替代速度限制或接触标定；[实际验收](docs/reward_options_acceptance.md)不代表长期训练收敛。

## 坐标与模型转换

**V32-01 状态（2026-09-17）：用户认为当前安装方案已基本完成，作为仿真训练迁移基线接受。** 真实安装测量留作后续部署复核，不阻塞 MJLab 开发；既有虚拟腕部、驱动、接触及正式右手 TODO 继续保留。

原始 V3.2 基座的掌部固定关节为 xyz `(0.008, 0.0069, -0.045)`，rpy `(0, 0, -1.5708)`。为延续 V3 安装约定，将根坐标转换到规范安装系：

- 旋转 `Rz(+1.5708)`；平移约 `(0.0306000294, -0.0014999747, 0.0891380070)` m。
- 转换只改变根坐标表示，保留左手全部连杆的相对运动、质量和惯量；原始模型独立正运动学与惯量测试已通过。
- 规范系掌部锚点沿用 V3 的 `(0.0237, 0.0065, 0.0441380069844)` m。这是方便比较的安装约定，并非实机测量。
- 世界安装位置为右 `(0.18, 0.15, 0.12)`、左 `(0.18, -0.15, 0.12)` m；四元数 wxyz `(0, 0.70710678, 0, -0.70710678)`。这些位置指**转换后的安装系**，不是原 CAD 基座原点。
- 规范局部 −Y→世界 +Y（沿键盘）；+Z→世界 −X（琴键深度）；−X→世界 +Z（高度）。

小指 MCP／swing／PIP／DIP 的暂定初值为 0.32／0／0.45／0.24 rad，按屈曲／侧摆作用迁移。拇指前两关节初值为 1.0／0.4 rad，其余轻弯目标见 `tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py`。V3 的 0.8 rad 拇指第二关节初值会使新结构额外自碰撞，已根据扫描结果调整。

右手按规范系 XZ 平面反射：位置与网格为 `S p`，姿态为 `S R S`，转动关节轴为 `−S a`，惯量为 `S I S`，其中 `S=diag(1,-1,1)`；同时翻转三角面顶点顺序、变换法线。保留同一关节角范围和初始角度。**这一数学镜像不能证明真实右手的内部结构、驱动方向或惯量相同**（V32-10）。

左手 index／ring MCP 网格各有 208154 个三角面，超过 MuJoCo 二进制 STL 的 200000 面限制。包内保留 STL，并生成相同顶点和三角面的 OBJ；加上镜像右手，共 4 个 OBJ，没有简化几何。

### 可复现生成

```bash
.venv/bin/python scripts/prepare_models.py
```

只读取包内 `assets/source/left_original.urdf` 与 `assets/meshes_left/*.STL`，生成规范左 URDF、临时右 URDF、右 STL、4 个 OBJ，以及 [转换参数与生成文件散列](tsensei_v3_2_pianist/assets/model_derivation.json)。该命令会重建这些派生文件；定制生成逻辑时应修改脚本并记录日志。正常安装已包含派生文件，无需再次生成。

## 指尖与碰撞限制

十个指尖定位点由各末节局部 +Z 最远端顶点的包围盒中点计算，暂用于距离奖励；它们不是经实测确认的触键中心。当前极值面有 4～8 个顶点，详见 [几何测量](docs/fingertip_measurements.md)。球半径 4 mm，仅显示，不参与碰撞。

当前只排除左右各一对掌部／`thumb_rotaback_link2` 碰撞，未继承 V2 的 8 对 MCP 排除。在当前姿态恢复装配碰撞后最大约 2.51 mm 重叠，尚需区分 CAD 凸包假碰撞和真实装配干涉（V32-06）。排除在整个运动范围内有效，其他手指、双手和琴键碰撞保留；默认姿态在此设置下为 0 接触。

## 来源与维护

- 源码直接来自 V3 独立环境，其 RoboPianist 抽取来源保存在 [source_manifest.json](docs/source_manifest.json) 的来源链中；保留版权头与 `LICENSE.robopianist`。
- 原始左 URDF 和 22 个左 STL 与用户文件字节一致；规范左 URDF、右模型与 OBJ 为显式派生资源。用户模型不自动适用 RoboPianist 源码许可证。
- 未复制或导入 `external_never_copy_only_reference/`。坐标参考沿用 V2／V3 已建立的安装约定。
- `AGENTS.md` 与本机日志技能要求每次修改记录问题、实际参考、修改位置与原因、验证及 TODO；独立复制后仍维护自身日志。

## 当前模型修订

2026-09-16 接入 `/home/houjue/A3.2 URDF/urdf/A3.1V6 26.8.4-2.urdf` 及配套 22 个 STL，重新生成左右规范 URDF、22 个右手 STL 和 4 个大网格 OBJ。右手是数学镜像，仍待正式厂商模型确认。来源及 SHA-256 见 [来源清单](docs/source_manifest.json)。旧模型及相关映射／快照已备份到 `model_backups/before_a32_20260916_141929.tar.gz`。独立复制时可排除 `model_backups/`。

### 小指命名约定

左右手均采用下表，并同步运行 URDF 的 joint、link、parent/child 引用及 Control 标签。此次仅改名，保持关节顺序、轴、限位、几何、惯量与初始角度不变。

| 原始 A3.2 名称 | 当前运行名称 |
| --- | --- |
| rotaback | MCP |
| MCP | swing |
| swing | PIP |
| PIP | DIP |

`assets/urdf/TsenseI_left.urdf` 和 `TsenseI_right.urdf` 是更新后的运行模型。`assets/source/left_original.urdf` 与 STL 文件名保留原始输入以便追溯；`scripts/prepare_models.py` 会同时应用整张映射表，重新生成不会恢复旧名称。小指指尖 site 跟随重命名后的 DIP 连杆。此次改名不改变已有四个动作槽位的物理含义，但按字符串绑定关节／执行器的代码需要使用新名称。

### 物理参数与原版Shadow对照

[当前冻结场景／GPU运行值与原版Shadow物理对照](docs/physics_shadow_comparison.md)：包含步长、接触合并、琴键止挡、摩擦、碰撞几何和实际GPU容差，核对日期2026-09-18。
## 2026-09-28 奖励单项对照

当前 `run.sh` 的 Actor／Critic／温度学习率均为 `3e-4`。新增三个独立参数：`--anticipatory-fingering-time .15`、`--staged-fingering-reward`、`--substep-rewards`；默认仅开启子步漏采修复。前两项仍等待实验选择，整指净空控制及按压许可保持可选。

使用 `python3 scripts/run_reward_comparison.py --samples 1000000` 顺序运行基线和三个单项组，保存完整参数、源码归档和比分。提前定位组增加 50 维未来用指观测，因此需要匹配的新检查点。公式、采样范围、可选关闭方式及实际结果见 [本轮对照说明](docs/reward_comparison_20260928.md)。


## 当前 TsenseI_A3.2 提前定位训练

带 ID 新模型的接入、单位／碰撞修复和启动记录见 [本轮说明](docs/tsensei_a32_training_20260928.md)。
此处记录2026-09-28在node4使用`run_tsensei_a32.sh`的历史训练；该包装入口保留兼容，不用于新训练。当前唯一入口为`all_use_this_train.sh`，机器、分配与本轮参数以Slurm及config为准。
当时通用`run.sh`的提前定位默认为0；其文件头是历史CLI记录。
初始姿态图在 `preview/outputs/TsenseI_A3.2_initial.png`。旧模型单独备份，新源码不兼容旧模型 checkpoint，回放旧训练须使用对应源码归档。


## SwanLab 与最新训练时序分析

`run.sh` 默认开启 SwanLab，上传顺序为 F1、recall、precision、启用奖励的字母序；可用 `--no-swanlab` 关闭。监控环境单独安装，不改变训练依赖或 SAC 参数。

- 新机器安装：`bash scripts/setup_swanlab.sh`；密钥使用 `SWANLAB_API_KEY` 或仓库外 `~/.config/tsensei-pianist/swanlab.key`。
- [上传参数、指标口径与失败补传说明](docs/swanlab_training_metrics_20260928.md)
- [本轮 SwanLab 实验](https://swanlab.cn/@blackswan17/TsenseI_A3.2/runs/qgbeil7c)
- [完整训练与逐音时序报告](training_mjlab/outputs/TsenseI_A3.2_anticipatory_20260928/run-20260928-105754-540954/analysis_timing_20260928/report.md)
- [奖励机制与 F1 改进候选](docs/tsensei_a32_timing_reward_review_20260928.md)

本轮 best 音乐 F1 为 0.978843，final 回退；改进优先考虑周期内正确按压与保持。报告区分原 MIDI 和 20 Hz 目标网格，新的奖励候选尚未实施。


## 正确按键奖励修复（2026-09-28）

此节为旧schema的历史修复；下列子步公共奖励CLI已从当前源码删除，新训练会明确拒绝。当前Shadow控制帧评分读取唯一奖励总表，旧checkpoint仍匹配原归档。

`run.sh` 新增显式 `--target-key-substep-reward --target-key-activation-weight 0.5`：正确键正分按200Hz物理子步平均，连续键程和真实触发各占一半，减少迟按/短按和浅按未触发仍拿高分的空间。总权重、正式F1、20/200Hz和三个3e-4学习率保持。

恢复旧正确键公式可追加 `--no-target-key-substep-reward --target-key-activation-weight 0`。旧checkpoint按原源码回放；更换奖励须显式迁移。106项测试和10.24万样本短训练通过，尚未声称已提高长期F1。

[公式、参数与验收](docs/target_key_reward_fix_20260928.md) · [其他F1影响因素](docs/f1_additional_audit_20260928.md)


## 本地网页 online_control（2026-10-04）

本地WSL三页中文控制台：训练结果与预览、左右臂手控制、相机记录与钢琴标定。启动默认hardware但不连接/使能设备，mock需显式`--mock`。本地WSL运行 `bash online_control/install.sh --hardware`、`bash online_control/run.sh --port 18765`，Windows浏览器访问 `http://127.0.0.1:18765`。远程仅负责SSH文件浏览/Slurm导出，本地后端调用网口/USB SDK并记录。可生成独立下载包，见 [WSL部署](online_control/WSL.md)。

运行步骤、测试和明确的硬件门禁见 [online_control/README.md](online_control/README.md)；训练结果/旧NPZ转换见 [导出说明](online_control/EXPORT.md)，checkpoint重放仍通过 `scripts/hpc/submit.sh online-export` 在Slurm中执行。策略action、控制目标和实际状态分开，未知SDK映射/限值只可预览。本轮只进行mock和CPU离线验收，没有真机联调；网口独占控制、实机空间约束、右手SDK标定与物理停止能力未验证。原上位机、训练奖励和学习配置不变。
