# TsenseI A3.2 钢琴环境

十指任务点按方向A沿各自末节向掌根回退3mm，作用于指法距离和指尖位置观察；耦合方向与物理碰撞保持原定义。见[指尖调整与验收](docs/fingertip_setback_20261022.md)。

钢琴当前采用用户指定的1224 mm横向布局：白键22.5 mm，黑键上表面10×90 mm、底面12×95 mm、高18 mm，后端齐平、原中心基准不动；两/三黑键组中心排列保持。黑键实体与坐标原点见[形状说明](docs/black_key_shape_20261023.md)。尺寸公式、验证与局限见[琴键布局说明](docs/piano_layout_1224_20261022.md)。

本机查看新钢琴：`./launch_piano_preview.sh`。加载已验证的MuJoCo 3.11.0二进制模型，支持鼠标旋转/平移/缩放、1总览/2顶视/3近景、H显示双手、O显示钢琴真实原点及XYZ轴；仅静态显示，不训练、不运行物理积分。需要`preview/outputs/piano_black_key_20261023/`模型包；启动器保留3GiB内存上限及至少4GiB可用内存检查。

当前统一训练入口默认安装参数：`height=0.09 m`、`yaw=15°`（0.2617993877991494 rad）、`ty=0.04 m`。命令末尾显式参数仍可覆盖；已提交作业保留其冻结配置。


指法与时间编辑：运行 `./launch_fingering_editor.sh`，默认导入用户的 exper024 示例 `.proto`；支持逐音修改十指编号、起音/松键时间、撤销及另存，并在远端CPU求解当前A3.2三维按键姿态。使用方法与预览局限见 [指法编辑器](docs/fingering_editor.md)。

指法编辑器连接设置保存在 `configs/fingering_editor/`，旧日志目录内配置自动迁移；当前目标为手指原色，上一组为浅色实填，下一组为下一手指对应的浅色斜线提示，两类提示均可关闭。

2026-10-21：修复指法编辑器的相机上下方向与遮挡，增加左右手标识及本地渲染512MiB内存上限。本地完整源构建压力测试被保护主动终止（OOM=0），默认继续远端求解；设计、命令和实测见 [本地压力测试](docs/fingering_editor_local_stress_20261021.md)。

统一训练入口现采用学习器TF32（采样/评估仍为IEEE FP32）；TX/TY每轴硬速度1.99m/s、硬加速度1.99m/s²，软阈值分别1.90m/s和1.90m/s²。用户015-5新实验exper025从头开始；exper024不改动。完整参数以实际config和[奖励总表](docs/reward_defaults_20261006.md)为准。

当前钢琴独立模型包：[piano_1224_black_key_20261023.zip](exports/piano_1224_black_key_20261023.zip)，包含今天的后端齐平梯形黑键、独立MJCF XML、MuJoCo 3.11.0 MJB、Python生成器、88键尺寸数据、图和验证记录；使用说明见[包内README](exports/piano_1224_black_key_20261023/README.md)。2026-10-22旧包保留，但不含本次黑键形状。

## 获取项目

仓库地址：<https://github.com/AGIMani/TsenseIianist>。模型网格及统一训练入口所需的固定源码归档使用 Git LFS 保存；安装 Git LFS 后执行：

```bash
git lfs install
gh repo clone AGIMani/TsenseIianist
cd TsenseIianist
git lfs pull
sha256sum logs/piano_layout_model_20261022/source_piano_layout_1224_v1.tar.gz
```

归档 SHA256 须与 `all_use_this_train.sh` 中的 `training_archive_sha` 一致。虚拟环境、缓存、实验 `results/`、生成的 `outputs/`、部署副本和重复项目 ZIP 不纳入 Git；文中指向这些本地产物的链接需要对应实验文件。源码、配置、测试、模型资产和历史维护文档纳入版本控制；本地已有实验和历史文件保持原样。依赖环境需按项目安装说明另行准备，GPU 操作仍须通过 Slurm。

当前手模型以用户指定的 `/home/hj/TsenseI_Hand_URDF/TsenseI_hand_A3.2_withid_upper_thumb5mm_noupper_mid/` 目录为来源，左右手分别读取包内URDF，保留其关节树、质量和惯量；当前按用户指定同名目录核对来源，掌根采用CAD零件凸包＋凹形掌壳分解（每侧79片），其余42个link保留原凸包，共200个碰撞组件；详情见[掌根碰撞修复](docs/collision_mesh_repair_20261022.md)。拇指第二旋转关节按用户要求恢复旧轴向与正角坐标，右侧连接件按实测网格四孔位置修正8mm。指尖继续按URDF关节链推导，八指耦合范围按2026-10-22用户要求改为10°～80°；左右十指MCP训练目标统一为0～1.4rad（含拇指），启动校验与Euler保持。详情及验收见[本次模型替换](docs/model_replacement_20261020.md)。旧模型checkpoint不可加载到新模型，下一轮从头训练。

当前按键奖励按用户选择恢复exper001的正确键连续键程评分，同时保留原始MIDI逐物理子步采样、真实触键错键计数和子步平均；旧50%激活混合公式已删除。统一入口绑定新归档，旧奖励合同检查点不兼容，下一轮须从头训练。腕TX/TY保留0.02kg armature并校验编译惯量；当前恢复exper031数值：TX/TY Kp=300、阻尼34.15259872981849、力限±50N；Pitch为8/0.5/±5N·m，保留指尖回退3mm、100%激活和1224mm键盘，见[默认驱动恢复与exper034](docs/exper034_default_wrist031_20261022.md)。Euler保持。公式见 [唯一奖励总表](docs/reward_defaults_20261006.md)，验证与限制见 [本轮修复](docs/training_consistency_20261015.md)。

当前 `tsensei_collision_v1` 跨手碰撞按每个200Hz物理子步采样；20Hz控制步包含10个子步，每个出现任意跨手接触的子步扣−0.05，整步最多−0.5，周期末刷新不重复计分。实现和验证见 [唯一奖励总表](docs/reward_defaults_20261006.md)。

2267 小星星低 F1 的配置与同次回放诊断见 [诊断数据](results/exper002/diagnosis_20261013/diagnosis.json) 和 [变更日志](docs/change_log.md)。原0.9913实验与本轮模型、奖励及控制合同不同；预算完成不等于性能达标。

当前机械臂只维护 `controller=wuji_cartesian_v1`：20Hz 腕部目标经过目标制动、200Hz 连续 IK、SDK 速度/加速度限制及 URDF 位置限制，驱动 Link7—法兰—四孔连接件—手的完整刚性链。竞争的腕 PD 已移除，手指与琴键仍执行物理仿真。默认无臂。

单双臂在线IK计算的瓶颈、可选CPU编译实现及200Hz验收见 [IK实时性分析](docs/arm_ik_realtime_20261018.md)。带臂场景可显式选择 `--arm-ik-backend numba_cpu`；默认后端仍为Torch。固定朝向的 `wuji_cartesian_v1` 没有Pitch动作，统一入口带臂运行须在末尾显式加 `--wrist-pitch-max-velocity 0 --wrist-pitch-max-acceleration 0` 清除通用JSON对该固定通道的运动限值；其余硬限保持。原生带臂checkpoint导出使用 `--video1 native front --plots hand-joints`，直接回放原训练装配和mocap轨迹。

带臂默认新增掌根跟随误差惩罚：双侧各物理子步原始请求到实际掌根误差取最大值，按 `−min((e/1mm)²,5)` 每控制步计分，理想0mm。1mm为诊断超限阈值，允许超限且记录次数/持续子步/进入事件；不改变刚性连接和关节硬约束。具体参数见唯一奖励总表。

2026-10-18 的v9单场景跟随验收连续30分6秒、361200周期零截止超时，全部关节约束通过；范围与原始记录见上述实时性分析。修复训练入口依赖传递后，统一入口短训4341完成32条采样、6次SAC更新及完整评估，已提交小星星双M6约1000万样本 exper012/Slurm4351，见 [提交记录](results/exper012/submission.json)。其完整带臂有声视频及同次原谱对比由后台脚本自动导出回传；仅提交不代表训练完成。此前exper011在采样前失败，诊断记录保留。

## 当前配置

[当前双臂配置](configs/preview/current_arm_training_20261011.json) 保存模型、乐谱、奖励、控制限制、准备段、耦合、安装变换和源文件指纹。双侧 J1–J7 reset 初始角，单位度：

| 侧 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 右 | −84.27055 | −68.29417 | 61.39675 | −89.05803 | −64.834334 | 27.92905 | −7.30834 |
| 左 | 84.27055 | −68.29417 | −61.39675 | −89.05803 | 64.834334 | 27.92905 | −7.30834 |

带臂训练仍须显式提供双侧七角；这组数值是当前配置的初态，不是从历史 checkpoint 恢复的状态。固定基座由初角、已抬高的手根初态及安装变换一次反算。

[当前奖励 JSON](configs/rewards/default_20261006.json) 与 [唯一奖励总表](docs/reward_defaults_20261006.md) 维护现用公式和参数。`humanoid_finger_gesture_reward` 只有闲指目标相对 reset 目标的 L1 惩罚：系数 −0.02，未来 20 控制步用指豁免。没有旧姿态、几何、任务条件等可切换分支。

## 使用与维护

2026-10-22左手控制记录的目标关节速度、加速度及实际电机反馈分析，见 [手指速度与加速度报告](docs/hand_control_velocity_acceleration_20261022.md)。报告区分控制周期和主机时间抖动，以及关节角度与直线电机行程。

已导出的MuJoCo录制回放按2026-10-20用户最新授权默认在本地运行`interactive_replay/launch_local.sh`，直接加载版本匹配的`model.mjb`、使用NVIDIA OpenGL及本地声音，不重新编译XML或仿真。启动器以systemd用户scope保护桌面（内存上限3GiB、禁止swap、至少4GiB可用内存），旧`launch_remote.sh`名称也转为本地启动。2026-10-22新回传交互采用exper027指定viewer的操作方式：底部进度条点击/拖动、1～4总览/近景/侧面/顶视、A切换实际手根/法兰坐标轴、方向键跳转1秒、R重播和倍速；时长、准备段和模型均取本轮真实录制。物理仿真、场景编译、策略推理/训练和其他大型分析仍默认使用远端`volc-a32`，GPU任务经Slurm；具体边界见[AGENTS.md](AGENTS.md)。

从强化学习轨迹到实机执行的输入输出、脚本、命令与完成状态，见 [轨迹标定补偿与机械臂执行流程](docs/rl_to_hardware_trajectory_workflow_20261013.md)。当前已有双侧离线补偿与限幅目标；定位验收、进场、手部映射和实机下发仍待完成，文档区分只读核查、条件复现及待实现入口。

视频通过 `export_video.sh` 提交 Slurm。历史录制的 `goal/fingers` 标注遵循各自冻结的 Shadow 控制网格，逐帧与同次录制核对；2026-10-22新指法合同的评分目标是原谱物理子步，旧网格标注仅属诊断，不能当作新奖励目标。已有完整录制时可用 `--recording` 复用同一轨迹，`--fingering` 显示目标琴键颜色；所有活动视频／回放入口不再绘制手指和琴键悬浮标签。每次交付仍须附带同次实际触键与原始 MIDI 对比 PNG；可用 `scripts/plot_recorded_midi_comparison.py` 对保存的物理按键数据离线绘图。

仅有已录制手部轨迹、希望展示当前模型时，可使用 `export_video.sh --trajectory-bundle /绝对路径/trajectory_bundle.zip --recorded-data /匹配原记录数据目录 --output /新输出目录`（可先加 `--validate-only`）。该入口按关节名称回放位置，并精确恢复相对钢琴的手根变换，不加载旧checkpoint、不运行旧源码、不推进动力学；当前只支持已核对原曲的TwinkleTwinkleRousseau轨迹包。数据目录需提供匹配SHA的 `physics_trace.npz`、`original_config.json`、`original_report.json`、`original_scene.xml`；XML仅作为通道/琴键限位数据读取。声音、绿色触键和对比图来自原回放200Hz实际琴键；原谱连续时序指法只通过琴键颜色提示，不称为当前训练控制帧目标。原记录F1不代表当前模型性能。

远端训练结果默认由本地后台脚本 [`scripts/watch_training_transfer.py`](scripts/watch_training_transfer.py) 自动回传到 `results/experXXX/`，命名和启动验证要求见 [`AGENTS.md`](AGENTS.md)。该脚本依据实验的 `submission.json` 查询Slurm、在结束后同步结果及诊断并校验散列；不需要创建Codex定时任务。回传进度、错误和校验结果保存在对应实验目录的 `transfer.log` 与 `sync_status.json`。

2026-10-13 起，完整交付还要求自动使用本轮 `best.pt` 导出带琴键颜色提示、无悬浮标签的有声视频，并回传同次实际触键与原始 MIDI 对比 PNG 和数据；仅回传训练文件不算完整交付。后台脚本在 submission.json 的 export_required=true 时串联训练终态、Slurm 导出、CPU 原谱对比、回传和音视频/SHA验收；训练文件先行回传时不创建 TRANSFER_COMPLETE。当前自动原谱恢复支持注册小星星；其他曲目须补充明确来源，失败保留诊断且不伪报完整交付。

MJLab 预览的 `finger_clearance=null` 与当前统一训练入口一致；胶囊专用净空模块不适用于当前整link凸包。预览配置已于2026-10-14更新为当前工作区源码指纹，并通过源码、依赖、机械臂、奖励配置校验以及远端完整CPU场景重建；双臂2409与无参手部2413的GPU初始化/reset/单步均通过，交互窗口尚未验收。下面是图形会话中的底层命令，CUDA执行须先满足Slurm分配要求：

```bash
# 无参：从当前源码构建无M6机械臂预览
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py

# 显式双臂预览
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --training-config configs/preview/current_arm_training_20261011.json
```

无参模式继承当前预览preset的手部、乐谱及控制参数，将arm_follow设为null并关闭机械臂运动罚；不用历史冻结场景目录。显式--scene须提供已有manifest，单独--task缺少--scene会提前报错。预览preset不是统一训练默认值总表；当前训练数值仍以统一入口和本轮完整config为准。

训练设计要求统一使用根目录 `all_use_this_train.sh`，经 `scripts/hpc/submit.sh` 进入 Slurm；固定源码路径及SHA由该入口维护，独立部署须携带归档。每次开始训练必须打印最终完整奖励配置。

当前统一入口恢复双卡布局：`cuda:0`负责512个环境采样，`cuda:1`负责TF32学习，通过`--gpus 2`申请并启用单批流水。保持batch1024、每条经验0.25次更新和全部物理/奖励诊断。学习器compile/fused optimizer/CUDA Graph保持开启；物理图和评估指标图默认关闭，额外加速须有独立Slurm速度与数值验收依据，见`logs/dual_gpu_20261020/`。正在运行的exper024/025仍保持各自提交时的四卡冻结配置，不因入口修改自动缩卡。

发现设计错误后直接修改活动实现、配置、校验和文档，不再创建递增的 schema 或 mode 来保留失败方案。旧机械臂方案、奖励配置、实验脚本、测试与说明放在 [deprecated/](deprecated/README.md)，活动入口不读取，也不做旧 checkpoint 兼容转换。JSON 中剩余的 schema 字段用于序列化格式；其他仍可选择的控制/模型功能见环境说明，不能把它们解释成当前机械臂的旧版本。

进一步阅读：[环境与实际边界](docs/training_environment.md)、[刚性机械臂设计](docs/optional_arm_wrist_ik_requirements_20261008.md)、[物理参数报告](docs/physical_parameters_technical_report_20261011.md)、[工作约定](AGENTS.md)、[当前变更日志](docs/change_log.md)。

离线对比绘图也支持 `--reference-events` 与 `--label`：参考事件必须先按训练config／归档核对原始MIDI并记录config SHA；该输入用于注册曲目和已明确记录谱面修复的曲目，不能用量化训练目标代替原始事件。2267／2263实例见 `results/video_exports_2267_2263_20261013/`。

### 2026-10-13 指间碰撞与掌侧网格

同手碰撞奖励只统计同一手不同手指：排除掌部外壳和同指全部关节间接触（包括拇指两个旋转输出），保留物理碰撞与全接触诊断；权重0.1、子步去重累计、每控制步上限5不变。活动URDF引用的base_link矩形凸起已从STL/OBJ删除，左右凸包和代理重建。当前模型及奖励须新训练，旧检查点不兼容。复核说明见[奖励总表](docs/reward_defaults_20261006.md)与`logs/finger_collision_mesh_20261013/`。

左手轨迹补偿与首帧装配导出见 [说明](results/left_wrist_compensation_20261013/README.txt) 和 [完整ZIP](results/left_wrist_compensation_20261013.zip)：附件左wrist在钢琴系补偿[+10,+9.6,-26.5]mm，保留已有yaw=0.15rad，并按当前刚性安装导出法兰轨迹、t=0左臂七角及Slurm2347静态四视图。仅首帧做臂IK，未验证全路径或实机精度。

补偿后的左臂全程关节轨迹已按用户选择保留200Hz／9.15秒，使用当前训练`wuji_cartesian_v1`限制J1–J7速度180度/秒及加速度450／900度/秒²。交付见[完整包](results/left_arm_joint_trajectory_20261013_hard_limited.zip)、[主关节CSV](results/left_arm_joint_trajectory_20261013/controller_limited/left_arm_joints_200Hz_deg.csv)和[使用说明](results/left_arm_joint_trajectory_20261013/controller_limited/README.txt)。允许跟踪误差，另附25ms可选收停段；起点仍为已验证t=0逆解，未规划从实机当前姿态进入。

左臂轨迹的SI单位完整列导出：[q/v/a弧度CSV](results/left_arm_joint_trajectory_20261013/radians_with_derivatives/left_arm_qva_200Hz_rad.csv)，共22列（时间、七角rad、七速度rad/s、七加速度rad/s²），保留1831帧／9.15秒。[下载包](results/left_arm_joint_trajectory_20261013/left_arm_qva_rad.zip)另含25ms收停段及合并版。

右手包的对应导出：[右臂22列弧度q/v/a](results/right_arm_qva_20261013/hard_limited/right_arm_qva_200Hz_rad.csv)、[含原20手指角度的42列合并版](results/right_arm_qva_20261013/hard_limited/right_arm_qva_and_hand_200Hz_rad.csv)、[完整包](results/right_arm_qva_20261013_hard_limited.zip)。保留200Hz／9.15秒，采用原包连续IK参考＋当前训练制动／硬限环节，另附20ms收停段；具体控制区别、跟踪误差与限制见包内README。

完整双手同步数据：[左右手关节＋wrist导出包](results/both_hands_joints_wrist_20261013.zip)，包含每侧7臂＋20手指的54关节q/v/a、钢琴系wrist请求／执行位姿、分侧文件与关节映射，统一1831帧／200Hz／9.15秒。总表为[177列CSV](results/both_hands_joints_wrist_20261013/both_joints_and_wrist_200Hz.csv)；手指缺历史的初始派生导数标NaN，原记录速度另存，详见包内说明。

本机稳定性排查见 `logs/codex_stability_20261014/`：已证实Codex界面及大型Python/Git任务遭遇系统OOM。大型本地CPU分析统一使用 `scripts/run_cpu_analysis.sh COMMAND ...`，超出预算须转远端；Git自动打包已关闭并限制打包内存。额外4GiB交换空间的管理员脚本为 `scripts/setup_local_swap.sh`，实际启用状态以 `swapon --show` 为准。

### Prelude 修订指法训练（2026-10-18）

本轮 exper010 使用 `training_mjlab/inputs/prelude_bwv847/candidate_fingering/prelude-bwv-847_fingered.proto`（61处指法修订），固定 yaw=30°，当前 Pitch/MCP 目标范围；从头约2500万步。实际提交参数、固定源码SHA与自动视频/原谱对比回传状态以 `results/exper010/submission.json`、`sync_status.json` 为准。对比参考为与checkpoint逐字节校验的注册原谱proto事件，未经控制网格量化；更早 timing_normalized.mid 有5个音符松键时间差，不用它替代注册原谱。

Prelude续训状态（2026-10-18）：用户撤回中断时exper010已取消，现以其6500向量步best.pt在exper013恢复（Slurm4394）；按用户最新要求累计约2000万步，续训16,672,256 transition，原网络/优化器/RNG恢复但经验池重建。原导出4391已取消，exper013结束后的完整自动导出回传配置见 `results/exper013/submission.json`。

Prelude采样性能排查（2026-10-18）：同为6820批时，exper010采样200.4分钟、exper009为25.9分钟，学习更新均约27分钟。原因是200Hz耦合子步退出整步CUDA Graph后，硬限位后的forward逐步走Python原生路径。修复版只在普通子步循环复用MJLab forward Graph，保留200Hz耦合、限位、物理采样与奖励；H100 Slurm1840同状态forward结果差异为0，512环境采样提速2.11倍。已按用户要求迁移到H100 exper015/1841，20小时限时，状态和来源记录见 `results/exper015/submission.json`。

2026-10-20 Turkish March：当前无臂 A3.2 指法比较、原谱1000 Hz实验及局限见 [曲目说明](training_mjlab/inputs/turkish_march_advanced/README.md)；训练与自动交付登记见 `results/exper023/submission.json`。

2026-10-20 当前015-5训练：用户已批准47处提前松键、指法不变，exper024／Slurm6251，20Hz控制／200Hz物理。详见 [谱面批准与训练说明](training_mjlab/inputs/pig_015_5/README.md)。

### 015-5 无强化学习 IK 物理轨迹（进行中）

本轮使用原始 MIDI，通过独立关节 IK 和真实接触驱动琴键；模型工作在 `volc-a32`，当前尚未达到整曲 F1≥0.95。用户已允许此独立实验关闭八指控制耦合，并授权修订不可达指法、修复拇指键侧/键底接触；修订谱单独保存在`training_mjlab/inputs/pig_015_5/ik_revised_v2/`，原始音高、时值及训练输入不变。用户最新要求保留少量拇指指法、闲置参考initial；v6候选从实际触键证据筛选保留10个拇指音符（右1左9），新轨迹仍须完整复验；开局避碰、闲置回位与整曲F1继续优化。全禁用拇指的避碰版本只作对照，旧v55的“零键接触”不代表跨手避碰通过。执行合同、来源与进度见 [IK 物理轨迹说明](docs/ik_015_5_physical_20261020.md)。

当前指法奖励按原谱200Hz物理子步评分，先对当时用指平均，再对10子步平均；当前与未来预告分开，20Hz动作及学习预算不变。旧指法端点/提前/分阶段配置与checkpoint拒绝，新训练从头开始；完整定义与验收见[奖励总表](docs/reward_defaults_20261006.md)。本轮仅修复并验收，未启动训练。

当前指法奖励目标（2026-10-22）：十指统一白键距世界+X前端40mm、黑键20mm，Y/Z定义不变；详见奖励总表。

### 2026-10-21 同指量化冲突

2026-10-21历史exper027沿用原版RoboPianist floor起音/ceil松键转换；strict_v1不再因同一控制帧同指多键拒绝启动，全部目标保留，finger_conflict记录为warnings。音符缺失、重复音无释放帧及非法输入仍拒绝。原谱指法和时值不改；exper027使用新冻结源码从头训练。

2026-10-22后续统一入口无臂训练的左右Palm_Base_Pitch范围为0～0.5rad（初值0）；运行中的exper031仍使用其冻结的0～0.3rad配置。目标速度/加速度限制保持3.1416rad/s和15.708rad/s²。

琴键实际激活统一为100%归一化角行程（白/黑键标称约10/8mm）；`key_press_reward` 正确键连续满分门槛保持95%。训练子步、F1、声音、高亮和原谱对比图共用该判据；旧0.5°阈值的配置/检查点须使用原冻结依赖，当前入口拒绝静默迁移。

2026-10-22：双侧四指闲指/reset参考指尖升高8mm（拇指、腕、SW不变）；闲指L1中八指SW显式`swing_scale=0.5`，等效系数−0.01，其余−0.02。运行时八指耦合范围10°～80°，参考姿态不强制该角度约束。当前训练归档以统一入口SHA为准，详情见奖励总表。

后续交互导出包的启动模板采用用户指定的 `results/exper027/right_arm_prep_smooth_20261022/replay/launch_local.sh`，维护于 `scripts/launch_local_recorded_replay.sh`：同目录 `model.mjb` + `viewer.py`、NVIDIA/OpenGL及3GiB独立scope；播放器读取本轮匹配录制，不使用exper027专属轨迹数据。

- 当前钢琴尺寸总图（未按下、mm）：[PNG](logs/piano_dimensions_20261022/piano_dimensions.png)、[可缩放SVG](logs/piano_dimensions_20261022/piano_dimensions.svg)、[全部键对距离](logs/piano_dimensions_20261022/all_key_pair_distances_mm.csv)。

- [DIN8996待测尺寸图](logs/din8996_reference_20261022/din8996_dimensions_pending.png)：未取得标准全文，未知项均待测；[来源及钢琴URDF检索](logs/din8996_reference_20261022/sources.txt)。

- [NVlabs局部5黑键全部10组横向距离](logs/nvlabs_black_key_distances_20261022/black_key_all_distances.png)：中心距与投影边缘距离分开标注，非当前训练模型。

- [用户Piano.obj尺寸图（白键宽中位数标定22.5mm）](logs/user_piano_obj_20261022/piano_obj_dimensions_mm.png)：原模型不改，完整标定与测量数据同目录。

exper033为exper032腕部驱动单因素对照：通过统一入口追加专用源码归档/SHA恢复exper031腕部驱动，其他冻结配置与1000万经验预算保持032；不修改统一脚本默认值。提交及自动回传状态见results/exper033/submission.json与sync_status.json。

- [1224mm理想黑键布局图](logs/piano_layout_1224_20261022/piano_layout_1224.png)：白键22.5mm、黑键12mm、均匀缝隙1.058824mm；非实琴标定。

2026-10-23：交互回放打包固定使用独立的纯琴键着色模块，不再复制训练归档中的旧指尖/目标标签逻辑；验收逐显示采样检查无文字和悬浮球，详见[回放标签清理](docs/replay_display_cleanup_20261023.md)。
