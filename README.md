# TsenseI A3.2 钢琴环境

## 获取项目

仓库地址：<https://github.com/AGIMani/TsenseIianist>。模型网格及统一训练入口所需的固定源码归档使用 Git LFS 保存；安装 Git LFS 后执行：

```bash
git lfs install
gh repo clone AGIMani/TsenseIianist
cd TsenseIianist
git lfs pull
sha256sum logs/mount_alignment_20261013/source_mount_aligned.tar.gz
```

归档 SHA256 须与 `all_use_this_train.sh` 中的 `training_archive_sha` 一致。虚拟环境、缓存、实验 `results/`、生成的 `outputs/`、部署副本和重复项目 ZIP 不纳入 Git；文中指向这些本地产物的链接需要对应实验文件。源码、配置、测试、模型资产和历史维护文档纳入版本控制；本地已有实验和历史文件保持原样。依赖环境需按项目安装说明另行准备，GPU 操作仍须通过 Slurm。

当前按键奖励已恢复 Job1724 的原谱物理子步评分：正确键50%键程＋50%真实激活，子步奖励取均值。周期末纯键程算法已删除；其他奖励/模型/硬限不随此变更回退。统一入口绑定新归档，旧奖励合同检查点不兼容，下一轮须从头训练。公式与验证索引见 [唯一奖励总表](docs/reward_defaults_20261006.md)。

2267 小星星低 F1 的配置与同次回放诊断见 [诊断数据](results/exper002/diagnosis_20261013/diagnosis.json) 和 [变更日志](docs/change_log.md)。原0.9913实验与本轮模型、奖励及控制合同不同；预算完成不等于性能达标。

当前机械臂只维护 `controller=rigid_wrist_command`：20Hz 腕部目标经过目标制动、200Hz 连续 IK、SDK 速度/加速度限制及 URDF 位置限制，驱动 Link7—法兰—四孔连接件—手的完整刚性链。竞争的腕 PD 已移除，手指与琴键仍执行物理仿真。默认无臂。

## 当前配置

[当前双臂配置](configs/preview/current_arm_training_20261011.json) 保存模型、乐谱、奖励、控制限制、准备段、耦合、安装变换和源文件指纹。双侧 J1–J7 reset 初始角，单位度：

| 侧 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 右 | −84.27055 | −68.29417 | 61.39675 | −89.05803 | −64.834334 | 27.92905 | −7.30834 |
| 左 | 84.27055 | −68.29417 | −61.39675 | −89.05803 | 64.834334 | 27.92905 | −7.30834 |

带臂训练仍须显式提供双侧七角；这组数值是当前配置的初态，不是从历史 checkpoint 恢复的状态。固定基座由初角、已抬高的手根初态及安装变换一次反算。

[当前奖励 JSON](configs/rewards/default_20261006.json) 与 [唯一奖励总表](docs/reward_defaults_20261006.md) 维护现用公式和参数。`humanoid_finger_gesture_reward` 只有闲指目标相对 reset 目标的 L1 惩罚：系数 −0.05，未来 20 控制步用指豁免。没有旧姿态、几何、任务条件等可切换分支。

## 使用与维护

从强化学习轨迹到实机执行的输入输出、脚本、命令与完成状态，见 [轨迹标定补偿与机械臂执行流程](docs/rl_to_hardware_trajectory_workflow_20261013.md)。当前已有双侧离线补偿与限幅目标；定位验收、进场、手部映射和实机下发仍待完成，文档区分只读核查、条件复现及待实现入口。

视频通过 `export_video.sh` 提交 Slurm。指法标注使用训练实际应用的 Shadow 控制帧 `goal/fingers`，逐帧与同次录制核对；精确音符事件用于原谱诊断，不能替换标注目标。已有完整录制时可用 `--recording` 复用同一轨迹，`--fingering --finger-labels` 显示指法与指尖编号。每次交付仍须附带同次实际触键与原始 MIDI 对比 PNG；可用 `scripts/plot_recorded_midi_comparison.py` 对保存的物理按键数据离线绘图。

仅有已录制手部轨迹、希望展示当前模型时，可使用 `export_video.sh --trajectory-bundle /绝对路径/trajectory_bundle.zip --recorded-data /匹配原记录数据目录 --output /新输出目录`（可先加 `--validate-only`）。该入口按关节名称回放位置，并精确恢复相对钢琴的手根变换，不加载旧checkpoint、不运行旧源码、不推进动力学；当前只支持已核对原曲的TwinkleTwinkleRousseau轨迹包。数据目录需提供匹配SHA的 `physics_trace.npz`、`original_config.json`、`original_report.json`、`original_scene.xml`；XML仅作为通道/琴键限位数据读取。声音、绿色触键和对比图来自原回放200Hz实际琴键；指尖编号与原谱连续时序指法作为展示注释，不称为当前训练控制帧目标。原记录F1不代表当前模型性能。

远端训练结果默认由本地后台脚本 [`scripts/watch_training_transfer.py`](scripts/watch_training_transfer.py) 自动回传到 `results/experXXX/`，命名和启动验证要求见 [`AGENTS.md`](AGENTS.md)。该脚本依据实验的 `submission.json` 查询Slurm、在结束后同步结果及诊断并校验散列；不需要创建Codex定时任务。回传进度、错误和校验结果保存在对应实验目录的 `transfer.log` 与 `sync_status.json`。

2026-10-13 起，完整交付还要求自动使用本轮 `best.pt` 导出带指法标注的有声视频，并回传同次实际触键与原始 MIDI 对比 PNG 和数据；仅回传训练文件不算完整交付。后台脚本在 submission.json 的 export_required=true 时串联训练终态、Slurm 导出、CPU 原谱对比、回传和音视频/SHA验收；训练文件先行回传时不创建 TRANSFER_COMPLETE。当前自动原谱恢复支持注册小星星；其他曲目须补充明确来源，失败保留诊断且不伪报完整交付。

MJLab 桌面预览（需要图形会话）：

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --training-config configs/preview/current_arm_training_20261011.json
```

训练设计要求统一使用根目录 `all_use_this_train.sh`，经 `scripts/hpc/submit.sh` 进入 Slurm；固定源码路径及SHA由该入口维护，独立部署须携带归档。每次开始训练必须打印最终完整奖励配置。

发现设计错误后直接修改活动实现、配置、校验和文档，不再创建递增的 schema 或 mode 来保留失败方案。旧机械臂方案、奖励配置、实验脚本、测试与说明放在 [deprecated/](deprecated/README.md)，活动入口不读取，也不做旧 checkpoint 兼容转换。JSON 中剩余的 schema 字段用于序列化格式；其他仍可选择的控制/模型功能见环境说明，不能把它们解释成当前机械臂的旧版本。

进一步阅读：[环境与实际边界](docs/training_environment.md)、[刚性机械臂设计](docs/optional_arm_wrist_ik_requirements_20261008.md)、[物理参数报告](docs/physical_parameters_technical_report_20261011.md)、[工作约定](AGENTS.md)、[当前变更日志](docs/change_log.md)。

离线对比绘图也支持 `--reference-events` 与 `--label`：参考事件必须先按训练config／归档核对原始MIDI并记录config SHA；该输入用于注册曲目和已明确记录谱面修复的曲目，不能用量化训练目标代替原始事件。2267／2263实例见 `results/video_exports_2267_2263_20261013/`。

### 2026-10-13 指间碰撞与掌侧网格

同手碰撞奖励只统计同一手不同手指：排除掌部外壳和同指全部关节间接触（包括拇指两个旋转输出），保留物理碰撞与全接触诊断；权重0.1、子步去重累计、每控制步上限5不变。活动URDF引用的base_link矩形凸起已从STL/OBJ删除，左右凸包和代理重建。当前模型及奖励须新训练，旧检查点不兼容。复核说明见[奖励总表](docs/reward_defaults_20261006.md)与`logs/finger_collision_mesh_20261013/`。

左手轨迹补偿与首帧装配导出见 [说明](results/left_wrist_compensation_20261013/README.txt) 和 [完整ZIP](results/left_wrist_compensation_20261013.zip)：附件左wrist在钢琴系补偿[+10,+9.6,-26.5]mm，保留已有yaw=0.15rad，并按当前刚性安装导出法兰轨迹、t=0左臂七角及Slurm2347静态四视图。仅首帧做臂IK，未验证全路径或实机精度。

补偿后的左臂全程关节轨迹已按用户选择保留200Hz／9.15秒，使用当前训练`rigid_wrist_command`限制J1–J7速度180度/秒及加速度450／900度/秒²。交付见[完整包](results/left_arm_joint_trajectory_20261013_hard_limited.zip)、[主关节CSV](results/left_arm_joint_trajectory_20261013/controller_limited/left_arm_joints_200Hz_deg.csv)和[使用说明](results/left_arm_joint_trajectory_20261013/controller_limited/README.txt)。允许跟踪误差，另附25ms可选收停段；起点仍为已验证t=0逆解，未规划从实机当前姿态进入。

左臂轨迹的SI单位完整列导出：[q/v/a弧度CSV](results/left_arm_joint_trajectory_20261013/radians_with_derivatives/left_arm_qva_200Hz_rad.csv)，共22列（时间、七角rad、七速度rad/s、七加速度rad/s²），保留1831帧／9.15秒。[下载包](results/left_arm_joint_trajectory_20261013/left_arm_qva_rad.zip)另含25ms收停段及合并版。

右手包的对应导出：[右臂22列弧度q/v/a](results/right_arm_qva_20261013/hard_limited/right_arm_qva_200Hz_rad.csv)、[含原20手指角度的42列合并版](results/right_arm_qva_20261013/hard_limited/right_arm_qva_and_hand_200Hz_rad.csv)、[完整包](results/right_arm_qva_20261013_hard_limited.zip)。保留200Hz／9.15秒，采用原包连续IK参考＋当前训练制动／硬限环节，另附20ms收停段；具体控制区别、跟踪误差与限制见包内README。

完整双手同步数据：[左右手关节＋wrist导出包](results/both_hands_joints_wrist_20261013.zip)，包含每侧7臂＋20手指的54关节q/v/a、钢琴系wrist请求／执行位姿、分侧文件与关节映射，统一1831帧／200Hz／9.15秒。总表为[177列CSV](results/both_hands_joints_wrist_20261013/both_joints_and_wrist_200Hz.csv)；手指缺历史的初始派生导数标NaN，原记录速度另存，详见包内说明。
