# 当前变更日志

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
