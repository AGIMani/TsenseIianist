# 普通静态预览与当前训练模型核对（2026-10-28）

运行 `./launch_piano_preview.sh`；启动时显示双手可用 `./launch_piano_preview.sh --show-hands`，窗口内H切换双手，1/2/3切换视角，O切换原点坐标轴。入口现在读取`preview/outputs/piano_tcp_soft_20261029/`；旧`piano_black_key_20261023`包完整保留。

新包由当前训练冻结归档`5573700d6cfdd855ecaaef7173c08246b88138bc06c1d31902b316c6f638cfd6`与本轮统一入口参数在volc-a32独立目录构建，使用训练的完整source recipe及`SceneBundle`结构性移除TZ/Yaw。前一版5a5a21db模型在H100构建，旧包保留。本次Pitch[-0.3,0.6]rad、镜像局部Z轴Roll[-0.3,0.5]rad、Ty[-0.05,0.05]m，模型源Tz[-0.6,0.46]m但默认TZ仍结构移除；Pitch/Roll力矩均±18N·m；原谱及legacy覆盖诊断保持；静态原生模型不是GPU批量训练过程的执行验收。

## 已同步及核对

| 项目 | 旧普通静态包 | 新普通静态包／当前训练模型配置 |
|---|---|---|
| 白／黑键标称键程 | 10／8mm | 12／10mm，检查编译joint角上限及几何长度反算 |
| 黑键外形／琴宽 | 已是新外形 | 保持顶10×90、底12×95、高18mm；琴宽1224mm |
| 双手来源 | 已是用户新A3.2双侧URDF | 相同资产；冻结归档逐文件核验后重建 |
| 碰撞几何 | 每侧100组件、掌壳79组件 | 保持全手200组件、两侧掌壳各79组件 |
| 双侧Pitch机械／执行器范围 | 0～0.5rad | −0.3～0.6rad，初值0 |
| 双侧Roll | 无 | 右局部+Z、左−Z；范围[-0.3,0.5]rad、原始初值0 |
| 原始掌位 | 世界X=.14m；Ty=.04m | 世界X=.16m；Ty=.02m；Y=±.15m、Z=.09m、Yaw15°保持 |
| 手指驱动 | Kp=1、通用阻尼/摩擦、armature=.0002 | Kp=2、逐轴Kd/力矩限及拟合摩擦、armature=.0001；40轴识别通过 |
| 腕驱动 | exper031数值 | Tx/Ty Kp300、阻尼34.15259872981849、力限±50；Pitch/Roll Kp8、阻尼.5、力矩±18N·m |
| 模型时间步 | .005s（200Hz） | .002s（500Hz）；源控制配置50Hz |
| MCP目标／本体范围 | 静态显示不执行动作映射 | 核对10个训练目标0～1.4rad；本体URDF0～1.57rad不混为同一范围 |
| 源码守卫 | 4个来源散列，已与当前不匹配 | 更新为本次模型相关来源散列与统一入口SHA；变化时明确拒绝旧包 |

机器可读对比包括编译后的44个变化关节记录、42个变化执行器记录；大部分为手指PD/摩擦变化，Pitch执行器范围也更新。详见`preview/outputs/piano_training_20261028/`内`model_audit.json`、`previous_model_audit.json`和`model_changes.json`；这是新增Roll前的对比记录。当前Roll包核对见其model_audit.json及下方新验收。

## 静态显示与训练的明确边界

显示的是**真实首音准备开始前的原始参考姿态**，不是准备结束后策略学习的reset姿态。预览不执行动力学、首音准备、曲尾回程、策略推理、耦合动作映射、目标速度/加速度限制或奖励累计；参考指姿态本来就不要求满足运行中的10°～80°耦合范围。当前训练配置保留这些过程，本次没有为了显示而改变它们。

原生MuJoCo加载MJB，调用运动学和相机更新；不在本机编译XML或调用`mj_step`。训练采用GPU批量求解，静态加载成功不能证明训练稳定性或数值轨迹等价。

## 另一个入口尚未同步

`./launch_piano_preview.sh --self-collision`仍进入原来的自碰撞检查工具，固定归档`1d4a67f…`、模型`920475c…`，本次未重写其远端合同、MJB或计分协议。它已经包含新琴键外形、12/10mm键程和新双手URDF，但还有：

- 20Hz控制／1000Hz物理、每步50子步；当前训练配置50／500Hz、每步10子步。因此单对持续整步的同手静态估算旧工具为−5，当前公式在10子步下为−1。
- Pitch滑条仍0～0.5rad；当前训练为−0.3～0.6rad。
- 原始Ty=.04m（世界X=.14m）；当前训练Ty=.02m（世界X=.16m）。
- 来源`drive_settings.json`与当前不一致。普通静态预览此次更新不自动更新这个入口。
- 第一次预览核对时`volc-a32` SSH连接被关闭；后续连接已恢复，但未重新启动该工具。以上依据其仍被配置绑定的MJB/source_config及已保存ready响应核对。

## 严格覆盖拒绝与用户后续选择

首次尝试按完整训练recipe重建时，`export_task → physical_coverage`在500Hz下明确报错：`Repeated key 64 has no release sample at physics resolution`。逐音审计发现6处约1.041667ms的松键间隙；200Hz、500Hz均缺失松键采样点，1000Hz均有覆盖。因此当时严格模式无法通过完整任务构建，与内存容量无关。

为完成用户要求的静态模型更新，随后显式只调用训练同一个模型导出函数，不构建音乐任务。`acceptance.json`清楚记录`full_music_task_built=false`；这是模型验收，不是绕过训练校验。随后用户明确选择保留50/500Hz、去除严格校验。当前入口已改为legacy覆盖诊断，保留原谱和exact_note_timing；导出/加载/运行时统一处理，缺失采样点不再阻止构建，也不会补造松键帧。6处时间与音高见`logs/static_training_preview_20261028/score_sampling_audit.json`。

## 前一版静态模型验收与来源

- 远端：`h100:/data/share/houj/pianist_static_preview_20261028/`；CPU systemd scope `pianist-static-build-20261028-v2`，MemoryHigh10GiB、MemoryMax12GiB、禁止swap、CPUQuota400%，CUDA不可见，无GPU训练/渲染。复用已核实的独立依赖环境，冻结源逐文件核验，代码来源不使用远端旧工程。
- 新MJB SHA：`2b7e015d7589a4221b23d8f9afee84415998484f878addc44b8e055d4f86fa30`。回传全部10项SHA通过；源守卫与本机当前文件核对通过。
- 本机：MuJoCo3.11.0、NVIDIA RTX3050静态显示6秒／210帧，状态未变，物理步0，峰值2026156032字节（约1.89GiB），内存事件及OOM计数均0；保留原启动器MemoryHigh2300M、MemoryMax3G、禁止swap和至少4GiB可用内存门槛。
- 记录：`logs/static_training_preview_20261028/`中的部署/命令/构建成功与失败日志、采样审计、SHA验收、`preview.png`。完整参数和依赖来源另存新包`source_config.json`、`scene_manifest.json`、`launch_command.json`。

## 关闭严格覆盖拒绝后的完整构建验收

- 远端`volc-a32:/home/houj/projects/pianist_score_coverage_20261028/`，CPU scope `pianist-score-build-20261028`，10/12GiB内存阈值、swap=0、CPUQuota400%；复用用户独立依赖环境，代码由父归档加8项明确文件覆盖冻结，2252个归档成员逐项核对。
- 完整recipe构建、TaskBundle加载及ExactScore CPU目标表通过；11137控制步、13帧松键尾段、2544原音符逐项完全相同。物理漏音0、漏松键间隙6，控制网格覆盖通过；export/load/runtime报告一致，valid=false如实保留。没有运行物理步、策略或训练。
- 最新包`acceptance.json`为`full_music_task_built=true`，`score_acceptance.json`保存上述结果。MJB SHA仍为`2b7e015d7589a4221b23d8f9afee84415998484f878addc44b8e055d4f86fa30`；预览来源已同步当前入口和新归档。
- 本机新版加载6秒/210帧、物理步0、状态不变、峰值约1.89GiB，所有内存事件/OOM计数0。
- 归档/回传/SHA及本机受限显示记录见`logs/score_coverage_20261028/validation.json`；冻结来源以统一入口为准。该条为当时的静态验收记录；后续联合实分配/动力学验收见docs/roll_capacity_acceptance_20261028.md，未开始正式训练。

## Roll／±18N·m静态更新

此前±0.5范围普通包为`preview/outputs/piano_roll_20261028/`。在volc-a32 CPU限额scope重新完整构建，MuJoCo3.11.0，nq=136、nu=48；MJB SHA `af233c87e3efb416152dff6e2490ca4c3a3045edd971441c79ec9eb5368a8486`。模型审计逐轴核对Pitch/Roll范围、Roll镜像轴、Kp8/阻尼.5/±18N·m；全部12项产物回传SHA通过。仍是首音准备前姿态，运行时目标硬速度/加速度由控制器执行，不由静态显示执行。

本机通过原限额启动器显示6秒、311帧，物理步0、状态不变，峰值1.36GiB、内存high/max/OOM事件0。记录在`logs/roll_capacity_20261028/roll/static_viewer.log`及`static_contract.json`。普通预览来源守卫与最终统一入口SHA同步。远端首次CPU导入使用egl时缺少库，随后沿用已有CPU构建的MUJOCO_GL=disable，不渲染，完成构建；失败日志保留。

## 后续Roll范围[-0.3,0.5]更新

此前普通包piano_roll_range_20261028由ec43ec5b…新归档重新完整构建，核对双侧Roll机械/执行器/策略范围、Pitch±0.5及双轴±18N·m保持；MJB SHA `6cc67e8f4288f1d0fbe05b0ce9b8d6f11bceb7993d64158e12ff225c07fe236e`，仍MuJoCo3.11.0、nq136/nu48。原始Roll保持0，静态显示不执行动力学。源文件与入口SHA守卫同步，记录见logs/roll_range_20261028。


## Pitch及Ty/Tz最终范围

此前普通包为`preview/outputs/piano_palm_ranges_20261028/`，来源`5573700d6cfdd855ecaaef7173c08246b88138bc06c1d31902b316c6f638cfd6`；MuJoCo3.11.0，nq136/nu48，MJB SHA `6ca19a991eac850484966a8c50df8d8800f2485de362dc0399ee5b806a2c6ece`。双侧Pitch[-0.3,0.6]rad、Roll[-0.3,0.5]rad、Ty[-0.05,0.05]m；源模型Tz[-0.6,0.46]m已编译核对，默认训练和普通预览仍结构移除TZ。原始Pitch/Roll/Tz为0、Ty为0.02m；本包不积分或执行首音准备。12项产物SHA与源码/入口守卫通过；真实动力学短测见logs/palm_ranges_20261028/validation.json（Slurm10530）。

/data已满，新增包和归档通过项目内符号链接指向/home/hj/pianist_artifacts/，原启动命令不变；独立复制时需解引用新增目录链接。旧自碰撞查看器仍独立固定，未在本次同步。


2026-10-28 Tx/Ty目标加速度硬限改为5m/s²，统一入口覆盖值变化后，普通静态预览重新构建到`preview/outputs/piano_palm_acceleration_20261028/`，更新完整入口与命令校验。仍使用5573700d冻结源码；MJB SHA保持6ca19a99…，模型几何/范围/驱动未变。静态预览不执行目标限速或动力学；CPU目标限速验收见`logs/palm_acceleration_20261028/validation.json`。


2026-10-29：Tx/Ty加速度奖励软阈值改为4.75m/s²；当前包`preview/outputs/piano_tcp_soft_20261029/`已按新入口重新构建并校验12项产物与1479项源码守卫，MJB仍6ca19a99…，不执行物理仿真。当前训练归档目录已恢复为实体目录，旧主目录副本保留；本包及归档通过Git LFS发布，克隆后需拉取LFS文件。
