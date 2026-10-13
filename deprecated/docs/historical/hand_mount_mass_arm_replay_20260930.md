# 1078 回放、STEP 安装座、952 g 手模型与机械臂 IK

> 本文记录同日较早的 `arm_hand_connect.STEP` 窄安装座及其历史视频。
> 当前装配已更新为45.6×24.6 mm宽安装座，见[新版安装件说明](tianji_mount_20260930.md)。
> 下文原测量与视频保留，不能当作新安装件验收结果。

## 结果与文件

当前双手 URDF 的腕部安装座已直接替换为 STEP 对应零件，安装面从 **45.6×24.6 mm** 改为 **35.6×24.6 mm**。
中央孔与 STEP 参考手对齐；外轮廓和四个螺栓孔来自同一 STEP 零件。
质量更新为 **0.952 kg/手**，双手合计 **1.904 kg**。

- [新安装座对照图](../preview/outputs/connector_alignment/connector_urdf_replaced.png)、[完整双臂装配图](../preview/outputs/connector_alignment/assembled_arms.png)
- [1078 原物理回放，有声](../preview/outputs/1078_final_20260930_detailed/replay.mp4)
- [增加机械臂和连接件的原速回放](../preview/outputs/1078_final_20260930_arms/arms_original.mp4)
- [满足本次 TCP 限制的放慢回放](../preview/outputs/1078_final_20260930_arms/arms_limited.mp4)
- [左臂七关节速度图](../preview/outputs/1078_final_20260930_arms/left_joint_velocities.png)、[右臂七关节速度图](../preview/outputs/1078_final_20260930_arms/right_joint_velocities.png)
- [关节角度/速度 CSV](../preview/outputs/1078_final_20260930_arms/joint_velocities.csv)、[TCP 限制图](../preview/outputs/1078_final_20260930_arms/tcp_limits.png)、[TCP CSV](../preview/outputs/1078_final_20260930_arms/tcp_limits.csv)

两个机械臂视频沿用 1078 记录的手部轨迹，显示当前安装座。机械臂通过差分 IK 跟随。
未重新模拟新质量下的手部动力学，也未评测新模型 F1。

## URDF 质量与几何

实际依据为用户附件 `arm_hand_connect.STEP`、`质量修改说明.md`、`mass_update_report.json`、
`inertia_update_report.json` 和 `model_derivation.json`。另外三个接触/驱动/碰撞代理附件与项目文件逐字节相同，未修改。
附件作为模型数据使用，未据其文字执行额外训练或其他操作。

`scripts/apply_hand_mass_revision.py` 核验两侧各22个连杆，用新旧质量比缩放每个惯量张量，保留COM。
原质量约0.156939274 kg/手，修订后0.952 kg/手。惯量检查正定性、主惯量三角不等式和镜像关系。
质量修订生成的URDF与附件SHA256完全相同；之后再应用本次安装座替换：

| 模型 | SHA256 |
|---|---|
| 左手，仅质量修订 | `4d8f6c76e231a66fa78c6a38ca28e5729f1195265c1f3ee3b2ed83a9522f6424` |
| 右手，仅质量修订 | `694b15afdfd8dd8384f0d46a31b14d79633db2c7189ffb90ad600ff02b81c0ba` |
| 当前左手，含新安装座 | `c22437512d252dc407fd18f4d9b0377f24cb74332ed53c6f2c16d7d6f4c4b779` |
| 当前右手，含新安装座 | `4b4f1a13f1b985d8a74faf0cbfe754f0169f0af1bb7e8912765a52eac669084b` |

质量版备份位于包内 `assets/source/TsenseI_{left,right}_mass952_before_mount.urdf`，原厂商URDF保留。
新安装座的质量/惯量继续采用所给报告中的估计值，尚未按新CAD或实物重新标定。

`preview/scripts/import_hand_connector.py` 保留STEP实例变换、毫米转米、以0.1mm容差三角化。
装配含转接件1一个、转接件2四个、转接件3一个及参考手一个。
`scripts/replace_hand_mount_from_step.py` 从参考手提取唯一闭合的腕部安装座，装配坐标范围为
`[-16.5,-23,-51]` 至 `[8.5,13,-22]` mm，体积约3481.974 mm³。
中央孔对齐变换为：

```text
STEP_from_left_hand:
R = [[0,1,0], [1,0,0], [0,0,-1]]
t = [-0.008,-0.02175,-0.026] m
```

它将旧手孔中心`[16.75,4,-4]`mm映射到STEP孔中心`[-4,-5,-22]`mm；相对之前外轮廓中心对齐方案，沿STEP Y移动3.25mm。
旧安装座还重复出现在base网格中，因此从掌壳大组件减去旧座，并保留其他原组件。
布尔运算零碎片全部位于旧座范围，剩余掌壳闭合；原安装面残留面积为零。

两个URDF中`base_link`与`{Left,Right}_hand_palm_link`的visual/collision引用分别改为：

- `meshes_{side}/base_without_old_mount.STL`
- `meshes_{side}/wrist_mount_from_step.STL`

手指网格、关节坐标/轴/范围及惯性字段保留。右手仍是镜像推导。
`assets/connector_mount_revision.json`保存来源、变换、输入/输出散列，`model_derivation.json`同步。
`scripts/prepare_models.py`重建时自动重新应用质量和安装座修订；两URDF、四新STL临时重建逐字节一致。
加载已生成模型无需STEP附件或CAD环境，也不读取其他项目。

## 交互预览

有图形桌面的本机可用原入口：

```bash
training/.venv/bin/python preview/scripts/preview_pose.py
```

默认显示六件外部连接件，并有配套窗口：选择左/右手，调XYZ平移（mm）、Roll/Pitch/Yaw（度），恢复中央孔对齐，保存配置。
偏移使用各自手根坐标，连接件和Link7到手的工具变换一起更新；双臂重新拟合当前/重置姿态，不可达时回退。
保存会检查文件是否被其他进程修改，并创建时间戳备份。

默认配置为 `configs/preview/hand_connector_20260930.json`；`--connector-config`可指定副本。
`--no-connector-controls`关闭面板，`--no-hand-connector`关闭外部六件转接件显示；后者不会恢复URDF内已换的安装座。
云端渲染必须经Slurm：

```bash
bash scripts/hpc/submit.sh preview --headless --output /absolute/path/assembly.png
```

Slurm运行源码快照，交互保存应传入指向持久项目文件的`--connector-config`。
已通过真实模型偏移、恢复、保存和文件冲突测试，以及无窗口渲染；未在云端人工操作Tk窗口。

## 1078 严格归档回放

检查点为 `training_mjlab/outputs/hpc/coupling-20260930-full-20260930-012842-704278/final.pt`，
SHA256 `c61fc55ba33054b54906f1f84c22f1696a31f6802087d5f6c6ff47a4d2ba5c5b`。
使用 `logs/hpc/1078/source.tar.gz`，保留源码、依赖、检查点合同校验。

```bash
bash scripts/hpc/submit.sh replay \
  --source-archive /absolute/path/logs/hpc/1078/source.tar.gz \
  --checkpoint /absolute/path/final.pt --mode render --capture-physics \
  --output /absolute/path/new_recording
```

新增观察器只在每物理子步复制状态，保留原音频回调。`physics_trace.npz`含1830个200Hz状态和单独初态，
全长9.15秒，含qpos/qvel/mocap/踏板。保存同一已移除TZ/Yaw的`scene.xml`与`model.mjb`用于FK。

| 独立回放 | Slurm | 音乐平均帧F1 | 音乐键帧micro F1 |
|---|---:|---:|---:|
| 首次原回放 | 1086 | 0.934810 | 0.927273 |
| 带完整子步状态的回放 | 1091 | 0.932489 | 0.925319 |

两个GPU接触回放存在已知浮点轨迹差异，不能称为逐帧相同或替代训练三回合评估。
机械臂视频和曲线均使用1091记录，音频来自真实琴键触发，保留准备段误触。

## 机械臂与TCP限制

M6采用[厂商历史URDF提交](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/tree/1a9cf30770ef6654cc1edfc048e1d0861be27eea/URDF)中的
`Marvin M6-CCS-696-urdf_V4.0.rar`，压缩包SHA256为`8471d1d8c77c56d41101bdbc51ba38323c42a835e870658b363d9f6cb02a027c`。
运行资产/许可证/散列位于`preview/assets/tj_m6`，没有运行硬件SDK。
基座中心距93mm，临时肩座中心`[.52,0,.48]m`，Link7法兰面取网格`Y=-.095m`。
肩座和法兰转角待标定；六个外部连接件和机械臂仅运动学显示，无新增质量或接触反力。

TCP定义为手URDF的`base_link`原点，工具变换来自连接件配置。
初始有界位姿IK后，用阻尼雅可比增量跟随记录，保留厂商关节角范围。
1831个状态最大数值残差约`9.44e-9m`和`4.31e-8rad`，只代表数学求解精度。

用户限制按向量范数解释：**1m/s、2m/s²、1.5rad/s、4rad/s²**。
另外核验厂商URDF中的14个关节速度上限，各为**3.1416rad/s**。
原200Hz记录离散差分如下：

| 手侧 | 线速度m/s | 线加速度m/s² | 角速度rad/s | 角加速度rad/s² |
|---|---:|---:|---:|---:|
| 右 | .262551 | 8.42880 | 2.09596 | 293.920 |
| 左 | .279601 | 8.00937 | 1.47909 | 196.593 |

快速俯仰变化是主要限制，单独降低线速度不能消除角加速度超限。
本次保留原速版，并给出保留整条路径的保守放慢方案：关节序列拟合二阶连续样条，首尾速度为零；
由`v=J qdot`、`a=J qddot+Jdot qdot`算TCP导数，每原始秒1000次采样验收。
按速度比与加速度比平方根取共同时间缩放，再留2%余量。
连续样条峰值与上表离散差分不同，还包含末尾平滑停下的要求，数据分别保存。

缩放系数为**11.495614**，9.15秒路径变为**105.184867秒**，视频加末尾停留后约106.25秒；原速视频10.20秒。
这是可行演示，未进行最短时间轨迹优化。音频同比延长并保持音高，节拍已改变。

| 手侧 | 放慢后线速度m/s | 线加速度m/s² | 角速度rad/s | 角加速度rad/s² |
|---|---:|---:|---:|---:|
| 右 | .023207 | .314558 | .204092 | 3.844675 |
| 左 | .024486 | .325772 | .129200 | 2.252744 |

14关节全段峰值均低于各自限速。图左列为连续IK路径的原时间尺度，右列为放慢时间尺度；PNG/PDF/CSV/NPZ均保留。

```bash
MUJOCO_GL=disable training/.venv/bin/python preview/scripts/replay_arm_ik.py \
  --replay /absolute/path/recording --output /absolute/path/arm_result --analysis-only

bash scripts/hpc/submit.sh arm-replay \
  --replay /absolute/path/recording --output /absolute/path/arm_result --render-only
```

四个限制可在分析时用`--linear-speed-m-s`、`--linear-acceleration-m-s2`、`--angular-speed-rad-s`、`--angular-acceleration-rad-s2`覆盖。
render-only校验轨迹散列和安装参数，沿用保存的限制。
M6/连接件资产仅在preview/arm-replay中显式打包，普通A3.2训练包仍不含M6。

## 验证、依赖与TODO

- 34个相关测试通过：质量/惯量/镜像、STEP几何、归档、打包、原预览控制8项、新安装偏移/恢复/保存测试；TCP导数另与独立圆周运动解析解核对。
- 当前模型CPU实际编译134物理坐标/46驱动器，每手.952kg，惯量正值；见`preview/outputs/hand_mass_revision_20260930/compiled_with_step_mount_validation.json`。
- 1093完成A100装配PNG，1097完成两段有声机械臂视频，ffmpeg全片解码通过。
- 首次渲染1096因自动geom重名失败，已给覆盖层自动名称加前缀，CPU合并编译和1097重跑通过。早期ffmpeg/FluidSynth依赖失败已修复。
- Composer/Torch的MuJoCo版本不同，离线FK使用可移植XML，未混读MJB。桌面交互以程序化事件测试，未声称已人工操作云端Tk。

所有安装位于用户环境。可选绘图依赖在`preview/requirements-replay.txt`；CAD独立环境
`training_mjlab/cad_tools/.venv`使用cadquery-ocp7.8.1.1.post1、trimesh4.11.5、manifold3d3.5.4。
后续加载现成网格无需CAD库。
`preview/scripts/setup_piano_audio.py`将FluidSynth/音源解包到项目私有目录，云镜像不可用时可传
`--mirror https://archive.ubuntu.com/ubuntu/`，通过apt元数据SHA256校验。ffmpeg用现有imageio-ffmpeg二进制，系统库/驱动/CUDA未修改。

TODO(V32-MOUNT-01)：实机孔位、工具方向及更换安装座后的质量/惯量标定。
TODO(TJ-ARM-02/03/04)：临时肩座、工具坐标、真实机械臂动力学和避障。
TODO(V32-MASS-01)：新质量下策略F1/碰撞/驱动稳定性。
本任务未启动新训练或向硬件下发指令；文件、编译和运动学验收不替代上述未完成验证。
