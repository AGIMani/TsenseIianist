# TsenseI V3.2 钢琴演奏环境物理参数技术报告（历史记录）

> 历史参数记录：本文保留2026-10-11报告及其后续补记的测量、参数和工作区状态，正文“当时”均指对应记录时点，不再作为现行默认值表。特别是腕部3kg armature、旧编译惯量/阻尼以及“入口不存在”的描述均为历史状态。当前源码/配置链接会随工作区更新，不能据此把本文旧数值视为新训练合同；现行修改与验证见 [2026-10-15训练一致性修复](training_consistency_20261015.md)，奖励见 [唯一奖励总表](reward_defaults_20261006.md)，固定归档以 [统一训练入口](../all_use_this_train.sh) 为准。

**报告日期：** 2026-10-11

**适用范围：** 当时工作区源码、`current_arm_training_20261011.json` 和项目内 M6/连接件资产

**参数单位：** SI 制；角度为 rad，除非明确标注 degree

## 0. 参数口径和复现边界

本报告把参数分为三类：

1. **当时合同实际使用**：由当时预览/机械臂合同和源码路径共同确定。
2. **源码默认或历史兼容值**：代码仍保留，但当时合同可能覆盖或移除。
3. **名义/未标定值**：存在于 URDF、CAD 或仿真适配配置中，但不能解释成硬件实测值。

本报告编写时记录的工作区没有根目录 `all_use_this_train.sh`，也没有 AGENTS 指定的固定训练归档目录；这不描述2026-10-15后的工作区。因此，正式训练回放仍必须以作业保存的 `config.json`、`scene/manifest.json`、源码归档和 SHA 为最终准据。本报告不把工作区参数称为某个已冻结训练作业的完整运行快照。

报告当时参考的合同文件：

- [当时机械臂/预览合同](../configs/preview/current_arm_training_20261011.json)
- [动作映射](../configs/actions/direct_joint_v1.json)
- [手部左 URDF](../tsensei_v3_2_pianist/assets/urdf/TsenseI_left.urdf)
- [手部右 URDF](../tsensei_v3_2_pianist/assets/urdf/TsenseI_right.urdf)
- [钢琴常量](../tsensei_v3_2_pianist/models/piano/piano_constants.py)
- [钢琴 MJCF 构建](../tsensei_v3_2_pianist/models/piano/piano_mjcf.py)
- [手部 URDF→MJCF 构建](../tsensei_v3_2_pianist/models/hands/urdf_model.py)
- [手部驱动构建](../tsensei_v3_2_pianist/models/hands/tsensei_hand.py)
- [接触参数](../tsensei_v3_2_pianist/assets/contact_settings.json)
- [驱动参数](../tsensei_v3_2_pianist/assets/drive_settings.json)

## 1. 环境装配和时间离散

### 1.1 构建链

```text
TsenseI 左/右 URDF ─┐
                    ├─ Composer/MuJoCo 编译 ─ scene.xml + manifest
88 键钢琴 MJCF ─────┘

无臂模式：scene.xml 直接交给 MJLab

有臂模式：
scene.xml
  → 移除手腕 TX/TY/Pitch 物理自由度
  → 保留手指和琴键动力学
  → 插入 M6 mocap 几何
  → Link7→法兰→四孔连接件→手根刚性装配
  → 连续 IK 和目标制动
  → MJLab/MuJoCo Warp 执行物理子步
```

实现路径：[source_runtime.py](../tsensei_v3_2_pianist/mjlab_env/source_runtime.py)、[bundle.py](../tsensei_v3_2_pianist/mjlab_env/bundle.py)、[scene.py](../tsensei_v3_2_pianist/mjlab_env/scene.py)、[arm_scene.py](../tsensei_v3_2_pianist/mjlab_env/arm_scene.py)。

### 1.2 报告记录的时间参数

| 参数 | 当时值 | 来源 |
|---|---:|---|
| 策略控制频率 | 20 Hz | `current_arm_training_20261011.json` / `source_runtime.py` |
| 控制周期 | 0.05 s | 同上 |
| 物理频率 | 200 Hz | 同上 |
| 物理步长 | 0.005 s | 同上 |
| 每控制周期物理子步 | 10 | `control_dt / physics_dt` |
| 积分器 | Euler | `contact_settings.json`, `physics_settings.py` |
| `impratio` | 10.0 | 同上 |
| 重力补偿 | 手部开启 | `base.py`, 当时合同 `gravity_compensation=true` |
| MuJoCo 世界重力 | 源码未显式覆盖，使用 MuJoCo 默认值 | `base.py` / `physics_settings.py` |

`contact_settings.json` 的基础元数据仍记录 40/400 Hz；20/200 Hz 导出时，钢琴 `solref[0]` 会根据实际物理步长重新设置。正式冻结场景应以其 `manifest.json` 为准。

## 2. 手部物理参数

### 2.1 质量、惯量和链节

每只手有 22 个带惯量的 link，总质量 **0.952 kg**。右手采用左手镜像质量/惯量策略，不是厂商正式右手模型。

| 链节 | 质量 |
|---|---:|
| `base_link` | 0.150 kg |
| palm link | 0.017 kg |
| thumb rota block | 0.035 kg |
| thumb rotaback | 0.075 kg |
| thumb MCP / PIP | 0.040 / 0.015 kg |
| 每个 index/mid/ring/pinky 的 MCP / swing / PIP / DIP | 0.030 / 0.070 / 0.040 / 0.015 kg |

每个 link 的质心 `origin xyz/rpy` 和惯量张量 `Ixx/Iyy/Izz/Ixy/Ixz/Iyz` 均逐项保存在：

- [TsenseI_left.urdf](../tsensei_v3_2_pianist/assets/urdf/TsenseI_left.urdf)
- [TsenseI_right.urdf](../tsensei_v3_2_pianist/assets/urdf/TsenseI_right.urdf)

惯量来源和缩放方法见 [mass_update_report.json](../tsensei_v3_2_pianist/assets/mass_update_report.json) 与 [inertia_update_report.json](../tsensei_v3_2_pianist/assets/inertia_update_report.json)：保持原质心和相对质量分布，仅按新质量比例缩放惯量。连接件几何替换后没有重新进行 CAD/动力学标定。

### 2.2 手指关节和动作范围

| 关节类别 | URDF 机械范围 | 当时策略物理目标范围 |
|---|---:|---:|
| Thumb Rot | 0～2.09 rad | 0.05～1.57 rad |
| Thumb Rot2 | 0～1.74 rad | 0.05～1.57 rad |
| Thumb MCP/PIP | 0～1.57 rad | 0.05～1.20 rad |
| 四指 MCP | 0～1.57 rad | 0～1.20 rad |
| 四指 PIP | 0～1.57 rad | 0～0.90 rad |
| 四指 DIP | 0～1.57 rad | 0～0.80 rad |
| 四指 swing | −0.35～0.35 rad | −0.35～0.35 rad |

动作策略范围和名称见 [direct_joint_v1.json](../configs/actions/direct_joint_v1.json)。URDF 原始 `effort=0`、`velocity=0` 是供应文件占位值，不是有效硬件限制。

### 2.3 当时有效手指 PD

当时合同显式选择 `drive_profile=shadow_v1`，40 个手指关节统一使用：

| 参数 | 数值 | 单位 | 来源 |
|---|---:|---|---|
| position actuator Kp | 1.0 | N·m/rad | `drive_settings.json`, `tsensei_hand.py` |
| joint damping | 0.05 | N·m·s/rad | 同上 |
| armature | 0.0002 | kg·m² | `urdf_model.py`, `tsensei_hand_constants.py` |
| frictionloss | 0.01 | N·m | `drive_settings.json`, `tsensei_hand.py` |
| actuator force limit | ±0.6 | N·m | `PROVISIONAL_FINGER_EFFORT` |

这些是仿真适配值；`TODO(V32-04)` 仍要求硬件电机、减速比和真实力矩标定。

### 2.4 手腕虚拟自由度参数

源码默认值位于 [tsensei_hand_constants.py](../tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py)。当时 `wrist_pitch_range_scale=0.8` 将 Pitch 的原始 ±0.2 rad 缩为 ±0.16 rad。

| 自由度 | 当时范围 | 当时 Kp | 当时阻尼 | armature | 力限 |
|---|---:|---:|---:|---:|---:|
| TX | 右 −0.7605～0.4605 m；左 −0.4605～0.7605 m | 300 N/m | 61.68983524 N·s/m | 3 kg | ±50 N |
| TY | −0.05～0.15 m | 300 N/m | 61.68983524 N·s/m | 3 kg | ±50 N |
| Pitch | −0.16～0.16 rad | 8 N·m/rad | 0.5 N·m·s/rad | 0 | ±5 N·m |
| TZ | −0.025～0 m | 1000 | 50 | 0 | ±50 N；当时结构移除 |
| Yaw | −0.25～0.25 rad | 300 | 10 | 0 | ±5 N·m；当时结构移除 |

TX/TY 阻尼由编译后的 `dof_M0=3.17136314 kg` 计算：

```text
damping = 2 × sqrt(Kp × dof_M0) = 61.68983524 N·s/m
```

旧 `legacy_v1` 保留 TX/TY 阻尼 30、Pitch Kp/阻尼 50/3、Yaw Kp/阻尼 300/10、TX/TY armature=0、手指 frictionloss=0；当时合同不使用这些旧值。

### 2.5 手部安装和碰撞几何

| 参数 | 当时值 |
|---|---|
| 右手根初始位置 | `[0.18, 0.15, 0.10] m` |
| 左手根初始位置 | `[0.18, −0.15, 0.10] m` |
| 手根安装四元数 | `(0, 0.70710678, 0, −0.70710678)` |
| 碰撞 profile | `shadow_primitive_v1` |
| base housing | 原 mesh |
| palm mount | box 代理 |
| 非末节 | capsule 代理 |
| 末节 | 全网格等效惯性、指尖锚定 capsule |
| 末节近似直径 | 约 11.72～12.82 mm |

逐 link 的 capsule `pos/quat/size/fromto` 见 [collision_proxies.json](../tsensei_v3_2_pianist/assets/collision_proxies.json)。这是工程碰撞近似，不是完整外壳包络或实测接触垫。

2026-10-12 起，活动左右手 URDF 的 `left_hand_mid_PIP_link` / `right_hand_mid_PIP_link` 的 visual 与 URDF collision mesh 均引用对应侧的 `*_index_PIP_link.STL`；`mid_PIP` link 名、惯性、`mid_PIP_joint` 的父子关系、原点、轴和限位均保持不变。当时 `shadow_primitive_v1` 使用的独立 capsule 碰撞代理未随此视觉 mesh 替换自动改变，仍以 [collision_proxies.json](../tsensei_v3_2_pianist/assets/collision_proxies.json) 为准。

## 3. 钢琴物理参数

来源：[piano_constants.py](../tsensei_v3_2_pianist/models/piano/piano_constants.py)、[piano_mjcf.py](../tsensei_v3_2_pianist/models/piano/piano_mjcf.py)。

### 3.1 几何、质量和被动关节

| 参数 | 白键 | 黑键 |
|---|---:|---:|
| 数量 | 52 | 36 |
| 宽度 | 22.5 mm | 10 mm |
| 长度 | 150 mm | 90 mm |
| 高度 | 22.5 mm | 18 mm |
| 质量 | 0.04 kg | 0.02 kg |
| 最大行程 | 10 mm | 8 mm |
| 最大关节角 | 0.0665682 rad | 0.0886559 rad |
| 弹簧刚度 | 2 N·m/rad | 2 N·m/rad |
| spring reference | −1° | −1° |
| 关节阻尼 | 0.05 N·m·s/rad | 0.05 N·m·s/rad |
| armature | 0.001 kg·m² | 0.001 kg·m² |

钢琴总键宽为 `1.221 m`。底座 full size 为 `0.1 × 1.221 × 0.04 m`，代码没有显式指定底座质量/惯量，实际由 MuJoCo geom 默认密度处理，不能称作已标定底座动力学。

训练中 `add_piano_actuators=False`：琴键没有额外位置电机，只有手指接触、弹簧、阻尼和限位。

### 3.2 按键状态

| 参数 | 数值 |
|---|---:|
| 按键激活阈值 | `0.00872665 rad`（0.5°） |
| sustain 激活阈值 | 0.5 |

来源：[piano.py](../tsensei_v3_2_pianist/models/piano/piano.py)。

## 4. 机械臂物理和刚性装配

来源：[M6 左 URDF](../preview/assets/tj_m6/left/arm.urdf)、[M6 右 URDF](../preview/assets/tj_m6/right/arm.urdf)、[m6_sdk_joint_limits.json](../configs/preview/m6_sdk_joint_limits.json)、[arm_options.py](../tsensei_v3_2_pianist/mjlab_env/arm_options.py)、[arm_scene.py](../tsensei_v3_2_pianist/mjlab_env/arm_scene.py)。

### 4.1 M6 名义质量和惯量

每侧 URDF 名义质量为 **12.4427 kg**：

```text
Base  = 1.19620 kg
Link1 = 1.97050 kg
Link2 = 2.24398 kg
Link3 = 1.92420 kg
Link4 = 1.60850 kg
Link5 = 2.08830 kg
Link6 = 0.72496 kg
Link7 = 0.68606 kg
```

每个连杆的 `com xyz` 和完整惯量张量均在 M6 URDF 中。

**重要：这些不是当时动力学中的有效臂质量。** 有臂场景把 M6 link 作为 `mocap=true` 视觉链，机械臂 geometry 设置 `density=0`、`contype=0`、`conaffinity=0`，没有机械臂 torque actuator、动力学关节或反作用力。当时机械臂是运动学执行器。

### 4.2 关节位置、速度、加速度和名义 effort

| 关节 | 位置范围 | 名义 effort | 速度硬限 | 加速度硬限 |
|---|---:|---:|---:|---:|
| J1 | −3.1067～3.1067 rad | 108 N·m | π rad/s | 7.85398 rad/s² |
| J2 | −2.0944～2.0944 rad | 108 N·m | π rad/s | 7.85398 rad/s² |
| J3 | −3.1067～3.1067 rad | 66 N·m | π rad/s | 15.70796 rad/s² |
| J4 | −2.5307～1.0472 rad | 66 N·m | π rad/s | 15.70796 rad/s² |
| J5 | −3.1067～3.1067 rad | 18 N·m | π rad/s | 15.70796 rad/s² |
| J6 | −1.0472～1.0472 rad | 18 N·m | π rad/s | 15.70796 rad/s² |
| J7 | −1.5708～1.5708 rad | 18 N·m | π rad/s | 15.70796 rad/s² |

SDK 原始单位是 degree/s、degree/s²；当时合同转换成 rad 制。名义 effort 只来自 URDF/SDK，当时 kinematic arm 不使用。

### 4.3 当时初始角度和装配

当时初始角度：

```text
right = [-84.27055, -68.29417,  61.39675, -89.05803,
          -64.834334, 27.92905, -7.30834] degree

left  = [ 84.27055, -68.29417, -61.39675, -89.05803,
           64.834334, 27.92905,  -7.30834] degree
```

刚性链：

```text
Link7 → 固定法兰 → 四孔连接件 → 手根
```

| 参数 | 右侧 | 左侧 |
|---|---|---|
| Link7→手 xyz | `[-0.0135,-0.121,0.004] m` | `[0.004,-0.121,-0.0135] m` |
| Link7→手 rpy | `[π/2,0,0]` | `[π/2,-π/2,0]` |
| Link7 法兰中心 | `[0,-0.095,0] m` | `[0,-0.095,0] m` |

四孔安装面为 `24.6 × 45.6 mm`；安装板孔径 4.3 mm，连接件孔径 3.7 mm。连接件质量、惯量和螺纹/法兰真实装配仍未标定，见 [hand_connector_20260930.json](../configs/preview/hand_connector_20260930.json) 和 [connector_mount_revision.json](../tsensei_v3_2_pianist/assets/connector_mount_revision.json)。

## 5. PD / IK / 目标限制控制器

### 5.1 手指 PD

手指使用 MuJoCo native `position` actuator，参数见第 2.3 节。目标是角度，输出受 ±0.6 N·m 力限约束；没有单独的硬速度限制，URDF velocity=0 不提供有效限速。

### 5.2 无臂腕部 PD

无臂模式下 TX/TY/Pitch 使用 MuJoCo position actuator；参数见第 2.4 节。目标变化率限制属于外部 `TargetMotionLimiter`，不是 PD 参数。

### 5.3 当时有臂控制链

有臂使用当时 `controller=wuji_cartesian_v1`，实现位于 [arm_follow.py](../tsensei_v3_2_pianist/mjlab_env/arm_follow.py)：

| 参数 | 当时值 |
|---|---:|
| 控制器 | `wuji_cartesian_v1` |
| IK | 阻尼最小二乘 |
| IK 迭代 | 12 |
| 姿态误差权重 | 0.15 m |
| DLS damping | 0.008 |
| 参考种子 | 上一时刻连续 reference |
| 位置/速度/加速度硬限 | 强制执行 |
| J6/J7 耦合 | 强制执行 |
| 软限比例 | 0.95 |
| 满罚超限比例 | 0.10 |
| 臂运动罚上限 | −5/控制步 |

IK 数值误差使用位置误差和旋转向量误差；当时 IK 不是时间上的 PD 控制器。目标制动根据剩余目标距离、关节速度和加速度硬限计算可停速度，然后再执行 J6/J7 耦合线搜索。

### 5.4 腕目标限制

来源：当时合同中的 `control_limits`。

| 策略通道 | 目标速度上限 | 目标加速度上限 |
|---|---:|---:|
| TX/TY | 0.99 m/s | 1.99 m/s² |
| Pitch | 3.1416 rad/s | 15.708 rad/s² |

这些是离散目标差分限制，不等同于真实硬件 TCP 或关节动力学保证。

## 6. 接触、限位和碰撞筛选

来源：[physics_settings.py](../tsensei_v3_2_pianist/physics_settings.py)、[contact_settings.json](../tsensei_v3_2_pianist/assets/contact_settings.json)、[self_collision.py](../tsensei_v3_2_pianist/mjlab_env/self_collision.py)、[arm_scene.py](../tsensei_v3_2_pianist/mjlab_env/arm_scene.py)。

### 6.1 当时接触参数

手：

```text
solref  = [0.005, 1.0]
solimp  = [0.5, 0.99, 0.0001, 0.5, 2.0]
friction = [1.0, 0.005, 0.0001]
priority = 0
```

钢琴键/琴箱：

```text
solref  = [0.010, 1.0]       # 20/200 Hz 实际导出值
solimp  = [0.9, 0.95, 0.001, 0.5, 2.0]
friction = [1.0, 0.005, 0.0001]
priority = 0
```

琴键限位：

```text
solreflimit = [0.02, 1.0]
solimplimit = [0.9, 0.95, 0.001, 0.5, 2.0]
```

由于手和钢琴 priority 相同，手—键最终接触参数按 MuJoCo 规则混合，约为：

```text
solref = [0.0075, 1.0]
solimp = [0.7, 0.97, 0.00055, 0.5, 2.0]
```

### 6.2 结构性碰撞规则

- 钢琴键之间通过 piano 默认 `contype=0` 禁止相互碰撞。
- 手—键接触保留。
- 手内部非相邻自碰撞保留。
- 当时刚性臂转换会自动恢复原 hand-only 模型中因 weld 变化丢失的父子过滤关系，当时合同记录为 `parent_collision_filter`。
- 有臂场景的双手根/掌部跨手接触通过显式 geom pair 进行检测，但不产生机械臂动力学反作用。
- 机械臂 mesh 与连接件 mesh 当时是视觉/运动学几何，不作为动力学碰撞体。

## 7. 未标定、未生效或不能过度解释的参数

### 7.1 手部

- URDF 原始 effort/velocity 全为 0，不能作为硬件额定值。
- Kp=1、阻尼=0.05、±0.6 N·m 和 armature=0.0002 是仿真适配值。
- 手指真实电机、减速比、转子惯量、摩擦和速度上限未确认。
- 右手是镜像推导，不是正式厂商右手。
- capsule 碰撞几何是工程近似，不是外壳保守包络。

### 7.2 钢琴

- 底座没有显式质量/惯量；使用 MuJoCo 默认 geom 质量规则。
- 琴键质量、弹簧、阻尼和 armature 是建模常量，不是具体实体钢琴的逐键测量。
- 黑键行程是工程近似。

### 7.3 机械臂和连接件

- M6 URDF 质量/惯量是名义资产参数，但当时 arm dynamics 不使用。
- 当时没有臂 torque dynamics、控制器闭环参数、负载折减或真实伺服响应。
- 法兰钟向、螺纹配合、连接件质量/惯量、实机零位和编码器映射未完成硬件复核。
- 当时臂可达性和目标制动验收不能解释成硬件安全验收。

### 7.4 复现要求

正式训练/回放必须同时匹配：

1. 保存的完整 `config.json`；
2. 对应 `scene/manifest.json`；
3. 固定源码归档及 SHA；
4. 手模型、碰撞 profile、驱动 profile；
5. 当时 `controller=wuji_cartesian_v1`、双侧七角、连接件合同；
6. physics/control timestep 和 contact 参数。

活动设计变化后直接重新训练；`deprecated/` 中的历史 checkpoint 不进入当时流程。

