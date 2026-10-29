# 项目外部坐标关系清单

本清单响应用户要求：不展开手URDF内部link/joint坐标，也不展开机械臂内部关节链，只列外部接口、场景、目标/记录和显示相关坐标。依据2026-10-25当前唯一入口和活动源码；默认无臂，初始世界位置右(.16,.15,.09)m、左(.16,-.15,.09)m，palm_base_yaw=15°、TZ关闭。位置追加参数已统一为world_x/right_world_y/left_world_y/world_z，冻结源码以唯一入口及[本轮记录](coordinate_names_20261025.md)为准。下文符号为说明用记号，不表示代码为每个符号创建了独立body。

约定：`T_A_B` 把B系的齐次点坐标变换到A系，即 `p_A=T_A_B p_B`。平移默认用米；矩阵使用列向量。四元数必须看字段合同：MuJoCo惯用wxyz，部分CSV使用xyzw，不能按字段长度猜顺序。

## 当前无臂任务

| 坐标/对象 | 相对谁 | 关系 | 状态及来源 |
|---|---|---|---|
| 仿真世界W / Stage | 总基准 | +X朝演奏者、+Y低音到高音、+Z向上 | 每个并行物理world使用同一套约定，不把不同环境的数组行当成坐标变换。`models/arenas/stage.py`、`suite/tasks/base.py` |
| 钢琴整体P（`piano/`） | W | 当前直接挂到arena，无额外平移/旋转，`T_W_P=I` | `suite/tasks/base.py`；查看器读取`piano/base`的父body作为钢琴原点，见`scripts/view_static_piano.py` |
| 钢琴后支撑块Q（`piano/base`） | P | 固定平移`(-.127,0,.020)`、单位旋转 | 它是后方支撑块，不是整体钢琴原点；琴键与它是P下的同级body。`models/piano/piano_constants.py`、`piano_mjcf.py` |
| 左右手安装基准M_R/M_L | W | 平移右`(.18,+.15,.09)`、左`(.18,-.15,.09)`；安装旋转A见下 | 是定义手安装的固定参考，不是新的可动实体；源码模板安装Z=.10被入口world_z=.09覆盖。`models/hands/tsensei_hand_constants.py` |
| 平移动作基底D_R/D_L | W | Tx正方向`(0,+1,0)`、Ty`(-1,0,0)`、Tz`(0,0,+1)` | 这是三个滑轨轴的方向基底，没有额外“forearm实体坐标系”。Tz当前已结构性移除；`palm_base_*`是关节名（此前名为forearm_*）、`Palm_Base_*`是对应位置执行器/控制通道名。`tsensei_hand_constants.py` |
| 左右掌根H_R/H_L（`base_link`） | M_R/M_L | 随滑轨实际qpos及Pitch变化，完整位姿见下 | 是手模型的外部接口原点，不是掌心、质心、指尖或法兰。`models/hands/tsensei_hand.py`、`suite/tasks/base.py` |
| 88个琴键K_i | P | 各有静止安装位置，按键时绕局部+Y铰链旋转 | 原点在原键体中心；铰点在原点局部负X半键长处。`models/piano/piano_mjcf.py` |
| 琴键参考site S_i | K_i | site局部pos=0、quat=单位，故`T_K_i_S_i=I` | 随琴键一起运动；记录原中心及参考半尺寸。黑键目标不能从mesh主轴中心反推。`piano_mjcf.py`、`fingering_targets.py` |
| 外观/碰撞geom G_j | 所属body | `T_W_G=T_W_body T_body_G` | 黑键mesh可能经编译做主轴/质心重定位，不能把geom系当原键体参考系。手URDF内部附属几何按用户要求省略。`fingering_targets.py` |
| 指尖任务site与支撑点 | 手末节（内部链省略） | 由所属末节的完整世界变换得到世界位置；任务点比支撑点沿末节朝掌根回退3mm | 任务点和耦合支撑点是两个局部标记，不能当两个全局任务坐标系。`models/hands/tsensei_hand.py` |
| 仿真渲染相机C_view | P或W | 固定相机挂钢琴整体下；自由相机由world下的lookat/视角生成 | `piano/closeup,left,right,back,egocentric,topdown`；与实机手眼相机无关联。`models/piano/piano.py`、`preview_camera.py` |
| 屏幕像素U | C_view | 世界点经视图及透视/正交投影到像素，不是刚体4×4变换 | 改相机只改画面，不移动物理物体。`scripts/view_static_piano.py` |
| 临时接触局部系C_contact | W及接触几何 | 原点为接触点，第一轴为接触法向，其余轴为切向；每个接触动态生成 | 力/力矩可能在接触系表达；不能把`mj_contactForce`结果直接当世界XYZ。见MuJoCo官方[contact类型](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact)、[contactForce](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce) |

钢琴整体原点与白键“键面中心”也不同：当前白键静止中心Z=.01125，键面Z=.0225，整体P的Z=0位于白键静止底面所在高度。整体Y=0位于键盘横向中心。支撑块Q虽然叫`base`，不能替代P。

### 安装与掌根的完整关系

安装旋转把模型本地X/Y/Z分别指向世界−Z/−Y/−X：

```text
A = [ 0  0 -1
      0 -1  0
     -1  0  0 ]

B = [ 0 -1  0       # 三列分别为Tx、Ty、Tz在世界中的方向
      1  0  0
      0  0  1 ]
```

因此，设`p_M`是各手安装点，`q=[Tx,Ty,Tz]`是实际关节位移：

```text
p_W_H = p_M + B q
世界X = .18 − Ty
世界Y_R = +.15 + Tx
世界Y_L = −.15 + Tx
世界Z = .09（当前Tz关闭）

R_W_H_R = A Rx(+Pitch_R) Ry(−15°)
R_W_H_L = A Rx(−Pitch_L) Ry(−15°)
```

等价的未烘焙变换顺序是：`T_W_M · Trans(-Tz,-Tx,Ty) · Rx(±Pitch) · Ry(-Yaw)`。固定Yaw烘焙时，代码对其余轴作逆旋转补偿，保持滑轨方向不变，见`mjlab_env/bundle.py`。

这里A是安装/本体朝向，B是平移关节的动作方向基底；不能拿当前掌根旋转矩阵直接旋转原始滑轨轴来代替B。左右掌根之间不存在独立的固定安装变换；一般是`T_HL_HR=inv(T_W_HL) T_W_HR`，随两手动作变化。

### 琴键、目标点、准备与参考姿态

设键静止中心为`c_i`、局部铰点`h_i=(-半键长,0,0)`，则：

```text
T_P_Ki(q_i) = Trans(c_i) Trans(h_i) Ry(q_i) Trans(-h_i)
T_W_Si = T_W_P T_P_Ki
```

指法目标不是额外完整坐标系，它是在世界系中计算的点：

```text
p_target_W = p_reference_site_W + delta_W
白键 delta_W = (.035, 0, .005625)
黑键 delta_W = (.025, 0, .0045)
```

偏移是固定世界轴偏移，不是`R_key*delta_local`；随按键运动的是参考site中心，偏移方向保持世界轴。指尖误差、首音悬停目标最终也在同一W中表达。耦合使用静止琴键平面的世界法向，这是一根参考向量，不是新增可动坐标系。依据`fingering_targets.py`、`mjlab_env/task.py`、`finger_coupling.py`。

`origin`、`reset`、首音准备末态、曲尾返回目标都是同一W下的参考状态，不是重设世界原点。当前first_note_v1明确不创建mocap支架；只有另外的fixed/policy准备模式才会有`preparation_{right,left}_mount`外层支架，见`bundle.py`与`first_note_preparation.py`。

## 仅显式启用机械臂时增加的外部链

当前默认无臂；`first_note_v1`不支持带臂，不能将下列链画成当前已运行模式。略去臂内部关节，仅保留两端接口：

```text
W → B_side（固定基座）
  → Link7（随臂运动；内部链省略）
  → F_side（固定项目法兰）
  → C_side（固定结构连接件） ≡ H_side（掌根/控制TCP）
```

`T_W_H=T_W_B T_B_7(q) T_7_F T_F_H`。初始手根位置来自世界XYZ参数（清零移除滑轨不会再丢弃该位置）；基座由该完整位姿、用户显式初始七角和安装变换一次反算：`T_W_B=T_W_H0 inv(T_B_7(q0) T_7_H)`。它不是可从当前无臂初始XYZ直接抄出的通用固定位置。见`mjlab_env/arm_follow.py`。

| 固定关系 | 右 | 左 |
|---|---|---|
| Link7→项目法兰平移，mm | `(0,-95,0)` | 相同 |
| Link7→手根平移，mm | `(-13.5,-121,-4)` | `(+13.5,-121,+4)` |
| 法兰→手根平移，mm | `(-13.5,-26,-4)` | `(+13.5,-26,+4)` |
| 手根x轴在法兰系 | +X | −X |
| 手根y轴在法兰系 | +Z | −Z |
| 手根z轴在法兰系 | −Y | −Y |
| 连接件显示系相对H | 沿H局部Y +8mm，无旋转 | 单位变换 |

结构连接件body与手根H同位姿；显示系用于CAD安装件外观。当前装配通过`nominal_link7_to_hand @ inv(offset_transform)`计算，右侧+8mm已经纳入有效手根安装矩阵，不能重复补偿。见`preview_connector.py`、`configs/preview/hand_connector_20260930.json`、`arm_scene.py`。法兰与手根原点距离约29.567719mm。

带臂模式`wuji_cartesian_v1`复用Palm_Base_Tx/Ty/Tz通道名，但它们表示世界XYZ位移：`p_request_W=p_rest_W+[TX,TY,TZ]`，朝向固定。不能沿用无臂的Tx→+Y、Ty→−X映射。见`arm_scene.py`、`arm_options.py`。

W与P当前重合，但带臂合同仍保存完整`piano_world_pose`，基座在钢琴系的表达是`T_P_B=inv(T_W_P) T_W_B`，见`arm_scene.py`。在线实际手根姿态与请求姿态可以不同；这是同一坐标系下的跟踪误差，不是新增坐标系。

## 实机标定、离线轨迹和其他展示入口

这些关系不被当前无臂训练读取，数值必须绑定具体采集/配置；不能把模拟中的W=P当作实机标定。

| 坐标/表示 | 相对关系 | 当前边界 |
|---|---|---|
| 实体相机C_real、视觉检测钢琴V、用户约定真实钢琴P_real、左右机器人基座B_real | `T_B_Preal=T_B_Creal T_Creal_V T_V_Preal` | 视觉到用户轴约定：用户X=−视觉Y、用户Y=视觉X、用户Z=视觉Z；仍需原点平移。相机链仍需验证；2026-10-28 已登记独立右臂 P_real↔B_real 探针结果，见 [探针标定说明](right_arm_probe_calibration_20261028.md)，默认训练不自动读取 |
| SDK末端/法兰F_SDK | 离线应将手根目标转换为SDK接受的末端：`T_P_F=T_P_H inv(T_7_H) T_7_F` | 项目法兰与SDK真实法兰完整定义是否一致需当前标定；所引旧单点结果当前本地缺失，不在本审计中重构/猜填矩阵 |
| 控制TCP | 仿真轨迹/运动指标指H，即base_link原点 | 旧实机单点任务曾假定TCP=SDK法兰；不是当前工具偏置标定。不能无条件写F_SDK≡H |
| 钢琴系手根轨迹`T_piano_hand` | `T_W_H=T_W_P T_P_H` | 是完整手根姿态，不是相对第0帧累计位移；新palm_base_world_poses路径明确存世界位姿，旧wrist_poses属于历史名称，要看文件合同 |
| 离线目标补偿 | `T'_P_H=D_P T_P_H`，特定脚本D_P为纯平移 | 修改目标，不是新增实际坐标系，也不是实测状态/新训练默认 |
| online_control展示场景的`frame="base"` | 公共展示系中钢琴在原点，各臂各有root_transform | 这个泛称不是任一实机机器人基座；不是当前默认训练或硬件标定来源 |

依据：`docs/rl_to_hardware_trajectory_workflow_20261013.md:100,130,156,183`、`preview/scripts/render_trajectory_bundle.py`、`preview/scripts/replay_arm_ik.py`、`preview/scripts/export_video_job.py`、`scripts/compensate_{right,left}_wrist_20261013.py`、`online_control/models.py`。后者仍含过时手模型提示，不作为当前A3.2硬件事实。

## 不应额外算作空间坐标系的名称

- `palm_base_tx/ty/tz`与`Palm_Base_Tx/Ty/Tz`是对应关节/执行器名称，不是两套独立坐标系；qpos与ctrl也不必相等。
- world_x/y/z、初始七角、seed、首音目标、原始姿态、reset和曲尾返回点都是参数、点或参考状态。
- 策略31维动作、归一化数值、MIDI音高/时间轴、帧堆叠不是三维空间坐标系；控制步/物理子步是时间采样，不是空间frame。
- 接触法向、静止键面法向等可构成方向基底，但没有因此添加可运动实体。
- 世界单位m与显示mm、四元数wxyz与xyzw是单位/序列约定，不是新坐标系。

本次仅源码、已保存记录和小矩阵审计，不加载MuJoCo模型、不运行物理或训练；没有修改任何坐标变换、模型或入口参数。
