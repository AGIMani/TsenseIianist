# 右臂—钢琴坐标变换：结果与使用说明

**本次使用 p2～p10 九点解，探针偏移 `[0,0,156] mm`。** 版本号为 `right_arm_piano_rigid_probe_20261026_selected_p2_p10`，程序读取同目录 [transform.json](transform.json)。此说明覆盖坐标转换、工具补偿和当前项目接口；本次交付尚未把矩阵加载到控制程序。

## 1. 坐标系和输入约定

| 符号 | 含义 |
|---|---|
| P | 标记琴键真值时使用的用户钢琴坐标系 |
| B | 右臂 MDH 基座坐标系，即项目 `FlangeKinematics("right")` 的基座；硬件通道 B |
| F | 右臂法兰坐标系，当前项目设置零工具偏移，IK 的 TCP 是法兰原点 |
| H | 装回灵巧手时的手模型根坐标系 `base_link` |

`T_A_from_B` 表示把 B 系中的坐标转换到 A 系，使用列向量左乘。JSON 的二维数组按行存储；不是需要转置后才能使用的列存储格式。

```text
[p_B; 1] = T_B_from_P @ [p_P; 1]
p_B = R_B_from_P @ p_P + t_B_from_P
```

此处两边描述的是**同一个物理点**，只改变表达坐标系。尖端位置转换为 B 系后，仍然是尖端位置，尚不是法兰原点位置。

钢琴坐标定义必须沿用采点时的原点和轴向：白键面 `Z=22.5 mm`，不是 Z=0；p2→p5→p8 是 +Y 方向，p4→p3→p2 是 +X 方向。p1 真值为 `(55,0,22.5) mm`，p1 不是钢琴原点。三维模型、轨迹及目标点若使用其他钢琴原点或轴向，应先转换到本次 P 系。

## 2. 选定的正逆变换

### 钢琴 P → 右臂基座 B

下列矩阵的平移单位为 mm；旋转部分无量纲。说明文档保留九位小数，程序使用 JSON 的完整精度。

```text
T_B_from_P_mm =
[[-0.998083274,  0.011006723,  0.060898530,  616.322841293],
 [ 0.061047222,  0.013719564,  0.998040585, -358.127560354],
 [ 0.010149655,  0.999845301, -0.014365198,  -24.655882813],
 [ 0,            0,            0,             1          ]]
```

平移列 `[616.322841293,−358.127560354,−24.655882813] mm` 表示**钢琴原点在右臂基座系的位置**。

### 右臂基座 B → 钢琴 P

```text
T_P_from_B_mm =
[[-0.998083274,  0.061047222,  0.010149655,  637.254460604],
 [ 0.011006723,  0.013719564,  0.999845301,   22.781728079],
 [ 0.060898530,  0.998040585, -0.014365198,  319.538498536],
 [ 0,            0,            0,             1          ]]
```

平移列 `[637.254460604,22.781728079,319.538498536] mm` 表示**右臂基座原点在钢琴系的位置**。两矩阵互为逆矩阵；不能只对平移取负而保持旋转不变，逆变换为 `R_inv=R.T`、`t_inv=−R.T @ t`。

| JSON 字段 | 输入坐标 | 输出坐标 | 长度单位 |
|---|---|---|---|
| `T_B_from_P_mm` | 钢琴 P | 右臂基座 B | mm |
| `T_P_from_B_mm` | 右臂基座 B | 钢琴 P | mm |
| `T_B_from_P_m` | 钢琴 P | 右臂基座 B | m |
| `T_P_from_B_m` | 右臂基座 B | 钢琴 P | m |

从 mm 切换到 m 时，矩阵只将前三行的平移列除以 1000，旋转和最后一行不变；目标点与工具偏移也应使用 m。不要将整个 4×4 矩阵除以 1000。

## 3. 可运行的点转换算例

把整个文件夹复制到有 NumPy 的 Python 3 环境。进入该目录执行 `python example.py`，或者在同目录脚本中调用：

```python
import numpy as np
from example import load_calibration, transform_points, transform_vectors

cal = load_calibration()
T_BP = np.asarray(cal["T_B_from_P_mm"])
T_PB = np.asarray(cal["T_P_from_B_mm"])
p_P = np.array([55.0, 165.2, 22.5])
p_B = transform_points(T_BP, p_P)
print(p_B)  # [564.616788747, -330.047577915, 140.753574947] mm
np.testing.assert_allclose(transform_points(T_PB, p_B), p_P, atol=1e-9)

# N×3 数组，每行一个点：
points_P = np.array([[55., 165.2, 22.5], [0., 330.2, 22.5]])
points_B = points_P @ T_BP[:3, :3].T + T_BP[:3, 3]

# 位移/同一物理点的线速度仅旋转，不加原点平移：
delta_B = transform_vectors(T_BP, [0., 10., 0.])
# [0.110067227, 0.137195644, 9.998453008] mm
```

上述基座目标是由钢琴真值换算而来，与采点时实际 FK 重建尖端存在拟合残差；不要求二者逐位相等。线速度公式适用于静止 P/B 坐标系中同一物理点的速度表达；尖端与法兰是不同点，旋转运动还会产生工具杆臂速度项。

## 4. 从探针接触目标得到法兰目标

当前项目 IK 使用法兰原点。因此给出钢琴接触点 `contact_P` 后，还需要选择法兰目标朝向 `R_B_from_F_des`：

```text
contact_B = R_B_from_P @ contact_P + t_B_from_P
probe_tip_F = [0,0,156] mm
flange_origin_B = contact_B − R_B_from_F_des @ probe_tip_F
T_B_from_F_des = [R_B_from_F_des, flange_origin_B; 0,0,0,1]
```

156 mm 沿**法兰 +Z**；补偿向量会随法兰朝向变化，不能固定从基座 Z 或钢琴 Z 减去 156 mm。外参决定两个坐标系的关系，不决定每次接触的法兰朝向。

使用包内 p2 的记录朝向，可以复现以下离线算例：

```python
from example import flange_pose_from_contact

T_BF = flange_pose_from_contact(
    T_BP,
    contact_P=[55., 165.2, 22.5],
    R_B_from_F_des=cal["sample"]["R_B_from_F"],
    tip_in_F=cal["probe_tip_in_F_mm"],
)
print(T_BF[:3, 3])
# [506.289267541, -202.936920016, 209.867834824] mm
```

此朝向用于数值核对，不是所有目标点的通用姿态。若规划提供的是钢琴系下的法兰朝向，则先计算 `R_B_from_F_des = R_B_from_P @ R_P_from_F_des`。目标朝向、连续逆解和关节约束仍由轨迹规划处理。

如果另一个控制器已经把工具 TCP 设置为该探针尖端，应按该控制器的工具约定输入目标，不要再次减去 156 mm。当前仓库的 `FlangeKinematics.native()` 明确设置单位工具矩阵，采用本节法兰原点补偿方式。

## 5. 装回手之后如何转换模型轨迹

钢琴与基座相对摆放、P 系定义不变时，`T_B_from_P` 仍表示同一物理关系。手的工具链需要另外处理；156 mm 属于本次探针，不是手指长度。

| 模型输出的目标对象 | 转换关系 |
|---|---|
| 钢琴系下的法兰完整位姿 `T_P_from_F` | `T_B_from_F = T_B_from_P @ T_P_from_F` |
| 钢琴系下的手根完整位姿 `T_P_from_H` | `T_B_from_F = T_B_from_P @ T_P_from_H @ inv(T_F_from_H)` |
| 钢琴系下的指尖接触点 `contact_P` | 结合目标手关节角求 `tip_F`，再使用第 4 节公式求法兰位置 |

指尖目标的工具链可写成：

```text
[tip_F;1] = T_F_from_H @ T_H_from_D(q_hand_des) @ [contact_D;1]
flange_origin_B = (R_B_from_P @ contact_P + t_B_from_P)
                  − R_B_from_F_des @ tip_F
```

`D` 是手模型中的指端坐标系，`contact_D` 是明确的接触点，`q_hand_des` 为该轨迹时刻的目标手角；不同手姿态对应不同 `tip_F`。反馈重建时应使用同步反馈角。多个手指同时接触时，应由臂手联合规划满足这些约束。

当前仓库 `a3-cpp-sdk/a3_linux/hand_model/installation.json` 的 `sides.right.flange_from_base_link` 对应 `T_F_from_H`，平移单位为 **m**。与本说明 mm 算例连乘前，需将其平移转为 mm，并确认模型根坐标和实际安装一致。换手或安装变化后重新核对工具链；本次探针标定未验证手指模型和接触精度。

## 6. 当前项目的接入位置与字段方向

以下为本次交接时仓库代码的实际语义，路径均从仓库根目录起算。离线示例可直接运行；现有页面还没有完整导入本 JSON 的入口。

| 位置 | 当前含义 | 接入本结果时需要处理 |
|---|---|---|
| `a3-cpp-sdk/a3_linux/arm_cartesian.py` → `PianoFrames.transform("right")` | 返回基座→钢琴 `T_P_from_B` | 使用本包逆向矩阵的完整 R 和 t |
| 同文件 → `PianoFrames.save()` / `arm_cartesian_frames.json` | 只保存 `position_mm` 和 `calibrated`，旋转仍为固定 `BASE_ROTATIONS` | 要应用完整外参，需增加旋转的加载、保存和显示；只填 XYZ 不足以启用此次标定 |
| `a3-cpp-sdk/a3_linux/piano_calibration.py` → `_target_pose()` | `rotation=R_P_from_B`，`base_position_mm=t_P_from_B`；`target_mm+offset_mm` 为钢琴系法兰目标 | 同时使用本包逆向 R/t，输入经工具补偿后的法兰位置 |
| 同文件 → `update()` | 手工更新位置不会导入任意旋转 | 不能假设请求体添加 `rotation` 就会生效，需要实际增加导入处理 |
| 同文件 → `capture()` / `camera_frame()` | 相机读取会重新设置外参，右侧还应用 locator→用户钢琴系转换 | 本包已在用户 P 系，不能再次乘 `USER_FROM_LOCATOR`；切换来源时应更新版本并使旧预览失效 |

若某字段要求“基座在钢琴系的位置”，应使用：

```text
position_mm / base_position_mm = [637.254460604,22.781728079,319.538498536]
rotation = T_P_from_B_mm[:3,:3]
```

这组字段和值是接入语义说明，不是当前页面已支持的 JSON 导入格式。`PianoFrames` 仍使用固定旋转时，单独填上述平移不能复现完整变换。若采用直接生成基座法兰目标的路径，应把 `T_B_from_F_des` 交给接收 B 系位姿的 IK 接口；不要再把它当作钢琴位姿传入绝对钢琴目标入口，否则会发生重复坐标转换。

项目 `pose_matrix()` / `pose_values()` 的六维顺序为 `[x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg]`，固定轴 XYZ 欧拉角满足 `R=Rz(yaw) Ry(pitch) Rx(roll)`。传入某个坐标系的位姿时，位置和朝向必须都在该系表达。`_target_pose(..., target_rpy_deg=None)` 保持种子法兰朝向；填写角度时，角度表示钢琴系下的法兰朝向。

旧 `offset_mm` 是旧外参/目标下的经验补偿，不能直接叠加到新标定上作为固有误差。接入时先明确补偿是否仍适用，再通过新外参下的测量重新确定；文档和离线脚本不会修改现有运行配置。

## 7. 同事接手后的数值核对与实测

1. 在包目录运行 `python example.py` 和 `python verify.py`。依赖只有 NumPy，脚本不连接机械臂；后者从关节角和包内 MDH 快照复算，不依赖原仓库目录。
2. 接入后记录实际加载的版本 ID、R/t 方向、长度单位与工具 TCP。用第 3、4 节输入核对软件目标，确认没有重复转换或重复工具补偿。
3. 用不同于拟合点的新接触目标核对实际位置，并保留同步关节反馈、目标和误差。先验证原采点区域，再评估 p1 所在区域；换回手后另外验证指尖工具链。

九点拟合 RMS 为 **0.569797 mm**，最大残差 **1.006251 mm（p8）**，内部留一 RMS **0.713413 mm**。正确 p1 没有参与拟合，在 `Y=0` 的预测误差为 **2.094625 mm**。拟合覆盖 `X=−45～55 mm`、`Y=165.2～494.6 mm`、`Z=22.5 mm`；这些是模型与本批接触记录的残差，不能当成整个工作空间的独立定位精度。

钢琴/基座移动、坐标原点或轴向改变、机械臂运动学模型更新后，应重新核对外参。输入与模型来源哈希在 JSON 中；完整推导、误差定义、九点/十点对照及 156/160 mm 差异见 [ANALYSIS.md](ANALYSIS.md)。
