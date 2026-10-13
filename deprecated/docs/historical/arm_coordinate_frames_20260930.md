# 当前机械臂、法兰与手部坐标系图

首帧全部关节角：[t=0左臂7关节与左手20关节CSV](../preview/outputs/left_initial_joints_20261006/left_arm_and_hand_joint_angles_t0.csv)、[可读文本](../preview/outputs/left_initial_joints_20261006/left_arm_and_hand_joint_angles_t0.txt)。包含各关节URDF名称、度/弧度；角度以关节零位为参考，与钢琴系位姿分开。该原轨迹首帧四指swing均为0，手指沿用原记录qpos。

首帧对应图：[t=0秒手H／钢琴P／机械臂基座B四视图](../preview/outputs/hand_piano_arm_views_t0_20261006/hand_piano_arm_base_four_views_t0.png)。与下方t=1图使用同一安装和四视角，H=[140,−150,125]mm，B=[661.740,−22.843,347.558]mm，P=[0,0,0]mm；首帧处于抬高25mm的准备起点。完整矩阵、来源和相机参数保存于同目录`metadata.json`。

2026-10-06补充：[t=1秒手H／钢琴P／机械臂基座B四视图](../preview/outputs/hand_piano_arm_views_20261006/hand_piano_arm_base_four_views_t1.png)。沿用当前冻结安装和完整左臂，从钢琴+Z、−X、+Y及斜视方向同时标出三个真实原点与各自轴向。钢琴系坐标（mm）：P=[0,0,0]，H=[170.263,−158.389,100.001]，B=[661.740,−22.843,347.558]；H来自当前导出臂角FK，B与首帧标定一致。青/金/紫圆心分别为H/P/B；标记透过实体显示，虚线为原点连线投影。单张视图、完整矩阵、输入SHA及相机说明见同目录`metadata.json`；未移动基座或修改历史轨迹。

2026-10-06补充：[左手URDF `base_link` 原点四视图](../preview/outputs/hand_base_views_20261006/left_hand_base_link_four_views.png)：俯视（钢琴+Z侧）、正视（−X侧）、侧视（+Y侧）和斜视均取 `left_base_from_start_20261005` 的同一t=0冻结姿态。青色圆心为同一个真实H原点，在钢琴模型坐标系中为[140,−150,125]mm；红/绿/蓝为X_H/Y_H/Z_H。标记透过外壳显示，右下角钢琴轴仅表示方向、并非钢琴原点位置。仅隐藏上游臂几何，保留Link7/连接件；未改变安装或关节。单视图及来源/相机验证位于同目录的 `metadata.json`。

2026-10-06补充：[当前视频t=0左手—钢琴坐标系与原点距离图](../preview/outputs/hand_piano_frames_20261006/hand_piano_frames_distance.png)。使用原视频带ID左手与冻结安装，未更换模型或移动坐标系。P=[0,0,0]mm，H=[140,−150,125]mm，空间直线距离240.260mm；未按下白键顶面Z_P=+22.5mm，P在其下方。图中红/绿/蓝为X/Y/Z，金色连线是P—H，屏幕投影长度不等于空间长度；[数值和来源校验](../preview/outputs/hand_piano_frames_20261006/figure_metadata.json)。

2026-10-06补充：[当前视频t=0手原点—法兰中心俯视关系图](../preview/outputs/hand_flange_top_20261006/hand_origin_to_flange_top.png)。采用下节首帧反解基座后的冻结场景；H=[140.000,−150.000,125.000]mm，F=[163.691,−154.000,107.766]mm。空间距离29.568mm，俯视XY投影24.026mm，法兰比手原点低17.234mm；图中的连线长度表示XY投影。 [精确坐标／来源](../preview/outputs/hand_flange_top_20261006/relation.json)。


## 最新：首帧反算基座一次，随后固定（2026-10-05）

用户确认只在初始化时由原t=0手部完整位姿和指定七角反算一次基座，之后基座固定跟踪wrist。公式为`T_P_B = T_P_H(0) · inverse(T_B_J7(q0) · T_J7_H)`。当前左基座在钢琴系的位置为 **[661.739502, −22.842523, 347.557576] mm**，URDF固定轴xyz欧拉角 **[−89.694095, −7.201126, 178.209422]°**；整段演奏保持此完整位姿。手／法兰局部安装、右侧及指定七角seed不变。首帧手根为原[140.000001,−150.000000,124.999971]mm和原朝向，同时七角精确匹配用户值。

`preview_mount.json`通过`base_xyz_m.left`覆盖左侧位置，`base_rpy_rad.left`保存朝向；未指定侧保留原中心／间距规则。详见[求解JSON](../preview/outputs/left_base_from_start_20261005/base_calibration.json)、[T_P_B CSV](../preview/outputs/left_base_from_start_20261005/T_piano_base.csv)、[T_B_P CSV](../preview/outputs/left_base_from_start_20261005/T_base_piano.csv)和[固定基座回放](../preview/outputs/left_base_from_start_20261005/left_arm_hand_front.mp4)。矩阵作用于列向量：`p_P=T_P_B p_B`；逆矩阵是钢琴在固定基座系中的位姿。此为模型约束反解，未声称硬件标定。

以下SDK法兰、J7与手的局部坐标关系仍有效；历史静态图中的钢琴系绝对位置对应当时旧基座，应以本节更新后的放置为准。

## SDK 原生法兰系与当前左手（2026-10-05，最新坐标定义）

已找到项目内厂商SDK的独立法兰定义：`TJ_FX_ROBOT_CONTRL_SDK/CommonConfig/ccs_m6_40.MvKDCfg`第9行是法兰MDH `[90°,0mm,95mm,90°]`，第1行机型1017；`kinematicsSDK/FxRobot.cpp`的CCS初始化及FK明确使用独立`m_Flange`，默认工具／用户坐标变换为单位阵。URDF只导出到Link7，不代表SDK没有法兰坐标系。此处 **F** 改指SDK原生法兰系；此前“沿用Link7轴向”的临时显示系在下文记为 **F_legacy**，两者原点相同但轴向不同。

映射到当前URDF Link7：`X_F=Z_J7, Y_F=−X_J7, Z_F=−Y_J7`，原点位于Link7局部`[0,−95,0]mm`；原生`+Z_F`沿法兰朝工具一侧。当前已修正的刚性手／连接件组件恰好满足 **X_H=X_F、Y_H=Y_F、Z_H=Z_F**，故手基座与法兰同向，但原点不重合：

```text
p_F = R_F_H p_H + t_F_H
R_F_H = I
t_F_H = [−13.5, −4.0, +26.0] mm
原点间距 = 29.567719 mm
```

在钢琴系中F原点`[60.245290,−191.553004,179.433725]mm`，H原点`[34.468683,−188.370606,193.564858]mm`。[原生法兰与手坐标系图](../preview/outputs/native_flange_hand_20261005/left_native_flange_and_hand_frames.png)中F虚线、H实线，原点和姿态均取当前模型；[T_F_H矩阵](../preview/outputs/native_flange_hand_20261005/T_native_flange_hand.csv)、[T_J7_F矩阵](../preview/outputs/native_flange_hand_20261005/T_J7_native_flange.csv)、[SDK与URDF审计](../preview/outputs/native_flange_hand_20261005/audit.json)保存完整数值和来源SHA。

用独立MDH连乘、SDK `libKine.so`纯运动学FK和URDF链，在零位、当前七角、逐轴正负扰动及100组随机角共116姿态交叉核对；诊断性将URDF的1.5708／3.1416视为精确π/2／π后，与SDK矩阵差最大4.19e−10。保留原URDF数值时，当前残差0.004386mm／0.000955°，已在审计中分别保存SDK直接结果及映射到当前模型的结果，未修改URDF以消除误差。本次图采用后者，保持机械臂、手、连接件的所有实际姿态不变。软件定义已核对；实际设备零位、安装分度及控制器自定义TCP仍需实物核对，未连接硬件。复现脚本见`logs/native_flange_hand_20261005/{audit,render}.py`。

## 当前有效安装：手与连接件构成完整刚性组件（2026-10-05）

用户明确：先将手和连接件装成一体，再将完整组件安装到机械臂。后续调整应保持CAD手座四孔及配合面关系，在组件—法兰接口调整；不能单转手而把连接件留在原地。此前“手自身Z转90°、连接件固定”的输出已查出错位，以下旧图仅保留为历史记录。

当前通过固定法兰中心、绕F_legacy的−Y（即SDK原生F的＋Z）整体转90°实现掌面朝下。机械臂基座、J1–J7及所有连杆、法兰均不动；左连接件相对手的offset归零，新nominal_link7_to_hand承载整体装配变换。轴关系仍为X_H=Z_J7、Y_H=−X_J7、Z_H=−Y_J7；J7→H位置现为[4,−121,−13.5]mm，F_legacy→H位置为[4,−26,−13.5]mm。手原点现为钢琴系[34.468683,−188.370606,193.564858]mm，相比错位姿态移动19.9123mm以恢复孔位配合，手掌朝向保持。

[正确装配近景](../preview/outputs/left_mount_alignment_fix_20261005/left_hand_orientation.png) · [全臂](../preview/outputs/left_mount_alignment_fix_20261005/left_arm_initial_pose.png) · [更新J7坐标关系](../preview/outputs/left_mount_alignment_fix_20261005/frames/left_j7_and_hand_frames.png) · [J7→H矩阵](../preview/outputs/left_mount_alignment_fix_20261005/T_J7_H.csv) · [F_legacy→H矩阵](../preview/outputs/left_mount_alignment_fix_20261005/T_F_H.csv) · [装配复核](../preview/outputs/left_mount_alignment_fix_20261005/alignment_validation.json)。四孔配对、CAD配合平面与法兰中心/面法向验证通过；实际紧固孔分度和硬件安装仍待核对。

## 历史图：修正装配前的姿态和临时法兰系

修正装配前左手与J7的[坐标关系图](../preview/outputs/left_j7_hand_frames_20261005/left_j7_and_hand_frames.png)、[完整变换JSON](../preview/outputs/left_j7_hand_frames_20261005/frames.json)、[T_J7_H矩阵CSV](../preview/outputs/left_j7_hand_frames_20261005/T_J7_H.csv)已导出。J7取Joint7_L转动后的Link7_L随动系，原点为真实关节枢轴、当前q7=−25.69049°。p_J7=R_J7_H p_H+t_J7_H，其中t=[−13.5,−121,−4]mm；矩阵CSV使用米。轴映射X_H=Z_J7、Y_H=−X_J7、Z_H=−Y_J7。保持此前机械臂与手的姿态。

修正装配前掌面朝下姿态的[临时法兰F_legacy与手基座H同图近景](../preview/outputs/left_flange_hand_frames_20261005/left_flange_and_hand_frames.png)及[精确变换](../preview/outputs/left_flange_hand_frames_20261005/frames.json)：F轴虚线、H轴实线，各原点不平移；X_H=Z_F，Y_H=−X_F，Z_H=−Y_F。F系中的F→H平移为[−13.5,−26,−4]mm。采用当前冻结姿态且机械臂不动，F仍沿用Link7轴向及局部Y=−95mm的项目定义。

## 最新手安装：2026-10-05固定机械臂，左手自身Z轴+90°

在下节基座调整及用户给定七关节角基础上，手安装右乘Rz(+90°)。机械臂全部关节／连杆、法兰、安装件与手base_link原点保持；手指角也保留。对应连接件配置offsets.left.rpy_deg=[0,0,−90]，其逆变换使手实际+90°；preview_mount的左tool同步等价结果，右侧不变。

当前H原点仍为钢琴系[34.780893,−205.855831,184.042807]mm；X_H≈[−.024241,.000668,−.999706]，Y_H≈[.031000,−.999518,−.001420]，Z_H≈[−.999225,−.031025,.024209]。掌面朝下但仍是高于琴面的指定静态姿态。

[当前手朝向](../preview/outputs/left_hand_local_z90_20261005/left_hand_orientation.png) · [全臂](../preview/outputs/left_hand_local_z90_20261005/left_arm_initial_pose.png) · [同视角前后对照](../preview/outputs/left_hand_local_z90_20261005/hand_before_after.png) · [详细变换](../preview/outputs/left_hand_local_z90_20261005/pose_report.json)。变更前配置、CPU绘图脚本及固定机械臂的逐连杆验收在logs/left_hand_local_z90_20261005；未连接硬件或求解IK。

## 最新安装：2026-10-05按用户给定关系修改左臂

左臂基座原点保持钢琴系[520,−46.5,480]mm，现采用＋X_B=−X_P、＋Y_B=−Z_P、＋Z_B=−Y_P，URDF固定轴RPY=[−90°,0°,180°]。右臂安装保持此前配置。本节之后的旧快照仍按各自冻结安装解释，不能当作此次修改后的当前姿态。

用户指定左J1–J7初始角[77.04558,−70.57884,−61.54541,−82.70541,157.40033,5.54958,−25.69049]°已保存为左seed。当前手基座在P系为[34.780893,−205.855831,184.042807]mm；手轴约为X_H=[−.03100,.99952,.00142]、Y_H=[−.02424,.00067,−.99971]、Z_H=[−.99923,−.03103,.02421]。保留原URDF关节符号、原手／法兰安装和指定轨迹首帧手型，手掌仍侧立。

查看[左臂总览](../preview/outputs/arm_base_remount_20261005/left_arm_initial_pose.png)、[手部朝向近景](../preview/outputs/arm_base_remount_20261005/left_hand_orientation.png)、[俯视](../preview/outputs/arm_base_remount_20261005/left_hand_top.png)、[完整报告](../preview/outputs/arm_base_remount_20261005/pose_report.json)。备份、CPU渲染脚本与验收在logs/arm_base_remount_20261005。用户仅授权左臂此次安装变更；非实物标定或全轨迹重新规划。

以下前三张图是2026-09-30的reset＋IK姿态快照；2026-10-05新导出的全零姿态和正转方向见文末，不能混用两种姿态的世界轴。

使用当前源码reset姿态、A3.2_tianji安装配置和双臂初始IK解绘制，未运行策略或推进动力学。
透明外形来自实际编译模型的可视网格；绘图时进行顶点聚类以降低面数，模型文件不变。

| 图 | 高清PNG | PDF |
|---|---|---|
| 左臂总览与J1–J7分别放大 | [图片](../preview/outputs/arm_frames_20260930/joints_left.png) | [PDF](../preview/outputs/arm_frames_20260930/joints_left.pdf) |
| 右臂总览与J1–J7分别放大 | [图片](../preview/outputs/arm_frames_20260930/joints_right.png) | [PDF](../preview/outputs/arm_frames_20260930/joints_right.pdf) |
| 左右法兰F与手根H近景 | [图片](../preview/outputs/arm_frames_20260930/flange_hand.png) | [PDF](../preview/outputs/arm_frames_20260930/flange_hand.pdf) |

红/绿/蓝分别是局部正X/Y/Z。坐标轴起点是真实模型原点，未为排版移动坐标系。
每个关节放大图独立选取视角，避免沿轴观察导致箭头缩成一点；不同子图不能直接比较屏幕方向。
J1/J2原点重合，J5/J6/J7原点重合，故分图表示。总览的J3/J4只作邻近关节组标记，
两者原点实际相距18 mm，并非重合。

## 坐标定义

- **Ji**：URDF中`Jointi_L/R`的随动子连杆`Linki_L/R`坐标系，已包含当前关节角旋转。
  不采用DH坐标，也不是去除当前q旋转的关节零位系。两臂七个关节的URDF轴均为`[0,0,1]`，
  所以蓝色+Z是关节正转轴；正角按右手定则。
- **F**：将当前Link7坐标系沿其局部Y平移−95 mm，轴向不变。
  这是项目临时法兰面中心的显示约定，URDF未单独提供厂商法兰/TCP坐标系。
  此约定下朝工具方向为−Y，不把+Z称为法兰法向。
- **H**：当前左右手的规范化`base_link`坐标系；不是CAD原始坐标，也不是惯性质心。

F到H的变换为（平移分量用F坐标表达）：

| 手侧 | 平移mm | 旋转 |
|---|---|---|
| 左 | [-13.5, -26, -4] | 绕X +90° |
| 右 | [-13.5, -26, +4] | 绕X +90° |

注意与Link7到H的`Y=-121 mm`区分：法兰F本身已经在Link7的`Y=-95 mm`处。
右手和连接件仍是镜像推导模型；法兰实机标定、肩座位置和工具方向继续保留
TODO(V32-MOUNT-01)、TODO(TJ-ARM-02/03)。

## 验证与复现

`preview/scripts/draw_arm_coordinate_frames.py`读取当前配置，重建实际预览并求初始IK。
另直接沿URDF链计算`parent × origin × rotation(axis,q)`，与MuJoCo全部14个子连杆变换比较；
矩阵元素最大误差左5.55e-16、右6.66e-16。逐关节+Z与MuJoCo实际旋转轴一致（容差1e-10），
J7乘安装变换与实际手根一致（容差2e-8）。每侧绘图实际包含8个臂体、22个手体和1个连接件体。
首次图片检查发现手体命名筛选不匹配，已修复为实际模型前缀，重新生成并检查全部三张图。

所有18个坐标系的世界变换、14个关节角、视角与输入SHA256保存于
[frames.json](../preview/outputs/arm_frames_20260930/frames.json)。PNG为3240×2160（关节）
和2880×1620（法兰），PDF保留文字/坐标箭头，外形三角面栅格化。

```bash
training/.venv/bin/python preview/scripts/draw_arm_coordinate_frames.py
```

这是CPU离线FK及Matplotlib Agg绘图（`MUJOCO_GL=disable`），不使用CUDA/OpenGL/GPU推理。
未修改机械臂、手、安装偏移、训练参数或已有检查点；未进行硬件控制。

## 2026-10-05：当前模型的零位和正方向导出

用户要求导出当前环境机械臂零位及正方向。读取当前项目左右M6 URDF、preview_mount与online_control.models场景安装合同，以全部J1–J7=0 rad定义模型零位。与上一节历史reset＋IK图不同，本次所有零位变换均显式覆盖为q=0。当前preview_pose的来源散列检查未通过，现有场景加载器按原规则使用seed=0；导出保留实际初始值与该提示，不修改初始姿态配置。

| 交付 | 路径 |
|---|---|
| 左臂零位及七关节正转图 | [PNG](../preview/outputs/arm_zero_axes_20261005/zero_axes_left.png)、[PDF](../preview/outputs/arm_zero_axes_20261005/zero_axes_left.pdf) |
| 右臂零位及七关节正转图 | [PNG](../preview/outputs/arm_zero_axes_20261005/zero_axes_right.png)、[PDF](../preview/outputs/arm_zero_axes_20261005/zero_axes_right.pdf) |
| 机器可读合同 | [JSON](../preview/outputs/arm_zero_axes_20261005/arm_zero_axes.json)、[14关节CSV](../preview/outputs/arm_zero_axes_20261005/joint_axes.csv) |
| 查看与打包 | [HTML](../preview/outputs/arm_zero_axes_20261005/index.html)、[ZIP](../preview/outputs/arm_zero_axes_20261005/arm_zero_axes_bundle.zip) |

十四个URDF关节局部轴均为[0,0,1]，正角按右手定则。从正轴端朝原点看为逆时针；图中橙色弧线沿局部＋X到＋Y表示＋q。红／绿／蓝为局部＋X／＋Y／＋Z；子图相机不同，不能拿屏幕左右当作关节正负号。

| 零姿态世界轴近似 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| 左臂 | −Y | −Z | −Y | −Z | −Y | −Z | ＋X |
| 右臂 | ＋Y | ＋Z | ＋Y | ＋Z | ＋Y | ＋Z | ＋X |

这里的世界系是scene manifest的base，钢琴在该系单位变换，不等于机械臂自身Base_L/R。两个基座原点分别为左[.52,−.0465,.48]／右[.52,.0465,.48] m，绕世界X为左＋90°／右−90°。表格仅作零姿态近似，JSON/CSV保留URDF原始RPY导致的微小轴分量；其他姿态须用父链FK求轴，不将这些世界方向固定用于全轨迹。

JSON包含每关节原始parent origin XYZ/RPY、关节局部轴、零位基座／世界轴、基座安装／各关节／Link7／手根4×4变换、当前初始值、角限、来源SHA；CSV适合直接核对同事左臂joint_0…6→J1…J7映射。sources保存URDF、配置及导出代码快照，不包含网格，交付不是独立仿真模型包。模型零位不代表实机编码器回零，肩座／法兰标定TODO仍保留。

```bash
training/.venv/bin/python preview/scripts/export_arm_zero_axes.py \
  --output /absolute/path/new_output
```

通过CPU Matplotlib Agg绘图，MuJoCo GL关闭；未调用旧脚本main/reset/IK。14个零位及单关节＋1e-5 rad姿态独立FK对MuJoCo最大矩阵误差8.881784197001252e-16，正轴误差6.661338147750939e-16，正扰动旋转向量与正轴最大误差3.3306690738754696e-16；两幅PNG/PDF分别2880×1920及18×12英寸，人工检查通过。机器可读表、来源散列、manifest和ZIP完整性验收记录在logs/arm_zero_axes_20261005/export_validation.json。没有动力学推进、GPU、训练、硬件动作或模型修改。

### 同日：实机SDK轨迹的安装对应关系诊断

用户随后指出手应平放在钢琴上，怀疑J1需转180°，并要求检查其余关节；新确认CSV来自实机SDK／编码器。该信息使此前“零位／方向一致”的直接映射成为待核实假设，旧视频保留为原映射对照。

对6201帧完整CSV，在当前URDF及安装下检查以下候选。水平度采用固定初始手型的五个既有distal-tip标记拟合平面；该平面不等于已标定掌面或接触指腹。倾角是该平面与水平面的锐角，中位数／最大值同时保存在报告中，另保留有向法线以识别翻面。

| 候选 | 指尖平面倾角中位数 | 指尖平均世界Z范围 |
|---|---:|---:|
| 原SDK直接映射 | 89.74° | .745–.780 m |
| J1减π，J2–J7不变 | 89.74° | .180–.215 m |
| 再将手／连接件整体绕当前法兰安装轴＋90° | 8.13° | .117–.152 m |

J1减π在此轨迹与加180°方向等价，且全部角度在原限位内。安装候选以Link7中的临时法兰中心[0,−.095,0] m为枢轴、[0,−1,0]为轴，`T_new=T_F R_axis(+π/2) inv(T_F) T_original`；实际旋转的是手及安装件，不是将J7加90°。反向法兰−90°的有向平面法线约171°，会翻面。J7单独＋90°仍接近竖直，并使111个源样本越限，不能替代安装修正。

对J2–J7分别做单关节正负翻转及±90°／180°候选检查，每次仅改变一关节，其余保留J1减π方案；逐候选倾角、位置范围及越限计数见报告。这不是全部组合搜索或实机标定。目前支持优先核对J1零位与法兰安装转角，尚无证据需改J2–J7默认映射。法兰＋90°候选的平均指尖仍高于白键面94–129 mm，且X方向仍需安装位置核对，不能称已放平贴键。仅凭关节序列和期望平放，不能唯一求出关节零位／正负号、肩座变换和工具变换；需SDK工具定义、实际安装基准／已知法兰姿态进一步确认。

交付[对照页面](../preview/outputs/arm_csv_alignment_20261005/index.html)与[详细报告](../preview/outputs/arm_csv_alignment_20261005/report.json)，含0／15.5／31秒原映射、J1偏置、J1＋法兰候选三列同世界相机图及分别居中的左手近景、ZIP和来源场景。CPU llvmpipe渲染、无动力学推进，独立FK对实际绘图矩阵误差1.7763568394002505e-15；SHA／ZIP／PNG验收见logs/arm_csv_alignment_20261005/export_validation.json。候选仅存在本次内存场景与报告，未写回URDF、安装默认或训练配置，未生成实机控制指令。

## 2026-10-06：按新首帧关节角反解固定基座

上述旧图、零位表及安装候选保留其原冻结配置。本次[新轨迹标定文件](../preview/outputs/left_replan_smooth_20261006/base_calibration.json)按用户给定的77.48315／−70.55979／−61.07445／−81.20139／160.95941／14.69720／−25.86782°反解左臂基座。公式 `T_P_B = T_P_H(0) inv(T_B_H(q0))`，保留原t=0手URDF base_link在钢琴系 `[140.000001,−150.000000,124.999971]mm` 的完整朝向与刚性工具安装。

新基座钢琴系位置 `[618.618028,−19.096110,412.253914]mm`，固定轴xyz RPY为 `[−89.939536,0.737832,178.585872]°`；完整矩阵与逆矩阵已保存。只有这次首帧反解调整基座，轨迹开始后基座不动；右臂安装、手与连接件／法兰关系不变。当前preview_mount仅写入新左臂基座和seed，冻结导出仍独立保存全部配置。本次为模型反解，尚未实物测量，不改变SDK零位／符号或URDF，不把旧零姿态钢琴方向表作为新基座的方向表。

同日用户要求导出t=0除手关节外的环境信息：[完整环境快照说明](../preview/outputs/environment_t0_20261006/README.md)、[总JSON](../preview/outputs/environment_t0_20261006/environment_t0.json)、[各坐标系位置／角度CSV](../preview/outputs/environment_t0_20261006/frames_t0.csv)。来源为最新附件视频`left_csv_video_20261006`冻结安装；B／F／H原点分别为[618.618028,−19.096110,412.253914]／[163.690649,−154,107.766190]／[140.000001,−150,124.999971]mm。F与H方向同为固定轴XYZ RPY[0,81.405683,180]°，法兰在手系[+13.5,+4,−26]mm。包内保存七臂关节及模型、88键状态／参数、相机／地面／光照／物理环境与安装矩阵，排除全部手指／虚拟腕内部关节状态；物理琴键积分环境与视频显示代理分开标记，不改变任何场景或轨迹。
