# M6 双臂＋V3.2 双手预览与差分 IK

**2026-09-22 更新：**此前 18 cm／单向跟随方案已更新为 93 mm／可达腕部约束，并新增 GPU 运动学机械臂场景和检测奖励。P 默认保存实际姿态。[当前方案](m6_forearm_93mm_20260922.md)。下方为 9月21日历史方案及验收，未接 RL 的描述已被本次用户决定覆盖。

更新：2026-09-21。用户已确认 Marvin M6-CCS-696、仅 V3.2、左右安装中心距 18 cm，并授权先采用可调临时安装，之后标定。本轮只接入姿态预览，不修改训练策略。

## 已完成与范围

- 原生 `preview/scripts/preview_pose.py` 默认显示白色双臂、双手及完整钢琴；`--no-arms` 可关闭。
- 双臂各 7 关节，使用包内厂商左右 M6 URDF 的关节轴、位置限位和速度限位。手仍采用项目现有 V3.2 模型；右手仍为镜像推导，不能称为厂商正式右手。
- 手腕是目标：手腕平移／Pitch／Yaw → 差分 IK → 双臂 Link7 及转接点跟随。目标是手腕当前实际世界姿态，而非未到达的控制目标。
- 本轮为**运动学装配预览**：独立 FK 模型解算关节，mocap 网格同步显示。没有实体动力学连接、机械臂碰撞、避障或硬件控制。不会把机械臂质量或接触传入钢琴仿真；训练仍为 51 动作，原模型 nq/nv/nu=138/138/50。
- GPU 冻结任务预览 `preview_mjlab.py`、历史训练／视频仍保持原模型；本功能位于源姿态预览。

## 启动与操作

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
.venv/bin/python preview/scripts/preview_pose.py --paused --no-contact-controls
```

1. 在右侧 Control 面板拖动 `R_Wrist_*`／`L_Wrist_*`，双手与机械臂随动。暂停时可直接摆放手腕，不推进钢琴物理；恢复运行后按原驱动器和物理运动。手指滑块仍在运行物理时驱动手指。
2. 也可双击腕部的**绿色球**选中，再按 `Ctrl＋鼠标右键拖动` 平移，`Ctrl＋鼠标左键拖动` 旋转；`Ctrl＋Shift＋右键` 改为水平面平移。绿色球可能被手腕遮挡，可转动视角后选中。鼠标操作依据 [MuJoCo 官方说明](https://mujoco.readthedocs.io/en/stable/programming/samples.html#mouse)。
3. 绿色球通过局部 IK 转换成已有的五个手腕控制量。超出范围或不支持的 Roll 旋转会投影到当前允许范围，不新增手腕自由度。
4. `Space` 暂停／继续，`R` 恢复打开时状态，`F` 恢复全琴＋双臂视角（MuJoCo 原生 F 也可能切换接触力显示）。
5. `P` 保存双手初始姿态及备份；**不保存**安装 JSON 或机械臂关节角。重新打开后重新求解双臂初姿。

恢复仅手预览：

```bash
.venv/bin/python preview/scripts/preview_pose.py --no-arms --paused
```

## 临时安装配置／待标定

配置：[preview_mount.json](../tsensei_v3_2_pianist/assets/tj_m6/preview_mount.json)。修改后重启预览，也可通过 `--arm-config /绝对路径/mount.json` 加载其他配置。

| 字段 | 当前值 | 含义 |
|---|---|---|
| shoulder_spacing_m | 0.18 m | 左右基座安装原点距离，沿世界 Y 对称；不等于外壳边缘或第一关节中心距离 |
| shoulder_center_xyz_m | [0.52, 0, 0.48] m | 两安装原点的世界坐标中点 |
| base_rpy_rad | 右 [−π/2,0,0]；左 [+π/2,0,0] | 基座相对世界的姿态 |
| link7_to_hand.xyz_m | 两侧 [0,−0.11,0] m | 手腕原点在 Link7 局部坐标系的位置 |
| link7_to_hand.rpy_rad | 两侧 [π/2,0,0] | 手腕坐标系相对 Link7 的姿态 |
| seed_q_rad | 左右各 7 值 | 初始化 IK 的种子，并非最终关节角 |
| damping / orientation_weight_m | 0.008 / 0.15 | 阻尼／旋转误差权重 |
| iterations_per_frame | 8 | 每帧差分 IK 最大迭代次数 |

保留当前双手初始位置，选取可达的临时肩座／手臂姿态。Link7 局部 −Y 约 95 mm 的网格端面仅作为临时法兰参考；当前手腕放在 −110 mm，简单圆柱显示约 15 mm 转接段。**这些不是实测法兰、安装方向或加工尺寸。** 灰色肩座同样只是示意。

- **TJ-ARM-01 已确认**：只做预览随动，机械臂暂不进入强化学习。
- **TJ-ARM-02 待标定**：肩座世界高度、前后位置、基座朝向；18 cm 安装间距已确认。
- **TJ-ARM-03 待标定**：Link7 到手腕的真实刚性变换及转接件尺寸。
- **TJ-ARM-04 未实施**：实体链动力学、碰撞与避障 IK。本轮 IK 不保证无自碰撞或避开钢琴；这些需在后续实体仿真接入时单独完成。

## 求解与可达性

初始化采用有界非线性最小二乘，必要时使用固定随机种子的多个初值；持续随动使用 6D 位姿误差与 MuJoCo 末端 Jacobian：

`Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e`

其中角误差为旋转向量并乘 0.15；每帧总关节变化受 URDF 速度限制，关节位置受 URDF 范围限制。移动很快会有跟随延迟；目标不可达时保留真实解算残差，手与法兰可能分离，不拉长机械臂伪装跟随。虚拟手腕原有全琴 TX 范围不保证全部可达。终端约每 120 帧检查一次，误差超过 5 mm 或 3° 时提示；退出后姿态 JSON 的 `arms` 字段保存配置、关节角与残差。

## 验收和产物

- [当前全景](../preview/outputs/tj_m6_preview/pose.png)／[配置与初姿误差](../preview/outputs/tj_m6_preview/pose.json)。
- [5 秒随动演示](../preview/outputs/tj_m6_follow_demo_final/arm_follow.mp4)／[演示截图](../preview/outputs/tj_m6_follow_demo_final/motion.png)。这是脚本生成的腕部运动，不是训练策略。
- 150 帧、30 fps、1280×720 H.264；ffprobe 和全片解码通过。此段轨迹最大位置误差右 0.00793 mm、左 0.00737 mm，最大角误差 <0.0017°，不能外推为全工作空间保证。
- `MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_preview_arm_ik.py tests/test_preview_controls.py tests/test_preview_backend.py tests/test_preview_pose_config.py -q`：**25 项通过**。包含 Jacobian 数值对照、速度／位置限位、不可达目标、鼠标目标转换、暂停运行循环中滑块／目标球跟随、原预览回归。[日志](../preview/outputs/tj_m6_preview/preview_regression.log)。
- wheel 包资源检查通过：16 STL、2 URDF、2 JSON、1 LICENSE；不依赖相邻 TJ_forearm。测试中仍有既有 audioop／NumPy 弃用警告。
- 当前工具会话无有效桌面 X11 授权，窗口尝试失败；**尚未现场验证鼠标拖动**。离屏图像已查看，控制转换和预览循环通过程序测试。请在桌面终端执行上述命令，用滑块和绿色球分别验收。

重生成全景与演示（演示必须使用新目录）：

```bash
.venv/bin/python preview/scripts/preview_pose.py --headless --output preview/outputs/m6_check.png
.venv/bin/python preview/scripts/demo_arm_ik.py --output preview/outputs/m6_demo_new
```

## 文件和来源

- `tsensei_v3_2_pianist/preview_arm_ik.py`：URDF 运动学构建、差分 IK、拖动目标转换。
- `tsensei_v3_2_pianist/preview_runtime.py`、`preview/scripts/preview_pose.py`：现有界面接入、相机、暂停腕部控制。
- `preview/scripts/demo_arm_ik.py`、`tests/test_preview_arm_ik.py`、`tests/test_preview_controls.py`：演示及验收。
- `tsensei_v3_2_pianist/assets/tj_m6/`：独立资源、配置、LICENSE；`source.json` 记录原始路径及 SHA256。
- [厂商 SDK 仓库](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK)，模型来自历史提交 `1a9cf30770ef6654cc1edfc048e1d0861be27eea`，左右原始 URDF／STL 保留字节内容。SDK 本体未执行，无硬件连接。

论文训练配置、奖励和动作合同未随本功能改变。修改问题及验证详见[变更日志](change_log.md)。
