# 当前双臂安装图与关节角度余量（2026-10-11）

本次导出直接读取 `configs/preview/current_arm_training_20261011.json`，通过 `build_bundle_from_config` 构建与 MJLab 预览／训练相同的刚性装配，然后在保存的 `arm_follow.initial_q_rad` reset 姿态做 CPU FK 和几何绘图。没有推进物理、运行策略或连接硬件。

## 交付

- [左臂、连接件和左手 PNG](../preview/outputs/arm_installation_limits_20261011/installation_left.png) · [PDF](../preview/outputs/arm_installation_limits_20261011/installation_left.pdf)
- [右臂、连接件和右手 PNG](../preview/outputs/arm_installation_limits_20261011/installation_right.png) · [PDF](../preview/outputs/arm_installation_limits_20261011/installation_right.pdf)
- [左右总览 PNG](../preview/outputs/arm_installation_limits_20261011/installation_overview.png) · [PDF](../preview/outputs/arm_installation_limits_20261011/installation_overview.pdf)
- [精确关节余量 CSV](../preview/outputs/arm_installation_limits_20261011/joint_margins.csv)
- [安装变换、基座和来源散列 JSON](../preview/outputs/arm_installation_limits_20261011/installation_report.json)
- [完整 ZIP](../preview/outputs/arm_installation_limits_20261011/arm_installation_limits_bundle.zip)

## 读图约定

图中的安装链是 `Base → J1…J7 → fixed flange → fixed four-hole connector → fixed hand root`。法兰在 Link7 局部 `[0, −95, 0] mm`；机械臂 link 是当前模型中的 mocap 可视链，手指和琴键仍是动态物理体。

每个关节的当前角度是训练合同保存的 `q₀`。所有 M6 URDF 关节轴为局部 `+Z`，按右手定则定义；从 `+Z` 轴端朝关节原点看，增加 q 为逆时针。因此：

```text
顺时针余量     = q₀ − lower
逆时针余量     = upper − q₀
```

图中橙色段为顺时针余量，绿色段为逆时针余量。CSV/JSON 同时保存弧度和角度。J6/J7 的 CCS-696 耦合约束、目标制动、速度和加速度硬限还会进一步缩小可执行区域，图中的独立角度余量不能单独作为全局可达性证明。

复现：

```bash
MUJOCO_GL=disable training_mjlab/.venv/bin/python \
  preview/scripts/export_arm_installation_limits.py
```

输出脚本为 `preview/scripts/export_arm_installation_limits.py`。导出过程只做模型构建、MuJoCo `mj_forward`/几何读取和 Matplotlib Agg 绘图，未改变训练入口、模型、奖励或历史归档。
