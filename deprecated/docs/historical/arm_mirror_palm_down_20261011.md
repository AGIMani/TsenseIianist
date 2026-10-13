# 左臂镜像、法兰刚性跟随与掌面朝下场景（2026-10-11）

此前的 `arm_mirror_palm_down_scene_20261011` 是旧左侧后三轴姿态的基线快照。根据用户最新要求，新的候选改为复制右臂 J1–J7 的角度绝对值，尤其 J5/J6/J7 的绝对值严格相同；候选输出位于 [`arm_left_mirror_mount_candidate_20261011`](../preview/outputs/arm_left_mirror_mount_candidate_20261011/)。候选在固定当前左基座的前提下重算 Link7→手安装变换，主导工具局部 Z 时钟约 `−90°`，并保留四孔连接件的刚性链。候选不会覆盖默认训练配置、checkpoint 或硬件状态。

## 结果

- 左侧七轴角度取右侧绝对值并使用符号 `[+,-,-,-,+,+,-]`：左侧 J5/J6/J7 为 `[+64.83433,+27.92905,-7.30834]°`，右侧对应绝对值完全相同。世界镜像面为 `Y=0`，Link7 链原点最大镜像位置误差约 `0.0069 mm`，五个指尖站点最大镜像误差约 `3.82e-13 mm`。
- 手根由 `Link7 > fixed_flange > fixed_four_hole_connector > fixed_hand_base` 固定连接。候选左侧 Link7 到手根的实际矩阵与配置最大绝对误差约 `1.16e-15`，因此请求法兰旋转时手根随同一刚性链旋转。
- 左手五个 distal-tip 站点拟合平面法向接近世界 `+Z`，与钢琴键面倾角为 `0°`，指尖高度范围约 `0.0787661403–0.0787661404 m`。这里的“掌面朝下”按当前场景的指尖平面与键面平行定义；是否符合实物掌面、孔位和法兰时钟仍需硬件复核。

## 交互文件

- [left_mirror_mount_candidate.html](../preview/outputs/arm_left_mirror_mount_candidate_20261011/left_mirror_mount_candidate.html)：新的候选 Plotly 3D 轨道，内嵌实际编译的 M6 连杆、四孔连接件和 A3.2 手部网格，可拖动旋转、滚轮缩放，并切换全部／左手／右手／真实网格／安装链／指尖；右侧面板固定显示 J5/J6/J7 镜像角度和掌面安装结果。
- [left_mirror_mount_candidate.png](../preview/outputs/arm_left_mirror_mount_candidate_20261011/left_mirror_mount_candidate.png)：候选静态总览。
- [candidate_report.json](../preview/outputs/arm_left_mirror_mount_candidate_20261011/candidate_report.json)：候选角度、安装偏移、验证和来源散列。
- [旧基线 HTML](../preview/outputs/arm_mirror_palm_down_scene_20261011/training_scene_mirror_palm_down.html)：保留用于对比，不代表本次新角度候选。
- [scene_report.json](../preview/outputs/arm_mirror_palm_down_scene_20261011/scene_report.json)：镜像、安装链、掌面平面、初始角度和输入文件散列。
- [arm_mirror_palm_down_scene_bundle.zip](../preview/outputs/arm_mirror_palm_down_scene_20261011/arm_mirror_palm_down_scene_bundle.zip)：HTML、报告、说明和manifest。

首次打开 HTML 需要网络加载 Plotly CDN；场景数据已经写入 HTML，不依赖运行中的预览进程。

## 复现

```bash
MUJOCO_GL=disable training_mjlab/.venv/bin/python \
  preview/scripts/export_arm_mirror_palm_down_scene.py
```

脚本只做 CPU 编译、`reset_cpu()` 和 FK/站点几何检查：`physics_steps=0`、`policy_evaluations=0`、`hardware_commands=0`。机械臂碰撞过滤和无臂自碰撞合同仍由训练源码保持，未在本次导出中改动。 :codex-annotation{index="1"}
