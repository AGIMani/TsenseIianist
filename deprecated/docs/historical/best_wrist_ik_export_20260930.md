# 1103 当前训练 best：腕部差分IK视频与关节速度

2026-09-30。当前最新正式训练为1103十倍gesture/40–400Hz组，已结束；
本次使用其`best.pt`（step 19400、9,932,800条样本），不是final，也不是跨历史实验重新排名。
选模指标是三回合逐帧音乐F1最小值0.92444448，均值0.92514997。
旧1078的20Hz分数不与当前40Hz量化直接排名。

## 交付

目录：`preview/outputs/1103_best_20260930_wrist_ik/`。

- [带M6双臂的原速有声视频](../preview/outputs/1103_best_20260930_wrist_ik/arms/arms_original.mp4)
- [原始策略有声视频](../preview/outputs/1103_best_20260930_wrist_ik/hand_replay/replay.mp4)
- [右臂角度/速度图](../preview/outputs/1103_best_20260930_wrist_ik/arms/right_joint_velocities.png)
- [左臂角度/速度图](../preview/outputs/1103_best_20260930_wrist_ik/arms/left_joint_velocities.png)
- 每臂J1–J7各有独立PNG；汇总图另有PDF。
- `arms/joint_angles.csv`：时间及14关节角，单位rad，3651行。
- `arms/joint_velocities.csv`：区间中点时间及14关节速度，单位rad/s，3650行。
- `arms/wrist_poses.csv`：两手世界XYZ和xyzw四元数，3651行。
- `arms/arm_trajectory.npz`：原始关节角、腕部目标位姿、IK残差；`joint_velocity_samples.npz`保存速度。
- `arms/report.json`及`validation.json`保存参数、限位/速度/FK验收；`export_bundle.zip`包含两段视频、全部图表、CSV和报告。

视频均40FPS、406帧、10.15秒，机械臂视频1400×900；包含1秒策略准备、完整音乐、.25秒释放及1秒末尾停留。
音频来自真实400Hz琴键触发，保留末尾钢琴衰减，不使用目标乐谱代替真实弹奏。
本次单回合音乐逐帧F1 **0.92550265**，micro **0.92603266**，音乐错键帧0；与训练三回合统计分开记录。

## 求解与曲线定义

手部物理完全由1103匹配归档重建，严格检查源码/配置/检查点合同。
记录3650个400Hz子步及初态，共3651个状态，物理时长9.125秒。
对每个状态做FK，读取左右手URDF `base_link` 世界位姿作为wrist目标；
使用项目内M6-CCS-696七关节URDF及当前A3.2_tianji连接件工具变换。
基座间距93mm、肩座中心[.52,0,.48]m，当前工具安装配置原样保留。
手腕姿态也参与解算，避免只跟随位置造成法兰朝向不确定。

先求一次有界初态IK，再逐物理帧使用阻尼雅可比增量，逐次限制数值修正并保持URDF角范围。
**速度 = (q[i+1]−q[i])/0.0025秒**，时间戳位于区间中点；没有滤波、时间拉伸或速度裁剪。
视频使用相同关节序列的线性插值，避免三次样条在关节限位附近过冲。
图中红线为URDF的±3.1416rad/s。各关节最大绝对速度：

| 关节 | 右臂rad/s | 左臂rad/s |
|---|---:|---:|
| J1 | 0.331844 | 1.267729 |
| J2 | 1.071127 | 0.740797 |
| J3 | 0.306422 | 1.786197 |
| J4 | 0.446448 | 0.683675 |
| J5 | 0.187226 | 1.287167 |
| J6 | 1.803287 | 1.253479 |
| J7 | 0.267957 | 1.098609 |

14关节采样角度及区间速度全部满足URDF范围。独立FK核对最大位置误差
**0.029106mm**、最大旋转误差**0.000239057rad（约0.01370°）**；
线性插值区间中点最大位置误差约0.029069mm。
左J7有298个采样点触及角度限位，冗余分支没有做避限优化。
这不是全路径加速度、动力学或避障验收；原腕部离散角加速度峰值右154.10、左140.91rad/s²，
不能因为关节速度低于上限就宣称真机可直接执行。

## 实现、复现和验证

`preview/scripts/replay_arm_ik.py`新增`--original-samples --render-timing original`：
保留原采样时间、导出14个单关节图和CSV；视频标签参数化，采样率读取真实physics_dt。
视觉安装件从当前URDF选取，通过记录场景body/visual引用定位mesh，兼容旧资产命名。
原平滑/限速方案保留，当前导出只用原速采样方案。

```bash
bash scripts/hpc/submit.sh replay \
  --source-archive "$PWD/logs/hpc/1103/source.tar.gz" \
  --checkpoint "$PWD/training_mjlab/outputs/hpc/gesture10-comparison-20260930-140658-445247/best.pt" \
  --mode render --capture-physics --output /absolute/new/hand_replay

MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 training/.venv/bin/python preview/scripts/replay_arm_ik.py \
  --replay /absolute/new/hand_replay --output /absolute/new/arms \
  --analysis-only --original-samples --render-timing original --label '1103 best step 19400'

bash scripts/hpc/submit.sh arm-replay \
  --replay /absolute/new/hand_replay --output /absolute/new/arms \
  --render-only --render-timing original --fps 40 --label '1103 best step 19400'
```

Slurm1124完成策略回放；1126完成双臂渲染。CPU合并场景保留134物理坐标，22个mocap含视觉覆盖层。
相关轨迹导数、安装件、连接件回归6项通过；独立全帧FK、CSV/NPZ一致性、范围、插值中点检查通过。
两段最终视频完整解码通过，PNG已目视检查；Slurm作业和源归档保留。

首次平滑分析因左臂样条超限终止，改用上述未经平滑的原始采样；
首次渲染1125因旧mesh名假设失败，修复后CPU编译及1126渲染通过，失败场景保存为`scene_failed_1125`。
训练源码/奖励/检查点未变；新安装件只出现在覆盖层，不把旧训练F1说成新装配动力学F1。

TODO(TJ-ARM-02/03/04)：肩座/法兰标定、机械臂动力学及碰撞避障未验收。
TODO(V32-MOUNT-01)：真实安装及质量惯量待标定；右手仍为镜像推导。
本次增加的关节角仅用于离线运动学回放，没有向硬件发送指令。
