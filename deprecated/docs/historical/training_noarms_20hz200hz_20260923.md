# 小星星：无机械臂、20/200 Hz（2026-09-23）

## 本轮配置

用户明确要求新增去除机械臂参数，并按20 Hz控制／200 Hz物理重新训练小星星。本次从头训练，不加载旧检查点。

|项目|值|
|---|---|
|训练参数|`--no-arms`|
|曲目|TwinkleTwinkleRousseau，原有指法，158帧|
|控制／物理|20／200 Hz，0.05／0.005秒，每动作10子步|
|前瞻／观测／动作|0.5秒＝10步，1180维观测，51维动作|
|动作映射|mixed_v1，Pitch物理范围±0.2 rad|
|碰撞奖励|同手自碰撞权重0.2；跨手掌指仍无接触+0.5、有接触0|
|机械臂|无M6几何、IK、可达腕部约束或臂碰撞查询|
|接触|保留Shadow规则；钢琴solref时间常数2×dt＝0.01秒|
|SAC|沿用已确认Stage1；256环境，batch4096，每向量步32次更新|
|采样预算|19,532向量步×256＝5,000,192条transition|
|GPU加速|物理子步CUDA graph、学习器编译、融合Adam均启用|

论文pp.6–8采用40/400 Hz和子步插值；本轮时序按用户新决定覆盖该次训练，未加入插值或自动重调gamma／奖励权重。物理时间折扣跨度会变化。源预览contact_settings.json和旧冻结包保持原设置。

## 参数行为

训练入口 `python -m tsensei_v3_2_pianist.rl_torch.train --no-arms ...`：

- 若输入M6 scene/task，在本次run的`hand_only_bundle/scene`、`hand_only_bundle/task`生成独立副本，去除tj_机械臂／肩座／目标球和相关资产、m6元数据、mocap及臂碰撞名单，恢复原跨手掌指名单。旧输入保持不变。
- 若输入已无机械臂，直接校验并使用；训练和评估使用相同的有效包。config.json记录有效路径和哈希，回放也使用这些无臂包。
- 不加此参数时继续尊重输入冻结包。参数本身不修改频率；频率由新导出的scene/task决定。
- 去臂副本不复用原M6 CPU轨迹参考。未知布局失败报错，不默默忽略机械臂。

## 可重复命令

在本独立项目目录执行；导出目录必须未存在。

```bash
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_scene.py --output training_mjlab/outputs/scene_twinkle_noarms_20hz200hz_20260923 --control-hz 20 --physics-hz 200
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_task.py --scene training_mjlab/outputs/scene_twinkle_noarms_20hz200hz_20260923 --output training_mjlab/outputs/task_twinkle_noarms_20hz200hz_20260923 --midi TwinkleTwinkleRousseau --lookahead 10
/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/.venv/bin/python -u -m tsensei_v3_2_pianist.rl_torch.train --scene /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/scene_twinkle_noarms_20hz200hz_20260923 --task /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/task_twinkle_noarms_20hz200hz_20260923 --no-arms --stage 1 --device cuda:0 --num-envs 256 --vector-steps 19532 --batch-size 4096 --gradient-steps 32 --seed 42 --physics-graph --compile-learner --fused-optimizer --progress --output-dir /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-noarms-20hz200hz-20260923-123052 --self-collision-weight 0.2 --action-profile mixed_v1
```

再次训练需换新的`--output-dir`。不必重复导出已存在的冻结包。

## 本次启动与验收

- 启动：2026-09-23T12:30:52.055984（本机时间），PID `2152433`。
- 输出：`/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-noarms-20hz200hz-20260923-123052`。
- 日志：`/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-noarms-20hz200hz-20260923-123052.console.log`，同目录`.launch.json`保存命令与PID。
- scene导出往返检查和task模型／初始姿态／接触数组一致性检查通过；task使用scene实际physics_dt，避免沿用默认400 Hz。
- 2环境GPU验收：M6去臂副本及新20/200包各执行20个控制步，观测和奖励均有限；确认nmocap=0、arm_ik=None、arm_collision_model=None及原始M6输入哈希不变。
- 新20/200包观测1180，物理dt=.005、10子步；跨手各21geom。验收原始日志：`training_mjlab/outputs/noarms_validation_20260923.log`。
- 正式训练已进入SAC更新；记录时vector_step=1210，transitions=309760，updates=6752。首轮评估F1约0.0886，尚非最终结果。短步测试不代表长期收敛或严格CPU/GPU全轨迹一致。

## 保留事项

V32-SAC-03回放与训练评估差异、V32-06软接触限制仍保留；本轮不宣称解决手部动作突变、实体机械臂动力学或可达性。移除机械臂后，策略没有机械臂可达性限制。


## 前期视频（2026-09-23）

选本轮最早保存的`step-1000.pt`：256,000条采样、32次SAC更新，并非训练完毕或最佳模型。使用当前冻结20/200 Hz无臂包，演奏者前方正交视角、完整钢琴、实际触键与踏板音轨。

[播放视频](../preview/outputs/replays/twinkle_noarms_early_step1000_20260923/replay.mp4)；同目录report.json、rollout.npz及played.mid保存复现信息。

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py --checkpoint training_mjlab/outputs/sac-noarms-20hz200hz-20260923-123052/step-1000.pt --camera-view front --camera-framing full --audio --output preview/outputs/replays/twinkle_noarms_early_step1000_20260923
```

验证：158控制帧完成，视频8.95秒、1280×720、20 fps、H264＋AAC；NVIDIA硬件渲染，1580物理子步记录267个实际note-on，音轨非乐谱配音。回放F1=0.073116，与训练记录0.088638存在差异，继续保留V32-SAC-03，不能视为完全复现训练评估。未修改训练参数或训练代码。


## 最佳检查点视频（2026-09-23）

训练已完成5,000,192采样，耗时2949.49秒（49分9秒）。训练日志最佳step-17000，F1=0.510970；最终F1=0.479325。本次按用户要求导出最佳检查点。

[播放最佳视频](../preview/outputs/replays/twinkle_noarms_best17000_20260923/replay.mp4)

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py --checkpoint training_mjlab/outputs/sac-noarms-20hz200hz-20260923-123052/step-17000.pt --camera-view front --camera-framing full --audio --output preview/outputs/replays/twinkle_noarms_best17000_20260923
```

完整158控制帧，视频8.95秒、1280×720、20fps、H264/AAC；演奏者前方正交全琴视角，音轨来自1580物理子步的实际按键与踏板（69次note-on），非乐谱配音。回放F1=0.506751，与训练0.510970略有差异，V32-SAC-03继续保留。报告和轨迹位于视频同目录。未修改训练／奖励／渲染代码。
