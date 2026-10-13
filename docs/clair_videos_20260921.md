# 《月光》最佳检查点视频（2026-09-21）

用户要求导出《月光》视频。按上一轮检查结果，分别导出原动作映射和最新mixed_v1两轮的最佳已保存检查点，均为step-1000；选择依据为训练日志最高逐帧按键F1，详见[训练结果报告](training_results_20260921.md)。视频保留实际错音／漏音，不代表已经学会曲目。

| 配置 | run | 日志最佳F1 | 视频 |
| --- | --- | ---: | --- |
| 原动作映射legacy，Pitch±0.15rad | sac-20260920-204729-873996 | 0.110728 | [完整有声视频](../preview/outputs/replays/clair_legacy_best_20260921/replay.mp4) |
| 最新mixed_v1，Pitch±0.2rad | sac-20260920-212536-832002 | 0.093584 | [完整有声视频](../preview/outputs/replays/clair_mixed_best_20260921/replay.mp4) |

## 导出设置与文件

- 正对演奏者、俯视30°的正交全琴视角，避免透视／鱼眼效果；完整键盘和双手可见。
- 保留彩色目标指法（拇指粉红色）、实际按键绿色高亮和逐帧指标。
- 音频根据400Hz物理子步的实际琴键按下／释放及踏板事件，用钢琴音色合成；没有用目标MIDI替换实际演奏。力度仍沿用既有合成规则，不声称已经标定真实声学。
- 自动加载检查点对应冻结场景、任务、动作profile与自碰撞0.2；不使用当前源模型覆盖历史范围。
- 每个输出目录保存selection.json（选择依据）、report.json（实际回放指标／相机／音频）、rollout.npz（轨迹）、played.mid／played.wav、audio_events.json与audio_substeps.npz。
- 导出日志在preview/outputs/replays/export_logs/clair_legacy_best_20260921.log及clair_mixed_best_20260921.log。

## 重复导出命令

从独立项目根目录执行，输出目录自动带时间戳：

```bash
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir training_mjlab/outputs/sac-20260920-212536-832002

training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir training_mjlab/outputs/sac-20260920-204729-873996
```

本次沿用现有导出程序和已确认评估口径，不修改训练／奖励。V32-TRAIN-01（月光退化原因）、V32-SCORE-02（候选指法可达性）、V32-SAC-03（跨次物理差异）、V32-06（接触标定）、V32-AUDIO-01（力度）继续保留。


## 实际验收

- 两个导出进程均正常退出，整曲2587个控制步／25870个物理子步，视频2628帧、40fps、65.7秒（含初始帧及结尾停留）。ffprobe确认1280×720 H.264＋AAC，音视频时长一致；两段视频完整解码无错误，截图已查看完整钢琴／双手／彩色指法。
- legacy实际回放F1=0.12735270，mixed_v1实际回放F1=0.10795062；训练日志分别0.11072775／0.09358424。两者并非逐值复现，保留V32-SAC-03，不替换原训练曲线或把本次回放称为日志精确复现。
- 实际音轨note-on事件分别1053／1059，48kHz双声道，RMS分别0.05186／0.05215，非静音。事件数含错键和重新触发，不等于弹对的音符数；力度127尚未标定。音视频测量保存为各目录report.json／ffprobe.json。


## 双手相撞定位（2026-09-21）

用户问为何手看起来撞在一起。检查两段视频的原始rollout.qpos，用对应冻结SceneBundle恢复模型，对每个控制帧做原生MuJoCo几何接触检测（mj_forward，不重新积分，不推断原GPU接触力）。[原始诊断JSON](../preview/outputs/diagnostics/clair_inter_hand_20260921.json)保存接触对、时间、负距离、首条腕目标和录像分项奖励。

| 指标 | legacy | mixed_v1 |
| --- | ---: | ---: |
| 初始姿态左右手接触 | 无 | 无 |
| 首个采样到双手接触的时间 | 0.075秒 | 0.075秒 |
| 存在双手接触的保存帧 | 2558/2587（98.88%） | 2564/2587（99.11%） |
| 最深负接触距离 | −8.57mm | −9.68mm |
| 首条右TX目标 | −148.48mm | −163.36mm |
| 首条左TX目标 | +161.51mm | +153.33mm |

首个接触对为左右拇指根部rotaback_link2；较深交叠出现在左拇指PIP与右中指DIP之间。负距离是碰撞几何的交叠，不等同真实手表面穿透深度；这里采样间隔25ms，实际首次接触可能在两个保存帧之间。本次没有重建GPU子步接触力。

### 原因链

1. 两段是日志最佳step-1000，但仅32次更新，策略仍接近初期输出。TX使用绝对位置范围映射，右范围[−760.5,460.5]mm、左[−460.5,760.5]mm；动作0对应右−150mm／左+150mm，并非保持初始位置。两手初始安装在琴宽方向±150mm，故上述首条TX目标让它们向中央靠拢。初始掌部中心琴宽间距313mm，mixed在0.1秒仅105.44mm；期间同时TY向琴内约73～76mm、TZ向下约12mm、Yaw从初始0.25rad被命令到接近0，也改变接触姿态。
2. mixed_v1没有改TX；TY/TZ只是策略坐标平移，网络近零经shift后仍映射物理中点，不会自动把起始物理目标变成初始姿态。scene.step直接设绝对ctrl，没有子步目标插值或目标变化率限制。
3. 已启用self_collision_reward=−0.2×同手body对数，SelfCollisionTracker的self类别不含inter_hand。原forearm_reward只检查两块指定掌部geom：无掌掌接触得0.5、有则0。跨手手指—手指／手指—掌部并不由这个项完整覆盖。故不能说完全没有双手相关奖励，但当前覆盖不足。mixed视频同手惩罚平均仅−0.00541，仍有99.11%的采样帧跨手接触，两者并不矛盾。
4. 物理采用已确认的Shadow软接触，不是绝对不可穿透约束；持续位置伺服推向重叠目标时，接触反力不保证几何零交叠。接触奖励也是软约束，无法单独保证不碰撞。

后续可考虑完整跨手接触惩罚、让初期命令对应初始姿态、目标限速／插值，但它们分别涉及奖励接触名单及动作／控制设计。论文式19名单未披露、用户要求自行填写未知映射，故本次只解释定位，不代填名单、不自行启用新奖励或改映射。V32-TRAIN-01尚不能仅凭这次定位判定所有训练退化都由相撞造成；V32-06／V32-SAC-03继续保留。


## forearm检测区域可视化

[检测凸包四视图](../preview/outputs/forearm_regions_20260921_final/forearm_regions.png)：红色右掌、蓝色左掌，灰色为不属于该奖励检测名单的手部上下文。上排初始姿态，下排mixed视频21.575秒保存姿态；右列隐藏其他模型，仅显示两块检测凸包。读取冻结task.json的palm_geoms，不以手绘边界代替碰撞模型。MuJoCo凸包显示标志开启；图是几何示意，不重建原GPU接触力。

复现（使用新的输出目录）：

```bash
training_mjlab/.venv/bin/python preview/scripts/draw_forearm_regions.py   --recording preview/outputs/replays/clair_mixed_best_20260921   --time 21.575 --output preview/outputs/forearm_regions_new
```

对应geom为right_/Right_hand_palm_link_collision_0与left_/Left_hand_palm_link_collision_0（本冻结模型ID93／137），详见产物regions.json。只改渲染器内存中的颜色／显示组，不改变源模型、奖励、训练或原视频。
