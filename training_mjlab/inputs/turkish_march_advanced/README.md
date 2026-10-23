# Turkish March：当前 A3.2 训练指法

当前输入：`model_xy_1000hz_candidate/turkish-march-advanced_fingered.proto`。
原谱：`original.mid`，SHA256 `83707e4f40347d0a51402349a8ef8af7cf59c357656b50432591b3e8fceb88eb`。

2026-10-20 用户明确选择保留原谱，采用 1000 Hz 物理仿真；控制仍为 20 Hz。2544 个音符的音高、起止时间、力度、轨道等全部与原始 MIDI 逐项核对一致，只改变 `NoteSequence.part` 指法。轨道 0 为右手（1460 音符），轨道 1 为左手（1084 音符）；右手 0–4、左手 5–9。4450 控制帧加 0.25 秒松键尾段，总计 4455 帧；控制网格无一指多键冲突。

## 选择依据与边界

`scripts/fit_turkish_fingering.py` 在固定 v4 源码中按训练的 A3.2、无臂、0.06 m 高度、0.04 m TY、30° yaw、移除 TZ/yaw 和 60°～80° 耦合配置编译模型，读取实际耦合初始指尖位置及任务琴键目标。`model_geometry_1000hz.json` 保存完整 recipe、几何、来源 SHA。没有猜测新的腕方向或自然姿态。

比较原启发式候选和三个几何动态规划候选；保持持续音指法不变，最小化共享腕平移后的 XY 对齐残差与相邻事件换位代价。按统一 `posture + 0.35 × movement` 指标选中 `model_xy_movement_0.35`：综合代价 5133.638→3831.658（−25.36%），平均对齐残差 18.428→16.684 mm，换位代价 11218.828→8120.672（−27.62%）。完整比较见 `model_xy_1000hz_candidate/selection_report.json`，逐音符表见同目录 `fingering.csv`。

这是四个候选中按当前模型中立姿态几何指标选出的方案，不是全局最优、关节空间可达性、无碰撞或实际触键成功证明。最大中立姿态残差仍有 71.22 mm，关节运动是否补偿需后续训练/物理回放评估；不能把几何指标写成 F1。

训练沿用根目录 `all_use_this_train.sh`，本轮末尾覆盖 `--physics-hz 1000 --no-physics-graph --replay-capacity 3000000` 及本曲目路径。源码归档 `source_cross_hand_substep_v4.tar.gz` 的 SHA256 为 `2de5639f3fed9c872f54de3e00759e83d27ccf1a48b083387fd0f9c703cc9c65`。Slurm6187 四卡短测完成 384 transitions、92 次学习更新，仅证明启动与流水链路，不证明学会整曲。完整训练登记在 `results/exper023/submission.json`。

## 历史失败诊断

`generated_20hz_candidate/` 为时值不变的旧启发式基线，`generated_v1/` 是更早 40 Hz 候选；`strict_v1_candidate/` 曾擅自缩短六处结束时间，违反保留原谱要求，已撤销其活动注册地位，仅保留失败诊断。exper020 编译失败；exper021 回放池不足；exper022 CUDA Graph 捕获失败，只有训练前初始评估检查点。旧运动学回放直接设置琴键位置，其 F1=1 不能用于评判该指法的物理触键能力。
