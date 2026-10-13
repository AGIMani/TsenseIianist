# 准备帧参与学习与执行加速（2026-09-26）

## 本轮要求

用户要求准备帧也进入训练，以学习避免曲首误触；训练提速只优化执行，不改变其余学习设置。本页覆盖同日早先“固定准备、不进入 replay”的默认决定。原固定模式保留用于旧合同和对照。

## 准备从第 0 帧就是训练

`run.sh` 使用 `--preparation-mode policy_v1 --preparation-time 1 --preparation-lift 0.02`。

1. 保留当前完整姿态为 origin，整体沿世界 Z 抬高 2 cm 作为 initial。
2. reset 真正放在 initial，关节速度与原源姿态一致；没有预先模拟完再复制终态。
3. 从第一帧开始，策略控制全部 47 个原有动作。整体支撑按五次曲线在 1 秒内下降，TZ/Yaw 仍物理移除。
4. 前 20 个控制帧是空谱，实际动作、奖励、下一观测和结束标记照常进入 replay，并参与 SAC 更新。
5. 第 21 个控制帧开始原曲目。1 秒末支撑回到 origin **高度**，手指、TX/TY/Pitch 由策略决定，不强制回写原关节角。

仅把固定动作塞进 replay，策略无法学习其动作对准备期误触的影响。因此本次采用真实策略控制。准备段不是视频前缀：训练与视频都真实从 initial 开始，琴键在准备期被按响时会如实发声和记录。

### 奖励与可见信息

采用现有 Shadow 公式：空谱期间无激活错键时 `key_press_reward=0.5`，有任何激活错键时为 0；踏板目标为关闭；能耗和掌部碰撞项照常。准备期无指法目标，指法项为 0。没有添加新的惩罚、改权重或改变按键阈值。

现有观测只有关节位置，没有移动支撑的高度。为了让策略区分准备进度，新模式在完整旧观测末尾追加一个 `preparation/remaining_fraction`：初始为 1，1 秒后为 0。当前观测为 **1181 维**，原动作仍 **47 维**；frame stack 保持原规则，额外进度是当前值。原 TZ 零槽不改作其他含义。

进度和保存的 duration/lift 唯一确定下降曲线的高度与速度。此次新增观测是让可训练准备过程可见的必要接口，不是性能优化中的网络扩宽；Actor/Critic 隐层、激活、LayerNorm、dropout 和 SAC 超参数保持。

### 对采样与评估的实际影响

- 原曲 158 步，新回合 178 步；去掉前 20 帧后，目标与指法数组和原版逐项相同。
- 512 环境、总 5,000,192 条 transition、前 4999 条随机探索、总 4,995,193 次 SAC 更新不变；准备样本包含在总采样预算内，不另加预算。
- 因回合变长，同样样本数内完整曲目的次数减少约 11.2%。前 4999 条随机样本分到 512 个环境，覆盖约 0.5 秒，当前都会落在准备前半段。这是保留原采样预算的结果。
- 0.5 秒前瞻不变：准备前半秒还看不到首音，从约 0.5 秒开始在前瞻窗口中出现首音；当前指法目标仍按原规则在对应音符帧提供。
- 原总 F1 仍含整个回合；新增 `music_only_f1`、`music_only_micro_f1` 和准备误触统计。安静的空谱帧会给原逐帧 F1 加分，不能拿新 178 帧总分直接声称超过旧 158 帧演奏。
- `--initial-buffer-time` 是准备以外的额外空谱。例如准备 1 秒＋额外 0.25 秒共 25 个空帧，首音在第 25 帧；曲目指标排除这两部分。

### 并行与回放

每个环境根据自己的控制步数更新 mocap 支撑，每物理子步都更新；支持物理 CUDA graph。局部 reset 只把指定环境重置到 initial，不推进其他环境。准备期真实 transition 与正式演奏使用同一个 replay 插入路径。

新回放不再重复添加 CPU 准备前缀：`rollout.npz` 包含全部 178 帧，音频采样来自同一 GPU 轨迹，`policy_time_offset_seconds=0`。报告分别列准备、额外空白和曲目段；准备误触不会被强制静音。

旧 `fixed_v1` 仍保留固定准备轨迹及独立前缀，`--preparation-time 0` 可完全关闭准备。完整准备合同、源码散列和观测维度继续检查，旧 1180 维检查点不能直接加载到新 1181 维策略；需要对应源码回放或新训练。

## 执行加速的范围

加速保留原 critic → Polyak → actor → temperature 顺序、随机采样、Adam 参数、batch256、replay1M、每新增样本一次完整更新、三个学习率3e−4、gamma0.8、alpha初值1、256×3隐层、20/200Hz、512环境及评估频率。

`--learner-graph` 将完整更新捕获为 CUDA 图，减少主机调度和同步；replay 随机索引使用实时有效容量，不能在捕获时固定旧容量。构图所需的试跑会恢复参数、Adam 状态、随机数和更新计数，不计入训练。每次更新仍检查有限值，图内累计状态，在当前向量更新块结束时读回，发现异常后停止，不执行下一轮采样/保存。

`--optimizer-layout` 进一步优化 GPU 上 Adam 的分块与标量偏差校正计算布局，直接使用本机 PyTorch 的原始 `adam_math`；不改 epsilon、动量、方差、step 或公式顺序。它是可选执行路径，必须同时开启 `--learner-graph --fused-optimizer`。首次构建需要本机 CUDA Toolkit/nvcc、C++ 编译器和固定版本 ninja，后续读取编译缓存。没有 CUDA 编译器时可以关闭此项、保留普通 learner graph。

没有启用 TF32、混合精度或低精度 optimizer state。图和执行布局不存入模型参数，保存标准模型/Adam/RNG 状态；恢复后重新建立图。不得在现有图仍存活时替换其参数或 optimizer storage。

### 相同配置的完整训练入口实测

两次都使用512环境、1181观测、batch256、replay1M、相同奖励/SAC/准备流程，采样40个向量步=20,480条transition、完成15,481次SAC更新；唯一设置差异为两个执行加速开关。

| 指标 | 原执行方式 | CUDA图＋Adam布局 |
| --- | ---: | ---: |
| 短跑总耗时，含训练器初始化/评估/保存 | 69.72秒 | 26.44秒 |
| 稳定阶段每向量步中位耗时 | 1.969秒 | 0.532秒 |
| 稳定阶段采样吞吐 | 260条/秒 | 962条/秒 |

短跑整体快2.64倍，稳定阶段快3.70倍。独立学习器微基准约3.95倍；三种口径不同，不混用。两次都有2次完整178帧评估，预算一致；短跑无完整训练回合结束，另由2环境190步验收reset。

**尚未恢复到历史约46分钟。** 按本轮稳定步骤及评估开销粗估，当前5M学习预算约102分钟；这是短测外推，不是已跑完5M，未覆盖长期速度波动和周期保存。历史46分钟只有约59万次更新，当前约500万次；本次没有通过减少更新兑现旧时间。FP32矩阵计算和评估仍占时间。

[完整两次运行与预算核对JSON](../preview/outputs/diagnostics/policy_preparation_acceleration_20260926/paired_training.json)；[学习器细节、数值测试与未采用候选](learner_execution_20260926.md)。

新默认已开启执行加速：

```bash
bash run.sh --output-dir training_mjlab/outputs/my-run
# 对照原执行方式（相同学习预算）
bash run.sh --no-learner-graph --no-optimizer-layout --output-dir training_mjlab/outputs/reference
```

新加速布局累计88次更新最大权重差约1.19e−7，属于实测浮点舍入差异；普通CUDA图对照逐bit相同。随机数状态和更新次数保持，不能据短数值测试承诺长训练动作完全一致。更激进的max-autotune候选虽然再快13%，首次更新权重偏差约1.21e−5超出本轮阈值，未启用。TwinQ批量矩阵乘候选首次更新偏差6.45e−6，同样未通过2e−6阈值，停止验证并未接入。

参考：[PyTorch 2.10 CUDA Graph 文档](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs)、[该版本 fused Adam 原始数学实现](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/fused_adam_utils.cuh)。实际行为以本机固定依赖源码与数值测试为准。

## 验收与剩余问题

准备核心验证包括逐子步高度、物理图开/关、异步局部 reset 隔离、1181 维观测、真实准备样本入库和原休止帧误触扣分。源码构建验证 158→178 帧，追加额外 0.25 秒为 183 帧；前缀外所有目标/指法逐项相同。短训/回放/加速数值与计时记录见本轮日志。

[正式入口验收视频](../preview/outputs/preparation/policy_active_final_acceptance_20260926/replay.mp4)与[独立验证报告](../preview/outputs/preparation/policy_active_final_acceptance_20260926/presentation_validation.json)使用380条样本短训检查点：178个真实策略帧，20准备＋158音乐，没有CPU准备前缀，音频偏移为0。支撑高度0.12→0.10m，1秒到位，逐子步曲线误差小于6e−9m。首秒实际发生4次错误发音、16/20控制帧误触，声音和状态原样保留。本视频是当前policy_v1流程验收；此前fixed模式的安静示范不能代表当前策略效果。

当前源码另存于`training_mjlab/source_versions/policy_preparation_acceleration_20260926.tar.gz`，此前归档保留；只归档源码、入口及依赖清单，不保存永久编译场景。最终散列、语法及文档链接核对见[完整性记录](../preview/outputs/diagnostics/policy_preparation_acceleration_20260926/final_integrity.json)。

V32-START-01：准备期现在能够学习，但只有长训策略实测才能判断是否减少开头误触；未训练好的策略仍会误触。V32-START-02：奖励仍按 20Hz 控制末端采样，音频按 200Hz，短错音盲区未自动消除。原驱动/接触校准及 GPU 接触非逐bit重复的限制继续有效。
