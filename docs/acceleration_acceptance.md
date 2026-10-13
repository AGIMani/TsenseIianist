# 多子步 CUDA graph、Torch 编译与融合 Adam 验收

日期：2026-09-18。适用于本独立项目 MJLab＋Torch SAC，未调整控制频率、物理步长、求解器、更新次数、batch、FP32／TF32 或论文 SAC 参数。

## 1. 完成内容

| 开关 | 新训练默认 | 实际实现 |
| --- | --- | --- |
| `--physics-graph` / `--no-physics-graph` | 开 | 将一个控制周期的 100 次 `mujoco_warp.step` 原生核记录到同一 Warp CUDA graph，一次提交执行；仍然积分 100 次，每次 0.5 ms |
| `--compile-learner` / `--no-compile-learner` | 开 | `torch.compile(fullgraph=True)` 编译 critic／actor 张量目标函数及其自动微分 backward；Adam 调度、Polyak、温度更新、有限值检查仍由原 runner 执行 |
| `--fused-optimizer` / `--no-fused-optimizer` | 开 | actor、critic、temperature 三个 Adam 使用 `fused=True`；lr／eps／动量／更新顺序不改 |

`SAC` 类本身默认 eager，以便 CPU 测试和旧脚本复用；训练入口显式传入默认开启的参数。评估只做 actor inference，无需编译学习器。

随机噪声和 dropout mask 在编译区域外按原顺序、使用独立 update RNG 生成。没有将网络包成带 `_orig_mod` 参数名的新模块。旧 checkpoint 可按原名读入；切换 fused/eager 时规范化 optimizer group 与 step tensor 的设备；不改变检查点权重含义。保存的训练 config 包含三个加速开关。

### 物理实现细节

本机 Warp 1.17 无法将现有 MJLab 单步 graph 的 launch 嵌入另一段 capture，实测报 CUDA error 900。最终实现直接捕获 100 次原生 `mjwarp.step`，不是嵌套 launch，也不是把 dt 改成 50 ms。首次 step 懒捕获；MJLab 重建其 step graph 时，外层 graph 相应重建。

有逐子步 callback（录音、诊断）或者启用 MJLab NaN guard 时，自动执行原逐步路径；每个 callback 仍调用 100 次。自碰撞监测核被捕获进 graph，并在回调路径逐步运行。每周期的 actuator force 保存仍在最终 forward 之前，保留能耗语义。

## 2. 实测速度

RTX PRO 6000 Blackwell；Torch 2.10.0+cu128／MJLab 1.6／MuJoCo Warp 3.11／Warp 1.17。256 环境、4096 batch、每向量步 32 次更新，FP32，TF32 关闭。相同 step-18000 checkpoint，独立启动两次，replay 为 262144 条、由 16 批有效预热 transition 重复填满；各测 12 个预热后周期，阶段之间同步 GPU。碰撞权重 **0**，没有其他 GPU 任务并行。

| 每向量步分项 | 三项关闭 | 三项开启 |
| --- | ---: | ---: |
| actor | 0.270 ms | 0.403 ms |
| 100 子步物理＋状态读取 | 244.405 ms | 246.897 ms |
| 其余任务计算 | 2.140 ms | 2.090 ms |
| replay 插入 | 0.744 ms | 0.667 ms |
| 32 次 SAC 更新（包含 replay 抽样） | 245.672 ms | 126.452 ms |
| **总计** | **493.232 ms** | **376.510 ms** |
| **吞吐** | **519.0 transition/s** | **679.9 transition/s** |

总吞吐约 **1.31 倍**，同等稳态工作量耗时约减少 **23.7%**。学习器一段约 **1.94 倍**；没有分离编译和 fused 各自贡献，不能单独给它们归因。物理段没有显著提速（本次约慢 1%），说明当前 GPU 接触／约束计算仍是主要成本，不能将“一次提交”理解为免除 100 次物理运算。

学习器 32 次预热更新耗时：关闭 0.355 s；开启 7.367 s，包含本次缓存条件下的首次编译。完全空缓存、不同 batch／模型可能更慢。这些耗时不包含初始化、评价、录音、视频、文件保存，也不代表一次新的 5M 正式训练实测；不要直接承诺 5M 完成时间。启用自碰撞还会增加监测开销，本表没有估算该开销。

原始记录：[baseline](../training_mjlab/outputs/acceleration_validation/timing_baseline.json)、[optimized](../training_mjlab/outputs/acceleration_validation/timing_optimized.json)。

## 3. 数值、状态和奖励验收

- 编译＋fused 与冻结的真实 JAX 更新参考比较：**1870 项通过**，dropout 0／0.01，各 5 次更新。参数最大绝对误差 `2.3842e−7`，loss／metric 最大误差 `2.2054e−6`，梯度最大误差 `2.5034e−6`。[JSON](../training_mjlab/outputs/acceleration_validation/sac_compiled_fused.json)
- GPU graph 与逐步执行：中立控制下 qpos／qvel／接触计数对照，包含局部 reset、graph 重建及 100 次 callback 验证。此验证不承诺长时接触轨迹逐位一致；既有 GPU 重播漂移 TODO(V32-SAC-03) 继续保留。
- 碰撞计数：同体、异手、琴键、正间隙、无效 padding 排除；重复接触和反向 pair 去重；环境隔离和局部 reset。新奖励进入实际 reward、上次 reward 观测和 episode return；旧／新 reward 合同冲突拒绝加载。
- 已知动作重播 60 步（3 s）：46 个周期检测到同手接触，25 对不同部件，每周期最多 7 对，候选权重 0.2 累计扣分约 −23。第三控制周期逐子步独立 CPU 读取接触表计数为 2，与 GPU 并集一致；reset 后计数 0。[结果](../training_mjlab/outputs/acceleration_validation/self_collision.json)。这是旧策略动作的诊断，**不是新奖励训练效果**。

## 4. 复查命令

以下在独立项目根目录执行：

```bash
training_mjlab/.venv/bin/python -m unittest \
  tests.test_acceleration_and_collision tests.test_torch_sac tests.test_mjlab_task -v

training_mjlab/.venv/bin/python scripts/validate_torch_sac.py \
  --compile-learner --fused-optimizer \
  --output training_mjlab/outputs/sac_compiled_fused_recheck.json

training_mjlab/.venv/bin/python scripts/profile_mjlab_training.py \
  --checkpoint training_mjlab/outputs/sac-20260917-150810-595009/step-18000.pt \
  --iterations 12 --output training_mjlab/outputs/timing_optimized_recheck.json

# 基线对照：另起进程，避免同时争用 GPU；输出必须用新路径。
training_mjlab/.venv/bin/python scripts/profile_mjlab_training.py \
  --checkpoint training_mjlab/outputs/sac-20260917-150810-595009/step-18000.pt \
  --no-physics-graph --no-compile-learner --no-fused-optimizer \
  --iterations 12 --output training_mjlab/outputs/timing_baseline_recheck.json

training_mjlab/.venv/bin/python scripts/validate_self_collision.py \
  --rollout preview/outputs/replays/sac-20260917-150810-595009_best-step-18000_20260918/rollout.npz \
  --output training_mjlab/outputs/self_collision_recheck.json
```

新奖励确认表：[rew. 总表与设计](reward_design_review.md)。后续建议先确认自碰撞权重，再进行固定曲目／seed 的长期 A/B 学习实验；速度测试不等于 F1 提升。

## 5. 短训练与恢复

`acceleration_validation/smoke_collision`：4环境、6向量步、24条transition、10次更新，三项加速开启，自碰撞权重0.2，batch8；每3步时间截断，完成8次环境截断reset、两次独立评价和checkpoint保存。训练日志中自碰撞平均分项约 −0.60～−1.20。随后 `smoke_resume` 从 `final.pt` 恢复，再插入8条新transition并更新2次，总更新次数10→12，reward权重和加速参数保留。

这些是执行链路验收，不足以说明新奖励能改善策略。正式训练必须另建运行目录，避免把这里的短测权重当作最优训练结果。


附加回放验收：从带0.2权重的smoke checkpoint导出完整158控制步视频，自动恢复该权重，`rollout.npz`为158×6分项奖励且逐帧求和等于总reward；实际琴键音轨采样15800物理子步，输出1280×720 H.264＋AAC／8.95s。[回放验收报告](../preview/outputs/acceleration_smoke_collision/report.json)。这是未充分训练的测试策略，不能当作新的最佳演奏结果。窗口交互未人工验收。
