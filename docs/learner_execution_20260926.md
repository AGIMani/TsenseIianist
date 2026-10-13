# SAC 执行加速：保持 Shadow 更新预算

日期：2026-09-26。用户要求恢复执行速度，同时保留全部算法、网络结构、batch 与学习预算。本轮保留每新增 transition 一次更新、batch256、replay 1M、FP32、TF32 关闭、原学习率／Adam／损失／随机抽样顺序。这里的优化不减少 4,995,193 次更新。

先前变慢原因见 `docs/training_speed_audit_20260926.md`。旧 46 分钟训练只有 593,088 次更新；不能通过偷偷回退该预算来声称等价提速。

## 已采用的执行方式

### 完整 learner CUDA graph

`rl_torch/learner_graph.py:CUDALearner` 捕获完整 Critic → Polyak → Actor → 温度更新，以及 replay gather、原始 dropout／Gaussian 随机生成和逐次健康检查。保留原有模块、参数名、梯度与 Adam state 格式，检查点不保存 CUDA graph。

replay 在早期逐渐变大，`torch.randint(replay.size, ...)` 的上界必须实时更新，因此这一个随机整数操作保留在图外，使用相同 generator、原 batch 形状和相同整数分布。输入 indices 写入固定 GPU 张量，图内读取所选 replay 行。没有把早期采样上界错误冻结为建图时容量。

Polyak 仍先乘 `1−tau`，再加 `tau×source`，只将各参数上的操作合并成两个 foreach kernel，不把两步替换为不同公式。单独普通 graph 对照中，累计 88 次更新的网络、目标网络、温度、指标以及两个 RNG 状态完全相同。

### Adam 的执行布局

`rl_torch/adam_execution.py` 是可选 CUDA 执行适配。它直接包含当前已安装 PyTorch 的 `ATen/native/cuda/fused_adam_utils.cuh`，调用其中原 `adam_math`，没有另写一套 Adam 公式。

原 fused Adam 对本网络的中小张量使用 chunk65536／512 threads，Profiler 中 Critic、Actor、温度三个 Adam kernel 合计约 430 μs／更新，占未改布局的完整图 GPU kernel 时间 36.4%。其 beta/lr/epsilon 为 double，使部分中间运算使用 FP64；大量线程重复计算相同 bias correction，且原分块下本网络并行块较少。

适配只改分块为 chunk1024／256 threads，在同一 block 内共享一次相同的 double bias correction，并继续调用原 PyTorch 算术。保留原 float32 参数／动量／step，原 double 中间运算、epsilon 位置、乘加顺序、step 递增及 checkpoint state 布局。不启用 `--use_fast_math`、AMP、TF32。

CUDA 编译／执行布局仍可能带来浮点舍入差异：1181 维、原 256×3 网络的对照中，第 1 次更新完全相同；累计 88 次更新最大权重差 **1.19e−7**，log-temperature 差 **1.86e−9**，RNG 状态始终完全相同。**不能声称优化布局之后逐 bit 相同或长期策略轨迹完全相同。** GPU 验收使用明确容差，并保留普通 graph／普通 learner 供对照。

## 接口与开关

```python
from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner

# 先加载检查点；replay 至少已有一个 batch 后才能创建。
learner = CUDALearner(agent, replay, optimizer_layout=True)
metrics = learner.update(update_count)  # 精确执行 update_count 次顺序更新。
```

训练入口采用 `--learner-graph`、`--optimizer-layout` 显式记录实际运行分支。不支持 CUDA 扩展构建的机器可用 `--no-optimizer-layout` 保留普通完整 graph；`--no-learner-graph --no-optimizer-layout` 使用原逐次更新执行。关闭执行加速不改变 update 数量／batch／算法。布局依赖 graph，不能无图单独开启。

建图中的预热更新是临时操作：网络、目标网络、Adam 原有动量和 step、learner RNG、replay RNG、更新计数全部原地恢复，避免丢失捕获地址。原来尚未初始化的 Adam state 只预分配等价零状态，不产生真实优化步骤。非空 Adam warm restart 也经测试。图存活期间不允许替换模型、replay 或 Adam state 存储；检测到地址改变会拒绝继续，必须重新建图。

返回的 `metrics` 是末次图执行的张量视图，下一次图更新会覆盖它。训练器在记录该步指标后再开始下一步，不应把这些视图直接保存在需要永久保留的历史列表中。

## 有限值检查

每次更新仍检查原有全部 scalar metrics 是否有限，在 GPU 上将结果做累计 AND。`update(count)` 返回前强制读回检查；任何一次出现 NaN／Inf 都使整块失败，即使后续值偶然恢复有限也不会清除失败。

错误检测从原来的“每次更新立即向主机报告”改为“当前向量步更新块结束报告”，最长为该块的更新数量；**下一次环境动作、评估或保存之前会失败**。失败后的 learner 禁止复用，不会继续从损坏参数训练。没有删掉检查。

## 实测性能（独立 learner）

RTX PRO 6000 Blackwell，Torch 2.10.0+cu128，FP32／TF32 关闭；Actor/Critic256×3，batch256，随机数、损失、顺序更新预算保持。下表为独立 synthetic transition 短测，512 次顺序更新，包含 replay 随机抽样，已预热，不含物理／评估／保存／初次编译；不是完整训练耗时。

| 执行方式 | 观测／动作 | 512 次更新 | 单次平均 | 相对同次普通执行 |
| --- | --- | ---: | ---: | ---: |
| 普通执行，已编译 loss＋fused Adam | 1180／47 | 1.900 s | 3.711 ms | 1× |
| 完整 graph，原逐参数 Polyak | 1180／47 | 0.706 s | 1.379 ms | 2.69× |
| 完整 graph＋foreach Polyak | 1180／47 | 0.689 s | 1.347 ms | 2.78× |
| 完整 graph＋Adam 布局 | 1180／47 | 0.484 s | 0.946 ms | 3.92× |
| 普通执行对照 | 1181／47 | 1.922 s | 3.755 ms | 1× |
| 完整 graph＋Adam 布局 | 1181／47 | 0.487 s | 0.951 ms | 3.95× |

1181 维是新准备阶段合同下的输入维度；本轮未改变隐藏层。短测 replay 容量32768、实际初始4096行，是隔离 learner 的固定输入实验；不能将它作为正式1M replay／完整512环境训练吞吐。

在未改 Adam 布局的普通 graph 中，单独只调用 `graph.replay()` 512 次用 0.687 s；加回图外 randint 为 0.691 s。动态上界采样只占很小部分，不能通过冻结 replay.size 换取速度。

按 0.951 ms×4,995,193 次计算，仅 learner 就约 **79 分钟**；这还没有物理、评估和保存，不能声称已恢复旧 46 分钟。单次微基准不保证长期耗时，真实训练入口的对照结果如下。

## 真实训练入口的同预算对照

两次实际运行使用**完全相同**的环境、初始准备合同与算法：512 环境，1181 观测／47 动作，batch256，replay 容量 1M，40 个向量步＝20,480 条 transition，前4,999条随机，共 **15,481 次完整更新**。区别仅为完整 learner graph／Adam 执行布局开关。

| 实测项目 | 原执行方式 | graph＋Adam 布局 |
| --- | ---: | ---: |
| 40步总时间，含初始化／评估／保存 | 69.722 s | 26.441 s |
| learner 建图／预热恢复 | 0 s | 1.419 s |
| 稳态普通向量步中位耗时 | 1.96895 s | 0.53217 s |
| 稳态普通步骤吞吐 | 260.0 条/s | 962.1 条/s |
| replay 实际填充量 | 20,480 | 20,480 |

短训总时间提速 **2.637 倍**，排除启动和周期评估影响的普通步骤提速 **3.700 倍**。显存中确实分配了1M容量，并非用小容量替代；短训尚未填满1M，也未覆盖完整训练阶段和所有episode reset。

用此次稳态步耗时，加一次实际测得的评估额外开销，按原5M预算外推约 **6,094秒＝102分钟**。这只是估计，**没有实际跑完新的5M**；不含周期检查点保存及长期吞吐／接触分布变化。原46分钟对应更少更新，本次保持Shadow预算仍未恢复到该墙钟时间。

原始对照：[paired_training.json](../preview/outputs/diagnostics/policy_preparation_acceleration_20260926/paired_training.json)。两个实测 run 为 `training_mjlab/outputs/policy-preparation-speed-baseline-20260926-010942-269460` 与 `training_mjlab/outputs/policy-preparation-speed-accelerated-20260926-011131-508283`。

## 已试验但未启用的 max-autotune

独立诊断将损失编译配置改为 `max_autotune=True`、`triton.cudagraphs=False`、`force_same_precision=True`，相当于禁止嵌套 graph 的完整形状搜索；TF32 仍 False，矩阵精度 highest，候选日志显示 `ALLOW_TF32=False`。

1181／47／batch256 测得 0.4216 s／512 次（0.823 ms／更新），比已采用布局再快约 13.4%，首次建图／调优约 24 秒。但 Actor 第一次更新的最大参数差达到 **1.21e−5**，超过本轮已采用方案的数值验收阈值。虽然 scalar loss 和 RNG 可相同，梯度微小差异仍会经 Adam 放大。因此保留诊断证据，**没有修改正式 SAC 编译选项，也没有静默启用**。

## 验证与构建依赖

实际命令：

```bash
training_mjlab/.venv/bin/python -m pytest tests/test_learner_graph.py -q
```

结果 **8 passed**，覆盖普通 graph／Adam 布局两分支：

- 建图不消耗真实更新、模型参数与 RNG；动态 replay 增长／环形覆盖后的抽样序列保持。
- 普通 graph 精确对照；布局分支在固定容差内。
- 非空 Adam state 的 warm restart → capture → 继续更新；图生成的状态可在普通 eager SAC 加载继续更新。
- 非正／非整数更新数、replay 不够、重新加载导致 Adam 地址改变时拒绝。
- 任何一次非有限值导致块失败，失败不能被后续有限值抹去，禁止继续复用。

本次测试修正过一个 fixture 问题：将同一个 CUDA optimizer state dict 同时加载给两个对象会在 PyTorch 中别名共享 tensor；改为各自 deepcopy，模拟实际两个独立 checkpoint 反序列化。没有通过放宽阈值掩盖该问题。

新增固定构建依赖 **ninja==1.13.0**，只安装到 `training_mjlab/.venv`；没有变动已有 Torch／MuJoCo／numpy。CUDA 布局首用需本机 CUDA Toolkit **12.8**（nvcc）、C++ 编译器及 PyTorch 开发头文件；编译结果缓存于 Torch 默认扩展缓存，当前为 `~/.cache/torch_extensions/py312_cu128/tsensei_adam_execution_v1/`。源码／Torch版本／CUDA架构变化需要重新编译；归档只有源码，不能把本机缓存当成随项目分发的依赖。

PyTorch 头文件还需 nvcc `--expt-extended-lambda`，已显式配置。若编译依赖缺失，明确报错，可显式关闭布局；不悄悄切到其他精度或算法。

## 实际参考来源与原始证据

只参考本地安装版本的实现：

- `torch/cuda/graphs.py`：捕获／side stream／replay／generator state 注册约定。
- `torch/optim/optimizer.py:_cuda_graph_capture_health_check`；`torch/optim/adam.py:_fused_adam`：capturable 和原 step／Adam 操作。
- `torch/include/ATen/native/cuda/fused_adam_utils.cuh`：原 `adam_math`、double bias correction、float状态算术。
- `torch/include/ATen/native/cuda/MultiTensorApply.cuh`：原 chunk65536／block512。
- `torch/_inductor/config.py`：本地 max_autotune、force_same_precision 配置，仅用于未启用候选。
- 项目 `rl_torch/sac.py`、`replay.py`、`checkpoint.py`：原 SAC 顺序、RNG、抽样和检查点合同。

证据目录：`preview/outputs/diagnostics/learner_graph_20260926/`。包括 `probe*.py/json/log`（1180、1181、Adam布局与未启用autotune）、`tests.log`、`profile_graph.py/log`、`cuda_trace.json`。Profiler 会额外引入采样开销，kernel 时间分解不等于独立墙钟时间。

未解决：同 Shadow 预算的完整5M墙钟时间、长期策略质量和所有接触轨迹仍需正式训练验收；不能将当前约4倍learner加速描述成已经复现旧46分钟训练。
