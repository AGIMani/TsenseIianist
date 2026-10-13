# A3.2 双卡采样与 SAC 学习流水

更新：2026-09-28。用户要求把当前 A3.2 源码迁移到火山云，使用两张 GPU 训练，并保留当前学习预算。原训练器的环境、Actor、SAC、replay 和评估都使用同一 `--device`；历史 `dual_a100_sac_plan.md` 中的双卡部分此前只是方案。

## 接口与保持的设置

本轮在 Torch 训练入口增加两个可选参数，`run.sh` 和 `run_tsensei_a32.sh` 已有末尾参数转发，无需修改默认命令：

```bash
# 同一策略的双卡流水：GPU0 采样，GPU1 学习。
bash run_tsensei_a32.sh --device cuda:0 --learner-device cuda:1 \
  --collector-pipeline --output-dir training_mjlab/outputs/a32-dual

# 同步分卡对照：相同设备分工，关闭并发和策略滞后。
bash run_tsensei_a32.sh --device cuda:0 --learner-device cuda:1 \
  --no-collector-pipeline --output-dir training_mjlab/outputs/a32-split-sync
```

不传 `--learner-device` 时仍使用 `--device`；`--collector-pipeline` 默认关闭。流水模式要求两个不同的可见 CUDA 设备，不能把同一设备的两个名字当作双卡。设备编号是作业实际 `CUDA_VISIBLE_DEVICES` 内的编号。

本实现保持来源脚本的模型、动作/观测合同、乐谱/指法、物理/控制频率、奖励、网络、三个学习率、全局 batch、replay 容量、环境数、warmup、整曲预填和每 transition 更新比。没有把环境数、batch 或学习预算乘二，没有启用 AMP/TF32。完整数值仍由实际源码、启动命令和 `config.json` 决定；不从历史方案恢复旧参数。

## 调度与数学边界

GPU0 运行物理、任务和独立 Actor 快照；GPU1 保存唯一 SAC 学习器及 replay。没有第二套独立策略，也没有梯度平均。

1. 在上次采样线程退出后，将已完成学习的 Actor 复制到 GPU0，等待复制完成。
2. 单个采样线程采集第 `t` 个向量批；主线程同时完成第 `t-1` 批对应的 SAC 更新。
3. 两边完成后，才把第 `t` 批写入 replay；第 `t-1` 批的更新只能看见前 `t-1` 批数据。
4. 保留末帧观测后，再 reset 已结束的世界。下一次采样不会与 reset 并发。
5. 评估、保存、正常结束和质量早停前，完成尚未执行的当前批更新。最多只有一个采样任务和一个待更新批，不会多采一个预算外批次。

第一次学习的编译、CUDA graph 捕获和该向量更新块先独立完成，避免捕获阶段与其他线程的 CUDA 操作冲突；之后才复用 CUDA graph 进行重叠。采样线程报错会终止训练，不将错误批写入 replay；学习器报错会等待采样线程退出，并禁止提交该新批或保存“正常完成”的结果。

**流水会改变采样调度。** 第 `t` 批动作使用第 `t-2` 批更新后的快照，与同步训练相比最多落后一个向量更新块。SAC 公式、每批可见 replay 和更新次数不变，但数据由稍旧的策略产生，因此不宣称逐次优化或训练轨迹等价。边界排空后下一批使用最新快照。同步分卡模式可用于区分设备迁移和策略滞后的影响。

Actor 快照不与 learner 共享可写参数。随机动作生成器归采样端所有；保存前将采样端 action RNG 写回原 checkpoint 字段。检查点仍是原有 learner warm restart：保存网络/优化器/温度/RNG，恢复时重新建立 replay、世界和调度，不宣称精确续接。

## 可观测信息

`config.json`/checkpoint 的 `execution_layout` 保存布局、设备、最大待处理批数、最大策略滞后和调度变化标记。`execution_environment.devices` 保存两张卡的名称和计算能力，`peer_access` 记录双向 CUDA P2P 能力；P2P 能力不等于已测得 NVLink 带宽。

`metrics.jsonl` 和最终 `summary.json` 的 `collector_pipeline_state`（指标行字段为 `collector_pipeline`）保存：

- 已采集和已完成更新的向量步、待更新步、Actor 对应的 learner update 版本。
- 采样结束时相对已完成 learner 的更新次数差，以及历史最大差。
- 采样、更新、Actor 复制、replay 写入/reset 的主机耗时。
- 采样与更新回调的重叠墙时、流水 step 墙时。step 墙时不含外部边界 flush，更新总耗时包含 flush。

这些时间是主机回调时间；采样尾部和 learner finite 检查会等待 GPU 完成，但仍不能把回调重叠直接当作 CUDA kernel 重叠。`overlapped_vector_batches` 是安排了双回调的批数，可能包括预填期没有实际更新的批。

最终 `torch_peak_allocated_bytes_by_device` 分设备记录 Torch 峰值显存；Warp 的部分显存不在 Torch 计数中，完整进程显存和 GPU 利用率还需作业内 `nvidia-smi`/分析器记录。保留原 `torch_peak_allocated_bytes` 字段供旧工具读取。

## 本机验证与远端验收

本机只有一张 RTX PRO 6000 Blackwell，不能据此宣称 A100 双卡实际加速或 GPU 并发已验收。

已执行：

```bash
training_mjlab/.venv/bin/python -m pytest -q \
  tests/test_collector_pipeline.py tests/test_shadow_training_budget.py \
  tests/test_preparation_training_entry.py tests/test_f1_training_repairs.py \
  tests/test_target_key_cli.py tests/test_torch_sac.py \
  tests/test_training_quality_control.py tests/test_run_entry_controls.py
```

结果：**60 passed，2 skipped，48 subtests passed**。新 CPU 调度测试用线程事件确认采样/学习确实并行进入，检查前后批 replay 隔离、fractional 更新计数、边界重复排空、早停无多采、采样/学习/提交/最终排空异常传播。现有预算、准备、F1 修复、奖励 CLI、SAC 和训练入口测试通过。两个跳过项要求真实双 GPU，覆盖 Actor/RNG 跨卡、replay 传输、普通 SAC 和 learner graph；可在两卡作业内运行同一个测试文件。

远端先在相同资源分配和软件版本下，使用相同种子、完整奖励/诊断、相同采样数，比较单卡、同步分卡和流水三种模式。短训必须超过实际整曲预填长度，否则没有学习更新，不能验证 learner 或计算训练吞吐。例如在当前脚本下选择 210 个向量步做首次功能验收，再用足够长的稳定窗口比较；最终以运行时解析的预填长度为准。

```bash
training_mjlab/.venv/bin/python -m pytest -q tests/test_collector_pipeline.py
bash run_tsensei_a32.sh --device cuda:0 --learner-device cuda:1 \
  --collector-pipeline --vector-steps 210 --no-swanlab \
  --output-dir training_mjlab/outputs/a32-dual-smoke
```

必须核对 `replay_insertions = vector_steps × num_envs`，最终 pending 为空，按当前预填边界和更新比计算的 updates 与实测一致，checkpoint 可加载且最新完整策略评估成功。随后比较排除首次编译后的 samples/s、updates/s、总墙时、两卡利用率/显存、音乐 F1 与碰撞。只把实测更快且功能正常的方案用于长期训练。

### 普通 eval.sh 与检查点回归

接口复查发现：原 `scripts/eval_from_source.py` 只允许既有训练进度字段出现在 checkpoint、缺席于 `config.json`。新增动态 `collector_pipeline_state` 会触发 `Incomplete config.json`。现仅把此动态字段加入既有白名单；静态 `execution_layout`、源码、奖励等仍须完整存在并与 checkpoint 一致，未知额外字段继续拒绝。新增 5 项测试覆盖初始/周期/最终流水进度以及布局缺失/被改写。

`eval.sh` 继续从保存的 `source_runtime` 重建，再在显式 `--device` 上创建普通 SAC。训练时保存在 GPU1 的权重先按 CPU 读取；优化器张量迁到评价设备，action/update RNG 按 CPU 状态恢复，`capturable` 恢复为普通执行。评价不要求两张 GPU，也不启动 collector 线程。双卡布局和 GPU 信息是保存的运行记录，不覆盖评价命令的设备。

必须在本次训练使用的完整源码副本内评价；`source_runtime` 对整个 Python 包和依赖版本有严格校验。修改训练包后不能直接回放旧 checkpoint，设备拆分不会放松这项检查。`eval_from_source.py` 本身位于包外，补其动态字段白名单不改变训练模型的源指纹。

```bash
# RUN 指向本次入口实际生成的时间戳输出目录。
bash eval.sh --checkpoint "$RUN/final.pt" --mode evaluate \
  --device cuda:0 --episodes 1 --output "$RUN/final-standalone-eval.json"
if test -f "$RUN/best.pt"; then
  bash eval.sh --checkpoint "$RUN/best.pt" --mode evaluate \
    --device cuda:0 --episodes 1 --output "$RUN/best-standalone-eval.json"
fi
```

远端断言应至少包含：

- 当前 A3.2 提前 0.15 秒时，`observation_size=1231`、`action_size=47`。
- 若保持 512 环境、183 帧整曲预填、210 向量步、更新比 .25，则采集 107,520 条、更新 3,456 次。一般情况从 `learning_start_transition_1_based` 和 `update_ratio_fraction` 计算，避免预填长度改变后沿用硬编码。
- 流水总结的 collected/updated-through 都等于实际结束步，pending 为 null，最大采样末端 Actor 版本差不超过单批更新预算（当前 128 次）。最终/周期/best checkpoint 同样排空；初始 best 的步数可为 0。
- checkpoint 的 agent update 数、metadata update 数和对应步数的预算一致，schema/kind 保持原定义。
- `best.pt` 仅在 `summary.best_checkpoint` 非空时要求存在。随机策略可能未满足键底资格，不能为测试强行放宽 best 门槛。
- 独立评价完整结束、无截断、诊断覆盖完整；独立回放 F1 允许物理数值误差，不要求 GPU 接触轨迹逐 bit 相同。

补充验证命令：`pytest -q tests/test_eval_source_config.py tests/test_collector_pipeline.py tests/test_torch_sac.py`，结果 **39 passed、2 skipped**（两项仍需真实双 GPU）。

### 火山云 Slurm 934 首轮稳态观察

2026-09-28 在两张 A100 的作业 934 中，实际双 GPU `test_collector_pipeline.py` 已通过全部 11 项。首个单卡短训完成 262,144 条采样、42,112 次更新，独立 `eval.sh` 重建/加载通过。以下流水数据读取自首个 `pipeline-0` 的指标，比较窗口均为第 190→510 向量步，即 163,840 条新增采样；避开首次编译和末尾评价。

| 项目 | 单卡 single-0 | 双卡 pipeline-0 |
| --- | ---: | ---: |
| 窗口墙时 | 120.191 s | 93.930 s |
| 稳态采样吞吐 | 1,363.17 条/s | 1,744.28 条/s |
| 相对单卡吞吐 | 1.000× | 1.280× |

从流水累计主机计时做同窗口差分，除以 320 个向量批：

| 每向量批的部分 | 平均耗时 |
| --- | ---: |
| 采样回调 | 89.221 ms |
| SAC 更新回调（128 次更新） | 291.033 ms |
| Actor 快照复制 | 0.489 ms |
| replay 提交及 reset | 1.751 ms |
| 流水 step 墙时 | 293.311 ms |
| 采样/学习回调重叠 | 89.221 ms |

该窗口采样回调耗时全部与学习回调重叠。它支持“采样已被学习阶段覆盖”的执行结论，仍是主机区间统计，不能替代 CUDA kernel 时间线。采样结束时最大 Actor 滞后为 128 次更新，符合一批上限。

**剩余瓶颈是单张 GPU 上的 SAC 更新。** Actor 复制只占流水 step 时间约 0.17%，replay 提交/reset 约 0.60%。即使将这两部分完全消除，该窗口的理想吞吐提升也仅约 0.77%；只继续加快已经被覆盖的物理采样不能显著提高端到端吞吐。因此暂不增加队列深度或更改 batch、更新比、环境数、精度来追求表面利用率。

后续保持学习设置的低风险检查可先比较 CPU 线程数和亲和性，并用分析器确认 learner 的 CPU 提交间隙、矩阵运算和图执行占比。现有脚本已启用物理/learner CUDA graph 和 Adam 执行布局；快照复制或 replay 传输的进一步改造应由其实际时间占比决定。要让另一张卡承担 SAC 计算，则需要独立设计和验证分布式 learner，不属于本次微调。

来源：作业目录 `logs/hpc/934/benchmark/results.json` 的 `single-0`，以及 `pipeline-0-20260928-190155-056814/metrics.jsonl` 第 190/510 步的累计计时差分。这是**首轮观察**，最终两轮交错顺序的均值、完整预算/检查点验证及最终选择以本次迁移报告为准，不用首轮值替代最终结果。

## TODO

- **V32-DUAL-01**：火山云两张实际分配 GPU 上执行真实跨卡测试、完整环境短训及保存/评估验证。
- **V32-DUAL-02**：测单卡、同步分卡和流水的稳定吞吐、显存、实际 GPU 重叠和质量；尚未证明两卡最大利用或端到端加速。
- **V32-DUAL-03**：若 learner 占绝大多数耗时，流水只能隐藏少量物理时间。需根据实测决定后续分布式 learner 或其他执行优化，不自动改 batch、环境数、更新比或精度来制造利用率。

修改位置：`rl_torch/train.py`、`rl_torch/collector_pipeline.py`、`tests/test_collector_pipeline.py`。参考当前训练主循环、replay 插入校验、`CUDALearner` 捕获/恢复逻辑和历史双卡方案；未使用或修改旧训练产物。
