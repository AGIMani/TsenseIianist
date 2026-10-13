# V3.2 Torch SAC＋GPU replay

> 2026-09-28 工作约定更新：用户已取消强制论文阅读、逐项比对和因论文内容重合而征询对齐决定的要求，后续训练不以论文参数为默认依据。下文表格、命令和验收结果是历史记录；新训练使用 [run.sh](../run.sh)，实际参数以启动命令、完整 `config.json` 和实测结果为准。历史 preset、旧检查点兼容及未知模型映射校验继续保留。

> 2026-09-24更新：新训练改用源码＋完整config.json重建，不保存持久化scene/task缓存；Actor256×3、温度1、gamma0.8、曲首准备1秒，指法保持原曲。见[source_config_training_20260924.md](source_config_training_20260924.md)。下文保留历史方案及验收。

更新：2026-09-17。**V32-GPU-04 / P3 单卡功能已实现并通过短程验收。** 现在可以在 MJLab GPU 任务上采样、写入 CUDA replay、更新 Torch SAC、独立评估、保存／加载检查点和预览策略。尚未完成正式曲目长训练、学习质量验收或双卡训练。

## 1. 2026-09-17 用户决定与验收参数（历史）

2026-09-17 用户在看到当时 JAX 配置与论文 pp.8–10／表2的差异后，选择“帮我设置为论文的参数”。因此当轮 Torch 验收采用下表，原 JAX 实现及其历史默认配置保留为数学参考。该选择记录于 [历史论文登记](references/biomimetics_2026_610_notes.md)，不代表当前主线参数或后续强制对齐要求。

| 设置 | Stage1 | Stage2 参数档位 |
| --- | --- | --- |
| Actor | 256×2，GELU，tanh Gaussian | 同左 |
| 双 critic | 256×3，Dropout 0.01 → LayerNorm → GELU | 同左 |
| actor / critic / alpha 学习率 | 3e−4 / 1e−3 / 1e−3 | 1e−4 / 1e−3 / 1e−3 |
| gamma | 0.85 | 0.90 |
| tau | 0.005 | 0.005 |
| alpha 初值 | 0.2 | 0.2 |
| 目标熵 | −0.5×51＝−25.5 | 同左 |
| 并行环境 | 256 | 256 |
| batch | 4096 | 4096 |
| replay | 1024向量槽＝262144条transition | 同左，重新建立 |
| 随机动作预热 | 300向量步 | 0 |
| 开始学习 | 第1000向量步，且replay至少有一个batch | 同左 |
| 每轮更新 | 每个符合条件的向量步，32次完整SAC更新 | 同左 |
| 默认运行预算 | 20000向量步＝5120000条transition | 同左 |
| 独立确定性评价 | 每1000向量步；结束时若未到间隔，再评价一次 | 同左 |
| 梯度裁剪 | 关闭 | 关闭 |

**这是论文参数适配，不是整篇复现。** 双手51动作、1180观测、20 Hz、当前奖励和discount语义继续采用已确认的任务包；没有切成论文单手21动作、2106观测、40 Hz或乘dt奖励。论文未完整披露critic层数、actor对双Q聚合、温度损失等细节，本轮保留现有三层critic、actor Q均值及现有温度损失。LayerNorm epsilon=1e−6，GELU使用tanh近似，Adam eps=1e−8，保留原Flax的运算顺序。

Stage2入口**只提供论文的参数档位与网络转移规则**：恢复Actor、在线critic和target critic，重新建立optimizer、replay、物理及乐谱状态，alpha按表2初始化为0.2。没有擅自启用论文Stage2奖励权重、正功率、速度／加速度或超速惩罚，故尚不具备论文完整的Motion Refinement课程。相关缺口继续由V32-PAPER-09/U01/U06跟踪。

## 2. 文件与数据流

| 文件 | 职责 |
| --- | --- |
| `tsensei_v3_2_pianist/rl_torch/sac.py` | 网络、稳定tanh log-prob、SAC损失、优化器、独立随机流；`SACConfig.paper(stage)`定义算法参数 |
| `rl_torch/replay.py`（同上包内） | CUDA批量环形写入、均匀有放回采样、结束标记与环境discount；也支持CPU规则测试 |
| `rl_torch/train.py` | 单GPU训练、向量步／样本／更新计数、部分reset、独立评估、周期保存；`parser()`定义运行预算 |
| `rl_torch/checkpoint.py` | 原子写入learner检查点，严格匹配场景与任务散列 |
| `rl_torch/evaluate.py` | 单独加载检查点，在独立GPU环境完整评价 |
| `scripts/export_torch_sac_reference.py` | 用真实本地JAX SAC生成受控随机输入下的参考更新 |
| `scripts/validate_torch_sac.py` | 对照动作、log-prob、TD target、损失、梯度、Adam矩及更新后的网络／温度 |
| `preview/scripts/preview_mjlab.py --checkpoint` | 确定性策略播放，同时保留指法颜色与绿色实际激活键 |

训练数据流：`obs → actor → normalized action → PianoTaskEnv.step → replay.insert → reset(done_ids) → replay.sample → SAC.update`。所有训练张量在同一张GPU上，没有逐环境NumPy/JAX转换；日志、检查点与预览允许低频主机拷贝。当前合法性／非有限值检查有批次级同步，未实现torch.compile、任务CUDA graph或异步多卡重叠。

Replay保存 `state/action/reward/discount/next_state/terminated/truncated`。**先插入结束帧，再reset**；采样结果独立持有内存，环形覆盖不会改写已取出的batch。可处理跨环尾、超容量批次（保留最后capacity条）、空批次，非法批次在修改缓冲区前报错。采样无优先级、无n-step。

2026-09-24 后的新 task 将 reset 观测中的上一动作恢复为 scene 实际中立控制；训练配置用 `reset_contract` 记录该语义。结束 transition、局部 `done_ids` reset 和未结束世界保持不变。旧 task/checkpoint 仍使用旧动作下限历史，不能 warm-start 到新 task；详见[并行环境 reset 修复](reset_recovery_20260924.md)。

SAC每次更新顺序为 **critic → target critic软更新 → actor → alpha**。target用双Q最小值，actor用双Q均值。目标为 `reward + gamma * environment_discount * (min_target_Q - alpha * next_log_prob)`；不额外乘`1-done`，保留曲目结束discount=1的已确认规则。alpha损失为 `alpha * (entropy - target_entropy)`，entropy停止梯度。

### 向量步与内存

一个向量步推进N个环境，产生N条transition；每轮32次更新不会随N自动改变。默认N=256时 `rho=32/256=0.125`、采样比`R=32×4096/256=512`。若只改成N=64而保留32和4096，则rho=0.5、R=2048，已经改变学习预算；有效配置会明确打印这些数值。运行时和论文的控制频率不同，不能用相同向量步数声称相同物理时间或学习效率。

默认D=1180/A=51时，262144条FP32数据＋两个bool标记的replay实际分配 **2,530,738,176字节**，另需网络、优化器、batch和Warp物理内存。报告的Torch显存不含全部Warp分配，不能当整卡总显存。

## 3. 启动与验收命令

以下命令均在独立项目根目录运行。使用已有 `training_mjlab/.venv`；新机器先按 [场景安装说明](mjlab_scene_acceptance.md) 建环境、导出场景，再按 [任务说明](mjlab_task_acceptance.md) 导出带指法任务。无需把JAX安装进GPU训练环境。

### 查看论文默认配置，不启动物理

```bash
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train --inspect-only
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train --stage 2 --inspect-only
```

### 短程验收

下面**主动缩小预算**检查整条链路，保留论文网络与优化器参数；不是正式论文训练。输出目录必须不存在，可改一个新名称：

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --task training_mjlab/outputs/task_toy \
  --num-envs 16 --vector-steps 48 --replay-vector-slots 32 --batch-size 256 \
  --warmup-vector-steps 8 --learning-starts 16 --gradient-steps 2 \
  --eval-interval 24 --log-interval 8 --save-interval 24 \
  --output-dir training_mjlab/outputs/my_sac_smoke
```

通过标准：退出0，`summary.json`为completed；768条插入、66次更新、32个完整episode、replay最终512条；loss／alpha有限；`metrics.jsonl`有独立评价，`final.pt`可加载。本轮首次训练内确定性评价键F1为0，后续独立两回合平均约0.0147，不能视为学会弹琴；预期改善须另做正式训练评估。

### 默认论文参数启动

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --task training_mjlab/outputs/task_twinkle \
  --output-dir training_mjlab/outputs/my_stage1
```

该命令使用默认256环境、4096batch、32更新、20000向量步。本次没有运行整个预算。用户自己的曲目须先导出带指法任务，将`--task`改为相应目录即可；场景与任务散列不符会报错。

### 评价与可视检查

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.evaluate \
  --task training_mjlab/outputs/task_toy \
  --checkpoint training_mjlab/outputs/my_sac_smoke/final.pt \
  --output training_mjlab/outputs/my_sac_smoke/evaluation.json

# 桌面终端：空格开始/暂停，R重置，再空格重播
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --task training_mjlab/outputs/task_toy \
  --checkpoint training_mjlab/outputs/my_sac_smoke/final.pt --paused
```

播放使用`tanh(mean)`，策略控制全部关节及踏板，Control滑块不参与策略动作；去掉`--checkpoint`恢复手动预览。`--checkpoint`要求提供匹配的`--task`，不与`--demo/--bottom-out`合用。窗口指法仍表示指定目标，绿色表示真实键激活。单独评价默认跑完整乐谱，训练时的可选时间上限不自动带入这个整曲评价。

评价使用帧级Precision/Recall/F1，再按episode平均；全局`key_frame_micro_f1`另按所有评价episode的TP/FP/FN汇总，保持两种口径的区别。确定性评价不消耗训练动作／更新随机流，也不复位训练环境。

## 4. 保存、恢复与Stage2

输出包括 `config.json`（论文配置、实际覆盖、版本、源／任务／场景散列）、`metrics.jsonl`、`summary.json`、周期 `step-N.pt` 和 `final.pt`。检查点保存actor、critic、target、alpha、三个Adam状态、动作／更新／探索RNG和累计learner更新数。以下两种加载明确不同：

```bash
# 同阶段warm restart：保留learner/optimizer/RNG，重建replay和环境
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --task training_mjlab/outputs/task_twinkle \
  --warm-start training_mjlab/outputs/my_stage1/final.pt \
  --output-dir training_mjlab/outputs/my_stage1_restart

# Stage2参数档位：加载Stage1网络，采用新学习率/gamma，清空optimizer/replay，alpha=0.2
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --stage 2 --task training_mjlab/outputs/task_twinkle \
  --stage2-from training_mjlab/outputs/my_stage1/final.pt \
  --output-dir training_mjlab/outputs/my_stage2_parameters
```

warm restart会重新计数本轮向量步、预热和学习起点，更新总数另外保留；不是精确续训。检查点不包含replay、物理状态、乐谱／历史、采样调度与全部模拟器状态。Stage2参数档位不能称为已经实施论文的奖励课程。两种加载都核验任务与场景；不能把Toy检查点直接用于不同散列的Twinkle、另一个姿态或不同 reset 动作历史合同。

## 5. 已完成的验证

原始结果与散列汇总：[Torch SAC验证记录](torch_sac_validation.json)。

- **JAX数值对照**：实际本地 `rl/sac.py` 的三个更新函数；仅将随机高斯采样与Dropout注入固定输入，没有替换损失。两层actor／三层critic的小网络，dropout=0／0.01各连续5次更新，Torch CPU和CUDA各1870检查通过。覆盖动作、log-prob、TD target、各损失／梯度、Adam一二阶矩、网络／target／温度。CUDA参数最大差2.3842e−7，梯度2.6227e−6。掩码在本对照样例中给双Q相同输入；正式训练对各Q独立抽样。不声称JAX与Torch相同seed产生相同随机序列。
- **规则测试**：5项unittest覆盖CPU/CUDA replay环绕、超容量与所有权、非法批次不写入、末帧discount、饱和tanh、检查点下一步更新一致、Stage2清空优化器／alpha、评价RNG独立与论文预算。
- **16环境完整链路**：48向量步／768采样／66更新，32曲目结束，回放容量512并发生环绕，独立评价与周期／最终保存通过。
- **论文规模短测**：256环境、batch4096、32更新／轮、262144容量；17向量步／4352采样／64次更新通过。缩短预热与学习起点，不代表完整20000步实验；此短测尚未到21帧曲末，结束路径由16环境样例覆盖。
- **恢复与转移**：warm restart 8向量步／128采样，继承66更新后新增10，总76；显式4步上限触发32个截断。Stage2网络转移24向量步／384采样／42新更新、16曲目结束，通过独立评价与保存。
- **独立评价与桌面播放**：实际检查点加载、整曲确定性评价和保留指法叠加的窗口播放，详见汇总记录。短训练质量不足属于后续学习验收，不把低loss当作演奏效果。

重跑规则／数值验收：

```bash
training_mjlab/.venv/bin/python -m unittest discover -s tests -p test_torch_sac.py -v
training_mjlab/.venv/bin/python scripts/validate_torch_sac.py
# 仅需重新生成JAX参考时执行；会覆盖该测试夹具
JAX_PLATFORMS=cpu training/.venv/bin/python scripts/export_torch_sac_reference.py
```

## 6. 下一步

- **V32-GPU-05**：在单卡正式曲目训练与性能测量基础上，再做采样／学习双卡分工、传输和更新预算基准；远程A100未验收。
- **V32-GPU-06 / V32-SAC-01**：使用用户正式指法曲目，确定演奏质量目标并进行长训练、多seed评价。当前默认Twinkle可作基线，但短测不证明收敛。
- **V32-SAC-02**：如需精确中断恢复，补齐replay、模拟器、乐谱、历史与调度状态；现在仅warm restart。
- **V32-PAPER-09/U01/U06（历史资料缺口）**：原文未完整披露奖励课程或 critic 结构；保留此记录，不将复现论文列为后续训练要求，也不以此阻塞项目自身的工程改进。

V32-04／05／06／07／10的硬件参数、接触／指尖及正式右手限制继续存在。当前工作完成了单卡算法与数据闭环，不替代这些模型标定或演奏质量验收。
