# 控制范围短训汇总工具（2026-09-27）

`scripts/summarize_range_trials.py` 只读取显式指定的训练目录，不导入 Torch、不创建物理环境，不运行 GPU，也不修改原始 run。它用于同父检查点／同预算的控制限制与学习率短测，完整参数和对照差异仍以各 run 的 `config.json` 为准。

```bash
python3 scripts/summarize_range_trials.py \
  --runs training_mjlab/outputs/range_trial_baseline_20260927-20260927-010422-311301 \
  --output preview/outputs/diagnostics/range_trials_20260927
```

`--runs` 后可以依次列多个目录。加 `--plot` 生成 PNG/SVG，此时使用已安装 matplotlib 的训练虚拟环境 Python。输出目录不能是输入 run 本身，避免覆盖真实 `summary.json`。

## 输出与口径

- `summary.json`：完整规范化控制配置及解析合同、三学习率、父权重SHA、采样／更新预算、实际样本／更新量、耗时、全部评估、配置一致性比较。
- `evaluations.csv`：每次初评／预填期间／学习后评估，含最差音乐F1、microF1、准备错键、手碰撞帧／子步、目标跳变与速度、clip／slew、诊断覆盖率和采样频率。
- `runs.csv`：初评、**真正新增更新之后的完整评估峰值**、最新评估及确认的末评，保留这些时点的全部指标。`learned_improved_over_initial` 才回答新学习有没有超过继承初始策略。
- 可选 `range_trials.png/svg`：逐次F1、碰撞、准备误触、TX目标跳变与slew比例曲线。

step0 best、尚未更新的prefill评估都不能计为短训改善。恢复旧global update计数时先减初始计数；截断／未完成评估不能成为学习后峰值。没有 `summary.json` 时标记 `in_progress_or_interrupted`，不凭文件缺失宣称进程仍运行，也不把 latest 当已完成final。仅忽略仍在写入的末尾不完整JSON行，完整坏行报错。

`elapsed_s` 来自训练summary，不含源码入口准备；无summary时只另外报告最后训练日志推算时间，不能当完整耗时。稳态采样速度复用已验证的日志时间差方法，排除评估／存盘边界及首次学习构图。运动极值按日志注明的采样率解释，软件目标跳变不能当真实关节速度。

## 基线明细

现有 baseline：512环境×512向量步，262,144样本，42,752新增更新，三学习率均3e-4，未启用控制限制，耗时122.473秒。

| 向量步 | 是否发生新增学习 | 音乐F1 | micro F1 | 准备错键帧 | 手碰撞帧 | TX目标最大跳变m |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 初始继承 | 0.973568 | 0.970639 | 1 | 9 | 0.246307 |
| 128 | 仍在预填 | 0.973568 | 0.970639 | 1 | 9 | 0.246307 |
| 256 | 是 | 0.874834 | 0.893238 | 0 | 19 | 0.505596 |
| 384 | 是 | 0.931585 | 0.946996 | 0 | 17 | 0.494441 |
| 512 | 是 | 0.917300 | 0.927434 | 0 | 10 | 0.568840 |

更新后峰值在step384，比初始音乐F1低0.041983；保存的best仍是step0。这个基线没有证明新学习改善。

## 三组完整短测结果

三组均已完成，同父检查点SHA、seed42、scene/task及奖励合同、512环境×512向量步、262,144样本、42,752新增更新、batch1024×0.25更新/样本。都只迁移网络、新建Adam/replay/RNG，初始温度显式为父学得的0.001959640998393297；一整曲91,136条预填后开始学习。Critic／温度LR均3e-4，网络和其他SAC参数相同。

此次用于完整对照的三个目录，均位于 `training_mjlab/outputs/`：

- baseline：`range_trial_baseline_20260927-20260927-010422-311301`；
- wrist highLR：`range_trial_wrist_highlr_retry_20260927-20260927-012126-614681`；
- wrist lowLR：`range_trial_wrist_lowlr_20260927-20260927-011820-702559`。

两个wrist组使用同一控制合同：右TX目标范围[-0.2,0.2]m、左TX[-0.1,0.2]m；左右TX/TY目标变化率1.2m/s，左右Pitch4rad/s。20Hz下对应每帧最多6cm／0.2rad目标变化，未限制手指。限制后的previous-action观测与reset初值规则见 [控制合同说明](control_checkpoint_contract_20260927.md)。

| 配方 | Actor LR | 初评音乐F1 | step256 | step384 | step512／末评 | 学习后峰值 | 峰值−本组初評 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline，无限制 | 3e-4 | 0.973568 | 0.874834 | **0.931585** | 0.917300 | 0.931585 | −0.041983 |
| wrist highLR，完整重跑 | 3e-4 | 0.975256 | 0.929777 | **0.942917** | 0.925739 | 0.942917 | −0.032339 |
| wrist lowLR | 3e-5 | 0.975256 | 0.942194 | 0.903315 | **0.953798** | 0.953798 | −0.021459 |

**三组保存的 best 都是初评step0，新增学习均未超过初始。** 控制合同在不学习时就将音乐F1从0.973568变为0.975256、准备错键从1帧变0、手碰撞从9帧23子步变7帧18子步；这是控制路径改变带来的轨迹差异，不是新权重学会了更好的演奏。

| 末评／耗时 | baseline | wrist highLR | wrist lowLR |
| --- | ---: | ---: | ---: |
| 音乐 micro F1 | 0.927434 | 0.937063 | **0.963605** |
| 准备错键帧 | 0 | 0 | 0 |
| 手碰撞帧 | 10 | **5** | 8 |
| 手碰撞物理子步 | **18** | 32 | 23 |
| TX目标最大跳变m | 0.568840 | 0.060000 | 0.060000 |
| TY目标最大跳变m | 0.127192 | 0.060000 | 0.060000 |
| Pitch目标最大跳变rad | 0.382552 | 0.200000 | 0.200000 |
| 指关节目标最大跳变rad | **1.518736** | 2.024904 | 1.797545 |
| 20Hz采样最大指关节速度rad/s | **11.9293** | 13.1183 | 14.3832 |
| 软件范围clip占比 | 未启用 | 0.3745% | 0.3745% |
| slew占比 | 未启用 | 6.4607% | 5.4307% |
| 训练summary耗时秒 | 122.473 | 122.018 | 121.500 |
| 日志推算稳态采样条/秒，中位数 | 2389.9 | 2394.6 | 2399.0 |

clip／slew比例分母为6个受限腕通道×178帧，诊断覆盖完整；禁用基线不能填写0%。每项运动极值是对应评估的跨回合最大值，速度为20Hz采样。限制控制目标没有保证真实速度上界，手指也没有被限制。三组耗时接近，不据约1秒差异宣称低LR让训练执行更快。

### 低LR可作为下一轮候选的依据与限制

在这次固定预算内，低LR组的**末评与学习后峰值音乐F1、末评micro F1最高**，比同限制高LR末评音乐F1高0.028059；腕目标跳变也符合本次限制。这支持采用“相同腕限制＋Actor3e-5”继续验证，目的在于减少微调造成的演奏退化。

这不是所有指标都改善：低LR在step384低于另外两组，最终手碰撞帧多于高LR，碰撞子步多于无控制baseline，手指速度最高。三组都未超过自己的初评、未达到F1≥0.99／零碰撞，也只有一个种子；没有“无控制＋低LR”的第四格，不能完全拆分控制与LR的独立贡献。保持初始高F1 best，不以短测改善的名义覆盖它；继续训练需复评、保留best和完整控制合同。

### 中断组与重复性

`range_trial_wrist_highlr_20260927-20260927-011510-827921` 在step510被主任务记录为SIGTERM终止。文件证据是最新日志step510、261,120样本／42,496更新、缺少 `summary.json`，没有step512完整末评。它**不参与三组完整预算对照**，不能用最近一次step384当作final。原始数据保留，排除说明见输出目录 `excluded_run.json`。

相同种子和配置的完整highLR重跑也不是逐位重复：首次／重跑step256音乐F1为0.92968663／0.92977702，step384为0.90316463／0.94291745。这里保留两个实测结果；接触与GPU浮点训练中的小差异可以随更新扩散，不把固定seed称为逐bit训练保证，也不凭该差异自动判定工程错误。完整对照采用重跑的实际512步结果；该变异性进一步限制单次短测结论强度。

### 生成最终图表的命令

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
training_mjlab/.venv/bin/python scripts/summarize_range_trials.py --runs \
  training_mjlab/outputs/range_trial_baseline_20260927-20260927-010422-311301 \
  training_mjlab/outputs/range_trial_wrist_highlr_retry_20260927-20260927-012126-614681 \
  training_mjlab/outputs/range_trial_wrist_lowlr_20260927-20260927-011820-702559 \
  --output preview/outputs/diagnostics/range_trials_20260927 --plot
```

结果：[完整JSON](../preview/outputs/diagnostics/range_trials_20260927/summary.json)、[逐次评估CSV](../preview/outputs/diagnostics/range_trials_20260927/evaluations.csv)、[每组摘要CSV](../preview/outputs/diagnostics/range_trials_20260927/runs.csv)、[对照图](../preview/outputs/diagnostics/range_trials_20260927/range_trials.png)。脚本还逐项列出父SHA、奖励SHA、物理合同、seed、计划／实际预算的相等性，便于检查对照是否公平。

## 验证与记录

`tests/test_range_trial_summary.py` 与已有 `tests/test_quality_trial_summary.py` 合计 **6 passed（0.02秒）**，覆盖继承计数、prefill／初始分数不计改善、截断峰值排除、运行中缺summary／部分末行、完整控制配置保存、CSV以及禁止覆盖输入目录。三组完成目录的JSON／CSV／PNG／SVG已生成，图已检查；修正了某组无该指标时各子图颜色不同的问题，使颜色始终对应同一组。命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
training_mjlab/.venv/bin/python -m pytest tests/test_range_trial_summary.py tests/test_quality_trial_summary.py -q
```

参考：`scripts/summarize_quality_trials.py` 的可靠日志读取与时间推算、baseline `config/metrics/summary/best.json`、`rl_torch/train.py` 现有评估字段。只新增脚本、测试和本文，正式训练包保持冻结。主任务统一合并根／项目修改日志；目标F1≥0.99和无手碰撞仍待候选／长训验收。
