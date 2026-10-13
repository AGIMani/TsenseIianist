# 2026-09-28：SwanLab 指标上传与本轮时序分析

## 已完成内容

用户指定 SwanLab。`run.sh` 现在显式开启上传，默认个人空间私有项目 `TsenseI_A3.2`；直接调用 Python 入口默认关闭。上传进程独立读取原有 `metrics.jsonl`，没有在 GPU 采样或 SAC 更新循环加入网络调用，没有改变奖励、学习率、模型、控制频率或更新预算。

本轮已完成训练的全部历史已补传：

- [SwanLab 本轮实验](https://swanlab.cn/@blackswan17/TsenseI_A3.2/runs/qgbeil7c)
- 本地目录：`training_mjlab/outputs/TsenseI_A3.2_anticipatory_20260928/run-20260928-105754-540954`
- 原始 1129 条 JSONL 合并为 1022 个不同采样步，最后横轴为 5,000,192 条 transition；SDK 完成刷新并报告 37,384 条上传记录。
- 上传状态、URL、指标顺序及文件内容散列保存在 run 内 `swanlab_upload.json`。云端回读确认状态 FINISHED、项目 PRIVATE、137 条指标序列；全部 11 条主曲线和 2 条音乐段曲线逐点匹配本地（最大数值差 1.12e-16），前三指标各 99 点、奖励各 976 点。证据另存 `swanlab_cloud_verification.json`。

## 指标顺序与含义

主图位于 `01_metrics` 分组，数字前缀保证按名称排序仍保持用户指定顺序：

1. `metrics/01_F1`
2. `metrics/02_recall`
3. `metrics/03_precision`
4. `energy_reward`
5. `false_press_penalty`
6. `fingering_reward`
7. `forearm_reward`
8. `humanoid_finger_gesture_reward`
9. `key_press_reward`
10. `self_collision_reward`
11. `sustain_reward`

前三项均来自原评估的全回合逐帧平均指标，包含准备帧，不能把 F1 换成音乐段值却继续配全回合 recall/precision。由于逐帧分别平均，平均 F1 不一定等于平均 precision/recall 的调和平均。

`evaluation/` 单独上传音乐段 F1/最低 F1、20 Hz 和 200 Hz 音乐 micro-F1/recall/precision、准备误触、碰撞和限速统计。最佳模型仍按原 `music_only_f1_min` 选择；本次主 F1 曲线与 best 选择值不是同一个口径。`train/` 记录损失、温度、熵等，`training/` 记录样本数、更新数、速度等，`train_episodes/` 保留训练回合统计。主奖励曲线是已加权项，不再重复乘权重；原始奖励另列。

缺失值和非有限值省略，不补零或沿用旧值。相同 transition 的训练、回合和评估数据合并后只提交一次，最后一个采样步在进程结束时刷新。实时图最多延迟一个本地日志采样步。

## 使用方法

独立监控环境已在本机安装 SwanLab 0.10.1。独立复制到其他机器后执行：

```bash
bash scripts/setup_swanlab.sh
```

这只创建 `training_mjlab/monitoring/.venv`，不修改 Composer 或 GPU learner 的依赖环境。

```bash
# run.sh 默认在线上传；run_tsensei_a32.sh 也继承此行为
bash run.sh --swanlab-run-name my-experiment

# 本轮关闭上传
bash run.sh --no-swanlab

# 指定项目／团队空间
bash run.sh --swanlab-project my-project --swanlab-workspace my-team

# 离线记录，暂不访问云端
bash run.sh --swanlab-mode offline

# 为已有训练补传，不启动训练
training_mjlab/monitoring/.venv/bin/python scripts/swanlab_upload.py \
  --run-dir training_mjlab/outputs/<actual-run-directory> \
  --swanlab-project TsenseI_A3.2
```

全部新增参数也写入 `run.sh` 文件头：`--swanlab/--no-swanlab`、`--swanlab-project`、`--swanlab-workspace`、`--swanlab-run-name`、`--swanlab-mode`、`--swanlab-key-file`、`--swanlab-python`。补传脚本额外提供 `--force` 创建新的上传实验；相同项目／workspace／配置／完整日志内容已经成功上传时，默认跳过，避免重复实验。失败后的补传创建新云端实验，不伪称续接原实验。

## 密钥、配置和失败处理

本机密钥存储在仓库外 `~/.config/tsensei-pianist/swanlab.key`，权限 0600；也可通过 `SWANLAB_API_KEY` 环境变量提供，环境变量优先。密钥不作为 CLI 明文参数，不进入源码、训练 config、checkpoint 或本说明。

只上传白名单超参数和数值指标：SAC、网络尺寸、种子、环境时序、准备和奖励配置等。没有上传完整源配方、源码、外部乐谱或本地路径。SDK 的终端、源码/Git、依赖及机器探针关闭。后续若希望附加视频或模型，需要单独扩展；当前只上传本次要求的指标和超参数。

上传进程的日志是输出目录旁 `<run-directory>.swanlab.log`；上传网络或认证失败保留本地训练结果，不把训练判为失败。训练退出后最多等待上传进程 30 秒刷新，然后结束上传进程并保留补传来源。训练本身的非零退出状态照常传回。正常实时上传一般只有末步需刷新；本轮完整历史补传耗时约 26 秒。尚未对带在线上传的新 5M 训练测量吞吐，因此不宣称开销绝对为零。

上传记录横轴采用原始采样数；历史补传的云端记录时间是上传时间，不能用它倒推出原训练耗时。真实训练用时看原 `summary.json`，实时速度另列 `training/loop_transitions_per_second`。

## 本轮训练分析与下一步

完整逐音报告及曲线位于上述 run 的 `analysis_timing_20260928/`；奖励机制审计见 [时序奖励审计](tsensei_a32_timing_reward_review_20260928.md)。

训练 5,000,192 样本，耗时 53.35 分钟。best 第 9400 步音乐 F1 平均／最低均 0.978843；final 平均降至 0.948955、最低 0.946775，总回报却从 551.8056 略升至 551.9679。奖励更高不能保证时序更准确。

导出 best 视频这条独立回放有 18 个漏按键帧：14 个量化目标末帧、2 个迟按帧、一个短 D5/R2 整音对应的 2 帧。相对原 MIDI，仅 3 个命中音提前释放；其他“早松”主要相对于延长后的 20 Hz 网格目标，不能简单要求所有音延长。命中音相对原 MIDI 起音延迟中位 21.667 ms，D5/R2 与 G5/R5 约 64 ms；200 Hz FN 主要来自起音延迟。

优先候选为在原目标键正奖励预算内加入 200 Hz 持续正确按压信号：先对照端点／子步连续键程混合，再比较加入实际激活占用率的版本，以区分时序和浅按阈值两种作用。当前正确键正奖励仍看 20 Hz 末端，且白键刚低于激活阈值仍能拿约 94.12% 连续目标分。该候选比继续增大 humanoid 更直接针对测到的问题，但尚未训练验证。

建议同一 best 分支、同种子和预算、三学习率保持 3e-4，分别测试新奖励与已有提前量 0.15/0.20 秒；不要同时改腕速或阈值。除正式原 F1，还检查原 MIDI 起落延迟、200 Hz micro-F1、错键和碰撞。该段记录当时的分析建议，后续已按用户要求实施[正确键奖励修复](target_key_reward_fix_20260928.md)。2026-09-28用户已取消论文比对和对齐确认要求，后续按项目目标、实测与任务授权推进；本次上传接入本身未更改训练算法。

## 验证与保留问题

- 29 项单元测试通过：10 项指标适配测试覆盖排序、统计口径、权重、非有限值、白名单和秘密排除；19 项上传与入口测试覆盖并发半行、合并/末步刷新、失败隔离、路径/去重、learner退出状态和不污染源码配方。
- 完整真实历史通过 SwanLab SDK 离线测试及在线补传；正式补传没有改变原 metrics/config/checkpoint。
- `bash -n` 与入口 `--help` 通过；上传进程独立于 learner 的执行和失败处理另有测试。
- 本次没有调用 RL 优化器，不把日志单元测试称为新的训练效果验收。
- `V32-TIMING-01/02/03`：奖励候选待决定和实测、MIDI/grid边界差异、短 D5 缺少逐子步接触因果记录。
- `V32-A32-DIAG-01`：新掌壳 base_link 未完整进入现有键底诊断；不能由现有键底0断言全手不卡键。

## 参考

- 本轮 config.json、metrics.jsonl、summary.json、best.pt/best.json、训练 source.tar.gz、best 视频 rollout/audio_substeps。
- `scripts/train_from_source.py`、`rl_torch/train.py`、`mjlab_env/rewards.py`、`task_logic.py`、`fingering_experiments.py`、论文决策笔记。
- 官方 [SwanLab Settings](https://docs.swanlab.cn/api/py-settings.html)、[log](https://docs.swanlab.cn/api/py-log.html)、[环境变量](https://docs.swanlab.cn/api/environment-variable.html)、[数据读取 API](https://docs.swanlab.cn/en/api/py-api.html)，及安装的 0.10.1 SDK 接口。
