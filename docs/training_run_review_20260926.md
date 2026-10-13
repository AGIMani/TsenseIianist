# 小星星 policy preparation 5M 训练审计（2026-09-26）

## 结论与范围

审计训练目录：`training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894`。

**优先保留并从 `step-8500.pt` 微调。** 它是历史日志中最好的已保存检查点，三次原评估路径的曲目 F1 为 **0.97209、0.96745、0.97209**。`final.pt` 三次都是 **0.93513**。最高日志值出现在 step 9560，曲目 F1 **0.97405**，但当时没有保存权重；不能从日志恢复它，也不能将 step 9500 当成它。

本次训练和复评都没有达到用户的曲目 F1 ≥ 0.99 目标。20 Hz F1 也不能保证没有短促错音、手指自碰撞或突然的策略目标变化。以下三类指标必须同时看：曲目按键准确性、200 Hz 的实际按键／接触、关节运动与驱动目标变化。

本报告只分析已完成训练及其旧合同。后续奖励、优化器、批量大小或新训练由本轮主任务的设计及验收文档记录，不以本报告默示批准。

## 数据、源码和复现边界

- 历史曲线：全部 `metrics.jsonl`，489 次评估，以及原 `summary.json`、`config.json`。
- 权重：step 8500／9000／9500／final，各 **3 次标准评估 + 1 次物理子步诊断**，共 16 个完整回合，每回合 178 控制帧。
- 对应旧源码：`training_mjlab/source_versions/policy_preparation_acceleration_20260926.tar.gz`，SHA-256 为 `15215d0db8b19fd6f3aac1780c73711a370befa4acb75b9e757d8c021176f0c3`。
- 严格检查 **163 个包内文件**、依赖版本、config／checkpoint 合同、完整 preparation 合同，以及重建的 scene／task 指纹。源码仅解压到临时目录；没有将当前新增奖励的包替换到旧检查点。
- Scene manifest SHA-256：`5052ab74ad2a0c20baa94df1004ed7bef42be1e42cbaf217335e22e969fb8fb5`。
- Task SHA-256：`ede90d1265b7d657262091490a7ea0ec6add0376d4d1a50f7a867c18410c7a08`。
- 标准评估使用归档 `rl_torch/train.py::evaluate` 和原 GPU 物理图。子步诊断使用原场景已经提供的只读回调，以获取每个物理子步；与视频路径一样，它绕过控制周期物理图。两条路径分别报告，不能承诺接触轨迹逐位一致。

评估中的确定性策略关闭动作采样，但浮点接触计算仍可能分叉：8500 的三次结果确有约 0.00464 差异。三次复评适合筛选候选，不是统计置信保证。历史训练没有保存每一条随机训练动作，不能据评估录像声称重现了整个训练过程。

## 训练过程

### 实际预算

| 项目 | 实际记录 |
|---|---:|
| 环境数／向量步 | 512／9766 |
| 收集 transition | 5,000,192 |
| SAC 更新 | 4,995,193 |
| 初始随机动作 transition | 4999 |
| 完整回合／截断回合 | 27,648／0 |
| Replay 最终容量 | 1,000,000 |
| 训练计时 | 6321.96 秒，约 105.37 分钟 |
| 全程平均吞吐，包含训练器内评估等 | 约 790.9 transition/s |

每回合前 20 帧是空谱准备，后 158 帧是乐曲。策略从抬高 2 cm 的 initial 开始控制，支撑在 1 秒内下降；全部真实 transition 都进入学习。本训练使用 Shadow 五项原奖励：按键、踏板、能量、指法、forearm。**同手自碰撞奖励、false_press_penalty 和各速度／加速度附加项均关闭**；不能误把配置中保留的候选权重当成已启用。

### 学习曲线

| 向量步区间 | 曲目 F1 均值 | 最小／最大 | 发生准备误触的评估次数 |
|---|---:|---:|---:|
| 1–1000 | 0.1812 | 0.0000／0.4238 | 34／50 |
| 1001–3000 | 0.7071 | 0.3545／0.8956 | 0／100 |
| 3001–5000 | 0.9197 | 0.8301／0.9606 | 13／100 |
| 5001–7000 | 0.9326 | 0.8142／0.9701 | 31／100 |
| 7001–9000 | 0.9436 | 0.8691／0.9717 | 34／100 |
| 9001–9766 | 0.9469 | 0.8862／0.9741 | 13／39 |

首次曲目 F1 达到 0.8／0.9／0.95／0.97，分别在 step 2100／3100／3780／6020；未达到 0.99。后期窗口均值仍上升，但单次结果波动，final 比最高日志低约 **3.89 个百分点**。因此结论是“最终权重不是最好的权重”，不能说后半程一直退化。

准备阶段误触在中段消失后又出现，说明学会弹奏与保持准备安静并非自动绑定。原 F1 混入空谱准备帧，安静时这些帧得到满分；筛选曲目质量应优先使用 `music_only_f1`。

完整图表：[训练曲线](../preview/outputs/diagnostics/policy_training_5m_20260926/training_curves.png)，[489 次评估 CSV](../preview/outputs/diagnostics/policy_training_5m_20260926/evaluations.csv)。

## 检查点复评

| 检查点 | 三次标准曲目 F1 均值［范围］ | 子步诊断的 20 Hz 曲目 F1 | 20 Hz 错／漏键采样数 | 200 Hz 诊断 F1 |
|---|---:|---:|---:|---:|
| 8500 | **0.97054［0.96745–0.97209］** | 0.96745 | **1／17** | 0.91359 |
| 9000 | 0.96218［相同］ | 0.96218 | 5／20 | 0.90329 |
| 9500 | 0.95591［相同］ | 0.95591 | 5／17 | 0.90924 |
| final | 0.93513［相同］ | 0.93513 | 9／25 | 0.89162 |

这里“错／漏键采样数”以一个键在一个采样点为一项，不是音符事件数；一个帧可包含多个错键。200 Hz F1 将同一个 20 Hz 目标保持十个物理子步，用于定位物理误触与按下／释放时序，**不是更换后的正式训练评分**。按键起落的物理时间会使它系统性低于控制周期末 F1，不应直接套用同一个 0.99 验收门槛。

数据：[比较 JSON](../preview/outputs/diagnostics/policy_training_5m_20260926/repeated/comparison.json)、[每个错误控制帧](../preview/outputs/diagnostics/policy_training_5m_20260926/repeated/error_frames_20hz.csv)、[最佳与最终的按键错误图](../preview/outputs/diagnostics/policy_training_5m_20260926/repeated/best_final_key_errors.png)。

## 错音的具体来源

### 20 Hz 检查漏掉的短音

8500 的这次物理子步诊断中，准备阶段在控制周期末完全安静，20 Hz 准备 F1=1；但是 **policy t=0.965–0.980 秒出现 MIDI72 的 15 ms 实际激活**，比曲首早 35 ms。这解释了“日志显示准备没错，音频仍能听见先乱弹”的一种实际情况。

该回放曲目阶段另有 4 次在开始时不属于当时目标的 note-on：

| 音符 MIDI | 策略时间 | 曲目时间 | 激活时长 |
|---|---:|---:|---:|
| 79 | 1.030 s | 0.030 s | 15 ms |
| 71 | 1.650 s | 0.650 s | 30 ms |
| 62 | 4.730 s | 3.730 s | 5 ms |
| 77 | 7.705 s | 6.705 s | 20 ms |

200 Hz 的 139 个曲目错键采样还包含前一个正确音符晚释放约 15–30 ms 的情况，不能将这 139 项全部叫作 139 次乱弹。完整 note-on／note-off 与误触区间在各 `detail-0/report.json`。

### 四个权重共同漏掉的片段

四次详细回放都在曲目 **0.50–0.70 s** 漏掉 MIDI60（C4）。既有 `preview/outputs/diagnostics/startup_20260926/reset_and_score.npz` 的 158 帧 goal 与本轮 `rollout.npz` 去掉 20 帧准备后的 goal 完全一致；该旧诊断内的指法是：

- 曲目 0.50–0.65 s：右拇指 MIDI72（C5）、左拇指 MIDI60（C4）、左小指 MIDI48（C3）。
- 曲目 0.65–0.70 s：右拇指 C5、左拇指 C4。
- 没有同一帧将两个目标键分配给同一根手指。

这指出一个值得独立核查的左手拇指／小指八度跨度与短时转换问题。它不是“已经证明不可达”；也不能据此改动乐谱或指法。静态几何／关节范围可达性及动态按键可行性需分别验证，由本轮对应检查记录最终结论。

## 自碰撞与动作突然变化

### 接触

先检查 reset 本身：四个详细回放的初始 qpos／qvel／ctrl／观测完全相同，qvel 全零。另用**严格匹配的旧归档模型**加载 8500 的初始状态、双手 mocap 世界 Z 抬高 0.02 m，在未施加策略命令时执行 CPU `mj_forward`；同手、不同 body、`dist <= 0` 的接触 **左右均为 0**。所以没有发现“初始必然重叠导致零自碰撞回合先天不可达”的问题，不能据此放宽门槛。该检查只回答初态，不保证随后的下降和策略运动无接触。

对应 GPU 回放中，同手接触最早分别出现于 8500 的 0.320 s、9000 的 0.120 s、9500 的 0.375 s、final 的 0.260 s。证据：[初态数组一致性](../preview/outputs/diagnostics/policy_training_5m_20260926/repeated/initial_state_consistency.json)、[CPU reset 接触检查](../preview/outputs/diagnostics/octave_reachability_20260926/reset_contacts.json)。

| 检查点 | 准备／曲目同手接触子步 | 全回合同手接触时间并集 | 最大接触穿入深度 | 跨手接触子步 |
|---|---:|---:|---:|---:|
| 8500 | 103／329 | 2.160 s | 5.401 mm | 0 |
| 9000 | 175／322 | 2.485 s | 4.878 mm | 0 |
| 9500 | 103／302 | 2.025 s | 3.463 mm | 0 |
| final | 95／289 | 1.920 s | 3.022 mm | 0 |

接触按 MuJoCo 求解器报告的不同 body 的手部 geom 对、`dist <= 0` 记录。时间并集是在同一子步有任意同手接触就计一次。穿入深度是模型接触距离，不是传感器测量，也没有从它反推力或受伤风险。

8500 最长的几组同手接触：左无名指 PIP—左小指 PIP **0.545 s**；右无名指 swing—右小指 PIP **0.370 s**；右中指 PIP—右无名指 PIP **0.300 s**；右无名指 PIP—右小指 PIP **0.220 s**。最大穿入来自右拇指 PIP—右食指 DIP，约 **5.40 mm**。这些是具体的无关手指姿态问题，forearm 的掌部代理奖励没有覆盖全部同手指碰撞。

final 同手接触时间反而比 8500 短，但弹奏准确性明显更差；少碰撞与高 F1 需要分别验收。

### 运动与控制目标

| 检查点 | 实际 TX／TY 最大速度 | 手指最大角速度 | 相邻 TX 驱动目标最大差 |
|---|---:|---:|---:|
| 8500 | 0.396／0.358 m/s | 12.495 rad/s | 15.55 cm |
| 9000 | 0.354／0.440 m/s | 14.472 rad/s | 12.03 cm |
| 9500 | 0.436／0.340 m/s | 12.260 rad/s | 30.69 cm |
| final | 0.555／0.386 m/s | 15.395 rad/s | 27.85 cm |

“驱动目标差”是策略通过动作映射给位置驱动器的连续两次指令差，**不是手真实瞬移这么远**。物理状态连续求解，但剧烈改变目标仍会产生较快加减速。在 8500：

- 第一条策略命令相对 reset ctrl：TX 最大差 1.116 cm、TY 2.243 cm、Pitch 0.139 rad、手指 0.709 rad。
- 曲首 policy t=1.00 s 的新 TX 目标差最大 15.55 cm。
- TX／TY 实际速度峰也出现在 t=1.00 s 的采样终点；该状态来自此前控制帧，不能把它直接归因于随后 t=1.00 s 的新指令。
- Pitch 最大角速度 4.337 rad/s；手指最大驱动目标差 1.543 rad。

因此现有证据支持两条需要分别处理的来源：策略目标阶跃和实际同手接触。仅靠更换 reset、增加一秒准备或减小碰撞反力，都不能证明已消除它们。此次没有做接触禁用反事实实验，不能断言每一次抖动都是被碰撞弹回。

## 对后续工作的建议与待验收

1. 从 8500 保留已学弹奏技能；每次评估按 `music_only_f1` 保存 best，另留 final，避免丢失 9560 这类峰值。
2. 将短错音、自碰撞、无任务手指姿态和实际运动速度与 F1 一同记录，防止一个分数掩盖其他退化。已有五项奖励并不直接表达所有“人性化”要求。
3. 优先核查共同漏 C4 的短片段；静态可达不等于能在给定时间、碰撞和琴键反力下稳定执行。
4. **V32-START-01**：新长期训练仍需验证准备到曲首的接管、首命令变化及完整录像；本次最佳旧权重仍有目标阶跃。
5. **V32-START-02**：当前旧合同存在 20 Hz 末态评分未采到的短误触。是否改变奖励采样是新设计，不能把本报告的 200 Hz 诊断偷偷当成已修改的训练规则。
6. **V32-SAC-03／V32-06**：GPU 接触重复回放并非逐位一致；严格源码合同保证模型／设置一致，不保证所有浮点轨迹一致。
7. 本审计没有新增最佳／最终视频；已有完整 NPZ 与事件记录可复现，视频需在 GPU 空档使用旧归档入口单独导出。达到 0.99、减少自碰撞和主观动作自然都尚未由新长期训练验收。

## 工具、参考及实际验证

新增诊断脚本：

- `scripts/analyze_training_run.py`：CPU 解析曲线、所有已保存权重的历史排名、准备误触和图表。
- `scripts/audit_checkpoint_trajectories.py`：严格归档重建、标准重复评估、200 Hz 实际物理状态／接触／音符／目标跳变记录。
- `scripts/summarize_checkpoint_audit.py`：已有轨迹的 CPU 汇总、逐帧错漏键 CSV 与对照图。

参考：本项目论文决策笔记 `docs/references/biomimetics_2026_610_notes.md`；原 run 的三个 JSON/JSONL 文件；归档中的 `rl_torch/train.py::evaluate`、`rl_torch/checkpoint.py::load_agent/reward_options`、`mjlab_env/task.py`、`mjlab_env/scene.py`、`preview_audio.py::events_from_states`；现有 `scripts/eval_from_source_archive.py` 严格解包与验证工具。本审计没有采用外部参考代码作为依赖，也没有修改奖励、SAC 或环境核心。

在独立项目根目录执行：

```bash
training_mjlab/.venv/bin/python scripts/analyze_training_run.py \
  --run training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894 \
  --output preview/outputs/diagnostics/policy_training_5m_20260926

training_mjlab/.venv/bin/python scripts/audit_checkpoint_trajectories.py \
  --run training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894 \
  --source-archive training_mjlab/source_versions/policy_preparation_acceleration_20260926.tar.gz \
  --output preview/outputs/diagnostics/policy_training_5m_20260926/repeated

training_mjlab/.venv/bin/python scripts/summarize_checkpoint_audit.py \
  --audit preview/outputs/diagnostics/policy_training_5m_20260926/repeated
```

重复 GPU 诊断需选择**新的输出目录**，避免覆盖既有证据。16 次回放全部完成；各条物理轨迹都检查每第十个子步的按键与 20 Hz 状态完全一致。脚本语法检查通过；图表已实际生成并查看。最初误用 Composer 的 `training/.venv` 运行绘图，遇到缺少 matplotlib，改用已有 learner `training_mjlab/.venv` 后成功，没有为此改依赖或环境。日志与 NPZ 均保存在 `preview/outputs/diagnostics/policy_training_5m_20260926/`。

## 同轮补充：新 best／final 的 config 回放校验

后续 A–D 微调试验新增了自动选择 best 的检查点元数据。`scripts/eval_from_source.py::validate_config` 原先只允许 `vector_step`、`updates` 不出现在启动时的 config 中，因而会误拒绝包含 `evaluation_selection_state`、`last_evaluation`、`last_evaluation_vector_step` 的新检查点。

仅在该入口加入上述 **3 个确切运行时字段**的名单；其余 metadata 字段仍必须在完整 config 中存在，所有 config 值仍与 checkpoint 逐项相等比较。即使名单字段出现在 config 中，也必须值相同。未修改包内训练／环境源码、源散列或模型合同。

新增 `tests/test_eval_source_config.py`，覆盖 best／periodic／final、尚无评估、旧检查点兼容、未知字段、源／任务／场景／奖励／SAC／评估配置差异及缺项拒绝。执行 `CUDA_VISIBLE_DEVICES='' training_mjlab/.venv/bin/python -m pytest -q tests/test_eval_source_config.py tests/test_eval_source_archive.py`，**26 项通过**。另 CPU 读取同轮 A–D 四次已完成试验的 best 与 final，共 **8 个真实检查点**，完整 config 校验通过；逐个注入未知字段及源散列差异均拒绝。结果见 [config 回放校验记录](../preview/outputs/diagnostics/policy_training_5m_20260926/config_replay_acceptance.json)。这是 CPU 入口兼容性验证，不代替新训练的完整视频验收。

只读检查确认 `preview_mjlab.py` 和 `render_mjlab_checkpoint.py` 没有另一份同类完整 config 字段集合比较。`eval.sh` 的 preview／render 均经过修正后的公共入口；之后的 `for_checkpoint`、`load_agent` 继续执行源、scene/task、维度和准备合同校验。新训练源码归档应包含修正后的入口脚本；旧归档仍按各自旧版本使用，没有回写旧归档。
