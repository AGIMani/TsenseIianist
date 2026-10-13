# 2026-09-27：最新腕限速续训的曲首卡键与全指净空审计

## 结论

最新 `twinkle_range_finetune-20260927-012613-20260927-012614-462007` 已完成。`best.pt` 在 step 2600 的音乐 F1 为 **0.98227856**，最终 step 4096 为 **0.96856545**。严格旧源码重建后各复评三次完全一致；没有达到 0.99。最佳策略仍将左拇指从琴面下方送入键底，腕目标限速没有根治这个问题。

原始姿态不是埋在琴里：origin 双拇指完整碰撞表面最低 Z=34.319 mm，白键顶 22.5 mm，净空 **11.819 mm**；initial 再增加 20 mm。十根手指对其 XY 范围内琴键的保守净空均为正。单看全局黑键 35 mm 高度会误报拇指低 0.681 mm，因为 origin 拇指不与黑键 XY 范围重叠。

## 来源与复现

- 训练结果：上述 run 的 `summary.json`、`best.json`、`metrics.jsonl`、`config.json`。`latest_range_launch.json` 是启动快照，里面的 step 0 分数不能当作最终结果。
- 旧源码：`training_mjlab/source_versions/control_limits_finetune_20260927.tar.gz`，SHA-256 `721816a28666c084d92c936f736f782504365a9746dc13f4f336f62a106fb846`；**164 个源码文件逐项匹配** checkpoint 合同。没有绕过源码、任务、模型、准备或控制合同校验。
- 规则参照：`docs/references/biomimetics_2026_610_notes.md`、旧归档 `mjlab_env/rewards.py::gesture_protection_mask`、`mjlab_env/bundle.py` 的 origin/reset、`rl_torch/train.py::evaluate`。本文为实测诊断，不自行改变论文设计。
- 工具：`scripts/audit_checkpoint_trajectories.py`、`scripts/analyze_motion_contacts.py`、新增 `scripts/analyze_finger_clearance.py`。只修改独立诊断脚本，未改正式包。
- 输出：`preview/outputs/diagnostics/quality_clearance_20260927/repeated/`。包括源合同、三次 canonical 评估、每个 checkpoint 独立详细回放的 178 控制帧／1780 物理子步、接触点/法向/关节目标/实际关节/力矩、原始与抬高姿态几何。

```bash
training_mjlab/.venv/bin/python scripts/audit_checkpoint_trajectories.py \
  --run training_mjlab/outputs/twinkle_range_finetune-20260927-012613-20260927-012614-462007 \
  --source-archive training_mjlab/source_versions/control_limits_finetune_20260927.tar.gz \
  --output preview/outputs/diagnostics/quality_clearance_20260927/repeated \
  --checkpoints best.pt final.pt --repeats 3 --detail-repeats 1 --extended
training_mjlab/.venv/bin/python scripts/analyze_motion_contacts.py \
  --audit preview/outputs/diagnostics/quality_clearance_20260927/repeated
training_mjlab/.venv/bin/python scripts/analyze_finger_clearance.py \
  --audit preview/outputs/diagnostics/quality_clearance_20260927/repeated --plot
```

第一次命令要求新输出目录，避免覆盖证据。后两个命令只读取已记录的数据并重算分析，不占 GPU。

## 训练是否继续改善

总计 2,097,152 条 transition、501,504 次更新、11,776 个正常完成回合、0 截断，耗时 1163.01 秒。完整 42 次评估表在 `training_history.json`。

| 向量步 | 音乐 F1 最小值 | 同手/跨手碰撞子步 |
|---:|---:|---:|
| 0 | 0.975256 | 18 |
| 400 | 0.936438 | 0 |
| 800 | 0.951477 | 0 |
| 1600 | 0.978903 | 7 |
| 2600（best） | **0.982279** | 10 |
| 3200 | 0.969831 | 8 |
| 4096（final） | 0.968565 | 6 |

分数先下降、恢复后又退化；不能假设多训同一个目标一定趋于 0.99。碰撞数更低也不等于弹奏更好或没有键底碰撞：这里的训练碰撞诊断主要统计手—手，键底另行分析。

## 最佳策略确实卡在哪里

以下时间从 policy 接管 initial 的时刻起算，音乐从 1.0 秒开始。

1. **0.90 秒：**左拇指指尖 Z≈−31.42 mm，整指最低 Z≈−31.43 mm，远低于白键顶；当时还在琴键 XY 外，所以没有琴键接触。
2. **1.09 秒：**左拇指末端与 G3（MIDI55）键底接触 1 个物理子步。
3. **1.16–1.365 秒、1.375–1.405 秒：**左拇指 PIP 碰撞几何与 F3（MIDI53）键底接触，共 49 个物理子步；局部外法向 Z 在 −1 到 −0.936 之间，明确是下表面，不是正常顶面触键。最大该键底接触穿透 0.948 mm。
4. **1.25 秒附近：**Rot2 请求约 0.056 rad，实际约 0.843 rad，执行器达到 −0.6 N·m 限幅，约束力矩反向；不是单纯播放插帧。整段存在大跟踪误差，但关节并非整个 0.25 秒完全静止。
5. **1.5 秒开始需要左拇指按 C4：**C4 在音乐 0.50/0.55/0.60/0.65 秒四个控制帧都漏掉。受阻时间与之后缺键关联，但没有反事实实验，不能把全部缺键只归因为这一次接触。

图：`preview/outputs/diagnostics/quality_clearance_20260927/repeated/best/detail-0/opening_finger_clearance.png`。红色区间为真实键底接触，包含整指最低高度、指尖高度、用指状态和目标/实际/力矩。

| 详细回放指标 | best | final |
|---|---:|---:|
| 钢琴琴壳接触 | 0 | 0 |
| 手—键侧/底接触 | 0.385 s | 0.555 s |
| 明确键底接触 | **0.250 s** | 0.215 s |
| 非末端 link—键接触 | 0.090 s | 0.085 s |
| 同手碰撞 | 0.050 s | 0.030 s |
| 跨手碰撞 | 0 | 0 |

“非末端 link”按模型几何命名区分，不宣称等同真实皮肤上的指腹位置。各时间按类别去重子步，不能把不同类别相加。

## 卡住时是否是闲指

best 的 50 个键底子步全部发生在左拇指**当前不用**时，但其中 **30/50** 已进入未来 0.25 秒用指保护窗，**50/50** 都在 0.5 秒观测前瞻内。左拇指随后要按 C4。

因此只把“完全闲置的手指”拉回初始角度会留下缺口：准备中没有谱面、即将用指被豁免、已经漏键时旧“所有键正确才给姿态奖励”的门槛关闭，都可能缺少抬指信号。对未来用指也必须约束接近路线，允许准备按键不等于允许从键底进入。奖励设计归主任务另行实现，本文没有改变它。

全指数据在各 `detail-0/finger_clearance.npz`：40 个碰撞 primitive 的严格轴对齐包络给出每根手指最低表面，另存当前/未来 0.25 秒/未来 0.5 秒用指掩码与接触。最佳回放十根手指都出现过在“当前及未来 0.25 秒不用”时低于其局部键面包络的样本；不是只存在左拇指一个特例。例如右食指有 145 子步，左拇指 43 子步。它们是保守高度违规诊断，不全是真实碰撞。

只监视指尖 site 也不充分：右小指存在 19 子步“指尖高于白键面、其余指节碰撞表面仍低于白键面”。应检查完整参与碰撞的四段指体。

## 评估盲区

best 准备段 20 Hz 误触为 0，但 200 Hz 仍有 E5/MIDI76 在 0.965 秒起响，共 10 ms。音乐 20 Hz F1=0.98228、错键为 0；200 Hz 诊断 F1≈0.91527、有 114 个错键子步。许多子步问题属于目标换帧附近的按下/释放延迟，不能把这个诊断直接替代当前训练 F1；也不能把 20 Hz 零错键描述成音频完全无杂音。

## 通用改进的验收建议

以下是待实现方案的验收定义，不是已达标承诺，不限定曲名、音高或某一根拇指：

1. **非用指全指净空：**检查左右十指所有实际碰撞 primitive，按明确的闲指/准备指/当前按键指掩码分类，空谱也有信号。明确白/黑键局部键面、净空 margin 和键盘范围外是否仍要求全局高度，不能用“无接触”替代“在琴面上方”。
2. **防止低处横向入键：**穿过键盘边界/相邻键 XY 范围前，整指高度需先满足接近约束。当前要按的手指可向目标键顶面下降，但不允许因用指豁免从侧底进入。只看控制周期末可能漏掉 200 Hz 路径。
3. **接触结果：**每次三回合 canonical+至少一次完整 200 Hz 详细回放；曲首、整曲分别报任意手指键底子步、最长连续区间、侧接触、非末端接触、200 Hz 准备误触。目标应包含键底为 0，不能只看手—手碰撞。
4. **受阻判据：**保留接触法向、目标跟踪误差、执行器限幅、反向约束力矩与连续时长。慢速度阈值需明示；本例严格 `|v|<0.2 rad/s & |lag|>0.3 rad & 饱和 & 键底` 仅 10 ms，不能因此否认其余 240 ms 底部接触，也不能把整段都称为完全堵转。
5. **通用回归样本：**空谱、十指分别单指、双手组合、不同黑白键/位置、目标到休止与休止到目标、近期用指保护边界、指尖安全但中间指节偏低、手指在琴外低位向琴内移动；在不同曲目/移调/指法输入重测，不写入 C4/F3/某时间的特例规则。
6. **音乐质量：**保留 best，候选原权重回放和短训后都检查音乐 macro/micro F1；训练未达到目标时如实报告。新奖励或观测合同须显式迁移，不让旧 checkpoint 静默加载新任务。

## 实际验证与边界

- 严格归档重建 exit 0，best/final 各三次标准评估＋一次完整详细回放，共八回合。GPU 已释放。
- CPU 净空分析与作图 exit 0，保存 original/reset 两静态姿态、40 primitive/10 指全部数据；origin→initial 所有表面均增加 20 mm。
- 静态姿态通过 CPU `mj_forward`，运动轨迹与接触来自记录的 GPU 世界；未用 CPU 重新仿真替代 GPU 接触力。`qfrc_constraint` 包含全部约束，不当作单个键接触力。
- 局部净空使用 primitive 与静止键的 XY AABB 重叠，是保守筛选，不能当真实几何穿透距离。键底结论来自实际接触法向，独立于高度近似。
- 正式包、模型、奖励、控制合同未在这份审计中修改；后续改进/训练由主任务验收。
- TODO **V32-START-01**：新策略接管和全指路径仍须证明无键底进入，现 best 明确未过；**V32-START-02**：200 Hz 短错音盲区仍存在；**V32-SAC-03**：0.99/零碰撞/通用曲目效果仍未达到或未验证。
