# V3.2 Torch 弹琴任务与验收

> 2026-09-28 工作约定更新：本文描述 2026-09-17 的任务迁移验收。用户已取消强制论文阅读、逐项比对和因论文内容重合而征询对齐决定的要求，后续训练不以论文参数为默认依据。当前任务及训练参数以 [run.sh](../run.sh)、启动命令、完整 `config.json` 和实测结果为准；下文历史数值不覆盖当前配置，旧检查点仍保留原合同。

更新：2026-09-17。**V32-GPU-03 / P2 已完成本轮约定的有指法任务迁移。** `PianoTaskEnv` 在现有 MJLab `Simulation` / MuJoCo Warp 物理层上计算 Torch 乐谱、观测、奖励、结束信息和评估指标，可向后续 SAC 提供 transition 数据。后续已接通 [Torch SAC、GPU replay与训练入口](torch_sac_acceptance.md)，尚未证明策略学会弹琴。

## 1. 2026-09-17 已确认的迁移范围（历史）

用户已确认，下列事项无需重复询问：

- 当轮保留 CPU 任务规则，未联动采用论文的新观测、40 Hz、乘 dt 奖励、正功率或完整奖励课程；当时后续 SAC 验收采用了用户选择的论文参数。见 [历史论文决策登记](references/biomimetics_2026_610_notes.md#本轮追加决定2026-09-17)。该记录不要求后续按论文配置训练或逐项比对。
- 用户将提供合适指法，本轮不实现或验收无指法最优指派分支。首版只接收完整指法乐谱，缺失或非法标注报错。
- 不使用错键终止；错键扣分，继续演奏。没有增加开启该行为的 CLI 参数。
- 保持双手＋踏板、51 维动作、20 Hz 控制、0.0005 s 物理步长、100 子步、腕部 TZ −25～0 mm，以及当前安装与关节初值。

物理配置以冻结场景为准。当前 `training_mjlab/outputs/scene` 的 `gravity_compensation=false`，任务导出自动沿用同一值，CPU/GPU 对照一致；原 JAX 训练候选默认 `true` 是另一配置。若后续选择重力补偿训练，须另导出场景和相配任务并重新验收，不能直接复用本次散列。

## 这些功能具体做什么

- **乐谱**：每50 ms确定这一刻应按的键、应松开的键和踏板目标，并提供当前＋未来10帧，帮助策略提前准备。
- **指法**：记录每个音符指定哪只手、哪根手指。观测里的10个数只是当前十指是否应参与；精确的“手指—琴键”分配另供指法奖励和预览使用，不等同于自动控制手指。
- **1180维观测**：给策略的数字输入：上次物理动作51、当前十指使用标记10、11帧×89目标979、双手关节位置50、琴键按压状态88、踏板状态1、上次奖励1，共1180。它不是1180个关节，也不是图像。
- **分项奖励**：分别评价按键、踏板、指尖靠近指定键、执行器功率和左右掌部碰撞，再直接相加，供后续学习改进动作。
- **结束逻辑**：标记曲目演奏完毕，或显式时间限制达到，保存结束帧。一次episode结束不等于停止整次训练。
- **独立reset**：例如64个环境中只复位第3个，恢复它的物理、乐谱和历史，其余环境继续各自进度。
- **F1**：衡量实际按键与乐谱是否一致，同时考虑错按和漏按；另有踏板指标。键F1高不代表指法一定正确，指法距离由单独奖励评价。

“错键只扣分”具体是失去当步0.5的无错键奖励，当前不是每错一个键再扣0.5，也不保证总奖励变为负数；曲目照常推进。“仅验收有指法”表示本轮使用带手指标注的曲谱，没有实现无标注情况下自动匹配手指的分支。“保留CPU规则”表示把现有规则迁移到GPU批量计算，保持任务含义；论文新增观测、40 Hz和奖励改法未自动采用。

## 2. 代码与数据

| 入口 | 职责 |
| --- | --- |
| `scripts/export_mjlab_task.py` | 用原 CPU Composer 预处理乐谱、记录实际观测合同和参考轨迹 |
| `tsensei_v3_2_pianist/mjlab_env/task_bundle.py` | 冻结乐谱／指法与场景关联散列；加载时校验资源 |
| `tsensei_v3_2_pianist/mjlab_env/task_logic.py` | Torch 批量乐谱、前瞻、历史、奖励、结束与指标 |
| `tsensei_v3_2_pianist/mjlab_env/task.py` | `PianoTaskEnv`：连接 GPU 物理与任务规则，提供 step/reset |
| `scripts/validate_mjlab_task.py` | 固定 CPU 状态数值对照、真实 GPU rollout、局部 reset 和末帧检查 |
| `preview/scripts/preview_mjlab.py --task PATH` | Control 交互＋指法颜色／左右手标签／音名＋实际激活琴键绿色＋终端奖励／F1 |

任务包包含 `task.json`、`score.npz`、`cpu_reference.npz`。`goal[T,89]` 保存 88 键和踏板目标，`fingers[T,10,88]` 保存指法分配计数；静音帧允许全零，同指多目标保留原 CPU 的逐音符权重。任务与冻结场景的 `manifest.json` 散列绑定，不能混用另一个姿态或动力学版本。

导出需要本项目的 `training/.venv`（Composer 与观测包装器）；运行使用 `training_mjlab/.venv`。运行时不导入 Composer/JAX，不依赖母项目、V2/V3 或参考附件目录。乐谱和规则张量留在 GPU；目前有批次级合法性／结束检查产生主机同步，尚未进行任务层图捕获或训练吞吐优化。实现为本项目 Torch 任务适配层，尚未改成 MJLab `ManagerBasedRLEnv`。

### 乐谱导出

以下命令均在独立项目根目录执行。输出目录必须为空／不存在，导出器拒绝覆盖已有验收包。已有本机输出可直接进入下一节验收。

```bash
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_task.py \
  --midi Toy --output training_mjlab/outputs/task_toy
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_task.py \
  --midi Toy --frame-stack 3 --output training_mjlab/outputs/task_toy_stack3
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_task.py \
  --midi TwinkleTwinkleRousseau --output training_mjlab/outputs/task_twinkle
```

用户曲目可用 `--midi /absolute/path/song.proto --output training_mjlab/outputs/task_my_song`。指法读取 `NoteSequence.notes[].part`：右手拇指到小指为 0～4，左手为 5～9。优先交付保留这些字段的 `.proto`；标准 `.mid` 通常不能保留 part。沿用当前 CPU 的 `has_fingering()` 检测：只有 part=0 的曲目会被视为缺少指法，需要单独处理这种输入后再使用，不自动猜指法。没有实际验证用户尚未提供的曲目。

## 3. 任务合同

### 动作、观测与时序

动作 `[N,51]`，归一化到 `[-1,1]`，依次右手25、左手25、踏板1；物理控制是各执行器的绝对位置范围，踏板映射到 `[0,1]`。错误形状、非有限值或越界输入在推进前报错。

默认观测 `[N,1180]`：51 上次物理动作＋10 当前指法使用标记＋11×89 目标窗口＋左右50关节位置＋88琴键归一化位置＋1踏板＋1上次奖励。具体顺序以 `task.json` 的 `contract.observation_fields_before_stack` 为准。窗口包含当前帧与未来10帧，尾部补零；没有新增速度或相位观测。三帧历史为3540维，按特征保存从旧到新的三帧。

每次 step 用当前乐谱目标计算本次奖励，再推进索引生成下一观测。曲目最后一步保留 CPU 的最后目标窗口。reset 清空历史，仅最后一帧填入初始观测；上次奖励为0。2026-09-24 起，新导出 task 的上次动作初值使用冻结 scene 的中立物理控制，并以 `reset_action_semantics=scene_neutral_physical_v1` 标记；它与 reset 后的实际执行器目标一致。旧 task 缺少该字段时仍按历史的动作下限语义加载，以保持旧检查点可复现。原因、主流并行环境对照与旧权重边界见[reset 修复记录](reset_recovery_20260924.md)。

### 奖励

| 分项 | 本轮规则 |
| --- | --- |
| 按键 | 目标键按压深度的 Gaussian tolerance 均值×0.5；没有错键再加0.5；静音帧的目标键部分为0 |
| 踏板 | 阈值0.5离散踏板后，与乐谱目标计算原 tolerance |
| 功率 | `−0.005 × Σ abs(actuator_force × actuator_velocity)` |
| 指法 | 指定指尖到原琴键目标点的 tolerance 均值，阈值0.01 m、margin 0.1 m；无目标为0 |
| 掌部碰撞 | 左右指定掌部碰撞时0，否则0.5；可在导出时关闭 |

各项直接相加，不乘控制 dt。物理推进后刷新几何位姿；功率使用最后积分子步的执行器力与刷新后的速度，保持 CPU `legacy_step` 的采样时点。按键激活来自真实关节状态，没有为命中目标强改琴键位置。

### 结束、reset 与 SAC 数据

- 曲目完成：`terminated=True`；显式设置 `max_episode_steps` 且提前达到限制：`truncated=True`。两者同时达到时优先曲目完成。
- 两种情况下 `discount=1`，忠实保留当前 CPU 的返回语义；`terminated` 在此表示曲目完成，不能套用某些 Gym runner 的自动折扣规则。
- 不自动 reset。调用者先保存末帧 transition，再 `reset(done_ids)`；若有结束环境未 reset，下一次 step 在物理推进前报错。
- `reset(ids)` 只重置选定世界的物理、乐谱、踏板、历史和累计指标，返回全部环境的当前观测。未选世界的历史不额外推进。
- 新 task 同时把选定世界的上次动作历史恢复为实际中立控制；旧 task 保留其冻结的历史下限值。
- 返回观测独立持有内存，后续 step/reset 不覆盖上一 transition。
- 曾实测原 CPU 的错键终止内部设置 discount=0，但 Composer 已提前读取，因此实际返回1。本轮按用户要求关闭错键终止，未修改旧 CPU 的这个行为。

`TaskStep` 提供 `observation/reward/discount/terminated/truncated/info`，`done` 为两个结束标记之并。后续 replay 保存归一化动作，SAC TD target 使用 **`gamma * result.discount`**。先保存结束帧的 next observation，再获取 reset 后的 actor 输入。

```python
import torch
from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv

env = PianoTaskEnv("training_mjlab/outputs/scene",
                   "training_mjlab/outputs/task_toy", num_envs=64)
obs = env.reset()
action = torch.zeros((64, 51), device=env.device)  # 接口示例：范围中点，不是中立姿态
result = env.step(action)
transition = (obs, action, result.reward, result.discount,
              result.observation, result.terminated, result.truncated)
# 此处可用 rl_torch.replay.Replay.insert 保存 Transition；完整接线见 rl_torch.train。
obs = result.observation
ids = result.done.nonzero(as_tuple=False).flatten()
if ids.numel():
    obs = env.reset(ids)
```

`info` 包含各项奖励、应用目标与索引、下一索引、实际激活键、错键标记、累计回报、步数、结束原因（0继续／1曲目完成／2时间截断）。`frame_metrics` 和 `episode_metrics` 顺序为键 Precision/Recall/F1、踏板 Precision/Recall/F1；后者逐帧平均，分母为零按原评估记1。另提供汇总键 TP/FP/FN 的 `key_frame_micro_f1`，其零总分母按原训练评估记0。两个 F1 口径不同，日志中须注明。

## 4. 如何验收

### 自动验收

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_task.py --num-envs 16
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_task.py --num-envs 64
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_task.py \
  --num-envs 16 --task training_mjlab/outputs/task_toy_stack3
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_task.py \
  --num-envs 16 --task training_mjlab/outputs/task_twinkle
training_mjlab/.venv/bin/python -m unittest discover -s tests -p test_mjlab_task.py -v
```

通过标准：命令退出0，终端显示 `Music task acceptance: passed`，对应 `training_mjlab/outputs/task_acceptance_*/report.json` 的所有检查为 true。报告包含固定 CPU 状态规则误差、真实 GPU 轨迹误差和单位、键激活一致性、指标归约、逐环境 reset、结束帧所有权和时间截断。

固定状态观测／各奖励误差阈值2e−5；真实物理存在浮点和求解器差异，单独检查滑动关节<0.002 m、铰链<0.05 rad、观测<0.1、总奖励<0.05且离散键激活完全相同。真实 rollout 容差不等于固定状态规则容差，也不是硬接触标定标准。

本机实测汇总见 [验证记录](mjlab_task_validation.json)：Toy 16／64、三帧历史、Twinkle 均通过；Toy 全21帧、Twinkle CPU/GPU前64帧对照，GPU中立控制完整158帧结束流程通过。另有4项解析规则测试、30项原 CPU 回归测试、64环境原物理验收和真实桌面预览检查。

### 窗口验收

在桌面终端执行：

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --task training_mjlab/outputs/task_twinkle --paused
# 可选：固定动作下压演示
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --task training_mjlab/outputs/task_toy --bottom-out
```

空格开始／暂停，Control 滑块调节关节和腕部目标。实际激活键变绿；终端每10步及结束时打印目标键编号（0～87）、总奖励和逐帧平均键 F1。按错键不会提前结束；到曲目末尾暂停，按 R reset，再按空格继续。未按下目标键按指法着色（拇指粉红／品红、食指红、中指青、无名指蓝、小指黄），指尖和键上方同时显示R1～R5／L1～L5标记；左上角列出图例与十指音名／键号。绿色表示实际按下，绿色键上的指法标记仍保留。默认启用，`--no-show-fingering`关闭；初始暂停也显示，跟随当前观测的待执行目标帧，曲末保留末帧。Toy只用食指，要看拇指目标可用Twinkle。显示详情见 [预览说明](../preview/README.md)。当前窗口没有独立踏板滑块；踏板任务逻辑已通过脚本中的0／0.5／1动作对照。

固定下压动作只是验收真实接触与任务数据，按到的键可能与曲谱不同，F1为0是允许的；这不是已训练策略。默认不传 `--task` 时仍为纯物理预览。

## 5. 下一步与待确认事项

- **V32-GPU-04已完成单卡功能验收**：Torch SAC、CUDA replay、训练／独立评估／检查点和策略预览见 [Torch SAC验收](torch_sac_acceptance.md)。
- **V32-PAPER-10已决定**：用户选择论文表2参数，Actor256×2、Stage1/2优化器与向量步预算已配置。V32-PAPER-09完整奖励课程仍未采用；未披露细节保持登记。
- **V32-SAC-01／02**：用户正式曲目与学习配置、效果指标和精确恢复仍需落实。本轮 Toy/Twinkle 只用于接口与规则验收。
- **V32-GPU-01／05／06**：远程 A100、双卡吞吐和多 seed 学习效果待完成。
- V32-04／05／06／07／10 的模型动力学、接触、指尖与正式右手限制继续保留，见 [修改日志](change_log.md)。

无需等待无指法分支即可推进上述工作。暂存的 `*_stale_reference` 是开发时旧参考证据，不能用于新训练或验收。
