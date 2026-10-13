# V3.2 SAC 实现与 Shadow 差异确认

> 2026-09-28 工作约定更新：本文保留历史 JAX 基线及当时的比较结果。用户已取消强制论文阅读、逐项比对和因论文内容重合而征询对齐决定的要求，后续训练不以论文参数为默认依据。当前入口是 [run.sh](../run.sh)，实际参数以启动命令、完整 `config.json` 和实测结果为准；历史模型与检查点兼容要求保留。

2026-09-16。当前交付为可独立运行的**单环境 CPU MuJoCo + JAX SAC 基线**；本机安装的是 CPU JAX 依赖。环境物理仍使用原 Composer；未实现 MJX、多环境 GPU 回放或双卡调度，也没有执行正式长训练。本文列出的训练默认值是待用户确认的候选基线；短程验证使用 Toy 和小网络。

## 2026-09-17 路线更新

后续采用 **MJLab + MuJoCo Warp + 自定义 PyTorch SAC + GPU replay**，见 [当前迁移方案](dual_a100_sac_plan.md)。本文中的 JAX 实现和命令继续作为现有可运行基线与算法对照；MJLab／Torch训练入口现已另行实现，见 [Torch SAC验收](torch_sac_acceptance.md)；本文命令仍是JAX参考，不会启动MJLab。V32-01 当前仿真安装阶段基本完成；单 GPU 场景已接通，见 [场景验收](mjlab_scene_acceptance.md)，16／64 环境验证已完成，见 [批量验收](mjlab_batch_acceptance.md)，有指法 Torch 任务已完成，见 [任务验收](mjlab_task_acceptance.md)，Torch SAC与GPU replay已通过单卡验收。C01～C05 仍用于确定正式学习配置，框架选择已按用户要求确定，不需重复确认。

## 已实现内容与使用

可选参考：[两阶段 MJLab＋SAC 历史论文记录](references/biomimetics_2026_610_notes.md)。论文的观测、奖励缩放、功率定义、网络和两阶段参数与本页历史基线存在差异。本页 JAX 候选值保持原状；2026-09-17 当轮 Torch SAC 曾按用户选择采用论文参数，见 V32-PAPER-10 历史登记，任务奖励与完整课程没有随之更改。该历史选择不作为当前训练默认配置或后续修改的论文比对要求。

实现位于包内 `tsensei_v3_2_pianist/rl/`：

| 文件 | 内容 |
| --- | --- |
| `sac.py` / `networks.py` / `distributions.py` / `specs.py` | 原版 SAC 核心、网络、tanh 高斯和接口；相对导入，可独立复制 |
| `environment.py` | 当前 V3.2 环境与原训练包装顺序的适配，输出真实动作/观测说明与来源散列 |
| `replay.py` | CPU 均匀随机回放，显式存完整 transition，避免跨 episode 拼接 |
| `train.py` | 单环境训练、独立评估、本地日志及 CLI |
| `checkpoint.py` | 保存三个网络优化器、目标网络、温度和 RNG；严格核验环境和算法配置 |

当前独立验证环境在 `training/.venv`。与原预览 `.venv` 分离，原预览依赖未升级。复制项目时可排除两个虚拟环境。

在项目目录安装（CPU 验证配方）：

```bash
python3 -m venv training/.venv
training/.venv/bin/python -m pip install -r requirements-sac-tested.txt
training/.venv/bin/python -m pip install -e '.[train,test]'
```

`requirements-sac-tested.txt` 是本次 Linux/Python 3.12 CPU 安装快照，包含实际解析出的第三方依赖；不是 A100/CUDA 安装证明或跨平台锁文件。远程 GPU JAX 需按驱动选择并验证对应依赖，见 [JAX 官方安装](https://docs.jax.dev/en/latest/installation.html)。

只查看待确认的真实配置，不训练：

```bash
MUJOCO_GL=egl JAX_PLATFORMS=cpu training/.venv/bin/python -m tsensei_v3_2_pianist.rl.train --inspect-only
```

短程验证示例（输出目录必须尚不存在）：

```bash
MUJOCO_GL=egl JAX_PLATFORMS=cpu training/.venv/bin/python -m tsensei_v3_2_pianist.rl.train \
  --midi Toy --max-steps 32 --warmstart-steps 8 \
  --batch-size 8 --replay-capacity 64 --hidden-dims 16 16 \
  --eval-interval 16 --eval-max-steps 100 --log-interval 8 \
  --save-interval 16 --output-dir runs/sac-smoke
```

安装后也有 `training/.venv/bin/tsensei-sac` 入口。`--help` 展示开关；布尔开关如 `--no-gravity-compensation`、`--no-action-reward-observation` 可显式关闭。正式训练命令应在下文配置确认后再确定；默认 500 万步不是本次已执行量。

输出有 `config.json`（实际字段顺序、动作名称/范围、模型/任务散列、依赖版本）、`metrics.jsonl`、`summary.json`、周期和最终 `.msgpack`。`--warm-start CHECKPOINT` 恢复 SAC 参数、优化器和 RNG，但重新建立回放与物理 episode，不能称为精确断点续训；需要相同网络、折扣、环境及动作/观测合同。

## 与原 Shadow 训练的逐项差异

这里的 Shadow 基准是母仓库 `robopianist-rl/run.sh` 的实际覆盖值，不是 `Args` 的所有默认值。核心文件来源散列保存在 [来源清单](sac_source_manifest.json)；可逐行查看 [源码差异](shadow_sac_comparison.diff)。这些对照文件只作审阅，不是待直接应用的补丁。

| 编号 | 原 Shadow | V3.2 当前实现／候选 | 原因与状态 |
| --- | --- | --- | --- |
| D01 环境入口 | `robopianist.suite.load(environment_name=...)` | 独立包 `suite.load(midi=...)` | 必须适配；运行不依赖母仓库或 Shadow 资产 |
| D02 动作 | 原手执行器，run.sh 开启 reduced_action_space | 51 维，右手 25、左手 25、踏板 1 | 按现有 V3.2 模型适配；25=20 手指+5 虚拟腕部；策略输出层、critic 动作输入自动跟随，不能复用 Shadow checkpoint |
| D03 动作归一化 | `CanonicalSpecWrapper` 映射 [-1,1] 至物理范围 | 同一线性映射；在物理规格精度下缩放 | V3.2 严格检查 ctrlrange，避免 float32 上限舍入越界；并拒绝错误形状/NaN，测试上下限与中点 |
| D04 动作缩减 | `--reduced-action-space` | 不传入该参数 | 该开关为 Shadow 专用，V3.2 会报错；未增加新的关节锁定或联动 |
| D05 指尖碰撞 | `--primitive-fingertip-collisions` | 使用当前 V3.2 碰撞网格，不传入该参数 | Shadow 专用开关不适用；本次未改网格、kp、力矩、摩擦、关节范围或碰撞排除 |
| D06 观测 | 原 Shadow 的关节/动作尺寸，10 步前瞻、历史动作/奖励 | 当前训练配置实测 1180 维，有指法；无指法为 1170；frame_stack 后倍增 | 327 仅是 V3.2 默认 Toy 环境，不是 run.sh 参数下的训练输入；按实际规格建立网络 |
| D07 包装顺序 | 历史动作奖励→拼接→可选 frame stack→动作归一化→float32→dm_control 接口 | 保持顺序；仅本地子类修复 NumPy 2 转换 | 拼接字段按名称排序；历史动作保留**物理目标值**，replay.action 则存归一化策略动作；reset 历史值继承上游 wrapper 的 generate_value 语义 |
| D08 SAC 数学 | 目标 Q 取 min；actor Q 取 mean；自动 alpha；critic→actor→temperature | 原样保留 | 三次原算法更新数值对照覆盖参数、优化器、目标网络和 RNG；没有替换成 PPO/其他 SAC 实现 |
| D09 参数 | run.sh：gamma=.8、256×3、critic dropout=.01、LayerNorm | CLI 候选相同；学习率3e-4、tau=.005、batch256 | 基于实际启动脚本；SACConfig 库默认仍保持原版 dropout0/无LN，CLI 显式覆盖；Args 默认 gamma=.99 不能混为原实验值 |
| D10 目标熵 | 默认 -0.5×动作维数 | 51 维对应 -25.5 | 保留公式；显式 target_entropy=0 现在被正确接受，原版 `or` 会意外回退，该边界修复不影响默认算法 |
| D11 回放 | `_prev/_latest` 拼接 timestep；全局 NumPy RNG | 显式 `(s,a,r,discount,next_s)`；独立 RNG | 均匀有放回采样保持；末帧先入库再 reset；探索与回放 RNG 分离意味着随机轨迹不逐步复现原全局 RNG |
| D12 评估状态 | 评估代码复用训练 `timestep` 变量 | 独立 eval 环境、局部 timestep 与种子 | 修复评估后策略读取错误观测的风险；测试开启/关闭评估不改变训练 agent 的最终状态 |
| D13 音乐指标 | 每帧 binary P/R/F1，zero_division=1，再平均 | 按键与踏板保留该统计方式；额外给 micro F1 | micro F1 独立命名，不能与原 F1 混用；评估设步数上限并报告 capped episodes |
| D14 输出与恢复 | W&B／音视频包装，原脚本无检查点保存 | 本地 JSON 日志和 SAC 状态检查点；未移植音视频/W&B | 本次交付算法基线；检查点明确为 warm restart，未保存 replay 或环境状态 |
| D15 现代依赖兼容 | `jax.random.KeyArray`、FrozenDict 特定 copy 方法 | `jax.Array` 类型标注，兼容 dict/FrozenDict | 接口修复；新增设备驻留动作接口供后续 GPU 环境使用，CPU 接口仍返回 NumPy |

原奖励公式沿用**当前 V3.2 包**，不是本次从 Shadow 重新定义奖励；指尖 site、掌部接触几何等沿用现有手模型。`disable_fingering_reward=True` 在当前任务中会进入 OT 指派分支，不等于取消所有指尖距离奖励，本次不更改此行为。末帧 discount 直接沿用实际 dm_env 返回值。

默认有指法训练观测的顺序和长度已实测：

| 字段 | 长度 |
| --- | ---: |
| action（物理目标历史） | 51 |
| fingering | 10 |
| goal（当前+10 个未来时刻，每时刻88键+踏板） | 979 |
| left_/joints_pos | 25 |
| piano/state | 88 |
| piano/sustain_state | 1 |
| reward（上一奖励） | 1 |
| right_/joints_pos | 25 |
| 总计 | 1180 |

## 请确认的正式训练基线

以下是建议，已经写成可检查的 CLI 默认值，但没有据此启动正式训练。确认可按 C01～C05 回复，或回复“全部按建议”；用户的确认决定之后的正式训练配置，不影响本次已授权的算法实现。

| 编号 | 建议 | 需要你确认的选择 |
| --- | --- | --- |
| C01 动作与模型 | 使用现有 V3.2 的全部 51 维动作，含每手5个虚拟腕部自由度 | 是否保持完整动作，还是需要锁定某些腕部/关节？新增锁定尚未实现 |
| C02 初始曲目 | 正式基线用 TwinkleTwinkleRousseau，与原 run.sh 一致；Toy 只作短测 | 是否沿用这首曲目，或改为你指定的曲目？ |
| C03 任务输入 | 控制周期0.05s、物理0.0005s；10步前瞻；启用历史动作/奖励和指法奖励 | 是否沿用？当前有指法输入1180维，改选项后自动重新计算 |
| C04 重力补偿 | 训练中开启 gravity_compensation，跟随原 run.sh | 是否开启？这与直接 `suite.load()` 的默认关闭不同，但不改变包内模型资产或预览默认值 |
| C05 SAC 参数 | 本页JAX历史候选gamma=.8、batch256、256×3等保留 | 2026-09-17 用户已选择新Torch主线采用论文表2；正式训练效果仍待验收，见Torch SAC说明 |

## 验证与后续

原算法对照数据来自未经算法修改的母仓库 SAC，用同一 CPU 依赖环境生成；仅在生成进程提供已移除的 `KeyArray` 类型标注别名。数据和版本在 `tests/data/shadow_sac_reference.npz` / `.json`，独立项目运行测试时不读取母仓库。

首次短测：Toy 32 步、25 次更新、1 个训练回合结束，执行了两次独立评估，保存周期和最终检查点。小网络16×16、batch8用于功能验证，不能推断默认大网络收敛或 A100 性能。完整测试、安装检查与最终验证摘要见 [SAC 验证记录](sac_validation.json) 和 [修改日志](change_log.md)。

- TODO(V32-SAC-01)：MJLab + SAC 框架路线已确定；落实 C01～C05 的任务／学习配置并进行正式效果评估，短测不证明可演奏。
- TODO(V32-SAC-02)：若需要精确断点恢复，补齐 replay、环境、采样与调度状态；目前仅 warm restart。
- TODO(V32-GPU-05～06)：沿用 [双 A100 方案](dual_a100_sac_plan.md) 的多卡基准及学习评估（04单卡功能已完成）；当前 JAX SAC 代码是其单环境参考实现。
