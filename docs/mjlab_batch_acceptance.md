# MJLab 第二阶段：激活颜色、接触止挡和 16／64 环境

## 本次行为

- MJLab 预览与原 CPU 预览中，实际激活的琴键显示绿色，松键后恢复原色。依据实际琴键角度判断，沿用距最大角度 0.5° 的激活阈值；绿色表示已激活，不表示一定已完全按到底。
- 琴键最大行程由物理关节限位产生反力，手—键碰撞与限位一起阻挡下压。控制目标仍可继续向下调，实际位置受接触和执行器力限约束；其他手指关节也可能弯曲或滑动，不能理解为把整只手固定住。
- 支持一个 MJLab Simulation 内的 1／16／64 个世界，每个世界有独立的 qpos、qvel、ctrl、时间、踏板和控制步计数，可只 reset 指定世界。

腕部范围按用户确认设为 **Palm_Base_Tz = −0.025～0 m**，在 `tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py::FOREARM_DOFS["forearm_tz"]` 修改；左右执行器、预览 Control 和新导出动作范围同步。初始安装位置和初始关节角未改变。

## 接触实现与范围

配置在 `tsensei_v3_2_pianist/suite/tasks/base.py`，CPU 预览、CPU 训练参考与新导出模型一致：

| 配置 | 当前值 |
| --- | --- |
| 物理步长 | 0.0005 s（原 0.002 s） |
| 控制周期 | 0.05 s；每次控制 100 个物理子步 |
| 积分器 | implicitfast |
| 琴键接触 solref / solimp | (0.001, 1) / (0.99, 0.99, 0.001, 0.5, 2) |
| 琴键限位 solreflimit / solimplimit | (0.001, 1) / (0.999, 0.999, 0.001, 0.5, 2) |
| 钢琴碰撞 priority | 1，确保与手接触时采用钢琴侧参数 |

MuJoCo 使用可调的软约束求解接触，不能保证任意速度／载荷下数学意义的零穿透。本次实现较硬的接触和最大行程止挡，保留微小数值误差；没有在 step 后裁剪 qpos、冻结手指、删除触键碰撞或增大执行器力矩。[官方求解器参数说明](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)

较硬约束直接配合旧 Euler 步长会产生抖动；因此降低步长并改用 implicitfast。子步数增至原来的 4 倍，物理成本会增加；这是本次接触精度选择，未宣称加速训练。V32-04／05／06／07／10 的驱动、速度、碰撞、指尖和正式右手资料仍需标定。

绿色显示使用纯视觉覆盖层，厚度扩张 20 微米以避免画面闪烁，不参与碰撞；继续使用 state-only viewer sync，避免每帧复制完整模型。离屏图直接修改渲染场景颜色。

## 1. 目视验收

本节记录当时较硬接触、腕部下压 25 mm 的冻结场景验收；历史旧版曾保存在 `training_mjlab/outputs/scene_before_hard_contact_20260917`，后按[快照清理记录](source_bundle_cache_20260924.md)删除。在项目根目录运行以下历史命令时，请先核对所需场景是否仍存在：

```bash
# 自动按到底并松开
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py --bottom-out
# 手动控制；Space 继续／暂停，R 复位
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py --paused
```

手动检查：

1. 按 Space 开始仿真；将左右 Palm_Base_Tz 从 0 调到 -0.025 m。腕部只能下压 25 mm，仅靠此动作可能尚未将键按到底。
2. 将左右 Mid_MCP 从初始 0.25 调到 0.35 rad；实际激活的键变绿。持续保持目标时，中指对应键到达最大行程，接触反力阻止中指达到控制目标。不要把滑块目标当作实际角度。
3. 将 Palm_Base_Tz 调回 0、Mid_MCP 调回 0.25，手抬起、琴键回弹，绿色消失。

原预览也可使用 `.venv/bin/python preview/scripts/preview_pose.py` 验收相同规则，保留 S 保存和坐标轴等功能。旧导出模型不会因源代码改变自动更新；另存新姿态后请重新导出到新目录，并用 `--scene` 指定。

## 2. 自动验收单场景、16 和 64 环境

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_env.py \
  --output training_mjlab/outputs/acceptance_wrist25 --render

MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_batch.py --num-envs 16
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_batch.py --num-envs 64
```

**通过标准**：第一条显示 `Acceptance: passed`，后两条分别显示 `Batch acceptance: passed`，退出码均为 0。报告分别在 `acceptance_wrist25/report.json`、`batch_16/report.json`、`batch_64/report.json`。`pressure_states.npz` 保留并行环境的控制目标、动作分组与受压状态。

并行验收覆盖：

- 相邻世界分别不按／只按右手／只按左手／双手同时按，踏板值也不同；检查每个世界的激活与踏板。
- 连续 1.5 s 施加 Palm_Base_Tz=-0.025 m、中指 MCP=初始值+0.1 rad，确认中指对应琴键到达限位（角度距上限 <0.001 rad），中指实际角度低于目标 >0.05 rad。
- 对施压阶段**每个物理子步**采集溢出位、接触／约束峰值和接触深度：溢出位全 0；手—键最大数值穿透 <0.5 mm；琴键前端最大超行程 <0.1 mm。前端误差采用键长×角度超限的保守换算，未裁剪原始 qpos。
- 同动作组激活琴键集合相同，滑动关节差 <0.05 mm、铰链差 <0.002 rad；GPU 接触浮点归约可能产生微小差异，不要求轨迹逐位相等。
- 各动作组 CPU 参考对照：最大滑动误差 <2 mm、铰链角误差 <0.05 rad。
- reset 偶数世界后，未选中的奇数世界 qpos、qvel、ctrl、时间、求解器 warm-start、踏板与计数器逐位不变；再向前一步与连续 CPU 参考比较。
- 通过实际松开控制让琴键回弹；倒序 reset 全部世界检查初值恢复。

这是固定场景的短程物理验收，不是所有姿态／撞击的极限证明，也不是训练吞吐基准。报告中的 Torch 显存峰值不包含全部 Warp 分配，不应当作整卡显存峰值。

## 3. 调用批量接口

```python
import torch
from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene

scene = PianoScene('training_mjlab/outputs/scene', num_envs=16)
action = scene.tensor(scene.bundle.normalize(scene.bundle.neutral_action)).repeat(16, 1)
state = scene.step(action)  # (16, 51) 归一化动作，范围 [-1,1]
scene.reset(torch.tensor([1, 7], device=scene.device))
snapshot = scene.cpu_snapshot(env_id=7)  # 仅显示或诊断；不推进 CPU 物理
```

`state` 中的物理状态可能是共享缓冲区视图；若要跨 step 保存，用 `clone()`。`substep_callback` 仅为可选验收钩子；正常调用不执行本次密集诊断。

## 本机结果（2026-09-17）

- RTX PRO 6000 Blackwell，MJLab 1.6.0、MuJoCo/Warp 3.11.0、Torch 2.10.0+cu128；原 CPU 预览使用 MuJoCo 3.13.0。
- 32 项 CPU 测试、45 项单场景 GPU 检查、16／64 环境各 49 项检查通过；冻结场景和动作范围均已更新。
- 64 环境施压阶段：最大手—键穿透约 **0.326 mm**，琴键前端最大超行程约 **0.0432 mm**；最多 186 个总接触、单世界最多 112 个约束，溢出位全为 0。中指 MCP 与持续目标的差约 0.121 rad，表明接触阻止其到达目标。
- 实际 X11 MJLab 演示按键 `[33,54]` 变绿后松开，正常退出；原 CPU 预览附坐标轴短测约 60 Hz，NVIDIA 硬件渲染、state-only sync。没有以鼠标实际拖动控件，手动检查仍按上节进行。
- [数值与检查记录](mjlab_batch_validation.json)；[当前按键图](../preview/outputs/hard_contact_press.png)。旧单环境报告保留为历史记录。

物理参数和腕部动作范围变化会改变学习问题及模型散列；旧训练检查点不应直接当作同一环境下的续训。本次只重新检查训练配置快照，没有进行 SAC 学习。

## 后续工作

V32-GPU-03 已按用户选择完成有指法 Torch 乐谱、观测、奖励、终止、历史和部分 reset，见 [任务验收](mjlab_task_acceptance.md)。V32-GPU-04 现已接通PyTorch SAC与GPU replay并完成短训练，见 [单卡验收](torch_sac_acceptance.md)。远程 A100 仍需单独核验；本阶段未开始强化学习训练。
