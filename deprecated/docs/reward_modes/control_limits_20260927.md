# 可选执行目标范围与变化率限制

用户于 2026-09-27 允许短测软件范围裁剪及每帧目标变化限制，保留原机械限位与碰撞，择有效方案续训。此模块提供执行方式，不把候选限速称为硬件额定安全速度。

## 合同

`PianoTaskEnv(..., control_limits=None)` 和 `PianoScene` 同名参数默认 `None`，沿用原执行／动作观测路径。显式启用示例：

```json
{
  "schema": 1,
  "mode": "target_clip_slew_v1",
  "channels": {
    "R_Wrist_TX": {"min": -0.2, "max": 0.2, "max_velocity": 1.2},
    "L_Thumb_Rot2": {"max_velocity": 16.0}
  }
}
```

这些数字是本轮待整曲短测的候选，尚不能因为通过代码测试而认定有效。

- `min/max` 为已有物理关节坐标，平移用 m，转动用 rad；未提供的边界沿用原动作合同。
- `max_velocity` 为**执行器目标**变化率，平移 m/s、转动 rad/s；省略即该通道不限变化率。它不保证实际 qvel 不超限，也不限制碰撞力。
- 所有数字必须有限；速度必须大于 0；范围须满足原 `physical_min≤min<max≤physical_max`，并包含模型真正 reset 下发的 ctrl。
- 只能填写实际启用的手关节名；不允许 sustain 或已移除的 TZ/Yaw。未知字段／通道拒绝。
- 单侧边界也可单独提供；每个通道至少提供一个边界或变化率。

`canonical(config)` 返回保存的规范参数，`resolve(...)` 对照真实模型／动作／reset 合同补全索引、单位、完整范围、控制周期，输出 resolved 合同。`None` 对应两者均为 `None`。模型 XML、机械 range、策略动作范围和 Actor 输出维度保持原样；新合同须由 checkpoint 显式记录，跨执行合同迁移不得静默放行。

## 每步执行顺序

先按原策略动作映射得到请求物理目标 `u_req`，然后仅对配置通道应用：

`u_clip = clamp(u_req, min, max)`

`u_apply = clamp(u_clip, u_previous − max_velocity×control_dt, u_previous + max_velocity×control_dt)`

`u_previous` 读取这个环境上一帧实际下发的 `data.ctrl`，不是上一帧原始策略请求，更不是实际关节 qpos。执行器对新目标仍零阶保持 10 个物理子步，没有加子步插值或改 20/200 Hz。未配置通道原样通过。

例如 20 Hz 下 `max_velocity=16 rad/s` 允许每帧目标改变最多 0.8 rad。策略仍输出原 `[-1,1]`，没有把缩小后的范围重新铺满该区间，因此已有权重输出相同动作时，其原物理含义保持不变，再由执行层按需限制。

## 观测与 reset

启用时，原 51 维物理 `last_action` 观测记录 **APPLIED** 目标；新增变换没有增加 1181 维观测或 47 维策略动作。奖励中涉及动作命令的项也收到相同 APPLIED 目标。策略 replay 动作仍是实际选出的原始策略动作，不能用 APPLIED 物理数组替换 SAC 的动作坐标。

历史 task 的 reset_action 若不等真正 ctrl，仅在启用控制限制时改成实际 reset ctrl，填入 reset 历史最新一帧；其余历史清零语义不变。单环境 reset 只恢复该行目标及历史，其他环境保持原状态。由于 slew 依赖上一下发目标，若任务没有动作历史观测则拒绝启用 slew，避免未被观察的隐藏状态。

每个 step info 增加：

| 字段 | 形状／含义 |
| --- | --- |
| `raw_policy_action` | N×47（当前合同），策略原始动作 |
| `requested_physical_action` | N×51，原映射后的请求目标 |
| `applied_physical_action` | N×51，最终实际下发目标 |
| `control_clip_mask` | N×51 bool，静态范围是否改变请求 |
| `control_slew_mask` | N×51 bool，变化率是否改变静态裁剪结果 |

返回诊断拥有自己的张量，后续 step/reset 不会改写已存结果。滤波只在 `scene.step` 调用一次；task 不重复映射已处理目标。

## 本轮实测与候选依据

归档重建及动作数据见 `preview/outputs/diagnostics/joint_workspace_20260927/`。best 左拇指 Rot2 在 t=0.90 s 目标从 1.7265 变为 0.0115 rad，同帧 PIP 从约 0.068 变为 1.530 rad，并伴随 TY／Pitch 变化。随后 t=1.14–1.365 s 出现 A3 键底接触；Rot2 实际角被保持在约 0.834–0.957 rad，而目标低至 0.054 rad，65.2% 子步力限饱和。这是限速试验的依据，不证明单轴阶跃是唯一原因。

本轮初始候选只限制左拇指目标变化率，或分别限制腕部大阶跃，所有手指角度范围保持原样。软件 TX 试验范围 R[-0.2,0.2] m、L[-0.1,0.2] m 包含 best 原请求范围，不由目标琴键位置直接推导。候选仍需完整闭环回放，因为动作观测变化后策略会改变后续请求。

## 验证

- CPU：34 项通过，1 项 GPU 跳过。包括 `None` 身份合同、无效字段／固定轴／非有限数拒绝、reset 包含校验、先 clip 后 slew、实际前目标、局部 reset，以及既有奖励／任务／碰撞回归。
- 已在 GPU 分别启用和关闭物理 graph 验证：第一帧真实目标从 0.1 限到 0.15，原请求 0.5；观测和 ctrl 相同。局部 reset 后两环境下一目标为 0.15/0.20，另一环境物理／history 未改变，旧返回诊断没有被覆写。
- GPU 测试用保留同样物理接口的合成标量模型，验证接口与图执行，不证明真实策略改善。真实 best 各候选整曲闭环评估由主任务另行记录。
- 真实 V3.2 追加验收：8 个环境，每条路径 24 控制步，分别关闭／开启物理 CUDA graph。局部 reset `[1,4,7]` 时，其余世界的 qpos/qvel/ctrl、准备 mocap、warmstart、控制计数、前目标、动作历史和任务累计等 19 项状态逐 bit 不变；真实 reset ctrl、返回观测 APPLIED 51 槽与执行命令一致。TX/TY 目标最大变化 0.06 m、Pitch 0.2 rad、左拇指 0.8 rad，FP32误差不超过 `7.2e-8`；两路径都实际触发 clip 和 slew，graph 路径确认捕获。报告为 `preview/outputs/diagnostics/joint_workspace_20260927/real_model_control_acceptance.json`。这是执行合同验收，不是无碰撞或策略改善证明。
- 独立恢复／CLI／质量评估回归：`test_control_checkpoint_contract`、`test_training_quality_control`、`test_control_limits` 共 18 项，其中 17 项 CPU 通过、1 项 GPU 测试因本轮 CPU 命令显式屏蔽设备而跳过。旧 None 身份兼容、严格 resolved 合同、仅显式 networks-only finetune 可改变执行合同均通过。

命令：

```bash
CUDA_VISIBLE_DEVICES='' training_mjlab/.venv/bin/python -m unittest \
  tests.test_control_limits tests.test_task_conditioned_gesture \
  tests.test_humanoid_finger_gesture tests.test_collision_diagnostics \
  tests.test_paper_rewards tests.test_mjlab_task
training_mjlab/.venv/bin/python -m unittest tests.test_control_limits.GPUControlLimitsTest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  training_mjlab/.venv/bin/python scripts/verify_control_limits_real.py \
  --checkpoint training_mjlab/outputs/twinkle_quality_continue-20260926-194555-20260926-194556-402944/best.pt \
  --limits configs/control_limits/combined_20260927.json \
  --output preview/outputs/diagnostics/joint_workspace_20260927/real_model_control_acceptance.json
```

最后一条是需排期独占 GPU 的诊断命令，输出已存在时会拒绝覆盖。它使用父检查点相同模型／任务参数重建，并严格检查导出 scene/task 哈希相同；不更新网络，也不写训练检查点。

TODO(V32-START-01/02)：软件目标限速不是动态无碰撞保证，需核验键底／键侧持续时间、200 Hz 错音、F1、力限和延迟；候选可能牺牲快速音符或手型转换。论文 §2.6 p.8 的子步插值与本次每控制帧 slew 不是同一方法，未因论文自行引入新公式或改机械参数。
