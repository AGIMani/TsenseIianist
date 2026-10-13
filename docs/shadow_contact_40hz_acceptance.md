# Shadow 接触参数、40/400 Hz 与 solimp 预览验收

实施：2026-09-18；文档核对完成：2026-09-19。用户要求接触参数同步原版 Shadow，并明确选择控制 40 Hz、物理 400 Hz、保留 0.5 秒前瞻。本次只替换相应物理／时序配置；TsenseI 几何、驱动器、腕部范围、奖励开关和未知映射约定继续有效。

## 1. solimp 是什么

MuJoCo 用软约束近似接触。`solimp` 描述约束阻抗随穿透深度变化的曲线，并不是单独一个摩擦或刚度系数。阻抗越接近 1，求解器越强地要求满足约束；实际穿透还受 `solref`、步长、驱动力、几何和求解精度影响，不能保证零穿透。

| 五个值 | 含义 |
| --- | --- |
| `d0` | 接触刚建立时的阻抗；预览主滑块 |
| `dwidth` | 达到过渡宽度时的阻抗 |
| `width` | 过渡宽度；本项目几何接触为 m，琴键转动限位为 rad |
| `midpoint` | 过渡曲线拐点位置，范围 0～1 |
| `power` | 曲线形状指数 |

`solref` 控制约束的恢复时间与阻尼。两物体接触时还会合并双方参数：本配置双方 `priority=0`、`solmix=1`，因此手—琴键的 solimp 为逐项平均，即 `[0.7, 0.97, 0.00055, 0.5, 2]`。调琴键的 `d0` 会改变这一平均值，不等于直接设置最终接触阻抗。详见 [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters) 与 [contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。

## 2. 当前参数和保存位置

源配置：[contact_settings.json](../tsensei_v3_2_pianist/assets/contact_settings.json)。加载、校验、备份和 GPU 更新逻辑位于 [physics_settings.py](../tsensei_v3_2_pianist/physics_settings.py)。

| 项目 | 新配置 |
| --- | --- |
| 控制／物理 | 40 Hz／400 Hz，即 0.025／0.0025 s，每控制步 10 个物理子步 |
| 积分器／impratio | implicitfast／10 |
| 手 solref／solimp | `[0.005, 1]`／`[0.5, 0.99, 0.0001, 0.5, 2]` |
| 琴键与琴箱 solref／solimp | `[0.005, 1]`／`[0.9, 0.95, 0.001, 0.5, 2]` |
| 双方 friction／priority | `[1, 0.005, 0.0001]`／0 |
| 琴键限位 solref／solimp | `[0.02, 1]`／`[0.9, 0.95, 0.001, 0.5, 2]` |
| 未来前瞻 | 20 步＝0.5 秒；当前帧另计 |
| 观测／动作维数 | 2070／51；旧观测为 1180 |
| 小星星任务 | 315 个控制步，7.875 s；有指法、错键不提前终止 |

原版钢琴用 `solref[0] = 2 × physics_dt`，其 200 Hz 下是 0.01 s，本次按同一规则在 400 Hz 下得到 0.005 s。手接触与琴键限位的其余参数按原版对齐，依据是[之前的原版对照](physics_shadow_comparison.md)。不替换 TsenseI 手为 Shadow 碰撞形状，也不照搬 Shadow 执行器。GPU FP32 求解容差仍有 MuJoCo-Warp 的最低 1e−6 限制。

论文 §2.4、2.6（pp.6–8）同样采用 40/400 Hz。本次用户决定覆盖此前保留 20 Hz 的时序决定，但没有授权子步动作插值、奖励自动乘 dt 或新的映射；这些没有联动启用。若用户已选择 `scale_by_dt` 或差分奖励，它们按实际新 dt 计算。SAC gamma 未自动重标定，因此其按秒折扣范围会随频率变化。

## 3. 预览和滑块验收

以下命令从独立项目根目录运行。

```bash
# 源模型：默认打开独立 solimp 窗口
.venv/bin/python preview/scripts/preview_pose.py

# 不显示接触面板
.venv/bin/python preview/scripts/preview_pose.py --no-contact-controls

# 已冻结的新 GPU 场景：显式打开实验用面板
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py --contact-controls
```

面板对象可选 `piano`（琴键／琴箱）、`hand`（双手）、`key_limit`（琴键限位）。主滑块调 d0，其余四个滑块在“高级 solimp 设置”中；下方显示实际手—键合并值。滑动后实时更新模型，继续运行物理才能观察接触运动变化。建议先复位、固定 Control 目标，再比较不同值的下压效果。

2026-09-24补充：两个预览入口现在从实际编译模型提取参数并与原版Shadow参考逐项比较，启动终端显示`MATCH`／`DIFF`；CPU `pose.json`保存完整差异、合并后的手—键参数和`refsafe`时间常数。面板新增“恢复原版 Shadow solimp”，一次恢复手、钢琴和琴键限位三组阻抗。该同步只覆盖接触环境参数；不修改Wrist、执行器、碰撞形状或时序。40/400 Hz默认包和20/200 Hz无臂包均已实测为`MATCH`。

- **P 仍然只保存初始姿态**，不会保存接触参数。
- 源模型窗口点击“保存接触配置”写入上述 JSON；旧文件备份到同级 `contact_settings_backups/`。如果文件被其他编辑器改过，会拒绝覆盖并要求重新打开。
- GPU 预览面板保存为 `preview/outputs/contact_settings/` 下独立实验 JSON；不会覆盖训练时的冻结包。回放检查点时改接触参数属于实验，结果不能当原训练环境评价。
- 关闭面板不会关闭主预览。源预览无窗口模式不创建 Tk 面板。

面板截图：[solimp_panel.png](../preview/outputs/solimp_panel.png)。

## 4. 新训练与旧检查点

已生成的新默认冻结包：

- `training_mjlab/outputs/scene_shadow_40hz400hz`
- `training_mjlab/outputs/task_twinkle_shadow_40hz400hz`

新训练默认读取这两份。旧 `scene`／`task_twinkle` 保留；评价、策略预览、视频导出和轨迹换视角会按检查点／录制报告的散列查找原场景和任务。找不到时明确报错，可显式传 `--scene`、`--task`。**1180 维旧网络不能直接用于 2070 维新任务，需要新训练。**

保存 JSON 不会修改已经导出的训练包。让保存后的配置进入训练，必须导出一对新包；例：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/export_mjlab_scene.py \
  --no-gravity-compensation --output training_mjlab/outputs/scene_contact_trial
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_task.py \
  --scene training_mjlab/outputs/scene_contact_trial \
  --midi TwinkleTwinkleRousseau --lookahead 20 \
  --output training_mjlab/outputs/task_contact_trial
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --scene training_mjlab/outputs/scene_contact_trial \
  --task training_mjlab/outputs/task_contact_trial \
  --num-envs 256 --vector-steps 19532
```

目录须不存在；修改后需另取目录名。GPU 实验 JSON 如要作为源配置，应先明确核对并替换源 JSON，再做上述导出。导出会核对源模型与冻结场景的接触参数，不能混配。

## 5. 已做的验证与未通过项

| 检查 | 结果 |
| --- | --- |
| 奖励、任务、SAC、加速、接触滑块、最佳选择、相机单元测试 | 34 项通过 |
| 钢琴颜色与旧冻结场景硬限位回归 | 2 项通过；旧硬限位通过不代表新软接触符合旧上限 |
| 新源模型／GPU 预览窗口 | DISPLAY=:10 均打开与关闭成功；真实 Tk Scale 事件、应用、临时文件保存及恢复通过 |
| GPU 滑块 | 原地更新 Warp 数组，CUDA graph 指针保持；后续图执行状态有限 |
| 新任务完整 CPU／GPU 对照 | **269/270 通过，严格物理轨迹对照未全通过** |
| 新 SAC 短训练 | 4 环境×6 步，24 transition、10 更新、8 截断、2 次评价，2070 维网络保存成功 |
| 新检查点完整有声回放 | 自动找到新包，315 控制步、3150 音频子步，40 fps、356 帧／8.9 s |

严格对照的失败项为 `press_physical_rollout`：观测最大差 0.1192299，超过现有 0.1 门槛；奖励最大差 0.001091，平移最大差 0.132354 mm、转角最大差 0.007936917 rad，16 个环境累计 288 个琴键激活位差异。中性轨迹约 1e−7 数值差；固定状态规则／分项奖励、局部 reset、完整 315 步结束逻辑与 CPU 指标对照通过。额外核对 CPU 参考的键状态和激活与保存 qpos 完全一致，未发现参考导出错帧。没有放宽阈值或隐藏失败，不宣称 CPU／GPU 物理等价。

恢复 Shadow 接触后比旧硬接触更软。CPU 固定下压 2 秒测得最大手—键穿透 **3.05335 mm**、琴键限位折算键端超行程 **1.42037 mm**，状态有限。这是该姿态／目标下的测试值，不是所有策略的误差上界。旧版本“到底后近似硬止挡”的误差上限不适用于新配置。

证据目录：

- [严格任务对照报告](../training_mjlab/outputs/shadow40_task_acceptance/report.json)
- [下压接触测量](../training_mjlab/outputs/shadow40_task_acceptance/contact_measurement.json)
- [短训练摘要](../training_mjlab/outputs/shadow40_sac_smoke/summary.json)
- [新配置短测策略有声视频](../preview/outputs/shadow40_smoke_replay/replay.mp4)／[报告](../preview/outputs/shadow40_smoke_replay/report.json)；这是接口验收策略，不是最佳演奏。

复查严格对照命令（会复现当前失败项）：

```bash
training_mjlab/.venv/bin/python scripts/validate_mjlab_task.py \
  --scene training_mjlab/outputs/scene_shadow_40hz400hz \
  --task training_mjlab/outputs/task_twinkle_shadow_40hz400hz \
  --num-envs 16 --output training_mjlab/outputs/shadow40_recheck
```

TODO(V32-06)：在真实手／接触目标下标定软接触和限位误差。TODO(V32-SAC-03)：继续定位 CPU／GPU 触键激活差异及既有跨次回放漂移。没有启动新的 5M 训练，没有测得这次时序更改的长期吞吐或演奏改善，不能沿用旧训练速度／最佳 F1 作为新配置结果。
