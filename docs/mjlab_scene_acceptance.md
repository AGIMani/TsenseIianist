# MJLab 第一阶段：场景迁移与验收

本阶段提供当前 V3.2 双手＋钢琴的完整冻结场景、MJLab 单环境物理接口、自动验收和交互预览。物理通过 `mjlab.sim.Simulation` 调用 MuJoCo Warp，在 CUDA 上执行；当前音乐任务见 [Torch任务验收](mjlab_task_acceptance.md)，SAC 训练属于后续阶段。

> 2026-09-17 后续更新：默认场景使用 0.0005 s × 100 子步、implicitfast 和 −25 mm 腕部范围。本文末尾的首次验收数值保留为历史结果；当前参数和报告以 [批量验收](mjlab_batch_acceptance.md) 为准。

## 1. 文件与环境

- `scripts/export_mjlab_scene.py`：在原预览环境中导出场景。
- `tsensei_v3_2_pianist/mjlab_env/bundle.py`：资产散列核验、按名称恢复初值和动作映射；不依赖母仓库。
- `tsensei_v3_2_pianist/mjlab_env/scene.py`：MJLab 物理接口，支持 num_envs 和指定 env_ids 的独立 reset。
- `scripts/validate_mjlab_env.py`：CPU/GPU 相同动作轨迹对照；退出码非零表示失败。
- `preview/scripts/preview_mjlab.py`：GPU 物理预览，Control 滑块直接驱动原有位置执行器。
- `training_mjlab/.venv`：独立 MJLab 环境；原 `.venv` 与 `training/.venv` 继续使用各自依赖。

以下命令均在独立项目根目录执行。当前机器已完成的安装与验收结果见末节；复制到其他机器时，运行 `bash training_mjlab/setup.sh`，按 `training_mjlab/requirements-tested.txt` 安装相同 Python 3.12 / Linux x86_64 / CUDA 12.8 软件组合。系统需有兼容 NVIDIA 驱动。

## 2. 冻结当前姿态

```bash
MUJOCO_GL=egl .venv/bin/python scripts/export_mjlab_scene.py \
  --output training_mjlab/outputs/scene
```

默认重力补偿关闭，与当前预览一致。训练候选需要补偿时，另导出一份：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/export_mjlab_scene.py \
  --gravity-compensation --output training_mjlab/outputs/scene_gravity
```

导出包含 `scene.xml`、所有模型资源和 `manifest.json`（qpos、qvel、ctrl、动作上下限／名称、安装位置、指尖参考位置、求解器选项、源文件散列）以及 `cpu_reference.npz`（原环境版本下的 CPU 参考轨迹）。不覆盖已有非空目录；重新保存姿态后使用新目录，再将后续命令的 `--scene` 指向它。导出时会重新加载 XML 并比较指尖世界坐标。

运行端只读取冻结场景。编辑原手常量不会热更新已导出的模型。原 CAD、母仓库或 Composer 无需出现在 MJLab 运行端。默认初值来自当前 `INITIAL_ANGLES_BY_SIDE`，左右手 pitch/yaw 方向与模型保持一致。

## 3. 自动验收

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_env.py \
  --scene training_mjlab/outputs/scene \
  --output training_mjlab/outputs/acceptance --render
```

首次运行需要编译 Warp CUDA 内核和捕获计算图，启动可能明显慢于后续运行；终端的编译输出不是训练进度。无桌面 GPU 服务器可使用 EGL。删除 `--render` 只取消 PNG，不取消任何物理验收。

**通过条件**：终端显示 `Acceptance: passed`，程序退出码 0，`report.json` 中所有 `checks` 的 `passed` 为 true。

检查包括：

1. nq/nv=138、nu=50、动作51；初值、双手根位姿、十个指尖位置和 ctrl 恢复正确，初值误差小于 1e-6。
2. 左右食指及三个腕部平移、pitch、yaw 的控制都实际产生预期方向的关节运动；动作边界与原位置控制一致。
3. 使用相同物理目标比较 CPU 与 GPU 短轨迹。一般控制的最大滑动误差 <0.5 mm、铰链角误差 <0.01 rad；专门按键／松键轨迹分别 <2 mm、<0.05 rad。两种误差分开报告，不把米与弧度混为一种物理量。
4. 双腕下移 0.025 m，同时中指 MCP 在初始值上增加 0.1 rad，产生真实 Warp 手—键接触，CPU/GPU 激活过的琴键集合一致且非空；抬手后全部松键。验收不直接改写琴键 qpos，也不禁用触键碰撞。
5. 踏板动作能写入独立任务状态；运动后 reset 能恢复 qpos、qvel、ctrl 和踏板。

`report.json` 记录各项误差、按键编号（钢琴内的 0～87 索引）、版本、GPU 和用时；`trajectories.npz` 保存全部动作与 CPU/GPU qpos。`initial.png` 和 `press.png` 渲染的是 GPU 状态的 CPU 拷贝，渲染器只做 forward，不负责推进物理。

这是场景迁移验收阈值，不能据此声称长轨迹完全一致或已学会弹奏。CPU 对照使用 MJLab 环境中的同版本 MuJoCo；同时对照导出时记录的原版本 CPU 轨迹、位姿，核对跨版本物理、几何和初值。

## 4. 目视验收

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --scene training_mjlab/outputs/scene --paused
```

- 初始姿态应与原预览一致；按 **Space** 开始／暂停，**R** 恢复冻结初值。
- 在右侧 Control 修改 `R/L_Index_MCP`，检查对应食指响应；修改 `R/L_Wrist_TX/TY/TZ`，检查整手位移。
- 两手 yaw 给相同符号应同向，pitch 给相同符号应反向。局部轴复合旋转不能仅凭屏幕左右判断，先复位并单独改变一个角。
- 原预览的姿态保存仍在原脚本中使用；本脚本用于验证冻结模型的 GPU 行为。

自动演示按键和松键：

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py --demo
```

演示循环为初始停留→双腕下压并弯曲中指→松开，可加 `--duration 10` 自动关闭。演示时 Control 每步被脚本目标覆盖，手动验收请去掉 `--demo` 和 `--bottom-out`。

## 5. 当前边界和下一步

场景包含完整网格、钢琴弹簧／阻尼、已有掌部碰撞排除和原 XML 执行器。导出保留物理选项；加载器遇到运行版本不支持的非默认选项会报错，避免默默采用不同设置。

本接口只提供物理状态、按键激活和踏板值；不提供乐谱推进、音乐奖励、episode 终止或 SAC 训练观察。16／64 环境的独立 reset／接触容量检查已通过，见 [第二阶段验收](mjlab_batch_acceptance.md)；音乐任务已另由 `PianoTaskEnv` 提供，见 [任务接口与验收](mjlab_task_acceptance.md)；Torch SAC 与 GPU replay 已另通过 [单卡功能验收](torch_sac_acceptance.md)。V32-04／05／06／07／10 等现有模型标定限制继续保留。

## 6. 首次单环境验收历史结果（2026-09-17，旧参数）

- NVIDIA RTX PRO 6000 Blackwell；Python 3.12.3、MJLab 1.6.0（指定提交）、MuJoCo/MuJoCo Warp 3.11.0、Warp 1.17.0、PyTorch 2.10.0+cu128。原导出环境为 MuJoCo 3.13.0。
- **45 项 GPU 检查通过；4 项 CPU 自动测试通过；pip check 通过。**
- 初始 qpos 最大误差 1.32e-8；按键／松键轨迹相对同版本 CPU 最大滑动误差约 0.0481 mm、铰链角误差约 0.00243 rad。相对原预览版本 CPU 分别约 0.0416 mm、0.00233 rad。
- CPU 与 GPU 激活集合均为 `[33, 54]`，松开后为空；实际 Warp 接触对包含双手食指／中指等末节与琴键。控制周期边界采样最多 12 个接触、130 个约束，分配容量分别为 1024／4096；这不代替下一阶段批量负载与每子步容量验证。
- 首次内核编译与加载约 92.3 秒；缓存命中后的加载约 1.64 秒，整套验收含 PNG 约 5.49 秒。这是本机单场景验收耗时，不是训练加速比。

实际 NVIDIA/X11 窗口自动演示 4 秒后正常退出（退出码 0），终端记录按键／松键循环；未代替用户手动拖动鼠标验收。无 DISPLAY 的远程终端使用自动验收 `--render`，交互窗口应从桌面终端打开。

详细、可随项目保存的数值见 [验收记录](mjlab_scene_validation.json)。本地完整输出位于 `training_mjlab/outputs/acceptance/`。

接口依据：[固定提交的 MJLab Simulation](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/sim/sim.py)。当前选择直接导入完整 `MjModel`，通过项目配置保留导出的 MuJoCo 选项；`PianoTaskEnv` 已继续使用此处冻结的场景，并单独保存相配的任务合同与参考轨迹。
