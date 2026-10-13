# 碰撞振荡定位与统一实体接触（2026-09-24）

## 结论与本轮决定

用户要求以 `~/oldproj/DaxianPianist` 的 `DAXIAN_DISPLAY=:10 bash preview_model.sh --training-script run.sh` 为参考，定位振荡原因，三类接触使用统一的实体碰撞方案。此决定覆盖同日较早的“同手／跨手只检测、无反力”默认设置。

**已复现的主要原因是轻量 CAD 连杆、缺失电机等效惯量和偏大的位置增益共同造成的接触数值振荡。** 当前使用 MuJoCo 原生位置驱动，没有另一套预览 PD 程序；不启动界面、不运行策略或奖励，固定 Control 目标同样可以复现。碰撞反力本身是正确的物理功能，关闭反力只移除了触发条件。

- 给双手共 40 个手指转动关节设置 `armature=0.0002 kg·m²`。
- 手指 `Kp` 从 3 改为 1 N·m/rad。阻尼仍为 0.05，力矩限幅仍为 ±0.6 N·m；腕部参数独立保留。
- 预览默认恢复同手、跨手、手—琴键真实接触。训练包装脚本也恢复实体接触并指向新冻结包，碰撞奖励公式和权重保持。

这些是仿真适配参数，**不是 TsenseI 电机实测值**。Shadow 的手指 Kp 随关节变化，本轮采用其范围内经对照验证的 1，没有声称逐关节复制 Shadow。

## 原版实际参考配置

只读解析 Shadow 预览脚本，并在已有容器中实例化它实际使用的环境，未修改该项目。根脚本的 `--training-script run.sh` 指向 `robopianist-rl/run.sh`，实际是 20 Hz 控制／200 Hz 物理，启用 reduced action space、gravity compensation、primitive fingertip collisions、TX/TY。

| 项目 | V3.2 修改前 | 指定的 Shadow 环境 | 本轮 V3.2 |
| --- | --- | --- | --- |
| 手指 armature | 0 | 0.0002 | 0.0002 |
| 手指 Kp | 统一 3 | 按关节 0.5／1／1.5 | 统一 1，暂定 |
| 手指 damping | 0.05 | 0.05 | 0.05 |
| 手指 frictionloss | 0 | 0.01 | 0 |
| 手指碰撞几何 | CAD mesh 凸包 | 多种简单几何；指尖启用 capsule | 保留 V3.2 几何 |
| 手指、跨手、琴键接触反力 | 前一方案分别关闭手—手 | 物理接触 | 三类均开启 |

参考文件：`preview_model.sh`、`robopianist-rl/preview_model.py::training_args/main`、`robopianist-rl/run.sh`、`robopianist-rl/train.py::get_env`、`robopianist/models/hands/third_party/shadow_hand/right_hand.xml`、`robopianist/models/hands/shadow_hand.py`。原版有效配置记录在 `preview/outputs/contact_stability_20260924/shadow_reference.log`，运行时不依赖原版项目。

V3.2 修改前手指 `dof_M0` 约为 `1.19e-7～3.22e-5 kg·m²`。`armature` 补充的是关节运动带来的附加惯量，常用于电机转子经减速器折算的惯量；它改变对力的加速响应，不是碰撞屏蔽或位置锁定。[MuJoCo 官方说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-armature)。本轮保留 URDF 原始连杆质量和惯量。

## 如何区分原因

1. 固定目标保持 20 秒，观察最后 2 秒，并逐物理步采样关节速度。无界面、无策略、无奖励也抖，排除了显示帧率和奖励回推作为本次振荡的必要条件。
2. 关掉碰撞后同一目标稳定，表明是接触激发的动力学问题，但这种做法不能保留真实接触。
3. 最初单因素试验中，改 `implicitfast` 没有消除振荡；只降低 Kp、只加摩擦或提高阻尼也没有完整解决。减小物理步长有帮助，但用户指定的 200／400 Hz 保留。
4. 加 `armature=0.0002` 后同手振荡大幅下降；进一步以完整 XML 重新编译复核，避免只修改运行时数组而未重算惯量相关常量的影响。
5. 12 cm、物理移除 TZ 的旧冻结姿态下，左右拇指持续挤压仍有残余振荡。该场景只补 armature 时 CPU 手指速度 RMS 为 0.136 rad/s；再把 Kp 改为 1 后降至 0.000786 rad/s。加入 Shadow 的 frictionloss=0.01 未改善这个案例，故没有追加该变化。

原始单因素结果：`sweep.json`；重编译结果：`compiled_checks.json`、`compiled_arm_kp1_checks.json`；跨手参数实验：`cross_grid.json`，均在本次诊断输出目录。早期 `cause_checks.json` 属运行时参数实验，最终数值以完整重编译的验收结果为准。

## 三类接触现在如何工作

| 接触 | 检测 | 物理反力 | 训练奖励 |
| --- | --- | --- | --- |
| 同手不同手指／部件 | 是 | 是 | 启用 self_collision_reward 时按原规则扣分；包装脚本权重 0.2 |
| 左右手掌／手指 | 是 | 是 | forearm 无碰撞 +0.5，有碰撞 0 |
| 手与琴键 | 是 | 是 | 原按键／指法等任务奖励，排除在手自碰撞扣分之外 |

三类接触统一进入 MuJoCo 约束求解器，奖励只读取碰撞结果。保留原有父子体过滤及两对掌部／拇指装配排除，不把正常安装相邻面加入碰撞。

“统一”指相同的实体接触机制；手材料与琴键材料仍按 Shadow 的不同参数及混合规则求解。手 `solref=[0.005,1]`、`solimp=[0.5,0.99,0.0001,0.5,2]`；琴键 `solref=[2*physics_dt,1]`、`solimp=[0.9,0.95,0.001,0.5,2]`。未用负 margin/inactive pair 隐藏本轮默认碰撞。

## 实际验收

固定目标 20 秒，下面是最后 2 秒所有手指中最大的角度峰峰值。CPU 使用完整编译模型、20/200 Hz、12 cm 手根、TZ 关节真正移除。修改前后姿态／动作完全相同。

| 接触案例 | 修改前 CPU | 修改后 CPU | 修改后 GPU |
| --- | ---: | ---: | ---: |
| 同手四指挤压 | 33.6623° | 0.007267° | 0.007319° |
| 左右手持续靠拢 | 2.2213° | 0.003114° | 0.001981° |
| 手指压琴键 | 2.6798° | 0.000116° | 0.0000032° |

![固定目标下的接触振荡对照](../preview/outputs/contact_stability_20260924/contact_stability_comparison.png)

图中每类选择修改前摆幅最大的同一个关节，分别减去各自时间均值；用于展示振荡，不能用来比较最终接触平衡角度。

- 新 GPU 场景的同手／跨手／手—键法向力峰值分别约 3.77／6.18／1.29 N，读取 GPU `efc.force`，没有用 CPU 重算力替代。证明稳定并非来自取消反力。
- GPU 三环境测试前段使用控制周期 CUDA graph，最后 2 秒逐物理子步记录，避免 20 Hz 采样漏掉高频振荡。自碰撞案例实际 reward=-0.4，跨手案例 forearm=0，键接触案例不产生同手扣分。
- 源 Composer 预览步进和额外 forward 也分别在 400／200 Hz 验证三类接触，6 例通过。源姿态与旧冻结包不同，持续挤压时仍有低速调整：跨手最后 2 秒摆幅约 0.475°／1.19°，速度 RMS 约 0.00119／0.00300 rad/s。不能把低速接触调整称为绝对静止。
- 源预览 headless 使用 NVIDIA 渲染成功，JSON 确认三类均 `physical`、Kp=1、armature=0.0002。旧检测模式的 3 项 CPU/GPU 回归仍通过。
- 额外检查新增益仍可压响琴键：无 TZ 的新 CPU 包，右手 TY=0.04、Pitch=0、Index MCP=1.2／PIP=0.3／DIP=0 rad，其他初始目标保留，保持 1 秒后实际激活 key index 52（按冻结任务阈值）。这是物理按键测试，不是策略演奏结果；记录在 `press_pitch_checks.json`。
- CPU/GPU 的最终接触组合与力值仍不完全一致，本次不是严格逐轨迹等价或训练收敛验收。

最终报告：`stable_cpu.json`、`stable_gpu.json`、`baseline_frozen_cpu.json`、`preview_regression.json`。其中 baseline 命令按设计未达到稳定阈值并返回失败，保留作为修复前对照。最初 hand-key 测试在旧 TY=0 姿态没有触键，已把测试目标明确设为 TY=0.04，并要求真实接触力大于 0.01 N 后才通过。

## 使用与复验

关闭旧窗口，重新启动当前源码预览；旧进程不会热更新模型：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
.venv/bin/python preview/scripts/preview_pose.py --no-arms
```

Control 中先保持四指 MCP/PIP/DIP 为 0.6 rad，让 Index/Ring SW 往上、Mid/Pinky SW 往下，制造接触后松开鼠标保持目标，观察是否仍持续快速抖动。原命令默认的两种 detection-only 开关现均为 false；只有显式传入时才启用历史模式。附带机械臂仍属运动学显示，本轮三类接触结论不扩展为 M6 实体动力学。

新训练包装脚本 `bash scripts/train_twinkle_h12_no_tz.sh` 已指向：

- `training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/scene`
- `training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/task`

保留 49 动作、20/200 Hz、既有奖励、乐谱和旧冻结初始姿态，只更新手指惯量／增益及关闭 detection-only 开关。**没有启动长期训练**。源预览仍采用当前源姿态及默认 40/400 Hz，二者共享新手指参数；旧冻结包／检查点保留原物理参数，不自动套用本次修复。新的场景散列阻止错误混用。

复验：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/validate_preview_contact_stability.py \
  --output preview/outputs/contact_stability_preview_recheck.json

MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_contact_stability.py \
  --scene training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/scene \
  --task training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/task \
  --backend gpu --disable-wrist-tz \
  --output preview/outputs/contact_stability_gpu_recheck.json
```

用 `.venv/bin/python`、省略 `--backend gpu` 可运行冻结场景 CPU 检查。复制项目后如未携带新输出包，可用 `scripts/export_contact_stability_bundle.py --scene <旧场景> --task <旧任务> --output <新目录>` 从自带冻结包派生；脚本拒绝覆盖原包，不依赖 Shadow 工程。

## 参数位置与保留 TODO

- `tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py`：`PROVISIONAL_FINGER_ARMATURE`、`PROVISIONAL_FINGER_KP`；`urdf_model.py` 应用于 40 个手指关节，未写入腕部或琴键。
- `preview/scripts/preview_pose.py`：实体接触默认、启动输出及 JSON 参数记录。
- `mjlab_env/task_bundle.py`：新任务导出校验 armature／阻尼／关节摩擦，防止新任务悄悄匹配旧动力学场景。
- **V32-04**：真实电机转子惯量、减速比、Kp／阻尼／力矩仍待实测标定；本轮参数会改变动态跟踪和接触平衡，应新训练。
- **V32-06**：凸包几何与装配排除仍需验收。尚未证明所有滑块极限组合均无振荡或无穿透。
- **V32-SAC-03**：CPU/GPU 接触细节及策略复播差异仍保留。本次没有重新测完整 SAC 的 F1／收敛。
