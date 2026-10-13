# V3.2 预览与训练启用手部重力补偿（2026-09-24）

## 当前行为

- 源姿态预览 `preview/scripts/preview_pose.py` 默认 `--gravity-compensation`；可用 `--no-gravity-compensation` 作对照，实际开关写入 `pose.json`。经下文同步后，当前手根高度 0.12 m、TY=0、左右 Yaw=0.25 rad，时序默认 20/200 Hz；源预览仍可显示仅运动学 M6 机械臂，补偿只给手部 body。
- 无检查点的 GPU 预览 `preview/scripts/preview_mjlab.py` 默认读取 `training_mjlab/outputs/contact_stable_h12_20hz200hz_gravcomp_20260924/scene`。冻结场景的手部初值、时序及补偿已与当前源预览对齐；源预览加 `--no-arms` 更接近无臂训练场景。
- `scripts/train_twinkle_h12_no_tz.sh`（49 动作）及其 `train_twinkle_h12_no_tz_no_yaw.sh`（47 动作）在2026-09-24 reset修复后改从当前源码缓存构建同姿态 scene/task；固定包继续用于旧检查点。Torch SAC 命令行不显式指定 scene/task 时仍默认选历史固定包。新包装场景仍为无臂、20 Hz 控制／200 Hz 物理、12 cm 手根、0.5 秒前瞻／1180 维观测；保留 `mixed_v1`、同手碰撞权重 0.2 和原 SAC 参数。运行时分别物理删除 TZ，或同时删除 TZ/Yaw。
- 源场景导出 `scripts/export_mjlab_scene.py` 和 `mjlab_env.bundle.export_scene()` 默认开启手部重力补偿；导出脚本可显式使用 `--no-gravity-compensation` 生成历史对照场景。已经冻结的旧包、旧训练及其检查点不受默认值变化影响。

新场景仅把左右手共 44 个 body 的 MuJoCo `gravcomp` 从 0 设为 1。编译模型的质量、惯量、几何、关节、执行器、接触参数及 20/200 Hz 选项保持一致；冻结 qpos/ctrl、乐谱、观测、奖励及 SAC 配置也保持旧小星星合同。重力补偿会改变运动轨迹和接触，因此旧检查点不能在新场景下直接续训；检查点散列会阻止混用。`mjlab_env.paths` 现可递归定位嵌套冻结包，旧检查点仍能匹配旧包。

## 生成和预览

当前新包位于 `training_mjlab/outputs/contact_stable_h12_20hz200hz_gravcomp_20260924/{scene,task}`。先用 `scripts/export_mjlab_scene.py --control-hz 20 --physics-hz 200` 生成开启重力补偿的**模板**，再用 `scripts/derive_gravity_scene.py` 从旧冻结 scene 派生新 scene，最后用 `scripts/derive_gravity_task.py` 从旧 task 派生新 task。派生脚本会验证两份编译模型除 `body_gravcomp` 外的关键数值合同以及初始 qpos/ctrl；新场景重新生成 CPU 场景参考轨迹。旧 task 的 CPU 物理参考轨迹不适用于新重力设置，新 task 不沿用它。

本次曾生成 `training_mjlab/outputs/scene_gravcomp_template_20260924`，仅作派生参考，**不是本轮训练场景**。它生成时的源码 TY/Yaw 初值为 0.04/0.15，而训练冻结场景是 0/0.25；首次尝试派生 task 时初值一致性检查据此拒绝。该历史模板已按[后续快照清理](source_bundle_cache_20260924.md)删除。

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
MUJOCO_GL=egl .venv/bin/python preview/scripts/preview_pose.py --headless --no-arms --output preview/outputs/pose_gravcomp.png
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py --paused
bash scripts/train_twinkle_h12_no_tz_no_yaw.sh --num-envs 256 --seed 42
```

如需复制独立项目到其他机器并运行以上冻结训练脚本，需一并复制新 `scene/`、`task/` 及其资源；普通源码安装可以排除整个 `outputs/`，但冻结训练入口不能缺少这两个包。

## 验证与运行

- `MUJOCO_GL=egl .venv/bin/python preview/scripts/preview_pose.py --headless --no-arms --output /tmp/v32-gravcomp-preview-20260924.png` 成功渲染；终端显示 `Hand gravity compensation: on`，Shadow 接触参数核对 `MATCH`。
- `derive_gravity_scene.py` 核对新旧编译模型的关键数组并验证 44 个 body 开关；`derive_gravity_task.py` 验证乐谱／动作／时间／初值合同，新 task 加载成功。对 47 动作结构性关闭 TZ/Yaw 后，新旧模型均为 nq=134、nu=46，手部 gravcomp body 分别为 44／0。GPU 以新包完成 2 环境×4 向量步、8 条采样、3 次 SAC 更新及检查点保存，47 动作接口通过。
- 新 47 动作小星星实验从头启动：`training_mjlab/outputs/sac-twinkle-no-yaw47-gravcomp-20260924/`；同级 `.launch.json` 保存准确命令，`.log` 为进度，`config.json` 为实际配置。启动时 PID 270182；配置确认 47 动作／1180 观测、目标熵 −23.5、新 scene/task 散列，并已进入 SAC 更新和首次完整评估。目标采样 5,000,192 条；此记录仅证明启动及短程运行，最终结果以 `summary.json` 为准。49 动作脚本已切换，但本次未另开 49 动作长期实验。
- 同步前的测试记录：`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_mjlab_bundle.py tests/test_contact_settings.py -q -k 'not gpu'` 为 8 passed、1 deselected。当时源码姿态 TY=0.04 m、Yaw=0.15 rad，固定目标 MCP/PIP=0.85 rad 能激活键 54；同步后改用新姿态下已验证的右食指目标，见下文。新旧检查点的散列查找分别解析到新旧冻结包。

## 源码与当前冻结场景同步

2026-09-24 按用户要求，以**正在训练的冻结 scene** 为基准同步后续源码入口，没有改写冻结 scene/task 或正在运行的进程。

- `models/hands/tsensei_hand_constants.py`：左右手安装高度从 0.10 调至 0.12 m；两侧 TY 从 0.04 调至 0，Yaw 从 0.15 调至 0.25 rad。其余关节初值和腕部范围不变。
- `preview/scripts/preview_pose.py`、`scripts/export_mjlab_scene.py`、`mjlab_env.bundle.export_scene()`：当前入口默认 20 Hz 控制／200 Hz 物理并继续开启手部重力补偿；源预览与导出脚本可显式指定 `--control-hz 40 --physics-hz 400` 查看／生成历史时序。导出默认目录改为 `scene_source_20hz200hz_gravcomp`，避免误用旧 40/400 Hz 名称。通用 `suite.load()` 仍从 `assets/contact_settings.json` 读取历史 40/400 Hz 默认时序，且 `PianoTask` 的裸调用默认关闭重力补偿；直接调用时须显式传入 `control_timestep=.05, physics_timestep=.005, gravity_compensation=True` 才与本轮场景一致。
- 使用默认 `export_scene()` 导出到临时目录，与当前训练 scene 对照：编译模型的 body 位置／姿态／质量／惯量／重力补偿、关节／执行器、geom／接触、物理选项，初始 qpos/qvel/ctrl、动作上下限、手部位姿及接触配置全部一致。源文件散列和导出包散列不应视作相同；新的源码导出包不能冒充旧训练 checkpoint 所绑定的冻结包。
- 源预览默认仍显示**仅运动学 M6 双臂**，训练场景无臂；加 `--no-arms` 可对照手和钢琴。源预览是完整 51 动作模型、无乐谱任务；本轮 47 动作训练在加载冻结包时物理删除左右 TZ/Yaw 并使用小星星 task、1180 维观测、`mixed_v1`、自碰撞奖励 0.2 和 SAC。这些属于训练入口，不需要写死到通用模型源码。
- 验证：源预览 NVIDIA EGL 离屏渲染通过，报告重力补偿开启、20/200 Hz 和 Shadow 接触 MATCH。显式 `--control-hz 40 --physics-hz 400` 离屏预览也通过、Shadow 接触 MATCH。`tests/test_mjlab_bundle.py tests/test_preview_controls.py` 共 12 项通过；`tests/test_environment.py tests/test_contact_settings.py -k 'not gpu'` 为 23 passed、1 deselected。旧固定触键测试动作对新姿态不再激活琴键；替换为 TY=0.04 m、右食指 MCP=1.2/PIP=0.3/DIP=0 的真实动作，键 52 可按下并松开，未放宽激活断言。

## 未解决事项

- **V32-GRAV-01**：长期训练结束后，对比新旧冻结场景的学习曲线、接触稳定性和有声回放；短训练与初次 F1 不足以证明效果改善。
- **V32-04／V32-06／V32-SAC-03**：真实驱动参数、碰撞几何全范围及 CPU/GPU／独立回放差异仍沿用既有 TODO。重力补偿是仿真模型设置，不代表实机手部驱动已经具备同等补偿。
