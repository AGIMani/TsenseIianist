# 五指水平 initial 姿态（2026-10-01）

用户要求在当前基础上只修改手关节角度，使四指和拇指的指尖构成与琴面平行的平面，并保持自然弹琴手型。实际只改 `tsensei_hand_constants.py` 的 `INITIAL_ANGLES_BY_SIDE`：左右各16个手指屈曲/拇指旋转角；四指swing、全部腕关节、手根安装位置、碰撞几何、驱动和奖励公式不改。

## 姿态选择与作用范围

以包内编译模型五个 fingertip site 为端点，固定腕部与安装，约束五点世界Z相等；在机械限位和现有 direct_joint_v1 控制目标范围内，最小化相对原16角的平方改变量。四指弯曲至少0.1 rad，保留原手型的连续弯曲和横向展开，拇指保持原侧向姿态。采用SciPy SLSQP数值求解，写入9位小数；最大单关节改变量为拇指Rot2约4.39166°。这只是自然外观的工程选择，TODO(V32-03)，不是人体动作采集或硬件标定。

当前initial定义为origin整体世界Z上抬0.025m，两者共用手指角。此次修改因此同时作用于后续源码构建的origin/initial手型，保持原抬升合同；source preview无height-offset时显示origin。旧归档/检查点和已有训练不变。读取场景初始角的姿态奖励参考会随新场景自然更新，未改奖励公式/权重；旧checkpoint仍需要匹配归档，不放松源码校验。

左右手对称，四指角单位rad：

| 手指 | MCP | PIP | DIP |
|---|---:|---:|---:|
| index | 0.274966111 | 0.376053926 | 0.191980845 |
| mid | 0.226118825 | 0.367534728 | 0.194547042 |
| ring | 0.231149339 | 0.396375195 | 0.208665629 |
| pinky | 0.258693121 | 0.422932915 | 0.228418149 |

拇指Rot=1.135929529、Rot2=0.476648962、MCP=0.241250147、PIP=0.345985140 rad。完整前后角度见[数值检查](initial_five_tip_plane_20261001.json)。

## 检查结果

CPU重建并reset/forward，左右结果相同：

| 状态 | 五指世界Z | 高于未按下白键面 | 五指最大高度差 | 拟合平面倾角 | 静态接触数 |
|---|---:|---:|---:|---:|---:|
| origin | 53.766140 mm | 31.266140 mm | 9.40e-8 mm以内 | 8.74e-9°以内 | 0 |
| initial | 78.766140 mm | 56.266140 mm | 9.40e-8 mm以内 | 8.74e-9°以内 | 0 |

白键面Z=22.5mm，初始琴键旋转为0，法线世界+Z。这是指尖标记点距白键平面的高度，不是手指整个碰撞体的最小净空。五个点包括拇指；左右镜像误差检查通过。32个修改角均在URDF机械范围与当前动作目标范围内；全部关节和驱动器目标限位、reset qpos/neutral ctrl一致性通过。AST对照确认只有每侧手指角赋值改变，其他常量和腕部值不变。

模型指尖为末节碰撞mesh局部+Z极值面标记，TODO(V32-07)：不等同真实软指腹接触中心，不保证五个指腹接触面的姿态完全一致。共面只保证reset瞬间，策略启动/60–80°从动耦合后手指继续运动；未修改耦合逻辑，不承诺演奏过程中始终水平或无碰撞。未启动训练或测量新F1。

## 回归与图像

执行 `MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' training/.venv/bin/python -m pytest tests/test_environment.py tests/test_preview_pose_config.py -q --tb=short`：22通过，3失败。三项失败均通过进程内加载修改前完整constants复现：旧mesh指尖检查不适配primitive默认、旧palm_link安装接触断言、原URDF惯性与现模型质量设置不一致。与姿态相关的旧硬编码thumb Rot2/pinky默认角断言已改为读取明确的侧别保存角，仍检查名称映射；未改上述三项既有模型问题。

静态真实模型预览通过 `scripts/hpc/submit.sh --allocation 1364 preview` 在Slurm步骤1364.13生成（A100 EGL）；参数 `--headless --no-arms --no-hand-connector --show-fingertips --height-offset 0.025`，无物理settle。height-offset仅为展示现有initial抬升，不写回安装坐标。

- [模型预览](../preview/outputs/initial_five_tip_plane_20261001/initial.png)
- [前后指尖高度](../preview/outputs/initial_five_tip_plane_20261001/height_check.png)
- [数值/角度/边界检查](initial_five_tip_plane_20261001.json)

实际参考：包内左右URDF、`models/hands/{tsensei_hand_constants,tsensei_hand,fingertip_geometry,urdf_model,collision_proxies}.py`、`suite/tasks/base.py`、`models/piano/{piano_constants,piano_mjcf}.py`、`mjlab_env/{preparation,scene}.py`、`configs/actions/direct_joint_v1.json`、run.sh、预览和Slurm入口及上述测试。指定可选日志技能路径不存在，按AGENTS.md维护本文及change_log。
