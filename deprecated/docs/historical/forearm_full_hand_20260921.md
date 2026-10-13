# forearm_reward 扩展跨手手指碰撞（2026-09-21）

2026-09-22：另提供含 M6 的新场景／任务，增加左右臂与臂—手检测。[新区域和使用命令](m6_forearm_93mm_20260922.md)。下述旧冻结任务仍保持掌指范围。

用户要求在原掌部碰撞检测基础上增加不同手的手指碰撞。现扩展为**左右手之间的掌—掌、指—指、指—掌**；每侧21个碰撞geom（掌部1个、五指各4个部件），包括拇指根部、四指侧摆部件与指尖。排除安装基座、视觉网格、琴键／琴箱，以及同一只手内部接触。

## 更新后的区域

![新检测区域](../preview/outputs/forearm_full_hand_20260921/forearm_regions.png)

红色右手、蓝色左手。上排初始姿态，下排借用旧mixed视频21.575秒保存姿态展示新区域，不是重新训练后的效果。右列仅显示碰撞凸包；左列叠加半透明模型。名单见[regions.json](../preview/outputs/forearm_full_hand_20260921/regions.json)。

## 奖励规则

- 右组与左组任一有效接触：`forearm_reward=0`；无接触：`+0.5`。现有外部权重1不变，不按接触数量追加扣分。
- 保持原控制周期结束时的接触判定，距离阈值`collision_margin=1e-8`。没有改成子步接触并集，周期内发生又分离的短暂接触仍可能漏检。
- `self_collision_reward`独立负责同手接触，正常按琴键不计入forearm。
- 不更改物理碰撞开关、solref／solimp、动作范围、SAC或终止。奖励不能保证绝对无碰撞，也不修复初期目标向中央跳变。
- 本次是用户指定的工程奖励扩展，不是论文式19的数量奖励；论文collision_reward未知映射仍不代填，原互斥规则保留。

## 实现与兼容

`forearm_collision.py`统一筛选掌部／手指碰撞名称，CPU任务采用相同名单。新task.json保存`forearm_collision_scope=cross_hand_palms_and_fingers_v1`和`forearm_collision_geoms`，保留`palm_geoms`原含义。Torch优先加载新名单，旧冻结任务缺少新字段时仍只检测掌掌。内部snapshot键名`palm_collision`为兼容保留，新任务下表示已选跨手区域接触。

CPU源码直接新建任务采用扩大名单；没有冻结旧规则的历史CPU/JAX源码运行不承诺旧奖励复现。已有Torch检查点依靠冻结task保留原规则。

已导出新完整月光任务：`training_mjlab/outputs/task_clair_mixed_forearm_full_20260921`（2587帧／2070观测）。物理场景继续使用`scene_clair_mixed_pitch02_20260920`。没有覆盖旧task，没有启动新训练。

## 新训练与绘图

在原月光mixed_v1训练命令中使用以下参数：

```bash
--scene training_mjlab/outputs/scene_clair_mixed_pitch02_20260920 \
--task training_mjlab/outputs/task_clair_mixed_forearm_full_20260921 \
--action-profile mixed_v1
```

旧命令指向旧task就仍采用旧掌部范围。新task散列已改变，旧检查点不能冒充在新任务上训练的策略。

从独立项目根目录重画区域（新输出目录）：

```bash
training_mjlab/.venv/bin/python preview/scripts/draw_forearm_regions.py \
  --recording preview/outputs/replays/clair_mixed_best_20260921 \
  --task training_mjlab/outputs/task_clair_mixed_forearm_full_20260921 \
  --output preview/outputs/forearm_full_hand_new
```

绘图校验新task与录像物理scene散列一致，只借用姿态，不改原视频。

## 验收

`training_mjlab/.venv/bin/python -m unittest tests.test_forearm_collision tests.test_mjlab_task -v`：6项通过。新GPU测试在8个独立世界覆盖掌掌、指指、指掌、同手、琴键、正距离、反向geom顺序、缓冲区无效槽；有效跨手接触奖励0，其余0.5。旧名单兼容测试只识别掌掌。8环境真实物理运行4步、奖励有限且为0或0.5，独立reset通过。

导出CPU neutral／press参考各8步，区域绘制完成，目视确认21×2块覆盖。日志保存在区域输出目录的export.log／tests.log。未进行长期训练或重跑已知不完全通过的CPU/GPU物理对照；不声称已改善演奏。V32-TRAIN-01、V32-SAC-03、V32-06继续保留。


补充验证：用新导出CPU neutral／press各8个固定状态输入Torch任务逻辑，16步总奖励及全部分项在2e−6容差内一致；旧mixed任务散列与原训练config一致。此检查隔离奖励逻辑，不代表CPU/GPU物理轨迹逐值一致。结果在区域目录validation.json。
