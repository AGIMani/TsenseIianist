# 12 cm手根与49维策略（2026-09-24）

> 同日后续修复：TZ 仍结构性移除，包装脚本现使用 `contact_stable_h12_20hz200hz_20260924/{scene,task}`，并恢复同手／跨手／手—琴键真实接触。本文后部“取消反力”属于已覆盖的历史设置。[最新驱动参数与碰撞验收](contact_stability_shadow_20260924.md)。

## 用户决定

双手手根初始高度从0.14改为0.12 m；用户确认左右Wrist_TZ固定0 mm，并从策略中删除两个通道，动作51→49。保留当前Yaw初值、Pitch范围及其他映射。新增开关`--disable-wrist-tz`，默认不加时仍是51维。未启动正式5M训练，仅作短程验收。

论文控制与动作接口相关V32-PAPER-04已有记录，本次按用户明确模型适配，不据论文另改控制、奖励或未知映射。

## 运行

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
bash scripts/train_twinkle_h12_no_tz.sh
```

包装脚本使用新12 cm冻结包、20/200 Hz、无机械臂、mixed_v1、同手自碰撞0.2、5,000,192采样，物理graph/学习器编译/融合优化器开启。可追加`--output-dir`等参数；默认自动生成run目录。检查参数而不训练：`bash scripts/train_twinkle_h12_no_tz.sh --inspect-only`。

新包：`training_mjlab/outputs/scene_twinkle_h12_20hz200hz_20260924`和`task_twinkle_h12_20hz200hz_20260924`。旧冻结场景和检查点不变；修改源常量不会改旧训练包。

## 接口与固定方式（已按用户要求改为物理移除）

- 源高度：`tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py`的RIGHT_POSITION／LEFT_POSITION。
- 49维按原顺序删除原零基索引22、47（R/L Wrist_TZ）；Actor／Critic／replay均为49维。当前mixed_v1压缩偏移也同步，踏板成为索引48。config.json保存policy_action_names。
- `--disable-wrist-tz`在编译MuJoCo模型之前，从MJCF输入中去掉左右`forearm_tz`关节和`R/L_Wrist_TZ`驱动器。实际物理模型nq=nv=136、nu=48；不加参数仍是138/138/50。原Composer手模型也支持用`forearm_dofs`名单不创建TZ。
- 不再使用qpos/qvel/warmstart投影清零，也不创建零位等式约束；垂直平移自由度在求解系统中不存在。手根受垂直外力也不会沿TZ移动，其他腕转动和手指弯曲仍可改变指尖高度。
- 策略／replay仍49维。为了保留当前任务观测和可选奖励的数据接口，上一动作保留51槽（TZ=0）、手部关节观测保留原槽位（TZ=0）、分组功率保留50槽（TZ=0）。这些零槽只是观测／奖励占位，不是物理关节或执行器；实际ctrl为48维、qpos为136维。
- 源冻结包保持不变。开关决定编译时拓扑，运行配置／检查点新增`wrist_tz_physics=joint_removed_v1`，和源包哈希共同标记物理合同。
- 旧“投影固定”的49维试验检查点被明确拒绝加载，不能静默按新动力学回放；原51维检查点仍保留原语义。新的49维检查点可正常训练／评估／回放。
- 当前固定TZ模式只支持无机械臂场景；M6需配合`--no-arms`。未在本侧任务启动长期训练。

## 冻结包与当前仓库并行变化

读取时发现当前task导出器已经增加fingering_target新设计。本轮新任务从上一轮20/200冻结task派生，仅替换scene哈希，保留原乐谱、指法、目标和奖励，未将该额外设计混入本次改动。派生源哈希已记录，旧CPU轨迹参考移除，不作为新高度的对照依据。源场景导出前后校验驱动器、质量／惯量、接触、几何、关节范围保持一致，仅两处手根body_pos的z改变。

## 验证与限制（物理移除版）

- `tests/test_tz_structural.py`：检查关节／驱动器不存在，CPU原生`mj_step`连续100子步，在两手各施加1000 N向下外力时手根高度不变；旧投影检查点合同拒绝测试。
- GPU CUDA graph和逐子步模式各2环境40个随机控制步，叠加1000 N向下外力，手根z保持0.12 m（浮点误差范围内）、观测／奖励有限、局部reset通过。不调用任何坐标纠正。
- 开启TZ分支仍为138物理自由度、50驱动器、51动作。
- 4向量步SAC／GPU replay短测，包含3次更新、评估和检查点保存；输出`training_mjlab/outputs/tz_removed_structural_smoke_20260924`。本次短测关闭学习器编译，保留物理graph／融合优化器；不声称本轮验收了学习器编译或训练收敛。
- 新检查点完整158帧视频回放位于`preview/outputs/replays/tz_removed_structural_smoke_20260924`，用于接口检查而非弹奏效果展示。
- TODO(V32-TZ-01)原“实体固定”缺口通过消除自由度解决；这不包含真实支架柔性／应力测量。V32-SAC-03、V32-06继续保留。

此前投影方案及其短测仅为历史试验，用户已明确要求替换。本版保留12 cm高度、20/200 Hz、现有初始Yaw/Pitch与旧指法规则，不修改并行任务的拇指目标点设计。


## 2026-09-24：同手自碰撞仅检测

用户随后要求取消同手接触反力，12cm无TZ训练包装脚本已增加`--self-collision-detection-only`；自碰撞惩罚0.2保留。预览重启后默认仅检测，可用`--no-self-collision-detection-only`恢复实体接触。见[说明与验收](self_collision_detection_only_20260924.md)。


## 后续覆盖：左右手也取消反力

2026-09-24用户进一步要求跨手碰撞只检测。当前预览默认`--hand-collision-detection-only`，同手＋跨手均无反力；12cm无TZ包装脚本同步使用该开关。旧仅同手开关语义保留。详见[当前说明](all_hand_detection_only_20260924.md)。
