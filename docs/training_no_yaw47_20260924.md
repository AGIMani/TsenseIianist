# 小星星：关闭TZ与Yaw，47动作（2026-09-24）

## 用户决定与实现

用户要求新增取消Palm_Base_Yaw的训练，并确认保持当前训练场景各自的初始角度。当前冻结包`contact_stable_h12_20hz200hz_20260924/scene`左右Yaw均为+0.25 rad（约14.3°），取manifest.qpos而非后来编辑的源默认值。

- 新增`--disable-wrist-yaw`，本版要求同时`--disable-wrist-tz`；不加时保留旧49/51动作行为。
- 在51通道顺序中删除R/L TZ、R/L Yaw（零基22、24、47、49），Actor/Critic/replay为47维，踏板最后；目标熵按现行规则为−23.5。物理nq/nv=134、nu=46，TZ和Yaw关节及驱动器实际移除，没有隐藏可运动Yaw或每步坐标投影。
- 将固定Yaw写入手根静态旋转，同时反向变换同body上其他腕部关节轴，确保TX/TY/Pitch的方向不改变。当前实现要求关节锚点和reference为0，遇到其他布局会拒绝，不静默近似。
- 1180维观测保留完整关节槽位，Yaw槽填各侧固定值；完整物理动作同样补固定值，移除通道功率填0。策略不再输出它们。
- checkpoint记录disable_wrist_yaw，评估／视频自动恢复47维配置；旧49维网络不能直接warm-start本实验。因此本轮从头训练5M，不继承上轮权重。

## 本轮配置与启动

保持12cm手根、20/200 Hz、0.5秒前瞻、无机械臂、mixed_v1、同手自碰撞0.2、三类实体接触和接触稳定驱动参数。SAC仍Stage1；不引入新的论文奖励设计或未知映射。

预算19,532向量步×256＝5,000,192采样。起始时间2026-09-24T12:01:55.809090，PID33512。此前49维续训继续在原进程运行，本轮共享GPU，吞吐可能下降。

```bash
bash /home/houjue/pianist_v2/tsensei_v3_2_pianist/scripts/train_twinkle_h12_no_tz_no_yaw.sh --num-envs 256 --vector-steps 19532 --seed 42 --output-dir /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-no-yaw47-20260924-120155
```

输出：`/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-no-yaw47-20260924-120155`。

```bash
tail -f /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-no-yaw47-20260924-120155.log
```

独立复制项目后可运行`bash scripts/train_twinkle_h12_no_tz_no_yaw.sh`，包装脚本从自己的项目位置解析路径。旧冻结包保持不变，移除关节在加载时按checkpoint配置完成。

## 验证

- CPU前向运动学：10组随机其他关节姿态，将原模型Yaw设为冻结初值后，与移除Yaw模型的全部geom位置／旋转一致，绝对误差阈值1e-10。确认134坐标／46驱动器。
- 2环境4向量步GPU短训练通过，包含47维编译SAC、3次更新、GPU replay、融合优化器、独立158帧评估与检查点保存。记录在training_mjlab/outputs/no_yaw47_smoke_20260924。
- 检查点回放与正式训练启动结果见修改日志及metrics.jsonl；短程验收不代表长期效果。

保留V32-SAC-03评估/回放差异、V32-04驱动标定、V32-06碰撞适用范围。此实验只检验固定Yaw组合的训练效果，不保证F1提升。

启动验收补充：正式run确认47动作／1180观测／目标熵−23.5、disable_wrist_yaw=true，已采样115,200条，当前补充replay、尚未到1000向量步学习起点；短程检查点独立视频完整158帧通过。
