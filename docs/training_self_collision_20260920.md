# 2026-09-20 自碰撞惩罚对照训练

**当前状态：用户随后指定改练上传的《月光》简易版，小星星本轮已于20:24:43停止；《月光》缺指法，尚未启动。**

用户要求“增加上手指自碰撞的惩罚再开一次训练”。本轮启用已经实现的 `self_collision_reward`，采用现有候选／模板权重 **0.2**。这是本轮显式启用，不修改全局默认奖励。

## 任务与惩罚

- run：`training_mjlab/outputs/sac-20260920-202151-452958`；2026-09-20 20:21:51启动，后台PID1851100。
- 对照：`sac-20260920-171514-880281`（无此惩罚，5M训练已完成）。
- 从头训练、seed42、256环境、19532向量步＝5,000,192采样；batch4096、每学习向量步32次更新。
- 40/400 Hz、0.5秒前瞻、2070维观测、51动作；场景／任务散列与对照轮相同。三项加速及进度条开启。
- 惩罚：`−0.2 × N_same_hand_pairs`。每个控制周期内10个物理子步及末次forward中，`contact.dist <= 0`的同手body对取并集；多个接触点／多个子步只算一次。
- 包含手指—手指及手指—掌部等同手部件接触，受原模型装配排除与碰撞过滤约束；不处罚正常手指—琴键，也不把双手之间碰撞归入此项。原掌部奖励保持。
- 这是已提供的纯负同手碰撞惩罚，不是论文p.12式19的`0.2×(1−N_undesired)`；未启用需要用户填写接触名单的`collision_reward`，未代填未知映射。
- 实际config对照：CLI仅`self_collision_weight: 0→0.2`及输出目录变化，reward_config仅这一项enabled变化。奖励不同，跨轮不要直接用总return判断优劣，应结合按键F1、碰撞次数及动作视频。惩罚不能保证零碰撞。

## 查看进度

在独立项目根目录执行：

```bash
V32_RUN=training_mjlab/outputs/sac-20260920-202151-452958
tail -f "$V32_RUN/metrics.jsonl"
# 含进度条的完整终端输出
 tail -f training_mjlab/outputs/launches/sac-20260920-202151-452958/console.log
```

日志中`reward_raw_terms.self_collision_reward`是负的平均碰撞对数；`reward_weighted_terms.self_collision_reward`是乘0.2后的实际扣分。第10步实测分别−4.8984375与−0.9796875，确认惩罚进入实际奖励。

## 导出本轮最佳视频

至少首次评价和检查点保存完成后：

```bash
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir training_mjlab/outputs/sac-20260920-202151-452958
```

回放会自动恢复本轮奖励。完整启动命令与PID记录：[launch.json](../training_mjlab/outputs/launches/sac-20260920-202151-452958/launch.json)／[command.sh](../training_mjlab/outputs/launches/sac-20260920-202151-452958/command.sh)。同配置重跑请更换输出目录。

## 保留的限制

V32-SAC-03 CPU／GPU激活与回放差异、V32-06接触标定仍存在；训练启动不意味着这些问题已解决。能否减少碰撞且保持弹奏效果，要等本轮完成后验收，不根据启动日志承诺改善。


## 改换曲目（2026-09-20）

用户随后上传clair-de-lune-beginner.mid并要求训练该曲。已对确认命令匹配的本轮PID1851100发送SIGTERM，确认进程退出；保留原输出、metrics和step-1000.pt，未删除数据。停止前最后完整日志1520向量步／389120采样／16672更新；独立评价及首个检查点已成功，惩罚实际进入奖励。interruption.json记录用户改曲原因，没有把此次中断写成训练完成。

新曲原始文件与检查报告、146行待填指法CSV见[曲目准备目录](../training_mjlab/inputs/clair_de_lune_beginner/README.md)。2轨、约65.087秒、音高36～79；两轨100／46音符，当前解析后所有part=0，has_fingering=False。MIDI只有非主轨调号引起的解析警告，读取成功；未修改原始字节。

当前GPU导出器task_bundle.py要求完整指法，原用户选择由自己提供指法。已询问本曲由用户补充，还是授权自动生成候选指法直接训练；答复前不擅自分配或关闭指法奖励。本曲尚未导出任务、尚未训练。自碰撞权重0.2、40/400 Hz及5M计划保留。


后续：用户已授权助手生成指法，完成146音符候选并开启《月光》训练，详见[最新记录](training_clair_de_lune_20260920.md)；上文缺指法状态为历史。
