# 2026-09-23：M6 小星星 5M 训练启动

## 最新结果：训练已完成，最佳视频已导出

- `sac-20260923-004554-861361` 完成5,000,192条采样、593,056次更新、15,872个训练回合；训练器计时5276.316秒＝1小时27分56秒。目前未发现训练进程。
- 20次整曲评估：首次F1=0.102751，最佳 **step-18000（4,608,000采样）F1=0.204550**；final F1=0.167407、Recall=0.180952。已记录train标量未发现NaN/Inf；完成采样不代表成功演奏。
- [F1／Recall曲线](../preview/outputs/training_reports/m6_twinkle_20260923/f1.png)；[完整指标](../preview/outputs/training_reports/m6_twinkle_20260923/analysis.json)。相较此前40/400 Hz小星星F1=0.859539明显更低，但本轮同时改变动作、Pitch、姿态、奖励与机械臂可达约束，不能将退化单独归因于机械臂。
- [最佳有声回放](../preview/outputs/replays/m6_twinkle_best18000_20260923/replay.mp4)／[四时刻预览](../preview/outputs/replays/m6_twinkle_best18000_20260923/contact_sheet.png)。正交前视，包含全部钢琴、肩座及双臂；实际触键音轨，非目标乐谱配音。
- **此次回放F1=0.193968、Recall=0.194180**，与训练日志最佳0.204550不一致，`logged_metrics_match=false`。保持两份结果，不把回放数字改成日志数字；V32-SAC-03差异原因仍待定位。
- 回放315个控制帧中约19.365%有琴键激活；音轨按3150个物理子步生成22次note_on，非静音，但这不等于22个正确音符。平均指法接近奖励0.884957、按键奖励0.545503，说明接近目标并未充分转化为有效演奏；奖励竞争或IK限制是否为原因仍需进一步对照。
- 所有保存控制帧的forearm_reward均为0.5，本段未检测到所选范围的碰撞；同手惩罚均值−0.013968。腕部连接最大位置残差右0.000142 mm、左0.000099 mm，未发现保存帧的手臂脱开。只检查了周期末状态，不保证全部物理子步无碰撞，也不包括未启用的同臂／臂—琴碰撞。[几何与分项检查](../preview/outputs/replays/m6_twinkle_best18000_20260923/review.json)。

### 最佳视频复现命令

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
MUJOCO_GL=egl training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py \
  --checkpoint training_mjlab/outputs/sac-20260923-004554-861361/step-18000.pt \
  --scene training_mjlab/outputs/m6_93mm_twinkle_20260923/scene \
  --task training_mjlab/outputs/m6_93mm_twinkle_20260923/task \
  --camera-view front --camera-framing full --audio \
  --output preview/outputs/replays/my_m6_twinkle_best
```

输出须选新目录。新M6包是嵌套目录，本命令显式传scene/task，避免旧自动查找层级不足。

验收：回放完整315帧，视频含初始帧和末帧停留共8.9秒；1280×720、40fps、H.264＋AAC，ffprobe及完整解码通过，截图已查看。仅M6全景渲染相机自动调整中心与默认距离；手部近景和旧无M6相机不变。原训练配置／奖励未修改，未启动新训练。

以下启动记录为历史快照。

用户要求在已接入机械臂的训练场景上开始小星星训练。本轮从头训练，沿用既有5M预算和已确认论文Stage1 SAC设置。

- 启动时间：2026-09-23T00:45:54.861990（本机Asia/Shanghai）。PID：1357075。
- 运行目录：`training_mjlab/outputs/sac-20260923-004554-861361`。
- 场景／任务：`training_mjlab/outputs/m6_93mm_twinkle_20260923/scene`、`task`。
- 乐谱：TwinkleTwinkleRousseau，有指法，315帧，40/400 Hz，2070观测、51动作。
- 机械臂：93 mm安装中心距、GPU批量运动学IK及周期末可达腕部约束；1175对forearm名单，无碰撞+0.5／碰撞0；没有机械臂接触力或实体动力学。
- 动作：mixed_v1，Pitch物理范围±0.2rad；同手自碰撞权重0.2。其余奖励延续既有规则，不引入未知论文映射。
- 256环境×19532向量步＝5,000,192条采样；batch4096、每次32更新、seed42，Stage1。前1000向量步收集样本，随后满足batch条件开始更新。
- 多物理子步CUDA graph、学习器编译、融合Adam与进度条开启。每1000向量步评价、保存检查点，末尾另评估保存final。

## 命令与进度

以下为本次实际命令存档，不要在当前进程运行时重复执行：

```bash
/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/.venv/bin/python -u -m tsensei_v3_2_pianist.rl_torch.train --scene /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/m6_93mm_twinkle_20260923/scene --task /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/m6_93mm_twinkle_20260923/task --stage 1 --device cuda:0 --num-envs 256 --vector-steps 19532 --batch-size 4096 --gradient-steps 32 --seed 42 --physics-graph --compile-learner --fused-optimizer --progress --output-dir /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-20260923-004554-861361 --self-collision-weight 0.2 --action-profile mixed_v1
```

查看进度：

```bash
tail -f /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/launches/sac-20260923-004554-861361/console.log
```

[启动信息及命令](../training_mjlab/outputs/launches/sac-20260923-004554-861361/launch.json) · [训练日志](../training_mjlab/outputs/launches/sac-20260923-004554-861361/console.log) · [指标](../training_mjlab/outputs/sac-20260923-004554-861361/metrics.jsonl)。首次编译可能暂停进度更新；没有评估前不报告F1。

## 导出与验证

以既有mixed_v1的手物理场景`scene_clair_mixed_pitch02_20260920`为基础重新导出小星星任务，再用`export_m6_bundle.py`添加机械臂；场景目录名来自历史月光实验，但新乐谱明确为小星星，未复制月光音符。导出核对315帧、2070观测、93 mm及1175检测对；不覆盖旧包。

原生预览`.venv`缺少dm_env_wrappers，初次导出失败；改用既有`training/.venv`完成CPU任务导出，M6资源合并用原生`.venv`，训练用`training_mjlab/.venv`，没有临时安装依赖。日志保存在启动目录。实际启动进展在本文件末尾补记。

本次只启动训练，不表示已完成5M或收敛；TJ-ARM-02/03标定、TJ-ARM-04实体动力学／避障、TJ-ARM-05吞吐／长期学习／冗余IK观测及V32-SAC-03／V32-06继续保留。

## 启动后实测

核验时间 2026-09-23T00:47:26.643379：PID 1357075存活，日志向量步660、采样168960、更新0；GPU replay持续增长，分项奖励有限且含forearm_reward／self_collision_reward。尚未出现评估结果，不报告F1或预计完工时间。
