# 小星星可训练准备段：5M正式训练（2026-09-26）

用户要求“帮我开启一次训练”。沿用当前run.sh默认，从头训练，未使用warm-start，未采用2048规模测速参数。本轮仅启动及检查，不将启动成功写成训练完成。

## 运行位置

- 启动时间：2026-09-26T02:01:40.957448+08:00。
- 后台入口PID：3388050；独立会话、输入断开、日志写文件，退出当前终端后仍继续。
- 输出：[本轮目录](../training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894)；完整参数：[config.json](../training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894/config.json)。
- 终端日志：[train.log](../training_mjlab/launches/twinkle_policy_preparation_5m-20260926-020140/train.log)。
- 启动与验证元数据：[launch.json](../training_mjlab/launches/twinkle_policy_preparation_5m-20260926-020140/launch.json)。

## 本轮配置

小星星TwinkleTwinkleRousseau；512环境×9766步=5,000,192样本；前4999条随机、第5000条起每新样本1次SAC更新，计划4,995,193次更新。batch256、replay1M、Actor/Critic256×3、三个学习率3e-4、初始温度1、gamma0.8；20/200Hz、无臂、TZ/Yaw结构移除、47动作、1181观测，Shadow任务/驱动/碰撞保持。

policy_v1：origin抬高2cm作为initial，1秒支撑平滑回origin高度，策略从第0帧控制，20空谱帧参与原奖励/replay/SAC；原乐谱指法、重力补偿保持。physics graph、compiled learner、fused optimizer、learner graph和Adam布局全部开启。每20向量步评估、500步保存、2步写指标。

## 启动核验与限制

2026-09-26T02:03:42.400569+08:00核验到step 164，83,968条样本、78,969次更新、8次评估；后台入口存活、配置断言及163个包内文件源码指纹通过。未发现启动阻断错误；既有capsule/mesh MULTICCD限制仍按原设置保留。

训练仍在运行，以上为启动快照，不会自动更新此文档。按前轮持续测速估计约1小时50分钟，实际以进度和最终summary.json为准。V32-START-01准备期误触改善、V32-START-02短错音盲区、V32-PERF-01长期吞吐及原接触/驱动TODO保留；尚不能宣称收敛或无爆冲。

查看进度：

```bash
tail -f training_mjlab/launches/twinkle_policy_preparation_5m-20260926-020140/train.log
```

## 02:24 最新已保存检查点的有声视频

训练期间又保存了 `step-2000.pt`，已重新导出当前最新视频：[观看 step-2000 有声回放](../preview/outputs/replays/shadow_sync_step-2000_20260926-022408-889856540/replay.mp4) · [指标与回放报告](../preview/outputs/replays/shadow_sync_step-2000_20260926-022408-889856540/report.json)。视频为 H.264＋AAC、20 fps、199 帧、9.95 秒；报告核对日志指标一致。整回合 F1=0.8076，单独乐曲段均帧 F1=0.7833；准备 20 帧无错键。这仍是训练中的阶段检查点。

## 02:22 前一检查点有声视频

用户要求导出最新训练视频。导出时该轮最新完整检查点为 `step-1500.pt`（768,000 条训练样本），用现有 `preview/scripts/export_latest_training_video.sh` 从检查点旁完整 `config.json` 和当前源码重建，未暂停或改动后台训练。

- [观看视频](../preview/outputs/replays/shadow_sync_step-1500_20260926-022153-947827789/replay.mp4)：H.264 画面＋AAC 实际琴键音频，20 fps、199 帧、9.95 秒。
- [回放报告](../preview/outputs/replays/shadow_sync_step-1500_20260926-022153-947827789/report.json)及[导出日志](../preview/outputs/replays/shadow_sync_step-1500_20260926-022153-947827789_render.log)。报告确认检查点指标和回放一致。
- 此检查点整回合 F1=0.6491；去掉准备阶段后，乐曲段平均逐帧 F1=0.6046。准备 20 帧本次没有错键，真实音频和奖励轨迹同步；这些是当前阶段回放，不代表最终 5M 效果。

实际回放包含 20 帧可训练准备＋158 帧乐曲；音频由真实物理子步的琴键事件生成，无另拼的固定准备前缀。导出后训练进程仍在运行，之后将出现更新的检查点。
