# 小星星 47 动作训练：阶段结果、视频与 F1 曲线（2026-09-24）

本报告读取已保存的检查点和 `metrics.jsonl`，按各自最高 `eval.f1` 导出确定性 GPU 全曲回放。**重力补偿轮已完成 5,000,192 条采样；无补偿续训仍在运行**，后者可能产生更高检查点。`eval.f1` 是逐帧平均的琴键 F1；图中的点是原始训练评价，没有平滑。

| 训练 | 选用检查点 | 日志最佳 F1 | 独立回放 F1 | 视频 |
| --- | ---: | ---: | ---: | --- |
| 无重力补偿 47 动作续训 | `step-7000.pt`，本轮 1,792,000 采样 | 0.853105 | 0.853105 | [含真实触键音频的视频](../preview/outputs/replays/twinkle_legacy47_continue_step7000_20260924/replay.mp4) |
| 手部重力补偿 47 动作从头训练 | `final.pt`，本轮 5,000,192 采样 | 0.844304 | 0.844304 | [含真实触键音频的视频](../preview/outputs/replays/twinkle_gravcomp47_completed_best_20260924/replay.mp4) |

![小星星 F1 训练曲线](../preview/outputs/training_reports/twinkle_gravity_comparison_step7000_20260924/f1_curves.png)

[高清 PDF](../preview/outputs/training_reports/twinkle_gravity_comparison_step7000_20260924/f1_curves.pdf) · [每次评价的 CSV](../preview/outputs/training_reports/twinkle_gravity_comparison_step7000_20260924/evaluation_points.csv) · [曲线来源和 SHA-256](../preview/outputs/training_reports/twinkle_gravity_comparison_step7000_20260924/report.json)

左图把无补偿首轮 5M 与其续训连接在累计采样轴上，同时显示从头训练的重力补偿曲线；右图显示当前两轮各自的采样进度。**无补偿续训从已学习 5M 的策略出发，而重力补偿实验从零开始**，不能据此图单独归因于重力补偿，更不能把本轮不同训练进度下的最高值当最终排名。

两段视频均使用各自检查点记录的冻结 scene/task、`mixed_v1`、47 动作、完整 158 控制帧。独立回放由 MJLab GPU 推进，CPU 快照仅作渲染；音频来自每个物理子步的真实琴键激活与踏板。`ffprobe` 确认两段均为 H.264＋AAC、1280×720、20 fps、8.95 秒。无补偿／补偿回放分别产生 34／37 次 note-on。音符力度仍固定为 127，尚未标定（V32-AUDIO-01）。

完整[无补偿选择](../preview/outputs/replays/twinkle_legacy47_continue_step7000_20260924/selection.json)、[无补偿回放指标](../preview/outputs/replays/twinkle_legacy47_continue_step7000_20260924/report.json)、[补偿选择](../preview/outputs/replays/twinkle_gravcomp47_completed_best_20260924/selection.json)、[补偿回放指标](../preview/outputs/replays/twinkle_gravcomp47_completed_best_20260924/report.json)保留检查点与场景散列。两段新视频的 F1 与各自日志相同；无补偿视频的其他回报指标仍有细微差异，`logged_metrics_match=false`。补偿轮选用 `final.pt`，回放报告按脚本约定没有对应 `step-N.pt` 日志行，`logged_metrics_match=null`，需结合 selection 中的最终评价查看。V32-SAC-03 的独立回放细微差异仍未完全消除。

验收过程：`export_best_mjlab.py` 对两轮按已保存评价选优，显式传各自冻结 scene/task；最终选用的两次导出均退出 0，完整 158 控制帧，场景/检查点散列验证通过；`ffprobe` 核对视频和音轨，人工查看中间帧及曲线图。未停止或修改无补偿续训。最新曲线快照在 2026-09-24 15:17:29+08:00 生成，含无补偿首轮 20 点、无补偿续训 7 点、重力补偿完成轮 20 点；此前阶段视频和曲线也保留在 `preview/outputs/`。后续进度以 run 的 `metrics.jsonl`／`summary.json` 为准。

以后可用以下命令生成新快照；脚本只读训练日志，不加载 GPU：

```bash
training_mjlab/.venv/bin/python scripts/plot_twinkle_gravity_comparison.py \
  --output preview/outputs/training_reports/另一个新目录名
```
