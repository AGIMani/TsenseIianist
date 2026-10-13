# 小星星5M最终视频和F1曲线（2026-09-26）

本轮训练已完成：5,000,192条样本、4,995,193次SAC更新，最终检查点为[final.pt](../training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894/final.pt)。本页导出的是该最终检查点的确定性完整回放；与训练期间保存的step-1500/2000阶段视频不同。

## 交付

- [观看最终有声视频](../preview/outputs/replays/twinkle_policy_preparation_final_archive_20260926-141944/replay.mp4)：1280×720、H.264＋AAC、20 fps、199帧、9.95秒；[视频报告](../preview/outputs/replays/twinkle_policy_preparation_final_archive_20260926-141944/report.json)、[渲染日志](../preview/outputs/replays/twinkle_policy_preparation_final_archive_20260926-141944_render.log)。
- [F1训练曲线 PNG](../preview/outputs/training_reports/twinkle_policy_preparation_final_20260926-141816/f1_curve.png) · [PDF](../preview/outputs/training_reports/twinkle_policy_preparation_final_20260926-141816/f1_curve.pdf) · [全部评估点CSV](../preview/outputs/training_reports/twinkle_policy_preparation_final_20260926-141816/evaluation_points.csv) · [曲线报告](../preview/outputs/training_reports/twinkle_policy_preparation_final_20260926-141816/report.json)。

曲线使用[该轮metrics.jsonl](../training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894/metrics.jsonl)中的全部489个评估点，横轴为累计环境样本数，原始点连线、没有平滑。整回合F1包含20帧可训练准备；乐曲F1只统计随后158帧，曲线同时展示两条，下面另画准备阶段错键帧数。

| 记录 | 整回合F1 | 乐曲段F1 | 准备错键帧 |
| --- | ---: | ---: | ---: |
| 日志最高评估：step9560，4,894,720样本 | 0.976966 | 0.974051 | 0 |
| 最终检查点：step9766，5,000,192样本 | 0.936798 | 0.935127 | 1 |

最高点是评估日志中的step9560，不能把它当成final.pt视频或未保存的step9560检查点。最终回放准备段实际发生2个错误发音事件，合计1个控制帧有错键；音频与显示保留了这些事件。

## 导出方法与实际验证

先尝试普通eval.sh：源码校验明确拒绝，因为训练之后包内`mjlab_env/reward_config.py`、`mjlab_env/rewards.py`、`rl_torch/train.py`发生变化。[首次失败日志](../preview/outputs/replays/twinkle_policy_preparation_final_20260926-141816_render.log)已保留。没有替换当前源码或关闭校验。

逐文件核对本地归档`training_mjlab/source_versions/policy_preparation_acceleration_20260926.tar.gz`与检查点记录的**163/163个源码SHA256完全相同**；随后使用现有`scripts/eval_from_source_archive.py`在临时目录恢复该归档，按检查点旁完整config.json重建scene/task并渲染。实际命令：

```bash
training_mjlab/.venv/bin/python scripts/eval_from_source_archive.py \
  --source-archive training_mjlab/source_versions/policy_preparation_acceleration_20260926.tar.gz \
  --checkpoint training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894/final.pt \
  --mode render --output preview/outputs/replays/twinkle_policy_preparation_final_archive_20260926-141944
```

归档入口和原评估器返回码均为0；FFprobe实测视频编码/尺寸/帧数/时长如上，音频为AAC；渲染报告中的整回合F1、Precision、Recall、key-frame micro F1、乐曲F1和准备错键帧数已逐项与最终`metrics.jsonl`评估记录核对一致。检查点SHA256：`dd3cbca75ca31064f673ca87d09f8133ed1a0bf81fd037ac714be1bfdfb96525`。曲线报告与CSV仅来自本轮日志，经过有限值、样本数递增和最终step核验；PNG已目视检查。

## 边界

曲线是确定性评估中的按键F1，不代表动作完全平滑。最终准备仍有1帧错键，V32-START-01的开头动作/接触效果仍需看视频和逐子步分析；V32-START-02的20Hz评分对200Hz短错音盲区仍保留。本次只导出与核验，不改变模型、奖励、训练参数或当前源码。
