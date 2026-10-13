# V3.2 通用整指净空训练：最新视频与 F1 曲线（2026-09-27）

本轮训练已完成：512 环境、9,766 向量步，共 5,000,192 条新 transition。导出两个检查点：按训练器资格规则选出的 **best.pt（step 7100）** 和时间上最后的 **final.pt（step 9766）**。最佳选择来自 `best.json`，要求 200 Hz 键底接触统计有效且为零；没有把日志中的整回合 F1 最高值直接当作最佳检查点。

## 交付

- [最佳有声视频](../preview/outputs/replays/generic_clearance_best_step7100_20260927/replay.mp4) · [报告](../preview/outputs/replays/generic_clearance_best_step7100_20260927/report.json) · [预览图](../preview/outputs/replays/generic_clearance_best_step7100_20260927/preview.png)
- [训练结束有声视频](../preview/outputs/replays/generic_clearance_final_step9766_20260927/replay.mp4) · [报告](../preview/outputs/replays/generic_clearance_final_step9766_20260927/report.json) · [预览图](../preview/outputs/replays/generic_clearance_final_step9766_20260927/preview.png)
- [F1 放大图 PNG](../preview/outputs/training_reports/generic_clearance_complete_20260927/f1_curve_zoom.png) · [PDF](../preview/outputs/training_reports/generic_clearance_complete_20260927/f1_curve_zoom.pdf)；[完整纵轴图 PNG](../preview/outputs/training_reports/generic_clearance_complete_20260927/f1_curve.png) · [PDF](../preview/outputs/training_reports/generic_clearance_complete_20260927/f1_curve.pdf)
- [99 个原始评估点 CSV](../preview/outputs/training_reports/generic_clearance_complete_20260927/evaluation_points.csv) · [绘图来源和口径](../preview/outputs/training_reports/generic_clearance_complete_20260927/report.json)

| 检查点 | 三回合最差音乐 F1 | 三回合最差音乐 micro-F1 | 20 Hz 准备错帧 | 200 Hz 键底子步上限 | 手碰撞帧上限 | 键侧/底子步上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best.pt，step 7100 | 0.990024 | 0.987993 | 0 | 0 | 0 | 134 |
| final.pt，step 9766 | 0.978843 | 0.981067 | 0 | 0 | 2 | 26 |

最佳音乐 F1 达到 0.99，但 micro-F1 仍低于 0.99；键侧接触也未消失。最终检查点分数回落并出现两帧手碰撞，因此观看效果优先使用最佳视频。图中蓝线采用训练器选择 best 的 `music_only_f1_min`：每次评估三回合中最低的乐曲段逐帧平均 F1。绿色为音乐 micro-F1；完整图另外显示包含 1 秒准备帧的整回合 F1，以及手碰撞和键底接触。横轴为本轮新增样本数，99 点原始连线，没有平滑或外推。论文 §2.10 pp.12–13 的 25 ms、88 键网格归约与本项目 20 Hz 逐帧平均口径不同，不跨项目直接横比。

## 实际导出与验证

- `preview/scripts/export_best_mjlab.py --select-only` 读取 `best.pt`、`best.json` 和完整选择合同，确认最佳为 step 7100。source 模式视频分别使用 `bash eval.sh --checkpoint training_mjlab/outputs/twinkle_generic_clearance_finetune_20260927-20260927-093425-663203/best.pt --mode render --output preview/outputs/replays/generic_clearance_best_step7100_20260927 --camera-view performer --camera-framing full` 和相同命令替换 `final.pt`、输出 `generic_clearance_final_step9766_20260927`；两次源码/配置重建成功。
- FFprobe 确认两段均为 H.264/AAC、1280×720、20 fps、199 帧、9.95 秒。目视检查两张预览图及放大曲线。回放音乐 F1 与对应训练评估分别相差 2.12e-8、4.69e-8，整回合 F1 逐项相同；准备错帧、键底统计与训练记录一致。
- `scripts/plot_training_f1.py` 的总览只绘制名为 `sac-` 的正式目录，当前自命名运行不会进入其所选曲线。因此本次直接读取本轮 `metrics.jsonl` 全部 99 个评估点，生成本轮单独曲线，未修改绘图源码。

参考文件：本轮 `config.json`、`best.json`、`summary.json`、`metrics.jsonl`，`eval.sh`、`scripts/eval_from_source.py`、`preview/scripts/{export_best_mjlab,render_mjlab_checkpoint}.py`、`scripts/plot_training_f1.py`，以及 `docs/references/biomimetics_2026_610_notes.md` 的 V32-PAPER-11 已决定口径。

**TODO**：V32-QUALITY-01 仍需改善 micro-F1、键侧接触和多曲泛化；V32-START-02 的 20 Hz F1 对 200 Hz 短错音盲区仍在。本次只导出和核验，不改训练、奖励或模型。
