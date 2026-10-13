# V3.2 最新从头训练：best 视频、F1 曲线与旧产物清理（2026-09-27）

本次只处理 2026-09-27 13:25 启动的最新从头训练，不混入此前微调运行。训练目录为 `training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927-20260927-132520-465250`，5,000,192 条新 transition，99 次完整评估。

## 导出结果

- **best checkpoint**：`best.pt`，step 9600；训练器资格选择指标 `music_only_f1_min=0.94345991`，3回合最差 micro-F1=`0.94138544`，键底子步0。该结果没达到0.99。
- [Best 有声视频](../preview/outputs/replays/generic_clearance_fresh_best_20260927/replay.mp4) · [实际回放报告](../preview/outputs/replays/generic_clearance_fresh_best_20260927/report.json) · [预览图](../preview/outputs/replays/generic_clearance_fresh_best_20260927/preview.png)。源码和config重建成功；视频H.264/AAC、1280×720、20fps、199帧、9.95秒。视频单次回放F1=.958186、micro-F1=.950530，准备阶段20帧无误触；200Hz键底0、键侧/底42子步、非末端19子步。单次回放分数和best三回合最差值口径不同。
- [F1曲线 PNG](../preview/outputs/training_reports/generic_clearance_fresh_20260927/f1_curve.png) · [PDF](../preview/outputs/training_reports/generic_clearance_fresh_20260927/f1_curve.pdf) · [99个评估点 CSV](../preview/outputs/training_reports/generic_clearance_fresh_20260927/f1_evaluations.csv) · [曲线选择规则和数据摘要](../preview/outputs/training_reports/generic_clearance_fresh_20260927/f1_report.json)。星标严格使用`best.json`中按三回合最差音乐F1及键底门槛选出的step9600。绘图同时给出最差回合音乐F1、平均音乐F1、micro-F1和0.99目标线。

绘图阶段发现通用旧分析脚本按平均F1排序并找不到权威`best.pt`，会把step9000当best。已删除该脚本生成的临时错选图/报表，仅保留按训练metrics及`best.json`校验后的F1曲线、CSV和摘要；没有更改通用绘图源码。

## 清理范围与影响

- 删除2026-09-17至09-24旧SAC运行中的273个冗余周期checkpoint；每轮保留训练指标最高的可用checkpoint及已有`final.pt`/`best.pt`，配置、日志、metrics、summary和训练目录保留。释放5,983,557,545 bytes（约5.57 GiB）。
- 删除旧训练/预览场景导出目录、`source_cache`及其中复制的STL/OBJ，共13项，释放约2.07 GiB；模型源文件`tsensei_v3_2_pianist/assets/`未动。编译场景可由源码重新导出。依赖已删冻结scene的历史checkpoint可能需重建场景才能回放。
- 删除清单（包含临时错误分析产物）：[cleanup_manifest.json](../preview/outputs/training_reports/generic_clearance_fresh_20260927/cleanup_manifest.json)。本次在`training_mjlab/outputs/`和`preview/outputs/`中已找不到残留STL/OBJ/MSH场景副本。当前best视频、曲线、原始评估点和运行记录均保留。

## 参考与验证

读取本轮`config.json`、`best.json`、`summary.json`、`metrics.jsonl`，使用`eval.sh`及`preview/scripts/render_mjlab_checkpoint.py`导出，使用`analyze_training_run.py`和本轮权威best合同核对曲线。FFprobe确认视频/音轨规格；检查F1曲线图及99点CSV。清理前生成精确候选清单，删除后核查输出目录大小与mesh副本。尚未验证删除旧冻结场景后每个历史checkpoint都能源码重建；此项属于上述回放限制。
