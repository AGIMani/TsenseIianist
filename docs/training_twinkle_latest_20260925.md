# 最新小星星训练与有声视频（2026-09-25）

## 结论

最新训练 `sac-twinkle-source-5m-20260924-1837` 已完成5,000,192条采样，用时2757秒（45.95分钟），593,088次更新，27,648个完整训练episode，无截断。最终逐帧按键F1为0.904307，本轮最佳step-7000为0.919101。主要剩余问题是起始准备阶段误触、部分帧漏按、仍有手部碰撞；不能称为全曲无误。

## 视频与复现

- [最新最终检查点视频](../preview/outputs/replays/twinkle_latest_final_20260925/replay.mp4)
- [本轮最佳step-7000视频](../preview/outputs/replays/twinkle_best_step7000_20260925/replay.mp4)
- [训练曲线](../preview/outputs/replays/twinkle_latest_final_20260925/training_curves.png)
- [抽帧对照](../preview/outputs/replays/twinkle_latest_final_20260925/contact_sheet.jpg)
- [逐帧统计与视频规格](../preview/outputs/replays/twinkle_latest_final_20260925/analysis.json)

两段视频均为1280×720、20fps、199帧、9.95秒、H.264＋AAC。NVIDIA硬件渲染，完整执行178个控制步（8.9秒，含1秒准备）并追加初始画面0.05秒和结尾停留1秒。音频来自200Hz实际琴键激活边沿和实际踏板，经FluidSynth合成，不是乐谱配乐。力度固定127、尚未标定，不能据此评价演奏力度。

在独立项目根目录执行：

```bash
bash eval.sh --checkpoint training_mjlab/outputs/sac-twinkle-source-5m-20260924-1837/final.pt --mode render --output preview/outputs/replays/twinkle_latest_final_20260925
bash eval.sh --checkpoint training_mjlab/outputs/sac-twinkle-source-5m-20260924-1837/step-7000.pt --mode render --output preview/outputs/replays/twinkle_best_step7000_20260925
```

使用源码＋完整config重建，源码和依赖指纹检查通过，两次退出码0。两个目录各包含report.json（检查点哈希、配置与指标）、rollout.npz、audio_substeps.npz、played.mid/wav及音频事件。回放按键precision/recall/F1与相应日志一致；return有约0.00073／0.00024的浮点差异，不声称所有值逐位相同。

## 本轮参数

512环境，direct_joint_v1逐关节映射，47动作，1180观测；无机械臂，TZ/Yaw结构锁止，20/200Hz，前瞻0.5秒，准备1秒。手根0.10m、TY0.04m、Yaw0.15rad，重力补偿开启，三类实体接触保留。Actor256×3、初始温度1、discount0.8；batch4096、gradient_steps64、replay容量262144。原五项奖励＋同手碰撞惩罚0.2启用；新加false_press_penalty仍关闭，乐谱原指法未改。

## 定量结果与边界

| 指标 | 最终step-9766 | 最佳step-7000 |
|---|---:|---:|
| 已采样数量 | 5,000,192 | 3,584,000 |
| 逐帧precision | 0.953183 | 0.955992 |
| 逐帧recall | 0.931648 | 0.933521 |
| 逐帧F1（选择最佳的依据） | 0.904307 | 0.919101 |
| 全局按键帧micro-F1 | 0.921466 | 0.922010 |
| 完全正确的控制帧 | 149/178（83.71%） | 149/178（83.71%） |
| TP／FP／FN按键帧数 | 264／14／31 | 266／16／29 |
| 存在误按／漏按的控制帧数 | 12／19 | 14／19 |
| 准备阶段误按键帧数 | 5 | 2 |
| 同手碰撞惩罚非零的帧数 | 21 | 22 |
| 跨手碰撞奖励为0的帧数 | 2 | 2 |

统计方法：将rollout.actual与goals前88列转为布尔值，逐键逐控制帧比较；FP表示非目标键被激活，FN表示目标键未激活。一个音符跨多帧，因此14个FP不是14个错音；逐帧F1也不是90.4%的音符完美弹出。两种F1的聚合口径不同，不能用平均precision/recall再取调和均值复算逐帧F1。

最终版14个误按键帧中11个在前32控制帧（1.6秒）内，准备1秒仍有5个。缓冲时间允许手移动，但未保证移动过程中不触键。其余3个误按键帧位于索引102的同一帧。漏按散布全曲，尚需逐音符分析区分到键偏迟、提前释放或接触不足，不能仅由F1确定机械原因。

最终版每帧平均奖励：按键0.856956，踏板0.988766，能量−0.089996，指法0.789653，跨手0.494382，同手−0.040449。总return含这些辅助项，不能当作纯演奏准确率。最终／最佳音频分别54／53次note-on，额外触发可能来自错键或同键重复激活；尚未做逐音符对齐，不能将额外触发全部认作错音。

抽查0、0.3、1、1.5、3、5秒画面，双手和键盘显示正常，开头有从初始姿态向演奏姿态的移动。单段完整episode没有拼接多次reset；抽帧不足以证明全程无抖动，尚未进行关节速度／加速度验收。

## 学习曲线与上轮比较

约150万采样时F1为0.611，约230万达0.848，358.4万达到本轮峰值0.919；后期在约0.85–0.92波动，最终较最佳低1.48个百分点。说明本曲学到了有效按键，但存在平台和波动，单纯延长训练不保证继续提高。

上轮 `sac-twinkle-source-5m-20260924` 的最佳日志F1为0.405065，本轮最佳0.919101，提高51.40个百分点；最佳precision由0.309057升至0.955992，误按显著减少。曲线按实际样本数对齐。上轮256环境／mixed_v1、本轮512环境／direct_joint_v1，并非只变一个变量，不能把改善全部归因于MCP映射或环境数。此处上轮数值是训练日志，未在本任务重新导出上轮视频。

每次评估仅同一短曲的一个episode；没有多随机种子／其他曲目验证。PyTorch峰值allocated为约2.66GiB，不包括Warp等分配，不能作为总显存峰值。

## 下一步建议（尚未实施）

1. 以step-7000作为本轮按F1选择的展示与对照基线，同时保留final；最佳F1并不意味着所有错误统计都最佳。
2. 先定位起始1.6秒和漏按帧，做音符起止对齐及关节速度分析；不要仅凭观感更改reset或控制规则。
3. 若用户决定启用已有false_press_penalty，可在相同预算、映射及其他参数下做对照实验，检查准备误触下降是否伴随漏按增加。本次没有替用户开启奖励或启动新训练。
4. 既有V32-SAC-03保留：本轮两检查点指标匹配，不代表所有历史／硬件回放差异已解决；V32-AUDIO-01力度标定、V32-EXPORT-02自动最佳导出入口与源码合同兼容仍保留。本次直接使用eval.sh成功导出。

参考：本轮及上轮config.json、metrics.jsonl、summary.json；eval.sh、scripts/eval_from_source.py、preview/scripts/export_best_mjlab.py；docs/references/biomimetics_2026_610_notes.md。本次仅导出与分析，没有变更训练、奖励或论文对齐决定。

补充验证：final.pt导出report中的logged_evaluation／logged_metrics_match为null，自动日志关联未识别final文件名；已直接按metrics.jsonl的step-9766逐项核对precision、recall、F1及micro-F1，完全一致。step-7000自动关联为true。不能将final报告的null当成自动校验通过；此元数据问题归入V32-EXPORT-02。
