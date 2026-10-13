# 2026-09-20 小星星新训练

用户要求开启新训练。2026-09-20 17:15:14（本机时间）后台启动，PID `1651371`；本页记录启动配置，不是实时状态页。

- 从头训练：Torch SAC Stage1，论文SAC参数，seed42。
- 小星星带指法任务：40 Hz控制／400 Hz物理、0.5秒前瞻、2070维观测、51维动作。
- 256环境×19532向量步＝5,000,192条采样；batch4096，每个符合条件的向量步32次更新。
- 默认五项奖励，额外自碰撞权重0，未启用未知映射或其他论文奖励。
- 预热300步，1000步开始学习；每1000步评价和保存检查点，日志每10步输出。
- CUDA graph、学习器编译、融合Adam、进度条均开启。GPU replay约4.10GiB。
- 训练脱离启动终端运行，输出目录`training_mjlab/outputs/sac-20260920-171514-880281`。
- 原有V32-SAC-03／V32-06仍保留，参见[新物理验收](shadow_contact_40hz_acceptance.md)。本次未修改代码、物理、奖励或任务，不代表CPU/GPU激活差异已解决。

## 查看进度

从独立项目根目录运行：

```bash
V32_RUN=training_mjlab/outputs/sac-20260920-171514-880281
tail -f "$V32_RUN/metrics.jsonl"
# 查看包含进度条的终端日志
 tail -f training_mjlab/outputs/launches/sac-20260920-171514-880281/console.log
```

第一次学习发生前，updates=0是预期行为；此时吞吐和ETA不包含SAC更新成本，不用于预测最终训练耗时。

## 导出该轮已保存的最佳结果

首个评价与检查点写出后：

```bash
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir training_mjlab/outputs/sac-20260920-171514-880281
```

完整启动命令、PID与时间保存在[launch.json](../training_mjlab/outputs/launches/sac-20260920-171514-880281/launch.json)，可复制命令在[command.sh](../training_mjlab/outputs/launches/sac-20260920-171514-880281/command.sh)。重复训练需要新的输出目录。

## 启动验收快照

启动约1分43秒时：进程存活；最近完整日志为1120向量步／286,720条采样／3,872次更新，loss与温度为有限值；已写出step-1000.pt。第1000步完成315帧整曲独立评价，F1=0.1210229，未截断。此为初期结果，不是最终训练指标。首个SAC编译已通过，训练继续后台运行。

终端有Torch提示可开启TF32，本轮按既有FP32合同保持关闭，不作为错误处理；暂无训练异常。

## 完成结果与最佳视频（2026-09-20）

- 本轮已正常完成：19,532向量步、5,000,192条采样、593,056次SAC更新。
- summary.json实测总时间 **3581.8859秒＝59分41.9秒**，平均约1396条采样/秒。计时包含物理／学习器初始化、首次编译、采样更新、独立评价和检查点保存；不含Python启动前开销及本次视频导出。
- 本机启动时间17:15:14.881，summary写入18:14:58.665，进程启动到完成文件约59分43.8秒；这与内部计时边界略有不同。
- 在本轮20个有保存文件的评价候选中，按日志mean-frame按键F1选出 **final.pt**（向量步19532）：F1=0.8595394、precision=0.9693651、recall=0.8322750、return=989.9627。这里的“最佳”限定本轮相同新场景／任务，不混入旧20Hz环境。
- [全钢琴正面有声视频](../preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920/replay.mp4)：1280×720、40fps、8.9秒、H.264＋AAC；正交−30°视角，含完整双手和钢琴。声音按真实激活键／踏板的3150物理子步合成，58次note-on，未播放目标乐谱。
- 视频回放315步，F1=0.8610209、return=990.3944；与训练日志F1相差约0.0014815（0.148个百分点），仍按V32-SAC-03记录，不声称逐帧复现。V32-AUDIO-01固定velocity127未标定保持。
- [选优依据](../preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920/selection.json)、[回放报告](../preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920/report.json)、[训练时间与指标对照](../preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920/training_comparison.json)。render脚本对final.pt未填logged_evaluation，因此单独保存selection／comparison作核对，没有把null当一致。

实际导出命令：

```bash
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir training_mjlab/outputs/sac-20260920-171514-880281 \
  --output preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920 --audio
```

ffprobe确认视频／音轨与时长，查看preview.png确认完整钢琴／双手、激活键绿色和指法图例；未修改模型、奖励、训练配置或检查点。
