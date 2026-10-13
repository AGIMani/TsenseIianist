# 从最佳检查点新增5M采样（2026-09-24）

用户要求在刚导出的最佳结果基础上再训练5M步。来源为`/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436/step-19000.pt`，源模型训练评估F1=0.673748，独立视频F1=0.660579。

## 续训方式

采用已有--warm-start：恢复Actor、Critic、target网络、优化器、温度、学习器及探索RNG；重新建立GPU replay和仿真状态，不是包含旧replay的逐步精确续跑。沿用Stage1，不切课程或奖励。

- 新增19,532向量步×256环境＝5,000,192条transition。
- 源step-19000已采样4,864,000；本分支结束后累计约9,864,192条。原run最后532向量步不属于该最佳检查点的参数历史。
- --warmup-vector-steps 0：直接用恢复策略采样，避免重新进行纯随机动作预热；--learning-starts 1000保留，先补充新replay，之后每向量步32次SAC更新。
- 49动作／1180观测、20/200 Hz、12cm手根、无臂、TZ结构移除、mixed_v1、同手自碰撞0.2、真实同手／跨手／手—琴键接触均保持。场景／任务哈希、完整reward_config、SAC参数与源run逐项相同。
- 每1000向量步评估和保存；新目录step编号从本轮0计数，updates继承源576,032次。

## 启动

时间：2026-09-24T11:55:25.182361；PID：17055。

```bash
bash /home/houjue/pianist_v2/tsensei_v3_2_pianist/scripts/train_twinkle_h12_no_tz.sh --warm-start /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436/step-19000.pt --warmup-vector-steps 0 --learning-starts 1000 --vector-steps 19532 --num-envs 256 --seed 42 --output-dir /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-continue5m-20260924-115525
```

输出：`/home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-continue5m-20260924-115525`。

```bash
tail -f /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-continue5m-20260924-115525.log
```

同目录.launch.json记录启动PID／命令。训练在后台继续，原检查点保持不变。按上一轮耗时参考约48分钟，实际受GPU负载与评估影响。

## 启动验收

启动前未发现其他Torch SAC训练进程。config.json的origin确认step-19000和576032更新恢复；scene/task哈希、奖励、动作、TZ与碰撞开关、SAC合同全部一致。已开始GPU采样，进一步更新状态见metrics.jsonl；本次尚未完成，不宣称F1提升。

保留V32-SAC-03评估／回放差异、V32-04驱动标定和V32-06碰撞适用范围。没有新论文参数决策或代码修改。

启动实测补充：已采样273920条，累计updates=578304（本轮新增2272次），第一轮评估F1=0.622739。GPU学习更新、独立评估、step-1000保存均已执行；属于续训初期。
