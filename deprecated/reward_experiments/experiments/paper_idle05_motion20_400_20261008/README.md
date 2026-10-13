# paper＋旧闲指−0.5＋目标运动20/400，从头训练

2026-10-08用户指定新增对照。Slurm **1931** 于北京时间08:38:14提交，双A100物理0/2；复用根 `all_use_this_train.sh`，没有新增训练入口。

| 项目 | 本次设置 |
|---|---|
| 初始化 | 从头随机初始化，seed42；无父checkpoint |
| 按键 | paper，和弦关闭 |
| 旧闲指姿态 | idle_posture_penalty_v1，整项十指合计每步最多−0.5，weight=1，sigma=.25 |
| 同手 / 跨手 | 每去重body对−.2 / 周期任意跨手接触−.5 |
| 独立误触罚 | 关闭 |
| 延音 / 能量 / 指法 | 开启，weight=1；legacy指法 |
| 目标运动惩罚 | 手指20rad/s、400rad/s²；腕旋转3rad/s、15rad/s²；超限10%整项封顶−5/步 |
| 位置及Pitch目标限制 | hard_v1；Pitch3.1416rad/s、15.708rad/s²；TX/TY 1m/s |
| 公共奖励采样 | 子步，实际键激活占比.5，不乘dt；旧闲指仍控制周期末 |
| 环境 | 原小星星/原指法，A3.2，无臂，20/200Hz，31动作/1407观察 |
| 初始姿态 | 同986完整准备合同；h=.10m、TY=.04m、Yaw=.15rad，五指共面，1秒/2.5cm准备 |
| 学习 | gamma=.8，三个LR3e-4，512环境，batch1024，replay2M，.25更新/样本 |
| 预算 | 19,532向量步＝10,000,384样本 |
| 评估/选模 | 每500步，3回合，原MIDI音乐200Hz micro F1最差回合；保存best及final |

相对986仅paper模式和旧闲指上限.1→.5两项奖励改变；相对635还改变了手指目标惩罚阈值10/200→20/400。历史两组为H100，本次A100，不能称逐位可复现的硬件同条件对照。目标差分限制不保证实际qvel/qacc。

本次显式使用匹配旧奖励合同的项目内固定归档 `logs/joint_motion_limits_20261006/source_motion_limits_v1.tar.gz`，SHA256 `821996010ffd4727558bdc4090f657677c5233cf39d686cbfed3f77f9bbda022`。未混入后来新默认的闲指L1、同手−.75、误触−.05或端点评分。根入口新增成对的 `--source-archive PATH --source-sha256 SHA` 前置参数；缺配对或SHA不符即拒绝。无覆盖时原默认归档/参数完全不变。

提交命令（重复执行会创建另一个实验，不用于查询）：

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --source-archive /home/houj/projects/tsensei_v3_2_pianist/logs/joint_motion_limits_20261006/source_motion_limits_v1.tar.gz --source-sha256 821996010ffd4727558bdc4090f657677c5233cf39d686cbfed3f77f9bbda022 --reward-config /home/houj/projects/tsensei_v3_2_pianist/configs/experiments/paper_idle05_motion20_400_20261008/rewards.json --reward-sampling auto --substep-rewards --target-key-substep-reward --target-key-activation-weight 0.5 --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_idle05_motion20_400_scratch_20261008
```

CPU预检确认36个非奖励有效区段与986一致，根入口原默认sbatch参数逐项不变，缺失/错误SHA均拒绝且CUDA未初始化。实际Slurm配置、源归档、场景/任务/动作/观察/控制/准备/耦合合同已经回读匹配；基础GPU检查通过，启动日志已完整打印奖励。

来源：`logs/hpc/paper_idle05_motion20_400_20261008/{request,entry_arguments,effective_config,submission,config,startup_validation}.json`。启动审计以 `startup_validation.json` 最新保存值为准；训练完成前不宣称最终F1或提分。

启动学习核验（北京时间2026-10-08T08:41:25.901623+08:00）：Slurm1931 RUNNING；向量步380，样本194,560，SAC更新25,088，训练数值有限。实际配置SHA `a578fafcbce49f2a2d681c4bac96db8fa6a985ea3c5dcf0ec5da356391a52506`。
