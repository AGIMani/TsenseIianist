# Nocturne Op. 9 No. 2 beginner：候选指法

来源文件：`/home/hj/下载/nocturne-op-9-no-2-beginner.mid`

输出文件：`generated_v1/nocturne-op-9-no-2-beginner_fingered.proto`

- 共 171 个音符，原始音符属性全部保留。
- 乐器轨道 0 按右手处理（78 音符），轨道 1 按左手处理（93 音符）。
- `NoteSequence.part` 使用项目约定：右手 0–4，左手 5–9。
- 以 20 Hz（50 ms）时间网格生成；20 Hz 与 40 Hz 的 `strict_v1` 覆盖检查均通过。
- `fingering.csv` 是逐音符可读表，`fingering_report.json` 保存源文件和输出文件 SHA256、指法计数及生成参数。

这是动态规划启发式候选，不是专业钢琴家校订，也没有经过 TsenseI IK、碰撞或实机可达性验收。训练前仍应复核跨手分配、和弦指法和长距离跳跃。

## 带臂训练启动记录（2026-10-12）

Job 2266 使用本谱面和显式双 M6 初始角从头启动。因指定固定源码归档的合同名称与当前 AGENTS.md 不一致，在启动审计后主动取消；没有替代训练在运行。检查记录与完整奖励配置见 `../../../results/nocturne_arms/job_2266/`，详情见 `../../../docs/change_log.md`。
