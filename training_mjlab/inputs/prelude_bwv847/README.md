# Bach Prelude BWV 847：候选指法

来源文件：`/home/hj/下载/prelude-bwv-847.mid`

当前训练输入：`candidate_fingering/prelude-bwv-847_fingered.proto`

逐音符表：`candidate_fingering/fingering.csv`

## 生成与验证

- 原始 MIDI 的 SHA-256 为 `9e29d5b640d875bcfa93824cc963a4d9419ce2139cfd4e786148d45127e602d8`。
- 当前训练 proto 已应用 `physics_release_gap_v1` 修复：5 个相邻同键重击的前一音符结束时间各提前 5ms，保留 200Hz 物理采样所需的松键采样点；起音、音高、轨道、指法和力度不变。修复前 proto 保存在 `deprecated/score_repairs/prelude_bwv847_20261012/prelude-bwv-847_fingered.original.proto`。
- 项目内的 `prelude-bwv-847_timing_normalized.mid` 保留全部 1090 个音符和音符 tick 事件，只把原文件中的 tempo、调号和拍号事件移到 conductor track，避免导入器漏读非零轨道的 tempo 变化。
- 轨道 0 按右手处理，轨道 1 按左手处理；`NoteSequence.part` 使用项目约定：右手 0–4，左手 5–9。
- 采用项目现有的动态规划启发式，以 20 Hz（`dt=0.05` 秒）时间网格生成；持续音保持同一手指，同一物理帧没有同指多键冲突。
- 生成结果包含 1090 个音符、1581 个 20 Hz 轨迹帧；20 Hz 和 40 Hz 的 `strict_v1` 覆盖校验均通过，连续时间指法冲突为 0。
- 生成报告保存源文件、输出 proto 的 SHA-256、各手指计数、轨道映射和校验结果：`candidate_fingering/fingering_report.json`；报告同时记录上述释放间隔修复。

这是可供训练使用的工程候选指法，不是专业钢琴家校订，也没有经过 TsenseI 逆运动学、碰撞或实机可达性验收。训练或导出任务时应加载带 `NoteSequence.part` 的 `.proto`，不要改用不含指法的普通 `.mid`。
