# Clair de lune beginner：待补指法

原始MIDI按字节复制，未改节奏、调号或音符。inspection.json保存检查结果。146个音符均无指法，当前有指法GPU任务无法直接导出。用户已授权生成指法；已生成 `generated_v1/fingering.csv` 与保留指法的 `.proto`，训练启动状态见项目 docs/training_clair_de_lune_20260920.md。

## 指法表

fingering_template.csv的note_index对应本项目MidiFile.from_file解析后NoteSequence.notes的序号；不要重新排序或修改音符身份列。仅填写finger_id：

| finger_id | 手指 |
| --- | --- |
| 0～4 | 右手拇指、食指、中指、无名指、小指 |
| 5～9 | 左手拇指、食指、中指、无名指、小指 |

每个音符必须填写一项。本表是待填模板，空白不能视为默认拇指；两条MIDI轨道不等于逐指分配。用户于2026-09-20要求由助手生成对应指法；保留空白模板，实际候选在 generated_v1/fingering.csv。

计划采用既有同手自碰撞奖励−0.2×控制周期内去重碰撞部件对数；40/400 Hz、0.5秒前瞻和5M预算保持。指法补齐后需要导出新的任务包，不能用小星星任务代替。

## 生成结果

- 第0乐器轨道按右手（100音符）、第1按左手（46音符）；这是根据当前曲目声部的工程分配。
- 动态规划枚举手指分配，固定持续音手指、避免同一手指同时按不同键，并偏好较小换位／较自然手指次序；允许必要的换指顺序。音程偏置和代价是启发式，没有通过TsenseI逆运动学或专业演奏者验收。
- 标准.mid不会保存本项目NoteSequence.part指法，训练应加载 generated_v1/clair-de-lune-beginner_fingered.proto，不要再用原始.mid。
- CSV finger_id为0～9；finger_number为人读1～5，与左右hand列合用。
- 原始146音符的音高、时间、力度、乐器完全保留；生成proto按时间排序，避免源文件按轨道排序导致原trim_silence漏掉左手开头。实际trim起点0.4347825秒，2587个40Hz控制帧。
- 连续时间及离散帧均无同指多键冲突，全部十指均有分配。完整机器验收见 generated_v1/fingering_report.json。

可重复生成到新目录：

```bash
training/.venv/bin/python scripts/generate_candidate_fingering.py \
  --midi training_mjlab/inputs/clair_de_lune_beginner/clair-de-lune-beginner.mid \
  --output training_mjlab/inputs/clair_de_lune_beginner/generated_v2 \
  --right-instrument 0 --left-instrument 1
```

上面命令从独立项目根目录执行。`generated_v2`必须是新目录。重新生成／改指法后必须重新导出task，已有冻结任务不会改变。
