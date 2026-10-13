# 曲谱完整性检查与真实松键尾段

日期：2026-09-28。用户授权直接修复已有确定证据的 F1 问题；主任务同时处理指法预算、replay 预填和参数映射。本文只记录谱面和尾段。

## 问题与取舍

旧 `music/midi_file.py:NoteTrajectory.seq_to_trajectory` 为区分长按和重复按键，会跳过相邻同键的第二个 onset 帧。第二音只有一个控制帧时会完全消失。直接删掉这段处理又会把两个音合并为长按，也不正确。

本次采用 **严格覆盖检查**，保留当前目标时间网格。它不会自动移动音符、延长音符或提高 F1 分数。不能在当前控制周期完整表达的输入会在 Composer 构建时明确报错；用户应提高控制频率或显式修订输入。`strict_v1` 保证“通过检查的谱面没有被静默删音”，不保证任意 MIDI 在 20 Hz 都可表示，也不保证物理上一定能弹出极短音。

## 参数及保存合同

```bash
--score-validation strict_v1
--release-tail-time 0.25
```

- 通用 Python 入口默认 `legacy`、`0`；旧配置缺字段也按这两个值读取。
- `run.sh` 由本轮主任务显式开启 `strict_v1` 和 0.25 秒尾段。
- `strict_v1` 检查每个原始音符在量化/重复音分隔之后有归属于自己的目标帧；检查重复音之间真实存在释放帧，防止合并为长按；检查同指同帧多键、同键重复指派，以及时间、音域、力度、指法和被谱长截断的音符。
- 报错附原 MIDI 序号、音高、起止时间或冲突帧。通过时导出 `score_coverage.json`，每个音包含原始时间、保留帧以及起落时间量化误差。该文件进入 task 的 SHA 校验。
- 尾段必须是有限、非负的整数控制帧。0.25 秒在 20 Hz 下为 5 帧；键、指法和踏板目标全部归零，策略和物理照常运行，奖励、错键惩罚、transition 与 replay 都照常处理。
- `music_end_step` 是演奏结束的 exclusive index，**包含准备前缀**；`release_tail_steps` 单独保存。TaskBundle 校验长度、合同及全空目标。
- 主任务把新增尾帧排除在既有 `music_only_*` 指标之外，并单列尾段误触和最终松键状态。总回合 F1 包含新增帧，因此跨实验应以音乐段指标和尾段指标分别比较，不能用额外静音帧抬高总 F1 作为音乐进步。
- 未修改 reset、终止 discount、20/200 Hz、手模型、关节范围或观测维度。旧 checkpoint 继续依赖原源码/配置，不绕过兼容校验。

## 已执行验证

```bash
training/.venv/bin/python -m unittest \
  tests.test_score_validation tests.test_score_conversion -q

training_mjlab/.venv/bin/python -m unittest \
  tests.test_score_validation tests.test_source_score_contract -q
```

第一组 11 项通过：真实 note-seq 转换复现三种短重复音，严格模式明确拒绝，历史模式保持；有效重复音保留释放帧；当前小星星严格检查通过，原始 34 音全部仍有目标，逐帧 notes/sustains 与原版完全相同。

第二组 13 项通过，验证真实 Composer 导出／TaskBundle 加载、缺字段兼容、边界篡改拒绝和重建参数往返。实测当前小星星从 178 帧增加到 183 帧，`music_end_step=178`、`release_tail_steps=5`；前 178 帧 goal/fingers 与原版逐元素相同。

实际源构建证据：[score_contract_verification.json](../training_mjlab/outputs/f1_integrity_fix_20260928/score_contract_verification.json)。该报告只有源码构建/曲谱验证，不是 GPU 训练收敛证据。

## 文件与后续事项

参考原补充审计 `docs/f1_additional_audit_20260928.md`，修改 `score_validation.py`、`music/midi_file.py`、CPU task/environment、`mjlab_env/source_runtime.py`、`mjlab_env/task_bundle.py` 与两个导出/训练 CLI；三个专用测试文件覆盖纯逻辑、实际转换和源码导出。

- `V32-F1-AUDIT-03`：已阻断静默删音，当前曲全部音符覆盖。低频控制无法表示的输入仍需显式提高频率或修订谱面，不自动猜测音乐意图。
- `V32-F1-AUDIT-04`：已提供真实释放训练尾段和明确音乐边界。最终松键学习效果、原 MIDI 起落误差改善需要新训练/回放验证，不能从目标全零推出手已松键。

根仓库与项目日志由主任务合并记录本轮所有修改。
