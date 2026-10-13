# Deprecated 方案隔离区

这里保存已经明确失败、弃用或仅用于历史审计的配置、脚本、文档、测试和候选产物。当前训练入口不会读取本目录，当前代码也不为这些方案做 checkpoint 兼容转换。

- `arm_control_experiments/`：旧机械臂 schema1/schema2 及候选安装、振荡分析和旧镜像安装导出。
- `reward_experiments/`：旧奖励配置、实验目录和历史奖励模式。
- `legacy_entries/`：旧训练入口和旧奖励脚本。
- `tests/`：仅针对旧方案的测试。
- `docs/reward_modes/`：旧奖励模式与实验说明。
- `docs/historical/`：旧机械臂和安装说明。
- `docs/change_log_history_20261011.md`：清理前的历史日志。

新的设计直接修改活动实现，发现错误后继续在活动实现上修正，不再新增递增 schema 或 mode 名称。

`current_arm_training_20261011_v8.json` 是本轮统一前生成的重复预览快照，已与旧臂方案一起隔离；活动配置只有 `configs/preview/current_arm_training_20261011.json`。

`arm_control_experiments/outputs/` 保存清理前的预览图、场景和安装快照；它们不是当前几何或初始角的来源。
