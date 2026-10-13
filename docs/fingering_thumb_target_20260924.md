# 拇指 Fingering 目标点（2026-09-24）

## 用户确认的坐标含义

用户通过键盘平面图确认：“底部朝上”指图中 X 方向，从琴键的 `+world X` 端向 `−world X` 移动，不是 Z 高度。

仅当乐谱指法为右拇指 R1（全局手指0）或左拇指 L1（全局手指5）时，目标 X 改为：

```text
白键 thumb_target_x = key_geom_x + key_half_size_x - 0.020 m
黑键 thumb_target_x = key_geom_x + key_half_size_x - 0.015 m
```

Y 与原规则相同，仍为琴键几何中心；Z 仍为 `key_geom_z + 0.5*key_half_size_z`。其他八根手指继续使用 `key_geom_x + 0.35*key_half_size_x`。

当前白键半长为0.075 m，因此拇指 X 为 `0.055 m`；旧默认点为 `0.02625 m`。当前黑键中心X为−0.030 m、半长0.045 m，因此拇指X为 `0 m`；旧默认点为−0.01425 m。

## 小星星实际拇指键

| 手指 | 音符 | key | 乐谱帧 |
|---|---:|---:|---|
| R1 | C5 | 51 | 10–14、149–157 |
| L1 | C4 | 39 | 10–13、104–116 |
| L1 | D4 | 41 | 83–95 |
| L1 | F4 | 44 | 41–55 |

本曲没有把任何黑键分配给拇指，因此小星星图里只出现白键的新拇指点；黑键1.5 cm规则已由单元测试直接验证。

![小星星所有拇指点](../preview/outputs/fingering_targets_thumb_20260924/twinkle_all_thumb_targets_zoom_final.png)

橙点为该键原有的普通手指目标，粉色圆环为新的拇指目标。图的底部是 `+world X`，所以粉色圆环位于橙点下方，并且距离白键底边2 cm。

第10帧同时使用左右拇指，可在三维场景中核对：

![小星星第10帧](../preview/outputs/fingering_targets_thumb_20260924/twinkle_frame10_thumb_targets_final.png)

## 冻结包与兼容性

- 新任务：`training_mjlab/outputs/task_twinkle_noarms_20hz200hz_thumb_targets_20260924`。
- scene、20/200 Hz、1180维观测、51维动作与乐谱均未改变。
- 旧 task、旧 checkpoint 和既有视频继续按旧目标点解释；新规则改变奖励，不允许拿旧 checkpoint 冒充按新规则训练所得结果。
- 新 task 的 manifest 冻结了坐标系、拇指编号及白／黑键距离。缺少该字段的旧 bundle 自动走旧目标逻辑。

论文 p.11 只公开了指法距离的1 cm成功区间和10 cm margin，没有公开琴键目标点位置。本次白／黑键拇指点是用户确认的 TsenseI 适配，不称为论文参数。

## 验证

- `tests.test_fingering_targets`、`tests.test_mjlab_task`、`tests.test_paper_rewards`：20项通过；CPU环境中的几何测试3项通过。
- 新task导出158帧、1180维观测。
- 1环境GPU完整任务验收通过；冻结CPU状态与GPU规则的 fingering reward 最大误差 `1.2381e-7`，总奖励最大误差低于 `2.4e-7`。
- 旧task和新task各执行GPU reset＋step：旧目标张量保持`1×88×3`，新目标为`1×10×88×3`，两者奖励均有限，兼容分支通过。
- 验收报告：`training_mjlab/outputs/task_acceptance_thumb_targets_20260924_v2/report.json`。
