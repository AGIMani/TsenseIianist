# Fingering 奖励目标点图（2026-09-24）

## 当前定义

当前 GPU 任务与 CPU legacy 任务对每个琴键定义一个三维目标点：

```text
target = key_geom_xpos + [0.35 * key_geom_half_size_x,
                          0,
                          0.5  * key_geom_half_size_z]
```

偏移量直接加在世界坐标轴上，不随琴键局部旋转。`geom_size` 是 MuJoCo 的半尺寸，因此旧默认 X 偏移是琴键完整长度的17.5%；Z 是从几何中心向上半个半厚度，即从键底到键顶的75%位置，不是上表面。奖励计算的是乐谱指定指尖 site 到对应目标点的三维欧氏距离。成功区间为 0～0.01 m，Gaussian margin 为 0.1 m；同一帧所有有效“手指—琴键”分配的得分取平均。无目标音符帧返回 0。

2026-09-24起，新导出的task对拇指增加了用户确认的X方向覆盖：白键从图中底边向上2 cm，黑键向上1.5 cm；旧冻结task保持上述默认点。详见[拇指目标点说明](fingering_thumb_target_20260924.md)。

论文 p.11 的指法距离奖励采用相同的 0.01 m 成功区间和 0.1 m margin，但没有公开 TsenseI 琴键上目标点的具体几何定义；本图展示当前 CPU-compatible 实现，不把目标点位置称为论文参数。

## 图示

- 小橙点：88 个琴键各自的精确奖励目标点。
- 大彩色点：第 120 帧实际参与 fingering reward 的目标。
- 彩色线：指定指尖 site 到目标点的距离。
- 第 120 帧为当前无机械臂小星星 `step-17000` 回放状态：R3→E5 距离 40.13 cm，L2→A3 距离 1.86 cm。

[正面图](../preview/outputs/fingering_targets_20260924/frame120_exact_targets.png)；[俯视图](../preview/outputs/fingering_targets_20260924/frame120_exact_targets_overview.png)。同目录 JSON 保存目标世界坐标和距离。

[88键逐键标注图](../preview/outputs/fingering_targets_20260924/all_88_keys_labeled_v2.png)直接在每根白键和黑键上写出音名与 `key` 编号。每个橙点都是该键的精确奖励点；第120帧的 `L2→A3` 与 `R3→E5` 另有红色／青色圆环。

## 复现

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python \
  preview/scripts/draw_fingering_reward_targets.py \
  --frame 120 --view front \
  --output preview/outputs/fingering_targets_20260924/frame120_exact_targets.png \
  --keyboard-map-output preview/outputs/fingering_targets_20260924/all_88_keys_labeled_v2.png
```

脚本只读取冻结 scene/task 和 rollout，不修改物理模型、奖励或训练结果。现有 `FingeringHighlight` 彩球是便于观看的指法标记，使用不同显示偏移；不能把它当作奖励目标点。本脚本单独按照奖励公式绘制。
