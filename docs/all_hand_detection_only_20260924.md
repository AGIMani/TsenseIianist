# 左右手之间也仅检测碰撞（2026-09-24）

> 历史方案：同日用户随后要求参考 Shadow 定位根因、统一实体接触，当前默认已恢复三类物理反力，本文默认行为和启动说明已被[最新修复](contact_stability_shadow_20260924.md)覆盖。开关仍可显式启用以兼容旧模式。

用户明确要求关闭左右手碰撞反力，保留手与琴键的真实接触。覆盖此前仅同手检测的默认预览范围。当前同手以及跨手的掌／指接触均不产生反力，仍由检测结果计算原有奖励；手与琴键仍可真实按压。允许手部互相穿过是所选模式的结果。

## 使用

关闭当前预览窗口后重新运行：

```bash
.venv/bin/python preview/scripts/preview_pose.py --no-arms
```

预览新增`--hand-collision-detection-only`，默认开启并涵盖同手＋跨手。`--no-hand-collision-detection-only`退回此前仅同手模式；再加`--no-self-collision-detection-only`恢复全部手部实体碰撞。未重启用户现有进程。

GPU训练通过`--hand-collision-detection-only`启用，现有`bash scripts/train_twinkle_h12_no_tz.sh`已采用此参数。旧`--self-collision-detection-only`仍保持仅同手，避免旧检查点语义改变。开关独立记录到config/checkpoint，评估／回放恢复并检查一致性。未更改12cm、20/200Hz、TZ物理移除、self_collision权重0.2及其他奖励参数，未启动长期训练。

## 实现与奖励

CPU预览将inactive接触对覆盖范围从同手扩展到任意手—手，保留原几何／父子／装配排除。GPU主模型碰撞位掩码屏蔽所有手—手组合，独立查询模型沿用原始掩码，既给每子步自碰撞去重计分，也在控制周期末给forearm判定。

跨手奖励仍是无碰撞+0.5、有碰撞0，未改成按数量扣分。即使self_collision_reward关闭，reset和控制周期末仍刷新独立查询，forearm不依赖同手跟踪器开关。琴键接触始终留在主物理求解；手与琴体的原接触配置也未作额外改变。本次不新增M6动力学支持，GPU模式仍要求无臂输入。

## 验证

- `scripts/validate_hand_detection_only.py`：CPU构造左右手球体相交及琴键接触，跨手接触efc_address=-1，琴键接触约束有效。
- 实际冻结模型GPU逐步和CUDA graph各2环境35控制步，将左右手横移到重叠位置，主GPU接触数组没有任何手—手组合，独立查询检测碰撞，forearm奖励为0；测试特意关闭self_collision_reward，验证奖励之间独立。观测均有限。
- 源预览headless渲染成功，NVIDIA GPU，生成/tmp/cross_hand_detection_preview.png。
- `all_hand_detection_smoke_20260924`完成4向量步、3次SAC更新、独立评估和保存（本短测关闭学习器编译）；回放产物`preview/outputs/replays/all_hand_detection_smoke_20260924`。不代表长期收敛。
- 原始验证日志：`training_mjlab/outputs/all_hand_detection_validation_20260924.log`。

## 保留事项

独立查询增加计算／显存，256环境吞吐未验收。V32-SAC-03、V32-06保留；此改动消除手—手反力来源，不保证位置驱动／琴键接触造成的所有抖动都消失。
