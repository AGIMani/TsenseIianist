# 同手自碰撞仅检测与扣分（2026-09-24）

> 历史方案：同日用户随后要求定位根因并恢复统一实体接触，本文默认行为已被[最新修复](contact_stability_shadow_20260924.md)覆盖。检测模式开关仍保留，默认关闭。

用户明确要求同手自碰撞只检测并扣分，完全不产生接触反力。本轮沿用既有self_collision_reward的同手不同body对范围及去重方式，不改变其权重、不启用论文未知碰撞名单。手—琴键、跨手接触仍为物理接触；同手手指可能互相穿过，这是所选行为。

## 使用

重启预览后默认生效（正在运行的窗口不会热更新）：

```bash
.venv/bin/python preview/scripts/preview_pose.py --no-arms
```

临时恢复原实体自碰撞：加`--no-self-collision-detection-only`。预览仅显示／模拟，不运行策略学习；训练中扣分需启用奖励。

GPU训练新增`--self-collision-detection-only`，通用训练入口不带此参数保持旧物理合同。现有`bash scripts/train_twinkle_h12_no_tz.sh`已加入此开关，并保留`--self-collision-weight 0.2`、12cm、TZ物理移除和20/200Hz。此次没有启动新的长期训练。

checkpoint/config记录此布尔字段，独立评估、回放自动恢复；新旧模式的检查点不能静默混用。原冻结包未重写。GPU目前仅支持无M6场景，若有M6须先--no-arms。

## 实现

- `detection_only_collision.py`：CPU预览为原本可碰撞的同手geom对增加inactive pair，保留父子体过滤、装配排除及碰撞掩码。负margin的绝对值大于有限手部几何包围尺寸，gap抵消检测阈值，重叠仍生成记录但无约束力。该实现要求当前MuJoCo>=3.11；在本机3.11、3.13检查通过。
- MuJoCo关于检测距离与激活距离的定义见[官方接触文档](https://mujoco.readthedocs.io/en/latest/computation/#margin-and-gap)。旧版本含不同gap语义，不直接沿用。
- GPU当前MuJoCo-Warp明确拒绝MULTICCD下的非零pair margin，因此不绕过库检查，也不关闭钢琴的MULTICCD。主模型用碰撞位掩码排除同手组合；独立模型沿用原碰撞几何、排除名单，在每物理子步查询碰撞并交给原去重计分器。
- 查询模型仅共享几何位姿，接触／约束数组单独分配，从不执行动力学求解。查询结果不注入主模型接触力。同手检测仍按穿透dist<=0，按控制周期body对去重。
- 保留原主物理中的手—钢琴和跨手相互作用。没有修改驱动增益、阻尼、接触材料或奖励权重。

## 验收

- `tests/test_detection_only_collision.py`共3项通过：CPU真实重叠仍有记录但接触力为0；琴键接触仍激活；碰撞掩码只排除同手；GPU逐步和CUDA graph均检查主接触数组没有同手接触，独立检测产生负奖励，reset正常。
- 另外两环境各40随机控制步测试，逐步自碰撞累计惩罚−166.8、graph −172.0；随机序列不同，不作轨迹一致性比较。CPU主模型琴键活跃接触记录分别171／267，观测奖励有限。
- 无臂源预览headless成功，NVIDIA渲染；4向量步SAC/Replay/融合优化器、独立评估与保存成功（本短测关闭学习器编译）。短测不代表学习收敛或长期吞吐验收。
- 短测输出`training_mjlab/outputs/self_detection_smoke_20260924`；完整回放输出`preview/outputs/replays/self_detection_smoke_20260924`。

## 限制

独立碰撞查询增加GPU检测及显存开销，长期256环境吞吐未测。V32-06几何／真实接触标定及V32-SAC-03回放差异保留。跨手碰撞若也需仅检测，应另行明确范围；本次按上下文只改同手自碰撞。


## 后续覆盖：左右手也取消反力

2026-09-24用户进一步要求跨手碰撞只检测。当前预览默认`--hand-collision-detection-only`，同手＋跨手均无反力；12cm无TZ包装脚本同步使用该开关。旧仅同手开关语义保留。详见[当前说明](all_hand_detection_only_20260924.md)。
