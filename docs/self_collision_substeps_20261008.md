# 同手碰撞子步累计与控制步封顶（更新2026-10-13）

当前唯一总表见[奖励与惩罚总表](reward_defaults_20261006.md)。统一入口默认：

```json
"self_collision_reward": {
  "enabled": true,
  "weight": 0.1,
  "params": {
    "aggregation": "substep_sum",
    "maximum_penalty_per_step": 5.0
  }
}
```

```text
C = Σ_physics_substeps N_unique_same_hand_distinct_finger_body_pairs(substep)
r_self = -min(weight × C, maximum_penalty_per_step)
```

仅计同手不同手指的接触；掌壳与手指、同一指MCP/PIP/DIP/Swing、同一拇指MCP/PIP/rota_block/rotaback_link2均排除。筛选只影响奖励，物理接触及完整诊断保持。

先对每个子步内同一body对的多个geom/接触点去重，再跨子步累计；**乘weight后才整项封顶**，不逐对封顶、不逐子步封顶、不除以子步数、不乘dt。当前每控制步20Hz对应10个200Hz物理子步。

| 合资格同手不同手指接触情况 | 累计对次C | 当前最终奖励 |
|---|---:|---:|
| 没有同手接触 | 0 | 0 |
| 一对只接触一个子步 | 1 | −0.1 |
| 一对连续接触10个子步 | 10 | −1 |
| 两对各连续接触10个子步 | 20 | −2 |
| 共24对次 | 24 | −2.4 |
| 共25对次 | 25 | −2.5 |
| 三对各连续接触10个子步 | 30 | −3 |
| 五对各连续接触10个子步 | 50 | −5 |

无臂／带臂同一规则。引擎的物理接触过滤、自动恢复的父子排除和非相邻自碰撞检查保持；跨手及手—琴键不计入。末端额外forward不重复计次；关闭诊断仍可累计。控制步开始清空计数，全量/局部reset只清理对应环境。改变物理频率会改变相同接触时长的对次及达到封顶的时间，仍不是按秒归一化罚率。

`self_collision_pair_substeps`和raw保留完整对次数（raw为−C），不被封顶截断；weighted/final为实际封顶分。`self_collision_pairs`仍是周期并集诊断，不用于奖励；`self_collision_substeps`仅表示发生同手接触的子步数，不等于对次。

本次按维护规则删除旧周期并集奖励路径；`params={}`或仅有aggregation而缺少maximum_penalty_per_step的旧不封顶配置在新源码中明确拒绝。当前实现不会自动补该字段；旧 JSON 和历史产物已移入 `deprecated/`，当前训练不做 checkpoint 兼容转换。新模板、默认JSON及`--self-collision-weight`入口均使用同一封顶路径。

历史验收归档：`logs/arm_ccs696_coupling_20261011/source_arm_ccs696_coupling_v5.tar.gz`，父归档为`logs/arm_parent_filter_fix_20261010/source_rigid_arm_parent_filter_v1.tar.gz`（SHA5580e1bffbcf75bbcd507af9b27af3b551bd0a6c0cdd6eacf1cece64ecbd1a06），只叠加本任务白名单文件，SHA由统一脚本固定。验证记录见同目录。CPU覆盖去重、累计、封顶边界、weight/上限为0、日志一致、dt不缩放、局部reset、旧配置拒绝和入口合同；本次未运行GPU或启动训练。
