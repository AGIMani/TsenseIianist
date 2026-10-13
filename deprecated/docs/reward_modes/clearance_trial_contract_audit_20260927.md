# 2026-09-27：净空短训与正式续训的CPU合同复核

本复核不占GPU、不构建环境、不修改正式包，只读取实际config、归档、检查点与日志。可复现脚本为 `scripts/audit_clearance_trials.py`，完整结果在 `preview/outputs/diagnostics/quality_clearance_20260927/trial_contract_audit.json`。

## 源码及物理身份

- `training_mjlab/source_versions/generic_clearance_trials_20260927.tar.gz` 的 SHA-256 为 `1caf48862dbfe1da1e699a521322155c2cf26c024df6bbba79ea578c6299da93`。
- 归档共432个普通文件，其中受source合同约束的包文件166个；两组短训及正式续训逐文件散列全部匹配，复核时当前包也逐文件相同。
- scene/task散列与各自父检查点完全相同，完整origin/initial/preparation合同、1181维观测、47动作及动作映射保持。
- 新软件腕约束只有TX/TY 1.2m/s、Pitch4rad/s；解析后的上下界等于原范围，没有旧R_TX[-.2,.2]/L_TX[-.1,.2]裁剪。
- 规范配置与resolved合同一致：`active_command_mode=active_contact_v1`、`geometry_clearance_v1`奖励及键底观察器确实启用；`max_key_underside_substeps=0`同时约束best候选与停止。

## 两组512向量步短训

每组512环境×512步=262144样本，先收集完整178帧×512=91136样本，再按0.25更新/transition，实际新增更新 **42752次**，batch1024。无均匀随机探索，预填仍是策略采样。每条训练/评估日志与预算相符；episode结束事件写在当向量学习块之前，不能用它误判少128次更新。

| 状态 | 音乐macro F1 | 音乐micro F1 | 键底子步 | 侧/底子步 | 手—手子步 | 准备错帧 |
|---|---:|---:|---:|---:|---:|---:|
| 同父网络接入新控制后的step0 | 0.915341 | 0.944544 | 0 | 96 | 7 | 0 |
| Actor3e-5，step512 best/final | **0.921730** | 0.939286 | 0 | 132 | 35 | 1 |
| Actor3e-4，step512 final | 0.867541 | 0.872340 | 0 | 137 | 16 | 1 |

低LR best=step512，文件与final逐字节相同（SHA `a37f4dd36ffc3663da17441d4866563a0d35172e9436fd065c860cb0d0cd71c9`）。高LR best保留step0。两组真实采样均完整，无键底导致的best拒绝事件；拒绝缺统计/卡底高分的行为另外由单元测试验证。

耗时低LR147.486秒、高LR147.983秒；以trainer总耗时计算分别1777.4、1771.5样本/s，包含该计时范围内的初始化和评估。日志中开始更新后的循环平均约2197.7和2193.3样本/s，不能混作纯学习器速度或GPU极限。

短训均从原best只迁移网络：父alpha=0.0016916337190195918，优化器/RNG重新初始化、replay重建。高LR保存的step0检查点提供直接证据：Actor/Critic/target Critic所有张量与父逐bit相同，三组优化器state为空、updates=0。低LR初始best已被更高分覆盖，不能声称其原step0文件现在仍可直接比较。

## 正式续训确实恢复完整学习器

运行目录：`training_mjlab/outputs/twinkle_generic_clearance_finetune_20260927-20260927-093425-663203`。

- 此次是 `learner_warm_restart`，从低LR step512 best恢复，不是再次只迁移网络。`allow_control_change=false`、`allow_reward_change=false`，父与子奖励和控制合同一致。
- 配置中的init-temperature保留0.0016916337190195918以匹配保存的SAC配置；**实际恢复的已学alpha为0.002877362770959735**，不是把温度重置为配置初值。
- 读取正式run step0 best时，Actor/Critic/target网络、action/update/exploration RNG逐bit等于低LR父；updates=42752，Actor/Critic/温度Adam均非空（10/28/1个参数state），所有step、第一/第二矩张量逐bit恢复。
- 原始optimizer dict仅`capturable`从True变False；`sac.py::load_state_dict`先清GPU图执行标志，`learner_graph.py`随后重开，不是重置Adam。历史step0比对记录保存在`*_initial_migration.json`和`warm_restart_optimizer_diff_20260927.json`；训练后来已把best替换为更高合格分数，旧step0文件未另复制。
- replay与world重新开始，仍先预填91136条完整策略transition。计划9766向量步/5000192新增样本/1227264新增更新；若跑满，总更新为父42752+1227264=1270016。不是精确恢复旧world/replay。
- 本报告快照读到step1100/累计160768更新；当时best=step1000、macro F1=0.938819，已有新增更新，不能把启动成功写成长期收敛。

## 选择低LR step512续训的证据与风险

相同新环境下低LR比高LR保分好，完整恢复已适应新奖励的critic/Adam有合理依据，没有合同层面的阻断。但短训不足以证明从step512必然优于step0：低LR只把macro从0.91534提高到0.92173，micro反而下降；准备错帧从0变1、手碰撞7→35、侧/底96→132。

键底为0本身不能验证所有未用指一直在琴面上方。低LR评估仍报告最大实际几何缺口约15.62mm，但该全指汇总没有按当前闲/用指拆分，不能仅凭此值断言是哪根闲指违规，仍需详细轨迹分类。新控制相对原无guard best 0.98228仍明显损失音乐分数。当前best资格只额外限制键底，手碰撞/准备误触仍属于整体达标门槛，不能把“best_eligible”翻译成全部质量目标已通过。

建议保持已保存合格best并持续看音乐macro/micro、准备误触、侧碰、手碰撞和实际缺口，只有完整验收后才宣布根治；不因一次短训优胜承诺0.99或普适无卡键。本复核没有修改正在进行的训练。

## 审计工具修正记录

运行期间第二组config曾处于GPU初始化前的未完成状态（尚无origin），工具现明确标记等待完整配置，不误报损坏。checkpoint内部tuple与JSON list按正式eval入口相同的JSON语义比较；episode日志按采集前后实际时序核预算。上述修正只在包外诊断脚本，正式源码166文件未变。
