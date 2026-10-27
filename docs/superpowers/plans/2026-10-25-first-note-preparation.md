# 首音准备实现计划

> 执行：本会话按 executing-plans / test-driven-development 实施，最终独立审查。

Goal: 准备控制器平滑移动到双手首音上方，真实物理终态交接，验证后更新冻结归档和默认入口，交付同次准备视频/实际按键对比。
Architecture: 新first_note_v1固定物理前缀在SceneBundle生成；目标从匹配任务的精确事件读取。CPU IK仅操作scratch data；动态数据仅mj_step推进。缓存的真实终态用于每回合reset，原始前缀用于导出，不进入SAC replay。
Spec: docs/superpowers/specs/2026-10-25-first-note-preparation-design.md（2026-10-25用户已批准）
Constraints: 远端物理/IK；GPU只Slurm；保留真实轨迹；原谱/指法/学习预算不变；不覆盖历史结果；只发布本任务验证的归档补丁。
Review focus: 单手/和弦指法；首音不可达；悬停实体误触；终态/限速历史交接；导出前缀时间偏移及归档源码一致。

## Task 1: 首音规划与物理准备
- [x] tests/test_first_note_preparation.py先覆盖事件选择、平滑曲线边界与速度/加速度、输入拒绝，见证失败。
- [x] first_note_preparation.py实现独立事件选择、IK、真实物理前缀、逐步诊断和失败报告。
- [x] preparation/bundle/options接入；远端验证默认曲目悬停、接触、终态，不回写实际状态。
## Task 2: 配置、训练与回放接入
- [x] source_runtime/train_from_source/rl_torch配置一致；准备上下文记录动作/耦合/目标限速。
- [x] reset沿用实际终态和下发目标；原谱/学习样本预算不变；不支持带臂时明确拒绝。
- [x] targeted CPU/GPU回归与异步reset、源重建/交接验证。
## Task 3: 冻结发布与视频
- [x] 从当前默认归档加本任务补丁构建新归档、SHA与来源清单；远端实际使用该归档验证。
- [x] Slurm同次准备轨迹渲染有声视频、原谱对比PNG和数组；音视频/SHA校验后回传。
- [x] 更新默认入口、README、奖励总表准备语义、变更日志；独立审查并修复必要问题。

## Ledger
- 用户已明确批准直接实施及视频导出，不再重复请求流程审批。
- Ruling: 保留现有工作区未提交变更，先备份涉及文件，再以当前入口冻结源码加明确补丁部署；避免把无关改动纳入训练归档。
- Ruling: 固定前缀每个环境构建时真实仿真一次；reset使用完整实际终态，重用可审计准备录制。这延续已有fixed准备的缓存机制，不伪造策略轨迹。

- 验收：Slurm10039运行最终冻结归档，21 tests/21 subtests通过，1项已知父源码music_only_f1失败独立排除并留存基线证据；Slurm10038同轨迹视频导出完成。
- 实际物理硬限与悬停验收分开：左腕8.3268e−5rad瞬态越界保留，未做状态投影；末端、无触键与GPU交接门禁通过。

- 最终发布：归档加入exper042真实物理导出修复，Slurm10040为最终SHA做联合回归：99 tests/21 subtests通过；1项既有基线失败排除。默认入口/SHA与回传文件一致。
