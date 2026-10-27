# 土耳其进行曲全曲检查与重复片段复用

用户明确纠正：需要检查整首曲子的指法，重复小节可以复用。上轮305拇指候选只完成全曲事件排程和前20秒物理验证；本轮补足全部2544音符、原谱约222.473958秒的检查和可交互交付，不启动训练。

沿用当前候选、固定exper038模型/归档、20Hz控制和已授权1000Hz物理；原谱音高、起止、力度、轨道与元数据不改。少用但保留拇指。TZ仍默认关闭；研究代码不自动成为训练源码。

## 方法选择

只看前20秒或只跑一个重复小节不能覆盖全曲；直接复制同音型的物理状态会抹掉边界与动力学误差。采用用户指定的重复复用思路：按原MIDI tick/节拍识别严格重复，区分移调和近似；在兼容的持键/手位上下文中复用指法和几何候选，重算边界衔接，再连续积分整首曲子。不是把局部成功拼接成完整物理回放。

全曲事件审计记录每音、每小节、重复组及边界：占指冲突、同指恢复间隔、跨度/静态残差、拇指次数、前后手位和原谱保持。受约束DP候选与当前305方案比较，失败或退选记录保留。仅在证据支持后替换默认外部指法。

对当前候选先取得连续全曲物理基线。改进候选若变化，必须用匹配模型/参考重新连续录制全曲；报告分段与整体实际100%键程的TP/FP/FN/F1、漏键/错键、受力同手/跨手接触及硬目标限幅。重复段的每次出现都保留实际测量，不共享物理成绩。

## 交付与边界

全曲交互MJB回放、同次实际声音、原谱/实际对比PNG与逐段数组，能直接跳至问题小节；几何指法查看与动态回放明确区分。记录精确输入/源码SHA、命令、PID、退出状态。全部模型/物理与大型求解在volc-a32独立目录CPU；本地只编辑、轻量文件检查、已录制数组分析和受保护MJB显示。

验收是全曲覆盖与真实证据，不预先保证全曲无漏键/无碰撞。若物理问题仍存在，逐个标明出现位置、重复组与证据，不能以运行结束声称已演奏正确。保留旧结果、其他会话修改和当前奖励/模型/训练归档。
# Added request: initial Palm_Base_Ty

The user explicitly authorized exploring a backward shift of the initial bilateral Palm_Base_Ty and directly applying it if helpful. Measure axis direction from the actual model. Compare the existing --ty0.04 with modest offsets using the newly frozen first_note_v1 preparation source, keeping its unprojected physical recording and original score. Selection must consider both hands' total travel, peak actual motion, terminal hover accuracy and unintended contacts. Initial placement is distinct from joint range and cannot by itself fix every mid-song wrist jump. The original full-song comparison continues unchanged for a fair baseline.
