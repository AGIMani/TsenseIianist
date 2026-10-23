# 默认腕驱动恢复exper031及exper034

按用户2026-10-22要求，默认 `shadow_v1` 腕驱动恢复exper031的数值；保留当前指尖任务点回退3mm、100%实际激活和1224mm琴键布局。由当前指尖归档（SHA256 `4f37d650a64e936484a8a7fadd30f51ac6c91e8dce7a480fc16b91f547d96f74`）派生，不能直接切回exper031旧源码。

| 双侧轴 | Kp | 阻尼 | 输出限幅 |
|---|---:|---:|---:|
| TX/TY | 300 N/m | 34.15259872981849 N·s/m | ±50 N |
| Pitch | 8 N·m/rad | 0.5 N·m·s/rad | ±5 N·m |

保留逐轴显式阻尼结构、平移armature=0.02kg及编译M0=0.972kg校验；阻尼数值与exper031当时的临界阻尼相同。手指驱动、MCP范围、Pitch范围、耦合10°～80°、目标运动限制、闲指参考、同手和跨手碰撞等均不变。恢复的是仿真驱动数值，不是硬件标定。

exper034继续使用exper033的注册015-5修订v2乐谱及原时值，原预算19532个向量步×512环境=10,000,384条经验（约1000万）；batch1024、每条经验0.25次更新、20Hz控制/200Hz物理、cuda:0采样/cuda:1学习、TF32学习器与原seed42保持，从头训练。通过唯一入口 `all_use_this_train.sh` 追加MIDI和输出目录参数。

同时补齐冻结包此前遗漏的本地录制回放启动器/查看器，纳入已完成的无悬浮标签显示修复及交互回放验收流程，保留目标键颜色、实际触键高亮和声音。显示文件不改变训练奖励/动力学；对应CPU导出合同测试一并验收。

远端独立项目 `/home/houj/projects/pianist_exper034_20261022`，冻结归档 `logs/wrist_default031_20261022/source_wrist_default031_v1.tar.gz`。验收命令、日志和来源在同目录；实际训练/导出作业ID、状态、SHA及回传进程见 `results/exper034/submission.json`、`sync_status.json` 和 `transfer.log`，不能把排队视为已经开始学习。

训练终态后，现有本地后台 `scripts/watch_training_transfer.py` 自动提交best.pt的Slurm导出，生成同次物理回放的交互有声回放和原谱对比PNG/事件数据，回传checkpoint、配置、指标、奖励启动日志、来源及SHA。全部验收完成才创建 `TRANSFER_COMPLETE`。后台进程不依赖Codex打开，但需要本机/SSH保持可用。

CPU编译及运行时检查结果由 `logs/wrist_default031_20261022/release_validation.json` 记录。F1和接触表现须等本轮完成后实测；本轮同时包含几何/阈值/布局更新，不是仅腕驱动单因素对照（该对照为exper032/033）。
