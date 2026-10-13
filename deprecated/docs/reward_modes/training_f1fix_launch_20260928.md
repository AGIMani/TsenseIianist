# TsenseI_A3.2 F1 修复后正式训练

启动时间：2026-09-28T17:39:16.659945+08:00（北京时间）。用户要求“帮我开启一次训练”，本轮按当前已验收配置从头训练小星星，不加载旧 best。

## 入口与状态

- 命令：`bash run_tsensei_a32.sh --output-dir training_mjlab/outputs/TsenseI_A3.2_f1fix_20260928-173915/run`
- 后台启动 PID：`417273`；独立 session、stdin=/dev/null，关闭当前任务界面不依赖终端会话。
- 实际训练目录：[run-20260928-173916-689809](../training_mjlab/outputs/TsenseI_A3.2_f1fix_20260928-173915/run-20260928-173916-689809)。
- [完整 config.json](../training_mjlab/outputs/TsenseI_A3.2_f1fix_20260928-173915/run-20260928-173916-689809/config.json)；[控制台日志](../training_mjlab/outputs/TsenseI_A3.2_f1fix_20260928-173915/console.log)；[启动记录和源码指纹](../training_mjlab/outputs/TsenseI_A3.2_f1fix_20260928-173915/launch.json)。
- [SwanLab 实时曲线](https://swanlab.cn/@blackswan17/TsenseI_A3.2/runs/otw79aa2)，项目 TsenseI_A3.2，保持 F1、recall、precision、字母序奖励的主图顺序。
- 这是启动时的状态记录，最终结果以本轮 metrics.jsonl／summary.json 为准。

## 本轮实际设置

| 项目 | 设置 |
| --- | --- |
| 预算 | 512环境×9766向量步＝5,000,192 transition |
| 动作／观测 | Direct47动作／1231维；八四指MCP上限1.3rad |
| SAC | Actor/Critic256×3，三LR3e-4，gamma.8，初始温度1，batch1024 |
| 学习调度 | replay1M；前4999条随机；完整183帧预填93696条后开始；稳态.25更新/transition |
| 时序 | 控制20Hz／物理200Hz，10子步 |
| 准备／释放 | 1秒可训练准备，initial抬高2.5cm；曲末.25秒真实松键 |
| 提前定位 | .15秒；当前指法优先开启；分阶段指法关闭 |
| 按键奖励 | 子步错键／功率汇总开启，正确键子步均值开启，真实触发占比.5 |
| 附加惩罚 | 自碰撞.2、错键下压1、geometry_clearance humanoid总预算≤.1 |
| 控制 | 腕TX/TY限速1.2m/s、Pitch4rad/s；闲指保护/用指许可关闭 |
| 验证与保存 | 每100向量步评估3回合，保存合格音乐best，每500步保存checkpoint |
| 质量目标 | 音乐F1和micro各.99，连续3评估且满足准备误触、碰撞、键底和最终松键门槛 |

提前停止仍须通过全部质量条件；不能仅凭F1曲线超过.99宣称训练达标。

## 启动核验

后续已观察到第400向量步／204,800样本、27,776次更新，学习损失数值有限；训练GPU进程417609仍运行，显存约14.6GiB。SwanLab状态为uploading，状态文件已确认25组指标提交（至117,760样本，状态每25组落盘，因此会落后于正在训练的步数）。第300步的3回合评估完整，音乐F1约.057，属初期结果，尚无收敛结论。

独立只读复查当前源码／模型／Direct JSON、33个学习器文件和奖励／控制JSON与14:57成功短训一致；当前脚本语法通过。实际新config确认1231/47、Direct1.3、三LR、183帧与93697学习起点。新源码配方的所有散列与本轮归档逐项相同。

源码归档 `source.tar.gz` 的SHA256为 `d256202c4753e68b4aac7d7ec37185a97166faa52faf40edbec449b7fa461167`；另保存本次 `run_tsensei_a32.sh` 入口副本。旧实验、best与源码归档未修改，未更改训练代码或超参数。密钥仍仅从既有外部文件读取，不写入源码、启动参数或日志。

后续关注 V32-QUALITY-01（正式F1／碰撞结果）、V32-F1-AUDIT-01/02（预填和当前指法优先的收益）与V32-F1-AUDIT-04（曲末释放效果）。本次启动不代表目标已达成。
