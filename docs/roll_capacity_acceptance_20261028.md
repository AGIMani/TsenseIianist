# 50/500Hz、4096环境与48M经验池远端联合验收（2026-10-28）

用户要求先验收当前配置、不要正式训练，并将默认Roll开放至±0.5rad。无Roll基线已通过，下文保留其原始证据。用户随后确认Roll右局部+Z、左局部−Z，并将Pitch/Roll力矩限均改为±18N·m；新版本验收见末节。

## 实测结果

- volc-a32 / TenseiNode1，Slurm10459，双A100-SXM4-80GB，16 CPU、640GiB主机内存；统一经scripts/hpc/submit.sh validate，COMPLETED/0:0，8分47秒。
- 50Hz控制、500Hz物理，每步10子步；4096环境。保留原首音物理准备、原谱/指法/奖励/碰撞诊断及双卡采样/学习分工，动作31维、观察1406维。
- 每环境257控制步（5.14秒），共10526720物理子步。全部状态/奖励有限，无接触/约束容量溢出，精确子步目标匹配，局部reset不污染已有经验。
- 单批双卡流水8批、8192次SAC诊断更新通过，尾批flush完整。仅诊断，无正式训练、没有生成训练checkpoint。

| 资源 | 实测占用 | 额度或余量 |
|---|---:|---:|
| FP32 CPU replay全部4800万行内存页 | 508.82GiB | 池保持驻留，与4096环境同时存在 |
| 进程RSS峰值 / Slurm报告峰值 | 514.85 / 519.74GiB | 作业申请640GiB |
| 结束前作业cgroup内存 | 517.48GiB | 剩余122.52GiB |
| cuda:0采样卡采样到的占用峰值 | 55.52GiB | 余23.80GiB |
| cuda:1学习卡采样到的占用峰值 | 0.66GiB | 余78.66GiB |
| Swap | 0 | 无交换依赖 |

先把整个池的所有字段写零，确认RSS真实增长，保持replay.size=0；随后仅插入真实采样的1052672条经验并据此抽样。容量写零不是虚构预填经验。没有完整演奏整曲、没有跑正式学习预算，也不证明接触安全、硬件标定或收敛。GPU占用来自CUDA total-free逐阶段/逐控制步采样，另保留2秒nvidia-smi日志，不宣称捕捉全部瞬时峰值。

## 来源与回传

- 训练默认源码仍790024f85cf6…；验收归档575d5b75b449…仅新增tests/test_roll_capacity_acceptance.py和来源说明，运行时文件不变，不替换默认训练归档。
- 远端目录：/home/houj/projects/pianist_roll_capacity_20261028，作业源码在project/logs/hpc/10459/source；原谱SHA66563e8b5e338c1ef3e426233647da2a008ac931939775b1ee87c36857339af0。
- logs/roll_capacity_20261028/submission.json保存资源与命令；baseline/保存完整config、奖励启动日志、物理/流水报告和GPU监控；后台watch_baseline.py已完成回传并退出，transfer_manifest.json逐文件SHA复核通过。

## Roll与±18N·m当前版本

用户已确认右掌根局部+Z、左掌根局部−Z，Roll范围±0.5rad；Pitch/Roll均Kp=8N·m/rad、阻尼0.5N·m·s/rad、力矩限±18N·m。目标速度3.1416rad/s、加速度15.708rad/s²；实际关节状态的瞬态不等于目标差分限制。原始姿态Roll=0，首音准备允许求解Roll，训练reset读取准备真实终态。

新增真实Roll关节及执行器，完整物理目标53维、默认耦合后动作33维、观察1410维（双方Roll位置及上一实际下发目标）。左手索引、16个PIP/DIP耦合从动通道、限速、准备/回程、动作归一化、奖励腕分组与运动诊断同步。固定Yaw只共轭前序关节，后序Roll保留局部轴；可选机械臂仍固定腕朝向。未迁移旧checkpoint。

当前固定归档`logs/roll_capacity_20261028/source_roll_v1.tar.gz`，SHA256 `afe79f92b803211c390d3a495dc83462ecb37d263a10b5139ada62f4c93f963f`；由790024f8父归档加29项明确覆盖冻结，2256成员审计通过。35项本机轻量检查、36子检查通过，重型测试本机跳过。

远端作业10477经统一submit.sh validate申请双A100/640GiB，COMPLETED/0:0、9分06秒。完整短测通过：每环境257控制步（5.14秒），10526720物理子步、1052672条真实transition、8192次诊断更新；局部reset/尾批flush通过，无非有限状态/奖励或接触约束溢出。本轮没有正式训练或训练checkpoint。

| 新Roll模型资源 | 实测 |
|---|---:|
| 48M FP32 CPU池全部内存页驻留 | 510.603GiB |
| 进程RSS峰值 / Slurm MaxRSS | 516.660 / 521.802GiB |
| 结束前作业cgroup占用 / 额度 | 519.192 / 640GiB |
| 结束前作业内存余量 / swap | 120.808GiB / 0 |
| 采样卡采样峰值 / 余量 | 55.768 / 23.550GiB |
| 学习卡采样峰值 / 余量 | 0.659 / 78.659GiB |

新Roll目标差分最大速度3.1415999867rad/s、最大加速度15.7079999985rad/s²，分别不超过3.1416与15.708。编译后四个Pitch/Roll执行器力限均[-18,18]N·m，Kp8、阻尼.5；Roll轴[0,0,+1]/[0,0,−1]、关节及目标范围±.5。沿用默认hard_v1机器人位置保护；这些目标限制不是实际qvel/qacc保证。初始准备与原谱完全按默认配置运行，短测并非整曲演奏、收敛或长期稳定性证明。

新普通静态包`preview/outputs/piano_roll_20261028/`由同一归档重新完整构建，nq136/nu48、MJB SHA af233c87e3ef…；键程12/10mm、A3.2及200碰撞组件保持。MuJoCo3.11.0本机受限显示通过、物理步0。默认入口绑定新归档及强SHA；相关测试、奖励总表/变更日志已同步。

本轮命令/作业/来源在`logs/roll_capacity_20261028/roll/submission.json`；结果自动回传至同目录acceptance/，传输与完整性结果见sync_status.json及transfer_manifest.json。后台监控曾随shell退出，由独立systemd用户服务pianist-roll-transfer-10477恢复；仍每60秒查询，完成后退出，不依赖Codex继续打开。
