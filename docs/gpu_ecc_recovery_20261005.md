# 2026-10-05 GPU ECC故障现场与恢复

## 已确认现场

北京时间18:29–18:33检查。1724已正常完成，best19500的SHA256为
`06c3abdb71db043b3bf44c0f7b69539ac690e5ee1008bc0ba2334f36be6f3cd8`。
1726在物理GPU 2、3上运行约5分57秒后报`CUDA error: uncorrectable ECC error encountered`并退出；CUDA调用异步，Python报错行不足以确认首次故障位置或具体GPU。

- GPU 2：`GPU-7ab6c8b2-f11c-553f-bc21-caa713f0d29c`，PCI `0000:69:00.0`。
- GPU 3：`GPU-82b1a3f9-080a-94f3-2cfe-066844e06fb7`，PCI `0000:6f:00.0`。
- 1727随后获配同一对卡，卡在`probe_gpus.py`，没有生成探测报告或开始训练。
- 1725获配GPU 5、6，日志停在预填183步／93696样本／0更新，主进程处于D状态、等待`uvm_gpu_release`。它与ECC的具体因果关系尚未由内核日志证实。
- 只读`nvidia-smi`查询返回`Failed to initialize NVML: Unknown Error`。
- 节点已为`MIXED+COMPLETING+DRAIN`，Slurm原因是其他用户作业1738无法清理，时间17:03:50；不是本次操作设置的DRAIN。还有其他用户运行中作业。
- 当前账户无免密sudo，不能读取dmesg／系统内核日志，无法确认Xid、ECC计数或执行GPU恢复。

ECC表示设备检测到无法纠正的存储数据错误。A100可隔离部分错误；是否需要重启应用、重置GPU或重启节点，应由Xid和恢复状态决定。不能仅凭这一次报错判定显卡永久损坏，也不能归因于奖励配置或具体Python报错行。此前1724在5、6卡成功不代表2、3卡当时或现在健康。

## 已执行的恢复准备

保存现场、原GPU探测、错误日志、命令和遥测末段于`logs/hpc/ecc_recovery_20261005/`。
复制并校验父best、config、best指标及summary。检查统一入口、固定源码归档和三份冻结奖励JSON的SHA与原提交一致。

用户要求修复后，对本用户卡死的1725和1727执行`scancel`，调度器接受取消请求；最初两项为COMPLETING，18:33复核已从队列移除且sacct为CANCELLED，但两个孤儿进程仍残留，均有SIGKILL待处理（SigPnd=0x100）。节点DRAIN原因现为1725无法清理；Slurm撤销分配不代表GPU／驱动资源实际释放。没有清除ECC计数、关闭ECC、改驱动、操作其他用户作业或重启机器。

`restart_plan.json`保存三组通过原`all_use_this_train.sh`重启的完整命令、奖励与校验信息，输出使用新的`_ecc_retry`前缀。尚未提交。1725没有记录学习更新，1726仅有起点评估，1727未进入训练；恢复时三组仍从同一父best网络重新开始原追加预算，不把各组step0 best当作训练收益或续训进度。

## 管理员恢复步骤

1. 保持节点停止接收新任务，先读取并保存内核Xid和Fabric Manager日志：

   ```bash
   sudo journalctl -k --since '2026-10-05 16:20:00' --no-pager
   sudo journalctl -u nvidia-fabricmanager --since '2026-10-05 16:20:00' --no-pager
   sudo nvidia-smi -q
   ```

2. 根据Xid和UUID确认受影响设备，并协调节点上其他用户的任务。A100的Xid94可能只需要重启受影响应用；Xid95、行重映射失败、驱动不可恢复等情况需要按NVIDIA指南进行GPU重置或节点重启。当前NVML失败且多个进程不能正常退出，不能以反复重提训练替代驱动恢复。不要在未核对使用者及NVSwitch/Fabric Manager条件前重置所有GPU。
3. 管理员完成设备恢复和必要健康检查后恢复Slurm调度；通过项目入口提交验收：

   ```bash
   cd /home/houj/projects/tsensei_v3_2_pianist
   bash scripts/hpc/submit.sh probe
   ```

   必须检查实际获配UUID、`gpu_probe.json`通过，且故障设备的ECC／Xid不再新增。项目probe只测试获配的两卡及基本计算／互传／Warp，不替代管理员对故障卡的硬件诊断。
4. 验收后按`restart_plan.json`逐组打印完整奖励并从统一入口重新提交；检查学习更新数持续增加和首个训练后评估完成。重置后若ECC复发，继续隔离故障卡并进行厂商诊断。

参考：[NVIDIA A100 ECC恢复机制](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/response-to-uncorrectable-contained-ecc-errors.html)、[NVIDIA节点故障处理与GPU重置条件](https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html)。本次尚未取得管理员内核日志，硬件／驱动恢复与重新训练均未完成。
