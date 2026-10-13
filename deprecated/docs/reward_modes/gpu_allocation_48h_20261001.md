# 两张GPU连续保留48小时

用户于2026-10-01明确要求保留2张GPU共48小时。已通过项目Slurm入口提交作业 **1364**，名称`TsenseI-reserve2`。

后续用户批准的精确时序修复测试安排复用1364，由后台协调器等待分配后依次执行，不新增两卡申请。共同参数为60°–80°耦合、10倍gesture得分、TX/TY1m/s；详细队列见[修复测试说明](note_timing_repair_tests_20261001.md)。单项结束或队列失败后保留进程继续等待。

## 分配规则

- 申请同一节点2张GPU、16个CPU、128GB内存；本集群GPU型号为A100。
- Slurm已接受`TimeLimit=2-00:00:00`。48小时从实际开始运行计算，排队时间不扣除。
- 这是普通Slurm资源分配，按现有优先级等候，不抢占其他用户作业，也不使用管理员reservation权限。
- 保留进程只等待，不运行GPU计算。后续任务结束后保留进程仍存在，资源到期或用户提前取消才释放。
- 保留申请本身不启动训练；后续队列通过独立srun步骤执行。

截至提交后核查，1364为`PENDING (Resources)`，`AllocTRES`为空，即申请已接受但尚未分到GPU。调度器给出的预计开始时间可能变化，实际起止时间以`scontrol show job 1364`为准。

## 在这两张卡上运行任务

等1364进入`RUNNING`后，通过同一入口指定分配号：

```bash
# 运行短验收；在现有分配内创建srun步骤，不另申请两张卡。
bash scripts/hpc/submit.sh --allocation 1364 validate tests/test_coupled_40hz_integration.py

# 需要训练时使用当前双卡配置。
bash scripts/hpc/submit.sh --allocation 1364 train
```

`--allocation`必须放在模式名之前，允许与`--source-archive`一起使用。入口核对作业属于当前用户、状态为RUNNING且名称为`TsenseI-reserve2`，拒绝与`reserve`或`--dependency`混用。

任务沿用`job.sbatch`中的GPU环境、归档、探针、训练/回放入口。`srun --overlap`让任务和不进行计算的保留进程共用已分配资源；文件锁限制通过此入口同一时间只运行一个子任务。每次任务有独立步骤编号，stdout/stderr写入`logs/hpc/1364/steps/<step>.out/.err`，源码及遥测写入`logs/hpc/1364/steps/<step>/`，避免连续运行覆盖。

这些命令前台等待任务结束。GPU分配总到期时间不会因为新任务开始而延长。单独执行原`submit.sh train`仍会申请另一份资源；使用已保留的两张卡时需要带`--allocation 1364`。

## 查看与提前释放

```bash
squeue -j 1364
scontrol show job 1364

# 用户决定提前释放时执行，也会终止该分配内尚在运行的任务。
scancel 1364
```

将来另需一份相同保留申请，可运行`bash scripts/hpc/submit.sh reserve`，会生成新的作业号；当前1364已提交，不需重复运行。

## 验证

- `bash -n`：提交、保留及任务三个脚本通过。
- CPU调度桩测试5项、7子测试通过：48小时显式参数、复用时只调用srun、参数含空格及步骤日志、拒绝他人/等待/非保留作业、拒绝冲突参数、原dependency/source-archive提交保持。
- 真实Slurm提交接受，核验`ReqTRES=cpu=16,mem=128G,gres/gpu=2`及48小时上限。
- 等待资源期间没有运行真实GPU子任务，分配内GPU执行验收仍待实际获得资源后进行。

参考文件：`scripts/hpc/submit.sh`、`job.sbatch`、项目AGENTS、Slurm当前分区和作业信息、`srun --help`；新增`reserve.sbatch`与`tests/test_hpc_allocation_entry.py`。
