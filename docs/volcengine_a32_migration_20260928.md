# A3.2 迁移至火山云两张 A100

## 当前状态

2026-09-28：本地准备迁移包、独立环境安装、Slurm 作业和双卡采样／学习入口。
**当前部署目标为 `115.191.75.48:22`（TenseiNode1，A100）**。已成功登录并按用户指定完成首次改密，A3.2 包已上传并校验，两套环境、私有CUDA编译器及EGL已安装；完整双卡验收通过，正式Slurm作业935已于北京时间19:11启动。账号为 `houj`，Slurm account 为 `research`；密码不写入仓库、脚本、迁移包或运行日志。

用户先指定 `101.47.9.106:22`，实际为 Tsensei2／H100 且8卡全部占用；用户随后明确要求排队时改用 A100，因此停止 H100 环境安装，转至当前 A100。H100 未提交 GPU 作业。A100 的 Slurm 作业931实测获得两张 A100-SXM4-80GB，驱动550.144.03、互联NV12。

资源依据是 [Tensei 集群概述](https://agimani.github.io/hpc_doc/)，公开配置为 8×A100-SXM4-80GB、NVSwitch、Ubuntu 22.04、驱动 550.144.03。实际分配、驱动、互联和吞吐须在用户账号内验证。[集群规则](https://agimani.github.io/hpc_doc/user/basis/)要求 GPU 程序经 Slurm 执行。

## 迁移范围

只迁移当前 `tsensei_v3_2_pianist` 内的 A3.2 源码、左右手模型、钢琴、乐谱、配置和必要工具。项目目录／Python 包名继续保留，避免破坏导入和检查点路径。打包排除旧手模型备份、其他手项目、M6 模型、既有训练产物、视频、虚拟环境和密钥；原工作区文件保留。

- 清单和打包方法：[A3.2 独立包](a32_migration_bundle_20260928.md)。
- 依赖安装与 GPU 探针：[环境说明](hpc_environment_20260928.md)。
- 并行实现和采样时序：[双卡说明](dual_gpu_pipeline_20260928.md)。

## 两卡分工与优化边界

`scripts/hpc/job.sbatch` 申请 **2 GPU、16 CPU、128 GiB 主机内存、24 小时**，仅使用 Slurm 分配的逻辑 `cuda:0/1`，不重写 `CUDA_VISIBLE_DEVICES`。

- GPU 0：512 个物理环境、观测／奖励和采样 Actor。
- GPU 1：百万条 replay、SAC Actor／Critic／温度更新。
- 同一个学习器训练同一个策略，队列最多一个向量步。采样与学习重叠，但快照有最多一批策略延迟；不是两份独立训练，也不保证逐步数值等价。
- 保持当前 A3.2 入口的 1231 观测／47 动作、20/200 Hz、batch1024、每 transition .25 次更新、三个 LR3e-4、5,000,192 条总样本、完整乐谱预填及全部奖励／控制参数。
- 沿用当前代码的 Direct MCP 上限 1.3rad、.15 秒提前定位、正确键子步奖励、曲末 .25 秒松键等最新修复；不从旧历史记录恢复参数。

两张卡不能把显存自动合并。若 SAC 比采样耗时更长，采样卡会等待；仅看利用率不足以证明加速。先测基础 CUDA／Warp／P2P，再测同样本、同更新次数的单卡／双卡短训，最终按端到端时间、稳定吞吐、F1 与碰撞评估。作业934两轮平均稳态吞吐：单卡1361.94、双卡1745.29条/秒，提升28.15%。首轮每批采样89.22ms已与291.03ms学习阶段重叠，快照和commit/reset合计约0.76%；当前限制主要在学习器，不通过无效占用显存追求利用率数字。最终两轮结果见下方验收记录。

## 远端执行顺序

上传和解压后，进入独立项目根目录（建议 `~/projects/tsensei_v3_2_pianist`）：

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/hpc/setup_envs.sh
bash scripts/hpc/setup_cuda.sh
bash scripts/hpc/setup_egl.sh
bash scripts/hpc/submit.sh probe
squeue -u houj
```

GPU 基础探针输出在 `logs/hpc/JOBID/gpu_probe.json`。成功后提交等预算短训：

```bash
bash scripts/hpc/submit.sh benchmark
```

默认交替顺序各测两次单卡／双卡，每次 262144 样本；保留原完整曲目预填、batch、更新比例和初末评估。比较 `logs/hpc/JOBID/benchmark/results.json` 与 `gpu_utilization.csv`。含编译的总时间和训练日志内稳定吞吐分别报告，不能把短训 F1 当长期收敛。

基准与完整训练路径均通过后：

```bash
bash scripts/hpc/submit.sh train
```

输出到 `training_mjlab/outputs/hpc/job-JOBID-时间戳`。每个作业先保存源码包，再在解压快照中执行；安装的两套环境通过符号链接复用。每次作业自己的源码、命令、探针和 GPU 用量均留在 `logs/hpc/JOBID/`，修改工作目录不会热改作业源码。

HPC 包装默认 `--no-swanlab`，本地指标／配置／检查点正常保存；不迁移本机 SwanLab 密钥。日后可在远端独立配置上传环境和凭据后显式开启。普通 `run.sh` 默认未改。

### 依赖兼容

实际安装 Torch2.10.0+cu128、Warp1.17、MuJoCo Warp3.11 与 MJLab固定提交；Composer与learner的依赖检查均通过。Slurm932已在原生550.144.03驱动通过两卡矩阵乘、CUDA graph、Warp实际内核及Torch互操作，P2P双向约199GB/s（含同步的复制短测，不代表训练吞吐）。未启用兼容驱动。随后作业934通过真实Adam编译/执行、MJLab物理、单卡及双卡完整短训和严格检查点重建。

系统缺nvcc与EGL dispatcher，使用setup_cuda.sh/setup_egl.sh仅安装到项目的training_mjlab/hpc，Slurm脚本自动加载。环境安装记录位于training_mjlab/hpc/environment/20260928T104745Z.yiYpTQ。MJLab下载过慢时使用经过SHA验证、258个运行文件与固定源提交逐字节一致的wheel；见环境说明。安装未更改系统驱动、CUDA或Python。

## Codex 连接这个服务器

在**运行 Codex 桌面端的电脑**的 `~/.ssh/config` 加入：

```sshconfig
Host volc-a32
    HostName 115.191.75.48
    User houj
    Port 22
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

先在该电脑终端验证 `ssh volc-a32`。如配置了 SSH 密钥，可补 `IdentityFile`；不要在 SSH 配置里存密码。集群若要求首次改密，按交互提示完成。

远端已安装 **Codex CLI0.158.0**，登录shell能找到`~/.local/bin/codex`。20:14按用户明确要求，经SSH单独传输node4已有ChatGPT登录缓存到远端`~/.codex/auth.json`（0600），`codex login status`显示`Logged in using ChatGPT`，真实模型请求返回`CODEX_A32_PROXY_OK`。凭证未进入仓库、迁移包或日志。

已打开的SSH终端需要加载新代理配置；重新连接的登录shell会自动加载：

```bash
source ~/.profile
codex login status
codex
```

### 代理连接与维护

远端最初没有代理，设备认证接口直连返回403/`unsupported_country_region_territory`。按用户要求复用node4已有代理，当前链路为：A100的`127.0.0.1:17999` → SSH加密转发 → node4的`127.0.0.1:17999` → node4原有代理出口。只监听远端回环地址。

node4用户服务`~/.config/systemd/user/volc-a32-proxy.service`已enabled/active；SSH每30秒保活，连续3次失败后退出，systemd间隔10秒重连。专用SSH公钥仅允许监听该代理端口，不允许执行远端命令；私钥留在node4。远端`~/.profile`追加有边界标记的HTTP/HTTPS/ALL_PROXY大小写变量，原文件已备份。

在node4查看或重启：

```bash
systemctl --user status volc-a32-proxy.service
systemctl --user restart volc-a32-proxy.service
```

A100依赖node4及其上游代理在线。node4当前Linger=no，服务随该用户的systemd用户管理器运行；未承诺无人登录或整机重启时仍保持连接。此设置只作用于后续shell，已经运行的Slurm935没有重启或修改环境。

登录缓存迁移依据[OpenAI无头环境认证说明](https://learn.chatgpt.com/docs/auth)，由用户本次明确授权；初次源码迁移包仍不包含认证文件。

然后桌面端 **Settings → Connections → SSH** 添加／启用 `volc-a32`，选择远端项目目录 `/home/houj/projects/tsensei_v3_2_pianist`。具体界面按安装版本显示。该流程依据 [OpenAI SSH 连接说明](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host) 和 [Codex CLI 安装说明](https://learn.chatgpt.com/docs/codex/cli)。本轮已使用用户授权的现有凭证登录，认证文件与项目源码分开保存。

如果桌面端无法交互输入SSH密码，可先在运行桌面端的电脑配置已有SSH公钥，或生成专用密钥后使用`ssh-copy-id -i ~/.ssh/id_ed25519_volc_a32.pub houj@115.191.75.48`安装公钥；再在上述Host块加入`IdentityFile ~/.ssh/id_ed25519_volc_a32`。私钥留在该电脑，不能把node4的SSH配置误当成桌面电脑配置。

### Windows SSH 公钥修复入口

用户Windows终端为CMD，直接执行PowerShell语法失败；`powershell`不在PATH，复制文本还出现`id\_ed25519...`错误路径。提供`scripts/hpc/fix_windows_ssh.cmd`，远端分发副本为`/home/houj/fix-a100.cmd`。在Windows CMD运行：

```cmd
scp -o User=houj -o PubkeyAuthentication=no 115.191.75.48:fix-a100.cmd "%TEMP%\fix-a100.cmd" && "%TEMP%\fix-a100.cmd"
```

按提示输入服务器密码。脚本用Windows PowerShell完整系统路径，不依赖其PATH；复用或创建`%USERPROFILE%\.ssh\a100key`，只上传公钥。原SSH配置先备份，保留其他主机；修复`volc-a32`/`volc-a32-key`的错误IdentityFile，使用临时配置实际验证BatchMode公钥登录输出`houj`后才保存。私钥留在Windows。成功后Codex主机别名选`volc-a32`。

服务器公钥安装部分已在隔离目录验证CRLF规范化、重复运行不重复追加、保留旧条目；当前任务在Linux node4，未声称已经写入Windows文件。**V32-HPC-WINSSH-01**：等待用户在Windows执行，完成客户端端到端验收。

连接后 Codex 可编辑远端源码并用 `sbatch` 提交作业。训练仍须通过 Slurm；普通 Codex／SSH 终端的 `nvidia-smi` 报设备访问错误属于集群隔离行为。

## 本轮基准与正式训练

基准Slurm作业为`934`，包括真实双GPU测试11项，以及交替顺序的两轮单卡／双卡短训和每轮独立检查点重建。正式作业`935`带`afterok:934`依赖，基准COMPLETED/0后已自动从头启动。它不加载任何本机权重；完整config核验512环境×9766向量步，即5,000,192样本，原质量目标达成后仍允许提前停止。

| 模式 | 第1轮稳定样本/秒 | 第2轮稳定样本/秒 | 平均 |
| --- | ---: | ---: | ---: |
| 单卡 | 1363.17 | 1360.72 | 1361.94 |
| 双卡流水 | 1744.28 | 1746.30 | 1745.29 |

每次262144条样本、42112次更新、93696条预填；四次计数/1231观测/47动作/最终排空校验及独立`eval.sh final.pt`重建加载全部通过。稳定窗口为向量步190→510（163840样本/40960更新），排除初次编译、预填、末评估。总耗时受缓存影响，不能把含冷编译的1.84倍当长期加速；采用稳态1.2815倍。短训不证明长期F1或碰撞达标。

远端证据：`logs/hpc/934/benchmark/results.json`、`collector_tests.log`和`gpu_utilization.csv`。本地只复制验收JSON/日志到`training_mjlab/hpc/reports/a100_20260928/`，该目录不进入迁移包。

正式输出目录：
`/home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/job-935-20260928-191156-583628`

作业源码包`logs/hpc/935/source.tar.gz` SHA256：`c14bb52f0131e1b0a3b832bb1b414e12710d6d0293b20f7bcc9492e296dccb5f`。模型/动作/准备/奖励合同与当前入口保持，显式双卡执行布局写入config；源码快照不会随之后工作目录编辑而变化。19:14核验到向量步300/153600样本/14976次更新（仍有1批待更新），所有损失有限，两卡同时工作、学习卡99%利用率。正式训练尚未完成。

## 日常查看

```bash
cd /home/houj/projects/tsensei_v3_2_pianist
squeue -u houj
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
tail -n 30 logs/hpc/JOBID.out
tail -n 30 logs/hpc/JOBID.err
```

训练中的细粒度指标在对应输出目录的`metrics.jsonl`，参数在`config.json`，结束后生成`summary.json`。`final.pt`为结束模型；`best.pt`按现有音乐F1和物理诊断资格保存，不放宽质量门槛。

## 验收与 TODO

- 本地功能与打包验证结果统一记录在 [修改日志](change_log.md)；本机只有一张正在训练的 GPU，不以本地测试代替双 A100 验收。
- **V32-HPC-01**：登录、上传、两环境、用户CUDA/EGL安装及Slurm两卡基础探针已完成。
- **V32-HPC-02 已完成**：真实双卡测试11通过，四次短训/预算/排空/检查点重建通过，平均稳态加速28.15%；935已正式开训。
- **V32-HPC-03**：长训 F1／碰撞与达到质量目标的时间比较；已验证短测执行加速；长期收敛、F1.99和碰撞目标仍待正式训练结果。
