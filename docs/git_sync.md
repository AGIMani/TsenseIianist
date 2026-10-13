# Git 源码同步与服务器环境

远端仓库：<https://github.com/ONKIALL26/tsensei_v3_2_pianist>。

本仓库保存可独立安装的 A3.2 训练源码、当前双手模型、乐谱、配置、测试和文档。首次提交以 `scripts/hpc/package_a32.py` 的轻量训练清单为依据，另加入 Git 配置和本说明。不提交 Python 环境、CUDA 工具、缓存、检查点、日志、视频、密码或私钥。

本机已有的 M6/安装件预览资产、完整在线控制应用及独立克隆的硬件 SDK 不在这次训练源码上传范围；相关历史说明保留，不能将其中的本地交付路径视为 clone 后可用。`online_control` 仅保留离线导出需要的四个模块。带机械臂的预览和导出需要另行提供相应资产；A3.2 无臂训练仍以包内模型为来源。

## 日常开发与同步

在开发工作目录修改后，按文件显式暂存并检查：

```bash
git status --short
git add path/to/changed_file
git diff --cached
git commit -m "说明本次修改"
git push origin main
```

每次修改仍须更新 `docs/change_log.md` 及相关说明。`.gitignore` 不是内容扫描器；上传新文档和配置前仍须确认不含凭据。训练服务器尽量只拉取已提交源码，机器专属配置使用被忽略的 `*.local.env` / `*.local.json`，后续入口需显式读取才会生效。

服务器首次克隆（已有旧目录时使用新目录，不覆盖）：

```bash
git clone git@github.com:ONKIALL26/tsensei_v3_2_pianist.git tsensei_v3_2_pianist_git
cd tsensei_v3_2_pianist_git
```

后续同步：

```bash
git status --short
git pull --ff-only
git rev-parse HEAD
```

如存在本地修改或分支分叉，先检查并处理，不用 `reset --hard` 自动覆盖。服务器认证建议各自使用独立的只读 Deploy key；开发端需要写入权限。公钥在 GitHub 仓库 Settings → Deploy keys 配置，私钥只留在对应机器。仓库地址不代表它必须公开。

## 环境安装和 GPU 运行

Git 只同步源码。首次克隆仍需按 `docs/hpc_environment_20260928.md` 安装独立环境；不要复制另一台机器的 `.venv` 或修改系统驱动。现有入口为：

```bash
bash scripts/hpc/setup_envs.sh
bash scripts/hpc/setup_cuda.sh
bash scripts/hpc/setup_egl.sh
bash scripts/hpc/submit.sh probe
```

等待探针作业成功并检查结果后，再通过 `submit.sh` 验收和训练。所有 CUDA/GPU 工作都在 Slurm 分配内执行。依赖未改变时复用本机环境；依赖文件改变后需同步安装并重新验收。2026-10-01后续已完成H100独立环境、GPU验收、512环境短训及检查点重建；正式训练按用户要求停止并保留两卡，待奖励确认后再开实验。当前状态见 [H100部署记录](h100_training_20261001.md)。

目前 `scripts/hpc/job.sbatch` 使用通用 `gpu` 分区、两卡、16 CPU 和 128 GiB 内存；并未实现自动选择 A100/H100/local 配置，Git 拉取也不会自动调整学习参数。普通 `run.sh` 与 HPC 双卡入口的实验默认不同，切换机器时应保留原实验入口和完整显式参数。

## 固定排队作业的源码

现有作业会从自己的源码快照运行。为防止排队期间拉取新源码改变待运行版本，在提交前先打包，再使用已支持的 `--source-archive`：

```bash
mkdir -p logs/submissions
revision=$(git rev-parse HEAD)
# 提交前确认 git status --short 为空；同一提交不要并发覆盖同名归档。
python3 scripts/hpc/package_a32.py --output "logs/submissions/$revision.tar.gz"
bash scripts/hpc/submit.sh --source-archive "logs/submissions/$revision.tar.gz" train
```

提交入口记录并验证归档 SHA-256，作业从归档解压目录执行，之后工作目录的修改不热切换训练源码。这个打包器读取工作目录，所以上述“工作区干净”的检查不可省略。归档固定训练负载源码；Slurm 提交/启动脚本仍位于工作目录，排队期间不要修改这些脚本。保留完整 `config.json`、环境记录和检查点合同，迁移续训时另行传输检查点并使用项目支持的恢复方式。

## 首次上传状态（2026-10-01）

本次只接入 Git 同步，不修改训练数值、不更新远端 H100 环境，也不启动新训练。推送状态以 `docs/change_log.md` 及实际远端分支为准。
