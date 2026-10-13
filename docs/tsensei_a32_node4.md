# TsenseI_A3.2 在 node4 的位置与环境

## 当前工作区已经位于 node4

2026-09-28 实测：

- `hostname` 返回 `FC500T-K`；这是机器的本地主机名。
- `hostname -I` 包含 `100.86.254.51`。
- `getent hosts node4` 返回 `100.86.254.51 node4.tail9f5c0d.ts.net`。

因此，当前 `/home/houjue/pianist_v2` 工作区与用户指定的 `node4` 是同一台机器。此前 `ssh houjue@node4` 认证失败只说明本机到自身的 SSH 登录未配置可用密钥，不代表工作区未到远程。用户从 Windows 上传后的目录已出现在此文件系统中，无需再次 SCP 或复制 Python 环境。

## 上传文件与训练模型

| 用途 | 位置 |
| --- | --- |
| 用户上传目录 | `/home/houjue/TsenseI_A3.2__hand_withid` |
| 当前项目原始模型目录 | `/home/houjue/pianist_v2/TsenseI_A3.2` |
| 独立训练项目 | `/home/houjue/pianist_v2/tsensei_v3_2_pianist` |
| 训练实际读取的派生模型 | 独立项目内 `tsensei_v3_2_pianist/assets/` |

附件 ZIP SHA256 为 `a86ee434844f4c37f84a61531cc78a4faeea23abd9920566996c915b469aeb53`。

按去掉 ZIP 顶层目录后的相对路径逐文件核验，共 30 个文件：

- 项目 `TsenseI_A3.2/` 的 30 个原始文件与附件逐字节相同。
- 用户上传目录的 29 个文件与附件逐字节相同；唯一不同文件是 `urdf/A3.1V6 26.9.27(6).urdf`。
- 该 URDF 的差异仅为 44 处 mesh 文件名从 `package://A3.1V6 26.9.27(6)/meshes/` 改成 `../meshes/`，其余字节相同；关节、惯量、坐标与网格内容没有因此变化。

训练读取包内派生模型，保持独立项目不依赖外部上传目录。模型几何、关节映射和派生结果的验收见本轮模型说明与修改日志。

## 已有训练环境

实测 GPU 为 **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**，总显存 97,887 MiB；不是其他规划文档中的多卡 A100 集群。`torch.cuda.is_available()` 为 `True`。

现有两套环境可用于当前源码入口，无需重建：

| 环境 | 已核实版本 |
| --- | --- |
| `training/.venv` | MuJoCo 3.13.0、dm-control 1.0.46、dm-env-wrappers 0.0.13 |
| `training_mjlab/.venv` | Torch 2.10.0+cu128、MuJoCo/MuJoCo-warp 3.11.0、MJLab 1.6.0、ninja 1.13.0 |

本轮训练使用新模型、提前定位 0.15 秒、子步误触奖励，三个学习率均为 `3e-4`，其余沿用 `run.sh`。模型替换和提前定位引入的新观测合同需要匹配的新训练；启动位置、PID、输出目录和验收结果由本轮主日志记录，环境可用不等于训练已完成或 F1 达标。
