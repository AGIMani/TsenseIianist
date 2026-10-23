# 本地开发与远程训练交接（2026-10-11 历史记录）

> 历史交接包说明：本文保留2026-10-11交接时记录的包内容、归档路径及SHA，不是现行训练配置。下文“当时”均指该交接记录；旧归档不用于新训练。当前入口及归档以 [all_use_this_train.sh](../all_use_this_train.sh) 为准，变更与验证见 [2026-10-15训练一致性修复](training_consistency_20261015.md)，当前奖励仅以 [唯一奖励总表](reward_defaults_20261006.md) 为准。

## 在本地 PC 解包

本包是可独立开发的 A3.2 项目源包。将压缩包下载到本地后执行：

```bash
mkdir -p /home/hj/pianist
tar -xzf tsensei_v3_2_pianist_local_dev_20261011.tar.gz -C /home/hj/pianist
cd /home/hj/pianist
```

归档根目录就是项目根目录，不需要再移动一层。解包后可用随包的 `.sha256` 和 `.manifest.json` 复核归档。

## 本地开发

- 默认入口仍是根目录 `all_use_this_train.sh`；本地主要进行 Python、配置、预览和 CPU 测试修改。
- 预览脚本使用 `configs/preview/current_arm_training_20261011.json`，旧候选左臂镜像安装配置和导出脚本已移入 `deprecated/`，当时预览只读取活动合同。
- `preview/outputs/` 下的导出图和 HTML 属于交付产物，不放入源包；本次候选产物另附 `left_mirror_mount_candidate_bundle.zip`。
- 修改算法源码后不能直接把工作区当作已生效训练源码。训练前必须生成新的不可变源码归档、记录 SHA256，并在入口中显式传入匹配校验值。

## 远程训练

把本地修改同步到远程项目目录后，在远程项目根目录使用 Slurm 入口：

```bash
bash all_use_this_train.sh
```

该入口会通过 `scripts/hpc/submit.sh` 申请双卡，`cuda:0` 采样、`cuda:1` 学习。当时默认仍为无臂训练；启用机械臂时，必须显式提供左右两侧各 7 个初始角：

```bash
bash all_use_this_train.sh --arms \
  --arm-initial-right-deg J1 J2 J3 J4 J5 J6 J7 \
  --arm-initial-left-deg  J1 J2 J3 J4 J5 J6 J7
```

远程训练入口当时固定校验：

```text
logs/arm_ccs696_coupling_20261011/source_arm_ccs696_coupling_v5.tar.gz
SHA256 5913682940f0e6a4a1dfacf3b9181f9910a625596dbd008e9ad49cb1070f82ed
```

如果本地修改了会影响训练的源码，先在远程生成并审计新的源码归档，再更新入口的归档路径和 SHA；仅同步工作区文件不会改变当时训练归档。GPU 训练、推理和渲染都应在 Slurm 分配内执行，不要在本地 PC 直接申请远程 GPU。

## 包含与排除

包内包含当时 A3.2 源码、训练入口、配置、测试、维护文档、预览脚本以及可选 M6/四孔连接件资产。为便于本地开发和传输，包内不包含 Python 虚拟环境、训练输出、checkpoint、运行缓存、历史源码归档或凭据。

当时配置已采用左臂 J1–J7 镜像角度与掌面朝下安装；该交接记录中固定训练归档尚未重打包，四孔紧固、法兰时钟和实机编码器零位需要硬件复核后再生成新的训练归档。
