# A3.2 独立迁移包（2026-09-28）

迁移入口是本独立项目 `run_tsensei_a32.sh`，模型是包内当前 **TsenseI_A3.2** 左手及派生镜像右手。无需上层仓库的 `robopianist/`、V2/V3、`TsenseI_A3.2/` 或其他手模型，也不迁移现有训练结果。厂家右手仍待提供，见 TODO V32-10。

## 打包与核验

在本独立项目根目录运行：

```bash
python3 scripts/hpc/package_a32.py --output /tmp/tsensei_a32_source.tar.gz
python3 scripts/hpc/package_a32.py --audit /tmp/tsensei_a32_source.tar.gz
cd /tmp
sha256sum -c tsensei_a32_source.tar.gz.sha256
```

输出包含 `.tar.gz`、同名 `.tar.gz.manifest.json` 和 `.tar.gz.sha256`。包内 `BUNDLE_MANIFEST.json` 列出每个源文件的相对路径、大小、模式和 SHA256；外部清单另含压缩包 SHA256。归档不带额外顶层目录，解压到一个新项目目录即可。已有同名输出会被拒绝。

文件顺序、gzip 时间、tar 时间、所有者及权限固定；相同源文件可得到相同压缩包字节。打包后重新核对选中文件，遇到并发修改会删除本次未完成压缩包并报错。脚本只读取原项目，不移动或删除原文件。

## 精确包含范围

完整逐文件清单以本次产物 `BUNDLE_MANIFEST.json` 为准。选择规则在 `scripts/hpc/package_a32.py` 中，不依赖 `.gitignore` 或 Git 跟踪状态。

| 路径 | 包含内容 |
| --- | --- |
| 项目根 | `AGENTS.md`、`README.md`、两份来源许可证、`pyproject.toml`、两个 tested requirements、`run.sh`、`run_tsensei_a32.sh`、`eval.sh`、两个基础预览/验证入口 |
| `tsensei_v3_2_pianist/` | Python 源码、当前 A3.2 原始/派生 URDF、左右 STL/OBJ、接触/驱动/碰撞 JSON、包内曲谱数据及来源说明 |
| `configs/` | 当前 JSON 配置及说明；不含 `archive/` |
| `scripts/` | `.py/.sh/.json/.toml/.txt/.md/.sbatch`；自动纳入后续 `scripts/hpc/` 部署脚本 |
| `preview/scripts/` | Python/shell 回放和渲染入口 |
| `tests/` | Python 测试、JSON/NPZ 数值测试 fixture；不含两个 M6 专用测试 |
| `docs/` | Markdown、JSON、CSV 维护记录/合同/模型来源与验收说明；排除参考资料目录和会话导出 |
| `training_mjlab/` | `setup.sh`、tested requirements、监控 requirements、`inputs/` 下曲谱/指法输入 |

Shadow 名称的奖励/驱动 profile 和 SAC 数值测试 fixture 是当前 A3.2 使用的算法/合同，保留这些源码不表示迁移另一只手的实体模型。

## 排除范围

- 所有 `.venv`、缓存、构建目录、egg-info、Git 目录及安装依赖。
- `outputs/`、`runs/`、`source_versions/`、`model_backups/`、`archive/`；不含 checkpoint、训练日志、视频或旧冻结 scene/task。
- 包内历史 `TsenseI_V3.2_hand_withid_fixed.zip` 及其他压缩归档。
- `assets/tj_m6/` 和 `tests/test_preview_arm_ik.py`、`tests/test_m6_gpu.py`。当前源导出 `source_runtime → export_scene → suite.load` 构建无臂场景；`run.sh` 显式 `--no-arms`，不需要 M6 资产。通用 Python 模块仍保留，使用臂预览前须另补资产。
- `docs/references/`、`external_never_copy_only_reference/`、`docs/chat_answers_20260924.md`。文档中的历史链接可能指向未迁移的实验资料。
- 音频系统运行库及音色：训练不需要；有声视频回放时另行安装。
- 家目录及 `.ssh/.aws/.config`，以及 `.env*`、私钥/证书、credentials/secrets/API-key 命名的文件。符号链接不跟随。账号和密码不写入包或部署命令。

## 两个 Python 环境

源码编译和 GPU 训练使用不同 MuJoCo 版本，现有入口要求保留两个路径：

| 环境 | 本机已验证关键依赖 | 作用 |
| --- | --- | --- |
| `training/.venv` | Python 3.12、MuJoCo 3.13.0、dm-control 1.0.46、dm-env-wrappers 0.0.13、NumPy 2.5.3、note-seq 0.0.5 | Composer 构建模型、乐谱及临时 scene/task |
| `training_mjlab/.venv` | Python 3.12、Torch 2.10.0+cu128、MuJoCo/MuJoCo Warp 3.11.0、Warp 1.17.0、MJLab 1.6.0 固定源码 commit | GPU 物理、SAC、replay、评估 |

纯训练部署可采用以下安装方式；应在目标集群的计算环境核验驱动、软件源和模块，再由部署脚本执行：

```bash
python3.12 -m venv training/.venv
training/.venv/bin/python -m pip install -r requirements-tested.txt
training/.venv/bin/python -m pip install 'dm-env-wrappers==0.0.13' -e '.[test]'
bash training_mjlab/setup.sh
training/.venv/bin/python -m pip check
training_mjlab/.venv/bin/python -m pip check
MUJOCO_GL=egl bash run_tsensei_a32.sh --no-swanlab --build-only
```

`requirements-sac-tested.txt` 是包含旧 JAX 基线的较完整快照，也可用于 Composer 环境；当前 PyTorch 主线不要求安装 JAX/Flax/Optax。`training_mjlab/setup.sh` 单独运行不会创建 Composer 环境。

系统需 Linux x86_64、Python 3.12/venv、支持 CUDA 12.8 软件栈的 NVIDIA 驱动、C++ 编译器及 CUDA Toolkit `nvcc`。当前 `--optimizer-layout` 会编译自定义扩展并使用 Torch 的 Adam 数学实现，还需要 ninja；仅有 PyTorch wheel 自带 CUDA runtime 不足以编译。设置正确的 `CUDA_HOME`/`PATH` 后再短测。EGL 用于无显示器渲染；只有训练数值验证时不需窗口。ffmpeg、FluidSynth/音色仅在有声视频导出时需要。

`run.sh` 默认启用 SwanLab。迁移验证可显式 `--no-swanlab`；如需云图表，运行 `scripts/setup_swanlab.sh` 创建独立监控环境，并单独配置自己的认证文件。监控密钥不进入本包。

## 新训练的来源合同与验证边界

包内 `source_runtime.source_files()` 对实际包文件及 `configs/actions/*.json` 做散列。排除历史 ZIP/M6 后，新训练会保存裁剪后源码的完整指纹；不改写旧训练的 config 或降低旧检查点校验。远端训练应保留本次源压缩包及校验清单，之后恢复时使用匹配源码和依赖。

普通 `run.sh` 会写完整 `config.json`，但不自动保存源码归档。本迁移包应在训练前生成并留档，避免以后代码更新后无法重建模型。解包后再改代码时需重新打包。

打包测试验收确定性、排除规则、必需文件、符号链接拒绝及并发修改检测；压缩包审计逐项核对 SHA256。它们不证明目标卡性能、双卡吞吐或训练收敛。远端还需执行源码构建、两卡各自短训/评估和实际吞吐/显存测量；不能用归档成功替代 GPU 验收。

### 本机实际验收

- `training/.venv/bin/python -m pytest tests/test_a32_migration_bundle.py -q`：5 项通过，包含篡改内容及路径逃逸拒绝。
- 首次真实源码审计快照包含 469 文件、37,073,564 字节，压缩后 12,731,445 字节（约 12.14 MiB）；后续部署脚本/文档更新会使数量和散列改变，以正式产物清单为准。
- 包内逐文件审计及外部 `sha256sum -c` 均通过。同一解包源码连续重打包两次，完整 `.tar.gz` 字节相同。
- 解包到项目外临时目录，唯一外部运行连接为两个依赖 venv 链接。执行 `bash run_tsensei_a32.sh --no-swanlab --build-only` 返回 0，成功重建当前 A3.2 场景及乐谱；确认不需要旧手 ZIP 或 M6 资产。实际输出仍为 20/200 Hz、shadow_v1、strict_v1、0.25 秒释放尾段。
- 原始审计目录为 `/tmp/tsensei-a32-bundle-audit-xfrps8oz`，CPU 日志为该目录 `build-only.log`；它是本机临时验证资料，不随迁移包传输。本项未占用 GPU、未启动长期训练。

TODO V32-HPC-BUNDLE-01：目标计算节点的依赖安装、双卡 smoke、吞吐与长期 F1/碰撞结果由本轮部署报告补充。已有 V32-10、驱动/接触物理标定等模型限制保留。
