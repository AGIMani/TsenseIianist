# 火山云 A3.2 环境安装与双卡探针（2026-09-28）

此页维护用户目录安装与探针流程；当前环境分工、依赖核对、固定源码和机械臂实验状态集中在[训练环境总览](training_environment.md)。2026-10-10核对learner原133包与安装记录一致、Composer原84包保持且追加6包分析依赖；当前没有整体重装的依据。两个环境分别用于模型构建和GPU学习，不随意合并。

## 范围与当前状态

本次为 A3.2 独立迁移包增加用户目录环境安装和 Slurm 内的 GPU 验收入口。现有两个虚拟环境及正在运行的本地训练不改动。未迁移训练产物、账号凭据或已编译 CUDA 缓存。

目标平台资料来自用户提供的 [HPC 使用文档](https://agimani.github.io/hpc_doc/)：Ubuntu 22.04、A100 80 GB、NVSwitch，GPU 作业通过 Slurm 分配。文档给出的驱动 550.144.03 / CUDA 12.4 与本项目 Torch CUDA 12.8 环境存在需要实测的版本差异；没有据此改动依赖版本或系统驱动。

后续连接已由主任务确认到 A100 节点 `TenseiNode1`，实际驱动为 `550.144.03`，没有可用的系统 `nvcc` 或 `/usr/local/cuda`。完整远端迁移与验收状态以同任务的部署记录为准；本说明不把环境脚本测试视为远端 GPU 验收。

## 安装

在迁移后的独立项目目录运行：

```bash
bash scripts/hpc/setup_envs.sh --print-plan
bash scripts/hpc/setup_envs.sh
```

安装脚本只在当前项目用户可写目录工作：

| 位置 | 用途 |
| --- | --- |
| `training/.venv` | Composer / CPU 模型构建，项目 `requirements-tested.txt` 加本机实际使用的 `dm-env-wrappers==0.0.13` |
| `training_mjlab/.venv` | `training_mjlab/requirements-tested.txt` 指定的 Torch / MJLab 环境 |
| `training_mjlab/hpc/tools` | 从官方安装器取得的 uv 0.12.15 |
| `training_mjlab/hpc/python` | uv 管理的 Python 3.12 |
| `training_mjlab/hpc/uv-cache` | 下载缓存，安装到环境时使用复制方式 |
| `training_mjlab/hpc/environment/<UTC 时间>.<随机后缀>/` | 每次成功安装后的两个实际 freeze 文件和 `installed.json` |

环境带 `.tsensei-a32-hpc-environment` 所有权标记；脚本只允许重复修复自己建立的环境，遇到已存在但无标记的 `.venv` 或符号链接立即退出，不覆盖旧环境。安装中断后可以重试同一脚本。并发安装由 `flock` 拒绝。Python 包版本冲突、下载失败会保留错误并退出，不静默改用另一版 Torch / MuJoCo / Warp。

本入口不安装 SwanLab；HPC 作业应使用 `--no-swanlab`。没有运行 `apt`、没有修改系统 Python、CUDA 或 NVIDIA 驱动，也不需要迁移本地 `.venv`。安装步骤不导入 GPU 库，可以在允许联网和安装用户软件的登录节点执行。

实际 freeze 文件记录远端真正装入的依赖；`installed.json` 同时记录 Python 版本、两份输入 requirements 的 SHA-256 和安装器版本。它们是安装记录，不是 GPU 兼容性证明。

### 下载缓慢时的固定版本离线输入

A100安装使用集群建议的TUNA镜像，版本不变。原MJLab codeload URL缺归档扩展名，被uv0.12.15拒绝，requirements已改为同一个提交的GitHub `.tar.gz` 地址。下载仍慢时可设置 `TSENSEI_MJLAB_WHEEL=/absolute/path/mjlab-1.6.0-py3-none-any.whl`；脚本只接受SHA256 `ccb22b92162e67991f104c6499626f681b43afb91b1cb98b2bae7ce73f420f32`。该wheel的258个运行文件与固定提交`8ee51fbcf806a7419189f706d9e394cbeb7790fa`源归档逐字节一致，不是另一版MJLab。实际离线requirements保存在`training_mjlab/hpc/learner-offline-requirements.txt`。

## 用户目录 CUDA 12.8 编译器

现有 `rl_torch/adam_execution.py::extension()` 使用 Torch `load_inline` 编译 CUDA Adam 布局，需实际的 `CUDA_HOME/bin/nvcc`、CUDA/CCCL 头文件和可链接的 cudart。Torch 的 `CUDAContextLight.h` 还包含 cuBLAS、cuSPARSE、cuSOLVER 头文件，因此单独下载一个 nvcc 可执行文件不足以编译。

先完成两个 Python 环境，再运行：

```bash
bash scripts/hpc/setup_cuda.sh --print-plan
bash scripts/hpc/setup_cuda.sh
source training_mjlab/hpc/cuda_env.sh
```

脚本从 [NVIDIA CUDA 12.8.1 组件清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) 固定以下 Linux x86_64 组件和 SHA-256：nvcc 12.8.93、cudart 12.8.90、CCCL 12.8.90，总下载约 81.3 MB。每个包按官方散列校验后解包，并保存许可证。需要的额外 math-library 头文件取自已锁定安装的四个 NVIDIA wheels，不再下载完整数学库；安装记录包含每个复制头文件的包版本及 SHA-256。这符合 NVIDIA [组件归档说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables)提供的按组件取用方式。

完整工具链保存在 `training_mjlab/hpc/cuda-12.8.1`，同级 `cuda_env.sh` 设置本进程的 `CUDA_HOME`、PATH 和运行库目录。脚本不安装驱动、不写 `/usr/local`、不执行系统包管理命令，也不把 CUDA stub 库加入运行时查找目录。无标记的既有目录会被拒绝；完整安装采用暂存目录后重命名，支持中断后重试。

### 可选的用户态兼容库

550 驱动对 CUDA 12.8 的 PTX JIT 支持不能只由 Torch 能识别 GPU 来证明。若 Slurm 实测确认需要兼容库，可明确安装并在该任务中激活：

```bash
bash scripts/hpc/setup_cuda.sh --with-compat
source training_mjlab/hpc/cuda_env.sh
source training_mjlab/hpc/cuda_compat_env.sh
```

该选项下载官方 `cuda-compat-12-8_570.211.01-0ubuntu1_amd64.deb`（约 37.95 MB），按 [NVIDIA Ubuntu 22.04 软件包索引](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz)中的固定 SHA-256 验证，只用 `dpkg-deb --extract` 解包到用户目录。它不会执行 `dpkg -i`、注册系统软件包或加载内核驱动。安装和激活分别操作；默认不启用这些兼容库。

[NVIDIA 的 forward compatibility 说明](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)允许符合条件的数据中心 GPU 使用用户态库来处理较新 CUDA/PTX 支持，并要求实际检查设备支持及错误。当前官方页已把 550 这类未列出的旧分支视为 EOL，因此本工程不宣称其组合得到当前支持保证。必须在 Slurm 中验证；若出现 803/804 或 PTX 编译错误，保留日志并停止，不绕过错误。不要将这组旧兼容库用于 580 及以上驱动。

本机只读调用 Warp 动态库的版本函数得到：当前 `warp-lang==1.17.0` 实际内置 CUDA 12.9，minor-version compatibility 开启，驱动仍未初始化。其生成 cubin 的兼容路径应先通过真实探针确认；不能认为装入 CUDA 12.8 兼容库就覆盖了所有 12.9 PTX 路径。GPU 探针现也记录 Warp 的 toolkit/driver 版本、作业的显式 compat 开关及实际加载的 `libcuda` / PTX JIT 库路径，以区分系统内核驱动和用户态兼容库。

## 用户目录 EGL dispatcher

实际远端环境有 `libEGL_nvidia.so.0` 和 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`，但缺少公共入口 `libEGL.so.1`。因此 `MUJOCO_GL=egl` 导入 MuJoCo 时，PyOpenGL 的 `PLATFORM.EGL` 为 None，访问 `eglQueryString` 报错。NVIDIA vendor 库不能单独替代 GLVND dispatcher。

```bash
bash scripts/hpc/setup_egl.sh --print-plan
bash scripts/hpc/setup_egl.sh
source training_mjlab/hpc/egl_env.sh
```

脚本固定使用 Ubuntu Jammy 的 [libegl1 1.4.0-1](https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libegl1_1.4.0-1_amd64.deb) 与 [libglvnd0 1.4.0-1](https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libglvnd0_1.4.0-1_amd64.deb)，总下载 102,146 字节。SHA-256 与远端已配置 Ubuntu 软件源的 `apt-cache show` 元数据核对后写死在脚本中，下载后再次验证。安装只用 `dpkg-deb --extract`，完整保留包内许可证，不调用 `apt install` / `dpkg -i`，也不修改系统包数据库。

两个包解包到 `training_mjlab/hpc/egl-jammy-1.4.0-1`；已有 NVIDIA vendor 库和系统 JSON 保持原样。新 `egl_env.sh` 仅为当前任务添加用户库目录，并设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。脚本不创建 EGL context、不访问 GPU；完整渲染仍须在 Slurm GPU 分配内检查。`libOpenGL.so.0` / `libGL.so.1` 等其他 GL 组件不由这个最小 dispatcher 修复代替。

## Slurm 内运行双卡探针

取得同一节点的两张 GPU 分配后，使用迁移包提供的 Slurm 入口，或在已分配 GPU 的任务内执行：

```bash
training_mjlab/.venv/bin/python scripts/hpc/probe_gpus.py \
  --output training_mjlab/hpc/reports/gpu_probe.json
```

探针在导入 Torch / Warp 前检查 `SLURM_JOB_ID`、GPU 分配变量及 `CUDA_VISIBLE_DEVICES`，不接受登录节点直接使用 GPU。默认要求 Torch 恰好看见两张分配的 GPU。Slurm 的可见设备编号可能重新映射，因此程序使用作业内的 `cuda:0` / `cuda:1`，不假定它们是宿主机的物理编号。

检查内容：

1. 保存驱动查询、GPU 名称/内存/计算能力、拓扑、Torch runtime、包版本、`nvcc` / C++ / Ninja 版本及 `CUDA_HOME`。
2. 每张卡执行 FP32 矩阵乘法并校验结果，关闭 TF32；逐卡捕获 CUDA graph、改动输入、重放并校验结果。
3. 检查两方向 peer access，复制默认 16 MiB 数据，逐元素校验，并记录 20 次复制的总耗时。计时包含最终同步，是该测试的实际传输速度，不能称为 NVLink 理论峰值。
4. 使用独立临时 Warp 缓存，编译并在两张卡执行小 kernel，验证 Torch / Warp 张量互操作和输出。不会清除其他任务的 Warp 缓存。

成功状态为 `passed_basic_gpu_checks`。错误时仍写 JSON，包含失败阶段和异常，进程返回非零。无分配时仅写拒绝原因，不查询 GPU。

**通过探针不等于完整训练通过。** 探针没有编译项目的 Adam CUDA 扩展，没有执行 MJLab 物理场景，也没有衡量 A3.2 的吞吐/F1。`nvcc` 不在 PATH 时会记录不可用；需要通过集群已有工具链/module 设置解决，不能直接改写系统 CUDA。驱动、Torch runtime、Warp 编译或 CUDA graph 任一阶段失败，需先定位相应错误，再决定下一步兼容方案。

## 实际本地验证

在当前工作区执行：

```bash
bash -n scripts/hpc/setup_envs.sh
bash scripts/hpc/setup_envs.sh --print-plan
python3 -m py_compile scripts/hpc/probe_gpus.py
training/.venv/bin/python -m pytest -q tests/test_hpc_environment.py
```

结果：**17 项测试通过**；覆盖无 Slurm/无 GPU/无可见设备拒绝、物理 GPU 编号 0 的正确识别、没有 site-packages 时仍能输出拒绝报告、安装计划不写环境、已有环境及符号链接不被改动，以及 CUDA 组件计划/兼容库显式开关/已有工具链保护和 EGL 计划/已有目录保护。未在当前单 GPU 机器执行 GPU 探针或重新安装 Python 依赖。

另在本机的新 `training_mjlab/hpc` 目录实际执行 `setup_cuda.sh --with-compat`：三个 tarball 和 compat deb 下载、官方 SHA-256/大小校验、用户目录解包及 nvcc 版本检查通过；`nvcc` 为 12.8.93。再次运行同命令无需重新下载，成功复用工具链。没有激活兼容库、没有访问本机 GPU，也没有修改原系统 CUDA 或两个已有 `.venv`。

使用 Python AST 读取项目 `adam_execution.py::_CUDA` 和已安装 Torch 的 `cpp_extension.py::COMMON_NVCC_FLAGS`，用新私有 nvcc 执行 `-c -O3 -std=c++17 --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -Xcompiler=-fPIC`，加上 Torch/Python 头文件目录与原 COMMON_NVCC_FLAGS。真实 Adam CUDA 翻译单元离线编译通过：31.09 秒，目标文件 175,720 字节，无编译警告。初次手写命令遗漏 Torch 默认的 `--expt-relaxed-constexpr` 导致 constexpr 报错，补齐原默认参数后通过；训练源码和编译数学设置未改。此项只证明编译器/头文件足够，不包含扩展动态链接、GPU 执行或优化器数值验收。

`setup_egl.sh` 的本地实际下载/SHA-256/解包/重复运行通过；`ctypes.CDLL` 加载私有 `libGLdispatch.so.0`、`libEGL.so.1` 并取得两个 EGL 符号通过。加载 `egl_env.sh` 后，PyOpenGL `platform.PLATFORM.EGL` 为有效 CDLL，`eglQueryString` 符号可取得。未调用该函数、未创建 EGL context，这些结果不代表完整渲染已验收。

## 参考与待完成

实际参考了两份 `AGENTS.md`、修改日志技能、两份 `requirements-tested.txt`、`training_mjlab/setup.sh`、`pyproject.toml`、`mjlab_env/source_runtime.py::dependency_versions`、`rl_torch/adam_execution.py`、Torch `cpp_extension.py` / `CUDAContextLight.h`、现有 uv CLI 帮助、Warp 安装源码，以及以上 HPC / NVIDIA 官方资料。

- **V32-HPC-ENV-01 已完成**：A100两环境安装与pip检查通过，版本保存在`training_mjlab/hpc/environment/20260928T104745Z.yiYpTQ`；Slurm932/934通过双卡Torch矩阵乘、CUDA graph、Warp真实编译/执行及P2P。实际加载原生550.144.03 libcuda，未启用兼容库。934中Composer完整导入和源码构建通过；普通SSH内无GPU权限时EGL显示初始化失败，不在普通shell执行GPU测试。
- **V32-HPC-ENV-02 已完成**：Slurm934实际完成Adam扩展/图捕获、A3.2物理、两轮单卡与两轮双卡短训，四次检查点独立重建全部通过；双卡稳定吞吐平均1745.29条/秒、较单卡提升28.15%。完整结果及935正式启动路径见[迁移说明](volcengine_a32_migration_20260928.md)。
