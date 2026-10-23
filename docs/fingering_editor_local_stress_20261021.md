# 本地指法预览压力测试（2026-10-21）

用户明确要求评估本机能否完成模型构建和求解。目标是在桌面继续运行时验证当前A3.2编辑预览的内存与交互表现；不进行训练、策略推理、动力学、GPU或硬件测试。

**当前结论：本地绘图通过；完整源构建未通过保护门槛，持续本地IK尚未验证。默认继续远端构建/求解、本地受限渲染。** 未提高预算、增加swap或关闭保护来取得通过结果。

## 固定预算和停止条件

| 对象 | MemoryHigh | MemoryMax | MemorySwapMax | 其他保护 |
|---|---:|---:|---:|---|
| 本地编辑/渲染窗口 | 384 MiB | 512 MiB | 0 | 独立scope，核对实际内核限制 |
| 本地源码准备、模型构建与IK | 1024 MiB | 1536 MiB | 0 | 独立scope，监控所有子进程与文件缓存 |

两者上限合计2GiB。启动完整测试/模型前要求 `MemAvailable≥4GiB`，不足则拒绝，不能关闭其他应用来伪装代表性桌面负载。CUDA可见设备为空、`MUJOCO_GL=disable`，BLAS/OMP线程数均为1；保留系统OOM保护。

监控每250ms读取cgroup `memory.current/peak/stat/events`、系统 `MemAvailable` 和 `/proc/pressure/memory`；以下任何条件触发就停止该worker scope：

- 当前cgroup内存超过硬上限80%（约1228.8MiB）；此值包含匿名内存及文件缓存。
- 全系统可用内存低于2GiB。
- 全系统内存 `full avg10` PSI超过1%。这可以在触及硬限之前停止内存回收压力，不等同于已经发生OOM。
- 初始化超过180秒，或资源监控不能继续工作。

cgroup硬限是监控轮询之间的最后屏障。worker在重型导入之前检查实际限制，不能只传一个环境变量来绕过；`memory.oom.group=1`作用于该独立scope。系统systemd255不支持scope的 `MemoryOOMGroup=` 参数，改为写入用户委派scope的同名内核文件并读回校验。未请求系统管理员权限或更改系统配置。

## 测试阶段与验收

| 阶段 | 工作负载 | 记录与通过条件 |
|---|---|---|
| 0 静态与保护检查 | 环境版本、模型源/worker SHA、cgroup限额、当前桌面负载 | 文件匹配、限制有效；错误时不启动模型 |
| 1 本地绘图 | 同一批已保存姿态绘制20帧，校验左右手及相机 | 512MiB内、OOM=0、画面正确；统计帧耗时 |
| 2 冷启动 | 从固定v6源码构建完整场景，再初始化CPU耦合与IK | 180秒内ready、无保护终止、OOM=0；记录启动时长、版本、PID与峰值 |
| 3 连续求解 | 默认120秒；10指单音、键域两端、多指和弦、同指冲突、示例原始起音时刻 | 有限状态、原限位检查、八指60–80°、琴键qpos不变；统计延迟与内存趋势 |
| 4 拖动风暴 | 连续10000个时间定位请求 | GUI实际队列只保留1个待处理请求，最新ID正确，不累计大消息队列 |
| 5 重启与延长 | 可选2次冷启动；每次持续时长最高600秒 | 比较每次峰值、错误、请求延迟；短测不能宣称长期无泄漏 |

各阶段有依赖关系；冷构建不通过就停止，不跳过它后宣称完整本地运行通过。IK大残差是姿态质量指标，单独记录，不通过放松限位/耦合阈值来降低残差。全程不写琴键按下状态；不产出真实触键F1。

## 命令与产物

```bash
# 默认120秒连续求解；前置阶段失败会提前终止
./.venv/bin/python scripts/stress_fingering_editor.py --seconds 120

# 冷启动通过后，可做两次重启、每次10分钟的后续测试
./.venv/bin/python scripts/stress_fingering_editor.py --seconds 600 --restarts 2

# 日常默认：远端构建/求解，本地512MiB内绘图
./launch_fingering_editor.sh

# 显式实验路径：本轮冷构建未通过，不建议日常使用
./launch_fingering_editor.sh --backend local
```

脚本自动创建不覆盖旧结果的 `results/fingering_local_stress_时间戳/`。`report.json` 包含源码/样例SHA、命令、作用范围、退出状态、资源采样和已运行阶段；只有冷构建和后续阶段实际通过才标为passed。通过后另存 `ready_*.json` 和 `poses.json`。源码准备、worker stdout/stderr与实际内核参数在 `logs/fingering_editor/pianist-fingering-worker-*.{log,json}`。

本地运行时由同一固定归档解包到独立 `logs/fingering_editor/local_runtimes/`，只复用现有依赖环境；不读取远端旧工程、当前并行修改中的工作区算法或旧checkpoint。固定源SHA为 `9299ab01d69563092b65cd4d91b09429ae99aa501cedbb797b9956bd098fece8`。本地Composer为MuJoCo3.13.0，求解依赖环境为3.11.0；必须由现有场景加载/选项验证接受，未放松版本或合同检查。

冷构建第二次测试去掉了编辑器不使用的乐曲训练任务，只调用同一源码的 `export_scene(build_reference=False)`；保留真实完整场景、凸包、惯量、关节、驱动及限位。先等待Composer退出再导入Torch，避免两个大运行时重叠。此优化没有改训练源码、奖励或归档。

## 本轮实测

- **保护参数初检**：第一次命令因systemd不接受scope `MemoryOOMGroup=` 在重型进程启动前失败；诊断保留于 `results/fingering_local_stress_20261021_120532/report.json`。修复为实际内核文件设置/校验，预算不变。
- **冷启动1**：`results/fingering_local_stress_20261021_120602/report.json`，峰值约1024.5MiB，因PSI full avg10>1%主动停止。`memory.events` 的oom、oom_kill、oom_group_kill均为0，硬限事件max=0；未达到ready。
- **冷启动2（只导出场景）**：`results/fingering_local_stress_20261021_120727/report.json`，峰值约1039.3MiB（1.015GiB），仍因相同PSI阈值主动停止。采样可见匿名内存升至约1030.5MiB；不是仅凭大虚拟地址判断。OOM与硬限max事件仍为0；持续IK与重启阶段未运行。
- **本地绘图**：`logs/fingering_editor/camera_fixed_20261021.json`，20帧峰值64.7MiB，耗时中位0.280秒/帧、最大0.347秒/帧，high/max/oom事件均为0；这是CPU简化三角形绘制，约3.5帧/秒，不称高帧率GPU渲染。
- **轻量验证**：`tests/test_fingering_editor_data.py` 和 `test_fingering_editor_safety.py` 合计7项通过；覆盖原谱保留、预算拒绝、上方相机/左右关系和10000次请求的队列上界。

后续要回答“本地求解单独是否可用”，可另做远端编译产物在本机受限加载的隔离实验；本轮没有该实测，不能据此宣称本地IK已经通过。全模型源码构建继续保留为待优化项，不能以当前短测排除其他桌面负载下的内存风险。
