# 2026-09-24：源码与完整配置训练／回放

本页覆盖此前 `source_bundle_cache_20260924.md` 的持久化 scene/task 缓存方案。新入口是项目根目录 `run.sh`，回放入口是 `eval.sh`。保留旧检查点的历史读取路径。

## 当前设置

| 项目 | 新训练入口实际设置 |
| --- | --- |
| 曲目 | TwinkleTwinkleRousseau |
| 曲首准备时间 | 0 秒（2026-09-25 用户取消缓冲）；每次 reset 后直接开始原乐谱 |
| 指法 | 保留乐谱原有指法；已按用户最新要求撤回覆盖指法的改动 |
| 手部重力补偿 | 开启；模型中 44 个手部 body 的 gravcomp=1 |
| Actor | 256、256、256，共 3 个隐层 |
| 初始温度 | 1，仍自动学习温度 |
| discount / gamma | 0.8；与终止 transition 的 discount 分开 |
| 姿态 | 当前源码 h=0.10 m、TY=0.04 m、左右 Yaw=0.15 rad |
| 时序与前瞻 | 控制 20 Hz／物理 200 Hz，0.5 秒前瞻，即 10 步 |
| 动作／观测 | 无臂、结构性移除 TZ 与 Yaw，47 动作／1180 维观测 |
| 奖励及驱动 | direct_joint_v1（四指MCP `[0,2]→[0,1.2] rad`）、自碰撞权重 0.2，三类实体接触；其他规则沿用当前源码 |
| 并行规模与总预算 | 512 环境 × 9766 向量步＝5,000,192 条 transition |
| Replay／batch | 512 向量槽 × 512 环境＝262,144 条；batch 4096 |
| SAC 更新强度 | 每向量步 64 次更新；稳态 0.125 update/transition，batch 样本复用比 512 |
| 样本日程 | 随机预热 150 向量步＝76,800 条；500 步＝256,000 条后开始学习 |
| 评估／保存／日志 | 每 500／500／5 向量步，即每 256,000／256,000／2,560 条样本 |

2026-09-25 用户取消曲首缓冲，`run.sh` 和 `scripts/train_from_source.py` 默认均为0秒；新小星星恢复158个控制步。旧检查点仍使用保存的配置，不修改已有训练进程。

历史设置：1 秒缓冲是乐谱开头的空白时间，物理、策略和奖励计算仍在进行；不会冻结手、插值动作或限制关节速度。其作用是给策略学习准备姿态的时间，不保证消除大幅动作。当时小星星整曲由 158 步变为 178 步。

用户后来明确“不需要改这处指法了”，本轮指法覆盖参数、音符 part 重写及绕过指法识别的分支已全部撤回。`PianoWithTsenseiHands` 和 `MidiFile.has_fingering()` 恢复原有识别逻辑，乐谱保留原始指法。指法数组保留原曲定义；历史1秒缓冲仅增加前置空白。

重力补偿在 `suite/tasks/base.py`、`rl/environment.py` 和 scene 构建函数中本来已默认开启。本轮将其显式写入训练命令、环境配方和实际配置，并验证编译结果，未把钢琴的重力关闭。

## 使用方式

在本独立项目目录运行：

```bash
bash run.sh --output-dir training_mjlab/outputs/my-run

# 将训练开始时打印的实际输出目录填在这里；以下时间仅是格式示例。
RUN_DIR=training_mjlab/outputs/my-run-20260924-180000-123456

# 完整整曲数值评价
bash eval.sh --checkpoint "$RUN_DIR/final.pt"

# 交互回放；可加 --paused
bash eval.sh --checkpoint "$RUN_DIR/final.pt" --mode preview

# 导出视频
bash eval.sh --checkpoint "$RUN_DIR/final.pt" \
  --mode render --output preview/outputs/my-run
```

`run.sh` 按 Shadow `run.sh` 风格逐项列出参数，末尾传参可以覆盖默认值。2026-09-24 用户将后续默认并行规模从 256 提高到 512，并确认其他按样本数／更新比相关参数一并调整。默认计划为 512 环境 × 9766 向量步，即仍为 5,000,192 个样本；`gradient_steps=64` 保持原来的 `64/512=0.125 update/transition`，`replay_vector_slots=512` 保持 262,144 条 replay 容量，batch 仍为 4096。预热、学习起点、评估、保存和日志间隔减半，使对应的样本数与旧 256 环境入口一致。

`--output-dir` 是运行目录的基础名称。训练入口在构建环境前为它追加北京时间 `YYYYMMDD-HHMMSS-微秒` 并打印最终目录；未传时使用 `training_mjlab/outputs/sac` 作为基础名称。重复使用同一基础名称会创建新目录，原检查点保留。回放时使用打印出的实际目录；`config.json` 的 `cli.output_dir` 也记录这个目录。

这一调整不是论文原始的 256 环境设置，而是保持本项目总样本预算和更新强度的 512 环境适配。由于学习条件从 `step >= learning_starts` 起算，新的完整预算产生 593,088 次更新，旧配置为 593,056 次，相差一次旧向量步的 32 次更新；样本学习起点均为 256,000 条。正在运行的 `sac-twinkle-source-5m-20260924` 已按旧 256 环境配置创建，不会因脚本修改而热切换；新默认只作用于后续启动。下面列出的 GPU 短程验证属于此前源码回放功能验收，不是 512 环境吞吐或收敛验收。

进度显示默认开启，无需额外参数：入口先显示读取配置、构建模型／乐谱的阶段提示，GPU初始化后显示训练百分比、已采样／总采样、速度、预计剩余时间、向量步、SAC更新数及最近一次评价F1。首次物理／SAC编译可能使进度暂时停留；F1在首次评价完成后显示。使用 `--no-progress` 可关闭阶段提示和训练进度条。进度信息写入 stderr；需要同时保存到日志时使用 `2>&1 | tee train.log`。

`eval.sh` 默认读取 checkpoint 同目录的 `config.json`，也可显式加 `--config /path/config.json`。可移动整个运行结果目录；回放不依赖训练时的场景目录。`config.json` 包含有效 SAC 配置、全部解析后的训练 CLI、奖励配置、动作开关、reset 合同、源姿态、乐谱处理参数、依赖版本和源码 SHA-256。外部乐谱文件以 base64 和校验值嵌入配置；不再依赖原文件路径。检查点也保存同一份元数据，并校验外部 config 与检查点一致。

源码配方覆盖包内模型和资产的散列。需要保存与训练相同的项目源码及依赖版本；配置里的散列不是源码备份。源码或依赖已修改时会明确报错，避免静默用新环境解释旧策略。Actor 层数已变化，旧两层检查点不能直接热启动到当前三层网络。

## 环境如何重建

训练由 `scripts/train_from_source.py` 创建完整环境配方，再由 `mjlab_env/source_runtime.py` 构建；回放通过 `mjlab_env/paths.py` 读取检查点中的同一配方。保存的记录是脚本、源码、完整 JSON 配置与权重。

本机 Composer 在 `training/.venv`，MJLab/Torch 在 `training_mjlab/.venv`。两个解释器之间仍用临时目录传递生成的 MJCF、资产及任务数组；这是当前实现的编译中间产物，正常退出时删除。运行目录不写持久化 scene/task 或 source_cache；回放会重新生成并核对模型和任务的散列。没有声称完全取消中间 MJCF 文件或将两个 Python 环境合并。

兼容历史记录的旧 `--scene/--task` 入口仍存在，新的 `run.sh` 不使用它。无臂处理产生的中间结果也放入临时构建目录。渲染报告包含源码配方，因此保存轨迹后重新取景也能重建环境。

参考了本机 `/home/houjue/DaxianPianist/robopianist-rl/eval_song.py` 读取 args.pkl 并调用 `train.get_env(args)` 的流程，以及原版 `robopianist-rl/run.sh` 的显式参数写法。没有找到名为 `eval_daxian_unlock` 的目录，未声称检查了该目录的专用实现；本次落实的是用户明确指定的“脚本＋源码＋完整配置”方法。独立项目没有新增对这些参考目录的导入或复制依赖。

## 验证结果

- `training_mjlab/.venv/bin/python -m unittest discover -s tests -p test_torch_sac.py`：5 项通过，包含新 CLI 的温度／Actor／gamma 和历史论文参数显式恢复。
- `training_mjlab/.venv/bin/python -m unittest discover -s tests -p test_source_runtime.py`：撤回指法修改后3项通过。验证20帧空白后乐谱与原指法逐元素一致、44 个补偿 body；删除首次构建目录后，从 JSON 重建所得 scene/task 散列一致；篡改源码散列或外部配置被拒绝。
- `MUJOCO_GL=egl training/.venv/bin/python -m pytest tests/test_environment.py -q`：19 项通过。先前误用 unittest 对该 pytest 文件未发现测试，随后已用正确入口验证。
- `bash -n run.sh eval.sh`：通过。
- 第一次短测 `source_config_smoke_20260924` 在撤回指法修改前完成：420 transition、207 次 SAC 更新；该临时检查点已被最终原指法短测替代，不用它验证当前方案。
- 最终短测 `training_mjlab/outputs/source_config_original_fingering_smoke_20260924`：2环境×210步，共420 transition、207次SAC更新、2次完整episode结束、0次截断；physics graph、Torch compile和fused optimizer均开启。47动作／1180观测，实际SAC温度1、三层Actor、gamma0.8。运行目录仅有config.json、权重、指标、summary及eval结果，无持久化scene/task。
- 原训练进程退出后运行 `eval.sh`：重新构建环境并完成178步整曲评价，F1=0.101063。
- `eval.sh --mode render --no-audio`：NVIDIA EGL渲染通过，178步整曲＋初始帧＋结尾停留，共199帧、9.95秒；产物为 `preview/outputs/source_config_original_fingering_smoke_20260924/replay.mp4`。修复了重建时先导入MuJoCo、后选择EGL导致OpenGL库未加载的问题；交互入口同样在重建前选择图形后端，未自动操作桌面窗口。
- 独立渲染运行F1=0.102686，与数值评价相差约0.00162；环境散列一致不等于GPU动力学逐位同轨迹，此轮未定位该小幅重复运行差异。以上只证明训练、原指法和源码重建回放可运行，不表示短训策略已学会演奏。

## 聊天记录与遗留事项

`scripts/export_chat_answers.py` 将当前会话与父会话可获得的助手最终回答和进度说明逐条导出至 `docs/chat_answers_20260924.md`，排除内部推理、工具输出和其他会话。再次运行可更新快照：

```bash
python3 scripts/export_chat_answers.py \
  --thread 01a0d243-10ba-7d30-83e9-d779743ccdb8 \
  --output docs/chat_answers_20260924.md
```

- V32-CHAT-01：父会话本地记录仅到序号 408，当前分支引用截止序号 1630，中间历史未找到；需要补充该历史段的原始可见聊天导出才能补齐，不能称为完整侧边栏历史。
- V32-RESET-01：1 秒准备时间与上一动作 reset 修复不保证策略动作平滑，仍需训练后测首步目标和完整视频。
- V32-SAC-03、V32-04/06：长期学习表现、硬件／接触标定与此前 CPU/GPU 接触差异仍保留，本轮短测没有解决这些事项。

## 当前训练阶段回放（2026-09-24）

`sac-twinkle-source-5m-20260924`仍在运行，本次快照读取到step-10000的第10次整曲评价，训练主循环随后已到step-10110／2,588,160条transition。已保存检查点中`step-5000.pt`仍是日志逐帧平均按键F1最高者：F1=0.3601208、precision=0.2630269、recall=0.8852997、micro-F1=0.3062843、return=429.4767；step-10000的F1=0.3594902，略低于它。选优只比较当前运行已保存且有有限整曲评价的检查点，训练完成前最佳值仍可能变化。

对`step-5000.pt`从保存的源码配方重建环境并独立执行确定性整曲回放，得到F1=0.3347617、precision=0.2461188、recall=0.8188204、micro-F1=0.2880711、return=423.5324，`logged_metrics_match=false`。复播F1比训练日志低约0.02536，继续保留V32-SAC-03；不能用训练日志值替代视频这次实际轨迹。高召回而低精确率，以及画面中同时出现多枚非目标绿色琴键，说明策略已覆盖较多目标帧，但误按／多按仍明显，当前阶段不能称为学会演奏。踏板precision／recall／F1均为1，只说明本次踏板目标匹配。

[完整有声视频](../preview/outputs/replays/twinkle_source_current_best_step5000_20260924/replay.mp4)为H.264 1280×720／20 FPS＋AAC，199帧、9.95秒，包含178个控制帧、初始帧和1秒结尾停留。音频由每个200 Hz物理子步的真实琴键状态和踏板状态生成，共134次note-on，不是目标乐谱配音；力度固定127且未标定。目录同时保存`selection.json`、`report.json`、`rollout.npz`、物理子步音频状态、MIDI、WAV和检查截图。视频SHA-256为`5cd590db070a65666c1cbb94ba21f72e0841bb4cdabcfc158e7bd2b395945e45`。

论文§2.10（pp.12–13）报告的是25 ms键状态网格上的Precision／Recall／F1；本项目本轮按已确认的20 Hz即50 ms规则计算逐帧平均F1，并另存micro-F1，因此本段结果只用于当前项目内部选优，不与论文数值直接横比。标准`export_best_mjlab.py`的选优逻辑可用于源码检查点，但其完整导出路径会先解析再显式传入scene/task，而源码检查点要求自行重建且拒绝显式路径；本次使用`--select-only`后直接调用`render_mjlab_checkpoint.py`，没有修改训练或回放源码。

- V32-EXPORT-02：如需让`export_best_mjlab.py`一条命令兼容源码检查点，应避免重复解析并补回归测试；本次用户需求已由直接渲染路径完成。
- V32-SAC-03：继续跟踪同一checkpoint训练评价与独立复播的轨迹／指标差异。
- V32-AUDIO-01：实际触键音符力度仍固定127，尚未完成力度标定。
