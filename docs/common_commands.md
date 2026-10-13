# V3.2 常用命令速查

## 当前训练入口（2026-09-28）

后续新训练使用项目显式配置，不按论文预设启动，也无需论文比对或对齐确认：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
bash run.sh --inspect-only
# 实际开训时移除 --inspect-only；A3.2提前定位入口为 bash run_tsensei_a32.sh
```

参数见 `run.sh` 文件头，运行结果以该轮完整 `config.json` 为准。当前为20/200 Hz、三个学习率3e-4、Actor/Critic 256×3、gamma .8、512环境、batch1024、每样本.25次更新；具体可选项按入口显式参数生效。下文早期冻结场景和论文预设命令仅用于历史解释，不是新训练推荐命令。

## 历史入口：源码构建与缓存（2026-09-24，已由run.sh替代）

当前 h10／TY0.04／Yaw0.15 的小星星 49／47 动作包装脚本先按源码及覆盖参数生成 SHA-256 缓存，再将缓存 scene/task 交给原 Torch SAC。先用 `--build-only` 构建，或用 `--inspect-only` 核查训练参数；不带这两个参数才会启动长期训练。详情和旧快照清理清单见[源码缓存说明](source_bundle_cache_20260924.md)。

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
bash scripts/train_twinkle_h10_ty04_yaw015_no_tz_no_yaw.sh --build-only
bash scripts/train_twinkle_h10_ty04_yaw015_no_tz_no_yaw.sh --inspect-only
```

下文早期冻结包、运行结果和命令是历史记录；部分旧快照已经清理，不能用它们直接回放旧检查点。

## 正在运行：修复接触后的新小星星（2026-09-24 08:14）

已按用户要求启动约 5M 样本的新训练：256 环境、20/200 Hz、12 cm、无臂、TZ 物理移除、49 动作、真实接触及自碰撞惩罚 0.2。运行名 `sac-twinkle-contact-stable-20260924-081436`。[配置、启动记录和查看日志命令](training_twinkle_contact_stable_20260924.md)。

## 最新：接触振荡修复（2026-09-24）

当前预览默认恢复同手／跨手／手—琴键真实接触，手指采用 armature=0.0002、Kp=1。关闭旧窗口后运行：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
.venv/bin/python preview/scripts/preview_pose.py --no-arms
```

`bash scripts/train_twinkle_h12_no_tz.sh` 现在从当前源码缓存构建12 cm、20/200 Hz包，默认关闭两种 detection-only 参数；新 task 使用中立 reset 动作历史。以下历史章节中的默认“仅检测”说明已被覆盖。旧检查点仍按原冻结场景回放。[接触原因与复验](contact_stability_shadow_20260924.md)；[reset合同](reset_recovery_20260924.md)。

2026-09-23更新：M6小星星5M已完成，最佳step-18000日志F1=0.204550，有声回放F1=0.193968。[结果、曲线及视频](training_m6_twinkle_20260923.md)。

2026-09-23：已启动含93 mm M6机械臂的小星星5M训练，mixed_v1、forearm扩展＋同手自碰撞0.2。[本轮命令与运行记录](training_m6_twinkle_20260923.md)。

## 93 mm M6 双臂与 forearm 区域（2026-09-22）

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
.venv/bin/python preview/scripts/preview_pose.py --paused --no-contact-controls --show-forearm-collisions
```

手腕目标受可达性及机械臂速度约束，P 默认保存实际姿态。新 GPU scene/task 在 `training_mjlab/outputs/m6_93mm_clair_20260922_final`；[新训练／GPU预览／导出命令及验收](m6_forearm_93mm_20260922.md)。旧命令指向旧包时不包含机械臂奖励。

## M6 双臂＋V3.2 随动预览（2026-09-21）

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
.venv/bin/python preview/scripts/preview_pose.py --paused --no-contact-controls
```

Wrist 滑块／绿色腕部目标球可调位置和姿态，双臂差分 IK 跟随。F 回到全琴双臂视角；P 保存手姿态。安装配置另存于包内 `assets/tj_m6/preview_mount.json`，需编辑后重启；`--no-arms` 可关闭。[全部操作及限制](tj_m6_preview_ik.md)。仅源姿态预览，不改变训练或检查点回放。

更新：2026-09-21。适用于本独立项目的 MJLab＋Torch SAC 主线。

## 最新检查与 F1 曲线（2026-09-21）

forearm已扩展跨手手指碰撞。新训练需显式使用新task；[区域图、规则与任务路径](forearm_full_hand_20260921.md)。旧task保持原掌部检测。

两轮《月光》最佳step-1000的完整有声视频及复现命令见[月光视频导出记录](clair_videos_20260921.md)，含最新mixed_v1与原映射对照。

6轮正式MJLab训练已检查：5轮完成、1轮因改曲主动停止。两轮月光均完成5M采样，但最终F1分别为0.022305／0.011787，最佳均为step-1000；不要默认用final导出最佳视频。完整表、最佳检查点及F1曲线见[训练结果报告](training_results_20260921.md)。当前未发现仍运行的训练进程。

从独立项目根目录重新生成所有已记录训练的F1曲线：

```bash
training_mjlab/.venv/bin/python scripts/plot_training_f1.py \
  --output "preview/outputs/training_reports/f1_$(date +%Y%m%d_%H%M%S)"
```

**新训练默认已改为 Shadow 接触、40/400 Hz、20步前瞻与2070维观测。** 第1节视频／指标为旧配置历史结果，旧检查点回放自动查找原冻结包，不能用于新2070维任务。滑块操作及严格物理对照尚有1项失败，见[参数与验收](shadow_contact_40hz_acceptance.md)。

## 最新运行：mixed_v1＋Pitch ±0.2（2026-09-20）

`sac-20260920-212536-832002` 已完成，混合动作输出、Pitch物理范围±0.2rad，自碰撞0.2，《月光》5M。完整映射／命令见[新映射说明](action_mapping_mixed_v1.md)，最终指标见上方9月21日报告。

## 上一轮运行：《月光》候选指法＋自碰撞（2026-09-20）

已生成146音符候选指法，`sac-20260920-204729-873996` 已完成，40/400 Hz、5M采样、自碰撞权重0.2；本轮用保存的双腕yaw=0.25和安装Z=0.14新场景。逐音符指法、验证、日志和导出命令见[《月光》训练记录](training_clair_de_lune_20260920.md)。此前小星星自碰撞短轮已因改曲停止。

## 最新训练结果（2026-09-20）

40/400 Hz新配置小星星5M训练 `sac-20260920-171514-880281` 已完成，耗时 **59分41.9秒**，本轮最佳为 `final.pt`（日志F1=0.8595394）。[最新全琴有声视频](../preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920/replay.mp4)及[训练记录／导出命令](training_run_20260920.md)。下方第1节保留旧配置的历史最佳结果。

## 0. 先进入项目目录

以下命令均从**独立项目根目录**运行，不是外层 `pianist_v2`：

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
```

复制项目到其他位置时，只改上面的路径；后面的命令使用相对路径。

| Python 环境 | 用途 |
| --- | --- |
| `training_mjlab/.venv/bin/python` | MJLab训练、GPU策略回放、评价、有声视频 |
| `.venv/bin/python` | 可编辑／保存初始姿态的原生MuJoCo预览、导出物理场景 |
| `training/.venv/bin/python` | 乐谱任务导出、历史JAX `.msgpack`训练／回放 |

桌面窗口需要在有 `DISPLAY` 的桌面终端运行；视频导出使用 EGL，不需要打开窗口。训练、视频和场景导出的输出目录须为新目录，重复运行请换名字。以下 `V32_RUN` 和 `V32_CHECKPOINT` 是便于复制命令的变量，在新终端需要重新设置。

## 1. 本次最新训练和最佳结果

截至2026-09-18 15:54，本轮训练仍在运行。以下是导出时已有日志与已保存检查点中的最佳结果。

```bash
V32_RUN=training_mjlab/outputs/sac-20260918-135539-423462
V32_CHECKPOINT="$V32_RUN/step-15000.pt"
```

- 日志最佳：`step-15000.pt`，对应3,840,000条采样、448,032次更新；按键F1 **0.9297470**，奖励 **512.0909**。
- 本次视频实测：F1 **0.9314348**，奖励 **512.7067**，158个控制步完整曲终。
- 当前检查点使用原五项奖励＋自碰撞权重0.2，回放自动恢复；不是上一轮新增论文奖励的全开训练。
- 视频为正面正交全钢琴、1280×720、20fps、8.95秒；音轨由实际物理按键事件生成，保留错音。
- [完整钢琴正面有声视频](../preview/outputs/replays/sac-20260918-135539-423462_best-export-20260918/replay.mp4)
- [回放报告](../preview/outputs/replays/sac-20260918-135539-423462_best-export-20260918/report.json) ／ [轮内选择依据](../preview/outputs/replays/sac-20260918-135539-423462_best-export-20260918/selection.json) ／ [同场景同曲目比较](../preview/outputs/replays/sac-20260918-135539-423462_best-export-20260918/comparison.json)

上一轮 `sac-20260917-150810-595009` 已完成5,000,192条采样，日志最佳 `step-18000.pt` 的F1为0.9173900。两轮奖励配置不同，因此跨轮按相同任务的按键F1比较，不用总奖励比较；短程截断验收和不同任务不参与本次选择。

“最佳”依据已记录的独立评价，不是重新测试全部策略后取最大值，也不是多seed效果承诺。跨次GPU回放有差异，本次F1相差约0.169个百分点，继续记录为V32-SAC-03。运行中的训练后续可能产生更好的检查点，可用下一节命令重新选择。

## 2. 一条命令导出该轮最佳视频

```bash
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir "$V32_RUN"
```

默认从该run日志中选择**有实际检查点文件**的最高按键F1；相同F1按奖励、再按较晚向量步排序。完成run的最后一次评价也可对应 `final.pt`。自动生成带时间戳的新输出目录，并打印 `replay.mp4` 路径；默认全钢琴正面正交视角＋真实按键事件音轨。

只查看最佳检查点，不启动GPU回放：

```bash
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir "$V32_RUN" --select-only
```

新一轮训练完成后，将 `V32_RUN` 改为新目录。该命令只在指定run内选优，不混入其他曲目或短程测试。自动按检查点散列查找训练时的冻结scene/task；若包移至其他目录，加 `--scene 路径 --task 路径`，散列不匹配会拒绝载入。

## 3. 导出指定检查点视频

```bash
training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py \
  --checkpoint "$V32_CHECKPOINT" \
  --camera-view front --camera-framing full --audio \
  --output preview/outputs/replays/my_step18000_full_audio
```

| 参数 | 用途 |
| --- | --- |
| `--camera-view front` | 正对演奏者的正交视角，默认俯视30°，方便观察离键高度 |
| `--camera-framing full` | 完整钢琴与双手，默认 |
| `--camera-elevation -30` | 指定俯仰角；用−60可恢复原高俯视角，导出最佳／指定检查点／轨迹换角均支持 |
| `--camera-distance 1.05` | 调整正交视野范围；数值越小放大越多，可能裁剪琴体 |
| `--camera-framing hands` | 放大双手，会裁掉部分琴体 |
| `--camera-view performer` | 从演奏者一侧看琴，正交 |
| `--camera-view overview` | 旧版斜视透视 |
| `--no-audio` | 关闭音轨 |
| `--finger-labels` | 显示浮动指法文字；默认关闭以减少遮挡 |
| `--soundfont /path/to/piano.sf2` | 更换钢琴采样音色 |

首次缺少音频依赖时：

```bash
training_mjlab/.venv/bin/python preview/scripts/setup_piano_audio.py
```

此安装脚本适用于本机Ubuntu，下载并解包到项目 `training_mjlab/audio_runtime/`，不使用sudo。声音来自实际物理子步的按键／松键／踏板事件，保留错音；不播放目标乐谱。力度目前固定127，尚未标定真实击键音量（V32-AUDIO-01）。输出同时保存WAV、MIDI、事件JSON和轨迹NPZ。

### 同一次演奏换视角，保持音轨和动作完全相同

默认front现为−30°。本次已生成[step-15000低角度全琴视频](../preview/outputs/replays/step15000_front_height_20260918/replay.mp4)，其轨迹和WAV与前一版完全相同。起始动作／速度诊断见[实测说明](startup_height_speed_20260918.md)；其中记录了该训练随后完成的总耗时。

```bash
training_mjlab/.venv/bin/python preview/scripts/reframe_mjlab_recording.py \
  --recording preview/outputs/replays/sac-20260917-150810-595009_best-step-18000_20260918 \
  --camera-view front --camera-framing hands \
  --output preview/outputs/replays/my_step18000_same_take_hands
```

只重新渲染保存的轨迹，不重新采样策略／推进GPU物理。自动按录制报告散列匹配原包；移至其他目录时可显式传 `--task`／`--scene`。

## 4. 历史40/400 Hz小星星5M命令（旧论文预设）

```bash
training_mjlab/.venv/bin/python -u -m tsensei_v3_2_pianist.rl_torch.train \
  --scene training_mjlab/outputs/scene_shadow_40hz400hz \
  --task training_mjlab/outputs/task_twinkle_shadow_40hz400hz \
  --stage 1 --device cuda:0 --num-envs 256 \
  --vector-steps 19532 --batch-size 4096 --gradient-steps 32 --seed 42
```

自动创建 `training_mjlab/outputs/sac-日期-时间/`；也可加 `--output-dir training_mjlab/outputs/my_new_run` 指定新目录。

- 一个向量步推进256个环境：19532×256＝**5,000,192条采样**，不是每个环境5M步。
- 每个符合条件的向量步执行32次SAC更新；只改环境数而不调整其余预算，会改变更新／采样比例。
- 这条历史命令当时采用论文SAC参数；双手、40 Hz控制／400 Hz物理、0.5秒前瞻（20步／2070维观测）、原任务奖励、错键不提前结束，不代表整篇论文复现。
- 查看配置而不启动物理／训练：

```bash
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --num-envs 256 --vector-steps 19532 --inspect-only
```

训练进行时看日志：

```bash
tail -n 5 "$V32_RUN/metrics.jsonl"
# 连续显示新日志；Ctrl+C只退出tail
tail -f "$V32_RUN/metrics.jsonl"
# 只看独立评价行
rg '"eval"' "$V32_RUN/metrics.jsonl"
# 训练正常结束后才有summary
cat "$V32_RUN/summary.json"
nvidia-smi
```

开始新训练后，将 `V32_RUN` 改为新目录再查看。`step-N.pt` 的N为向量步。按键F1和总奖励都要看，不能仅以loss下降判断演奏效果。

## 5. 在窗口回放已训练策略

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --checkpoint "$V32_CHECKPOINT" \
  --camera-view front --camera-framing full --paused
```

空格开始／暂停，R重置；曲终先R再空格重播。带 `--checkpoint` 时策略控制关节与踏板，Control滑块不接管策略。这个桌面入口目前不实时发声，有声回放用视频导出。

## 6. 预览姿态、手动驱动和保存初值

### 预览可编辑的源模型

```bash
.venv/bin/python preview/scripts/preview_pose.py
# 显示世界／左右腕，以及双侧 M6 J1–J7 坐标轴（启用机械臂时）
.venv/bin/python preview/scripts/preview_pose.py --show-coordinate-frames
# 遮挡时显示半透明模型
.venv/bin/python preview/scripts/preview_pose.py --show-coordinate-frames --transparent-model
# 无窗口截图（同名文件会覆盖）
.venv/bin/python preview/scripts/preview_pose.py --headless --output preview/outputs/my_pose.png
```

新增的独立 solimp 面板默认打开，主滑块调d0，另有四项高级参数；用 `--no-contact-controls` 隐藏。点击“保存接触配置”才写入包内 `assets/contact_settings.json` 并备份；P仍仅保存姿态。详见[滑块用法](shadow_contact_40hz_acceptance.md#3-预览和滑块验收)。

启动后终端的`Shadow contact parameters: MATCH`表示实际编译模型已逐项匹配原版Shadow接触规则；`pose.json`中有完整`shadow_contact_comparison`。滑块实验后点击“恢复原版 Shadow solimp”可恢复手／钢琴／琴键限位三组阻抗。

右侧Control调关节与腕部目标；空格暂停／继续，R恢复打开窗口时的状态。暂停时滑块改变的是目标，继续运行物理后才会移动。

**保存会修改初始姿态配置。** 在窗口中按P，把Control目标保存为下次初值；如果希望保存实际关节位置：

```bash
.venv/bin/python preview/scripts/preview_pose.py --save-pose-source qpos
```

仍需按P保存；`--save-initial-pose` 则在关闭窗口时保存。写入文件：`tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py`；备份在 `preview/outputs/initial_pose_backups/`。

### 预览已冻结的MJLab物理场景并手动控制

```bash
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --scene training_mjlab/outputs/scene_shadow_40hz400hz \
  --contact-controls --camera-view front --camera-framing full
```

去掉策略参数后可使用Control滑块；该入口不写回初始姿态文件。GPU接触面板保存到 `preview/outputs/contact_settings/` 实验文件，不覆盖冻结场景或源配置。源姿态保存后，已有冻结场景和已训练策略不会自动变化。

## 7. 修改姿态或接触配置后，导出新的训练场景和任务

只有需要让**新训练**使用已保存姿态／接触配置时，才执行下面两步；目录名可改，但必须是新目录。保留旧场景供旧检查点回放。

```bash
# 1. 用源模型与当前初值导出场景，重力补偿设置与当前主线相同
MUJOCO_GL=egl .venv/bin/python scripts/export_mjlab_scene.py \
  --no-gravity-compensation --output training_mjlab/outputs/scene_new_pose

# 2. 用同一源模型导出带指法的小星星任务
MUJOCO_GL=egl training/.venv/bin/python scripts/export_mjlab_task.py \
  --scene training_mjlab/outputs/scene_new_pose \
  --midi TwinkleTwinkleRousseau --lookahead 20 \
  --output training_mjlab/outputs/task_twinkle_new_pose
```

之后训练／评价／回放均传这一对新 `--scene` 和 `--task`。不能将旧检查点强行套用修改后的场景。导出校验失败时先处理报错，不跳过合同核验。

## 8. 独立评价和继续训练

### 不生成视频，只评价整曲

```bash
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.evaluate \
  --checkpoint "$V32_CHECKPOINT" --episodes 1 \
  --output preview/outputs/my_step18000_evaluation.json
```

评价JSON同名会覆盖。固定曲目／姿态重复评价不是多seed泛化验证。

### 同阶段继续学习（warm restart）

```bash
training_mjlab/.venv/bin/python -u -m tsensei_v3_2_pianist.rl_torch.train \
  --scene training_mjlab/outputs/scene \
  --task training_mjlab/outputs/task_twinkle --stage 1 \
  --warm-start "$V32_CHECKPOINT" \
  --num-envs 256 --vector-steps 19532 --batch-size 4096 --gradient-steps 32 \
  --output-dir training_mjlab/outputs/my_stage1_restart
```

上例明确使用第1节旧检查点的旧场景／任务；新40 Hz检查点须将两项改为对应的新包，不能跨观测维数warm-start。

保留learner／optimizer等状态，但重建replay、物理及乐谱状态，重新计数本轮预算并重新预热；**不是精确断点续训**。Stage2目前只实现参数档位／网络转移，完整奖励课程尚待决定，详见 [Torch SAC说明](torch_sac_acceptance.md)。

## 9. 常见定位

| 现象 | 检查 |
| --- | --- |
| 找不到模块／脚本 | 是否在独立项目根目录；是否选对上表中的Python环境 |
| 窗口打不开 | 是否在桌面终端，有DISPLAY；仅要视频可用离屏导出 |
| 输出目录已存在 | 改新目录名；自动选优导出默认附时间戳 |
| scene/task contract mismatch | 必须使用训练当时对应场景／任务，不要忽略哈希校验 |
| 调了初始姿态但GPU预览没变 | GPU读取冻结包，按第7节导出新的场景／任务 |
| 音频依赖缺失 | 第3节setup脚本；不要把目标MIDI当作实际弹奏音轨 |
| 复播F1与日志略有差异 | V32-SAC-03尚待定位；换镜头优先复用保存的轨迹 |

更详细说明：[预览](../preview/README.md)、[GPU任务](mjlab_task_acceptance.md)、[Torch SAC](torch_sac_acceptance.md)、[修改日志](change_log.md)。


## 10. 训练速度／开头碰撞定位

已对最新最佳step-18000定位：物理与SAC更新各约一半耗时；开头存在大幅位置目标跳变及手内自碰撞。详细证据、具体几何名称、接触时间／力及复查命令见 [训练耗时与开头运动诊断](training_speed_and_startup_diagnosis.md)。该诊断轮次没有修改训练／碰撞／奖励；后续加速与可选自碰撞惩罚见第11节。


## 11. 加速开关与待确认奖励（2026-09-18）

新训练默认开启多物理子步 CUDA graph、学习器编译与融合 Adam。已有第4节训练命令可直接使用；无需改 batch、更新次数或物理 dt。编译首次触发有等待时间。旧20 Hz配置下实测稳态吞吐约提高31%，物理段本身未明显加速；不是本次40/400 Hz的新测速；见[加速验收](acceleration_acceptance.md)。

```bash
# 保留现有奖励，显式开启三项加速；约5M transition。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --vector-steps 19532 --num-envs 256 \
  --physics-graph --compile-learner --fused-optimizer

# 仅检查候选自碰撞参数；不启动训练。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --self-collision-weight 0.2 --inspect-only
```

需要关闭时分别加 `--no-physics-graph`、`--no-compile-learner`、`--no-fused-optimizer`。录音／逐子步诊断自动使用逐步物理路径，仍采样全部子步。

[reward确认文档](reward_design_review.md)开头是 `rew.*` 总表。自碰撞惩罚已实现，全局默认 `--self-collision-weight 0`；2026-09-20用户要求新一轮启用，采用现有候选权重0.2。在需要此项的新训练命令中增加 `--self-collision-weight 0.2`；不能拿权重0的旧checkpoint直接warm-start成0.2。评估、窗口回放、有声视频会从checkpoint自动恢复权重，无需手工再传。

`metrics.jsonl` 新增 `reward_terms`，含启用时的 `self_collision_reward`。后续平滑、姿态、超速及正功率等全部公开公式已实现为可选项，见第13节。


## 12. 训练进度条

Torch SAC 训练现在**默认开启进度条**，原训练命令直接使用即可；可显式加 `--progress`，后台写日志时可用 `--no-progress` 关闭。

- 主进度以单环境样本 `sample` 计数：每完成一次向量步增加 `num_envs`，总数为 `vector_steps × num_envs`。
- 显示百分比、已运行时间、预计剩余时间、sample/s，以及 `vstep`（向量步）、`updates`、`replay` 和最近独立评价的 `eval_F1`。
- 初始化提示在物理场景创建前输出；首次学习器编译、评价和保存期间显示对应阶段，避免误以为卡住。
- 默认至少间隔1秒刷新，不因显示增加GPU张量读取／同步。显示写入stderr，`metrics.jsonl`仍保留原结构和分项奖励。
- ETA在预热／首次编译／周期评价后会波动。100%表示本轮样本预算已完成；末次评价或final.pt保存可能仍需少量时间。warm-start进度按本次新预算从0计数，updates保留检查点的累计值。

正在运行的旧Python进程不会动态加载修改；下次启动训练生效。


## 13. 自己选择论文奖励

[rew.总表及必填映射](reward_design_review.md)是奖励配置的主说明。全部20个配置项已有实现，每项的`enabled`／`weight`可分别编辑；未知映射按用户决定保留null，未填不能开启。

```bash
# 复制并编辑自己的完整配置；current保持当前五项组合。
cp configs/rewards/default_20261006.json configs/rewards/my_rewards.json

# 先检查配置，再启动训练。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --reward-config configs/rewards/my_rewards.json --inspect-only
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --reward-config configs/rewards/my_rewards.json --num-envs 256 --vector-steps 19532
```

还提供`paper_stage1.json`、`paper_stage2_canon.json`、`paper_stage2_other.json`：仅启用paper按键＋指法及总奖励乘dt，其他项默认关闭，权重已按表3准备。`--enable-reward 名称`／`--disable-reward 名称`可重复用于覆盖单项开关，但不会替你填映射。

新checkpoint保存完整奖励配置，评价／视频自动恢复。正常`--warm-start`要求配置一致；若明确转到不同Stage2奖励，使用`--stage 2 --stage2-from 检查点 --allow-reward-change --reward-config 新配置`，只继承网络、重新建立optimizer和replay。Stage2未知shaping须显式选择identity才可用公开正则公式，不声称完整复现未披露函数。


## 小星星前期前冲视频（2026-09-20）

已导出完整小星星run的首个step-1000（256000采样／32更新）：[有声视频](../preview/outputs/replays/twinkle_early_step1000_20260920/replay.mp4)／[开头2秒四分之一慢放](../preview/outputs/replays/twinkle_early_step1000_20260920/start_2s_quarter_speed.mp4)。[诊断说明](twinkle_early_startup_20260920.md)记录首条目标跳变、动作0映射中点、逐2.5ms接触先后和中性控制对照；没有修改运行中《月光》训练。


## 无机械臂小星星20/200 Hz（2026-09-23）

新增训练参数`--no-arms`，可自动从M6冻结包创建无臂副本；本轮使用新20/200 Hz冻结包，10步前瞻、1180维观测。完整导出／训练命令和本次PID见[本轮记录](training_noarms_20hz200hz_20260923.md)。


## 12 cm手根、49动作、TZ固定0（2026-09-24）

运行`bash scripts/train_twinkle_h12_no_tz.sh`，已配置`--no-arms --disable-wrist-tz`及20/200 Hz新包。完整配置、动作顺序、固定方式和验证限制见[说明](wrist_tz49_height12_20260924.md)。此次仅短程验收，正式训练需运行上述命令。


## 2026-09-24：同手自碰撞仅检测

用户随后要求取消同手接触反力，12cm无TZ训练包装脚本已增加`--self-collision-detection-only`；自碰撞惩罚0.2保留。预览重启后默认仅检测，可用`--no-self-collision-detection-only`恢复实体接触。见[说明与验收](self_collision_detection_only_20260924.md)。


## 后续覆盖：左右手也取消反力

2026-09-24用户进一步要求跨手碰撞只检测。当前预览默认`--hand-collision-detection-only`，同手＋跨手均无反力；12cm无TZ包装脚本同步使用该开关。旧仅同手开关语义保留。详见[当前说明](all_hand_detection_only_20260924.md)。


## 关闭TZ和Yaw的47维新训练

`bash scripts/train_twinkle_h12_no_tz_no_yaw.sh`。新增参数`--disable-wrist-yaw`与`--disable-wrist-tz`配合使用，左右Yaw固定为冻结场景各自初值；本轮均0.25rad。47维网络需要从头训练。见[实验记录](training_no_yaw47_20260924.md)。
