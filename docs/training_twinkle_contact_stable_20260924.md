# 小星星：修复接触振荡后的新训练（2026-09-24）

用户要求开启一轮新的小星星训练。本轮从头运行已配置的 5M 预算，未恢复旧策略。

## 启动记录

- 开始时间：2026-09-24 08:14:36（Asia/Shanghai）。
- 进程 PID：`3826442`，后台独立 session；退出当前终端不终止训练。PID 仅指本次启动。
- 输出：`training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436/`。
- 实时日志：同级 `sac-twinkle-contact-stable-20260924-081436.log`。
- 启动命令、PID、时间和路径：同级 `.launch.json`；实际配置见 run 内 `config.json`。

## 本轮配置

| 项目 | 采用值 |
| --- | --- |
| 曲目 | TwinkleTwinkleRousseau（带指法小星星，158 控制帧） |
| 预算 | 19,532 向量步 × 256 环境 = 5,000,192 条 transition |
| 时序 | 控制 20 Hz／物理 200 Hz，10 子步 |
| 手／机械臂 | V3.2 双手、手根 12 cm、无机械臂 |
| Wrist_TZ | 编译前删除两个关节和驱动器，49 动作、1180 维观测 |
| 动作映射 | mixed_v1 |
| 手指驱动 | armature=0.0002 kg·m²、Kp=1，沿用已验收的接触修复 |
| 接触 | 同手、左右手、手—琴键均有物理反力，两种 detection-only 均关闭 |
| 自碰撞奖励 | 权重 0.2；forearm 等原任务奖励保持 |
| SAC | 已确认的论文 Stage1 参数，seed=42，从头训练 |
| 网络／学习率 | Actor 256×2；Critic 256×3；Actor 3e-4，Critic/温度 1e-3 |
| gamma／初始 alpha | 0.85／0.2；49 动作对应目标熵 −24.5 |
| Replay／batch | GPU replay 262,144 条；batch=4096 |
| 预热／学习 | 前 300 向量步随机动作，第 1000 向量步开始 SAC，每步 32 次梯度更新 |
| 加速／显示 | 物理 CUDA graph、学习器编译、融合优化器、进度条均开启 |
| 评估／保存 | 每 1000 向量步独立整曲评估并保存 `step-*.pt`；结束保存 `final.pt` |

场景和任务使用 `training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/{scene,task}`。沿用这个冻结包的初始姿态、指法目标及乐谱规则，不随源预览后续保存而变化。详情见 [接触修复及适用范围](contact_stability_shadow_20260924.md)。

## 命令

本次实际启动命令（已在后台运行，勿重复启动）：

```bash
bash /home/houjue/pianist_v2/tsensei_v3_2_pianist/scripts/train_twinkle_h12_no_tz.sh \
  --stage 1 --num-envs 256 --seed 42 \
  --output-dir /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436
```

查看进度：

```bash
tail -f /home/houjue/pianist_v2/tsensei_v3_2_pianist/training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436.log
```

每次评估的 F1 写在 run 内 `metrics.jsonl` 的 `eval.f1`；最终耗时写在完成后生成的 `summary.json`。预热阶段进度条的 ETA 不代表加入 SAC 更新后的实际耗时。

## 启动验收与范围

启动前核对无其他 Torch SAC 训练进程；GPU 显存占用约 1.6 GiB。`--inspect-only` 确认预算、真实接触、无臂、TZ 物理移除及原奖励开关正确。运行配置确认 49 动作／1180 观测／目标熵 −24.5。

启动约 2 分钟时已完成 1410 向量步／360,960 条采样、13,152 次 SAC 更新，损失／温度数值有限，`step-1000.pt` 已落盘。首次整曲评估 158 帧完成，F1=0.08015475，未截断；属于训练初期。此后训练继续在后台运行，实时进度以日志为准。

本次仅启动新训练，不以早期 F1 判断收敛。保留 V32-04 驱动实机标定、V32-06 全范围碰撞验证、V32-SAC-03 CPU/GPU 接触及复播差异。


## 完成结果与最佳视频

本轮正常完成5,000,192采样、593,056次更新，耗时2860.57秒（47分41秒）。日志最佳step-19000 F1=0.673748，最终F1=0.669798。

[最佳视频](../preview/outputs/replays/twinkle_contact_stable_best19000_20260924/replay.mp4)：1280×720、20fps、8.95秒；前方正交全琴视角，真实物理触键事件合成音轨（98次note-on），完整158控制帧。回放F1=0.660579，与日志不同，保留V32-SAC-03。

```bash
MUJOCO_GL=egl training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py --checkpoint training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436/step-19000.pt --scene training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/scene --task training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/task --camera-view front --camera-framing full --audio --output preview/outputs/replays/twinkle_contact_stable_best19000_20260924
```

自动查找未找到嵌套冻结包，需如上显式传scene/task；脚本校验哈希通过。再次导出须换output目录。


用户已要求从最佳step-19000继续新增5M采样；新run和续训命令见[续训记录](training_twinkle_continue5m_20260924.md)。原run及检查点不覆盖。
