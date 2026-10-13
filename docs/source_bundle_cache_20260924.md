# 从源码构建并缓存训练快照（2026-09-24）

> 2026-09-24更新：新训练改用源码＋完整config.json重建，不保存持久化scene/task缓存；Actor256×3、温度1、gamma0.8、曲首准备1秒，指法保持原曲。见[source_config_training_20260924.md](source_config_training_20260924.md)。下文保留历史方案及验收。

新训练入口是 `scripts/train_from_source.py`。它读取本项目源码及本地复制的 assets，在独立进程中应用可选安装姿态，先导出 `scene`，再从同一源码导出 `task`，最后将两者交给原有 `rl_torch.train`。不需要手工修改源码，也不会覆盖已生成的训练快照。手部重力补偿始终开启；默认 20 Hz 控制、200 Hz 物理、0.5 秒（10 步）前瞻，默认乐谱为有指法的小星星。

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
# 只构建快照，不启动训练；重复命令显示 CACHE HIT
python3 scripts/train_from_source.py --height 0.10 --ty 0.04 --yaw 0.15 --build-only
# 47 动作小星星：本项目既有动作映射、接触奖励与 SAC 参数
bash scripts/train_twinkle_h10_ty04_yaw015_no_tz_no_yaw.sh --inspect-only
# 确定配置后，去掉 --inspect-only 才会真正启动训练
```

包装脚本 `train_twinkle_h10_ty04_yaw015_no_tz.sh` 与 `train_twinkle_h12_no_tz.sh` 默认 49 动作；各自附加 `_no_yaw.sh` 默认 47 动作。这些入口现在都调用源码缓存。所有 `rl_torch.train` 的参数可继续传给包装脚本，例如 `--num-envs`、`--vector-steps`、`--output-dir`、`--inspect-only`。姿态与时序可以用 `--height`（米）、`--ty`（米）、`--yaw`（弧度）、`--control-hz`、`--physics-hz` 覆盖；同名参数以最后一个为准。`--midi`、`--lookahead`、`--frame-stack`、`--reference-steps`、`--no-action-reward-observation` 和 `--disable-forearm-reward` 属于 task 构建参数。裸 `train_from_source.py` 未预设 47／49 动作或 `mixed_v1`；使用包装脚本可取得本轮小星星配置。2026-09-24 后的新缓存还包含[中立动作 reset 历史修复](reset_recovery_20260924.md)；旧冻结 task 不会被原地改写。

默认缓存位于 `training_mjlab/outputs/source_cache/<SHA-256>/{scene,task,build.json}`。键包含本地复制的包源码与 assets、两个导出脚本和启动脚本、导出环境版本、乐谱文件散列（如果提供文件）、姿态／时序／任务参数。`build.json` 记录这些输入及 scene/task manifest 散列。重复运行先校验已有缓存及其文件散列；损坏的缓存会报错，不会静默覆盖。并发构建持有文件锁，完成后原子发布快照。训练 `config.json` 仍记录 scene/task 散列和训练参数；当前不生成 `args.pkl`。默认缓存位于 `outputs` 内，因此新检查点可按散列找到对应快照。自定义 `--cache-dir` 若放在 `outputs` 以外，回放时需显式传 `--scene`／`--task`。

## 与上一份 h10 冻结 task 的区别

旧 `current_h10_ty04_yaw015_20hz200hz_gravcomp_20260924/task` 从更早的 task 派生，保留了旧指法目标点。新缓存**直接从当前源码导出**，因此启用用户已确定的拇指目标点：白键从 `+X` 端向 `−X` 2 cm，黑键 1.5 cm；新 task 也包含重新计算的 CPU 参考轨迹。实测两者 `score.npz` 字节相同、158 帧、1180 维、10 步前瞻；新旧 scene 的安装姿态、重力补偿、接触与驱动编译数组一致。新 task 的指法奖励目标变了，不能把新旧训练差异只归因于缓存方式；旧检查点的 scene/task 散列也不能直接用于新缓存的 `--warm-start`。旧运行未改、未重新启动。

## 已清理的旧快照

用户确认无需回放旧检查点后，删除以下八个无当前进程引用、无当前默认入口或必要自动化验收依赖的目录；旧训练 `config.json`、指标和检查点文件没有删除：

| 删除目录（相对 `training_mjlab/outputs/`） | 原因 |
| --- | --- |
| `scene_before_hard_contact_20260917`、`scene_hard_contact`、`scene_wrist25` | 历史接触／腕部中间快照；没有 task 或运行引用 |
| `scene_gravcomp_template_20260924` | 重力补偿派生时的临时模板，非训练输入 |
| `contact_h12_20hz200hz_20260924` | 旧未使用的 h12 scene/task 对 |
| `scene_clair_shadow40_pose_20260920`、`task_clair_de_lune_beginner_shadow40_fingered_v1` | 旧《月光》40 Hz scene/task；仅旧运行引用 |
| `m6_93mm_twinkle_20260923` | 历史 M6 小星星 scene/task；同散列的 M6 scene 在 `m6_93mm_clair_20260922_final/scene` 保留 |

旧 47 动作续训使用 `contact_stable_h12_20hz200hz_20260924/{scene,task}` 并已完成，因此该旧包继续保留用于复现。当前默认 h10 包、历史 h12 重力补偿包、以及现有测试／预览依赖的基线、40/400 Hz、无臂和 M6 验收包也保留。被清理包对应的旧检查点可能无法直接回放；这是用户本轮明确接受的范围。新 h12 包装脚本已改走源码缓存，并包含2026-09-24的中立 reset 动作历史修复。

**TODO(V32-CACHE-01)**：若以后要恢复被清理的旧检查点，必须找回散列完全一致的原始 scene/task；从当前源码重新导出不会产生旧散列。长期新训练的演奏质量仍需独立验收（TODO(V32-GRAV-01)、V32-SAC-03）。
