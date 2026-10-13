# 当前手部姿态与重力补偿（2026-09-24）

## 新默认配置

用户指定左右手根安装高度 **0.10 m**、腕部 TY 初值 **0.04 m**、Yaw 初值 **0.15 rad**；手部重力补偿在当前源码预览、新场景导出和新训练包中保持开启。控制／物理仍为 20/200 Hz，原有接触参数、手指驱动、`mixed_v1`、自碰撞奖励 0.2、SAC 参数和小星星乐谱不因这次姿态修改改变。

物理场景和任务在 `training_mjlab/outputs/current_h10_ty04_yaw015_20hz200hz_gravcomp_20260924/{scene,task}`。`mjlab_env.paths.CURRENT_GRAVCOMP_*`、无检查点 GPU 预览和 Torch 训练默认指向这对新包。新 `scripts/train_twinkle_h10_ty04_yaw015_no_tz.sh` 和 `..._no_yaw.sh` 分别提供 49／47 动作入口；历史 `train_twinkle_h12_no_tz*.sh` 仍指向旧 h12 包。**本轮没有启动新长期训练，也没有修改、停止或重启正在使用旧冻结包的训练进程。**

源姿态预览 `preview/scripts/preview_pose.py` 和场景导出 CLI 固定传 `gravity_compensation=True`，不再提供关闭补偿的命令行开关；裸 `suite.load()` 也默认开启。旧冻结场景本身保留原散列，旧检查点仍须按原场景回放。`mjlab_env.bundle.export_scene(..., gravity_compensation=False)` 保留低层历史导出能力，不是当前 CLI 默认。

## 任务合同与生成

本节记录当时的冻结包生成方式。后续训练包装脚本已统一切到 `scripts/train_from_source.py` 缓存入口，并在2026-09-24加入中立 reset 动作历史合同；旧派生包继续用于历史检查点，详见[源码缓存说明](source_bundle_cache_20260924.md)和[reset修复](reset_recovery_20260924.md)。

直接运行 `scripts/export_mjlab_task.py` 会按当前源码额外启用拇指指法目标点规则；这超出了本次姿态修改。初次直接导出的候选已保存在同级 `task_source_thumb_candidate/`，**不是新训练默认 task**。若后续希望使用该规则，应单独比较奖励差异。

```bash
cd /home/houjue/pianist_v2/tsensei_v3_2_pianist
MUJOCO_GL=egl .venv/bin/python preview/scripts/preview_pose.py --headless --no-arms --output preview/outputs/pose_h10_gravcomp.png
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py --paused
bash scripts/train_twinkle_h10_ty04_yaw015_no_tz_no_yaw.sh --inspect-only
```

## 验证和剩余区别

- 新冻结 scene 实测 20/200 Hz、左右手根 z=0.10 m、TY=0.04 m、Yaw=0.15 rad、44 个手部 body 的 `gravcomp=1`。47 动作结构锁止后 `nq=134, nu=46`，补偿 body 仍为 44。
- 新 task 加载成功，158 帧／1180 维；与旧 task 相比，除 scene 散列、派生说明和旧 CPU 参考不可用的限制外，奖励及观测合同相同。无拇指专用目标点覆盖。
- 源姿态预览 EGL 离屏渲染通过，重力补偿 on、20/200 Hz、Shadow 接触参数 `MATCH`。新 47 动作包装脚本 `--inspect-only` 通过。回归测试修正了 TY 非零时应相对初值施加 1 mm 扰动的问题；完整回归和长期效果以本轮日志为准。
- 源预览默认可显示仅运动学 M6 双臂；新训练包无臂。源预览保留完整腕自由度，新 49／47 动作训练在加载时分别物理移除 TZ 或 TZ＋Yaw。这是入口用途差异，不是姿态或重力补偿差异。

**TODO(V32-GRAV-01)**：新姿态的长期训练、接触稳定性及回放效果尚未验收。V32-04 驱动标定、V32-06 碰撞几何和 V32-SAC-03 CPU/GPU／回放差异继续保留。

## 后续训练入口变更（2026-09-24）

上述 h10 冻结包及派生 task 仍在本机保留，历史内容记录其生成时状态。h10 的 49／47 动作包装脚本现改用[源码构建与缓存入口](source_bundle_cache_20260924.md)，并从当前源码直接导出 task。新 task 含拇指目标点规则，因此与本页记录的旧派生 task 奖励目标不同；没有将旧 checkpoint 或正在运行的旧训练切到新包。
