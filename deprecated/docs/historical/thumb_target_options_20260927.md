# 拇指目标参数、模型结构与预览差异

本轮开始于 2026-09-27，参数决定及实现于 2026-09-28。用户已确认：**保持当前 Shadow 统一目标，只把拇指距离做成按需参数**。不默认恢复白键 20 mm／黑键 15 mm，也不修改关节、驱动、姿态或指法。

## 1. 目标到底在哪里

当前 `run.sh` 选择 `shadow_v1`。未传距离时，十指都用：

- `target_x = key_geom_world_x + 0.35 × key_geom_half_length`
- `target_y = key_geom_world_y`
- `target_z = key_geom_world_z + 0.5 × key_geom_half_height`

此目标是指法距离奖励的参考点，不是强制接触点，也不直接改变琴键识别。它沿世界坐标计算，保留原公式。白键半长 75 mm 时，X 偏移为 26.25 mm。

历史 `tsensei_v1` 默认仅拇指另设：从琴键世界 `+X` 端朝 `−X`，白键 20 mm、黑键 15 mm。白键目标相对中心为 55 mm，比 Shadow 目标向 `+X` 偏 28.75 mm；黑键半长 45 mm 时偏差为 14.25 mm。当前训练此前没有启用这一历史拇指偏移。

论文 p.11 公开指法距离成功区间 1 cm 和 margin 10 cm，但没有给出键面目标点；这些距离属于已获用户决定的 TsenseI 几何设置，不是论文复现，也未擅自修改奖励公式或系数。

## 2. 参数与使用

```bash
# 当前默认：Shadow 十指统一目标
bash run.sh --thumb-target-mode profile

# 只在本次实验使用曾经的拇指目标；单位是米
bash run.sh --thumb-white-target-distance 0.020 --thumb-black-target-distance 0.015

# 可独立覆盖某种键，另一种保留所选 task profile 的定义
bash run.sh --thumb-white-target-distance 0.025

# 不论 task profile 是什么，显式使用十指统一目标
bash run.sh --thumb-target-mode uniform
```

`--thumb-target-mode` 默认 `profile`；模式为 `uniform` 时不能同时传距离，防止互相矛盾。距离必须为有限非负数，且不能超过对应键长。只有 R1／L1 的 X 目标会变；Y、Z、其他八指、真实碰撞和全部机械范围保持。

参数由 `scripts/train_from_source.py` → `source_runtime.make_recipe()` → CPU 任务／task 导出 → GPU 目标偏移计算贯通。配置完整保存到 `config.json` 的 `source_runtime.environment.thumb_target`，task manifest 使用 schema 2 记录原始选项及解析后的几何。加载时校验选项与 CPU 环境合同一致。旧 schema 1 或缺参数的旧 recipe 保留所属 profile 的历史定义；旧源码指纹仍须用对应源码归档重建，不能静默套用当前源码。

`make_recipe(**saved_environment)` 也接受已保存的 canonical `thumb_target` 字典，原有六个诊断脚本可继续重建。如果同时传 canonical 字典和非默认平铺 CLI 选项则报错，不静默选择一个。未知关键字继续被函数签名拒绝。

`scripts/launch_clearance_training.py` 显式继承父模型的完整拇指选项，并拒绝尚未处理的未知 source 字段，避免将 JSON 字典误转为未知命令行参数。

## 3. 修正了一处目标绘图错误

`preview/scripts/draw_fingering_reward_targets.py` 以前读了 task manifest，却调用 `target_position()` 时没有传入 `task_profile`，因此即便实际任务为 `shadow_v1`，也会按默认 `tsensei_v1` 画拇指目标。新代码从 manifest 校验并传递 profile 和完整距离选项，使画出的目标与训练一致。

这只能证明该独立目标绘图脚本存在错画路径，不能仅凭这一点断言用户截图或当前漏键由它造成。它不改训练物理轨迹。

## 4. 拇指结构证据与限制设计含义

来源为包内 `assets/urdf/daxian_left.urdf`／`daxian_right.urdf`，右手是镜像派生。角度单位 rad，以下轴是各关节的局部轴，非世界轴。

| 拇指关节 | 左／右局部轴 | 机械范围 |
| --- | --- | --- |
| Thumb_Rot | +Z／−Z | 0～2.09 |
| Thumb_Rot2 | +X／−X | 0～1.74 |
| Thumb_MCP | +Y／+Y | 0～1.57 |
| Thumb_PIP | +Y／+Y | 0～1.57 |

两处基部转轴加 MCP、PIP 构成四个独立串联关节；base 与 MCP 的连接有非零固定旋转。改变 Rot、Rot2 会同时改变末端位置和弯曲方向，因此不能把某一个角度简单当作世界竖直下压量。拇指没有 DIP。其他四指则是 MCP、侧摆 swing、PIP、DIP；swing 为 ±0.35，屈曲机械上限均 1.57，当前策略对四指 MCP 另有 1.2 的目标上限。

拇指 reward site 挂在末端 PIP，局部坐标约 `[-0.0041, 0, 0.0166] m`，来自原 STL 的局部 +Z 极值面中点。现用 capsule 特意保留该极点：末轴点 Z=0.0146 m、半径 0.002 m，因此没有凭空把 site 移离碰撞末端。但实际触键可能发生在 capsule 侧面或其他指节；倾角变化后，局部 +Z 极点并不总是世界最低的接触点。拇指末端拟合宽仅 4 mm，更需要同时检查位置、指腹朝向和接触位置，不能只用单点距离或统一 PIP 硬上限代替。

当前 `shadow_v1` 所有手指采用 Kp=1、阻尼 0.05、armature=0.0002、摩擦损失 0.01、力矩上限 0.6 Nm。URDF 的 effort/velocity 原值都为 0，这些数值是仿真暂定值，不是厂家标定。相关 TODO(V32-04/05/07) 保留。

## 5. 手动预览可达为何不等于训练能稳定按到

裸 `preview/scripts/preview_pose.py` 调用 `suite.load()`，默认 `tsensei_v1` 任务和 `legacy_v1` 驱动；它没有结构性移除 TZ/Yaw，也没有训练的动作净空投影／腕目标限速。当前 47 动作训练则移除了 TZ/Yaw，并使用 Shadow 驱动。

| 设置 | 裸源姿态预览默认 | 当前 Shadow 训练 |
| --- | --- | --- |
| 腕部 TZ／Yaw | 创建，可手动调节 | 关节和驱动器已移除 |
| TX／TY 额外 armature | 无 Shadow 3 kg 设置 | 各 3 kg，临界阻尼 |
| Pitch Kp／阻尼 | 50／3 | 8／0.5 |
| 手指 frictionloss | 默认 0 | 0.01 |
| 指法目标 profile | tsensei_v1 | shadow_v1 |
| 净空／目标限速 | 不经 GPU 策略变换 | 按本次训练配置选择 |

碰撞 proxy 在当前源码裸预览默认同样为 `shadow_primitive_v1`，不能笼统说所有几何都不同。环境变量可覆盖驱动／碰撞选择，上表描述未额外设置环境变量时的默认。当前截图来自检查点回放，按对应检查点合同重建，不能把它与手调源预览混为一谈。

建议后续用相同的 47 动作模型、驱动和碰撞来记录用户能按成功的短示教轨迹，再检查哪些实际接触/关节组合有效。优先研究“接近目标有效触键区域→指腹落键→键行程成功”的分阶段、有界辅助项，而不是凭一个曲子的失败收紧所有拇指关节。尚未用本轮代码替换奖励公式；方案取舍见本轮主分析报告。

## 6. 验证与未解决项

- Shared target／Shadow 历史 profile 单测：12 项、16 个 subtests 通过。
- 距离覆盖白／黑与两 profile、非拇指不变、部分覆盖、uniform、非法距离、manifest 篡改均有测试。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 training_mjlab/.venv/bin/python -m pytest -q tests/test_fingering_targets.py tests/test_shadow_task_profile.py tests/test_clearance_launcher.py tests/test_source_runtime.py`：28 项、16 个 subtests 通过（63.68 秒），包括真实 CPU 源码重建、白黑 88 键目标、旧缺字段 recipe 回退及父 recipe 继承；不启动训练、不占用 GPU。
- 根任务实际跑 `run.sh --inspect-only` 时发现系统 Python 没有 NumPy，新增选项验证的顶层 NumPy 导入破坏了原 stdlib 启动入口。已改为 `math.isfinite`，仅 `target_position()` 几何计算时局部导入 NumPy。新增 `python -S` 无 site-packages 回归；修复后相关 19 项＋16 subtests 通过，实际系统 `python3 scripts/train_from_source.py --task-profile shadow_v1 --thumb-white-target-distance .02 --thumb-black-target-distance .015 --build-only --no-progress` 构建成功，日志位于 `preview/outputs/diagnostics/thumb_target_system_build_20260928.log`。
- 补齐 canonical 字典重建接线后，参数／launcher／真实源码 roundtrip 与旧 recipe 回退相关 20 项＋16 subtests 通过（25.20 秒）。
- TODO(V32-07)：真实硬件指腹位置／形变及实际触键面的标定尚未完成。
- TODO(V32-QUALITY-01)：目标参数化不证明 F1 提升；是否改距离应以相同模型、相同奖励和随机种子的对照结果选择，当前默认保留统一目标。

## 7. L1:C4＋L5:C3 同时触键的 CPU 静态可达性

新增 `scripts/probe_thumb_pinky_chord.py`，从当前训练父配置重建 47 动作模型，优化左腕 TX/TY/Pitch、拇指四轴、小指四轴，共 11 个关节。其余手指保持 origin。使用原机械限位并保留小指 MCP 软件目标上限 1.2 rad，TZ/Yaw 仍移除。与父配置的 origin joint values、诊断模型的全部 geom sizes、joint ranges、actuator gains 数值比较，最大误差均为 0。

该测试**预先将 C4、C3 琴键设到全行程角度**，只检验有没有一个可无穿透地接触这两块已压下键面的手姿态。它没有用手施力把键按下，没有推进动力学。

12 个起点的 least-squares 搜索找到两个精确候选。最佳候选：

| 检查 | 左拇指 C4 | 左小指 C3 |
| --- | ---: | ---: |
| 指尖 XY 到当前统一目标 XY 误差 | <0.000001 mm | <0.000001 mm |
| capsule 表面到全按下键顶面的误差 | <0.000004 mm | <0.000004 mm |
| 整指到指定键底的最小余量 | 22.5 mm | 22.5 mm |
| 整个末端 capsule 的 XY 落在指定键内 | 是 | 是 |
| 原 active-contact 净空许可 | 是 | 是 |
| 指尖到全按下状态实际奖励点距离 | 4.63 mm | 5.13 mm |

最佳候选无 MuJoCo 静态左手相关穿透接触（末端与目标键从外侧近似切触）；左食／中／无名指的净空亏损都是 0。模型原有相邻体／装配碰撞排除规则继续有效，不代表验证了硬件或所有未启用碰撞对。

| 关节 | 静态候选值 |
| --- | ---: |
| L_Wrist_TX | −0.000419 m |
| L_Wrist_TY | 0.073656 m |
| L_Wrist_Pitch | 0.198726 rad |
| L_Thumb_Rot | 0.734278 rad |
| L_Thumb_Rot2 | 1.000666 rad |
| L_Thumb_MCP | 0.073915 rad |
| L_Thumb_PIP | 0.724059 rad |
| L_Pinky_MCP | 0.906119 rad |
| L_Pinky_SW | −0.210517 rad |
| L_Pinky_PIP | 0.941880 rad |
| L_Pinky_DIP | 0.615174 rad |

这给出具体证据：**该八度双键的几何端点在当前 47 动作模型中可达**，不需要扩大拇指 PIP 上限。候选更依赖拇指基部复合旋转和腕姿态，Pitch 已接近现有 +0.2 上限。相对 origin 的腕目标变化为 TX −0.419 mm、TY +33.656 mm、Pitch +0.198726 rad，均不超过旧限速每帧允许的 60 mm／0.2 rad；旧限速不会禁止从 origin 在一个控制帧下发这组腕目标，但实际机构不一定能一帧到位。

没有验证到达这组姿态的安全路径、键反力下的力矩余量、200 ms 音符内的响应、连续切换或 F1。不能把候选角直接写成新 origin，也不能把上述静态误差当物理仿真精度保证。

运行命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
training_mjlab/.venv/bin/python scripts/probe_thumb_pinky_chord.py \
  --run-config training_mjlab/outputs/twinkle_generic_clearance_finetune_20260927-20260927-093425-663203/config.json \
  --audit-dir preview/outputs/diagnostics/thumb_options_audit_20260928 \
  --starts 12 --max-nfev 1500 \
  --output preview/outputs/diagnostics/thumb_options_audit_20260928/thumb_pinky_static_ik.json
```

约 10.44 秒完成 12 组优化（不含源码构建）。JSON 保存全部候选角、qpos、残差、实际 capsule/键底检查和碰撞明细；同名 NPZ 保存最佳候选姿态，`.log` 保存执行结果。不启动训练或使用 GPU。
