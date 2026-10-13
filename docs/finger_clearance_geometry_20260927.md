# 整指净空几何与动作目标投影

用户本轮允许几何约束和有界辅助惩罚，要求空闲手指整根保持琴面上方，并保留通用键盘工作区。这里记录几何模块、CPU 可行性及明确的不足；最终动态策略回放、训练选择和启动由本轮主文档记录。本模块不是论文自然姿态映射或论文子步插值复现。

## 真实模型与初始净空

检查当前 Shadow primitive profile 的 **40 个手指碰撞 capsule**，每指 4 个。包括中间链，不能仅用十个 fingertip site 代表整指；掌部不属于本次 finger 包络。capsule 的两个中心线端点加半径精确界定现有碰撞体，不代表 STL 所有可视细节或经过硬件标定。

键盘 rest 顶面：白键世界 Z=22.5 mm，黑键 Z=35 mm。origin 左右对称：

| 手指 | 全指最低处高于白键顶 | 高于全局黑键顶 |
|---|---:|---:|
| 拇指 | 11.819 mm | −0.681 mm |
| 食指 | 35.081 mm | 22.581 mm |
| 中指 | 30.102 mm | 17.602 mm |
| 无名指 | 25.797 mm | 13.297 mm |
| 小指 | 26.207 mm | 13.707 mm |

拇指最低处在黑键前方，其末节最低 X≈22.11 mm，黑键前缘 X=15 mm，故负的“全局黑顶差值”**不是初始穿透**。局部键盘表面规则下 origin 全十指合格。initial 整体 +20 mm，几何净空相应增加20 mm。

若需退回一个高于所有黑键顶加2.1 mm的姿态，CPU 约束求解在原关节范围内从origin得到两拇指的小调整，其余八指无需调整。该姿态通过现有几何计算得到，仅用作失败退路目标，没有改写 origin 或实际 qpos，也不把它登记为论文未知自然姿态。

## 模块与默认参数

`mjlab_env/finger_clearance.py`：

```json
{
  "schema": 1,
  "mode": "full_finger_clearance_v1",
  "clearance_m": 0.002,
  "contact_slack_m": 0.001,
  "projection_iterations": 6,
  "max_joint_correction_rad": 0.25,
  "active_command_mode": "strict_v1",
  "active_contact_slack_m": 0.005,
  "active_support_xy_margin_m": 0.002,
  "active_recovery_delta_rad": 0.2,
  "wrist_clearance_interlock": false
}
```

`canonical(None)` 为关闭；规范参数和完整 `guard.contract` 应随配置／checkpoint 保存。当前实现要求现有无臂47动作、TZ/Yaw物理移除及两个准备支撑节点。新旧执行合同不能静默互换。

运行接口：

```python
guard = FingerClearanceGuard(bundle, device=device, options=options)
applied, metrics = guard.filter(
    requested_physical_51,
    actual_qpos=qpos,
    mocap_pos=mocap_pos,
    assigned_keys=current_fingering_matrix,  # N×10×88
)
```

所有运行期计算为设备上的批量张量，不对每个环境调用 CPU IK。初始化时由 MuJoCo 编译模型推导串联变换、关节轴、capsule、琴键 rest/full-travel 平面及退路姿态。没有新的时序记忆或未来指法；调用者负责提供当前乐谱的指—键关系。

### 基础几何规则（`strict_v1`）

1. 整个空间都以白键顶+2 mm为基础，不因手指目前在键盘 XY 范围外就豁免。因此不能先在键盘外垂到键底再横向进入。
2. capsule XY包围范围与黑键 rest 占地重合时，要求高于黑键顶+2 mm。此包围检测偏保守，接近边界的违规不一定已有实体接触。
3. 当前有任务的手指也不整根豁免。仅该指**末节接触 capsule**的完整XY范围落在其当前指定琴键内，才允许该末节降到琴键完整按压行程的顶平面，预留1 mm软接触容差；其余三个中间链继续受净空约束。不能只把“当前活动手指”作为所有碰撞的豁免开关。
4. 请求目标按 origin 的最低支撑高度计算，防止准备下降消耗2 mm裕度；实际物理状态按当前真实支撑高度计算。准备 phase 本来就已可观测，没有加入隐藏的未来指法阶段。

串联 FK 覆盖每指三个腕关节和四个手指关节，保留同一 wrist body 上多关节的原始顺序。解析Jacobian根据最坏capsule端点和限制平面法线计算。每轮只修正每指4个目标角，最多6轮；失败时改用已验证抬起姿态。不会裁剪通用 TX 工作区，不改机械范围、实际 qpos/qvel或碰撞反力。

### 指标与奖励输入

- `finger_actual_deficit_m`：实际姿态每指最坏几何缺口。
- `finger_requested_deficit_m`：原请求目标在最低支撑高度的缺口。
- `finger_applied_deficit_m`：投影后目标的缺口。
- `finger_clearance_deficit_m`：严格模式为实际和原请求缺口最大值；`active_contact_v1` 的主动指只取实际危险缺口，详见下文，供有界辅助惩罚使用。
- `finger_projection_mask`／`finger_fallback_mask`：目标被改动／采用抬起退路。
- `finger_pose_reference_feasible`：原origin四角在当前腕位和角色下是否真的合格，用于允许或关闭姿态次项；不能直接假定参考姿态永远安全。

上述形状均为 N×10。scene/task 必须在实际物理步后刷新 actual 和 reference mask，奖励不能只使用步前状态。实际下发的完整51目标进入既有 action 观测；SAC replay保留47维原策略动作。

## CPU 验证

`tests/test_finger_clearance.py` 九项通过（单线程 CPU，11.807秒）：

- 明确启用、参数校验及None关闭。
- 原始／随机手指姿态、腕平移／转动及支撑抬升下，40capsule FK与原生 MuJoCo一致，最大容差8e−8 m。
- 净空Jacobian与有限差分一致。
- origin局部表面全指合格；错误指定键不豁免末端。
- 八组随机请求投影后目标缺口≤1.1e−6 m，腕目标、原输入、物理qpos不变；重复调用无隐藏状态。
- 已求得的左拇指白／黑键、左食指黑键完整行程端点在正确指法下合格，改为空闲或错键指派则拒绝豁免。
- `idle_only_probe` 消融只纠正空闲指，主动指原命令逐bit保持，同时保留完整实际／请求违规诊断；此模式不是主动指已受保护的结论。
- `active_contact_v1` 在真实支持位置允许 PD 原目标；原请求末端 FK 下穿超过1 cm时不因此产生主动指几何负罚，后物理反馈使用同一规则。
- 两世界独立验证：危险主动指恢复目标与当前实际每角之差≤0.2 rad；空闲指低垂时该手腕临时保持当前实际目标；原始 qpos 不变。origin 安全时，左 TX 请求0.3 m完整保留，没有变成曲目专属范围裁剪。

另做8个CPU独立按键端点：左右拇指白C4/C5、黑C♯4/C♯5、左右食／小指黑C♯。均满足guard且接近完整行程顶面，残差≤4.3e−9 m。3个 least_squares达到250次上限，虽已找到合格端点，不能声称所有数值优化收敛。这仅证明这些手指存在可用的白／黑键静态端点，不证明整曲、其他和弦或动态进出都可行。

`scripts/probe_clearance_keyboard_span.py`再将左右拇指白／黑端点沿原TX轴平移至全部88键，共176个独立静态姿态；162个位于该手原机械／驱动范围内（每手81），这162个均被`active_contact_v1`正确许可且缺口为0。将指派故意错开一八度时，162个均拒绝触键许可。左右候选的并集覆盖MIDI21–108全部88键，说明规则和允许区域不是只对当前曲目的几个键成立。此结果是独立单指静态端点覆盖，**不证明双手同时、各类和弦、边缘琴体碰撞或动态整键盘演奏可达**。

可重跑：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  training_mjlab/.venv/bin/python -m unittest tests.test_finger_clearance
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  training_mjlab/.venv/bin/python scripts/probe_active_pad_reachability.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  training_mjlab/.venv/bin/python scripts/probe_clearance_keyboard_span.py
```

## 已证实的不足：端点合格不等于路径合格

`scripts/probe_clearance_paths.py` 找到两类明确反例，代码和数据均保留，尚未因首轮端点测试通过就宣布解决卡键。

1. 左TY=.07 m时，TX从−.303805到−.250895 m，两个端点的拇指净空均合格，中间某处接近黑键包络欠2.681 mm。该中间姿态原生查询无实体接触，说明它首先是净空裕度越界。
2. 64组随机两端合格的姿态，在关节线性过渡中点有250/640个手指违反包络，最深24.63 mm。两端残差均低于1e−6 m。**这不是策略实际失败率**，但反证了“两个姿态都合格，所以中间一定合格”。

最小后续增强是对实际→候选的多个中间点做FK检查，沿已合格路径逐段推进；已处于违规状态时需要连续抬指恢复，不能因为一个远端退路姿态合格就直接称整条路径安全。主动按压→释放→横移也存在连续阶段，不能把当前指法切换当作物理手指瞬时抬起。

此外位置伺服有惯性、滞后和接触，即使采样的运动学路径合格也不构成动力学不越界保证。必须保留200 Hz实际净空／键底／键侧／错音审计；必要时再增加可解释的路径或物理反馈，不能直接写qpos掩盖违规。

### 第一轮动态反证与消融

主任务在同一best上测试严格目标约束：音乐F1从.98228降到.08333，键底子步50→0，但键侧／底合计77→406。同手碰撞也增加。applied目标FK几乎全部合格，实际控制末状态却在96帧违规，最大22.96 mm。这个候选不能开长期训练，静态验证不能替代动态检查。

在新轨迹的295个主动指帧中，258个被投影，平均每关节改变.2274 rad；原请求末节仅29帧符合严格assigned XY规则。对固定成功基线再检查，284个成功指定按键帧中265帧的PD目标FK需要低于fully-pressed顶面超过1 mm，目标中位低15.71 mm，而实际仅低1.73 mm；实际最深4.77 mm。这说明伺服的施力位置目标不能直接当作实际碰撞姿态。

因此加入显式 `active_command_mode="idle_only_probe"` 消融：只有空闲指受原目标投影，主动指力目标原样保留，违规仍记录。该消融F1=.94002、键底0、侧／底51子步，支持“严格主动目标封住正常下压力”是主要退化因素。它仍未达到父策略音乐能力，且主动指没有获得新保护，不作为全手安全已解决的证据。

### `active_contact_v1`：实际接近许可与连续恢复

通用模块仍默认 `strict_v1` 以保留旧实验合同，新候选配置显式选择 `active_contact_v1`。原 `idle_only_probe` 也保留用于消融，不静默改变历史实验。该候选只读取当前指法及当前真实姿态，未引入未来指法隐藏状态。

- **空闲指**保持已验证的原投影数学和2 mm表面裕度。
- **主动指触键许可**以真实末节capsule与指定键XY支持区重合为准，支持区外扩2 mm；取消“整个末节必须完全装入键宽”的原目标限制。实际末端不得低于完整行程顶面超过5 mm，整根手指不得低于其重叠琴键rest底面加2 mm。满足这些条件时保留策略原位置目标，使位置伺服可以正常传力。
- **尚未接近指定键**时，修正后的主动指目标限制在当前实际四角附近，每角最多0.2 rad。**实际已危险**时，从当前真实角做抬指梯度恢复，并保留同样0.2 rad总范围，不跳到远端退路姿态。这个限制是每个控制帧相对于实际角的局部目标范围，不是实际速度硬上限。
- 5 mm是本轮284个成功触键帧的实测候选余量：实际低于full-travel顶面的中位为1.731 mm，最深4.770 mm；3 mm会拒绝43个成功帧，4 mm会拒绝8个。不是论文公开参数或硬件标定，也不允许把侧面／键底接触笼统称为软顶面接触。

新可选 `wrist_clearance_interlock=true`：同一手至少一个当前空闲指实际低于其局部键面，并且腕请求与实际有差异时，暂将该手TX/TY/Pitch目标保持在当前真实腕角／位置，先抬指再移动。默认关闭，保持全局TX工作区与机械范围；没有对qpos、qvel或惯性作瞬时清零。它以空闲指**真的低于表面**为门槛，而非仅损失2 mm预留间隙。解除门槛后即允许原腕目标，不引入新的状态计时器。

新增指标：`finger_active_force_permitted`、`finger_assigned_support`、`finger_key_bottom_deficit_m`、`finger_actual_pad_plane_m`、`finger_recovery_mask`（N×10），以及 `wrist_clearance_interlock_mask`（N×2）。`finger_applied_deficit_m`仍记录严格目标FK，因此正常主动施力也可能有较大值，不能再将其直接计作投影失败或真实穿透。

实际反馈必须使用：

```python
actual = guard.evaluate(actual_physical_51, actual_lift, current_assignment)
cost_deficit = guard.reward_deficit(
    actual, requested_metrics["finger_requested_deficit_m"], current_assignment)
```

在 `active_contact_v1` 中，空闲指取实际与原请求缺口的最大值，主动指只计实际危险缺口。原主动请求FK仍完整保留供诊断，不能在scene的后物理刷新中再次无条件取max，把正常PD下压力又罚回来。当前scene已通过上述helper刷新，reference可行性也按后物理真实腕位置重新计算。

首轮同best实际probe的该模式音乐F1=.91534、键底0、侧／底96、非末节触键9；低于idle-only的.94002及父策略.98228，说明保护仍明显改变策略行为。尚未把这次无训练迁移当作效果完成；下一步是对有／无wrist interlock短训，比较学会新控制后的音乐、实际200 Hz接触与吞吐。

### 局部路径草案，尚未进入正式模块

`scripts/probe_local_clearance_paths.py` 在实际角附近每轴±.2 rad内搜索，使用8个路径点及安全前缀，再用64个点复核同一随机反例集。端点方案366/640条指路径违规，局部草案降为7/640；最大缺口24.98→.755 mm。不过113指选择不移动，可能停滞，且有限采样仍漏黑键边界。对377个原已违规的随机手指，用actual为起点的连续梯度恢复有374个改善、130个一次达到净空，平均缺口17.59→7.12 mm；64点复核未出现更坏缺口。这不是动力学证明，也未验证移动腕或音乐F1，不能直接替换正式控制器。

## 资料与产物

- `preview/outputs/diagnostics/finger_clearance_20260927/geometry.json`
- `preview/outputs/diagnostics/finger_clearance_20260927/active_pad_reachability.json`
- `preview/outputs/diagnostics/finger_clearance_20260927/path_counterexamples.json`
- `preview/outputs/diagnostics/finger_clearance_20260927/keyboard_span.json`
- `scripts/probe_finger_clearance.py`、`scripts/probe_active_pad_reachability.py`、`scripts/probe_clearance_paths.py`
- 参考当前 `assets/collision_proxies.json`、源模型编译的关节／geom／琴键，`mjlab_env/{bundle,preparation,scene,task}.py`，安装版本 MuJoCo/Warp 的 `kinematics`/`jac` 接口，以及 `docs/references/biomimetics_2026_610_notes.md`。

TODO：动态旧best的音乐能力保留、主动按压进出与角色切换、黑键边界、路径可行性、实际惯性越界，以及规模吞吐；不得把本模块的当前命令约束称为严格的物理避障保证。
