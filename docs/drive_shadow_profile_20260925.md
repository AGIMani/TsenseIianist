# Shadow 参照驱动配置（2026-09-25；当前平移参数更新于2026-10-11）

## 最终实施

`assets/drive_settings.json` 提供 `legacy_v1` 与 `shadow_v1`。包内默认仍为 `legacy_v1`；新方案由 `TSENSEI_V32_DRIVE_PROFILE=shadow_v1` 显式启用。入口保存的 source recipe 决定训练／回放采用哪个方案。

| 项目 | legacy_v1 | shadow_v1 |
| --- | ---: | ---: |
| V3.2 当前连杆质量 | 每手0.952 kg | 原值 |
| TX/TY 额外平移等效惯量 | 0 | 每自由度 armature=0.02 kg |
| TX/TY Kp | 300 N/m | 300 N/m |
| TX/TY damping | 30 N·s/m | `2√(Kp × dof_M0)`，当前34.15259873 N·s/m |
| Pitch Kp／damping | 50／3 | 8／0.5 |
| 手指Kp／damping／armature | 1／0.05／0.0002 | 相同 |
| 手指frictionloss | 0 | 0.01 N·m |
| 手指驱动力矩上限 | ±0.6 N·m | 原值 |
| TX/TY／Pitch 驱动力上限 | ±50 N／±5 N·m | TX/TY ±15.32 N；Pitch ±5 N·m |

TX和TY是正交平移轴。当前每个平移自由度加入0.02 kg `armature`，仅表示很小的附加等效质量；它在平移广义惯量中体现，**不增加物理刚体或碰撞几何，也不增加Pitch转动惯量**。当前A3.2手部URDF总质量为0.952 kg，因此有效平移惯量参考值为 `M0=0.952+0.02=0.972 kg`，按 `2√(Kp×M0)` 得到34.15259873 N·s/m临界阻尼。TX/TY力限同步从±50 N缩放为±15.32 N，保持旧参数在总有效质量上的最大加速度包络；这不是厂家力矩标定值。该临界阻尼只针对单坐标线性模型，不能保证整个接触系统不振荡。

未修改关节轴、关节范围、策略映射、初始关节角／安装位置。最终配置没有新增名为 `virtual_forearm_inertia` 的物体；原URDF质量保持。未启用的TZ/Yaw仍由训练入口结构移除；它们的原驱动参数未在本profile内改写。此配置没有加入目标插值或速度限幅。

## 参考依据与坐标限制

实际读取的本地参考：

- `/home/houjue/DaxianPianist/robopianist/robopianist/models/hands/third_party/shadow_hand/right_hand.xml`：forearm质量3 kg，局部COM `[0,0,0.09]`，主惯量 `[0.0138,0.0138,0.00744]`；WRJ1 Kp8、damping0.5；手指摩擦、阻尼和armature默认。
- `/home/houjue/pianist_v2/robopianist/robopianist/models/hands/shadow_hand.py::_add_dofs`：TX/TY Kp300，按编译 `dof_M0` 计算临界阻尼。
- 安装环境中的 `mujoco_utils/physics_utils.py::get_critical_damping_from_stiffness`：计算方法。

参考路径只用于阅读，本项目执行和复制使用不依赖它们。

Shadow默认安装坐标中local X→world−Y、local Z→world−X；V3.2安装坐标中local X→world−Z、local Z→world−X。因此V3.2的Pitch与Shadow WRJ1**不是相同的世界转轴**。这里保留用户的V3.2轴约定，只取WRJ1的增益／阻尼值作为工程参照。真实Shadow编译时WRJ1的 `dof_M0=0.00654617 kg·m²`，下游质量约0.694 kg，3 kg前臂位于该腕关节上游。

JSON中的 `shadow_forearm_reference_only` 仅记录参考原值；COM和惯性张量不作为V3.2新增下游刚体参数使用。当前方案是虚拟平台动力学适配，不是V3.2硬件标定或Shadow形态的完整复现。

## 已拒绝的下游质量方案

最初试验把固定3 kg质量挂在现有V3.2腕root下游。它随Pitch转动，使Pitch `dof_M0` 增至约0.038765 kg·m²。150 mm TX阶跃虽降低平移峰速，却把伴随Pitch峰速从0.582提高到2.273 rad/s；这是错误的上／下游耦合，最终方案已删除该物体并改为平移armature。

拒绝方案源码和测量保存于 `preview/outputs/diagnostics/shadow_sync_20260925/rejected_downstream_platform/`、`*_downstream_rejected.json`。`drive_upstream_probe.json` 是临时XML干预记录，不应当作最终源码的精度验收；最终源码复验见下方两个JSON。

## 验证方法与范围

最终源码验证脚本 `preview/outputs/diagnostics/shadow_sync_20260925/validate_drive.py`：

1. legacy／shadow两套驱动，分别配原CAD mesh和新primitive碰撞形状，组成2×2对照。
2. 编译后逐项断言驱动切换不改变关节范围、ctrl范围、初始qpos或初始ctrl。
3. 验证平移阻尼按包含armature的 `dof_M0` 得到；拒绝未知profile名字。
4. 中立目标保持2秒、左TX 150 mm阶跃4秒、旧首个物理目标保持2秒、旧47动作模型的完整8.9秒**物理ctrl**开环压力测试。

另外以 `scripts/validate_shadow_collision_proxies.py` 检查同手、跨手和手—琴键三类接触，各固定目标20秒，逐物理步记录末2秒稳定性，并检查真实MuJoCo接触力。未用策略输出坐标直接替代物理ctrl，未关闭碰撞反力。

输出：

- `preview/outputs/diagnostics/shadow_sync_20260925/drive_comparison_final.json`
- `preview/outputs/diagnostics/shadow_sync_20260925/collision_with_shadow_drive_final.json`

最终源码在新primitive碰撞下：150 mm TX阶跃的平移峰速从1.27533降至0.532330 m/s，伴随Pitch峰速从0.582228降至0.269148 rad/s。三类20秒固定目标的末2秒手指速度RMS分别为同手0.0001453、跨手0.0016846、手—琴键0.0000140 rad/s；初始接触数量为0。它们是上述特定动作的验收值，不是速度硬上限。

旧物理ctrl整曲压力测试在新碰撞下的手—琴键峰值：legacy驱动422.47 N，被拒绝的下游质量方案339.27 N，最终平移armature方案59.98 N；新旧动力学会使轨迹及接触组合分叉，这个比较不能证明所有轨迹都按同一比例降低接触力。59.98 N仍需作为冲击压力继续关注，不能写成“已经没有爆冲”。

同一旧ctrl压力测试中，新碰撞配最终驱动的Pitch峰速3.7919 rad/s，仍高于新碰撞配legacy驱动的2.2098 rad/s；这说明本方案并非每个关节、任意指令下都更慢。单独将新驱动配旧CAD mesh时还出现484.14 N峰值，因此不应把联合配置验收拆成“仅改驱动即可解决”。应由新策略闭环回放继续验证动作及动态碰撞。

`MUJOCO_GL=egl training/.venv/bin/python -m unittest tests.test_drive_profiles -q` 实跑5项通过，覆盖默认legacy相等、原连杆质量／Pitch惯量／初态／范围不变、仅TX/TY增加armature、临界阻尼、手指摩擦／力限与未知profile拒绝。测试在Composer实际编译环境执行。

旧ctrl在新驱动／碰撞下是压力测试，**不是旧策略在新环境中的闭环评价**。旧策略没有为新惯量、摩擦、Pitch驱动或碰撞形状训练。固定目标稳定和短训练无NaN，也不等于已证明任意学习策略不会形成冲击。

## 未解决事项

- TODO(V32-04)：平台等效惯量、驱动力、阻尼／增益尚非硬件标定。
- TODO(V32-06)：接触形状、软限位和策略动态触键压力继续验收，不能将固定目标测试扩大为全部组合保证。
- TODO(V32-SAC-03)：最终GPU／CPU子步一致性、短训与新策略完整回放由本轮总验收报告记录。
- 先前静态碰撞几何诊断只描述采样末态，不能凭末态无接触推断前一物理子步没有冲量；应以动态力轨迹和同前状态局部干预为因果证据。
