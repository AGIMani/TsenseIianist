# TsenseI A3.2 含 ID 模型审计（2026-09-28）

## 结论

用户说明新附件与当前训练手为同一只手，增加 ID 部分。**文件实测是同一系列、相同自由度/机械限位，但不是只给旧模型加标识或外观。** 新 URDF 同时改变关节原点、部分轴方向、惯性和全部 22 个 STL；因此按完整新模型接入，不能把新旧训练差异只归因于提前定位。

本报告只记录静态文件/运动学审计，不代表动态接触稳定性、可达性、硬件参数或新训练 F1 验收。附件内脚本和导出日志没有被执行。

## 来源及比较口径

- 用户附件：`TsenseI_A3.2__hand_withid.zip`；压缩包内部旧导出名 `A3.1V6 26.9.27(6)`，仅作为数据。
- 旧实际训练模型：备份 `model_backups/before_TsenseI_A3.2_20260928.tar.gz` 内 `assets/source/left_original.urdf`、`assets/urdf/daxian_left.urdf` 和 `assets/meshes_left/`。
- 新实际模型：`assets/source/left_original.urdf`、`assets/urdf/TsenseI_left.urdf`。
- 对照代码：`scripts/prepare_models.py`、`models/hands/{urdf_model,tsensei_hand_constants,fingertip_geometry}.py`、`scripts/generate_collision_proxies.py`。代码位置均相对包内 `tsensei_v3_2_pianist/`，脚本位置相对独立项目。
- 论文对照：已读 `docs/references/biomimetics_2026_610_notes.md`；本项是用户指定模型替换/实测审计，不自行变更奖励、SAC 或论文未知映射。

**仓库根原 `daxian_v3.2` 不是此前实际训练所用的完整模型。** 根 URDF 总质量 183.0471 g，包内旧 source 为 `A3.1V6 26.8.4-2`、171.3631 g。下表以包内实际训练模型为准；不以根目录名代替运行来源。

| 项目 | 旧实际训练 | 新附件/新归一化模型 |
|---|---:|---:|
| 手侧 | 左手 + 派生镜像右手 | 仍只提供左手 |
| link / joint / 可动关节 | 22 / 21 / 20 | 22 / 21 / 20 |
| URDF 各 link 总质量 | 171.3631 g | 156.9393 g（−8.4171%） |
| STL 总三角形（单左手） | 1,450,275 | 320,547（−77.8975%） |
| 与旧 SHA 相同的 STL | — | 0 / 22 |
| 可动关节机械限位 | 原有20组 | 20组逐项相同 |
| effort / velocity | 20组均为0占位 | 仍均为0占位 |
| 独立 ID link / joint | 无 | 无；ID/外壳并入原link网格 |

所有新惯性张量特征值均为正，且满足主惯量三角不等式；这只是代数有效性，不能证明 CAD 材料/硬件质量正确。全部字段和散列见 [机器可读审计](tsensei_a32_model_audit_20260928.json)。

## 必须修复的源文件单位与 link 角色变化

新附件 `base_link.STL` 的原始坐标跨度为 `[50.9105835, 99.0071831, 105.3059235]`，URDF 却没有 `scale`；其他 STL 已使用米。将这个 base 同样解释成米会产生数十米大的碰撞体，不能据此生成代理或开始训练。应保持附件/raw source 不变，仅在归一化派生 URDF 对此附件的 base visual/collision mesh 添加 `scale="0.001 0.001 0.001"`，明确记录按毫米导出的修正。下附 JSON 的新 bounds 已使用此修正。

此外，**base 和 palm 在新导出中交换了实际部件角色**：

| 部件角色 | 旧 source / 已训练模型 | 新附件 |
|---|---|---|
| 约46×40×29 mm刚性安装块 | `base_link` | `Left_hand_palm_link` |
| 覆盖手掌/指根的复杂壳体 | `Left_hand_palm_link`，约41.51×96.03×80.97 mm | `base_link`，约50.91×99.01×105.31 mm |

在共同掌部坐标下，新 palm 安装块的零姿态 bbox 与旧 base 安装块几乎重合；名称本身不能决定碰撞处理或奖励语义。旧生成器对 base 用单box、palm用mesh，若不迁移实际部件角色，就会把新大手壳及拇指空腔填成实心box。与既有策略相近的首个候选是新base保留掌壳mesh、新palm按安装块box处理，并重新验证接触；MuJoCo单mesh依然使用凸碰撞，不能保证保存非凸空腔。

新 base STL 按共享顶点可分8个组件；最大壳体占24,162/29,723个三角形且本身可能非凸。简单按组件拆分并不保证恢复所有空腔。如实测仍卡拇指，需要明确的多凸/compound壳体近似及接触回归，而非一律取消整个手壳碰撞。

还必须核对 `forearm` 奖励的掌部 geom 名单及原掌壳—拇指装配排除对是否跟随实际壳体迁移；只匹配 `palm` 字符串可能使新奖励只检查小安装块、漏掉真实手壳。实际集成应在主报告记录最终选择与接触验收，不能仅靠本文件审计推定通过。

## 关节名和方向

旧 source 的小指命名 `rotaback → MCP → swing → PIP` 需要旧转换才能成为运行时 `MCP → swing → PIP → DIP`。新附件已是 `MCP → SWING → PIP → DIP`，**只需 `SWING → swing` 大小写归一化，不能再套旧四段重命名**。父子树和顺序保持 4 自由度。

使用共同掌部坐标（掌固定关节位置 `[0.0237, 0.0065, 0.0441380069844] m`、姿态单位阵）消除根坐标换算后：

| swing 轴 | 新旧共同坐标轴点积 | 结论 |
|---|---:|---|
| left_hand_index_swing_joint | -0.999974491 | 方向近乎反转 |
| left_hand_mid_swing_joint | -1.000000000 | 方向近乎反转 |
| left_hand_ring_swing_joint | -0.999974491 | 方向近乎反转 |
| left_hand_pinky_swing_joint | -0.999974491 | 方向近乎反转 |

轴点积 −1 表示完全相反。旧非零 swing 姿态迁移时 `q_new ≈ −q_old` 才是对应物理方向的起点，但新轴约 0.409° 的偏差和几何变化仍然存在，因此这个符号变换不保证整个模型姿态完全一致。**当前四指 swing 初始值全为0，零角无需改符号。** 新模型训练应使用新URDF原轴并从头训练；旧检查点不能直接当作兼容策略。

拇指也存在实际变化，而非只换根坐标：

| 拇指关节 | 零姿态子link旋转变化 | 共同坐标关节轴点积 |
|---|---:|---:|
| left_hand_thumb_rota_block_joint | 4.46020° | 1.000000000 |
| left_hand_thumb_rotaback_joint2 | 5.96419° | 0.994587029 |
| left_hand_thumb_MCP_joint | 21.78296° | 0.952851357 |
| left_hand_thumb_PIP_joint | 21.78296° | 0.952851357 |

尤其 thumb MCP 原点/姿态变化，使零姿态末端方向累计改变约 21.783°。不能只通过翻转 swing 或重新命名恢复旧拇指空间姿态。

## 保留当前 initial 角度后的空间变化

审计读取当前 `INITIAL_ANGLES_BY_SIDE["left"]`，拇指角为 `[1.13, 0.4, 0.25, 0.35] rad`，四指 swing 均为0，其余值保留当前来源。逐 URDF 关节树进行前向运动学，末端点按同一 `measure_distal_tip` 定义（碰撞网格局部 +Z 极值点/极值平面中点）。下表单位 mm，坐标为**归一化手根坐标，不是钢琴世界坐标**；旋转到世界坐标不改变位移长度。

| 左手末端 | ΔX | ΔY | ΔZ | 位移长度 |
|---|---:|---:|---:|---:|
| left_hand_thumb_PIP_link | -21.705 | +21.052 | +18.773 | 35.591 |
| left_hand_index_DIP_link | -0.249 | -1.039 | +7.466 | 7.542 |
| left_hand_mid_DIP_link | +4.570 | -0.000 | +5.371 | 7.052 |
| left_hand_ring_DIP_link | +3.150 | +0.315 | +7.850 | 8.465 |
| left_hand_pinky_DIP_link | +3.675 | +0.917 | +7.855 | 8.720 |

因此，保留数值是“保留初始关节角和手根安装设置”，**不是“完全保留旧手的指尖空间位置”**。挂载方向包含局部 X 朝世界 −Z，拇指局部 ΔX 为 −21.705 mm，意味着新拇指可能明显高于旧位置；实际钢琴世界位置需含当前 Yaw、腕部关节、预备抬升后检查。

如果另需恢复旧指尖空间姿态，应单独做新模型受限 IK 并验证整指/键面接触，不能只调整一项 q 或把关节零点偷偷改回旧模型。本次审计没有改 initial、origin、关节限位，也没有生成或采用 IK 姿态。

## 新曲面指尖使旧碰撞生成器失效

全部5个新末端的 +Z 最高处都只有一个唯一顶点。旧生成器用这个“极值平面”Y宽度/2作 capsule 半径，会得到0；MuJoCo 随后拒绝 radius=0。旧平端几何成立的假设在新含ID外壳几何上不再成立。

最小兼容方案是保留原端点位置和胶囊布局：旧平端仍使用旧极值平面；曲面改用顶端1 mm薄带的Y宽度/2，胶囊远端球心为 `tip − r·Z`，从而 +Z 表面仍经过实际网格极值点。以下是新左手的实测候选半径（右手镜像同）：

| 末端 | 顶端1 mm薄带 Y 半宽 |
|---|---:|
| left_hand_thumb_PIP_link | 2.333181 mm |
| left_hand_index_DIP_link | 2.419175 mm |
| left_hand_mid_DIP_link | 2.011263 mm |
| left_hand_ring_DIP_link | 2.409054 mm |
| left_hand_pinky_DIP_link | 2.333181 mm |

1 mm是记录在生成规则中的工程测量带，不是厂家提供的接触垫尺寸。该胶囊仍是已采用 Shadow 圆角接触方式的几何近似，不是新 STL 精确凸包或保证外包的碰撞体。应在 `collision_proxies.json` 记录实际拟合方法/薄带深度/来源 SHA，并拒绝非正半径；不可用全末节网格的近端壳体宽度冒充指尖接触半径。

新网格全部更换，必须重新生成左右 mesh、指尖点和 collision proxy；不复用旧OBJ：旧同名OBJ会被加载器优先选择，导致表面换成新STL但实际MuJoCo仍载入旧几何。新STL最大40,603三角形，均低于本项目旧20万面转换阈值。

## 分部质量变化

单位 g。base/palm之间有重新分配，而若干远端质量增大；只看整手减重会掩盖关节下游惯性变化。

| Link | 旧 | 新 |
|---|---:|---:|
| base_link | 14.2939 | 51.6913 |
| Left_hand_palm_link | 74.4549 | 11.4679 |
| left_hand_thumb_rota_block_link | 8.8230 | 5.9132 |
| left_hand_thumb_rotaback_link2 | 7.5046 | 7.1982 |
| left_hand_thumb_MCP_link | 2.9251 | 2.3440 |
| left_hand_thumb_PIP_link | 1.3467 | 3.4245 |
| left_hand_index_MCP_link | 8.2302 | 8.8955 |
| left_hand_index_swing_link | 4.5717 | 5.6572 |
| left_hand_index_PIP_link | 2.9984 | 2.3603 |
| left_hand_index_DIP_link | 1.6065 | 4.0206 |
| left_hand_mid_MCP_link | 4.9588 | 5.6240 |
| left_hand_mid_swing_link | 4.6206 | 4.6201 |
| left_hand_mid_PIP_link | 3.2155 | 2.5345 |
| left_hand_mid_DIP_link | 1.2027 | 5.0682 |
| left_hand_ring_MCP_link | 8.2302 | 8.8145 |
| left_hand_ring_swing_link | 4.5474 | 4.6201 |
| left_hand_ring_PIP_link | 2.9984 | 2.3526 |
| left_hand_ring_DIP_link | 1.6065 | 4.8708 |
| left_hand_pinky_MCP_link | 4.9588 | 5.5430 |
| left_hand_pinky_swing_link | 3.9495 | 4.2502 |
| left_hand_pinky_PIP_link | 2.8781 | 2.2439 |
| left_hand_pinky_DIP_link | 1.4418 | 3.4245 |

## 验证与待办

本报告实际执行：安全提取附件数据；从旧assets备份读取；XML树/限位/来源SHA比较；逐STL二进制坐标统计；惯性特征值检查；共同掌坐标下零姿态及当前初始关节角的前向运动学；曲面顶端薄带半径计算。使用项目 Composer venv 的 NumPy/SciPy，未执行附件 ROS launch/CMake 文件。

- **V32-A32-01**：新含ID网格下 origin/initial 的实际键面距离、全程接触稳定性及长期训练F1由本轮集成验收补充；本静态报告不代替这些结果。
- **V32-06/07**：圆角碰撞代理与接触垫中心仍为工程近似，需要硬件几何标定。
- **V32-10**：附件没有厂家右手模型，继续保留关于镜像右手的明确标记。
- **V32-04**：effort/velocity仍为0，继续沿用已确认驱动适配不代表硬件标定。

旧源模型保留在备份/旧源码归档；新的1231维提前定位策略应从头训练或通过显式迁移流程，不应跳过模型来源/控制合同检查。
