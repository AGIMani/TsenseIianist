# 当前 V3.2 与原版 Shadow 物理参数对照

> 本文是更改前的20 Hz／2000 Hz硬接触快照。当前已按用户要求改为Shadow接触＋40/400 Hz，见[新配置及验收](shadow_contact_40hz_acceptance.md)；下文“当前”指对比时的旧冻结包。

核对日期：2026-09-18。本轮只读取／编译核对并保存报告，没有修改训练物理参数。

## 1. 比较范围与依据

- **当前**：实际训练冻结包 `training_mjlab/outputs/scene`，通过SceneBundle载入并实例化MJLab GPU场景核对；不是仅查看未重新导出的源码。
- **原版**：仓库中的 `robopianist-rl/train.py` 默认配置及 `robopianist/.../suite/tasks/base.py:PianoTask` 默认双手物理模型：20Hz、无重力补偿、不禁用双手碰撞、不启用简化指尖。不是之前引用的Biomimetics论文环境，也不是另一个Daxian改版。
- 本仓库原版缺少Shadow模型资源及音色／PyAudio依赖。临时诊断进程跳过声音发现与未使用的播放模块，仍调用原PianoTask构建物理；采用本机缓存Shadow资源。左右XML逐字节核对官方Menagerie的 `1afc8be64233dcfe943b2fe0c505ec1e87a0a13e` 提交，与原版install_deps.sh锁定版本一致。网格资源使用该本机缓存，没有逐文件远程核验；未宣称完整历史训练复现。
- 两模型编译提取使用MuJoCo3.13；当前GPU运行环境为MuJoCo/Warp3.11，已额外核对mjModel参数及Warp实际选项。原版历史MuJoCo版本未锁定，版本相关默认值以本次编译结果为准。
- 完整逐geom、逐joint及执行器参数、散列和实测接触合并结果见[原始JSON](physics_shadow_comparison_20260918.json)。只归档数据，不给独立项目增加原版模型的运行依赖。

## 2. 全局仿真与求解器

| 项目 | 当前 V3.2 GPU训练 | 原版 Shadow 默认环境 |
| --- | --- | --- |
| 物理步长 | **0.0005s，2000Hz** | **0.005s，200Hz** |
| 控制间隔 | 0.05s，20Hz | 0.05s，20Hz |
| 每次控制的物理子步 | **100** | **10** |
| 积分器 | implicitfast | Euler |
| 约束求解器 | Newton | Newton |
| iterations 上限 | 100 | 100 |
| 线搜索 ls_iterations 上限 | 50 | 50 |
| 求解容差 tolerance | mjModel/XML默认记录1e−8；**GPU实际约1e−6** | CPU为1e−8 |
| 线搜索容差 ls_tolerance | 0.01（模型配置） | 0.01 |
| 摩擦锥 cone | pyramidal | pyramidal |
| 摩擦／法向阻抗比 impratio | 1 | 10 |
| noslip_iterations | 0 | 0 |
| 重力 | `[0,0,−9.81] m/s²` | 同左 |
| 默认手部重力补偿 | 关闭 | 关闭 |

迭代数字是上限，不能理解成每个子步都固定完成100次Newton迭代。GPU容差差异来自本机 `mujoco_warp/_src/io.py:put_model` 的FP32保护 `max(tolerance,1e-6)`，不是本次修改；已读取实际wpModel确认。不能仅凭场景manifest中的1e−8声称GPU也使用1e−8。

`impratio`是约束参数，不是接触滑动摩擦系数，也不表示原版摩擦力固定是当前的10倍。

## 3. 碰撞几何上的配置值

`solref`为正值时是 **时间常数（秒）、阻尼比**，不是刚度N/m。`solimp`五元组是阻抗曲线参数；数值接近1表示更强的约束。其物理解释及混合规则见[MuJoCo官方说明](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。

| 参数 | 当前手 | Shadow手 | 当前琴键／琴箱 | 原版琴键／琴箱 |
| --- | --- | --- | --- | --- |
| solref | `(0.004,1)` | `(0.005,1)` | **`(0.001,1)`** | `(0.01,1)` |
| solimp | `(0.9,0.95,0.001,0.5,2)` | `(0.5,0.99,0.0001,0.5,2)` | **`(0.99,0.99,0.001,0.5,2)`** | `(0.9,0.95,0.001,0.5,2)` |
| friction三元组 | `(0.8,0.005,0.0001)` | `(1,0.005,0.0001)` | `(1,0.005,0.0001)` | 同左 |
| priority | 0 | 0 | **1** | 0 |
| solmix | 1 | 1 | 1 | 1 |
| condim | 3 | 3 | 3 | 3 |
| margin／gap | 0／0 | 0／0 | 0／0 | 0／0 |
| contype／conaffinity | 1／1 | 1／1 | 0／1 | 0／1 |

friction三元组依次为滑动／扭转／滚动摩擦参数；当前condim=3，启用法向及两个切向约束，不能把后两个参数理解成另两种同时启用的制动力。琴键之间关闭碰撞，手与琴键允许碰撞。地面在两环境均为0／0，不参与碰撞。

### 手指碰琴键时，实际合并后的值

| 项目 | 当前 V3.2 | 原版 Shadow |
| --- | --- | --- |
| 接触solref | **`(0.001,1)`** | `(0.0075,1)`，两侧等权平均 |
| refsafe保护后的时间常数 | **0.001s** | **0.01s**，不小于2×0.005s |
| 接触solimp | **`(0.99,0.99,0.001,0.5,2)`** | `(0.7,0.97,0.00055,0.5,2)` |
| 滑动摩擦系数 | **1.0** | **1.0** |
| 阻尼比 | 1 | 1 |

当前琴键priority=1高于手的0，所以手—键采用琴键参数，**不是手上写的solref=0.004或滑动摩擦0.8**。原版两侧priority相同，solref／solimp按solmix平均，摩擦分量取较大值。依据[官方接触合并规则](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)，并用提取参数构建两球接触小例检查了实际contact数组；小例只核验合并，不模拟原手的接触力。

同手或双手相互接触时，两侧均为手参数：当前时间常数0.004s、滑动摩擦0.8；Shadow接触配置0.005s，在5ms步长与默认refsafe下时间常数保护到0.01s、滑动摩擦1。

## 4. 琴键到达下压极限后的止挡

| 项目 | 当前 V3.2 | 原版 Shadow |
| --- | --- | --- |
| solreflimit | **`(0.001,1)`** | `(0.02,1)` |
| solimplimit | **`(0.999,0.999,0.001,0.5,2)`** | `(0.9,0.95,0.001,0.5,2)` |
| 白键角度范围 | 0～0.066568rad，设计行程10mm | 相同 |
| 黑键角度范围 | 0～0.088656rad，设计行程8mm | 相同 |
| 回弹弹簧刚度 | 2N·m/rad | 相同 |
| 关节阻尼 | 0.05N·m·s/rad | 相同 |
| armature | 0.001kg·m² | 相同 |
| springref | −1° | 相同 |
| 白键／黑键质量 | 0.04／0.02kg | 相同 |

区别主要在手键接触与关节限位的数值硬化，回弹弹簧和设计行程没有加大。当前仍是MuJoCo的软约束近似硬接触，并非数学意义上绝不穿透或绝不超限；接触标定TODO(V32-06)仍未解决。

## 5. 几何复杂度及驱动差异

| 项目（双手合计） | 当前 V3.2 | 原版 Shadow |
| --- | ---: | ---: |
| 手部参与碰撞的网格geom | **44** | **12** |
| 手部参与碰撞的简单geom | 0 | **64：24盒＋20胶囊＋14圆柱＋6球** |
| 琴键＋琴箱碰撞盒 | 89 | 89 |
| 全场景geom，含视觉 | 178 | 214 |
| 物理自由度 | 138 | 140 |
| 手部执行器 | 50 | 44 |
| tendon | 0 | 8 |
| 显式body碰撞排除项 | 2 | 4 |

原版即使 `primitive_fingertip_collisions=False`，也已经大量使用简单形状；这个开关只进一步替换特定指尖碰撞体。当前总geom更少，不代表碰撞检测更便宜。显式排除项数量不包含MuJoCo同体／父子等自动过滤，不能把这两项当成全部禁碰名单。

原对照时的驱动差异：V3.2 手指阻尼0.05、armature=0、frictionloss=0，位置kp=3、力矩限幅±0.6N·m；Shadow普通手指阻尼0.05、armature=0.0002、frictionloss=0.01，kp和力矩范围按关节不同（数据逐执行器存于JSON）。原版还有腱耦合，不能用一个kp代表逐关节对齐。2026-09-24随后经固定目标接触测试，新源码和新冻结包采用armature=0.0002、kp=1，详见[振荡定位与修复](contact_stability_shadow_20260924.md)；旧JSON保留原测量。腕平移阻尼TX/TY=30、TZ=50保留；原版默认仅增加TX/TY，其阻尼按质量计算临界阻尼，本次编译约67.48。

**自碰撞奖励权重0.2不是物理碰撞参数**：它影响策略得分，不改变solref、摩擦或接触求解器。

## 6. 对训练速度的意义

最明确的差别是相同控制周期从10个物理子步变成100个，再叠加44个手网格碰撞体。它们能解释当前物理负担大的方向，但本轮没有运行Shadow吞吐基准，不能据此声称实际训练恰好慢10倍或给出各因素贡献比例。

减少子步或采用简化碰撞形状可能提速，但会影响此前要求的硬接触、穿透和运动响应。论文pp.6–8的2.5ms×10、40Hz也不是这里原版Shadow的5ms×10、20Hz；三者不可混用。此次仅比较，没有自动对齐任何参数。

## 7. 当前参数在哪些文件

- 已用于训练的冻结场景：[scene.xml](../training_mjlab/outputs/scene/scene.xml)、[manifest.json](../training_mjlab/outputs/scene/manifest.json)。
- 生成时物理步长、琴键优先级／接触／止挡、积分器：[suite/tasks/base.py](../tsensei_v3_2_pianist/suite/tasks/base.py)，约29、65～72行。
- 手的摩擦、接触时间常数和默认阻尼：[urdf_model.py](../tsensei_v3_2_pianist/models/hands/urdf_model.py)，约22～25行。
- 琴键行程、弹簧与阻尼：[piano_constants.py](../tsensei_v3_2_pianist/models/piano/piano_constants.py)。
- GPU场景容量与载入：[mjlab_env/scene.py](../tsensei_v3_2_pianist/mjlab_env/scene.py)，当前nconmax=1024、njmax=4096是容量配置，不是力或刚度。

修改生成源码不会自动更新冻结场景或正在运行的训练；本报告没有做任何此类修改。
