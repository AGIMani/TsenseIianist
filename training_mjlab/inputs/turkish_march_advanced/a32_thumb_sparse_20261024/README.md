# 当前 A3.2 少拇指候选（2026-10-24）

原始输入为用户的 `/home/hj/下载/turkish-march-advanced.mid`，SHA256 `83707e4f40347d0a51402349a8ef8af7cf59c357656b50432591b3e8fceb88eb`。2544音符、222.475秒；轨道0右手1460音、轨道1左手1084音。仅更改proto的`part`指法字段，音高、起止、力度、轨道及全部其他proto字段保持；持续音不换指、同指真实重叠为0。`fingering.csv`包含逐音分配，`input_validation.json`为独立原谱与训练时序检查。

统一默认入口 `all_use_this_train.sh` 指向这里的 `turkish-march-advanced_fingered.proto`。当前是**待用户查看的训练指法候选，不是已通过物理演奏验收的轨迹**；本次没有提交训练。源码仍为入口原有exper038冻结归档（SHA `1d4a67f15c796a849387ccc967c4960f4ee12d0b18bde2ff81f27a591c23d560`），新规划工具不改训练算法，也不声称工作区脚本进入该归档。

## 手型与搜索

用冻结A3.2实际URDF、当前任务指尖点、真实滑轨轴/限位与黑白键目标采样。无臂，height=.09m、ty=.04m、yaw=15°，TZ/Yaw结构移除，MCP0–1.4rad、Pitch0–.5rad，八指10°–80°耦合。以原始事件进行动态规划，保留持键占用，结合自然非拇指顺序、可选边缘拇指、共同腕位姿、指尖残差、移动成本和拇指软成本。未套用人手跨度数值。

| 手指 | 右手音符数 | 左手音符数 |
|---|---:|---:|
| 拇指 | 181 | 54 |
| 食指 | 198 | 240 |
| 中指 | 212 | 378 |
| 无名指 | 411 | 305 |
| 小指 | 458 | 107 |

拇指235/2544=9.24%。相同模型/参数、拇指成本为0的基线为767/2544=30.15%，减少532音；两手均保留拇指。对照见`thumb_baseline_comparison.json`；有限候选与历史依赖锚点不构成连续全局最优证明。

## 需要重点查看

静态候选事件平均残差：右2.55mm、左1.65mm；最大右46.20mm、左18.58mm。右17个、左1个起音事件在候选集中找不到10mm内方案。对最难五组和弦共241次连续局部IK复核：右73/76/81/85最佳仍24.58mm，左45/49/52/57仍15.35mm，另三组约9.94–15.33mm。局部搜索未找到3mm内解，不能据此证明全局不可达。详见`reach_diagnostics.json`、`continuous_ik_probe.json`。

`candidate_wrist_anchors.json`是离散几何节点，**不是可下发轨迹**。最短原谱事件间隔下节点转移速度可达右48.16m/s、左7.61m/s；物理预览使用0.15s腕请求平滑和当前1.99m/s、1.99m/s²逐轴目标硬限，并记录真实跟踪误差。不能把平滑节点或静态指尖误差当作触键成功。

3秒对照确认：固定规划腕参考和关节种子后，IK平均误差由约0.393m降至0.0227m，但100%实体键程micro-F1仅0.02417；换键横拖、腕跟踪、碰撞和按压不足仍存在。仅开启先抬指控制的对照F1=0，未解决快速音符。本轮保留这些失败记录，不改键阈值/驱动/时值来制造成功。20秒交付回放指标以同次录制报告为准。

## 默认训练时序

沿用用户2026-10-20批准的本曲1000Hz物理频率；200Hz无法采到原谱某些重复键松键间隙，1000Hz通过，最短音有30个物理样本。控制20Hz、512环境、batch1024、每条可学习经验0.25次更新不变。旧默认Prelude预填充1585步，新曲4455步；总向量步由19532增至22402，补足新增2870步预填充，维持17947个可学习向量步、2,297,216次优化更新。4455控制步整曲预填充需2,280,960条经验，replay-capacity改为3,000,000。奖励算法/权重保持；每控制步50子步使跨手单子步为−.01、整步最多−.5，同手每对每子步−.1且整项最多−5。完整默认值在唯一奖励总表中维护。

## 回放与来源

`results/turkish_march_planning_20261024/interactive_replay/launch_local.sh`打开20秒同次物理录制，保留实际触键高亮与声音；对比图为`actual_vs_original_midi.png`。无checkpoint、无策略推理；这是离线IK控制的物理检查。物理采样1000Hz，原谱时间0–20s，负1秒准备段不在可见录制内。几何节点交互视图仅展示规划意图。原始动作、实际实体键角、事件、源码与SHA保存在包内，加载验收状态见交付目录。

执行与失败记录：`logs/turkish_hand_fingering_20261024/`。所有模型构建、静态IK和物理录制在`volc-a32:/home/houj/projects/pianist_turkish_hand_20261024`单线程CPU完成；本机只做数组检查与已录制MJB受保护显示。

## 开源方法参考

- [PianoPlayer](https://github.com/marcomusy/pianoplayer)：前瞻搜索与手移成本。
- [Automatic-Piano-Fingering](https://github.com/PRamoneda/Automatic-Piano-Fingering)：研究人类指法的自回归模型和beam search，未当作机器人可达证明。
- [Kanma/piano_fingering](https://github.com/Kanma/piano_fingering)：音符/和弦分层图和动态规划。
- [PyDactyl](https://github.com/dvdrndlph/pydactyl)：多候选最短路及可解释成本。
- [PianoFingering.jl](https://github.com/JuliaMusic/PianoFingering.jl)：指法MDP状态设计。
- [lumikey/piano-fingering-model](https://github.com/lumikey/piano-fingering-model)：Transformer候选生成与交互接口。

本轮独立实现，不安装或复制这些仓库算法代码，不采用其人手跨度或未经约束的逐音输出。
