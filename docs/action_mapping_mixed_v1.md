# mixed_v1 动作范围与新训练（2026-09-20）

用户要求：普通手指输出[0,2]，四指swing保持；腕TY输出[0,2]映射0～150mm，TZ输出[−2,0]映射−25～0mm。后续明确Pitch“改关节角度范围，策略映射的范围不用改”，因此Pitch策略仍[−1,1]，实际关节／执行器目标变为±0.2rad。未另行指定的TX、Yaw与踏板保持。

## 映射总表

| 通道 | 索引（右／左） | 新策略输出 | 物理目标 | 公式（a是对外策略动作） |
| --- | --- | --- | --- | --- |
| 拇指Rot | 0／25 | [0,2] | [0,2.09]rad | 1.045a |
| 拇指Rot2 | 1／26 | [0,2] | [0,1.74]rad | 0.87a |
| 其余手指屈曲 | 每手0～19中除Rot、Rot2、SW | [0,2] | [0,1.57]rad | 0.785a |
| 四指swing | 5/9/13/17，30/34/38/42 | [−1,1] | [−0.35,0.35]rad | 0.35a |
| 右TX | 20 | [−1,1] | [−760.5,460.5]mm | 610.5a−150 |
| 左TX | 45 | [−1,1] | [−460.5,760.5]mm | 610.5a+150 |
| TY | 21／46 | [0,2] | [0,150]mm | 75a |
| TZ | 22／47 | [−2,0] | [−25,0]mm | 12.5a |
| Pitch | 23／48 | [−1,1] | [−0.2,0.2]rad | 0.2a |
| Yaw | 24／49 | [−1,1] | [−0.25,0.25]rad | 0.25a |
| 踏板 | 50 | [−1,1] | [0,1] | (a+1)/2 |

[全部51通道CSV](action_mapping_mixed_v1.csv)从新冻结模型提取，物理平移使用m、角度rad。没有将150当成150m。TY方向世界−X，TZ世界+Z（负值下压）；yaw同符号同向、pitch左右反向的轴约定不变。

## 策略、SAC和回放如何配套

- `mjlab_env/action_mapping.py`定义版本`legacy`与`mixed_v1`。对外动作是`a=tanh(z)+shift`；普通手指／TY的shift=+1，TZ=−1，其他为0。它仍是绝对位置目标。
- Actor.sample／deterministic、预热随机动作、critic输入、GPU replay存储／校验都使用新坐标；不是仅在显示里改标签。物理与任务奖励统一使用`scene.physical_action`转换，避免奖励和ctrl不同步。
- 所有通道输出区间宽度仍为2，只平移，变换Jacobian为1，log概率无需增加缩放项，目标熵继续−25.5。SAC gamma／学习率等保持；critic直接接收平移后的动作，所以不承诺新训练与旧训练数值逐步相同。
- checkpoint同时保存元数据和learner的action_profile。回放／评价自动恢复，warm-start不同profile会拒绝；旧无此字段检查点按legacy恢复。原始tanh网络权重键不变，shift从profile恢复。
- GPU手动预览的Control转换、性能分析replay和视频动作诊断同样识别profile。视频report新增action_profile，录制动作仍为实际对外输出。
- CLI使用`--action-profile mixed_v1`；兼容默认仍为legacy，必须同时指定新scene/task。新任务观测依旧2070维，历史动作观测继续按既有合同保存物理控制值。

## 这次修改与起始前冲

新的**对外action=0**确实让TY／TZ目标为0。但网络内部z=0时，新TY动作是1、TZ是−1，仍然对应75mm向内和−12.5mm下压；普通手指仍在范围中点。TX没有改动，也仍会由近0输出指向两手中央。因此除Pitch物理范围扩大外，本轮属于动作坐标平移，**不能宣称消除了前冲**。没有自行加目标限速、初始偏置校准、增量动作或子步插值。

论文§2.6 pp.7–8描述统一[−1,1]动作和线性物理映射；本次混合范围按用户明确要求适配，不冒充论文原定义。Pitch物理范围是用户后续明确选择。

## 冻结包与正式训练

- 场景：`training_mjlab/outputs/scene_clair_mixed_pitch02_20260920`
- 任务：`training_mjlab/outputs/task_clair_mixed_pitch02_20260920`
- 曲目仍是146音符候选指法《月光》；2587控制帧，40/400 Hz、0.5秒前瞻。
- 保留当前安装Z=0.14m、左右yaw初值0.25rad、自碰撞权重0.2。
- 正式run：`sac-20260920-212536-832002`，2026-09-20 21:25:36后台启动，PID1923100。
- 从头训练256环境×19532向量步＝5,000,192采样；batch4096／每学习步32更新、seed42、三加速及进度条开启。
- 上一轮《月光》训练未被停止；两者同GPU并行时会共享算力，不能沿用单进程59分钟作为这轮ETA。

```bash
# 独立项目根目录执行；复制此命令会另开训练，不是查看当前进度。
training_mjlab/.venv/bin/python -m tsensei_v3_2_pianist.rl_torch.train \
  --scene training_mjlab/outputs/scene_clair_mixed_pitch02_20260920 \
  --task training_mjlab/outputs/task_clair_mixed_pitch02_20260920 \
  --action-profile mixed_v1 --self-collision-weight 0.2 \
  --num-envs 256 --vector-steps 19532 --batch-size 4096 --gradient-steps 32

# 当前进度
tail -f training_mjlab/outputs/sac-20260920-212536-832002/metrics.jsonl
# 已评价并保存后导出最佳视频
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir training_mjlab/outputs/sac-20260920-212536-832002
```

[启动命令／PID](../training_mjlab/outputs/launches/sac-20260920-212536-832002/launch.json)。旧包保持，不修改历史检查点或运行中的旧训练配置。

## 实际验证

- 31项unittest通过：新3项覆盖实际Actor采样／确定性输出及概率平移、replay边界与学习更新、GPU端点／中点／非法动作无状态修改；另含已有SAC、奖励、任务、加速、最佳选择回归。
- 第一轮检查发现旧reward_options返回字典多了legacy字段导致2项兼容断言失败，已改为仅非legacy附加profile；GPU断言把TorchArray直接与Tensor比较导致1项错误，改用其张量视图。复跑全部31项通过，未放宽映射阈值。
- 新配置短训练4环境×6步，24transition／10更新／8截断／2评价，编译与融合优化器开启，10.90秒完成；summary在mixed_v1_smoke_20260920。
- 新短测checkpoint自动查找新包并恢复mixed_v1，确定性4步回放动作均在对应范围内，逐步ctrl与映射公式吻合；旧step-1000 CUDA checkpoint按legacy成功恢复。一次尝试在CPU加载CUDA训练RNG失败，改用原CUDA设备通过，未声称跨设备RNG续训兼容。
- 源文件语法检查通过，任务导出合同通过，CSV包含51行。短测不是收敛结果；V32-SAC-03／V32-06／V32-SCORE-02等原TODO保留，尚未验证新范围是否改善演奏或起始运动。

## 正式训练启动快照

- **正式启动验收补记**：最新完整日志1050向量步／268800采样／1632次更新、loss与温度有限；step-1000.pt已保存。首个评价完整2587帧、未截断、F1=0.09358424，属初期学习结果。PID1923100存活，记录的全部训练源码SHA与交付文件一致。后台训练继续，未等待5M完成。
