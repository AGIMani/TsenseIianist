# 《月光》自动候选指法与自碰撞训练（2026-09-20）

用户要求助手生成上传曲目的指法，继续“增加自碰撞惩罚并训练这首曲子”的请求。本轮已生成指法并启动训练；不是专家指法或已通过完整逆运动学验证的方案。

## 指法文件

- [逐音符指法CSV](../training_mjlab/inputs/clair_de_lune_beginner/generated_v1/fingering.csv)：146行，含原音符序号、音高、时刻、左右手和中文手指名称。
- [带指法训练文件](../training_mjlab/inputs/clair_de_lune_beginner/generated_v1/clair-de-lune-beginner_fingered.proto)：NoteSequence格式，part保存指法。普通.mid不保存本项目part，应使用此.proto导出。
- [生成与验证报告](../training_mjlab/inputs/clair_de_lune_beginner/generated_v1/fingering_report.json)。原始MIDI和待填模板保留。

| 手指 | 右手音符数 | 左手音符数 |
| --- | --- | --- |
| 拇指 | 15 | 10 |
| 食指 | 18 | 14 |
| 中指 | 25 | 9 |
| 无名指 | 23 | 9 |
| 小指 | 19 | 4 |

CSV中finger_id为0～4右手、5～9左手，均按拇指到小指；finger_number是便于阅读的1～5。

## 生成方法与限制

根据曲目两轨声部分配：instrument0（100音符）给右手，instrument1（46音符）给左手。脚本[scripts/generate_candidate_fingering.py](../scripts/generate_candidate_fingering.py)用动态规划选择指法：约束持续音不换指、同时不同键不能占用同指；代价偏好自然指序、较小手位移动及较小和弦拉伸。手指半音偏置、次序代价12和换位系数0.35等为明确的工程启发式，不是论文披露数值或TsenseI实测参数；允许必要的拇指穿越／换位指序，不能宣称零自碰撞。

首次尝试把指序设为硬约束，在第608帧无可行路径（连续音和离散化重叠会限制换位）。改为软偏好、仍保留持续音和同指多键的硬约束后，全曲可行。未删音、改节奏或为了通过验收缩短音符。

原解析结果按轨道排序，而源trim_silence取首末音符确定边界。生成proto在保持音符属性的前提下按时间排序，避免漏掉更早的左手开头；去掉0.4347825秒初始静音后仍保留146音符。连续时间和实际40Hz轨迹检查均无同指多键冲突，2587帧；冻结score.fingers形状(2587,10,88)，每帧每指最多1键。音高／起止时间／力度／乐器／program等逐音符多重集一致，proto读取回写一致。

TODO(V32-SCORE-01)本曲缺指法的阻塞已解除；TODO(V32-SCORE-02)候选指法的运动学可达性、实际换指／碰撞和演奏质量待训练及回放验收。自动分配只针对本曲，不替用户填写其他奖励未知映射。

## 当前场景与任务

- scene：`training_mjlab/outputs/scene_clair_shadow40_pose_20260920`
- task：`training_mjlab/outputs/task_clair_de_lune_beginner_shadow40_fingered_v1`
- 当前源姿态与旧冻结包不同：双腕yaw从旧包0到当前保存0.25 rad；左右base_link安装Z从旧包0.12到当前保存0.14 m。未编辑姿态源码，按已有保存值新导出，旧包不覆盖。
- 接触solimp／solref、琴键限位、执行器gain、joint范围和body旋转数组对比不变，40 Hz控制／400 Hz物理，20步0.5秒前瞻。场景差异见[previous_scene_diff.json](../training_mjlab/outputs/scene_clair_shadow40_pose_20260920/previous_scene_diff.json)。
- 首次向旧冻结场景导出时reset qpos合同检查拒绝（双腕yaw差0.25），随后导出新场景和匹配任务，合同检查成功；未跳过校验。

## 训练与查看

- run：`sac-20260920-204729-873996`，20:47:29后台启动，PID1879735。
- 从头训练，256环境×19532向量步＝5,000,192条采样；seed42、论文SAC Stage1、batch4096、32更新／学习向量步。进度条及三项加速开启。
- 当前五项奖励＋同手自碰撞`−0.2×去重body对数`，在每周期10个子步及末次forward取并集，正常触键排除；没有联动启用论文collision_reward或未知映射。
- 相比原小星星基线，曲目长度、候选指法、保存姿态和碰撞奖励均不同，不能作为仅奖励项的严格消融实验。每次完整回合更长，相同5M预算下完整演奏次数更少。

```bash
# 在独立项目根目录执行
V32_RUN=training_mjlab/outputs/sac-20260920-204729-873996
tail -f "$V32_RUN/metrics.jsonl"
# 查看进度条
 tail -f training_mjlab/outputs/launches/sac-20260920-204729-873996/console.log

# 已保存并评价后，导出本轮最佳有声结果
training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py \
  --run-dir "$V32_RUN"

# 暂停打开任务预览，显示指法和对应颜色
training_mjlab/.venv/bin/python preview/scripts/preview_mjlab.py \
  --scene training_mjlab/outputs/scene_clair_shadow40_pose_20260920 \
  --task training_mjlab/outputs/task_clair_de_lune_beginner_shadow40_fingered_v1 --paused
```

完整启动参数见[launch.json](../training_mjlab/outputs/launches/sac-20260920-204729-873996/launch.json)与[command.sh](../training_mjlab/outputs/launches/sac-20260920-204729-873996/command.sh)。候选生成后MIDI解析warning是已有非主轨调号提示，原始文件未改；V32-SAC-03／V32-06既有物理差异及标定TODO保留。

## 启动验收快照

- **启动验收补记**：截至交付检查，最新完整日志1290向量步／330240采样／9312次更新、loss与温度有限；step-1000.pt已写出。首次独立评价完整2587帧、未截断，F1=0.11072775；这是刚开始学习的指标，不是最终结果。后台训练继续。
