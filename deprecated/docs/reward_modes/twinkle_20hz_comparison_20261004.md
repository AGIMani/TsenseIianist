# 小星星20/200Hz五组对照（2026-10-04）

**2026-10-04后续指尖修正**：当前统一脚本已切换 `source_v2_fingertips.tar.gz`，with ID十个末节胶囊直径修正为11.72–12.82mm，保留原指尖点及其他学习/双卡参数。下文旧v1几何与归档描述为历史；见[实测和叠加图](fingertip_collision_audit_20261004.md)。

用户要求训练脚本从40/400Hz改为20/200Hz，开启多个对照并寻找实测最高F1。本轮保留原34音/指法、A3.2模型、31动作耦合、Pitch±.16rad、腕速、实体碰撞及1秒/2.5cm策略准备；前瞻20→10帧以保留0.5秒，提前观察仍0.15秒，完整观测2297→1407。因此全部从头训练，不跨频率/观测合同恢复旧模型。

## 10月4日15:41进度检查

有效`_v2`运行中，S0–S3已正常完成全部10,000,384样本/2,476,672更新，Slurm均COMPLETED0:0，用时约2小时7–10分钟。S4已由协调器自动接续为1587，15:25:37开始；检查快照到2340/19532步（约12%，1,198,080样本），学习指标有限、无新核验错误。其余组与独立复评安排保持，当前没有最终`best_model.json`。

| 组 | 状态 | 保存best步数 | 保存best精确合并F1 | final精确合并F1 |
| --- | --- | ---: | ---: | ---: |
| S0原五项末端 | 已完成 | 13500 | .964542 | .964892 |
| S1子步评分 | 已完成 | 19500 | .980321 | .972680 |
| S2实际触键.5 | 已完成 | 16500 | .979322 | .977749 |
| S3同手惩罚.2 | 已完成 | 19532 | **.984377** | **.984377** |
| S4错键下压.1 | 训练中 | 暂为2000 | .853699 | 未完成 |

上表均为训练时每次3回合；best按最低回合F1保存，因此S0 final的合并F1略高于best并不矛盾。S4尚未跑满，不能按当前分数判断该设计失败。独立10回合复评将在五组全部完成后自动执行，当前只能称S3为已保存候选中的暂时最佳，不能宣布最终获胜或达成.99。

S3三回合TP/FP/FN=7624/21/221，最低回合F1=.984115。相比S2 best，补回70个目标采样、少7个错采样；58个补回来自三个短音，短音实际覆盖从42/108（38.89%）升为100/108（92.59%）。同手接触采样为0，跨手接触平均14/1830子步（约.765%），键底0、准备误触0，侧接触194子步/回合；不称没有任何接触。

当前S3漏采样：迟按151、早松69、音内中断1。第23音A3/L2（index22）占90/221个漏采样，主要迟按约149ms，是进一步提高F1的明显瓶颈。一个回合未通过150ms起音匹配，但后段实际有弹，按物理激活并集重建不将它算整音全漏。完整重建及配置/进度快照在`logs/hpc/twinkle_20hz_20261004/status_20261004_1539/{progress_snapshot.json,note_analysis.json}`。

按S4目前约1267样本/秒估算，剩余训练约2小时，随后还需独立复评；这是进度估算，不是调度完成时间保证。本次检查未修改训练参数、追加训练或中断任务。

## 第④组最佳模型视频

**后续最终最佳视频（10月4日晚）**：五组训练和独立复评完成后，S4_false01 best16000以10回合200Hz精确F1 .984573居本轮第一。已通过原训练归档导出[第⑤组最佳视频](../preview/outputs/twinkle_20hz_best_S4_front_20261004/videos/S4_best.mp4)及[播放页面／报告](../preview/outputs/twinkle_20hz_best_S4_front_20261004/index.html)。Slurm1619正常完成；1280×720、20fps、单正面原速有声、10.2秒、带原指法。全片解码/音轨非静音/183帧指法匹配/产物SHA/ZIP CRC通过，检查实际中段画面。本条F1是独立复评结果，不是视频单回合测分；下面S3视频保留为此前导出。

按用户要求已导出S3_self02的`best.pt`（step19532，与final.pt整文件相同），使用本组`source_v3.tar.gz`及完整config严格重建。Slurm作业1588正常完成，未改变正在运行的第五组。视频为单正面无臂视角、1280×720、20fps、原速有声，包含准备段、原谱指法标注和结尾停留，共10.2秒。

- [观看／下载MP4](../preview/outputs/twinkle_20hz_S3_front_20261004/videos/S3_self02_best.mp4)
- [播放页面与导出报告](../preview/outputs/twinkle_20hz_S3_front_20261004/index.html)
- [打包文件](../preview/outputs/twinkle_20hz_S3_front_20261004/export_bundle.zip)

183帧目标指法与录制一致，204视频帧完整解码、音轨非静音、产物散列与ZIP CRC均通过，核查1秒和4.5秒画面；验证记录在`logs/hpc/1588/video_export_validation.json`。上文.984377为训练时三回合合并精确F1，不将其标为此次单回放分数。

## 入口和实验设计

### 统一默认训练脚本

2026-10-05早间历史配置（后来关闭和弦，见本文末尾1724）：关闭 `false_press_penalty`，按键奖励选择 `paper`、`chord_bonus=true`、`chord_scope="on"`。根脚本以项目绝对路径传入 `configs/rewards/paper_chord_on_20261005.json`，固定算法/模型继续使用已验收 `source_v2_fingertips.tar.gz`（SHA以根脚本为准）。此项仅改显式奖励配置，没有修改算法源码或重写历史S4 JSON；下面S4原默认及旧归档验收均是历史记录。当时按键公式为 `mean_substep[0.6×m×正确键分＋0.4×exp(−1.25×错键数)]`，目标键数2时m=1.08、至少3时m=1.15、其余m=1；正确键内部50%键程容差/50%实际触发保持。此次从头训练，预算512×19532=10,000,384，cuda:0采样/cuda:1学习/流水；具体学习率与gamma直接读取根脚本，不从旧对照恢复。

用户于2026-10-04随后指定统一使用根目录 `all_use_this_train.sh`。参数集中在该文件的提交命令中，默认采用第⑤组 `S4_false01` 参数并将掌部碰撞替换为全手 `tsensei_collision_v1`，从头训练；可以直接修改脚本或追加覆盖参数：

```bash
bash all_use_this_train.sh
bash all_use_this_train.sh --actor-lr 0.0001 --seed 43
bash all_use_this_train.sh --vector-steps 10000 --output-dir /absolute/path/my_run
```

脚本按用户最新要求自动调用 `submit.sh --gpus 2 --source-archive ... train`，不必再外包一次submit。提交后输出目录自动追加时间戳。默认奖励为归档内 `configs/experiments/twinkle_20hz_20261004/S4_false01.json`：五项原奖励+错键下压.1，目标键实际激活.5，同手惩罚关闭，无dt缩放。学习/模型/时序/控制/选模参数在脚本中显式设置；每500步评估、完整10M预算与原S4一致。

用户随后明确要求把最新统一入口的旧掌部模式换成全手模式。现固定新的 `logs/hpc/tsensei_collision_20261004/source_v1.tar.gz`，SHA以根脚本内校验值为准；`--task-profile tsensei_collision_v1`使forearm全手跨手接触0/−.5，S4奖励JSON无需改写历史内容，params为空时采用新任务基础/子步公式。原谱、其他奖励/控制/学习参数保持；最新统一入口改用双卡流水，不能称与历史S4完全相同。工作区Python修改不会进入固定归档；后续改算法须验证并更新归档及脚本SHA。独立部署携带此项目内归档，自定义MIDI/JSON使用绝对路径。

`scripts/hpc/train_coupled_20hz.sh --submit-s4`兼容转发到统一入口；原双卡底层入口保留兼容用途，不再作为新参数维护位置。真实shell→Slurm提交捕获→原归档包装链→source/learner CLI的CPU核验通过：21项训练配置、4项source recipe与S4保存值一致；末尾覆盖/单卡提交正确、CUDA未初始化。证据 `logs/hpc/s4_saved_entry_20261004/acceptance.json`。本次没有提交GPU作业或开始新训练。

最终全手替换验收：真实根脚本提交链与S4实际config的21项训练区段相同，source recipe仅task_profile改名；归档166个runtime文件与当前源码一致。Slurm1617以根脚本同一固定新归档执行，20秒COMPLETED0:0，4项专项通过，包含S4模式forearm最终0/−.5。没有开启新的SAC训练；[范围图和完整验收](tsensei_collision_v1_20261004.md)。

2026-10-04随后按用户要求切为双卡：根脚本申请2卡，`--device cuda:0 --learner-device cuda:1 --collector-pipeline`。仅改变执行布局，完整训练预算、模型、奖励和固定归档保持。CPU假sbatch捕获确认只有GPU数量、学习设备、流水开关三个差异；未提交新作业。此前单卡参数核验和GPU碰撞验收作为历史证据保留，当前双卡具体吞吐与F1仍需实测。

### 本轮实验启动时的历史入口

新入口为`scripts/hpc/train_coupled_20hz.sh`，`submit.sh train`使用该入口。旧文件名`train_coupled_40hz.sh`仅兼容转发，显式末尾覆盖40/400/20仍可用。对历史归档，job.sbatch仅在归档内没有新入口文件时使用原40hz入口；新入口执行失败不回退重训。已归档的旧训练入口和检查点不变。普通`run.sh`原本就是20/200Hz。

## 实验与固定条件

| 组 | 相对哪组 | 唯一设计差异 |
| --- | --- | --- |
| S0_shadow_endpoint | 20/200Hz基线 | Shadow原五项、控制周期末评分 |
| S1_shadow_substep | S0 | 按键/踏板按200Hz子步评分，能耗均值、掌部接触并集 |
| S2_activation05 | S1 | 目标键正项50%连续行程＋50%真实触发 |
| S3_self02 | S2 | 仅开启同手碰撞惩罚，权重.2 |
| S4_false01 | S2 | 仅开启错键下压惩罚，权重.1 |

五组当前指法仍为控制末端评分；不称所有奖励都200Hz。未来指法信息保留为观察，提前指法奖励关闭，humanoid/分阶段/其他正则均关闭。所有JSON `scale_by_dt=false`，五个原奖励外部weight1，原forearm无碰撞+.5/有碰撞0，不混入旧HPC负碰撞配置。S3/S4明确独立选择用户已授权对比的候选，S0保持上一轮关闭所有附加项的基线。完整配置位于`configs/experiments/twinkle_20hz_20261004/`。

每组512环境×19532步=10,000,384样本，seed42、Actor/Critic256×3、三LR3e-4、batch1024、每样本.25更新、replay2M、初温.5、gamma.75。频率降低后同样样本覆盖的物理时间及gamma折扣时域改变，不声称与旧40Hz轨迹或学习问题等价，也不静默平方gamma。183步整曲预填=93696样本，第93697条开始更新，完整预算2,476,672次学习更新。9999条随机探索、前瞻/准备/曲尾秒数保持。

统一每500向量步/3回合评估，比旧40Hz每1000步保持每环境相同的评估间隔秒数；只保留forward/best/final。为保证相同样本预算，将连续达标早停次数显式设为19534，本轮不会因某组短期达到阈值而缩短预算；依然报告是否满足原质量条件。best按回合最小200Hz精确原谱micro F1及键底0资格选择，另报告合并F1和所有接触诊断。新增采样最高分不等于多训练种子最优或物理安全。

## 资源、验收与自动汇总

所有GPU执行通过Slurm的`scripts/hpc/submit.sh`，每组一张A100、同卡采样学习，无双卡流水，最多四组并行，第五组等待空闲。正式训练前分别验证原末端基线和新增碰撞子步组。固定归档位于`logs/hpc/twinkle_20hz_20261004/`：

| 用途 | 归档 | SHA256 |
| --- | --- | --- |
| 两组GPU短训 | source_v1.tar.gz | `0ed7fbd2870a4c0b1cf54a64dbdedb20eed834e1ad397147fe3b360524846a4f` |
| 正式五组 | source_v2.tar.gz | `03741f41799c6f3c68b86641ea7493fd1b72e3687f2bd648249ec5c05d7a803b` |

v2补入CPU协调器/测试/说明和旧归档入口兼容，逐文件校验39项学习源码及167项source_runtime散列与实际smoke一致；模型/任务/学习/奖励代码不因打包变更。完整manifest SHA `80a88db58a4658c51d3057935311bad0e57e3306cd8c7b2c97a8d0874958d5e4`，协调器SHA `932c4e7123dbe3fd8ef70beb527abac6a78c04f6f125d3fe26da023b985e9baa`。后续工作区编辑不进入归档内训练。

短训 **1562=S0、1563=S3均COMPLETED0:0**，各512×240=122880样本、7296次更新，耗时约2分24秒。实际20/200Hz、1407观察/31动作、183帧及指定奖励均正确；六份forward/best/final保存文件和完整诊断通过，CUDA仅在Slurm分配内执行。末次F1分别0和.032777，只属功能验收，不代表收敛效果。`smoke_acceptance.json`、`coordinator_actual_smoke_acceptance.json`记录完整结果。

实际源构建还确认34音原始时间/指法逐项保持，模型钢琴solref随physics_dt变为`.01`，物理控制各10子步。入口/五组奖励/协调器联合47项CPU测试及43子测试通过；新增旧归档fallback在14项/38子测试中通过（批次有重叠），验证缺新文件才回退、运行错误保留退出码。两组真实smoke配置、全部指标流、完整结束检查和weights_only CPU检查点加载也通过协调器代码验收。

**2026-10-04 13:01:14启动CPU协调器，PID/session683184**。实际五组状态文件是`training_mjlab/outputs/hpc/twinkle_20hz_comparison_20261004/coordinator_state.json`；13:01:33时：

| 组 | 作业 | 状态 |
| --- | --- | --- |
| S0_shadow_endpoint | 1564 | RUNNING，初始化 |
| S1_shadow_substep | 1565 | RUNNING，初始化 |
| S2_activation05 | 1566 | RUNNING，初始化 |
| S3_self02 | 1567 | RUNNING，初始化 |
| S4_false01 | 尚未提交Slurm | 协调器待办，先四组并行，空卡后提交 |

13:04:09曾核对S0–S3四组config/340步174080样本20096次更新正确。但随后首轮被协调器错误中止，不能将这一早期状态称持续正常运行。

**首轮核验错误与修复**：1564–1568均在第二回合第366步被自动检查误判后取消。训练器先写`episodes`行再执行当步学习，故该行23296次更新正确对应365步；旧检查却要求366步后的23424。现有SAC日志有限、无CUDA异常，没有发现学习器数值故障。修复只让纯episodes行严格检查step−1，train/eval仍严格检查step，不放宽任意偏差。五份真实失败日志修复后全部通过，旧输出/状态完整保留并标为失败尝试，不纳入完整预算排名。缺陷和修复验收在`first_launch_audit_failure.json`、`coordinator_episode_update_acceptance.json`以及独立审计`coordinator_episode_phase_independent_audit.json`。

修复后协调器24项单测通过，包含第366和3660步同一步先episodes后train的记录；联合入口/奖励/回退共52项、43子测试通过。新增**1569=S0、1570=S3的600步多回合验收**，由修复版协调器全流程执行，覆盖366/549两个学习中的回合结束边界，并包含训练结束/检查点/自动独立复评。源v3 SHA `3873a973b36368afd300e055d7ddaf1fbe5264902b6e04405c8846cbbf2895f2`，学习器/模型源码散列仍与最初两组smoke相同。该短验收输出为`training_mjlab/outputs/hpc/twinkle_20hz_validation600_20261004/`，不纳入五组正式排名。

修复版正式启动使用新`manifest_v2.json`（SHA `e0dfa2318ae08950d6343809467d1c0ddaf2fd7202ba5086793b8efcd5b57889`）和新输出根`training_mjlab/outputs/hpc/twinkle_20hz_comparison_20261004_v2/`，协调器SHA `e7682409f6a19a61750ef9680e543eb0ad1b24e4fc1753c064eb60c5f100b8cb`。原首轮目录不覆盖。

**重新启动前的完整验收已通过**：1569/1570正常完成600步/307200样本/53376更新及4个评估点，366/549回合日志、train/eval、结束计数和best/final合同全部通过；自动复评1571/1572各10回合正常结束，最终排名和模型引用生成。S0 best=final600去重复评F1=.339818、键底0；S3 final600键底12不进入复评，best400复评F1=.197892、键底最大105，进一步被独立复评资格检查排除。54项独立验收在`actual_long_smoke_acceptance.json`。这里是早期功能/流程验收，不作为五组长期收敛结论，也不称短策略物理安全。

**正式重启时间2026-10-04 13:18:29，PID/session710615**。前四组S0/S1/S2/S3分别对应作业 **1573/1574/1575/1576**，13:19实际config全部通过并开始采集；S4保持协调器待办，前四组完成空出卡后提交。当前有效进度和排名均使用上述`_v2`根目录，首轮无后缀目录明确为误判取消的历史失败尝试。

13:22:04独立启动复核通过：四组均实际跨过第366步，已至少440步/225280样本/32896更新，config与各自完整JSON和1407/31接口一致；S4仍待办，无核验错误。`startup_acceptance_v2.json`保存新有效启动证据，当前进度持续由`_v2/coordinator_state.json`更新。

使用`--max-parallel 4 --replay-top-k 10 --detach`运行，不依赖本次对话终端存活。提交前保存请求和回执；有歧义的提交不自动重投，异常只处理自己记录的作业，不干扰其他用户作业。恢复必须同一manifest与协调器SHA；运行中不修改协调器文件。

对照协调器在CPU读取实际config/metrics/summary/checkpoint，核对归档/模型/奖励/预算与精确F1覆盖。训练后对全部合格best/final（同组同一步去重，最多10个）独立复评各10回合，最终按精确合并micro F1排序，最低回合F1及接触作为次序参考，键底0资格保持。实时与独立复评排名分别写`leaderboard.json`，仅在全部训练及选定复评正常完成且存在复评合格候选后写`best_model.json`，其中保存模型/config/源归档路径及散列，不覆盖旧模型。最高F1只能指本轮实测候选，0.99/1仍须结果证明。

200Hz原谱全部34音仍有覆盖，三个短音各约12/10/14个物理采样；20Hz末端各只有一个目标帧。新实验内部用相同200Hz评分，旧400Hz结果单列。旧轨迹离线抽样到200Hz可作评分参考，不能称旧策略已在新20/200Hz物理环境复测。

TODO：正式五组完整预算与独立复评排名、200Hz键底/手接触/误触、早松/迟松/迟按分解、多训练种子与.99目标验收。

## 2026-10-05：paper/chord最佳模型继续训练三组奖励对照

用户指定从本轮训练继续开启闲指姿态、同手自碰撞及二者组合。三组均从Slurm1620的best15500（PT SHA `080b01d3608bdb2b6d8442abff4a31dbeec1b0e6d9f14a7cb2bfa19c3fe74956`）加载同一网络，统一调用根目录`all_use_this_train.sh`并在末尾追加参数；默认脚本未修改，未创建新训练脚本。

| 组别 | Slurm | humanoid idle posture | self collision |
|---|---|---|---|
| idle_only | 1681 | 开启 | 关闭 |
| self_only | 1682 | 关闭 | 开启，weight=.2 |
| idle_and_self | 1683 | 开启 | 开启，weight=.2 |

北京时间08:00:13核验：1681/1682 RUNNING，各102400样本/2048次更新、指标有限，实际config和父网络加载合同通过；1683 PENDING(Resources)，待分配后核验实际config。实际startup证据保存在上述证据目录的startup_acceptance.json。

闲指项采用归档现有`idle_posture_penalty_v1`：weight=1、maximum_penalty_per_step=.1、sigma=.25。每个控制步扣 `-.01 × Σ闲指 [mse/(mse+.25²)]`，mse为每指四个实际关节相对场景origin参考的均方角度偏差；当前用指及未来.15秒内用指豁免，固定十指分母，全手最多-.1/步。只评姿态，不启用几何保护/投影。self项继承本轮已保存weight=.2，每个去重同手body对每控制周期扣-.2；200Hz子步取整个周期并集，多接触点/多子步不重复。跨手仍独立0/-.5。

其他参数与父训练一致：512环境、20/200Hz、31动作/1407观察、原曲/指法、gamma=.8、三LR3e-4、seed42、双卡采样/学习流水；每组新增19532向量步／10,000,384样本，评估每500步×3回合。false_press仍关闭，按键paper/chord_scope=on及正确键真实激活占比.5保持。阶段仍为1。

奖励变更使用`--finetune-from ... --allow-reward-change`：保留actor/critic/target_critic网络，重建优化器、温度、随机数、回放池和物理状态；不是原训练逐状态连续恢复。严格保留scene/task/源码/动作/观察/控制检查，未传`--allow-control-change`。父PT、奖励JSON和源码SHA已保存并核验；启动前打印每组完整启用奖励及performance选项。

配置在`configs/experiments/paper_chord_reward_ablation_20261005/`；实际提交读取`logs/hpc/paper_chord_reward_ablation_20261005/`内只读奖励快照和父PT副本，防止排队期间编辑实验配置影响提交。该目录同时保存request、三组实际参数链捕获、固定归档CPU构建/effective配置、entry_acceptance及逐组提交回执。

实际提交命令（已执行，勿为查看状态重复提交）：

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --finetune-from /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_reward_ablation_20261005/parent_best.pt --allow-reward-change --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_reward_ablation_20261005/idle_only_rewards.json --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_finetune_idle_only_20261005
```

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --finetune-from /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_reward_ablation_20261005/parent_best.pt --allow-reward-change --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_reward_ablation_20261005/self_only_rewards.json --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_finetune_self_only_20261005
```

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --finetune-from /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_reward_ablation_20261005/parent_best.pt --allow-reward-change --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_reward_ablation_20261005/idle_and_self_rewards.json --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_finetune_idle_and_self_20261005
```

三组训练结果待预算结束后按同频率精确原谱F1、同手/跨手接触、底部接触及吞吐比较；新增负罚会改变return，不能以return直接排名F1或宣称已安全。没有额外启动未请求的继续训练基线，父1620指标保留作参考。



## 2026-10-05：续训前基线关闭和弦，从头训练1724

基线为Slurm1620，原完整配置位于`training_mjlab/outputs/hpc/paper_chord_on_fullhand_20261005-20261005-004515-261777/config.json`，实际命令为`logs/hpc/1620/command.txt`。后三组1681/1682/1683均由它的best15500网络续训。本次按用户“续训前基线”要求采用1620原始训练条件，从头初始化，不加载后三组网络，也不添加闲指姿态/同手碰撞惩罚。

唯一奖励差异是`key_press_reward.params.chord_bonus: true → false`和`chord_scope: "on" → null`；null是归档解析器支持的无范围值，不能写成字符串"off"。paper公式、实际触键占比.5、其余奖励和权重保持；不按dt缩放。当前启用key_press/sustain/energy/fingering/forearm五项，权重均1，其中forearm在tsensei_collision_v1中为全手跨手接触周期并集惩罚−.5，无接触0。

已更新唯一入口`all_use_this_train.sh`默认奖励及输出名，独立JSON为`configs/rewards/paper_chord_off_20261005.json`。保持gamma=.8、三LR=3e-4、seed42、512环境、19532向量步（10,000,384样本）、每500步3回合评估、双卡采样/学习流水、原小星星和指法、20/200Hz、1秒准备段及固定ae7e29f3源码包。原1620基线命令/完整config/入口快照及实际逐字段差异保存在`logs/hpc/paper_chord_off_20261005/`。

正式启动前打印全部奖励及performance，使用只读JSON快照保证本次提交可复核；14:47:40经以下原入口提交Slurm1724：

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_off_20261005/submitted_rewards.json
```

归档奖励解析器CPU校验、真实入口假sbatch捕获及与1620实参比较通过；除奖励JSON和输出路径外实参完全相同，未传checkpoint恢复选项。长期F1需等训练评估，不预先判断关闭加成是否改善。


## 2026-10-05：关闭和弦基线的三组奖励续训对照

用户要求以1724为基线，增加与此前相同的对比训练。1724尚在运行，因此三组均设置`afterok:1724`，等待基线成功结束、best.pt停止更新后，自动加载同一个最终选模网络；不会加载提交时的中途best。14:52三组已提交并核实PENDING(Dependency)，不是已开始GPU续训。父路径来自本轮实际输出目录，未来最终PT散列/父步数由各组启动的checkpoint_origin记录；当前不伪填最终散列。保留严格源码/模型/动作/观察/控制检查，仅显式允许奖励变化。

| 对照 | 作业 | 闲指姿态惩罚 | 同手碰撞惩罚 | 追加样本 |
|---|---:|---|---|---:|
| idle_only | 1725 | 开启 | 关闭 | 10,000,384 |
| self_only | 1726 | 关闭 | 开启 | 10,000,384 |
| idle_and_self | 1727 | 开启 | 开启 | 10,000,384 |

三组保留chord_bonus=false、chord_scope=null、paper公式及全部五项基础奖励。姿态沿用idle_posture_penalty_v1、weight=1、maximum_penalty_per_step=.1、sigma=.25；同手碰撞weight=.2，每去重body对每控制周期−.2。与此前1681/1682/1683对应JSON逐项比较，仅和弦两个字段不同；与本轮1724相比仅指定附加奖励项不同。三组从父网络加载actor/critic/target critic，重建优化器、温度、随机状态、回放池及环境，保持上一轮相同续训方式；不称完整学习器状态恢复。

仍使用all_use_this_train.sh末尾参数，无新训练脚本。为等待父基线，统一入口新增开头`--dependency EXPR`，透传给既有submit.sh，不进入训练解析器，也不改变无依赖时的默认训练参数。各组双卡流水、19532×512预算、gamma=.8、三LR3e-4、seed42、每500步3回合评估和固定ae7e29f3归档保持。

配置在`configs/experiments/paper_chord_off_reward_ablation_20261005/`，提交使用`logs/hpc/paper_chord_off_reward_ablation_20261005/`内只读奖励快照。该目录保存父完整config、真实参数捕获、请求/回执、逐组完整奖励打印及Slurm依赖读回。CPU归档解析/对照参数一致性、普通入口无依赖路径、缺失依赖参数拒绝和shell语法通过；待获配后再验收实际config/父PT/数值与完整训练结果。父作业失败时afterok不会启动子作业。

实际提交命令：

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --dependency afterok:1724 --finetune-from /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_off_fullhand_20261005-20261005-144747-246686/best.pt --allow-reward-change --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_off_reward_ablation_20261005/idle_only_rewards.json --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_off_finetune_idle_only_20261005
```

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --dependency afterok:1724 --finetune-from /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_off_fullhand_20261005-20261005-144747-246686/best.pt --allow-reward-change --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_off_reward_ablation_20261005/self_only_rewards.json --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_off_finetune_self_only_20261005
```

```bash
bash /home/houj/projects/tsensei_v3_2_pianist/all_use_this_train.sh --dependency afterok:1724 --finetune-from /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_off_fullhand_20261005-20261005-144747-246686/best.pt --allow-reward-change --reward-config /home/houj/projects/tsensei_v3_2_pianist/logs/hpc/paper_chord_off_reward_ablation_20261005/idle_and_self_rewards.json --output-dir /home/houj/projects/tsensei_v3_2_pianist/training_mjlab/outputs/hpc/paper_chord_off_finetune_idle_and_self_20261005
```
