# 可选论文奖励实现验收

日期：2026-09-18。主说明：[rew.总表／字段与命令](reward_design_review.md)。

## 1. 实现与用户决定

论文pp.10–12已公开公式均接入Torch任务，含按键／指法、前臂／腕／手指的功率／命令速度／命令加速度、手指超速、碰撞、自然姿态、腕pitch、前向按压，以及总奖励乘dt和和弦倍率选项。保留四个项目兼容项，共20个可开关项。

用户明确要求未确定的映射必须亲自填写后才能启用。因此正式配置模板的动作单位、接触类别、自然参考、左右pitch方向、十指等效角和和弦作用范围均保留null或关闭；绝不把下面测试中的合成映射提升为默认。原命令保持原五项规则。

## 2. 实际测试结果

环境：本机`training_mjlab/.venv`，Torch2.10.0+cu128，MJLab1.6，MuJoCo Warp3.11，Warp1.17，RTX PRO6000 Blackwell。

| 验收 | 实际结果 |
| --- | --- |
| 公式／配置／兼容／GPU接触单元检查 | 26项通过，见[测试日志](../training_mjlab/outputs/reward-validation-20260918/tests.log) |
| GPU任务＋17分项 | 2环境、6控制步，通过CPU／GPU逐分项对照，最大绝对误差`2.9802322e−8` |
| 局部reset | 第4控制步计分索引为`[0,3]`；仅reset环境的两步命令历史回到neutral，另一环境历史保持 |
| legacy默认 | 等价current配置选择原执行分支，仍为5项，未创建碰撞跟踪器 |
| Stage1短训练 | 4环境×6向量步，24transition、10次更新，8次时间截断reset、两次独立评价和检查点保存 |
| 显式Stage2跨奖励转移 | 4环境×2步，8transition、新增2次更新；总update从新优化器的0开始，最终2；来源记录原10更新及旧奖励配置 |
| 新检查点完整有声回放 | 158控制步、15800音频物理子步，分项数组158×17，逐帧求和等于总reward；视频8.95s，1280×720 H.264＋AAC |

### 单元检查覆盖

- 论文0.6/0.4按键组合、指数错键数量、两种显式和弦作用范围、静音空目标；活跃指平均、1cm成功区、10cm margin和空指法帧。
- 负功率不计罚，6／4／40动作分组和踏板排除；三组首步速度／加速度的解析值、恒定命令后的二阶差分、独立reset、超速阈值。
- 明确的合成姿态、腕方向与等效角值；阈值比较与body pair去重；同手、跨手、琴箱、正常琴键排除和显式忽略项；自碰撞在自定义dt配置中只计一次。
- 原始值、外部加权和dt缩放后的总和；新reward进入下一观测和episode return；关闭全部项时reward为0。
- 缺失映射、非法字段、非有限数值、重复能耗／碰撞配置被拒绝；20项模板以及表3三个阶段权重。
- 旧检查点兼容；新检查点保存完整配置；普通加载拒绝奖励变化；只有显式networks-only转移才允许换奖励，重置优化器／alpha。

CPU／GPU对照在相同GPU物理状态上比较奖励算术，用于检查批处理／设备差异；具体公式的独立解析值由单元检查提供，不能把同代码跨设备对照单独当作论文语义证明。

## 3. 产物与复查

- [GPU对照报告](../training_mjlab/outputs/reward-validation-20260918/report.json)
- [Stage1短测](../training_mjlab/outputs/reward-validation-20260918/smoke_stage1/summary.json)
- [Stage2短测](../training_mjlab/outputs/reward-validation-20260918/smoke_stage2/summary.json)
- [新配置回放报告](../preview/outputs/reward-options-smoke-20260918/report.json)
- [仅检查示例](../training_mjlab/outputs/reward-validation-20260918/inspect_example.json)：paper Stage1按键／指法加三个正功率项，没有启用任何未填写映射的项。

下面的验收脚本会在**新的输出目录**内生成标有`test_fixture_only`的配置，用于公式和接口测试，不修改正式模板，也不启动长期训练：

```bash
training_mjlab/.venv/bin/python scripts/validate_reward_options.py \
  --output training_mjlab/outputs/reward-options-recheck

training_mjlab/.venv/bin/python -m unittest \
  tests.test_paper_rewards tests.test_mjlab_task tests.test_torch_sac \
  tests.test_acceleration_and_collision -v
```

## 4. 边界与未解决事项

- 测试用腕方向和单关节等效角只是算术fixture，既未标定也未获用户批准；用户真实映射仍待填写，不能直接用于正式实验。
- 没有运行新的5M训练，没有证明新奖励提升F1、降低碰撞率或具备实机安全性；测试回放是弱训练策略，不是新的最佳演奏视频。
- 没有改控制频率；开启scale_by_dt使用实际0.05s，不是论文0.025s。观测仍为1180维，第二步命令历史只在任务内部维护。
- Stage2 shaping未公开（V32-PAPER-U01），只提供明确选择identity的路径；作者动作单位（U02）和模型映射（U05）仍是资料缺口。
- 延续当前模型物理接触／速度参数和镜像右手限制，未进行硬件、长时收敛或人工GUI交互验收。
