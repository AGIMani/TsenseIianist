# V3.2 按本仓库 Shadow 同步（2026-09-25）

## 授权与范围

用户要求当前项目的参数和碰撞方式同步完整 Shadow 训练，并明确“同步之后能够不再出现爆冲即可，之后你直接开启训练”。新实验从头训练；保留 MJLab/Torch 并行后端、V3.2 关节结构/尺寸/坐标及已确认腕范围，不用 Shadow 的手模型覆盖 V3.2。碰撞反力保留。此前版本已归档到 `training_mjlab/source_versions/before_shadow_sync_20260925.tar.gz` 及同名 JSON。

同步基准是本仓库 `../robopianist-rl/run.sh`、`train.py`、`sac.py` 和 `../robopianist/robopianist/` 任务源码。注意本仓库 Shadow 常量所指向的 third_party XML 目录缺失；实体参数参考机器已有 `/home/houjue/DaxianPianist/robopianist/robopianist/models/hands/third_party/shadow_hand/` 原生资产，并记录来源。新项目运行不导入这些路径。审计合同见 [JSON](shadow_reference_contract_20260925.json)。

## 参数变化

| 项目 | 此前入口 | Shadow 同步入口 |
|---|---|---|
| Actor/Critic | 256×3/256×3 | 相同；GELU，Critic LayerNorm、dropout0.01 |
| 学习率 Actor/Critic/温度 | 3e-4/1e-3/1e-3 | 均3e-4 |
| gamma/初始alpha/tau | 0.8/1/0.005 | 相同 |
| batch/replay条数 | 4096/262144 | 256/1000000 |
| 随机探索/起学 | 76800/约256000条 | 前4999条随机，第5000条起学 |
| 每条新增样本更新数 | 0.125 | 1 |
| 512世界边界 | 每步固定64次更新 | 第10向量步121次，其后每步512次 |
| 控制/物理/前瞻 | 20/200Hz，0.5s | 相同 |
| 曲首缓冲 | 用户已于9月25日改为0 | 0，与本仓库Shadow一致 |
| 策略坐标 | direct_joint_v1混合范围 | shadow_v1统一[-1,1] |
| 八个四指MCP物理目标 | [0,1.2]rad | 保留V3适配；机械限位不改 |
| 奖励 | 原5项、自碰撞；脚本尾另开错键惩罚 | 原5项；自碰撞/false_press_penalty关闭，可自行再次启用 |
| forearm范围 | 跨手掌与手指 | 掌部代理，+0.5/0；V3没有实体Shadow前臂 |
| 指法目标 | 拇指有单独目标 | 所有指使用原Shadow琴键目标公式，指法表和手指tip site不变 |
| 腕惯性 | 仅0.171363kg手部 | TX/TY各加3kg等效平移惯量(armature)，按编译有效惯量设置临界阻尼 |
| 手指碰撞 | CAD凸包 | 按V3 CAD拟合的胶囊/圆柱；掌保留mesh |

三种 model/task profile 均显式保存到完整config.json。旧任务语义和旧驱动/碰撞模式有具名入口；源代码散列不匹配仍拒绝回放，需恢复对应源码，不能静默拿新模型播放旧checkpoint。

## 仍然不同于 Shadow 的部分

- 512个环境先一起采集再集中学习，不能等同单环境逐条交错更新。5,000,192条样本计划4,995,193次更新。按每样本预算对齐，不声称学习路径数值相同。512环境同步从曲首出发，所以前4999条随机样本只覆盖约10个控制步，而原单环境4999步覆盖多回合；这是采样覆盖差异，未擅自随机化乐谱起点或重置状态。
- V3.2双手47动作、1180观测，保留独立指关节，没有移植Shadow腱耦合；TZ/Yaw结构性移除。原乐谱指法保留。重置使用初始qpos、零qvel和neutral动作历史，未重引入旧wrapper动作历史下限问题。
- V3的Pitch轴与Shadow WRJ1不是同一世界轴，增益只能作为数值参照。Shadow前臂3kg作为上游平台参照，以TX/TY滑动关节armature表达，不增加刚体或Pitch惯量；这并非TsenseI实测硬件质量，驱动器保留V3暂定0.6Nm手指力矩上限和50N平移力限。需要硬件标定后才能用于实机。
- 掌CAD形状的凹槽不宜用单个盒子填平，故掌仍为mesh。新指尖几何按真实末端面拟合，未将CAD近端关节的最大宽度当成指腹宽度。
- 没有新加子步插值、速度硬裁剪、qpos投影或关闭手—键碰撞。原Shadow同样直接保持位置目标；训练后的轨迹平滑性必须实测。
- 原Shadow日志每1000条、评估每10000条；512并行下分别在1024/10240样本边界执行。保存间隔仍500向量步，与学习预算无关。

## 爆冲验收原则与已发现的问题

之前诊断已区分目标命令跳变和真实碰撞反弹，见 [训练运动诊断](training_motion_audit_20260925.md)。固定姿态稳定不能证明任意策略无冲击；CPU/GPU一致也不能证明接触合理。

碰撞代理第一稿按整件CAD横宽包裹，导致拇指基座过宽，旧ctrl压力回放出现831.6N，已否决。修正后非末端按MuJoCo惯性拟合、末端按实际面宽拟合，并保留失败实验结果。修正版在旧相同qpos/qvel下消除了已定位的0.700秒拇指互撞和1.645秒小指夹键，2.465秒左食指触键力由23.83N降至3.46N。但轻腕驱动在整曲旧目标回放仍产生路径分叉后的422.5N新接触峰，因此碰撞更换单项不算最终通过，也不迁移旧策略权重。

另否决了“根部下游直接挂3kg刚体”的驱动候选：Shadow的前臂在腕旋转上游，直接挂在V3根部会跟随Pitch转动，阶跃的被动Pitch峰速反而从0.582增至2.273rad/s。最终改为TX/TY各加3kg等效平移armature，保持原URDF质量和Pitch惯量，TX峰速0.532m/s，Pitch峰速0.269rad/s。临界阻尼61.68983524N·s/m，以含armature的dof_M0=3.171363kg计算。旧ctrl全曲触键峰力由该被拒方案339.27N降至59.98N。失败候选短训和原始代码片段保留用于审计，未作为正式训练。

最终需要同时验证新驱动+碰撞的初始接触、三类保持、旧物理ctrl压力、CPU/GPU、短训及源码配置回放。周期评估新增motion指标跟踪目标跳变和速度；这些是20Hz末帧采样值，不等于200Hz子步峰值。正式训练进展与验收结果在下节记录。

## 验证与运行结果

- 最终模型合同测试5项、碰撞代理测试3项在Composer环境通过；源码重建、Shadow任务/动作、预算、SAC、旧映射共25项在MJLab环境通过。第一次混用环境调用碰撞测试因Torch环境没有dm_control导入失败，随后在正确Composer环境3项通过；这是测试环境选择问题。
- 完整源码配方显式覆盖shell中相反的profile设置，删除临时编译目录后再重建scene/task散列相同；未新建持久化冻结场景。run.sh语法检查及所有CLI名称在文件头覆盖检查通过。
- 同手/跨手/触键各20秒固定目标，末2秒手指速度RMS为0.0001453/0.0016846/0.0000140rad/s，初态0接触。TX150mm阶跃1.27533→0.53233m/s；同阶跃伴随Pitch0.58223→0.26915rad/s。旧ctrl整曲还有约59.98N触键峰和3.7919rad/s Pitch峰，不能据此宣称所有动作无冲击。
- 最终同模型CPU/GPU四种目标各1秒、总800子步：有限状态、按键激活0差异。静止/同手/触键的最大手指角差8.18e-6rad、平移6.22e-8m；跨手暂态最大1.50°/0.437mm，角差>0.01rad累计45ms，到1秒降至0.000227rad。两端手指峰速均4.78506rad/s，腕平移峰速均0.496841m/s。分叉发生在掌mesh开始参与接触后，但不能仅凭CPU几何查询断言GPU碰撞算法错误。严格跨手一致性仍未通过。
- 最终短训：`training_mjlab/outputs/shadow-sync-final-smoke-20260925-170357-534665`，512×40=20,480样本，4999随机样本，15,481更新，73.80秒（含初始化），Torch显存峰9.72GB（不含全部Warp）。两次158帧独立整曲评价正常结束、没有数值异常；40向量步只占一次训练回合的前段，不称为收敛。
- 保存后的源码+config+checkpoint回放重建成功，但不是逐bit轨迹复现：训练末评估F1=0.0470464，第一次独立回放F1=0.0337553；额外同进程两回合首obs/首action逐bit一致，首物理步后obs已有1.19e-7差异，后续策略反馈放大。两回合F1均0.0470464，但return仍不同。审计见`replay_audit/report.json`；本机MJLab明确记录MuJoCo Warp尚非完全确定性，约束浮点原子累加和接触次序可能引入细微差异。此证据不支持再次归咎于reset初态或权重未正确保存。保留V32-SAC-03重复性限制，没有为追求表面一致再次冻结视频动作或修改reset。

诊断原始数据见 `preview/outputs/diagnostics/shadow_sync_20260925/`；碰撞与驱动拒稿、最终版名称分开，最终采用 `drive_comparison_final.json`、`collision_with_shadow_drive_final.json`、`cpu_gpu_substeps_final.json`。

## 正式训练已启动

北京时间2026-09-25 17:11:19，已从头启动：

```bash
bash run.sh --output-dir training_mjlab/outputs/sac-twinkle-shadow-sync-5m
```

实际目录：`training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333`。launcher PID/进程组2620645；启动记录`training_mjlab/shadow-sync-5m-launch-20260925-171119.json`，启动日志`training_mjlab/outputs/shadow-sync-5m-startup-20260925-171119.log`。已确认进入SAC更新并完成独立评价；不是仅生成命令。

预算512×9766=5,000,192条，计划4,995,193次更新。保存每500向量步；新配置的更新次数比原入口多，不能沿用旧5M耗时。退出聊天不停止该独立进程。完整配置在run目录config.json；最终保存final.pt，周期保存step-N.pt，使用`bash eval.sh --checkpoint <实际检查点路径> --mode preview`或render回放。

最终源代码归档`training_mjlab/source_versions/shadow_sync_20260925.tar.gz`及散列JSON，旧版归档`before_shadow_sync_20260925.tar.gz`。这是源码版本备份，不是冻结scene/task；回放仍从对应源码和完整config重建。日志见[change_log.md](change_log.md)。

## 参考与未解决事项

参考：上列Shadow源码/资产；`models/hands/{urdf_model,tsensei_hand,tsensei_hand_constants}.py`；`mjlab_env/{source_runtime,action_mapping,task_bundle,task,scene}.py`；`rl_torch/{train,sac,replay}.py`；前述运动诊断、碰撞代理诊断和本轮测试文件。

论文§2.7/表2（pp.8–10）的batch4096、critic/温度1e-3以及0.125 update/transition作为历史方案保留；用户本轮已明确选择Shadow而非论文SAC参数，不重复请求批准。论文§2.6(p.8)归一化动作、§2.9(pp.10–12)奖励与当前有重合，但本轮采用可查证Shadow源码规则，没有猜测论文未知映射或Stage2 shaping。

保留TODO(V32-04)驱动力/惯性硬件标定、TODO(V32-06)几何及掌部碰撞验收、TODO(V32-SAC-03)长期轨迹与GPU接触一致性。短训通过不代表收敛，更不能保证所有后续策略永不撞击。
