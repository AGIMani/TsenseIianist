# 逐关节PD与拟合摩擦迁移（2026-10-23）

来源：AGIMani/a3-hand-pd-sim 的 resistance-model 分支，提交 e91ea8870dd5431e88cc4e565813bbaf4a1597a1。读取 baseline_right.json 与 resistance_right_20j_20261020.json 的参数数值；本项目独立实现每子步更新，不导入或运行外部工程。

用户确认默认补偿全部有效重力，保留拟合摩擦。当前配置要求完整重力补偿：手部各body gravcomp=1，手指joint actuatorgravcomp=false，使补偿在电机限矩之外；alpha仍记录来源值，但有效重力被完全补偿，alpha不额外施力。显式关闭重力补偿会拒绝该配置，防止未补偿却误用本合同。

每个手指关节：u=clip(Kp*(q_target-q)-Kd*qvel, -effort, effort)。Kp=2，Kd逐轴来自源基线，armature=0.0001 kg·m²；MCP力矩±0.6、四指PIP±0.48、DIP和拇指PIP±0.33、侧摆与拇指两旋转轴±0.344 N·m。Kd属于执行器PD，限幅外的被动黏性阻尼为拟合b。每个物理子步在当前实际姿态计算名义重力矩G，干摩擦阈值C=T0+kg*abs(G)，交给MuJoCo静/动摩擦约束求解。不能把Kd和b简单合并进被动阻尼，否则饱和行为不同。

G由该轴下游各刚体当前世界质心、质量、关节世界轴/锚点和世界重力求矩，不使用含科氏项的运动中qfrc_bias。GPU阈值为每环境独立数组，不跨环境共用；CPU Composer before_substep使用相同公式。直接调用裸mj_step的离线工具应在每子步先调用finger_resistance.Resistance.update；未调用的工具不属于此动态摩擦验收路径。

小指源旧名按功能转换：rotaback→MCP，MCP→swing，swing→PIP，PIP→DIP。左右各20轴用独立的运行时阈值，沿用同一右手拟合系数；G按当前各自的独立URDF模型计算。模型几何、源惯量、关节限位、腕驱动、20/200Hz、Euler、动作/耦合、乐谱和奖励公式保持。

全部系数唯一存于tsensei_v3_2_pianist/assets/finger_resistance.json。原仓库PD为仿真先验；阻力来自单条右手轨迹，左手未独立标定，无名指动态参数未充分激励。这次迁移不是当前新手模型的重新硬件标定，不能承诺响应更快。

验证记录：logs/resistance_drive_20261023。CPU当前完整场景40轴参数、Jacobian独立重力矩对照与短步进；GPU通过Slurm执行多环境阈值与短步进检查。最终通过状态以日志与冻结来源记录为准。
