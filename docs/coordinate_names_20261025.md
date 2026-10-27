# 初始世界位置参数与掌根命名

2026-10-25按用户要求统一位置追加参数。世界W与当前钢琴整体P重合；X朝演奏者，Y低音到高音，Z向上。位置单位为米，准备抬升之前的掌根指 `right_/base_link`、`left_/base_link` 的实际原点，既不是质心，也不是安装基准或法兰。

## 当前唯一入口

```bash
bash all_use_this_train.sh \
  --world-x 0.16 --right-world-y 0.15 --left-world-y -0.15 --world-z 0.09 \
  --palm-base-yaw 0.2617993877991494
```

以上即当前默认位置，未启动训练。命令末尾可覆盖，旧`--height/--ty/--yaw`在资源申请之前明确报错。`--palm-base-yaw`单位rad，是明确命名的旋转关节角，并非世界欧拉yaw；本次统一世界坐标的是初始**位置**接口。

| 旧名 | 新名/行为 | 说明 |
|---|---|---|
| `--height .09` / pose.height | `--world-z .09` / pose.world_z | 初始实际掌根世界Z，内部安装Z取此值、Tz初值0 |
| `--ty .02` / pose.ty | `--world-x .16` / pose.world_x | 数值必须换算：world_x=.18−旧Ty，不能只换标签 |
| 隐含Tx=0、安装Y=±.15 | `--right-world-y .15`、`--left-world-y -.15` | 分别填写左右初始世界Y；内部Tx=所填Y−对应安装Y |
| `--yaw` / pose.yaw | `--palm-base-yaw` / pose.palm_base_yaw | 保留原关节角及旋转定义 |
| `forearm_tx/ty/tz/pitch/yaw` | `palm_base_tx/ty/tz/pitch/yaw` | 五个由项目生成的虚拟关节；轴、顺序、行程和驱动不变，非URDF内部关节 |
| `FOREARM_DOFS` / forearm_dofs | `PALM_BASE_DOFS` / palm_base_dofs | 同步构造参数及动态前缀分类 |
| height-offset / depth-offset | world-z-offset / world-x-offset | 轻量预览CLI的安装偏移明确标注世界方向 |
| 保存字段observed_wrist_world_* | observed_palm_base_world_* | 实际掌根世界位姿，区别于ctrl目标 |
| 新导出wrist_poses.* | palm_base_world_poses.npz/csv | 新NPZ附frame、body_frames、单位和旋转表示；旧文件不重命名 |
| 新报告tcp_frame指手根 | palm_base_frame | 与项目法兰、SDK法兰和工具尖端区分 |

大小写不同的`palm_base_tx`（关节）与`R/L_Palm_Base_Tx`（控制通道）是一一对应的对象名，不是两套空间坐标系；ctrl目标与qpos实际位置仍须区分。

## 内部换算与带臂

安装参考XY保留右(.18,.15)、左(.18,-.15)。无臂内部滑轨正向仍为Tx→世界+Y、Ty→世界−X、Tz→世界+Z；例如世界X=.20时Ty=−.02，右世界Y=.11时Tx=−.04。不能把动态`Palm_Base_Tx`控制通道解释成世界X。

新recipe带`pose_contract=initial_palm_base_world_xyz_v1`，只接受完整明确的世界位置字段。source_pose分别记录mount_world_xyz_m、initial_world_xyz_m和initial_palm_base_joint_q，不再用right_position/right_ty猜含义。旧recipe须使用原匹配归档，当前源码拒绝静默升级。

带臂安装反算现在也使用所填的实际初始世界XYZ。移除虚拟滑轨后，其Cartesian零请求仍对应这一个安装位置；原流程清零滑轨时忽略Ty偏移的问题被修复。该模式的动态动作仍为`p_rest+[Tx,Ty,Tz]`世界XYZ**位移**，与初始位置的绝对世界坐标不同。无臂与带臂仍有不同动作合同，本次没有交换或旋转控制通道。

## 其余同义名称与保留边界

- `wrist`在许多内部控制限速/变量名中仍是“掌根控制”的功能类别；已有`--wrist-*-max-*`和`--disable-wrist-*`不是新的坐标系。显示、保存的实际位置及新轨迹文件已使用palm_base。未把数百个功能名称泛化成世界XYZ。
- `base_link`是模型外部掌根body的规范名，保留以对接URDF；不改手或臂内部坐标。
- `T_piano_hand`是钢琴系完整掌根位姿，当前W=P时数值重合，但不能删除其参考系含义。相机、接触系、琴键系、法兰与掌根也不能合并。
- `forearm_reward/forearm_collision_*`属于现有全手跨手碰撞奖励/诊断合同，不是五个虚拟关节的名称，保持奖励公式和配置。
- 日期固定的凸包/自由度研究`piano_dof_contact_study.py`、原录制分析`analyze_ik_tracking.py`、两份20261013补偿脚本及固定9299ab归档的手型图使用其原数据/原代码命名，未改历史输入。通用准备轨迹工具明确区分新名和archived_forearm，拒绝混名录制。
- 日期`configs/poses/*`、旧实验JSON、旧MJB、旧归档、旧图保留原字节。`current_arm_training_20261011.json`虽带current字样，实际是已落后的日期合同，不能当作本轮新默认；旧预览须用匹配归档或显式新建的完整配置。

验证记录及冻结源码来源见`logs/coordinate_names_20261025/`；唯一入口绑定经过远端实际模型验收的归档/SHA。只编辑工作区不代表既有作业或历史回放已改变。
