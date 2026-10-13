# TsenseI V3.2 修改日志

## 2026-10-11：预览用全链路凸包与真实碰撞网格开关

- **用户要求／问题**：用户观察到单个 capsule 不能覆盖电机壳体、连接凸台和指节侧面的完整碰撞区域，要求保留原电机形状并改用简化凸包，同时在预览脚本中可以切换到真实 collision mesh 观察接触来源。
- **实现／位置**：新增 `scripts/generate_collision_hulls.py` 和 `assets/collision_proxies_convex_hull.json`；每侧除 `base_link`、`thumb_rota_block_link`、`thumb_rotaback_link2` 外的19个 link 使用完整 URDF collision STL 顶点生成的三角化凸包，三个保留项继续使用原始 URDF collision mesh。`models/hands/collision_proxies.py` 与 `models/hands/urdf_model.py` 增加 opt-in `convex_hull_v1` mesh profile，不改变当前 `shadow_primitive_v1` 训练合同。
- **预览开关**：`preview/scripts/preview_pose.py` 新增 `--collision-profile {shadow_primitive_v1,convex_hull_v1,legacy_mesh_v1}`、`--real-collision-mesh` 和 `--show-hand-collisions`。`--real-collision-mesh` 选择原始 URDF collision mesh 并自动以橙色显示；`convex_hull_v1` 同样自动显示凸包。输出 JSON 记录 profile、是否真实 mesh 以及显示状态。
- **验证／边界**：左右手各22个碰撞 geom 编译通过；凸包 profile 质量、惯量、关节和执行器数组与原始 mesh profile 一致，新增6项碰撞 profile 测试与原13项预览控制／后端测试通过，py_compile通过；`MUJOCO_GL=egl` 无臂 headless 预览分别生成凸包和真实 mesh PNG。凸包会填充凹腔，是外包近似，不等于原始 CAD 的凹槽；当前训练、源码归档、checkpoint、历史回放未切换到该 profile，尚未做长期接触／吞吐／收敛验收。详见 [碰撞 profile 说明](collision_geometry_profiles_20261011.md)。

## 2026-10-11：按当前0.952 kg手部质量修正TX/TY有效惯量

- **问题／来源**：上一条同日修改仍把已废弃的旧手部质量 `0.171363 kg` 带入 `M0`，与当前A3.2 URDF每手 `0.952 kg` 不一致。用户要求彻底移除这一旧质量假设。
- **修改**：`shadow_v1` 平移驱动显式记录 `hand_mass_kg=0.952`、`armature=0.02 kg` 和 `effective_mass_reference_kg=0.972 kg`。`Kp=300 N/m`保持不变，临界阻尼更新为 `34.15259873 N·s/m`；TX/TY力限按旧 `50 N / 3.171363 kg` 的最大加速度包络重新缩放为 `±15.32 N`。
- **验证／边界**：左右当前URDF的22个link质量求和均为 `0.952 kg`；同步更新驱动测试、当前预览配置SHA、BUNDLE manifest和驱动说明。历史3 kg/0.171363 kg记录仅保留为历史实验事实，不能作为当前默认参数；未启动训练或修改历史checkpoint。

## 2026-10-11：去除无臂模式的3 kg上游等效平移质量

- **用户要求／范围**：用户确认当前无臂模式不应模拟 Shadow 前臂，要求把 TX/TY 的等效 `armature` 从3 kg降为尽可能小且仍可控振的质量，并同步阻尼与力限。
- **修改／原因**：`tsensei_v3_2_pianist/assets/drive_settings.json` 的 `shadow_v1` 平移参数改为 `armature=0.02 kg`、`Kp=300 N/m` 保持不变。按编译有效惯量 `dof_M0≈0.191363 kg` 计算临界阻尼 `15.15373221 N·s/m`；TX/TY力限按旧 `50 N / 3.171363 kg` 的加速度包络缩放为 `±3.02 N`。不增加刚体、Pitch惯量或碰撞几何。
- **验证／兼容**：同步更新驱动profile测试、source runtime测试及当前驱动说明。历史3 kg训练配置、checkpoint、归档和旧报告不改；已有旧模型必须继续使用匹配旧源码与配置，不能静默迁移到新驱动。

## 2026-10-11：生成本地开发与远程训练源包

- **用户要求／范围**：用户要把当前项目下载到本地 `/home/hj/pianist` 开发，训练继续在远程执行。本轮只打包当前 A3.2 源码与必要预览资产，不改变训练参数、checkpoint、历史归档或运行中作业。
- **实现／交付**：使用 `scripts/hpc/package_a32.py --include-preview-assets` 生成 `dist/tsensei_v3_2_pianist_local_dev_20261011.tar.gz`，同时生成 manifest 和 SHA256 校验文件；新增 `docs/local_development_handoff_20261011.md`，记录本地解包、远程 Slurm 训练、机械臂显式初角和固定训练归档校验方式。左臂镜像掌面朝下的真实网格 HTML 作为独立候选 bundle 交付，不混入训练归档。
- **排除／安全**：包内不含虚拟环境、训练输出、checkpoint、缓存、历史源码归档或凭据；可选 M6 和四孔连接件网格随预览资产打包。训练仍由远程 `all_use_this_train.sh` 经 `scripts/hpc/submit.sh` 提交，当前固定 `source_arm_ccs696_coupling_v5.tar.gz` 及 SHA 不变。
- **验证**：生成器内置 archive audit 通过；最终文件数、大小和 SHA256 以包旁的 `.manifest.json`／`.sha256` 为准。未启动训练或 GPU 作业。

## 2026-10-11：导出当前双臂安装图与每关节顺逆时针余量

- **用户要求／范围**：用户要求导出当前左右手与机械臂的安装图，并标出每个关节当前角度在位置范围内顺时针、逆时针剩余余量。使用当前同步预览配置 `configs/preview/current_arm_training_20261011.json`；不运行训练、策略或硬件。
- **实现／交付**：新增 `preview/scripts/export_arm_installation_limits.py`，通过 `build_bundle_from_config` 和 `SceneArmFollower` 读取同一 MJLab 刚性链，绘制 `Base→J1…J7→法兰→四孔连接件→手根` 的实际编译网格，另导出左右 PNG/PDF、双侧总览、CSV、JSON、README 和 ZIP 至 `preview/outputs/arm_installation_limits_20261011/`。角度余量按局部 +Z 右手定则计算：`CW=q₀−lower`、`CCW=upper−q₀`；同时记录 J6/J7 耦合值／余量、基座／安装矩阵、配置及 URDF／限位／连接件来源SHA。同步 README 和安装图说明。
- **验证／边界**：`MUJOCO_GL=disable training_mjlab/.venv/bin/python preview/scripts/export_arm_installation_limits.py` 成功，左右各7个关节均在独立硬位置范围内；PNG/PDF、CSV/JSON读回及 ZIP CRC 校验通过，`py_compile` 通过。导出零物理步、零硬件命令；独立关节角余量不替代 J6/J7 耦合、路径／速度／加速度、碰撞或实机安全验证。

## 2026-10-11：在线IK加入CCS696 J6/J7耦合约束

- **要求／实际参考**：用户要求在线IK也执行此前审计到的CCS696 SDK耦合约束。依据 `logs/arm_workspace_audit_20261011/coupled_limits.json` 及项目当前 schema=3 `ArmFollower`，约束为 `1.025×abs(J6)+abs(J7)≤110.5°`，并保留J6±60°、J7±90°独立位置范围。
- **实现**：`mjlab_env/arm_options.py` 将耦合常数写入带臂合同，初始七角在申请GPU前校验；`mjlab_env/arm_follow.py` 的有界IK回溯把违反耦合的候选代价置为无穷，最终200Hz下发目标沿当前状态到候选轨迹做48轮二分可行线搜索，再执行速度／加速度／位置硬限复核。不会再先独立夹角、再把J7事后推回而破坏轨迹差分。`configs/preview/current_arm_training_20261011.json` 同步合同字段和源码指纹。
- **验证**：`py_compile` 通过；Slurm单A100作业2165使用含M6预览资产的不可变测试归档，`tests/test_optional_arm_follow.py` 通过14项、跳过4项。新增测试覆盖初始角拒绝、静态IK候选拒绝、连续120步轨迹耦合满足及每步速度／加速度硬限。未启动训练、未修改奖励和腕目标数值、未改历史checkpoint或历史归档；当前训练归档已同步为本轮耦合修复归档，入口SHA与同目录归档校验文件一致。
- **边界／TODO**：该约束按项目审计确认的SDK J6/J7坐标约定执行；若实机零角或轴向映射不同，需先更新映射及来源校验。它约束关节配置可行域，不等同于全局位姿可达、碰撞或实机动力学安全证明。

## 2026-10-11：MJLab预览同步当前双臂训练环境并加入可达性扫描

- **要求／实际参考**：用户明确预览必须使用当前MJLab训练环境，以便判断机械臂不可达位置。核对新的AGENTS约定、根入口 `all_use_this_train.sh`、当前 schema=3 刚性装配、固定 `self_collision_cap` 归档、训练运行时 `source_runtime`、`PianoTaskEnv`／`SceneBundle`／`ArmFollower` 及现有 `arm_workspace_audit_20261011`。没有启动训练，没有把旧 `preview_pose.py` 或独立 `preview_arm_ik.py`当作当前环境。
- **实现／位置**：新增 `mjlab_env/preview_contract.py`，让 `preview/scripts/preview_mjlab.py --training-config` 先验证源码／依赖指纹，再按同一 `source_runtime` 构建任务，并传入奖励JSON、控制限幅、手指耦合、准备段、禁用腕轴、物理图开关和完整 `arm_follow` 合同。新增 `mjlab_env/arm_reachability.py`，用同一个 `ArmFollower` 在固定安装上扫描 TX/TY/Pitch 与准备高度，输出左右CSV、PNG和汇总JSON；窗口显示请求与实际FK的毫米／毫弧度误差，区分跟踪暂态和静态不可达证明。新增 `configs/preview/current_arm_training_20261011.json`，明确标为从统一入口生成的预览合同，不冒充历史checkpoint config。
- **HPC／兼容**：`scripts/hpc/submit.sh` 和 `job.sbatch` 增加 `mjlab-preview` 单卡模式；作业仍解包显式传入的预览归档，MJLab脚本在 `training_mjlab/.venv` 中运行，审计模式关闭Composer GL并可无窗口执行。旧 `preview` 模式保持原脚本兼容。训练入口继续固定 `self_collision_cap` 归档；预览专用归档为 `logs/mjlab_preview_sync_20261011/source_mjlab_preview_sync_v7.tar.gz`，旧PT、视频和历史归档原样保留。
- **验证／边界**：`tests/test_mjlab_preview_sync.py` CPU小网格、20对父子排除合同和当前奖励封顶通过；CLI小扫描输出8点/侧的分类CSV/PNG/JSON通过；`py_compile`、HPC shell语法检查通过。完整5535点/侧扫描已完成并保存于 `logs/arm_workspace_audit_20261011/mjlab_preview_scan_v6/`：左3191解／2309几何证明不可达／35未确定，右3180／2309／46；Slurm2162在预览归档中完成带臂GPU一控制步验收。静态 `proven_outside_bound` 只证明保守几何外界之外；`unresolved`、IK见证、路径／速度／加速度、碰撞、连接件／手质量惯性和实机动力学不能互相替代。奖励公式、参数、无臂自碰撞路径和训练默认未改。

## 2026-10-11：腕部可达性扫描及M6 SDK动力学参数审计

- **用户确认／边界**：继续手指＋腕部策略、臂7轴由IK；实际腕部须遵守原位置／角度／禁用自由度，执行不了时缩小腕动作并对原请求罚分。用户要求先探究不可达位置再讨论方案；本轮仅独立审计与名义算例，未改控制、奖励、默认参数、基座、旧实验／归档，未启动训练或GPU作业。原expe00002已停。实机SDK参数配置／型号通过异步问题待确认。
- **依据／隔离**：读取当前all_use_this_train.sh/AGENTS和3a177bb0…self_collision_cap归档，核对与旧2109运动学一致性（arm_scene仅父子接触过滤新增）；当前SDK项目内源码／手册／robot.ini/ccs_m6_40、双URDF及用户初角。logs/arm_workspace_audit_20261011保留配置、哈希、差分、脚本、数值及图。并行只读审计交叉复核FK坐标、解析外界和SDK语义。
- **可达扫描／验证**：CPU当前ArmFollower按41TX×9TY×5Pitch×3高度=5535目标／侧，原12次求解加最多12轮多初值96迭代；URDF关节角盒内、0.1mm/.001rad门槛，独立NumPy/scipy FK逐成功点核验。左3191成功、右3180；利用J3/J4组合偏移和其余杆长得肩→Link7严格上界.6020794723m，按同IK误差容差增加.221816mm裕量，每侧2309点证明不可达，左35/右46未确定。零误差上界外多2点与求解容差边缘重叠，单独记录不混分类。音乐高度各1845点中左右798不可达；未假称全部连续空间／碰撞／实时路径覆盖。当前固定TY=.04、Pitch=0的左TX+.7605需.929707m，明显超界；右镜像对应同类结果。
- **SDK发现／仿真**：有动力学和阻抗字段，但URDF effort[108,108,66,66,18,18,18]与示例ini BASIC TorqueMax[35,35,18,18,18,18,18]不一致；MvKDCfg部分零惯量不可直接当物理张量。D文档单位冲突，README为模态阻尼比，未猜粘性阻尼。J6/J7耦合度制1.025|J6|+|J7|<=110.5尚未进当前IK；只对现有解按同零角假设筛查，右2个见证违反，不宣称该目标无其它解。独立CPU URDF臂自身重力逆动力学/势能有限差分误差≤2.19e−9Nm，左J1约−14.263、右13.981Nm、J4约−8.5Nm；不含手/连接件载荷、接触、摩擦或伺服，不宣称实机完整模拟。
- **交付／TODO**：workspace_map.png、两侧workspace.csv/scan.npz、scan_summary与更新后的workspace_summary、independent_bound_audit、sdk_audit/coupled_limits、nominal_gravity_simulation及脚本。图红色为解析证明不可达、橙色为未解；绿色为当前模型数值解，非实机认证。相关机械臂要求文档已同步“待实施”用户合同。后续应讨论安装/可用空间、把实际腕流形和关节双限联合规划、纳入经确认的J6/J7映射以及选择匹配实机的动力学参数；不以腕部独立投影或IK后逐关节夹限冒充严格可执行。

## 2026-10-10：同手碰撞weight改为0.2，整项每控制步最多扣5

- **要求／实际参考**：用户指定self_collision_reward权重0.2、惩罚封顶−5。参考当前统一入口/默认JSON、父arm_parent_filter_fix归档、reward_config/rewards/task与子步tracker、默认/公式/碰撞/指标测试、唯一奖励总表及AGENTS替换旧模式规则；可选日志skill不存在。
- **修改／原因**：模板与默认JSON显式aggregation=substep_sum_v1、maximum_penalty_per_step=5、weight=.2；先对完整子步接触对数乘weight，再在最终奖励单位封顶，25对次起−5。raw保留负计数，weighted/final/观察/回报使用封顶值，不乘dt。活动旧周期并集评分和不封顶配置删除，旧params={}或缺cap明确拒绝；CLI self-collision-weight亦强制RewardManager，不能走旧捷径绕过封顶。物理计数/过滤、臂和其他奖励不变。
- **归档／文档**：从SHA5580e1bf…bd1a06父归档仅叠加本任务代码、配置、测试和文档，生成logs/self_collision_cap_20261010/source_self_collision_cap_v1.tar.gz，统一入口同步路径/SHA。唯一奖励总表、同手专题、AGENTS及README原地更新；不修改历史实验/归档/PT或运行中作业。
- **验证／边界**：CPU去重累计→奖励、0/24/25/26/50对次、权重和预算0、乘权重后封顶、不缩放dt、日志/观察/回报、reset隔离和旧配置拒绝测试；归档审计及入口CPU参数核对记录保存同目录。工作区63项通过／7跳过、38子测试通过；归档隔离61项通过／9跳过、38子测试通过（多出的2个外置入口跳过已在工作区通过）。初次隔离因未链接用户环境而构建失败，补齐原有venv链接后全套通过，未改环境包；无臂／带臂fixture／末尾覆盖3种真实入口经stub与归档parser核对通过，其他奖励逐项保持。具体结果见validation.json。本次不启动训练、不运行GPU；父Slurm2140的接触过滤验证保持历史原权重语义，不冒充新罚分GPU验收。

## 2026-10-10：恢复刚性臂装配前的父子接触过滤，保留无臂自碰撞

- **要求／参考**：用户要求修复相同初态无臂0对、刚性臂12对同手接触，并明确保留不安装机械臂时的自碰撞检查。参考原训练归档、training_environment_audit_20261010的CPU逐对审计、arm_scene转换与MuJoCo／MuJoCo Warp本地父子过滤实现、self_collision子步计数、默认奖励JSON及唯一奖励总表；本机可选日志skill不存在。
- **修复／位置**：`mjlab_env/arm_scene.py`在结构转换前后按编译模型的weld父子关系自动比较，仅为丢失的20对原有关系补充XML物理body排除。mocap手根weld=0导致的原生过滤缺口由此恢复；不硬编码手指忽略名单，不在奖励层隐藏接触。尊重原FILTERPARENT开关及显式geom pair优先级，无臂构建路径不变，非相邻同手、跨手及指键接触保留。
- **兼容／训练归档**：resolved臂合同记录`parent_collision_filter`来源、原开关与补充body对；继续schema3但严格合同比较拒绝缺该字段的旧PT，不放松源码散列。新`source_rigid_arm_parent_filter_v1.tar.gz`以原rigid_arm_v1派生，仅替换经测arm_scene运行代码、增加测试及同步文档；根`all_use_this_train.sh`固定新路径和SHA，具体散列见入口及`logs/arm_parent_filter_fix_20261010/validation.json`。其他既有导出／预览代码不混入，旧归档、实验及checkpoint原样保留。
- **CPU验收**：新增`tests/test_arm_parent_collision_filter.py`覆盖12种手指姿态×2高度的完整接触对等价、正常初态、非相邻自碰撞和指键接触、关闭FILTERPARENT、显式geom pair优先级及旧臂合同拒绝。与机械臂、同手子步、跨手碰撞及默认奖励回归合计38通过、6跳过、24子测试通过；用户原显式初角／原姿态单独构建，无臂和修复后刚性臂均0对同手接触。
- **GPU验收**：统一submit入口Slurm2140／单A100，5通过、1跳过、2子测试通过，退出0:0。真实20Hz控制／200Hz物理周期的中性带臂姿态无伪接触、同手奖励0；无臂和带臂受控非相邻接触姿态都保留5对，对相同物理接触缓冲做10次子步采样均累计50对次，原−.75系数扣−37.5。既有无臂执行、跨手接触、训练更新及局部reset验收通过；未启用13秒长轨迹复测，控制算法未改。
- **说明／边界**：同步AGENTS、README、环境说明、机械臂合同、HPC入口说明及唯一奖励总表的接触来源／采样定义；奖励公式、模式、开关、weight和params不变，无臂默认仍8开14关，显式带臂9开13关，开训仍打印完整最终奖励。验收、归档清单／差异、入口CPU检查保存`logs/arm_parent_filter_fix_20261010/`。未重启正式训练，未宣称F1／吞吐改善；运动罚饱和、实际腕工作空间、长程学习与保存策略仍为后续TODO。

## 2026-10-10：核查expe00002视频实际wrist工作空间

- **要求／来源**：用户认为腕部超出应有范围。读取原best5000合同、scene动作映射／双限位、arm_scene.requests与arm_follow求解／制动，复用2120同次20Hz qpos、mocap_pos/quat。仅CPU原合同重建和mj_forward，不积分物理、不重跑策略、不启动训练。
- **量化**：原TX范围右[−.7605,.4605]m、左[−.4605,.7605]m，TY双侧[−.05,.15]m，Pitch±.16rad。实际投影TX右[−.37125,.14485]m、左[−.37339,−.00323]m，TX/TY未出界。Pitch按最近允许轴旋转计算，左最低−.161108rad，轻微超出约.001108rad；另有禁止轴姿态残差，不能把三轴旋转都叫Pitch。
- **关键缺口**：手根已结构性固定Link7；原腕关节已删除，范围只限制请求，SDK限制约束臂关节，不自动保证实际手根保持原高度／固定姿态流形。已减去规定准备抬高后，左／右离面最大25.534／19.843mm，均t=.30s；准备结束后最大4.524／12.121mm。非Pitch姿态最大约4.48°／2.82°；准备结束后约.707°／2.374°。IK阻尼软目标及逐臂关节制动未加入实际手根工作空间等式／不等式约束。所有数值限20Hz端点，非完整200Hz峰值。
- **产物／边界**：logs/expe00002_result_analysis_20261010/wrist_range.py、json、npz、png保留原范围、投影／残差、时间及原策略未限速请求（不冒充最终请求）；未判为安装脱离或臂SDK硬限失效。建议将实际FK工作空间、禁止方向和关节双限统一求解，不通过独立投影手根破坏刚性连接。仅分析未实施修复。
- **诊断修正**：同目录之前analysis.json内部动作近限统计改为逐通道上下界归一化，正确兼容[0,2]及[−1,1]；此前绝对值>.95不适用于全部通道。接触对、回报、F1与已报告结论不变。

## 2026-10-10：梳理当前训练环境与机械臂训练

- **问题／参考**：用户要求梳理混乱的训练环境，重点机械臂。核对AGENTS、根统一入口、submit/job.sbatch及归档底层入口、setup_envs与20260928安装freeze、当前固定归档及manifest、机械臂options/follow/scene/self_collision、2075／2109原config/metrics/best与结果分析、Slurm实时查询、旧振荡及控制验收。可选pianist-v2-change-log技能路径不存在。
- **整理／位置**：新增docs/training_environment.md作为环境、入口、源码／实验用途及机械臂三版合同的导航，不复制奖励默认总表。README增加当前入口与停止状态，将旧启动更新明确归入历史；修正旧默认来源、1180维观察与node4现状措辞，标明删除的子步CLI为历史。更新机械臂专题的2109停止结果、旧schema2振荡分析的适用范围、HPC入口及环境说明；submit.sh、run.sh、run_tsensei_a32.sh只修正注释和注释示例，不改变命令行为。
- **环境／实验事实**：北京时间晚间用户队列为空，2109已于18:58:08取消；最后已记录6000步／3,072,000样本／744,576更新，最后完整评估5500步三回合原谱合并及最差F1均0%。saved best5000合并1.1309%、最差0%，与未保存的step2000开始学习后最高合并8.9955%区分，初始未学习网络合并F1为11.4524%；forward仍为500步、无final／summary。learner133包与原安装记录一致、Composer原84包不变追加6包分析依赖，无整体重装依据。固定归档SHA正确；235个选定Python／JSON中仅既有export_videos和preview_arm_ik后续工作区改动不同，机械臂／训练核心七文件一致。
- **接触过滤定位／证据边界**：新增CPU对照使用原2109源码、best合同与相同基础场景，无策略推理／物理积分。共同动态关节reset qpos差0、左右手根位姿差约1e−16；无臂0对同手接触、schema3刚性臂12对。这12对在原native weld／parent关系下均被过滤，手根删除腕DOF并挂mocap后weld=0，原生父子过滤不再适用；SelfCollisionTracker的conceptual分组没有恢复物理过滤。证据是模型转换前后的明确差异，不是已证明F1下降的唯一原因，也不冒充GPU200Hz逐对接触明细。旧动作abs>.95统计忽略[0,2]通道，不用于新的饱和结论；两项运动罚满−5依据原同次reward_terms。
- **验证／保存**：logs/training_environment_audit_20261010保留environment_snapshot、source_comparison、entry_validation、arm_contact_filter_audit及两个CPU复现脚本。真实入口经stub sbatch／归档底层入口／归档parser核验：无臂8/22项奖励、显式fixture臂9/22、schema3，末尾覆盖生效；缺双侧七角exit2且未调用sbatch。七个shell入口bash -n通过；CPU初态接触对照复跑通过，文档相对链接核验通过。fixture不代表用户初态，本轮无真实作业提交／GPU运行／环境安装，未改训练算法、奖励、归档、历史PT或原实验。
- **未解决TODO**：按原模型已确定的父子接触关系恢复物理过滤，CPU及Slurm复验后同步归档／SHA；定位两项运动罚的主导通道与饱和分布，区分手指／腕旋转的最终下发目标差分与TCP／臂罚的限幅前需求；下一轮明确best资格、选优口径和中间存档，以免高分评估无对应PT。长期训练效果及实机动力学仍未验收，本轮未启动训练。

## 2026-10-10：expe00002停止后的训练结果分析

- **范围／依据**：用户要求分析当前结果。只读原2109完整metrics/config、best5000同次2120录制及原1c8b3147源码；训练保持中断，未改变奖励／几何／源码归档或重跑策略。
- **训练证据**：初始200Hz原谱三回合汇总F1=11.45%，训练最高9.00%，best5000=1.13%，5500=0%；总回报却由−20647升至−18122。best视频F1=0%、14个实际音符事件，音乐窗口TP0/FP130/FN2615。saved best按最差回合F1并受琴键底面接触筛选，不是全历史无条件最高模型。
- **离线定位**：logs/expe00002_result_analysis_20261010/analyze.py在CPU按原PT源码／依赖／模型合同重建，只对原183个20Hz qpos做mj_forward接触查询，未积分物理／推理／运行GPU。每帧12对同手接触：双侧base_link对食／中／无名／小指MCP及拇指旋转块各5对，双侧掌对拇指旋转块各1对。最大CPU代理穿入12.83mm（base与拇指旋转块）；183帧全部存在。该CPU端点结果不冒充GPU200Hz全部子步的独立接触明细，但与原GPU同次每步self_collision=−90及12×10×.75吻合。
- **奖励含义／局限**：同次视频均值按键+.3816、踏板+1、指法+.00048，同手−90、关节／腕TCP−5和臂−5。两项运动罚183步全饱和，腕/臂罚基于限幅前请求，不能据此认定已下发臂轨迹突破SDK硬限。paper无误触部分可贡献+.4，因此少按键仍可获得部分奖励；恒定−90本身不改变理想最优策略，数值大不单独证明因果。需优先审计装配代理／接触筛选，再记录限幅前需求的主导关节与饱和分布，受控对照后才修改训练。
- **交付／验证**：analysis.json保留接触对、帧计数／最大深度、奖励和动作统计；evaluation_curve.csv与training_analysis.png由原日志绘制、无平滑。原合同哈希重建通过，保存qpos维数匹配，JSON奖励与GPU回放一致。未把该结果宣称实机穿透或全部动作不可避免接触；高分缺档和训练收敛尚未解决。

## 2026-10-10：导出expe00002 best并中断训练

- **要求／依据**：用户明确改为导出best后中断。核对2109实际config、指标与checkpoint；冻结第5000步／256万样本／616,576更新的best.pt（f8334214…c967b）。训练三回合汇总200Hz原谱F1=1.130902%，与未保存的历史最高第2000步8.995490%区分。
- **导出／隔离**：Slurm2120单A100使用原1c8b3147…265dea7f归档录制完整1830子步；2121仅读同次qpos／音轨调整正面完整双M6取景，1280×900／20fps／10.2秒／H264+AAC。展示派生包仅修改render_recorded_native的取景、标题、零mocap兼容及清单；模型／奖励／控制全部原字节保持，未重跑IK／策略生成第二条轨迹。产物preview/outputs/expe00002_best_20261010/，审计logs/expe00002_best_export_20261010/。
- **评分／验收**：原谱34音符、1秒准备、9.15秒物理记录、末帧展示1秒；同次200Hz键程阈值重建音轨、事件、PNG/PDF/NPZ/CSV和原MIDI来源。本视频音乐micro F1=0%，TP0／FP130／FN2615，不能用训练1.13%替代。原PT/config/源码合同、原谱逐子步目标、真实键与控制端点、音频事件、视频全片解码/AAC及ZIP CRC/manifest均通过；人工查看正面完整装配画面和对比图。
- **中断／局限**：导出验收后18:58:08执行scancel 2109，sacct确认CANCELLED；最后已记录第6000步／3,072,000样本／744,576更新，不宣称停止时未记录的实际步数。保留原run、检查点和新独立best备份，未更改训练算法／奖励／原归档。高F1缺档无法恢复、碰撞和饱和罚分问题本轮未修复。同步视频说明；离线绘图使用training/.venv（training_mjlab/.venv缺note_seq）。

## 2026-10-10：启动expe00002刚性双M6训练（Slurm2109）

- **用户要求／参考**：用户在跟随器修复及视频确认后要求开启训练。核对唯一入口、固定1c8b3147…265dea7f源码包、当前奖励总表／JSON、修复验收、用户左初角与expe00001镜像审计、train_from_source及实际Slurm资源；可选日志skill未安装。新臂观察／合同已变，按当前入口从随机初始化，未沿用旧PT。
- **提交／保持**：只用all_use_this_train.sh末尾覆盖arms、双侧显式7角、臂运动罚开启、冻结奖励输入路径和独立输出expe00002；没有新训练入口、没有改统一默认数值／运行算法或历史归档。2026-10-10 13:58提交2109，双A100/512环境/20Hz控制/200Hz物理、cuda0采样/cuda1学习流水、10,000,384样本。当前源码归档散列与作业源包一致。
- **预检／奖励**：logs/expe00002保留真实入口fake-sbatch捕获、不可变输入快照、完整最终奖励打印、CPU构建／刚性初态FK、初始镜像与预检结果。实际运行config再次核对9开13关、完整奖励等于预检，schema3／31动作／1461观察、双硬限与CPU固定基座匹配，origin为空且三个恢复入口为空。没有更新奖励公式／参数，总表仍为唯一维护来源。
- **运行状态／文档**：作业RUNNING、双GPU基础检查通过，实际输出training_mjlab/outputs/hpc/expe00002-20261010-135845-564168/。启动阶段先初始评估，再预填93,696样本，第93,697样本起满足更新条件；尚未把启动或初始网络评估称为训练收敛。初始审计startup_validation.json记录实际阶段和指标行数；后续以metrics.jsonl为准。同步README及机械臂当前文档的新实验记录。长期吞吐、F1与指键接触改善仍待训练结果。

## 2026-10-10：导出刚性机械臂修复后的测试视频

- **要求／参考**：用户要求测试视频，随后明确演奏者前方视角。参考当前1c8b3147…265dea7f训练包、统一入口、用户expe00001初角／镜像审计、机械臂修复长轨迹测试、原左腕导出与camera/audio/source_runtime；遵循同次按键／原MIDI图要求。可选日志skill未安装。
- **实现／隔离**：logs/rigid_arm_video_20261010捕获真实入口完整配置，冻结脚本与输入。只给父包增加本次record/render执行文件，全部原文件字节保持，CPU编译预检通过。Slurm2103记录双环境13秒200Hz完整qpos/qvel/ctrl/mocap、腕请求／实际、法兰、臂q/v/a及琴键；scene.step预设42cm停止／48cm返回，手指主动指令保持初态，未运行策略／学习器。奖励配置完整打印，任务奖励不作为此诊断成绩。
- **视角／产物**：取消未执行的待依赖2104，2105仅从同次状态生成用户指定front正面及连接件近景。两段14.05秒、合并28.1秒，1280×900/20fps/H264/AAC；每段末帧静止1.05秒。交付preview/outputs/rigid_arm_test_20261010的合并／分段视频、运动图、左臂7轴v/a图／CSV、真实触键音轨、原始MIDI／实际对比PNG／事件／NPZ、来源、HTML与ZIP。右侧仅隐藏，完整88键声音／图包含右手接触；不称仅左手F1。
- **验收／观察**：2103、2105均COMPLETED；停止／返回后0.355／0.945秒停稳，SDK双硬限和URDF位置通过，安装残差3.1229e−8m。独立URDF FK位置误差最大8.56e−8m、旋转矩阵差<2.72e−7；原谱34事件来源、同次音频／按键数据、视频全解码／时长、manifest及ZIP CRC通过，正面关键帧／对比图已查看。保持指令下确实出现拖键和重复触键（507／242个触键事件），原数据保留；参考F1 12.6181%／13.8665%仅为预设测试与原谱差异，未冒充训练。
- **文档／范围**：同步README与video_export_cli本次索引；训练源码归档、默认参数和旧实验／视频不改，没有新训练。机械臂仍运动学，指键接触抖动与长期演奏效果未在本导出中修复或宣称验收。

## 2026-10-10：按四孔刚性装配修复腕目标IK跟随与目标制动

- **用户要求／参考**：用户要求严格跟随，并明确手—四孔连接件—法兰固定，目标是RL得到同一装置的臂／手轨迹。此最新合同替代旧实际腕独立PD、机械臂只跟随显示的要求。参考arm_options/follow/scene、scene/task/bundle/finger_clearance、target_motion_limits、旧200Hz振荡记录／原因对照、包内URDF／SDK／preview_mount／四孔装配JSON与STL、checkpoint／统一入口／归档流程；MuJoCo官方body/joint文档核对无关节子体的刚性连接语义。
- **安装／控制实现**：保留Link7法兰中心[0,−.095,0]m和29.567719mm固定手根偏移，连接件配置／12个网格SHA入合同。arm_scene将法兰—连接件—手根作为Link7无关节后代；删除旧腕TX/TY/Pitch物理DOF／执行器，保留策略腕请求，200Hz先连续IK＋受限制动，再唯一装配根更新及手指／琴键物理。按名字重映射状态／执行器并恢复编译舍入精度。固定基座、手指／琴键物理输入保持，臂／连接件不假定动力学或碰撞。
- **振荡修复**：arm_follow以之前IK参考为连续种子，执行层按目标剩余距离反解离散可停速度，同时保留URDF边界制动与SDK v/a交集；双臂验证通过才提交。新增54维q/v、参考、完整位姿误差；scene/task与局部reset同步。finger_clearance支持真实刚性手根完整姿态，保留键底诊断，旧动作投影在臂模式明确拒绝。臂运动罚数值／最大值公式保持，输入为腕请求的连续IK参考；能量项只计实际存在的手指执行器。更新唯一奖励总表对应行／语义、当前机械臂说明、HPC当前节、AGENTS与README。
- **上游追加修复**：Slurm2096的13秒反向轨迹虽然最终停稳，但曲线揭示旧20Hz腕部目标限幅器自身反复超调；不是刚性链脱离。仅带臂模式在target_motion_limits增加面向限幅前原始目标的离散制动，scene传入原始请求避免一步slew遮住远端目标，硬限数值／奖励公式不改，无臂分支保持。新增随机反向硬限与单调回程测试，36项目标限制CPU测试通过（含12个subtests）；更新带臂target_motion_contract.schema=2。
- **接触补漏**：CPU强制双手重叠对照发现，旧MuJoCo将两侧无自由度根都归入world weld并过滤掌部互碰。arm_scene为原根／掌部几何补4个显式跨手接触对，self_collision按最近独立mocap根区分无自由度碰撞组；保留同手刚性连接内部不计碰撞。修复前42个跨手接触且无根／掌部对，修复后与原物理手模型同为51个（含9个根／掌部接触点）；不添加臂几何接触或运动学反作用力，唯一奖励表同步说明。新增CPU及Slurm回归覆盖原根接触、tracker和奖励快照。参考MuJoCo官方collision selection及实际CPU/GPU版本行为。
- **CPU验收**：初始臂核心／原物理参数／实际胶囊FK12项通过；补准备合同与根碰撞2项通过；目标限制36项／12个subtests和同手计数6项通过。当前相关回归67项通过（58个subtests）；另3个旧几何姿态子断言及2个旧入口schema错误，在修改前代码按相同测试重现，保留日志，不用放宽阈值掩盖。原记录腕轨迹作输入的CPU控制器13秒复测已停稳，原记录和全部历史模型不动。
- **GPU最终验收**：Slurm2100单A100通过长轨迹、无臂默认、SAC更新与局部reset三项。两环境各13s／2600子步：42cm左移4s停止后4.355s稳；48cm远请求6s返回后6.945s稳（请求位置误差<.1mm、全部臂速度<.001rad/s且持续至13s）。末1秒臂速度0；独立q差分复核SDK v/a通过，20Hz腕目标v/a最大比例.930282／.999998。固定装配误差峰值3.1229e−8m；极端反向请求误差可达145.7mm，未冒充连接间隙。Slurm2102最终补碰撞版本三项通过，覆盖GPU原根碰撞、tracker／奖励快照、有限物理步、无臂、SAC和reset。2100后只补接触对和碰撞组，控制文件逐字一致；新增对在原长轨迹上严格分离，trajectory_scope.json保存保守包围半径证据。早期2095／2096／2097／2099验收日志保留用于追溯。
- **交付／归档**：preview/outputs/arm_follow_repair_20261010/validation.png与validation.json，CPU/GPU数组及差分／来源SHA在logs/arm_follow_repair_20261010。最终固定source_rigid_arm_v1.tar.gz从797385b8…8ad0d17原包仅覆盖本轮源码、测试、当前文档和四孔装配12个STL；生成完整清单／SHA、审计并与Slurm2102受测运行时代码比对，再同步唯一入口。真实入口以fake sbatch及最终归档解析器核对默认无臂／显式有臂、原奖励数值、双卡流水和缺角申请前拒绝，未提交训练。
- **兼容／边界**：活动臂schema3／rigid_wrist_command_v1，带臂腕目标制动合同schema2，旧臂schema1／2明确拒绝，旧PT须原归档。原手部观察的已移除腕轴为常数占位，54维实际臂q/v／参考／误差承担完整根状态表达。仍为运动学臂，未增加臂动力学／连接件惯量或臂碰撞；根间互碰只检测，不产生运动学根的反作用力。未启动长期训练／恢复expe00001，长期F1、吞吐、收敛及硬件标定仍待验证。


## 2026-10-10：左臂持续振荡的原因定位与隔离对照

- **问题／实际参考**：用户要求认真分析上一轮左腕加速测试停止后为何持续振荡。读取Slurm2085同次200Hz物理记录、20Hz腕请求／下发目标、完整臂合同／基座／初态、固定797385b8…8ad0d17归档与其arm_follow/arm_scene/scene/target_motion_limits、包内左臂URDF／SDK／工具安装。核对173项源码指纹；34个记录前态的CPU单步IK重算，下发q与原GPU最大差4.20e−15rad。无策略学习、无新GPU／物理重录，所有操作为CPU分析。
- **定位**：4s停止请求没有预减速，腕目标硬限生成452.858mm下发位置，实际腕超调至432.165mm。包内几何肩心到腕心保守上界602.079mm，请求峰值606.574mm，4.275–4.590s完整位姿不可达；4.310s加权Jacobian条件数约3689。最终位姿经独立有界IK验证可达，6–9.15s腕位置严格静止，但臂误差仍1.167–103.570mm。当前求解只优化下一步位姿，速度只参与可行边界；stopping_speed面向URDF位置边界，没有面向目标制动，导致反复过冲。8.865s误差1.167mm时J1仍90.105deg/s，离散最低制动位移约8.80deg。
- **隔离／限制**：从同一6s臂q/v额外运行8s：固定腕原算法6.340s停稳；记录目标延长6.950s；初v清零5.385s；固定同一IK解继续原限幅追踪6.420s；固定解＋位置速度反馈保留原硬限2.585s。仅取消加速度边界0.085s，但峰值95.39倍SDK，只作为诊断反例。停稳定义误差持续<.1mm且各速度<.001rad/s；延长CPU最终收敛，不宣称永久极限环。近奇异/切换处CPU与GPU会积累数值差，长程停稳时间不能冒充原视频的未来预测。
- **产物／验证**：新增logs/arm_oscillation_analysis_20261010的可复现分析／后处理及preview/outputs/arm_oscillation_analysis_20261010的原记录原因PNG、对照PNG、全部数组／指标／来源SHA。独立NumPy FK、对照q/v/a差分、SDK／URDF边界复核通过（明确取消加速度的诊断项除外），图已查看；新增docs/arm_follow_oscillation_20261010.md，同步README与机械臂合同的问题状态。
- **范围／TODO**：仅调查和文档，当前运行源码、训练奖励、固定归档/SHA、旧checkpoint及视频保持。诊断位置速度反馈尚未作为动态腕跟随修复部署；后续需保留腕部主导／固定基座／双硬限，完成目标制动和连续IK、长时间停稳／近奇异／不可达恢复／双侧隔离等CPU及Slurm验收，再更新归档。不能靠继续训练、加大奖励罚或取消硬限替代控制修复。

## 2026-10-10：左腕逐渐加速左移的受控视频测试

- **用户要求／实际参考**：从初始位置让左手向左越来越快，观察当前训练环境并导出视频。参考当前统一入口及797385b8…8ad0d17固定归档、用户expe00001左七角与已确认镜像右七角、arm_scene/follow、scene/task、preview_camera/audio、原Twinkle注册曲与173个训练源码指纹；全部参数通过真实入口fake-sbatch捕获并用冻结源码CPU构建，没有新建训练入口或加载旧schema检查点。
- **测试范围／方向**：模型核对低音方向为世界−Y，对应L_Wrist_TX减小。保留1秒原准备段、原腕PD、手指耦合、接触、奖励及SDK双硬限；1–4秒位置需求为−0.42×((t−1)/3)^3米，速度从0增至0.42m/s，4秒后目标保持，其他主动指令取reset值。完整183控制步／1830个200Hz子步，未运行学习器、未启动或恢复训练。右手只在显示隐藏，物理和奖励保持；无策略checkpoint，不将该测试称为训练F1。
- **实现／导出**：任务代码仅在logs/left_wrist_ramp_20261010及展示派生归档，父归档的训练源码逐字保持。2084因学习环境无note_seq在物理开始前失败，改用既有Composer环境从匹配源码导出34个原始音符事件，未安装/更改环境。Slurm2085单A100成功，逐子步保存真实qpos/qvel/ctrl/mocap、腕位姿、IK原始需求和最终q/v/a、残差、琴键；原速度1280×900/20fps/10.2秒，叠加速度/间隙/罚分曲线。初版相机从演奏者对面观察；2086重绘脚本漏csv导入，修正后2091仅从同次状态重绘为演奏者方向，让左移在画面向左；没有重跑物理/策略/IK，复用同一音轨。
- **实测现象**：最终左移41.99997cm，实际左向腕速峰值0.36730m/s。加速段1–4秒左臂残差最大0.000601mm；4秒后停止指令、手部短暂超调到43.2165cm，4.245秒臂残差超过1mm，此后出现持续振荡；最大119.3626mm（4.605秒），末态46.1011mm。183控制步中左臂101步有超限罚、100步满−5；J4速度触及πrad/s，全部关节下发速度/加速度仍通过SDK硬限复核。观察到的是跟随稳定性问题，根因尚未在本次定位；不能把“硬限通过”说成“跟随稳定”，也没有修改求解器隐藏现象。
- **交付／验证**：preview/outputs/left_wrist_ramp_20261010，最终performer_view/replay.mp4、motion_overview.png、left_arm_joints.png/200HzCSV、同次原MIDI/实际按键PNG/事件/NPZ、HTML/ZIP。保持手指指令下没有触发琴键，音轨静音，参考F1=0%仅作原谱对比，不代表学习结果。独立URDF FK、真实腕qpos/mocap的CPU复算（最大3.75e−8m）、全部173源码、配置/用户初角、同次MIDI目标/实际键、全片视频音轨解码、manifest SHA和ZIP CRC通过；已查看最终初帧/4秒帧及对比PNG。仅本次测试产物和文档更新，统一训练归档/奖励/停止的expe00001保持。

## 2026-10-10：恢复腕部主导，机械臂单向IK跟随

- **用户要求／参考**：用户明确纠正“腕部带动机械臂来训练，不是机械臂直接驱动手”，替代旧方案A中FK反写实际wrist的语义。核对统一入口、arm_options/follow/scene、scene/bundle/preparation、腕PD与目标限制、checkpoint臂合同、奖励总表、旧expe00001原归档与GPU验收；未启动或恢复生产训练。
- **实现／位置**：arm_scene保留原腕TX/TY/Pitch关节、PD执行器、原动作/状态ID和准备支架，新增的只有零质量、无接触臂显示体；SceneArmFollower只读物理后的真实wrist xpos/xmat，publish只写机械臂显示mocap。scene每子步先物理积分、关节硬位置修正与forward，再IK、SDK联合硬限、显示；恢复准备kernel、原finger-clearance guard与键底接触诊断，当前投影开关保持关闭。arm_follow诊断保存本次实际目标位姿，独立运动罚数值/最大比例公式不变，输入改为实际腕位姿求出的IK需求；SDK速度/加速度、28维观察和动作维度保持。
- **兼容／物理参数**：臂合同schema=2、唯一actual_wrist_ik_follow_v1，活动代码拒绝旧schema1，旧模型继续匹配历史归档，不放松SHA/模型/控制检查。CPU对照发现mj_saveLastXML会舍入体位姿、惯量、轴、阻尼等；追加臂后恢复原模型浮点输入/原网格数据，通过各body/mesh地址恢复BVH界并重算质量常数，保留新整数资源地址。原关节/执行器/体/几何/网格等名称前缀强校验，零质量臂显示不改变总质量；原手积分轨迹严格对照通过。
- **CPU验收**：60项通过、1项GPU按保护跳过，另22个subtests通过；覆盖限速/限加速、制动边界、实际目标不读取PD需求、受限臂产生残差时不能反写手、局部reset、旧schema拒绝、当前奖励、TCP/Pitch目标限制及checkpoint控制合同。独立核对原模型质量9.588kg保持、BVH节点分组一致，证据logs/actual_wrist_follow_20261010/。
- **GPU与归档验收**：Slurm2081通过步进/SAC后，测试误把MJLab TorchArray包装器直接交assert_close导致类型错误；仅测试改为读取[:]。2082全部子步合同通过，但沿用旧“远PD请求必然造成IK超限”断言失败；新合同实际腕缓动时可无超限，改为显式扰动真实腕位置来测试受限跟随，不改算法/阈值、不放宽数值容差。Slurm2083完成GPU短集成（A100、1项通过、22.75s测试），覆盖20个物理子步、真实目标、非零超限与SDK双硬限、一次SAC更新及局部reset隔离。最终归档与受测版本运行时代码逐字核对，审计记录在同目录validation.json与archive_audit.json；统一入口在验证后切换新归档SHA。
- **文档／边界**：同步当前奖励总表对应行/功率语义、AGENTS、README、可选臂说明及HPC当前段落；旧要求文本另存previous_requirements.md且原训练归档原样保留。臂受限可跟不上手，允许残差/暂时脱离；没有臂力矩动力学或反作用力，没有新增残差罚，不保证新架构长期F1/吞吐。expe00001仍停止，未把旧模型说成新模式续训结果。

## 2026-10-10：中断expe00001并导出最新带臂视频

- **用户要求／参考**：中断训练、导出最新视频。参考2075实际config/metrics和checkpoint原子保存逻辑、固定source_arm_follow_v1归档、render_mjlab_checkpoint/capture与回放入口、scene/task闲指诊断分支、preview_camera/audio及原注册Twinkle MIDI；遵循视频同时交付同次实际按键与原谱对比图要求。
- **停止／模型保全**：用户请求时首次500步评估尚未结束、没有任何PT；评估期间学习器不更新，等待原子forward.pt写入后立即scancel2075，状态CANCELLED。已CPU完整读取核对最新PT：step500、256,000样本、40,576次更新，SHA1906bd98…3aff6b。没有停止其他训练、热改运行作业或伪造final/best文件。
- **导出修复／位置**：2076在首帧因渲染器把非空“禁用诊断合同”当成实际guard而读取不存在的finger_projection_mask失败；preview/scripts/render_mjlab_checkpoint.py改为检查实际finger_clearance_guard。2077的初次派生包漏更新清单，被原散列审计拒绝；重新生成BUNDLE_MANIFEST后审计通过，没有放松检查。2078单A100按严格匹配的全部173个模型源码文件、config和控制合同完成策略回放及同次200Hz音频记录。
- **同轨迹重新取景**：原相机裁掉基座上沿，新增preview/scripts/render_recorded_native.py及--recording展示入口，从该录制的qpos/ctrl/mocap读取状态、只做mj_forward与渲染，复用原played.wav；不调用策略、mj_step或IK。完整相机lookat=[.30,0,.30]、distance1.35、elevation−25。2079在CPU加载检查点时CUDA RNG状态不兼容，改为分配内cuda:0严格加载，2080成功27秒；不修改检查点RNG或合同。派生v2/v4归档只修改展示脚本／增加展示脚本及清单，全部fingerprinted模型/奖励/训练/控制文件逐字保持。原训练入口归档/SHA不变、历史归档原样保留。
- **交付／验收**：preview/outputs/expe00001_step500_20261010_v2/，最终视频full_scene/replay.mp4，1280×720、20fps、10.2秒、H264/AAC、原速，含1秒可学习准备、1830个200Hz子步及1秒末帧停留。同次原谱音乐micro F1=5.2290006%（TP141／FP2637／FN2474）；原34音符，实际218次起音。PNG/PDF、事件CSV、实际/目标NPZ、原MIDI、来源、HTML和ZIP齐全；原谱注册流程去除14.692708s原前导静音、视频音乐偏移1s，未用控制网格冒充原谱。图与原始精确子步目标、实际控制末态和音频事件逐项一致。
- **检查／边界**：CPU检查有/无实际guard分别保留5项／0项诊断，源码/配置/PT/臂合同核验通过；2080全片音视频解码、实际预览帧/对比图查看、22个manifest文件SHA与ZIP CRC通过。本机无ffprobe，使用项目既有ffmpeg检查双流/时长并完整解码；归档原MIDI时间戳为epoch，ZIP显式strict_timestamps=False，仅归档时间表示调整，文件字节保持。日志／审计在logs/expe00001_stop_export/。未恢复训练；低F1和此前最佳模型筛选诊断门槛问题未在本次修改，单视频分数不替代三回合训练评估。

## 2026-10-09：expe00001 双M6镜像初态训练

- **用户要求／参考**：给定左臂J1–J7为[84.27055,−68.29417,−61.39675,−89.05803,158.27133,6.45346,−28.12498]度，右臂镜像，命名expe00001开训。参考统一入口、当前奖励JSON、固定source_arm_follow_v1归档、双侧M6 URDF、preview_mount明确Link7安装、手模型reflection、arm_options/follow/scene及可选机械臂需求记录；无可用日志skill。
- **实现／原因**：保留用户左角、双侧模型与安装，用URDF反射推导右前四轴，再对右后三轴有界求解完整镜像末端朝向。右角[-84.27055,−68.29417,61.39675,−89.05803,−64.8343338991625,27.929049544537655,−7.308340216508003]度，全部在位置范围内。这不是更改IK训练算法；保留原归档/SHA9adb5f69…274ce。统一入口仅本轮末尾覆盖--arms、显式双七角、开启机械臂运动罚、冻结奖励输入路径和expe00001输出目录；无新训练脚本／默认表。
- **验收**：CPU实际固定归档构建、inspect-only解析和初态FK对齐通过，世界镜像对应关节原点误差最大6.89µm（保留URDF近似π）；实际运行config中完整机械臂参数、奖励及反算基座与CPU审计完全一致。Slurm2075已分配双A100并完成基本GPU核验，观察1435／动作31，从头10,000,384样本、512环境、20/200Hz、采样学习流水；183控制步/93,696条transition预填后学习。完整最终奖励已在日志打印，9开13关；paper、和弦关闭、独立误触关闭，新增机械臂独立−5预算，其余奖励/限制保持。
- **记录／边界**：输出training_mjlab/outputs/hpc/expe00001-20261009-230640-597425；命令参数、奖励输入快照、CPU解析、镜像／基座审计和startup_validation保存logs/expe00001。同步README、可选机械臂说明及奖励总表对应机械臂行。现有无参数默认、历史模型/实验与其他作业保持；当前完成启动验收，完整预算吞吐及收敛效果待训练，未宣称初始高F1或物理安全保证。

## 2026-10-08：纠正SDK四指swing零位到URDF最小值

- **用户要求／参考**：用户查看最大值图后明确“计算反了，调整sdk0对应最小值”。按最新要求替换当前网页SDK映射；核对冻结A3.2四swing范围[−.35,+.35]rad、上一轮原SDK标准CSV及固定基座/安装，不更改实机SDK原角度、时间、七臂或其他手关节。
- **实现／合同**：contracts当前唯一SDK公式改q_URDF=deg×π/180−.35，SDK0→−.35rad、SDK20.0535deg→0、SDK40.107deg→+.35rad；不裁剪反馈。CSV结构schema仍1，方向标记hardware_sdk_v2；旧hardware_sdk_v1新导入明确拒绝，不能把相同字节按新方向静默解析/命中旧视频。模板与字段说明、README、网页提示同步。新CSV只改变首行方向标记，SDK数值/时钟逐点完全相同，独立规范化数组断言仅4swing反向、基座/安装精确保持。旧归一化文件、最大值图和视频保留历史原样，不修改原数据、训练奖励/算法/源码归档。
- **回放／交付**：新增logs/left_sdk_zero_min_20261008准备/静态图/包装/浏览器核验，新preview/outputs/left_sdk_zero_min_20261008。Slurm2032静态SDK0图四指全−.35rad，其他23角不变、无物理积分；2033名义74s仅键，2034原20指动力学/臂规定，2035下发目标仅键，2036反馈仅键，均经统一入口单A100 EGL完成0:0。后三片保留88.092815933s真实时钟/89.12s视频，前片75s，25fps H264/AAC、左臂左手面对演奏者＋俯视；无额外.15rad或腕/base重定位。每片独立同次400Hz实际琴键生成音频/事件/图，新起音数67/39/68/55，全部输入SHA/规定mocap/独立FK/完整视频解码通过。只积分手指的对照不称全臂动力学。
- **验证／限制**：28项CPU网页测试通过，新增SDK0/中心/最大值三点独立方向验证与旧标签拒绝；正常空位/严格场景缓存仍通过。真实浏览器三片播放/同步、模板新标记、同次图/轨迹Range206、文件上传→新版缓存完成、旧媒体仍可访问核验保存于本轮logs；当前8787启动新版dataset/runtime，历史dataset只追加已验证媒体。原始MIDI/音乐起点仍缺，F1=null，原谱完整对比未完成；已有强制接触/软键限超程仿真限制保留报告，不把模拟力称硬件力。


## 2026-10-08：左手四指swing最大值姿态图片

- **用户要求／参考**：用户确认SDK swing零度对应模型最大值后要求导出图片。参考left_flange_recalibration_20261008/replays/plan74的冻结scene_manifest、首帧command_motion及资产SHA，四指关节limit.upper=.35rad，已确认SDK约定q_URDF=.35−deg×π/180。
- **实现／边界**：新增logs/left_swing_max_20261008/{render,test_render}.py，仅将食指/中指/无名指/小指swing改为各自模型上限+.35rad（+20.054deg，SDK等效0deg），其他23个关节、基座旋转/平移及安装保持首帧；只mj_kinematics/mj_camlight，无物理积分/策略/额外.15rad。输出preview/outputs/left_swing_max_20261008的正面/俯视1200×980 PNG、关节角对照CSV、来源和SHA，未修改原轨迹/视频/网页、训练源码/奖励/归档。
- **验收**：经scripts/hpc/submit.sh单卡Slurm2031，COMPLETED0:0，分配内1项导出验收通过；NVIDIA A100 EGL非软件渲染，独立CPU FK与显示完整链接位姿一致，原输入/资产SHA前后不变，恰4关节设上限及其余位置保持断言通过；已查看两张图片。图片是静态姿态示意，不是击键/动力学评估，无F1或声音导出要求。


## 2026-10-08：法兰首点反算左臂基座、统一CSV与真实网页导出

- **用户要求／实际参考**：用户给法兰首点钢琴XYZ=[207.090,−150.552,99.158]mm，明确“保持当前基座朝向，只反算位置”，并要求继续计算轨迹及标准CSV网页上传导出。参考10.8_14.12.zip内left_arm_200Hz(2).csv首帧、原wzx冻结scene_manifest、项目Link7局部[0,−95,0]mm法兰定义、已确认SDK swing约定、原20指／88键XML与网页接口；未使用附件作为执行指令，未连接硬件。
- **基座／轨迹**：使用74s原臂首帧q=[86.82750,−67.17148,−60.92138,−92.05105,159.11995,7.30998,−28.51734]deg，按p_P_B=p_P_F0−R_P_B p_B_F(q0)一次反算。新基座XYZ=[609.880265798,−19.281045732,418.768079248]mm，旧RPY=[−89.939536,0.737832,178.585872]deg及Link7→手安装保持，误差<3e−17m，无额外0.15rad/准备段/平滑。已有完整七关节轨迹时保留原q/qd/qdd字节并在新基座做FK，未伪称另造wrist目标做IK；生成wrist_plan_200Hz、基座CSV/矩阵/三方向坐标图。名义74s手序列从实机下发启动1000点后37001点重建，缺发送值沿前值；另外88.092815933s命令/反馈文件保留真实时钟，不与名义规划混用。
- **网页／格式**：新增contracts.py、templates/trajectory_v1及字段说明CSV；schema1首行标定XYZ、可选RPY/标定七角，27原生关节_q_deg和明确hardware_sdk_v1/urdf_v1。SDK四swing=.35−deg×π/180，其余直接；首帧不空，后续同关节前值，固定字段不许中途变化，截断CSV明确拒绝。已标定同一首点的场景保持精确矩阵，避免重复上传的浮点加法改变SHA；缓存仍严格核对原文件、模式及场景SHA。新增可启用的Slurm exporter／worker快照、规范化数组及场景冻结、持久化队列/已提交作业恢复、完成文件SHA验收与媒体恢复；前端处理submitting并刷新继续查询任务、区分输入/实际回放轨迹下载。媒体追加CLI仅载入已验证旧dataset的media，保留旧WZX及历史URL，不套用旧分数。
- **真实GPU／验收**：Slurm2017完成74s规定机器人视频，75s/1875帧；2026/2027分别完成真实时钟目标/反馈，仅88键动态、机器人DOF0；2030完成20指动态对照，臂DOF0。后三片89.12s/2228帧，25fps、720×900、面对演奏者保留左臂左手＋俯视，NVIDIA A100 EGL；全片解码/AAC音轨、独立FK误差<1.9e−15、输入数组与规定mocap每步不变、模型/安装/键记录SHA通过。声音、实际事件和PNG同片同次400Hz键录制，实际起音分别73/40/74/60次。初次动态2022因本轮排队期间更新CSV校验被依赖SHA保护拒绝，未进入物理；保留失败记录，用新快照2030重试通过。无使用其他用户分配；2015仅取消本轮尚未运行的两帧模板烟测。没有修改训练奖励/统一入口/归档、历史视频或正在运行的训练。
- **交付／验证／TODO**：preview/outputs/left_flange_recalibration_20261008含基座、模板、标准74s及88s文件、四片完整记录和合并dataset；当前8787服务用该dataset和隔离runtime，保留旧WZX媒体。27项CPU测试通过；Chromium--disable-gpu真实三片播放、同步、横纵缩放、模板下载、刷新任务和实际文件上传→严格场景缓存完成通过，旧3WZX媒体及当前视频/图/轨迹Range206，证据logs/left_flange_recalibration_20261008。原始MIDI与音乐起点仍缺失，四片F1=null，PNG原谱层明确缺项，完整原谱对比未完成；规定机器人强制深接触及琴键软限超程是现有仿真边界，真实力/硬件标定不宣称。第一路不是完整机械臂动力学。


## 2026-10-08：默认按键切换paper，关闭独立误触罚

- **用户要求／参考**：统一脚本按键legacy改paper、关闭和弦放大、关闭默认误触−.05，并说明更新配置。核对all_use_this_train.sh、外部default_20261006.json、固定source_arm_follow_v1归档内rewards/train及启动包装器、当前唯一奖励总表；可选日志skill不存在。
- **修改**：外部JSON仅改key_press_reward.params.mode=paper和false_press_penalty.enabled=false；chord_bonus原本false，继续保持，chord_scope=null。关闭项保留未启用weight=−.05。paper为0.6正确键连续行程分＋0.4exp(−1.25错键数)，本身仍降低错键得分；fingering_reward仍legacy。统一脚本通过绝对路径读取该JSON，更新其注释；总表原地更新公式、例子、准备静音基准1.4、8开14关及可选有臂9开13关，同步README、AGENTS和现有默认合同测试期望。
- **验证**：bash语法和改动格式检查通过；现有默认奖励CPU测试12项全部通过。用fake sbatch捕获真实双卡入口、独立临时解包固定归档、调用其真实训练parser/effective_config核验外部JSON完全生效，归档SHA9adb5f69…274ce匹配；归档RewardManager核验正确键全按下且错键0/1/2时按键得分1/0.714602/0.632834。核验奖励差异恰为上述两个字段、预算10,000,384、采样学习流水及TCP/Pitch双硬限制保持。证据与最终CPU解析配置在logs/paper_defaults_20261008/；审计中source-builder参数由外层包装器消费，未构造GPU场景或将CPU解析文件冒称实际运行config。
- **范围／TODO**：本次选择现有受支持公式，不修改算法或固定源码归档；外部JSON覆盖归档内部旧默认。保留启动时打印完整最终奖励机制。未提交训练、未改历史实验或运行作业；新组合的收敛/F1尚未训练验证。

## 2026-10-08：修复工作台切换数据后旧视频地址404

- **问题／实际参考**：用户截图提示“视频暂时无法读取”，仍显示9.125s/3651点旧示例与22.10%分数。检查8787当前bootstrap已是wzx 88.092815933s新数据，原三个历史视频内容地址实际均404；新视频正常，原因是旧页面保留旧catalog而新服务只登记新media。参考trajectory_compare/__main__/app/static/app.js、旧load_catalog原SHA校验、新dataset与本次截图，未重新导出或改变轨迹。
- **修改／原因**：新dataset启动时额外登记通过旧manifest/来源SHA校验的默认历史媒体，保留只读旧URL，不导入旧slots/metrics/curve/cache或把旧视频替换为新视频。旧依赖损坏／缺失不放松校验、不阻断新数据集。Store生成服务session_id，增加无缓存/api/session及bootstrap字段；首页Cache-Control:no-store、前端API fetch:no-store，每15秒／恢复焦点／视频错误核对session，变化提示“重新载入”，用户点击前保留播放位置和已选数据。新错误提示说明服务切换后的页面重载办法。同步工作台README。
- **验收／边界**：20项CPU网页测试通过，新增会话重启标识和无缓存头测试；实际新3／旧3视频Range均206/video/mp4，Chromium--disable-gpu验证最新三视频可播放、89.12s、F1明确缺原MIDI无旧分数，模拟session变化出现重载按钮、点击前播放位置不变、点击后恢复新组，证据logs/trajectory_compare_media_fix_20261008/{check_browser.py,validation.json,session_changed.png,current_dataset.png}。服务4143896仍监听8787，新dataset/runtime保持；修复前旧JS没有会话检查，用户需刷新一次。未运行GPU、修改媒体/输出manifest或训练源码/奖励/归档。


## 2026-10-08：10.8_14.12上传轨迹接入三路比较并核对小指映射

- **用户要求／实际参考**：用户要求把wzx_experiment轨迹导入当前三视频框架测试，目录本机不存在后上传10.8_14.12.zip；映射澄清回复“你帮我检查一下”。读取ZIP三文件原头／单位／异步时钟和所有有效位置、原始规划200/500Hz、此前明确SDK四指swing零度对应URDF最大值的合同、当前冻结scene_manifest与安装、1359原手／键XML、hardware_key_contact和rollout_prescribed的同次录制／音频／绘图／渲染实现及网页app/data/static。没有从附件文档执行指令，也未连接硬件／读取外部旧项目。
- **核验／转换**：安全解包ZIP并记录SHA7c4c4673…3853、三个输入SHA。实机七臂各15201发送样本、二十指各38001，不按27行循环；臂启动400点后14801点与规划q逐点差≤1.19e−12deg，手启动1000点后37001序列确认旧小指rotaback/MCP/swing/PIP→当前MCP/swing/PIP/DIP。四指规划到SDK gain≈.947464、offset≈19deg、整段最大残差<2e−6deg，其余小编码偏差单独记录，不改硬件零位或臆测标定。日志回放SDK→URDF沿用已确认swing .35−rad，其余直接转单位；不追加拇指24deg/腕.15rad或移动基座。每关节自身指令/反馈时钟ZOH、缺数值沿上个真实同关节值，开头未知不填；源1.907642593–90.000458526s无缩时，35239个400Hz状态，age/缺失/完整映射证据保留。
- **本次真实视频／边界**：新增任务logs/wzx_import_20261008_1412/{common,prepare,record,postprocess,render,finish,validate_mapping,test_pipeline,browser_check}.py；独立preview/outputs/wzx_import_20261008_1412。第一路下发驱动原20手指DOF，机械臂规定（未实施全臂动力学）；第二下发、第三反馈完全规定机器人、只有88键动态、22world-mocap手连杆。模型保留原质量/惯量/阻尼/驱动/左手排除，经CPU逐项对照；Slurm1957一次录制三路，规定q与mocap每步及SHA不变。三个25fps/2228帧/89.12s H264/AAC视频，左臂左手、面对演奏者＋俯视；声音与PNG实际按键来自相应同次400Hz琴键激活，未重跑策略。保留深度强制接触与软限位超程（command/feedback峰值求解力约115108/133744N），不当硬件受力。
- **框架修改**：trajectory_compare新增显式--dataset校验加载离线catalog（媒体在目录内、manifest/模型/几何SHA），启动到同一8787本地服务，独立runtime保留历史默认。明确第一路边界/缺MIDI F1状态与“实际按键记录”链接；原谱缺失时不把实测图叫原谱比较。动力学缓存的实际执行曲线与输入目标分离：job返回executed curve并持久化/恢复，前端切换该曲线，避免目标曲线配反馈录像。仍未接入未知新文件通用Slurm worker，capabilities.export_adapter=false。同步工作台README和video_export_cli。
- **验证／修复／TODO**：19项CPU网页测试通过，新增缓存执行曲线恢复和数据集散列/越界测试；27角CSV全部精确读回、键状态/事件、非零音频和三视频完整H264/AAC解码、独立FK误差1.78e−15、模型资产SHA/交付manifest通过。浏览器三路播放、26点切换、独立XY缩放、同步seek/play、单路音频与手机不溢出通过并留截图。1957声音阶段私有依赖路径缺失、1958缺pulseaudio子目录均记录失败，补私有LD路径后CPU同次录制后处理成功；1959渲染反复NPZ解压取消自有作业，数组一次载入后Slurm1960成功1passed78.68s，A100 EGL明确分配内，不重新录键或改旧训练归档。原始MIDI和音乐起点仍未提供/无法识别，F1=None、实际PNG原谱层显式缺失，**完整原谱对比尚未完成**；提供后可在同次keys记录补分/图。训练奖励/源码归档/默认入口和旧导出均未改。


## 2026-10-08：可选机械臂方案A、子步IK／硬限与wrist同步

- **要求／实际参考**：用户先要求记录显式七角、抬高后t=0反算固定基座完整位姿、安装、观察和同步问答，随后授权实现；最终选择方案A、实际wrist使用限幅后臂FK完整位姿、每200Hz物理子步求解／检查、机械臂独立额外最多−5/控制步。参考本轮确认、AGENTS、统一入口／父固定TCP归档、项目内M6 URDF与SDK快照、坐标／旧预览说明、bundle/scene/task/rewards/control_limits/checkpoint等；可选日志技能缺失，不读其他工程运行源码。
- **实现／位置**：新增arm_options/arm_follow/arm_scene，初始角默认None，统一入口和源构建器在申请GPU／训练前校验；安装只取明确Link7变换，不取旧seed/base。CPU双精度由真实抬高首态反算固定双基座，保存世界／钢琴坐标；子步连续种子阻尼IK、有界回溯、SDK速度／加速度与URDF位置/制动共同约束，方案A下发FK驱动实际手根。机械臂模式结构性移除腕PD自由度但保留动作通道，手指／键继续积分；28维q/v观察，局部reset保留其他环境。原始需求、速度／加速度和位姿残差可诊断；数值异常在提交该子步新臂状态前抛错，不引入B/C或额外失败罚。
- **奖励／接口**：新增独立arm_motion_limit_penalty（默认无臂false；--arms默认true），weight1、95%SDK、超软阈值10%满−5、双臂/关节/导数/子步取最大，每控制步一次不乘dt。当前表22项无臂9开13关，有臂默认10开12关。train/evaluate/preview/render和checkpoint携带并严格比较初始角、安装、基座、观察／执行合同；不放松旧模型兼容。同步唯一奖励总表、需求问答、README、AGENTS、HPC说明及固定归档入口。
- **验证**：CPU相关79项及124子测试通过（GPU项在本机跳过）；补充位置边界600次饱和／反向测试发现1e−16rad区间浮点交叉，增加制动余量后通过。独立原生MuJoCo验证首态全位姿和子步FK同步；A100 Slurm1961基础步进/局部reset通过，1962通过原谱子步诊断、耦合、腕/TCP硬限、独立惩罚、逐子步同步和一次SAC更新。最终不可变归档SHA `9adb5f697017461826c3eb4cdcbb3b5bbf10debb8d88a5c773455847874274ce` 经Slurm1963同项集成复验1passed、COMPLETED/0:0；隔离归档CPU77项/124子测试通过（GPU项与归档外统一入口两项跳过）。真实入口用sbatch stub捕获无臂／测试七角两套参数，由最终归档解析完整配置，缺初始角在sbatch前拒绝；没有提交生产训练。GPU同步验收容差2µm/2e−5rad；测试角仅fixture，生产初态留空。记录在`logs/optional_arm_follow_20261008/`；本条最终验收结果在归档完成后补记，归档内文档保留打包时记录，运行时代码与默认JSON逐文件SHA一致。
- **部署／边界**：从父c3908914…47a8归档仅按本任务白名单派生`source_arm_follow_v1.tar.gz`并携带项目M6资产，更新SHA；旧归档／实验／运行作业保持，没有启动生产训练。机械臂无动力学或接触力模型；旧腕几何防护两项投影须关闭，其额外几何诊断在臂模式明确不可用。当前实际wrist是完整FK，旧TCP/Pitch目标限制约束IK请求，不冒充实际根运动的额外硬限。控制周期整体capture在臂模式停用，单物理图／学习器配置可保留；512环境长期吞吐、收敛与硬件可行性未验收。

## 2026-10-08：调整TCP硬上限与软阈值

- **要求/参考**：用户指定TX/TY硬速度0.99m/s、硬加速度1.99m/s²，惩罚速度阈值0.9m/s、加速度阈值1.90m/s²（末尾输入结合上下文按m/s²解释）。核对统一入口、当前奖励JSON、rewards/control_limits/target_motion_limits及固定07d8764d…89d4归档；沿用已确认机械臂式限幅前需求口径。
- **修改/保持**：只改根入口四个双侧硬限制CLI数值、两个TCP奖励阈值及默认JSON对应两字段；不改算法、模式、运行时Python、手指/腕参数、10%线性封顶−5、控制频率或训练预算。直接同步唯一奖励总表当前参数/公式/实例/硬控制行，README、AGENTS和HPC说明。速度满罚需求0.99m/s，加速度满罚需求2.09m/s²；需求加速度1.99时单独扣约2.3684。20Hz从静止首步下发最大4.975mm。
- **部署/验证/TODO**：从上一TCP归档按配置/文档白名单派生`logs/tcp_thresholds_20261008/source_tcp_motion_limits_v2.tar.gz`，保留全部运行时代码与历史归档，更新根入口路径/SHA。数值CPU核验、实际shell入口stub→新归档parser完整配置对照、父/新哈希差异记录保存在同目录；这只是配置解析与CPU验证，未提交训练、未执行GPU。真实GPU物理集成/CUDA graph仍待Slurm验收。

## 2026-10-08：TCP TX/TY目标双硬限制与限幅前需求惩罚

- **要求/实际参考**：用户指定双侧TX/TY硬速度0.95m/s、加速度2m/s²，并加入10%超限罚；澄清回复要求按当前机械臂惩罚口径。只读核对隔离项目`/home/houj/projects/tsensei_arm_986_experiment_20261007/tsensei_v3_2_pianist/mjlab_env/arm_motion_limits.py`中`IKMotionLimiter.apply`：原始需求减前次下发目标求速度，再减前次下发速度求加速度。另参考当前control_limits/target_motion_limits、task、reward_config/rewards、train与根入口及父闲指归档；不导入隔离机械臂代码、资产或运行环境。可选日志skill未安装。
- **实现/原因**：现有最终目标投影扩展到左右TX/TY，CLI与canonical/resolve支持平移加速度、逐通道单位为m/m/s/m/s²。根入口双侧每轴硬值0.95/2，保留位置边界离散刹停与float32向内舍入。新增运动项参数tcp_velocity_limit=.95、tcp_acceleration_limit=2；真实task.step将限幅前完整物理需求传给RewardManager，TCP以该需求与前两步最终下发历史求导，避免硬限幅把超限罚归零。全体手指/腕旋转/TX/TY两类超限仍取共同最大比例，10%扣满整项−5，每控制步一次、不乘dt、不额外叠加TCP预算；初始/局部reset为对应中性下发目标、静止历史。旧含运动项但缺新字段JSON明确拒绝，不增加可切换旧公式模式。
- **保持/文档**：手指20/400、腕旋转3/15及Pitch硬3.1416/15.708、hard_v1、闲指20步/L1−.1、其他奖励/预算/模型保持。直接更新唯一奖励总表的参数/公式/硬控制行与实例，同步README、AGENTS、HPC说明和本记录；历史归档、JSON、checkpoint及运行作业不改。HPC说明归档同时收录已存在的584历史验收文字，不改变运行源码。
- **验证**：CPU相关回归96通过、148子测试通过，1个GPU相关测试跳过；覆盖四轴三种dt、1000步随机/满幅/换向与边界刹停、实际float32下发差分、部分reset、±阈值/5%/10%罚分、混合单位共用预算、TZ/踏板排除、缺需求/非法参数拒绝、真实task.step配CPU场景替身传递原始需求、每步单位/权重及现有奖励/控制合同。新归档解包回归与真实入口stub解析结果保存在`logs/tcp_motion_limits_20261008/`；stub不提交作业。
- **部署/TODO**：从固定`c7d0df085f1b14ed004b91c8efc3f851236b41bc63c60a7e288cd22f1098369e`父归档按明确白名单生成`source_tcp_motion_limits_v1.tar.gz`，统一入口同步新路径/SHA；其余父文件逐字保留，归档散列/文件差异见archive_audit.json。保留启动完整奖励打印。没有训练/GPU执行，GPU物理集成与CUDA graph待Slurm验收；目标差分合规不代表真实TCP/qvel/qacc或实机安全保证。

## 2026-10-08：闲指改为当前及未来20控制步均未用指，L1系数−0.1

- **用户要求/参考**：当前MIDI未分配且lookahead未来20步也未用到才算闲指，目标偏差系数从−.04改为−.1。核对统一入口、默认奖励JSON、reward_config/rewards/task/task_logic、奖励总表及固定父归档`e821fb42…cfb0e0`；可选日志skill不存在。
- **实现/位置**：在现有`idle_target_l1_v1`直接替换仅当前用指的判定，不新增模式或保留该当前帧退路。新增必填正整数`lookahead_steps=20`；基于训练MIDI已保存指法/Shadow控制网格预计算当前k至k+20的用指并集，按每个环境当前索引取掩码，曲尾零填充不回绕。缺掩码、缺窗口的旧L1配置明确拒绝。保留最终展开物理目标相对真实reset目标的L1、从动关节、weight=1、不乘dt、不平均、不封顶。
- **边界/保持**：20Hz未来20步为1秒，包含第20步不含第21步；准备段可豁免即将弹首音的手指，休止/曲尾依同一规则。仅修改该项参数和判定，策略观察`--lookahead 10`、0.15s提前指法观察、其他奖励、预算、控制限制/模型均保持；不把奖励窗口变更静默扩大为策略观察改动。历史姿态模式不是本次修改对象，历史JSON/归档/检查点与运行作业保持。
- **验证**：新增/更新24项CPU测试通过，覆盖窗口边界、左右手、持续音、准备段、休止、曲尾、不同环境索引/局部reset、旧配置/无效掩码拒绝、无提前奖励时仍豁免、耦合从动/物理rad/权重和dt。扩展回归112项及145子测试通过；另2项旧task_conditioned测试在父归档同样失败（旧fixture错误假设template仍为Gaussian），未更改无关旧测试掩盖问题。默认JSON差异核对仅系数与lookahead_steps。
- **部署/文档/TODO**：同步唯一奖励总表对应行和准备段定义、README、AGENTS及根入口注释；从父归档按白名单生成`logs/idle_lookahead20_20261008/source_idle_lookahead20_v1.tar.gz`并同步根入口SHA。构建差异/父文件哈希、CPU回归、独立解包与入口stub最终配置核验保存在同目录。启动完整奖励打印机制保留；本次未提交训练、未运行GPU，GPU集成与长期效果待Slurm验收。

## 2026-10-08：同步原版Shadow的乐谱目标与奖励时序，删除旧公共采样模式

- **用户要求/参考**：用户要求修复当前目标/奖励计算，与原版Shadow同步并删除被替代的错误计算模式。核对官方RoboPianist `piano_with_shadow_hands.py` 的NoteTrajectory、`_goal_current`和末态奖励，以及官方`music/midi_file.py`、`piano_roll.py`；本项目task/task_logic/task_bundle/rewards/performance_options、train/evaluate、统一入口及父固定归档1334522e…56cf9。可选日志skill未安装。同步范围是乐谱目标与公共奖励时序，不将A3.2动力学、额外罚项或硬限制冒称原版配置。
- **实现/原因**：训练goal/当前及提前指法始终使用原版NoteTrajectory控制网格；第k步末态对动作前第k帧评分，之后才推进观察。删除精确端点覆盖goal/fingering、公共子步平均及正确键激活率混合公式；删除4个performance/CLI开关，schema升级2，旧字段/旧schema明确拒绝。保留原MIDI事件、176维观察和200Hz实际按键诊断，记录`goal_at_control_start`仍是原谱诊断。控制帧F1准备/曲尾归类改回同一帧索引，原谱物理指标保持独立。严格谱覆盖现在同时校验控制网格及原事件；无法量化而丢失/合并的音符明确报错，不改写原谱或指法。
- **保持项**：9开12关、各奖励weight/params、同手每对每物理子步−.75累积、手指20/400、腕软阈值3/15和目标硬限制3.1416/15.708均保持。未改物理模型、控制频率、预算或现有运行作业。更新唯一奖励总表、AGENTS、README与相关当前测试；旧JSON/归档/检查点原样保留并使用匹配旧源码。
- **验证**：工作区CPU回归156通过、7个GPU/图相关项跳过，另乐谱转换/严格覆盖12通过；覆盖音符切换、矛盾精确端点不能覆盖动作目标、静音/终止/局部索引、连续键程不混激活率、公共末态与瞬态诊断隔离、准备20帧/曲尾5帧、旧配置拒绝、同手累计与目标运动回归。实际Twinkle34音符控制网格覆盖全部通过，开启原谱诊断与关闭时goal/fingers逐元素相同。首次OSMesa库缺失后使用`MUJOCO_GL=disable`进行无渲染CPU检查，不涉及GPU。
- **部署/证据/TODO**：从原固定归档仅按白名单替换本任务文件，生成`logs/shadow_score_sync_20261008/source_shadow_score_sync_v1.tar.gz`；统一脚本同步路径/SHA。同目录保留构建、差异审计、独立解包CPU验证和stub捕获真实双卡入口/新归档parser最终配置。启动仍打印完整奖励。没有提交训练或GPU验收，CUDA graph/真实GPU集成及长期F1效果待Slurm验证；CPU通过不宣称物理安全或收敛提升。

## 2026-10-08：奖励完整总表与直接替换旧模式的维护规则

- **用户要求/实际参考**：用户要求在对话和文档给出当前最新奖励/惩罚完整表，参数需逐项清楚，并写入AGENTS：今后修改奖励直接删除被替代旧模式，不再不断叠加humanoid式多模式。参考根统一入口、外部default_20261006.json、固定self_collision_substeps_v1归档，以及reward_config/rewards/task/task_logic/self_collision/exact_score/performance_options、train、任务常数、琴键触发和指法目标代码；日志skill指定路径不存在。
- **修改位置/原因**：将已有 `docs/reward_defaults_20261006.md` 原地整理为唯一持续维护总表，覆盖21项（9开12关）的enabled、weight、全部params、公式/单位/采样/上限/依赖，额外详解key_press参数、准备/休止/终止、控制与真实状态区别。明确当前legacy、闲指目标L1、手指20/400及10%封顶、同手每对每子步−.75，旧四模式表不再作为当前默认。AGENTS新增同步维护与替换旧模式规则，README添加唯一入口链接。
- **验证**：CPU fake sbatch/python捕获真实统一入口，使用固定归档真实parser/effective_config解析，最终奖励与外部JSON完全相同；归档SHA1334522e…56cf9匹配，645个隔离归档成员与tar一致，45个相关运行文件及默认JSON与工作区逐字一致。审计/最终配置/参数证据保存 `logs/reward_catalog_20261008/`；另核对表格字段完整性、分值例子与文档链接/格式。
- **范围/边界**：本次仅文档与未来维护规则，不更改算法、数值、脚本、归档、当前旧分支、历史实验或运行作业，未运行GPU或提交训练。删除旧模式规则适用于后续对奖励算法的替换，历史归档/旧JSON/检查点及匹配回放必须保留；当前同手新增子步累计的GPU验收状态不因本次CPU文档核对而改变。

## 2026-10-08：同手接触按物理子步累计扣分

- **用户要求/参考**：用户指出原周期并集罚无法区分接触持续时间，要求同一对连续碰10个子步、每次−.2时合计−2。核对原固定reward_defaults_v1归档、`self_collision.py`、`scene.py`、`task.py`、`reward_config.py`、`rewards.py`及碰撞/奖励CPU测试；可选日志技能不存在。
- **修改位置/原因**：新增显式`self_collision_reward.params.aggregation=substep_sum_v1`，tracker按每个物理子步内body对去重后累加，任务快照/信息新增`self_collision_pair_substeps`，RewardManager按−weight×累计数计分、不除以子步数或再乘dt。周期末forward只补原并集诊断，不多扣一次；局部/全量reset清理各自计数。碰撞诊断关闭时也能累计。当前默认JSON保持weight=.75，因此连续10次为−7.5；闲指、跨手和其他奖励/物理/控制不变。空params保留历史并集与dt合同，未放松checkpoint检查。
- **部署/文档**：从原`1bd541f1…523e9`归档仅替换9项本次文件，生成`logs/self_collision_substeps_20261008/source_self_collision_substeps_v1.tar.gz`（SHA `1334522eaffa128dbe93c32996d65b93227a3bda4ab978916cd8c95fb1a56cf9`）；父归档与其他文件不变，统一入口更新默认归档/SHA，保留同期新增的显式归档覆盖。新增专题说明，同步README、奖励说明和AGENTS当前规则。既有作业1931及其旧奖励JSON/固定归档未改。
- **验证**：新增CPU Warp接触缓冲→tracker→奖励测试覆盖连续10次、短接触、多对、多geom、跨手/琴键过滤、接触释放、末端刷新不重复扣分、诊断开关、reset隔离、逐子步单位和旧合同。工作区相关回归68项及121子测试通过；真实统一入口经sbatch stub捕获并用新归档parser解析，奖励合同/权重/公共端点、双卡预算和hard_v1正确，启动仍打印完整最终奖励，未提交作业。归档隔离验证及散列审计保存在同目录。
- **边界/TODO**：扩展回归发现`test_real_wrapper_chain_preserves_explicit_reward_only_overrides`使用已删除项的旧`configs/rewards/current.json`，父版本同测试也失败，保存父/新失败日志并在相关通过集单独排除，没有改旧JSON掩盖兼容拒绝。GPU场景/CUDA graph和训练效果待Slurm验收，未执行GPU或训练。本次逐子步计次数随物理频率变化，不宣称相同秒数跨频率罚率不变。

## 2026-10-08：新增paper／旧闲指−0.5／20与400从头训练1931

- **用户要求/参考**：按上轮对比表新增paper、旧闲指整项惩罚上限−.5、手指目标运动20/400、从头训练。参考986实际config、635奖励说明、统一入口、submit/job包装、原82199601…bda022运动限制归档及其原训练解析器。可选日志技能此前确认不存在。
- **修改/原因**：新增`configs/experiments/paper_idle05_motion20_400_20261008/{rewards.json,README.md}`及`logs/hpc/paper_idle05_motion20_400_20261008/`审计。为统一入口添加成对且SHA强校验的前置`--source-archive/--source-sha256`覆盖，允许显式旧奖励实验使用匹配归档；没有新训练脚本、算法/归档修改或默认奖励改变。README和实验对照补充新增作业链接。
- **执行/验证**：真实shell→submit捕获→固定归档CLI CPU预检通过，相对986仅paper模式与旧闲指上限两字段不同，36个非奖励有效区段相同；默认参数逐项不变，缺配对/错SHA拒绝，未初始化CUDA。08:38:14通过根入口提交双A100 Slurm1931，从头seed42、10,000,384样本、20/200Hz/512环境；物理0/2基础GPU检查通过，实际配置与预检、原源码/场景/任务/动作/观察/控制/准备/耦合合同匹配，无warm-start/finetune来源。预检、提交和作业启动均打印当前奖励，08:41:25首次学习核验为380向量步、194,560样本、25,088次SAC更新，训练数值有限，记录在startup_validation.json。
- **边界/TODO**：保持旧子步公共评分、实际激活.5、同手−.2、跨手−.5、独立误触关、旧闲指姿态封顶和目标惩罚最大−5，不混入新L1默认。1931在A100，635/986在H100；奖励有两项差异，不能作为只改一个参数的对照。长期F1、接触与运动指标待完成核验；本次没有操作旧作业、修改历史checkpoint或导出视频。

## 2026-10-08：全部现有实验参数与F1分组对照

- **问题/实际参考**：用户要求统计奖励、环境、初始姿态、机械臂等参数，判断哪些改动有助F1。读取根统一入口、实际config/best/summary、10回合复评leaderboard、近期H100完成快照、耦合/时序/奖励消融/月光中止专题及tactile_benchmark说明；只读取本项目保存证据。可选`pianist-v2-change-log`技能文件不存在，按本项目日志要求记录。
- **修改位置/原因**：新增`docs/experiment_comparison_20261008.md`及`logs/experiment_comparison_20261008/`的50项主实验CSV/JSON、78份本地config盘点、原始来源/散列与CPU构建脚本；同步README入口。失败、仅启动快照、烟测/性能、回放、未运行计划及触觉固定轨迹诊断分别标记，避免作业数被误当正式实验数。逐行区分200/400Hz、原MIDI/量化目标、训练最差/合并、独立复评和视频单回合F1。
- **结论/核验**：50项主表实际config SHA全部匹配；三组逐字段审计确认976→635仅legacy→paper、632→976仅旧闲指上限1→.1、976→986仅目标阈值10/200→20/400。best分别提升11.0909、4.3559、9.4378个百分点，均为单seed同10M证据。近期21组完整准备合同相同且全部无臂；S3/S4 F1接近但手接触差异显著。生成CSV回读、F1范围/缺项和文档链接检查完成；审阅修正R5实际是网络微调而非恢复学习器。未新增算法测试、CUDA、训练/回放或历史归档改动。
- **未解决/TODO**：729/988仅有历史启动快照，免密码只读远程查询认证失败，当前状态/最终F1未知；当前L1/同手−.75/误触−.05/端点完整默认未发现长期验证，不能借用旧986分数。初始姿态/机械臂没有干净单因素提分证据，多seed及paper×20/400组合仍待验证。本次仅统计建议，未改默认、提交训练、GPU推理或重新导出视频。

## 2026-10-07：单左手摆位、实际琴键高亮及五档接触力录制

- **用户要求／指法确认**：用户要求只保留一只手，先摆好手指，显示实际激活的琴键，按不同velocity下压记录接触力；随后明确选择原始左手，小指C4、中指E4、拇指G4。实际参考benchmark的physics/env/config/recording/video/reporting/audio模块，固定归档内bundle.py、琴键和手模型、此前velocity试压数据；没有读取禁止的旧模型目录或修改固定源码归档。无训练、CUDA或GPU渲染。
- **单手环境**：新增`single_hand_model.py`，从原始全精度MJCF移除另一手的body树与引用约束／驱动，沿用已验证TZ/Yaw结构移除及编译后的锁定姿态，得到111个qpos/qvel（88琴键＋23手关节）和23驱动。避免mj_saveLastXML对惯量等参数舍入；剩余质量／惯量和驱动经原双手模型逐名比较。physics/env/config支持显式left/right/both；单手动作23、本体感知46、触觉5×2，旧both缺省保留调用兼容，不给单手填虚拟右手。CLI支持`--hand`和对应动作维数。
- **摆位与控制**：`single_hand_chord.py`按完整末节胶囊最低表面求解白键前部上方4mm左右的初始摆位，食指／无名指抬起，腕Pitch沿用0、Yaw沿用归档0.15，TX/TY由几何求解后在击键过程中冻结。琴键自然释放，初始姿态仅在reset设置；之后全部由物理积分。33个关节目标路径点对应14mm名义下压，再加入固定仿真静态试压补偿（中指MCP+.045、小指MCP+.12及拇指两个轴的拟合偏置，随下压比例平方施加），不改PD参数、不注入力或琴键位置。实际参数保存在`configs/left_c_major_placement.json`与每段metadata，不能称硬件标定或自然姿态。
- **测量与显示**：physics额外保存`tip_key_normal[sample,finger,key]`的MuJoCo法向求解力，不进入策略观测；reporting导出NPZ。video适配5指／40Hz控制、真实观测周期25ms；指定三键蓝色、同次200Hz真实键程事件对应的激活键橙色、接触点绿色，侧边显示C4/E4/G4的实测velocity。高亮帧数由同次实际事件核验，未按动作或目标谱着色为已激活。
- **最终实测**：`single_hand_export.py`冻结预备姿态，录制0.75秒无接触预备段，随后1.6/.8/.4/.25/.20秒下压、.5秒保持、.5秒返回和.5秒释放。C4实测velocity18/40/86/127/127，E4为15/32/68/115/127，G4为12/27/57/95/121；每段仅有指定3个NoteOn和3个NoteOff，超过.01N的指—键接触对恰好符合指法。三音起音跨度58.455/26.246/12.513/7.617/5.978ms，当前近似参考谱的50ms起音容差note F1均1.0；名义velocity16/32/64/96/127不是实际值，不宣称音量或时值完全命中。最快段全程单键力峰值5.37735N，预备力0。
- **产物／验收**：最终全量物理录制`outputs/single_left_recordings_20261007_v3/`；交付`outputs/single_left_chord_20261007/`含5段有声视频、预备图／实际激活截图、并排逐键力图和CSV、三音总览、同次实际／原始参考MIDI时序PNG、事件与原始矩阵、HTML和ZIP。无准备偏移／裁剪／倍速；视频40fps，200Hz物理，gain .8且不归一化。109项CPU测试通过；`validation/single_hand_audit.py`检查右手物理移除、所有接触对、逐键矩阵求和、200Hz键程精确重建事件、音符松开、激活高亮逐帧计数、CSV/NPZ逐值一致、原视频／配音帧及时间戳一致、音轨与完整解码、载荷SHA/ZIP CRC。实际查看预备帧和三键同时激活帧。网页仅绑定服务器127.0.0.1:18771，请求打开Codex右侧浏览器返回queued；HTTP资源另验证，未将queued当作客户端渲染证明。
- **试验保留／TODO**：最早未补偿试压只有中指触发；较快.15秒试验拇指短暂接触F4但没有触发其NoteOn，因此最终最快档改为.20秒，预试与中间录制均保留。XML舍入问题在最终v3前修复，最终全量重录，未拿中间轨迹配最终图。高velocity存在饱和、三音起音及力度不同，当前是固定诊断轨迹，触觉闭环策略／velocity精确跟踪和真机标定仍未实现。根README与benchmark说明同步；未改统一训练入口、奖励、历史视频或已运行作业。

## 2026-10-07：修复和弦总览的网页访问入口

- **问题／参考**：用户点击聊天中的HTML路径未见网页。检查原`chord_velocity_force_20261007/index.html`存在，实际参考benchmark README、`docs/video_export_cli.md`已有HTTP预览约定、当前`open_in_codex`工具定义及OpenAI官方Browser文档。文件链接不等于已打开网页，不能据此断言客户端不支持浏览器或发生版本故障。
- **处理／验证**：在TenseiNode1用benchmark独立Python启动只监听`127.0.0.1:18769`的静态服务，根目录仅为本次交付目录；首页与36个引用资源HTTP200，检查HTML/视频/图片等Content-Type，记录在`logs/hpc/benchmark_chord_browser_20261007/`。请求在当前线程右侧浏览器打开HTTP总览，结果queued，未宣称客户端渲染成功。benchmark README增加HTTP入口、远程端口转发和完整ZIP离线打开说明。
- **边界**：没有更改原始导出、视频、图、manifest或ZIP，无训练/GPU操作。客户端可达性仍需实际浏览器确认；远程项目的回环地址不自动等于用户电脑的回环地址。

## 2026-10-07：和弦velocity对照及视频旁同次逐键接触力

- **要求／实际参考**：用户要求不同velocity的和弦测试，并在视频旁附每次velocity对应的接触力变化。参考benchmark内`audio.py`、`velocity.py`、`physics.py`、`env.py`、`recording.py`、`video.py`、`reporting.py`、`validation/velocity_probe.py`及已有FluidSynth/TimGM6mb集成；遵循当前AGENTS，可选日志技能缺失。CPU物理、音频和OSMesa/llvmpipe软件渲染，无训练、GPU或真机操作。
- **音源测试**：新增`audio_chords.py`及CLI `audio-chords`；合成C4/E4/G4六档统一velocity32/48/64/80/96/127，以及平衡／根音／中音／高音突出四组。固定音色、1秒持音、每档2.5秒、gain0.8，无归一化；PCM第一秒RMS严格递增，首末24.0000dB。混合音量结果与单独合成的声部测量明确区分，保存音频、MIDI、JSON、图、CSV及SHA；独立读取MIDI核对全部时值和velocity。原独立试听在`outputs/chord_velocity_20261007/`。
- **物理和导出修改**：新增`chord_force.py`组织5组固定MCP/PIP渐变动作、原始录制、同次视频配音、力曲线和并排HTML。`physics.py`按实际琴键几何所属关节映射88键，在原求解接触循环中额外记录`key_contact_normal`；不改变动作、动力学或观测。`reporting.py`保存此200Hz评测向量，`video.py`显示实际各音velocity。各键力为所有手—键法向接触力大小之和，含侧面／底面，区别于视频原有完整末节胶囊全部接触的逐指力。新增独立几何核对测试；没有从velocity倒推或合成接触力。
- **实际结果／边界**：0.05/0.2/0.5/1/2秒目标渐变，各轨迹2.30/2.45/2.75/3.25/4.25秒、实际10/9/8/8/8音。共同键MIDI56按慢到快velocity11/26/48/111/127，实际音符持续段该键法向力峰值5.7178/5.6258/6.3380/6.5431/7.7406N；不是单调一一映射。所有误触保留。参考为原始合成C4/E4/G4（0.5–1.5s，velocity64，无指定指法），固定动作不读它或触觉，五组本视频note F1均0；音源和弦演示与物理诊断明确标注，不能称机械手准确完成指定和弦或触觉收益评估。
- **产物／同源检查**：原始完整模型与状态在`outputs/chord_force_recordings_20261007/`；交付`outputs/chord_velocity_force_20261007/`含5段有声视频、旁边的逐键力PNG、共同键起音对齐总览、逐音velocity／起音附近力／持续段峰值CSV、88键200Hz力CSV、同次原始MIDI／实际按键PNG和事件、完整试听及ZIP。无checkpoint，准备偏移0，无裁剪／倍速／末停；力取积分前求解时刻，比对应积分后键位早5ms，起音力取最近样本且偏移不超过2.5ms，未求解初始零值排除。
- **验证／TODO**：完整CPU回归106项通过（42条现有依赖警告）；5段视频46/49/55/65/85帧完整解码、音轨有效，配音前后帧与时间戳完全相同，琴键与接触点求和通过。`validation/chord_force_audit.py`从200Hz真实键程重建所有事件完全一致，CSV逐值对应原NPZ、交付与原始按键PNG逐字节一致，MIDI与音源RMS、载荷SHA和ZIP CRC通过；日志和审计JSON在交付validation目录。README和benchmark说明同步。没有更改训练源码归档／默认训练入口／历史产物。精确和弦控制、触觉闭环学习及真机velocity／力标定仍待实现。

## 2026-10-07：986最佳模型追加1500万样本988

用户授权986继续训练1500万样本。经H100原根all_use_this_train.sh末尾追加warm-start、完整原奖励和vector-steps=29297提交Slurm988；512环境批次共15,000,064追加样本。父986 best19532、F1三回合最差91.576674%，PT SHA350594d0731e20f20c11e70637161fa2471b6100bb14a781e988fe74483c11d9，config ef2e0a82…c0b59、源码82199601…bda022保持。恢复网络/优化器/温度/RNG和2,476,672更新计数，经验池/物理场景重建。

CPU预检仅requested_vector_steps和requested_transitions不同，其余有效参数及完整奖励与986相同：legacy、闲指−.1、手指20/400、腕3/15、运动超限10%整项−5、同手−.2、跨手−.5、原20/200Hz子步评分及位置hard_v1/Pitch目标3.1416/15.708保持。奖励在预检打印并由作业启动再次打印，没有兼容豁免、算法或默认改动。审计在logs/hpc/continue_986_15m_20261007/；预算仅本轮末尾覆盖，未新增训练脚本。北京时间14:04:56启动验收通过：988 RUNNING，225,280追加样本、32,768新增学习更新，总更新2,509,440；全部合同/奖励/来源匹配，数值正常，双H100物理1/5。 效果待后续评估。

## 2026-10-07 13:58：986／987训练完成核查

986随机初始化于13:53:05完成，best/final19532原谱音乐F1三回合最差91.576674%；987从976网络微调于13:52:01完成，best18500为96.690817%、final19532为96.356116%。两组各10,000,384样本、2,476,672更新，Slurm COMPLETED/0:0；20/400和976其他合同再次核验通过，所有训练数值有限、未发现ECC/CUDA/致命日志。CPU读取，无新策略回放或续训。

同手接触仍存在：986 best/final三回合最多9个物理子步；987best最多4、final最多27；跨手和键底接触均0。因此不宣称碰撞消失或物理安全。986较同预算976best82.14%提升9.44个百分点；987额外训练1000万样本，不能把其提升全部归因于阈值变化。审计保存在logs/hpc/finger20_400_20261007/{scratch,finetune}/current_status.json。

## 2026-10-07：benchmark接入velocity敏感钢琴音频

- **要求／参考**：用户要求接入音源并听到velocity导致的音强变化。实际参考`tactile_benchmark/{src/tactile_piano_benchmark/{cli,events,velocity,reporting,video}.py,validation/velocity_probe*,tests/test_portability.py}`、主项目`preview_audio.py`及`preview/scripts/setup_piano_audio.py`、现有用户目录FluidSynth/TimGM6mb运行库与包清单／许可。遵循当前AGENTS；可选日志技能缺失。CPU离线合成，不启动训练、GPU或真机。
- **修改／独立性**：新增`audio_runtime.py`校验并提取独立音频归档，`audio_worker.py`以本目录动态库在子进程运行FluidSynth2.2.5，`audio.py`提供事件校验、固定增益合成、单键六档演示及同次视频配音，`audio_probe.py`从原200Hz轨迹重建核对事件并生成物理速度试听。CLI新增`audio-demo/audio-probe/audio`及rollout/smoke的`--audio/--audio-gain`。音频运行库归档SHA`fd72370dbc5ebd42799eaef93c1189acdc3904dac340b9ae867eb90b22a2085f`，原包清单和许可保留；独立复制无需父项目音频文件，不改系统或原训练归档。根README、benchmark及vendor说明同步。
- **声音合同**：实际事件velocity原样传入，不采用旧预览默认127；固定音色、bank/program0、48kHz双声道PCM16、关混响／合唱、post_gain1；不归一化、压缩或限幅。浮点峰值达到.999报错，统一调整所有对照的显式gain。保留原始事件时值和截断，不伪造末端NoteOff；派发取最近音频样本，音源内部块延迟未标定。重导出已含音频的记录时，排除旧WAV／报告，防止覆盖新增益声音；空事件可合法导出静音。
- **实测／产物**：先以0.2确认波形范围，再以整套相同0.8增益生成`outputs/velocity_audio_20261007_listen/`。同C4、同0.8秒按键、每2秒一次，velocity32/48/64/80/96/127，数字RMS严格递增且首末相差约24dB。原物理探针从键程重新构造事件逐项一致，音高56的velocity11/26/48/111/127按慢到快播放，保留各片段内真实时值；单键听音版为事件子集，全键音频另存，用统一0.3秒起音窗测RMS。原force/none录制分别配AAC有声视频，保持原视频帧／时间戳和同次按键PNG；原快速诊断全部127，两组音频一致，不能用它们演示力度梯度。数字强度不是已标定声压或真机力度。
- **验证／TODO**：新增非法事件、velocity／固定gain真实音频响应、实际回放音频、重复导出增益和独立目录复制音频测试；原94+录制2项与本轮音频10项一起回归106项通过，日志在产物validation。两段MP4全片解码、非零AAC音轨、原视频逐帧MD5及时间戳一致已检查；音频样本数／事件／源码和运行库SHA保留。当前提供离线音频与视频，不提供实时声卡播放；更高质量多层采样、音源选择、实际声压和真机映射仍待后续。

## 2026-10-07：触觉benchmark接触力视频与无触觉对照

- **要求／参考**：用户要求实际测试并导出视频、仿真实际接触力及无触觉结果。读取当前AGENTS、`tactile_benchmark/{README.md,configs/force.json,src/tactile_piano_benchmark/{cli,physics,env,bootstrap,reporting}.py}`、`scripts/hpc/submit.sh`、`docs/video_export_cli.md`、现有CPU软件渲染流程与用户目录OSMesa/ffmpeg。可选日志技能缺失。未启动训练或加载旧checkpoint。
- **修改位置／原因**：CLI增加`--record-video`；新`recording.py`只读保存同次回放的MuJoCo完整控制帧状态、编译模型、原始末节接触点及求解力，避免视频另跑轨迹。新`video.py`强制OSMesa/llvmpipe软件渲染，检查文件SHA、完整姿态中琴键位置与物理记录一致、接触点力求和与200Hz力记录一致；显示力来自原始记录，不使用渲染时mj_forward重算值。新`compare_export.py`校验并打包双组物理／观测CSV、HTML、ZIP、总览PNG，保留各组同次原始目标／实际按键图及JSON/MIDI。新增只读记录与求解力一致性测试；根README和benchmark说明同步命令与边界。
- **测试设置／结果**：相同合成25音连续时间原谱、seed42和固定servo_probe，仅force／none观测不同；各265控制步、2651条200Hz物理记录、13.25秒。两组物理、完整状态、动作、目标和接触点记录完全一致；单末节法向力峰值19.981775986N，法向力总积分250.824242443Ns；force观测非零，none全零。各实际90音、匹配0、漏25／多90、note F1=0，velocity MAE=null。固定动作不使用谱面或触觉，数据不能证明学习收益。力是完整末节胶囊所有接触的MuJoCo求解值，非已标定指腹／真机力。
- **产物／时间合同**：`tactile_benchmark/outputs/contact_comparison_20261007/`保存双组1280×720、20fps静音视频，接触点与力数值、同次按键／原始目标图、200Hz法向／切向力CSV、20Hz实际触觉观测CSV、NPZ、实际／目标MIDI、配置和SHA。无准备偏移、裁剪、倍速或末停；265视频帧覆盖13.25秒，终点状态仍保留。接触力时标比对应积分后姿态早5ms；接触点20Hz显示，原始力200Hz，初始零值注明初始化。未改既有模型／驱动／动作／奖励合同、训练入口、归档或历史产物。
- **验证／TODO**：录制新增测试及原CLI测试23项通过，完整回归96项通过；两段视频全片解码通过，琴键和逐指力核验通过。测试日志及来源校验结果在产物validation及各组video_validation.json保存；打包器复核各组载荷SHA并检查ZIP CRC。CPU OpenGL实际为llvmpipe，未使用GPU。接入触觉闭环策略、真机接触力和velocity标定仍待实施；当前无触觉只屏蔽观测，不移除接触物理。

北京时间12:45:09启动验收：986与987均RUNNING，各187,392样本、23,296次学习更新，数值有限；物理卡分别1/5与2/3。两组完整奖励、20/400及父合同再次匹配。

## 2026-10-07：默认手指惩罚20/400及986从头／987继续976

- **请求/参考**：用户授权默认手指速度阈值20rad/s、加速度400rad/s²及两组新训练。参考本地统一入口/默认JSON、976完整config/best与原源码、H100原统一入口和奖励变化finetune合同。
- **修改/执行**：all_use_this_train.sh的手指覆盖和configs/rewards/default_20261006.json的两阈值更新；AGENTS/奖励说明/README/H100说明同步。H100原统一入口只同步两覆盖值。两组使用976完整奖励JSON仅改20/400，提交986随机初始化、987从976 best19000继承网络，两组各10,000,384样本；987因奖励变更只开启allow-reward-change，优化器/温度/RNG/经验池重建，不更改控制合同。审计CPU辅助/请求/来源/实际配置在logs/hpc/finger20_400_20261007/，说明和JSON在configs/experiments/finger20_400_20261007/。
- **验证**：两组CPU捕获入口和原归档解析，仅reward_config的两阈值改变，其他有效配置相同。实际config/所有来源/场景/任务/控制/动作/观察/原谱/指法/准备/耦合合同与976相同，原源码82199601…bda022及父PT a5e73111…190999散列匹配，双GPU基础检查通过，完整奖励打印。bash语法、审计辅助编译及git diff --check通过。
- **限制/TODO**：20/400为手指目标惩罚阈值，没有新增手指速度/加速度硬限制。保留976闲指−.1与其他奖励以及腕3.1416/15.708目标硬限制；没有应用后来默认闲指L1/端点评分，未改算法/归档/历史模型/硬件/其他作业。效果待后续评估。

## 2026-10-07：976最佳视频、同次原谱比较及F1曲线交付

- **请求/参考**：用户导出视频和F1；参考976完成核查/best19000完整config/原82199601…bda022归档、统一export_video.sh/Slurm入口及683同次比较验收辅助。保持原谱、奖励与控制合同。
- **执行/位置**：单H100作业985录制/渲染，CPU生成同次200Hz真实按键与原始MIDI比较；另读41次训练评估三回合最差绘制F1，无平滑。审计辅助在logs/hpc/export_976_20261007/，输出preview/outputs/experiment_976_best_20261007/，独立曲线docs/figures/experiment_976_f1_20261007/；本地HTML/ZIP加入曲线并重建manifest。README/视频说明同步，无训练/默认/驱动修改。
- **验证**：PT SHA a5e73111…190999、config8f69185f…274a2和源码匹配；1280×720/20fps/10.2秒H264/AAC完整解码；1830个200Hz真实样本、1秒准备和末停1秒，原谱/声音/实测子步/控制端点同源。远端21项和本地加入曲线后25项manifest/ZIP通过。视频单回合F1=81.57%，训练best82.14%/final81.86%另计；查看视频4.5秒帧、对比图及F1图，脚本语法/git diff --check通过。
- **限制**：仍有漏音；本次视频单回合和训练三回合最低不可混用。127.0.0.1:18768服务仅本机回环，浏览器打开请求queued不代表客户端渲染验收。目标限幅不等于物理安全保证。

## 2026-10-07 12:25：检查976对比训练完成结果

- **请求/参考**：用户检查训练情况；参考976 Slurm完成记录、实际config/归档、best.json及metrics.jsonl，和632原最佳同口径指标。
- **结果/位置**：CPU复核976于09:52:23正常完成10,000,384样本，best82.14%/final81.86%（200Hz原谱音乐三回合最差），较632best提高4.36个百分点。所有训练数值有限、无致命/ECC/CUDA日志，best/final碰撞及准备误触检查0。完成审计保存在logs/hpc/legacy_idle01_compare_632_20261007/completion_status.json，README和H100说明同步。
- **边界**：未提交续训、GPU推理或更改其他作业；仍有漏音，单seed尚不能推断显著性，训练评估不代表视频分数或所有状态安全。

## 2026-10-07：635视频左臂指定初始角差分IK及运动超限曲线

- **请求/参考**：用户要求635机械臂角度、速度和加速度曲线，并给定左J1–J7为77.48315/−70.55979/−61.07445/−81.20139/160.95941/14.69720/−25.86782°。635原no_arms没有臂策略数据；本次由同视频wrist_poses.npz（manifest验证）离线求解，参考preview_arm_ik.py、preview_trajectory.py、joint_motion_plots.py、arm_motion_limits.py和left_replan_smooth_20261006冻结安装／基座及项目内SDK限值。
- **位置/方法**：新CPU辅助与验收在logs/ik_635_left_20261007/，交付在preview/outputs/experiment_635_left_differential_ik_20261007/。使用原0–9.15秒、1831个200Hz位姿点，初角精确保持、已有基座固定，阻尼最小范数Jacobian逐点完整位姿跟踪；无随机首解、重定时、速度裁剪或滤波。保存角度/速度/加速度CSV和NPZ、SDK峰值／超限区间、PNG/PDF、来源、manifest、ZIP。未修改训练、历史导出、全局安装、归档或其他作业。
- **结果/验证**：J1–J7速度峰值34.28/45.70/25.54/46.74/33.29/12.51/60.61°/s，均低于SDK180°/s；加速度峰值1094.90/2902.02/1746.95/2334.65/3001.32/1126.41/6128.12°/s²，全部超过相应450/450/900/900/900/900/900参考值，最大J7为6.809倍（t=6秒）。独立URDF FK逐1831点检查最大误差5.275e−8mm／3.243e−8°，初角、差分重算、SDK原文件SHA、PNG、manifest、ZIP CRC通过。相关视频说明同步。
- **边界/TODO**：结果是该原时序轨迹的一组运动学IK解，非机械臂训练数据／实机测量；未检验负载、动力学、碰撞、控制器比例限速或采样间峰值，也未新增速度／加速度优化。仅分析左臂，右臂未授权初角。

## 2026-10-07：632闲指整项上限−0.1从头对比976

- **请求/参考**：用户要求开启632配置下闲指最高−.1的新对比训练。参考632实际config、奖励、原固定运动归档及H100根all_use_this_train.sh、既有CPU预检/启动验收流程；按632相同随机初始化、seed42和预算执行。可选日志技能缺失。
- **修改/提交**：仅修改本轮奖励JSON的humanoid_finger_gesture_reward.params.maximum_penalty_per_step从1到.1，完整奖励保存/打印。新配置说明在configs/experiments/legacy_idle01_compare_632_20261007/，审计CPU辅助/请求/实际config在logs/hpc/legacy_idle01_compare_632_20261007/。通过H100原统一入口末尾覆盖奖励/独立输出提交976，双H100物理1/5；没有新增训练脚本、修改全局默认、历史结果、算法归档或其他作业。
- **验证**：CPU捕获真实入口参数和有效配置，仅reward_config一个标量变化；实际config与632源码/场景/任务/动作/观察/原谱/指法/准备/耦合/控制合同匹配，hard_v1及双Pitch目标3.1416/15.708保持。原82199601…bda022归档散列匹配，双GPU基础检查通过，完整奖励作业内打印，从头模式无checkpoint或兼容豁免。README/H100说明同步。
北京时间08:47:28启动验收：976为RUNNING，340向量步、174,080样本、19,968次学习更新，训练数值有限；完整配置与GPU检查通过。

- **边界/TODO**：完整预算10,000,384样本、seed42，效果待训练评估；目标差分硬限制不保证真实qvel/qacc。新旧默认奖励没有混用，单seed对比不代表统计显著性。

## 2026-10-07：683导出页面浏览入口

- **问题/参考**：用户截图中的index.html入口显示为代码文件链接，未直接呈现网页；核对现有683 A100导出HTML/相对资源及OpenAI官方Browser说明。
- **处理/位置**：保持导出文件和manifest不变，仅在TenseiNode1启动127.0.0.1:18767静态HTTP服务，目录限定为683 A100导出目录，并请求Codex右侧浏览器打开index.html。PID、服务日志及HTTP检查保存在logs/hpc/export_683_browser_20261007/；视频说明增加访问办法。
- **验证/限制**：index.html、有声MP4、对比PNG的HTTP HEAD均200且类型正确。UI工具返回queued，仅代表打开请求排队，不宣称客户端已渲染；若桌面客户端与服务器不是同机，需SSH本地端口转发。仅CPU服务，无GPU或训练修改。

## 2026-10-06 20:32：683最佳模型原配置继续训练729

- **请求/参考**：用户在683视频后继续训练。参考683实际best19000/config/完成核查、H100原根统一入口与固定运动归档、既有632续训预检/恢复验证流程；可选日志技能此前确认缺失。
- **执行/位置**：通过已有根`all_use_this_train.sh`追加冻结父模型warm-start、原完整奖励JSON和独立输出，Slurm729于20:29:19提交，双H100物理1/5、512环境、追加10,000,384样本。网络/优化器/温度/RNG恢复，经验池重建；legacy、闲指−1、运动惩罚−5、位置hard_v1及Pitch3.1416/15.708双硬限制、其余所有配置保持。完整奖励打印，未新增训练入口、修改算法/全局默认或历史结果。审计辅助/请求/实际配置/来源验证位于`logs/hpc/continue_683_20261006/`，README/H100说明同步。
- **来源/验证**：父best19000训练三回合最差F1=.8923141891891891，PT SHA`aef93caab214301f83dea08b92ff6d41afb84677cae2fbf721f447b82cda4f00`；固定源码82199601…bda022，实际config SHA`bcf8ff1786b83bfe3c20032b8b0b0f9c45af1b74e2f53d44b73649caa603d3e7`。CPU预检全部有效配置与683一致，实际源码/场景/任务/奖励/控制/动作/观察/指法/准备合同一致、无兼容豁免，GPU探测通过；初始更新数4,881,152与父匹配。20:32:02已256000样本、新40576更新，总4921728，训练数值有限，辅助语法与git diff --check通过。
- **边界**：续训效果待后续评估；目标离散差分硬限制不代表真实qvel/qacc保证，本次没有GPU渲染或硬件操作。继续采用H100训练环境；上一次A100选择属于视频导出。

## 2026-10-06 19:41：683最佳视频切换A100并完成

- **请求/参考**：用户明确改用A100导出当前视频。参考A100 ECC历史、实际Slurm节点/队列、1791 GPU探测、683原PT/config/归档SHA、既有export_video.sh/submit.sh、同次MIDI比较及交付验证脚本。
- **执行/位置**：从H100复制原best19000/config/best.json，SHA逐项保持；源码仍82199601…bda022。网络传输较慢，独立用户目录安装SSH传输依赖，多连接分段复制后再次校验原PT；密码仅终端读取、未写文件/源码/包。新增审计/后处理辅助在`logs/hpc/export_683_a100_20261006/`。通过已有单卡Slurm入口1797在物理2号A100完成录制/渲染，CPU从同次记录生成对比图；未修改算法、训练配置、驱动或其他用户作业。
- **交付/验证**：`preview/outputs/experiment_683_best_a100_20261006/`包括有声MP4、同次原谱/实测PNG/PDF/CSV/NPZ/MIDI、HTML和ZIP。1797 COMPLETED/0:0/66秒，1280×720/20fps/10.2秒；1830个200Hz实际物理样本、1秒准备及末停1秒。单视频音乐micro F1=.8976842105263158，TP2132/FP3/FN483，独立于训练89.23%。原始源码/PT/config/目标/音频事件及控制端点匹配、全片H264/AAC解码、PNG/manifest21项与ZIP CRC/SHA通过；实际查看视频4.5秒帧与图。源码辅助语法及git diff --check通过，README与视频说明同步。
- **迁移/边界**：H100718取消，停止其CPU等待器和本地收集器，原待完成页面改指向A100成果。1791分配内6/7和1797物理2卡检查通过，但登录会话全卡NVML仍曾失败，不宣称全节点ECC修复；原记录保留，无硬件操作。可选日志技能缺失；本视频评估不代表所有状态安全。

## 2026-10-06 19:03：683当前视频导出718排队及自动交付

- **请求/参考**：用户导出当前视频。选择683 best19000；实际结果/源归档/config，既有export_video.sh、submit.sh及585/五组视频CPU比较与验证脚本，AGENTS同次实际按键和原始MIDI交付规则。
- **执行**：保持683原PT/config及82199601…bda022归档，经现有单卡Slurm入口提交718，正面双手、有声、1280×720、原速、准备段保持。用户原始授权内自动后处理与下载：远端CPU等待导出completed后生成同次200Hz按键图/数据、检查合同/媒体/散列/ZIP并打包；本地CPU收集校验manifest及视频解码后更新本地输出。未取消其他训练、修改checkpoint或在分配外用GPU。
- **位置/状态**：脚本、请求与工作状态集中`logs/hpc/export_683_20261006/`；本地`preview/outputs/experiment_683_best_20261006/index.html`目前为明确待完成占位页，完成后自动替换。远端同名输出；19:03 Slurm718仍PENDING/Priority，八卡由其他训练占用。文档同步。
- **验证/未完成**：输入PT/config/归档SHA及脚本语法检查通过，远端等待器和本地收集器状态确认运行；最初SSH连接过期导致启动失败已重新认证并恢复，凭据未写入源码/迁移包。当前没有视频/按键数据可验收，不宣称完整导出完成；未来实际回放F1与训练F1分别记录，GPU排队结束时间未知。最长自动等待24小时，失败状态可复核。可选日志技能缺失。

## 2026-10-06 18:46：检查683续训结果并导出F1曲线

- **请求/参考**：用户检查当前训练结果并导出F1。读取Slurm683实际sacct、metrics.jsonl、best.json、summary.json、config与启动SHA、错误日志；父632完成记录作为同口径参考。
- **结果**：18:29:03正常COMPLETED/0:0，新增10,000,384样本、2,476,672次更新；配置SHA保持、全部训练数值有限、未检出致命错误标记。best19000三回合最差原谱音乐micro F1=.8923141891891891，final=.8917949799620333，较父最佳提升11.4485个百分点。best同手/跨手/键底0，final同手最多7物理子步、其余0。
- **产物/验证**：`logs/hpc/results_683_20261006/status.json`保存状态、来源、逐评估、散列与检查；`docs/figures/experiment_683_f1_20261006/`保存PNG/PDF/CSV/source.json，41次完整三回合评估、200Hz统一最差口径，标注best/final及父632参考。图检查可读、数值与best/metrics一致；README/H100说明同步，git diff --check通过。
- **边界**：仅现有训练评估及CPU绘图，没有GPU推理、重跑策略、新训练或视频导出；未达到99%目标，零接触best评估不推广为全部轨迹安全。可选日志技能此前确认缺失，按项目日志要求记录。

## 2026-10-06：导出当前轨迹t=0左臂与左手全部关节角

- **要求/参考**：用户要求t=0各关节角；参考`left_base_from_start_20261005/left_arm_200Hz.csv`、`hand_trajectory/finger_joints_{rad,deg}_200Hz.csv`和冻结场景URDF关节映射及manifest。可选日志技能缺失。
- **产物/原因**：`logs/left_initial_joints_20261006/export.py`导出`preview/outputs/left_initial_joints_20261006/left_arm_and_hand_joint_angles_t0.{csv,txt}`与metadata/manifest；逐关节包含URDF名称、度和弧度，左臂7加左手20共27项。臂七角保留用户初值，手指使用原记录首帧实际qpos；四指swing在该轨迹首帧全部为0，不套用历史另一个姿态预览。固定基座/法兰/连接件为刚性变换，不另伪造关节。
- **验证/边界**：所有输入SHA匹配冻结manifest，源CSV均t=0，27项计数、度/弧度一致及导出CSV逐值回读通过。纯CPU提取，无轨迹/安装/训练修改或新回放；角度为各关节相对URDF零位的关节坐标，不是相对钢琴的欧拉角。同步坐标说明。

## 2026-10-06：t=0与t=1秒左手／钢琴／机械臂基座四视图

- **请求/参考**：接续用户t=1秒问题，要求多个方向同时标出手URDF base_link、钢琴和臂基座位置。读取`left_base_from_start_20261005`冻结场景、同一t=1（400Hz索引400）臂/手/琴键数据与manifest，参考此前CPU四视图工具、base_calibration及独立树FK；可选日志技能已确认缺失。
- **产物/修改**：`logs/hand_piano_arm_views_20261006/draw.py`生成`preview/outputs/hand_piano_arm_views_20261006/hand_piano_arm_base_four_views_t1.png`、四张独立PNG、metadata与manifest。保留完整左机械臂和手，标注H/P/B真实原点、坐标轴和钢琴系毫米坐标；同步坐标系说明。H=[170.262735,−158.388894,100.000600]mm，B=[661.739502,−22.842523,347.557576]mm，P为钢琴模型原点。
- **首帧补充**：用户随后要求t=0；`draw_t0.py`使用同一冻结录制索引0和相同四视角生成`preview/outputs/hand_piano_arm_views_t0_20261006/hand_piano_arm_base_four_views_t0.png`及独立图/metadata/manifest。H=[140,−150,125]mm，B/P保持。相同SHA、时间、独立FK、固定基座、原点可见、qpos不变与PNG检查通过，拼图实际查看；保留t=1产物。
- **验证/边界**：来源SHA/时间对齐、独立URDF树FK与渲染矩阵一致、B矩阵与初始化标定一致、所有原点在四视图内、qpos未改、PNG解码通过，拼图实际查看可读；文档diff检查通过。CPU OSMesa llvmpipe且CUDA不可见，无物理积分、策略推理、新视频或安装/训练修改；图示运动学位姿不等于硬件标定，轴和原点标记透过外壳显示。

## 2026-10-06：当前轨迹左手base_link原点多视角标注

- **请求/参考**：用户要求从多个方向标出左手URDF base_link位置。使用`left_base_from_start_20261005`冻结scene_manifest及t=0臂/手/琴键NPZ、原manifest SHA；参考本日手—法兰/钢琴图CPU工具及独立URDF树FK。可选日志技能不存在。
- **产物/原因**：`logs/hand_base_views_20261006/draw.py`生成`preview/outputs/hand_base_views_20261006/`四视图拼图及独立top/front/side/oblique PNG、metadata/manifest。青色圆心与引线定位H，透过外壳显示；H轴红绿蓝，钢琴方向小图只表示方向。仅隐藏上游机械臂以便看清连接位置，不改变Link7/连接件/手/关节/安装；同步坐标说明文档。
- **验证/边界**：来源SHA、独立FK与渲染H矩阵一致、H=[140,−150,125]mm、静态qpos不变、原点落在各视图可见范围及PNG解码检查通过。实际查看拼图并调整镜头范围/标注位置。OSMesa/llvmpipe软件渲染、CUDA不可见，无物理积分、策略推理或新视频；未修改训练与历史轨迹。位置是模型坐标，不宣称硬件标定。

## 2026-10-06 17:25：632最佳模型原配置续训683

- **请求/参考**：用户要求632继续训练。参考632实际config、best19500、完整F1检查记录，远端统一根入口、固定运动限制归档及既有585 warm-start预检/验证流程。可选日志技能已知缺失。
- **执行/配置**：通过H100现有`all_use_this_train.sh`末尾追加冻结632 best19500、原完整奖励JSON与独立输出路径，Slurm683于17:23:11提交，双H100物理0/2；未新增训练入口/修改算法、默认奖励或历史结果。恢复网络、优化器、温度与RNG，经验池/物理场景重建，追加10,000,384样本。legacy、闲指整项−1、目标超限整项−5、同手−.2、跨手−.5、hard_v1与Pitch3.1416rad/s及15.708rad/s²全部保持；启动完整打印奖励。
- **来源/产物**：父checkpoint SHA`628192eec4aa42c4d8759dd977628570dd191f997a5d2970bebf40e0f775fd74`，best19500原三回合最差F1=.7778293135435993；固定源码82199601…bda022。实际config SHA`5fa26a5ed312d662c10e7ed7af902aceda261588c1bf99b41e7891e4fdaa6b16`。审计辅助、请求、原始奖励、实际config和验证记录位于`logs/hpc/continue_632_20261006/`。README/H100说明同步。
- **验证**：实际入口CPU预检全部有效配置与632一致；启动核对源码、场景、任务、奖励、控制、指法、准备、动作、观察合同不变，未启用奖励/控制变更豁免。双GPU基本检查通过，initial updates等于父2,472,576，恢复来源和非重置优化器标志正确。17:25:26本轮320向量步、163840样本、新17408次更新，总2489984，学习数值有限；日志未见Traceback/ECC/CUDA/OOM标记，语法检查及git diff --check通过。
- **效果/边界**：初始化三回合最差F1=.7661176470588236，合并值=.7728264256777813；这是恢复后的重新评估，不宣称与父保存评估逐值相同。续训效果待后续检查，目标差分限制不代表真实qvel/qacc保证；未新导出视频或操作硬件。

## 2026-10-06：单独导出632的F1曲线

- **请求/参考**：用户要求632单组F1曲线；使用已核验的完整训练记录`logs/hpc/training_status_20261006_1715/status.json`中的41次评估，来源为632实际metrics.jsonl及best.json，未重跑策略。
- **产物**：`docs/figures/experiment_632_f1_20261006/`保存PNG、PDF、CSV及source.json；横轴向量训练步，每步512样本；纵轴为200Hz原始MIDI音乐区间micro F1的三回合最差值，标注best19500=77.782931%与final19532=77.450980%。CSV另保留三回合合并值并明确字段，不混用；来源JSON保存checkpoint SHA与完整评估。
- **验证/边界**：确认19532步、10,000,384样本、配置校验和全部三回合评估完整；图已检查可读，图表与来源数值一致。纯CPU绘图，无GPU作业、训练改动或视频导出；结果为训练内评估，未执行独立复评。可选日志技能缺失，按项目约定记录。

## 2026-10-06 17时：三组目标运动限制训练完成核查

- **请求/参考**：用户检查当前训练效果。读取H100 Slurm632/635/638的sacct、原始metrics.jsonl、best.json、summary.json、config、启动验证及错误日志；参考582父模型和本轮实验配置说明。
- **结果**：三组均COMPLETED/0:0、10,000,384样本、2,476,672次更新，预算耗尽正常结束；配置SHA保持，全部学习数值有限，未检出Traceback/ECC/CUDA错误标记。统一200Hz原谱音乐micro F1三回合最差，best分别.7778293135/.9322978893/.8776707164；final分别.7745098039/.9249201278/.8689596961。三组best19500；best/final全覆盖评估的同手/跨手/键底接触子步均0，末批训练仍有小量同手负罚，不推广为所有轨迹安全。
- **产物/修改**：CPU读取辅助及状态JSON存`logs/hpc/training_status_20261006_1715/`；仅用原有评估数据生成`docs/figures/training_status_20261006_1715/{f1.png,f1_data.csv}`，同步README与H100结果说明；未修改训练/源码/历史结果、未GPU推理或启动续训。
- **验证/边界**：状态、更新数、完整评估、best score与逐回合最差交叉核对；PNG检查可读、曲线统一口径且明确旧582控制不同。638由初始最差6.43%到best87.77%，此前6.99%为合并值；仍低于635，末段整体上升但不据此保证继续训练收益。可选日志技能缺失，依项目要求记本日志；独立多种子复评和实际qvel/qacc安全未在本次执行。

## 2026-10-06：当前视频首帧手原点—法兰中心俯视关系图

- **要求／参考**：用户要求从俯视方向导出当前手原点与法兰中心关系。读取 `left_base_from_start_20261005` 冻结场景、臂／手／琴键首帧NPZ及manifest；法兰采用此前SDK核对的Link7局部[0,−95,0]mm原点／原生轴向，参考 `docs/arm_coordinate_frames_20260930.md`、`online_control.models.forward_kinematics` 与原CPU渲染工具。可选日志skill不存在。
- **交付**：`logs/hand_flange_top_20261006/export.py`以CPU OSMesa生成 `preview/outputs/hand_flange_top_20261006/hand_origin_to_flange_top.png`、无标注PNG、T_P_H／T_P_F CSV、relation.json与manifest。正交沿钢琴−Z俯视，屏幕右为＋Y、下为＋X；标出H／F真实投影点和连接尺寸，保留手、连接件及Link7，仅隐藏上游臂几何便于查看。
- **数值／验证**：H=[140.000001,−150.000000,124.999971]mm，F=[163.690649,−154.000000,107.766190]mm；H→F=[23.690648,−4.000000,−17.233781]mm，空间距离29.567719mm、XY投影24.025961mm。输入SHA、独立URDF树FK对渲染FK、首帧qpos不变、F/H同向、正交方向与像素长度／XY距离一致检查通过。图已查看；未积分仿真、改安装／轨迹、修改训练源码或导出新视频。


## 2026-10-06下午：582最佳模型新增目标运动惩罚微调续训638

- **用户要求/确认**：用户选中582组合奖励99.39%视频，明确这轮为续训；确认沿用双侧Pitch目标速度3.1416rad/s、加速度15.708rad/s²硬限制。保留582 paper、无和弦、闲指整项−.1、同手weight=.2及子步奖励，奖励仅新增目标速度/加速度超限项（10/200与3/15，超限10%整项每步封顶−5）。既有632／635保持运行。
- **实际参考**：AGENTS、582实际config与best.json、已验收582视频checkpoint SHA、H100运动限制部署根`all_use_this_train.sh`及82199601…bda022固定源码、归档`rl_torch/{train,checkpoint,sac}.py`、632配置及启动核验、585续训流程。代码规定更改奖励/控制须网络微调，已向用户说明加载策略/价值网络并重置优化器、温度、RNG、经验池；不放松模型/任务/动作/观察合同。可选日志技能缺失。
- **位置/执行**：新增`configs/experiments/continue_582_motion_20261006/{rewards.json,README.md}`。使用远端已有统一根脚本追加`--finetune-from --allow-reward-change --allow-control-change`、本轮JSON、双硬限制和独立输出；未新增训练入口或修改算法/全局默认/历史结果。审计辅助文件集中在`logs/hpc/continue_582_motion_20261006/`。Slurm638于14:47:32提交，物理H100卡4/7、512环境、seed42、追加10,000,384样本。
- **父模型**：582 best13500，原训练精确F1=.9938697318007663，checkpoint SHA`16a64b0dfd683a45b7840513c18c5ecf6cccb9379bdab7cf982a7252d20034c7`，父config SHA`eda59aa11dbbfedb018177c26a3bd6a604531a1e37a28a6091c1eead31ded095`，远端冻结副本复核一致。原模型1704576次更新；本轮更新计数从0开始，不冒充恢复优化器状态。
- **验证**：实际入口CPU预检35个其他有效段与582一致、仅新增一项奖励及两侧速度/加速度四字段；CPU调用固定源码加载器，策略/双价值/目标价值全部张量逐项与父checkpoint完全一致，优化器空状态、温度.5、updates0。实际Slurm配置与预检一致，场景/任务SHA和动作/观察/指法/准备合同保持；原始加载检查通过，双GPU探测通过、完整奖励已打印。实际config SHA`8741a409bc0dc2829dea94c6d2868aac14ae153a14f7fd445851cc77cf1acdfd`。14:49:58已340向量步、174080样本、19968次更新，训练数值有限。文档及README同步，git diff --check通过。
- **初始表现/边界**：新硬限制下step0原谱音乐micro F1=.06991274372508888，明显低于原控制配置的99.39%，已告知用户；网络逐项加载一致，新配置需要适应。初始评估完整保存，训练后效果待验收；奖励和控制均有变化，不能称严格单变量奖励消融。目标离散差分限制不是实际qvel/qacc保证，无硬件操作或新视频导出。

## 2026-10-06下午：新增582奖励基础的目标运动惩罚对照635

- **用户要求/实验定义**：用户引用582“两者组合99.39%”要求再开速度/加速度惩罚实验。沿用582完整奖励，仅新增目标运动超限项；相对632恢复paper按键、闲指整项上限−.1/步。沿用最新随机初始化和双侧Pitch硬限制均开启的要求，速度3.1416rad/s、加速度15.708rad/s²，hard_v1位置限制保持。开始工作时已说明此定义，不加载582 checkpoint。
- **实际参考**：当前AGENTS、582实际config、632实际config与既有CPU预检/启动验证脚本、H100已有根`all_use_this_train.sh`及82199601…bda022固定运动限制归档。可选日志技能路径缺失。新增`configs/experiments/paper_idle01_motion_20261006/{rewards.json,README.md}`，审计证据在`logs/hpc/paper_idle01_motion_20261006/`；README和H100执行说明同步。未修改算法、统一默认、历史结果或运行中632。
- **执行**：现有根入口末尾覆盖本轮reward JSON、Pitch速度/加速度及独立输出目录，14:39:11提交双H100 Slurm635；512环境、seed42、19,532向量步/10,000,384样本。目标运动项阈值手指10/200、腕旋转3/15（rad/s及rad/s²），超限10%整项封顶−5/控制步，其他582奖励权重/开关/公式/子步采样保持。没有新增训练入口，启动完整打印奖励。
- **验证**：CPU捕获实际统一入口参数，确认35个其他有效配置段与582一致，奖励仅新增目标运动项；与632有效配置相比，仅按键模式和闲指上限两字段不同。基础双GPU检查通过，实际config与预检、归档SHA、源配方、随机初始化、双侧最终目标运动合同均核对；实际config SHA `d419e51b4290a33cff7140538a8c3d7f2bdf15a909987e6cafe9123ddff6dbcf`。北京时间14:41:09启动核查：Slurm635为RUNNING，物理H100卡0,2；200向量步、102,400样本、2,048次学习更新，训练数值有限。20条已记录奖励均值在各自范围。无算法修改，复用上一轮CPU/GPU实现验证，未重复相同测试；git diff --check通过。
- **TODO/比较边界**：等待后续训练F1与碰撞结果。历史582使用迁移起点、4rad/s且加速度硬限制关闭，因此与历史99.39%的差异不能仅归因于惩罚；632与635均随机起点、同种子/硬限制/预算，仍同时改变按键与闲指两项奖励。目标差分硬限制不代表实际qvel/qacc保证。

## 2026-10-06下午：582基础legacy／闲指−1／目标运动惩罚从头训练632

- **用户最终要求**：沿用582其余参数，按键改legacy、闲指整项最多−1/控制步，开启目标速度／加速度超限惩罚；明确随机初始化。后续明确双侧腕部Pitch目标速度3.1416rad/s，速度与加速度硬限制都开启，最终加速度15.708rad/s²。
- **实际参考**：`AGENTS.md`、`logs/hpc/h100_ablation_restore_20261006/582_config.json`、H100运动限制部署根`all_use_this_train.sh`及其`source_motion_limits_v1.tar.gz`、Slurm585展开源码的奖励/控制合同和现有测试、原585启动验证脚本、`docs/hpc_training_options_20260930.md`。可选日志技能路径缺失。工作区无Git未合并条目或冲突标记；现有其他任务修改保留。
- **修改/消除配置冲突**：新增`configs/experiments/legacy_idle1_motion_20261006/{rewards.json,README.md}`，仅改582的三项奖励。使用H100已有根统一脚本末尾显式覆盖奖励和双侧Pitch速度/加速度，避免582旧4rad/s/加速度关闭及本地后来新默认奖励相互覆盖。未新增训练脚本、改算法、修改历史配置或替换全局新奖励默认。保留582同手weight=.2、旧闲指姿态公式、20/200Hz子步采样、原曲及指法。最终完整奖励在CPU预检和训练启动均打印。
- **归档/实际执行**：运动限制归档SHA`821996010ffd4727558bdc4090f657677c5233cf39d686cbfed3f77f9bbda022`，根入口SHA`b35f15c65965e4b3c2217bd90c183ef039bbcfc8483659fca032ff78daae1ca0`。Slurm632于14:31:52提交，物理H100卡1/5，随机初始化seed42，512环境、双卡流水、10,000,384样本；没有恢复或放松checkpoint合同。完整config SHA`c935816c404a4dc700331d8e15ace6e8e0163044693d9c0dfad4956d0a7b7664`。记录集中于`logs/hpc/legacy_idle1_motion_20261006/`。
- **验证**：归档CPU测试31通过、1跳过、125子测试通过；额外闲指公式检查全手封顶−1。提交前实际入口捕获配置确认35个有效段与582一致，仅三个奖励差异及双侧速度/加速度四字段；运行保存配置与预检一致，源码配方仅源码散列变化。实际GPU探测通过，最终target_motion_contract确认两侧3.1416/15.708、位置hard_v1，随机起点和奖励打印通过。14:34:03已280向量步、143,360样本、12,288次更新，训练数值有限。README与H100说明同步。
- **TODO/边界**：新训练效果/F1待后续评估；最终目标离散差分限制不代表实际qvel/qacc保证。复用已在584验收的运动限制实现；本任务无GPU渲染、旧模型重放或硬件操作。

## 2026-10-06：独立A3.2触觉与MIDI velocity benchmark环境

- **问题/授权**：用户在前述“触觉输入＋MIDI velocity目标/评分，先仿真再真机”讨论后要求新建目录并搭建benchmark环境。本轮完成可运行环境与评测链路，没有启动训练或硬件。根目录已有其他任务未提交修改，按文件局部增量保留。
- **实际参考**：当前AGENTS、`all_use_this_train.sh`、其固定`logs/reward_defaults_20261006/source_reward_defaults_v1.tar.gz`；`suite/__init__.py`、`suite/tasks/{base,piano_with_tsensei_hands}.py`、`models/hands/{tsensei_hand,tsensei_hand_constants}.py`、`models/piano/{piano,piano_mjcf,midi_module}.py`、`mjlab_env/{bundle,source_runtime,joint_limits}.py`、`music/{midi_file,__init__}.py`、`rl_torch/note_timing_metrics.py`、`preview_audio.py`、`scripts/hpc/{submit.sh,job.sbatch,package_a32.py}`、`tests/test_environment.py`、现有依赖锁和README。可选日志技能路径缺失，遵循本日志约定。
- **位置/隔离**：新`tactile_benchmark/`包含`src/tactile_piano_benchmark/`、六个观测/扰动配置、测试、安装脚本、依赖锁、README和vendor源码副本。复制固定归档SHA `1bd541f1d54bd7a33a7afb6ccf0a9ebf98611eebe9274b07afc639d87cc523e9`（约15MB），仅从它校验/提取A3.2源码，拒绝混用已导入的其他来源及被修改的提取件。独立`.venv`安装Python3.12.14/88项依赖及本包，无Torch/GPU依赖；没有修改现有训练环境、归档、默认参数或checkpoint。`.gitignore`仅增加新目录白名单；根README同步入口。
- **物理与观测**：复用归档CPU场景构建及TZ/Yaw编译前结构移除，134坐标/46实体驱动器，Yaw烘焙.15rad，shadow几何/驱动、重力补偿及八MCP目标[0,1.2]。独立benchmark使用46维归一化直接目标、原生MuJoCo软关节限位，不假称继承统一训练31维耦合、hard_v1状态投影、目标运动限制、准备或奖励。20Hz控制/200Hz物理；十个完整末节胶囊全部接触的求解法向/切向力真实采集，不按接触对象过滤。策略三组同形观测，none零、binary接触占比/是否接触、force周期均值/峰值；关节状态/历史动作/目标事件共同输入，无全键行程、接触对象ID或上步reward泄漏。真实力和键程分别保存求解起点/积分终点时标。
- **事件与评分**：真实归一化键程双阈值测速，未标定固定速度→1..127映射，触发/释放滞回防止持续持键刷分；修复触发前回撤仍使用旧速度的问题。原始JSON/MIDI/proto时值/velocity及已有指法规则保留，合成校准谱单独生成，不猜填指法；踏板、打击乐、同音高重叠明确拒绝。逐音符音高/起音最大匹配后最小起音误差，力度不影响配对，报告事件F1/联合F1/起音和力度MAE/漏多音与接触指标；无匹配MAE=null。错键继续、末帧discount=1/截断后由调用者reset。CLI完整打印本benchmark奖励。
- **输出**：保存原始与实际事件JSON/MIDI、真实物理NPZ、策略实际观测NPZ、独立动作NPY、指标、完整配置、模型与benchmark源码散列及逐文件SHA。PNG由同次记录绘制共享音高/时间轴原谱与实际层及力度/力曲线；末尾仍按住的音符明确标记区间截断，原始开关事件另存。不是视频导出；没有请求或伪造视频/音轨。短smoke明确diagnostic/partial，不当完整排名。
- **验证**：`tactile_benchmark/.venv/bin/python -m pytest -q tactile_benchmark/tests` 94通过；涵盖真实MuJoCo接触、动作/结构锁止、reset/随机化、触觉噪声/延迟/丢包、不泄漏琴键状态、原始MIDI节拍变化与零力度NoteOff、指法proto、事件回撤、匹配/奖励和报告SHA。完整目录复制到临时独立位置后仍从自带归档编译/步进，未导入Torch/Warp，修改提取源码被拒绝。Gymnasium checker通过（无界Box建议与既有dm_control/pydub依赖警告保留，非失败）。三模式相同种子与诊断动作各265控制步/2651物理记录/13.25秒，所有物理数组逐项完全一致；none触觉0、binary≤1、force真实非零，各12项载荷SHA通过，PNG已查看。证据在`tactile_benchmark/validation/`及`outputs/acceptance_{none,binary,force}_v2/`。诊断动作不依赖谱面/触觉，F1=0且快速击键velocity饱和127，不能当作策略性能对照。追加0.05/0.2/0.5/1/2秒实际目标下压，音高56的velocity为127/111/48/26/11，证明固定代理可区分速度；无琴键状态注入，少数误触仍饱和，保留原映射。慢速证据和原始NPZ在validation/velocity_probe*；最后补齐API同音重叠拒绝并通过94项测试，已有回放保留各自源码散列。
- **TODO/边界**：TODO(V32-TACTILE-01)真实琴映射/速度范围/琴槌机构与高阈值悬停锁存限制，TODO(V32-TACTILE-02)真实指腹传感区域与带宽，TODO(V32-TACTILE-03)扰动范围实测；另有力度任务可达性、采样收敛、释放评分、多种子学习与GPU后端。右手镜像及URDF原始effort/velocity=0、工程碰撞/驱动标定边界保持。本轮不宣称触觉提高F1或真机力度已校准。

## 2026-10-06下午：训练完成核查与五组最佳视频

- **要求／参考**：用户要求检查当前训练并导出所有视频，按本轮580／581／582／585／620的best执行；读取Slurm账目、完整config／best／summary／metrics、原训练归档及已有585回放工具。五组均正常按预算结束，620 best=.9474100435、final=.9208245039；逐组配置SHA与启动记录一致。
- **导出／恢复**：585复用619同次产物；623／624／626分别录制580／581／620。582的625在物理5号卡无帧产出并长时间停滞，未见明确ECC报错；627获配0号卡通过基础探测后取消625，并用相同原始输入成功重录。未修改训练、物理合同或历史结果，停滞现场另存，不判定硬件修复。
- **产物／验证**：本轮五组有声MP4、各自同次200Hz原谱／实际／误差PNG与PDF、CSV／NPZ、原／实际MIDI、来源和视频单回合F1汇总至preview/outputs/all_training_best_videos_20261006，提供HTML与ZIP。核对41次训练评估、预算／best最差回合、配置／源码／PT、按键与音频事件、原谱目标和控制端点一致、H264/AAC全片解码／PNG及文件散列。证据与原录制数组保存logs/hpc/all_videos_20261006；README、H100记录和视频说明同步。视频单回合F1与训练3回合最差F1分列；训练已完成，未新开训练。

## 2026-10-06：从585最佳模型继续训练620

- **要求／参考**：用户承接585曲线要求继续训练；读取585完整config／best及启动记录、原根入口／归档、checkpoint和SAC恢复实现、训练预填规则。本地最新默认已另有奖励修改，本次以585匹配合同为准。
- **执行**：复用H100既有根all_use_this_train.sh末尾追加warm-start与独立输出目录，申请双卡Slurm620，再训练10,000,384样本。冻结best19500及config副本，恢复网络／优化器／温度／随机状态，继承2,472,576更新；replay／物理场景重新初始化。原hard_v1、Pitch目标运动限制、六项开启奖励、20/200Hz、gamma／学习率保持，未新增训练脚本／算法或变更历史产物。
- **验证／记录**：CPU通过真实入口假提交与归档parser核对有效配置，提交前及启动时打印完整奖励。实际config逐项通过父模型源码／场景／任务／控制／动作观察合同和恢复来源检查，物理0／2双卡探测通过。请求、入口实参、奖励、完整配置和启动更新证据保存logs/hpc/continue_585_20261006；README与H100记录同步。正在运行，最终F1收益待本轮实际评估。

## 2026-10-06：本地最新默认按键奖励恢复legacy

- **要求／参考**：用户明确要求本地最新默认key_press_reward恢复原公式、停用paper模式；读取统一入口、default_20261006.json、reward_config／rewards／task_logic、固定归档及现有默认奖励测试。可选日志skill缺失。
- **修改**：外部默认JSON仅将key_press_reward.params.mode从paper改为legacy，保持weight=1及和弦关闭。原公式为`.5×目标键行程容差得分均值+.5×无错键指示值`；Shadow端点评分、独立误触／同手／闲指奖励和硬关节／目标运动限制保持。同步入口注释、默认奖励说明与现有测试预期；固定归档已支持legacy，归档／SHA及历史配置不修改。
- **验证／边界**：真实统一入口以假sbatch捕获，不提交作业，再由固定归档parser解析；与原有效配置逐字段比较，仅上述mode改变。当前默认奖励CPU测试12项通过，证据logs/legacy_key_reward_20261006。本次仅本地默认调整，未启动训练、未更改远端或历史实验；本次组合的GPU效果尚未实测。

## 2026-10-06：585训练F1曲线导出

- **要求／参考**：承接585视频，用户要求导出F1曲线。读取已完整保存的远端状态／评估快照、best选模合同及逐回合精确F1，核对checkpoint.py中合并micro F1与最差回合的定义。
- **产物／口径**：preview/outputs/experiment_585_f1_20261006含完整及后期放大PNG、PDF、41行CSV、来源JSON和SHA清单；主线为200Hz原谱音乐区间三回合最低值，辅线为逐回合算术平均、阴影为范围，不平滑，标出best19500及final19532。CSV单独保留日志合并计数F1，避免把它称为算术平均；不混入视频回放F1。
- **验证／边界**：初始／500步间隔／最终记录完整、每次三个完整回合、有限值、最差值、best步数／分数及最终样本预算核对通过；PNG已打开检查标签和标记。仅CPU处理已有记录，无新回放／GPU／训练，脚本位于logs/hpc/plot_585_f1_20261006/plot.py，H100文档已同步。

## 2026-10-06：585最佳模型视频导出

- **要求／参考**：用户要求导出585视频；读取AGENTS、视频导出说明、585完整配置／best记录／源码归档和既有单卡导出、原MIDI对比与验收工具。选择best19500，保留原hard_v1和目标运动限制。
- **执行／产物**：H100单卡Slurm619通过统一submit入口完成同次物理录制和有声正面视频，COMPLETED 0:0；CPU复用1830个200Hz真实键程采样生成原谱／实际／差异图、原始MIDI、CSV／NPZ、PDF、HTML和ZIP，交付preview/outputs/experiment_585_best_20261006。本视频精确音乐micro F1=.8435576310762046，明确区别训练三回合评分。
- **验证／修改**：源码／配置／PT匹配、音频事件／子步目标／控制端点逐项一致；H264/AAC全片解码、PNG、21项manifest、ZIP及下载后散列通过，视频帧与图人工查看。任务后处理脚本修正默认fps为None时使用媒体实测帧率；未改通用源码、训练参数或旧归档。原录制及验证保存logs/hpc/export_585_20261006，同步视频说明。无本次交付缺项，未启动新训练。

## 2026-10-06：08:46训练状态与实际配置核查

- **要求／参考**：用户要求检查当前训练及配置；读取H100 Slurm队列／账目、580–582及585的实际config／summary／best／metrics／checkpoint散列和错误日志末尾，对照根统一脚本、新默认奖励JSON、归档SHA及原启动快照。
- **结果**：四组均COMPLETED 0:0，各10,000,384样本／2,476,672更新、预算耗尽正常结束；完整配置SHA逐一匹配启动记录，best／final均已保存。H100队列空、8卡idle。保存统一200Hz原谱音乐区间3回合最差F1、末5次均值和接触诊断；区分三组父模型续训与585从头训练，区分旧子步奖励与尚未开训的新端点默认。
- **修改／验证**：证据保存logs/hpc/training_status_20261006_0842，更新README和H100执行记录。CPU核对预算、best分数与逐回合最小值、四份config散列通过；本次未改算法／脚本／参数或历史产物，没有启动或取消作业。新默认奖励的GPU集成与训练效果仍待验收，本次未复测GPU健康或宣称A100 ECC修复。

## 2026-10-06：同手／误触／闲指默认奖励调整与Shadow端点采样

- **要求与参考**：用户要求同手碰撞每对−.2改−.75并默认开启、误触权重−.05（追问确认同时开启）、闲指使用weight=1的−.04×下发目标与初始目标L1偏差；删除collision_reward及natural_posture_reward且无需兼容删除项，公共采样同步Shadow，确认PIP/DIP开关。读取统一脚本、当前motion_limits归档、奖励／task／配置／指标／CLI源码、相关CPU测试、包内Shadow参考合同及已审计端点说明；未运行时引用外部历史工程。可选日志skill缺失。
- **实现**：新default_20261006.json启用三项。同手周期内去重body接触对每对−.75；误触负权重乘非负行程代价，保留错键行程公式，半程单键−.004、全程−.0075。闲指新idle_target_l1_v1以当前乐谱用指为掩码、最终40物理手指目标（含耦合从关节）对实际reset目标求L1，排除腕／平移／踏板，不用实际qpos、未来手指豁免、平均或封顶。两删除项从模板／校验／计算／任务接线／指标移除，旧含删除项JSON明确拒绝；旧归档与配置不改。
- **采样／入口**：control_endpoint、substep=false、target_key_substep=false、activation_weight=0、scale_by_dt=false；提前／分阶段奖励关闭，精确原谱诊断与.15秒观察保留，当前paper按键公式与全手碰撞范围保持。跨手由周期并集变为端点计罚，同手计数是独立周期并集项。PIP/DIP仍由--finger-coupling开启，--no-finger-coupling关闭，完整观察开关不等于耦合开关。保留读取时脚本已存在的运动罚项full_excess=.1、腕硬阈值、学习预算、双卡流水。新归档仅从82199601…bda022覆盖本任务文件，路径logs/reward_defaults_20261006/source_reward_defaults_v1.tar.gz，SHA同步根入口。
- **验证／边界**：最终归档隔离CPU公式、21项模板、删除项拒绝、负权重符号、实际目标/初始参考、PIP/DIP计分、当前闲指掩码、dt、端点对瞬态历史独立、旧正则和腕目标限制回归85项／149子测试通过；2项外置入口案例在工作区真实脚本链单独通过，fake sbatch/python确保不提交。643个归档文件中628项逐字继承父归档，15项任务文件变更／新增；校验最终根入口→归档脚本→归档parser的完整奖励一致，validation.json、effective_config_cpu.json与归档SHA证据保存在同目录。README、AGENTS及专题说明同步。本次未调用GPU或提交训练，580–582、585等其他聊天运行中作业保持；父运动限制的H100验收不能代替本次新奖励集成／效果验收。

## 2026-10-06：按当前统一入口新增H100目标运动限制实验

- **要求／参考**：用户要求按当前训练脚本再开一组。核对当前all_use_this_train.sh、固定82199601…bda022源码归档、原reward JSON、新运动项／硬目标控制代码和既有CPU验收；保留580–582既有训练。当前入口无迁移参数，新增实验从头训练，512环境、双卡流水、20/200Hz、gamma .8、三LR .0003、seed42与10,000,384样本均保持。
- **实际参数差异**：本轮读取的根入口SHA b35f15c6…ae1ca0，已将full_penalty_excess覆盖为.1；历史CPU报告和文档为.2。按用户“当前脚本”要求执行.1，即最大相对超限10%扣满−5。独立capture_current.py截获实际sbatch和归档训练入口参数，用归档parser做CPU解析，完整奖励保存在logs/hpc/h100_motion_experiment_20261006/printed_rewards.txt，未重写历史验收。
- **部署／验收**：隔离H100目录`/home/houj/projects/tsensei_motion_experiment_20261006`，原入口／奖励／归档与提交工具逐项SHA固定。新增Slurm测试使用真实模型、最终物理ctrl重建腕目标一／二阶差分、独立NumPy奖励计算、46关节位置限位、普通／CUDA图与局部reset。583被导入路径守卫拒绝：外置测试使pytest加载部署工作区而非作业归档；未提交训练。将测试和解析配置纳入单独验收归档，运行时代码逐字继承训练归档，保留严格路径断言。
- **GPU验收／提交**：584 COMPLETED 0:0，57项／111子测试通过，3项归档外置入口案例跳过（本次实际入口另捕获解析）。普通／CUDA图每种160步×2环境，腕目标最大速度2.21576512rad/s、最大加速度15.708rad/s²，46关节硬位置检查通过；独立NumPy奖励误差≤5.96e−6、罚项[−5,0]、局部reset隔离通过。校验运行时代码逐字一致后，再次打印完整奖励，原样执行当前统一入口提交585，双卡／128G／24小时，从头开始新预算。启动读回确认物理6／7卡、hard_v1与目标运动合同、当前奖励、20/200Hz、31/1407接口、seed42及从头训练生效。00:49快照480步／245,760样本／37,888更新，训练数值有限、新罚项实际计入（初期均值−5）；随后660步／337,920样本／60,928更新，完成500步首次训练后评估，更新持续增长。完整配置SHA9a1cf863…f1b96e与启动证据已保存；不将早期罚值或F1视为最终收益。不把目标限制当作真实qvel/qacc保证。

## 2026-10-06：下发目标超限奖励与腕部硬速度／加速度限制

- **要求／明确选择**：新增目标速度／加速度超限惩罚，手指阈值10rad/s、200rad/s²，最大相对超限达到20%时每步最多−5。用户确认无第二项；明确接受虚拟腕工程阈值，随后指定腕硬限制3.1416rad/s／15.708rad/s²、惩罚阈值另设3rad/s／15rad/s²。不是为TJ_M6关节猜填映射；URDF只有3.1416rad/s速度，SDK快照J3–J7有900deg/s²，15.708由用户明确选定。
- **实际参考**：读取当前AGENTS、根统一入口及固定hard_v1归档、reward_config/rewards/task/scene/control_limits、train/checkpoint/SwanLab指标、source_runtime/package_a32、指法／奖励／控制／硬限位测试，项目内TJ_M6左右URDF、configs/preview/m6_sdk_joint_limits.json、对应HPC和硬位置文档。指定日志技能路径缺失，照常维护本日志。
- **修改／原因**：新增默认关闭的joint_motion_limit_penalty、六个阈值／预算参数、独立CLI及通用启停兼容；统一入口显式启用。最终完整物理目标含耦合从关节，按control_dt计算一／二阶差分，分组阈值归一后共同取最大超限；线性饱和−5，最终控制步单位、不乘全局dt，weight≤1。reset使用实际初始ctrl，两帧目标历史归零速度，局部重置隔离；未改旧正则含义、策略动作频率或观察维度。SwanLab补新项名称，原始／加权日志和启动最终奖励打印沿用。
- **腕部硬目标控制**：显式max_acceleration通道增加最终位置／速度／加速度可行目标投影，位置边界提前减速；在coupling／guard／hard位置裁剪后应用，历史来自最终下发目标。只给当前可动Pitch配置，Yaw结构锁止保持，TX/TY原1m/s保持。新字段／resolved contract只在开启时出现，不回填历史控制；这是命令目标硬限制，不是对真实物理qvel/qacc的速度投影。
- **兼容／部署**：旧22项JSON补新项为关闭，历史文件不改；已有reward/control合同仍严格比较，新启用或改阈值不允许静默载旧checkpoint。以SHA a627c43becf48e3b9917c41a9e664a8e8caca7727032c028c001254ea711d93d的hard_v1归档为底座，只覆盖本任务文件到logs/joint_motion_limits_20261006/source_motion_limits_v1.tar.gz，脚本路径／SHA同步；不混入并行任务的机械臂、视频、Slurm改动。README、AGENTS、HPC说明同步。完整差异、归档和验收日志保存在同目录。
- **验证／TODO**：最终归档隔离解压CPU163项／249子测试通过；4项跳过为1项GPU和3项外置入口测试，后者在工作区真实脚本链另测3项全通过（fake sbatch/python，不提交作业）。4种dt×1600步×8环境×2旋转通道直接从最终FP32目标求差分，验证位置、速度和加速度（仅双精度算术1e-12容差），包含随机、换向、触边与局部reset。归档640文件，其中621文件逐字继承父归档，19项任务代码／测试／文档与原奖励JSON变更或新增；archive_audit.json、validation.json及解析后effective_config_cpu.json在同目录，最后一份是CPU配置核对而非新训练config。当前本地Slurm节点draining，未运行本任务GPU物理验收或启动训练；后续须在Slurm验证真实模型集成、实际关节速度／加速度、F1／碰撞／吞吐。原已在其他聊天推进的H100验收／训练不由本任务修改。当前frame_stack1未新增前两帧目标速度观察，不宣称加速度奖励无隐藏历史依赖。

## 2026-10-06：H100三组奖励对照，保留硬关节限位

- **用户要求／参考**：将ECC中断的三组实验迁至H100；随后明确保留当前硬关节限位。参考原1725–1727冻结request／奖励／父best、ECC恢复计划、当前统一入口与hard_v1源码、GPU限位及控制合同测试。三组仍为idle_only、self_only、idle_and_self；从1724 best19500仅迁移网络，显式allow-reward-change与allow-control-change，优化器／alpha／replay重新初始化；不称原动力学的精确断点恢复。
- **部署／不变量**：独立H100目录`/home/houj/projects/tsensei_ablation_restore_20261006`，执行当前`all_use_this_train.sh`经submit.sh申请每组2卡；父best SHA06c3abdb…6f3cd8、hard_v1归档SHAa627c43b…11d93d与三份原奖励SHA逐项核对。每组512环境、20/200Hz、gamma .8、三LR .0003、10,000,384样本，奖励完整打印后提交。部署记录`logs/hpc/h100_ablation_restore_20261006/`；原训练／检查点及导出目录保持。
- **验收发现／修正**：H100576双卡基础探测通过；577真实A3.2硬限位、CUDA图／callback、耦合目标及动作检查通过，但旧control_limits测试夹具缺少forearm_collision_geoms，35通过／1失败。仅修正`tests/test_preparation_policy.py`的合成夹具，明确已有命名球体代理（物理接触全部关闭），不是为训练猜填碰撞范围；初次空列表夹具在578暴露旧通用索引不支持空组，改为夹具实际代理名单。独立验收归档只覆盖这一测试文件，训练运行时代码与a627c43b归档逐字一致，不修改训练归档或放宽合同。
- **GPU验收／提交**：最终579 COMPLETED 0:0，36项／28子测试通过、零跳过；训练代码与原hard_v1归档逐字一致，仅验收夹具不同。逐组打印完整奖励后通过当前统一入口提交580（idle_only）、581（self_only）、582（idle_and_self），各申请双卡／128G。三组50项配置不变量逐项匹配，奖励、父best SHA及网络迁移、原动作／观察／准备合同保持，46个机器人关节的hard_v1合同生效，六个GPU UUID互不重复。00:16左右快照三组均640步／327,680样本／58,368学习更新，已完成500步首次训练后评估；初始化／预填后确实推进，不再停在0更新。证据startup_validation.json及各组完整config。训练持续运行，尚无收敛或最终优劣结论。

## 2026-10-05：机器人关节硬位置限位、策略审计和左手限位导出

- **要求／实际参考**：用户明确要求仿真和导出角度都遵守URDF／规定限位，检查策略是否输出非法负值。参考当前 `control_limits.py`、`scene.py`、`task.py`、`action_mapping.py`、`sac.py`、通用曲线导出、冻结source_v2_fingertips归档及1359同次原始录制／原动作合同；未从新动作范围推断旧记录。可选日志skill路径缺失。
- **修改**：新增 `joint_limits.py`／`joint_limits_warp.py`，显式hard_v1合同，机器人有限位标量关节每物理子步投影、向外速度清零、清受影响warmstart并刷新接触；有效范围为模型／动作／显式位置限制交集，float32界限向内舍入。GPU图、callback、reset和最终目标共用限位；琴键不投影、移除腕轴不重建。通过已有control_limits配置及checkpoint合同传递，不向历史配置插默认字段。通用视频关节CSV写前校验位置，17位有效数字写后读回；超限记录须另作派生指令。训练启动打印最终奖励配置和状态限位模式。
- **审计／交付**：原365控制帧的31策略通道和46物理ctrl通道均0越界；四指PIP/DIP由有界耦合器控制。新 `preview/outputs/left_bounded_joint_trajectory_20261005/` 包含20指关节200/400Hz rad/deg CSV、旧臂轨迹原样副本、比较PNG、逐通道报告／来源SHA。仅食指PIP38点、DIP465点修正到0，最大改动0.325439°／0.549968°；合法负swing保留，所有CSV读回限位检查通过。旧轨迹／视频／音频未覆盖。
- **验证／限制**：最终归档隔离解压CPU测试41通过、3项GPU测试跳过、40个子测试通过；入口参数／散列与Bash检查通过。新归档 `logs/hard_joint_limits_20261005/source_hard_limits_v1.tar.gz` SHA256 `a627c43becf48e3b9917c41a9e664a8e8caca7727032c028c001254ea711d93d` 已由统一入口固定，仅在上一归档上覆盖本任务文件。GPU真实模型图／普通／callback与耦合验收通过submit.sh提交1746（旧候选1745已取消），但唯一节点因既有作业1725清理失败draining，待运行，不声称GPU通过。状态投影改变动力学；限位CSV是派生指令，不继承原视频击键或F1验证。手指速度／加速度限制尚无已确认数值。未启动训练或更改已有训练作业。实现／验收详见 `docs/hard_joint_limits_20261005.md`，打包、审计证据在 `logs/hard_joint_limits_20261005/`。


## 2026-10-05：H100验收与关闭和弦最新final单卡视频导出

- **要求／参考**：用户要求导出当前关闭和弦的最新视频，随后询问CPU或单卡方案。核对1724实际final／config／summary、完整源码归档、原视频导出CLI／worker／Slurm入口及CPU测试、此前同次MIDI对比生成器。最新终点为19532步final，不是best19500；三组异常续训没有新结果。匹配归档scene明确要求CUDA，现有final无物理录制；CPU软件渲染不能补出不存在的轨迹。
- **实现／提交**：export_videos.py增加--gpus 1|2（原默认2保持）并保存计划／透传submit；submit.sh与job.sbatch允许export单卡，探测要求同步为1。只改作业资源入口，固定训练算法／归档与合同不变。原1742双卡PENDING取消，同一冻结PT／config改为1743单卡导出；1744是afterok:1743的无GPU CPU绘图／验收，失败依赖自动取消。非训练脚本位于logs/hpc/paper_chord_off_final_export_20261005，无新训练入口。
- **计划／证据**：preview/outputs/paper_chord_off_final_20261005保存PT副本SHA 8984d4be…c1db5b9、config／plan／submission及完整交付状态；正面无臂1280×720／20fps、原速有声及手曲线。从此前已验收生成器制作本轮compare.py，严格绑定本PT／1724源码／关闭和弦配置，待同次1830个200Hz物理子步生成原MIDI对比PNG/PDF、事件CSV／数组NPZ、HTML／ZIP；finish.py还核验H264/AAC全片解码与SHA／ZIP。没有替换为旧和弦开启录制或伪填单视频F1。
- **验证／边界**：tests/test_hpc_allocation_entry.py与test_video_export.py合计69通过、38子测试通过；涵盖单卡export许可／探测、CLI→计划→提交及默认双卡和旧功能回归。Shell语法、后处理AST、PT/config/归档散列及CLI dry-run通过；CPU读取既有视频验证已有imageio-ffmpeg可获取H264/AAC元数据，无须安装系统ffprobe。Slurm读回1743请求gres/gpu=1，1744仅4CPU／8G、依赖正确。同步video_export_cli；不改其他既有未提交修改。
- **H100实测／迁移完成**：经用户授权连接101.47.9.106，Slurm566分配物理0／2卡，Torch矩阵、CUDA graph、双向P2P与Warp互操作探测通过（4秒，0:0）；普通SSH的NVML错误不能代替分配内结果，仅验收两卡而非全部8卡。独立目录`/home/houj/projects/tsensei_video_export_20261005_2334`部署当前导出工具，严格核对原final/config/ae7e29f3归档。568单卡完成1830个200Hz物理子步，但附加手关节曲线被位置限位校验拒绝；保留原轨迹，不裁剪关节或放宽校验，574单卡复用568原录制、`--plots none`完成视频（23秒，0:0）。H100运行后取消尚未启动的A1001743／1744。隔离复制的Composer环境添加绘图依赖，pip check通过，未修改系统或已有训练环境。
- **交付／验证**：`preview/outputs/paper_chord_off_final_h100_video_20261005`含有声MP4、同次原MIDI对比PNG/PDF及事件CSV/NPZ、HTML／ZIP。视频1280×720、20fps、10.2秒；含1秒准备和末帧停留，本次200Hz原谱音乐区间micro F1=.9896392939370683。167源码文件、PT/config、原谱与录制目标、音频事件与实际键程、控制帧与物理末帧一致性全部通过；H264/AAC全片解码、PNG、21项manifest和ZIP CRC/SHA通过。文件取回后21项SHA再次通过，ZIP本地按已核对文件重建并通过CRC/SHA；对比PNG及视频4.5秒解码帧已实际查看。证据及CPU脚本在`logs/hpc/h100_video_export_20261005/`。
- **边界／TODO**：A100 ECC／节点DRAIN未修复；H100基础探测和本次单卡回放不是双卡训练压力测试或全部卡健康证明。未启动新训练。此次不交付被限位检查拒绝的附加手关节曲线；CPU核对5个关节出现超限，最大为左小指swing的0.02310836rad（1.324012°），详见joint_overshoot.json。旧训练物理状态保持原样，不将按键F1视为限位／碰撞安全保证。原生MuJoCo CPU替代训练后端仍未实现或验证。

## 2026-10-05：首帧反解一次左臂基座，随后固定基座跟踪wrist

- **补充左手轨迹**：用户要求提供手的轨迹。`logs/left_base_from_start_20261005/export_hand.py`读取本回放冻结场景、臂／手NPZ和原wrist目标，新增输出子目录`hand_trajectory/`：20指关节rad／deg的200／400Hz CSV、钢琴系手`base_link`的FK位姿及原目标位姿200／400Hz CSV、完整变换NPZ、五指角度曲线和单位／来源报告。位姿列为时间、米制XYZ、xyzw四元数及连续展开的固定轴xyz RPY（度）；明确FK是模型重建而非传感器反馈。时间0–9.125秒，400Hz3651行、200Hz1826行，和臂／视频同步；原目标时间仅有1.78e−15秒表示舍入差。CSV读回、首帧位姿、基座恒定、旋转矩阵回构及400Hz FK误差0.095219mm／0.476270°通过；不替代上一条1600Hz密集验收。原视频、原CSV和状态SHA保持；只更新HTML索引／manifest并同步轨迹专题，未重新规划或运行物理。
- **最终要求／参考**：用户明确“只有第一次的基座坐标系可以动，从初始的结果反解之后的基座就应该固定，然后开始通过wrist的轨迹来IK解算”。读取原`export_best_constrained_20261001_151434/data/wrist_poses.npz`、指定七角、当前安装／URDF、SDK限值、`online_control.models`／`preview_arm_ik`、上一轮约束求解与左声部400Hz按键证据。用`T_P_B=T_P_H(0) inv(T_B_H(q0))`反算一次完整基座位姿，后续不逐帧移动基座。
- **修改**：`preview_arm_ik.py`及`online_control/models.py`支持可选按侧`base_xyz_m`，未指定侧保持原肩座中心／间距放置；当前`preview/assets/tj_m6/preview_mount.json`只更新左基座位置和朝向，右侧、七角seed、手／法兰／连接件不变。新基座在钢琴系为[661.739502,−22.842523,347.557576]mm，xyz固定轴RPY[−89.694095,−7.201126,178.209422]°。首帧七角精确等于用户值，手根位置／朝向等于原t=0准备首帧。旧配置备份、求解／导出／渲染／验证脚本保存在`logs/left_base_from_start_20261005/`。
- **规划／验收**：取消额外2秒衔接；原9.125秒腕轨迹、1秒准备段和20指关节保持。原网格候选在1600Hz审计中误差0.101360mm被拒绝，采用0.09mm内部拟合裕量重加权后独立选择第6候选；正式0.1mm／0.5°验收未放宽。14601点最大误差0.09770066mm／0.47667567°，RMS位置误差0.00683939mm；连续速度／加速度凸包最高为SDK71.65723%／98.00533%，最小角限余量11.156874°。端点速度／加速度为零，使用显式导数样条避免直接二阶求值的1.16e−10浮点抵消残差。两种基座加载器的旧默认／新覆盖与独立URDF树FK差9.99e−16，渲染FK差2.00e−15；逐帧基座恒定、首帧手位姿、右侧不变和工具SHA通过。
- **交付**：`preview/outputs/left_base_from_start_20261005/`含200Hz／1826行与400Hz／3651行q/qd/qdd CSV、度制角度、原20指关节、预览JSON、连续样条、基座正逆变换CSV／JSON、曲线和HTML。视频保留左臂／连接件／左手、相机面向演奏者，1600×1000、40FPS、405帧／10.125秒（含1秒末帧展示）；配左手WAV/MIDI及实际按键／原MIDI PNG。PT/config/录制/原tar内全部合同SHA、CSV读回、H264/AAC全片解码、图／音频事件一致、PNG和manifest通过；AAC/WAV相关系数.9999474。首帧、最终视频3.25秒解码帧及对比图实际查看。同步坐标、机械臂轨迹和视频导出专题。
- **边界／TODO**：本次CPU IK及llvmpipe渲染，模型基座仅初始化调整一次；没有GPU或硬件动作。声音和琴键沿用同次原物理录制15次左手真实激活，仅移除新增2秒偏移，原踏板与原谱时值保持；原左声部micro F1=.966120566不冒充新IK的接触评价。实物标定、完整碰撞／耦合干涉、力矩和新轨迹接触动力学未验证。训练入口、归档／SHA、运行作业及历史产物不改。

## 2026-10-05：三组续训ECC故障排查与恢复准备

- **要求／参考**：用户要求修复首次遇到的GPU不可纠正ECC错误。核对1724–1727的Slurm状态、实际config／metrics／best、probe与GPU UUID、1726错误栈、1725遥测、进程等待状态、统一入口与原提交request；查阅NVIDIA A100 ECC机制及节点故障恢复指南。可选日志技能路径缺失，按本项目规则记录。
- **定位**：1726获配物理2／3卡并报CUDA ECC；不能仅凭异步调用栈确定具体故障卡。1727获配同一对卡后GPU探测卡死；1725使用5／6卡，预填183步／93696样本／0更新后无进展，进程D状态等待uvm_gpu_release。只读nvidia-smi返回NVML Unknown Error，节点因其他用户1738无法清理而已DRAIN。无免密sudo／内核日志权限，Xid和根因未确认，不把奖励开关当硬件故障原因。
- **执行／修改**：logs/hpc/ecc_recovery_20261005保存现场、错误、探测与遥测末段，备份父best／config／summary并核对best SHA 06c3abdb…6f3cd8。按本轮修复授权scancel本用户1725／1727；请求成功，18:33最终调度状态CANCELLED；两个孤儿进程仍残留且有SIGKILL待处理，节点DRAIN原因更新为1725无法清理，不能宣称驱动资源释放成功。新增docs/gpu_ecc_recovery_20261005.md及README状态说明；restart_plan.json保存通过原统一入口的新目录恢复命令，未另建训练脚本或提交新训练。
- **验证／TODO**：父PT、统一入口、ae7e29f3源码归档及三份冻结奖励SHA与原提交一致；CPU核验JSON可读及文档diff。未进行普通会话CUDA计算、改训练算法／归档、改系统驱动、关闭ECC、清空计数、重置设备、重启节点或操作其他用户作业。管理员须获取Xid并协调恢复GPU／节点，恢复后经submit.sh做GPU验收，再打印完整奖励并重新提交原三组。硬件恢复、残留清理和训练推进尚未完成。

## 2026-10-05：固定当前坐标安装，以指定左臂七角生成轨迹并回放

- **后续视角／声部调整**：用户要求左侧视频与击键音乐、相机面对演奏者，随后明确保留机械臂。新增`logs/left_only_facing_performer_20261005/{common,prepare,render,validate}.py`，输出`preview/outputs/left_only_facing_performer_20261005/left_arm_hand_front.mp4`、`left_keystrokes.wav/.mid`及同目录实际按键／原MIDI对比图、事件／状态／来源、HTML和验收报告。完整保留左臂、刚性连接件及左手，右侧无可视模型；相机在钢琴−X侧朝＋X演奏者方向，MuJoCo azimuth=0°、elevation=−38°，正交固定视角。新视频1600×1000、40FPS、485帧／12.125秒；上一版臂／手NPZ逐字节保持、安装SHA保持，不重新规划。
- **左声部证据／验收**：原400Hz、3650个物理子步的保存状态在原冻结`kinematic_scene/scene.xml`中仅做CPU前向接触重建，不推进物理。31次原激活均有唯一手接触归属，左右触键音高集合不相交；保留15次左手击键、排除16次右手击键。音符起止仍由原键程及阈值决定，接触只用于归属；包括52/56等未达触发阈值的左手接触键程，不将接触本身变成音符。音频直接使用通过SHA检查的原归档preview_audio，保留原踏板、48kHz立体声SoundFont，音量仍为原velocity=127，未声称力度标定。原谱以已确认part/fingering 5..9取左手，保留原起止；图、视频高亮和声音使用同一15事件及2秒新增偏移。原录制左声部音乐段micro F1=.966120566（TP3037/FP35/FN178），仅是原录制指标。H264＋AAC全片解码、WAV时长／首音前静音／非零音量、AAC与WAV相关系数.9999858、事件逐一匹配、PNG及manifest SHA通过；实际查看正面首帧、中帧、最终视频解码5.25秒帧及对比图。核验发现1359解压工作副本14项与原合同不同，改为直接检查1103原源码tar内全部合同文件，检查通过后取原音频脚本；没有放松兼容检查或修改历史。CPU llvmpipe渲染，完整机器人逐帧网格边界至少110px余量；当前安装及轨迹不变。同步`docs/video_export_cli.md`与机械臂轨迹专题；新机械臂接触动力学及实机仍未验证。
- **要求／范围**：用户再次指定左J1–J7为[77.04558,−70.57884,−61.54541,−82.70541,157.40033,5.54958,−25.69049]°，要求生成此前20261001轨迹并导出回放，明确当前坐标系不要再变动。指定姿态到原手根起点差131.568637mm／7.421777°；解释后采用2秒平滑到原起点，再接9.125秒原演奏。当前左右基座、原生法兰／手／连接件安装、右侧和历史产物均不改；只生成左侧新轨迹。
- **实际参考**：当前preview_mount／hand_connector JSON、M6左URDF与SDK默认位置／速度／加速度参数、native_flange_hand及left_mount_alignment_fix冻结姿态、online_control.models／trajectory、preview_arm_ik／preview_trajectory、constrained_arm_motion／replay_arm_csv，原export_best_constrained_20261001_151434的手CSV／腕NPZ／manifest／config／plan，以及1359同次physics_trace、audio_substeps、played.wav、rollout与匹配归档的原始MIDI／library／midi_file／rl.environment。可选日志技能缺失，按AGENTS直接维护本日志和专题文档。
- **规划／修改位置**：`logs/left_arm_replan_20261005/{plan,solver,path_utils,export,render,validate}.py`及冻结输入／验证日志。原差分种子触及SDK J5=170°，默认误差框QP超时；保留失败证据，任务局部改用J5=160°连续IK种子和逐样本误差重加权，最终J5不锁定，未改全局默认优化器。位置范围取URDF／SDK交集；三次B样条连续速度／加速度上界、0.1mm／0.5°验收和原节奏均不放宽。2秒五次关节过渡严格从用户七角出发，起止qd/qdd=0，与演奏样条C2相接。
- **结果／验收**：演奏段1600Hz、14601点FK最大位置误差0.093475mm、最大姿态误差0.471432°、RMS位置误差0.008137mm；连续速度／加速度上界最高为SDK66.0574%／98.0155%，最小角限余量7.21018°。独立URDF树对渲染FK差1.89e−15，刚性安装矩阵差3.89e−16；首帧与用户七角及上一张手坐标图一致。原输入／当前安装SHA保持；两个CSV读回、解析导数、五次过渡极值／接点、图片解码、H264＋AAC全片解码、manifest SHA及ZIP CRC全部通过。CPU OSMesa llvmpipe，无GPU／Slurm新任务或硬件连接。
- **交付**：`preview/outputs/left_arm_replanned_20261005/`含12.125秒／40FPS／485帧有声双视图（左臂总览＋手近景），11.125秒轨迹200Hz／2226行、400Hz／4451行，均含q/qd/qdd；附度制角度、左手20指关节、控制台预览JSON、样条与状态NPZ、曲线、首中末PNG、HTML、来源与报告、8.2MB ZIP。原1秒准备段保留，新音乐起点为3秒，视频末帧停留1秒。首帧、衔接末帧、演奏中帧和时序图均实际查看。专题`docs/arm_motion_constraints_20261001.md`同步。
- **按键证据／限制**：视频机械臂为新FK轨迹，手指／琴键／声音复用同次原400Hz物理录制；未新跑策略／接触物理。原PT、源码归档、config和录制SHA校验通过，MIDI资产及相关解析代码与config一致，原谱14.692708秒前导静音裁剪后365个量化目标全部匹配，但绘图保留原音符起止而非量化目标。实际状态从录制键程和原0.00872665rad阈值重建、控制帧逐项匹配原actual，图／视频共享新增2秒偏移；保存31实际事件和34原谱事件及数组。原物理回放音乐段400Hz／原谱micro F1=0.942818（TP4765/FP113/FN465），不冒充新臂轨迹的物理表现。画面只显示左侧机器人，琴键／音频保留原双手；当前实物零位／安装、腕部耦合干涉、完整碰撞和动力学／负载验收仍未完成。

## 2026-10-05：查证SDK原生法兰坐标系并与当前左手对比

- **问题／实际参考**：用户指出法兰有自身坐标系并要求找到后与手对比。读取项目内SDK的`CommonConfig/ccs_m6_40.MvKDCfg`、`kinematicsSDK/FxRobot.cpp`的LOADMvCfg／CCS初始化／FK、`FXMatrix.cpp:Tmat`、`FXMath.cpp`、`SDK_PYTHON/fx_kine.py`与`libKine.so`，以及当前左URDF、安装配置和`left_mount_alignment_fix_20261005`冻结姿态。此前图中F只平移Link7并继承其轴向，是临时显示系；本次查到机型1017配置第9行原生法兰MDH为[90°,0mm,95mm,90°]。
- **结果／交付**：在URDF Link7上原生F的原点为[0,−95,0]mm，轴向X_F=Z_J7、Y_F=−X_J7、Z_F=−Y_J7；原生＋Z指向工具。当前完整手／连接件安装使H三轴与F同向，但F→H在F系的平移为[−13.5,−4,+26]mm，距离29.567719mm。新增`logs/native_flange_hand_20261005/{audit,render}.py`，在`preview/outputs/native_flange_hand_20261005/`输出真实模型近景坐标PNG、干净图、两个变换CSV、来源SHA和SDK／URDF完整审计JSON、116姿态FK数组及manifest。图中F虚线、H实线，真实原点未挪动。同步`docs/arm_coordinate_frames_20260930.md`，明确旧F_legacy与SDK原生F的区别。
- **验证／边界**：只加载纯运动学库，不加载控制SDK；零位、当前角、14个单轴扰动、100组随机角的独立MDH／SDK／精确角URDF交叉核对，矩阵误差最大4.19e−10。最初1e−12断言因SDK数值残差失败，实测记录后按1e−9验证；保留原URDF的1.5708／3.1416时，当前位姿残差0.004386mm／0.000955°，原URDF与直接SDK结果分别保留，未改源模型以消除差异。CPU OSMesa llvmpipe绘图，原qpos逐位不变、当前配置／原冻结来源SHA一致、变换合成／CSV／PNG／交付SHA检查通过，主图已查看。机械臂基座／七角、手和连接件安装、右侧、历史产物及训练均未修改；SDK软件定义已查证，实物零位、安装孔分度、控制器自定义TCP仍待现场核对。

## 2026-10-05：修正左手与连接件拆开旋转造成的装配错位

- **用户要求／原因**：用户指出安装错位并明确“手和连接件是一体的，然后再安装到机械臂”。前一操作把手单独绕自身Z转90°、连接件留在法兰，保持了手原点却破坏手座四孔配合。审计显示安装面中心错开19.9123mm，四孔最佳配对最大误差34.4093mm。读取当前安装JSON、原CAD装配及手座STL/URDF、install_tianji_mount.py／prepare_connector_mount.py／preview_connector.py、当前冻结姿态；已向用户承认并修正该处理。
- **实际修改**：以原完整装配为基准，在固定法兰中心绕Link7的−Y轴整体转90°，T_new=T_F R_−Y(90°) inv(T_F) T_original；hand_connector配置nominal_link7_to_hand.left更新为该完整组件的安装，offsets.left归零以维持手／六件连接件内部配合。preview_mount左tool同步。新J7→H平移[4,−121,−13.5]mm，F→H为[4,−26,−13.5]mm，掌面朝向与前一步一致。两配置备份与fix_and_validate.py位于logs/left_mount_alignment_fix_20261005；左右臂基座／关节、右手和全部网格／URDF均未修改。
- **位移／验证**：机械臂七关节及渲染全部qpos逐位保持，左右臂全部连杆／右手矩阵差0；法兰中心和面法向不变。手原点因恢复正确装配由[34.780893,−205.855831,184.042807]改为[34.468683,−188.370606,193.564858]mm，总位移19.9123mm；这不是机械臂运动。四孔世界中心配对误差0（浮点模型），CAD孔位／配合平面实际几何测试tests/test_step_mount_revision.py三项通过；原生MuJoCo对独立FK最大误差1.33e−15。
- **交付**：preview/outputs/left_mount_alignment_fix_20261005包含正确组件的总览、手近景／俯视、前后对照、更新J7关系图、T_J7_H与T_F_H CSV、配置／源状态／装配证据、HTML和ZIP；主图已实际查看。PNG、SHA、ZIP及来源验证通过；详细证据alignment_validation.json和validation.json。同步docs/arm_coordinate_frames_20260930.md，以此节作为最新安装合同，旧错位图仅留历史记录。
- **TODO**：CAD手—连接件四孔与法兰中心／平面已复核，实物法兰螺孔分度、紧固件与完整运动碰撞仍待标定；无硬件动作、IK、训练或完整轨迹重新规划。

## 2026-10-05：左手H与J7随动坐标系关系导出

- 用户要求导出手和J7坐标关系；复用最新left_hand_local_z90_20261005冻结姿态、当前安装SHA及此前F/H绘图逻辑。J7明确为URDF Joint7_L转动后的子连杆Link7_L坐标系，当前q7=−25.69049°；采用实际关节枢轴原点，不使用法兰面中心。
- logs/left_j7_hand_frames_20261005/render.py经CPU OSMesa导出preview/outputs/left_j7_hand_frames_20261005/left_j7_and_hand_frames.png、clean PNG、frames.json及T_J7_H.csv。主图J7虚线、H实线、红X绿Y蓝Z，标出J7系的J7→H偏移[−13.5,−121,−4]mm、间距121.82mm和轴映射X_H=Z_J7、Y_H=−X_J7、Z_H=−Y_J7。
- 验证J7原点等于MuJoCo实际关节anchor、Z轴等于实际正转axis，URDF树FK与原生Link7矩阵一致；图中原点位置未挪动、旧qpos逐位一致。原来源和交付SHA、两PNG解码、CSV/JSON矩阵一致及世界变换合成通过；六轴标注已实际查看，证据validation.json。同步docs/arm_coordinate_frames_20260930.md。无配置、模型、关节角或硬件修改。

## 2026-10-05：当前左法兰F与手基座H的坐标系近景

- 用户要求查看当前二者坐标系。读取最新left_hand_local_z90_20261005冻结场景／状态及当前安装SHA，复用CPU场景构建，F按Link7局部[0,−95,0]mm并继承Link7轴向，H取真实base_link。
- logs/left_flange_hand_frames_20261005/render.py使用CPU OSMesa导出preview/outputs/left_flange_hand_frames_20261005/left_flange_and_hand_frames.png，并附clean PNG、frames.json及manifest。F轴用虚线，H轴用实线；红X绿Y蓝Z，箭头放在各自真实原点，金线连接两原点。侧栏列出轴向映射与F系偏移[−13.5,−26,−4]mm、原点间距29.57mm。
- 当前配置与冻结来源SHA一致；原qpos逐位不变，FH旋转／偏移计算复核通过，原点投影在画面内，PNG解码／交付SHA通过，实际查看六轴与文字清晰。未修改安装、关节角或模型；法兰定义仍为项目约定。证据validation.json，相关文档docs/arm_coordinate_frames_20260930.md同步。

## 2026-10-05：固定机械臂、左手绕自身Z轴+90°朝下

- **要求／参考**：用户要求当前机械臂完全不动，仅手绕Z轴90°使掌面向下。以紧前arm_base_remount_20261005姿态为基准，读取既有手／法兰变换、preview_connector.py的逆偏移合同、online_control.models及此前静态渲染脚本。按左手自身Z轴、经过原手base_link原点旋转，采用T_after=T_before Rz(+π/2)。
- **实际修改**：configs/preview/hand_connector_20260930.json的offsets.left.rpy_deg从[0,0,0]改为[0,0,−90]；由于合同中使用inv(offset)，实际左手为+90°。同步preview/assets/tj_m6/preview_mount.json中左link7_to_hand的等价变换，保证不使用连接件模块的预览也一致。两配置原件备份在logs/left_hand_local_z90_20261005；未调用IK或ConnectorPreview.apply，不改变臂关节。
- **交付／验收**：CPU OSMesa输出preview/outputs/left_hand_local_z90_20261005的臂总览、手近景／俯视、同视角前后对照、角度CSV、状态NPZ、配置／坐标／验证来源JSON、HTML和ZIP。相对旋转为局部[0,0,+90]°，手基座仍[34.780893,−205.855831,184.042807]mm，手+X近似钢琴−Z。新旧整份渲染qpos逐位相同，左臂各连杆矩阵无变化，6件连接件世界变换差小于1e-12，右臂／右手不变；回退tool与连接件tool一致。PNG已查看，散列及ZIP CRC通过；详细证据validation.json。同步docs/arm_coordinate_frames_20260930.md。
- **边界／TODO**：这是用户指定的可视安装修正及静态运动学姿态；机械臂、法兰、手基座原点均固定。未移动到键面、未重规划完整轨迹、未进行新动力学或实机标定；历史交付保持原记录。

## 2026-10-05：按用户轴向关系修改左臂基座并预览指定初始角

- **要求／参考**：用户截图规定基座＋X→钢琴−X、＋Y→−Z、＋Z→−Y，随后确认仅左臂修改、右臂安装保留；初始角取用户给出的[77.04558,−70.57884,−61.54541,−82.70541,157.40033,5.54958,−25.69049]°。读取截图、当前preview_mount、左右URDF、online_control.models、连接件配置、原20261001手指轨迹首帧、已有CPU场景构建与渲染脚本。
- **实际修改**：preview/assets/tj_m6/preview_mount.json仅修改base_rpy_rad.left为[−π/2,0,π]、seed_q_rad.left为上述角度的弧度值；基座位置仍[520,−46.5,480]mm。原配置备份logs/arm_base_remount_20261005/preview_mount_before.json，附change_request.json记录来源与前后SHA。右臂、关节零位／符号、法兰连接变换和历史导出均保持。旧preview_pose来源检查本来已失效，本次保留检查规则；验证场景实际读取新seed。
- **交付／验证**：同目录render.py经CPU OSMesa llvmpipe生成preview/outputs/arm_base_remount_20261005的左臂总览、手近景、俯视、并排PNG、初始角CSV、完整变换／来源JSON、状态NPZ、场景清单、HTML与ZIP。手指按指定hand_joints CSV首帧，无此前swing+15°预览修改。新手基座P坐标为[34.780893,−205.855831,184.042807]mm；全部臂／手连杆独立FK对MuJoCo最大误差低于1e-12。当前配置仅两处left字段变更、七角匹配、PNG解码、SHA、ZIP CRC通过，三主视图已查看，证据validation.json。同步docs/arm_coordinate_frames_20260930.md。
- **边界／TODO**：图中手指朝琴键−X伸出，手掌仍侧立；本次仅落实用户基座轴向，未自行补法兰90°或关节180°偏置。安装仍待实机核对；静态运动学预览无动力学推进、避碰验收、硬件动作或新训练。此工作尚未重新规划完整轨迹。

## 2026-10-05：钢琴与左右臂基坐标系关系图

- 用户要求一张展示三坐标系轴向相对关系的图。读取preview/assets/tj_m6/preview_mount.json、online_control/models.py、preview_arm_ik.py及指定20261001轨迹冻结arm_config.json；核对当前与冻结基座安装一致，钢琴为世界原点。
- CPU绘图脚本logs/arm_base_coordinates_20261005/draw_frames.py导出preview/outputs/arm_base_coordinates_20261005/piano_and_arm_base_frames.png。左／右基座在P系为[520,−46.5,480]／[520,+46.5,480]mm，旋转分别Rx(+90°)／Rx(−90°)；红绿蓝分别表示各自局部XYZ，附轴映射。coordinates.json记录位置来源；此前二维位置草图也保留。
- PNG已实际查看，轴映射由配置旋转矩阵计算，与代码安装定义一致；未修改模型、安装或轨迹。安装仍待实物标定。同步docs/arm_motion_constraints_20261001.md图链接。

## 2026-10-05：左手四指 swing 向左的静态姿态预览

- 用户要求基于此前俯视投影将四指 swing 朝左。参考1720录制initial状态、匹配归档XML、此前frames.json和export_top.py；核对四关节正方向均使指尖向钢琴−Y移动，限位±0.35rad。本预览统一取+15°，未将该角度视为用户确认的训练默认值。
- logs/idle_only_first_frame_swing_left_20261005/export_swing_left.py使用CPU OSMesa，只将食指／中指／无名指／小指swing由0改为0.261799387799rad。新图、角度CSV、编辑状态NPZ、来源JSON、HTML和ZIP位于preview/outputs/idle_only_best_left_swing_left_20261005；图上明确标为编辑姿态。
- 验证仅4个qpos变化，手基座／拇指不变，四指指尖Y分别减少23.695、25.945、23.517、20.571mm；两主图已查看，PNG、manifest及ZIP CRC通过。证据见同日志目录validation.json。
- 此为静态预览，未改原录制／训练／IK轨迹；不构成触键或碰撞验收。相关说明同步docs/video_export_cli.md。

## 2026-10-05：指定左手第一帧的正交俯视与琴面投影

- **要求／参考**：用户附上best19000左手第一帧坐标图，要求俯视并直接观察手在琴面上的投影。读取该帧frames.json、1720同次完整物理录制／归档XML、原第一帧绘图脚本及MuJoCo分割接口。输入checkpoint／物理轨迹／XML散列与前次交付一致，使用同一t=0 initial状态。
- **处理／交付**：logs/idle_only_first_frame_top_20261005/export_top.py在CPU OSMesa llvmpipe将相机设为钢琴＋Z向下正交视角，屏幕右为＋Y、下为＋X；手根及手指／琴键状态不变。隐藏右手、关闭光照阴影与反射；根据仅左手可见几何的分割mask，叠加青色40%轮廓到原琴键图，标出五个原模型指尖投影。preview/outputs/idle_only_best_left_first_frame_top_20261005含1600×1200标注俯视图、无标注图、半透明投影图、mask四PNG，以及来源/坐标JSON、HTML、manifest和ZIP。
- **验证／说明**：手根矩阵逐值匹配此前第一帧；同XY点沿Z投到白键面Z=.0225m前后像素误差0，mask172754像素且未裁剪，五指site齐全。两张交付主图实际查看、四PNG解码、原来源SHA、6文件交付SHA、ZIP内容/CRC通过，证据logs/idle_only_first_frame_top_20261005/validation.json；docs/video_export_cli.md同步。
- **边界／TODO**：几何正投影不是光照阴影或实际接触，黑键保留真实外观，琴键区域之外的手掌投影保留。无新策略推理、GPU、物理推进、IK、姿态／模型／安装／训练修改；原始斜视、原录制及先前IK结果不变。此项静态视图交付不构成新的演奏视频或F1评估。

## 2026-10-05：以关闭和弦1724为父基线排队三组奖励对照

- **要求／参考**：用户要求以本轮基线增加与之前相同的对比训练。读取1724实际config/输出及进度、此前三组request和完整JSON、固定归档reward_config、根入口与submit/job Slurm依赖实现。当前父训练未结束，采用afterok等待最终best，与此前“基线完成后同一best网络续训”的实验方式一致。
- **修改／原因**：新增configs/experiments/paper_chord_off_reward_ablation_20261005/{idle_only,self_only,idle_and_self}.json，从1724实际奖励仅开启对应姿态/同手项；与此前三组逐项比较仅和弦false/null不同。姿态weight1、总最大罚.1、sigma.25；同手weight.2，其余基础奖励、训练数值/预算、物理和源码不变。all_use_this_train.sh新增前置--dependency透传已有submit能力，训练参数仍统一维护，无新增训练入口，无算法/归档变更。README与20Hz说明同步。
- **提交／打印**：每组提交前打印全部奖励weight/params及performance，保存只读JSON快照。14:52经统一入口提交1725 idle_only、1726 self_only、1727 idle_and_self，各双卡流水、19532×512新样本，--finetune-from指向1724输出best.pt，--allow-reward-change明确授权奖励差异。Slurm读回三组均PENDING、Dependency=afterok:1724；父成功完成后才自动启动。父尚未完成，不保存中间PT冒充最终父模型，不伪填最终步数/SHA；启动时原checkpoint合同检查与origin记录保持。
- **验证／证据**：归档解析器CPU验证三组JSON、真实根脚本假sbatch捕获及与此前逐组实参比较通过（仅父路径/奖励路径/输出路径变化），全部关闭和弦。默认无依赖命令保持，缺失依赖参数拒绝，bash语法及diff检查通过。logs/hpc/paper_chord_off_reward_ablation_20261005保存request/父config/入口快照/奖励打印/参数捕获/回执/Slurm读回/entry与submission_acceptance。普通会话未执行GPU。
- **TODO／边界**：三组尚待父完成，不把排队或CPU预检称实际训练启动通过；后续核对实际config/父散列及学习指标，再按同频率原谱精确F1和碰撞比较。续训与上一轮相同为网络加载，重建optimizer/alpha/replay/RNG/physics，不称完整状态续训；父失败时依赖不会放行。没有额外第四组。

## 2026-10-05：续训前1620基线关闭和弦并重新训练1724

- **要求／参考**：用户要求找到当前续训前基线脚本和参数，关闭key_pressed_reward的chord_bonus/chord_scope后重开训练；实际配置名称为key_press_reward。读取AGENTS、根唯一入口、1620实际command/完整config、后三组续训父配置、原paper_chord_on奖励JSON及匹配归档reward_config/rewards解析；指定日志技能不存在。确认1681/1682/1683父基线为1620 best15500，本次采用其原从头训练条件。
- **修改／原因**：新增configs/rewards/paper_chord_off_20261005.json，严格从1620实际reward_config仅将key_press_reward.params.chord_bonus=true→false、chord_scope="on"→null；paper模式、activation=.5、其他全部字段保持。根all_use_this_train.sh切换该JSON绝对路径及新输出标签，未另建训练入口。false_press/闲指姿态/同手碰撞保持关闭，既有五项奖励weight均1、跨手接触−.5/控制周期保持。同步README及20Hz实验说明，旧配置、模型、checkpoint与源码归档不改。
- **执行／验证**：固定ae7e29f3归档解析器CPU校验、真实根入口假sbatch捕获、基线实际实参比较及shell语法通过；正式提交前打印完整奖励weight/params/performance并保存printed_rewards.txt，冻结只读submitted_rewards.json。14:47:40通过原入口提交Slurm1724，双卡RUNNING；512×19532=10,000,384样本从头预算，gamma=.8、三LR3e-4、seed42、20/200Hz、原曲/指法保持。GPU只在Slurm内执行。
- **实际启动／证据**：实际完整config与1620仅reward_config及CLI奖励/输出路径不同，所有其他顶层字段（含源码/物理/控制/动作/观察合同及SAC）逐项一致，确认无checkpoint加载。logs/hpc/paper_chord_off_20261005保存基线脚本/命令/config快照、字段差异、提交回执、完整奖励打印及startup_acceptance；输出training_mjlab/outputs/hpc/paper_chord_off_fullhand_20261005-20261005-144747-246686。14:50前确认已280向量步／143360样本／12288次学习更新，全部已写指标有限。
- **TODO**：本轮完整预算后的原谱精确F1、接触与吞吐待实测，不能从配置变更或早期指标宣称改善；此任务未额外启动其他对照。

## 2026-10-05：补齐三组续训best视频并汇总基线对照

- **要求／参考**：用户要求将当前视频全部导出，按当前上下文补齐idle_only、self_only、idle_and_self三组best，并附此前父基线。读取1682／1683实际best/config、现有1678／1721导出和manifest、当前video_export_cli、统一入口及同次MIDI对比生成器；仅处理本轮视频交付，不扩展历史实验或训练。
- **执行／交付**：经export_video.sh→submit.sh新增1722 self_only与1723 idle_and_self，两者best18500，79／80秒COMPLETED 0:0。PT SHA分别309534d773bb52a79e7f27f56d8a6be08bad56b1998eaf57dcaee69f11aa13f7、969772efa1845a5465497e8223283d772d9a8ae06ebefeb57a069ac53de4a818；匹配同一ae7e29f3指尖归档及各组冻结config，回放实际完整奖励逐项匹配。每段正面无臂、原速有声、原谱指法、1280×720／20fps／204帧／10.2秒，并附手关节曲线、同次原MIDI对比PNG/PDF/CSV/NPZ/MIDI、HTML和ZIP。原基线与idle视频只读复用，重新核验文件SHA。
- **同视频结果／验证**：两次各1830个200Hz真实键程重建音频事件、实际控制末帧及原注册精确目标逐项匹配。self视频TP/FP/FN=2589/11/26，原谱音乐区间F1=.9929050815；idle＋self=2591/14/24，F1=.9927203065。两次各34原音符均有激活覆盖、35次实际起音，不把覆盖统计称标准note-F1。源码/PT/config/奖励/原MIDI来源核验通过；两片全片H264解码及AAC 48kHz双声道通过，音量均值−27.2／−27.0dB、峰值−11.0／−11.1dB。实际查看两片4.5秒画面及两张对比图，各219文件SHA和ZIP内容／CRC通过；logs/hpc/1722、1723保存验收。
- **汇总／边界**：新增preview/outputs/paper_chord_all_best_20261005四组可查看页面，含从头播放／全部暂停控制及独立单视频分数；ZIP包含四段视频、每组原按键对比图／数组／事件／来源，逐组复制与原导出SHA一致，全部HTML链接可解析，manifest和ZIP通过。脚本／验收保存logs/hpc/paper_chord_all_best_export_20261005，相关CLI说明同步。汇总单视频F1与先前训练三回合结果分开，不由导出分数重定训练排名；无训练/归档/模型/硬件变更，未宣称视频整体无碰撞。

## 2026-10-05：最佳姿态惩罚第一帧的左M6七关节IK

- **要求／参考**：用户要求利用刚导出的左手第一帧解机械臂IK并得到每关节角。读取同次best19000第一帧frames.json／左手关节CSV、1720物理录制及冻结config/PT来源、当前online scene manifest、左M6 URDF、preview_mount、hand_connector、既有ArmPreview求解实现。保持记录的完整手根目标[.140000000894,−.15,.124999970627]m及原朝向，当前安装配置不叠加此前J1／法兰候选。
- **求解／选择**：在CPU执行URDF矩阵FK＋SciPy bounded least_squares，6维位置／SO(3)误差权重.15m/rad，当前seed与固定随机种子共7次得到首个精确可达分支。首解J7余量仅.033°，再用SLSQP在6维位姿等式与原关节范围内增大最小归一化限位余量，6次迭代成功，最小绝对余量9.635949°。最终J1→J7为[.719112879812,.983340640778,−2.774078527222,−1.461788226503,2.701710571748,−.792145257048,−1.402620964547]rad；七自由度有多解，本次是一组局部可达解，不声称唯一／全局最优。
- **交付／验证**：preview/outputs/idle_only_best_left_first_frame_ik_20261005含关节CSV、完整IK JSON／初值和优化记录／来源矩阵、场景快照、1440×900机械臂带左手PNG、HTML、manifest与ZIP。使用第一帧20个手指关节角，手URDF SHA匹配训练归档；独立MuJoCo FK位置误差2.5172e-16m、旋转误差4.4067e-16rad，6位rad回代误差2.1913e-7m／7.8960e-7rad，图实际查看通过。7角在限位内、CSV/JSON、输入与源SHA、PNG、交付SHA和ZIP CRC通过；证据及三个脚本保留logs/idle_only_first_frame_ik_20261005，docs/video_export_cli.md同步。
- **边界／TODO**：全部CPU数值运算及OSMesa llvmpipe软件渲染，无新训练、GPU、动力学推进或实机操作；未改原模型／安装默认／训练归档。未检查避碰，数值误差不代表实机精度，SDK编码器映射及肩座／工具安装标定仍待确认；仅交付原URDF约定角度，不将其称已验证实机命令。

## 2026-10-05：仅姿态惩罚best19000左手第一帧与钢琴相对坐标

- **要求／参考**：用户要求当前仅姿态惩罚训练最佳轨迹第一帧及手／钢琴坐标关系，随后明确仅左手。读取1681完整config、best/summary、三组实验入口记录、视频CLI、严格归档回放／被动捕获脚本、原生坐标框架及物理录制重建实现。核实idle_posture_penalty_v1开启、self_collision_reward关闭，其他原基础奖励保留；最佳为19000，训练三回合最差原曲精确F1=.9942528735632183。
- **执行／来源**：冻结best.pt/config/best.json至logs/idle_only_first_frame_20261005，PT SHA cebd7ffb504294f85f43bbdd85677ce606a100fbc23393a77f25f5262969c9aa。submit.sh --gpus 1 replay提交1720，使用ae7e29f3固定匹配归档，167源码文件及检查点合同通过，34秒COMPLETED 0:0。原模型双手物理回放记录1830个200Hz子步，另有动作前initial状态；取t=0准备段抬升25mm的第一帧，未把准备结束或第一个按键帧冒充首帧。
- **交付／坐标**：preview/outputs/idle_only_best_left_first_frame_20261005含左手斜视／侧视原图与坐标图四张1280×900 PNG、frames.json、左手23关节CSV、HTML及ZIP。P=`piano/`实际根系、H=`left_/base_link`；P在世界为单位阵，H在P的位置约[140,−150,125]mm，精确旋转／4×4矩阵／逆变换见JSON。红绿蓝XYZ在真实原点上投影，虚线连P/H；P原点低于白键面22.5mm。离线仅隐藏右手可视几何，未修改录制状态，无M6或上轮SDK安装候选叠加。
- **验证／文档**：GPU仅在Slurm1720运行，首帧再绘用CPU llvmpipe。归档模型body/joint树加记录qpos/mocap的独立FK与MuJoCo位姿误差1.3877787807814457e-17，合成／逆变换、检查点SHA、四张PNG、左手CSV、7交付文件SHA及ZIP内容／CRC通过；实际检查斜视与侧视。诊断绘图脚本与export_validation.json保存在任务logs；docs/video_export_cli.md同步。
- **边界／TODO**：第一帧策略尚未动作，不能评价姿态惩罚效果；原最佳评分是训练选模分数而非静态图分数。本次仅交付静态图，内部原回放视频保留logs，无用户要求的视频交付，不伪造首帧MIDI比较。初始化与实际URDF手根是本次坐标定义，未声称为实机TCP标定；训练归档、模型、作业和默认安装均未改。

## 2026-10-05：仅姿态惩罚续训1681最佳模型视频导出

- **要求／参考**：用户在侧对话中明确要求导出当前仅姿态惩罚的视频。读取当前AGENTS、video_export_cli、export_video入口、1681完整config/best/summary、匹配v2指尖归档与1681展开源码、既有同次回放MIDI对比脚本；可选日志技能路径缺失。
- **执行／交付**：选择1681的best19000，PT SHA cebd7ffb504294f85f43bbdd85677ce606a100fbc23393a77f25f5262969c9aa，根export_video.sh经submit.sh提交Slurm1721，79秒COMPLETED 0:0。输出preview/outputs/paper_chord_idle_only_best_20261005：正面无臂有声MP4、原谱指法、实际手关节曲线、原MIDI与实际按键对比PNG/PDF/CSV/NPZ/MIDI、HTML和ZIP。1280×720、20fps、204帧／10.2秒，保留1秒准备、9.15秒物理轨迹及末帧停留；不改训练、模型、安装配置或其他作业。
- **同次数据／验证**：源归档SHA ae7e29f3778ba08ee7ec589a7568320cb24ef2b4e34688dc28ea76d898bc5fd3，与冻结config/PT及录制合同匹配；回放实际reward_config逐项一致，仅gesture项启用idle_posture_penalty_v1，self项关闭。CPU读取本次1830个200Hz真实键程及阈值，音频事件逐项重建一致，原注册曲精确事件与全部录制子步目标、控制末帧真实按键完全匹配；原MIDI SHA af583c13275d4fd1477f7c6d19b20714f470a40ccf84084b525fef027c14ed98。音乐[1,8.854166667)内TP/FP/FN=2590/11/25，单视频精确F1=.9930981595，34原音符均有实际覆盖，35次实际起音；未把覆盖计数称note匹配F1。
- **验收／说明**：全片H264解码通过、AAC 48kHz双声道，音频均值−27.2dB／峰值−11.1dB；人工检查4.5秒视频帧及对比图，219项交付SHA、ZIP内容／CRC通过，无STL/XML/MJB。证据logs/hpc/1721/video_export_validation.json；同步docs/video_export_cli.md。训练三回合分数单独标注于data/training_selection.json，不冒充视频分数；本次未额外复评或推断整体无碰撞。

## 2026-10-05：实机SDK左臂轨迹的J1及其余关节／法兰安装诊断

- **要求／参考**：用户认为手应平放在钢琴上，怀疑J1转180°并要求检查其余关节，随后确认CSV为实际机械臂SDK／编码器读数。读取当前online场景、左右零位导出、原CSV回放及报告、左M6 URDF、连接件配置、手部distal-tip测量与初始五指平面说明、琴键尺寸。此新反馈使此前直接映射的零位／方向确认成为待核实假设，保留原视频。
- **检查／结果**：CPU沿完整6201帧计算原映射、J1减π、J2–J7逐关节符号翻转及±90°／180°、法兰绕现有安装轴±90°／180°候选，保存logs/arm_csv_alignment_20261005/{analyze_mapping.py,metrics.json}。J1减π全程合法，将指尖平均Z从.745–.780 m降至.180–.215 m，但五指标记拟合平面仍约89.74°。J2–J7保持原值，手及连接件整体绕Link7临时法兰中心[0,−.095,0]、局部−Y轴＋90°后，倾角中位8.13°／最大8.37°，但平均指尖仍高于白键面94–129 mm；反向安装候选翻面。J7单独＋90°仍竖直且111样本越限，不能用来替代安装转角。其余逐项统计保存在完整报告，未将局部候选检查称穷举标定。
- **交付／验证**：preview/outputs/arm_csv_alignment_20261005包含0／15.5／31秒三列同世界视角与分别居中近景，共六张1800×670 PNG、JSON/场景/诊断脚本/HTML及ZIP。OSMesa llvmpipe CPU软件渲染，18次视图的实际MuJoCo矩阵对独立FK误差≤1.7763568394002505e-15，PNG解码、11个交付文件SHA、ZIP内容与CRC通过；开头整臂／近景实际查看。证据logs/arm_csv_alignment_20261005/export_validation.json，内部坐标说明同步。
- **边界／TODO**：倾角只描述既有五指末节标记平面，不是已标定掌面或软指腹接触。当前数据支持优先核对J1零位及法兰安装转角，J2–J7尚无确定的改动依据；剩余高度／X位置须核对肩座与钢琴及工具安装。单一关节轨迹与“平放”不能唯一识别SDK零位／方向和安装变换，仍需SDK工具定义或实机安装／已知法兰姿态。候选只用于本次诊断，未改原CSV、URDF、安装默认、训练归档或作业，未进行GPU／动力学推进／硬件操作。

## 2026-10-05：当前左右机械臂零位和正方向导出

- **要求／实际参考**：用户要求导出当前环境机械臂零位及正方向。读取AGENTS、online_control/models.py、左右本地M6 URDF、preview_mount.json、preview_pose.json、hand_connector_20260930.json、当前手URDF/初始关节常量、原坐标系绘图脚本/文档与刚完成的CSV回放场景构建器；指定日志技能路径不存在。按已有同事左臂CSV确认保留joint_0…6→J1…J7映射。
- **修改／原因**：新增preview/scripts/export_arm_zero_axes.py，显式将两臂14关节q设为0，保留当前固定手姿态及安装变换。导出每侧零位总览和七关节放大图，红／绿／蓝为局部＋X／＋Y／＋Z，橙色弧线为局部＋X向＋Y的正转；全部URDF轴[0,0,1]，正角按右手定则。另存原parent origin XYZ/RPY、局部轴、零位臂基座／世界轴、4×4矩阵、角限、Link7/手根安装、当前初始值、来源SHA及CSV映射。区分模型q=0与实机回零、个体Base_L/R与scene manifest世界base；零位世界轴近似左[-Y,-Z,-Y,-Z,-Y,-Z,+X]／右[+Y,+Z,+Y,+Z,+Y,+Z,+X]，精确文件不人为归零URDF微小分量。
- **交付／验证**：preview/outputs/arm_zero_axes_20261005含两张2880×1920 PNG及PDF、arm_zero_axes.json、14行joint_axes.csv、README/HTML、来源URDF/配置/代码快照、scene manifest、manifest及ZIP，不打包网格。14关节零姿态及逐关节＋1e-5 rad扰动的独立FK对MuJoCo最大矩阵误差8.881784197001252e-16；原生正轴误差6.661338147750939e-16，正扰动旋转向量误差3.3306690738754696e-16。Python编译、两张图片人工检查、CSV/JSON逐值一致、21个交付文件SHA、原始来源/副本/网格SHA、ZIP内容与CRC通过，证据logs/arm_zero_axes_20261005/export_validation.json。同步docs/arm_coordinate_frames_20260930.md，明确旧图属于历史reset＋IK姿态，不能当作本次全零姿态。
- **边界／TODO**：Matplotlib Agg CPU绘图、MuJoCo GL关闭，未调用旧绘图main或求IK，未推进动力学、GPU、训练、硬件控制。现有preview_pose来源散列已变化，场景构建器依原规则回退seed=0，实际初始值及警告已保留，不修改加载规则。肩座／法兰标定TODO与右手镜像限制保留；模型零位不是实机编码器回零，交付sources不含网格，不称独立仿真迁移包。模型、安装配置、训练归档、checkpoint和历史导出未修改。

## 2026-10-05：同事200Hz左臂CSV带当前左手的视频

- **要求／实际参考**：用户上传left_arm_200Hz.csv并明确确认marvin_0_arm_joint_0…6依次对应预览Marvin M6左臂J1…J7，零位及正方向一致。实际读取该CSV、online_control/models.py、preview/assets/tj_m6/left/arm.urdf及preview_mount.json、当前A3.2左手URDF/initial_positions、hand_connector_20260930.json、Piano、arm_motion_limits及video_export_cli；指定日志技能路径不存在。
- **修改／原因**：新增preview/scripts/replay_arm_csv.py，严格检查SI列名、200Hz时钟、有限数值及原URDF角限，逐样本应用原q/qd，不插值、不裁剪、不重定时；保留原qdd并绘制角度／速度／加速度PNG。使用项目本地左臂、左手及现有安装件，20个左手活动关节固定为当前场景原始姿态，随法兰运动，未猜填手指轨迹。整臂全局视角与完整左手跟随近景同步拼接，取景依据全部6201个源姿态的几何边界。同步docs/video_export_cli.md。
- **执行／产物**：当前环境已有用户目录OSMesa/llvmpipe软件渲染，强制CUDA不可见，实测Mesa llvmpipe；未执行GPU渲染或推理，未安装系统依赖。preview/outputs/colleague_left_arm_200hz_20261005含1440×720、40fps、H264静音MP4；原速31秒＋末帧停留1秒，共1280帧。40fps每5个200Hz样本取一帧，包含31秒末样本；保存原CSV、6201行q/qd/qdd/时间NPZ、逐视频帧源时间、固定手姿态、场景及模型网格SHA、SDK参考限值、三幅源轨迹图、三时刻截图、HTML、manifest及ZIP。
- **验证**：全部6201组源q在MuJoCo qpos精确保留；65个姿态×30个臂／手连杆与独立URDF矩阵FK核对，最大矩阵误差1.887379141862766e-15。CSV错误单位列和非200Hz时间间隔拒绝检查、Python编译通过。交付CSV逐字节及NPZ四组数组逐值匹配附件SHA 6d69227ad89fb4967c2cb71201b96859e1a53ae60f0d2f39ea7ef50bc9582dde；视频时间数组及实际1280帧、32秒、尺寸、帧率核对通过，全片解码通过，15个交付文件SHA和ZIP内容／CRC通过。人工检查0／15.5／31秒画面及角速度曲线。证据logs/left_arm_csv_20261005/{fk_validation,export_validation}.json；输出目录及ZIP均无STL/XML/MJB。
- **边界／TODO**：外部CSV运动学展示，无动力学步进、策略、checkpoint、原MIDI或真实按键记录，MIDI对比及F1不适用，未捏造演奏指标。肩座／法兰继续使用现有暂定变换，不宣称实机标定；无手指动作来源。训练默认、源码归档和运行中作业未修改。首次绘图参数异常及近景取景的未完成缓存保存在logs/left_arm_csv_20261005，已修正后使用新目录完整导出，不将部分输出交付。

## 2026-10-05：paper/chord最佳网络追加三组姿态／同手碰撞续训

- **要求／实际参考**：用户要求基于当前训练，在统一脚本末尾追加参数，分别开启humanoid_finger_gesture_reward（idle_posture_penalty_v1）、self_collision_reward及两者组合。读取AGENTS、all_use_this_train.sh、1620完整config/best/PT、固定v2指尖归档展开的train/checkpoint/reward_config/rewards/task/self_collision/source入口、当前submit/job、既有idle模式参数说明；指定日志技能路径不存在，依项目要求记录。
- **参数／原因**：新增configs/experiments/paper_chord_reward_ablation_20261005/{idle_only,self_only,idle_and_self}.json，仅从1620实际reward_config改变指定两项。idle沿用归档模式默认weight=1、maximum_penalty_per_step=.1、sigma=.25；固定十指平均，当前及未来.15秒用指豁免，参考原场景origin的每指四个实际关节，每控制步至多全手-.1。self继承当前保存weight=.2，每去重同手body对每控制周期-.2，子步并集，不计跨手/正常触键。既有五项奖励、paper和弦on、false关闭、gamma=.8、LR3e-4、seed42、原曲/指法、20/200Hz、双卡流水及各10M追加预算保持。
- **续训合同／提交**：同一best15500 PT SHA 080b01d3608bdb2b6d8442abff4a31dbeec1b0e6d9f14a7cb2bfa19c3fe74956，复制只读parent_best.pt并保存父config、逐组只读奖励快照。通过原all_use_this_train.sh末尾追加--finetune-from／--allow-reward-change／--reward-config／--output-dir提交1681 idle_only、1682 self_only、1683 idle_and_self，每组19532×512=10,000,384新样本。原脚本未改，无新训练脚本；旧检查点/归档/作业未改。网络加载保留actor/critic/target_critic，但重建optimizer/alpha/RNG/replay/physics，不称逐状态恢复；scene/task/源码/动作/观察/控制严格检查保持，没有allow-control-change。
- **验证／打印**：三组真实根脚本假sbatch捕获（未提交）及固定归档CPU源码构建/inspect-only通过，effective_config对父配置仅reward_config不同，双卡实参/末尾覆盖/SHA通过。CPU构建关闭GL及CUDA可见性，未在普通会话执行GPU；无需算法修改或重打包当前ae7e29f3归档。每次正式提交前打印当前全部启用奖励、weight/params、performance及不按dt缩放。实际GPU训练仅Slurm分配内执行。
- **实际启动**：北京时间08:00:13，1681/1682 RUNNING，各vector_step200、102400样本／2048学习更新，指标有限且reward_terms分别实际包含gesture/self项。两组实际完整config逐项匹配预检，origin确认finetune_networks_only/父15500及PT SHA，11类物理/源码/控制/指法/观察合同与父模型一致。1683 PENDING(Resources)，尚无实际启动config，不把CPU预检称GPU启动验收。
- **证据／文档／TODO**：logs/hpc/paper_chord_reward_ablation_20261005含request、父快照、三组参数捕获、effective_config、entry_acceptance、逐组submission/job_id、startup_acceptance；docs/twinkle_20hz_comparison_20261004.md保存三组实际命令及公式，README同步父完成状态及新对照。待1683获资源后核验实际启动，三组长期F1、碰撞和吞吐待预算完成实测；负罚改变return，不据此宣称F1改善或物理安全，未额外开用户未请求的第四组。

## 2026-10-05：最新paper/chord训练完成检查与最佳视频导出

- **要求／实际参考**：用户要求检查最新训练并导出视频。读取AGENTS、all_use_this_train.sh、video_export_cli、Slurm1620日志/command、完整config/best/summary/metrics、匹配v2指尖归档及1620展开源码、export_video_job和既有同次回放compare.py；指定日志技能路径不存在，继续按项目要求记录。
- **训练实测**：1620 COMPLETED 0:0，19532向量步／10,000,384条样本，训练计时6200.725秒（约1612.777样本/秒）；1123条指标无非有限数值。预算正常耗尽，target_reached=false。best15500按三回合最差原曲精确F1选模=.9908081195，三回合合并=.9908725346；final合并=.9912263849、最差=.9902017291。best评估跨手接触/琴键底部子步/准备错键控制帧均0，但同手自碰撞最多179控制帧、1678子步，不能称整体无碰撞或已满足全部目标；不将final合并指标与best最差指标混排，不宣称双卡倍速。
- **执行／交付**：经export_video.sh→submit.sh export提交Slurm1678，79秒COMPLETED 0:0，使用source_v2_fingertips.tar.gz（SHA ae7e29f3778ba08ee7ec589a7568320cb24ef2b4e34688dc28ea76d898bc5fd3）与本轮best/config。输出preview/outputs/paper_chord_on_best_20261005：正面无臂原速有声MP4、原指法标注、手部实际关节速度/加速度图、HTML及ZIP。PT SHA 080b01d3608bdb2b6d8442abff4a31dbeec1b0e6d9f14a7cb2bfa19c3fe74956；无新训练，无M6加载，不修改历史作业/检查点/源码归档。
- **同视频原谱对比**：CPU读取同次1830个200Hz真实键程与触发阈值，重建audio_events逐项一致；原曲来自匹配源码内注册TwinkleTwinkleRousseau及原MIDI（SHA af583c13275d4fd1477f7c6d19b20714f470a40ccf84084b525fef027c14ed98），去掉14.692708333秒原曲首静音，加1秒策略准备偏移。原始事件逐子步匹配录制目标、所有控制末帧实际状态匹配，图含全曲及6.5–8.3秒细节、CSV/NPZ/MIDI/来源/compare.py。精确音乐范围[1,8.854166667)，视频物理范围[0,9.15]另停留1秒（编码10.2秒）；未量化/拉伸原谱。
- **视频单次结果／验收**：精确音乐TP/FP/FN=2590/25/25，F1=.9904397706；34原音符/35实际起音，150ms同音高最大匹配下34匹配、0漏音、1额外A3重触发（6.835–7.330秒），3个短音均匹配。这是诊断起音匹配，不是标准note-F1。该录制跨手奖励周期并集接触帧0、琴键底部子步0，同手自碰撞没有单独录制，明确使用训练评估记录而非冒充本视频指标。204帧1280×720/20fps H264全片解码通过、AAC 48kHz双声道，音频均值−27.3dB/峰值−12.1dB；PNG/画面实际查看、交付manifest和ZIP CRC通过。
- **位置／TODO**：data/training_status.json保存训练结果及完整奖励；data/video_contact_status.json与midi_comparison/note_timing.json保存本视频口径，logs/hpc/1678/video_export_validation.json保存验收。同步docs/video_export_cli.md。采样精度5ms，不含踏板余响；训练自碰撞问题与自动绘图尚未集成通用CLI的既有TODO保留，本次视频交付已手动补齐对比图。

## 2026-10-05：关闭false_press并启动paper和弦on双卡训练

- **要求／参考**：用户明确要求当前统一脚本关闭false_press_penalty，key_press_reward采用paper的0.6正确键/0.4指数错键公式，chord_bonus=true、chord_scope=on，开一轮训练。实际读取all_use_this_train.sh、S4原JSON、submit/job、source/learner解析、reward_config/rewards及既有paper/target-key测试；指定日志技能缺失。
- **修改／原因**：新增完整显式配置configs/rewards/paper_chord_on_20261005.json，相对S4仅false_press.enabled=false和key.params三个指定字段改变；根统一脚本改为绝对路径传入该JSON，输出标签paper_chord_on_fullhand_20261005。原S4配置与历史结果保留。按键公式mean_substep[.6×m×正确键分+.4×exp(-1.25×错键数)]，m=1/1.08/1.15（目标数≤1/2/≥3），只放大正确键部分，内部activation占比.5保持；其他当前参数以根脚本为准。同步README及20Hz内部说明。
- **验证／归档**：当前固定source_v2_fingertips.tar.gz SHA ae7e29f3778ba08ee7ec589a7568320cb24ef2b4e34688dc28ea76d898bc5fd3；奖励/解析五个源码文件逐字节匹配，算法源码未改变，因此未重打包。外部JSON通过绝对路径实际进入归档运行，提交时另保存不可覆盖的奖励快照及请求/回执。CPU19项/16子测试通过；真实根脚本假sbatch捕获和effective/source解析通过，确认双卡流水、20/200Hz、原曲/指法及10,000,384样本从头预算。五个数值案例1/1.048/1.09/.7626019/.4通过；shell语法和diff检查通过。证据logs/hpc/paper_chord_on_20261005/entry_acceptance.json。
- **启动**：北京时间2026-10-05 00:45:08经all_use_this_train.sh提交Slurm1620，双A100分配RUNNING，cuda:0采样/cuda:1学习/流水；GPU基础检查通过。输出training_mjlab/outputs/hpc/paper_chord_on_fullhand_20261005-20261005-004515-261777。提交只执行一次，未操作其他作业。
- **实际启动核验**：00:47实际config逐项匹配提交奖励/学习配置，31动作/1407观察、cuda:0/cuda:1流水、v2指尖归档SHA及无checkpoint恢复均通过。已200向量步/102400样本/2048次学习更新，全部已写指标有限，reward_terms不含false_press。证据logs/hpc/paper_chord_on_20261005/startup_acceptance.json。
- **TODO**：完整预算后的F1/碰撞/吞吐仍需实测，不能把CPU公式检查或初期更新视为效果改善。

## 2026-10-05：S4最佳视频同次回放MIDI对比图

- **要求／参考**：用户要求导出MIDI对比图。按新AGENTS规则使用第⑤组best16000视频的Slurm1619同次录制，读取report/config/audio_substeps/audio_events/rollout及原1587匹配源码内music/library、midi_file、preview_audio；以原曲未量化事件为参考，未重新运行策略。
- **产物／位置**：`preview/outputs/twinkle_20hz_best_S4_front_20261004/midi_comparison/`保存全曲及6.5–8.3秒局部PNG/PDF、原始MIDI/实际MIDI、实际/原曲音符CSV、逐音覆盖CSV、200Hz实际/目标NPZ、summary与可重建compare.py。原曲去除14.692708333秒曲首静音，再按该视频加1秒准备偏移，所有图使用视频原时间；音高按实际出现的琴键排列，轴标明只列这些键。同步导出HTML、report、manifest及ZIP。
- **验证／实测**：源指纹/PT SHA匹配；从实际键程与阈值重建的音频事件和原记录逐项一致；原始MIDI目标与1830个物理子步、末帧实际键与183个控制帧逐项一致。该视频音乐区间TP/FP/FN=2535/7/80，micro F1=.983129727；34个原音符、32个实际起音，两次E5（视频7.557–7.620及7.708–7.776秒）完全未触发。此单回放分数不混同独立10回合.984573。全曲/局部图检查、产物SHA和ZIP CRC通过，视频文件SHA保持；旧MIDI归档时间戳早于1980的打包问题已用ZIP兼容时间戳解决。
- **边界／TODO**：仅CPU处理既有记录，无CUDA/新训练；图为琴键触发时序，不含踏板延音余响，采样精度5ms。通用导出生成器尚未集成自动绘图，本次已按规则补齐当前视频，无本次交付缺项。

## 2026-10-04：视频必须附带原MIDI对比图及AGENTS整理

- **要求／参考**：用户要求写入AGNETS.md并清理旧AGENTS；核实项目活动指令文件为根目录`AGENTS.md`，未发现需另建的AGNETS。读取完整当前约定、统一训练脚本（含新指尖归档与用户最新参数）、video_export_cli及录制/原谱标注说明。
- **修改／原因**：AGENTS从225行整理为49行，保留统一双卡入口、全手碰撞/新指尖、Slurm、旧检查点兼容、未知映射、独立部署、日志与验证规则。删除活动指令中的重复/被覆盖旧入口、旧频率/奖励默认及长篇历史实验叙述；清理前完整原文连同SHA保存到`docs/agents_history_20261004.md`，明确仅作历史参考，不改归档内部文件。
- **视频规则**：每次交付视频必须附同一回放真实按键对原始MIDI的时序图（至少PNG），要求真实激活数据、原始事件时间、准备/曲尾/裁剪/倍速对齐、采样口径标注、图/数据/来源同目录和最终视频＋图双链接；有HTML/ZIP则一起纳入。同步`docs/video_export_cli.md`，工具未自动生成时要求补充绘图，缺项不可称完整交付。
- **验证／边界**：原文完整保留检查、当前约束及链接存在性检查、diff检查通过。本次只整理文档，不改训练/奖励/视频生成代码，不启动GPU或追补旧视频。后续导出须按新工作流生成并核验配图；不宣称CLI已实现自动生成。

## 2026-10-04：当前20/200Hz最佳S4视频导出

- **要求／依据**：用户要求导出当前最好训练视频。检查最新best记录、五组最终best_model/leaderboard及Slurm队列；当前20/200Hz对照独立10回合最高为S4_false01 best16000，合并F1 .9845730987。采用原config与source_v3归档回放，不将最新全手碰撞或双卡训练设置套入历史检查点。
- **执行／产物**：经export_video.sh→submit.sh export提交1619，46秒COMPLETED0:0。生成`preview/outputs/twinkle_20hz_best_S4_front_20261004/videos/S4_best.mp4`及index/report/ZIP；单正面无臂、1280×720、20fps、原速有声、原指法标注，完整9.15秒轨迹加结尾停留为10.2秒。未开训或修改训练参数。
- **验证／限制**：PT SHA与最终选模记录一致（cbffb53677e9334ab1fe588a00957fc01aca50ccff268391bd9ae11373c35d33），183帧目标指法匹配、204帧全片解码、9项产物SHA及ZIP CRC通过；音频均值−27.6dB/峰值−12.5dB非静音，4.5秒画面检查通过。证据`logs/hpc/1619/video_export_validation.json`。复评F1不标作本次单条视频轨迹分数，不与不同频率旧实验混排；无新增导出TODO。

## 2026-10-04：with ID指尖胶囊过窄复查与修正

- **问题/参考**：用户要求重新检查A3.2碰撞外壳，指出指尖胶囊不应过窄。读取当前AGENTS/all_use_this_train.sh、URDF/STL/model_derivation、collision_proxies/生成器/urdf_model/fingertip_geometry、指尖与全手测试、净空胶囊接口、旧A3.2审计与安装座说明；只使用包内资产。指定日志技能路径缺失，照常记录。
- **实测根因**：with ID末节STL总Y宽14.56–16.00mm，旧生成器只测最高1mm薄带，将4.02–4.84mm窄截面作为整根末节直径并沿Z放置。此前未充分说明该规则造成的整根异常收窄，本轮用两投影实网格叠加证实。
- **修改/原因**：generate_collision_proxies新增完整末节等效惯性半径/主轴拟合并锚定原指尖和局部Z极值；只更新左右10个末节capsule，直径11.72–12.82mm。新增--terminal-only/--output用于候选与精确控制改动范围；资产记录whole_mesh_inertia_tip_anchored_v1/原生拟合/端点/源SHA，删除薄带宽度字段。其余34项、模型/视觉/惯性/驱动/关节/指法/奖励/SAC参数不改；仍为近似，最大顶点外漏从11.33–12.23mm降到4.72–6.36mm，不冒充完整外壳包络。
- **图/整手审计**：新增draw_fingertip_collision_audit.py输出docs/figures/fingertip_collision_audit_20261004.png及JSON，另导出更新后的整手范围图。掌壳实际加载更新后的base_without_old_mount；JSON旧base来源描述不决定mesh加载。安装座实际primitive仍是旧box，未联动修改，登记TODO(V32-MOUNT-COLLISION-01)。44项来源/修改前后/初始接触记录在logs/fingertip_collision_20261004。
- **验证**：CPU碰撞代理4项通过，覆盖实际编译/完整网格拟合/镜像/原指尖和Z端点/非碰撞物理合同；重生成JSON字节一致。源码场景22/22名单、初始穿透接触0。Slurm1618 34秒COMPLETED0:0，5项通过：全手检测/S4分值/子步并集/局部reset/graph一致及8环境120步（6秒）真实运行，含后半小幅有界目标变化。qpos/qvel/奖励有限，max|qvel|2.067535、max|qpos|1.167660、跨手接触0；不视为硬件速度标定或长期策略安全。图已实际查看，shell及diff检查通过。
- **归档/文档**：根all_use_this_train.sh只切换至已验收1618同一源码包source_v2_fingertips.tar.gz，SHA ae7e29f3778ba08ee7ec589a7568320cb24ef2b4e34688dc28ea76d898bc5fd3，保留最新双卡与S4其余参数。166个runtime文件与当前源码匹配、归档审计通过；最终假sbatch捕获统一入口确认新归档/SHA、双卡流水与末尾覆盖，未提交训练，证据logs/fingertip_collision_20261004/entry_acceptance.json。AGENTS/README/HPC/全手说明同步，详情docs/fingertip_collision_audit_20261004.md。
- **TODO/边界**：本次未开新长期训练，旧v1归档/检查点不回写。保留工程碰撞/指垫标定V32-06/07、安装座新box拟合、完整策略F1/碰撞/吞吐验收；旧净空IK/梯度失败未改，两动作保护仍关闭。没有声称全壳保守包络或无碰撞保证。

## 2026-10-04：统一all_use_this_train.sh切换双卡流水

- **要求／参考**：用户要求将当前统一脚本改成双卡。读取根脚本、submit.sh/job.sbatch执行与GPU分配规则、最新AGENTS、README和五组实验说明；保留已确认的新全手碰撞source_v1归档及SHA。
- **修改／原因**：仅将根脚本的`--gpus 1`改为2、learner-device从cuda:0改为cuda:1、开启collector-pipeline；cuda:0继续物理采样，cuda:1承担SAC学习，不属于双学习器梯度并行。其余S4学习参数、10M预算、奖励、控制、全手碰撞模型及归档不变。AGENTS/README/实验说明同步新的双卡默认。
- **验证**：bash语法和diff检查通过；使用假sbatch分别捕获修改前后的真实根入口→submit链，确认仅GPU数量、学习设备、流水开关三个实参差异，末尾覆盖顺序正确；证据`logs/hpc/s4_saved_entry_20261004/dual_gpu_entry_acceptance.json`。未运行CUDA、未提交作业，不改变已运行任务。
- **TODO／边界**：本次只改启动执行布局，未做新配置双卡测速；吞吐、F1及碰撞以之后Slurm内实测为准，不承诺翻倍加速。

## 2026-10-04：全手跨手碰撞替换与最终统一训练入口

- **问题/授权**：用户要求当前双卡forearm_reward覆盖左右手任意部位接触，任务名改tsensei_collision_v1，无碰撞0/有碰撞−.5，绘制范围图；统一HPC/当前训练配置、run.sh archived、AGENTS约定以后只在当前双卡脚本后追加参数。用户随后明确不要保留旧掌部模式，当前源码直接替换，历史归档不回写。
- **实际参考**：读取AGENTS、run.sh/run_tsensei_a32.sh、scripts/hpc/{train_coupled_20hz,train_coupled_40hz,submit,job.sbatch}、forearm_collision/fingering_targets、suite任务、mjlab_env/{task_bundle,task_logic,task,source_runtime,rewards,substep_rewards,scene}、当前HPC奖励JSON、真实A3.2编译geom、既有绘图/入口/源码/奖励测试与README/HPC说明。本机指定日志技能路径缺失，照常维护本日志。
- **实现/原因**：tsensei_collision_v1选择手子树全部_collision_实体geom（22/手，含掌壳/掌段/五指），替换旧shadow_v1任务选项和缺名单退回掌部的路径；保留统一指法目标及shadow驱动/primitive物理数值。CPU/Torch基础及子步奖励同步0/−.5，控制周期取接触并集、不按部位/子步累加。新manifest核对实际模型完整名单，旧任务使用原归档，不静默迁移。更新CLI、相关诊断工具及使用任务profile的验收fixture；旧动作范围测试1.3预期纠正为当前配置1.2，未改动作范围。
- **入口/规则**：train_coupled_20hz.sh合并原run→A3.2→HPC实际默认，直接调用train_from_source.py，保存双卡/耦合/20-200Hz/10M预算/奖励/控制参数，末尾覆盖保持。run_tsensei_a32.sh和旧40hz文件仅转发；run.sh只加ARCHIVED标识，退出当前依赖链。AGENTS新增当前优先规则：每轮新训练走submit.sh train并追加参数、不得新建训练脚本，后续默认仅改当前双卡脚本。README/HPC说明及docs/tsensei_collision_v1_20261004.md同步。
- **范围图**：preview/scripts/draw_collision_scope.py纯CPU使用编译模型mesh凸包与primitive采样绘制双视角，输出docs/figures/tsensei_collision_v1_20261004.png和JSON；实际查看通过，44个实体geom逐项记录。未调用GPU绘图或生成猜测手模型，未修改实体碰撞/驱动/关节限位。
- **验证结果**：实际CPU源码构建22/22与实体名单一致；主回归65项/85参数子测试通过，1 CUDA项跳过。扩展回归65项/52子测试通过，4 CUDA项跳过；净空保护的两组旧IK与一个梯度检查失败，切回原tsensei_v1范围复测仍同样失败，保留证据logs/tsensei_collision_20261004/clearance_legacy_scope_control.txt。新增入口/任务检查复测通过。Slurm1615单卡专项4项通过（21秒COMPLETED0:0），包含GPU8世界掌掌/指指/指掌/反向、同手/琴键排除、重复接触、末端分离仍扣分、局部清空和实际3控制周期；1614首次是测试snapshot漏endpoint导致缺精确目标历史，修正测试调用后通过。最终去掉旧名单fallback后的Slurm1616中GPU专项继续通过；一个入口静态断言因另一工作并发加入--submit-s4转发而失败（原断言禁止出现任何exec bash，过宽），收窄为禁止依赖run.sh/run_tsensei_a32.sh后，CPU14项/8子测试通过、1 CUDA项跳过。shell语法与git diff --check通过。
- **用户最终入口选择**：另一工作新增all_use_this_train.sh并在AGENTS指定单卡S4原归档。用户明确选择最新根入口，并要求替换其中旧掌部模式。本轮保留全部S4其他参数，改为task-profile tsensei_collision_v1并固定新归档logs/hpc/tsensei_collision_20261004/source_v1.tar.gz，SHA 8c2fd369a8fcc36825cb384138b15a6d2d562b6425637c2e912ac8809c882df3；原归档/原S4 JSON不改。AGENTS/README/HPC及五组说明改为最终根脚本优先，后续只改根脚本或追加参数。归档内runtime源码相对原S4仅本轮7个碰撞/任务/指法兼容文件变化，未混入其他模型/学习器变更。
- **最终根入口验收**：假sbatch捕获all_use_this_train.sh真实提交链（未提交训练），与S4实际config的21个训练区段完全一致，4个source recipe区段除environment.task_profile改名外相同；单卡/末尾覆盖、固定SHA及归档166个runtime文件与当前源码逐项一致。证据logs/hpc/tsensei_collision_20261004/unified_entry_acceptance.json。直接以该固定归档提交Slurm1617，20秒COMPLETED0:0、专项4项全部通过，新增验证S4空forearm params路径最终同样0/−.5。最终shell与diff检查通过。
- **边界/TODO**：本次只修改和短验收，未启动新SAC长期训练，未改已运行作业/协调器/归档。未实测全手惩罚的F1/碰撞率/吞吐收益；TODO(V32-CLEARANCE-REGRESSION-01)记录现有净空保护三项失败，当前入口两个动作保护开关关闭，未擅改其算法。绘图和接触单测不能视为全程避障或硬件标定。

## 2026-10-04：第⑤组配置保存为统一默认all_use_this_train.sh

- **用户要求／参考**：先要求追加第⑤组实际参数到训练脚本，随后明确命名`all_use_this_train.sh`并以后统一从该脚本追加/修改参数。读取S4实际config/summary、source_v3归档、当前train_coupled_20hz/submit/job/source入口及AGENTS/README；日志技能路径缺失，按本项目日志约定执行。
- **修改／原因**：新增根目录可执行`all_use_this_train.sh`，集中显式保存S4完整学习/模型/控制/奖励/评估参数，末尾`"$@"`覆盖；默认单卡submit、从头10M，不加载best。当前工作区已移除旧shadow_v1任务，因此校验并固定原v3归档，保持原任务/奖励语义；不回滚工作区全手碰撞实现。旧HPC脚本增加`--submit-s4`兼容转发，默认参数维护位置改为新脚本。同步AGENTS优先约定、README及五组实验说明，明确归档依赖与源码更新方法。
- **验证**：bash语法检查通过；使用假sbatch捕获真实提交参数（无作业提交），展开原归档shell链并用原source/learner解析器核对，21项训练配置和4项source recipe与S4保存值一致，单卡、从头初始化和末尾覆盖正确，CUDA未初始化。验收脚本中先规范tuple/list的JSON表示，并对临时目录缺失的依赖版本查询使用已保存版本fixture；不涉及模型构建或GPU验收。证据`logs/hpc/s4_saved_entry_20261004/acceptance.json`。
- **边界／TODO**：未开启新训练。`.gitignore`加入新根脚本白名单，确保可纳入版本管理。入口默认使用已验收原归档，不能宣称工作区新源码自动生效；后续算法源码变更必须验证并更新归档/SHA，部署携带项目内归档。无新增F1收益或硬件安全结论。

## 2026-10-04：训练入口统一选定奖励的子步评分

- **问题/用户范围**：用户要求训练脚本采用子步评分，随后明确只统一按键、误触、能耗与碰撞，指法和姿态保持现状。普通入口已有部分开关，Shadow专用入口仍强制端点，h10/h12旧入口未开启。
- **实际参考**：`run.sh`、`run_tsensei_a32.sh`、`scripts/train_twinkle_h*.sh`、`scripts/hpc/{train_coupled_20hz,train_coupled_40hz,submit_twinkle_shadow_rewards}.sh`；`mjlab_env/{performance_options,rewards,substep_rewards,task,task_logic,scene,exact_score,anticipation,finger_clearance}.py`、`rl_torch/train.py`；现有入口、五组实验、Shadow、正确键、指法实验与精确时序测试。可选日志技能路径不存在，按本项目日志要求记录。
- **修改位置/原因**：普通/HPC/h10/h12基础脚本显式`reward-sampling auto + substep-rewards + target-key-substep-reward`，派生入口继承；Shadow五项专用入口切换同样采样，保留五项名单及activation_weight=0，新默认输出目录加substep/当前日期以区分历史。只改启动选择，没有更改评分内核：连续项均值、碰撞周期并集/去重，指法/姿态/几何端点保持。README及HPC说明同步；现有入口验收测试覆盖派生脚本并更新Shadow预期。
- **验证结果**：CPU运行`test_hpc_40hz_entry`、`test_shadow_only_training_entry`、`test_twinkle_20hz_experiments`、`test_shadow_reward_only`、`test_target_key_reward`、`test_fingering_experiments`、`test_exact_note_timing`，55项及35子测试通过。真实参数链和五组显式配置保持，奖励均值/瞬时事件、旧合同通过；CUDA屏蔽，未进行GPU执行。
- **边界/TODO**：没有启动、取消或重启训练，没有改正在运行的协调器/归档/检查点或历史实验JSON。当前代码采样能力已存在，本次仅统一入口；新默认训练的F1与吞吐需后续实际训练确认，不宣称达到0.99。

## 2026-10-04：第④组S3最佳模型视频导出

- **要求／参考**：用户要求导出第④组视频。读取有效`twinkle_20hz_comparison_20261004_v2/S3_self02-20261004-131836-186923`的best/config及`source_v3.tar.gz`，参考`docs/video_export_cli.md`、`scripts/export_videos.py`、`preview/scripts/export_video_job.py`和HPC入口；独立CPU审计确认best/final整文件一致，step19532，39项训练源和167项运行源散列匹配。
- **执行／产物**：标准`export_video.sh`经`submit.sh export`提交1588，46秒COMPLETED0:0。单正面无臂、1280×720、20fps、原速有声，保留准备/曲尾及原谱指法，关闭图表；输出`preview/outputs/twinkle_20hz_S3_front_20261004/`内MP4、index、报告和ZIP。未修改训练代码或运行参数，1587继续训练。
- **验证／边界**：训练归档SHA `3873a973b36368afd300e055d7ddaf1fbe5264902b6e04405c8846cbbf2895f2`；PT SHA `8498ee9355937c186924d627b983e430d679867bb5aed65a2ad8d113bfe5df4e`。1830物理子步/9.15秒完整录制，183帧指法与原记录逐项匹配；视频204帧/10.2秒含结尾停留，H264/AAC双声道非静音（均值−27.3dB、峰值−12.4dB）、全片解码、9项产物SHA和ZIP CRC通过，检查1秒/4.5秒画面。证据`logs/hpc/1588/video_export_validation.json`；同步五组实验说明。训练三回合F1 .984377不冒充此次单回放精确F1；长期对照独立复评仍按原协调器执行，无新增导出TODO。

## 2026-10-04：20/200Hz五组F1对照

- **要求／参考**：用户要求40/400Hz改20/200Hz并开启多个对照寻找最高F1。实际读取run/HPC/Slurm入口、原五项JSON、performance/rewards/source_runtime/task、1517和R6独立10回合结果、现有对照协调器与测试。保留模型/指法/机械与控制约定，无外部目录依赖；本机可选日志技能缺失仍按项目日志约定执行。
- **入口／实验**：新增train_coupled_20hz.sh，job.sbatch切新入口，旧40hz文件名兼容转发；20/200与lookahead10保持.5秒，当前耦合31动作、完整1407观测。普通run.sh原已20/200。新增五份显式JSON：S0原五项末端、S1子步聚合、S2加真实激活.5、S3对S2仅self.2、S4对S2仅false.1。全部无dt缩放/无未来指法奖励/无姿态奖励，观察前瞻保持，不加载旧2297模型。
- **预算／验证计划**：每组从头seed42、512×19532=10,000,384样本、batch1024/.25、三LR3e-4、gamma.75/初温.5保持；183帧预填后预计2,476,672更新，eval500×3，完整预算对照，明确禁用实际提前结束。入口21项及16子测试、五组实际包装链8项通过；原34音200Hz仍有采样，不将20Hz端点一次目标帧称短音完整覆盖。GPU短验收1562/1563以固定归档执行，SHA `0ed7fbd2870a4c0b1cf54a64dbdedb20eed834e1ad397147fe3b360524846a4f`，实际1407/31、20/200/183帧和奖励匹配。
- **短训／协调器验收**：1562/1563各240步122880样本/7296更新完整完成，scene/task/动作/准备/源散列一致、奖励各自匹配、六份PT有限及诊断完整；final F1=0/.032777只作功能验收。真实CPU源构建34音183帧、1407/31、准备20/尾5、solref.01通过。新增run_twinkle_20hz_comparison.py及22项fakeSlurm/实际schema测试，联合入口与奖励47项/43子测试通过；实际两smoke经协调器增量/结束/检查点检查通过。补历史source归档缺20入口时回退原40入口，仅缺文件回退、不吞新入口错误；14项/38子测试通过（批次重叠）。
- **正式启动**：v2加入协调器/测试/说明与归档兼容，39学习源码+167 source_runtime SHA与实际smoke一致；正式归档SHA `03741f41799c6f3c68b86641ea7493fd1b72e3687f2bd648249ec5c05d7a803b`，manifest SHA `80a88db58a4658c51d3057935311bad0e57e3306cd8c7b2c97a8d0874958d5e4`。13:01:14 PID/session683184 detached启动，13:01:33 S0–S3作业1564–1567均RUNNING初始化，S4协调器待办，最多四张卡并行。协调器持久化提交/逐config与样本数审核，训练后全部合格best/final最多10个各独立10回合，最终按pooled精确F1/min/接触排序并保存引用；失败/未知提交不冒充完整对照，旧模型不覆盖。状态、leaderboard及最终best_model位于training_mjlab/outputs/hpc/twinkle_20hz_comparison_20261004，来源/验收位于本轮logs。
- **实际学习核验**：13:04:09四组config/source/奖励/模型合同均通过，实际340步174080样本/20096更新，已跨183步预填，协调器无错误；S4仍待空闲。本轮startup_acceptance.json保存证据，不把初始化或排队称已学习；当前仍为早期训练，未宣布最高F1或.99达成。
- **首轮误判／纠正**：随后1564–1568均在第366步被协调器错误取消：纯episodes记录写于当步学习前，23296更新应按365步核对，而旧增量验证误用step366要求23424。上条只代表13:04早期状态，首轮不能继续称运行正常；最初startup_acceptance未生成，改保存first_launch_audit_failure.json。独立检查五组SAC有限/无CUDA错误，仅该日志时序误判。修复仅纯episodes按step−1，其余train/eval仍严格step；23项回归及五份实际失败日志CPU重验通过，保留全部失败产物。增加600步S0/S3多回合GPU短测后使用新输出目录从头重启，不改变奖励/学习预算。
- **修复验收／正式重启**：新增3660同step多记录用例后24项协调器单测通过，联合入口/奖励/回退52项及43子测试通过。1569/1570各600步307200样本53376更新完成，真实366/549边界、4评估点和best/final完整检查通过；1571/1572自动各10回合复评完成，最终排名生成，S3短策略键底最大105被资格排除（功能通过不称物理安全）。v3归档SHA `3873a973b36368afd300e055d7ddaf1fbe5264902b6e04405c8846cbbf2895f2`，learner/model source与最初smoke不变；修复版manifest_v2 SHA `e0dfa2318ae08950d6343809467d1c0ddaf2fd7202ba5086793b8efcd5b57889`、协调器SHA `e7682409f6a19a61750ef9680e543eb0ad1b24e4fc1753c064eb60c5f100b8cb`。13:18:29 PID/session710615在新twinkle_20hz_comparison_20261004_v2根目录从头重启，S0–S3作业1573–1576 RUNNING，S4待空卡；13:19四组实际config均验证通过，旧首轮不纳入完整对照或覆盖。README/本轮说明已改为有效_v2状态。
- **重启后实际核验**：13:22:04四组均超过原误判位置，至少440步225280样本32896更新，完整奖励JSON/1407观测31动作/10M预算与原档一致，S4待办，无新错误；证据startup_acceptance_v2.json。600步流程另有54项独立审计通过。S3短训final键底12先被排除，best400再在独立复评检出105而拒选，说明功能检查通过并不宣称策略物理安全；正式结果仍待完整学习预算及复评。
- **15:41只读检查**：用户询问当前情况，读取_v2 state/leaderboard、实际config/metrics/summary/best、sacct和协调器日志。S0–S3完整10,000,384样本/2,476,672更新正常完成，保存best pooled F1=.964542/.980321/.979322/.984377，final=.964892/.972680/.977749/.984377；各完整物理诊断与计数复算通过，协调器已核验检查点。S4于15:25:37自动接续1587，快照2340/19532步约12%，当前best2000=.853699，SAC有限、无新核验错误，尚未独立复评/最终选模。S3 TP/FP/FN=7624/21/221、min=.984115，短音100/108覆盖92.59%，self0/inter14/side194/under0/准备误触0；相对S2补70TP少7FP。漏采样迟按151/早松69/中断1，A3/L2第23音约149ms迟按占90FN，非全漏。证据在logs/hpc/twinkle_20hz_20261004/status_20261004_1539，更新README和本轮说明；未改运行参数或开新训练，五组完成后按既定自动10回合复评确定排名。
- **文档／TODO**：同步README、HPC说明及docs/twinkle_20hz_comparison_20261004.md。五组已启动自动执行，所有GPU只经submit.sh；后续完整预算、独立复评最高F1/碰撞与时序失分仍待实际结果。不能跨200/400Hz混排名，不能保证.99或全局最优；当前最高模型仍保留原R6直到新结果验证。

## 2026-10-03：关闭附加奖励开训与提前松键实测（10-04完成结果检查）

- **要求／实际参考**：用户要求原Shadow基础外的奖励惩罚全部关闭并开一轮训练，查清提前松键是否更高奖。读取包内Shadow参考合同/current.json、当前HPC/run入口、reward_config/performance/rewards/task/task_logic、原R6 config/PT/source_v1及逐音误差；不读取或运行外部项目。指定日志技能本机缺失。
- **实施／原因**：五项原公式、weight1、末端评分/无dt；关闭自碰撞/false_press/humanoid和其余附加项。新增独立reward_sampling与anticipatory_fingering_reward开关，解除精确谱面强制子步奖励和提前观测/得分耦合；保持2297观测、精确时序、模型和控制。增加独立实验JSON/submit入口、同状态分支诊断脚本及CPU测试，旧缺字段合同不回写。
- **资源／参数**：扩展submit --gpus1支持train/validate/reward-diagnostic；train自动同卡采样学习、关闭双卡流水，默认双卡保留。从头seed42、512×19532=10,000,384样本、预计2,453,376新更新，batch1024/.25、三LR3e-4、gamma.75/初温.5保持，400Hz原谱最小回合F1选best。GPU只通过submit.sh，不热改其他运行作业。
- **CPU验证**：奖励及兼容路径70项唯一测试通过；实际包装链/HPC35项及38子测试通过；三个新增专项合跑16项/6子测试通过（批次有重叠）。旧JSON不新增字段、2297观测逐值保持、末端与子步矛盾拒绝、实际包装参数/五项均验证。相关README、HPC说明和docs/twinkle_shadow_rewards_20261003.md同步。
- **GPU短训／正式提交**：1507完成512×400样本、4480更新，实际scene/task SHA、动作/控制/耦合/准备合同与R6一致，forward/best/final及3次完整评估验收通过；155.49秒，final F1=0，仅功能短验收、不代表收敛。源码归档SHA `8a7c4e2e17e30ebc41515d2e2d4adc53a8081962277d10442e19023cbb99d178`、奖励SHA `d32cc4ba58cd110e6b9b62916ebf9543d6022bccc57e2336a006e76ecf98e2aa`。17:45:28提交正式单卡1517，17:49仍PENDING(Resources)，不可称已开始更新。CPU单次watch_training.py核验实际新config与smoke合同/从头初始化/预算，并等首次updates>0后写验收退出；独立复审确认实际metrics字段和SHA匹配。回执及状态在logs/hpc/twinkle_shadow_only_20261003，未运行监控CUDA。
- **提前松键实测**：原R6 best/原source_v1严格回放，1515诊断72秒完成，F1=.984520、TP/FP/FN=5088/18/142；18/18完全相同起点、54/54动作分支对照通过，重复最大qpos差6.71e-7/奖励1.19e-7、实际离散评分一致。1506首版因重复容差过严拒绝，结果保留；v2起点仍逐值一致、重复容差1e-5并检查TP/FP/FN/键bits，效应显著大于浮点噪声。39个补回目标的干预中34个150ms回报下降、5个上升。D4/L1在5.725s多保持25ms补7个目标却多5个晚松错采样：当步奖励+ .024518，150ms总回报−.099380；分项按键−.084847、false−.004977、能耗−.010041、指法−.002528、姿态+.003014。更早一帧短持则补3个目标/无新错键、总奖+.014371，故非所有早松最优。
- **10-04完成检查／实际参考**：按用户要求读取1517 sacct/stdout/stderr、实际config/summary/metrics/best.json、三个PT、固定源SHA和奖励合同，比较旧R6十回合记录。1517在10-03 18:01:56–20:29:57完整10,000,384样本/2,453,376更新结束，COMPLETED0:0，target=false；启动watcher曾在18:04:18验证380步/1920更新后退出。21次评估、选模/完整预算、三PT有限及合同75项CPU检查通过。保存best19000精确pooled/min=.980483/.979826、final=.967118/.948681，final第三回合G5全漏；40Hz best=.995026不能宣称精确达标。
- **独立复评／新增只读产物**：10-04 02:57经submit.sh固定原归档提交单卡1557，76秒完成10回合，严格源/奖励/控制/PT匹配与25项验收通过。best pooled=.977806/min=.973253，TP/FP/FN=51085/1104/1215；旧R6同10回合=.984785、50874/146/1426。新FN迟按931/早松224/中断60/全漏0；旧496/720/0/210。早松−68.9%但错采样7.56倍，767个FP位于结束后50ms内；短音30/30匹配、643/730覆盖(88.08%)，旧499/730。新手接触3154/3650=86.41%子步，主要同手3153.6，跨手24.7，侧接触982.7，键底0；准备40Hz0但400Hz4.2误触子步。best保存门槛不要求手接触0（只有达标早停要求），没有门槛执行矛盾。产物在本轮logs/results_20261004，新增plot_progress.py及CSV/JSON/PNG/PDF，全部计数复算一致；报告/README更新为已完成，无源码/奖励修改或新训练。
- **结论／TODO**：只补74个仍目标的音尾漏采样且其他项固定，总奖励+.476830、无负增益帧、false变化0；没有发现仍应按时直接鼓励松键的公式反转。真实延长动作可能因晚松而失去原按键的完整无错键奖励，超过补齐目标均值的收益；附加false不是示例主因。完整五项训练改善早松/短音却未超过R6整体F1且接触增多；原40Hz末端奖励存在子步盲区。不能将150ms/单轨迹干预或不同初始化/预算/多项配置的结果称单奖励因果；gamma等独立贡献、多训练种子和.99仍未验证达成。当前保留R6整体最佳，未擅自续训。本轮全部证据及历史状态保留。

## 2026-10-03：小星星 F1 奖励与微调实测

- **要求／范围**：用户授权尝试上一轮建议。本轮四组各4096×512新样本：原参数对照、仅false_press 1→.25、仅真实触发占比.5→.8、仅gamma .75→.90；同一A best、相同网络迁移和alpha=.0006。保留模型/指法/频率/观测/其余奖励及学习参数，不同时实现速度观测或新奖励。
- **实际参考**：上一轮分析、A完整config/best/PT与原归档、submit/job/train_coupled入口、run.sh/source构建、rl_torch train/checkpoint/SAC/evaluation、现有消融协调器及相关测试。CPU安全加载父PT确认保存alpha=.0005938008544、actor/critic/target结构和场景/任务合同。可选日志技能缺失，按本目录要求记载。
- **实施／原因**：新增四份独立完整奖励JSON与`docs/twinkle_f1_finetune_20261003.md`，同步README；补可选400Hz精确F1选模及配套队列/验证，默认旧40Hz合同保留。全部组采用fresh optimizer/alpha/RNG/replay的相同网络迁移，避免把默认初温.5或旧优化器混入单变量比较。
- **验证／启动**：74项CPU测试/109子测试通过，捕获真实入口确认仅指定单参数差异，真实父网络四组迁移逐tensor一致。1364.34完成512环境400步/4480更新短验收，实际scene/task合同不变、forward200/updates0、完整config/PT/精确选模/流水排空通过；final F1暂降.87773，step0 best保留，不声称烟测增益。1457.3原归档父模型10回合复评pooled=.973818、min=.970357、max=.977022，短音0/30。训练归档SHA`1709a25fbefee1d0b8e90f49a2064a3536766e23c71a1c7cc595f30da00d808e`，01:27:09独立PID/session2712292启动R0=1364.35/R1=1457.4，R2/R3按空闲双卡接续；新协调器逐config/SHA/预算/诊断/PT核验、失败隔离/不重试/只清理已归属步骤及其本地进程组，保留两个预留。状态在`training_mjlab/outputs/hpc/twinkle_f1_finetune_20261003/runtime.json`，完整验收在`logs/hpc/twinkle_f1_20261003/`。所有GPU执行均通过submit.sh。
- **用户追加／CPU验收**：四组运行中用户进一步授权完整学习器恢复或更低Actor LR，增加R4完整warm-start与R5仅Actor3e-5，各同4096步/477568新更新，仍同一A父模型/原奖励，与R0比较。R4严格保留父SAC名义初温.5合同、实际恢复已学alpha=.0005938008544，累计更新2257280→2734848，新replay；R5保持fresh状态和alpha=.0006。冻结源码下真实父PT/Adam/RNG字节恢复、LR和CPU一次更新验收通过；脚本/JSON保存在本轮logs，未执行CUDA。相关说明同步本轮文档/README。
- **原四组结果／复评**：02:26四组全部完成相同预算和22次评估，流水/PT/config/诊断验收通过；best合并精确F1依次.974481/.980641/.976094/.975785（R3为200步、零新增更新父网络），final为.974141/.978031/.973805/.965531。过程键底违规评估次数0/2/1/4均被选模拒绝；所有best/final短音实际覆盖仍0、键底0。1364.37在固定归档独立复评R1 best10回合：pooled=.981096、min=.980831，相对父10回合.973818/.970357改善，手接触7→1.3、侧接触259.5→262.9、键底0；实际源谱/奖励/控制合同通过。88评估点加父10回合9316逐音与复评20回合680记录CPU重构吻合，CSV/JSON/PNG/PDF在本轮logs分析目录。用户再次要求其余组也继续，原四组均正常跑满，追加两组继续准备。
- **追加启动／验收**：独立`scripts/hpc/run_twinkle_learner_finetune.py`及8项新增测试，连同原11项回归共19项通过；使用同一固定GPU源码，新协调器SHA`61fe8857500e2e955d5eef5932e11427e2895f4ba104c9653ec6ff56fbbd40c8`。02:38 PID/session2793654在1364.38/1457.6启动R4/R5，完整config和forward200立即核验通过：R4网络/Adam/alpha/update_rng均恢复父值、本轮更新0；R5网络一致/空Adam/实际Actor3e-5。原四组队列和结果保持，新状态位于`training_mjlab/outputs/hpc/twinkle_learner_finetune_20261003/runtime.json`，追加分析明确扣除继承更新数。
- **学习条件最终结果／下一分支**：03:07 R4/R5各完整4096步/477568新更新结束，best精确合并F1=.975326/.975317，final=.972939/.973818，短音均0。低Actor LR前800步最大掉分由R0 .196842减为.035170，最终未超过R1；完整恢复掉分.227819，仍有3次键底违规评估，均被拒绝，不能归因为Adam单项。全部44评估诊断完整、PT/config/选模/导出核验通过。用户要求继续找最佳方法，沿最初“false.25有效再测.1”分支追加R6 false.1与R7 false.25+Actor3e-5，同原A父/相同fresh迁移与预算；这是后续强度和交互对照，不预先宣称增益。
- **R6/R7实施／启动**：新增两份完整奖励JSON、独立`scripts/hpc/run_twinkle_reward_refine.py`与9项测试，连同前19项回归共28项通过；真实冻结包装链及CPU迁移/config/LR检查通过。协调器SHA`29c7f23e60d56c6b75b4dd7827d74b8f2c390004118ba1e37092e556a35edac2`，GPU归档仍source_v1。03:23 PID/session2841826启动1364.39/1457.7，两组完整config及forward200均通过、父网络一致/空Adam/alpha.0006/新更新0，前六组结果保持。命令、输入哈希和启动记录在本轮logs的reward_refine_*，只读分析为analyze_reward_refine.py。
- **R6/R7完成／汇总**：03:52两组完整预算完成且PT/流水/诊断验收通过；best合并精确F1=.984605/.980178、final=.981543/.979318。R6两个短E5覆盖150/156，但中间短D5仍0/63；总短音150/219，best侧接触324.33/手接触0/键底0，final侧接触增至695.67。R6过程400/600步键底107/22均被拒绝；R7全评估键底0。final_comparison.py汇总八组176评估/528回合，400Hz计数全部复算吻合，R4历史更新与R3父网络best明确分开。R6 best经CPU全合同验收。
- **单卡复评入口／实测**：原预留到期且仅剩1张空闲卡，新增submit.sh --gpus 1（仅新replay作业）、job.sbatch传预期数量、probe_gpus支持单卡并跳过peer；默认两卡及学习参数不变。33项CPU/34子测试、shell/diff检查通过；取消自己pending1498后提交1499，单卡GPU probe及固定归档回放76秒正常完成。R6十回合独立精确F1=.984785、min=.984016，高于父.973818与R1 .981096；短音499/730、匹配20/30，手碰撞0/键底0，侧接触306.7。检查点/源谱/奖励/控制散列与全部计数重构通过，完整三队列和GPU作业已结束；更新本轮文档、README和HPC入口说明。
- **TODO／边界**：初筛单seed不代表长期/多seed收益。完整恢复不得将历史更新数误记为本轮学习；R6尚漏短D5和长音边缘，理想补全短音也仅.987050，仍需额外303个FN修复才到.99（现有FP固定），不宣布达标。R1保留为错键/侧接触更少备选。逐音FN分解为早松720/迟触496/整音漏210，.1组比.25组短音多499TP、其他音少44TP；下一窄对照.05惩罚仅记录为候选，保留当前指法奖励合同，未开新组。best规则中手碰撞覆盖缺失仅限制早停的潜在边界本轮未触发，所有真实诊断完整，追加协调器终检独立检查全部覆盖。旧训练与原始日志保持。

## 2026-10-03：当前小星星 F1 0.99／1 候选审计

- **问题／范围**：用户要求依据当前训练找提高 F1 的奖励或其他条件。本轮分析最新已完成 A/C/D/E，各约 10M，同 seed42；不混用旧 9 月 .99 或不同合同的历史实验，不启动训练或更改训练默认。
- **实际参考**：四组 config/best/summary/metrics、原源码归档及 SHA，双卡入口；mjlab_env 的 rewards/fingering_experiments/task_logic/task，rl_torch 的 note_timing_metrics/checkpoint/train；现有消融、速度和时序实验文档。核对 SAC 折扣／熵与优先重放的原始资料，仅作一般机制说明。可选日志技能路径缺失，仍按本目录要求维护日志。
- **新增分析／原因**：`docs/twinkle_f1_candidates_20261003.md` 和 README 记录当前证据、单因素候选及验收。A best TP/FP/FN=14933/4/757；三个短音只贡献219个FN，理想修复后精确F1=.98242884；固定FP到.99至少修451个FN。D的B3/A3“未匹配”为超过150ms窗的迟响，两处占474/853个FN。当前观测缺速度、gamma=.75远期折扣、40Hz选best与400Hz目标不一致是另外的待测改进条件。独立复审补充网络微调公平性：改奖励入口重置Adam/alpha/RNG，基线也须相同迁移；共同初温.0006仅为精修候选，不能静默沿用.5或声称已恢复父学习器。
- **验证**：执行`python3 logs/analysis/twinkle_f1_20261003/recompute.py`，含只改变执行方式的B参照在内，五组10份best/final评估、30回合、1020逐音记录的400Hz TP/FP/FN/F1与原日志全部一致；保存来源散列、CSV/JSON，重复运行产物散列一致，`git diff --check`通过。候选为false_press 1→.25、activation .5→.8、gamma .75→.90和独立速度观测实验，不宣称已提升。没有执行GPU／新回放，也不改旧模型／运行结果。
- **TODO／限制**：后续单因素初筛、从头和多seed复验，精确F1选模／达标合同及新增奖励／观测需显式实施验收；.99／1与低碰撞均尚未达成。理想补音是计数反事实，非已实现控制效果；单seed对照不作统计因果保证。

## 2026-10-02：D组单视角演奏视频

- **要求／参考**：用户要求D组视频，后续指定单视角。读取消融runtime/实际D目录及best/final、统一export_videos/export_video_job工具、上一轮导出脚本和指法校验说明。
- **实施**：D best17000/final19532使用原训练归档，在1364/1457分配内回放；收到单视角要求后复用已有物理录制，通过1364.33/1457.2改为单正面1280×720/40fps/原速有声/原谱指法。输出preview/outputs/twinkle_D_front_20261002及独立HTML/ZIP/manifest，未修改源代码或训练配置。
- **验证／边界**：两段均406帧/10.15秒，严格源/PT/365帧目标匹配、非静音音频、单front相机和完整解码通过，ZIP校验通过，best预览已检查；相关实验文档已更新，diff检查通过。独立回放不等同训练评估轨迹，两个GPU预留保留。

## 2026-10-02：关闭提前定位／独立下压惩罚的三组对照

- **要求／参考**：用户要求三组同配置对照，并明确只关闭false_press_penalty、保留key_press_reward。读取已完成A的完整config/runtime/metrics、原固定归档、train_coupled_40hz.sh/run.sh、train.py、task.py/rewards.py/reward_config.py及已有速度协调器/相关测试。
- **实施／原因**：新增scripts/hpc/run_twinkle_ablation.py，用A原固定源码和seed42串行执行C定位0、D关闭false_press、E双关闭，每组19532×512样本；真实完整config除指定reward字段与定位关闭派生的2297→2247观测/replay字节数外逐项相等。保持按键奖励、.5秒谱面前瞻/精确倒计时、其他学习设置和1000步评估/三类PT。通用默认保持。同步docs/twinkle_ablation_20261002.md及README。
- **验证／启动**：真实入口隔离和奖励回归33项/15子测试通过；首次检查遇config分两次写入，1364.29停止，修正等待完整learner/replay字段，新增就绪检查后6项通过。旧初始化输出留档。18:56:01正式PID2327355独立session启动，C在1364.30，D/E等待；18:58 C完整配置严格校验通过，2247观测/定位0/false_press开启，其余非CLI字段符合基线；已到400步、204800样本，4352次学习更新记录，构图及学习已推进。git diff --check通过。归档SHA及入口/学习/奖励文件一致，所有GPU操作通过原预留入口。
- **后续并行迁移（19:04）**：用户要求利用空闲卡。scontrol/squeue实查8张A100中3张未分配，经submit.sh新增双卡预留1457，EndTime同步原1364的10月3日11:50:28。新增parallel_twinkle_ablation.py，仅停止旧CPU协调器并接管原runtime/锁，C的1364.30及提交子进程连续运行；D移至1457.0，E在C/D任一先完成的双卡接续。新PID2337673心跳及两组实际config验证通过，Slurm内GPU UUID确认两对互不重叠，C1520步/D180步；剩1张空闲。8项调度/配置测试、语法和diff检查通过，同步README/实验文档。原串行状态备份保留，未改变训练合同或释放任何预留。
- **完成结果复查（23:35）**：按用户检查请求读原runtime及A/C/D/E的config/summary/best/metrics；C/D/E全部正常跑满10M/2453376更新，22:59队列完成。final精确F1 .943633/.971421/.963444，均低于A .973910；D短音9/9匹配、E8/9、C0/9，但D/E侧面接触增多。复核9份PT安全加载、config/步骤、完整诊断、预算和排空流水通过，forward均1000，每组只三类PT。更新实验文档/README及输出JSON，diff检查通过；未新跑GPU评估或开训。
- **TODO／边界**：.99联合目标均未达；关闭独立下压惩罚与短音改善同时出现，但总体F1/侧面接触未优于基线，仍需逐音机制及多种子验证。定位关闭含对应观测与未来用指豁免的既有联动。四张卡均继续预留至10月3日11:50:28。

## 2026-10-02：双卡对照视频与F1曲线导出

- **要求／参考**：用户要求导出当前两组视频和F1训练曲线。读取正式runtime/config/metrics/best/summary、统一export_video.sh/export_videos.py/export_video_job.py、export_score_annotations.py、exact_score.py/task_logic.py、旧绘图脚本及导出说明。
- **实施／位置**：为A/B各导出best与final，原速正面+俯视双视角、实际触键音频、原谱指法；通过submit.sh --allocation1364执行，策略始终使用原训练归档。logs/hpc/twinkle_media_20261002保存可复现导出/绘图脚本；F1曲线仅取这两组21×2个原评估点，分别展示400Hz精确micro与40Hz三回合最低音乐F1，标注实际best，并导出PNG/PDF/CSV/JSON。
- **修复／验证**：首次1364.24完成物理录制，但旧指法恢复误用网格目标而拒绝精确时间录制。scripts/export_score_annotations.py改为调用已验证归档的sample_targets，按policy_time生成周期末目标/指法；task/source/score散列及全帧目标一致校验均保留。补测子帧起音、端点释放、踏板、错谱拒绝和旧网格兼容，tests/test_video_export.py共50项通过。A best复用原录制继续渲染，不重跑该策略。
- **交付／边界**：输出preview/outputs/twinkle_comparison_20261002：1364.25–28完成四个正面+斜俯视视频，均1920×540/40fps/406帧/10.15秒，音频非静音、完整解码及365帧指法一致校验通过。B best/final原文件SHA相同，仍分别导出；42个曲线点与原日志对应。总览index.html、PNG/PDF/CSV/JSON与11,756,494字节ZIP生成，ZIP校验通过；不打包PT/场景资产。已查看曲线和视频预览，git diff --check通过。曲线使用原训练评估，视频为检查点独立物理回放，可能因接触浮点误差与训练评估不同；不把回放当成原训练轨迹。不改学习设置、不释放GPU。

## 2026-10-01晚：双卡小星星、Pitch 80%、gamma .75与执行速度对照

- **要求／参考**：用户要求双卡Pitch原范围×.8、换原小星星、gamma .75、确认进一步速度收益并启动对照；后续明确80%及同配置执行速度对照，并要求评估稀疏、仅保留final/best及第1000步首次定期评估forward。读取双卡/run入口、source_runtime/bundle/action_mapping/scene、训练/evaluate/CouplingMetrics/NoteTimingMetrics及当前日志/归档、已有五指共面姿态说明、相关测试和H100交接。指定日志技能本机缺失。
- **实现／位置**：双卡wrapper显式TwinkleTwinkleRousseau、--wrist-pitch-range-scale .8及gamma .75；源码recipe经独立编译进程同时缩小Pitch机械/驱动/目标范围至±.16，manifest/action_contract严格记录，通用默认与旧归档保留。新增evaluation_graph.py及训练开关，只合并只读统计的CUDA提交；构图恢复状态、全部诊断保持，默认关闭。耦合误差索引改为同列stack以支持捕获。音符误差转Python float修复新检查点weights_only读取，不改指标值。新增run_speed_comparison.py按固定归档/相同seed预算顺序执行，失败停止、保留资源。train.py新增save-forward及checkpoint_retention，首次定期评估保存一次forward；双卡入口eval1000/save0，保留初评/终评及best资格，保存前排空流水。同步README、submit注释、双卡说明及本轮文档；历史月光测试显式固定原参数。
- **验证**：CPU56项/5子测试通过；首次Slurm实景捕获发现tuple索引不支持，修复后1364.15的16项/8子测试全部通过，含±.16端点、34音/365帧整曲3回合诊断、graph/eager同指标及错误拒绝。1364.16–19的A1/B1/B2/A2全部完成，短测评估仍200步：中位吞吐1184.44→1196.92样本/秒（+1.05%），每次评估13.2666→12.6887秒（−4.36%），收益有限。稀疏/存档改动后CPU44项/7子测试通过、3CUDA项明确跳过；1364.20的1001步存档验收确认仅三份PT，forward=1000/final=1001且不覆盖，安全加载/config一致；1364.21归档重建forward完整回放通过。23:30:53正式独立协调器PID1239861启动，A在1364.22、B排队；同19532步/seed42/1000步评估。语法检查和git diff --check通过。
- **10月2日结果复查**：按用户检查请求读取正式runtime、两组config/metrics/summary/best及checkpoint.py选择规则；两组各10000384样本/2453376更新正常结束。A/B final精确F1 .973910/.973010，best .975153/.973010，均漏E5–D5–E5三个52–68ms短音，.99未达；final手碰撞1/0子步、键底0/0，但侧面/非末端接触仍存在。B总耗时仅少14.86秒（.20%），不追认短测1.05%为长训收益。CPU安全加载6份PT并严格校验config、步骤和排空流水，forward均1000、仅三类PT。更新本轮文档/README和输出目录JSON核验记录；git diff --check通过。本次未进行新GPU评估/训练；1364仅batch保留至10月3日11:50:28。
- **边界／TODO**：两组都采用当前五指共面源码姿态，不回写旧模型；不改学习预算/奖励/控制物理频率/精度；正式两组同为1000步评估，短测200步口径不追改。H100当前认证失败，本轮先用A100原预留；旧月光队列保持停止。F1/碰撞和速度收益以实测为准。详见docs/twinkle_pitch_speed_20261001.md。

## 2026-10-01：停止所有训练、保留GPU并导出当前视频

- **要求／实际参考**：用户要求导出当前视频、中断所有训练但不释放卡。读取实际Slurm步骤、两边metrics/config/checkpoint、H100两份runtime及协调器，参考scripts/hpc/submit.sh、job.sbatch、eval_from_source_archive.py、eval_from_source.py与render_mjlab_checkpoint.py；指定change-log技能本机缺失。
- **操作／原因**：先停止H100两个协调器，再取消396.6；取消A100 1364.10，保留1364/396两个两卡预留。H100原队列及追加组状态明确改为用户停止/取消，不自动续跑。新增docs/training_stop_video_20261001.md并同步H100交接说明。视频选最新完整11000/6000步检查点，原Slurm分配内调用归档回放。
- **验证／兼容**：确认训练步骤消失、预留RUNNING。首次导出发现评估记录含NumPy标量导致weights_only拒绝；日志脚本生成独立等值副本，两边187张量全部相等、其他值递归相等、原PT保留且新副本安全加载通过。保留166文件源码散列验证；H100用户音频目录补齐libXss并核验SHA，不改系统库。1364.12/396.9完成两个66.95秒、40fps视频，均有实际触键音频、26370物理子步，完整解码通过；H100复制到当前项目后SHA一致。最后只剩两个预留batch，训练进程及协调器均已退出。
- **TODO／边界**：不声称未存档状态已保存，不把新回放视为原训练轨迹逐bit复现；保存端NumPy标量兼容修复另行处理。本轮不改学习设置、不重启训练、不释放或延长资源。

## 2026-10-01：H100追加gamma=.8单变量实验

- **要求／实际参考**：用户在现有R0–R3队列运行时要求再验证gamma=.8。读取`docs/h100_training_20261001.md`、原协调器/远端runtime与396步骤、`scripts/hpc/train_coupled_40hz.sh`、`tests/test_hpc_40hz_entry.py`和`tests/test_moon_f1_h100_contract.py`，本机指定change-log技能缺失。
- **实施／原因**：新增独立等待器`logs/hpc/h100_moon_gamma08_20261001/run_queue.py`，R3成功后接续`R4_gamma08`；复用36d87ac和原归档，唯一学习参数变化为每个40Hz transition的gamma=.8。R0奖励、144音谱、512×10000=5.12M、seed42、预填1350144、完整942464更新不变。原协调器及H100源码不热改；等待器检查前置完成、源SHA、锁及空闲分配，失败停止并保留396。扩展现有单变量CPU合同到.8，更新README和交接表/命令。
- **验证／实际状态**：真实启动链/谱面/单变量CPU测试6项、2子测试通过；协调器CPU模拟19例与py_compile通过，覆盖成功交接、上游失败/死亡/陈旧、源或资源不符及错误gamma拒绝。等待器SHA d4cb367902de776d0b1122e18ce430c3bf10c18640e872705b389a259bc964f1；17:34:11 H100独立PID/session3971706启动，重新连接确认waiting_upstream心跳持续且原R0/396.6运行，未新增占卡步骤。部署状态和精确命令已写入交接说明；新增组尚未采样，不能声称gamma=.8效果已验证。
- **TODO／边界**：待同预算实测R4与同机R0/R3的精确音乐F1、漏错键、碰撞/键底和耗时；质量早停须单独标记。未追加提前定位、多种子或其他奖励变化。

## 2026-10-01：H100接收144音月光R0/R1/R2/R3对比实验

- **要求／参考**：用户交接当前144音谱、奖励与R1同手weight.2/R2姿态上限.02/R3gamma.97，各5.12M/seed42；沿最新交接执行并补同机R0。实际读取Git main8bcf927、H100交接、HPC提交/训练/协调入口、新奖励/动作配置、原谱及GPU/CPU验收源码。
- **同步／资源**：17:10重新登录H100，396仍RUNNING且两卡空闲；新Git目录通过已提交历史bundle从b8bc390 fast-forward至8bcf927，原环境必需依赖核对一致，无需重装。保留现有A100基线和H100两卡预留，未传输凭据或修改系统软件。
- **验收改动**：发现原交接test_exact_timing_gpu/test_coupled_40hz_integration仍默认34音小星星。新增tests/test_moon_f1_h100_contract.py覆盖实际月光144音/2637帧/31动作2297观测、R0/R1/R2/R3单变量及新idle-posture奖励、完整400Hz物理计数与诊断。测试不改训练数值或曲谱；协调器在忽略logs目录按固定归档依次运行validate与四组，失败停止、无自动重试/重复启动。
- **验证／启动状态**：新增谱面/预算测试及相关入口/idle-posture测试在CPU共16项、8子测试通过，未访问GPU；新增GPU测试待H100执行。协调器使用当前月光真实config/metrics及合成预算完成/早停案例做CPU检查，拒绝错误预算/谱SHA/JUnit跳过。H100396.5实际36项、35子测试通过，零跳过(131.12s)；新增144音/2637帧/26370子步及实际姿态罚/精确计数/诊断通过。固定训练归档SHA b0767513ecf58ae57413e7f20e966f63b0579c5ccafe95097093cc8672e0f2b9，源码提交36d87ac(仅新增测试/文档)。nohup独立session/PID3956692已存活并启动首步；17:19:57 R0正式396.6开始，完整config通过，R1/R2/R3顺序等待，尚无训练结论。状态/验收路径写入交接文档并同步README；每组结束校验完整预算或显式标记质量早停，异常停止后续。
- **TODO／边界**：同预算末次/末5次精确音乐micro F1、漏错键、时序/短音、碰撞/键底与耗时需实测；质量早停单独标记。此批只seed42各5.12M，不自动增设R4、seed43/44或10M复验。

## 2026-10-01：提交当前源码及H100对照交接

- **要求**：用户要求提交当前代码以便H100训练，并把方案交接到指定会话01a0f597-5eb7-77f3-a6fa-406e78c3ad33。本次同步当前训练奖励/关节范围/144音谱，以及工作区已完成的导出与约束曲线修改；不提交环境、训练结果或凭据。
- **状态更正**：上一轮nohup协调器启动未存活，queue.log为空、无新Slurm步骤；此前日志“已启动等待队列”的表述不成立。已标记本地manifest为handed_off_to_h100，避免重复运行；A100基线1364.10继续。未修改H100运行状态。
- **交付/参考**：从基线实际config派生R1/R2完整JSON纳入configs/experiments/moon_f1_20261001，R3使用discount .97；tests/test_hpc_40hz_entry新增单参数差异检查。扩展docs/h100_training_20261001交接最新默认、谱面修复、H100同机R0建议、R1/R2/R3各5.12M/seed42/942464更新预算、固定源码/Slurm验收和顺序训练命令、故障停止及评估标准；同步README和双卡说明。参考实际基线config/源码归档、Git同步说明、H100部署记录与当前测试。
- **验证**：当前改动相关CPU测试67项及60子测试通过，2项GPU测试因CUDA屏蔽未执行；新增实验参数测试4项通过。完整训练GPU已在1364.10运行，本次未新启动GPU验收或实验。实际暂存38文件；独立导出298个Python语法、18个shell语法、66个JSON、凭据模式扫描及独立打包/归档审计通过。推送后的远端提交一致性在交付时核验。
- **通信限制**：当前工具可读取/打开目标会话的文档面板，但没有向已有会话发送聊天消息的工具。将交接说明在目标会话打开；不能宣称已向另一代理发消息或唤醒它。H100实际拉取/验收/开训需目标会话接续，不能把推送源码当已开训。

## 2026-10-01 16:47：启动R1/R2/R3单项F1对照自动队列

- **要求/资源**：用户询问多余GPU并授权开始推荐实验。scontrol确认A100节点8/8已分配、1364.10基线继续运行，未发现本地空卡；只读尝试查询H100 101.47.9.106被SSH publickey/password认证拒绝，未读凭据或更改远端，不把历史396保留当作当前可用证据。
- **实际参考/实现**：读取基线实际config/metrics、1364 Slurm分配/步骤、submit.sh/job.sbatch、source归档及源码入口，保持基线不变。在logs/hpc/moon_f1_comparison_20261001生成R1/R2/R3奖励配置、manifest及queue.py；R1只改self_collision weight1→.2，R2只改姿态max .1→.02，R3只改gamma→.97。三组512×10000=5.12M、seed42、预期942464更新。全部复用基线153927...de0ac2源码归档及144音谱，每次启动前核对归档/奖励SHA，默认训练脚本不改。同步README和双卡说明。
- **验证**：在基线冻结源码目录捕获真实wrapper链并解析，各组SAC/reward/control/coupling/layout相对基线恰好指定的一项差异；parameter_validation.json留存。CPU模拟验证三组依次完成、首组失败阻止后两组，未提交模拟作业；SHA检查通过。
- **执行/边界**：nohup协调器已启动，waiting_for_baseline；等1364.10 COMPLETED/0且达到比较预算后通过submit.sh --allocation 1364按顺序执行。任务结束保留分配；异常、分配过期、预算不符停止队列，不自动重试。当前新三组尚未采样，只有队列运行；既有质量早停保留，提前结束会单独标记并暂停。后续以实际manifest/summary记录训练进度与F1，不宣称已提高F1。H100并行启动因认证未完成。

## 2026-10-01：原地合并月光连音并使用保留双卡开训

- **要求/参考**：用户询问能否忽略时序报错开训或连接MIDI，随后明确“直接替换原谱”。读取MidiFile、exact_score、source_runtime/task_bundle、当前训练wrapper及Slurm入口、原proto/CSV/指法报告；不关闭校验。保持上一轮用户指定的闲指姿态、跨掌0/−.25、同手weight1、提前定位.15/no-current-priority及全部其他训练配置。
- **修改**：直接替换generated_v1/clair-de-lune-beginner_fingered.proto中B4原索引130/132、C5原97/99两对同指连音，各保留首音起点/末音终点和原指法。第一次合并B4后构建又报key51，经完整时钟扫描定位第二处C5并一并处理；146→144音。同步fingering.csv和fingering_report.json，原生成统计/成本保留标为历史，当前散列/数量更新。原谱/CSV/报告完整备份在logs/hpc/moon_b4_merge_20261001，附具体时刻、散列和exact_coverage.json。更新wrapper注释、README及双卡说明，入口路径不变。
- **验证**：两个间隔均1.811594ms；同键、同指且除起止外元数据完全一致，其余音符不变。原CSV顺序与proto不同，按音高/起点定位同步，未按proto索引修改CSV。CPU精确源构建通过144/144音，无同键/同指重叠、无未覆盖音或缺少松键采样，音乐2587帧/含准备释放2637帧不变。后续运行状态记录于本节。
- **启动**：已通过 `submit.sh --allocation 1364 train` 启动Slurm步骤1364.10，输出父目录training_mjlab/outputs/hpc/moon_idle_posture_20261001；仍使用保留两卡，任务结束不释放分配。18项CPU测试及18项子测试、CSV/proto/手指计数一致性和shell/diff检查通过。实际config已确认31/2297、512环境、10,000,384预算、1,350,144预填、144音精确覆盖、60–80°及最新三项奖励，初始评估运行中。实际输出moon_idle_posture_20261001-20261001-154749-604163，源码归档SHA256为153927fdcde9598ebe60b4246410513eb1a4c3771f6af4d48d452719a7de0ac2。
- **边界**：合并后这两处被评为持续音，不再要求二次击键；这是用户授权的谱面修改。候选指法、收紧从动范围可达性和长期F1仍待训练，未改物理频率/模型/奖励以绕过报错。

## 2026-10-01：用户审阅后指定闲指姿态与碰撞负罚为双卡默认

- **要求**：用户在B参数说明后明确改为跨掌无碰撞0/控制周期内有碰撞−.25、同手weight=1/每去重body对−.5、gesture仅闲指姿态且每指最多−.01/十指最多−.1每步；当前指法优先关闭、当前与未来共同平均，保留提前定位。覆盖上一条任务的B奖励默认，未授权开训。
- **参考/修改位置**：读取双卡wrapper、submit注释、coupled_40hz_idle_posture_20261001.json、mjlab_env/{rewards,fingering_experiments}.py、入口/姿态/时序测试和既有说明。切换train_coupled_40hz.sh的reward-config到现有闲指配置，保持0.15秒提前定位/no-current-priority及其他参数；更新submit注释、入口合同测试和GPU集成奖励预期（姿态不再乘dt），同步README及双卡配置说明。既有奖励实现已满足要求，无需改公式。
- **分值核对**：跨掌读取palm_collision_any周期并集，原−.5乘dt/.05=.5得到−.25；同手去重body对每对−1×weight1×.5=−.5。gesture为−.01×Σ闲指[mse/(mse+.25²)]，固定十指分母，当前及0.15秒内未来用指豁免、weight1、最终每步单位；无几何成本和十倍倍率。当前与未来指法分按共同分母平均，未来单指仍按已有最高.5接近分，不再受(1−当前分)抑制。
- **验证**：25项CPU测试和36项子测试通过，覆盖实际脚本链参数、跨掌周期并集、同手pair分值、闲指固定分母/当前未来豁免、姿态dt独立与上限、weight≤1、时序和旧合同兼容；shell语法和git diff --check通过。GPU集成未运行，未提交训练或同步远端。
- **保留/TODO**：《月光》、最新关节范围、40/400Hz、60–80°、1m/s、B的SAC参数及约10M预算保持。上一轮发现的《月光》B4间隔1.812ms导致精确时序构建失败仍未解决，未改变谱面或绕过校验；新奖励长期F1/碰撞效果待验。旧B归档和配置比对文件保留为历史快照。

## 2026-10-01：双卡默认采用B方案，保留当前曲目与范围，等待用户决定开训

- **要求/范围**：用户要求把B配置写入当前训练脚本并说明双卡参数，明确由用户决定是否开训；随后确认保留《月光》和当前关节范围，只采用B奖励/时序/学习参数。保留当前10,000,384预算，未开启、提交或排队任何训练，也未同步远端。
- **实际参考**：读取B_exact_timing-20261001-123516-570119/config.json、run.sh/run_tsensei_a32.sh、scripts/hpc/{train_coupled_40hz.sh,submit.sh,job.sbatch}、两套奖励JSON、direct_joint_v1、rl_torch/{train,sac,checkpoint}.py、mjlab_env/{source_runtime,exact_score,task_bundle}.py及当前MIDI、相关入口/时间/奖励测试。
- **修改**：双卡wrapper恢复coupled_40hz_gesture10_20260930.json，显式关闭staged和upcoming permission。B的geometry gesture乘10/weight1、跨掌+.25/0、同手weight.2（40Hz每对-.1）恢复；保留40/400Hz、60–80°、TX/TY1m/s、三LR3e-4、batch1024/.25、当前曲目及当前范围。更新submit注释、README和双卡说明；入口测试恢复B合同，闲指模式校验改为显式选自身JSON，GPU集成的奖励预期恢复所有项dt缩放（本次未执行GPU集成）。其他工作区修改保留。
- **验证**：CPU捕获实际wrapper链参数，经JSON规范化后SAC、reward_config、finger_coupling、control_limits、finger_clearance与B逐项完全相同。25项CPU测试及36项子测试通过；shell语法及diff检查通过。当前曲目旧grid模式CPU构建通过：146音/2637帧，整曲预填1,350,144条，计划第1,350,145条进入学习、全预算约2,162,560次更新。检查配置和比较结果保存在logs/hpc/b_defaults_review_20261001/。
- **未解决项/实际失败**：当前B精确模式CPU构建被physical_coverage拒绝：Repeated key 50 has no release sample at physics resolution。原proto中B4/MIDI71前音结束56.085130906s、后音开始56.086942500s，间隔1.811594ms，没有400Hz松键采样点。未关闭校验、未修改原MIDI或伪称可开训；需确定该处重击/松键目标表达。仅grid检查通过不能验证exact合同。新曲目/收紧从动范围下60–80°可达性、F1及吞吐未测；历史B小星星3M结果不能直接推广。

## 2026-10-01：原节奏机械臂速度／加速度约束导出

- **要求／参考**：用户要求SDK速度、加速度范围内解IK，并明确保持原节奏、误差很小且不累积。读取原视频1359记录及冻结PT/肩座/连接件、`preview_trajectory.py`、`preview_arm_ik.py`、统一导出与补导脚本、SDK快照及`FxRobot.cpp`/`AxisPln.cpp`、OSQP官方Python接口说明。旧差分IK只有角限和数值步长，未结合dt约束速度或加速度。
- **修改／原因**：新增`constrained_arm_motion.py`固定时间三次B样条整段优化，以每时刻绝对FK残差避免积分漂移；位置、速度、加速度系数界覆盖连续区间，起终点保留原关节角且v/a为零，SDK约束留2%数值余量。默认位置误差0.1 mm／姿态0.5°，独立复核；未收敛／超限则停止。初试0.001 rad姿态未通过，不宣称达成；整体减速和超时／超限候选未交付。只修改离线机械臂叠加轨迹，不改手部／训练／原视频／硬件。
- **接入**：CLI默认`--arm-ik constrained`，可显式raw诊断；提供误差参数，约束臂视频保持speed1。视频／补导图共用保存样条，臂速度／加速度图采用解析导数、SDK红线，手部仍用原数据差分。增加1600 Hz实际渲染插值FK审计、误差CSV/图及连续证书。独立Composer环境增加`osqp==1.1.3`并锁入可选requirements。Slurm分配步骤使用各自工作目录，避免重复导出冲突。README和导出说明同步，详见`docs/arm_motion_constraints_20261001.md`。
- **CPU验证**：78项相关测试通过。原9.125秒／400 Hz轨迹13/14关节加速度超限，最坏12.86倍；新轨迹14关节连续角度／速度／加速度均合格。14,601点帧间最大位置误差右0.060294 mm、左0.093272 mm，姿态0.4426°／0.4903°，末帧位置误差均<0.00001 mm。最高连续速度／加速度系数界分别为SDK的52.2441%／98.0327%；未用绘图差分低估峰值。
- **GPU交付**：确认1364训练队列已结束、仅保留空闲分配后，经`submit.sh --allocation 1364 export`在步骤1364.9执行，锁避免与其他任务重叠；COMPLETED/0，耗时1分38秒，未释放保留卡。新目录`preview/outputs/export_best_constrained_20261001_151434`：1280×720／40FPS／406帧／10.15秒（含末帧显示停留），H264+AAC原节奏有声、正交琴宽铺满、365帧原指法恢复校验通过。全片解码通过；14臂＋46手关节的128 PNG/128 PDF及误差图导出，SDK标线目视检查通过。287交付文件SHA和288 ZIP成员校验通过，0 STL/XML；原视频SHA、原始IK初值与全部手部曲线数值保持。新视频离线补导的所有臂关节数组与作业内导出逐项相同，验收在`logs/arm_motion_planning_20261001/delivery_validation.json`。
- **边界／TODO**：SDK默认值不是实物负载降额值；没有jerk／力矩／避障／硬件执行验收。TJ-ARM-02/03/04保留，手关节确认限值仍缺，不冒用M6值；较严容差或其他曲目可能被明确拒绝。连续运动限值有系数证书，位姿误差是离散／加密FK验收，未声称连续位姿误差证明。

## 2026-10-01 15:05：1364四组最终结果检查

- **要求/参考**：用户要求检查当前训练结果。读取四组实际config/metrics/best/summary、队列manifest和协调日志、Slurm步骤、D实际command与源码SHA、上一轮统计脚本和note timing定义；遵循AGENTS，未运行新仿真或GPU推理。
- **结果**：四组各3,000,320样本、703,360更新、replay插入相同，预算正常耗尽；队列14:41:38全部完成，1364只剩48小时保留进程。A/B/C/D最终精确F1 .656314/.677958/.247906/.593851，匹配13/15/4/14音，手碰撞240.33/17.67/22.67/202.33子步/回合，短音均0/3、键底均0。B最好但仍晚起62.18ms/早松26.18ms；C比B低43.01百分点，D低8.41百分点。D best .605934/91.67碰撞，final .593851/202.33，即使按best也没有超过B；末5次B/C/D F1均值 .648016/.248552/.543515。
- **验证/交付**：`review_20261001_final/`冻结四组原日志/配置/选模/总结及散列，CPU脚本重算各评估精确micro、验证每回合5230目标采样、诊断完整和数值有限，输出JSON/CSV/PNG并目视检查。B/C只差分阶段，B/D只差请求许可；控制/SAC/source相同，D归档仍0b982c...ca3261。更新既有时序文档与README，其他用户改动保留。
- **结论/边界**：保留B作后续候选基线，当前证据不支持合并C/D；未启新训或释放用户保留卡。3回合为同一seed42策略重复，3M初筛未证明最终收敛；未匹配可能含迟到，不能全部等同从未触发。B均值结束偏差含一例D4提前455ms断开，中位早10.21ms。工作区后续新闲指奖励/碰撞/关节范围不在本批实验中，效果需另测；V32-TIMING-03保留。

## 2026-10-01 14:11：1364前三组最终结果与D进度核查

- **要求与参考**：用户要求检查当前训练结果。读取A/B/C/D实际config、metrics、best、summary，协调器状态/日志、Slurm实际步骤1364.8的command和归档SHA，以及既有时序报告和分阶段奖励源码。未执行新GPU推理、仿真或训练。
- **结果**：A/B/C均完成3,000,320样本、703,360更新，原MIDI精确F1为.656314/.677958/.247906，手碰撞240.33/17.67/22.67子步/回合，匹配13/15/4个音（每回合34音），三个短音均0/3、键底均0。B比A高2.16百分点；C比B低43.01百分点，best .249624、末5次均值.248552，已形成低平台，不能因只学会4音而将较小起止误差当整曲改善。D快照890,880样本，最近819,200评估F1=0；B同预算也为0，继续原测试。
- **验证与交付**：`review_20261001_1410/`保存14:11:21输入快照/SHA、离线analyze.py、JSON/CSV及四组曲线。重算每次共同micro，确认每回合5230目标采样、诊断有效、训练标量有限，目视检查曲线。B/C奖励配置只差分阶段开关；实际D步骤SHA仍为0b982c...ca3261。同步既有时序说明和README。
- **范围/TODO**：本批仍是60–80°/geometry10倍/TXTY1m/s的原冻结合同，不代表工作区后来闲指奖励、碰撞或关节范围修改效果。单种子3M初筛，D尚未完成，晚弹/早松/短音仍待解决；V32-TIMING-03保留。保持现有队列，不以本次检查启动额外实验。

## 2026-10-01：手部物理目标范围与耦合从动范围调整

- **要求／更正**：用户要求按核对表修改左右手物理目标，并明确PIP/DIP不由策略输出时也使用新范围；随后纠正拇指前两项下限为0.05而非0.5 rad。最终Rot/Rot2 `[0.05,1.57]`，Thumb MCP/PIP `[0.05,1.2]`，四指MCP `[0,1.2]`、PIP `[0,0.9]`、DIP `[0,0.8] rad`，双腕TY `[-0.05,0.15] m`，Swing仍±0.35 rad，TX/Pitch/踏板保持。
- **实际参考**：`AGENTS.md`、`configs/actions/direct_joint_v1.json`、`models/hands/tsensei_hand_constants.py`、`mjlab_env/{action_mapping,scene,finger_coupling,source_runtime,control_limits}.py`、`tests/test_{action_mapping,finger_coupling,action_checkpoint_contract,action_source_fingerprint,control_limits,source_runtime}.py`、既有动作映射说明和README。本机指定change-log技能不存在。
- **修改位置与原因**：更新direct_joint_v1的51通道源合同；耦合器已有从PianoScene接收完整物理边界的路径，自动将八指follower范围改为PIP0–0.9／DIP0–0.8，无须改求解器。TY旧模型下限0不允许负目标，故同步扩展虚拟TY关节／驱动器范围和源码配方初始TY校验。更新现有映射断言及耦合回归，检查从动边界与机械限位分离；同步README和`docs/action_mapping_direct_joint_v1_20260924.md`。
- **验证**：Slurm1364.7 CPU-only（gres=none、MUJOCO_GL=disable）29项及45子测试通过，3项GPU测试未运行；覆盖映射、八指从动FK/边界/不可达诊断、机械范围保持、完整源重建及动作/源码兼容拒绝。日志`logs/hpc/action_ranges_20261001_cpu_no_render.out`。git diff --check通过。1368单卡验证在前置探测因入口要求两卡退出，未执行测试；1369双卡资源排队后主动取消。1364.6首次CPU步骤16项及35子测试通过、13项因缺EGL库在构建时出错，随后禁用渲染完成上述回归；未占用在训GPU。
- **边界／TODO**：策略输出区间、origin/initial、PD、限速、耦合60–80°配置、奖励和在训进程不改；当前Rot2初始0.4 rad在最终0.05–1.57范围内。手指URDF机械限位保持，TY模型范围改变并受源SHA/场景合同保护；旧checkpoint须用原归档回放。本次按用户要求改完返回完整表确认，未开训、未同步远端或提交Git。TODO(V32-START-01/V32-COUPLING)：收紧目标后的末节可达性、实际接触及音乐F1待动态验证，目标范围不等于实际关节角硬约束。

## 2026-10-01：H100两卡预留、Git源码同步与独立环境开训

- **要求／问题**：用户要求先预留两张H100，再同步当前项目环境并启动训练；H100原有目录为9/28旧源码且依赖未完成，当前GitHub main为`b8bc3906d7094b8f80f5bd98be294913f264a5f5`。
- **实际参考**：`AGENTS.md`、`docs/git_sync.md`、HPC迁移/环境说明、`submit.sh`、`reserve.sbatch`、`job.sbatch`、三套setup脚本、`package_a32.py`、`train_coupled_40hz.sh`、Torch训练/评估/流水源码及6组相关验收测试。本机指定change-log技能不存在，继续按本项目日志约定执行。
- **执行／修改位置**：通过仅含已提交main历史的Git bundle同步到H100新目录`~/projects/tsensei_v3_2_pianist_git`，旧目录保留；Slurm预留396在10月1日13:04:35分配两卡/16CPU/128GiB，到10月3日13:04:35。hpc下载缓存/工具链及训练输出放在/data/share/houj/tsensei_h100，用户子目录0700。新增`docs/h100_training_20261001.md`并同步README，协调器/基准/审计脚本保存在忽略的logs目录，不改训练源码/参数，不传输凭据、不修改系统驱动。
- **环境验证**：TUNA403后用官方PyPI按原固定版本安装；MJLab wheel及CUDA/EGL下载件复用并重新校验SHA，未复制.venv。Python3.12.14，Composer84包/learner133包，必需训练依赖和requirements SHA与A100一致；A100额外5个绘图包作为可选差异保留。nvcc12.8.93及EGL通过，日志在远端logs/hpc/deployment-20261001。
- **GPU验证**：396.0确认两张H100 80GB HBM3、sm_90，Torch矩阵/CUDA图、双向P2P、Warp内核/互操作通过；396.1共35项测试和8子测试通过，零跳过，包含真正双卡、Adam、CUDA图/RNG、批量评估与精确时序/40-400Hz物理；既有MULTICCD和Torch弃用提示未改变物理合同。
- **短训／正式状态**：396.2短训420步/215,040样本/7,040次更新、2M replay实分配、流水排空、三回合诊断全覆盖通过；采样/学习卡2秒监测显存峰值5,823/36,411MiB。396.3独立源码重建和final.pt加载评估通过。13:20正式396.4从头启动，13:23用户改为只需确认H100可训练、奖励确认后再开对比；立即取消396.4并保留396两卡。Slurm确认CANCELLED，最后记录760步/389,120样本/50,432更新；协调器退出137为主动取消，不是训练故障，且不会自动重启。输出保留于`/data/share/houj/tsensei_h100/outputs/h100-20261001-allocation396/formal/run-20261001-132000-608784`。同一源码归档SHA `4304062658aeb72cb2f021376d7630459f7c43110b17459989951e478ad0ed37`；短训低F1不作收敛声明。
- **TODO／边界**：正式训练保持当前31动作/2297观测、40/400Hz、512×19532约10M样本、batch1024/.25更新比及三LR3e-4；从头训练。待用户确认奖励后才开启新的对比实验；长期F1/碰撞和相对A100吞吐待实测；396预留在子任务完成后继续保留到期。H100长期GitHub只读认证尚未配置，此次bundle同步不等于git pull已认证；A100其他运行及并行文档/视频修改不合入本次源码。

## 2026-10-01：1364训练中期效果与下压干预检查

- **13:55再次检查**：新增`review_20261001_1355/`原始快照/散列、离线重算JSON/CSV和三组图，同步既有报告及README。A/B3,000,320均正常完成、703,360更新、best=final；共同精确F1 .656314/.677958（最终+2.16百分点）、碰撞240.33/17.67（−92.65%）、短音0/3。共同9音起音60.65/62.59ms、结束−4.31/−11.99ms，不沿用先前中途起音改善结论。C已2,764,800，F1 .248158，对比同预算B .649393落后40.12百分点；best .249624，约1.8M后平台，只匹配4左手音。D待运行。实际冻结配置/归档与原协议一致，后续工作区奖励/MCP编辑未进入本批；未发现非有限训练值。顶层manifest状态滞后于实际1364.5/C及协调日志，采用summary/metrics/实际Slurm交叉核验，未覆盖协调器状态，来源待查。重算各评估TP+FN=5230×回合数、F1公式和诊断覆盖通过，图表目视检查；未启动/停止作业或更改训练参数。
- **要求/参考**：用户要求检查当前训练效果。读取1364协调manifest/Slurm状态、A/B实际config/metrics/best/summary、下压干预report/cases、旧1103同预算metrics，以及当前`note_timing_metrics.py`的匹配和micro定义、已有时序修复说明；遵循AGENTS，未执行新仿真/GPU或修改在训参数。
- **交付**：`training_mjlab/outputs/note_timing_tests_20261001_1364/review_20261001_1315/`保存13:13:54日志/配置快照及SHA、离线脚本、CSV、JSON、F1/碰撞图；补充既有修复文档及README。脚本重算每次精确micro、验证每回合5230目标采样、三个诊断覆盖有效及训练数值有限；配对共同成功音时核对音高/目标起点，图表已目视检查。
- **实测**：A3,000,320完成且best=final，原MIDI精确F1 .656314；B快照2,785,280采样，最近评估2,764,800，C/D等待。同2,764,800预算A/B精确F1 .586608/.649393（+6.28百分点）、手碰撞316.33/37.33（−88.2%）子步/回合、键底0/0、短音匹配0/3。B匹配音晚起68.94ms、早松20.50ms；共同7音起音68.93→61.07ms、结束+.95→−17.02ms。三回合合计匹配30→40/102，未匹配包括超过150ms才弹的音，不能等同完全没按。较早2.56M差距+14.84百分点随后收窄，采用最新共同预算，未挑最大差值。
- **干预结果**：旧60°31成功音0/25/50ms提前的起音MAE32.11/14.01/15.48ms，50ms造成20音提前；三个旧短音只提前仍0/3，替换完整成功屈曲三元组后3/3，但松键仍晚。34重复基线逐位相同，CPU与原GPU激活差异0；局部重叠窗口总数不是整曲F1/碰撞。新60–80°训练效果与旧60°干预分开解释。
- **TODO**：A/B仍处3M初筛且B未结束，不能据此判断最终收敛；C分阶段、D请求许可继续原队列。保留V32-TIMING-03，多种子、完整预算和实际角度/物理安全效果待验证。未更改模型、奖励、腕速、耦合或队列，未上传分析结果到外部服务。

## 2026-10-01：接入 GitHub A3.2 训练源码同步

- **要求／问题**：用户提供 `ONKIALL26/tsensei_v3_2_pianist` 仓库并要求代为上传，当前工作目录没有 Git 历史，包含环境、训练输出、缓存和另行克隆的硬件 SDK。
- **实际参考**：`AGENTS.md`、`scripts/hpc/package_a32.py` 的源码白名单/排除规则、`scripts/hpc/submit.sh`、`job.sbatch`、`setup_envs.sh`、`train_coupled_40hz.sh`、`pyproject.toml` 和现有 README。可选 change-log 技能指定路径不存在，按本项目约定记录。
- **修改位置与原因**：新增 `.gitignore` 排除环境/产物/凭据及本机覆盖配置，新增 `.gitattributes` 保留模型二进制并确保 Linux 脚本 LF；新增 `docs/git_sync.md` 并链接 README。首次暂存以现有 A3.2 轻量包清单为依据，不上传 M6/安装件资产、完整在线应用和第三方硬件 SDK。训练源码、奖励及参数保持现场版本。
- **验证**：首次暂存572文件，约40.10 MiB；按实际暂存内容导出到临时独立目录，294个Python文件语法、18个shell/Slurm脚本语法、62个JSON解析及A3.2打包/归档审计全部通过，压缩包约13.86 MiB；常见凭据模式扫描0命中，非暂存/非忽略文件0。全量首次diff存在历史CSV/URDF的CRLF及旧代码空白告警，保留原模型和历史文件字节，不作为本次行为改动处理。未执行GPU工作或训练。
- **认证／状态**：用户已添加仓库专用Deploy key，`git ls-remote --symref origin`成功，远端为空；私钥只在用户 `.ssh` 目录，仓库本地配置使用该密钥。Git提交身份仅在本仓库设置为用户提供的GitHub账号ONKIALL26及其noreply地址，不改全局配置。首次源码提交 `c3cc462cdd20` 已成功推送至 `origin/main`，`git ls-remote` 核验远端提交号与本地一致；随后仅补充本段交付记录。
- **TODO／边界**：H100 环境安装/验收、服务器配置自动选择及续训迁移不在本次提交中实施；后续模型/F1/碰撞标定边界保持。

## 2026-10-01：精确音符时序、下压干预与四组分项训练

- **问题／授权**：用户批准修复原MIDI起止时序、同状态0/25/50ms提前下压与三个短音幅度测试、独立分阶段指法、独立即将用指请求目标许可，并要求排队训练。随后明确共同使用60°–80°耦合、gesture得分10倍、TX/TY1m/s，复用已申请的1364两卡48小时分配，不等待3M训练结束。
- **实际参考**：本目录AGENTS、`docs/note_timing_diagnosis_20261001.md`、1103完整配置及归档、1353原400Hz录制和MJB、源Twinkle MIDI；`mjlab_env/{source_runtime,task_bundle,task,task_logic,scene,rewards,performance_options,finger_clearance}.py`、`rl_torch/train.py`、源码训练和渲染入口、`run.sh`、`scripts/hpc/{submit.sh,job.sbatch,train_coupled_40hz.sh,reserve.sbatch,package_a32.py}`及相关回归。可选日志技能不存在，按AGENTS维护本记录。
- **时序修改**：新增`exact_score.py`，导出原始note/pedal事件，半开区间起止在每个物理积分后时刻采样；键正项/触发/错键/false_press/踏板和子步TP/FP/FN使用相同时钟。增加176维下一on/off秒倒计时，完整观测2121→2297，动作仍31；精确目标端点与起点分开，准备/音乐/尾段按采样时刻划分。40Hz奖励dt归一、gamma、温度和原非键奖励频率保持。只通过奖励JSON开启时也同步源构建标志；Composer导出不引入Torch依赖。旧配置缺字段和旧checkpoint合同保留，新任务SHA/观测/奖励合同严格隔离。
- **独立选项／默认**：新增`--upcoming-press-permission-time`（默认0、最多150ms），只放宽正确即将用指的目标键末节请求几何得分，近端/其他键/键底/实际风险保持；要求exact+足够提前窗口及关闭两项物理投影。分阶段复用原独立开关，默认关闭。HPC默认改为精确时序、八指60°–80°新JSON及gesture10倍JSON，weight=1/≤1校验保留；普通run.sh单卡频率与默认时序保持。
- **诊断／评估**：新增`strike_timing_diagnostic.py`，34音×三提前量+3短音×三供体幅度命令，另重复34基线；从相同录制状态克隆完整CPU积分状态，在Slurm内测真实接触/激活/起止/错音/碰撞，单列CPU相对原GPU漂移。新增`note_timing_metrics.py`，所有组按原MIDI计`exact_original_music_micro_f1`及逐音误差/漏音/短音召回；保留原三种指标，不把目标口径变化当作提升。旧记录重算校准通过，150ms匹配不宣称标准音符F1。
- **队列／源码**：新增`queue_note_timing_tests.py`、`run_timing_allocation_queue.py`，冻结统一源码SHA后，在1364分配内依次运行验收、下压诊断、功能短训及A旧量化/B精确/C精确+分阶段/D精确+50ms请求许可；四组均seed42/512环境/3,000,320样本，其他参数相同。后台协调器仅查询Slurm，RUNNING后经submit.sh/srun执行；失败阻止后续，锁避免重复，不自动释放保留分配。submit/job支持固定归档与步骤日志，普通独立提交仍可用。README、双卡/范围/保留说明及`docs/note_timing_repair_tests_20261001.md`同步。
- **CPU验证**：核心/时序/奖励/入口/源码等137项通过、1项CUDA测试跳过、73子测试；额外指标/批量评估/队列33项通过、12子测试；源/回放兼容另62项通过；新队列/分配/60°–80°及10倍入口21项通过、19子测试。上述组有重叠，不合计为唯一测试数量。实际Composer源码导出365帧/34精确音/5230目标子步通过；旧量化5590。干预dry-run核对录制、配置、时间轴和MJB散列通过。Python/Shell语法通过。旧finger_clearance的3个fixture失败已用1103旧类复现，新增许可9项均通过，历史fixture另列待办。
- **末次边界修复**：精确模式40Hz评估/视频改按终点分段，实际音符事件按自身时间分段；39准备/315音乐/11尾段端点与365执行区间一致，尾段完成改以完整步数判断，旧模式保留。针对性33项通过、2项GPU跳过、5子测试。
- **用户再次要求开启训练后的实测核查**：北京时间12:15核验1364已于11:50:28获得两卡、截止10月3日11:50:28。1364.0真实验收83项/58子测试通过；1364.1下压干预和1364.2功能短训0退出，短训13,440样本。1364.3的A组从11:54持续训练，已记录3,040向量步/1,556,480样本/342,272更新，B/C/D由原协调器顺序等待。保持现有运行；TODO(V32-TIMING-01)初次GPU/学习器验收通过，完整对照和干预效果分析仍待完成。更新本轮说明，未重复启动、改动冻结源码或运行参数。
- **实际排队／未解决**：北京时间11:48已挂接1364队列，输出`training_mjlab/outputs/note_timing_tests_20261001_1364/manifest.json`，后台协调PID487273；11:49核验1364仍PENDING/Resources、7步骤waiting。统一源码566文件审计通过，SHA256=`0b982c129b39e1c4c128968bad87d0bd4f7c1ca18f17baf99eae6f4f36ca3261`，归档内参数及端点评估修复已复核。未重复申请两卡，GPU验收与F1收益尚未得到结果。TODO(V32-TIMING-01/02/03)：真实GPU接口/学习验收、配对提前与幅度效果、四组同口径F1/时序/碰撞/吞吐及后续多种子验证。诊断使用原60°命令，不宣称等同新区间求解器；本轮训练全部显式60°–80°，A组也重训。既有镜像右手、暂定动力学与硬件标定限制保持。

## 2026-10-01：两张GPU保留48小时申请

- **要求／问题**：用户明确要求锁住2张GPU，并确认分配后保留48小时。当前8张卡均由其他用户作业分配，houj无管理员reservation权限，按普通Slurm队列申请。
- **实际参考**：项目AGENTS、`scripts/hpc/submit.sh`、`job.sbatch`；Slurm分区最大7天、当前作业/权限及`srun --help`。未调用其他代理或变更他人作业。
- **修改位置与原因**：新增`reserve.sbatch`，申请2GPU/16CPU/128GB/48小时，用等待进程保持分配，不运行GPU负载。`submit.sh reserve`显式提交48小时；新增`--allocation JOB_ID`，校验本人RUNNING保留作业后通过srun复用，flock防止本入口重复并行任务。`job.sbatch`在复用模式按Slurm步骤隔离源码/监控/日志，普通提交与source-archive/dependency逻辑保持。同步README和`docs/gpu_allocation_48h_20261001.md`。
- **验证／提交**：三个脚本bash语法通过；新增CPU调度桩测试5项、7子测试通过，涵盖48小时提交、srun复用、参数/路径、访问条件和旧提交行为。真实申请**1364/TsenseI-reserve2**已被Slurm接受，TimeLimit=2-00:00:00、ReqTRES含2GPU/16CPU/128GB；截至核查PENDING/Resources，尚未分配GPU。
- **TODO／边界**：分配内真实GPU执行需等获得资源后验收；不能把排队申请称为已占用。48小时从实际开始计算，任务运行不延长截止时间；可用scancel1364提前释放。本次未启动训练、取消或迁移其他任务。

## 2026-10-01：1103最佳回放晚弹、松键与短音漏弹审计

- **问题／要求**：用户根据34原音／31实际音的钢琴卷帘图，要求彻底分析为什么晚弹、抬迟。
- **实际参考**：AGENTS、1103归档源码及完整config／best／score_coverage；1353原始rollout、400Hz audio_substeps／physics_trace、编译model.mjb和音频事件；截图原导出目录的逐音CSV／比较脚本；归档music/piano_roll、task/task_logic/scene、anticipation、fingering_experiments、finger_coupling、finger_clearance、rewards、preview_audio及流水采样代码。可选日志技能指定路径不存在，直接执行项目日志要求。
- **修改位置与原因**：新增 `docs/note_timing_diagnosis_20261001.md` 及README入口；新增 `training_mjlab/outputs/timing_diagnosis_20261001/` 的三个CPU分析脚本、逐音／命令／奖励CSV、汇总JSON、被动FK数据和两张图。保留实际事件边沿与控制区间目标两种参考，避免把量化松键偏差当成统一动作延迟。未修改训练、奖励、模型、MIDI或启动脚本。
- **实测结果**：31/31成功音相对原MIDI平均晚起32.110ms，相对目标晚45.081ms；28/31相对原谱晚松，平均11.210ms，但相对训练目标25早／2准／4晚，均值−2.903ms。量化目标结束延长14.113ms解释上述均值差；两处短交替音额外重叠25ms。400Hz TP4888/FN702/FP0，micro F1 .93300248，FN中起音528、释放74、整音100；这是采样计数而非连续缺失时长。25个成功耦合音主要MCP跃增全在起音帧，24个在结束前25ms释放；三个短音MCP请求较弱、实体键程接近0，末节到目标键最小间隙3.812／3.666／3.250mm，原始单指分仍.886／.969／.891。
- **公式／排除证据**：实际两个目标键时，单个完全漏按键项损失.124985/帧，额外完全按键损失.325/帧；所有激活位由真实86.89%阈值计算。提前定位目标为悬停5mm、staged关闭；gesture潜在提前下压冲突列为待验证机制，未把实际小额罚认作完整因果。当前指法优先确实关闭；没有多一帧观察／音频延迟，全部八MCP ctrl=action×.65，无MCP额外限速或截断。
- **验证／范围**：三个脚本CPU运行成功；另两分析任务独立核对事件、数组端点、FN、公式、模型geom类型及FK距离。仅对已保存qpos做运动学，无积分、无GPU/Slurm提交或训练。旧视频交付目录在后段已不可用，改用与原config SHA一致的训练config及已核验逐音表，保留旧CSV SHA与来源说明，并再次以物理激活边沿核验31组起止。CPU复算首次导入旧music库缺note_seq，改为读取已与原MIDI核对的score_coverage原始起止及指法，避免安装额外依赖，最终成功。未用新回放替换旧数据。原键程软限位过冲保留在图中，不裁剪为100%。
- **未解决TODO**：V32-TIMING-01连续事件时间合同；V32-TIMING-02提前命令／短音幅度动态干预；V32-TIMING-03奖励、速度观测、耦合范围的分项长期对照。单条轨迹不能分配gesture／耦合／折扣的因果贡献，也不验证新50°–80°候选；既有标定与物理安全边界保持。

## 2026-10-01：四指耦合改为每指独立50°–80°许可区间

- **要求／问题**：用户要求原末节尽量60°改为50°–80°均许可，四指范围分别独立。实现左右食/中/无名/小指共8项独立配置，拇指沿原控制。
- **实际参考**：`mjlab_env/finger_coupling.py`、`scene.py`、`task.py`；`rl_torch/train.py`、`checkpoint.py`、`coupling_evaluation.py`；`run.sh`、HPC训练/提交/归档入口；耦合CPU/GPU与检查点测试、历史耦合说明。项目约定提及的本机change-log技能路径不存在，仍按项目要求维护本日志。
- **修改位置与原因**：`finger_coupling.py`新增`distal_angle_range_v1`，验证完整8项范围，分别求两边界DIP解并纳入可行区间内部/方向极值候选；区间内误差0，沿用origin位移偏好和有界回退。旧单角度分支与旧合同保持。`train.py`新增`--finger-coupling-angle-ranges`，启用耦合后默认50–80；显式`--finger-coupling-angle`可覆盖为旧控制。新增`configs/finger_coupling/angle_ranges_50_80.json`并接入双卡默认。同步入口注释、README、HPC说明及`docs/finger_coupling_angle_ranges_20261001.md`。
- **验证**：CPU定向28项、51子测试通过；覆盖原生FK、八指独立、内部姿态保留、越界/不可达、CLI及新旧检查点合同；3个脚本bash语法通过。修改前即存在HPC奖励测试期待gesture10但当前脚本实际原强度的失败，修改后仍为同一失败；未改奖励或该断言，定向通过数排除此项。Slurm1357已提交单项40/400Hz短验收，截至记录因Resources等待，GPU结果未获得。
- **TODO／边界**：实际触键夹角、F1、碰撞和512环境吞吐待验证；许可范围约束从动目标，不是硬改实际qpos。保留V32-COUPLING-01/02及V32-07标定限制。本次无新长期训练，未修改旧实验或其他训练参数；既有gesture脚本/测试不一致留待相关任务处理。

## 2026-09-30：当前双臂J1–J7、法兰和手根坐标系出图

- **要求／问题**：用户要求当前机械臂J1–J7及安装法兰、手坐标系的适当视角图。J1/J2与J5/J6/J7各自原点重合，全轴叠画难辨，采用总览加逐关节放大，并另画F/H双侧近景。
- **实际参考**：两侧`preview/assets/tj_m6/*/arm.urdf`、`preview_mount.json`、连接件配置、双手URDF/初态常量；`preview_arm_ik.py`、`preview_connector.py`、`preview_frames.py`、`preview_pose.py`和新安装件说明。
- **修改位置与原因**：新增`preview/scripts/draw_arm_coordinate_frames.py`，从真实编译网格/初始IK读取随动Link1–7系，以RGB箭头表达XYZ，逐图选视角；F明确为Link7局部[0,-95,0]mm、轴向不变的临时法兰系，H为规范化base_link。输出3张PNG、3张PDF及18坐标系世界变换/关节角/来源散列JSON。新增`docs/arm_coordinate_frames_20260930.md`并在装配文档链接。
- **验证**：直接解析URDF独立FK与MuJoCo14个连杆矩阵一致，最大元素误差左5.55e-16/右6.66e-16；各+Z等于真实正关节轴，J7工具变换与手根一致。首次图检查发现手体前缀不匹配，已按实际模型名称修复并重新出图；每侧包含8臂体/22手体/1连接件体，三图已目视检查。CPU Agg绘图退出0，未使用GPU；图片分别3240×2160/2880×1620。
- **TODO／边界**：法兰轴向是显式项目约定而非厂商标定TCP，V32-MOUNT-01/TJ-ARM-02/03继续保留，右手仍为镜像；各关节图为含q的随动系而非DH/零位系。无模型、安装参数、奖励、训练或硬件控制变更。

## 2026-09-30：安装新版 A3.2_tianji 连接件与宽安装座

- **问题／授权**：用户提供最新`A3.2_tianji.stp`要求安装。新板四孔19×40mm，与同日旧窄座19×30mm不匹配，需一起更新手侧安装座，不能直接沿用旧中央孔变换。
- **实际参考**：用户STP、`preview/assets/arm_hand_connect/assembly.json`、现有连接件配置、M6配置/URDF、双手URDF/质量版备份、质量/惯量/安装座来源清单；`import_hand_connector.py`、`prepare_connector_mount.py`、`replace_hand_mount_from_step.py`、`prepare_models.py`、`preview_connector.py`、`preview_arm_ik.py`、`draw_connector_alignment.py`、HPC打包/提交脚本及质量/安装/预览测试。可选日志技能路径缺失。
- **修改位置与原因**：导入器支持识别新参考手、跳过整手网格及按名称递归提取安装座。新`install_tianji_mount.py`核对四孔、保存旧文件备份、生成六件双侧运行网格及45.6×24.6mm宽座，更新URDF引用、安装清单和模型散列、连接件配置及M6工具变换。位置为左[-13.5,-121,-4]mm/右[-13.5,-121,+4]mm，沿用X旋转π/2。手指/关节/惯性/质量(.952kg每手)/肩座保持，右手仍为镜像。预览面板改用基准装配措辞。新STP保存在项目内，运行无需附件路径；备份`model_backups/before_tianji_mount_20260930.tar.gz`。
- **验证**：7项测试通过（几何/镜像/双侧四孔与法兰、关节惯性保持、质量兼容、真实模型调整/恢复/保存）。临时独立重建两URDF及四个修订STL逐字节一致。Slurm1119以0:0完成A100装配PNG（14秒），普通源码预览138坐标/50驱动器/51动作、0初态接触；双侧IK位置残差<1e-6mm，仅数值精度。CPU细节图、CAD检查及重建证据在`preview/outputs/tianji_mount_20260930/`。README、新`docs/tianji_mount_20260930.md`及历史装配文档同步。
- **未解决／边界**：初次整手导出5542面未三角化，未使用该参考手，改用命名闭合安装座。六件外部连接板之一有8条非流形边/0开放边，保留仅显示、不用于质量或实体接触；其他五件及手侧座闭合。TODO(V32-MOUNT-01) 实机孔/螺纹、法兰转角、新座质量惯量及连接板实体化前修复；TJ-ARM-02/03/04临时肩座、动力学和避障保留。无真机控制、新训练或策略F1验收，旧检查点仍须旧源码归档。

## 2026-09-30：检查1099/1103完整训练结果

- **问题与依据**：用户要求检查当前训练结果。读取Slurm1099/1103完成状态、两组完整config/summary/best及99次三回合评估metrics，复核gesture对照清单、F1聚合源码与几何/碰撞诊断语义。
- **核验结果**：两组均正常退出0、10,000,384样本、2,453,376更新、replay插入一致、末端流水排空；315音乐帧/559目标键帧/400Hz一致。所有评估3回合完整，按键/手碰撞/琴键诊断覆盖1，训练损失有限。训练参数除倍率和来源元数据外相同、scene/task散列相同；均预算耗尽且未通过F1 .99/零手碰撞联合目标，墙钟2:15:19/2:15:35。
- **发现**：十倍组best逐帧音乐F1 .91048→.92515，final .90942→.92381；但final 40Hz micro .93117→.92431、400Hz micro .94332→.93209，400Hz FN599.67→710.67。final手碰撞164.33→1.33子步，跨手164.33→0，键侧接触/实际几何缺额减少；非末节接触、用指角误差和释放尾段短误触增加。末100万同方向，不把逐帧F1改善当作累计漏键减少或整体最优。
- **文档与产物**：在`docs/gesture_score_comparison_20260930.md`补充完成结果、三种F1解释、末段/接触/尾段及检查点建议，README同步。派生JSON和曲线保存`training_mjlab/outputs/gesture10_comparison_20260930/completed_analysis.json`、`completed_training_curves.png`，含输入SHA-256、逐次评估、等预算校验；CPU分析断言通过并检查图像。未修改训练源码/默认或启动新训练。
- **TODO**：仅seed42训练内评估，尚无新逐音物理回放/具体音符归因；V32-GESTURE-10X-01保留多种子及建议3/5倍单项对照。1103 step-19000在记录的3回合手碰撞及误触为0，可作为复核候选，但有键侧/非末节接触，不能称物理安全已保证。

## 2026-09-30：humanoid得分10倍、外部weight≤1及双卡对照

- **用户要求与语义**：将当前双卡humanoid影响提高10倍，明确保留weight≤1，通过得分10倍实现并开启对照。当前geometry_clearance_v1是负罚，因此违规扣分幅度放大10倍；原单音/静音dt后上限.05→.5，旧辅助预算不超过一次漏键1/4的保证不再适用于新实验，已在说明中明确。
- **实际参考**：`AGENTS.md`、`scripts/hpc/{submit.sh,job.sbatch,train_coupled_40hz.sh}`、原`configs/rewards/coupled_40hz_20260930.json`、`mjlab_env/{reward_config,rewards}.py`、检查点reward合同、`scripts/swanlab_metrics.py`、几何姿态/时间单位/GPU耦合/入口测试，1099归档config/metrics。可选日志技能指定路径缺失；不影响本日志要求。
- **修改位置与原因**：geometry_clearance_v1新增可选score_multiplier，在基础和弦预算之后乘倍率，缺字段仍为1且不改旧canonical配置。外部weight≤1及基础预算校验严格保留，倍率单独校验有限正数。新增`coupled_40hz_gesture10_20260930.json`，与原JSON仅差score_multiplier=10；双卡默认指向新文件，普通run.sh及1099归档保持。公开指标配置同步保存倍率；README/HPC说明及`docs/gesture_score_comparison_20260930.md`记录公式、实验和预算变化。
- **验证结果**：CPU38项及44子测试通过，覆盖精确10倍/零点/梯度/和弦/dt/其他项不变、weight>1及无效倍率拒绝、旧合同与checkpoint恢复。Slurm1101真实40Hz耦合物理环境及奖励回归28项及30子测试通过，0退出；Shell语法通过。
- **对照启动**：1099继续作为原强度基线；新组1103于14:06从头启动，同seed42/10,000,384样本、31/2121、40/400Hz和相同SAC/腕速/评估设置。实际输出`training_mjlab/outputs/hpc/gesture10-comparison-20260930-140658-445247`，config除倍率及来源/路径元数据外逐项一致，包含相同scene/task散列；清单保存于`training_mjlab/outputs/gesture10_comparison_20260930/manifest.json`。归档另外包含同期preview_connector.py界面队列修正，仅由预览导入，不改变训练模型；该现场修改保留。14:10已观察到245,760样本/14,592更新，损失有限；gesture raw=weighted=−.43289813、dt后贡献−.21644907，实际weight=1及dt缩放核对一致。按用户要求不等待完整预算。
- **TODO**：V32-GESTURE-10X-01长期三种F1、漏弹/早松、跨手/同手/键侧碰撞及强辅助惩罚的取舍、多种子复验。功能验收不代表F1收益。

## 2026-09-30：1078回放、STEP安装座与质量修订、机械臂速度图

- **用户要求**：导出1078 final，安装上传连接件，按报告改质量，增加机械臂差分IK及全程速度图；TCP上限1m/s、2m/s²、1.5rad/s、4rad/s²。用户查看候选图后要求中央孔对齐、增加交互调整，随后明确直接在URDF将安装座换成STEP对应尺寸。
- **实际参考**：上传STEP/质量说明/六份JSON，规范左右URDF/原始左URDF/网格，1078 final/config/源码归档，现有预览IK/碰撞/控制/音频/Slurm脚本。M6采用厂商历史URDF提交`1a9cf30770ef6654cc1edfc048e1d0861be27eea`，来源/散列随预览资产保存。接触/驱动/碰撞代理三个附件与现有文件相同，未据附件扩大操作范围。
- **模型修改**：新增质量修订脚本，22连杆/手核对质量比惯量缩放、COM、正定/镜像；每手.952kg。质量版SHA与附件完全相同，备份后再改几何。安装面45.6×24.6→35.6×24.6mm，孔中心对齐；base内重复旧座一并去除，新增4份左右网格，URDF visual/collision引用同步，手指/关节保留。`prepare_models.py`可重建相同结果，质量/几何报告及派生manifest同步。
- **预览修改**：六件转接件仅显示，增加左右XYZ/欧拉角、恢复孔对齐、保存/备份/文件冲突检测。工具坐标随偏移更新，IK不可达回退。恢复可选M6资产；修正A3.2掌壳更名后的碰撞名单遗漏与新MuJoCo缺可选mesh graph时的只读线框显示。
- **回放与限速**：新增严格归档replay入口及200Hz只读状态观察器。离线阻尼差分IK导出两侧7关节角度/速度PNG/PDF/CSV与TCP曲线。原记录右/左线加速度8.429/8.009m/s²，角加速度293.920/196.593rad/s²。连续关节样条和静止首尾以1kHz核验，共同时间缩放11.495614后满足四项限制与URDF关节限速，9.15s路径变105.184867s；原速版保留，放慢版音频同比伸长。显示新安装座，物理/F1仍来自1078原模型记录。
- **验证**：34个相关测试通过；当前模型CPU编译134坐标/46驱动器、每手.952kg，惯量正值。临时重建两URDF/四STL字节一致，旧安装面残留面积0。Slurm1086原回放、1091完整记录、1093装配PNG、1097双机械臂视频完成，全片解码通过。1096自动geom重名失败已修复重跑；早期音频/ffmpeg路径问题已补私有依赖。交互以真实模型程序化事件测试，未人工操作云端Tk。
- **文档与范围**：README、`docs/hand_mount_mass_arm_replay_20260930.md`保存产物、命令、来源及两次独立回放F1差异。preview/arm-replay按需打包M6，普通训练包不含M6；可选CAD/音频/绘图库仅用户环境。此任务未启动新训练、未改已运行作业、未向硬件发送指令。
- **TODO**：V32-MOUNT-01实机孔位/工具方向/新几何惯量；V32-MASS-01新质量F1/接触稳定性；TJ-ARM-02/03/04肩座、工具及动力学/避障。右手仍为镜像推导，放慢演示不表示原节拍或硬件执行达标。

## 2026-09-30：双卡加速幅度与评估开销审计

- **问题／要求**：用户询问双卡训练为何没有大幅提速，要求检查原因。
- **实际参考**：Slurm934的benchmark/results.json及四次metrics；935的config／summary／metrics、GPU遥测／UUID映射／探针；当前及935归档的`rl_torch/{collector_pipeline,train,learner_graph,replay}.py`、`scripts/hpc/benchmark.py`及旧速度说明。查询Slurm当前无作业。
- **分析与修改**：新增`docs/dual_gpu_speed_audit_20260930.md`及JSON证据（含输入SHA256），README链接同步。核对双卡每批采样89.22ms、学习291.03ms、稳定加速1.2815倍；正式99次单环境三回合评估暂停学习。按累计吞吐还原时间，减去采样／学习／同步／commit计时并集，96个评估边界累计剩余1537.74秒，平均16.02秒；全程阶段外1590.44秒。分别检查无保存、best保存、周期保存，剩余均约16秒。解释单学习器关键路径和评估停顿，提出评估批量化与学习器profile优先级。
- **验证结果**：CPU解析原始JSON／JSONL／CSV，四组预算均42112次更新；正式9766步排空完成、1226624次更新，主机回调时间核算闭合。快照与commit共22.12秒，采样96.7%已重叠；遥测学习卡中位99%、低于5%占36.95%，P2P双向约200／196GB/s。已核对collector_pipeline／learner_graph／replay与935归档字节一致；训练评估解释使用935归档。原始数据和SHA可复核。
- **边界／TODO**：未提交GPU任务、未跑新benchmark、未改训练代码／默认值。数据对应旧1.2m/s；当前1m/s未测。剩余时间含评估／保存／管理，非独立CUDA内核计时；推算87.32分钟单卡不是完整单卡实测。TODO(V32-HPC-PERF-01) 评估批量化及学习器profile待验收；长期质量目标与性能分开判断。

## 2026-09-16 — 按 V2／V3 模式建立 V3.2 独立环境

### 目的与范围

新增 `tsensei_v3_2_pianist/`，复制已有 V3 环境的必要钢琴、舞台、音乐、奖励和 Composer 任务，使用用户提供的 V3.2 模型，完成独立安装、预览和验证。未启动策略优化训练。

### 问题、处理及依据

1. **缺少正式右手**：`daxian_v3.2/` 只有一份左手 URDF 和 22 个左 STL。已通过异步问题告知用户；在未收到选择回复时，按已说明的临时镜像方案完成双手环境。右手清楚标注为推导模型，正式文件保留 TODO(V32-10)，没有使用旧 V3 右手冒充。
2. **导出根坐标不同**：V3.2 掌部固定关节 xyz=`(0.008,0.0069,-0.045)`、rpy=`(0,0,-1.5708)`；直接套用 V3 安装四元数会导致方向和位置错误。用旧 V3 的掌部锚点定义规范安装系，对原模型仅做根坐标变换，保留左手内部运动与动力学；完整矩阵记录在包内 `assets/model_derivation.json`。这是坐标约定，不是实机标定。
3. **零驱动参数**：20 个活动关节全部 effort=velocity=0。保留原 URDF 值；MJCF 构建时对无有效 effort 的关节显式使用暂定 ±0.6 N·m（参考 V2 effort 上限），kp=3、damping=0.05 延续 V2／V3。该值未验证为 V3.2 硬件参数，速度约束未实现。
4. **ROS URI 与大网格**：原资源为含空格的 `package://A3.2 URDF 0908-2011/...` 路径。生成 URDF 时转为包内相对路径；保留原始 URDF 字节。食指／无名指 MCP 各 229740 三角面，超过 MuJoCo STL 200000 面上限，生成同顶点、同三角面的 OBJ；连同镜像右手，共 4 个 OBJ。未简化模型。
5. **结构和姿态差异**：拇指连接位置和姿态、各 DIP 偏移、小指结构改变，且小指名称为 `SWING` 大写，均保留实际 URDF 名称与结构。直接沿用 V3 拇指前两关节 1.0／0.8 rad 会产生额外 rota_block／MCP 碰撞；扫描 45 组限位内姿态后选择 1.0／0.4 rad。两侧安装仍为规范系 `(0.18,±0.15,0.12)` m。
6. **装配碰撞**：调整姿态后仍有左右掌部／thumb_rotaback 各一对约 1.32 mm 凸包重叠，暂沿用局部排除方式；可用 `--include-assembly-collisions` 恢复检查。其他碰撞保留，没有扩大到四指 MCP。真实装配干涉与凸包误差尚需确认。
7. **指尖定义未标定**：V3.2 末节极值面分别有 4／4／8／4／4 个最远 Z 顶点，采用其包围盒中点作为暂定距离奖励定位点；独立检查编译凸包和镜像关系，保留真实触键中心 TODO。
8. **验证过程问题**：测试有 pydub/audioop、dm_control/NumPy 弃用警告，无测试失败。一次辅助脚本写入命令误用了相对工作目录，随后改为绝对路径创建并运行成功。初始批量替换产生的 `daxian_v3.2.2` 说明文字已修正。MIDI 无指法时会打印 OT 提示，汇总报告从 JSON 正文起始位置解析。

### 实际参考文件

- 用户模型：`daxian_v3.2/urdf/A3.2 URDF 0908-2011.urdf`、同名 CSV 的导出字段、`meshes/*.STL`。读取关节树、axis、origin、限位、质量／惯量、网格顶点和三角面数。
- V3：`AGENTS.md`、`pyproject.toml`、`preview_pose.py`、`smoke_env.py`、`tests/test_environment.py`、`docs/source_manifest.json`；包内 `models/hands/{tsensei_hand,tsensei_hand_constants,urdf_model,fingertip_geometry}.py`、`suite/__init__.py`。
- V2：`scripts/convert_large_meshes.py` 的等几何转换方式；先前已读取的 V2 手模型常量与 URDF effort 基线。
- 根 `AGENTS.md`、README、日志；本机 `pianist-v2-change-log/SKILL.md`，此前读取的 `skill-creator/SKILL.md` 规则及其验证脚本。
- `external_never_copy_only_reference/` 本次没有读取、复制或导入，沿用 V2／V3 已确定的坐标约定。RoboPianist 必要文件经 V3 副本继承，完整来源链见 `source_manifest.json`。

### 具体改动

- 新项目与 Python 包统一命名 `tsensei_v3_2_pianist`，版本 0.3.2；本地 URDF、STL、OBJ、模型派生报告、MIDI 全部打包，无旧项目运行依赖。
- `assets/source/left_original.urdf`、22 个 `assets/meshes_left/*.STL` 为原始字节副本；新生成两个规范 URDF、22 个临时右 STL 和 4 个 OBJ。
- `scripts/prepare_models.py`：只用包内原始左手输入进行根坐标变换、完整镜像及大网格转换；反射位置、姿态、转动轴、惯量、法线，并反转面顶点顺序，记录派生文件 SHA-256。
- `models/hands/urdf_model.py`、`tsensei_hand_constants.py`：显式零 effort 替代值、新拇指初值与 TODO；保留 V3 任务和局部碰撞检查开关。
- `preview_pose.py`：普通／线框／定位球预览、冻结窗口、位姿偏移、步进物理及 XML 导出；JSON 明示临时右手、暂定力矩和根坐标转换。
- `tests/test_environment.py`：适配 V3.2 参数；新增直接对原始 URDF 的运动学／质心／惯量验证、随机姿态左右镜像验证、全部 STL 顶点／法线／面顺序验证。
- `scripts/check_installed.py`：在禁止访问源仓库及导入旧项目时检查 wheel 中的代码、模型和 MIDI。
- README、指尖说明、预览 PNG／JSON、来源清单、验证结果及本日志。根 README、AGENTS 和日志同步；日志技能扩展覆盖 V3.2，新目录自身的 AGENTS 保证独立复制后仍维护日志。

### 实际验证与范围

环境：Linux、Python 3.12.3、MuJoCo 3.13.0、dm_control 1.0.46、NumPy 2.5.3、独立 `.venv`。完整版本见 `requirements-tested.txt`。

- 创建 `.venv` 并 `pip install -e '.[test]'` 成功，`pip check` 无依赖冲突。
- `MUJOCO_GL=egl .venv/bin/python -m pytest tests -q`：**12 passed**，62 条第三方弃用警告。包含全部连杆随机限位内运动学、原始左手 COM／惯量、左右镜像、动作／限位／力矩、reset/step、随机重置、指尖凸包位置和显示开关。
- `MUJOCO_GL=egl .venv/bin/python smoke_env.py --steps 100`：通过，完成 4 回合，奖励／qpos／观测有限。动作 51、默认 Toy 观测 327、nq=nv=138、nu=50。
- 默认、碰撞线框＋定位球、恢复装配碰撞三种离屏预览均成功。默认与线框场景均为 0 接触／穿透；恢复装配模式显示实际掌部／拇指重叠，报告见同名 JSON。
- `--export-xml outputs/scene` 后使用 `mujoco.MjModel.from_xml_path` 重新编译成功，维数一致。
- `pip wheel --no-deps --no-build-isolation .` 成功，安装到仓库外 `/tmp/tsensei_v32_wheel_check`。从 `/tmp` 用 `python -I .../scripts/check_installed.py <target> --forbid-root /home/houjue/pianist_v2` 执行，禁止源仓库文件访问（仅允许本项目 venv 的第三方依赖）并禁止 V2／V3／RoboPianist 导入；Toy 与打包 Twinkle MIDI 各运行 40 步通过。
- 重跑 `scripts/prepare_models.py`，28 个派生文件（2 URDF、22 STL、4 OBJ）SHA-256 与首次生成一致；所有直接复制来源散列、原始 23 个模型文件字节一致性检查通过。
- `skill-creator/scripts/quick_validate.py` 检查日志技能通过。

机器可读摘要见 `validation_results.json`。未验证桌面 GUI、音频、长时间随机控制、训练收敛、真实右手或真机动力学。临时镜像和短步仿真通过不等于硬件模型已标定。

### TODO

| 编号 | 需要补充／确认 | 当前处理 |
| --- | --- | --- |
| V32-01 | 当前仿真安装坐标与姿态；真机部署时复核 | 2026-09-17 用户认为阶段基本完成，接受当前方案作为 MJLab 迁移基线；真机复核不阻塞迁移 |
| V32-02 | 机械臂／平移台模型、真实行程和驱动 | 每手 5 个虚拟腕部自由度 |
| V32-03 | 关节联动／独立驱动关系与真实准备姿态 | 20 个手指关节独立控制，轻弯初始目标 |
| V32-04 | 有效 effort、电机／减速比、控制增益、阻尼和摩擦 | 原始为 0；暂用 ±0.6 N·m、kp=3、damping=0.05 |
| V32-05 | 有效速度／加速度限位和执行机制 | 原始 velocity=0；未实现速度约束 |
| V32-06 | 装配关系、接触专用网格或凸分解 | CAD 凸包，暂排除两对掌部／拇指接触，可关闭检查 |
| V32-07 | 触键指垫模型／位置、材料与形变参数 | 几何端面定位点，不作为真机接触标定 |
| V32-08 | MIDI、指法、算法、算力和训练目标 | 已有单环境 JAX SAC 基线；2026-09-17 后续主线定为 MJLab + PyTorch SAC，待迁移及正式学习评估 |
| V32-09 | 模型后续对外再分发许可 | 保留用户来源，不套用源码许可证 |
| V32-10 | **正式 V3.2 右手 URDF 和全部关联网格** | 显式镜像临时右手；需重新核对几何、驱动与惯量 |


## 2026-09-16 — 统一 preview/scripts 与 preview/outputs 目录

- **问题**：预览主脚本原来位于项目根目录，输出散落在根 `outputs/` 与文档快照中；迁移后需修正默认输出位置及从其他工作目录启动时的本地包搜索路径。V2 使用说明还引用了旧目录名。
- **参考文件**：本项目原 `preview_pose.py`、README、对应包的 `preview_geometry.py` 导入接口，以及根／子项目 AGENTS、根 `.gitignore` 和本机 pianist-v2-change-log 技能；V2 另检查其内部 `.gitignore`。
- **修改位置与原因**：实际预览逻辑迁至 `preview/scripts/preview_pose.py`；根据脚本绝对路径确定项目根，并加载本项目的 Python 包；默认 PNG／JSON 为 `preview/outputs/pose.png`、`pose.json`。原根 `preview_pose.py` 改为 runpy 兼容入口，转发全部命令行参数。新增 `preview/README.md`、输出目录 `.gitkeep` 与局部 Git 忽略规则，更新项目 README 中的命令及输出路径。历史 `docs/` 预览快照保留；模型与控制参数未改。
- **实际验证**：从 `/tmp` 使用本项目 `.venv/bin/python` 运行新入口 `--headless`，默认输出成功落在本项目 `preview/outputs/`；随后运行根兼容入口 `--headless --show-pad-collisions --show-fingertips --output <本项目绝对路径>/preview/outputs/pads.png` 成功。普通／线框 PNG 与 JSON 均生成，十个末节线框可用，两次输出的手部状态元数据一致、均 0 接触。检查 `.gitkeep` 可纳入 Git、生成文件被忽略。沿用既有第三方弃用警告；未实测桌面窗口。本次为目录和入口整理，未重跑训练或模型全量测试。
- **TODO**：本次无新增未解决事项；原模型、驱动、安装／接触标定及训练相关 TODO 继续保留，详见前述条目。


## 2026-09-16 — 修复 Control 面板不能驱动关节和手部位置

- **问题与原因**：原交互分支只调用 `viewer.sync()` 和 sleep，没有执行物理步进；Control 面板更改了 `data.ctrl` 但执行器没有机会驱动关节。碰撞线框只在打开窗口时生成，也无法跟随运动。
- **参考文件**：本项目 `preview/scripts/preview_pose.py`、`tsensei_v3_2_pianist/preview_geometry.py`、手模型关节／执行器和虚拟腕部常量；已安装 MuJoCo `viewer.py` 的 `launch_passive`／`sync`／`lock` 接口，dm_control `engine.py` 的 `Physics.step`；[MuJoCo 官方 Passive viewer 文档](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)；根／子项目 AGENTS 和 pianist-v2-change-log 技能。
- **修改位置与原因**：新增 `tsensei_v3_2_pianist/preview_runtime.py`，由 viewer.sync 接收 Control 输入，在 viewer.lock 内推进真实物理并 forward 更新几何，按约 60 Hz 进行步进分组与节流。保留用户 ctrl，不调用会覆写动作的任务 step。每帧清空并重建线框，防止停留原位或不断累积。键盘事件使用线程安全队列；空格暂停／继续，R 恢复打开时完整 MjData（含时间和控制目标）。
- **预览入口与说明**：`preview/scripts/preview_pose.py` 默认使用实时交互；新增 `--paused` 保留暂停打开方式。`--headless` 仍生成静态快照，且明确拒绝与 `--paused` 同用。JSON 新增 capture 标记，说明保存的是交互前状态。更新项目及 preview README，说明 Control 控件、位置轴向、现有关节范围和恢复操作。模型、力矩与训练参数没有修改。
- **验证过程问题**：新增测试首次将单执行器 ctrl 当成数组索引，实际绑定返回标量，已改正；V2 默认高度下触键碰撞使位移达不到测试设定阈值。为单独验证 Control 到执行器的传递，将控制测试双手提高 0.1 m，保留真实动力学；原环境测试仍使用实际默认高度检查模型。没有通过修改预览力矩或禁用碰撞来绕过问题。
- **实际验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests -q` 中既有 12 项环境测试通过；修正控制测试后 `MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_preview_controls.py -q` 为 **3 passed**。新测试模拟 GUI 输入传递、执行真实 MuJoCo 仿真：验证双手食指关节及 tx／ty／tz 三轴响应、ctrl 不被覆盖、暂停／继续、R 恢复，以及移动线框不累积且与实时几何一致。测试使用模拟 viewer，未假称真实 GUI 操作。另从 `/tmp` 运行新入口 `--headless` 成功生成 PNG／JSON（0 接触），旧兼容入口对 `--headless --paused` 返回明确参数错误。沿用第三方弃用警告。
- **未覆盖与 TODO**：本机无 DISPLAY／Xvfb，未实测桌面鼠标拖动和键盘事件；需在桌面环境复核。受碰撞、重力与力矩限制时关节可能不到达目标，符合当前真实物理预览行为；原模型／驱动／接触标定 TODO 继续保留。未验证训练收敛。


## 2026-09-16 — GPU 渲染与预览卡顿修复

- **问题**：本机远程桌面默认 GLFW 上下文实际使用 Mesa llvmpipe CPU 软件渲染；此前 EGL 离屏使用 GPU，不能证明窗口也使用 GPU。预览每帧完整同步大模型，以及逐条重建碰撞线框，进一步占用控制循环。定时退出测试还复现 MuJoCo daemon 渲染线程尚未清理、Python／GLFW 已退出导致的段错误。
- **参考**：`preview/scripts/preview_pose.py`、`tsensei_v3_2_pianist/preview_runtime.py`、`preview_geometry.py`、`tests/test_preview_controls.py`、本机 `.venv` 的 `mujoco/viewer.py`（Handle.close、launch_passive、_launch_internal）和 MuJoCo `simulate.cc::Sync`；[官方状态同步文档](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)、[NVIDIA PRIME 环境变量说明](https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/primerenderoffload.html)；AGENTS 与 pianist-v2-change-log 技能。
- **改动**：新增包内 `preview_backend.py`，在导入图形库前选择驱动，隐藏上下文探测后恢复窗口可见性，输出实际 OpenGL renderer；新增 `--gpu auto/nvidia/system`。`preview_runtime.py` 优先使用状态同步，旧 API 退回完整同步；新增 `--full-sync`、`--duration`、`--profile-output` 和有界耗时采样。`preview_geometry.py` 缓存线框，运动时只更新世界变换，静止时复用。关闭时等待本次启动的渲染线程，避免退出竞争；此处隔离使用 MuJoCo 私有 `_launch_internal` 身份识别线程，未修改第三方安装。新增后端选择及关闭线程测试，扩展线框两端点一致性验证；README、preview 说明与本日志同步维护，代码仅依赖本项目和安装依赖。
- **实测环境与结果**：在用户现有 X11 桌面 `DISPLAY=:10`／本用户 XAUTHORITY 上实测，三个项目入口都显示 NVIDIA RTX PRO 6000 Blackwell，驱动 580.173.02。V3.2 窗口最终正常关闭（退出码 0），控制更新 62.2 Hz；同步中位数 0.231 ms、物理及 forward 0.649 ms、整次处理 2.900 ms。模型加载／reset 2.21 s，窗口创建 0.97 s。这些是服务器侧短时测量，不是客户端显示 FPS 或网络延迟保证。
- **验证命令**：`.venv/bin/python -m pytest tests -q`：21 项通过（第三方弃用警告保留）。桌面运行 `preview/scripts/preview_pose.py --duration 3 --profile-output preview/outputs/performance_gpu.json`，并开启 `--show-pad-collisions`；从 `/tmp` 用本项目绝对路径运行 `--headless --show-pad-collisions --show-fingertips`：PNG／JSON、EGL NVIDIA 渲染及退出码 0 均通过。控制测试模拟 GUI 输入、执行真实 MuJoCo 物理，覆盖手指／三轴平移、暂停／恢复／重置；线框两端点与原逐线算法一致。实际桌面验证覆盖窗口创建、仿真与关闭，未自动拖动真实鼠标滑块。详细命令、原始计时与验证范围见 [性能记录](preview_performance.json)。
- **同机对照**：同一 NVIDIA GPU、同样缓存线框的完整同步为 20.90 ms／循环约 43.2 Hz，状态同步为 0.23 ms／约 62.2 Hz；完整同步用相同命令追加 `--full-sync`。563,310,940 字节模型有 3,059,266 个网格三角面。独立 60 次线框计时（物理步不计入）从重建 10.17 ms 降到运动更新 1.47 ms，静止复用 0.013 ms；与原算法逐线验证两端点。早期无叠加线框的同步微基准为 21.31→0.049 ms，最终运行数据以本条含线框结果为准。

- **TODO／限制**：此次性能修复无新增阻塞 TODO；沿用原有模型／标定 TODO。标准 MuJoCo 物理仍在 CPU，GPU 加速 OpenGL 绘制；未验证远程客户端网络体验或训练收敛。旧 MuJoCo 缺少 state_only 参数时会退回完整同步；第三方升级后需回归窗口关闭流程。


## 2026-09-16 — 可选显示 World 与双手 wrist 坐标系

- **目的与范围**：为 V3.2 预览加入世界和双手 wrist 坐标轴；按用户要求通过 `--show-coordinate-frames` 显示，默认隐藏。仅影响本项目预览与相关说明。
- **问题**：模型没有单独命名 wrist 的 body，腕部自由度实际挂在规范化 `base_link`；世界原点位于琴键几何内部，部分轴在实体模式下受遮挡；原碰撞线框缓存将额外场景元素误判为需要重建，必须允许坐标轴追加在其后。
- **参考**：本包 `models/hands/tsensei_hand.py`（root_body 和 FOREARM_DOFS）、`urdf_model.py`、`tsensei_hand_constants.py`、`assets/urdf/daxian_left.urdf`、`assets/model_derivation.json`；现有 preview 脚本、preview_geometry、preview_runtime 和控制测试；[MuJoCo 渲染几何类型](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvgeom)、[官方渲染源码](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/render/render_gl3.c)（箭头与标签渲染）；根／子 AGENTS 与 pianist-v2-change-log 技能。未复制或导入附件参考目录。
- **修改**：新增 `tsensei_v3_2_pianist/preview_frames.py`，读取实际 body xpos/xmat，绘制 World／Right wrist／Left wrist 的 RGB 正向箭头与端点标签，共 18 个装饰元素。World 固定为 `(0,0,0)`，wrist 使用规范化 base_link 的实时位姿。脚本新增 `--show-coordinate-frames`、`--axis-length`（默认 0.09 m，须正且有限）、`--transparent-model`；最后一项仅设置 MuJoCo 显示透明选项，用于观察被遮挡的轴。窗口和离屏均支持；JSON 增加显示状态、坐标定义、原点及世界旋转矩阵，仍为交互前快照。
- **实现与性能**：preview_runtime 在碰撞线框之后更新固定的坐标轴槽位并记录 axes_ms；preview_geometry 允许尾部附加元素，保持线框缓存。沿用 GPU 状态同步。未添加物理几何／关节或修改动作空间；透明选项也仅影响渲染。同步 README 与 preview/README，并纠正已过时的“本机无 DISPLAY”说明。
- **验证**：`.venv/bin/python -m pytest tests -q` 共 23 项通过。新增两种组合（有／无线框）的真实物理控制回归，核对固定世界轴、随手运动的轴端点、物理状态不变及线框仅初始化一次。分别从 `/tmp` 运行默认关闭的旧兼容入口和显式开启坐标轴的主入口，均成功输出 PNG／JSON、退出码 0；核对开关前后双手位姿／接触／物理维度一致。普通和半透明图片均已人工查看。实际 NVIDIA 桌面运行 `--show-coordinate-frames --show-pad-collisions --duration 3` 并正常退出；循环约 62.2 Hz，轴更新中位数 0.062 ms，沿用 state_only 同步。详见 [验证记录](coordinate_frames_validation.json)。
- **限制与 TODO**：无新增阻塞项。坐标轴表示当前规范化模型和虚拟腕部，沿用 V32-01／V32-02 的安装与真实腕部标定 TODO；右手仍为临时镜像（V32-10）。不透明模型会正常遮挡坐标轴，可启用半透明显示；未实测远程客户端延迟或真实鼠标拖动，不将控制循环 Hz 视为显示 FPS。


## 2026-09-16 — 缩短 MuJoCo Control 执行器标签

- **目的与问题**：Control 面板标签栏较窄，原编译名如 `right_/A_right_hand_index_swing_joint` 只能显示相同的长前缀，无法分辨关节。用户提供的截图仅用于确认界面现象，不包含需要执行的文档指令。
- **参考**：`models/hands/urdf_model.py` 的手指执行器生成、`tsensei_hand.py` 的腕部执行器和动作顺序、两侧包内 URDF 的关节名称、任务中按关节查找腕部执行器的代码、dm_control `namescope.py` 的附加模型前缀规则、Control 回归测试，以及根／子项目 AGENTS 与 pianist-v2-change-log 技能。未修改第三方安装或读取外部参考项目代码。
- **修改**：在 `tsensei_hand_constants.py` 建立完整的 25 项／每手显示缩写映射；手指示例为 `R_Index_SW`，腕部为 `R_Wrist_TX`。`TsenseiHand.promote_actuators()` 在手模型附加到场景后，将 50 个 actuator 提升到场景根节点，保留对原 namespaced joint 的引用，因此 Composer 不再给 Control 名称加 `right_/`、`left_/`。保留 URDF／MJCF joint、body、网格名称。新增 `actuator_for_joint()`，让内部任务按稳定的源关节名解析执行器；更新任务、测试、控制台提示、项目 README 和 preview 说明。
- **行为影响**：Control 直接显示 `R_Index_SW`、`L_Index_SW` 等。50 维动作数量、左右手顺序、ctrlrange、forcerange、关节与物理参数均不变；动作规格中的执行器字段名称改为无命名空间短名，按旧字符串名称读取动作字段的外部程序需要同步更新。
- **验证**：`.venv/bin/python -m pytest tests -q` 为 **24 passed**；新增断言核对 `R_Index_SW`、`L_Index_SW`、`R_Wrist_TX`、`L_Wrist_Yaw`，50 个名称唯一，且全部不含 `/`、`_hand_`、`_joint`。测试仍覆盖真实 MuJoCo 控制、暂停／恢复、重置和显示几何；第三方弃用警告保留。在本机 X11 `:10` 以 NVIDIA 窗口运行 `--duration 2`，Control 模型加载、物理循环及窗口正常退出（退出码 0）。
- **TODO**：本次无新增模型 TODO。程序侧已确认标签中没有命名空间；实际字体观感由用户当前桌面窗口最终确认。


## 2026-09-16 — 替换最新 A3.2 左手并重新生成右手

- **请求与范围**：将用户提供的 `/home/houjue/A3.2 URDF/urdf/A3.1V6 26.8.4-2.urdf` 和同目录包的网格接入当前 V3.2 独立项目，生成可运行的镜像右手。按用户命名视为新的 A3.2；源 robot 名称仍保留 `A3.1V6 26.8.4-2` 作为来源记录。
- **实际问题**：小指拓扑改变，旧 `pinky_DIP` 和大写 `pinky_SWING` 已不存在，新增 `pinky_rotaback` 和小写 `pinky_swing`；MCP 变为 X 轴 ±0.35 rad 侧摆，末节改为 PIP。直接覆盖资产会使指尖查找和 Control 名称映射失败。全部 effort/velocity 仍为 0；新网格掌部／拇指凸包最大约 2.51 mm 重叠。两份左 MCP STL 各 208154 面，仍需等几何 OBJ 转换。
- **参考**：新 URDF 与 22 个配套 STL；旧包内 source/left_original.urdf 与网格；scripts/prepare_models.py、models/hands/tsensei_hand.py、tsensei_hand_constants.py、urdf_model.py、tests/test_environment.py、smoke_env.py、preview/scripts/preview_pose.py；README、docs/source_manifest.json、fingertip_measurements.md；根／子项目 AGENTS 与 pianist-v2-change-log 技能。附件仅作为模型数据；未读取或复制 external_never_copy_only_reference。
- **修改**：先归档旧 assets、手模型／控制常量、来源清单和文档快照至 `model_backups/before_a32_20260916_141929.tar.gz`；在独立临时目录用现有生成器重建新规范左手、规范镜像右手、22 个右 STL、4 个 OBJ，检查当前资产未被并发改动后替换活动 assets。清除活动资产中的旧小指 DIP／SWING 文件；归档保留原件。右手完整反射位姿、转动轴、惯量、法线和三角面绕序，仍明确为临时镜像。新增模型输入 SHA-256 并将旧来源移入 manifest 的修订历史。
- **小指适配**：tsensei_hand.py 使用 PIP 作为小指末节；常量为新 rotaback／MCP／swing／PIP 提供短名 Pinky_Rot／MCP／SW／PIP，并按关节作用设置暂定初值 0.32／0／0.45／0.24 rad。保持现有安装坐标、腕部、拇指姿态和两对掌部碰撞排除；未扩展排除范围。源 URDF／左 STL 均保留新输入字节，初值在 Python 层定义。
- **说明与交付**：更新 README 的当前源文件、动作顺序／旧策略兼容性、208154 面数、2.51 mm 穿透、备份位置；同步 preview 控制说明与末节测量文档。刷新默认／线框＋指尖／原始碰撞三套 docs PNG/JSON；新增 [本次验证记录](a32_model_replacement.json)。早期性能和首次适配记录作为历史保留。
- **验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests -q` 原有 24 项全部通过；新增左右小指链／关节轴限位／末节 site／初值／短名的 2 项测试，通过 `-k updated_a32` 执行。现有测试覆盖原始左手正运动学与惯量、镜像运动／质心／惯量、网格顶点／法线／绕序、Control 和坐标轴。`smoke_env.py --steps 100` 通过，4 回合，动作 51／观测 327／nq=nv=138／nu=50；默认 0 接触。三种离屏预览成功，查看默认双手 PNG；恢复装配碰撞后 4 个接触、最大穿透 0.002510672 m。23 个新直接输入字节一致，28 个派生文件重复生成 SHA-256 一致；旧归档可读且原 URDF 散列一致。未进行真实 GUI／真机或训练收敛验证。
- **TODO／限制**：沿用 V32-01～10，无新增编号。特别是 V32-04／05 的真实力矩／速度、V32-06 的碰撞模型和 V32-10 的正式右手仍需补充。新小指动作语义发生变化，旧策略不能仅凭 51 维形状相同直接复用。


## 2026-09-16 — 左右小指统一 MCP／swing／PIP／DIP 命名

- **请求与问题**：按用户完整映射，将 rotaback→MCP、MCP→swing、原 swing→PIP、原 PIP→DIP；左右手同时修改。旧 swing 已存在，必须一次应用映射避免重名或连锁替换。一次辅助测试编辑命令因工作目录重复拼接路径失败，已更正后执行，未影响模型资产。
- **参考**：活动 `assets/urdf/daxian_left.urdf`／`daxian_right.urdf` 的真实小指链；`scripts/prepare_models.py`；手模型、控制常量与初始化逻辑；`tests/test_environment.py` 原始 URDF 正运动学／惯量比较及小指测试；README、preview/README、末节测量和模型来源清单；AGENTS 与 pianist-v2-change-log 技能。
- **改动**：生成器新增一次性小指 joint／link 映射，同步 parent/child 引用，先规范左手再生成镜像右手；重新生成两个运行 URDF，`model_derivation.json` 增加 `left_source_name_map` 并刷新生成散列。包内原始输入副本及外部附件保持原始字节作为来源记录；STL／OBJ 文件名沿用来源名称，URDF 引用仍有效。每次重跑生成器都会应用新命名。
- **运行适配**：`tsensei_hand_constants.py` 的小指 Control 改为 R/L_Pinky_MCP、SW、PIP、DIP；初值按原物理槽位保持 0.32／0／0.45／0.24 rad。`tsensei_hand.py` 的小指末节 site 绑定改为 DIP。更新小指断言与原始 URDF 比较测试，使其通过显式来源映射查找运行关节／连杆；正运动学与惯量仍独立由未改名原始文件计算。更新当前 README、preview 说明、末节文档、来源清单及三份文档 JSON 快照的名称。
- **验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests -q`：26 项通过。将改名前左右 URDF 仅替换 name/link 属性后，与新生成 URDF 完整 XML 一致，确认轴、限位、树结构、质量／惯量与网格引用无额外变化；原始 URDF 和所有网格 SHA-256 不变，派生文件散列与清单吻合，名称唯一且父子引用有效。`preview/scripts/preview_pose.py --headless --output preview/outputs/pinky_renamed.png` 成功；默认 0 接触，新报告的中立目标和指尖世界坐标与此前快照完全一致。预览与控制测试覆盖当前模型，未重做真实 GUI 或训练验证。
- **TODO／兼容性**：无新增 TODO，保留既有 V32 模型标定事项。此次只改名，数值动作槽位与物理定义不变；按旧名称绑定关节／执行器的外部代码需更新到新名称。原始来源文件和历史修订报告仍使用原名。


## 2026-09-16 — 双 A100 保留 SAC 的并行训练设计

- **目的与范围**：为最新 V3.2 制定双卡 GPU 并行训练方案。用户补充整机为 8×A100-SXM4-80GB、NVLink/NVSwitch、64 物理核/128 逻辑 CPU、约 2 TiB RAM；方案沿用最初的两卡预算。本次只修改文档。
- **问题**：现有母仓库 SAC 尚未接通独立 V3.2；采样动作转换为 NumPy、回放只拼接单环境状态、无指法奖励依赖 SciPy CPU 指派。并行环境数量增大后，若不定义每条样本的更新预算，会大幅改变学习强度；原训练器还复用训练/评估 timestep。上述问题为源码审阅结果，尚未修复或运行验证。
- **参考**：母仓库 `robopianist-rl/train.py`、`run.sh`、`sac.py`、`networks.py`、`distributions.py`、`specs.py`、`replay.py`；本包 README、任务基类/双手任务、复合奖励和钢琴状态代码；[MJX](https://mujoco.readthedocs.io/en/latest/mjx.html)、[Warp 调优](https://mujoco.readthedocs.io/en/latest/mjwarp/#performance-tuning)、[JAX 多卡](https://docs.jax.dev/en/latest/201/shard-map.html)、[显存](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)、[测速](https://docs.jax.dev/en/latest/benchmarking.html)；根/子 AGENTS 和 pianist-v2-change-log 技能。未读取或复制只作参考目录代码。
- **修改**：新增 [双 A100 SAC 方案](dual_a100_sac_plan.md)，提供分卡采样/学习流水线与双卡同步学习对照、原 SAC 算法合同、batch/更新比例公式、80 GB 环境数搜索、回放显存预算、GPU 奖励/reset/接触要求、P0～P5 验收、检查点和独立包组织建议；README 训练接口增加入口。明确文档不是已实现的训练命令，参数是待测起点。
- **验证**：在本地 Linux 通过 `python3` 标准库检查方案/README 的相对链接、代码围栏、6 个稳定 TODO、硬件规格和入口；重新计算一百万条 replay 的 D=327/1180 占用分别为 2.828/9.652 GB，以及 E=1024 时三档 update 数 1024/256/128，全部通过。文档/日志新增段的格式与链接完成检查。未安装依赖、连接远程、测试 GPU 物理、多卡通信或训练收敛；本次无运行代码修改，未重跑物理测试。
- **TODO**：新增 V32-GPU-01～06，分别覆盖分配资源/互联核验、Warp 物理、批量任务、SAC/replay、双卡选优与多 seed 学习效果；既有 V32-01～10 模型标定事项继续保留。


## 2026-09-16 — 独立 SAC 基线实现与 Shadow 差异确认

- **目的与范围**：按用户要求先在独立 V3.2 中实现原 SAC，提供相对 Shadow 的具体修改清单，等待正式训练配置确认。仅新增训练包、可选依赖、测试与文档，不修改手模型、任务奖励或预览逻辑；不启动长训练或远程双卡任务。
- **参考**：母仓库 `robopianist-rl/{sac,networks,distributions,specs,replay,train}.py`、`run.sh`、LICENSE；原 Shadow 手常量/动作缩减和 `robopianist/wrappers/evaluation.py`；本项目 suite/任务/模型/音乐库；实际安装的 dm-env-wrappers 0.0.13 的拼接、动作归一化、历史动作/奖励、float32 转换；[JAX 安装文档](https://docs.jax.dev/en/latest/installation.html)；根/子 AGENTS 与 pianist-v2-change-log 技能。未读取或复制 external_never_copy_only_reference。来源散列见 [清单](sac_source_manifest.json)。
- **遇到问题**：原训练尚未接通 V3.2；旧 JAX KeyArray 类型标注已移除，REDQ 子采样仅适配 FrozenDict；旧 SinglePrecisionWrapper 的 `np.array(copy=False,dtype=...)` 在 NumPy 2 下报错；V3.2 严格动作限位需避免 float32 物理端点舍入；原训练循环评估复用 timestep。前两个 smoke/inspect 尝试因 NumPy 转换失败，修复本地适配层后通过，未改第三方安装代码。
- **算法修改**：新增包内 `rl/` 的 SAC/网络/分布/specs，相对导入并保留双 Q、actor 取 Q 均值、目标取最小值、自动温度和原更新顺序。新增参数合法性检查，显式 target_entropy=0 不再错误回退；类型标注/字典兼容；新增设备驻留动作接口。原默认数值行为与 Shadow 参考三次更新对照通过。携带 LICENSE.robopianist-rl 与来源清单，运行不读母仓库。
- **训练适配**：新增 V3.2 环境包装、显式 transition 回放、独立评估、CLI、原子 SAC 状态检查点。复用原包装顺序，训练配置实测动作51/观测1180；历史动作仍是物理目标，replay 动作为归一化值。评估保留原逐帧 P/R/F1 与踏板统计，额外独立命名 micro F1。探索/回放 RNG 独立，修复评估污染采样状态。恢复只为 warm restart，明确不恢复 replay/物理状态。
- **依赖与说明**：pyproject 增加可选 train extra 和 tsensei-sac 入口，在 `training/.venv` 安装 CPU 验证环境，保持预览 `.venv` 不变；保存 requirements-sac-tested.txt。新增 [确认表](sac_shadow_review.md)、[源码差异](shadow_sac_comparison.diff)、[真实默认合同](sac_default_contract.json)、[验证记录](sac_validation.json)，更新 README 和双 A100 计划进度。C01～C05 包含动作、曲目、输入、重力补偿和 SAC 超参数，均未据此执行正式长训练。
- **验证**：`MUJOCO_GL=egl JAX_PLATFORMS=cpu training/.venv/bin/python -m pytest tests -q --disable-warnings`：**34 passed**（原26+新增8），184项第三方弃用警告，44.20s；pip check 通过。原 Shadow 算法生成独立参考 NPZ，测试3次更新后的75个状态叶子、各步损失、随机动作与确定性动作；覆盖 checkpoint 后续更新一致、回放末帧/环绕、动作上下限、frame stack/无指法和评估不影响训练。Toy 小网络32步/25次更新/1回合及2次评估成功；默认256×3网络12步/5次更新与评估成功；热启动8步后累计26次更新（原25+新1）。检查源码散列、Markdown链接、JSON、格式、虚拟环境忽略规则。详细配置、版本与局限见验证记录。
- **未覆盖／TODO**：未做收敛、A100/CUDA、多环境/MJX、多卡或本次新 wheel 安装验证。V32-SAC-01 等待 C01～C05 确认及正式学习评估；V32-SAC-02 精确恢复待实现。V32-GPU-04 的独立 SAC/CPU 回放部分已完成，GPU replay 等其余 GPU TODO 仍保留；既有手模型标定事项不变。


## 2026-09-16 — 从预览保存左右手初始姿态到常量文件

- **目的与范围**：解释当前初始角度、安装位置和 pitch／yaw 来源，并支持在 V3.2 预览中写回同一个配置文件。
- **问题**：共享 INITIAL_ANGLES 无法表达左右手不同姿态；Control 目标与接触／力矩约束下实际关节值可能不同；安装位置与腕部平移若都写成世界坐标会重复位移；保存整个 Python 文件可能覆盖其他配置或用户并发编辑。测试中还遇到 0.1+0.02 的浮点精确比较失败，改用近似数值断言后通过。
- **参考**：`models/hands/tsensei_hand_constants.py` 的安装位置／四元数／FOREARM_DOFS／INITIAL_ANGLES，`tsensei_hand.py::_initial/initialize_episode/actuator_for_joint`，`suite/tasks/base.py::_add_hand`，现有 preview 脚本／运行循环及环境／控制测试；根／子 AGENTS 和 pianist-v2-change-log 技能。最初按旧目录寻找 tasks/piano_task.py 不存在，实际追踪到 suite/tasks/base.py。未读取或复制只作参考目录。
- **配置修改**：在共享 INITIAL_ANGLES 中显式列出原本隐含为 0 的腕部平移、pitch、yaw；新增默认空的 INITIAL_ANGLES_BY_SIDE。手模型按“单手值→共享值→0”读取初值，reset 同时初始化 qpos 和 ctrl。现有默认姿态、限位、力矩和安装旋转数值保持不变。
- **保存功能**：新增 `preview_pose_config.py`；S 键显式保存，`--save-initial-pose` 在窗口关闭／离屏渲染后保存，`--save-pose-source ctrl/qpos` 默认 ctrl。按当前真实关节引用提取每手所有关节，兼容简短 Control 名称；安装位置含 CLI 偏移，腕部平移单独记录。只替换原文件 RIGHT_POSITION、LEFT_POSITION、INITIAL_ANGLES_BY_SIDE 三个赋值的值，保留其他代码与注释。验证有限值、关节／控制范围和 Python 语法，拒绝越限而非静默裁剪。
- **备份与冲突**：每次先保存完整旧文件和 JSON 记录到 preview/outputs/initial_pose_backups，包含来源、实际值／目标、修改字段、参考路径及前后散列；临时文件写入并原子替换。打开窗口后如原文件被外部编辑则拒绝覆盖；重复保存自己的修改可继续。S 写入失败在终端说明，窗口继续运行。保存不改变当前物理状态；重启预览／训练后生效，R 仍还原本次入口状态。README、preview/README 同步新增完整初值／单位／优先级及恢复步骤。
- **验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_environment.py tests/test_preview_backend.py tests/test_preview_controls.py tests/test_preview_pose_config.py -q`：33 项通过，116 项第三方弃用警告，36.69 s。新增 ctrl／qpos 两种模式的左右不对称姿态保存与新进程加载，核对根位置、旋转、指尖误差 ≤1e-9；覆盖偏移不重复、暂停 S、失败后继续、非有限／越限拒绝、外部编辑与模拟原子写入失败、备份完整性和源码注释保留。
- **入口验证**：在 /tmp 隔离代码副本（资源仅引用原 assets）运行真实 CLI：离屏保存 height_offset=0.01、depth_offset=0.02 后，不带偏移重启，手部 JSON 完全相同；原项目常量未被这些写入测试覆盖。实际 NVIDIA X11 窗口 `--paused --duration 2 --save-initial-pose` 成功保存并退出码 0。S 键通过模拟 GUI 事件结合真实物理测试，未向桌面窗口发送真实按键。详见 [验证记录](initial_pose_save_validation.json)。本次未修改／重测 SAC 算法或执行训练收敛。
- **TODO／限制**：无新增阻塞项；沿用 V32-01／02／03 等安装、虚拟腕部和关节姿态标定 TODO。ctrl 保存的是目标，下一次 reset 直接放到目标姿态，未保证受阻时等同当前可见姿态；qpos 若受软限位影响越界则拒绝保存。已有训练／预览进程不热加载配置。


## 2026-09-17 — wrist yaw 同向、pitch 反向

- **目的／问题**：用户要求相同符号的 yaw 使左右手同向转动，相同符号的 pitch 使左右手反向转动。原实现只将左手 yaw 轴取反，导致 yaw 必须取相反数才同向，pitch 则同号同向。
- **参考**：`models/hands/tsensei_hand.py::_build`、`tsensei_hand_constants.py::FOREARM_DOFS/INITIAL_ANGLES_BY_SIDE`、现有 `tests/test_environment.py`／控制与姿态保存测试、preview/README、README；只读检查 `rl/environment.py` 的模型源码散列包含手模型和常量；根／子 AGENTS 与 pianist-v2-change-log 技能。
- **修改**：`tsensei_hand.py` 中左手关节轴取反条件由 forearm_yaw 改为 forearm_pitch。最终局部轴：yaw 两手均 −Y；pitch 右手 +X、左手 −X。常量补充注释，README 和预览说明同步新方向、局部坐标／复合转动含义，以及旧数据迁移方法。角度范围、控制增益、力矩及动作维度不变。V2／V3 未改。
- **初始值与兼容**：核对当前实际保存字典，两手 pitch／yaw 均为 0，无需数值迁移，保持当前初始姿态。若恢复此前的非零配置，为保持原物理姿态需将左手 pitch 和 yaw 都取负，右手不变；历史备份不改写。本次变更作用于物理模型及训练对应动作语义，旧策略／动作数据需要处理符号变化，不能仅凭相同维数视为兼容。
- **验证**：新增 4 项真实 MuJoCo 正向运动学测试，分别以正／负角度验证 yaw 左右世界增量旋转矩阵相同、pitch 互为逆旋转，并排除未发生运动的假通过。两手对齐、另一腕部角置 0；矩阵容差 1e-12。`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_environment.py tests/test_preview_backend.py tests/test_preview_controls.py tests/test_preview_pose_config.py -q`：37 项全部通过，128 项第三方弃用警告，40.03 s；包括原控制、坐标轴、独立姿态保存／新进程重载回归。`git diff --check` 通过。本次未重新运行桌面鼠标操作、训练或真机验证。
- **TODO**：无新增阻塞项；保留既有 V32 安装、虚拟腕部与硬件标定 TODO。启动中的预览需重启以重新编译关节轴。


## 2026-09-17 — TODO-01 阶段验收与 MJLab + SAC 方案调整

- **目的与范围**：按用户判断将 V32-01 标为当前仿真安装方案基本完成，将后续主线调整为 MJLab + MuJoCo Warp + 自定义 PyTorch SAC + CUDA replay。本次交付方案、状态及说明更新；尚未实现新训练器。
- **遇到的问题**：原双卡文档采用直接 MJX-Warp + JAX 路线，与本次框架选择不符。MJLab 现成训练集成主要面向 RSL-RL/PPO，不能直接当作 SAC；默认奖励乘步长、自动 reset 和物理派生量采样时序也可能改变当前学习行为。阅读时首次按旧路径查找 composite_reward.py 未找到，随后定位到 suite/composite_reward.py。
- **本地参考**：docs/dual_a100_sac_plan.md、sac_shadow_review.md、sac_default_contract.json；rl/environment.py、replay.py、sac.py、networks.py；suite/tasks/piano_with_tsensei_hands.py 和 suite/composite_reward.py；models/hands/tsensei_hand_constants.py、scripts/prepare_models.py；README、preview/README、两级 AGENTS 与 pianist-v2-change-log 技能。未读取或复制只作参考目录代码。
- **外部参考**：查询官方远端 HEAD，审阅提交 `8ee51fbcf806a7419189f706d9e394cbeb7790fa` 的 [环境配置与 step](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/envs/manager_based_rl_env.py)、[奖励管理器](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/managers/reward_manager.py)及依赖声明；参考官方[架构](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)、[训练集成](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)、安装和执行器说明。该提交仅是 API 审阅基准，尚非已安装验证的版本锁。
- **具体修改**：重写 [MJLab + SAC 方案](dual_a100_sac_plan.md)，保留原文件名和稳定 V32-GPU-01～06 编号；明确独立依赖、场景导出、51 维动作、当前初值和 yaw/pitch 约定、Torch 任务与 SAC 数值对照、GPU replay、更新预算、单卡到双卡阶段验收。首版规划显式关闭奖励时间缩放和自动 reset，保存末帧及原环境 discount。同步 README、preview/README、SAC 确认表及本日志 TODO 表。prepare_models.py 和手常量仅调整 V32-01 状态注释，运行语义和数值未变。
- **实际验证**：使用 python3 标准库核对四份说明的 29 个本地链接、Markdown 围栏、框架/API/稳定 TODO 状态及回放显存和更新预算算术；两个 Python 文件修改前后 AST 相同，语法有效；git diff --check 通过。新增 [本次验证记录](mjlab_sac_plan_validation.json)。没有安装 MJLab、实现 Torch SAC、连接远程 A100、运行 GPU 物理／双卡训练或验证收敛；本次只有说明与注释变更，未重跑物理测试。
- **下一步／TODO**：先实施 P0/P1（V32-GPU-01／02）：锁定依赖、冻结当前场景与 CPU 样例，在一张 GPU 验证 MJLab 的 1／16／64 环境 reset、控制、触键和批量独立性；随后移植任务、单卡 SAC，再评估双卡。V32-01 的真机测量作为后续部署复核，不阻塞 GPU 迁移；其他模型标定、正式右手、V32-SAC-01 正式学习配置与 V32-SAC-02 精确恢复继续保留。框架选择已获本次用户指令授权。


## 2026-09-17 — MJLab 第一阶段场景迁移与验收

- **目的与范围**：实现上一方案的第一项：将当前接受的 V3.2 双手＋钢琴场景导入 MJLab，在本机单 GPU 验证初始姿态、控制和触键，提供可重复验收与预览。V2/V3、原预览和已有 JAX SAC 运行逻辑未改。
- **遇到的问题**：只导出 XML 会丢失 reset 的 qpos／ctrl；MJLab 默认配置会覆盖部分物理选项，原预览 MuJoCo 3.13.0 与 MJLab 3.11.0 也有版本差异。初次依赖解析使用单一 Torch 索引停滞，改为固定官方 CUDA wheel URL＋PyPI 依赖，独立安装共约 8 GB 缓存。新 API 需先 wp.init；Torch 默认 CUDA stream 不能直接用于本机 Warp 图捕获，改为 Warp 自有流与 Torch 显式同步。首次 GUI 缺 DISPLAY；在用户桌面会话运行后发现 MuJoCo 3.11 渲染守护线程未退出就开始解释器清理，产生退出时 SIGSEGV，修复为 finally 等待对应线程完成。两次辅助文件写入因项目路径重复拼接失败，更正工作目录后重做。
- **本地参考**：suite/__init__.py、suite/tasks/base.py、models/piano/piano.py 的激活规则、手常量／已保存初值、rl/environment.py 的训练重力补偿候选；原 preview/scripts/preview_pose.py、preview_runtime.py、preview_backend.py 及控制测试；两级 AGENTS 和 pianist-v2-change-log 技能。没有读取或复制只作参考目录的代码。
- **外部参考**：官方 MJLab 提交 `8ee51fbcf806a7419189f706d9e394cbeb7790fa` 的 [Simulation](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/sim/sim.py)、scene、entity、XmlActuator、动作／观测／事件管理器和环境 step；安装版本的 MuJoCo viewer.py 与 MuJoCo Warp types/io。查阅其接口和生命周期，自行编写本项目适配层。MJLab 从固定官方源码包安装为第三方依赖，不依赖 /tmp 参考克隆或其他手项目。
- **具体修改**：新增 mjlab_env/bundle.py、scene.py、acceptance_cases.py；scripts/export_mjlab_scene.py、validate_mjlab_env.py；preview/scripts/preview_mjlab.py；tests/test_mjlab_bundle.py；training_mjlab/setup.sh 和 requirements-tested.txt。导出完整资产、命名映射、初值、全部可读物理选项、源散列及原版本 CPU 轨迹；拒绝覆盖非空导出目录／资产篡改。MJLab 使用 Simulation(model=...) 直接承载整体场景，不重复增加 PD 或修改碰撞和安装参数；同名映射 50 执行器＋踏板、归一化动作、0.002×25=0.05 s，只有一个世界。保留默认预览重力补偿关闭，可显式导出训练候选补偿版本。
- **预览与文档**：新窗口可用 Control、Space、R 或 --demo 运行 GPU 物理；CPU 拷贝仅用于显示，不推进物理。保留原脚本的保存／坐标轴用途。同步 README、preview/README、MJLab 方案和 SAC 说明；新增 [操作与验收说明](mjlab_scene_acceptance.md) 及 [数值记录](mjlab_scene_validation.json)。完整场景和 PNG／NPZ／运行报告位于 training_mjlab/outputs，未纳入 Git。
- **依赖验证**：本机 Python 3.12.3，RTX PRO 6000 Blackwell，驱动 580.173.02；MJLab 1.6.0 固定提交、MuJoCo/MuJoCo Warp 3.11.0、Warp 1.17.0、Torch 2.10.0+cu128。独立环境 pip check 通过；版本锁移除安装时临时源码路径，使用官方固定提交 URL。setup.sh 通过 bash -n；未另建第二个环境重复全量安装。原预览和 JAX 依赖环境保留。
- **实际验收**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_mjlab_bundle.py -q`：4 passed，8.07 s，11 项第三方弃用警告；覆盖导出初值／映射、目录移动、篡改／覆盖拒绝、真实 CPU 手按键与松开。`MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_env.py --render`：45 项 GPU 检查通过，包括正负 yaw 同向／pitch 反向、全部腕部坐标与左右食指响应、踏板、重置、CPU/GPU 及跨版本轨迹。初值最大误差 1.32e-8；按键／松键相对同版本 CPU 最大滑动误差约 0.0481 mm、角度 0.00243 rad。激活集合均 [33,54]，释放后为空；实际 Warp 末节—琴键接触通过。控制边界最多 12 个接触／130 个约束；没有宣称覆盖全部子步峰值或批量极限。首次编译加载约 92.3 s，缓存后约 1.64 s，整套缓存验收含两张 PNG 约 5.49 s；不作为训练吞吐基准。
- **窗口验证**：实际用户 X11 桌面运行 --demo --duration 4，观察两轮 []→[33,54]→[]，显式等待渲染线程后退出码 0。查看初始／按键两张 GPU 状态 PNG；未实测鼠标拖动或桌面按键事件。已为无 DISPLAY 提供早期错误提示，避免加载 GPU 后才失败。 补测 --paused --duration 2 正常退出。最终检查 52 个本地链接、Python／Shell 语法和 git diff --check 通过；根历史日志含改名前 tsensei_pianist 链接，本次仅核对新增根日志段，未改写历史记录。
- **TODO／下一步**：V32-GPU-01 本机版本与设备验证完成，远程 A100 分配／互联仍待核验；V32-GPU-02 单场景已完成，下一步 16／64 环境、每环境 reset 与接触容量压力验证。V32-GPU-03～06 的任务管理器、Torch SAC／replay、双卡、学习效果均未实现或验证。现有模型标定、镜像右手、SAC 正式配置／精确恢复继续保留。本实现是物理场景接口，未伪装为已完成音乐任务或策略训练。


## 2026-09-17 — 绿色琴键、接触止挡、25 mm 腕部与 MJLab 批量验收

- **目的与范围**：完成当前 V3.2 预览的激活琴键绿色显示和物理止挡，继续 V32-GPU-02 的 16／64 环境及独立 reset。用户确认 `[0,-25]` 指 Wrist_TZ 下压 0～−25 mm，因此将左右范围设为 `[-0.025, 0]` m；初始安装位置、初始角度保持当前值。
- **问题与处理**：原接触与琴键限位较软，旧参数探针中手—键穿透约 1.72 mm；只提高阻抗仍配合原 0.002 s Euler，会有抖动和更大误差。采用 0.0005 s、implicitfast、较高阻抗和更硬琴键限位。MuJoCo 是软约束求解器，不能保证数学意义零穿透，验收必须检查原始 qpos 与真实 Warp 接触，而非裁剪状态。原 −40 mm 腕部验收在收紧至 −25 mm 后不再合法；按用户范围改为腕部 −25 mm＋中指 MCP 初值增加 0.1 rad，未移动安装位置。初次同动作组使用混合米／弧度的 1e-5 容差过严，按滑动和铰链分别设为 0.05 mm／0.002 rad，实际差异在数值容差内；reset 未选世界仍要求逐位一致。
- **执行中问题**：CUDA 显存计数需先初始化设备；辅助命令曾因相对路径重复、系统只有 python3、误用预览不存在的 --output-prefix 参数失败，修正为实际路径、python3、--output 后成功。首次松键检查在 reset 后运行，不足以验证物理回弹，已改为真实受压状态下释放控制。没有把这些失败写为验收通过。
- **实际参考**：根／子 AGENTS、pianist-v2-change-log 技能；`suite/tasks/base.py`、`models/piano/piano.py`、`models/hands/tsensei_hand_constants.py`；`preview_runtime.py`、`preview_geometry.py`、`preview/scripts/preview_pose.py`／`preview_mjlab.py`；`mjlab_env/bundle.py`／`scene.py`／`acceptance_cases.py`；导出、GPU 验收、控制／场景测试及 SAC 配置检查入口。阅读已安装 MJLab Simulation.reset、MuJoCo Warp reset_data／types 的世界掩码与容量字段；未复制或导入 external_never_copy_only_reference。外部依据为 [MuJoCo 求解器参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters) 和 [joint solreflimit](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-solreflimit)。
- **接触与显示修改**：`suite/tasks/base.py` 默认物理步长 0.0005 s、控制 0.05 s（100 子步）、implicitfast；钢琴 geom solref=(0.001,1)、solimp=(.99,.99,.001,.5,2)、priority=1；键 joint solreflimit=(0.001,1)、solimplimit=(.999,.999,.001,.5,2)。未增大执行器力矩或删除触键碰撞。新增 `preview_keys.py`，CPU 与 MJLab 预览按实际键角激活阈值添加绿色纯视觉覆盖，离屏直接更新场景颜色；继续 state-only sync，保留坐标轴和姿态保存逻辑。新增 `--bottom-out` 自动按键／松键演示。
- **批量修改**：`mjlab_env/scene.py` 支持 num_envs、(N,51) 动作、每世界踏板／控制计数、指定 env_ids reset、单世界 CPU 显示快照和可选逐子步诊断。新增 `diagnostics.py`、`scripts/validate_mjlab_batch.py`，检查四组独立动作、持续受压、CPU 对照、接触／约束溢出、穿透、琴键超限、部分 reset、下一步状态与实际回弹。未选世界的 qpos/qvel/ctrl/time/qacc_warmstart/踏板/计数必须逐位不变。
- **范围与资源**：`FOREARM_DOFS['forearm_tz']` 改为 [-.025,0]；共享 `press_action` 驱动导出 CPU 参考、预览与批量检查，测试同步。默认冻结场景重新导出到 `training_mjlab/outputs/scene`，旧场景保存在 `scene_before_hard_contact_20260917`；最终 XML 与已验证的 scene_wrist25 相同，仅更新注释后的 source_hashes 不同，并重跑默认路径验收。按键／初始图复制到 preview/outputs/hard_contact_*.png。更新 README、preview/README、MJLab 方案、SAC 说明、单场景文档，新增 [第二阶段验收说明](mjlab_batch_acceptance.md) 和 [数值记录](mjlab_batch_validation.json)。`rl.train --inspect-only` 重新生成 sac_default_contract.json，动作边界和源码散列反映当前参数。
- **CPU 验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_mjlab_bundle.py tests/test_environment.py tests/test_preview_controls.py tests/test_piano_contact_and_color.py -q`：32 passed，126 个第三方弃用警告，39.71 s。新增颜色不改变物理、松键消色、合法目标下真实物理止挡测试；已有安装／控制和独立导出测试通过。
- **GPU 验证**：本机 RTX PRO 6000 Blackwell，既有锁定 MJLab 1.6.0／MuJoCo-Warp 3.11.0／Torch 2.10.0+cu128。最终默认场景的 `validate_mjlab_env.py --output training_mjlab/outputs/acceptance_wrist25 --render`：45 检查通过；`validate_mjlab_batch.py --num-envs 16`／`--num-envs 64`：各 49 检查通过。64 环境施压 3000 子步中最大手—键穿透约 0.326 mm、键前端超行程约 0.0432 mm，最多总接触 186、单环境约束 112，溢出位全零；中指关节与持续目标差约 0.121 rad。原始报告和 pressure_states.npz 保存于 outputs/batch_16、batch_64。诊断开销不代表训练吞吐，Torch 显存统计不含全部 Warp 分配。
- **窗口验证**：真实 X11 桌面 MJLab `--bottom-out --duration 4` 按键 []→[33,54]→[]、退出码 0；检查按键 PNG 确认绿色。原 CPU 预览 `--show-coordinate-frames --duration 2` 正常关闭，NVIDIA 硬件渲染、state-only sync，短测约 60 Hz；未实际操作鼠标滑块。保存功能未写入用户当前初值。本次没有运行 SAC 学习、完整训练回归或远程 A100。
- **最终静态核验**：13 个修改／新增 Python 文件语法通过、43 个当前说明的本地链接有效、Markdown 围栏成对，git diff --check 通过；16／64 报告散列与最终默认冻结场景一致。
- **TODO／后续**：本机 V32-GPU-02 的批量物理验收完成；下一步 V32-GPU-03 乐谱推进、Torch 观测／分项奖励／终止／历史和部分 reset 的 CPU 数值对照，再做 V32-GPU-04 Torch SAC 与 GPU replay。V32-GPU-01 的 A100 复测、V32-04／05／06／07／10 硬件动力学／接触／指尖／正式右手，以及正式学习配置与精确恢复继续保留。步长减小使子步成本增至原来 4 倍；范围、物理和源码散列变化后，旧 checkpoint 不能宣称为相同环境下精确续训。


## 2026-09-17 — 用户论文阅读、事实归档与逐项对齐约定

- **目的与范围**：阅读用户附件 `biomimetics-11-00610-v2.pdf`，回答其是否覆盖后续 Torch 乐谱／观测／奖励／终止迁移，记录必要方法与参数，并按用户要求在后续遇到重合设计时先提示、由用户决定是否对齐。此次仅资料、工作约定和说明变更，没有修改运行代码、物理、奖励或训练参数。
- **实际参考**：用户上传的26页 PDF（原附件路径与 SHA-256 见 [来源清单](references/biomimetics_2026_610_manifest.json)），全文文本及 pp.9–12 的原页图像，重点核对表2、表3、公式10–20；根／子 AGENTS、日志技能；当前 `suite/tasks/piano_with_tsensei_hands.py`、`suite/composite_reward.py`、`suite/tasks/base.py`、`models/hands/tsensei_hand.py`／常量、`music/midi_file.py`、`rl/environment.py`／`train.py`、`mjlab_env/scene.py`、SAC配置快照及现有迁移说明。DOI由附件记录，未联网核验在线版本、读取视频或作者实现；未读取或复制 external_never_copy_only_reference。
- **发现的问题**：论文与当前项目同用 MIDI／MJLab／SAC，但单手21动作、2106观测、40 Hz、乘dt的奖励、正功率惩罚、两阶段配置与当前双手51动作、候选1180观测、20 Hz、直接求和、绝对功率存在差异。论文未完整披露 Stage2 shaping、差分动作单位、指法编码边界、逐环境 reset／终止折扣、碰撞集合和部分实现／指标归约。不得把训练预算结束混为episode结束，也不能把256环境的向量步与CPU步直接比较。读取辅助脚本时一个预想的 wrappers/evaluation.py 不存在，实际评估位于 rl/train.py；PDF首次渲染使用 singlefile 只产生首页，随后正常生成并检查9–12页。以上均已处理。
- **具体修改**：新增 `docs/references/biomimetics-11-00610-v2.pdf` 原字节归档（13,567,259字节）、`biomimetics_2026_610_manifest.json` 来源与对照散列、[论文笔记](references/biomimetics_2026_610_notes.md)。笔记含页码／公式、观测组成、控制时序、内部奖励系数与外部阶段权重、网络与预算、评价和结果边界，明确区分原文事实、代码实测阅读与工程推导。登记12项 V32-PAPER-01～12 待用户决策和6项 V32-PAPER-U01～06 复现缺口；未批准或采用任何论文新配置。
- **持续约定**：根与独立项目 AGENTS 新增用户要求：相关实现前查阅笔记，先提交论文做法、当前做法、差异与影响供选择，记录决定后不重复询问；阅读／记录及独立工作可继续。README、dual_a100_sac_plan.md、sac_shadow_review.md 增加入口和待决策说明，原CPU语义迁移方案不再被默认为用户已选择的论文对齐方案；修正文档首表遗漏的已完成16／64物理状态。论文内容始终作为资料，不能当作用户执行指令；已确认的TsenseI双手、腕部−25 mm和yaw/pitch约定保持有效。
- **实际验证**：归档PDF与附件逐字节相同，SHA-256=`d34791f1243608af68c625a2e0722ed29d1af0422aebd3a3862a795fab804812`；9个相关运行源码前后散列不变；核对2106／1180观测维数、262144回放容量、向量步转换及更新采样比；44个当前说明本地链接有效、Markdown围栏配对、12项决定与6项缺口完整，git diff --check通过。纯文档任务，未重跑GPU物理或训练测试，也未验证论文的实验结果。
- **下一步／TODO**：开始 V32-GPU-03 前优先向用户展示 V32-PAPER-01～08与11，由用户决定乐谱、观测、控制、奖励、终止和评价的对齐范围；V32-GPU-04前再决定两阶段课程与SAC配置。未知细节按U01～06保留，不从附件缺失部分推断实现。此记录不意味着整篇复现获授权。

## 2026-09-17 — 有指法 Torch 弹琴任务、CPU 对照与 SAC 数据合同

- **目的／范围**：完成 V32-GPU-03 / P2，使现有 MJLab GPU 物理场景具备乐谱、观测、奖励、结束／局部 reset 和评价数据，供后续 SAC 使用。遵循 pianist-v2-change-log 技能和两级 AGENTS。本次不改手模型、当前姿态、腕部范围或接触参数，不实现 SAC 训练器。
- **用户决定**：用户选择“先完整迁移当前规则，论文改动留作后续单独决定”；用户提供合适指法，本轮无需验证无指法分支，首版缺失／非法指法报错；实测发现错键终止折扣问题后征询，用户决定“无需使用按错键就停止训练的参数”，因此错键只参与奖励、不终止。以上写入论文决策表、AGENTS 和验收说明；后续 SAC／课程 V32-PAPER-09／10 尚待对应阶段选择。
- **问题与修复**：CPU Composer reset 会替换底层 MjData，初版导出器保留旧指针导致参考位姿不变，现每次 snapshot 重新读取数据；旧导出包移为 `*_stale_reference` 保留证据，正式任务包重新导出。Composer 的 legacy_step 更新位置／速度但保留最后积分子步的执行器力，直接完整 forward 后算功率会有时序差异；GPU 场景新增末子步力缓存，用刷新后速度计算绝对功率。原包装器 reset 的“上次动作”实际为物理 action_spec 下界，而非中立姿态；frame stack 是特征优先、初始补零，均保留实测规则。
- **终止发现与取舍**：CPU 在开启错键终止时先读 discount，再执行任务终止判断；任务内部虽设0，实际返回给训练器仍为1。用户选择不用该功能，所以未修复旧 CPU，也未迁移或暴露开启错键终止的选项。曲目完成与显式时间截断均保留 discount=1；返回两类结束标记和末帧，禁止直接替换为1-done。任务结束后必须显式局部 reset，防止自动 reset 覆盖 SAC 的 next observation。
- **实际参考**：`suite/tasks/piano_with_tsensei_hands.py` 的 before/after_step、目标与指法、各奖励及终止函数；`suite/composite_reward.py`；`models/piano/piano.py`、`models/hands/tsensei_hand.py`；`music/midi_file.py`；`rl/environment.py`、`rl/train.py`、`docs/sac_default_contract.json`；已有 `mjlab_env/bundle.py`、`scene.py`、`acceptance_cases.py` 和预览／批量验收。阅读已安装 dm_control Composer/environment 的 step 顺序、dm-env-wrappers 动作／奖励历史与 frame stack、dm_env BoundedArray.generate_value、mujoco_utils 碰撞判定及 MJLab／Warp 张量与接触接口。论文依据已归档笔记，未采纳新的论文规则。用户排除无指法前曾阅读 [SciPy rectangular LSAP 原始实现](https://raw.githubusercontent.com/scipy/scipy/v1.17.0/scipy/optimize/rectangular_lsap/rectangular_lsap.cpp) 作可行性参考；没有复制代码、实现 OT 或引入运行依赖。未读取、复制或导入 external_never_copy_only_reference。
- **离线与运行实现**：新增 `mjlab_env/task_bundle.py` 和 `scripts/export_mjlab_task.py`，冻结 T×89 目标、T×10×88 指法计数、字段顺序、奖励常数、初始历史、场景散列与 CPU 实际轨迹。导出拒绝覆盖，加载校验场景／资源散列和乐谱形状／指法一致性。导出遵循冻结场景 gravity_compensation=false，与原训练候选true明确区分。新增 `task_logic.py`、`task.py`：51动作、默认1180观测、可选3540历史、目标前瞻、五项奖励直接求和、绝对功率、十指距离、左右掌部碰撞、踏板与键指标、每世界计数和末帧。运行时仅 Torch + MJLab/Warp，不依赖 Composer/JAX；是自定义任务适配层，未接 ManagerBasedRLEnv，未声称完成训练热循环优化。
- **物理接口／预览**：`scene.py` 新增末子步力缓存、复用 env_ids 校验和空 reset 处理，不变更物理参数。`preview/scripts/preview_mjlab.py` 增加可选 `--task`，报告目标键、奖励、F1；到曲目末尾暂停，R复位、空格继续。Control 及实际激活键绿色保留；不传 task 仍为纯物理模式。无独立踏板滑块，踏板通过脚本动作验证。包说明同步。
- **验证入口与文档**：新增 `scripts/validate_mjlab_task.py`、`tests/test_mjlab_task.py`、[任务验收说明](mjlab_task_acceptance.md)、[数值验证汇总](mjlab_task_validation.json)。更新 README、preview/README、场景／批量验收、dual_a100_sac_plan、sac_shadow_review、论文笔记与 AGENTS。短曲目验收脚本兼容一帧曲目完成优先于时间限制；实际 GPU 曲目本轮为 Toy 和 Twinkle，未另测用户一帧曲目。
- **CPU/Torch 数值验收**：`MUJOCO_GL=egl training_mjlab/.venv/bin/python scripts/validate_mjlab_task.py` 的 Toy16、Toy64、Toy三帧16分别98检查通过，Twinkle16为270通过。实际命令参数见验收说明／原始日志。固定 CPU 物理快照的观测及总奖励最大误差均2.3842e−7，低于2e−5阈值；包含逐项奖励、结束与折扣对照。真实GPU以相同控制执行，离散键激活与CPU完全相同，逐帧／累计指标一致；Toy完整21帧，Twinkle前64帧对照并跑完158帧中立控制的结束流程。选定世界reset后，其他世界qpos/qvel/ctrl/time/qacc_warmstart和乐谱／历史／指标逐位不变，末帧不被reset覆盖。
- **真实轨迹误差**：最终四份报告最大观测差0.04263、总奖励差0.001823、滑动关节差0.05465 mm、铰链差0.003779 rad；分别低于0.1／0.05／2 mm／0.05 rad验收阈值。接触数值轨迹在多次运行间有小幅变化，汇总保存最终报告及其SHA-256，不把固定状态误差当作真实物理轨迹误差，也不把该容差当硬件标定标准。
- **回归与窗口实测**：`training_mjlab/.venv/bin/python -m unittest discover -s tests -p test_mjlab_task.py -v`：4项解析规则／历史／结束测试通过。`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_environment.py tests/test_preview_controls.py tests/test_mjlab_bundle.py -q`：30 passed，118个第三方警告，35.77 s。`validate_mjlab_batch.py --num-envs 64 --output training_mjlab/outputs/batch_64_after_task`：49项物理检查通过。真实 X11 的 `preview_mjlab.py --task training_mjlab/outputs/task_toy --bottom-out --duration 3` 退出0，打印0／10／20帧目标、奖励、F1，实际激活[33,54]，完整到末帧暂停并正常关闭。固定动作未跟谱，F1=0不是训练结果；未实际拖动鼠标或开始SAC学习。
- **其他执行问题**：个别辅助读取命令误用了内部Python包的相对路径、读取了不存在的本地.gitignore；随后按实际目录读取，未影响验证。没有将失败命令记为通过。旧参考指针和功率时序问题修正后才生成最终数值报告。
- **最终静态检查**：9个新增／修改Python文件语法、空白与汇总源码散列一致；10份当前Markdown说明的69个本地链接和围栏检查通过；两级日志各一条本任务记录，git diff --check通过。四份任务报告共564项检查通过，均绑定当前冻结场景和相应任务包。
- **后续／TODO**：V32-GPU-03 在用户约定的有指法范围内完成；下一步 V32-GPU-04 Torch SAC数学对照、GPU replay、采样／评估／检查点与短训练。无指法不阻塞。用户正式指法曲目、正式学习配置、V32-SAC-02精确恢复、V32-GPU-01远程A100、05双卡、06多seed学习效果尚未完成；V32-04／05／06／07／10硬件参数、接触、指尖与正式右手限制保留。本轮验证不是训练收敛证明。

## 2026-09-17 — 任务功能解释与 MJLab 指法可视化

- **目的与范围**：解释乐谱／指法／1180维观测／分项奖励／结束／独立reset／F1的用途；按用户要求在当前GPU任务预览显示指法颜色与对应目标，拇指粉红／品红。仅改变预览和说明，未改模型、乐谱、观测、奖励、终止或训练参数。
- **问题**：原 `preview_mjlab.py` 虽已读取完整任务，但只画实际按下的绿色琴键，未画 `TaskBundle.fingers`；原CPU任务的指法着色不会随冻结XML自动按乐谱更新。把键全部直接染成指法色会覆盖既有绿色激活含义；已采用未按下目标键着色、按下绿色优先、悬浮指法标记始终保留。Toy只指定左右食指，不应为展示粉红而伪造拇指分配，拇指实测使用Twinkle第10帧。
- **参考**：两级AGENTS与pianist-v2-change-log技能；`preview/scripts/preview_mjlab.py`、`preview_keys.py`；`mjlab_env/task_logic.py`、`task_bundle.py`与冻结任务清单；CPU任务 `_colorize_fingertips`／`_colorize_keys`、`models/hands/tsensei_hand_constants.py` 的既有五指配色；已安装MuJoCo3.11 `viewer.Handle.set_texts`；论文笔记的已确认迁移范围。没有新增论文对齐决策，也未读取／导入只作参考附件目录。
- **修改**：新增 `tsensei_v3_2_pianist/preview_fingering.py`，基于当前帧(10,88)指法分配绘制未激活目标键色层、十指彩色定位标记、键上指法／音名标签，窗口十指分配表含R/L、1～5、音名及0～87键号。配色沿用本项目：拇指(.8,.2,.8)、食指红、中指青、无名指蓝、小指黄；显示模块独立于Composer手包。多指同键保留各自标记，不随意选指；静音清除目标色层。
- **接入与时序**：任务预览默认 `--show-fingering`，可 `--no-show-fingering`；不传task仍为纯物理预览。初始暂停/reset显示第0帧；每步显示当前观测的待执行目标（Next target），区别于终端奖励对应的刚完成Score frame；曲末保留Final target。指法分配变化时终端也输出十指列表。仅在显示文本改变时更新文本，不添加物理步骤或全模型sync。
- **说明更新**：README、preview/README、docs/mjlab_task_acceptance.md解释各功能、1180组成、指法使用标记与精确分配的区别；明确错键会失去0.5无错键奖励，并非逐错键累扣、也不等于总奖励必负；F1评价键／踏板而非实际手指身份；补充颜色、标签、开关、按下优先和Toy指法范围。既有任务验收JSON保留为当时源码快照，本次预览新散列另记 [预览验证](fingering_preview_validation.json)。
- **实际验证**：使用 `training_mjlab/.venv` 的MuJoCo3.11。真实X11 `preview_mjlab.py --task training_mjlab/outputs/task_toy --paused --duration 12` 退出0，截图确认图例、左右食指红色目标键及十指标签；`--bottom-out --duration 3` 退出0，指法在0／10／20帧更新、末帧静音清空、真实激活[33,54]仍绿并完成曲目；`--no-show-fingering --paused --duration 1` 退出0，无指法输出。未操作或关闭用户已有预览窗口。
- **离屏核验**：读取Twinkle真实指法第10帧（右拇指C5、左拇指C4、左小指C3），生成 `preview/outputs/fingering_thumb_target.png` 和 `fingering_thumb_pressed.png`，人工查看确认粉红／品红目标。断言视觉叠加前后qpos/ctrl/model.geom_rgba不变，目标键变绿后仍保留拇指标记，静音仅剩10个指尖标记。按下截图人为设置键角只用于验证颜色优先，不作为物理接触或演奏测试。结果在preview/outputs/fingering_validation.json及docs汇总。两个修改／新增Python文件语法、当前说明本地链接和围栏、git diff --check通过。
- **TODO／边界**：本次显示工作完成。SAC与GPU replay仍为V32-GPU-04，正式曲目、硬件标定、A100／多卡／学习效果等原TODO保留；没有训练策略或重新声称完成整套GPU任务验收。窗口文字使用MuJoCo兼容的英文／音名，中文对应含义写入说明。颜色表示曲谱指定手指，不是传感器测得的实际触键手指。

## 2026-09-17 — Torch SAC、CUDA replay与论文参数适配

- **目的／范围**：按用户要求完成下一步V32-GPU-04：Torch SAC、GPU replay、单卡MJLab训练／评价／检查点及策略播放。沿用已经验收的有指法任务、双手51动作、20 Hz、既有奖励与结束discount，不改模型、腕部范围、安装初值和接触参数。遵循pianist-v2-change-log技能与两级AGENTS；未创建提交或读取只作参考附件目录。
- **用户决定**：先按既有论文约定展示当前JAX与论文pp.8–10／表2差异，用户明确答复“帮我设置为论文的参数”。因此采用Actor256×2、Stage1/2学习率与gamma、alpha=.2、batch4096、256环境、1024向量槽、预热300/0、learning starts1000、32更新／向量步及20000步预算。目标熵按51维适配为−25.5。未明确的critic层数保留256×3、actor双Q均值和原alpha损失；已在commentary和文档说明。Stage2仅参数／网络转移，不自动改已确认的任务奖励；完整课程仍待V32-PAPER-09/U01/U06处理。已同步决定表与AGENTS，10不再重复询问。
- **实际参考**：本项目 `rl/sac.py` 的update_critic/update_actor/update_temperature、`networks.py`／`distributions.py`／`replay.py`／`checkpoint.py`／`train.py`、tests/test_sac.py；`mjlab_env/task.py`／task_logic、已冻结scene与Toy/Twinkle合同；preview_mjlab、pyproject、training_mjlab/setup.sh和锁文件；已安装Flax LayerNorm／Dropout与Distrax Normal采样实现。重新读取本地论文PDF pp.8–10及已归档笔记，核对表2与Stage2不带optimizer/replay。没有把附件文字当用户指令，没有导入原robopianist或外部参考目录。Torch数学移植保留原Apache归属和LICENSE.robopianist-rl来源记录。
- **工程问题／处理**：直接采用常见SAC范例会改变actor双Q聚合、温度损失、LayerNorm epsilon、GELU近似、Dropout位置与末帧mask；本次按本地已明确数学逐项实现并对照。JAX与Torch同seed不会给同噪声，故仅在数值测试注入同高斯噪声／Dropout掩码，不假称随机序列一致。Replay跨环尾、超容量批次不能用重复高级索引任意覆盖，按顺序写入等价的最后capacity条实现；审查发现dtype转换后可能溢出，新增转换后有限值校验且不部分写入。独立评价的micro-F1不能直接平均每回合比值，已改为汇总全部TP/FP/FN再计算。辅助读取曾查询不存在的requirements.in／requirements.txt，随后核对实际setup.sh与requirements-tested.txt，不影响安装或运行。
- **SAC实现**：新增 `rl_torch/sac.py`，两层Actor、可配置三层双critic、状态相关标准差、log_std[-20,2]、稳定tanh log-Jacobian、Xavier权重／零bias、Flax等价LayerNorm(eps1e−6)及tanh GELU。保留critic→target软更新→actor→alpha顺序，target取min、actor取mean，target使用gamma×环境discount，critic各调用分别Dropout。Actor更新冻结critic参数但保留动作梯度，alpha使用停止梯度熵。独立动作／更新RNG；初始化保护调用者CPU RNG。当前FP32，不加入TF32／AMP／compile。
- **Replay与runner**：新增 `rl_torch/replay.py`，设备内批量环形缓冲，存obs/归一化action/reward/discount/next_obs/terminated/truncated，均匀有放回采样、样本独立所有权、空批次／超容量／非法输入处理。`train.py` 在插入末帧后才局部reset；按向量步、transition、update分别计数，默认固定32更新不会随环境数自动改变。独立GPU评价环境不消耗训练RNG，日志包含各损失、alpha、episode指标、实际更新／样本比、配置、依赖与源散列。默认容量262144条实际2,530,738,176字节；Torch显存不含全部Warp。
- **保存与播放**：新增 `checkpoint.py` 原子保存actor/critic/target/alpha、三个Adam、动作／更新／探索RNG与更新数；核验任务／场景散列及同阶段SAC配置。warm restart重建replay、物理、乐谱和调度；Stage2-from仅载Stage1网络，optimizer/replay/alpha重新初始化。新增 `evaluate.py` 单独整曲评价；MJLab预览新增 `--checkpoint` 确定性策略控制全部关节／踏板，保留指法和绿色键，参数冲突报错，去掉检查点恢复手动Control。pyproject声明Torch训练／评价入口，实际命令用当前项目根目录python -m，无需混装JAX进GPU环境。
- **数值参考与测试**：新增 `scripts/export_torch_sac_reference.py`，实际调用本地JAX的三个更新函数，仅替换随机采样点，输出2263个数组及原始来源SHA。不同actor／critic层数，dropout0/.01各5次连续更新；fixture给双Q相同外部mask以固定比较，生产独立抽样。新增 `validate_torch_sac.py` 和 `tests/test_torch_sac.py`。CUDA与CPU各1870项数值检查通过：CUDA网络参数最大误差2.3842e−7、梯度2.6227e−6、损失／指标2.3246e−6；包括target、log-prob、Adam一二阶矩、温度梯度及目标参数。5项unittest覆盖CPU/CUDA ring／超容量／所有权／非法写入、终止discount、饱和tanh、检查点下一步一致、Stage2清空状态、评价RNG和论文预算。
- **真实GPU训练**：本机RTX PRO 6000 Blackwell及既有Torch2.10/MJLab1.6/Warp1.17锁环境。Toy16：48向量步、768transition、66更新、32完整episode，容量512环绕，周期与最终检查点及独立评价通过（24.64s含初始化／评价）。论文规模短测：默认256环境／batch4096／32更新／262144容量，缩短到17向量步、4352transition、64更新，通过（9.33s）；此样例未到曲末，结束由Toy16覆盖，不是20000步训练或正式吞吐基准。warm restart：8向量步／128transition，继承66后新增10＝76更新，4步时间上限下32截断。Stage2转移：24向量步／384transition／42新更新／16曲目结束，通过保存和独立评价。四个最终检查点张量均有限，计数与任务／场景散列核对通过。
- **独立评价／桌面**：加载Toy16 final.pt，独立两回合完整21帧，平均键F1约0.01474、micro-F1约0.01619、踏板F1=1；训练内首次最终键F1为0，接触轨迹可能小幅不同，均明确表示尚未学会演奏。真实X11 `preview_mjlab.py --task .../task_toy --checkpoint .../torch_sac_smoke16/final.pt --duration 3`退出0，策略驱动真实物理，指法叠加与实际绿色键保留。未操作用户已有预览窗口。未把loss下降或短测当作学习收敛。
- **文档与证据**：新增 [Torch SAC验收与命令](torch_sac_acceptance.md)、[配置／原始报告／检查点散列汇总](torch_sac_validation.json)；同步README、预览说明、任务／物理验收、迁移计划、JAX参考说明、论文决策、两级AGENTS。原JAX默认合同和历史验收保持其历史身份，新Torch每次运行生成独立config。每次短测覆盖参数和原始命令在对应config.json中，当前交付源散列与运行时源快照分开记录。
- **最终静态检查**：10个新增／修改Python文件语法与空白检查通过，11份当前说明86个本地链接和围栏检查通过，pyproject入口可解析，git diff --check通过；源与检查点散列写入验证汇总。
- **TODO／下一步**：V32-GPU-04单卡功能验收完成；V32-GPU-05双卡／A100资源与传输基准、06正式曲目长训练／多seed演奏质量仍待实施。V32-SAC-02仅warm restart，精确恢复缺replay／模拟器／乐谱／调度状态；V32-SAC-01正式曲目与质量目标待落实。完整论文奖励课程及未披露细节继续按09/U01/U06单独讨论。V32-04／05／06／07／10硬件动力学、接触、指尖和正式右手限制不变。未执行完整20000向量步实验、远程A100或双卡训练。


## 2026-09-17：回放旧 JAX 训练中最好的已保存检查点

- 目的／范围：用户提供 `runs/sac-20260917-111902-962786/step-100000.msgpack`，要求该训练最优结果的视频。按日志按键 F1 选择 90000 步，未启动训练或修改当前训练配置。
- 实际问题：附件为旧 JAX／Flax SAC 格式，不能使用新 Torch SAC 加载器；当前腕部下限与接触配置已改变，直接构建环境会与旧检查点不兼容。100000 步不是本 run 最佳 F1。
- 参考：run 的 `config.json`、`metrics.jsonl`、检查点元数据；`rl/{checkpoint,environment,train,sac,specs}.py`；`suite/tasks/{base,piano_with_tsensei_hands}.py`；`models/hands/tsensei_hand_constants.py`；`mjlab_env/bundle.py` 和 `training_mjlab/outputs/scene_before_hard_contact_20260917/manifest.json`；`preview_{backend,keys,fingering}.py`、`preview/scripts/preview_pose.py`；论文决策表 `docs/references/biomimetics_2026_610_notes.md`。同仓 V3 的 base.py 仅用于比对历史注释，无运行时依赖或跨项目导入。
- 修改：新增 `preview/scripts/render_jax_checkpoint.py`，在临时独立包副本中恢复两处历史源码；恢复结果的 SHA-256 与训练记录逐字节一致，并严格核验全部已记录源码及完整 runtime contract 后载入策略。`--best` 选择最高日志 F1（奖励打破平局），输出新目录防覆盖。使用 GPU EGL 渲染绿色激活键、指法颜色／标签、当前计分目标和时间／指标；保存视频、截图、轨迹及 JSON 报告。更新 `preview/README.md`。
- 验证命令：`training/.venv/bin/python preview/scripts/render_jax_checkpoint.py --checkpoint runs/sac-20260917-111902-962786/step-100000.msgpack --best --output preview/outputs/replays/sac-20260917-111902-962786_best`。实际用 NVIDIA RTX PRO 6000 EGL，JAX CPU／MuJoCo 3.13.0；158 步正常曲终，F1=0.3231676200030628、return=398.8167999982834，precision、recall、踏板指标及 micro-F1 也与原评估一致。历史 physics_dt=.002、control_dt=.05、Wrist_TZ −.04～0；当前工作区参数未回退。
- 视频验收：`ffprobe` 确认 H.264、1280×720、20 FPS、178 帧／8.9 秒（演奏 7.9 秒＋末帧 1 秒）；检查中间截图，钢琴、双手、激活绿键与指法图例可见。视频无音轨，不使用原曲音频掩盖错键；回放指标不代表收敛。运行中仅出现依赖弃用警告，不影响结果。
- 未解决事项：本次视频交付无新增阻塞 TODO；既有 V32-02/04/05/06/07/10 模型标定事项与 V32-PAPER 决策保持不变。脚本只支持可逐字节恢复的已知历史源码，其他旧版本需提供对应历史文件；当前任务未要求 Torch 视频或音频合成。


## 2026-09-17 — 核验 −25 mm 腕部范围并运行旧版小星星训练

- **目的／范围**：用户要求确认当前 wrist_tz 为 [-0.025,0] 后再开旧版单环境 CPU MuJoCo + JAX SAC 小星星训练。预算沿用上轮给出的 100000 步，seed=42，从头初始化；不用旧限位检查点。保持当前20 Hz、现有有指法任务奖励、重力补偿和关闭错键终止。
- **参考与决策**：阅读论文笔记及 V32-PAPER 决策表、torch_sac_acceptance.md、手常量、rl/environment.py 和 rl/train.py；根／子 AGENTS 与 pianist-v2-change-log 技能。论文 pp.8–10／表2的新SAC配置已经用于Torch；本次用户明确选择旧版，因此保留JAX 256×3、gamma=.8、batch256、三个lr=3e-4、alpha初值1，不擅自改成论文网络和向量预算。
- **遇到的问题／处理**：检查时发现本次同配置运行已于15:36:47启动，PID2149105，并有 launch.json 和源码快照，因此直接核验这一运行，未重复启动相同任务。首次进程列表过宽导致工具输出截断，随后按训练模块过滤。既有 MJLab 进程保持运行。仅有第三方弃用警告，未观察到启动错误。
- **运行配置与位置**：`runs/legacy-twinkle-tz025-20260917-153647/launch.json` 记录完整命令、CPU环境变量及PID；实际命令使用 `training/.venv/bin/python -u -m tsensei_v3_2_pianist.rl.train --midi TwinkleTwinkleRousseau --max-steps 100000 --seed 42 --no-wrong-press-termination --output-dir .../training`。预热5000步，每10000步评估及保存；输出config、metrics、step-N.msgpack和完成后的final.msgpack。源码快照位于该run的source_snapshot；本次没有修改模型、算法或限位。
- **实际验证**：使用旧版training/.venv、MUJOCO_GL=egl、JAX_PLATFORMS=cpu真实创建环境并reset，分别断言左右MuJoCo joint range和actuator ctrlrange为[-.025,0]（容差1e-12），初始qpos均0；整个运行contract与当前重建contract完全相同，包含全部已记录源散列。设备cpu:0，51动作、1180观测、physics_dt=.0005。结果保存到该run的wrist_verification.json。进程存活且日志已超过3600采样步；当时处于预热阶段，不把预热回报当SAC学习效果。
- **文档修改／TODO**：更新README运行说明和两级日志。V32-SAC-01／GPU-06的收敛与演奏质量仍需完整训练后评估；本次尚未完成100000步。V32-SAC-02仍仅warm restart；硬件、碰撞和正式右手等TODO保留。本次未新增阻塞项。

- **交付前复核**：PID2149105仍在运行，最新已记录5688步／688次SAC更新，已越过预热并进入实际学习；git diff --check通过。训练继续后台运行，完整预算和收敛尚未完成。


## 2026-09-17：MJLab step-3000 完整回放视频

- 目的：回放正在训练的 run `sac-20260917-150810-595009` 的指定 `step-3000.pt`（768000条采样），不替换为其他检查点。
- 问题：现有 MJLab 预览只有桌面窗口，旧视频脚本仅支持 JAX `.msgpack`，缺少 Torch GPU 完整视频导出；回放实测指标与历史评价存在小差异。
- 参考：`preview/scripts/preview_mjlab.py`、`render_jax_checkpoint.py`；`rl_torch/{checkpoint,train,evaluate,sac}.py`；`mjlab_env/{task,scene,task_logic}.py`；`preview_{backend,keys,fingering}.py`；run config／metrics、任务与场景 manifest；论文决策登记。保留已确认任务与论文SAC参数，本次不修改算法或奖励。
- 修改：新增 `preview/scripts/render_mjlab_checkpoint.py`，校验检查点／场景／任务散列，独立单环境CUDA确定性演奏；CPU只接收渲染快照，不推进物理。复用绿色激活键和指法图层，指法目标与计分帧一致；输出MP4、截图、报告和NPZ轨迹／分项奖励。更新 `preview/README.md`。运行中的训练进程未停止，训练代码／配置未改；同GPU期间短暂共享算力。
- 实际验证：`training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py --checkpoint training_mjlab/outputs/sac-20260917-150810-595009/step-3000.pt --output preview/outputs/replays/sac-20260917-150810-595009_step-3000` 完成158步整曲；NVIDIA EGL硬件渲染。F1=0.8436105251，precision=0.8775315881、recall=0.8734176159，return=486.2795104980；踏板F1=1。训练日志F1=0.8450871706、return=486.33456420898，不标记为完全匹配。
- 视频验证：ffprobe读取 H.264／1280×720／20 FPS／178帧／8.9秒（7.9秒演奏＋1秒末帧）；ffmpeg全片解码无报错，实际检查截图双手、绿键与指法目标可见。无音轨。
- 额外验证：使用现有 `python -m tsensei_v3_2_pianist.rl_torch.evaluate` 对相同检查点做无渲染整曲评价，输出 `evaluation_without_render.json`，F1=0.8369347453、return=485.2206115723。与训练时记录的所有 learner／任务源码散列比较，无变化。
- TODO(V32-SAC-03)：不同进程确定性策略评价存在小幅轨迹／指标差异；尚未隔离物理计算顺序、初始化历史或执行设置的贡献。后续需要逐帧状态／动作对照才能定位，不声称已证明是GPU浮点误差。此项不阻塞交付本次真实回放视频，但不承诺逐位重复；既有模型标定TODO保持。


## 2026-09-17：正面全钢琴正交视角与实际按键音轨

- 需求与后续修正：用户要求看清手指方向、去除鱼眼感并听到实际弹奏；随后明确必须看见全部钢琴。默认改为正面略俯视的全琴正交投影，另保留同轨迹手部近景。
- 实际问题：旧视频是斜视透视且无音轨；本机缺少FluidSynth原生库。相机探测发现当前MuJoCo自由相机会用 `model.vis.global_.orthographic` 覆盖抽象相机标记，单设 `MjvCamera.orthographic` 仍然透视；正交时fovy单位为世界长度。最初近景不满足用户随后补充的全琴要求，已改默认全琴并重渲染。
- 参考：`preview/scripts/{preview_mjlab,render_mjlab_checkpoint}.py`、`preview_{keys,fingering,backend}.py`、`mjlab_env/{scene,task,task_bundle}.py`、`models/piano/midi_module.py`、step-3000检查点／报告／已保存轨迹；论文决策登记（未改变任何任务／SAC设计）；FluidSynth官方音频及MIDI API（见preview/README链接）；Ubuntu包版权文件。
- 修改：新增 `preview_camera.py` 共用相机配置，默认完整88键／琴体与双手，支持front／performer正交、overview透视和full／hands取景。更新桌面与视频入口相机参数。新增 `preview_audio.py` 实际物理键沿／踏板事件、MIDI与采样钢琴WAV合成；回放使用已有substep_callback，只在独立实例采集0.5 ms子步，无训练代码修改。新增 `setup_piano_audio.py` 本地解包依赖；新增 `reframe_mjlab_recording.py` 从已保存真实轨迹换视角，保留同一音轨；同步预览说明与依赖版权信息。
- 输出：`preview/outputs/replays/sac-20260917-150810-595009_step-3000_full_ortho_audio/` 和 `..._hands_ortho_audio/`。中间透视测试输出报告已纠正标记，不作为最终版本。
- 验证：运行新版render脚本对step-3000完整158步，收集15800子步、99次note-on；每个控制边界的子步键激活与任务actual数组一致。4项音频测试覆盖长按不重触发、松开重按、和弦／非目标音不被过滤、踏板／曲终释放与静音。实际 `mjv_updateScene` GL相机测试确认正交标记及投影高度，1项通过。
- 媒体验收：最终两个版本复用逐字节相同WAV与轨迹；H.264 1280×720 20 FPS、179帧／8.95秒；AAC双声道48 kHz同为8.95秒；ffmpeg全片解码无错误，MIDI由mido独立解析为99个note-on、7.9秒事件范围。WAV峰值15595、RMS约2242（int16），没有削波；全景截图中88键与两侧琴体完整，近景用于看手指。
- 保留事项：TODO(V32-SAC-03)跨次GPU回放差异仍存在，本次F1=0.81823385、return=481.02084351，未宣称匹配旧日志或证明差异来源；两个最终视角之间没有重跑物理。新增TODO(V32-AUDIO-01)：音符时序／按键／踏板来自仿真，velocity固定127沿用原CPU MIDI模块，力度映射与真实声学未标定；当前提供采样音色而非实琴录音。未加入桌面实时发声。训练进程保持运行，奖励／任务／物理源文件未更改。

## 2026-09-18：最新最佳训练视频与常用命令速查

- 目的：导出最新正式MJLab训练中最好的已保存策略视频，并建立可复制使用的训练／导出视频／姿态预览命令文档。
- 实际发现：最新正式run仍为 `sac-20260917-150810-595009`，已正常完成19532向量步／5,000,192采样／593,056更新，用时10287.11秒；当前无训练进程。20个有效日志评价中step-18000的按键F1=0.9173900485最高，final.pt为0.8860760331，不能默认最后权重最好。原文档中的操作分散在多处，Python环境和源姿态／冻结GPU场景容易混淆。
- 参考：本run `config.json`／`metrics.jsonl`／`summary.json`；`render_mjlab_checkpoint.py`、`preview_mjlab.py`、`preview_pose.py`、`reframe_mjlab_recording.py`、`preview_camera.py`；`rl_torch/train.py`／`evaluate.py`；场景／任务导出脚本；`docs/torch_sac_acceptance.md`、预览README及论文决策登记。保留用户已确认SAC和任务参数，不新增论文对齐决策。
- 修改：新增 `docs/common_commands.md`，覆盖环境选择、当前最佳、自动／指定检查点视频、同轨迹换镜头、小星星5M训练、日志检查、策略窗口、手动姿态／坐标轴／保存、新场景任务导出、独立评价和warm restart；明确目录覆盖行为、冻结场景和源姿态关系、不是精确续训。README与preview/README新增入口。新增 `preview/scripts/export_best_mjlab.py`，在指定run中按已保存检查点的日志F1选优（平局按奖励、再按较晚步），支持完成run的final、忽略尚未写完的日志末行；默认正面全琴有声音轨、唯一时间戳目录，`--select-only`不分配GPU。新增两项检查点选择测试。
- 视频实际命令：`training_mjlab/.venv/bin/python preview/scripts/render_mjlab_checkpoint.py --checkpoint training_mjlab/outputs/sac-20260917-150810-595009/step-18000.pt --scene training_mjlab/outputs/scene --task training_mjlab/outputs/task_twinkle --camera-view front --camera-framing full --audio --output preview/outputs/replays/sac-20260917-150810-595009_best-step-18000_20260918`。完整158步，F1=0.9167571068、precision=0.9847046137、recall=0.9024262428、return=510.3649597168；与日志有小幅差异如实保留。选择依据另外保存selection.json。
- 验证：两项选择测试通过，覆盖较晚但较差、缺失权重、完成final映射、日志半行和无评价拒绝；真实run的select-only选中step-18000。ffprobe验证H.264 1280×720 20 FPS 179帧／8.95秒及AAC 48 kHz双声道同长；ffmpeg完整解码通过，实际检查全琴与双手截图。15800子步共43次note-on，158个控制边界键激活与视频轨迹一致，音频峰值0.3071（浮点）、无削波。CLI帮助与训练inspect-only核对命令参数及5M预算；未启动新训练、未修改权重／任务／物理参数。
- TODO：本次无新增阻塞项。V32-SAC-03跨次GPU回放差异、V32-AUDIO-01力度／真实声学未标定及既有手模型标定事项保持，文档明确说明。选择的“最佳”指日志最高F1，不声称所有检查点重新多seed评价后仍最佳。


## 2026-09-18：训练耗时分解与step-18000开头自碰撞定位

- 需求：解释训练慢，定位最佳视频前几秒大幅运动是否由碰撞造成。无授权变更奖励／论文未决设计，本轮仅诊断。
- 实际发现：正式训练5M用时10287.11秒，593056次SAC更新。独立256环境／4096batch／32更新短测每向量步482.79 ms，物理242.17 ms（50.16%）、学习含采样237.46 ms（49.19%）；任务与replay写入很小。确认本地MJLab已有单物理步CUDA图，不能误称整个物理没有CUDA图；100子步循环与eager SAC仍有额外开销。
- 接触发现：保存视频初始qpos无手部负距离接触，保持初始控制0.5秒的5ms采样没有手部负距离接触。第一命令26个手指伺服及双腕yaw要求饱和驱动力，右中指DIP／食指MCP目标改变76.9°／73.8°。原始视频0.15秒右拇指PIP与食指swing几何重叠约2.23mm，前三秒最大3.34mm。GPU固定动作诊断重放确认右掌与小指根、拇指与食指根、左无名指与小指根自接触，单接触峰值约12.9～20.7N；未采到双手互撞／手撞琴壳。不能将所有大幅运动仅归因于碰撞，未做去碰撞因果消融。
- 问题／限制：GPU固定动作重放前三秒仍与原视频最大平移差4.60mm／转角差0.260rad，力／时间属于诊断重放，几何另用原始保存状态核对；接触每5ms采样，非全子步覆盖。标注图第一次因NumPy非标量int转换报错，改用bodyid[0]后生成。网格凸包和驱动未硬件标定，不据此自动屏蔽自碰撞。
- 参考：run summary／metrics、最佳视频rollout；`rl_torch/{train,sac,replay}.py`、`mjlab_env/{scene,task,task_logic,bundle}.py`、手模型／常数／执行器代码；本地 `mujoco_warp/_src/support.py:contact_force` 读取真实GPU接触力、`mjlab/sim/sim.py:create_graph/step`；论文说明pp.11～12式16～20及决策登记。
- 修改：新增 `scripts/profile_mjlab_training.py`（独立进程内learner短测，不保存权重）、`scripts/diagnose_replay_start.py`（保存动作GPU重放、5ms接触力、初值控制基线、原视频几何与首帧命令分析）；新增 `docs/training_speed_and_startup_diagnosis.md` 和诊断JSON／碰撞标注PNG；常用命令文档增加入口。
- 验证：实际profile12轮、GPU重放前60控制帧＋10帧初始控制；读取报告与原qpos核对上述指标，标注图实际目视检查。统计首次力矩需求时使用位置误差×kp与执行器forcerange比较。测试均未改变训练源文件／场景／任务／权重。性能短测显式同步且replay由16组有效预热批次重复填充，不宣称历史训练的精确分解。新增脚本语法／help检查通过。
- TODO：V32-04/05驱动力／速度限制标定、V32-06自碰撞几何校验、V32-SAC-03复播差异继续存在；若启用插值、速度／加速度／自接触／姿态奖励，按V32-PAPER-04/07/09由用户决定，本次未改。无新增编号或自动参数切换。

## 2026-09-18 — 多子步整体执行、编译／融合学习器与手内碰撞奖励审查

### 目的与范围

按用户要求实现多物理子步整体执行、Torch学习器编译、融合优化器；新增同手自碰撞惩罚，并按论文与TsenseI模型整理以rew.总表开头的确认文档。用户要求先审查奖励，候选权重0.2保持可选，默认0；没有启动新的5M长期训练。

### 遇到的问题与依据

- 当前Warp1.17不能在capture内直接launch现有MJLab单步graph，实测CUDA error900。改为直接捕获100次原生`mjwarp.step`，保留0.5ms物理步长和20Hz控制。新代码在MJLab重建graph后重新捕获；录音／诊断回调和NaN guard使用逐步路径。
- 编译函数直接持有Generator会增加编译障碍；将noise/dropout显式输入，在区域外按原顺序使用update RNG生成。只编译张量目标及backward，保留模块参数名以兼容已有checkpoint。
- 旧Adam状态中的fused标志和step tensor位置不能直接沿用新runtime选项；load时规范化group和step设备。算法配置／学习率／更新顺序不变。
- 原forearm_reward只罚指定左右掌部碰撞，漏掉同手的拇指—食指、小指—掌等。本次按同手不同非焊接body对统计100子步接触并集，mesh多接触点和反向pair去重，避免按mesh细分数量反复扣分；不罚手—键接触。
- 论文p.11式19没有完整接触名单，式16–18的动作单位及Stage2 shaping也未完整披露。自碰撞建议`−0.2N`不含论文常数+0.2，其余正则项只作具体设计等待确认；不把工程适配称为论文完整复现。
- 一次辅助编辑调用使用了本机不存在的`python`命令，未执行写入；已改用`python3`，项目执行均使用明确venv。

### 实际参考

`mjlab_env/{scene,task,task_logic}.py`、`rl_torch/{sac,train,checkpoint,evaluate,replay}.py`、预览render／preview脚本、`scripts/profile_mjlab_training.py`、`tests/test_torch_sac.py`及冻结JAX参考、`training_mjlab/outputs/scene/{manifest.json,scene.xml}`、包内`assets/urdf/daxian_left.urdf`、`docs/references/biomimetics_2026_610_notes.md` pp.10–12／式12、15–20／表3。另阅读本机MJLab`sim.py`与`nan_guard.py`、mujoco-warp`types.py`；未从外部参考目录复制代码。

### 具体修改

- `mjlab_env/scene.py`：100子步整体capture、逐步fallback、reset隔离；新增`self_collision.py` GPU接触并集核。
- `mjlab_env/task.py`／`task_logic.py`：可选`self_collision_weight`、reward／上次reward观测／return／info接通，保留原五项与错键不终止语义。
- `rl_torch/sac.py`：编译actor／critic目标及反向、外部显式随机量、三组fused Adam与读档runtime规范化。
- `rl_torch/train.py`：三项默认开启、可分别关闭；自碰撞默认0；config和checkpoint记录选项，metrics增加分项reward。
- `checkpoint.py`／`evaluate.py`／预览render与窗口脚本：旧checkpoint缺省0，验证reward权重；播放自动恢复新reward权重，并导出包含新增项的reward数组，避免只改训练不改评估观测。
- 新增GPU行为测试、自碰撞真实动作验收脚本；扩展SAC数值验证及速度脚本开关。README、常用命令、论文决策补记、`docs/acceleration_acceptance.md`和`docs/reward_design_review.md`同步。

### 实际验证

- `training_mjlab/.venv/bin/python scripts/validate_torch_sac.py --compile-learner --fused-optimizer ...`：1870项通过，参数最大绝对误差2.3842e−7；dropout0／0.01各5更新。结果`training_mjlab/outputs/acceleration_validation/sac_compiled_fused.json`。
- `python -m unittest tests.test_acceleration_and_collision tests.test_torch_sac tests.test_mjlab_task -v`（上述venv）：13项通过，覆盖graph／逐步中立数值、局部reset、100次callback、graph重建、接触去重／环境隔离、reward进观测、奖励合同和fused/eager读档。
- 256环境、batch4096、32更新／向量步，12轮稳态隔离benchmark：493.232→376.510ms／向量步，519.0→679.9transition/s，吞吐约+31%；学习器245.672→126.452ms。物理244.405→246.897ms，**没有明显提速**。首次学习器预热0.355→7.367s，含本次缓存条件下编译；不当作长期训练实测。详细原始JSON与限制见加速验收文档。
- `scripts/validate_self_collision.py`回放已有step18000动作60步：46周期碰撞，25对不同部件，每周期最多7对，权重0.2合计约−23；第3周期独立读取逐子步接触表计数2，与GPU一致；reset清零。该数据不表示新奖励训练效果。
- 新建smoke_collision：4环境×6步、24transition、10更新、8次时间截断reset、2次独立评价并保存。smoke_resume恢复后8transition、2更新，总更新10→12。三项加速及权重0.2贯穿训练／保存／加载。没有改已有正式run或checkpoint。

- 新reward checkpoint完整有声回放通过：158控制步、15800音频子步、179视频帧／8.95s；6个分项逐帧和等于reward，自动恢复权重0.2，ffprobe确认H.264 1280×720＋AAC。验收产物`preview/outputs/acceleration_smoke_collision/`，仅为未充分训练的测试策略，不替代正式最佳视频。

### 未解决／待确认

V32-REW-01～05见reward确认表：自碰撞权重／允许接触白名单，速度与加速度／自然姿态，正功率及dt尺度，超速阈值／腕pitch／等效按压角。平滑等项**设计已给出但没有实施**。原V32-SAC-03长时GPU重播漂移、V32-04/05驱动与速度标定、V32-06接触标定及正式右手来源仍保留。未做长期新奖励收敛、实际硬件、桌面交互验收。

## 2026-09-18 — Torch SAC训练默认开启进度条

- **问题与目的**：原runner仅写metrics.jsonl及结束summary，长时间训练时终端缺少进度／ETA。首次读取依赖文件用了不存在的requirements.txt，随后确认实际文件为requirements-tested.txt，已包含tqdm==4.70.1，无需安装新依赖。
- **参考**：`rl_torch/train.py`训练循环、`training_mjlab/requirements-tested.txt`、`docs/common_commands.md`；核对论文决策登记，本次只是终端显示，不改变论文参数、任务规则、奖励或更新次数。按pianist-v2-change-log技能维护日志。
- **修改**：训练入口默认开启tqdm，新增`--progress`／`--no-progress`；主进度为vector_steps×num_envs条样本，显示计数、百分比、时间／ETA、速率、向量步、更新次数、replay大小及最近eval_F1。增加初始化提示和首次编译／评价／保存阶段提示；上下文管理保证退出时关闭显示。仅读取已有Python计数，不为显示新增GPU同步；常规刷新最小间隔1秒，阶段切换立即刷新。同步常用命令第12节。
- **验证**：MJLab venv默认开启运行`progress-smoke-20260918`，2环境×4向量步，8条transition／3次更新完成；终端到达100%及8/8，包含首次编译、周期评价、step-3保存和最终评价，生成final.pt／summary。保留原FP32设置，Torch出现建议TF32的提示，不据此改变精度。另验证`--no-progress`关闭显示，见对应progress-disabled-smoke运行。仅为短程功能验收，不是收敛测试。
- **未解决**：无新增TODO。ETA会受首次编译／评价影响；100%表示样本预算完成，最终评价／保存可能仍在继续。旧运行中的Python进程不会动态载入本改动，需在下次启动时生效。

## 2026-09-18 — 预览姿态保存快捷键 S 改为 P

- **问题／目的**：按用户要求，将当前V3.2姿态预览的保存键从S更换为P。
- **参考**：`tsensei_v3_2_pianist/preview_runtime.py`键盘事件分派、`preview/scripts/preview_pose.py`参数提示，以及README、preview/README.md、docs/common_commands.md；沿用pianist-v2-change-log记录约定。
- **修改**：保存回调仅绑定P／p，S／s不再触发姿态保存；同步终端、--help和三处使用说明。保存内容、备份、文件路径及关闭窗口保存选项保持原逻辑；历史日志中的旧快捷键不改写。
- **验证**：`.venv/bin/python preview/scripts/preview_pose.py --help`已显示P；两个修改的Python文件通过语法解析，确认保存分支只有P／p。未人工操作GUI，也未写入实际初始姿态配置。
- **未解决**：无新增TODO；已打开的预览窗口需要重启后使用新快捷键。

## 2026-09-18 — 论文全部公开奖励可独立启用，未知映射由用户填写

### 用户决定、范围与问题

用户要求按论文实现所有奖励、由用户自己选择启用；随后明确选择“实现公式接口，但未确定的映射必须由我填写才能启用”。本轮实现所有公开公式和表3权重，保留原命令默认规则；未代填腕方向、等效按压角、自然参考或不期望接触类别。Stage2 shaping未公开，不能补猜成某个函数；必须显式选identity才能用公开的未shaping正则。

实际开发中，一次测试文件编辑误用了相对于外层目录的路径，导致补写未执行、旧测试半成品缺少必要映射而报错；改为绝对路径、完成fixture后通过。该错误未改变生产配置。发现新自碰撞配置需要与原alias分开计算，避免重复叠加；增加GPU测试覆盖自定义dt下只计一次。

### 实际参考

- 直接读取用户附件`biomimetics-11-00610-v2.pdf` pp.10–12（pdftotext）：式12–20、腕pitch／前向按压文字、表3及Stage2 shaping注释；同步核对`docs/references/biomimetics_2026_610_notes.md`。
- `mjlab_env/{scene,task,task_logic,bundle,self_collision}.py`，`rl_torch/{train,checkpoint}.py`，回放脚本的reward_options／reward_term_names入口。
- 当前冻结scene/manifest.json、scene.xml中的action_names／neutral_action／body名称；包内手模型的镜像pitch约定；现有reward审查文档及测试fixtures。
- 使用pianist-v2-change-log维护约定；论文只作事实参考，未从external_never_copy_only_reference复制或导入代码。

### 修改位置与原因

- 新增`mjlab_env/reward_config.py`：20项完整schema、独立enabled／weight／params、Stage1/Stage2表3模板、缺失映射门禁、重复惩罚及非法数值检查、legacy等价配置识别。
- 新增`mjlab_env/rewards.py`：论文按键／和弦／指法、9项分组功率与命令差分、超速、碰撞、自然姿态、腕pitch、等效角及dt计算；双手动作按命名分组，踏板排除；每环境独立两步历史，reset以真实neutral_control填充。
- 扩展`scene.py`／`self_collision.py`：用户选择同手／跨手／琴箱类别及显式忽略body对；仍按全部子步的body对并集计数，正常手键接触排除，原self-only路径兼容。
- `task.py`／`task_logic.py`：配置接入真实reward、上次reward观测、episode return，提供raw／weighted／实际贡献及接触计数；全部关闭返回0；原规则与self_collision_weight等价配置仍走原计算分支。
- `train.py`：`--reward-config`、可重复enable/disable单项开关；config与checkpoint保存完整内容，训练日志分开记录内部公式值／外部权重后／dt后分项。
- `checkpoint.py`：校验完整奖励合同，旧checkpoint按原规则解释；评估／预览／视频通过现有reward_options入口自动恢复配置，不依赖原JSON路径。`--allow-reward-change`仅与networks-only的Stage2转移配合，记录旧配置并新建optimizer/replay。
- 新增四份`configs/rewards/*.json`模板；新增`tests/test_paper_rewards.py`、扩展GPU计数测试及`scripts/validate_reward_options.py`。所有已填写映射的验收配置明确为test_fixture_only，不是用户选择。
- 重写`docs/reward_design_review.md`，开头保留rew.总表；新增`docs/reward_options_acceptance.md`。更新README、预览说明、常用命令第13节、论文决策记录与内外AGENTS.md中的用户映射决定。

### 实际验证

- 26项单元／GPU检查通过，包含解析公式、开关／权重／dt、负功率、命令历史及局部reset、所有关闭、缺失映射拒绝、碰撞去重／类别／忽略项和checkpoint合同。结果`training_mjlab/outputs/reward-validation-20260918/tests.log`。
- 2环境×6控制步，17个启用分项的CPU／GPU逐项对照最大误差2.9802322e−8；局部reset后计分索引[0,3]，未reset环境的两步历史保持；legacy配置原分支确认。验收脚本不读取原项目或外部参考目录。
- Stage1短测4环境×6向量步，24transition／10更新／8次截断reset／两次独立评价；Stage2显式换奖励仅网络转移后8transition／2次新更新，总更新2，来源原10更新和完整旧配置已记录。原始／加权／dt后日志逐项核对通过。
- 新奖励checkpoint完整有声回放通过：158×17分项、逐帧和等于总reward，15800实际音频子步，8.95s、H.264 1280×720＋AAC。自动恢复完整配置，未手动重填；产物`preview/outputs/reward-options-smoke-20260918/`仅为接口测试策略。
- CLI inspect-only实测paper Stage1按键／指法＋三个正功率项；无GPU分配，不触发未填写映射。

### 未解决／后续

V32-PAPER-U01/U02/U05仍保留：缺作者Stage2 shaping、动作单位细节、碰撞与角度／自然姿态映射。公开公式接口已完成，不等于这些资料缺口已解决。V32-REW-01～05中“代码未实现”的部分由本轮可选实现替代，具体用户启用选择与真实标定仍待用户配置。继续保留原硬件速度／力矩／接触／正式右手及长时GPU漂移TODO。未运行5M长期新奖励实验，未宣称收敛、F1提高或碰撞消除，未人工验收GUI。

## 2026-09-18 — 导出当前最佳小星星训练视频（step-15000）

- **目的／范围**：按用户要求找到当前MJLab＋Torch SAC训练最好的已有结果，导出全钢琴正面有声视频；训练进程继续运行，未改代码、模型、奖励、检查点或训练配置。
- **参考**：`preview/scripts/export_best_mjlab.py`选优规则、`render_mjlab_checkpoint.py`回放／音轨／报告、`rl_torch/checkpoint.py`奖励恢复、两轮sac目录的config.json与metrics.jsonl、`docs/common_commands.md`，以及论文记录V32-PAPER-11已决定保留当前评价口径。按pianist-v2-change-log技能记录。
- **筛选**：按相同scene/task散列、完整曲目评价、有保存文件的检查点比较按键帧均值F1；两轮正式训练共有35个可选择检查点。新轮`sac-20260918-135539-423462/step-15000.pt`为0.9297469854，高于旧轮step-18000的0.9173900485。新轮对应384万transition／448032次更新，自碰撞权重0.2。不同奖励配置不以return作跨轮优劣依据。
- **遇到的问题**：初步扫描包含旧短测，其日志有缺return或无对应已评估检查点；依据任务散列／完整曲终及保存文件筛选排除，不修改历史日志。导出运行成功；回放F1与日志不完全相同，沿用V32-SAC-03记录，不把两者混称一致。
- **产物／文档**：`preview/outputs/replays/sac-20260918-135539-423462_best-export-20260918/`保存replay.mp4、played.wav／MIDI、实际按键事件、15800子步轨迹、rollout.npz、report.json、selection.json；另存comparison.json记录跨轮筛选、frame-check.png用于人工查看构图。更新`docs/common_commands.md`第1节到当前检查点及视频链接。
- **实际命令**：`training_mjlab/.venv/bin/python preview/scripts/export_best_mjlab.py --run-dir training_mjlab/outputs/sac-20260918-135539-423462 --output preview/outputs/replays/sac-20260918-135539-423462_best-export-20260918 --audio`。使用现有NVIDIA EGL硬件渲染、确定性tanh(mean)策略；未启动新的训练。
- **验证**：完整158控制步曲终，F1=0.9314348102、return=512.7066650；checkpoint散列f0cb316b8687a9a98d8ca7ff417768dfa2c887b53665c0e869f4aaa6aacd9e24。ffprobe确认H.264 1280×720＋AAC、8.95秒；179帧，实际音轨46次note-on、非静音RMS约0.048。查看3秒截图确认全琴与双手入镜、按键绿色及指法颜色。音频基于实际键状态，非目标乐谱配音。
- **未解决**：V32-SAC-03跨次GPU回放差异仍存在，本次比日志高约0.169个百分点；V32-AUDIO-01音量力度未标定（固定velocity127）。本次最佳为导出时已有结果，训练尚未结束；不宣称新论文奖励已训练或所有seed最优。

## 2026-09-18 — 定位step-15000起始前冲、降低正面视角、复测训练瓶颈

- **目的／范围**：用户要求分析刚导出视频的首帧前冲、从演奏者前方看清离键高度，并判断训练是否达到速度极限。本轮修改显示与诊断工具，未调整控制／奖励／reset／模型／训练参数；继续按pianist-v2-change-log记录。
- **参考**：`preview_camera.py`、`reframe_mjlab_recording.py`／`export_best_mjlab.py`／`render_mjlab_checkpoint.py`；`mjlab_env/scene.py`／`self_collision.py`、`rl_torch/sac.py`；`diagnose_replay_start.py`／`profile_mjlab_training.py`；step-15000原视频rollout/report、冻结scene动作轴／初始控制、当前run的metrics/config/summary；论文记录pp.6–11及V32-PAPER-04/07/08/10已有决策；旧性能验收报告。
- **问题与定位**：第一条绝对目标直接跳变，右腕TX／TY目标+85.04／+71.50mm，50ms后实际+27.84／+22.37mm；26指执行器与4腕转动执行器初始误差要求超过力矩限值。初始无负距离手接触，保持初始控制0.5s也无；逐0.5ms检查前100ms，0.5ms已运动，首次手键接触在8ms，首100ms无手内／跨手／琴箱接触。首秒每5ms采样首次自碰撞在0.430s，因此主因是主动命令跳变、后续触键反作用，而非初始穿插弹出。所有力与速度为仿真值，不宣称实机标定。
- **代码改动**：`preview_camera.py`将front默认−60°改为−30°，仍正交／全琴；其余视角不变。最佳导出与轨迹重构两个脚本新增elevation／distance透传。`diagnose_replay_start.py`新增正整数`--sample-every-substeps`，默认10保留，1时逐物理步取证；报告使用实际采样间隔。
- **视频**：`preview/outputs/replays/step15000_front_height_20260918/`重构原轨迹，相机−30°，158控制步＋初始／末停留，共179帧／8.95s，ffprobe确认H.264 1280×720＋AAC；查看preview.png确认全琴及双手可见。rollout／WAV／音频子步／事件与原录制SHA-256相同，保存在preservation_checks.json。
- **性能实测**：新训练已完成512万transition／608032更新，总8778.72s，约583transition/s。无其他训练或回放进程时，单独12轮profile，256环境／batch4096／32更新／自碰撞0.2／三项加速开启，物理252.09ms（63.53%），学习140.92ms（35.51%），总396.84ms（约645transition/s）。短测replay为重复预热样本，排除初始化／评价／保存，不冒充完整训练计时归因。代码仍有每次更新有限值检查主机同步和逐参数Polyak等可研究点；未测进一步优化收益，不宣称速度极限或保证倍数。
- **验收／产物**：相机与选优原测试3项通过；首秒和前100ms诊断、世界坐标首步摘要、当前profile原始JSON在preview/outputs/diagnostics。前100ms重播与原视频最大平移误差约0.00060mm、角度约3.58e−5rad。新增`docs/startup_height_speed_20260918.md`，更新常用命令／preview说明及旧诊断历史标注。
- **实际执行问题**：一次只读进程检查没有匹配项使rg返回1，后续GPU查询未执行，已单独重查；不涉及运行失败。其余诊断、profile、重构视频均退出0。
- **未解决**：V32-SAC-03跨次GPU微小漂移、V32-AUDIO-01力度标定保持。命令插值／硬速度限制／预备空拍／平滑奖励是控制或任务变化；论文pp.6–11重合项及用户映射决定已写入文档，未自行启用。物理核进一步分解和学习器同步融合尚未实施；本轮没有新训练或性能加速实现。更新视角后仍能看到原策略首步快速动作。

## 2026-09-18 — 当前GPU场景与原版Shadow物理参数核对

- **目的／范围**：按用户要求对比实际V3.2冻结训练场景与原版RoboPianist/robopianist-rl默认Shadow双手物理参数；只生成数据和说明，没有改变场景、控制时序、碰撞参数或训练。
- **参考**：当前scene.xml/manifest、SceneBundle/PianoScene、suite/tasks/base.py、urdf_model.py、piano_constants.py；原版robopianist-rl/train.py、PianoTask/ShadowHand、安装脚本锁定的Menagerie提交1afc8be64233dcfe943b2fe0c505ec1e87a0a13e；论文记录中的控制参数决策；MuJoCo官方modeling接触合并／solver说明；本机MuJoCo-Warp `_src/io.py`／`constraint.py`。按pianist-v2-change-log维护记录。
- **实际问题与处理**：仓库原版缺Shadow资产、音色和PyAudio；最初诊断导入分别因包初始化属性、音频依赖缺失失败。诊断进程设置仅用于模块发现的包路径／属性，并跳过未使用的音频播放模块，实际物理仍由原PianoTask构建。使用本机独立参考缓存的Shadow模型，左右XML逐字节核对官方固定提交成功；未复制资产／源码或新增独立项目运行依赖。网格资产未逐文件远程核验，历史MuJoCo版本未知，限制明记文档。一次Warp选项转JSON直接np.asarray失败，改用wp.array.numpy()后成功；首次缺目录／无rg匹配等只读命令非零退出没有被当作验收成功。
- **结果**：两模型编译后提取逐geom/joint/actuator参数。当前0.5ms×100、implicitfast，对比原版5ms×10、Euler；Newton上限均100。当前手键优先级由琴键1主导，实际solref0.001/1、solimp0.99/0.99/.001/.5/2、滑动摩擦1；原版合并solref.0075/1，经refsafe得到时间常数.01，solimp.7/.97/.00055/.5/2。当前键限位solref.001/1、solimp.999/.999/.001/.5/2，对比原版.02/1及默认阻抗。当前44手网格，对比Shadow12手网格＋64简单形状；琴键行程／回弹弹簧／阻尼不变。
- **GPU运行时重要发现**：实际MJLab模型参数与冻结包配置一致，但Warp put_model将FP32 tolerance从mjModel的1e−8提升至至少1e−6；读取实际wpModel约9.999999975e−7确认。报告将CPU配置与GPU实际值分列，没有修改依赖。
- **产物**：新增独立项目docs/physics_shadow_comparison.md及physics_shadow_comparison_20260918.json（含编译版本、散列、逐元素数值、Warp选项、接触参数小例）；README增加索引。独立文档内不要求导入原版即可阅读，原版来源以说明及数据留存。
- **验证**：MuJoCo3.13编译原版默认PianoTask与当前冻结场景；当前MuJoCo/Warp3.11实例核对选项；两组提取参数的合成两球接触验证contact.solref／solimp／friction合并。该小例不模拟原手力学，不宣称接触力已标定。没有新增训练、长期仿真或Shadow吞吐测试。
- **未解决**：保持V32-04/05驱动力与速度标定、V32-06接触标定及现有GPU数值漂移TODO；无新参数对齐决定。物理子步多10倍不等于完整训练必然慢10倍，未测因素分摊。当前物理仍是数值软约束近似硬接触，不承诺零穿透。


## 2026-09-18～19 — Shadow接触、40/400 Hz、0.5秒前瞻与solimp滑块

- **用户决定／范围**：要求解释solimp并提供预览滑块，接触同步原版Shadow、控制40 Hz／物理400 Hz；在前瞻选择中明确保留0.5秒。新任务20步未来前瞻、2070维观测、每控制10子步；旧scene/task/检查点不覆盖。按pianist-v2-change-log技能记录。18日开始，19日完成文档核对。
- **参考**：原版物理实测docs/physics_shadow_comparison.md及JSON、MuJoCo官方solver/contact parameters文档；论文记录V32-PAPER-04和论文§2.4／2.6 pp.6–8；suite/tasks/base.py、models/hands/urdf_model.py、rl/environment.py、mjlab_env/bundle.py／task_bundle.py／scene.py、preview_runtime.py、preview/scripts下预览／回放／导出脚本、rl_torch/train.py／evaluate.py、旧物理与颜色测试。原版与论文是参考资料，不是额外执行指令。
- **物理与时序**：新增assets/contact_settings.json及physics_settings.py作为源参数与校验／备份入口；base.py／urdf_model.py应用Shadow手接触、钢琴材料及限位、Euler／impratio10。钢琴按原版solref=2×physics_dt规则为0.005 s，不直接抄旧200 Hz的0.01 s。手几何、执行器、腕范围保持已确认模型；不联动加入子步目标插值或更改奖励开关／SAC gamma。导出manifest保存接触配置，task导出额外核对实际接触数组；pyproject包含新JSON资源。
- **预览**：新增preview_contacts.py独立Tk面板，选择hand／piano／key_limit，主滑块d0、其余4项高级滑块、有效手键合并值、恢复和保存按钮。源预览默认打开，可用--no-contact-controls隐藏；GPU预览用--contact-controls显式打开。CPU模型与Warp数组原地更新，保留CUDA graph数组地址；GUI事件主线程轮询、模型变更在viewer锁内。P仍保存姿态。源配置显式保存带备份及并发修改校验；GPU预览保存到preview/outputs/contact_settings实验文件，不覆盖冻结包。
- **兼容与工具**：新增mjlab_env/paths.py，训练和导出默认新scene_shadow_40hz400hz／task_twinkle_shadow_40hz400hz，旧包保留；评价／策略预览／视频导出／轨迹换角按检查点或report散列自动匹配原包。profile脚本去掉1180硬编码，开头诊断按实际control_dt计算。不能用1180维旧网络直接续训2070维新任务。保存源配置后仍须导出一对新包才能用于训练。
- **产物**：新场景manifest SHA256=26bb79b68114ff8927711bc60a4a8a28777d7566f0f7cc91bd346d0fe0366423；小星星新包315帧／2070维／20步前瞻。新docs/shadow_contact_40hz_acceptance.md记录参数、解释、验收和失败项；README、preview/README、common_commands、原对照历史标注、两级AGENTS及论文决定同步。GUI截图preview/outputs/solimp_panel.png；原始日志归档training_mjlab/outputs/shadow40_task_acceptance/logs/。
- **单元／GUI验证**：training_mjlab/.venv中unittest的test_paper_rewards、test_mjlab_task、test_torch_sac、test_acceleration_and_collision、test_contact_settings、test_best_checkpoint_selection、test_preview_camera共34项通过；.venv中pytest tests/test_piano_contact_and_color.py为2项通过。旧硬限位测试改为明确加载旧冻结包，原阈值不放宽；新软接触另行测量。新5项接触测试覆盖参数、实际contact混合变化、不改变qpos/ctrl、保存／备份／冲突、旧包散列匹配、新观测、GPU数组指针与图执行。两个真实GUI入口开关成功，真实Tk Scale事件／临时文件保存／恢复通过；未改用户初始姿态或实际源JSON。
- **训练／回放验证**：新配置4环境×6向量步短训练，batch8、梯度2、warmup1、learning-starts2、max-episode-steps3，24transition／10更新／8截断／2评价，9.6926s完成，结果shadow40_sac_smoke。其final.pt未传scene/task即可完整有声回放315步／3150音频子步；ffprobe确认H.264＋AAC、40fps、356帧、8.9s，产物preview/outputs/shadow40_smoke_replay。短测不是新最佳演奏，也没有新5M训练或吞吐承诺。
- **未通过／实际问题**：DISPLAY=:1授权失败，切换本机桌面DISPLAY=:10、用户XAUTHORITY后成功；未修改系统权限。严格validate_mjlab_task新包16环境对照为269/270，press_physical_rollout失败：max_obs_error=0.1192299超0.1门槛，max_reward_error约0.001091，平移0.132354mm／转角0.007936917rad，累计288个键激活位差异。中性轨迹近1e−7，固定状态规则／奖励、局部reset、完整315步曲终／CPU指标均通过。额外检查CPU参考key_state／activation与保存qpos一致，未发现错帧；没有放宽阈值或声称GPU物理与CPU完全一致。
- **接触限制／TODO**：新配置固定下压2秒CPU测试有限，最大手键穿透3.05335mm、琴键限位折算键端超行程1.42037mm，数据见contact_measurement.json；Shadow参数比此前硬配置更软，旧误差上限不再适用。保留V32-06接触标定、V32-SAC-03 CPU／GPU激活及跨次回放漂移；尚未隔离根因。未知奖励映射仍由用户填写，动作插值未启用。全部验证详见新验收说明，不把接口打通等同于训练收敛。

- **交付前检查**：24个相关Python文件语法解析通过，新验收文档本地链接全部存在；实际旧step-15000.pt自动匹配scene／task_twinkle，新短测final.pt匹配scene_shadow_40hz400hz／task_twinkle_shadow_40hz400hz。


## 2026-09-20 — 启动新40/400 Hz小星星5M训练

- **目的／决定**：用户要求开启新训练，沿用已确认40/400 Hz、0.5秒前瞻及论文SAC Stage1参数，从头训练5,000,192条采样（256×19532）。默认五项奖励、自碰撞权重0，不替用户开启论文未知映射项。
- **参考**：AGENTS、pianist-v2-change-log技能、论文登记V32-PAPER-04／10、新物理验收、rl_torch/train.py参数与启动逻辑、mjlab_env/paths.py默认包。启动前确认无其他SAC训练进程且GPU有可用容量。
- **操作／修改**：后台独立会话启动PID1651371，目录training_mjlab/outputs/sac-20260920-171514-880281，控制40 Hz／物理400 Hz、2070观测／51动作、batch4096、32更新／向量步、seed42，CUDA graph／compile／fused Adam／进度条全开。完整argv、PID、时间、console.log在outputs/launches/sac-20260920-171514-880281/；新增docs/training_run_20260920.md、更新common_commands最新启动索引及两级日志。未改代码、物理或奖励源配置。
- **验证**：首个编译和学习成功，启动约1分43秒最近日志1120向量步／286720transition／3872更新，loss与温度有限；step-1000.pt存在，第1000步独立评价315帧完整曲终、F1=0.1210229。进程仍在后台运行，未等待整个预算完成；不将初期采样ETA当最终耗时。
- **问题／TODO**：无新启动错误；Torch提示TF32可用但未开启，按既有FP32合同保留。既有V32-SAC-03 CPU／GPU激活差异与V32-06接触标定仍未解决，本次启动不改变其状态。不承诺收敛或新最佳表现。


## 2026-09-20 — 导出新40/400 Hz训练最佳结果与耗时

- **目的／参考**：用户要求最新最好训练视频并报告耗时。参考最新run的config／metrics／summary和launch时间、export_best_mjlab.py选优规则、render_mjlab_checkpoint.py报告与音轨实现、train.py计时边界、AGENTS及论文既定评价口径；按pianist-v2-change-log技能记录。
- **筛选／时间**：sac-20260920-171514-880281为最新正式run，已完成5,000,192采样／593,056更新。20个已保存候选里final.pt日志F1最高0.8595393896，对应19532向量步；不以旧20Hz不同任务结果混排。内部总耗时3581.8859198s（59分41.9秒），包括初始化／编译／训练／评价／保存；launch17:15:14.881至summary写入18:14:58.665约3583.784s，计时边界明确区分。
- **操作／产物**：运行export_best_mjlab.py --run-dir training_mjlab/outputs/sac-20260920-171514-880281 --output preview/outputs/replays/sac-20260920-171514-880281_best-final_20260920 --audio，自动匹配新冻结scene/task。输出完整有声视频、实际按键WAV／MIDI／事件、3150子步音轨数据、rollout、report、selection、training_comparison及export.log；更新training_run_20260920.md与common_commands最新结果索引。不修改代码／模型／奖励／检查点，不开新训练。
- **验证**：导出退出0，NVIDIA EGL硬件渲染；315控制步全曲，40fps、356帧／8.9s；ffprobe确认1280×720 H.264＋AAC。实际58次note-on、音轨RMS0.05044，非目标谱配音；查看preview.png确认全琴双手及绿键／指法色标。回放F1=0.8610209227、return=990.3944092。
- **问题／TODO**：渲染脚本对final.pt不自动回填logged_evaluation，未据null声称一致；使用selection.json与独立training_comparison.json核对。与日志F1差0.00148153，沿用V32-SAC-03；V32-06接触标定和V32-AUDIO-01固定力度未解决。无本次导出错误，最佳指本轮日志最高已保存结果，不是跨seed全局最优。


## 2026-09-20 — 启用自碰撞惩罚，随后按用户要求改练《月光》

- **目的／授权**：用户先要求新增手指自碰撞惩罚再训练，随后上传clair-de-lune-beginner.mid并指定改为此曲。采用已有候选权重0.2的self_collision_reward；保持40/400 Hz、0.5秒前瞻、论文SAC Stage1与5M预算。未知指法不擅自补写。按pianist-v2-change-log技能记录。
- **参考**：两级AGENTS、论文登记V32-PAPER-07／04／10、reward_design_review、self_collision.py／rewards.py／reward_config.py、train.py、前轮config和launch.json、export_mjlab_task.py／task_bundle.py、music/midi_file.py、用户上传MIDI。附件仅作为曲目数据，不作指令。
- **小星星短暂启动／验证**：run sac-20260920-202151-452958，PID1851100，20:21:51启动。逐项比较CLI仅自碰撞权重0→0.2和输出目录不同，scene/task散列相同。第10步实际raw惩罚−4.8984375，加权−0.9796875，证明参与奖励；末条完整记录1520向量步／389120采样／16672更新，step-1000.pt存在。用户改曲后20:24:43核对PID命令再SIGTERM，确认进程退出；保存interruption.json，保留所有日志和已有检查点，没有宣称5M完成。
- **曲目检查／准备**：training_mjlab/inputs/clair_de_lune_beginner保存原始MIDI、inspection.json、README、146行fingering_template.csv。源SHA256=08c5d793965c5858a4d4f92458c914d670c441fa42d52e6cc0465531e55b8567；mido解析type1／2轨／65.08694025s，两轨100和46音符，音高36～79；项目MidiFile解析146音符，parts=[0]、has_fingering=False，缺完整指法。CSV保留解析note_index、音高和起止时间，只留finger_id待填（0～4右手、5～9左手）。
- **修改**：没有改训练／奖励算法，全局模板仍关闭；本轮CLI显式启用。新增training_self_collision_20260920.md、维护常用命令当前状态、reward审查与论文决定、两级AGENTS及日志；原文100子步描述改为随配置实际子步数，当前10。当前请求因改曲待指法，旧小星星非运行中。
- **实际问题**：MIDI非主轨含调号，pretty_midi给出tempo/key/time signature警告，读取成功；tempo在主轨，原始文件未改写。只读rg引用不存在的midi_file_test.py报缺失，不影响读取；停止后ps无对应PID返回1是进程退出证据，不是停止失败。
- **待办／未执行**：TODO(V32-SCORE-01)：本曲146音符缺指法，等待用户指定或明确授权自动候选指法。按既有用户“由我提供指法”决定和GPU完整指法合同，未擅自生成／关闭指法项，也未导出不完整task或用小星星任务冒充。已发出明确两选一问题；《月光》训练尚未启动。V32-SAC-03／V32-06保留，负碰撞奖励不保证零碰撞。


## 2026-09-20 — 自动生成《月光》指法并启动带自碰撞训练

- **授权／参考**：用户在缺指法问题后要求“你帮我生成对应的指法”，据此为该曲生成候选并继续上一条训练请求；其他未知奖励映射授权不变。参考AGENTS／日志技能、论文决定、MidiFile.from_file／trim_silence／NoteTrajectory、piano_roll时间离散化、task_bundle合同、现有scene导出、SAC启动及刚停止run配置；不读取外部禁复制目录，也不声称启发式来自论文。
- **实现／指法**：新增scripts/generate_candidate_fingering.py，显式指定instrument0右手（100音符）、1左手（46音符）。动态规划枚举分配，以持续音保持和同指不同键互斥为硬约束，指序／半音姿态偏差／换位距离为软代价；允许必要的换位次序，参数为工程启发式。输出generated_v1/fingering.csv、带part指法.proto与报告；CSV含中文手指、左右手及原序号。右拇到小指15/18/25/23/19音符，左手10/14/9/9/4。
- **保护曲目／验证**：146音符的音高／起止时间／力度／乐器／program／is_drum多重集完全保持，原MID未改。源解析按轨道排序，旧trim用首末音符会漏更早左手；生成proto按时间排序，trim起点0.4347825s，仍保留146音符。连续时间音符重叠、实际40Hz轨迹均无同指多键；proto回读等于保存序列，has_fingering=True。冻结score形状2587×10×88、每帧每指最多1键，TaskBundle合同通过，2070维观测／51动作；新增脚本语法及文档链接检查通过。
- **导出实际问题**：首次严格手指次序的候选在frame608无解，改软次序成本但不放宽占指／持续音约束。第一次task向旧scene导出因双腕yaw0.25 vs旧0的reset检查失败，未绕过；按当前保存源姿态新导出scene_clair_shadow40_pose_20260920和task_clair_de_lune_beginner_shadow40_fingered_v1，成功。进一步核对发现当前安装Z=0.14 vs旧0.12（双手），已告知并记录；没有编辑／覆盖用户姿态源码。接触solref／solimp、key限位、actuator gain、joint范围和body旋转对比不变；新scene差异报告保存。一次只读TaskBundle调用漏传scene导致TypeError，补齐后通过；独立比较body_pos预设相同的断言失败后改为逐项报告，未伪称相同。
- **训练**：run sac-20260920-204729-873996，2026-09-20 20:47:29启动后台PID1879735；小星星旧进程保持已停止。新曲从头训练、seed42、256环境×19532向量步＝5,000,192采样，batch4096／每学习向量步32更新、三加速和进度条，self_collision_reward外部权重0.2。完整argv／console在outputs/launches同名目录。初期1000步／256000采样／32更新有限，实际raw碰撞−5.87109375、加权−1.17421877，证明进入损失前奖励；后续快照见下条。
- **文档／产物**：新增docs/training_clair_de_lune_20260920.md，更新曲目README／inspection、common_commands当前运行、上轮改曲记录、两级AGENTS、论文授权和两级日志；生成报告与导出成功／失败日志均保存于inputs/clair_de_lune_beginner/generated_v1。旧输入与冻结包保留。
- **TODO**：V32-SCORE-01本曲缺指法阻塞已解除；新增V32-SCORE-02候选指法完整运动学可达性／换指／碰撞与演奏质量待训练回放，非专家或实机标定。V32-SAC-03／V32-06保留。未重跑已知CPU/GPU严格物理轨迹对照，也未将启动当训练完成。相较小星星曲目、姿态、奖励均不同，不能称仅自碰撞单因素对照。

- **启动验收补记**：截至交付检查，最新完整日志1290向量步／330240采样／9312次更新、loss与温度有限；step-1000.pt已写出。首次独立评价完整2587帧、未截断，F1=0.11072775；这是刚开始学习的指标，不是最终结果。后台训练继续。


## 2026-09-20 — 小星星训练前期视频与前冲定位

- **目的／参考**：用户要求导出小星星前期视频查看爆冲。使用最新完整小星星sac-20260920-171514-880281首个step-1000（256000样本／32更新），读取config／metrics／冻结scene；参考render_mjlab_checkpoint、diagnose_replay_start、scene.step动作映射、sac.Actor、AGENTS与论文控制决定。遵循日志技能。本次仅导出／测量／文档，未改代码、模型、奖励、策略，月光PID1879735继续运行。
- **产物**：preview/outputs/replays/twinkle_early_step1000_20260920保存完整有声回放、原轨迹／音轨事件、startup_summary、export与diagnose日志；另导出开头2秒四分之一速度无声片段。新增docs/twinkle_early_startup_20260920.md和命令索引。逐2.5ms诊断首40控制步与保持初始ctrl0.5秒对照，JSON在preview/outputs/diagnostics/twinkle_early_step1000_20260920.json。
- **定位**：动作是全范围绝对位置；早期输出近0映射中点，首条右TX−123.71／左TX+142.26mm，右TY+75.74／左TY+70.44mm，TZ约−11.4／−12.76mm。无变化率限制／子步插值；30个手指初始比例力需求超过力矩限值。初始无负距离手接触，2.5ms已运动，10ms采到首次触键，50ms保存姿态出现手内接触（动态缓存52.5ms）、100ms双手接触；首秒未采到手—琴箱碰撞。保持初始目标0.5秒无手接触、腕最大偏离1.68mm。先主动目标运动、后接触反作用，不能称初始穿插弹出。
- **精度／验证**：导出完整315步、40fps／356帧／8.9s，ffprobe H.264 1280×720＋AAC；慢放7.975s。音轨313次实际note-on，RMS0.14372；查看截图全琴双手及指法。回放F1=0.123432 vs日志0.121023。重放首100ms与视频最大平移差0.000381mm／角度7.75e−6rad；首秒差增至14.38mm／0.5235rad，未将后段重放峰值力冒充原视频受力。原视频100ms内向中间位移81.66／75.51mm、向内32.64／27.24mm来自原qpos。
- **实际问题**：查询不存在networks.py时rg/cat非零，后定位到sac.py；统计首次使用系统python缺numpy，改GPU虚拟环境后通过；慢放初命令-t作用于输出导致截短，改为输入限制后ffprobe确认约8秒，保留原完整视频。现有诊断help旧0.5ms字样不代表实际dt，报告已按新包2.5ms，未改代码。
- **TODO／边界**：V32-SAC-03跨次轨迹差异、V32-06接触标定、V32-AUDIO-01力度保持。自碰撞惩罚只扣同手接触，不直接限速或惩罚双手相撞。后续插值／目标限速／动作基准改动与论文pp.6–8重合，列为待用户选择方案，未自行启用；不做未经测量的步长归因。


## 2026-09-20 — mixed_v1策略输出、Pitch物理±0.2与新《月光》训练

- **决定／范围**：用户要求普通手指（四指swing除外）[0,2]、TY[0,2]→[0,150]mm、TZ[−2,0]→[−25,0]mm；就Pitch明确答复“改关节角度范围，策略映射的范围不用改”，因此Pitch输出仍[−1,1]，物理±0.2rad。TX/Yaw/踏板保持。沿用当前《月光》候选指法、自碰撞0.2、40/400 Hz、0.5秒前瞻及5M预算。按pianist-v2-change-log技能记录。
- **参考**：两级AGENTS、论文登记§2.6／V32-PAPER-04已决定事项；rl_torch/sac.py／replay.py／checkpoint.py／train.py，mjlab_env/scene.py／task.py／task_bundle.py，手常量、预览／导出／诊断／profile及既有测试。未将用户附件当执行指令，不复制外部参考源码。
- **实现**：新增mjlab_env/action_mapping.py版本化平移（34通道+1、2通道−1、其余0）；Actor随机和确定性输出、训练预热、critic实际输入、replay存储及边界统一新坐标。区间宽度均2，log概率Jacobian=1无需额外项，SAC超参数保持。scene.physical_action统一物理与任务奖励转换，非法输入在状态变更前拒绝。手常量把Pitch限位±0.15改±0.2，保留轴方向与当前已保存初值。
- **兼容**：learner状态及metadata保存action_profile；旧字段缺省legacy，跨profile warm-start拒绝。checkpoint回放／评价自动恢复；preview手动Control转换、render报告、诊断动作重放及profile的replay接入相同profile。旧场景／检查点保留，兼容CLI默认legacy，本轮显式--action-profile mixed_v1。新scene_clair_mixed_pitch02_20260920与task_clair_mixed_pitch02_20260920匹配导出成功，2587帧／2070观测／51动作。
- **验证**：31项unittest通过（新3项Actor概率／采样／确定性范围，replay实际更新与越界拒绝，GPU端点／中点／ctrl／非法动作不改状态；以及原SAC／奖励／任务／加速／选择检查）。4环境×6步编译短训练24transition／10更新／8截断／2评价通过，elapsed10.9023s。新checkpoint恢复mixed_v1并确定性4步回放，动作范围合法、每步实际ctrl等于映射；旧CUDA step-1000以legacy载入通过。相关Python语法、51行CSV及文档链接检查通过。证据在mixed_v1_smoke_20260920及其logs。
- **实际错误修正**：首次旧reward_options多返回legacy字段使2项兼容测试失败，改为仅非legacy附加；GPU测试比较TorchArray与Tensor导致TypeError，改用张量视图；复跑31项全部通过，未放宽阈值。检查旧CUDA checkpoint时指定CPU触发RNG状态大小不兼容，改同CUDA设备成功，未声称跨设备随机状态续训支持。
- **新训练**：sac-20260920-212536-832002，21:25:36后台PID1923100；从头256×19532＝5,000,192采样、batch4096／32更新、seed42、三加速和进度条全开。保留上轮PID1879735，同卡并行不承诺旧单进程耗时。启动先完成约25万采样，正式学习／保存验证在后续补记。
- **文档**：新增docs/action_mapping_mixed_v1.md与51通道CSV；更新README、preview当前Pitch范围、common_commands、两级AGENTS、论文决定及日志。文档明确：对外action=0现在TY/TZ为0，但网络tanh零输出经平移成为TY1／TZ−1，物理仍为75mm／−12.5mm；仅平移坐标不能保证解决爆冲。未联动限速、插值、残差动作或初始化偏置校准。
- **TODO／限制**：V32-SAC-03／V32-06／V32-SCORE-02继续；短测不证明演奏改善或碰撞消除。未重复做已知不完全通过的CPU/GPU长期物理轨迹对照。本次扩大Pitch物理范围，并改变critic所见动作坐标，不能声称新旧训练完全等价。

- **正式启动验收补记**：最新完整日志1050向量步／268800采样／1632次更新、loss与温度有限；step-1000.pt已保存。首个评价完整2587帧、未截断、F1=0.09358424，属初期学习结果。PID1923100存活，记录的全部训练源码SHA与交付文件一致。后台训练继续，未等待5M完成。


## 2026-09-21 — 全部近期训练结果与F1曲线

- **请求／范围**：检查V3.2最新全部训练并绘制F1。按pianist-v2-change-log技能记录；只读现有训练产物，新增绘图和报告，不改策略、奖励、动作映射或训练参数。
- **参考**：两级AGENTS、论文notes已有评估决定；20个目录的config／metrics／summary及主动停止interruption；冻结task.json；rl_torch/train.py:evaluate和mjlab_env/task_logic.py:binary_metrics；三份20260920训练记录、common_commands与README。沿用已确认逐帧F1口径，未另设论文实验。
- **发现**：6轮正式MJLab（5完成、1因切换曲目停止），2轮历史JAX，12轮功能短测试；当前无训练进程。正式101个完整评估点、历史20点。旧小星星最高0.933966；40/400小星星最高0.859539。两轮月光5M已完成，最佳均step-1000，最终F1降至0.022305／0.011787，召回0.018754／0.011596；当前逐帧Precision的无预测帧定义为1，不能单看该值判断质量。
- **修改／产物**：新增scripts/plot_training_f1.py，自动扫描日志、按曲目／时序／后端分图，原始点与最佳星号，拒绝覆盖已有输出；生成preview/outputs/training_reports/20260921_f1内PNG总图、PDF、月光放大图、全部评估CSV、20目录清单、含原文件SHA的JSON。新增docs/training_results_20260921.md，更新README和common_commands已完成状态及重绘命令。
- **验证**：GPU虚拟环境Python运行绘图成功（无需GPU），两个PNG已目视检查中文／图例／坐标；语法解析、报告全部文件链接、101正式＋20历史点数、20目录清单通过。6轮最佳文件存在；101点F1有限且0～1、完成且无截断、采样数递增；5轮完成日志最终评价与summary采样一致，正式train标量无NaN／Inf。未重新仿真全部检查点，未把日志归档视作演奏验收。
- **实际问题**：系统python别名不存在，统计改用python3，绘图使用training_mjlab/.venv/bin/python；新产物目录原先不存在，脚本正常创建。主动停止轮没有summary，采用interruption及最后完整日志并明确其最终评价停在1000而非1520步。
- **TODO／边界**：新增V32-TRAIN-01（月光训练F1退化根因：后续对比首个／最终策略实际按键、指法可达性、碰撞与分项奖励）；V32-SCORE-02／V32-SAC-03／V32-06保留。单seed、曲目／物理／姿态差异及同时改变Pitch，使本次曲线不能作单因素因果结论；月光同卡并行耗时不作纯性能对照。没有未经授权调整奖励／课程。


## 2026-09-21 — 两轮月光最佳检查点有声视频

- **请求／参考**：用户要求导出月光视频；按上轮F1报告导出legacy与最新mixed_v1各自最高已记录检查点（均step-1000）。读取两级AGENTS、日志技能、论文已确认动作／评估范围、common_commands、export_best_mjlab.py、render_mjlab_checkpoint.py、preview_camera.py及原run日志。未更改训练或奖励。
- **产物**：preview/outputs/replays/clair_legacy_best_20260921和clair_mixed_best_20260921，含完整replay.mp4、选择依据、回放报告、轨迹、实际按键音频事件／MIDI／WAV、截图；export_logs保留导出日志。新增docs/clair_videos_20260921.md并更新common_commands入口。
- **验证**：两个GPU导出正常结束，NVIDIA硬件渲染，自动恢复各自冻结场景与action_profile；全曲2587步／25870子步，1280×720、40fps、2628帧、65.7秒。ffprobe确认H.264＋AAC且时长一致，两文件完整ffmpeg解码无错误；查看两张截图，全琴双手和指法可见。实际note-on1053／1059，48kHz双声道RMS0.05186／0.05215。保持正面−30°正交全琴视角，音轨来自实际接触激活，不用目标乐谱替音。
- **问题／限制**：回放F1为0.12735270／0.10795062，与日志0.11072775／0.09358424不同；如实分别记录，不修改原训练曲线。没有导出失败。声音力度127未标定，不把多次触发当正确音符；并未解决演奏质量。
- **TODO**：V32-SAC-03跨次物理差异、V32-TRAIN-01月光退化、V32-SCORE-02指法可达性、V32-06接触标定、V32-AUDIO-01力度继续。


## 2026-09-21 — 月光视频双手相撞原因定位

- **参考／范围**：读取两段原视频rollout、report及冻结SceneBundle／task；self_collision.py分类，scene.physical_action／step，task.snapshot掌部集合，task_logic的forearm_reward，rewards与Actor/action_mapping，论文已确认范围及日志技能。仅诊断，不改训练配置。
- **实测／产物**：preview/outputs/diagnostics/clair_inter_hand_20260921.json记录全部2587帧×2的原保存qpos原生MuJoCo几何接触。初始无跨手接触，首个保存接触时刻均0.075秒，分别2558／2564帧存在跨手接触，最深负距离−8.57／−9.68mm。完整更新docs/clair_videos_20260921.md原因链和后续待选项。
- **定位**：TX近零输出映射右约−150／左+150mm，引导双手向中央；mixed没有改变TX，TY/TZ平移仍保留网络中心对应物理中点。当前self只罚同手；forearm仅覆盖指定掌掌geom，未全面覆盖跨手手指。持续绝对位置目标配合软接触允许几何交叠。首个检查点仅32更新，不能视为成熟策略。
- **验证／问题**：逐帧几何统计完成；核对原录像首条ctrl及分项奖励，mixed同手平均罚−0.00541不代表跨手接触少。不是重新积分／重放GPU力，不把凸碰撞几何负距离称实机表面穿透。查询不存在tsensei_constants.py时rg报错，实际使用已冻结manifest和装配源码，无此文件依赖。
- **TODO**：V32-TRAIN-01退化其他原因、V32-06接触标定、V32-SAC-03保持；跨手接触名单与插值／限速等方案尚待用户决定，未代填未知论文映射。本次无策略或奖励修改。


## 2026-09-21 — forearm碰撞检测区域绘图

- **请求／参考**：用户要求画出forearm碰撞区域；遵循日志技能，参考task.snapshot与冻结task.json的palm_geoms、SceneBundle、preview_camera／backend、原mixed视频保存qpos与已确认奖励说明。无需新的论文奖励选择。
- **实现／产物**：新增preview/scripts/draw_forearm_regions.py，读取实际掌部碰撞geom并启用MuJoCo碰撞凸包显示，右红左蓝，其他手部半透明上下文；初始与21.575秒各绘上下文／仅区域两图。最终2000×1540四图及4张单图、区域JSON在preview/outputs/forearm_regions_20260921_final；更新docs/clair_videos_20260921.md复现命令。显示设置仅在诊断模型内存修改。
- **验证／问题**：实际EGL渲染完成，语法解析、两geom ID93／137与task名单一致、5个PNG存在；目视确认区域覆盖掌部而不含手指。首版视野偏远且天空盒有亮斑，调近相机并关闭天空盒，保留初版／v2作为过程输出，文档只指向final。不修改训练模型或扩大碰撞名单。
- **TODO**：V32-06接触几何标定、V32-SAC-03跨次差异及V32-TRAIN-01继续；图显示碰撞凸包，不等于厂家精确表面或原GPU动态接触力。


## 2026-09-21 — forearm扩展跨手手指并绘制区域

- **授权／范围**：用户明确要求forearm增加不同手的手指碰撞；实施跨手掌掌／指指／指掌，保持原无接触+0.5／接触0及周期末检测。读取日志技能、两级AGENTS、论文决定、CPU任务、Torch任务／导出／奖励／映射和既有绘图脚本。未知论文接触映射不代填，不启用论文collision_reward。
- **代码**：新增forearm_collision.py名称选择与冻结名单兼容；CPU_compute_forearm_reward采用掌部与手指组，task_bundle导出scope／forearm_collision_geoms并让CPU参考快照采用相同组；GPU task优先新组、旧任务回退palm_geoms。每手21geom（1掌＋20指），不含基座／同手／琴键。train元数据加入新helper源码hash；绘图脚本支持--task且核对scene散列。
- **产物／文档**：新task_clair_mixed_forearm_full_20260921沿用scene_clair_mixed_pitch02_20260920，2587帧／2070观测、CPU两案例各8步；旧task未覆盖。新区域图及名单在preview/outputs/forearm_full_hand_20260921，使用旧视频初始／21.575秒姿态展示新范围，不冒充新训练。新增docs/forearm_full_hand_20260921.md，更新rew总表、common_commands、preview/README、两级AGENTS、论文登记及两级日志。未启动训练。
- **验收**：tests.test_forearm_collision和tests.test_mjlab_task合计6项通过。8个GPU世界覆盖掌掌／指指／指掌、同手、琴键、正距离、反序geom与无效槽，正确返回0／0.5；旧名单保持仅掌掌，8世界4步真实物理及独立reset通过。CPU固定状态16步Torch总奖励／分项容差2e−6全部通过；旧mixed任务散列与原config一致。PNG目视检查，源码语法解析通过。export.log／tests.log／validation.json保存在图目录。
- **实际问题**：文档补丁附带错误preview标题锚点而整体未写入，核对后重新应用正确锚点；临时CPU参考对照首次未将float64转float32导致类型错误，按运行时合同转换后16步通过，未放宽奖励阈值。查询不存在training/README.md无影响。导出有上游NumPy弃用警告但成功。
- **TODO／边界**：V32-06／V32-SAC-03／V32-TRAIN-01继续。周期内短暂接触仍可能漏检，未自行改成子步并集；奖励不能保证无碰撞，未改初期动作跳变。旧Torch冻结规则保留，直接使用新版源码建立CPU任务则采用新规则，不承诺历史JAX源码回放原奖励复现。下次训练须显式指向新task。

## 2026-09-21 — M6 双臂白色装配预览与差分 IK 随动

- **请求／决定**：用户指定 M6、V3.2、基座安装中心距 18 cm，要求现有预览可见双臂，手腕平移／旋转时机械臂通过差分 IK 跟随，暂不进入强化学习；随后确认先使用可调临时肩座／法兰方案，后续标定。TJ-ARM-01 范围已确认。
- **实际参考**：根／本项目 AGENTS、pianist-v2-change-log 技能，`docs/references/biomimetics_2026_610_notes.md`，M6 原始左右 URDF／STL 与来源 inspection，`models/hands/urdf_model.py`、手模型与 `tsensei_hand_constants.py`，现有 `preview_pose.py`、`preview_runtime.py`、几何／姿态保存及测试；[MuJoCo 鼠标说明](https://mujoco.readthedocs.io/en/stable/programming/samples.html#mouse)。图片只参考外观，不推断精确尺寸。没有复制 external_never_copy_only_reference 代码。
- **改动**：新增包内 `assets/tj_m6` 左右 URDF、16 STL、来源散列／许可证及 `preview_mount.json`，安装中点暂定 [0.52,0,0.48] m，中心距 0.18 m，Link7 到手腕临时平移 [0,−0.11,0] m。新增 `preview_arm_ik.py`：独立 14 关节 FK、初始化有界 IK、6D 阻尼差分 IK、每帧累计速度和关节范围限制、实际手腕残差报告、绿色 mocap 拖动目标到原五腕控制量转换。主场景仅增加无碰撞视觉件，保留 nq/nv/nu=138/138/50、51 动作。
- **界面／独立性**：`preview/scripts/preview_pose.py` 默认 `--arms`，支持 `--no-arms`／`--arm-config`，白色双手双臂与全景相机；`preview_runtime.py` 增加绿色球拖动和暂停腕部控制、F 全景与残差提示。P 仍只保存手姿态，不写肩座配置。新增 `demo_arm_ik.py` 和 6 项 IK 测试，既有控制测试增补真实暂停循环随动测试；`pyproject.toml` 打包资产。手常量仅修正旧 S 保存注释，无数值改动。
- **文档**：新增 `docs/tj_m6_preview_ik.md`，同步 README、preview/README、常用命令、论文决定登记、两级 AGENTS、根接入记录和 TJ_forearm 本地说明；本项目无运行时跨目录模型依赖。
- **遇到的问题**：初次不可达测试选择的正 TX 实际可达，改用另一侧真实超范围目标 TX=−0.7605，保持 >5 mm 残差断言；原保存回归仍按 S，失败 1 项，按既有用户决定更新为 P 后通过。桌面 DISPLAY 缺失，尝试现有 X11 显示遭授权拒绝，未绕过；改用离屏渲染和程序化 GUI 事件验证，桌面鼠标实操尚未覆盖。文档辅助命令发现全局无 `python`，改用 `python3` 完成。依赖已有 audioop／NumPy 弃用警告未在本轮修改。
- **实际验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_preview_arm_ik.py tests/test_preview_controls.py tests/test_preview_backend.py tests/test_preview_pose_config.py -q` 为 **25 passed**（23.72 s），包括数值 Jacobian、位置／速度限制、目标不可达、鼠标位置／旋转映射、暂停滑块／拖动随动及原控制／姿态回归。日志 `preview/outputs/tj_m6_preview/preview_regression.log`。原生离屏全景 `tj_m6_preview/pose.png` 与演示截图已查看；`tj_m6_follow_demo_final/arm_follow.mp4` 150 帧／5 秒／1280×720，通过 ffprobe 和全片解码；该轨迹左右最大位置残差 <0.008 mm，角残差 <0.0017°。这是脚本腕部运动演示，不是学习策略或真实机械臂精度。
- **资源打包验证**：`pip wheel --no-deps --no-build-isolation .` 成功，确认 16 STL、2 URDF、2 JSON、LICENSE 均进入 wheel；该过程用于资产收录检查，未发布安装包。无需父目录 SDK 即可加载包内模型。
- **未解决 TODO／边界**：TJ-ARM-02 肩座位置／朝向、TJ-ARM-03 法兰变换及转接尺寸待实测；TJ-ARM-04 实体动力学链、碰撞和避障尚未实施。虚拟手腕全范围不保证可达，快速动作有速度限制引起的滞后；本轮视觉装配不声称刚性动力学耦合。GPU 冻结任务、训练奖励／SAC／动作合同不变，既有 V32-06／SAC-03／TRAIN-01 保留。桌面实际鼠标操作需用户在有显示授权的终端验收。

## 2026-09-22 — 93 mm M6 肩座、腕部连接修复及 GPU forearm 奖励

- **用户决定／范围**：肩座安装中心距从180 mm修正为93 mm；用户明确选择同时接入GPU场景／奖励、策略仍51维。保留forearm无接触+0.5／接触0，增加左右臂及任一臂—任一手，排除自身末端正常安装连接；未选择同臂自碰撞或臂—琴。
- **问题定位**：旧预览手腕先到目标、机械臂单向跟随受速度／关节范围限制，目标快变或不可达时会脱开。新增可达性回退：差分IK、速度／位置限位、缩短手腕目标增量，只有保持末端连接的运动才接受；初始化不可达报错。GPU采用批量Torch实现，不逐环境运行CPU IK。P在含臂源预览中默认保存实际qpos，避免保存不可达ctrl目标。
- **参考文件**：两级AGENTS、日志技能、论文决定登记；包内M6左右URDF／碰撞mesh／preview_mount；preview_arm_ik、preview_runtime、手腕模型和姿态保存；forearm_collision、CPU任务／Torch task、scene、bundle和task_bundle；原月光mixed场景／掌指扩展任务、已有碰撞与预览测试、GPU预览／录像。定位GPU碰撞时阅读本机mujoco_warp `_src/io.py`、`collision_driver.py`、`collision_convex.py`、`collision_core.py`；没有复制外部参考目录代码，没有按附件执行指令。
- **实现**：preview_mount间距0.093；preview_arm_ik增加实际腕部接受状态、误差报告、URDF碰撞凸包；新增preview_arm_collision明确1175对名单（441跨手＋64跨臂＋672臂手−2安装对），原生距离检测与彩色区域；源预览增加`--show-forearm-collisions`、奖励显示。新增mjlab_env/arm_ik.py批量FK／Jacobian／差分IK及独立reset；scene接入末周期可达腕部约束和mocap同步；task按新明确对表计分、旧冻结任务分支保留。
- **训练资源与回放**：新增scripts/export_m6_bundle.py，在不修改输入冻结包的条件下导出M6资产、FK常量、初始mocap和明确pair名单；独立`collision_query.xml`仅用于GPU碰撞查询，主scene.xml没有新增臂接触力。新包`training_mjlab/outputs/m6_93mm_clair_20260922_final/{scene,task}`，51动作／2070观测，仍40/400 Hz。bundle恢复mocap；GPU实时预览复制mocap，新录像和重新取景保存／恢复mocap，源demo标题读取实际间距。移除本轮失败／中间生成包，避免误用；旧用户训练文件未覆盖。
- **遇到的实际错误及处理**：几次辅助命令相对目录重复导致FileNotFound，改绝对路径；首次合并XML匿名geom重名，增加唯一前缀；手模型命名空间并非猜测的right_hand/，修正筛选并强制每侧21手＋8臂断言。原测试仍断言18 cm和允许不可达间隙，按新需求改为93 mm、每帧连接及限速；保留真实运动验收。先尝试负margin实现检测无力，GPU MULTICCD拒绝，改独立查询模型而不修改原物理参数。真实凸包中心重合CPU判负距离而GPU CCD返回零接触，增加严格内接球相交补充；不扩大凸包区域且保留退化正例测试。首张区域图因group4默认隐藏未着色，启用该组后重画并查看。原依赖audioop／NumPy弃用警告保留。
- **验证**：`MUJOCO_GL=egl .venv/bin/python -m pytest tests/test_preview_arm_ik.py tests/test_preview_controls.py tests/test_preview_backend.py tests/test_preview_pose_config.py -q`：26 passed（33.37 s）。GPU用training_mjlab/.venv执行`python -m unittest tests.test_m6_gpu tests.test_forearm_collision tests.test_mjlab_task -v`：11 passed（4.620 s），含真实臂臂重合、臂手碰撞、安装／同臂／钢琴排除反例、Torch FK和原生FK对照、8世界8步及局部reset。GPU改为读取配置参数后5项M6再次通过（3.553 s）。碰撞查询前后约束力不变；不可达快速预览请求每帧位置残差<0.02 mm、角残差<0.01°，并保留URDF速度限位。区域图及主CLI离屏渲染成功，后端NVIDIA RTX PRO 6000；JSON确认0.093 m、1175对。相关脚本py_compile、文档本地链接检查通过。
- **SAC接入短测**：新M6包、mixed_v1，8环境×4步、batch16、warmup1、learning-starts1、每步1次更新、关闭评估／学习器编译，成功产生32条GPU replay样本、3次优化更新和final.pt；耗时3.802 s，输出`training_mjlab/outputs/m6_93mm_sac_smoke_20260922`。只是接口验收，没有正式长训练或演奏收敛结论；其短测参数不替代用户既定正式SAC设置。
- **文档／产物**：新增docs/m6_forearm_93mm_20260922.md，区域图与完整名单／测试日志在preview/outputs/m6_forearm_93mm_20260922_final；同步rew总表、项目／预览README、常用命令、旧方案顶部状态、论文决定、两级AGENTS、根接入说明和日志。
- **TODO／实际边界**：TJ-ARM-02/03肩座其余坐标与法兰实测保留；TJ-ARM-04实体机械臂动力学、接触力及避障仍未接入。GPU仅周期末运动学投影，不能声称物理子步的实体刚性连接；机械臂IK隐藏状态尚未进入观测。TJ-ARM-05完整曲目／大批量吞吐／长期学习及冗余IK观测验收待做，原V32-06/SAC-03/TRAIN-01保留。桌面实际鼠标未覆盖；此前X11授权限制，用户可按文档在桌面终端验收。新功能必须选择新scene/task，旧冻结包不自动获得机械臂奖励。

## 2026-09-23 — 启动含M6的小星星5M训练

- 用户要求机械臂已接入时开始小星星训练。核对93 mm M6 GPU IK／1175对forearm规则已实现；沿用当前mixed_v1、同手自碰撞0.2、40/400 Hz、论文Stage1 SAC与既有5M预算。原策略仍51动作、2070观测，无机械臂实体动力学／接触力。
- 参考AGENTS、日志技能、论文既有决定、M6验收、train.py、export_m6_bundle.py、export_mjlab_task.py、前轮launch.json及冻结物理合同。使用原mixed物理场景重新导出315帧小星星有指法任务，再生成m6_93mm_twinkle_20260923/scene和task。没有覆盖旧包或启动重复训练进程。
- 初次用原生.venv导出遇到dm_env_wrappers缺失；换既有training/.venv成功，M6导出用.venv，训练用training_mjlab/.venv；未安装依赖。核对0.093 m、1175对、315帧、2070观测通过。
- 启动sac-20260923-004554-861361，PID 1357075，256环境×19532向量步＝5,000,192条采样，batch4096／gradient32／seed42，CUDA物理graph、学习器编译、融合优化器和进度条开启。从头训练，未warm-start。完整argv、命令脚本、后台console及导出日志已存launches/sac-20260923-004554-861361；新增docs/training_m6_twinkle_20260923.md并更新README和命令速查。
- 实际验证：核验时间 2026-09-23T00:47:26.643379：PID 1357075存活，日志向量步660、采样168960、更新0；GPU replay持续增长，分项奖励有限且含forearm_reward／self_collision_reward。尚未出现评估结果，不报告F1或预计完工时间。 此次启动不等于完成5M或训练收敛，采样阶段吞吐不代表优化更新后的速度；无正式新F1。
- TODO：TJ-ARM-02/03标定、04实体动力学／避障、05长期学习／吞吐／冗余IK观测和V32-SAC-03／V32-06保留。后续检查完成摘要、完整曲目F1及最佳检查点。

## 2026-09-23 — M6小星星训练结果与最佳视频检查

- 用户要求查看最新结果与视频；遵循日志技能，参考AGENTS、论文既有决定、训练summary／config／20次eval日志、渲染／相机／音频脚本及冻结M6几何。只检查／回放，不修改奖励、策略或重新训练。
- 确认5,000,192采样完成，593,056更新，耗时5276.316秒。最佳step-18000 F1=0.204550，最终0.167407。日志train标量无非有限值；新曲线与analysis.json在preview/outputs/training_reports/m6_twinkle_20260923。
- 原视频相机围绕琴键，新增仅M6 full模式的全臂全琴中心[.22,0,.28]、默认距离1.65／仰角−25；其余视角保持原配置。用现有render_mjlab_checkpoint.py导出best18000到preview/outputs/replays/m6_twinkle_best18000_20260923，显式新scene/task，保留真实触键音轨及mocap轨迹。
- 实测回放F1=0.193968，不等于日志0.204550，明确分别记录。315控制帧中19.365%有键激活；3150子步音频有22次note_on、RMS0.033，未用目标谱配音。控制周期末forearm检测0碰撞帧，腕部最大连接残差<0.00015 mm；仅针对保存帧，不宣称全子步或全类别无碰撞。手指接近奖励高而激活稀疏是观测现象，不单独归因于IK或奖励。
- 验证：渲染完整退出；8.9秒、1280×720／40fps H.264＋AAC经ffprobe及完整解码通过；四时刻截图已查看。辅助分析全局python缺numpy，改用项目training_mjlab/.venv成功。更新训练文档、README及命令速查；V32-SAC-03回放差异、TJ-ARM-05学习效果与既有标定／动力学TODO保留，未自行更换论文设计。


## 2026-09-23：无机械臂参数与20/200 Hz小星星新训练

- 问题：原训练入口只能按冻结包自动加载M6；任务导出只传control_dt，改场景物理步长后会与CPU默认400 Hz不一致。首次临时GPU检查误用legacy归一化动作验证mixed_v1，被范围检查拒绝；修正检查脚本使用mixed_v1偏移后通过，未改策略映射。
- 参考：scripts/export_m6_bundle.py、scripts/export_mjlab_scene.py、mjlab_env/{bundle,task_bundle,scene,task}.py、rl/environment.py、rl_torch/train.py、suite/tasks/base.py及docs/references/biomimetics_2026_610_notes.md。
- 修改：rl_torch/train.py新增--no-arms；mjlab_env/without_arms.py为M6生成独立无臂scene/task副本并恢复跨手掌指名单；无臂输入直接校验使用，训练/评估/回放元数据用有效包。导出脚本增加--control-hz/--physics-hz；bundle写实际时序和solref；EnvironmentConfig新增可选physics_timestep，task导出传场景值。源40/400配置不改。
- 新包：scene_twinkle_noarms_20hz200hz_20260923、task_twinkle_noarms_20hz200hz_20260923；20/200 Hz、10子步、前瞻10、1180观测、51动作；小星星158帧，mixed_v1、自碰撞0.2、5,000,192采样从头训练。命令与PID见独立项目docs/training_noarms_20hz200hz_20260923.md。同步common_commands、AGENTS与论文决策登记。
- 实测：CPU场景导出往返和任务模型一致性通过；M6去臂副本及新包分别2环境20个GPU控制步通过，观测/奖励有限、无mocap/IK/臂查询、原M6输入哈希不变。证据training_mjlab/outputs/noarms_validation_20260923.log；正式run已产生309760条采样、6752次SAC更新，首轮F1约0.0886。长期训练尚未完成。
- TODO：保留V32-SAC-03与V32-06；此次不是全CPU/GPU轨迹对照或手部平滑改造。去臂策略不再受机械臂可达性限制。


## 2026-09-23：导出当前无臂小星星前期视频

- 需求／范围：当前sac-noarms-20hz200hz-20260923-123052最早保存step-1000，256000采样／32更新；只导出回放和文档。
- 参考：preview/scripts/render_mjlab_checkpoint.py、docs/common_commands.md、docs/references/biomimetics_2026_610_notes.md、本轮config.json/metrics.jsonl和生成report.json。
- 操作：既有GPU脚本，front/full正交相机，实际触键音轨；产物preview/outputs/replays/twinkle_noarms_early_step1000_20260923/replay.mp4及轨迹／报告。内部说明同步docs/training_noarms_20hz200hz_20260923.md。
- 验证：脚本退出0、完整158帧；ffprobe确认1280×720、20fps、8.95秒、H264/AAC；NVIDIA渲染，实际267次note-on。
- 问题／TODO：回放F1=0.073116，训练该检查点评估0.088638；保留V32-SAC-03，不声称精确一致。未改运行中训练或奖励配置。


## 2026-09-23：无臂小星星最佳step-17000视频

- 需求：导出本轮训练视频，选择已核实训练F1最高的step-17000（0.510970）。
- 参考：本轮metrics.jsonl／summary.json、preview/scripts/render_mjlab_checkpoint.py、docs/references/biomimetics_2026_610_notes.md及生成report.json。
- 操作：既有GPU回放脚本以front/full正交视角、实际触键音轨输出preview/outputs/replays/twinkle_noarms_best17000_20260923/replay.mp4；更新内部docs/training_noarms_20hz200hz_20260923.md的完成结果、视频和命令。未修改代码或参数。
- 验证：脚本退出0，完整158控制帧；ffprobe确认8.95秒、1280×720、20fps、H264/AAC，NVIDIA硬件渲染，69次实际note-on。
- 问题／TODO：回放F1=0.506751，与训练0.510970略有差异，V32-SAC-03保留；不声称精确一致。
## 2026-09-24：绘制 fingering reward 精确目标点

- 遇到的问题：现有 `preview_fingering.FingeringHighlight` 的彩球是显示标记，使用键局部坐标和额外悬浮高度，与奖励实际采用的世界轴偏移不同，不能直接回答奖励目标点位置。
- 参考：`mjlab_env/task.py` 的 `key_offsets`／`snapshot`、`mjlab_env/task_bundle.py` 的 CPU 快照、`suite/tasks/piano_with_tsensei_hands.py::_compute_fingering_reward`、`preview_fingering.py`、冻结 scene/task/score 及 `docs/references/biomimetics_2026_610_notes.md` p.11 登记。论文给出距离容差但未公开本手目标点几何，继续使用已确认的 CPU-compatible 定义。
- 修改：新增 `preview/scripts/draw_fingering_reward_targets.py`，从冻结模型按 `geom_xpos+[0.35*size_x,0,0.5*size_z]` 精确绘制88点、有效指法点和指尖距离；按用户后续要求增加 `--keyboard-map-output`，在每根白键／黑键上直接写音名与key编号并放置橙色奖励点。新增 `docs/fingering_reward_target_20260924.md`，并在 `preview/README.md` 区分显示标记与奖励点。
- 验证：以当前无臂小星星最佳回放第120帧生成正面／俯视1600×900 PNG、5200×1500逐键标注图和JSON；逐键图包含88个带音名／编号的琴键与88个橙色目标点，R3→E5 40.1287 cm、L2→A3 1.8571 cm，脚本输出坐标与冻结公式一致。只读渲染，未修改任务、奖励或训练。
- 未解决 TODO：V32-PAPER-05 中论文目标点几何仍未公开；现有交互预览彩球继续作为视觉标记，如需让其代表奖励点应另行修改显示语义。

## 2026-09-24：拇指白／黑键专用 Fingering 目标点

- 遇到的问题：用户最初的“底部朝上”可能被理解为Z高度；用户用键盘平面图明确为图中纵向X，从`+world X`底边向`−X`。首次GPU验收暴露`task.py`函数内残留局部`import numpy`导致`UnboundLocalError`，移除局部导入后重新验收通过。等待会话工具两次返回解析错误，但进程实际已完成，随后用产物和报告确认。
- 参考：用户上传的键盘标注图仅用于澄清方向；实际修改参考`fingering_targets.py`、CPU `_compute_fingering_reward`、GPU `task.py/task_logic.py/rewards.py`、task bundle和论文笔记p.11/V32-PAPER-05。附件不含执行指令。
- 修改：新增共享`fingering_targets.py`；仅R1/L1的X目标改为白键`geom_x+half_x−0.020m`、黑键`geom_x+half_x−0.015m`，Y/Z与其他手指不变。CPU、legacy GPU和paper可选奖励统一支持逐手指目标；新task manifest冻结规则，旧bundle无字段时保留旧逻辑。绘图脚本增加逐手指目标和小星星全拇指键放大图；新增说明并更新rew.总表、预览说明、论文决策登记和AGENTS。
- 新产物：`task_twinkle_noarms_20hz200hz_thumb_targets_20260924`（158帧、1180观测）；小星星R1=C5，L1=C4/D4/F4，均为白键，本曲没有黑键拇指。图见`preview/outputs/fingering_targets_thumb_20260924/`。
- 验证：20项Torch规则测试和3项CPU几何测试通过；新task导出通过；1环境GPU完整验收通过，固定CPU/GPU fingering奖励最大误差`1.2381e-7`、总奖励最大误差低于`2.4e-7`。白键测试X=0.055m、黑键测试X=0m，精确对应底边向上2/1.5cm；旧／新task分别以`1×88×3`／`1×10×88×3`目标执行GPU reset＋step通过。
- 未解决TODO：论文没有公开目标点几何，继续标为用户指定TsenseI适配。旧checkpoint没有按新奖励训练，若需效果必须使用新task重新训练。


## 2026-09-24：手根12 cm与49维关闭Wrist_TZ

- 用户确认：双手根高度0.12 m；左右TZ固定0 mm、移除两个策略通道，新增训练开关--disable-wrist-tz，旧51维路径兼容。
- 问题：仅ctrl=0不能保证固定；尝试等式约束后40随机步偏移57.3 mm，不合格。最终采用每物理子步固定虚拟TZ坐标投影（qpos/qvel/warmstart=0），graph和逐步路径一致。中间h12_tz49_smoke已标记INVALID_EXPERIMENT。读取发现并行新增fingering_target，派生任务保持上一轮冻结规则以隔离变更。
- 参考：models/hands/tsensei_hand_constants.py、mjlab_env/{bundle,scene,task,task_bundle,action_mapping}.py、rl_torch/{train,sac,replay,checkpoint,evaluate}.py、preview/scripts/render_mjlab_checkpoint.py、论文决策登记及上一轮冻结scene/task。
- 修改：源高度14→12 cm；scene/task接49动作并补齐51物理控制；action_mapping适配压缩索引；train新增开关、policy_action_names、自动49维Actor/Critic/replay与−24.5目标熵；checkpoint恢复/验证TZ合同。新增scripts/train_twinkle_h12_no_tz.sh与20/200 Hz h12新包，原Yaw/Pitch/奖励保持。更新内部docs/wrist_tz49_height12_20260924.md、common_commands及AGENTS/论文登记。
- 验证：源场景导出往返通过；与上一包主要物理数组相同，仅两处根body_pos.z改动。graph和逐步模式各2环境40随机步，TZ位置/速度/控制均0，观测/奖励有限且局部reset通过；2环境8向量步、7次编译SAC更新、融合Adam/replay和全曲独立评估通过，保存49维checkpoint。正式5M未启动。
- TODO：V32-TZ-01固定坐标投影不是移除实体关节的精确动力学，反力需求需独立模型改造；保留V32-SAC-03、V32-06，不声明长期收敛。

验收补充：保存后的49维检查点视频回放完整158帧，actions形状(158,49)，TZ qpos/ctrl全为0；旧51维检查点在原CUDA设备恢复通过。首次尝试CPU恢复旧CUDA RNG触发已有设备状态不兼容，改在原CUDA验证；未扩展跨设备RNG迁移范围。


## 2026-09-24：按用户要求将TZ投影替换为不创建关节

- 问题：先前--disable-wrist-tz保留实体自由度并每子步强制归零，不符合用户要求的物理锁止。不得静默重解释旧投影检查点。
- 参考：本地robopianist/models/hands/shadow_hand.py的_DEFAULT_FOREARM_DOFS/_add_dofs（仅阅读）；本包models/hands/tsensei_hand.py、mjlab_env/{bundle,scene,task,task_bundle,rewards}.py、rl_torch/{train,checkpoint}.py及论文决策登记。
- 修改：SceneBundle在MuJoCo编译前从输入XML排除两TZ关节和驱动器，模型136/136/48；按名字重建reset和控制索引。scene删除投影kernel和所有纠正调用；task为兼容观测／奖励补零槽，物理控制48维。train/checkpoint新增joint_removed_v1合同并拒绝旧49维投影检查点，原51维可用。未改源冻结包、并行拇指目标点修改或其他奖励。
- 验证：CPU原生100子步及GPU graph／逐步各2环境40控制步，施加两手各1000N向下力，高度保持0.12m；无TZ关节／驱动器，观测奖励有限、局部reset通过、51维分支模型完整。新增tests/test_tz_structural.py验证拓扑、外力和旧合同拒绝。4向量步SAC/Replay/融合Adam及评估保存通过（学习器编译本短测关闭）。完整回放结果见内部说明。临时测试命令一次使用错误工作目录，修正路径后重跑。
- 文档：更新docs/wrist_tz49_height12_20260924.md和决策登记，保留历史投影记录并明确被本次替代；验收日志training_mjlab/outputs/tz_structural_validation_20260924.log。没有启动长期训练。
- TODO：V32-TZ-01结构性锁止已解决，真实支架柔性未建模；V32-SAC-03、V32-06继续保留，短测不代表训练收敛。


## 2026-09-24：同手自碰撞取消反力、保留检测扣分

- 需求：用户明确选择同手自碰撞仅检测／扣分；保持手—琴键实体接触。没有启动长期训练或重启用户现有窗口。
- 参考：mjlab_env/self_collision.py、scene/bundle/task、preview/scripts/preview_pose.py、rl_torch/train/checkpoint、physics_settings.py、论文决策登记；MuJoCo官方computation/#margin-and-gap、已安装MuJoCo-Warp collision_core/io实现。
- 实现：新增detection_only_collision.py；CPU预览用inactive pairs并默认开启（可--no-self-collision-detection-only）；GPU训练新增--self-collision-detection-only，主模型掩码取消同手接触，原几何独立查询每子步供奖励去重，独立接触数组不参与力求解。checkpoint冻结／核验模式；12cm无TZ启动脚本启用。原模型资产、接触材料、跨手／琴键接触及奖励权重保持。
- 问题：GPU MULTICCD不支持负pair margin，未绕过检查或全局关闭MULTICCD，改用独立查询模型。首次测试命令工作目录错误，修正后重跑成功。
- 验证：3项回归测试通过，包含CPU重叠零反力／琴键约束正常和GPU graph／逐步主接触无同手且奖励负值；各2环境40控制步补充测试有效。源预览headless渲染通过；SAC 4步、3次更新、评估／保存通过；回放结果见内部说明。日志training_mjlab/outputs/self_detection_validation_20260924.log、self_detection_tests_20260924.log。
- 文档：docs/self_collision_detection_only_20260924.md、common_commands、wrist_tz49说明和论文决定登记同步。使用pianist-v2-change-log技能记录。
- TODO：256环境长期吞吐和显存开销未测；GPU有M6输入需先--no-arms；V32-06与V32-SAC-03保留。允许同手穿透是用户所选检测模式，不宣称实体接触真实性。
## 2026-09-24 — 预览逐项核对原版 Shadow 接触参数

- **目的／范围**：按用户要求只同步并核对原版 Shadow 的接触环境参数，更新 V3.2 CPU／GPU 预览；不修改腕部自由度、动作映射、手部几何、训练任务或控制／物理频率。
- **问题与参考**：当前 `assets/contact_settings.json` 已是 Shadow 参数，不能把40/400 Hz钢琴的0.005秒误改成原版20/200 Hz时的0.01秒；原版规则是钢琴 `solref[0]=2×physics_dt`。参考 `docs/physics_shadow_comparison.md`、原版 `robopianist/suite/tasks/base.py`、本包 `physics_settings.py`、冻结 scene manifest 和两个预览入口。基础 `.venv` 没有Torch，训练 `.venv` 没有dm_control；两次把CPU／GPU测试混在单一环境执行分别出现依赖导入失败，随后按各自配套环境拆开重跑全部通过。
- **修改**：`physics_settings.py`新增与步长解耦的Shadow参考构造、实际模型逐项比较、合并手—键参数／`refsafe`报告和三组Shadow solimp复位；CPU预览从编译模型读取实际值，终端打印MATCH／DIFF并把完整结果写入`pose.json`；GPU预览始终打印相同核对，接触面板新增“恢复原版 Shadow solimp”。同步预览、验收和常用命令说明。
- **验证**：训练环境接触测试5 passed，CPU环境预览控制回归8 passed；两套冻结场景只读核对均MATCH：40/400 Hz合并solref `(0.005,1)`，20/200 Hz合并solref `(0.0075,1)`且refsafe为0.01秒；CPU NVIDIA/EGL无臂离屏预览成功，报告MATCH并输出PNG／JSON；两预览脚本语法和GPU入口帮助检查通过。
- **未解决**：V32-06接触力／真实材料标定继续保留；Shadow参数仍是MuJoCo软约束，不等于零穿透或实体TsenseI材料标定。滑块只编辑solimp，其他Shadow参数通过启动核对显示，不在面板中编辑。


## 2026-09-24：左右手碰撞反力关闭、保留forearm检测

- 需求：用户明确要求左右手之间也只检测碰撞，保留手—琴键真实接触。此前只有同手不产生反力。
- 参考：detection_only_collision.py、mjlab_env/{bundle,scene,task,self_collision}.py、rl_torch/{train,checkpoint}.py、preview/scripts/preview_pose.py与论文决策登记。
- 修改：新增hand_collision_detection_only全链路开关和检查点合同；CPU inactive pair扩大到跨手，GPU主模型掩码屏蔽所有手—手；forearm改从独立查询接触读取，reset／周期末无条件刷新查询，确保关闭同手奖励时仍能检测跨手。preview默认全手仅检测，12cm无TZ包装脚本同步。旧同手模式合同不改，原冻结包不覆盖。
- 验证：新增scripts/validate_hand_detection_only.py：CPU左右手相交无约束而琴键约束有效；GPU graph／逐步各2环境35步，主接触无手—手，forearm识别重叠为0，self_collision_reward关闭时同样有效。headless预览成功；4步SAC/Replay/评估/保存通过（学习器编译关闭），回放见内部说明。日志all_hand_detection_validation_20260924.log。
- 文档：docs/all_hand_detection_only_20260924.md；common_commands、12cm说明、旧同手说明追加覆盖说明，AGENTS／论文登记同步。遵循pianist-v2-change-log日志技能。
- TODO：V32-SAC-03、V32-06及长期吞吐验收保留；未启动长期训练或重启用户窗口。琴键接触或驱动导致的抖动不在本次保证范围。


## 2026-09-24：参考用户指定 Shadow 预览，定位接触振荡并恢复三类实体碰撞

### 问题与决定

用户撤回默认无反力处理，要求查明原版 Shadow 稳定、V3.2 手指接触抖动的原因。无GUI固定目标也能复现：导入的轻量CAD关节armature=0、暂定Kp=3，在接触中出现持续振荡。最初只加armature改善同手但旧12cm／移除TZ姿态的跨手拇指仍抖，CPU速度RMS=0.136rad/s；进一步Kp=1通过复验。Shadow frictionloss=0.01或增加阻尼没有稳定解决所有案例，未采用。

### 参考与修改

- 按用户指定命令只读解析 `/home/houjue/oldproj/DaxianPianist/preview_model.sh`、`robopianist-rl/preview_model.py`、`robopianist-rl/run.sh`、`robopianist-rl/train.py`、其Shadow XML/hand构建；在现有Docker容器实例化其实际20/200Hz环境，原工程未修改。Shadow日志保存在本次诊断输出目录。
- MuJoCo官方joint armature定义：https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-armature。参考本项目 `models/hands/urdf_model.py`、`tsensei_hand_constants.py`、`preview_runtime.py`、`physics_settings.py`、`mjlab_env/bundle.py/scene.py/task.py/task_bundle.py`、论文笔记、旧检测模式及高度/TZ说明。
- 40个手指关节armature=0.0002kg·m²、原生位置驱动Kp=1；原CAD质量/惯量、腕部、0.05阻尼、0.6Nm限幅及接触材料保留。这是仿真暂定参数，不是硬件实测。源预览两种detection-only开关默认false，JSON和启动信息说明实体接触及新参数；旧模式仍可显式启用。
- `scripts/export_contact_stability_bundle.py`派生不可覆盖的新包 `training_mjlab/outputs/contact_stable_h12_20hz200hz_20260924/{scene,task}`，只改手指参数，保留冻结姿态/乐谱/奖励并更新hash；不复用旧CPU轨迹。`scripts/train_twinkle_h12_no_tz.sh`换新包、去掉无反力开关。未启动训练。
- `mjlab_env/task_bundle.py`新增armature/阻尼/关节摩擦比对，避免新任务和旧物理参数静默混用。新增两个持续接触验证脚本；README、preview说明、常用命令、Shadow对比、历史方案覆盖提示、论文决定登记、AGENTS及本日志同步。

### 实际验证

证据目录 `preview/outputs/contact_stability_20260924/`；CPU MuJoCo3.13.0，GPU MuJoCo3.11.0/MJLab/Warp/RTX PRO 6000。

- `scripts/validate_contact_stability.py`：固定目标20秒、末2秒逐物理步统计，200Hz无TZ。同手／跨手／手—键CPU峰峰值由33.6623°／2.2213°／2.6798°降至0.007267°／0.003114°／0.000116°；GPU分别0.007319°／0.001981°／0.0000032°。三类实测约束力非零，GPU自碰撞reward=-0.4，跨手forearm=0。前段CUDA graph、末段子步回调；非逐轨迹严格等价验收。
- `scripts/validate_preview_contact_stability.py`：当前Composer源预览额外forward节奏、400及200Hz各三类，6例通过。跨手仍有低速调整（400Hz末2秒0.475°，200Hz1.19°），不能声称任何姿态都完全静止。
- headless新源码预览NVIDIA渲染成功，`stable_preview.json`确认physical三类、armature0.0002、Kp1；未关闭或重启用户当前桌面窗口，旧进程需重启。
- 新CPU包真实按键：TY=.04、Pitch=0、右食指MCP1.2/PIP.3/DIP0，1秒后激活key52；仅接触保持案例不保证激活阈值，另做了此测试。
- `training_mjlab/.venv/bin/python -m pytest -q tests/test_detection_only_collision.py`：3 passed，历史模式保留。最初在不含Torch的CPU venv执行该混合测试失败，切换正确环境后通过。
- 验证中修正诊断脚本对Composer reset后失效Physics指针的复用；手—键案例旧TY=0未触键，设明确TY=.04并断言非零反力，未把无接触静止算作通过。旧配置按设计失败，原始结果保留。
- 对照图 `contact_stability_comparison.png` 已生成并检查；具体命令和试验说明见 `docs/contact_stability_shadow_20260924.md`。

### 保留事项

V32-04电机惯量／Kp／阻尼和限幅标定，V32-06凸包/装配与全部极限姿态验收，V32-SAC-03CPU/GPU接触差异。没有长期SAC训练/F1验收。原冻结包／已运行进程仍保留旧参数；源当前姿态未被旧冻结姿态覆盖。奖励公式／权重未改变，论文未知映射不代填。


## 2026-09-24 08:14：启动修复接触后的小星星5M训练

- 用户要求：开启新的小星星训练；按现有包装脚本从头训练，不恢复旧检查点。
- 问题：启动前无其他Torch SAC训练进程，GPU仅约1.6GiB显存占用；本次启动未遇到阻断错误。预热阶段ETA低于进入32次SAC更新后的耗时，不据预热速度推算全程。
- 参考：`scripts/train_twinkle_h12_no_tz.sh`、`rl_torch/train.py`、新冻结场景/任务manifest、`docs/contact_stability_shadow_20260924.md`、`docs/wrist_tz49_height12_20260924.md`、论文决策登记与AGENTS。沿用已确认论文Stage1及本轮接触修复，不引入新论文参数。
- 操作：2026-09-24 08:14:36+08:00通过独立后台session启动，PID3826442；输出`training_mjlab/outputs/sac-twinkle-contact-stable-20260924-081436`，同级`.log`和`.launch.json`记录进度/精确命令/启动时间。预算19532向量步×256环境=5,000,192条；20/200Hz、12cm、无臂、TZ物理移除、49动作mixed_v1、自碰撞0.2、三类实体接触；物理graph、学习器编译、融合Adam和进度条开启。
- 修改：仅新增本次运行产物和`docs/training_twinkle_contact_stable_20260924.md`，更新README、常用命令及根/内部日志；未修改训练算法或冻结包。
- 验证：inspect-only确认实际开关/预算；实际config确认49动作、1180观测、目标熵−24.5。启动约2分钟时1410向量步/360960采样、13152梯度更新，损失有限，GPU训练进程仍运行。第1000向量步保存`step-1000.pt`，首次158帧完整评估F1=0.08015475，无截断；尚不能判断最终收敛。
- 保留：V32-04驱动标定、V32-06全范围碰撞验收、V32-SAC-03接触/复播差异，长期F1待训练完成。任务当前后台继续，无新增代码测试需求。


## 2026-09-24：接触稳定训练最佳step-19000视频

- 范围：导出sac-twinkle-contact-stable-20260924-081436日志最佳检查点，F1=0.673748；不改模型、训练、奖励。
- 参考：该run的config/metrics/summary、preview/scripts/render_mjlab_checkpoint.py、mjlab_env/paths.py错误栈、论文决策登记及生成report.json。
- 问题：默认查找未找到嵌套scene/task；改为显式传原冻结包路径，哈希校验通过。未改查找器。
- 产物：preview/outputs/replays/twinkle_contact_stable_best19000_20260924/replay.mp4及轨迹／报告／音轨；同步docs/training_twinkle_contact_stable_20260924.md记录完成统计及准确导出命令。
- 验证：渲染退出0，完整158控制帧；ffprobe确认8.95秒、1280×720、20fps、H264/AAC；98次实际note-on。回放F1=0.660579。
- TODO：V32-SAC-03训练/独立回放差异保留，不声明完全复现日志评估。


## 2026-09-24：从接触稳定最佳step-19000续训5M

- 用户要求：在最佳视频对应策略基础上再训练5M。沿用原场景、49维／TZ结构移除、20/200 Hz、自碰撞0.2及Stage1 SAC。
- 参考：scripts/train_twinkle_h12_no_tz.sh、rl_torch/train.py与checkpoint.py的warm-start机制、源run config/metrics、docs/references/biomimetics_2026_610_notes.md。
- 操作：启动`sac-twinkle-contact-stable-continue5m-20260924-115525`，PID 17055，新增5,000,192采样；恢复网络/优化器/温度/RNG，新replay/仿真；warmup设0用恢复策略采样，learning-starts保留1000。没有改训练代码和奖励；新增docs/training_twinkle_continue5m_20260924.md及原训练文档链接，launch.json保存精确命令。
- 验证：启动前无其他Torch SAC训练，源和新run scene/task哈希、reward/SAC/动作/TZ/接触合同一致，origin正确恢复step19000／576032次更新；GPU已开始采样，训练继续。
- 问题与TODO：无启动异常；warm-start不含旧replay，不是精确续跑。长期结果尚待完成，保留V32-SAC-03、V32-04／06，不声称提升。

启动实测补充：已采样273920条，累计updates=578304（本轮新增2272次），第一轮评估F1=0.622739。GPU学习更新、独立评估、step-1000保存均已执行；属于续训初期。


## 2026-09-24：新增关闭Yaw的47维训练

- 用户确认：在无TZ训练基础上关闭Yaw，固定各手冻结初始角；本轮两侧0.25rad。从头新增5,000,192采样，原续训不停止。
- 问题：Yaw与其余腕关节共用body，直接删除Yaw会恢复零位，直接只旋转body又会改变其余关节轴。实现静态旋转和其余轴反向变换，保持运动学；47维与旧49维权重不兼容，明确从头训练。
- 参考：mjlab_env/{bundle,scene,task,action_mapping}.py、rl_torch/{train,checkpoint,sac,replay}.py、当前冻结scene/task manifest、scripts/train_twinkle_h12_no_tz.sh、论文决策登记。
- 修改：train新增--disable-wrist-yaw（需--disable-wrist-tz）；bundle结构删除Yaw/驱动并固化初始旋转；scene/task补固定观测及51物理控制槽；mapping增加47维；checkpoint恢复和检查Yaw合同；新增scripts/train_twinkle_h12_no_tz_no_yaw.sh，内部说明docs/training_no_yaw47_20260924.md及common_commands、AGENTS/论文登记。
- 验证：10组随机其他关节CPU运动学对照，所有geom位置/旋转在1e-10内一致；2环境4步编译SAC/replay/融合优化器、3次更新和158帧评估保存通过。正式run sac-twinkle-no-yaw47-20260924-120155，PID33512，launch.json保存命令。
- TODO：训练仍进行中，共享GPU影响速度；保留V32-SAC-03、V32-04/06，不声明效果提升。Yaw单独关闭但保留TZ的49维布局暂不支持，参数检查明确拒绝。

启动验收补充：正式run确认47动作／1180观测／目标熵−23.5、disable_wrist_yaw=true，已采样115,200条，当前补充replay、尚未到1000向量步学习起点；短程检查点独立视频完整158帧通过。


## 2026-09-24：导出当前小星星训练视频

- **目的与参考**：按用户要求将当前 V3.2 训练导出视频。前一轮对话涉及同时运行的 47 维新训与 49 维续训，故两轮分别导出。参考各自 `config.json`、`metrics.jsonl`、已保存检查点、`preview/scripts/export_best_mjlab.py`／`render_mjlab_checkpoint.py`、冻结 `contact_stable_h12_20hz200hz_20260924/{scene,task}`、论文笔记 V32-PAPER-11 已决定保留的逐帧评价口径，以及日志技能。
- **遇到的问题**：导出脚本自动查找只遍历 `training_mjlab/outputs/*/manifest.json`，未找到嵌套 `.../scene/manifest.json`；首次 47 维选择成功但导出未开始。随后显式传原冻结 scene/task 完成两段导出，未改源码或训练配置。复播 F1 与训练日志略有差异，报告如实标记 `logged_metrics_match=false`，保留 V32-SAC-03。
- **具体产物**：`preview/outputs/replays/twinkle_47_best_20260924/` 选择 `step-18000.pt`（日志 F1 0.8615433，复播 F1 0.8611215）；`preview/outputs/replays/twinkle_49_best_20260924/` 选择 `step-16000.pt`（日志 F1 0.9090417，复播 F1 0.9039783）。每目录包含 `replay.mp4`、`selection.json`、`report.json`、过程截图及轨迹／音频资料；更新 `preview/README.md` 的视频链接和显式场景路径说明。
- **验证**：两次脚本退出码 0；`ffprobe` 分别确认 H.264 1280×720／20 FPS、AAC 音轨、8.95 秒；报告均为完整 158 步，实际琴键触发音符，47／49 维复播按键 F1 如上。视频为冻结检查点的独立确定性复播，不是训练中的实时画面；音符力度尚未标定。训练进程在检查时均已结束，本次未启动训练。
- **TODO**：V32-SAC-03 继续跟踪复播与日志的数值差异；V32-AUDIO-01 音符力度标定未完成。其余既有模型／接触 TODO 不变。

## 2026-09-24：预览／训练默认开启手部重力补偿

- **范围与问题**：用户要求两处均开启。旧 49／47 动作训练包关闭重力补偿且两轮运行已结束。当前源码初值 TY=0.04、Yaw=0.15 与旧冻结初值 TY=0、Yaw=0.25 不同；直接导出后 `derive_gravity_task.py` 的初值断言失败。改从旧冻结 scene 派生，只开启左右手 44 个 body 的 `gravcomp`，保持原姿态与任务合同；旧检查点不跨场景续训。原 `test_mjlab_bundle.py` 固定触键动作在当前源码姿态的 40/400 Hz 导出下不再触键；改用本轮 20/200 Hz、TY=0.04／MCP-PIP=0.85 的实际触键目标作物理验收。
- **参考**：`preview/scripts/preview_pose.py`、`scripts/export_mjlab_scene.py`、`mjlab_env/{bundle,paths,task_bundle}.py`、`rl_torch/train.py`、`scripts/train_twinkle_h12_no_tz*.sh`、旧冻结 scene/task manifest、`models/hands/tsensei_hand_constants.py`、`docs/references/biomimetics_2026_610_notes.md` 和日志技能。论文未规定本次补偿值；没有变更论文 SAC／奖励决定。
- **修改**：源姿态预览增加默认开启的可选开关及 JSON 记录；新导出默认开启；新增 `scripts/derive_gravity_scene.py`、`scripts/derive_gravity_task.py`，生成 `training_mjlab/outputs/contact_stable_h12_20hz200hz_gravcomp_20260924/{scene,task}`。GPU 预览、Torch 默认路径、49／47 动作包装脚本指向新包；`mjlab_env/paths.py` 递归查找嵌套冻结包以按散列回放旧检查点。更新物理触键回归测试、README、`preview/README.md` 与 [配置／验证说明](gravity_compensation_20260924.md)。
- **实际验证**：Python 编译检查通过；`MUJOCO_GL=egl` 的源预览离屏渲染成功，打印补偿 on 且 Shadow 接触核对 MATCH；派生脚本核对编译模型的关节、驱动、接触、质量、惯量、初值，仅 44 个 hand body 的 gravcomp 改变；新 task 加载通过。相关 CPU 回归 `8 passed, 1 deselected`（GPU 滑块项未运行），含当前 20/200 Hz 源导出触键／松键；新旧检查点分别找到新旧冻结包。2 环境×4 GPU 向量步、8 采样／3 SAC 更新及保存通过。新 47 动作长期实验 PID 270182 已启动并进入学习及首次评估；`training_mjlab/outputs/sac-twinkle-no-yaw47-gravcomp-20260924/` 的配置是 47 动作、1180 观测、目标熵 −23.5，目标采样 5,000,192 条。49 动作长期实验未启动。
- **TODO**：V32-GRAV-01 等长期结束再比较学习／碰撞／回放；V32-04、V32-06、V32-SAC-03 保留。新 task 未继承旧物理 CPU 参考轨迹；本轮未验收实机重力补偿。

## 2026-09-24：以当前训练冻结场景同步源码手根与预览／导出

- **目的和问题**：用户要求冻结训练与源码一致，并特别确认同步手根高度。实际源安装高度0.10m、TY=0.04、左右Yaw=0.15，训练冻结scene为0.12m、TY=0、左右Yaw=0.25；源预览／导出默认40/400Hz，训练20/200Hz。旧触键测试固定动作在同步姿态下产生接触但不激活琴键。`urdf_model.py` 的源散列与旧冻结记录不同，核查编译物理数组未见除上述姿态外的差异。
- **参考**：当前 `contact_stable_h12_20hz200hz_gravcomp_20260924/{scene,task}` manifest及47维run config，`models/hands/tsensei_hand_constants.py`、`mjlab_env/bundle.py`、`preview/scripts/preview_pose.py`、`scripts/export_mjlab_scene.py`、`assets/contact_settings.json`、`tests/test_mjlab_bundle.py`、论文时序笔记和本机日志技能。
- **修改**：常量中的左右手根Z改0.12m、两侧TY=0／Yaw=0.25；源预览与新导出默认20/200Hz，保留显式40/400Hz参数及手部重力补偿；导出默认目录名与时序一致。真实触键回归用TY=0.04、右食指MCP=1.2/PIP=0.3/DIP=0从冻结初值按下键52。更新README、preview说明、重力补偿对照、论文登记和根／内部AGENTS；冻结scene/task及正在运行的进程均未写入。
- **验证**：临时默认导出与训练scene在19类模型数组、物理选项、qpos/qvel/ctrl、动作范围、接触配置和手位姿上完全相同；源预览EGL离屏成功，20/200Hz、重力补偿on、Shadow接触MATCH，显式40/400Hz也通过；`pytest tests/test_mjlab_bundle.py tests/test_preview_controls.py -q` 为12 passed，环境／接触CPU测试23 passed、1个GPU项未运行。冻结scene/task SHA-256仍与训练config一致，未进行新的长期训练效果验收。
- **剩余差异／TODO**：裸 `suite.load()` 继续沿用历史40/400Hz contact_settings默认，且裸任务默认关闭重力补偿；调用者需显式传20/200Hz及 `gravity_compensation=True` 才能复现本轮物理场景。源预览M6只是可选运动学显示，47维TZ/Yaw锁止和任务／奖励仍在训练入口。V32-GRAV-01及既有驱动／碰撞／CPU-GPU差异TODO保留。

## 2026-09-24 14:47：无重力补偿47动作小星星再续训5M

- **目的与参考**：用户明确选择旧47动作分支，与当前重力补偿47动作训练对照。读取 `sac-twinkle-no-yaw47-20260924-120155` 的 config／summary／final checkpoint、旧 `contact_stable_h12_20hz200hz_20260924` scene/task、`rl_torch/{train,checkpoint}.py`、旧49动作续训说明、论文决策登记与日志技能。
- **问题与处理**：`train.py` 现在默认重力补偿包，旧checkpoint无法匹配。显式传无补偿旧包；启动前检查checkpoint schema和source scene/task散列一致，且47动作、1180观测、mixed_v1、TZ/Yaw物理锁止、self_collision=0.2、Stage1设置与旧运行相同。热启动恢复网络／优化器／温度／探索RNG，但重新创建GPU replay和仿真，不能视为逐步精确续跑。两轮并行共享GPU，速度可能降低。
- **操作**：后台PID336820，目标额外5,000,192采样；`--warm-start`旧 `final.pt`（step19532、updates593056）、`--warmup-vector-steps 0 --learning-starts 1000`，其他原47动作训练参数保留。运行目录 `training_mjlab/outputs/sac-twinkle-no-yaw47-legacy-continue5m-20260924-144738`；同级 `.launch.json`／`.log` 记录命令和进度。[独立运行文档](training_no_yaw47_legacy_continue5m_20260924.md)及README同步；无算法、模型或旧冻结包源码改动。
- **实际验证**：进程运行且已开始GPU采样；实际 `config.json` 显示旧scene/task散列、47动作、1180观测、origin正确恢复旧step／updates。本轮仅启动，未达到5M完成或效果验收；仅改运行产物与文档，未跑新代码测试。另有重力补偿47动作新训并行；旧47首轮与旧49链已完成。
- **TODO**：V32-LEGACY-47-CONT-01 完成后比较训练曲线、最佳F1与独立复播；V32-SAC-03的复播差异仍保留。

启动后核对：新run已过learning_starts=1000，向量步1020／261,120采样、累计updates=593,728；首轮完整评估F1=0.762553，`step-1000.pt`已保存。训练继续。

## 2026-09-24：当前47动作小星星两轮视频和F1曲线

- **目的与参考**：用户要求查看当前训练结果并导出视频、F1训练曲线。读取正在运行的旧无补偿47动作续训、重力补偿47动作新训以及已完成旧47动作首轮的 `config.json`／`metrics.jsonl`／checkpoint；使用 `preview/scripts/export_best_mjlab.py`、`render_mjlab_checkpoint.py`、现有绘图约定 `scripts/plot_training_f1.py` 和论文笔记V32-PAPER-11；按日志技能记录。
- **问题及边界**：两轮训练未完成，最新最佳可能改变；旧续训已从5M预训练策略启动、补偿轮从零训练，不能将图上差异单独归因于重力补偿。`logged_metrics_match=false`：补偿日志／独立回放F1=0.829958／0.830591，旧续训两者均0.845509但回报有约0.00589差异。保留V32-SAC-03，不称逐位复现。
- **产物与修改**：`preview/outputs/replays/twinkle_legacy47_continue_snapshot_20260924_1457/`选`step-2000.pt`，`preview/outputs/replays/twinkle_gravcomp47_snapshot_20260924_1456/`选`step-14000.pt`；各含视频、音频来源、selection／report、轨迹。`preview/outputs/training_reports/twinkle_gravity_comparison_snapshot_20260924_1456/`含PNG／PDF双面板F1图、评估CSV、源日志SHA-256与快照报告。新增[结果说明](training_video_f1_snapshot_20260924.md)，更新README和preview说明；无训练代码／参数／场景变更。
- **验证**：两次显式匹配各自旧／新scene/task导出退出0，完整158控制帧，检查点散列验收通过；ffprobe均为H.264＋AAC、1280×720、20fps、8.95秒。回放F1分别0.845509／0.830591；真实琴键note-on为39／41，人工查看两张中间帧与曲线图。图为原始`eval.f1`，旧首轮20点、旧续训2点、补偿16点；两轮在导出后仍运行。
- **TODO**：V32-LEGACY-47-CONT-01、V32-GRAV-01待最终5M完成后重新评价；V32-SAC-03独立复播差异、V32-AUDIO-01固定音符力度保留。

后续更新：重力补偿轮已完成5,000,192采样，最终`final.pt`成为本轮最佳，日志／独立回放F1均0.844304；旧无补偿续训仍进行，`step-5000.pt`最佳日志／回放F1均0.848885。新增`preview/outputs/replays/twinkle_gravcomp47_completed_best_20260924/`和`twinkle_legacy47_continue_updated_snapshot_20260924/`两段最新版视频，ffprobe再次确认H.264/AAC、1280×720、20fps、8.95秒，note-on分别37／36。新增`scripts/plot_twinkle_gravity_comparison.py`可只读重新生成曲线；最新快照含旧首轮20、续训6、补偿20个评估点，PNG／PDF／CSV及日志散列见结果文档。早期快照保留供历史追溯。V32-GRAV-01继续比较同等训练预算与接触稳定性，不能把当前旧续训和补偿从头训练的F1差额归因于补偿。

交付前再更新：旧续训刷新至`step-7000.pt`，日志和独立回放F1均0.853105；导出`preview/outputs/replays/twinkle_legacy47_continue_step7000_20260924/`，实际34次note-on、H.264/AAC、8.95秒。最新曲线改为`preview/outputs/training_reports/twinkle_gravity_comparison_step7000_20260924/`，含旧续训7个评估点；旧训练继续运行，报告已链接本阶段检查点，未称最终最佳。

## 2026-09-24：清理旧训练场景，改为源码构建并缓存快照

- **目的与范围**：用户要求删除无用训练场景，采用“脚本＋可选参数覆盖源码＋源码导出 scene/task＋缓存”的新训练入口；随后确认旧检查点不再需要回放。没有停止、改参或重启正在使用 `contact_stable_h12_20hz200hz_20260924` 的旧 47 动作训练。
- **问题及处理**：16 个旧场景中很多被 task、训练配置、测试或预览引用，先按 manifest SHA-256 和源码路径审计。导出环境分离：`.venv` 有 dm-control 但缺 `dm_env_wrappers`，`training_mjlab/.venv` 有 Torch 但缺 dm-control；最终分别使用 `.venv` 导出 scene、`training/.venv` 导出 task、`training_mjlab/.venv` 训练。旧 h10 派生 task 不含拇指目标点，直接源码 task 会启用它；保留此实际差异并写入文档，不宣称只变更缓存。源码常量此前已按用户指定改为 h0.10／TY0.04／Yaw0.15，但尚未形成完整日志，本条一并补记；`suite/tasks/base.py` 已设重力补偿默认开启，源预览／导出 CLI 始终开启。
- **参考**：`models/hands/tsensei_hand_constants.py`、`suite/tasks/base.py`、`mjlab_env/{bundle,task_bundle,paths}.py`、`rl_torch/{train,checkpoint}.py`、`scripts/{export_mjlab_scene,export_mjlab_task,derive_pose_task}.py`、`preview/scripts/draw_fingering_reward_targets.py`、旧运行 `config.json`、各 scene/task manifest、`docs/pose_h10_ty04_yaw015_gravcomp_20260924.md`、`docs/references/biomimetics_2026_610_notes.md`、本机日志技能。论文 SAC／奖励决定未更改；拇指目标规则来自用户此前确认的 TsenseI 适配。
- **修改**：`models/hands/tsensei_hand_constants.py` 增加仅在导出子进程读取的 `TSENSEI_V32_EXPORT_POSE`（height／ty／yaw，校验物理范围）；`mjlab_env/bundle.py` 将有效源姿态写入新 manifest；新增 `scripts/train_from_source.py`，将源码／assets、导出脚本、依赖版本、外部 MIDI 散列及覆盖参数编入缓存键，完整性校验、文件锁、临时目录原子发布后调用原 Torch SAC。h10 的 49／47 动作包装脚本改用此入口。清理八个目录：`scene_before_hard_contact_20260917`、`scene_gravcomp_template_20260924`、`scene_hard_contact`、`scene_wrist25`、`contact_h12_20hz200hz_20260924`、`scene_clair_shadow40_pose_20260920`、`task_clair_de_lune_beginner_shadow40_fingered_v1`、`m6_93mm_twinkle_20260923`。运行配置、指标和检查点文件保留。更新 README、常用命令、历史模板说明、h10 姿态说明、根／内部 AGENTS 及[完整设计与清理清单](source_bundle_cache_20260924.md)。
- **实际验证**：源码缓存首次构建成功，重复运行返回 `CACHE HIT`；47 动作包装脚本 `--inspect-only` 显示 mixed_v1、自碰撞0.2、5,000,192 目标采样，未启动新训练。新 scene 为20/200 Hz、h0.10／TY0.04／Yaw0.15、44个手部 body 重力补偿；task 为小星星158帧／1180维／10步前瞻，含拇指目标点。与旧 h10 包对比，14类编译模型数组及 reset qpos/ctrl 相同，`score.npz` 散列相同；task 的指法目标和 CPU 参考不同。CPU 环境／场景／预览／接触回归 `35 passed, 1 deselected`；结构锁止测试在 Torch 环境 `2 passed`。首次把 GPU 接触测试放入不含 Torch 的 `.venv` 时出现 `ModuleNotFoundError: torch`，已改为在该环境只运行 CPU 项；GPU 接触测试本轮未跑。旧训练 PID336820 在清理后仍运行，指向保留的 scene/task。没有做新长期训练或收敛验收。
- **TODO**：V32-CACHE-01 若需恢复已删场景的旧检查点，需找回散列完全相同的原包；当前源码重导出不能替代。V32-GRAV-01 新姿态长期效果和 V32-SAC-03 回放数值差异仍未验收；V32-04／06 模型驱动与接触标定继续保留。

## 2026-09-24：并行环境 reset 动作历史修复并从头重训

- **问题与结论**：物理 reset 恢复冻结 scene 的中立 qpos/qvel/ctrl，但旧 task 将 51 维上一动作观测填为 `BoundedArray` 物理下限。旧 47 动作策略探针显示，改成中立历史会使首步策略输出最大变化 1.2076；旧／新历史下首步物理目标相对中立姿态仍分别最大跳变 1.3256／1.2894 rad。因此错误历史是回放首步大动作的诱因，但旧 actor 本身已经不连续，不能仅改输入后声称旧检查点平滑。
- **参考**：Gymnasium 1.1 向量环境 autoreset 模式与 `final_obs`、Isaac Lab `ManagerBasedRLEnv`／`ActionManager.reset(env_ids)`／位置动作 reset 规则、本项目 `mjlab_env/{task_bundle,task_logic,task}.py`、`rl_torch/{train,checkpoint}.py`、旧 task／checkpoint、论文 V32-PAPER-U04/08 和日志技能。论文未披露历史初值，本次是工程一致性修复。
- **修改**：新 task 用 scene `neutral_action` 初始化上一动作，标记 `reset_action_semantics=scene_neutral_physical_v1`，校验51维有限值及逐项匹配；CPU reset 参考的最新历史帧同步修正。训练配置新增 `reset_contract`。旧 task 无字段时保留 `legacy_action_lower_bound_v1`，借 task 散列阻止旧 checkpoint 跨合同热启动。h12 49／47动作包装脚本改走源码缓存。更新 README、preview、task/SAC验收、缓存说明、论文决定登记及[完整说明](reset_recovery_20260924.md)。未加入插值、EMA或限速。
- **验证**：任务规则7项、TZ/Yaw结构锁止2项通过；新缓存 task 的 reset action 对 scene neutral 最大误差0，CPU参考误差1.31e-8。16环境验收的固定状态、局部reset、终止帧所有权均通过；既有`press_physical_rollout`仍因观测误差0.13092>0.1失败，其余通过。2环境×4步、47动作GPU smoke完成。诊断产物在 `training_mjlab/outputs/reset_action_diagnosis_20260924/`。
- **训练**：旧无补偿续训自然完成5,000,192采样，最终整曲eval F1=0.864497，未中止或改写。修复合同从头启动 `training_mjlab/outputs/sac-twinkle-reset-neutral-no-yaw47-20260924-154950`，trainer PID447204，256环境、47动作、1180观测、5,000,192目标采样；配置已确认新reset合同。旧权重未迁移。
- **TODO**：V32-RESET-01 待新训练保存检查点后测首步逐关节目标跳变并导出视频；若仍过大，再由用户单独选择动作插值／变化率限制／平滑奖励并作为新控制合同重训。V32-SAC-03、V32-04/06及接触CPU/GPU差异继续保留。

早期检查补充：新run的`step-1000.pt`已保存；中立reset观测下确定性首步目标相对中立值最大为手指0.62994 rad、腕平移0.14644 m、腕转动0.00760 rad、踏板0.55615。它仍是刚过学习起点的早期策略，不能代替最终视频验收；尤其腕平移仍大，V32-RESET-01继续保留。


## 2026-09-24：源码＋完整配置回放、准备时间和SAC调整（指法修改已撤回）

- **目的与最终范围**：按用户要求新训练增加1秒曲首空白，显式开启并记录源码手部重力补偿，Actor256×3、初始温度1、gamma0.8；用脚本＋源码＋config.json回放并整理当前聊天。用户后续明确“不需要改这处指法了”，本轮所有指法覆盖与识别绕过均已撤回，保留原曲指法。
- **遇到的问题**：旧持久化scene/task缓存不符合用户希望；原MidiFile.has_fingering对单一part判无指法，曾为覆盖指法调整后按用户决定撤回；新重建流程提前导入MuJoCo导致无头渲染EGL配置过晚，已调整初始化顺序；本机父会话原始聊天仅存到ordinal408而当前分支引用1630，不能恢复缺失原文；精确名为eval_daxian_unlock的目录未找到。
- **参考文件**：DaxianPianist/robopianist-rl/{eval_song.py,train.py}的args.pkl→get_env重建方式、本仓库robopianist-rl/run.sh；本项目rl/environment.py、suite/tasks/{base,piano_with_tsensei_hands}.py、music/midi_file.py、mjlab_env/{bundle,task_bundle,paths,without_arms}.py、rl_torch/{train,sac,checkpoint,evaluate}.py、preview/scripts/{preview_mjlab,render_mjlab_checkpoint,reframe_mjlab_recording}.py；论文笔记V32-PAPER-10／pp.8–10；仅沿当前本地聊天的父会话链读取可见助手记录，不导出推理／工具结果。
- **具体修改**：环境配置透传initial_buffer_time；scene/task导出可跳过离线参考轨迹用于临时构建；新增source_runtime构建／版本／源码散列校验和外部乐谱嵌入；train_from_source取消持久化缓存，新run.sh显式列参数，eval.sh和eval_from_source校验完整JSON并重建回放；训练SAC采用CLI有效配置，新checkpoint/config保存source_runtime及原reset合同，no-arms中间产物也留在临时目录；渲染报告记录重建配方并修正GPU后端选择时机；新增聊天导出脚本与文档。重力补偿原本True，确认实际44个body为1；没有修改钢琴重力。
- **最终指法状态**：撤回fingering_override的所有源码／CLI／任务导出参数及note.part重写，恢复原识别分支。原始MIDI未改；测试验证缓冲20帧后fingers数组逐元素等于原曲。早期全7短测目录标记SUPERSEDED，不用于当前回放。
- **验证**：SAC单元5项、源码重建3项、CPU环境pytest19项通过；run.sh/eval.sh语法通过。最终2环境×210步GPU短训产生420采样／207更新／2个完整episode，0截断；退出清理临时目录后数值eval和NVIDIA EGL视频回放均完成178步。实际config为47动作／1180观测、1秒缓冲、温度1／Actor256×3／gamma0.8。数字eval F1=0.101063，独立render F1=0.102686；不宣称GPU逐位重现或已收敛。详见[source_config_training_20260924.md](source_config_training_20260924.md)。本轮未启动新5M长训。
- **文档与TODO**：同步README、preview说明、旧缓存与SAC验收的替代标记、论文决定及AGENTS。聊天记录见[chat_answers_20260924.md](chat_answers_20260924.md)；V32-CHAT-01缺失历史需原始导出补齐。保留V32-RESET-01策略大动作验证、V32-SAC-03与V32-04/06及既有接触差异；配置只保存源码散列，长期重现需另保存本项目对应源码和依赖版本。


## 2026-09-24：训练入口进度显示

- **问题与范围**：用户要求给最新训练脚本增加进度条。检查发现run.sh已传--progress，rl_torch/train.py已有tqdm采样进度条，但源码构建阶段缺少提示，入口--help也未列出进度开关。本次仅完善启动脚本的显示和开关转发。
- **参考文件**：run.sh、scripts/train_from_source.py、tsensei_v3_2_pianist/rl_torch/train.py的parser/run进度实现、docs/source_config_training_20260924.md、本机pianist-v2-change-log技能。既有论文/SAC决定保持，本次不涉及训练算法或环境参数。
- **修改位置及原因**：scripts/train_from_source.py显式提供默认开启的--progress/--no-progress，将实际选择转发给训练器；读取配置、创建模型/乐谱及GPU初始化前即时输出阶段提示；进度选项排除于环境配方。run.sh补充进度说明，内部使用文档说明采样数、速度、ETA、更新数、首次评价后F1及stderr日志用法。包内训练源码未改，避免仅因显示改动改变源码重建散列。
- **验证**：4种默认/开启/关闭/重复覆盖的入口检查通过，确认环境配方不含progress；bash -n run.sh与入口--help通过。通过run.sh执行2环境×4向量步GPU短测，实际8个样本、3次SAC更新；捕获到构建1/2、2/2提示及训练进度100%和8/8，config.json记录progress=true。临时验证输出：/tmp/v32-progress-check-ad1ar2tr/console.log及run/summary.json。未启动长期训练；该短测只验证显示链路，不用于学习效果判断。
- **TODO**：无新增TODO。此前V32-RESET-01、V32-SAC-03和模型/接触标定事项保持原状态。

## 2026-09-24：后续默认512环境并同比调整5M训练参数

- **目的与影响范围**：用户要求把后续训练环境数从256提高到512，并确认其他相关参数一并更改。本次只改源码训练入口的默认计划及说明；没有停止、重启或改写已经运行的`sac-twinkle-source-5m-20260924`，该进程及其`config.json`继续是256环境旧合同。
- **遇到的问题**：`num_envs`同时参与总transition数、replay容量、稳态更新比和各向量步日程；只改成512会把总样本／replay翻倍，并将`gradient_steps/num_envs`从0.125降至0.0625。按用户选择同步换算后，总样本和主要学习预算保持不变。因更新条件包含`step >= learning_starts`，新计划为593,088次更新，旧计划为593,056次，相差32次；选择保持精确的256,000条学习起点。
- **参考来源**：`run.sh`；`rl_torch/train.py::effective_config`及训练循环；`rl_torch/replay.py::Replay`；当前运行`training_mjlab/outputs/sac-twinkle-source-5m-20260924/config.json`；论文对照`docs/references/biomimetics_2026_610_notes.md`的§2.4、§5及V32-PAPER-10。论文使用256环境，本次512是用户指定的工程适配。
- **具体修改**：`run.sh`默认改为512环境×9766向量步＝5,000,192条、replay-vector-slots=512（262,144条）、batch4096、gradient_steps=64（0.125 update/transition、复用比512）；warmup／learning-starts／eval／save／log改150／500／500／500／5，使对应样本数保持76,800／256,000／256,000／256,000／2,560。同步README、`docs/source_config_training_20260924.md`、根／内部AGENTS和论文决定登记。
- **实际验证**：`bash -n run.sh eval.sh`通过。`bash run.sh --inspect-only`退出0并从源码重建临时环境；有效配置为num_envs=512、requested_transitions=5,000,192、replay_capacity=262,144、batch=4096、gradient_steps=64、steady_updates_per_transition=0.125、steady_sample_ratio=512，预热／学习起点150／500。进程检查确认现有训练命令和config仍为256环境、19532步、32更新，未被热切换。本次未分配512物理环境，也未测训练吞吐、显存峰值、接触容量或收敛。
- **TODO V32-512-01**：在没有其他训练争用GPU时执行512环境短程物理／学习profile和容量检查，记录吞吐、完整GPU显存、overflow及与256的公平对照；长期效果只能由新运行的评价结果确认。V32-SAC-03、V32-04/06等既有事项继续保留。

## 2026-09-24：当前源码训练最佳检查点有声回放

- **目的与范围**：用户要求查看当前训练效果并回放最好的视频。只读选择仍在运行的`sac-twinkle-source-5m-20260924`已保存检查点，独立重建环境并渲染；训练进程PID616684未停止、未改参，模型／奖励／训练源码均未修改。
- **遇到的问题**：读取到step-10000共10次整曲评价后，`step-5000.pt`仍以日志F1=0.3601208最高。独立复播F1=0.3347617，与日志相差约0.02536且`logged_metrics_match=false`，保留V32-SAC-03。检查`export_best_mjlab.py`发现其完整导出会为源码检查点重复解析并显式传scene/task，而源码重建接口拒绝显式路径；本次采用`--select-only`后直接调用renderer，未顺手修改工具。
- **参考来源**：当前run的`config.json`／`metrics.jsonl`／`step-5000.pt`，`preview/scripts/{export_best_mjlab,render_mjlab_checkpoint}.py`、`mjlab_env/{paths,source_runtime}.py`、论文笔记§2.10 pp.12–13及V32-PAPER-11、日志技能。论文25 ms指标与当前20 Hz逐帧平均F1口径不同，不跨项目横比。
- **产物与结论**：新增`preview/outputs/replays/twinkle_source_current_best_step5000_20260924/`，含有声`replay.mp4`、selection／report、轨迹、实际触键MIDI/WAV、200 Hz物理子步状态和截图；更新README及源码训练说明。日志precision／recall为0.263027／0.885300；独立复播为0.246119／0.818820，画面显示多枚非目标键同时按下，故当前高召回但误按明显，尚未学好。
- **验证**：源码＋config临时重建及CUDA确定性回放完整结束178步；NVIDIA硬件EGL渲染。`ffprobe`确认H.264 1280×720／20 FPS＋AAC、199帧／9.95秒；音频来自实际触键，共134次note-on，视频SHA-256=`5cd590db070a65666c1cbb94ba21f72e0841bb4cdabcfc158e7bd2b395945e45`。人工查看3秒截图确认全琴和双手入镜，并能看到误按绿色琴键。首次最终校验从仓库根误用了子项目内相对路径，三项读取报文件不存在；随即改用绝对路径复核selection、report、ffprobe及SHA-256均通过，产物未丢失。训练快照已到step-10110／2,588,160条，仍在运行，当前最佳不是最终最佳。
- **TODO**：V32-EXPORT-02让通用选优导出器兼容源码checkpoint；V32-SAC-03继续定位评价／复播差异；V32-AUDIO-01完成力度标定。训练完成后需重新选优，不能把本阶段快照当最终结果。


## 2026-09-24：新增可选旧Daxian错键下压惩罚

- **目的／范围**：用户要求按旧项目错键惩罚规则和公式新增奖励，由用户决定启用。当前MJLab/Torch新增false_press_penalty，四份JSON模板及代码模板均默认关闭，外部权重1；run.sh默认命令和正在运行的训练未改，未新增正式训练。
- **问题与差异**：旧错键项不是−0.15×激活错键数，而是对非目标键下压比例x进行阈值及深度计罚：x≤0.001为0，否则−(0.01+0.14x)，先clip[0,1]后逐键累加。现有默认按键奖励仅在无激活错键时加0.5，论文pp.10–11式13–14按数量指数衰减；已向用户说明本轮沿用其指定旧公式并独立可选。旧JSON严格20项schema需兼容扩展；GPU临时测试首轮误读不存在policy_high/low属性，改为实际action_high/low后完成。
- **参考文件**：/home/houjue/DaxianPianist/robopianist/robopianist/suite/tasks/piano_with_shadow_hands.py中常量、_false_press_cost、_compute_false_press_penalty、_key_goal；旧models/piano/piano.py的normalized_state、robopianist-rl/run_daxian_unlock_pip_dip.sh；本项目mjlab_env/{reward_config,rewards,task,task_logic,scene}.py、rl_torch/{train,checkpoint}.py、tests/test_paper_rewards.py、论文notes §4.1／V32-PAPER-05、日志技能。只参考旧项目，独立运行无外部导入依赖。
- **修改**：rewards.py新增批量Torch公式及RewardManager分支；reward_config.py新增默认关闭配置，并对历史JSON缺失新项补关闭（不改变调用者原字典；其他缺项仍拒绝）；train.py在CLI启用／禁用前规范化旧JSON，复用现有--enable-reward／--disable-reward和检查点恢复。configs/rewards四份模板加入关闭项；更新奖励总表为21项、README、论文决定、AGENTS及[完整使用说明](false_press_penalty_20260924.md)。原CPU/Composer奖励组合、指法表、当前帧时序、SAC和512环境参数未改。
- **验证**：training_mjlab/.venv/bin/python -m unittest tests.test_false_press_penalty tests.test_paper_rewards tests.test_mjlab_task tests.test_torch_sac：28项通过（新增5项）。覆盖CPU／CUDA边界及超限clip、目标键和踏板排除、轻压未激活仍计罚、多键加和、休止帧／当前目标帧、局部reset、奖励进入上次reward／return、权重与dt、旧JSON／CLI及checkpoint开关恢复与禁止静默改合同。临时GPU物理验收2环境×30步、47动作／1180观测，60样本非零惩罚；逐步与独立NumPy公式最大误差1.4812e−7，奖励和观测有限，总奖励与分项一致；报告training_mjlab/outputs/false_press_penalty_validation_20260924.json。
- **限制／TODO**：正式启用由用户决定，未以测试开启替代用户选择；未验证新奖励对长期F1的影响，V32-SAC-03、V32-04/06等既有TODO保留。旧奖励配置可读并不绕过源码checkpoint既有版本散列校验；当前持续训练仍执行原先载入的源码，不热更新奖励。


## 2026-09-24：run.sh文件头补齐全部接受参数

- **目的**：用户要求将脚本接受的所有参数以注释写入文件头。仅修改run.sh注释及说明，不改实际命令和训练配置。
- **问题／参考**：run.sh同时经过scripts/train_from_source.py与rl_torch/train.py两个解析器；底层默认值与脚本显式传参不同，store_true参数不能用末尾false取消，源码入口还拒绝scene/task/source-config/cache-dir。核对两个入口、mjlab_env/source_runtime.py的姿态时序校验、reward_config.py的21项及约束、rl_torch/sac.py校验、论文笔记§4.1及既有用户决定；没有改变论文参数或任何奖励设计。
- **修改**：文件头按姿态／乐谱／时序、模型动作、奖励、SAC预算、设备加速、评价保存、恢复、帮助检查列出选项、单位、run.sh有效默认值、关键底层默认差异和限制；列出全部21个奖励名及必填映射，说明不支持的路径覆盖和单向布尔选项。README新增入口链接。
- **验证**：bash -n通过；用AST枚举两个解析器的add_argument并补齐BooleanOptionalAction反向开关，共61种选项拼写均在注释中出现（含明确标为内部拒绝的3个底层路径选项）；执行体与修改前逐字节一致；bash run.sh --help成功退出，未构建环境、未启动训练。无新增测试文件。
- **TODO**：本项无未完成工作；以后新增CLI需同步参数表。既有V32-SAC-03／V32-512-01等验收事项不受本次注释修改影响。

## 2026-09-24：四指MCP改为逐关节直接映射[0,1.2]

- **目的与范围**：用户确认左右食指／中指／无名指／小指共8个MCP改为策略`[0,2]`直接映射物理目标`[0,1.2] rad`，不再通过`mixed_v1`分组shift表达。拇指MCP保持`[0,1.57] rad`，MuJoCo关节／执行器机械硬限位不改；当前正在运行的旧profile训练未停止、未热切换。
- **问题与处理**：原Actor、随机预热、replay和GPU scene都从单一shift推导上下限，不能独立表达“策略范围不变、某关节物理目标上限缩小”。新增逐通道四边界后，首次合并SAC回归暴露通用3维legacy测试被51／49／47布局校验误限制，已恢复legacy任意正动作维支持，仅direct合同限制TsenseI布局。2环境短训本体成功，但首个结果提取脚本误读summary中的`updates`（实际为`updates_total`）而退出1；直接读取已生成配置／summary复核成功，训练产物未损坏。最初用系统Python跑测试缺Torch，随后统一切换项目`training_mjlab/.venv`。
- **参考**：`configs/actions/direct_joint_v1.json`、`mjlab_env/{action_mapping,scene,bundle}.py`、`rl_torch/{sac,replay,train,checkpoint}.py`、预览／诊断转换脚本、旧`action_mapping_mixed_v1.md`、当前scene manifest、论文笔记§2.6 p.8与V32-PAPER-04、本机日志技能。论文公开统一tanh仿射但未公开TsenseI MCP 1.2 rad，本次记录为用户指定工程适配。
- **修改**：新增`direct_joint_v1`的51通道显式JSON；Actor、随机预热、replay校验、GPU物理转换及逆转换统一读取逐通道范围，通用仿射log-Jacobian一并实现。训练`config.json`保存解析后的活动通道合同；`run.sh`后续默认新profile。更新GPU预览、接触／诊断脚本、README、源码训练说明、论文决定、两份AGENTS及[47维完整范围表](action_mapping_direct_joint_v1_20260924.md)。保留legacy/mixed_v1及其旧checkpoint恢复；profile不匹配继续拒绝。
- **实际验证**：`tests.test_action_mapping tests.test_torch_sac`共11项通过，含CUDA端点`0→0`、`1→0.6`、`2→1.2`、8个MCP名单、47维删TZ/Yaw、拇指MCP1.57和Pitch±0.2不变、逆映射、旧profile及checkpoint／replay回归。`bash -n run.sh eval.sh`通过；`bash run.sh --inspect-only --no-progress`确认512环境、47动作结构开关和`action_profile=direct_joint_v1`。临时GPU短训`/tmp/v32-direct-joint-smoke-p8PU44/run`完成2环境×4步＝8条transition、3次更新，config记录8个MCP均为`[0,2]→[0,1.2]`。
- **TODO**：本轮只验证映射和短训闭环，未启动新的5M训练，不能声称收敛、减少误按或改善演奏。V32-512-01的512环境吞吐／显存／overflow与长期效果仍待无训练争用时验收；V32-SAC-03、V32-04／06等既有项保留。


## 2026-09-24：训练输出目录追加北京时间时间戳

- **目的／问题**：用户用既有 `--output-dir training_mjlab/outputs/sac-twinkle-source-5m-20260924` 重启训练，在模型构建后才收到 `FileExistsError`，又由源码入口包装为 `CalledProcessError`。该目录包含已完成训练的检查点，必须保留。
- **参考文件**：`scripts/train_from_source.py` 的参数转发、`run.sh` 的输出目录说明、`rl_torch/train.py` 的 `output.mkdir(exist_ok=False)` 与训练配置写入、`docs/source_config_training_20260924.md`；本机 `pianist-v2-change-log` 技能。
- **修改**：源码入口在模型构建前解析末尾有效的 `--output-dir`，使用 `zoneinfo.ZoneInfo("Asia/Shanghai")` 生成 `YYYYMMDD-HHMMSS-微秒` 后缀并打印最终路径；未指定目录时用 `training_mjlab/outputs/sac` 作基础名称。目标目录若碰巧已存在，再加 `-01` 等序号。把最终路径传给 Torch 训练器，故 `config.json` 的 `cli.output_dir` 与实际目录一致。更新 `run.sh` 注释、README 与源码训练说明中的回放路径示例。旧目录不重命名、不覆盖。
- **验证**：临时目录测试覆盖 `--output-dir PATH`／`--output-dir=PATH`、北京时间格式、已存在的基础目录、同微秒碰撞和默认名称；`bash -n run.sh`、`py_compile` 通过。端到端 `bash run.sh` 2 环境×4 向量步完成 8 条样本／3 次更新；日志显示北京时间实际目录，配置记录同一目录，已存在的旧目录及标记文件未改。短测位于 `/tmp/v32-beijing-output-check-0hx8bbke/`。未以短测推断学习效果，未启动新的 5M 训练。
- **TODO**：无新增；V32-512-01、V32-SAC-03 及原有模型／接触验收仍按之前记录执行。


## 2026-09-25 — 最新小星星5M视频导出与训练分析

- 目的：导出最新final与本轮最高日志F1的step-7000，分析训练及错误分布；不改训练参数。
- 问题：自动best导出入口对source_config检查点的显式scene/task路径仍有兼容问题（V32-EXPORT-02），本次使用eval.sh直接重建成功。系统／CPU Python缺matplotlib，改用已有GPU venv；一次分析命令工作目录错误后改为仓库根相对路径运行成功，无新增依赖。
- 参考：AGENTS.md、pianist-v2-change-log技能、论文事实与决策登记；eval.sh、scripts/eval_from_source.py、preview/scripts/export_best_mjlab.py；两轮config/metrics/summary及新导出的report/rollout。
- 修改：独立项目preview/outputs/replays/twinkle_latest_final_20260925与twinkle_best_step7000_20260925保存有声视频和回放数据，最终目录增加training_curves.png、contact_sheet.jpg、analysis.json；新增docs/training_twinkle_latest_20260925.md并更新README。
- 验证：两个eval.sh导出退出0，源码／依赖检查通过，关键按键指标与日志一致。ffprobe确认两片均1280×720、20fps、9.95秒、H264/AAC；实际硬件渲染及物理触键音频。numpy重算TP/FP/FN、完整正确帧、准备期误触和碰撞帧；抽查六个时间点画面。最终F1=0.904307、最佳=0.919101，均149/178帧完全正确。return有不到0.001浮点差异。曲线仅用日志评估，未声称多种子收敛或动作平滑验收。
- TODO：V32-SAC-03历史回放差异、V32-AUDIO-01力度标定、V32-EXPORT-02自动导出兼容仍保留；起始误触、漏按及同手碰撞见分析报告。未启用false_press_penalty，未启动新训练。

补充验证：final.pt导出report中的logged_evaluation／logged_metrics_match为null，自动日志关联未识别final文件名；已直接按metrics.jsonl的step-9766逐项核对precision、recall、F1及micro-F1，完全一致。step-7000自动关联为true。不能将final报告的null当成自动校验通过；此元数据问题归入V32-EXPORT-02。


## 2026-09-25 — 取消新训练曲首1秒缓冲

- 用户要求：取消当前训练的1秒准备缓冲；后续run.sh与源码训练入口均默认0秒。
- 问题与依据：run.sh显式传1秒，scripts/train_from_source.py也默认1秒，需要同步，避免直接调用Python时仍增加缓冲。参考这两个入口、docs/source_config_training_20260924.md、docs/references/biomimetics_2026_610_notes.md以及项目AGENTS和修改日志技能。
- 修改：run.sh参数及头部说明改为0；train_from_source.py默认值改为0；内部训练说明和论文决策登记记录2026-09-25用户决定，覆盖旧1秒设置。未修改当前文件中的其他训练选项（含已存在的false_press_penalty开关），未改旧运行配置、检查点或正在运行的进程。
- 验证：bash -n通过；Python AST语法检查与参数断言确认两处均为0。未启动训练，未执行完整模型构建／收敛验收。
- 未解决事项：本次无新增TODO；已有V32-RESET-01动作平滑性问题仍需实际训练验收，取消缓冲不代表解决突然运动。


## 2026-09-25 — 训练全链路与爆冲／抖动诊断

- **目的**：用户要求梳理训练并解释与旧eval_daxian_unlock／Shadow平和动作的差异；本次不修改训练设计或启动新训练。
- **问题及定位**：旧eval_daxian_unlock实际为Daxian，另有Shadow原生结果，不能混淆。旧项目没有完整源码散列，不声称重建历史物理参数逐位一致。最新reset中立动作合同与终止transition保存正确，12项相关逻辑测试通过。真正抓到大幅目标阶跃：首步左TX目标+152.859mm、50ms实际+57.747mm，5.20s同回合目标跳169.240mm；CPU/GPU同模型同ctrl均腕峰速约1.296904m/s、手指21.0869rad/s。中立固定稳定；首目标固定后稳定；曲中目标保持仍有残余接触调整，不能声称所有姿态绝对不抖。一次查旧模型路径漏了一层robopianist后已更正；未更改旧项目。
- **参考**：根及独立AGENTS、pianist-v2-change-log技能、论文对照登记；当前run/config/rollout及rl_torch/{train,sac,replay}.py、mjlab_env/{scene,task,task_logic,action_mapping,rewards,task_bundle}.py、模型常量、render_mjlab_checkpoint.py；docs/reset_recovery_20260924.md及contact_stability_shadow_20260924.md。旧项目实际args/metrics、run_daxian_unlock_pip_dip.sh、Daxian/Shadow手源码、常量、XML、policy_joint_map及technical_report只读参考。
- **修改**：新增独立docs/training_motion_audit_20260925.md、README入口；preview/outputs/diagnostics/motion_20260925保存专用诊断脚本、模型参数、逐子步轨迹、CPU/GPU目标与速度图和报告。训练源码、奖励、动作与模型参数均未改。
- **实际验证**：GPU venv执行tests.test_mjlab_task与tests.test_torch_sac共12项通过；以当前源码重建47动作模型，178×10子步GPU与CPU保存控制开环重播，中立／首目标各保持2秒、曲中最大腕跳变后保持2秒。单手质量实测0.171363kg，腕TX/TY Kp300、阻尼30、力限50N；单自由度近似阻尼比2.092，并非简单欠阻尼。GPU重播与原视频全曲最大腕偏差0.160mm、手指0.01014rad，CPU分别0.249mm/0.01337rad；没有声称全轨迹逐位相等或GPU完全无数值差异。已查看CPU/GPU轨迹图，文档本地链接检查通过。
- **TODO**：保留V32-04驱动/惯量标定、V32-06碰撞几何和V32-SAC-03轨迹差异；新增V32-MOTION-01动态平滑性指标与控制约束方案待定，见报告。论文插值及式16–18仅列对照，不自行启用或重训。


### 同一任务补充：原版Shadow与“卡住后弹回”的接触因果验证

- 用户进一步描述弹回／卡住后，增加逐子步接触力检查，**纠正初期过宽归因**：0.705s手指21.09rad/s峰值来自前5ms跨手拇指撞击，不是仅由位置命令变化造成。0.705s终点几何已分离，不能用终点无接触推断积分过程中无碰撞。
- `preview/outputs/diagnostics/contact_20260925/`新增4组CPU全曲诊断、8事件×4分支同状态单5ms验证、力／接触事件、曲线、只读诊断脚本及接触着色截图。基线单步分支相对原CPU轨迹误差均0；0.700s仅去跨手接触，拇指速度由−21.0869变+6.27349rad/s；1.645s小指PIP侧撞黑键73.52N并受两白键挤压，名义SW限位超出4.92°；2.465s食指MCP真实瞬时受阻。正式训练实体接触与奖励均未改。
- `motion_20260925/contact_geometry_limits_20260925.py/json`保存GPU轨迹静态碰撞／软限位扫描，明确不能将静态查询力误认原GPU力；独立接触追查使用CPU积分实际力。持续0.1s的卡死阈值未命中，但单步分支证明短暂受阻真实存在。
- `shadow_comparison_20260925/shadow_tx_step.py/json`记录native Shadow独立手阶跃，实际手指为capsule/sphere/cylinder碰撞、PIP/DIP腱长度控制、单手3.794kg。首50ms位移12.327mm，只作机械响应旁证（无钢琴、gravity0、版本不同）。旧suite缺pyaudio，改实例化ShadowHand，无安装；早期按传动ID误选腱执行器的probe作废，最终按名字与传动类型核验。
- 总报告、运动诊断README和项目README已同步修正，区分两类机理，不把CPU/GPU复现误说成排除物理接触。补充参考MuJoCo官方mesh凸包及soft constraint文档：https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh 与 https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters。截图在EGL重建模型后渲染，已人工查看；本地文档链接和脚本语法检查通过。
- 后续V32-06须核对已定位接触对的实际凸包包络，接触存在不自动等于几何错误；V32-MOTION-01需同时验收命令阶跃和碰撞反弹。没有将关闭接触作为正式修复，也未擅改SAC或重训。


## 2026-09-25：同步本仓库Shadow并开启新5M训练

- **目的/授权**：用户要求把V3.2训练参数、碰撞方式同步完整Shadow，并授权验证后直接开训。保留MJLab/Torch、512环境、V3尺寸/关节/坐标；采用新具名profile，旧源码备份供历史回放。
- **遇到的问题**：本仓库Shadow Python源码完整但third_party XML缺失，实体参数只读参考本机DaxianPianist原生Shadow资产；原脚本注释称false_press关闭而末尾实际启用，已按本次Shadow原五项统一关闭。第一稿碰撞包围体过大，旧ctrl跨手力831N，否决并改为惯性拟合/末端面宽；下游挂3kg刚体造成Pitch错误耦合，否决并改为TX/TY armature。一次碰撞测试误用无dm_control的Torch环境，改正确Composer环境通过。新回放仍因GPU物理数值非确定性产生轨迹差异，未错误归咎于reset或权重。
- **参考**：根/项目AGENTS、pianist-v2-change-log技能、论文事实及决策登记；本仓库robopianist-rl/{run.sh,train.py,sac.py}、robopianist/robopianist/{models/hands/shadow_hand.py,suite/tasks/piano_with_shadow_hands.py}；本机DaxianPianist原生Shadow XML及mujoco_utils临界阻尼实现；原运动/接触诊断、当前URDF/OBJ/STL、rl_torch与mjlab_env。外部仅参考，无新运行时导入依赖。
- **修改位置/原因**：run.sh完整参数头及默认Shadow配置；train_from_source.py/source_runtime.py保存task/drive/collision profile、覆盖环境变量、严格源码重建。rl_torch/train.py增加三LR和按transition预算、边界随机行/更新数、评估motion指标。task_bundle/task/environment、forearm_collision/fingering_targets及action_mapping新增Shadow任务/动作合同；统一策略[-1,1]但保留8个MCP目标上限1.2。手模型新增collision_proxies/drive_settings包内JSON和生成/验证脚本，手指用V3尺寸primitive，掌mesh保留；TX/TY等效惯量3kg、临界阻尼，Pitch8/.5、手指摩擦.01。pyproject补齐JSON包数据。README、AGENTS、论文决策、本轮总说明和子说明同步更新。
- **实际验证**：最终33项相关测试通过（25源码/任务/动作/SAC；3碰撞；5驱动），bash语法/CLI注释覆盖通过；删除临时场景后同配置重建散列一致。三类20s固定接触收敛，TX150mm阶跃1.275→0.532m/s。800个CPU/GPU子步键状态一致，强跨手接触仍有暂态1.50°/0.437mm差异。512×40短训20,480样本、15,481更新完成，无非有限数，Torch峰9.72GB不含全部Warp。源码checkpoint回放成功，但完整轨迹非逐bit相同：同初obs/首action一致，物理一步后已有1.19e-7差异，后续策略反馈放大。详见docs/shadow_sync_20260925.md及preview/outputs/diagnostics/shadow_sync_20260925/。
- **正式启动**：2026-09-25北京时间17:11:19，从头运行`bash run.sh --output-dir training_mjlab/outputs/sac-twinkle-shadow-sync-5m`；输出`training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333`，512×9766=5,000,192样本、计划4,995,193更新。launcher PID2620645，进程组同号，启动记录training_mjlab/shadow-sync-5m-launch-20260925-171119.json；已核验进入更新及独立评估，持续运行。每样本更新数由.125升1，batch4096降256，耗时不能沿用旧5M约46分钟。新旧源码分别归档training_mjlab/source_versions/，未持久化冻结scene/task。
- **未解决TODO**：V32-04硬件动力学标定、V32-06掌部形状与动态接触、V32-SAC-03严格CPU/GPU/重复性、V32-MOTION-01新策略长期运动验收。压力测试仍有59.98N触键峰；不能宣称所有未来策略无爆冲，训练完成后须看新视频与子步运动。512并行采集/更新及早期乐谱覆盖与原单环境不等价，文档明确记录。


## 2026-09-25：导出当前Shadow同步训练的整曲视频

- **任务/范围**：用户要求导出正在运行的V3.2 MJLab训练视频。读取当时最新完整检查点`training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333/step-1500.pt`，不改正在运行的训练进程、源码或配置。
- **问题**：视频回放F1=0.682278，与训练日志该检查点整曲评估F1=0.681434接近但非逐位一致，`logged_metrics_match=false`。GPU物理接触的数值非确定性此前已记录为V32-SAC-03；不能把两次F1当同一条轨迹。无模型缺失或渲染错误。
- **参考**：该运行目录`config.json`和`metrics.jsonl`、`eval.sh`、`scripts/eval_from_source.py`、`preview/scripts/render_mjlab_checkpoint.py`及`tsensei_v3_2_pianist/preview_audio.py`；读取项目变更日志技能。没有外部来源。
- **修改**：通过现有回放脚本从源码+config+检查点重建，新增`preview/outputs/replays/shadow_sync_step1500_20260925/`中的`replay.mp4`、`preview.png`、`rollout.npz`、音频/MIDI事件及`report.json`；README加入产物入口，未改训练源码。音轨按每个物理子步的真实琴键状态和踏板合成。
- **验证**：`bash eval.sh --checkpoint .../step-1500.pt --mode render --output preview/outputs/replays/shadow_sync_step1500_20260925`退出0；`ffprobe`确认H.264/AAC、1280×720、8.95秒；报告158控制帧、32个真实note-on、F1=0.682278；人工查看`preview.png`，渲染完整。训练进程仍在运行。
- **TODO**：V32-SAC-03保留逐位重复性问题；本视频为第1500步阶段性策略，完整5M训练后的平稳性仍待验收。

### 同一视频导出任务补记：第2000步

- 用户再次要求“导出现在的视频”。新最新完整检查点为`training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333/step-2000.pt`；参考原运行`config.json`/`metrics.jsonl`、`eval.sh`及`preview/scripts/render_mjlab_checkpoint.py`。无新错误，未改训练源码或进程。
- 新增`preview/outputs/replays/shadow_sync_step2000_20260925/`：整曲H.264/AAC视频、真实按键音频、PNG、轨迹、MIDI及报告；此前第1500步产物保留。内部README入口更新到最新视频。
- 命令`bash eval.sh --checkpoint .../step-2000.pt --mode render --output preview/outputs/replays/shadow_sync_step2000_20260925`退出0。`ffprobe`确认1280×720、8.95秒；报告179视频帧、30次真实note-on、回放F1=0.7685049，与该检查点评估日志F1相同；人工检查预览图，训练进程仍在运行。
- TODO：V32-SAC-03逐位重复性仍保留；该视频是阶段检查点，不代表5M最终效果。


## 2026-09-25：当前视频一键导出与短训权重清理

- **目标/范围**：用户要求立刻导出当前MJLab训练视频、提供终端一键命令和位置，并清理一些无用训练结果。当前正在训练的512环境进程、正式训练检查点、历史正式模型、源码版本、视频均保留。
- **遇到的问题**：视频导出期间正式训练从第2500步保存到第3000步，因此最终一键命令自动选择更新且已写完的`step-3000.pt`。历史`outputs/`约8.9GiB，包含正式训练、验证短训和场景；无法仅凭目录大小断言正式结果无用。已有两个短训权重明确作废：第一版Shadow下游3kg错误耦合、2026-09-24已撤回指法修改前的source_config短测。其他短训的中途step-N权重有最终权重可替代。
- **参考**：正在运行的`training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333/{config.json,metrics.jsonl,step-3000.pt}`，现有`eval.sh`、`scripts/eval_from_source.py`、`preview/scripts/render_mjlab_checkpoint.py`、`README.md`、`docs/shadow_sync_20260925.md`、`docs/source_config_training_20260924.md`及本机pianist-v2-change-log技能。未修改训练算法或接触模型。
- **修改**：新增`preview/scripts/export_latest_training_video.sh`，默认绑定当前Shadow同步训练目录，自动选完整的最新step-N或final.pt、创建时间戳输出并打印绝对路径；README写入一键命令。新视频在`preview/outputs/replays/shadow_sync_step-3000_20260925-185519-618567837/replay.mp4`，其报告、实际琴键音频、PNG、轨迹在同目录。清理清单`docs/training_result_cleanup_20260925.json`精确记录删除21个.pt及其字节数：已完成短训的中间检查点，以及上述两个作废短训的final.pt；各目录config/metrics/summary和其余final.pt保留，共释放446,628,013字节（425.9MiB）。
- **实际验证**：一键脚本实跑选`step-3000.pt`并成功导出；`bash -n`通过，`ffprobe`证实H.264/AAC、1280×720、8.95秒。视频报告179帧、48次真实note-on、F1=0.8582278且等于该检查点评估日志。清理后重新核对manifest 21项状态complete，正式训练仍有6个检查点、进程PID2621063持续运行；有效的shadow-sync-final-smoke和torch_sac_smoke16的final与所有短训元数据仍在。总体outputs由约8.9GiB降至8.5GiB。
- **TODO**：V32-SAC-03 GPU接触逐位复现限制及V32-MOTION-01完整5M运动验收仍在；不把第3000步视频当最终策略。若将来需要更大空间，应逐个审查旧正式训练/冻结场景的引用后再删除，不能自动扫目录名。


## 2026-09-26 — 曲首审计、origin/initial 1秒准备及训练速度分析

### 用户要求与范围

检查视频开头乱弹是否从训练早期存在、解释同步Shadow后变慢；用户进一步确认“当前完整姿态记origin，抬高姿态记initial，从initial用1秒平滑回origin”。按该决定实现固定准备，保留已确认Shadow奖励、SAC每样本一次更新、时序和动作合同。本轮只做短训验证，没有启动新的5M。

### 遇到的问题与证据

1. 旧最终策略reset速度0、琴键全释放，但第一条右TX目标跳到−0.157759m，手指最大目标阶跃1.331078rad；乐谱t=0即有首组目标。旧final首秒包含全曲7个错键帧中的5个，以及200Hz下6次错音起音。早期step100/500整曲目标阶跃和低F1证明早期已有问题；未保存全部探索期轨迹，不能声称逐帧复原。
2. 200Hz音频采到0.270–0.290s和0.805–0.845s两次短错音，而20Hz控制末端奖励可能遗漏；踏板早期开启延长了听感。只加空白乐谱仍允许策略动作，无法保证准备安静。
3. 当前TZ/Yaw已物理删除；+2cm不能写进不存在的TZ或偷偷扩大[-0.025,0]范围。采用运动学父节点，只在固定准备使用，保留134坐标/46驱动器/47动作。
4. 如果在批量reset里推进200个物理子步，会连带推进其他世界。改为独立CPU世界计算一次准备终态，再局部复制；GPU隔离测试确认其他世界不变。
5. 同步Shadow后实际SAC更新593088→4995193，耗时2757.02→20064.90秒；独立计时学习器约占正常步骤98.4%。GPU仿真和编译优化原已开启。评估489次也有成本，约5%为日志差分估算，不是完整profiler精确分解。
6. 初版回放只检查minimal准备参数；独立复核发现应同时核对完整origin/initial/terminal。已补load_agent及全部使用点，重跑最终短训；中间短训保留并标注，未混作最终验收。
7. 发现既有47维交互预览将47个active控制与51维完整上下限直接裁剪，触发广播错误。改为policy直接act、手动模式以完整neutral填槽再更新active部分。窗口无DISPLAY/Xvfb未实测，离屏流程通过。
8. 新包源码使旧checkpoint严格散列校验拒绝直接重建。修改前完整备份源码；新增显式归档回放工具，保留严格校验。旧默认视频入口一并选用其匹配归档。首次归档render缺音频库/SoundFont，已通过引用本机已有audio_runtime解决，记录音色散列；没有从新代码偷换旧模型。
9. 独立预览最初source_recipe未包含实际启用的准备参数，已补齐并用最终源码重新构建，逐元素核对全部准备数组一致。参数时长增加整数控制帧校验，避免训练成功后视频无法对齐。

### 实际参考

- 根及独立项目AGENTS.md、本机pianist-v2-change-log/SKILL.md；论文笔记§2.6 p.8、§2.7 pp.8–10/表2、§2.9 pp.10–12、§2.10 pp.12–13及已有决策。论文没有本次固定准备方案，不因论文自行改策略插值/奖励。
- 当前/旧run的config.json、metrics.jsonl、summary.json、final/step500及已有step1500/2000/2500/3000回放；详见startup_20260926与speed_20260926原始诊断。
- 包内models/hands/tsensei_hand_constants.py；mjlab_env/{source_runtime,bundle,scene,task,task_bundle}.py；rl_torch/{train,sac,replay,checkpoint,evaluate}.py；源码训练/评估、preview_mjlab、render_mjlab_checkpoint、preview_audio、export_latest_training_video及相关测试。
- 本轮未复制或导入external_never_copy_only_reference内容，没有新增原项目运行时依赖。

### 改动位置及原因

- 新mjlab_env/preparation.py：五次曲线、两个mocap父支撑、独立CPU真实200子步、完整命名origin/initial/terminal/report与201状态trace；末态不强制投影。
- mjlab_env/{bundle,scene,task}.py：准备参数传递、实际终态reset及mocap/warmstart局部恢复；固定准备不进入RL动作/奖励。
- source_runtime.py、scripts/train_from_source.py、rl_torch/train.py、run.sh：--preparation-time/--preparation-lift、finite/整数时长/结构约束、源码recipe一致性，run.sh默认1s/.02m并完整注释；score空白仍0。
- rl_torch/checkpoint.py/evaluate.py及scripts/profile_mjlab_training.py：完整准备合同重放/热启动验证，旧缺项None兼容，新旧初始化不静默混用。
- preview/scripts/{render_mjlab_checkpoint,preview_mjlab,preparation_frames}.py、scripts/preview_preparation.py：真实准备画面、独立轨迹数组、实际200Hz音频1秒偏移；policy rollout仍只记录正式乐谱。
- scripts/eval_from_source_archive.py、export_latest_training_video.sh：显式严格旧源归档回放和历史默认导出入口兼容；tar路径/成员集合/sha256检查，临时源码退出删除。
- configs/poses/{origin,initial}_20260926.json：当时源参数、全手关节目标和实际世界位姿留档；不是运行配置的第二入口。
- README、AGENTS、论文决策、初始化/速度/归档三个说明文档及根/内部日志同步。源归档before_preparation_20260926和preparation_20260926保留历史与当前代码，不保存持久scene/task。

### 实际验证

- preparation核心5项（含GPU局部reset）、preparation_checkpoint 6项（真实小SAC保存/加载与配置约束）、原Torch SAC 5项、TZ结构2项、归档安全/一致性6项通过。
- `MUJOCO_GL=egl training_mjlab/.venv/bin/python -m unittest tests.test_source_runtime tests.test_shadow_training_budget tests.test_shadow_task_profile -q`：14项通过，45.802s；包含删除原临时构建目录后从source/config重建和Shadow参数保持。日志保存在preview/outputs/diagnostics/preparation_20260926/source_regressions.log。
- bash语法及所有本轮入口Python编译通过。
- 当前Shadow profile独立准备视频：200物理步/201状态，激活键0、音频峰值0；手关节终态误差3.88e−9rad、腕滑动2.17e−14m。0.000623rad全模型误差来自琴键静态回落，不误称手部误差。NVIDIA EGL硬件渲染通过。
- 最终短训目录training_mjlab/outputs/preparation-final-acceptance-20260926-004025-659259：2环境×170向量步=340 transition，replay插入340，324次更新，正常结束2回合、截断0；elapsed12.282s为训练器计时，不含所有前置源码构建。短训缩小batch/replay及探索预算、关闭compile仅为流程验收，未改变run.sh长训默认。
- 短训命令：`bash run.sh --num-envs 2 --vector-steps 170 --batch-size 16 --replay-capacity 512 --warmup-transitions 16 --learning-start-transitions 17 --eval-interval 170 --save-interval 170 --log-interval 20 --no-compile-learner --no-progress --output-dir training_mjlab/outputs/preparation-final-acceptance`。
- 正式eval.sh完整回放通过，full准备合同与checkpoint完全相同。policy_handoff_final_acceptance_20260926：20准备帧+1初态+158策略帧+20停留=199帧/9.95秒；音频200准备+1580策略子步，偏移1秒，准备触键0，首音1.06秒。340样本策略F1=.03481，仅验证流程，不是改善演奏质量的证据。
- 旧final通过归档独立evaluate与默认脚本完整有声render：158策略步，F1=.9405063391，与旧final日志相同。新旧源校验均未关闭；默认视频shadow_sync_final_20260926-004021-833751768/replay.mp4。

### 未解决与边界

- V32-START-01：origin与原初态相同，1秒准备不消除旧策略第一条大目标；需新策略长训和接管后首秒验收。动作过渡/限速属于后续控制决定，本轮未暗加。
- V32-START-02：200Hz实际声学事件与20Hz奖励端点的短错音盲区，未修改Shadow奖励采样。
- V32-PERF-01：保留Shadow更新预算的完整学习器图执行/减少同步优化尚未实现或验收，不承诺恢复46分钟。未擅自降update/transition。
- V32-04/06、V32-SAC-03及真实驱动/接触标定继续保留；运动学准备不代表机械臂动力学；GPU接触回放非逐bit保证。MULTICCD对部分capsule/mesh仅单接触的既有警告仍存在；不据短训宣称长期稳定。
- GUI窗口未实测，离屏预览与有声回放已通过。

交付：[曲首分析与准备使用](startup_preparation_20260926.md)、[速度审计](training_speed_audit_20260926.md)、[旧源回放](archive_replay_20260926.md)、[准备视频](../preview/outputs/preparation/lift_2cm_1s_shadow_20260926/preparation.mp4)。


## 2026-09-26 — 准备帧实际参与SAC学习、保持算法预算的CUDA执行加速

### 本轮要求与决定

用户要求“准备帧也进训练过程”，并要求恢复速度但“只加速、不改其他”。本轮按已说明的policy_v1实施：47维策略从抬高initial第0帧控制，支撑1秒回origin高度，前20空谱帧按原奖励进入replay；手指/原腕轴不再强制回写origin。新增1维剩余准备进度使运动支撑可观察，1181维；原观测1180维；物理134坐标/46驱动器/47动作保持。加速单独保留全部SAC/奖励/采样预算和FP32。只执行短训/对照，不新开5M。

### 问题、原因与处理

1. 固定准备段不受策略控制且不入replay，不能使策略学会其动作与误触的关系。改为真实GPU准备阶段；移动支撑原本不在1180观测里，新增一个remaining_fraction，未篡改已移除TZ的零槽。
2. 准备/额外空谱可能被重复添加或回放再次插CPU前缀。源码只在policy_v1导出总空谱=preparation+extra；渲染从GPU initial开始，offset0，真实音频不静音、不拼接另一段固定动作。旧fixed_v1保留原合同。
3. 原总逐帧F1会被安静准备帧抬高。保留原指标，新增music_only_f1/micro_f1及准备错键统计，额外buffer也排除；不改变奖励/评价公式。原0.5秒前瞻和指法提供时序未改。
4. live preview曾调用logic.observation直接取得旧1180维，未包含新phase。修复为保存env.reset/step的完整1181观测。
5. 每次完整SAC更新大量Python调度及GPU/CPU同步。新增完整learner CUDA graph，图外保留动态replay.size的randint；图内原顺序更新/每次finite累计，块末主机检查。构图试跑恢复参数、Adam矩/step、RNG和计数；禁止存储地址替换后继续用旧图。
6. 原Torch fusedAdam chunk65536/512threads对当前中小网络并行度低，三个Adam合计约430us/update。增加可选chunk1024/256threads和block共享相同bias的执行布局，直接调用安装Torch原adam_math，保留float32状态和double中间计算，无TF32/AMP/fast-math。
7. CUDA扩展首次编译缺ninja和extended-lambda编译开关：独立venv新增ninja==1.13.0并补开关，Torch/MuJoCo/numpy版本未变。无编译器机器可--no-optimizer-layout保留普通graph。
8. 数值测试最初把同一个CUDA optimizer state同时加载给两个对象导致共享tensor，修正为独立deepcopy反序列化；没有放宽阈值掩盖问题。原普通graph累计88次更新逐bit同值，Adam布局累计88次最大权重差1.19e−7、RNG同值，明确不承诺长期逐bit轨迹。
9. 尝试max-autotune和TwinQ batched-GEMM：分别首更新Actor偏差1.21e−5、6.45e−6，超2e−6阈值，均未接入正式代码。前者虽再快约13%，不能为了速度静默牺牲本轮数值约束；后者达到拒绝条件后未继续测速。

### 实际参考来源

- 根/独立AGENTS、本机pianist-v2-change-log技能、论文笔记§2.5 pp.7–8/§2.6 p.8/§2.7表2/§2.9 pp.10–12/§2.10 pp.12–13及此前用户Shadow决策。
- 现有mjlab_env/{preparation,bundle,scene,task,task_bundle,task_logic,source_runtime}.py，rl_torch/{train,sac,replay,checkpoint,evaluate}.py，run.sh、源码训练入口、实时预览及有声render。
- 本机Torch2.10的cuda/graphs.py、optim/{optimizer,adam}.py、include/ATen/native/cuda/{fused_adam_utils.cuh,MultiTensorApply.cuh}和_inductor/config.py；原包更新顺序/optimizer/RNG状态。
- PyTorch2.10官方CUDA文档及v2.10.0 fused_adam_utils.cuh链接见[实现说明](policy_preparation_acceleration_20260926.md)。未复制/导入external_never_copy_only_reference代码。
- 历史500万Shadow run与本轮paired同配置运行的config、metrics、summary；独立probe/profiler，不用不同学习预算伪装提速。

### 具体修改

- preparation.py/bundle.py/scene.py：policy_v1、raised initial reset、每world逐子步quintic、运行时phase、旧fixed规范化兼容；物理CUDA graph开/关均支持。
- task.py/task_bundle.py：验证前N帧空谱、append当前phase观测、info准备标记；真实transition沿原loop入库，不改reward公式。
- source_runtime.py、scripts/train_from_source.py、rl_torch/train.py：模式参数、总buffer构建、reset合同/观测维度、music-only评价；--learner-graph/--optimizer-layout和完整execution_environment配置。
- sac.py、新learner_graph.py/adam_execution.py：保持原SAC公开更新接口，额外图内有限值/foreach Polyak/原数学Adam布局；恢复标准checkpoint、图外动态抽样和构图状态复原。
- run.sh文件头及实际命令：policy模式/准备1s+2cm默认，learner graph/layout默认开；全部新参数、约束和关闭方式写入注释。Shadow LR/batch/replay/预算/奖励开关不改。
- preview/scripts/{render_mjlab_checkpoint,preview_mjlab,preparation_frames}.py及scripts/preview_preparation.py：模式区分、全策略178帧、三段统计、0音画前缀；固定无策略演示明确标注不代表新policy准备效果。
- training_mjlab/requirements-tested.txt新增ninja固定版本。新测试与诊断脚本、README/AGENTS/论文决策及本条日志同步。

### 实际验证

- 准备相关17项通过，含policy physics graph开/关、200个子步的高度、异步局部reset隔离、20/21帧边界、1181维、首条准备样本进replay、空谱误触原key项减少0.5；最终3项CPU归一化检查通过。
- `training_mjlab/.venv/bin/python -m pytest tests/test_learner_graph.py -q`：8 passed，含动态replay增长/覆盖、RNG和计数保持、非空Adam warm-start、图checkpoint→原SAC续训、finite粘滞失败、storage替换拒绝。
- `CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl training_mjlab/.venv/bin/python -m unittest tests.test_torch_sac tests.test_preparation_checkpoint tests.test_source_runtime tests.test_shadow_task_profile tests.test_preparation_training_entry -q`：22项通过，46.302秒；Shadow预算另7项联合检查通过。CPU隔离用于避免干扰GPU性能测试；GPU核心另行已验。
- 源码CPU三组真实构建：baseline158帧/首音0；policy1s178帧/首音20；policy1s+extra.25共183帧/首音25。去前缀后全部goal/fingers逐元素等于baseline，临时scene退出清理。
- 最终流程短训：policy-preparation-graph-acceptance-20260926-010838-328481，2环境×190步=380 replay样本、364更新、2正常178步回合、截断0；graph/layout/compile均开，17.610秒（含7.239秒图建立）。SAC及reward_config与原Shadow5M逐字段相同。该380样本策略总F1=.1861、music_only_f1=.1844，准备16/20帧误触，仅验流程，不能称已学会避免误触。
- 正式eval.sh有声回放：policy_active_final_acceptance_20260926/replay.mp4，178策略帧=20准备+158音乐，CPU前缀0，音频offset0，1780真实物理子步（准备200）。1秒支撑归零；首秒真实错音4次，保留声音/数组，不伪造安静视频。
- 真正同配置512环境pair：baseline policy-preparation-speed-baseline-20260926-010942-269460；加速policy-preparation-speed-accelerated-20260926-011131-508283。均40向量步/20480样本/15481更新/4999随机、batch256/replay1M、2次整曲评估；预算、SAC、奖励、动作/准备合同逐字段断言相等。
- 总时间69.722→26.441秒（2.637倍，含初始化/评估/保存），稳定向量步1.96895→0.53217秒（3.700倍），260→962.1样本/s。独立learner3.755→.951ms/update（3.95倍）；不混淆两种口径。
- 代码语法、来源散列及文档链接最终检查另见diagnostics/policy_preparation_acceleration_20260926。旧归档保留，新源码也归档，不生成持久训练scene/task。

### 未解决与结论边界

- V32-PERF-01：尚未恢复旧约46分钟。按本轮稳态及评估时间估算5M约6094秒/102分钟，尚未实跑，未包含长期漂移/周期存盘；旧46分钟仅约59万更新，当前约500万，未擅自减预算。FP32 GEMM仍为重要开销，更激进候选已拒绝。
- V32-START-01：准备已实际可学习，是否减少开头误触需要新5M策略效果验收；不把短训或静音帧导致F1变化当作改善证明。
- V32-START-02：20Hz端点评分仍可能遗漏200Hz可听短误触，奖励采样规则未擅改。
- 布局舍入误差边界、GPU接触轨迹分叉、原V32-04/06及V32-SAC-03继续保留；GUI无DISPLAY/Xvfb未实测，EGL有声完整回放通过。

交付：[本轮使用/吞吐](policy_preparation_acceleration_20260926.md)、[CUDA执行与数值证据](learner_execution_20260926.md)、[paired数据](../preview/outputs/diagnostics/policy_preparation_acceleration_20260926/paired_training.json)。


## 2026-09-26 — GPU最大吞吐诊断、满replay与同预算规模复核

### 需求与范围

用户要求解释速度变慢、检查继续提高稳定吞吐的可能性并测试GPU最大训练速度。遵循此前“只加速、不改其他”：本轮正式包163文件、run.sh默认、SAC、FP32、batch256、每样本1次更新、奖励/碰撞/准备均未改。扩大环境数只用于诊断，不热改或替换正式512配置；没有新开5M。

### 问题与实际参考

- 上轮962条/s来自40步、未跨reset、replay仅20,480行，不能代表全部训练或满库；进度条累计速度和排除评估的普通步骤速度口径不同。本轮补240步真实训练和满1M学习器。
- 历史46分钟实际593,088更新，Shadow5小时34分4,995,193更新（8.42倍）。满库前后/训练后期没有历史掉速证据；当前开测前也没有用户新增长训进程。
- 容量诊断首次在CUDA初始化前调用reset_peak_memory_stats失败，发生于环境创建前。仅修扫描脚本初始化顺序，保留失败collection_scan与成功collection_scan_v2证据。
- 更严格128次合成输入对照发现当前Adam布局对原fused的2/2,708,674个moment元素超原严格容差，参数最大差仍1.19e−7，loss/RNG一致；没有抹除失败或放宽阈值。分块候选对当前布局逐bit，但没有速度收益，因此不改正式实现。
- 参考根/独立AGENTS、日志技能、论文笔记§2.7及2026-09-25/26已确认SAC/执行范围；包内rl_torch/{train,sac,replay,learner_graph,adam_execution}.py、mjlab_env/{task,task_logic,scene,source_runtime}.py；旧/新完整config、metrics/summary和本轮实际GPU记录。
- GPU利用率解释核对NVIDIA官方nvidia-smi文档及本机帮助：https://docs.nvidia.com/deploy/nvidia-smi/ 。无external_never_copy_only_reference代码复制/依赖，没有修改驱动、功耗、精度或依赖版本。

### 修改位置

- preview/scripts/benchmark_training_speed.py：调用真实run.sh并以200ms采样GPU，保留源码/进程总时间及完整命令，不覆盖已有证据目录。
- preview/scripts/analyze_training_speed.py：从原metrics累计时间恢复普通/评估/reset窗口，报告不同口径及硬件统计；可配置预热排除窗口，不改训练日志公式。
- preview/scripts/plot_training_speed.py：从原始JSON生成PNG/SVG图，不启动GPU任务。
- preview/outputs/diagnostics/gpu_ceiling_20260926：两个真实run的调用/监控/分析、历史审计、学习器与Actor候选脚本/原始数值/报告、最终完整性记录。
- preview/outputs/diagnostics/gpu_capacity_20260926：128至4096六点独立进程扫描、真实1181观测样本、容量报告及初始化失败证据。
- README及docs/gpu_training_ceiling_20260926.md：方法、原始结果、数值失败、吞吐上限定义及局限；根/子日志合并记录本任务。

### 验证与实测

- `python3 preview/scripts/benchmark_training_speed.py --output preview/outputs/diagnostics/gpu_ceiling_20260926/default512_sustained --vector-steps 240`：512×240=122,880条样本、117,881次更新、12次评估，512正常回合、0截断。普通中位967.55条/s，10%–90%分位962.57–973.32；训练器总159.933秒/768.32条/s，含源码进程总174.793秒。未继续掉速。
- `python3 preview/scripts/benchmark_training_speed.py --output preview/outputs/diagnostics/gpu_ceiling_20260926/scale2048_same_budget --vector-steps 60 -- --num-envs 2048 --eval-interval 5 --save-interval 125`：同122,880条/117,881更新/12评估，每10,240样本评估保持；20个关键配置字段相等断言通过。普通1010.37条/s（+4.43%），训练器总156.961秒/782.87条/s，总时间仅少1.86%。该60步未跨训练reset；2048 reset由容量扫描真实覆盖。分组与随机探索在每个世界的时间覆盖改变，不据此修改默认。
- 满1M replay独立初始化9.646GB合成数据，3组×512顺序更新：128k有效行1045.82updates/s、1M1050.17，无满库变慢。Adam128/64/512线程候选对当前布局128更新逐bit，满库分别1046.5/1023.0/1037.3，无收益；原fused对照严格失败证据独立保留。
- 20-update profiler：矩阵主核及辅助60.33%，Adam6.65%，gather2.51%。GPU活动时间不当作墙钟或硬件占用率；正式512实跑GPU-util中位96%、230W、最高58℃，运行中没有激活降频原因，未测到温度/功耗限制。
- 容量扫描每点178步预热、356步策略采集、356步动作重放，两个计时段均跨2次reset。128/256/512/1024/2048/4096策略+环境分别4299/7790/13779/22274/31588/37978条/s，进程显存2.13/3.44/6.11/11.38/22.04/42.98GiB。均无容量错误；不含学习/replay，不冒充训练吞吐。
- Actor原样确定性forward，50真实+50随机观测，batch1/512各100组动作逐bit；参数原址更新/恢复与RNG检查通过。batch1同步延迟98.56→37.96µs，但178步评估仅推算省10.8ms，未集成。batch512是确定性形状对照，不当作随机训练收益。
- 新脚本语法及文档链接、源散列/原run.sh保持检查见本轮final_integrity.json；两组实跑/扫描/候选均正常退出，GPU已释放。图表已生成并查看。

### 结论与未解决

本次所测完整训练普通窗口最高1010条/s（2048规模诊断），当前固定512约968；独立学习端约1050，4096纯采集3.8万不能算训练速度。未发现值得合入的新候选，不更改任何正式训练默认。

V32-PERF-01保留：更深入矩阵/归约与完整评估张量调度仍可研究，必须保护RNG、检查、reset、旧新观测独立存储及数值约束；本轮没有穷尽所有实现或显存上限。没有完成新5M，不能据短测保证长期吞吐/收敛或误触改善；原V32-START-01/02和驱动/接触TODO继续。

交付：[详细报告及复测](gpu_training_ceiling_20260926.md)、[吞吐图](../preview/outputs/diagnostics/gpu_ceiling_20260926/throughput_comparison.png)。


## 2026-09-26 — 启动当前默认小星星可训练准备5M

- 用户明确要求开启一次训练；按当前run.sh从头启动512×9766=5,000,192样本，保留1秒policy_v1、1181观测/47动作、原SAC/Shadow接触及所有已启用执行加速，不采用2048测速设置。
- 参考：run.sh实际参数、源码训练入口、论文笔记已确认SAC/准备决定、日志技能、当前config.json和启动metrics。无新增设计或论文参数选择；未修改正式训练源码/算法参数。
- 问题：启动前GPU无计算进程；模型构建和GPU初始化正常，无新增阻断问题。既有MULTICCD capsule/mesh单接触限制不变。
- 修改：新增training_mjlab/launches/twinkle_policy_preparation_5m-20260926-020140/launch.json与train.log，生成本轮运行目录；新增docs/training_policy_preparation_5m_20260926.md并更新README/根及内部日志。后台独立会话PID 3388050，stdin断开，输出持久写文件。
- 实际命令：`bash run.sh --output-dir training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140`；run.sh自动追加时间戳，实际目录为training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894。
- 核验：2026-09-26T02:03:42.400569+08:00，step164／83968样本／78969更新／8评估；进程存活，512/5M/准备/动作观测/执行开关/batch/replay/更新比及无warm-start断言通过，163文件源码指纹匹配。启动成功不等于训练完成。
- TODO：本轮最终summary/checkpoint/视频与准备误触效果仍待训练完成；V32-START-01/02、V32-PERF-01和原驱动接触限制保留。
- [本轮训练记录](training_policy_preparation_5m_20260926.md)。

## 2026-09-26 — 导出当前运行的最新训练视频

- **需求与问题**：用户要求最新训练视频；训练仍在运行，需从当前轮最新已经完整保存的检查点选取，而不是旧 Shadow 默认历史轮或尚未完成的下一检查点。导出时最新为 `step-1500.pt`，训练指标已超过该保存点。
- **参考**：当前轮 `config.json`、`metrics.jsonl`、`step-1500.pt`、`preview/scripts/export_latest_training_video.sh`、`eval.sh`、生成的 `report.json`；本机 `pianist-v2-change-log` 技能。源码和配置严格校验按现有入口执行。
- **修改**：新增 `preview/outputs/replays/shadow_sync_step-1500_20260926-022153-947827789/` 下的有声视频与报告，并把实际检查点和视频写入 `docs/training_policy_preparation_5m_20260926.md`；训练代码和运行中进程未改。
- **验证**：`bash preview/scripts/export_latest_training_video.sh training_mjlab/outputs/twinkle_policy_preparation_5m-20260926-020140-20260926-020140-983894` 返回成功；FFprobe 核对 H.264/AAC、20 fps、199 帧、9.95 秒，报告 `logged_metrics_match=true`。完整回放 F1=0.6491，音乐段均帧 F1=0.6046，20 帧准备期零错键；导出后训练仍在更新（核验 step1886）。
- **TODO**：V32-START-01 仍需对完成 5M 后的最终策略验收，V32-START-02 的 200 Hz 可听短错音与 20 Hz 奖励采样差异继续保留；最终视频需等待最终检查点。此视频仅是 step1500 阶段快照。
- **最新补导**：训练随后保存 `step-2000.pt`，同一入口再导出[step-2000有声视频](../preview/outputs/replays/shadow_sync_step-2000_20260926-022408-889856540/replay.mp4)，保留前一阶段快照。FFprobe 核对 H.264/AAC、199帧/9.95秒；报告 `logged_metrics_match=true`，整回合F1=0.8076、乐曲段均帧F1=0.7833，准备20帧无错键。训练至step2096仍在运行，最终效果未验收。
## 2026-09-26 — 当前奖励公式与空闲手指初始姿态奖励

- **任务/范围**：列明当前 `run.sh` 实际启用的五项奖励及公式；将源码场景 reset 时每手20个手指关节角记录为标准姿态，增加显式 `--humanoid-finger-gesture` 空闲手指奖励。正在运行的旧训练进程及默认五项不热改。
- **问题**：奖励管理器原先只处理动作参考的论文 `natural_posture_reward`；物理关节名与策略动作名不同（如 `right_/right_hand_thumb...`），不能按动作名前缀推导实际 `qpos`。论文 pp.11–12 式20的自然基准仍未公开，不能把此次用户指定的初始姿态当作论文映射。
- **参考**：`run.sh`、`mjlab_env/task_logic.py`、`mjlab_env/rewards.py`、`mjlab_env/reward_config.py`、`mjlab_env/task_bundle.py`、`models/hands/tsensei_hand_constants.py`、源码实际构建的scene/task manifest、当前5M轮末 `metrics.jsonl` 的 `reward_terms`，以及 `docs/references/biomimetics_2026_610_notes.md` 式20与既有决策；本机日志技能。未复制只读外部参考目录。
- **修改**：`reward_config.py` 加入默认关闭的第22项及旧JSON兼容；`rewards.py` 从scene reset `qpos` 按task真实关节名抽取每指4角，计算当前物理角均方误差的高斯正奖励，只给当前谱帧无指法分配的手指；`rl_torch/train.py` 新增快捷开关并把实际40值写入新训练config/checkpoint元数据。四个奖励模板显式加关闭项；`run.sh`、README、奖励总表、论文决策、[公式与范围](humanoid_finger_gesture_20260926.md)、[40角快照](humanoid_finger_gesture_pose_20260926.json)同步说明。两手四关节分组与现有10指指法索引一致，最高每指0.2、全部空闲总上限2.0；标准差0.25rad，可由JSON调节。
- **验证**：`training_mjlab/.venv/bin/python -m unittest tests.test_humanoid_finger_gesture tests.test_paper_rewards tests.test_mjlab_task -v`，20项通过；按当前run参数真实源码构建并检查快照40值／左右各20；用真实scene/task manifest在CPU实例化奖励管理器，得到两侧索引均5×4、启用时六项列表；`--humanoid-finger-gesture --inspect-only`确认仅新增第六项。未进行GPU新奖励短训或长期收敛／误触对照；测试中的合成角不是用户标准姿态，正式标准值见快照。
- **TODO(V32-GESTURE-01)**：新奖励的训练学习效果及准备空谱的实际动作变化，需在用户选择启动新训练后比较；不能从公式／单元测试推断收敛。原V32-PAPER-U05等论文未知映射仍未解决。


## 2026-09-26 — 最终5M视频与F1曲线导出

- **需求/问题**：用户要求最新视频和F1曲线；本轮已经完成，最新检查点是final.pt。训练后3个包内源码文件变化，普通eval.sh按源码散列拒绝回放，保留失败日志；现有总览F1绘图脚本只选择`sac-`正式目录名，会漏掉当前自命名运行目录。
- **参考**：本轮final.pt/config.json/summary.json/metrics.jsonl、已有源码归档和`scripts/eval_from_source_archive.py`、`preview/scripts/render_mjlab_checkpoint.py`、`scripts/plot_training_f1.py`的数据口径及本机日志技能。未参考或复制只读external目录。
- **修改/产物**：用匹配归档在临时目录运行原评估器，生成`preview/outputs/replays/twinkle_policy_preparation_final_archive_20260926-141944/`有声视频/报告；用本轮全部489个日志评估点导出`preview/outputs/training_reports/twinkle_policy_preparation_final_20260926-141816/` PNG/PDF/CSV/JSON，整回合与乐曲段F1并列，不平滑；新增`docs/training_final_video_f1_20260926.md`并更新README/根及本日志。未改正式训练源码或检查点。
- **实际验证**：归档163/163源码文件散列与训练一致，config/checkpoint由归档评估器再次校验；渲染成功。FFprobe核对1280×720 H.264/AAC、20fps、199帧、9.95秒，图片目视检查；视频F1=.9367977381、乐曲F1=.9351265823、准备错键1帧/2次发音，与最后一次日志评估逐字段吻合。最高记录step9560整回合F1=.9769663215，不误称最终视频表现；训练summary完成5,000,192样本/4,995,193更新。
- **TODO/边界**：V32-START-01最终视频准备仍1帧错键，V32-START-02的20Hz评估对200Hz短错音盲区保留；F1高不代表动作完全平和。本轮只交付导出，没有逐子步运动分析。
- [最终视频、曲线、复现和限制](training_final_video_f1_20260926.md)。


## 2026-09-26 — 侧边请求导出当前训练最高已保存F1视频

用户要求导出当前训练F1最高视频。只新增导出产物并追加日志，不改主线程源码/参数/训练状态。参考当前5M的metrics.jsonl、summary/config和现存*.pt、eval_from_source_archive.py、归档render脚本、论文笔记的已有评估约定及日志技能。

已确认本轮5M完成；日志全局最高step9560/F1=.97696632没有权重，不能回放。按所有已保存权重对应eval.f1排序，最高step8500/F1=.9711075425，独立回放F1精确一致，音乐段F1=.96745027。使用policy_preparation_acceleration_20260926.tar.gz，163文件来源及完整配置/准备合同校验保留，避免采用主线程后续新源码。

首次预建元数据目录与render“输出目录必须不存在”要求冲突；保留失败日志，改为新video子目录重试成功，未改导出器。ffprobe确认1280×720 H.264/AAC、199帧/9.95秒，预览图人工查看正常；1780物理子步真实按键音频保留，准备20控制帧无端点错键但音频1次短误触。V32-START-02采样盲区保持，未宣称完全无误触；未保存的最高step无法恢复。

产物：preview/outputs/replays/twinkle_policy_best_saved_20260926-173850/{selection.json,README.md,video/replay.mp4,video/report.json}；只追加本条维护记录，不覆盖其他任务文档。


## 2026-09-26 — 完整5M审计、质量奖励与加速对照后开启新训练

- **任务**：用户要求曲目F1目标0.99、少碰撞、短训练，先详细计划并调用子任务实现后直接开训；另明确授权batch1024/.25候选经短测择优。
- **实际问题**：峰值step9560未保存、final差于8500；20Hz端点漏掉15ms准备误触；同手接触与TX目标阶跃；四个检查点共同漏C4。旧每指最高.2的姿态正奖可能抵消漏音代价。网络微调短测出现显著暂时退化，确认加载/alpha/原奖励正确，恢复Adam的E也未改善，不能归咎单一因素。新检查点多出的评估运行元数据使旧config完整性检查误拒绝回放，已作明确字段修复；归档首次引用不存在的requirements.txt/setup.py失败，改用实际requirements-tested系列，临时文件成功后原子替换并校验163源码文件。绘图误用无matplotlib环境后改现有learner环境，无新增依赖。
- **参考**：父5M config/metrics/summary与8500/9000/9500/final；匹配旧归档；`mjlab_env/{reward_config,rewards,task,scene,self_collision,task_logic}.py`、`rl_torch/{train,checkpoint,sac}.py`、source/eval入口、原reward模板及reset40角；论文pp.8–12相关决策、本机日志技能。SAC及奖励塑形原文链接见详细计划，未复制或导入外部只读参考项目。
- **修改**：三个子任务分别完成审计、奖励/子步诊断、训练/检查点工程。新增task_conditioned_v1（仅正确有音帧、未来.25秒指法豁免、全手最多.05候选）；旧公式保留。可选200Hz同手/跨手接触计数不改反力且保留CUDA图。新增逐评估原子best、最差回合联合质量门槛、初评、严格同合同finetune。eval_from_source仅允许明确的3个checkpoint运行字段缺于config，未知字段/源码差异仍拒绝。新增launch_quality_training.py/run_quality.sh与三个完整奖励JSON、三个旧运行审计脚本和短测汇总脚本；run.sh仅补CLI头说明及核实既有显式humanoid开关，未改变其默认命令。维护README、AGENTS和论文决定。
- **验收**：旧源码163文件严格匹配的12标准+4逐子步完整回合；新奖励/碰撞CPU30项，训练工程26项/39子测试，回放配置及归档26项，汇总3项。GPU24步graph开关对照reward差0、qpos4.66e-10/qvel2.98e-7、coverage10、局部reset隔离；物理/乐谱散列与旧父相同。CPU静态八度触键8组解满足动作范围、无额外穿透，不能当成动态轨迹。A–E各262144样本真实短训完成，A/B总208.96→104.87秒（1.99倍）、稳态972→2631样本/秒；C全组合相对B碰撞377→51子步，但最差音乐F1 .9274→.9019。C优于无humanoid对照D的.8689/66子步；所有短测未超过初始音乐分数。C final经eval.sh三回合严格回放与训练一致，8个best/final配置验收通过。
- **新训练**：从C final完整恢复同奖励learner/Adam/温度，fresh replay先填完整曲目；512×4096=2097152新样本、.25更新比、batch1024、原三LR3e-4/网络/gamma/20-200Hz/47-1181/实体碰撞保持；false1+self.2+有条件humanoid.05。每100步评估3回合、音乐F1改善保存best，连续3次音乐/micro≥.99且准备零错帧/200Hz零手碰撞才停止。后台启动和首批新增更新已验证；小Actor学习率未采用，训练未完成。
- **TODO**：V32-QUALITY-01联合0.99/无手碰撞验收；V32-GESTURE-01长期净收益；V32-START-01策略目标阶跃／动态转位；V32-START-02短音采样盲区；V32-04/06及V32-SAC-03继续保留。静态IK与短测不证明长期收敛或硬件安全。
- [详细计划、对照与启动目录](training_quality_optimization_20260926.md)、[旧训练审计](training_run_review_20260926.md)、[humanoid公式与验收](humanoid_task_conditioned_20260926.md)、[A–E数据](quality_trial_summary_20260926.md)。

## 2026-09-27 — 导出当前质量训练的 best.pt 有声视频

- **任务／范围**：用户要求生成当前训练 `best.pt` 视频。只读取已完成续训目录的最佳检查点和完整配置，新增独立回放产物；不修改训练代码、模型参数或检查点。
- **问题与选择**：启动元数据仍标记为运行中，但该输出目录已有 `summary.json`；以其当前 `best.json` 标记的 `best.pt` 为准，评估主指标为音乐段最差回合 F1=0.97356849（step4000，531968更新）。不将这项分数称作0.99或无碰撞达标。
- **参考**：该训练目录的 `best.pt`／`best.json`／`config.json`，`eval.sh`、`scripts/eval_from_source.py`、`preview/scripts/render_mjlab_checkpoint.py`，以及本机 `pianist-v2-change-log` 技能。回放从源码及完整配置重建，未引用 `external_never_copy_only_reference/`。
- **产物**：新增 `preview/outputs/replays/twinkle_quality_continue_best_20260927/`：`replay.mp4`、真实物理子步音频、MIDI、事件、轨迹、报告及预览图。采用 performer／full 视角，确定性 Actor 均值动作，包含20帧可训练准备和完整乐曲。
- **验证**：`bash eval.sh --checkpoint …/best.pt --mode render --output … --camera-view performer --camera-framing full` 成功；FFprobe确认 H.264/AAC、1280×720、20fps、199帧、9.95秒；目视检查预览图。回放整回合F1=0.97092026，音乐段均帧F1=0.97356841、micro-F1=0.97063903，和保存best评估相符。准备1个错误控制帧，音乐3个错误控制帧；同手接触9个控制帧／23物理子步，跨手为0。
- **TODO**：V32-QUALITY-01、V32-GESTURE-01、V32-START-01/02与驱动／数值限制保持；视频仅呈现当前best，不证明长期策略、0.99或无碰撞目标已经实现。


## 2026-09-27 — best失分、关节范围／卡键审计与小学习率续训

- **目的／决定**：用户要求完整分析当前best失分和大动作卡琴，给方案并在best基础续训；明确批准软件目标范围／变化率限制、Actor3e-5短测择优。已先记录论文p.8§2.6子步插值与本轮slew的差别、表2Stage2 LR1e-4与本轮候选差别，未改未获决定的论文设计。
- **实际问题**：原best音乐F1=.97356849，final=.95352627，前0.75秒占66.25%失分。没有碰琴体，左拇指进入A3键底及持续顶C4侧面，驱动与约束力矩支持被阻住；原TX跨度1.221m允许24.66cm目标跳变。Pitch实际绕世界Z，TZ移除，不能靠整手抬高退出。奖励缺乏专门键底／侧堵反馈，energy在堵转低速下未必大。softlimit实际可轻微越界。三组短训均退化，best均保留初始；低LR只是退化较轻。
- **遇到的工程问题**：审计初次重复collision_diagnostics关键字已改显式合并；Rot2下限.6/.9不含origin的.4rad，严格拒绝并留档，没有改origin迎合范围。首次高LR限制短训在step510收到SIGTERM，来源未确定，无final/summary，明确排除；以脱离前台会话的进程重跑完整同预算试验，旧证据保留。重跑曲线有浮点训练分叉，不承诺逐bit复现。
- **参考**：原运行config/metrics/summary/best/final及匹配quality_optimization源码归档；mjlab_env/{scene,task,task_logic,action_mapping,rewards,reward_config,source_runtime}.py，rl_torch/{train,checkpoint,replay,sac,evaluate}.py，原source/eval/preview/render入口及模型manifest/真实reset；论文事实/决策笔记；本机pianist-v2-change-log技能；MuJoCo官方modeling/computation文档与CAPS论文仅作说明比较，链接在主报告。未复制或导入external_never_copy_only_reference。
- **核心修改**：新增control_limits.py严格canonical/resolve与clip→slew，物理目标单位及reset包含校验，默认None保留历史行为；scene/task使用真实上一ctrl，奖励/51槽观测记录APPLIED，SAC replay保留raw47，1181观测不增加；局部reset不影响其他world。train/checkpoint/evaluate及preview/render完整保存/校验输入和解析控制合同；仅显式finetune允许改控制，其余物理/乐谱/准备/形状校验保留。低Actor LR沿用已有SAC参数，不改算法损失、Critic／温度LR、网络或预算。
- **入口／文档**：run.sh和run_quality.sh头补全部新CLI；launch_quality_training.py支持显式Actor LR/控制限制/完整预填，resume默认读取父LR。新增/扩展audit_checkpoint_trajectories.py、analyze_motion_contacts.py、plot_recorded_key_contacts.py、compare_contact_candidates.py、probe_control_limits.py、probe_limited_contacts.py、verify_control_limits_real.py、summarize_range_trials.py及相关tests；8个有效候选JSON。同步README、AGENTS、论文决定和独立控制/审计/训练曲线/短训说明。
- **实际验收**：CPU汇总38 passed、1 GPU skipped、69 subtests passed；汇总脚本另6项通过。真实GPU8world×24步×graph开关通过，局部reset对其他world19项状态逐bit隔离、真实ctrl/APPLIED观测一致、限幅限速误差≤7.2e-8。9组含原控制×3标准回合；审计子任务20完整回放，1780物理子步/回合。源归档164文件严格匹配；新best标准eval.sh3回合成功。三完成短训各262144样本/42752更新，末尾F1原控制高LR .91730、腕限制高LR .92574、腕限制低LR .95380，耗时约121–122s；均未超过初始。所有详细候选都未降低原best键底时长，wristgentle消除后段C4堵转、总侧底1.245→.800s但键底.190→.305s；不称为卡键已解决。
- **选用／开训**：本轮显式wrist_gentle（右TX[-.2,.2]m、左[-.1,.2]m、TX/TY1.2m/s、Pitch4rad/s），Actor3e-5；通用默认不自动开启。新父为低LR试验step0 best，三网络逐张量与原best相同，新控制下音乐F1=.97525623。北京时间01:26启动twinkle_range_finetune-20260927-012613-20260927-012614-462007，512×4096=2097152样本，原八奖励/batch1024/.25更新比/gamma.8/20-200Hz/47-1181/实体碰撞/origin保持；新replay完整曲目预填。PID及完整argv在latest_range_launch.json，已验证首批新增更新；启动状态快照不作实时监控。原latest_quality指针已留备份再更新。
- **回放留档**：control_limits_finetune_20260927.tar.gz含源码/资产/脚本/配置、无冻结物理场景；SHA256=721816a28666c084d92c936f736f782504365a9746dc13f4f336f62a106fb846。新旧checkpoint严格分合同，原best及归档完整保留。
- **未解决TODO**：V32-RANGE-01仅改善部分动作/接触，键底堵转仍在；V32-QUALITY-01音乐/micro≥.99和少碰撞未达成；V32-START-01目标跳变与动态避障仍需续训验收；V32-START-02的20Hz错帧0不等于200Hz零短错音（wristgentle仍30ms准备误触）；V32-04/06、V32-SAC-03标定和浮点接触限制保持。既有训练早停不检查键底，达门槛后仍要做详细接触审计。后续明确接触名单的堵转反馈、难点课程、相同合同replay sidecar仅为计划，未擅自实施。
- **详细结果**：[主报告](best_range_finetuning_20260927.md)、[卡键审计](motion_contact_audit_20260927.md)、[短训汇总](range_trial_summary_20260927.md)、[控制合同](control_limits_20260927.md)。
# 2026-09-27：导出当前续训 best.pt 视频并核对训练速度

- **任务**：用户要求导出当前训练的 `best.pt` 视频，并报告实际训练速度。
- **参考**：当前运行目录的 `summary.json`、`metrics.jsonl`、`best.json`、`best.pt`；`eval.sh`、`preview/scripts/render_mjlab_checkpoint.py` 及源码配置合同。
- **结果**：训练已完成 2,097,152 条样本、501,504 次更新，耗时 1,163.01 秒（约19分23秒）。总吞吐约 **1,803 条样本/秒**；日志稳定段中位数约 **1,853 条样本/秒**，最后记录约1,821条/秒。最佳 checkpoint 为 step2600，音乐段 F1 **0.98227856**，音乐 micro-F1 **0.98100173**。
- **产物**：[replay.mp4](../preview/outputs/replays/twinkle_range_finetune_best_20260927_v2/replay.mp4)、`report.json`、`rollout.npz`、音频和预览图。视频为1280×720、20fps、9.95秒，含1秒可训练准备段和完整乐曲；回放从源码和保存配置重建，控制限制合同通过校验。
- **验证**：`ffprobe` 确认 H.264/AAC、1280×720、20fps；回放报告显示准备20帧无20Hz错键帧，音乐段F1与 best 评估一致。准备阶段仍记录1个短音符事件，不能称200Hz音频完全无误触。
- **TODO**：F1尚未达到0.99；键底／键侧接触和200Hz短音仍需独立审计，不能仅凭视频或20Hz F1宣布完成。


## 2026-09-27 — 通用整指净空、曲首卡底与有界humanoid

- **任务/授权**：用户要求解决曲首拇指卡琴、非按键整指净空、增强humanoid但不盖过弹琴，不准曲目专属X范围，方案验证后写run.sh开训。已说明论文pp.11–12式19/20差异，用户明确选择几何约束＋有界辅助惩罚；随后询问更可行方案，按实测迭代为实际位置许可的按压控制。
- **问题/根因**：旧best .98227856仍左拇指键底250ms；origin净空正，策略先在琴外下降再横移。未来.25s姿态豁免覆盖30/50底碰子步。旧humanoid在准备/弹错0、平均贡献.28%。初版严格PD目标FK候选F1跌.08333，被拒绝；265/284成功用指帧的目标末端低于充分按压面>1mm，目标过驱中位15.71mm而实际仅1.73mm，不能混为物理穿透。原始任意安全端点间有路径下穿反例。低位停腕候选侧碰增多，未选；两LR短训高LR退化。初始probe helper误名已修；observer测试曾硬编码MuJoCo重排后的geomID，改按名称；新launcher无效早停开关已删除，未以静默降级掩盖。
- **参考**：164文件匹配旧归档best/final完整config及8回合；源码scene/task/task_logic/bundle/action_mapping/rewards/reward_config/preparation/self_collision/source_runtime与train/checkpoint/evaluate/sac、原preview/eval入口；论文事实/决定及本机日志技能；MuJoCo官方仿真/计算文档（链接主报告）。不复制/导入外部只读目录。
- **核心改动**：新增finger_clearance.py解析40碰撞体FK和几何投影；最终active_contact_v1根据实际支持区/顶底位置许可正常按压力，闲指2mm净空、实际用指5mm软接触余量/2mm支持边距、局部恢复≤.2rad。几何CUDAgraph只优化执行。scene/task保留真实物理、raw动作写replay、APPLIED记录观察/奖励，当前帧目标对齐、局部reset隔离。新geometry_clearance_v1负辅助项80%几何/20%姿态、总≤.1并受理想漏键预算约束；准备/错键有反馈，用指不罚合法PD请求穿透。其余7项保持，论文未知映射不填。
- **接口/诊断**：可选guard默认None，run.sh显式JSON；去旧曲目TX区间，仅保留腕速度。canonical/resolved合同完整保存，显式网络迁移才准改控制/奖励。新增200Hz只读键底/侧底/非末端观察；target-max-key-underside-substeps=0对新best资格及早停均生效，缺数据失败。训练/评估/交互/视频参数一致；新增完整审计、几何/路径/全键盘探针、续训启动及比较脚本。export_best进一步同步新权威best语义。run.sh头含92个接受参数并更新默认值；README、AGENTS、论文决定和独立文档已维护。
- **验证**：集中53 passed/76 subtests含真实GPU图/eager、APPLIED观察、准备负罚/局部reset、FK≤8e-8m、奖励/历史模式/新旧合同/诊断资格；脚本13 passed/2subtests另通过。8个黑白键端点、全键盘162个限位内端点许可/错键拒绝均通过，双手并集88键，但不当动态全曲保证。strict/idle-only/active-contact/interlock同best各1标准+1详细回放，F1分别.08333/.94002/.91534/.91218，底均0；idle-only缺完整用指保护，interlock侧碰更高，均不作为最终选用。
- **短训**：同父、同物理/谱面、512×512=262144条、batch1024/.25、各42752更新；Actor3e-5末尾F1 .921730/底0/手35/准备错1，3e-4末尾.867541/底0/手16/准备错1，约147.49/147.98秒（训练循环，不含外层源码构建）。选低LR完整学习器续训；未把低LR超过新控制初评当作超过原父.98228。
- **回放/留档**：低LRbest严格eval.sh重建及有声视频通过，音乐分数与训练一致，200Hz底0/侧底132/非末端93/1780子步。generic_clearance_trials_20260927.tar.gz SHA256=1caf48862dbfe1da1e699a521322155c2cf26c024df6bbba79ea578c6299da93，432总文件/166包源码逐项匹配；保存源码/配置，无冻结scene。
- **正式启动**：09:34:25已启动twinkle_generic_clearance_finetune_20260927-20260927-093425-663203，512×9766=5000192新样本；父为低LRstep512 best，网络/Adam/实际alpha .00287736/RNG逐张量恢复，新replay先整曲预填。明确不是精确物理/replay恢复。已验证step356累计65408更新（父42752）；09:41快照best step1100音乐F1 .94113951、底0，仍在训练。最新argv/PID/日志见latest_clearance_launch.json，latest_quality旧指针备份后更新。
- **TODO**：V32-CLEARANCE-01有限目标与许可不保证连续动力安全，真实净空暂态不足/侧碰仍有；V32-CLEARANCE-02多曲/和弦/高速动态泛化；V32-QUALITY-01的.99/少自碰撞仍未达；V32-START-02的20Hz短音盲区保持。短训/局部模型不能代替长期或硬件验收。
- **报告**：[主报告](generic_finger_clearance_20260927.md)、[基线审计](finger_clearance_audit_20260927.md)、[几何规则](finger_clearance_geometry_20260927.md)、[辅助公式](humanoid_clearance_reward_20260927.md)。

- **最终归档与导出复核**：`training_mjlab/source_versions/generic_clearance_20260927.tar.gz`，SHA256 `bbce9ac3affecc0e0db6e59bfa719e31432cdd4ad17d4006fef2e8bcc2d3418a`；435个文件，166个包源码逐项匹配正式运行config。仅源码、配置、脚本、模型和文档，无冻结scene/task。权威best导出独立8项测试/4子测试通过，旧range与新短训实际CPU选择验证通过。
- **后续进度快照（2026-09-27T09:50:27+08:00）**：训练step2848，累计updates384384；合格best step2700音乐F1=0.961121281，三个回合键底子步[0.0, 0.0, 0.0]、手部碰撞子步[0.0, 0.0, 0.0]、侧/底子步[22.0, 22.0, 22.0]、20Hz准备错帧[0.0, 0.0, 0.0]。训练仍运行，当前音乐表现尚低于旧父.982279，侧接触与真实净空暂态仍有；不宣称达到.99或连续物理安全。

## 2026-09-27 — 完成通用净空训练后导出最佳／最终视频及F1曲线

- **目的／问题**：用户请求最新视频和训练F1图；本轮训练已完成，best在step7100而final在step9766，两者效果不同。现有总览绘图脚本仅把`sac-`命名目录纳入曲线，当前自命名run会被漏绘。
- **参考**：`training_mjlab/outputs/twinkle_generic_clearance_finetune_20260927-20260927-093425-663203/`内`config.json`、`best.json`、`summary.json`、`metrics.jsonl`；`eval.sh`、`scripts/eval_from_source.py`、`preview/scripts/export_best_mjlab.py`、`preview/scripts/render_mjlab_checkpoint.py`、`scripts/plot_training_f1.py`，论文笔记V32-PAPER-11与本机日志技能。
- **修改**：新增`docs/generic_clearance_video_f1_20260927.md`；`preview/outputs/replays/generic_clearance_{best_step7100,final_step9766}_20260927/`各输出有声视频、预览、轨迹/报告；`preview/outputs/training_reports/generic_clearance_complete_20260927/`输出全图、放大PNG/PDF、99评估点CSV和来源JSON。未改训练源码、模型、权重、配置。
- **验证**：`export_best_mjlab.py --select-only`权威选择best通过；两次`bash eval.sh --mode render`严格源码+完整配置重建成功。FFprobe确认双片H.264/AAC、1280×720、20fps、199帧、9.95秒；报告音乐F1相对训练记录误差分别2.12e-8／4.69e-8，整回合F1相同；目视检查预览及曲线。best音乐F1=.990024、micro=.987993、键底0、手碰撞0但侧底134子步；final音乐F1=.978843、手碰撞2帧，训练没有在结束时保持最佳表现。
- **TODO**：V32-QUALITY-01微平均F1、键侧接触及多曲泛化未达验收；V32-START-02仍有20Hz评分盲区。详细产物链接与口径见[独立说明](generic_clearance_video_f1_20260927.md)。

## 2026-09-27 — 按当前通用净空配置开启从头训练

- **目的/问题**：用户要求新开训练；当前run.sh是从头配置，上一轮正式运行额外warm-start过短训best，直接复用其命令会变成续训。检查时无训练进程，GPU空闲；启动中无实际报错。
- **参考**：`run.sh`、新旧两轮`config.json`、新run的`best.json`/`metrics.jsonl`、论文决定笔记及本机pianist-v2-change-log技能。
- **修改**：执行`bash run.sh --output-dir training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927 --no-progress`，后台PID2195168，自动目录`training_mjlab/outputs/twinkle_generic_clearance_fresh_20260927-20260927-132520-465250`；新增同名前缀`.launch.log`、`.launch.json`及`docs/generic_clearance_fresh_training_20260927.md`。只生成新训练及说明，不改模型、run.sh、旧检查点或旧run。
- **验证**：新进程仍在，源码+config已构建；2026-09-27T13:27:11+08:00指标step230、采样117760、更新28190，不只是启动空进程。新config无任何检查点迁移，seed42、初始温度1、预热4999/学习5000、512×9766/batch1024/.25；与旧run相比source/reward/control/evaluation合同一致。step0初评F1 0.158228仅为随机策略，不能称效果。
- **TODO**：V32-FRESH-01长期训练完成后验收best/final及视频；V32-QUALITY-01、V32-START-02保留。[独立说明](generic_clearance_fresh_training_20260927.md)。


## 2026-09-27 — 最新从头训练 best 导出及旧产物清理

用户要求导出最新一轮 best 视频和F1曲线，并删除较早训练产物、无用输出及旧场景STL等。用户澄清仅处理13:25最新从头训练，不混入此前微调曲线。使用pianist-v2-change-log技能；参考本轮config/best/summary/metrics、eval.sh、render_mjlab_checkpoint.py、analyze_training_run.py及训练器权威best选择合同。

导出`twinkle_generic_clearance_fresh_20260927-20260927-132520-465250/best.pt`（step9600，三回合最差music F1=.94345991，micro=.94138544，键底0）有声视频与单轮F1曲线。FFprobe确认H.264/AAC、1280×720、20fps、199帧/9.95秒；回放F1=.958186、micro=.950530、准备帧无误触、键底0、键侧/底42子步。曲线覆盖99评估点并按best.json标星。发现旧分析脚本按平均F1选step9000且忽略best.json；输出采用一次性分析和权威选择校验，未改脚本。

清理2026-09-17至09-24的273个冗余周期checkpoint，保留每轮指标最佳权重及best/final；删除13个旧生成场景/cache目录，源mesh assets、训练配置/metrics/summary和当前输出不动。共释放约7.64 GiB。明确记录已删冻结scene的旧检查点可能需重新导出场景才能回放。清理清单、视频、图表、CSV及限制见`docs/generic_clearance_fresh_video_cleanup_20260927.md`；本项目详细记录见其同名内部日志。未解决事项为历史冻结scene的逐checkpoint源码回建尚未逐项验收。另外删除一次性通用分析器因忽略best.json生成的3个临时错选图表/报表，保留按权威选择合同生成的F1 PNG/PDF/CSV/JSON。清理manifest合计289项、7.641GiB；残余场景mesh副本数核验为0。


## 2026-09-28 — 独立保护开关、拇指漏键/采样审计及入口清理

- **任务与决定**：用户要求7项审计/修改；明确选拇指保持Shadow统一点、可选距离。应用pianist-v2-change-log技能。两动作保护改默认关闭；现有geometry奖励/self0.2/false1与腕速保持，不擅改新奖励/频率/SAC或开启训练。
- **问题**：wrapper默认保护与Python默认None不同；self快捷0被误读成实际权重0且F1图真有错标；现场lift.025与注释.02漂移；Shadow目标绘图遗漏profile；新thumb模块导入NumPy破坏系统Python轻量源码入口（真实run inspect发现，改局部导入并无site测试）；第一次临时消融漏连虚拟环境，补齐后重跑；一次pytest文件名输错更正后实际验收。不能把普通51轴legacy预览可达当47轴Shadow驱动同合同动态证据。
- **参考**：两轮generic_clearance finetune/fresh完整config、summary、best/final权重和166文件匹配源归档；run/train/evaluate/checkpoint、finger_clearance/source_runtime/task_bundle/task/rewards/task_logic/self_collision、URDF/驱动/碰撞代理/fingertip_geometry、原preview脚本；用户截图；论文笔记及MuJoCo/Composer/原Shadow官方资料（主报告含链接）。外部目录只读未导入。
- **修改**：两个独立BooleanOptionalAction与JSON可选字段，旧缺字段不扩写；关控制保留geometry反馈/诊断，真实目标identity；run新默认JSON并删除冗余self0实参、100参数头同步；权重图优先JSON；拇指mode/两距离贯通源recipe、CPU/GPU、manifest与回放/launcher，目标图profile修正；新采样/消融脚本与完整结构/优化文档。模型、origin、乐谱/指法、实体碰撞、原奖励公式不改。
- **事实/实测**：上午续训best .990024但micro .987993，下午从头best最差.943460；均已完成。上午best/final左拇指C4 score30–33全漏，下午best详细回合后两帧能按到；四个零学习消融（关腕速/闲指/用指/双控制）均未修复旧策略C4，F1 .973568/.975678/.984539/.977607，原.990024。best的20Hz错键0但200Hz有119key-substeps被末端漏采；false_press末端和周期均值分别−1.98814/−4.24084（仅离线，不改训练公式）。自碰撞已有200Hz并集，不将所有奖励笼统视为漏采。
- **清理**：8个未采用JSON移入configs/archive/20260927_control_trials，旧路径/内容SHA见index；保留所有checkpoints、runconfig、模型、归档、视频以及动态引用模板，删除可重建pytest缓存。配置不是显存/慢训练主因。
- **验证**：集中40 passed/59subtests含真实GPU图双控制关闭identity、APPLIED观察/局部reset、奖励/兼容合同/目标；CLI21/23、补充18/43、thumb/source28/16均通过（组间重叠不相加）；修system入口19/16，真实system build-only成功；实际run.sh inspect-only已成功。8次旧源best/final标准/详细回放+8次控制消融回放；所有消融严格同物理/谱面hash、0学习。采样脚本实际数据＋合成25/100Hz/空漏键/坏映射验证。静态/动态界限见主报告。
- **TODO**：V32-THUMB-08短音提前定位/接触shaping和跨曲多seed验证；V32-START-02子步奖励盲区；V32-CLEARANCE-01连续动力安全；V32-07/04实体接触/驱动标定。论文新设计仅提出方案待选择。新manifest散列不同，旧检查点用归档，未放松task恢复检查。

[完整报告](training_controls_thumb_review_20260928.md)。

- **双指端点补充**：CPU 12起点11轴静态IK找到当前47动作原限位内L1:C4＋L5:C3切触全行程键顶端点，XY误差<3.1e−10m／顶面误差<3.2e−9m，静态左相关接触0、闲指净空通过；与父origin/geom尺寸/ranges/gains逐项差0。只证明预设全按下键位置的几何端点，不证明200ms动力触键；候选未写入起姿。完整候选与关节角见thumb_options_audit_20260928/thumb_pinky_static_ik.json。
- **最终入口确认**：真实run.sh inspect exit0，两控制False；reward JSON self enabled/.2、false enabled/1、humanoid enabled/1且mode geometry_clearance_v1。旧快捷字段仍0仅作历史原始参数，不用于实际奖励判断。8归档JSON SHA全部复验一致。

- **源配方最终回归**：make_recipe支持保存的canonical thumb_target字段，与非默认平铺选项冲突时拒绝，六个解包environment的诊断脚本继续兼容；标准库隔离/源构建最后20 tests＋16 subtests通过，最终run.sh inspect再次exit0。


## 2026-09-28：三项独立训练对照、恢复3e-4及错键漏采修复

- **任务/授权**：用户明确要求提前定位、分阶段指法、子步误触惩罚分别参数化并开训比较；三个学习率恢复3e-4。遵循 pianist-v2-change-log，沿用用户已批准工程设计，无重复确认。当前默认子步修复开，另两项关；四组对照各约1M样本。
- **问题**：20Hz端点遗漏200Hz瞬态错键/能耗/掌部接触；旧1cm指法容差可悬停满分；未来目标缺少逐指分配观测；Actor还停留在前轮3e-5。测试发现归一化键程反算激活阈值可能浮点翻转，改为直接原始qpos判断。首次补测误写测试文件名，无测试执行，已更正重跑。
- **参考**：本地 task/task_logic/scene/rewards/reward_config/source_runtime、train/evaluate/checkpoint、原指法与键几何、此前200Hz离线审计、论文笔记p.8控制/pp.10–12奖励/pp.12–13评价。未导入外部参考代码。
- **修改**：新增 anticipation/fingering_experiments/performance_options/substep_rewards；task增加未来50维可选观测及真实键面高度，奖励仍按当前执行谱帧，下一观测用下一谱帧。奖励配置可选performance字段严格保存，旧缺字段不扩写。子步按键无错部分/false_press/能耗均值，forearm接触并集；既有自碰撞不重复，humanoid小预算不加权。CLI/run头包含独立开关及关闭方式；run三LR均3e-4。新增顺序四组对照脚本、源归档/SHA及源码漂移拒绝，区分合格best/观察峰值/final；评估保留20Hz F1并另报200Hz分阶段误触和micro-F1。
- **验证**：集中76 tests＋54 subtests通过；原始阈值修复后补测17 tests＋9 subtests通过；另CLI/旧质量控制21tests通过（组间有重叠不相加）。真实GPU验证graph/eager采样、仅reset指定世界、未来用指时间对齐、关闭功能原奖励/观察相同、阈值边界。三项组合512环境短训24步=12288transitions退出0，实际学习器更新及final评估完成，子步诊断覆盖100%；这不代表收敛。
- **正式实验**：training_mjlab/outputs/reward_comparison_20260928，四组baseline/anticipatory/staged/substep，各1954向量步=1000448transitions、seed42、从头训练。源码69MB归档随manifest；启动后不变更锁定源码/配置。比分与结束状态见comparison.json/comparison.md及本轮报告后续实测章节。
- **TODO/限制**：V32-QUALITY-01的0.99/少碰撞未由短训验证；V32-START-02只修复错键/能耗/掌部，目标键正奖励/指法/几何humanoid仍20Hz端点；M6子步合同未实现，明确拒绝而不静默退化。V32-REWARD-ABLATION-01需正式比分和后续多seed/完整预算；单次短预算不能确定长期优胜。

- **正式结果完成**：四组均1000448样本/248862更新，训练子进程全部returncode0，未提前停止；约45分钟完成。best/final音乐F1：baseline .556329/.500844、anticipatory .601055/.601055、staged .363080/.363080、substep .578692/.578692。final错键子步均值107.67/58.67/73/19；手碰撞最大138/49/175/118；final键底全0但不代表全训练期零。分阶段准备误触37子步，当前候选不推荐作为从头默认。
- **最终验证/交付**：258项合同审计全通过，三项组合短训检查点经eval.sh从源码+完整config重建、整曲GPU评估退出0，子步覆盖100%。保存manifest、4组fullconfig/best/final、source.tar.gz、comparison表/曲线、分析与合同审计JSON；输出目录内两个分析脚本不修改锁定训练源。源码SHA见主报告。结果汇总脚本重跑退出0。
- **运行记录补充**：外层工具会话在结束后返回143；四训练子进程均记录退出0，runner已写Comparison complete，四组summary/final齐全且无训练进程残留，审计确认预算完整，未因此重跑训练。外层会话回收原因未定位。一次辅助JSON读取误将list按dict读取，已更正，无文件/训练影响。
- **结论及剩余TODO**：保留默认substep修复，ahead/staged仍默认关；下一步优先测试ahead+substep组合及更长预算，不预先宣称组合收益。分阶段候选需重新考虑逐步引入，未自动修改奖励/重启长训。首轮对照已完成，V32-REWARD-ABLATION-01剩多seed/完整预算；V32-QUALITY-01的.99及零碰撞仍未达到。末200k均值已记录，避免只挑final或best。

[完整方案与实测](reward_comparison_20260928.md)。


## 2026-09-28：TsenseI_A3.2 替换、node4 核验与提前定位新训练

### 问题与参考

- 用户最新附件 TsenseI_A3.2__hand_withid.zip（SHA a86ee434844f4c37f84a61531cc78a4faeea23abd9920566996c915b469aeb53）；参考厂家 URDF/STL、旧活动模型备份、prepare_models.py、urdf_model.py、generate_collision_proxies.py、forearm_collision.py、source_runtime.py、run.sh 和提前定位实现，以及论文决策登记。外部参考目录未复制或导入。
- 最初把 node4 当另一台主机导致无效 SSH 认证排查；网卡及上传目录确认工作区本就在 node4/FC500T-K，无需重复 SCP。上传原件29/30逐字节相同，唯一URDF仅44处mesh相对路径替换，模型数据相同。
- 新附件并非纯外观：质量171.363→156.939g，关节坐标/轴变化。新小指已语义命名，旧连锁rename会错位；圆尖旧最高平面宽度为0；base单独毫米漏scale导致GPU NaN；新base/palm部件角色互换，旧box造成16mm初始穿透。

### 修改

- 原件解包为根TsenseI_A3.2；独立包全量替换source/leftmesh，正确镜像右手，输出TsenseI_left/right.urdf，更新当前模型加载/预览/README/项目描述及测试。旧V2/V3与历史奖励来源名称保留。旧活动资产备份为model_backups/before_TsenseI_A3.2_20260928.tar.gz；旧根daxian_v3.2归档为model_backups/legacy_vendor_daxian_v3.2。
- 转换器识别新旧小指schema；仅派生base几何补.001缩放及尺寸检查/来源SHA；没有缩放其他指段、质量或关节坐标。清理旧派生OBJ以免加载旧几何。
- Shadow碰撞继续每手21primitive+1mesh，按真实部件采用ID掌壳mesh/安装块box；同步forearm名单和既有装配排除。圆指尖顶部1mm薄带测宽并记录近似局限。无关节机械限位/学习率/奖励公式改动。
- 新run_tsensei_a32.sh显式提前.15s，通用run.sh仍默认0；子步奖励开、阶段奖励关、三LR3e-4、512环境/5M其余配置保留；新1231/47合同从头训。详细机制、姿态35.59mm差异及局限见docs/tsensei_a32_training_20260928.md。

### 实际验证

- 8项模型转换/网格尺寸/碰撞测试、5项Shadow合同通过。完整环境16pass/3个既有mesh-only失败，与旧模型基线一致；FK测试现归一化vendor非单位axis，小指轴按原件推导，没有改模型轴来过测试。
- 修复前GPU smoke以Nonfinite task state中止；未绕过检查。修复后reset接触0、CPU固定目标400子步全状态有限（峰值qvel .09995、最大qpos差.000623）。
- 最终512环境smoke完成12,288transition/1,822update/178帧评估，1231观测/47动作，随后eval.sh通过源码+完整config重建回放。短测不代表F1或全运动域安全。
- 2026-09-28 10:57:54 已在node4启动正式5,000,192样本训练，启动PID 4022146；输出父目录 training_mjlab/outputs/TsenseI_A3.2_anticipatory_20260928；console.log、launch.json、start.sh、源码归档和run子目录完整config/checkpoint留档。source.tar.gz SHA 0aea1d70f4b44e11c16b52c675f9134cf791b2784fb03e0a55ccdce13ee17106。

### 未解决 TODO

- V32-10 厂家真实右手；V32-04 驱動/质量惯量实测；V32-06/07 曲面接触与凸掌壳近似/指垫标定；长期学习F1.99、手碰撞与卡键仍需本轮真实评估。新模型非零swing方向与旧模型不同，旧权重不能直接使用；保留角数值不等于空间姿态不变。

启动后核验：入口 PID 4022146／GPU learner PID 4022415 存活，输出子目录 `run-20260928-105754-540954`；截至 2026-09-28T11:00:00.135434+08:00 已采样 153,600 条、37,150 次更新。实际 config 为 1231 观测／47 动作、提前 .15s／substep true／staged false，三学习率均3e-4；全部运行源码散列与保存配方一致。此为启动状态，不是最终分数。


## 2026-09-28：当前训练、best.pt 与 F1 曲线检查（11:47 快照）

- 参考：本轮run的metrics.jsonl、best.json、best.pt、config.json，key_contact_diagnostics.py，以及论文决策记录；使用pianist-v2-change-log技能。
- 工作：在 `training_mjlab/outputs/TsenseI_A3.2_anticipatory_20260928/review_20260928` 保存一致性检查后的best快照、metrics快照、CSV、PNG/SVG曲线、summary.json及report.md。仅新增分析产物，没有更改训练源码、参数或停止进程；未另起GPU回放。
- 验证：11:47:23采样4,659,200/5,000,192（93.18%），91个评估点；best step8600音乐最差F1=.970404、平均=.972514、200Hz音乐microF1=.935029，最新step9000最差=.964497。best CPU可加载，Actor/Critic/target张量全部有限，步数及1231/47合同一致。图表人工查看完成。
- 结论：后段提升放缓，端点失分主要为漏按；best每回合17–19个漏按键帧、21–24个短暂错键子步、3个自碰撞控制帧。不是17–19个音符，不能由聚合数据定位具体手指或认定爆冲已完全消失。
- 新TODO V32-A32-DIAG-01：新掌壳base_link被key_contact_diagnostics按名字筛选漏掉；当前键底0只覆盖已列手指/安装块，不能代表全手。F1采样不受此名单影响，物理接触未移除。此次仅记录，后续修复需要同步新导出合同及回放审计，不热改正在运行的训练。


## 2026-09-28：导出最新 best.pt 视频（第9400步）

- 参考：本轮best.pt/best.json/config.json、训练source.tar.gz、eval.sh、eval_from_source.py及render_mjlab_checkpoint.py。
- 问题：当前包目录新增TsenseI_V3.2_hand_withid_fixed.zip，严格源码校验拒绝回放。未移动/删除用户文件或放宽校验；在临时目录解包训练源码，复用既有两venv和audio_runtime。第一次归档回放缺音色库，补已有音频运行目录链接后成功。
- 产物：preview/outputs/TsenseI_A3.2_best_20260928_114948保存一致性核验的best/config副本、来源SHA、导出日志；video/replay.mp4、report.json、rollout.npz及实际按键音频。没有修改训练源码/参数或打断训练。
- 验证：导出退出0，ffprobe确认1280×720 H264＋AAC、20fps、199帧、9.95秒。完整178策略帧含20准备帧，另有显示停留。best第9400步保存音乐最低F1 .9788429；本次单次GPU回放音乐F1 .9742013，视频全段F1 .9771003；准备误触0，回放分数不冒充保存的3次评估分数。
- TODO：V32-A32-DIAG-01掌壳键底诊断遗漏仍保留，本次仅视频导出。


## 2026-09-28：最新训练时序审计与 SwanLab 指标上传

### 问题、范围与参考

- 用户要求定位迟按/早松、提出 F1 改进方案，并确认使用 SwanLab 上传 F1、recall、precision 和字母序奖励。使用 pianist-v2-change-log 技能；本轮不改奖励/SAC/模型/频率，不启动新训练。
- 参考本轮 10:57 训练的 config/metrics/summary/best/source.tar.gz、11:49 best 回放 rollout/audio_substeps、归档乐谱/library/NoteTrajectory，以及 rewards/task_logic/fingering_experiments/anticipation/train 源码、论文事实/决策笔记。外部参考为 SwanLab 官方 Settings/log/API 文档，链接见内部上传说明。
- 关键发现：20 Hz 末端正确键正奖励与 200 Hz 错键惩罚时间口径不对称；阈值以下浅按仍能获高连续分。不能把量化目标末帧缺失都称为比原 MIDI 早松。提出正奖励子步均值/激活占用率候选，尚未应用新的论文重合设计。
- 集成中实际修复：相同step多条JSONL会造成覆盖；处理并发半行/尾行；相对输出目录必须按learner工作目录解析；去重须同时检查workspace/配置/日志内容；SwanLab离线Run.url属性会抛异常，改为仅在线访问。上传器失败不能覆盖learner的退出状态。

### 修改与产物

- scripts/swanlab_metrics.py：纯标准库指标适配、稳定数字顺序、同step合并、严格公开配置白名单。主F1/recall/precision保持同一全回合逐帧统计；音乐段和micro口径另列；奖励取真实加权值，不重复乘权重，不补缺失零。
- scripts/swanlab_upload.py：独立跟随/补传进程，固定官方域名、私有项目、关闭SDK源码/Git/机器/终端探针，凭据只从环境或仓库外文件读取；状态/散列/URL留档。上传失败保留原JSONL供重试，原训练退出状态保持。
- run.sh、scripts/train_from_source.py：包装脚本显式开启SwanLab，直接Python默认关闭，新增参数说明完整写入文件头；上传参数不进入模型配方/learner CLI/checkpoint。setup_swanlab.sh和monitoring/requirements.txt使用独立venv，安装SDK0.10.1，不改两训练环境依赖。
- tests/test_swanlab_metrics.py、test_swanlab_upload.py：新增有意义的指标/并发读取/进程隔离/入口合同测试。
- docs/swanlab_training_metrics_20260928.md、docs/tsensei_a32_timing_reward_review_20260928.md、README新增配置和分析说明。run内analysis_timing_20260928保存完整99次评估曲线、逐音20/200Hz/MIDI双基准CSV、报告、重现脚本；保留旧11:47快照。

### 实测结果与验证

- 完成训练5,000,192样本/53.35分钟；best9400音乐F1 mean=min .978843；final mean .948955/min .946775，总回报反而略升。独立best回放.974201：18个漏键帧=14末帧+2迟按+短D5/R2整音2帧；200Hz漏采328中261迟按、47尾部、20整音。相对原MIDI只有3音提前释放，命中音起音延迟中位21.667ms，最明显约64ms。
- 归档重建全部目标/指法匹配录像；1780个物理样本每10个末端与178个控制帧激活逐项相同。PNG图表人工检查完成。候选效果不是实测收益，不把离线oracle估算当训练已到.99。
- python3 -m unittest discover -s tests -p 'test_swanlab*.py' -q：29项通过（10指标+19上传/入口），bash -n和入口--help通过。完整1129条历史通过隔离SDK离线端到端测试，1022个不同transition提交。
- 正式补传成功：https://swanlab.cn/@blackswan17/TsenseI_A3.2/runs/qgbeil7c 。云端回读FINISHED/PRIVATE、137条序列；11条主曲线+2条音乐曲线逐点对照通过，99评估点/976奖励点，最大误差约1.12e-16。凭据文件权限0600、在仓库外，无源码明文key。
- 未运行带新上传器的长期RL吞吐测试；独立进程避免在learner循环阻塞网络，不承诺零资源开销。新奖励候选、模型或学习参数没有变更。

### 未解决 TODO

- V32-TIMING-01：原目标键预算内200Hz连续/激活奖励候选待用户决定与受控短测；从best分支保留三个LR3e-4。
- V32-TIMING-02：MIDI与20Hz评分网格边界不同，保留双基准，不通过平移目标/放宽阈值提高分数。
- V32-TIMING-03：短D5整音漏按还缺逐子步请求/实际手姿/接触记录，未把腕速统计当根因。
- V32-A32-DIAG-01：掌壳base_link漏入键底诊断仍保留；不能用现有键底0声称全手不卡键。V32-QUALITY-01的.99与零碰撞目标仍未达成。


## 2026-09-28：修复正确键子步与触发奖励，补充F1根因审计

### 用户要求、问题与参考

- 用户明确要求修复“正确按键只看20Hz末端、未触发白键仍获94%连续分”，并找出其他F1问题。已说明论文pp.10–12式13–14与工程子步/真实激活混合差异，本轮用户请求是该修复的决定，不重复询问；使用pianist-v2-change-log技能。
- 参考：已完成A3.2 10:57训练config/summary/best/source归档、best视频rollout/audio_substeps、上轮逐音和奖励审计；mjlab_env的task/scene/reward_config/rewards/performance_options/substep_rewards/key_contact_diagnostics，rl_torch train/checkpoint/replay/SAC，music midi_file/piano_roll、fingering_experiments/anticipation，以及论文决策登记。没有复制external参考目录代码。
- 发现的实施风险：通用子步开关关闭时也需为新正确键开关采样；旧JSON不能补新字段造成合同漂移；run.sh新默认会污染旧三项单因素对照；新奖励进入下一步reward观测，不能保证旧actor零更新F1不变。

### 修改

- 新performance参数target_key_substep_reward(bool)、target_key_activation_weight([0,1])；缺字段runtime False/0，旧三个字段canonical序列化保持。配置拒绝非有限/非法权重及无有效按键奖励。
- rewards.target_key_positive_reward使用原高斯连续键程与物理实际激活混合，对目标键和物理子步取均值；当前run.sh显式True/.5。保留legacy .5/paper .6外层预算、原错键、权重/dt/和弦选项；空谱正项0，旧关闭路径不重算。task独立请求已有采样缓冲，未增加物理step。
- rl_torch/train.py、run.sh头部及实参、SwanLab参数白名单贯通；原三项run_reward_comparison脚本显式关闭新两项，旧对照含义保持。run_tsensei_a32继承新默认，三个LR仍3e-4、其他学习/模型/频率/谱面/观测/腕速不改。
- KeyContactDiagnostics明确补两侧base_link_collision_*掌壳到非末端列表，排除视觉/腕座；只读诊断，无碰撞反力或奖励变化，新contract保存名单；历史掌壳零计数不追认为全覆盖。
- 新test_target_key_reward/test_target_key_cli，扩充test_key_contact_diagnostics/test_performance_task_integration/test_reward_comparison_entry。
- 新scripts/audit_target_key_reward.py及recorded_best_audit逐帧CSV/summary；补充审计docs/f1_additional_audit_20260928.md与outputs/f1_additional_audit_20260928中的CPU复核脚本/JSON。
- 新docs/target_key_reward_fix_20260928.md，README、两层AGENTS和论文决策记录同步本轮已决定范围；未擅改其他候选设计。

### 实际验证与证据

- 12组核心/历史兼容/CLI/上传/碰撞/检查点回归共103项通过；真实GPU环境3项通过，包括关闭通用子步/诊断但正确键采样仍生效、独立NumPy公式比对、局部reset隔离。另有28项CPU乐谱/任务/SAC/replay/指法审计测试通过。日志保存在outputs/target_key_reward_fix_20260928。
- 原best相同轨迹178帧/1780子步中，149有目标音乐帧的r_on均值：旧末端.979282、子步连续.932537、子步混合.925273；实际激活占用率.918009。末端激活与视频逐项相同。这是固定轨迹重新计分，不声称新策略F1提高。
- 真实512环境从头短验收：smoke-20260928-132827-668385，102400样本/24350updates/512完整回合/0截断，训练循环69.54s；1231/47、20/200Hz、anticipation .15、三LR3e-4和True/.5落盘。所有网络张量有限，最终2/2评估完整，掌壳名单实际生效。短初始化策略音乐F1 .06899仅验收流程，不与5M best作效果比较。显式关闭SwanLab上传内部短测，正式run.sh上传默认保持。
- 新短测scene/task SHA、准备/控制限制/净空合同与原best一致，包源码与本次config逐文件散列一致；奖励变更仍要求显式迁移。原best/视频/日志未改。源归档SHA256 558f5aa4ff75c555f60f401c94907036ba4e442b521f89c35ab6b23e55411a7e，282文件。
- CPU实际/合成审计：4999随机样本全在前.5s准备期，首个音乐样本前1310更新；未来指法分母影响45/149有音帧，当前份额最低1/3；连续50ms重复音可被量化后的分音规则整音删除，本曲短D5目标未删，后续D5 onset帧135被移除；曲末C5无释放尾段。未发现末帧/reset/replay污染或需改既有discount的错误。

### 未完成与适用范围

- V32-TIMING-01：两项奖励修复完成代码/公式/流程验收；音乐F1收益/.99仍待受控续训和其他曲目测试。只短验收，没有新5M正式训练。
- V32-A32-DIAG-01：当前掌壳名单遗漏已修复；旧best全手物理卡底情况仍须重新采集，历史统计不可升级为新覆盖范围。
- V32-F1-AUDIT-01/02：整曲replay预填、固定当前/未来指法预算需各自对照，未改4999/5000预算或现指法公式。
- V32-F1-AUDIT-03/04/05：重复短音输入覆盖与评分合同、尾音释放验收、速度/当前指键观测候选保留，未更改原乐谱/F1来抬分。
- V32-TIMING-02/03：MIDI/量化时间差与短D5逐子步接触因果保留；不能仅凭腕限速统计下根因结论。


## 2026-09-28：取消强制论文比对与论文配置基准

- **用户要求**：以后不按论文配置作为训练依据，删除论文比对需求。
- **问题**：根和独立项目AGENTS要求修改前读论文、逐项报告页码/差异并等待对齐决定；README、迁移计划、验收及旧分析文档仍有主动门槛或把历史论文参数称作当前默认。日志技能本身没有该要求。
- **参考**：两级AGENTS、独立README、references/biomimetics_2026_610_notes.md，dual_a100_sac_plan/sac_shadow_review/torch_sac_acceptance/mjlab_task_acceptance/common_commands及近期F1/SwanLab报告；本机pianist-v2-change-log技能。检查TJ_forearm说明，无相关门槛。
- **修改**：删除两级AGENTS的论文对照约定，改为以当前源码、显式启动配置、用户目标和F1/碰撞/吞吐实测为依据；取消强制阅读/公式比对/逐项登记及仅因论文重合产生的确认流程。保留模型、坐标、未知映射和检查点兼容约束。论文笔记移除主动提示流程，历史对齐表标为归档参考。同步README、四份入口/验收说明、common_commands的当前run.sh入口、近期奖励/SwanLab/F1说明及四份旧报告中的面向未来门槛句子。
- **历史保留**：论文原文、历史实验数值/配置/检查点与修改日志留档；旧paper preset代码保留兼容用途，不自动用于后续新训练。没有借此次撤销流程改学习率、奖励、控制时序或模型。
- **验证**：独立子任务复查主动要求；定向rg确认旧强制门槛措辞已清除（历史日志保留）。282个训练包/脚本/配置/启动入口文件逐项SHA256前后完全一致，证明运行行为和参数未改。本次为文档规则修改，未运行训练或新增测试。
- **未解决事项**：无新增TODO；既有F1与模型验收任务仍按实际工程问题管理，不再以论文对齐为前置条件。
## 2026-09-28：动作映射统一为 Direct，四指 MCP 上限改为 1.3 rad

- **用户要求与问题**：用户先要求双手食／中／无名／小指 MCP 目标 `[0,1.2]→[0,1.3] rad`，随后要求查清 Shadow 与 Direct 的实际映射，并指定以后采用 Direct，不保留旧动作映射回放。核查发现原 `run.sh` 传 `--action-profile shadow_v1`，但 `physical_bounds()` 又从 `direct_joint_v1.json` 读取物理目标，导致名称与数据来源混淆；原映射测试还依赖已不存在的旧冻结场景目录。
- **参考**：`run.sh`、`scripts/train_from_source.py`、`tsensei_v3_2_pianist/mjlab_env/{action_mapping,scene,source_runtime}.py`、`tsensei_v3_2_pianist/rl_torch/{train,checkpoint}.py`、`configs/actions/direct_joint_v1.json`、当前 A3.2 10:57 训练的 `config.json`、`tests/test_action_mapping.py`、A3.2 URDF 与本地修改日志技能。保存的旧 config 显示 `action_profile=shadow_v1`、八 MCP 目标上限 1.2，属于旧训练事实。
- **修改**：`run.sh` 显式改用 `direct_joint_v1`，任务／驱动的 `shadow_v1` 和 `shadow_primitive_v1` 碰撞参数保持各自含义；`configs/actions/direct_joint_v1.json` 中八 MCP 上限统一改为 1.3，策略范围仍 `[0,2]`，中点目标 0.65。`action_mapping.py` 删除 `shadow_v1` 动作选项，未保留临时 Shadow 兼容配置；`README.md` 和 `docs/action_mapping_direct_joint_v1_20260924.md` 改为当前数值及清楚的参数角色。其余关节目标、机械硬限位与 20/200 Hz 未改。
- **配置与检查点校验**：`source_runtime.source_files()` 将 `configs/actions/*.json` 加入源码指纹，归档回放按该文件校验；训练在加载父检查点前解析并比较完整 `action_contract`，拒绝同名 Direct 但不同目标范围的旧权重迁移。`scripts/launch_quality_training.py` 使用 Direct 并拒绝非 Direct 父检查点；相关测试／CPU-GPU 对照脚本改用当前动作映射。历史运行文件与旧归档未修改；用户不要求用当前源码兼容旧回放。
- **验证**：`bash run.sh --inspect-only --no-progress` 实际输出 `action_profile=direct_joint_v1`；源码 `--build-only` 成功。动作／归档／启动测试 27 通过，检查点／训练质量测试 20 通过及 33 子测试，源码构建／对照入口测试 13 通过及 7 子测试；当前模型 GPU 动作端点测试 1 通过，GPU 任务／子步／几何集成 7 通过。`bash -n`、JSON 解析和 Python 编译通过。静态核对 47 动作、八 MCP `[0,2]→[0,1.3]`。未运行长训，F1／碰撞收益尚无实测结论。
- **未解决 TODO**：无新增 TODO；旧 V32-04/05/07 的电机与接触参数标定仍在原记录中。旧 `shadow_v1` 动作检查点不能用当前入口直接续训／回放，符合用户本轮决定。

## 2026-09-28：F1 证据修复与启动参数一致性复核

- **授权／范围**：用户由新建任务改为在当前任务修复此前有确定证据的 F1 问题，并复核其已尝试修复的 Shadow／Direct 不匹配；采用当前源码和实测，不恢复已取消的论文审批流程。
- **问题与参考**：参考 `docs/f1_additional_audit_20260928.md`、旧 A3.2 配置／回放、`run.sh`、Direct JSON、action_mapping/source_runtime、train/checkpoint/evaluate、fingering_experiments、midi_file/TaskBundle、SwanLab 上传及本机日志技能。确认早期 replay 只覆盖准备、提前定位共用分母、极短重复音可能静默消失、缺曲末释放。用户的动作 Direct／四指 MCP1.3 和动作 JSON 入源码指纹已正确实现；剩余 task/drive Shadow 属于不同设置，未改回旧动作语义。
- **修改**：`run.sh` 显式 `--replay-prefill-episodes 1 --anticipatory-current-priority --score-validation strict_v1 --release-tail-time .25` 并补齐文件头。train 按实际183帧预填93696条、保留4999随机动作，之后 .25 更新／transition，不追补预填更新；保存请求／实际学习起点。performance/fingering/rewards 采用有界当前优先公式 C+(1-C)*.2*F（F≤.5）；旧字段缺项保持历史公式。score_validation、midi_file、源码入口和TaskBundle严格报告不能表达的音符，拒绝静默删音，不改当前原始目标。尾段5帧真实进入物理／奖励／replay；train和视频均将尾段与音乐分开，目标达成检查最终松键。
- **合同／参数修复**：`scene.action_contract` 与 `checkpoint.load_agent()`、评估／视频／预览／审计／性能分析入口补齐完整物理范围校验；训练侧使用同一实际合同。修复 SwanLab 驱动／碰撞枚举漂移、补充预填／释放／奖励开关和实际指标。历史三项对照显式关闭本轮新默认，避免实验定义漂移；旧冻结场景测试替换为当前源码临时构建。训练模型、机械限位、三LR3e-4、20/200Hz、腕限速、几何保护关闭及原乐谱指法保持；旧检查点未回写。
- **集成时遇到的问题**：谱面 helper 最初放在 music 包触发 note_seq 的 Composer 专属依赖，已移动为无依赖的顶层模块，GPU不额外安装 Composer 依赖；旧最小测试fixture缺control_dt，按真实历史导出补齐。首个验收命令的 eval-interval0 与已有目标早停互斥被正确拒绝，改为240；测试调用误写模块名后纠正，未为通过测试放宽合同。
- **验证**：实际 Bash argv 71项，经两级真实解析器0未知、0文件头遗漏；真实GPU八MCP 0/.65/1.3端点测试2项通过。130项集中回归通过、3项GPU任务集成通过、16项不同谱面／源码检查通过，另外历史准备与归档测试通过（批次有重叠，不累加）。当前34/34音覆盖；尾段前178帧goal/fingers逐元素不变。512环境×240步=122880样本短训完成512回合、0截断、7296更新、4999随机，37.334秒；第180步仍0更新、第190步896更新；所有日志数值有限，200Hz采样覆盖1。独立进程严格重建加载 final.pt 成功，音乐F1=.1242616、尾段5帧、最终激活键0。仅执行功能短训，未启动新的5M，低F1不作为收敛收益证据。
- **文档／留档**：更新README、旧审计状态和 `docs/f1_evidence_repairs_20260928.md`；谱面专题 `docs/score_integrity_release_tail_20260928.md`。输出 `training_mjlab/outputs/f1_integrity_fix_20260928/acceptance.json`、完整config/checkpoint/score_coverage/独立评估和283文件source.tar.gz；归档SHA256为aebca0112aee4b5e23a19a86f7e6b69dd4524168446bc97efb503cfa69b071a1，与新训练源码指纹逐项一致。无持久冻结scene；验收显式no-swanlab，正式默认上传保持。
- **未解决TODO**：V32-F1-AUDIT-01/02的训练收益须正式对照；03已阻断静默丢音，但20Hz无法表达的MIDI仍需提高频率或显式修谱；04释放训练已接入、学习效果待实测；V32-QUALITY-01的F1.99／低碰撞尚未达成。无速度观测等候选因素未证实为根因，本轮不扩观测或擅改终止discount。

## 2026-09-28：启动 F1 修复后的 A3.2 正式5M训练

- **用户要求**：开启一次训练；采用上一任务已验收的新配置从头启动小星星。
- **问题／参考**：无新增启动阻碍。读取两级AGENTS、pianist-v2-change-log技能、run.sh/run_tsensei_a32.sh、source_config与上一轮成功smoke，以及SwanLab上传入口；独立子任务只读核对源码／模型／Direct／学习器散列和奖励控制JSON无漂移，GPU无现有训练竞争。
- **操作**：北京时间2026-09-28T17:39:16.659945+08:00，独立后台session PID417273运行run_tsensei_a32.sh，新目录training_mjlab/outputs/TsenseI_A3.2_f1fix_20260928-173915/run-20260928-173916-689809。512×9766=5000192样本；保留提前.15s、Direct1.3、三LR3e-4、20/200Hz、1秒准备+2.5cm抬高、一轮预填、当前指法优先、正确键子步/.5触发与.25秒松键，SwanLab默认启用。没有修改训练源码或默认参数，没有加载/覆盖旧checkpoint。
- **留档／验证**：本轮source.tar.gz、独立入口副本、launch.json、完整config已写；source_runtime散列与归档逐项匹配，实际1231观测/47动作、预填93696条、第93697条开始学习已核验。SwanLab已创建run otw79aa2；初期预填更新0符合合同。随后观察到400向量步/204800样本/27776更新，损失有限，GPU进程417609存活；SwanLab uploading且25组指标已提交。启动状态详见docs/training_f1fix_launch_20260928.md。
- **未解决TODO**：V32-QUALITY-01、V32-F1-AUDIT-01/02/04效果待本轮完成评估；本条仅记录启动，不宣称F1.99、零碰撞或训练完成。


## 2026-09-28：A3.2 火山云双卡迁移

- **任务／范围**：用户要求只迁移当前 A3.2 源码和运行依赖、排除现有训练产物与其他手，并使用两卡训练及提供 Codex 连接方法。按当前 run.sh／配置实施，不改模型、乐谱、奖励、学习率、batch、更新比例或总学习预算。
- **服务器／实际问题**：公开文档对应 A100，但用户首次指定101.47.9.106实为H100且8卡全占用；用户明确排队时改A100，最终部署115.191.75.48（TenseiNode1、houj、research）。两服务器均依管理员要求和用户指定完成首次改密，密码不入文件。H100安装已停止、未提交GPU作业；A100的Slurm931实际取得两张80GB A100及NV12互联。实际A100无nvcc，驱动550.144.03。PyPI个别路径超时改用文档建议的TUNA镜像，锁定版本保持；uv拒绝MJLab原无扩展URL，改同提交archive.tar.gz，并提供严格SHA校验的离线wheel路径以解决慢下载。
- **参考**：两级AGENTS、本机pianist-v2-change-log技能；run_tsensei_a32.sh/run.sh、source_runtime、train/SAC/replay/CUDALearner/checkpoint/eval；requirements及旧dual_a100_sac_plan。外部为 https://agimani.github.io/hpc_doc/ 的access/basis/slurm/gpu-training/uv、OpenAI官方SSH/CLI文档，以及NVIDIA官方CUDA12.8.1 redistrib与兼容包（链接见迁移／环境说明）。
- **实现**：scripts/hpc/package_a32.py确定性打包和逐文件SHA，排除旧手/M6/旧模型zip/训练产物/venv/密钥，仅保留独立A3.2；setup_envs.sh安装Composer与Torch两环境，setup_cuda.sh用户目录安装带SHA校验的nvcc/cudart/cccl及必要wheel头文件，可显式启用用户态兼容库，不改系统驱动；probe_gpus.py仅Slurm内检测两卡/真实Warp/CUDA graph/P2P。submit.sh/job.sbatch申请2GPU+16CPU+128G，保存源码并在独立解包快照内运行；benchmark.py等预算单卡/双卡对照、排除初次编译的稳态区间、计数和配置验证及独立checkpoint重建加载。
- **双卡实现**：train.py新增可选--learner-device/--collector-pipeline及collector_pipeline.py，GPU0采样Actor/物理，GPU1 SAC/replay；最多1批策略滞后，有真实采样线程及commit边界，首次编译独占，评估/保存/结束排空。保持512环境、batch1024、.25更新/transition、5,000,192样本；调度变化明确写config，不声称逐步等价。保留原单卡默认、同步分卡对照；记录两卡显存、Actor版本和主机重叠时间。eval_from_source.py仅增加动态流水状态白名单，静态布局和其余合同继续严查。
- **迁移证据**：初次上传源码包475文件／12,743,908字节，SHA256 96cf17516fefa5fc3fade1dc14c16cb26c0b6bd9a9eb3c391d521e2275f2a507，后续工具修复通过明确文件同步。离线MJLab1.6.0 wheel SHA256 ccb22b92162e67991f104c6499626f681b43afb91b1cb98b2bae7ce73f420f32，与固定提交8ee51fbc...原tar的258个运行文件逐字节相同；原tar SHA256 824d91fbb3efe0fcc3c34f244f698df6a51d2c95e4efc1d9fca94940976f8a8f。未复制旧训练权重、replay或视频。Codex CLI0.158.0已在A100安装且登录shell可用；初次源码迁移未包含认证，20:14后续按用户明确授权单独复制登录缓存（见下条）。
- **本地验证**：打包5项测试、环境脚本17项CPU测试；解包后仅链接双venv的--build-only成功，去M6/旧zip后可独立构建，重复包字节一致。主任务最终合并回归56通过/2真双卡跳过；实现方流水/预算回归60通过/2跳过/48子测试，eval修复39通过/2跳过（各批有重叠，不相加）。私有CUDA12.8.93实装、重入及真实Adam翻译单元sm80离线编译通过（31.09秒），不是GPU执行验收。Slurm提交argv/含空格路径、错误mode及无allocation拒绝验证通过。
- **已完成远端基础验收**：Composer84包、learner133包，两环境pip检查通过；用户nvcc12.8.93和EGL dispatcher安装成功。Slurm932/934在原生550.144.03驱动通过两卡Torch计算、CUDA graph、Warp真实内核/PyTorch互操作和P2P双向约199GB/s短测，未启用兼容库。缺EGL dispatcher已用固定Ubuntu包在用户目录补齐；Slurm内Composer导入和源码构建通过。作业933因遥测UUID缺GPU-前缀退出，补前缀后934遥测正常；实际双GPU流水测试11项通过。
- **完整远端验收**：Slurm934 COMPLETED/0，17分46秒；两轮单卡/双卡每次262144样本、42112更新，四次计数、合同、排空、独立checkpoint重建全部通过。稳态单卡1363.17/1360.72、双卡1744.28/1746.30条/秒，均值提升28.15%。首轮每批采样89.22ms已藏入291.03ms学习，快照/提交约0.76%，未用TF32/AMP或改变学习设置。含冷编译总耗时不作为长期加速。
- **正式训练**：Slurm935在afterok:934完成后19:11自动启动，路径training_mjlab/outputs/hpc/job-935-20260928-191156-583628；512环境/9766步/5000192样本、1231观测/47动作、batch1024/.25、三LR3e-4及原完整模型/奖励/控制合同核验。源码包SHA256 c14bb52f0131e1b0a3b832bb1b414e12710d6d0293b20f7bcc9492e296dccb5f。19:14观察到300向量步/153600样本/14976更新，损失有限，两卡同时工作。无旧权重迁移；HPC显式关闭SwanLab在线上传，本地指标正常保存。
- **TODO**：V32-HPC-01/02已完成；V32-HPC-03长期音乐F1/碰撞和达到质量目标的耗时待正式结果。完整证据与连接步骤在docs/volcengine_a32_migration_20260928.md；不把启动或短训提速称为长期质量达标。

- **20:14 Codex代理与凭证后续**：用户报告device-auth 403，实测远端代理变量均空、认证接口明确返回unsupported_country_region_territory；用户随后要求像node4一样走本地代理并直接使用凭证。参考node4现有.profile的17999代理、OpenAI官方auth无头凭证迁移说明。新增node4用户systemd服务volc-a32-proxy和仅转发专用SSH密钥，A100仅127.0.0.1:17999经SSH连接node4同端口，配置保活/断线重连；远端.profile保留备份并追加代理变量。经SSH stdin单独复制已有ChatGPT登录缓存到远端~/.codex/auth.json，0600，未输出凭证内容。远端登录shell代理生效、服务enabled/active、login status已登录，实际Codex最小请求两次均返回CODEX_A32_PROXY_OK且退出0。首次诊断请求直连超时，node4代理下设备码GET返回530；最终以真实已认证模型请求成功为验收。935仍RUNNING，未修改训练进程。迁移说明同步当前登录方式；V32-HPC-PROXY-01保留node4/上游在线依赖及Linger=no下整机重启/无人登录的持续性未验收。

## 2026-09-28：导出本轮正式训练 best.pt 有声视频

- **要求／参考**：用户要求导出当前训练best。读取latest_f1fix_launch.json、本轮summary/config、eval_from_source_archive.py、归档渲染入口和既有日志技能；按准确run定位，不用latest脚本自动选final。
- **操作**：导出第9700步best.pt，训练保存音乐F1=.9002111；使用本轮匹配源码归档和完整config，保留20/200Hz。输出preview/outputs/TsenseI_A3.2_f1fix_best_20260928-210504/video/replay.mp4、rollout、音频事件/MIDI、report与ffprobe；新增docs/best_video_f1fix_20260928.md及输出README，同步启动指针为已完成。没有修改训练源码/参数/检查点或再开训练。
- **验证／问题**：无导出失败。归档170文件和checkpoint散列校验通过，渲染退出0并完整终止，NVIDIA硬件渲染。视频1280×720、20FPS=.05秒/帧、204帧10.2秒，含48kHz AAC真实键触发音频；1830物理子步采样，20准备+158音乐+5松键。预览画面检查通过。独立回放音乐F1=.8981013，与训练评估.9002111分别记录，不宣称逐位复现。
- **TODO**：V32-QUALITY-01继续保留；本轮预算耗尽且未达全部质量条件，本次只导出未额外进行失分研究。

- **Windows SSH连接修复后续**：用户要求直接修改`C:\Users\ONKIALL\.ssh`。确认当前工具在Linux node4，Windows目录未挂载且无跨主机执行工具；服务器密码登录实测正常，原authorized_keys仅有node4代理专用受限公钥。新增scripts/hpc/fix_windows_ssh.cmd（ASCII/CRLF）并放到A100的~/fix-a100.cmd，使用完整系统PowerShell路径解决PATH问题，专用a100key避开粘贴转义，公钥经SSH安装且去重；备份原配置、仅清理目标别名的IdentityFile、临时配置实际BatchMode登录验证后才保存。打包允许.cmd，迁移说明同步。隔离目录实测公钥安装的CRLF处理/幂等/旧条目保留通过，打包5项测试通过；Windows运行条件不具备，V32-HPC-WINSSH-01保留用户本机执行及端到端连接验收。未声称已经改动Windows配置。

## 2026-09-28：整音漏弹与提前松键修复方案

- **用户要求／问题**：根据用户提供的历史退化分析制定修复方案。主要候选机制是当前优先公式用`1-C`抑制未来定位，以及距离指法允许悬停高分；多项同期变化尚不足以做单因果归因。本次范围为方案制定。
- **实际参考**：`AGENTS.md`、`run.sh`、`run_tsensei_a32.sh`、`scripts/hpc/{submit.sh,job.sbatch,benchmark.py}`；`mjlab_env/{fingering_experiments,performance_options,task_logic}.py`、`rl_torch/{train,checkpoint}.py`；`scripts/{run_reward_comparison,audit_checkpoint_trajectories,analyze_thumb_sampling,audit_target_key_reward}.py`；`tests/{test_fingering_experiments,test_f1_training_repairs}.py`；`docs/{f1_evidence_repairs_20260928,reward_comparison_20260928}.md`；A100作业935的config／summary／best及用户本轮分析。可选本机日志技能路径缺失，按AGENTS维护日志。
- **修改位置与原因**：新增`docs/f1_regression_repair_plan_20260928.md`，明确R基线、A仅关闭当前优先、B仅开分阶段、后续AB交互试验；初筛各3,000,320样本，完整预算与多种子确认，补充逐音／早释／奖励系数／200Hz物理验收。明确B也会经C影响未来奖励，旧历史对照脚本会覆盖正确键奖励／预填／尾段，不能直接用于此次消融；3M warm-start再训2M不等同不中断5M。README新增方案链接，并区分935已完成结果与历史本地.9002运行。
- **证据边界**：用户提供的逐音原始回放本次未重跑；已核对现有公式和历史分阶段1M负结果。935实际best均值.926381、最低.925467，作为当前A100基线。没有更改训练源码、奖励JSON或默认值，没有提交GPU作业。
- **验证**：文档3个相对链接可访问；提交模板经`bash -n`静态语法检查，三组命令差异仅为预定布尔开关；预算独立换算122880／7296、3000320／726656及5000192／1226624（样本／更新）。935记录的149个运行源码／模型／动作文件SHA-256全部匹配。此次只做文档及静态核验，未运行训练测试，不将方案视为修复收益已验证。
- **未解决TODO**：TODO(V32-F1-REPAIR-01) 专用计划／指标工具及等预算、多种子、逐音验证待实施；TODO(V32-QUALITY-01) .99与物理质量目标仍待实测。固定份额公式只作条件触发的后续候选，不是已采用配置。

## 2026-09-29：run.sh 目标限速参数

- **用户要求／问题**：把当前TX、TY及其他限速作为run.sh参数，方便启动时调整；此前必须修改控制JSON。
- **实际参考**：`run.sh`、`scripts/train_from_source.py`的参数转发、`rl_torch/train.py`、`mjlab_env/{control_limits,action_mapping}.py`、`configs/control_limits/generic_wrist_rate_20260927.json`和README、`tests/test_run_entry_controls.py`。可选日志技能路径不存在，遵循本项目日志约定。
- **修改位置与原因**：`control_limits.py`新增腕部五轴、双手手指组及逐关节CLI参数和覆盖解析；`train.py`注册参数并将解析结果保存至现有控制合同；`run.sh`参数表及控制配置README说明单位、默认值、优先级和示例。支持`--wrist-tx-max-velocity`、`--wrist-ty-max-velocity`、`--wrist-pitch-max-velocity`等；`--finger-max-velocity`设置40个手指关节；可重复`--joint-max-velocity NAME=VALUE`逐关节覆盖。JSON低于组参数，组参数低于逐关节参数，同名重复最后生效。
- **行为边界**：不传参数时保持当前JSON及TX/TY1.2m/s、Pitch4rad/s；0只删除对应限速并保留位置范围。负数／非有限值／未知关节报错；当前结构移除的TZ/Yaw拒绝正限速。限制执行器目标变化率，不代表实际关节速度上限或硬件标定。无新增默认手指限速，不更改奖励、训练预算或检查点兼容规则；历史检查点继续使用匹配源码归档。
- **验证结果**：实际Bash入口替身捕获＋真实参数解析／有效配置的10项CPU测试通过，含新增4项及多个非法输入子用例，覆盖默认等价、覆盖优先级、全关闭、位置范围保留、输入JSON不变、合同JSON往返和无JSON使用；`bash -n run.sh`通过。未执行GPU验收或提交训练。
- **未解决TODO**：无新增功能TODO；用户自定限速的F1／吞吐／碰撞影响须后续Slurm实测，既有物理和训练质量TODO保留。

## 2026-09-29：run.sh 显式设置双手TX/TY为1m/s

- **用户要求／问题**：将当前左右TX、TY目标限速均设为1m/s，直接写入run.sh。现场脚本末尾还缺失续行符，导致`"$@"`未进入exec命令，追加覆盖参数无法转发。
- **实际参考**：`run.sh`、`mjlab_env/control_limits.py`的组／单关节覆盖顺序、`rl_torch/train.py`参数入口、`configs/control_limits/{generic_wrist_rate_20260927.json,README.md}`、`tests/test_run_entry_controls.py`、README及现有F1修复方案。读取Slurm队列确认当时无运行／排队作业。
- **修改位置与原因**：run.sh命令新增`--wrist-tx-max-velocity 1 --wrist-ty-max-velocity 1`，同时更新参数注释，恢复末尾续行符及参数转发；原控制JSON的1.2历史值保留。现有入口测试的默认预期更新为四个平移通道各1，仍验证末尾覆盖、逐关节优先级和其他控制／奖励。同步README、控制参数说明及F1方案：后续各对照组共同使用1m/s新基线，不把935原1.2配置当作完全相同的对照。
- **验证结果**：`bash -n run.sh`通过；CPU实际Bash入口捕获＋真实配置解析测试10项及12个subtests全部通过。有效配置核对左右TX/TY均1.0m/s、左右Pitch均4.0rad/s；20Hz下平移目标每帧变化上限为0.05m。没有执行GPU验证或提交训练。
- **未解决TODO**：无新增实现TODO。目标限速不等于实际关节速度上限；1m/s对F1／碰撞／吞吐的影响留待Slurm对照，V32-F1-REPAIR-01及V32-QUALITY-01保留。

## 2026-09-30：双卡评估加速与可选四指耦合实验

- **问题与授权**：用户要求优化双卡利用率、HPC默认TX/TY各1m/s并保持提前定位，增加可选MCP控制/PIP-DIP从动并启动F1测试。用户明确确认四指末节朝指尖轴线与琴面约60°、拇指保持原控制；保留PIP/DIP物理关节。
- **实际参考**：`AGENTS.md`、`run.sh`、`run_tsensei_a32.sh`、`scripts/hpc/{submit.sh,job.sbatch,benchmark.py}`、`mjlab_env/{scene,task,task_logic,bundle,action_mapping,control_limits,finger_clearance,source_runtime}.py`、包内V3.2编译模型/关节轴/指尖site、`rl_torch/{train,evaluate,checkpoint,sac,collector_pipeline}.py`、预览/轨迹审计/性能回放入口、旧作业934/935性能记录及`docs/dual_gpu_speed_audit_20260930.md`。本机可选日志技能缺失。
- **执行修改**：HPC显式传TX/TY1、提前定位.15、eval-num-envs3；新增performance/validate入口。评估按独立world并行，保留3回合、100步频率、逐回合/最差质量门槛、全部200Hz诊断；保存评估/检查点计时。性能对照包含单卡、双卡采样学习流水、双独立策略，固定每策略更新预算，后者只比较总实验吞吐。普通run.sh保持单卡。未改变奖励公式、网络、学习率、精度、SAC更新预算。
- **控制修改**：新增finger_coupling.py，按实际编译几何解算有界PIP候选和DIP解析根，目标末节向琴面下倾60°。四指16个输出变从动，47→31策略动作，134物理qpos/46驱动器保留；拇指/SW/腕保留。可选删除16个实际qpos+16个上次物理动作槽，含提前定位1231→1199；实际从动受动力学影响，此删观测可能损失信息。配置、几何、观测和动作合同严格保存/回放，跨合同迁移拒绝。物理功率/奖励仍包括从动关节；兼容腕限速，拒绝从动限速及冲突几何目标投影。角度诊断为20Hz端点，区分当前谱面用指，不能据此宣称每个200Hz击键都达到60°。
- **说明与实验**：run.sh参数注释、README和`docs/{hpc_training_options,finger_coupling_experiment}_20260930.md`同步。计划同seed42/每组3,000,320样本比较47/1231基线、31/1231耦合完整观测、31/1199耦合压缩观测；按现有自动熵规则31动作目标熵-15.5，基线-23.5，完整与压缩耦合两组相同。新增CPU只读`scripts/summarize_coupling_comparison.py`及6项通过的合成文件测试，按清单汇总预算/维度/best/final/末50万/角度/碰撞，未完成不判断优胜，查询一次Slurm而不自动监控。
- **验证进度**：CPU合同/入口/检查点/批量评估59项及65子测试通过，HPC脚本/预算23项通过；耦合几何7项CPU检查通过，含真实镜像非平行轴FK角度误差<0.0006°。Shell语法通过。Slurm1073验收83项及57子测试通过；1076真实31动作/两种观测SAC-replay-checkpoint往返1项及2子测试通过。1075的512环境耦合短训完成131,072样本/9,344更新，诊断全覆盖。1074已完成五模式/六策略性能对照，每策略262,144样本/42,112更新且六次独立回放通过：稳定含评估吞吐单卡顺序896.71、单卡批量1150.96、双卡顺序1040.74、双卡批量1391.34、双独立合计2316.17样本/秒。双卡评估15.434→5.676秒，吞吐+33.69%；相同批量评估双卡对单卡+20.89%；双独立实验总吞吐对流水+66.47%，不等于同模型加速。首组冷构图55.4秒/后续约2.6秒、共享节点及短预算限制明确记录。1077/1078/1079均以0退出，分别用时35:26/35:46/35:25，完整达到3,000,320样本/726,656更新；配置核对共享奖励/控制/源码/SAC/评估一致。1073至1079的151个运行源码/模型/动作文件与当前内容SHA-256完全一致。
- **三组F1结论**：新增`docs/coupling_f1_analysis_20260930.md`并更新机器汇总。基线、完整观测耦合、压缩观测耦合best音乐F1分别.89212/.93115/.92024，final 200Hz分别.86949/.91516/.89993；完整观测耦合的best漏键帧59→43，但final手碰撞10.67→79子步/回合。压缩观测比完整观测best低1.09个百分点且未缩短约35分钟训练，说明实际从动状态不是无损冗余。60°目标求解误差接近0，但当前谱面用指实际角误差约12.7°；诊断仅为20Hz端点，不能代表触键瞬间。1078 final虽比best的20Hz F1低.27个百分点，但200Hz、尾段释放和碰撞更好，作为当前回放候选更均衡。本轮仅一个训练种子，未做逐音回放或跨种子显著性结论。
- **TODO**：V32-COUPLING-01保留多种子、逐音漏弹/早松及碰撞修复后复验；V32-COUPLING-02保留物理接触时的200Hz角度验证；V32-HPC-PERF-01保留完整预算/独占节点重复性能与物理数值差异验证；包内指尖标志继承TODO(V32-07)，未宣称接触面角/硬件标定。既有V32-QUALITY-01等保留。

## 2026-09-30：关闭当前指法优先并保留0.15秒提前定位

- **用户要求／问题**：移除当前优先公式对未来定位奖励的抑制，保留双卡入口提前0.15秒。
- **实际参考**：`run.sh`、`run_tsensei_a32.sh`、`scripts/hpc/job.sbatch`、`mjlab_env/fingering_experiments.py`、`rl_torch/train.py`及三个相关CPU测试文件；README和双卡配置说明。
- **修改位置与原因**：`run.sh`改为显式`--no-anticipatory-current-priority`并更新参数注释；A3.2/HPC继承关闭设置，原有0.15秒参数保留。恢复已有当前／未来用指共同平均，未来项不再乘`1-current`；没有新增公式。通用run.sh提前时间仍为0，分阶段指法仍关闭。README与`docs/hpc_training_options_20260930.md`同步当前默认；历史实验及显式开启旧公式的兼容路径保留。
- **验证结果**：三个入口Shell语法检查通过；项目`training_mjlab/.venv`内33项CPU测试通过。实际Bash入口替身捕获并交给真实配置解析器，确认run.sh为0秒/优先关闭、A3.2为0.15秒/优先关闭，末尾显式开启仍可覆盖。HPC命令仍明确传入0.15秒。首次用系统Python检查因缺Torch无法导入，切换项目环境后通过。没有执行GPU任务或启动训练。
- **未解决TODO**：本次验证配置与既有公式行为；F1、整音漏弹及提前松键收益仍需新训练对照，V32-F1-REPAIR-01和V32-QUALITY-01保留。

## 2026-09-30：耦合完整观测切换40/400Hz并更新双卡默认

- **问题与授权**：用户要求40/400Hz、完整耦合观测、关闭当前指法优先，核对dt相关公式；分析全部现有训练证据后更新双卡默认并启动一轮。沿用无需等待长训结束的要求。
- **实际参考**：`run.sh`、`run_tsensei_a32.sh`、`scripts/hpc/{submit.sh,job.sbatch,benchmark.py,package_a32.py}`、`mjlab_env/{source_runtime,reward_config,rewards,task,task_logic,anticipation,control_limits,finger_coupling,bundle,task_bundle}.py`、`physics_settings.py`、`suite/tasks/base.py`、`rl_torch/{train,replay,sac,checkpoint}.py`；935、1077、1078、1079的config/summary及已有F1汇总、性能1074、历史退化与分阶段对照报告。保留同期机械臂软件/回放相关修改。
- **修改位置与原因**：新增`train_coupled_40hz.sh`，由Slurm train调用；31动作/2121完整观测、前瞻20帧、提前.15秒、优先关闭、TX/TY1m/s。新奖励JSON增加可选reference_dt=.05及scale_by_dt=true，最终奖励统一dt/.05=.5；旧缺字段配置语义保持。gamma=√.8、初温.5；10,000,384样本、replay2M、探索/评估/存档间隔按Hz换算，保留旧5M的仿真时间。三LR/batch/.25更新比保持，总更新预算随样本增加。普通run.sh单卡默认未改。
- **设计判断**：1078在等预算三组F1最好，压缩状态没有收益；935 final碰撞更少，因此不能称已有综合最优。此次保留既有碰撞/按键权重及分阶段关闭，未猜测未知映射。40Hz新接触常数和谱面量化均会改变任务；跨频率需要时间归一碰撞和统一采样网格。README、HPC说明及`docs/coupled_40hz_training_20260930.md`同步完整证据与策略。
- **验证与启动**：CPU26项及37子测试通过；真实CPU构建365总帧/315音乐帧、559目标键帧、34个MIDI音符全部严格覆盖、钢琴solref5ms。Slurm1094共46项及41子测试通过，包括真实40Hz耦合物理环境、2121观测、跨准备段、局部reset、实际ctrl、.5奖励、旧奖励与流水回归。Shell语法通过。1095短训以0退出：512环境/262,144样本/18,816次更新，实际31/2121、2M replay=34,204,000,000字节、预填186,880样本，流水排空；训练内部204.71秒，PyTorch学习卡峰值32.0GiB。1098从匹配归档独立载入final并完成3回合，400Hz全部诊断覆盖1，以0退出。1099于12:49按新默认从头启动10,000,384样本训练，输出`training_mjlab/outputs/hpc/job-1099-20260930-124943-674629`；完整config与短训主合同及162文件散列一致，已完成整曲预填。用户明确不必等待长训完成，因此只核对启动/实际学习。机器验收记录`training_mjlab/outputs/coupled40_validation_20260930.json`。
- **实际学习与补充元数据**：1099已观察到420向量步/215,040样本/6,912更新，损失有限、奖励缩放.5核对一致、最大策略延迟一批。`scripts/swanlab_metrics.py`公开配置白名单保留reference_dt，13项及9子测试通过，实际新config导出.05核对通过。仅影响后续可选上传配置；1099显式关闭SwanLab且使用归档，不改变已启动训练。
- **TODO**：V32-COUPLING-03长期F1/碰撞/逐音/触键角及多种子验证；40Hz吞吐与学习收益待实测，短训仅验工程链路。既有V32-QUALITY-01、V32-COUPLING-01/02保留。

## 2026-09-30：克隆 MARVIN 机械臂软件 master

- **问题／要求**：用户要求将指定 `cynthia-you/TJ_FX_ROBOT_CONTRL_SDK` 仓库的 `master` 分支及仓库内机械臂软件克隆到当前项目。
- **实际参考**：本项目 `AGENTS.md`、`README.md`、`docs/change_log.md`；上游 `README.md`、`.gitattributes`、`marvinSDK_ubuntu.sh`、`marvinSDK_ubuntu_100343.sh`、`contrlSDK100343/makefile`、`kinematicsSDK/makefile`、`MarvinPlatform_EN/ui_EN.py` 及 Python SDK／示例的导入项。本机可选日志技能指定路径不存在。
- **修改位置与原因**：根目录新增完整独立 Git 克隆 `TJ_FX_ROBOT_CONTRL_SDK/`，使用 `git clone --branch master --recurse-submodules`，包含控制／运动学／干涉检测源码、预编译动态库、Python/C++示例、Windows/Linux上位机和手册；不修改上游文件。根 README 增加路径、版本、内容及运行依赖边界，便于后续使用。
- **验证结果**：克隆退出0；当前分支 `master`，HEAD、origin/master 和远端实时 `refs/heads/master` 均为 `a0f8c1134895a1b3ab2a5621bb93ae4ba0634990`；`git fsck --full` 退出0，工作树干净。无 `.gitmodules`、子模块或 Git LFS 配置；上位机二进制和 SDK `.so` 实际文件已下载，目录含历史约561MiB。此次只做文件／Git静态核验，未运行编译、硬件示例、GPU作业或训练。
- **未解决TODO／范围**：实际运行前仍需匹配控制器版本并在独立用户环境确认运行依赖；固件／机型专用参数由上游 Releases 另行提供，本次未选取或刷写，未声称真机可用。现有A3.2迁移清单与训练代码不变，不自动将新SDK纳入轻量迁移包。无新增克隆问题。


## 2026-09-30：当前1103最佳检查点、M6腕部差分IK与关节速度导出

- **问题/授权**：用户要求导出当前训练最好视频，依据wrist位置加入机械臂差分IK并导出每关节速度图。按当前最新1103正式训练的best.pt选取step19400，未混用final或跨频率历史最佳。
- **实际参考**：`AGENTS.md`、README、1103的best/config/summary及匹配源码归档；`docs/{gesture_score_comparison,tianji_mount,hand_mount_mass_arm_replay}_20260930.md`；`scripts/{eval_from_source_archive,capture_archive_render}.py`、`scripts/hpc/{submit.sh,job.sbatch}`、`preview/scripts/replay_arm_ik.py`、`preview_trajectory.py`、`preview_arm_ik.py`、包内M6 URDF/preview_mount及hand_connector配置、当前左右手URDF。指定可选日志技能路径缺失。
- **修改位置与原因**：replay_arm_ik增加原始采样/原速导出、左右14单关节图、角度及速度CSV、动态标题/真实采样率；视觉mesh从当前URDF读取并通过记录场景body视觉引用定位，避免旧名称与旧安装座硬编码；原速音频保留末尾衰减。同步README和`docs/best_wrist_ik_export_20260930.md`。
- **实际结果**：输出`preview/outputs/1103_best_20260930_wrist_ik`。Slurm1124严格归档回放400Hz，单回合音乐F1 .92550265；1126原速机械臂视频1400×900/40FPS/406帧/10.15秒，含真实琴键音频。双M6各7关节；3651位姿/3650速度区间，采用delta-q/.0025s，不滤波/不变速。峰值右1.803287/左1.786197rad/s，均低于URDF3.1416；最大位置/旋转残差.029106mm/.000239057rad；左J7有298采样点触限。
- **验证/修复**：相关6项CPU回归通过。独立全帧FK、关节范围、CSV/NPZ、插值中点检查通过；合并当前安装件CPU编译134坐标/22mocap，图像检查及两段最终视频全片解码通过。首次三次样条分析过限，保留原速样本避免过冲；1125初次渲染因mesh命名假设失败，修复后1126以0退出。失败场景/作业记录保留。新增报告/原始数据/图表/完整打包及SHA256。
- **未解决TODO**：TJ-ARM-02/03/04、V32-MOUNT-01保持；运动学解只测角限/速度/跟踪，未做冗余避限、机械臂动力学/避障或真机执行。原腕部加速度并未被限制。旧训练物理/模型/配置/检查点不改，当前安装件只是视觉覆盖层，不宣称新动力学F1通过；右手仍镜像推导。


## 2026-10-01：统一PT、多视频/多视角及速度曲线导出入口

- **用户要求/问题**：按指定PT导出可配置视频，采用`--videonums N --video1 arms backend --video2 noarms forhead`式参数；可选机械臂、同片多视角、机械臂与关节速度图。9月30日开始实现/验收，跨日完成说明。
- **实际参考**：`AGENTS.md`、前轮1103导出报告及真实400Hz物理记录、1103完整config与匹配source.tar.gz；`scripts/{eval_from_source_archive,capture_archive_render}.py`、`preview/scripts/{replay_arm_ik,render_mjlab_checkpoint,reframe_mjlab_recording,preparation_frames}.py`、`preview_trajectory.py`、`preview_camera.py`、`preview_arm_ik.py`、`preview_connector.py`、当前左右手URDF/机械臂/连接件配置、`scripts/hpc/{submit.sh,job.sbatch,package_a32.py}`与现有打包/轨迹测试。可选指定日志技能仍不存在，直接执行AGENTS日志要求。
- **修改位置/行为**：新增根`export_video.sh`、stdlib规划器`scripts/export_videos.py`和Slurm执行器`preview/scripts/export_video_job.py`。支持显式PT/config、源码SHA匹配、固定PT输入副本、1–16视频、每视频1–4同步视角、grid/横向/竖向、backend/forhead兼容别名、单视角相机自定义、命名、尺寸/帧率、原时间裁剪/播放倍速、音频/文字/高亮开关。只做一次严格归档策略回放，其余摄像机复用；可按PT/源码/轨迹/场景/音频SHA复用已有录制。普通会话只规划和提交，GPU推理/渲染均在submit.sh export分配内。HPC打包包含新入口，仅需机械臂视频或arm-joints时才附加预览资产。
- **图表/交付**：`--plots arm-joints tcp hand-joints`分别选择M6 14关节、双腕TCP XYZ/模长线/角速度、手指及腕部实际物理关节；`all/none`支持。角度/速度CSV、PNG/PDF、原始NPZ、参数/状态/校验报告、index.html及可选ZIP。曲线使用原始时间差分，倍速只改变视频/音频；全量数据与裁剪片段图表区分，F1标明仍为完整回合。速度超限/触限次数记录，不自动平滑/裁剪控制。失败保留文件，同名输出目录拒绝覆盖。同步README及`docs/video_export_cli.md`，提供直接可复制命令/全部参数/限制。
- **CPU验证**：`tests/test_video_export.py`覆盖实际用户语法、多视角、非法参数/编号/路径、归档字节及额外合同文件、录制SHA不符、输出防覆盖、裁剪/音频倍速、mocap四元数SLERP、相机、真实MuJoCo关节类型筛选、固定准备前缀和Slurm边界。连同既有轨迹/打包共31项通过；三个Shell入口语法通过。
- **Slurm实测**：1214从PT自动匹配1103归档重新推理，保存3650物理子步，成功导出640×360/20FPS/21帧静音视频，关闭文字/高亮。1215复用原1103记录，0.3–2.3秒/2倍速/20FPS/hold.2，后视+顶视双画面机械臂1280×360及前视无臂640×360均25帧/1.25秒且AAC有声；三类图表65 PNG/5 PDF、14关节差分CSV、46手关节、裁剪端点、ZIP及全片解码均验证。1216验证仅noarms视频仍能导出arm-joints、近景/自定义命名、49项复用输入SHA及触限/超速报告。1217验证无臂前视+顶视竖向480×720视频、作业归档不含M6；四个作业全部COMPLETED/0:0。画面人工检查通过。验收目录均位于`preview/outputs/export_cli*20260930`。
- **失败修复**：首次CPU四元数测试因安装SciPy的单轴数组形状要求失败，修正fixture为N×1后通过；首次Slurm1213因NumPy2.5标量与MuJoCo enum比较全False导致手关节筛选为空，显式转int并增加真实模型回归，1215后续通过。失败记录保留，未覆盖旧导出。
- **未解决TODO/边界**：当前支持有完整source_config合同的无臂标量关节手部PT；旧含实体/运动学机械臂的PT拒绝额外叠加。临时肩座/法兰、TJ-ARM-02/03/04与V32-MOUNT-01校准/动力学/避障仍待验收；右手仍镜像推导。arms只是当前装配的运动学覆盖层，不修改旧策略物理/奖励/训练、不宣称新装配F1或真机可执行。未启动训练或下发硬件指令。


## 2026-10-01：导出目录去除STL与正交琴宽铺满

- **用户要求/问题**：STL不需要出现在导出结果中；相机太远，要求钢琴宽度与视频一致并去除鱼眼/透视变形。现场最新1253导出目录含recording与scene两套中间模型，共120个STL；相机原为固定45°透视且距离1.55/1.12m。
- **实际参考**：`AGENTS.md`、`scripts/export_videos.py`、`preview/scripts/{export_video_job,replay_arm_ik}.py`、`preview_camera.py`、`preview_arm_ik.py`、`tests/{test_video_export,test_preview_camera}.py`、`docs/video_export_cli.md`、1253导出计划/报告/编译场景，以及包内MuJoCo实际geom_aabb/geom_xmat和mjv_updateScene生成的GL相机参数。可选指定日志技能缺失。
- **修改位置/原因**：统一导出worker的recording和合并scene改存原项目`logs/hpc/JOB_ID/export_work`，结果目录/ZIP不含模型文件。规划器`--recording`可接受完整交付目录并解析报告缓存位置，保留输入SHA检查。相机改正交，按实际钢琴可见几何投影宽和子画面宽高比自动取景；新增`--piano-fill`默认1.0，支持.1–1。distance保留相机空间位置含义，正交下不控制大小。取景竖直中心按片段21个姿态的手/臂/琴几何范围计算，宽度优先，不因加机械臂拉远；多视角逐帧恢复各自全局fovy。同步README和参数说明。
- **旧交付整理**：用户最近目录`preview/outputs/export_best_20261001_003535_801190`的recording/scene移到`logs/hpc/1253/export_work`，120个STL移出；保留视频/图表/CSV。更新报告迁移记录、依赖预览报告、manifest和ZIP。未删除源模型或录制，旧/新交付目录作为`--recording`均dry-run通过；未批量清理其他历史导出。
- **验证结果**：51项CPU测试通过，含五视角×三宽高比的15组实际MuJoCo GL相机正交/左右投影边界测试、填充比例、distance不缩放、干净交付缓存解析与原参数/归档/轨迹/打包回归。Slurm1294短视频验证前后及双视角拼接，0退出；1299从1103匹配归档完整回放并导出两段有声视频/原两类图表，COMPLETED/0:0、1分17秒。每段1280×720/40FPS/406帧/10.15秒，前后钢琴投影宽1.221m、正交视野高.6868125m、横向占比1.0，画面检查通过。新输出`preview/outputs/export_best_orthographic_20261001`及ZIP均0 STL/XML/MJB；两视频全片解码、ZIP CRC和交付SHA通过。输出计划/相机/缓存位置均可追溯。
- **TODO/边界**：图像取景不改变物理/奖励/动作或速度数据定义。宽度优先时，极端侧视/更长机械臂可能超出竖直画面；可选更高画布或较小piano-fill，报告记录21个采样姿态的竖直裁剪状态，不声称连续路径全几何可见。既有TJ-ARM-02/03/04、V32-MOUNT-01标定/动力学/避障边界保留。没有新训练或真机控制。

## 2026-10-01：本地 online_control 控制台、SDK适配与离线验收

- **问题/授权**：用户要求在当前仓库交付可运行本地中文网页，用于强化学习机械手弹钢琴的导出、预览、执行、记录、相机、钢琴标定与回放；默认mock，不得在开发验收中连接/使能/驱动真机。用户补充Vision2HandControl包含关节转行程实现。本机指定的可选日志技能路径不存在，直接执行AGENTS日志要求。
- **实际参考**：本仓库`AGENTS.md`、`TJ_FX_ROBOT_CONTRL_SDK/SDK_PYTHON/fx_robot.py`、原`MarvinPlatform_EN/ui_EN.py`、`contrlSDK/Robot.cpp`及实际示例/接口文档；附件`A3.2-SDK-Windows-Linux-x64-20260930.zip`内`a32_driver/{device,native}.py`、`a32_kinematics/{mapper,a32_fast,a32_angles_to_inputs,joint_coordinates}.py`和本地几何数据；`preview_arm_ik.py`、`preview_trajectory.py`、`preview_connector.py`、包内双手URDF/钢琴常量、`preview/assets/tj_m6`及现有安装配置；`scripts/{eval_from_source_archive,capture_archive_render}.py`、`preview/scripts/{render_mjlab_checkpoint,replay_arm_ik}.py`、`mjlab_env/task_logic.py`及1103已有rollout/原编译场景/臂轨迹。Vision2HandControl网页缓存可读，但实时Git/API/raw访问受限/404，未假称读取其最新代码，继续复用附件实际转换实现。
- **新增代码及原因**：功能集中在`online_control/`。`backends.py`薄封装原Marvin/A32 API，SDK经许可证/散列归档到`vendor/a32_sdk`，运行不读附件路径或其他旧项目；`trajectory.py/export.py/models.py`实现统一时间轨迹、原action/ctrl/实际状态分离、按编译模型位置伺服和gear映射、原控制区间时间及零阶保持、当前谱面参考音符、臂手独立导出/对齐、真实URDF及Link7安装链。`control.py/app.py`提供独立调度与独立超时监视、单控制权/进程锁、显式解锁/确认、完整限值/起点/恢复复核、深复制快照、优先停止撤销与锁存、缺测拒绝及同源请求检查。浏览器预览时钟从不发送伺服命令。
- **记录/标定/前端**：`recording.py/camera.py/calibration.py/replay_video.py/midi.py`提供有界异步日志和写盘告警、可配置的关键相机故障独立停止/锁存、原始序号/时间戳/发送结果、SHA归档、可选RGB/RGB-D和MIDI、半自动ROI深度/琴面/已知键对应点/独立复核、固定和腕载外参求解、手调重验/版本回退、回传驱动三维骨架视频及逐帧缺测/插值证据。原仓库只有仿真摄像机接口，未找到物理相机实现；因此新增可选OpenCV/RealSense薄接口，不安装重型平台。中文控制台使用本地Three.js、原有72个网格、本地Noto字体，明确计划/实时/历史来源、琴位与执行隔离。附最小/可选依赖、独立安装/启动脚本、mock/硬件配置骨架、示例和说明。
- **既有文件改动**：`scripts/hpc/submit.sh`、`job.sbatch`新增`online-export`分配内入口；`package_a32.py`仅纳入离线导出必要源码，避免网页依赖/SDK/运行数据进入训练包。根README增加入口；详细说明为`online_control/{README,EXPORT,ADAPTERS,CAMERA,ACCEPTANCE}.md`。原上位机、训练/奖励/模型参数及已运行实验不改。
- **本地PC访问补充**：用户截图在本地8765显示“Form2”标签。核对当前应用终端、`__main__.py`、`app.py`、`static/app.js`及服务器监听，实际服务在127.0.0.1:8765正常返回HTTP200和“TsenseI · 本地控制台”；截图标题与本服务不符，未武断确认本地端口占用进程。`online_control/README.md`补充本地18765→服务器8765的SSH转发、不同电脑回环地址含义及标题检查，继续保持回环监听和来源校验。未操作用户PC或验证其SSH隧道，也未连接设备。
- **验证结果**：相关CPU测试与既有打包回归78项通过；Shell/Python/JS语法通过。实际浏览器加载72网格、JS/HTTP错误0，预览拖动/倍速/琴位保存发送0；执行快照不变，急停后计数331→331。真实旧rollout离线转换手20+20关节/365状态帧，31 action、46 ctrl、134 qpos分离；臂7+7，琴键88路、62实际事件、34谱面参考音符。实际mock回传日志重建83帧10FPS视频，全片解码和SHA通过、日志丢失0；API回放与视频共用实际回传时间基准并逐帧核验。早期测试夹具遇到浮点等式、JSONL读取方式、新外参确认字段预期，以及软件渲染墙钟假设错误，均修正夹具后通过；首轮缺系统CJK字体/浏览器库，补本地字体和用户目录浏览器测试库后通过。开发全程未连接真机，未提交训练或GPU推理。
- **未解决TODO/明确门禁**：TODO(V32-ONLINE-01)实物SDK坐标/关节顺序/动态限值、经验证空间与碰撞控制器；TODO(V32-ONLINE-02)网口跨主机独占及控制器看门狗/停止时延/断线物理停止能力；TODO(V32-ONLINE-03)实体相机外参/深度/MIDI与视频同步实测；TODO(V32-ONLINE-04)新checkpoint导出Slurm端到端运行。当前网口运动因无可验证独占租约被阻止，缺少现场空间验证器的实机执行保持拒绝；硬件停止统一报告未确认，不能以SDK发包成功证明静止。右手仍是镜像推导，SDK未校准右手，URDF effort/velocity=0不猜作硬件限制；原TJ-ARM-02/03/04、V32-MOUNT-01继续保留。Vision2HandControl最新源码需有可读路径/权限后才能比对，本轮不宣称联调成功。


## 2026-10-01：指定回放 MIDI 导出与原始乐谱逐音对比

- **问题/授权**：用户要求导出 `export_best_20261001_101143_985592/videos/video2_noarms_front.mp4` 的MIDI并与原始乐谱比较。
- **实际参考**：该导出目录的config/report/export_plan；作业1353 recording中的played.mid、audio_events.json、audio_substeps.npz、rollout.npz和report；源 `music/data/rousseau/twinkle-twinkle-trimmed.mid`；`preview_audio.py`、`music/{library,midi_file,piano_roll}.py`、`rl/environment.py`、`preview/scripts/{render_mjlab_checkpoint,export_video_job}.py`。指定可选日志技能不存在，遵循AGENTS日志要求。
- **修改位置与原因**：仅新增原导出目录下 `midi_comparison/` 的离线compare.py、5个MIDI（原始逐字节/原始归零/实际视频时间/实际归零/40Hz量化目标）、逐音CSV、JSON、PNG/PDF、HTML报告、manifest及独立ZIP；同步 `docs/video_export_cli.md`。保留原视频和原导出manifest/ZIP。原文件首音14.692708333秒，实际准备1秒，分别扣除且不拟合时间平移/拉伸，分开报告原曲与量化基准误差。
- **实测结果**：原曲34音、实际31音；同音高与起音差≤150ms唯一匹配31音、漏3音、额外0音。漏音E5/D5/E5的原曲归零时间6.557292/6.656250/6.708333秒，原时长62.5/52.083/67.708ms。已匹配音相对原曲起音延迟中位29.792ms，note_off晚13.125ms；相对40Hz目标起音晚45ms、note_off早5ms。实际统一力度127，原曲17～61；音乐逐帧F1仍0.9255026455。此前聊天的最后激活采样时刻与本次真正note_off边沿相差2.5ms，不能混作同一统计。
- **验证结果**：原曲SHA与checkpoint源配置一致；400Hz键程/踏板重新生成事件与原JSON完全一致，生成MIDI与原played.mid SHA一致；原始/实际归零MIDI回读音高、起止时间校验通过（<0.1ms）；每个原曲音唯一对应量化目标，匹配无歧义。检查PNG和ZIP内容/CRC/manifest及HTML链接。首次ZIP因原始MIDI复制的历史mtime早于1980失败，改用strict_timestamps=False后成功，仅ZIP条目时间受限，文件内容不变。全程仅CPU离线分析，无仿真、GPU推理或新训练。
- **未解决TODO/限制**：不把本次宽容起音匹配计数称为标准音符F1；踏板余响未合入按键时长，物理力度/音量未标定（V32-AUDIO-01保留）。原始归零MIDI采用固定tempo保留秒级事件时序，完整元数据/速度图请用逐字节原文件。重建脚本仍依赖该作业原录制，交付ZIP内MIDI/CSV/图/HTML可独立打开。


## 2026-10-01：恢复视频指法标注并清理旧预览产物

- **用户要求/原因**：用户指出视频没有乐谱/MIDI指法，随后要求完成后删除`preview/output/`中的无用历史产物并自行重新导出；实际目录为`preview/outputs/`。新多视角worker只调用了ActiveKeyHighlight，漏接原render_mjlab_checkpoint中的FingeringHighlight；指法训练数据本身没有丢失。
- **实际参考**：`AGENTS.md`、`preview/scripts/{export_video_job,render_mjlab_checkpoint,replay_arm_ik}.py`、`scripts/{export_videos,eval_from_source_archive,eval_from_source,capture_archive_render}.py`、`mjlab_env/{source_runtime,task_bundle}.py`、`preview_fingering.py`、`preview_keys.py`、1353实际录制report/rollout/physics_trace、1103完整config和匹配源码归档、原导出计划；`tests/{test_video_export,test_preview_camera,test_preview_trajectory,test_a32_migration_bundle}.py`、`scripts/hpc/{submit.sh,job.sbatch,probe_gpus.py}`及安装工具`prepare_connector_mount.py`、`replace_hand_mount_from_step.py`。指定可选日志技能路径仍缺失。
- **修改位置/行为**：规划器新增默认开启`--fingering/--no-fingering`，`--finger-labels/--no-finger-labels`控制额外三维悬浮编号（默认关闭）。新`scripts/export_score_annotations.py`从精确匹配归档临时重建原task，检查完整task SHA和每帧录制goals，恢复真实指法/指尖名称，不重新分配指法、不重复策略/物理回放。复用录制也核对rollout SHA；缺少匹配输入明确报错。数据保存在交付`data/score_annotations.npz/json`，含校验信息。
- **显示/时间**：worker恢复指尖/目标琴键颜色和底部左右手指法，例如R4:G5；绿色实际按键仍独立。按原控制区间及评分端点选择指法，固定准备前缀/初帧为空、policy准备空谱/释放尾段保留原定义，裁剪/倍速/末帧停留与各视角同步。顶部文字开关不影响指法；为底栏预留竖直取景空间，保留正交和琴宽铺满。README与参数文档同步。
- **验证结果**：58项CPU回归通过，新增覆盖源任务/录制目标不符、时间错位、左右手和弦、准备/释放/停留、裁剪/倍速、复用旧录制、标注独立开关及实际MuJoCo相机底栏净空。1103归档恢复365帧（40Hz）指法，task SHA `b5e9accc0ee695fa3e3c19664e4e387bbaca1bbf836b296b5dd35cf9c1ca4285`与1353记录一致，所有目标逐项一致。CPU OSMesa/llvmpipe实际重渲染后视+顶视机械臂2560×720和无臂前视1280×720两段有声视频（各18帧/10FPS/1.8秒），另测无指法静音分支；全片解码和帧图检查通过。报告在`logs/fingering_cpu_validation_20261001`。私有OSMesa仅解包到用户cache，按Ubuntu包索引SHA校验，不改系统驱动/CUDA；首次镜像下载502，官方HTTPS下载成功。
- **GPU边界**：Slurm1356经规定入口提交，但8张GPU全部被其他用户分配，未启动；完成CPU软件渲染后取消，不占后续资源。本轮没有GPU推理/渲染验收，不将软件渲染称为GPU通过。原策略/物理/奖励/机械臂IK合同和检查点未修改，未启动训练/硬件控制。
- **清理结果**：确认无进程打开旧输出，删除16个完整旧目录及connector_alignment中的旧图/网格/报告，共1070文件、2,072,455,555字节（约1.93GiB）。仅保留`preview/outputs/connector_alignment/hole_aligned_mount.json`（888字节），因两个安装工具仍读取它；保留文件SHA验证不变。删除范围不含训练PT、源码/模型资产、Slurm归档/录制缓存或online_control产物；旧MIDI对比/视频交付路径一并按用户清理授权失效。清理清单`logs/preview_outputs_cleanup_20261001.json`，清理后原PT的新导出命令dry-run通过，指法默认为开启。
- **未解决TODO**：新代码在GPU渲染路径的复验因资源占用未进行；旧肩座/法兰/动力学与右手镜像等边界保持。现有极端侧视竖直裁剪限制保持；悬浮编号在近距离可能重叠，因此默认用底部明确指法说明。用户将自行生成新视频。


## 2026-10-01：用户 Vision2HandControl 源码与 Linux SDK 独立提取

- **问题/授权**：用户在侧对话上传 Vision2HandControl-online_motion_play.zip，要求传到服务器并提取 A3.2 下发必要文件，随后补充 Windows/Linux x64 SDK 并要求继续。仅执行此独立提取任务，不改动主线程控制后端或运行服务。
- **实际参考**：当前 AGENTS.md；新源码包的 `src/a32_driver/{device,native,recording}.py`、`a32_hardware.json`、`src/a32_kinematics/`、`hand_profiles.py`、两种 A3.2 型号配置、`online_control/online_motion_play/a32_session.py`、记录/时钟工具和对应 SDK/动作/范围说明；补充 SDK 的 Linux C++/CMake、C ABI 头文件、两个离线检查脚本及原包散列清单。附件说明仅作源资料，不执行其中设备操作指令。
- **位置/原因**：原 ZIP 持久保存于 `/home/houj/projects/source_archives/`；独立目录 `online_control/imports/vision2handcontrol_online_motion_play_20261001/` 下 `source/` 原样提取81文件/884170字节，`sdk/` 保存完整跨平台SDK78文件/2313593字节。保留转换几何/URDF/通道表/Windows DLL/Linux so/原生源码/记录和播放参考，略去原项目其他手型、重型视觉/网格/UI/部署与缓存。添加README、两份SHA清单、最小离线依赖、可重复离线校验脚本和验证JSON；不覆盖现有vendor或训练模型。
- **验证结果**：原 ZIP 副本 SHA一致；SDK自带77项SHA全部通过；81个源码提取文件SHA及29个Python语法通过。裸手/withid×左右手共4组20关节→14直线mm+6旋转deg，LUT/精确线性输出最大差1.251e-7mm，缺关节拒绝。SDK另完成4组20关节→18self+2MIT准备。Linux so实机主机加载ABI2/16导出成功；Vision原版Python封装显式加载该so后完整/快速未连接状态读取通过。两包转换/硬件配置/源URDF文本一致（字节差异来自CRLF/LF），平台选库逻辑在SDK native.py。未连接、使能、下发或寻零真机；未启动服务/训练/GPU。
- **限制/纠正**：先看到Vision源码只附Windows DLL不代表用户未提供Linux支持；随后SDK已补齐并验证Linux库加载。网页独立环境缺PyYAML，改用已有训练环境仅做CPU数值验证，没有修改运行环境。提取件非完整上游网页/视觉平台，原始完整源码仍在ZIP；右手硬件标定、真实通信与停止时序未验证，后续主线程可按文档引用这些已校验文件。


## 2026-10-01：指定视频关节速度/加速度与臂手默认图表

- **用户要求**：为`preview/outputs/export_best_20261001_111043_212264/videos/video1_arms_front.mp4`导出机械臂逐关节速度、加速度图；随后明确要求导出参数默认包含机械臂和手的每个关节两类图表。本轮同时补导指定视频的臂/手图表并修改后续默认，未重新推理或渲染该视频。
- **实际参考**：`AGENTS.md`、该1359视频/manifest/report/export_plan和`data/{arm_trajectory.npz,arm_joint_angles.csv,arm_joint_velocities.csv,arm_report.json}`；1359缓存中的physics_trace、原report和kinematic_scene；`preview/scripts/{replay_arm_ik,export_video_job}.py`、`scripts/export_videos.py`、`preview_trajectory.py`、实际编译MuJoCo joint type/qpos地址；`tests/{test_video_export,test_preview_trajectory,test_a32_migration_bundle}.py`、README与`docs/video_export_cli.md`。指定可选日志技能路径不存在，直接遵循项目日志要求。
- **修改位置/原因**：`scripts/export_videos.py`的`--plots`默认由none改为`arm-joints hand-joints`；显式选择仍替换默认，none关闭、all额外含原TCP速度。新增`preview/scripts/joint_motion_plots.py`共用数值/图表导出，worker的臂/手图表统一调用；双手按原编译模型选实际hinge/slide及单位，从动关节包括在内，已结构移除的TZ/Yaw不虚构。默认即使noarms画面也求臂IK用于曲线；只需手可显式选择hand-joints。普通回放仍按原Slurm入口，训练参数/模型和IK不改。
- **数值定义**：采用原采样位置差分v=Δq/Δt，时间为位置区间中点；加速度a=Δv/Δt_v，时间为速度采样中点。无滤波、样条、解绕、限速或倍速换算；保留尖峰。旋转单位rad/s与rad/s²、平移单位m/s与m/s²；视频末帧停留不追加人为零值。裁剪端点沿既有轨迹插入；少于3位置样本明确无加速度，不伪造零曲线。每关节两类独立PNG/PDF、左右汇总图、位置/速度/加速度CSV、NPZ和峰值/时间/RMS报告。
- **离线入口/当前产物**：新增`preview/scripts/export_joint_curves.py --video PATH`，只读取匹配的原导出数据/缓存，核对视频/轨迹SHA、时刻和机械臂列顺序，不另求一条IK；新目录防覆盖，独立manifest/ZIP。当前输出在原导出目录的`video1_arms_front_joint_curves/`：14臂关节+46手/腕关节，共128 PNG、128 PDF，原时间0–9.125秒、400Hz、3651位置/3650速度/3649加速度样本。HTML索引、CSV/NPZ、报告和约34.9MB ZIP已生成；原视频及原交付manifest/ZIP不变。README和导出参数说明同步。
- **验证结果**：64项CPU测试通过，含均匀/非均匀时间网格上解析二次轨迹/匀速、重复/倒序/非有限输入拒绝、极短片段、实际MuJoCo平移/旋转列及默认/覆盖开关。指定视频全量位置与原NPZ/物理记录逐项相同；新速度与旧机械臂速度CSV一致；用独立400Hz中心二阶差分核对全部加速度、所有CSV与NPZ一致。原/新manifest全部SHA、269项ZIP CRC通过，汇总及平移关节图人工检查通过；默认导出dry-run确认两组图表。全部为CPU数组分析/绘图，无GPU推理、新训练或硬件下发。
- **实测摘要/边界**：右臂速度峰值J6=1.851502rad/s、加速度峰值J6=140.162941rad/s²；左臂速度峰值J3=1.786192rad/s、加速度峰值J5=202.039726rad/s²。加速度为已记录IK采样的离散估计，未假设URDF未提供的加速度上限，不表示机械臂动力学/真机可执行验收；手曲线来自真实仿真位置，不是策略目标。原TJ-ARM-02/03/04及镜像右手/安装标定边界保持。原视频帧率40Hz不作为求导采样率。


## 2026-10-01：远程 Codex 曲线报告打开方式

- **问题/原因**：用户在 Codex 浏览器打开 `export_best_20261001_111043_212264/video1_arms_front_joint_curves/index.html` 失败。当前浏览器状态为该绝对路径的 Google 搜索结果；报告在 TenseiNode1，项目为远程 SSH 项目，本地 Windows 浏览器不能直接读取该服务器文件路径。
- **实际参考**：当前 AGENTS.md、报告 HTML 与相对图片/数据链接、`docs/video_export_cli.md`、`online_control/README.md` 的既有 PC 转发说明、Codex 项目元数据及 `open_in_codex` 工具说明；OpenAI Docs 技能及官方 Browser 文档（https://learn.chatgpt.com/docs/browser?surface=app，地址栏搜索与本地页面预览）。可选项目日志技能路径不存在。
- **处理/修改**：在服务器 127.0.0.1:18766 启动 Python 静态服务，根目录仅为该曲线报告目录；进程/启动命令记录 `logs/joint_curves_http_18766.json`，访问日志同名前缀 `.log`。请求 Codex 在侧栏打开报告网页，工具仅返回 queued，不据此宣称渲染成功。参数文档增加服务启动、本地 SSH 转发和整包下载后打开的说明。原 HTML、图表、视频、manifest 与 ZIP 均未改。
- **验证结果/边界**：服务器 HTTP 首页与全部264个唯一相对资源链接均200，PNG/PDF文件头正确，验证记录 `logs/joint_curves_http_18766_validation.json`；无需 GPU、推理或重导。未操作用户 Windows SSH 客户端，无法验证本地转发与实际浏览器渲染；最终提供本地 PowerShell 转发命令及明确的 HTTP 地址，不把远程回环地址等同于 PC 直连。服务需持续运行，主机重启后可按文档重新启动。


## 2026-10-01：机械臂 SDK 速度/加速度限值默认标图

- **用户要求**：从SDK查找各机械臂关节速度与加速度限制，并在今后的曲线图中默认标注；同时更新当前 `export_best_20261001_111043_212264/video1_arms_front_joint_curves` 报告。
- **实际参考/溯源**：项目内 `TJ_FX_ROBOT_CONTRL_SDK/CommonConfig/ccs_m6_40.MvKDCfg`（SDK提交 `a0f8c1134895a1b3ab2a5621bb93ae4ba0634990`，配置SHA `ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52`）；`kinematicsSDK/FxRobot.cpp` 的LOADMvCfg/FX_Init_Robot_Lmt_CCS、`AxisPln.cpp`的OnSetPNVA及规划公式、`SDK_PYTHON/fx_kine.py`的load_config/initial_kine、`contrlSDK/MarvinSDK.h`；厂商同一固定提交的公开配置和Python文档；预览URDF/source.json及现有 `joint_motion_plots.py`、两个导出入口和相关测试。未加载或连接SDK控制驱动。
- **查证结果**：配置块0/1分别对应左右臂，7个关节速度均180°/s；J1/J2加速度450°/s²，J3–J7为900°/s²。读取器将每轴文件第7/8列明确放入PNVA[2/3]，随后赋给m_JLmtVel/m_JLmtAcc，规划中使用v²/(2a)、sqrt(位移/a)；据此澄清SDK概述的加速度/加加速度及单位笔误。换算为π rad/s、2.5π/5π rad/s²；不以URDF速度近似值3.1416替代SDK精确度制换算。
- **修改位置/原因**：新增 `configs/preview/m6_sdk_joint_limits.json` 保存度制数值、提交/文件SHA和代码解释依据；`preview/scripts/arm_motion_limits.py` 按完整左右J1–J7名字校验并转SI，无硬件SDK运行依赖。该配置位于正常源码选择范围，保持独立打包。`joint_motion_plots.py` 接受明确单位和名字匹配的参考限值，每关节及左右汇总PNG/PDF画正负红虚线、超限红点与峰值倍数，纵轴保留全量曲线；单位错配/零或非有限限值拒绝。两导出入口默认传入SDK限值，HTML由同一数据生成表格；输出 `joint_limits.json`、逐关节超限采样数/比例及首末时刻。手/腕缺少确认值时标注Reference limit unavailable，不套用臂限制。README和参数文档同步。
- **当前报告更新**：纯CPU从原视频导出/缓存重绘14臂+46手/腕，保留原400Hz时间/位置/速度/加速度；128 PNG、128 PDF及HTML/JSON/manifest/ZIP全部更新。原URL `http://127.0.0.1:18766/index.html` 保持，仍需用户本地SSH转发。临时重绘验证通过后替换交付目录，旧报告保存在 `logs/joint_curves_sdk_limits_20261001/previous_report`，不在preview/outputs新增遗留目录；原视频及其原导出根manifest/ZIP未改。
- **验证结果**：70项CPU测试通过，涵盖SDK原配置SHA/列数值、deg→rad转换、单关节和两符号实际绘图线/超限点/轴范围、原始尖峰不裁剪、单位/名字/无效限值拒绝及独立包选择。与旧报告逐CSV SHA和NPZ逐数组比较完全一致；新269项manifest SHA及270项ZIP CRC通过，265个页面相对资源HTTP200，抽查右J2加速度及左臂速度汇总图可读。验证/替换/HTTP记录在 `logs/joint_curves_sdk_limits_20261001/`。14个臂关节速度均未超SDK参考，13个有关节加速度超限采样；最大参考倍率左J5=12.86225（202.039726rad/s² / 15.707963rad/s²）。
- **边界/TODO**：限值为SDK M6默认规划配置参考，未读取实物控制器、按负载降额或运行时百分比缩放，不声称PVT/直发命令强制执行此上限或已通过真机动力学验收。未修改轨迹/滤波/倍速、IK、训练、奖励、原SDK、在线控制门禁或硬件配置。手/腕实际速度/加速度限制仍待确认；既有安装/右手镜像/动力学TODO保持。


## 2026-10-01：双卡新训练改为碰撞负罚与固定闲指姿态扣分

- **用户要求/范围**：解释外部权重及两个指法开关；双卡入口跨手掌部改为无碰撞0/有碰撞扣分，关闭当前指法优先，同手自碰撞weight改1，gesture去几何仅保留闲指姿态。用户随后确认最终每控制步固定上限0.1，十指均摊，不保留十倍倍率与额外时间缩放。仅修改后续新训练默认，未提交、停止或改写正在运行的训练/冻结队列。
- **实际参考**：AGENTS.md、run.sh、run_tsensei_a32.sh、scripts/hpc/{submit,train_coupled_40hz}.sh，mjlab_env/{reward_config,rewards,performance_options,fingering_experiments,finger_clearance,task}.py，rl_torch/train.py，原coupled_40hz及gesture10奖励JSON；相关CPU公式、HPC入口、精确时序和GPU集成测试；README及docs/hpc_training_options_20260930.md。可选日志技能路径此前核对缺失，遵循本目录日志规则。
- **修改位置/原因**：新增configs/rewards/coupled_40hz_idle_posture_20261001.json并设为双卡入口默认；旧JSON不改。reward_config新增显式collision_penalty_v1和idle_posture_penalty_v1合同，保留旧模式校验与序列化。rewards中掌部继续用控制周期接触并集，原始0/-0.5经40Hz原时间系数变为0/-0.25；同手weight=1后每去重body对最终-0.5。当前指法优先此前已关闭，新JSON再次显式false，当前/未来共同平均；保留旧优先实现用于旧合同。
- **姿态最终公式**：-0.01×Σ闲指[mse/(mse+0.25²)]，mse为该指四个实际关节相对原场景姿态的均方角差。当前和0.15秒内即将用指豁免，固定十指分母，准备/休止也计罚。删除此模式的几何风险、几何参考可行性掩码、score_multiplier及dt二次缩放；weight=1且严格校验≤1。其他奖励保持原dt/.05。每步上限固定意味着换频率后每秒强度会变化；40Hz全十指上限累计4分/秒。物理接触、几何诊断和原best键底门槛保留。
- **开关解释/兼容**：分阶段指法仍默认关，30%接近+70%对齐/近键面/实际键程；零键程时当前用指原始分最多0.3。提前请求许可只放宽旧几何奖励的末节请求风险，不提前动作或MIDI、不豁免实际/近端/其他键/键底风险。新姿态模式没有几何评分，该开关默认0，强行启用由既有校验明确拒绝。60–80°耦合、完整观测、1m/s、40/400Hz、精确时序和学习预算保持。外部权重是乘数而非概率，无需总和为1。
- **验证结果**：65项CPU测试及71项参数子测试通过，命令覆盖test_idle_posture_rewards、test_paper_rewards、test_geometry_clearance_reward、test_gesture_score_multiplier、test_reward_time_units、test_fingering_experiments、test_target_key_reward、test_hpc_40hz_entry、test_exact_timing_entry。新增解析值测试验证最终分数、固定分母、当前/未来豁免、与几何无关、跨dt不额外缩放、外部权重只乘一次、碰撞并集、旧掌部基线与非法合同；旧gesture10及checkpoint兼容回归通过。两个shell入口bash -n与git diff --check通过。GPU集成预期同步姿态dt例外但未执行GPU测试。
- **文档/TODO**：README与双卡参数说明同步最终分值、选项依赖和历史实验边界。新默认GPU训练/检查点重建及长期F1、短音、碰撞表现尚未验证；不把CPU公式通过表述为收敛收益。跨掌基线整体减0.25/步会降低return，后续应按音乐F1和碰撞实测比较。未修改其他侧任务的预览/机械臂代码及实验产物。


## 2026-10-01：双卡默认曲目切换《月光》初学者版

- **问题/授权**：用户要求将当前双卡训练脚本改为训练月光；本次为入口修改，未要求启动训练。
- **实际参考**：`scripts/hpc/{train_coupled_40hz.sh,submit.sh,job.sbatch,package_a32.py}`、`run_tsensei_a32.sh`、`run.sh`、`scripts/train_from_source.py`、`mjlab_env/source_runtime.py`、`music/midi_file.py`、`score_validation.py`、`tests/test_hpc_40hz_entry.py`，以及 `training_mjlab/inputs/clair_de_lune_beginner/generated_v1/` 的既有proto和指法报告。
- **修改位置与原因**：`train_coupled_40hz.sh`新增默认`--midi`，指向由当前源码根目录解析的`clair-de-lune-beginner_fingered.proto`，沿用用户此前授权的146音工程候选指法，末尾CLI仍可覆盖。`submit.sh`注释、README和`docs/hpc_training_options_20260930.md`同步曲目；文档移除只适用于小星星的固定365帧预填说明。仅双卡默认曲目改变，普通run.sh默认、SAC、奖励、40/400Hz、31动作/2297观测、耦合60–80度、10,000,384样本及流水配置保持。
- **验证结果**：4个入口Bash语法检查通过；现有双卡入口CPU测试3项通过。捕获真实wrapper链确认最终月光proto绝对路径有效，末尾`--midi TwinkleTwinkleRousseau`可覆盖。原proto SHA与既有指法报告相同；Composer音乐处理的40Hz strict_v1覆盖146/146音、2587音乐帧、无覆盖错误；打包collect_files确认归档包含完全相同的proto字节。source runtime已有外部谱面嵌入完整config机制，无需依赖旧项目路径。未执行仿真、GPU推理或提交新训练。
- **未解决TODO/限制**：当前A3.2耦合模型的候选指法运动学/碰撞/长期演奏表现未验收。1秒准备与0.25秒释放使回合约2637帧，整曲预填及每次评估变长；保持原预算，不宣称与小星星耗时相同或已达目标F1。已运行或指定旧归档的作业沿用各自冻结配置。


## 2026-10-01：仅手指角度调整五指水平 initial

- **用户要求**：基于当前姿态只改手关节角，让四指和拇指指尖构成与琴面平行的平面，保持自然弹琴手型并检查。
- **实际参考**：AGENTS.md、包内左右URDF、`tsensei_hand_constants.py`/`tsensei_hand.py`/`fingertip_geometry.py`/`urdf_model.py`/`collision_proxies.py`、suite base和piano常量/构建、preparation/scene、direct_joint_v1动作范围、run.sh、preview_pose/Slurm入口、test_environment和test_preview_pose_config。指定可选日志技能路径缺失。
- **修改位置/原因**：只改源码INITIAL_ANGLES_BY_SIDE中左右各16个手指角，最小化相对原角度改变量并约束五site等高；最大调整约4.392°。四指swing、腕部、手根、2.5cm抬升、模型/驱动/控制/奖励公式不改。同步两处测试的旧角度常量断言及README，详细角度/证据写入docs/initial_five_tip_plane_20261001.{md,json}。保留用户其他未提交改动。
- **验证结果**：新源码CPU重建origin/initial，双侧五指最大高度差<1e-6mm、拟合平面倾角<1e-7°；initial距白键面56.26614mm，origin31.26614mm；两状态静态接触均0，限位/reset/镜像与仅手指角AST核对通过。回归22通过、3既有模型失败，三项在旧姿态独立复现（primitive/mesh假设、旧掌壳接触名称、原URDF惯性）；未隐瞒或联动修复。真实模型图经submit.sh在Slurm1364.13/A100渲染，已检查图像及高度对比图。
- **边界/TODO**：initial/origin共用手指角，因此后续新源码两种姿态及依赖其初始角的姿态奖励参考共同更新；旧归档/已运行实验不变。TODO(V32-03/07)保留自然手型和指腹标定，site共面不等同完整指腹接触面齐平；仅验收reset，策略/从动耦合后不保证保持水平，未启动训练/修改耦合/声称动态避碰或F1提升。


## 2026-10-04：本地WSL硬件三页控制台、远程PT导出与分步标定

- **问题/授权**：用户要求把现有mock为主界面改为本地PC/WSL网口与USB实机控制，远程服务器只选择PT并导出；三页分别为轨迹/预览、左右臂手简化控制、记录/逐步标定。用户补充WSL已能通过网口连接机械臂和相机，尚未提供USB路径/相机型号/固定标定位姿；开发不得连接设备，仍按授权仅离线实现/验收。
- **实际参考文件**：`AGENTS.md`、现有`online_control/{app,control,backends,models,trajectory,export,camera,calibration,recording,replay_video}.py`及前端；原`TJ_FX_ROBOT_CONTRL_SDK/MarvinPlatform_EN/ui_EN.py`、`SDK_PYTHON/{fx_robot,fx_kine,fx_interference}.py`、运动学LOADMvCfg实现、`CommonConfig/*.MvKDCfg`、`DEMO_PYTHON/showcase_drag_interference.py`与干涉配置/源码；附件SDK已归档`vendor/a32_sdk`；原`submit.sh/job.sbatch`和导出合同。WSL部署命令核对Microsoft官方networking/connect-usb说明。指定可选日志技能此前已确认缺失，直接维护本日志。
- **实际修改/原因**：`__main__.py`默认hardware但零设备I/O，`--mock`显式选择测试；三页本地界面保留单控制权、常驻停止、预览隔离。`hardware_setup.py/backends.py`加入WSL只读诊断、配置持久化接口、实际厂商参数候选读取、分组件使能/模式和反馈FK TCP；网络运动改为明确现场独占安排+本机进程锁/UDP独占，明确没有厂商跨主机租约。SDK A/B运动学按实际加载数组0/1取值，未照抄原GUI右侧索引0问题。停止后必须复核并明确重新使能，无自动续跑。
- **远程/映射/空间**：新增`remote.py/remote_helper.py/remote_routes.py`，仅显式操作使用本地OpenSSH别名/known_hosts；允许根目录限制、符号链接检查、固定stdin协议、Slurm原入口、后台监控/取消/散列下载，重启不自动联网，提交回执丢失不重复提交。`commissioning.py`和逐关节编辑表保存明确单位/符号/零点/模型版本，原轨迹不变；反映射仅用于模型显示并保存到记录。`spatial.py`复用原厂干涉模块与已审定的文件/覆盖对，未知手/琴覆盖仍拒绝；慢预检不占用反馈线程，结束再核对起姿；未选保持部件姿态锁入快照并在执行时监视。
- **标定/交付**：`robot_calibration.py`导入已验证的分步轨迹，每步独立确认并复用原控制器；稳定实际姿态、末位姿和相机时差通过后采集，不能以发包成功当到位；固定相机用明确T_tool_marker和回传FK/点击深度，独立检查点求解。现有五步视觉向导加画面指示，实时/历史/回传视频保持原始数据来源。`package_local.py`按源文件/模型资产白名单生成独立WSL包，排除训练环境/权重/实验/私钥；新增WSL.md/REMOTE.md/配置样例，更新README/ADAPTERS/ACCEPTANCE和本地中文字体。原上位机和训练/奖励/模型参数不改。
- **验证结果**：114项相关CPU测试通过，最后映射单位/速度字段修正再跑6项通过；Shell和JS语法通过。三页Chromium软件渲染mock执行/硬件只读及配置表单均通过，JS/HTTP错误0，预览0发送、执行快照隔离、停止后计数冻结、hardware不自动连接。ZIP隔离解压验证4模型/72网格/3原生库及所有必要本地接口，无Torch/MuJoCo/设备/SSH调用。最初旧STL1970时间戳令ZIP失败，修正ZIP时间兼容而未改模型；现有mock样例起姿与新源码seed不同，按当前预览seed更新明确mock fixture后通过。证据`online_control/test_artifacts/wsl_20261004/`。
- **未解决TODO/边界**：本地WSL设备、真实时延/软件停止/看门狗、相机型号与深度尺度/去畸变、实物限值/映射/装配/钢琴禁入空间、实际SSH/Slurm导出仍未真机验证；不声称联调成功。SDK仍未校准右手。厂商15碰撞对不含当前手/钢琴，必须提供现场空间配置；没有给未知范围编造通过值。原preview_pose.json手常量SHA已因用户后续源码变化过期，原仓库/便携包均显式seed回退，未擅改姿态。TJ-ARM-02/03/04、V32-MOUNT-01和V32-ONLINE相关实测事项继续保留。

## 2026-10-06：当前视频首帧左手—钢琴坐标系与距离图

- **最终要求／范围**：用户最终仅要求当前视频t=0手与钢琴坐标系及原点间距，取消其余工作。不带ID模型此前仅创建隔离候选文件，未完成渲染、未替换活动模型；本次交付使用原视频带ID左手。
- **参考／实现**：冻结`preview/outputs/left_base_from_start_20261005/inputs/scene_manifest.json`与同次400Hz机械臂、手、琴键首帧记录，既有手—法兰关系校验；读取`piano_constants.py`及真实几何。新增`logs/hand_piano_frames_20261006/draw_frames.py`，输出同名目录内坐标系PNG、来源与数值JSON，同步坐标文档。CPU MuJoCo仅mj_forward，不执行物理步进；坐标轴与P—H尺寸为真实三维点的投影叠加，附白键侧剖示意。
- **验证／结果**：四项输入SHA与原manifest一致；H独立与此前FK值一致，[140.000,−150.000,125.000]mm；P—H空间距240.260261mm（XY投影205.182846mm）。52个未按下白键实际geom顶面全部22.5mm；qpos前后完全一致；llvmpipe软件渲染、PNG可读取并视觉检查标签。机械臂／手安装和钢琴原点均未变动。模型坐标测量不代表硬件标定；未新生成视频或轨迹。


## 2026-10-06：635最佳模型独立有声视频导出

- **请求/参考**：用户在侧对话要求导出635视频；参考`docs/video_export_cli.md`、635实际best19500/config/82199601…bda022归档、已有585同次回放比较与验收流程。未继续侧对话边界前的训练请求，未改动运行中的训练。
- **执行/产物**：原视频部署`export_video.sh`经submit.sh申请单H100 Slurm685，62秒COMPLETED/0:0；635 PT SHA `51d5c5d82309c8cf32782a3ad77f715bb37e35be2dfa5697a9feec3a66045f51`，原奖励及hard_v1、Pitch目标3.1416rad/s与15.708rad/s²保持。`preview/outputs/experiment_635_best_20261006/`交付正面双手1280×720、20fps、原速有声10.2秒视频及同次实际按键/原始MIDI PNG、PDF、CSV、NPZ、MIDI、来源、HTML和ZIP；保留1秒准备与末帧1秒停留。
- **验证**：1830个200Hz真实琴键子步重建实际事件，与原MIDI精确时序比较；同次视频音乐micro F1=.9289137380191693（92.89%），区别于训练三回合最差93.23%。PT/config/源码合同、目标/实际控制端点及声音事件逐项检查；21项manifest及ZIP CRC/内容SHA、全片H264/AAC解码、PNG检查通过。下载后再次复核全部文件及解码，查看4.5秒视频帧和对比图。证据`logs/hpc/export_635_20261006/`。
- **边界**：仅本次导出及附属日志/文档；不改模型、训练、物理限制或历史视频，不把本次F1和未记录的碰撞指标外推为硬件安全。

## 2026-10-06：新左臂初始角、固定反解基座及手指适度平滑

- **请求／范围**：用户给定新左臂七关节77.48315／−70.55979／−61.07445／−81.20139／160.95941／14.69720／−25.86782°，要求重生成轨迹，随后明确机械臂和手指都优化、手指适度降低速度与加速度。沿用此前“仅首帧反解基座，之后固定跟踪原wrist”合同，不改变右臂、原曲节奏及手—法兰刚性安装。本轮没有新训练、设备控制或视频要求。
- **实际参考**：AGENTS.md、`export_best_constrained_20261001_151434`冻结wrist/config/plan、`left_base_from_start_20261005`冻结arm/hand/mount/scene/connector及manifest；Slurm1359原physics_trace/XML；1103源码归档与原checkpoint；`online_control/{models,trajectory}.py`、preview_arm_ik、preview_connector、constrained_arm_motion及M6 SDK参考限值JSON、既有坐标和运动约束文档。指定可选日志技能路径缺失，不影响本日志维护。
- **实际实现**：任务脚本／尝试与失败证据放在`logs/left_replan_smooth_20261006/`，最终交付独立目录`preview/outputs/left_replan_smooth_20261006/`。保持原t=0手完整位姿，由 `T_P_H inv(T_B_H(q0))` 一次反算左基座 `[618.618028,−19.096110,412.253914]mm`，轨迹全程固定。更新当前preview_mount仅左base_xyz/base_rpy/seed；写入前核对旧SHA、工具SHA及其余字段完全一致。右侧、连接件、URDF、旧结果和当前训练源码不改。
- **机械臂**：较强80%加速度约束及过强SQP候选未满足原容差，不交付。最终C2样条对候选系数混合后独立审核1600Hz／14601点，最大0.099603mm／0.484043°；连续角度／速度／加速度系数界通过SDK参考。相对同一新首帧、同一基座未平滑基线，整体速度峰值−4.265%，整体加速度峰值−0.612%，各关节加速度RMS−16.0～32.7%；所有臂加速度峰值下降，J2/J5速度峰值略升，逐项如实交付。
- **手指**：原实际20关节轨迹C2适度平滑，原接触几何±25ms保护指尖0.5mm／1°、闲时1mm／3°，纳入URDF角范围；原食指PIP/DIP稍低于零限位如实记录。最终单独接触保护段最大偏移0.494962mm；全20关节同频率400Hz差分速度与加速度峰值均下降，整体速度−0.415%、加速度−14.258%。额外约束连续解析加速度到原差分峰值95%（独立审核95.001%，容许数值精度），解析全局上界1737.345695rad/s²与差分峰值1568.036872rad/s²分别说明，不能混报或称硬件限值。
- **验证／交付**：PT/config/源码归档全部源文件散列、原录制、冻结URDF和依赖散列通过；原9.125秒／1秒准备、无新增衔接、七角精确首帧及原手首帧保留。200／400Hz臂／手／组合CSV含位置与解析导数，另有角度、手根钢琴系FK和原目标、样条NPZ、目标语义JSON、关节映射、基座矩阵、曲线PNG和峰值对比表。CSV读回、FK跨实现、连续范围、共享时间、PNG、manifest和ZIP校验由任务脚本独立完成；详细审核记录见delivery_validation.json。
- **边界／TODO**：合并臂姿态误差后保护段指尖最大偏移1.167783mm，全程1.533312mm。原录制键状态下音高接触集合相同，原激活键接触样本保留96.477%（新q0未平滑基线97.755%）；不是新键激活或F1。没有新物理回放／音轨，不能用旧音频和按键记录证明新手指效果完全一致；旧注释中的F1／音轨不转用于新轨迹。SDK限值、手指硬件速度／加速度、负载／力矩、装配和伺服／动态击键仍待实测；TJ-ARM及既有安装TODO不变。同步两份坐标／运动约束文档，未改训练或奖励。

## 2026-10-06：附件左臂轨迹的有声视频及新琴键回放

- **请求／参考**：用户上传200Hz七关节CSV要求导出视频，数组逐项确认与`left_replan_smooth_20261006/left_arm_200Hz.csv`完全一致，使用同一规划的400Hz连续臂样条／20关节手指、冻结基座和连接件。参考本目录AGENTS、video_export_cli、上述manifest/preflight、1359冻结physics_trace/XML及原MIDI、原audio后处理和上一版CPU render。PT/config/源码归档散列再次检查，不重新运行旧策略或覆盖旧视频。
- **实际实现**：新增`logs/left_csv_video_20261006/{common,record,render,finish}.py`，交付`preview/outputs/left_csv_video_20261006/`及ZIP。400Hz规定手指位置/解析速度与手根运动，每步恢复非琴键状态，使用原XML由mj_step重新积分琴键；右手移离钢琴，仅左手影响按键。不把旧键程/音轨、新手指目标或原谱作为实际按键，阈值沿用原0.00872665rad距键底。原MIDI逐字副本及原左手5–9指法事件保留，1秒准备；踏板关闭，因为附件不含踏板。
- **视频／声音／图**：左机械臂和左手完整可见，相机在钢琴−X侧向+X演奏者，1600×1000／40FPS／405帧／10.125秒，原速，9.125秒轨迹及1秒末帧展示。CPU llvmpipe软件渲染，无GPU。新3650物理子步／3651含初始帧实际琴键qpos/qvel记录用于动画、实际激活阈值、FluidSynth声音及同次原MIDI三层PNG；实际/原始事件JSON/CSV、绘图NPZ、输入SHA、音频和视频同目录保存。音符力度固定127，不声称物理力度标定。
- **验证**：附件数值、精确七角首帧、固定基座逐帧、400Hz共同时间轴与键激活重算、音频事件从同次键程重算、PT/config/归档和原MIDI SHA通过；CPU相机方向及完整臂手网格边界检查，已查看4秒帧／对比PNG。视频H264/AAC全片解码及405帧、音轨、10.125秒检查通过，manifest／ZIP SHA及CRC核对通过。第一次后处理缺本地音频库搜索路径，仅复用同次新键程修复音频环境，未重新模拟；原MIDI旧归档时间戳不适于ZIP，打包容许旧时间戳，不改源文件。验证详见任务delivery_validation.json。
- **指标／边界**：本次音乐区间[1,9.125)单次400Hz键状态micro F1=.9607780612244898，TP3013／FP44／FN202，不是原checkpoint策略评估或训练分数。规定手根／手指运动、每步恢复非键状态，机械臂运动学显示；未验收PD/伺服、整臂惯性、力矩或真机。运行时读取冻结XML而非历史版本不兼容MJB，不宣称训练动力学逐位复现或原效果完全保持。新实际琴键声音／图取代本轮先前“仅几何”的导出缺项，但规划目录及旧证据不改，当前安装、训练／奖励和所有旧视频不变。同步视频导出说明。

## 2026-10-06：最新轨迹t=0环境信息导出，排除手关节

- **用户要求／参考**：导出t=0除手关节外的环境信息。按最新`left_csv_video_20261006`视频而非旧安装读取冻结manifest/mount/connector、同次臂／手／琴键数据；参考规划`left_replan_smooth_20261006`及标定、1359原XML／初始trace、原生法兰矩阵、原臂URDF、online_control.models和视频构建／相机脚本。
- **实现／交付**：新增`logs/environment_t0_20261006/export.py`，独立输出`preview/outputs/environment_t0_20261006/`与ZIP。总JSON、中文README、全部臂连杆／基座／钢琴／原生法兰／手根／连接件位姿CSV与矩阵，七臂关节状态及URDF质量／惯性／限位，88键初始真实状态和尺寸／质量／弹簧／阻尼，安装配置与网格来源SHA，琴键物理环境及视频显示环境分别导出；含重力／求解器／时间步长、相机／灯光／地面／非手指几何／摩擦／材料。手指与虚拟腕内部关节的角度／速度／加速度／目标、完整qpos/qvel/ctrl及指节网格均未写入文件；保留独立的手base_link空间位姿。
- **验证／数值**：两个冻结导出manifest逐项通过；臂URDF与物理XML SHA、独立FK／显示基座／物理mocap手根、原生法兰相对手变换及旋转矩阵／四元数一致；28项交付文件及ZIP CRC/SHA通过，并逐文件检查20个手关节名均未泄漏。左B=[618.618028,−19.096110,412.253914]mm，F=[163.690649,−154,107.766190]mm，H=[140.000001,−150,124.999971]mm；手／法兰同向，法兰在手系[+13.5,+4,−26]mm。88键t=0均未激活，踏板关；仅CPU建模/FK，无物理步进、渲染或GPU，证据见任务delivery_validation.json。
- **边界／文档**：显示用臂惯性代理明确与原URDF惯性区分；物理模拟仅规定手运动后积分琴键，右手mocap上移2m隔离并非右臂安装改变，右臂安装仅作冻结参考。未改安装、轨迹、训练／奖励或旧导出。排除手内部关节后不是完整手姿态恢复包；网格仅按路径／SHA引用，不打包。同步坐标文档；既有实物标定与动力学TODO保持。

## 2026-10-07：原网络控制目标与同次物理执行、F1及有声视频

- **请求／实际参考**：用户追问原网络actions／最终ctrl是否保存，并要求导出目标轨迹、仿真实际执行、F1和视频。读取Slurm1359的rollout／physics_trace／audio_substeps、归档render_mjlab_checkpoint和scene控制循环；核对1103的PT／config／source.tar.gz、原始注册MIDI及指法。复用export_best_constrained_20261001_151434来源记录、left_base_from_start_20261005原左键接触归属与声音事件、left_replan_smooth_20261006冻结安装及未平滑臂基线、left_csv_video_20261006 CPU绘制框架。实际来源为1103 gesture10-comparison的best step19400，geometry_clearance_v1倍率10；不能沿用早期图片的idle-only／19000标签。
- **实现／交付**：新增logs/network_targets_execution_20261007/{common,export,render,finish}.py，输出preview/outputs/network_targets_execution_20261007及同名ZIP。365周期原31维action、46维最终ctrl及左23通道目标逐值导出40Hz CSV；同次400Hz实际qpos含t=0共3651帧，实际qvel仅3650已录子步，未补造初始速度。policy_time为控制周期末，CSV明确interval_start/end；ctrl在10个物理子步间保持。另交付目标／实际端点并列表、23通道曲线、原实际手根钢琴系位姿、完整原始NPZ、通道单位／映射和来源SHA。
- **视频／指标**：保留新七角和固定反解左基座，臂采用未平滑基线IK作显示，手指取原实际物理位置，不对原目标／执行再平滑。仅显示左臂左手、相机朝演奏者，CPU OSMesa／llvmpipe，1600×1000／40fps／405帧／10.125秒，原左键声音。同次400Hz实际键程与原始MIDI三层PNG、原始谱／事件／计分数组保存；音乐区间[1,8.875)，1秒准备、0.25秒释放、额外1秒末帧展示。左键micro F1=96.6121%（TP3037／FP35／FN178）；原同次双手400Hz原谱micro F1=94.2072%，双手40Hz端点／量化目标micro F1=92.6033%，分开说明。
- **验证**：PT／config／源码归档、原始录制、原MIDI及冻结安装manifest校验通过；原actions／ctrl／qpos／qvel CSV精确读回，F1独立重算、图／声音同源、时间语义检查通过；查看4秒帧及按键PNG，视频H264／AAC全片解码405帧、47交付文件manifest、ZIP SHA／CRC通过，证据为logs/network_targets_execution_20261007/delivery_validation.json。初次renderer发现旧键档字段为key_limits而非limits，修正读取后重绘，不重新推理／模拟。
- **边界／文档**：原物理回放为双手，左键来自同次接触归属；隐藏右侧不代表独立左手仿真。原训练no_arms=True，七臂关节CSV／画面只是实际wrist的IK显示，非网络输出或臂动力学反馈。未重新运行策略／物理、未使用GPU、未改变当前安装、训练／奖励／旧结果；不把此前新规定运动回放的96.08%混入此次原执行。同步video_export_cli及交付README；实物映射、安装及伺服验收TODO保持。

## 2026-10-07：附件实机下发参考与反馈的当前场景运动学回放

- **问题／授权／参考**：用户要求附件command_aligned_20261006_210206_8add5170_angle_all.csv的实际参考和反馈在当前无动力学场景各跑一遍。读取60102行原数据、online_control.models／replay_video／commissioning、preview/scripts/replay_arm_csv的场景构建、现有原网络录制CPU绘制框架、当前左臂／A3.2左手URDF及preview_mount／hand_connector；不读取外部旧项目或连接设备。用户最新补充“实机四指swing零度对应原模型最大值”，取代先前“减20°”答复，最终统一为q模型=0.35−实机角度×π/180，其余关节按确认的同名直接映射。
- **修改／交付**：新增logs/hardware_kinematic_replay_20261007/{prepare,render,finish}.py，独立输出preview/outputs/hardware_kinematic_replay_20261007及ZIP。只显示左臂和左手，实际下发参考／实际反馈左右并排，另有两个独立视频；前视相机面对演奏者，下方同尺度手部俯视。直接设置27关节角，mj_kinematics／mj_camlight更新位置；无动力学／积分／IK／平滑。当前基座、钢琴、连接件、URDF及模型资产不改，原CSV逐字归档。
- **时间／数据合同**：指令与反馈分别使用各自原采样时刻；每关节2226条发送，臂独立反馈2226点（中位约5.008ms），手约1174独立快照（中位约10.495ms），不把发送频率当反馈频率。原实测126空行；仅≤30ms反馈间隙作线性显示插值，更大间隙及初始缺失隐藏相关子树、逐帧说明，禁止用下发值填补。指令零阶保持。t=0对应原1.268903879秒，末帧12.995417858秒；含精确末帧471个显示采样点，40fps／510视频帧／12.75秒，末帧展示约1秒。
- **参考／图表**：完整参考视频使用实际下发位置。原驱动参考位置缺左臂7关节及拇指两旋转关节，另存CSV并绘三色原时间曲线，不以指令拼接冒充；其时间使用原反馈快照，没有独立参考时钟。保存原始去重长表、映射、三类显示CSV／NPZ、逐帧来源、关节采样缺失说明及40Hz显示口径RMS／最大角差。36个去重负swing实测点映射后略超URDF上限，最大0.0091394°，保留不裁剪并记录幅度。
- **验证**：原文件SHA／60102行、全部三类去重原采样与映射、三个CSV读回、缺失逐帧状态、当前URDF／基座／连接件／网格来源及固定基座核对通过，独立FK与绘制模型矩阵最大误差1.9984e-15；软件渲染及三个视频全片解码／各510帧、PNG、27文件manifest／ZIP SHA与CRC核对均通过。查看首帧和4秒双列／俯视截图；证据在logs/hardware_kinematic_replay_20261007/delivery_validation.json。第一次截取手部视野不足、中文字库为子集，改为完整手部边界取景与英文画面标签；输入附件只读mode被copy2保留，重建时改为SHA一致复用；重新导出前应用最终swing映射，没有修改原CSV。
- **边界／文档**：该文件为实机日志而非checkpoint回放，没有真实琴键状态／MIDI来源或声音；钢琴静态背景，视频静音，不能推导实际击键／F1或套用旧按键图与音乐，交付README和video_export_cli明确缺项。未训练／推理／GPU／新物理回放／设备发送，所有训练／安装和旧结果保持；实物安装、负载／伺服与接触验收不在本轮范围。

## 2026-10-07：合并核对后按授权补空，重新导出参考与反馈

- **请求／变更／实际参考**：用户最初要求从某组起直到末尾实测位置全部非空，并补充self_9单通道；最后明确改为“空的位置和上一帧一样”。最新angle_all附件与原全表SHA一致，self_9的2226行逐项为同一表中指PIP副本，合并没有新增／冲突／新可补值。核对原CSV及各自指令／反馈时钟、原hardware_kinematic_replay脚本和来源记录、当前models／URDF／安装以及本目录AGENTS。严格不补空的后缀原仅剩121组／约0.63秒，该未交付尝试由用户最新补值要求取代。
- **实现／数据**：新增logs/hardware_ffill_replay_20261007/{fill,prepare,render,finish}.py，独立交付preview/outputs/hardware_ffill_replay_20261007及ZIP。仅同一关节实测位置空项沿用上一帧有效反馈，不用指令补值；完整文件111项补齐，15个初始未知不猜填，非空实测与全部其他原字段不变。新增填充方式／原有效实测时间两列及逐项来源JSON。跳过开头11组，从第12组完整指令时刻1.326636539秒开始；保留2215组／59805行，选中部分83补值，27关节实测位置字段全非空。原始两附件逐字保存，9个全空驱动参考通道不填。
- **回放／时间**：对真实反馈快照维持≤30ms显示插值，补值区间始终保持上一帧真实值，不提前插值到恢复后的新反馈，原始来源时间独立保留。源末帧12.995417858秒，实际11.668781319秒；含精确末帧468个显示采样点，40fps／507视频帧／12.675秒，约1秒末帧展示。三个参考／反馈并排及独立视频，上方面对演奏者的整臂前视、下方手部俯视；全程27关节可用，不隐藏手指，补值帧标Held。保留四指swing 0.35−实机角度×π/180，当前固定基座和工具／钢琴安装不变。
- **验证／交付**：逐行独立核对所有111补值只来自同关节先前实测、15未知未补造、其他字段原样；原附件／派生表SHA、self_9重复关系、裁切表59805行及零空实测、原生采样填充标记／来源时间、三个显示CSV读回、468帧完整反馈、固定基座／模型资产SHA及独立FK通过。FK最大矩阵误差1.88738e-15；查看首帧和4秒前视／俯视，三个H264静音视频全片解码各507帧、PNG、31文件manifest／ZIP SHA与CRC通过，证据在logs/hardware_ffill_replay_20261007/delivery_validation.json。绘制用状态字符串显式U32，避免补值标记被NumPy短字符串截断。同步video_export_cli及交付README。
- **边界**：补值有原始来源标记，不称新增实测；40Hz显示角差指标含补值，不当作全频反馈误差或音乐F1。继续只调用CPU运动学与相机更新，无动力学／接触／IK／网络／GPU／实物发送。文件无真实琴键激活及匹配音乐时钟，钢琴静态，因此没有击键声音／F1／原MIDI按键图；已解释二者需要同次琴键接触仿真或实测按键数据。未改训练、当前安装和此前视频；既有实物标定／负载／接触TODO保持。


## 2026-10-07：实机轨迹严格固定，仅琴键接触动力学生成声音与F1

- **请求／实际参考**：用户选择方式一，明确实测机械臂和手指不能被动力学改变。读取最新angle_all、self_9附件和上一轮hardware_ffill_replay的已验证填充／裁切／原生时钟、当前冻结scene_manifest/URDF/基座/连接件；参考1359冻结scene.xml／physics_trace、1103匹配PT/config/source.tar.gz、原始TwinkleTwinkleRousseau MIDI及归档library明确指法，使用既有软件渲染／音频模块。没有改训练、默认奖励、当前安装或旧输出。
- **实现／不变保证**：新增logs/hardware_key_contact_20261007/{common,prepare,record,postprocess,render,finish,test_record}.py和piano_only_dynamics.xml，独立交付preview/outputs/hardware_key_contact_20261007及ZIP。将左手22个连杆变成逐步规定FK的world mocap刚体；物理模型只有88个piano关节，机器人nq/nv=0，无机器人质量积分、PD或力反馈修正、无臂动态关节、无equality/actuator，排除同手碰撞保留冻结手—琴形状。每一步mocap位置／四元数array_equal，位置误差0m／旋转矩阵误差1.22e−15；原27关节数组、NPZ和FK文件SHA回放前后不变。机械臂只按文件FK显示，绘制模型从不mj_step。不是每步先受动力学影响再复位的旧做法。
- **采样／作业**：同前从CSV1.326636539到12.995417858秒，2215组59805行、选中83授权前值补空；四指swing沿用0.35−deg×π/180，其余直接映射。按原指令／反馈时钟直接重建400Hz：指令零阶保持、真实反馈间≤30ms插值、补值区间前值保持；未从40Hz视频表插值或声称400Hz硬件实测。400Hz4668物理步／4669状态，末步0.001281319s，只有键积分。经scripts/hpc/submit.sh单卡validate提交Slurm1877，COMPLETED 0:0；专用CPU MuJoCo录制在分配内运行，GPU只做申请内环境探测，无CUDA推理。CPU OSMesa/llvmpipe离线绘图／渲染。
- **视频／同源指标**：两个独立轨迹各自录琴键qpos/qvel/activation/contact_counts/峰值力，不借用旧键程或旧音轨。三个有声H264/AAC视频：并排1920×1080及独立960×1080，40fps507帧12.675秒，左臂左手完整可见、面向演奏者，下方手俯视；并排声音取右侧反馈，单独视频各自按键声音。按距键底0.00872665rad阈值取激活，FluidSynth力度固定127、无踏板。两张原谱／真实模拟按键／叠加三层PNG、实际／原谱事件JSON/CSV、计分数组、原MIDI、原生填充表和400Hz27关节CSV同目录。全段[0,11.668781319)400Hz micro F1：参考0.27030539311241064(TP832/FP2109/FN2383)，反馈0.24927085358740034(TP641/FP1287/FN2574)；音乐区间另为0.2797579018和0.2544660580，均不是训练或硬件F1。
- **同步／模型边界**：硬件CSV没有乐谱时钟，询问后未收到新时间，按下发七臂关节匹配原规划t=1s估计音乐起点CSV4.431607860s／视频3.104971321s；拟合仅用下发臂角，不用键状态调分，两条共用起点。原MIDI时值与播放率1.0保留；原下发匹配时间速率0.9485793说明硬件执行约慢5%，未拉伸原谱抵消。固定手轨迹在黑键限位处强制压入约7.2–7.4mm；原软琴键限位上下超程最大参考0.0829986rad／反馈0.1289506rad，峰值求解力约149812／154042N。已核对完整冻结手模型FK与规定FK差≤约1.2微米，未发现明显帧错位；不移动／裁剪机器人或调接触参数来修分。力不能作为真实硬件力，F1仅约束模型诊断；接触形状来自原1359而非新训练默认。限位／实物接触标定仍TODO，声音非麦克风录音／力响度标定。
- **验收／文档**：结构性88键only/22mocap断言、全部原生重采样及补值规则、27关节CSV逐值读回、各步规定位置不变、输入／规定FK SHA、当前安装与资产SHA通过；PT/config/源码归档以及原MIDI／归档指法实际音符时间重新核对。实际键激活、两种区间F1、音频事件/MIDI字节、共用时间轴独立重算；三个视频强制映射H264/AAC全片解码各507帧、非零WAV、PNG、61交付文件manifest、ZIP SHA/CRC通过。首帧及7秒前视／俯视和反馈原谱图已查看。验收日志logs/hardware_key_contact_20261007/delivery_validation.json，逐步不变证据data/trajectory_invariance.json；同步video_export_cli与交付README。最初原谱平移的浮点结合顺序产生约1e−15s差，验收改为1e−12s时间容差，不改任何时间数据、输入SHA或模型合同。ZIP保留数据／绝对资产依赖记录，不声称独立无依赖安装包。


## 2026-10-07：原rollout的ctrl／qpos在固定机器人、琴键only环境对比

- **用户要求／参考**：用户引用原始rollout.npz，要求在刚才无机器人动力学环境导出；确认“下发ctrl和原实际qpos两条都导出作对比”。核对1103 gesture10-comparison best19400及Slurm1359匹配rollout/physics_trace/XML、PT/config/source归档、原《小星星》MIDI／归档library指法；参考network_targets_execution的通道语义、上一轮hardware_key_contact的0机器人自由度模型／record/audio/软件render、当前冻结scene_manifest／URDF／安装及ArmPreview FK/Jacobian。保持当前安装、训练归档／奖励和所有旧输出，未重新运行策略。
- **实现／轨迹合同**：新增logs/rollout_prescribed_comparison_20261007/{common,prepare,record,test_record,postprocess,render,finish}.py和冻结piano_only_dynamics.xml，独立交付preview/outputs/rollout_prescribed_comparison_20261007及ZIP。ctrl为原最终40Hz目标，按[policy_time−.025,policy_time)零阶保持，包含准备／映射／保护后的目标，不以31维actions直接冒充关节角；actual用同次完整400Hz trace含初始帧，原rollout40Hz端点另存并核对差<2e−7rad/m（float32记录精度）。原20指关节逐值保留，是URDF原生弧度，不再套硬件swing映射。原准备mocap400Hz共享两条，1秒lift、原0.15rad安装倾角保留，不追加或重新估计音乐时钟。
- **机械臂／安装**：原训练no_arms=True，无J1–J7数据；在当前固定基座/法兰/连接件下分别对目标／actual的手base_link位姿逐点IK，七角只用于当前机械臂换算和显示。最大位置误差9.97e−9／9.99e−9m、旋转9.63e−8／9.77e−8rad，手指不改、不平滑或更改时序；actual首七角精确保留用户77.48315等角度，ctrl首帧已有不同腕目标，所以其七角不同，不插入平滑段去改变目标。手根source初始对齐仅3.25e−16m浮点舍入（数值恒等），没有移动或反解新基座。两类输出均保存目标根／IK误差；不宣称该IK CSV直接可安全下发硬件。
- **严格不受动力学改变**：使用同上一轮只含88键关节、22个规定FK的world mocap手连杆、无机器人自由度／PD／equality／actuator的物理模型。臂不加入动力学。输入27角和规定FK在每一步及前后SHA不变，mocap位置／四元数array_equal；身体位置误差0m／矩阵约1.22e−15。经scripts/hpc/submit.sh单卡validate提交Slurm1884，COMPLETED0:0，1passed；两条琴键新录制均在分配内完成，普通会话只CPU FK/IK／后处理／软件渲染。
- **交付／时间／分数**：原0–9.125s，0–1s准备，1–8.875s音乐，0.25s释放；400Hz3650步/3651状态。三个有声H264/AAC视频405帧40fps10.125s：targets_vs_actual1920×1080及targets/actual960×1080，只有左臂和左手、前视朝演奏者、下方俯视，额外1秒只展示末帧；并排声音来自右actual，独立视频各自声音。主27角400Hz CSV（七臂IK＋20原指）与原23通道ctrl40Hz、原rolloutqpos40Hz、原actual400Hz、原NPZ、通道单位另存。两张原谱/实际模拟按键/叠加PNG、原/实际事件JSON/CSV、计分数组、原始MIDI及音频同目录。声音／F1／绿色按键来自对应同一次新400Hz键程，阈值距键底.00872665rad，力度127、无踏板。原MIDI时间和指法原样，音乐起点1s不拟合硬件；本次音乐区间原谱micro F1：ctrl22.096850%(TP470/FP569/FN2745)，actual95.578340%(TP2983/FP44/FN232)，全段分数相同；非原训练／旧物理分数或硬件F1。
- **模型边界**：固定目标位置与原受PD驱动的实际轨迹不同；直接把ctrl当位置时出现较深琴键软限位超程，ctrl/actual最大偏差0.238271/0.0288071rad，峰值求解力206394/2466.72N，记录如实保留，不当实物受力。没有改机器人位置、接触参数、时序或误键过滤去修分。F1是各自规定运动下琴键模型诊断；接触几何为原1359冻结版本，非当前训练新默认。结果包以资产SHA和项目绝对依赖路径记录，不称独立安装包。
- **验收／文档**：PT/config/源码/MIDI/库指法/全部录制来源SHA、当前基座工具URDF资产SHA、原ctrl保持、原actual位置、20指逐值和全部CSV读回、IK精度、88键only/22mocap模型与逐步不变、原谱实际音符时间/1s偏移、键激活/F1/音频边沿及MIDI字节独立重算通过；三个视频全片强制映射H264/AAC各405帧/40fps/10.125s与非零WAV、PNG、64文件manifest/ZIP CRC和SHA通过。查看首帧、7秒对比／俯视及actual原谱图；证据任务delivery_validation.json和交付validation/media/trajectory_invariance。同步video_export_cli和交付README，render字段明确“只在绘制阶段无IK”，避免误称准备阶段未做IK。


## 2026-10-07：三路轨迹／视频／F1 对比网页框架

- **请求／实际参考**：用户要求先实现整体网页框架：三路文件选择与视频，第一路考虑机器人动力学，另两路固定机械臂和手指、仅琴键动力学；各自 F1，下方按手关节／指尖切换空间位置曲线，独立横纵轴缩放。读取本目录 AGENTS、online_control/models.py 的 URDF/FK、包内 fingertip_geometry.py、已完成 network_targets_execution_20261007 与 rollout_prescribed_comparison_20261007 的冻结场景、manifest/report/metrics、原生 27 关节 CSV／NPZ、同次视频与按键／原谱图。可选 pianist-v2-change-log 技能路径缺失，按本目录日志规则执行。没有读取其他项目、重新运行策略或连接硬件。
- **实现**：新增 trajectory_compare/{app,data,__main__,browser_check}.py、独立运行／测试依赖及 README；static 的离线中文三栏界面实现文件选择／拖放、任务提交／状态、视频／F1／按键图／轨迹下载、同步播放／定位／速度／声音来源、26 个点下拉（五指尖、手 base_link、20 个手关节子链接原点）、钢琴坐标系 mm 的 XYZ 曲线、横纵轴分别缩放、拖动／滚轮／Shift 滚轮、复位及点击定位视频。使用固定版本官方 Noto Sans SC 本地字体并携带 OFL／来源 SHA，不依赖 CDN。根 .gitignore 放行新目录而排除运行数据，README 增加入口。
- **数据／任务合同**：只接受明确原生关节名和单位的 27 关节 CSV、禁止 pickle 的 NPZ（展开限 128 MiB）及显式单位 JSON；文件限 32 MiB，有限数字／严格递增时间／唯一名称校验，不猜硬件 swing 零位或缺值。CPU FK 保留安装与原输入角度；图仅均匀抽取最多 1400 个原采样点含首尾，不当完整接触记录。运行目录持久化上传、曲线、场景快照／SHA 与任务，媒体允许列表支持 Range。任务合同明确 robot_dynamics／prescribed、机器人自由度 0（后者）和 Slurm scheduler；新输入返回 pending_backend，无视频／声音／F1，不实际提交作业。只有原字节 SHA＋相同模式匹配已验证缓存才返回既有结果。
- **真实示例／边界**：三默认示例同次音乐段 400 Hz micro F1 为 96.6120566%／22.0968500%／95.5783403%，对应各自原按键图与有声视频。第一路原训练 no_arms=True，只有手的状态来自原物理执行，七臂角为腕轨迹 IK 显示；页面显著标注，不能称完整机械臂动力学。另两路严格规定机器人位置，只有键物理自由度；原 1 秒准备段、0–9.125 秒轨迹及额外 1 秒视频末帧展示不变。既有固定接触软限位／力值及非实机 F1 边界保留。只在本工作台 cache 生成首路精确七臂＋20 手组合 CSV，没有修改历史导出。
- **验收**：tests/test_trajectory_compare_{app,data}.py 共 17 项 CPU 通过；含独立 FK 点值／冻结模型与视频来源 SHA、明确角度映射、非法／超大输入、按字节及模式隔离、HTTP 视频 Range、任务与上传恢复、场景快照 SHA。Chromium 软件模式 --disable-gpu 真实浏览器验收三卡／26 点／对应分数、文件选择缓存命中、新文件 pending 且无旧分数、同步播放暂停定位、独立 X/Y 缩放／滚轮／平移／复位／关节选择、刷新状态恢复、390px 手机无横向溢出；中文字体加载、无 JS 错误或 HTTP 5xx。查看桌面及手机截图，证据 trajectory_compare/test_artifacts/{browser_report.json,desktop.png,mobile.png,pending.png}。JS 语法、Python 编译及任务文件 diff 检查通过；本机服务默认 localhost:8787，8788 仅隔离验收后关闭。
- **未解决 TODO／范围**：后续接入 Slurm 仿真／渲染 worker 与同次声音、按键图／F1／实际执行轨迹；第一路机器人动力学控制合同、实机长表映射与各轨迹时钟仍待研究。框架当前不把上传当作已物理执行，不伪造动态回放。所有训练参数、源码归档、当前模型安装、运行作业和旧视频保持。


## 2026-10-07：修复对比网页重合轨迹遮盖蓝线

- **问题／实际参考**：用户截图指出“原始物理执行”的蓝线未显示。读取 trajectory_compare/static/app.js 的绘制／悬停／图例逻辑、同页 bootstrap 的实际曲线以及上传截图。三路数据均存在，第一和第三路各 1400 点；食指尖 X 差值中位数仅 0.00479mm、90% 小于 0.06052mm，等宽 1.7px 蓝→绿→橙的绘制顺序让橙线覆盖蓝线。接近截图时刻的 4.5325s，两路 X 差约 0.08367mm；不是数据缺失。
- **修改／原因**：app.js 用蓝 4px／绿 2.7px／橙 1.5px 嵌套线宽显示重合部分，悬停点用 5／3.8／2.6px 半径，保留 XYZ 方向虚实线型及原坐标。图例增加三路勾选显隐，可单独看任一路；显隐保持当前横纵轴视图、切换关节点保持显隐，全隐藏有明确空态。index.html／styles.css 更新操作提示、图例控件和响应式布局，trajectory_compare/README.md 同步说明。
- **验证**：Node 语法与 diff 检查通过；真实 Chromium 软件模式用同一份实际数据对比旧等宽／新嵌套线宽，食指尖 X 图蓝色像素从 12 增至 2948，单独蓝线为 6132。实际截图确认蓝橙两线均可见；显隐保留范围、全部隐藏提示、关节点切换后保留显隐、恢复三路、缩放／重置及 390px 手机无横向溢出均通过，无 JS 错误。截图与报告在 trajectory_compare/test_artifacts/overlap_fix_20261007/{all_tracks.png,blue_only.png,report.json}。未新增模拟测试或重跑物理视频。
- **范围／TODO**：仅修改可视样式和图例显示，不偏移／平滑／修改轨迹，不改模型安装、视频、F1、训练或导出合同。此前待接入的新仿真导出器与硬件映射 TODO 保持。
# 2026-10-11：导出左臂镜像、法兰跟随与掌面朝下训练场景

- **要求／依据**：用户要求将左臂与右臂镜像、让左手随法兰刚性旋转、把左手安装到掌面朝下，并导出训练场景交互图。核对当前 `current_arm_training_20261011.json`、`arm_scene.py` 的 schema=3 刚性装配、已记录的镜像FK审计和当前编译模型的五个 distal-tip 站点。
- **结论／处理**：当前左七轴初始角已经是记录的右到左镜像解，未用未经验证的七轴全变号；手根已经是 `Link7 > fixed_flange > fixed_four_hole_connector > fixed_hand_base` 的固定后代，会随法兰旋转。左侧五个指尖站点拟合平面与键面倾角为0°，作为当前场景的掌面朝下几何判据。未修改默认训练配置、训练归档、checkpoint或硬件状态。
- **交付／位置**：新增 `preview/scripts/export_arm_mirror_palm_down_scene.py`，从同一 `build_bundle_from_config` 生成可旋转缩放的 Plotly HTML、静态总览PNG、机器可读 `scene_report.json`、README、manifest与ZIP，位于 `preview/outputs/arm_mirror_palm_down_scene_20261011/`；同步README及 `docs/arm_mirror_palm_down_20261011.md`。
- **验证**：`MUJOCO_GL=disable training_mjlab/.venv/bin/python preview/scripts/export_arm_mirror_palm_down_scene.py` 成功；脚本 `py_compile` 通过；Link7→左手根最大矩阵误差 `1.11e-16`，左指尖平面倾角 `0°`，Link7链镜像位置最大误差约 `0.0069 mm`；HTML／JSON／ZIP均写出并含输入源码散列。导出未推进物理、未运行策略、未发送硬件命令。机械臂父子接触过滤修复及无臂自碰撞合同保持。 :codex-annotation{index="1"}
# 2026-10-11：按右臂绝对值镜像重算左臂角度与安装时钟候选

- **用户修正／依据**：用户明确要求左臂 J1–J7 按右臂镜像复制，特别是 J5/J6/J7 的角度绝对值必须相同；此前导出的左后三轴沿用末端姿态反解，不符合这个规则。核对当前 `current_arm_training_20261011.json`、URDF 链、刚性装配代码和四孔连接件合同。
- **候选角度**：左侧采用 `[+,-,-,-,+,+,-]` 的符号约定并复制右侧绝对值，得到 `[84.27055,-68.29417,-61.39675,-89.05803,64.83433,27.92905,-7.30834]°`。固定当前已标定左基座，重新求解 Link7→手的安装变换；主导工具局部 Z 时钟约 `−89.9991°`，并包含 `[40.1638,17.4973,-2.9063] mm` 的连接件坐标补偿及小角度补偿。
- **实现／交付**：新增 `configs/preview/current_arm_training_left_mirror_palm_down_20261011.json` 与独立 `configs/preview/hand_connector_left_mirror_palm_down_20261011.json`；新增 `preview/scripts/export_left_mirror_mount_candidate.py`，在不覆盖批准四孔文件的前提下，使用生产 `arm_scene.augment_bundle` 构建候选并导出 `preview/outputs/arm_left_mirror_mount_candidate_20261011/` 的真实编译网格 Plotly HTML、真实网格 PNG、角度/安装面板、报告、候选输入和 ZIP。旧 `arm_mirror_palm_down_scene_20261011` 保留为基线并在说明中标明已过时。
- **验证／边界**：候选 Link7→手最大矩阵误差约 `1.16e-15`，左指尖平面倾角 `0°`，镜像 Link7 链最大位置误差约 `0.0069 mm`，物理步／策略／硬件命令均为0。候选尚未覆盖固定训练归档或默认配置；四孔紧固、法兰时钟和实机编码器零位必须硬件复核后才能晋升训练默认。无臂自碰撞与已修复的父子接触过滤合同未改。 :codex-annotation{index="1"}
## 2026-10-11 — 将当前接触积分器切换为 implicitfast

- **请求／修改**：用户要求把当前物理积分器从 `Euler` 替换为 `implicitfast`。
- **实际参考**：`tsensei_v3_2_pianist/assets/contact_settings.json`、`tsensei_v3_2_pianist/physics_settings.py`、当前接触验收说明。
- **修改位置**：接触配置与 Shadow reference 生成器的 `integrator` 均改为 `implicitfast`；同步 `docs/shadow_contact_40hz_acceptance.md` 的当前值表。
- **影响**：当前新编译场景的接触积分器发生变化，旧 Euler 场景、checkpoint、归档和历史结果不变，不能直接视为同一物理合同续训。
- **验证**：已核对源码中当前配置和 reference 均为 `implicitfast`；尚未重新编译 GPU scene、运行物理回归或启动训练。

