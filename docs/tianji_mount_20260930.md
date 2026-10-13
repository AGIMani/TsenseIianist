# A3.2_tianji 新安装件

2026-09-30，按用户新附件 `A3.2_tianji.stp` 更新双手与 M6 的仿真装配。
本说明覆盖上一版 `arm_hand_connect.STEP` 的窄安装座和中央孔定位。

![安装细节](../preview/outputs/tianji_mount_20260930/installation_detail.png)

[完整双臂图](../preview/outputs/tianji_mount_20260930/assembled_arms.png)；
[装配数值](../preview/outputs/tianji_mount_20260930/alignment.json)。

当前姿态的[J1–J7、法兰与手根坐标系图及PDF](arm_coordinate_frames_20260930.md)已单独导出。

## 实际修改

- 保留新CAD的六件外部连接件实例变换，提取命名零件
  `A3_2_HOLDER_WITH_WIDE_CONNECTOR`，替换此前的窄腕部安装座。
- 新安装面为 **45.6×24.6 mm**，四孔中心间距 **40×19 mm**。
  连接板孔径4.3 mm，手侧孔径3.7 mm；只核对CAD孔轴同心，不由网格推断螺纹配合。
- 按四孔中心和安装面定位，保留手根坐标、指尖方向及全部关节树。
  大开口中心偏离四孔中心3.25 mm，不能沿用旧中央孔居中方案。
- 左右URDF的安装座网格改为 `wrist_mount_tianji_20260930.STL`；
  继续使用去除重复旧座的 `base_without_old_mount.STL`。手指几何、关节、惯性字段不改。
- 配置仍为 `configs/preview/hand_connector_20260930.json`，默认加载新版；
  面板保留平移、旋转、保存及恢复基准装配。六件外部连接件和M6仍仅运动学显示。
- 右侧连接件与安装座仍是左侧镜像推导，不能称为厂商正式右手装配。

STEP坐标与左手根的关系（m）：

```text
assembly_from_left_hand:
R = [[0,1,0], [1,0,0], [0,0,-1]]
t = [-0.004, -0.0135, -0.026]
```

Link7到手根沿用临时法兰面 `Y=-95 mm`，旋转仍为X轴+π/2：
左手平移 `[-13.5,-121,-4] mm`，右手 `[-13.5,-121,+4] mm`。
肩座中心和93 mm间距保持，初始手部世界位置保持，双臂重新求解IK。
配置中小于1e-6 mm的差异来自STL float32表示，不是实机测量精度。

## 来源、保存与复现

原附件及选定零件保存在 `preview/assets/a32_tianji_verified_20260930/`，源文件SHA256：
`91090eee6c2012f8c617f1d37d428d639963d9c4a607f2ef4dc2af7e933df8d3`。
左右运行连接件位于 `preview/assets/hand_connector/tianji_20260930/`。
运行网格、配置和来源散列均在项目内，预览不读取附件路径或其他项目。
原始142 MiB CAD不进入轻量Slurm包；已有预览打包规则递归包含新运行网格。

旧模型、配置、网格和来源清单备份为
`model_backups/before_tianji_mount_20260930.tar.gz`，旧连接件文件也保留。
`assets/connector_mount_revision.json`与`model_derivation.json`记录新来源、变换及散列。

首次离线提取可用CAD独立环境执行（输出目录必须不存在）：

```bash
training_mjlab/cad_tools/.venv/bin/python preview/scripts/import_hand_connector.py \
  --step preview/assets/a32_tianji_verified_20260930/A3.2_tianji.stp \
  --output /tmp/tianji-cad-new \
  --reference-product A3_2_0915-2233_ASM --skip-reference-mesh \
  --extract-product A3_2_HOLDER_WITH_WIDE_CONNECTOR
```

`scripts/install_tianji_mount.py`保存本次安装过程，检查当前版本和交互偏移，拒绝重复安装。
已有模型使用 `scripts/prepare_models.py`重建即可，无需再次处理CAD。

云端渲染仍经Slurm：

```bash
bash scripts/hpc/submit.sh preview --headless \
  --output /home/houj/projects/tsensei_v3_2_pianist/preview/outputs/tianji_mount_20260930/new_preview.png
```

有图形桌面时，原 `training/.venv/bin/python preview/scripts/preview_pose.py` 入口默认显示新装配。

## 验证与限制

- 7项测试通过：安装面尺寸、镜像绕序、双侧四孔/法兰一致性、关节与惯性保持、
  质量兼容、真实编译模型上的偏移/恢复/保存及并发修改拒绝。
- 临时目录从原始模型完整重建后，两URDF及四个修订网格逐字节一致；
  见 `preview/outputs/tianji_mount_20260930/rebuild_validation.json`。
- 左右手质量仍各0.952 kg。新安装座为闭合正体积网格，约4161.019 mm³。
  未依据体积猜测材料密度，惯量保留原报告估计值。
- Slurm **1119 COMPLETED / 0:0**，14秒，A100生成完整装配图。
  普通源码预览为138坐标/50驱动器/51动作，初态0接触；这不是训练31动作合同的验收。
  左右腕部位置IK残差均小于1e-6 mm，仅表示数值求解误差。
- 首次整体三角化参考手时，OCP报告5542个面无三角网格；该参考手未用于安装，
  已改为按命名零件提取安装座。首次摘要保存在`initial_import_audit.json`。
- 外部连接板`part_06`有8条四面共边，0条开放边，严格网格流形检查未通过；
  位于中间孔交会处，不在四孔安装面。保留CAD显示几何，不用于质量估计或实体接触。
  其余五个外部零件及手侧安装座闭合，见`cad_solid_validation.json`。

TODO(V32-MOUNT-01)：实机孔/螺纹配合、法兰转角、工具坐标与更换座后的质量惯量标定；
连接板进入实体碰撞/质量计算前需处理非流形边。
TODO(TJ-ARM-02/03/04)：临时肩座、真实机械臂动力学和避障待验收。
本轮未运行真机、策略或新训练；旧检查点仍须匹配旧源码归档，不能静默换用新几何。
