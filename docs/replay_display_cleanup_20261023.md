# 回放标签清理（2026-10-23）

用户截图来自 exper033 的交互回放。其部署仍有旧版 preview_fingering.py，打包器直接复制该文件，因此先前只修当前工作区未覆盖这个已导出包。此前验收没有检查附加几何的文字和形状，导致标签漏检。

scripts/replay_key_highlight.py 只生成目标琴键着色盒，不读取指尖位置，不生成 R1–R5/L1–L5、目标键文字或悬浮球。scripts/package_interactive_replay.py 固定将其安装到回放包，不再从训练源码复制显示辅助模块。实际触键绿色高亮、声音、相机和时间控制保留。训练模型、激活门槛、记录轨迹和评分不变。

scripts/interactive_recorded_replay.py 的 validate/render-check 检查全曲显示采样中的附加几何：必须是无文字的琴键盒；结果记录 display_contract、floating_label_count、annotation_sphere_count 和 display_samples_checked。本机 render-check 另保存 render_check.png。

修复覆盖本地11个独立回放包、远端8个回放包及21个部署目录；不改历史训练作业的冻结源码及归档，防止破坏其来源记录。既有录制、模型、音频、检查点未更换。历史包只更新显示代码；完整加载/渲染验收针对截图中的 exper033，其余包不声称已逐个播放验收。

远端两项回归通过，覆盖十指着色、实际触键优先、冲突分配和旧训练标签模块不得重新被打包。exper033 远端 MJB 加载/记录验证、本机受限 systemd scope 内 NVIDIA RTX3050 OpenGL 渲染均通过：24.3秒、200Hz、4861物理帧、487显示采样，文字和悬浮球均0。保留原回放单回合物理 micro-F1=0.916988331336794。

本次以默认 wrist_default031 归档为父，仅叠加显示脚本/测试及说明形成 source_replay_key_colors_v1 新归档；训练运行时文件逐字不变。统一入口仅切换归档和强校验SHA，已提交 exper034 作业配置不变，其部署导出打包器已经修复。证据和SHA保存在 logs/replay_labels_removed_20261023/。
