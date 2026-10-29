# 右臂—钢琴探针标定（2026-10-28 导入）

当前离线标定位于 `configs/calibration/right_arm_piano/transform.json`。
完整用户交付包保存在同目录，登记和来源哈希见 `configs/calibration/right_arm_piano_current.json`。
版本为 `right_arm_piano_rigid_probe_20261026_selected_p2_p10`，取 p2～p10 九点解。

`T_P_from_B_m` 将 SDK 右臂基座坐标转换到用户钢琴坐标；`T_B_from_P_m` 为其逆，
平移单位 m。应使用完整旋转和平移，不能只填写基座 XYZ。探针尖端位于法兰
`[0,0,156] mm`，仅用于采点，不是灵巧手的工具长度。手的安装关系继续由当前
四孔连接件配置提供，不能叠加旧手模型的安装变换或旧经验偏移。

本次在 pianist 中没有找到此前活动指尖标定的副本或读取入口，因此这里是登记
新的项目内标定来源；不把本项目的训练、静态预览或根据初始角反算的模拟基座
强行改成实机坐标。当前默认无臂训练及已冻结作业不读取该实测外参。
当前 TurkishB 回放目录的 `right_arm/inputs/calibration.json` 已切换到相同文件，
旧指尖标定和旧复拟合诊断已移除，旧机械臂派生回放需重新计算。
第三个项目 TsenseIianist_platform 的历史记录和硬件配置未修改。

数值检查：交付包 13 文件清单哈希通过；从采点关节角和 MDH 快照复算通过。
九点 RMS 0.569797 mm、最大残差 1.006251 mm；p1 留出误差 2.094625 mm。
这些是拟合和留出残差，不是全工作空间或换回手之后的独立物理精度保证。
源包的 `not_deployed` 状态保留原样以维持哈希；本次未部署硬件。

在有 NumPy 的 Python 环境执行：

```bash
python configs/calibration/right_arm_piano/verify.py
python configs/calibration/right_arm_piano/example.py
```

当前来源包不依赖旧手 URDF。新旧手约 8 mm 的根坐标定义差异应在当前手的
安装链中解释；探针外参本身不应用额外 8 mm 补偿。
