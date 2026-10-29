# 右臂—钢琴标定交接包：选定九点解

**交接版本：`right_arm_piano_rigid_probe_20261026_selected_p2_p10`。** 使用 p2～p10，刚性探针沿法兰 +Z 偏移 156 mm。正确 p1 按用户选择不参与拟合，保留用于外区检验。本包是结果与接入资料，控制程序尚未自动加载这套完整外参。

| 同事需要做的事 | 文件 |
|---|---|
| 了解数据、误差、p1 取舍及与同事报告的差异 | [分析报告 ANALYSIS.md](ANALYSIS.md) |
| 获取矩阵方向、单位、算例、TCP 补偿及项目接入说明 | [使用说明 INTEGRATION.md](INTEGRATION.md) |
| 在代码中读取完整精度正逆变换 | [transform.json](transform.json) |
| 运行离线点转换与法兰目标算例 | [example.py](example.py) |
| 从关节角和 MDH 参数复算并验证结果 | [verify.py](verify.py) |
| 查阅记录、逐点误差和九点/十点对照 | [calibration_points.json](calibration_points.json)、[point_errors.csv](point_errors.csv)、[comparison.json](comparison.json) |

`transform.json` 只保存本次选定九点变换，钢琴→右臂基座使用 `T_B_from_P_mm`，反向使用 `T_P_from_B_mm`；以米为单位时选择对应 `_m` 字段。诊断用的十点统计不作为运行时配置。

| 指标 | 结果 |
|---|---:|
| 九点三维拟合 RMS | 0.569797 mm |
| 九点最大三维残差 | 1.006251 mm（p8） |
| 九点内部留一预测 RMS | 0.713413 mm |
| 正确 p1 未参与拟合的预测误差 | 2.094625 mm |

拟合区域为钢琴 X=−45～55 mm、Y=165.2～494.6 mm、Z=22.5 mm。这些残差反映本批接触数据与名义机械臂模型的一致性，尚无独立物理定位精度验证。

将整个文件夹复制给同事，在有 NumPy 的 Python 3 环境中运行：

```bash
python example.py
python verify.py
```

脚本不依赖原仓库路径或机器人 SDK。图表提供 [PNG](selected-fit.png)、[SVG](selected-fit.svg)、[PDF](selected-fit.pdf)，九点与十点对比见 [nine-vs-ten.png](nine-vs-ten.png)。
