# 四指耦合：独立50°–80°许可范围

同日后续更新：用户明确指定新双卡训练和本轮时序对照使用**60°–80°**，配置为`configs/finger_coupling/angle_ranges_60_80.json`。下方50°–80°数值为最初实现和验证记录；求解与合同机制相同，旧配置保留。当前入口另有[精确时序修复](note_timing_repair_tests_20261001.md)。

2026-10-01，按用户要求将新训练的耦合默认从单一60°目标改为每根非拇指独立的角度区间。

## 当前配置

`scripts/hpc/train_coupled_40hz.sh`显式读取
`configs/finger_coupling/angle_ranges_50_80.json`。该文件包含两手共8项：

| 手指 | 右手字段 | 左手字段 | 默认范围 |
| --- | --- | --- | --- |
| 食指 | right_index | left_index | [50, 80]° |
| 中指 | right_mid | left_mid | [50, 80]° |
| 无名指 | right_ring | left_ring | [50, 80]° |
| 小指 | right_pinky | left_pinky | [50, 80]° |

每项独立上下界，也允许左右手不同。修改一项不要求其他项同步；例如把`right_index`设成`[55, 75]`，其他7项仍可保留`[50, 80]`。文件必须包含全部8项，每项满足有限数值`0 < lower <= upper < 90`，布尔值、缺项、未知手指和反向区间均拒绝。

通用入口仍以`--finger-coupling`选择是否启用；启用后不提供角度参数时，各指默认50°–80°。HPC入口默认启用耦合，并保留完整观测。自定义范围参数为：

```bash
bash scripts/hpc/submit.sh train \
  --finger-coupling-angle-ranges /absolute/path/to/ranges.json
```

显式`--finger-coupling-angle 60`可选择原单一目标控制，优先于范围文件，便于覆盖包装脚本进行旧方案对照。`--no-finger-coupling`仍能关闭整个耦合控制。

## 控制与诊断

角度沿用既有定义：从DIP关节轴心指向包内指尖标记的末节轴线，相对静止琴面向下的夹角。它不等于DIP关节本身的弯曲角，标记仍有TODO(V32-07)真实指垫标定限制。

每根手指独立计算区间误差：

`error = max(lower − angle, 0) + max(angle − upper, 0)`

因此50°、60°、70°和80°都满足默认许可范围。控制器为每个候选PIP解析求取两个角度边界的DIP解，同时考虑origin、实际DIP、物理端点及方向极值。区间内按PIP/DIP相对origin的位移平方和选择姿态，保留可行的区间内部解。仍采用17点PIP搜索和9点局部细化，0.05°为数值可行容差。找不到区间内候选时选越界误差较小的有界解，并报告`coupling_infeasible`。

目标误差和实际角度误差均改为到许可区间的距离，区间内为0。实际角度仍在控制周期末记录；`active`仍表示谱面指定用指，不表示已经接触琴键。旧单目标实验的误差是到60°的距离，不能直接与新区间误差比较。

从动目标继续每个控制帧更新，准备段和闲指也沿用此控制。拇指、四指MCP/SW策略控制及PIP/DIP物理驱动保留；范围选择不增加策略输出，HPC仍为31动作/2121观测、40/400Hz。范围属于从动目标的几何要求，实际接触、惯性和驱动滞后可能造成越界，不是对实际qpos的硬投影。奖励权重、dt换算、学习参数和腕部限速未由本次改动调整。

新配置使用`distal_angle_range_v1`，完整8项范围、求解器和观测合同保存至config/checkpoint。旧`distal_60deg_v1`配置、求解及检查点语义保留；跨模式或改变任一手指范围的恢复会被合同检查拒绝，应为新控制从头训练。

## 验证与未完成项

- CPU定向验证：28项通过、51个子测试通过。覆盖原生MuJoCo FK、左右手非平行关节轴、8根手指分别改范围、可行内部origin保留、角度边界误差、不可达有界回退、CLI、旧60°恢复和新范围合同检查。脚本`bash -n`通过。
- 修改前已存在的`test_time_horizons_budget_and_learning_units`失败：当前脚本使用`coupled_40hz_20260930.json`，测试仍期望gesture的`score_multiplier=10`，实际JSON缺省为原强度。首次基线为1失败/5通过；修改后同一失败仍存在。上述28项定向通过是在排除此既有失败项后获得，未改变奖励脚本或放宽该断言。
- Slurm **1357**仅提交`tests/test_coupled_40hz_integration.py`，检查新模式CUDA graph与直接求解一致、31/2121布局、45步物理执行、实际角度区间误差、dt奖励和局部reset。截至本次记录仍因资源等待而PENDING，不能宣称GPU验收已通过。
- 本次未启动长期训练，F1、触键瞬间实际夹角、碰撞及512环境吞吐仍待测试。TODO(V32-COUPLING-01/02)、TODO(V32-07)保留。

CPU复查命令：

```bash
MUJOCO_GL=disable PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
  training_mjlab/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_finger_coupling.py tests/test_coupling_checkpoint_contract.py \
  tests/test_hpc_40hz_entry.py tests/test_run_entry_controls.py \
  -k 'not test_time_horizons_budget_and_learning_units'
```

实际参考：`mjlab_env/finger_coupling.py`、`scene.py`、`task.py`、`rl_torch/train.py`、`checkpoint.py`、`coupling_evaluation.py`；HPC包装/提交/归档入口；`tests/test_finger_coupling*.py`、`test_coupling_checkpoint_contract.py`、`test_coupled_40hz_integration.py`、`test_hpc_40hz_entry.py`和历史耦合实验文档。
