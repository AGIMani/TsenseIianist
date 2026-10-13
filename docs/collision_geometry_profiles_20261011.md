# 碰撞几何预览候选（2026-10-11）

`convex_hull_v1` 读取当前 URDF 的 collision 引用、缩放和原点，给原 capsule/box 部件生成完整 STL 的凸包，以保留电机壳体和指节外形的外边界。每手22个碰撞geom：掌壳 `base_link` 与拇指根部 `thumb_rota_block_link`、`thumb_rotaback_link2` 保留原始 mesh，其余19个使用三角化凸包。URDF没有独立的电机分类标签，这里的两段按链接名记录，不宣称是厂商确认的全部电机名单。

生成器是 `scripts/generate_collision_hulls.py`，配置是 `tsensei_v3_2_pianist/assets/collision_proxies_convex_hull.json`。38个凸包文件记录源SHA、输出SHA和面数；已验证源网格凸包顶点位于输出凸包内（容差0.1微米）和所有三角面朝外。它们是完整凸包去掉内部三角面后的简化，没有使用会削去边角的减面算法。视觉、惯量、关节和驱动保持不变。

**MuJoCo原生mesh接触仍按凸包计算。** 即使选择原始URDF mesh，也不代表非凸CAD逐三角面碰撞；单个凸包会填满孔洞和凹槽。要保留这些空腔须做多凸体分解，本次没有声称实现这一点。旧聊天二维图还使用了旧掌壳/安装座来源并遗漏掌壳原点平移；它只能作历史示意，不能用来判断当前装配。新候选以当前URDF为准。

## 预览参数

| 入口 / 参数 | 行为 |
| --- | --- |
| `preview_pose.py --collision-profile convex_hull_v1` | 切换到新凸包物理，并自动显示接触凸包、接触点和力箭头 |
| `preview_pose.py --real-collision-mesh` | 切换到原URDF mesh物理（legacy_mesh_v1），自动显示其实际接触凸包 |
| `preview_pose.py --show-hand-collisions` | 只显示当前模型的碰撞体、接触点和力箭头，不更换几何 |
| `preview_mjlab.py --show-hand-collisions` | 只显示已加载训练场景的真实碰撞几何；不改变旧checkpoint或物理合同 |

碰撞体为半透明橙色，视觉外壳降低透明度以便观察边界。Space暂停/继续、R复位；接触点和力箭头由原生viewer显示。显示开关不改变接触掩码和排除关系；`--include-assembly-collisions`是另一个会改变物理的开关，不默认开启。接触力属于模拟结果。

远程GPU预览仍通过Slurm入口执行，例如导出凸包静态图：

```bash
bash scripts/hpc/submit.sh preview --headless --no-arms --no-hand-connector \
  --no-contact-controls --collision-profile convex_hull_v1
```

在已有GPU分配和图形会话中，源姿态交互入口为：

```bash
.venv/bin/python preview/scripts/preview_pose.py --no-arms --no-hand-connector \
  --no-contact-controls --collision-profile convex_hull_v1 --paused
```

将 `--collision-profile convex_hull_v1` 换为 `--real-collision-mesh` 可比较原URDF；换成 `--show-hand-collisions` 可检查当前capsule/box。源姿态预览是独立几何实验，不等于同步MJLab训练回放。输出JSON记录所选profile、显示状态和初始穿透对。

## 验证与生效范围

本次为可选预览候选，当前默认训练入口/固定源码归档未切换，不宣称训练已生效。同步MJLab预览继续严格校验原源码/config，工作区有修改时会拒绝过期源码指纹；不覆盖历史指纹绕过校验。

CPU验收覆盖源顶点包含、面朝向、SHA、惯量/关节合同不变、真实碰撞显示开关与预览循环。尚无Slurm GPU接触轨迹/吞吐/收敛验收。现存早期静态PNG在显示修正前生成，不作为最终凸包接触验收证据。
