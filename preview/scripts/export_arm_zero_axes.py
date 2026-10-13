#!/usr/bin/env python3
"""Export the current local arm URDF zero pose and positive rotation convention."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'disable'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from online_control.models import build_scene_manifest, forward_kinematics
from preview.scripts.replay_arm_csv import build_scene
from preview.scripts.draw_arm_coordinate_frames import draw_meshes, draw_frame, camera_for, setup


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def visual_meshes(model, data, side):
    result = []
    for gid in range(model.ngeom):
        name = model.geom(gid).name or ''
        if not name.startswith((f'arm_{side}__', f'hand_{side}__')):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start+count].astype(float)
        start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        faces = model.mesh_face[start:start+count]
        # Illustration-only vertex clustering; full model remains unchanged.
        _, first, inverse = np.unique(np.round(vertices/.001).astype(np.int64), axis=0,
                                     return_index=True, return_inverse=True)
        faces = inverse[faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        vertices = vertices[first] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        result.append((vertices[faces]*1000, '#b8c4d1', name))
    return result


def positive_arc(ax, frame):
    angle = np.linspace(0, 2*np.pi/3, 45)
    local = np.column_stack([42*np.cos(angle), 42*np.sin(angle), np.full(len(angle), 9)])
    points = local @ frame[:3, :3].T + frame[:3, 3]*1000
    ax.plot(*points.T, color='#e88915', lw=2.8, zorder=45)
    tangent = frame[:3, :3] @ np.array([-np.sin(angle[-1]), np.cos(angle[-1]), 0])*12
    ax.quiver(*points[-1], *tangent, color='#e88915', lw=2.8, arrow_length_ratio=.7, zorder=45)


def figures(side, rows, meshes, output):
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.suptitle(f'{side.capitalize()} M6 arm | zero pose and positive joint rotation', fontsize=22, y=.975)
    fig.text(.5, .93, 'All J1-J7 = 0 rad | URDF joint frames | +q: right-hand rotation about local +Z', ha='center', fontsize=12)
    axes = [fig.add_subplot(2, 4, i+1, projection='3d', computed_zorder=False) for i in range(8)]
    all_points = np.concatenate([v.reshape(-1, 3) for v, _, _ in meshes])
    low, high = all_points.min(0), all_points.max(0)
    center = (low+high)/2
    radius = np.max(high-low)/2+50
    setup(axes[0], center, radius, (25, -40 if side == 'left' else 40))
    draw_meshes(axes[0], meshes, center, radius, alpha=.65)
    origins = np.array([r['origin_world_m'] for r in rows])*1000
    axes[0].plot(*origins.T, color='#40566b', lw=1.5, zorder=25)
    axes[0].set_title('Zero pose + fixed current hand pose', fontsize=12)
    axes[0].text2D(.02, .04, 'Joint origins may coincide; see individual panels.', transform=axes[0].transAxes, fontsize=9)
    for i, (row, ax) in enumerate(zip(rows, axes[1:]), 1):
        frame = np.array(row['T_world_joint_zero'])
        center = frame[:3, 3]*1000
        setup(ax, center, 105, camera_for([frame[:3, :3]]))
        draw_meshes(ax, meshes, center, 105, alpha=.22)
        draw_frame(ax, frame, f'J{i}', 60, label_origin=False)
        positive_arc(ax, frame)
        ax.set_title(f'J{i} | q = 0 rad', fontsize=14)
        axis = np.array(row['axis_world_at_zero'])
        ax.text2D(.5, .01, 'world +axis: ['+', '.join(f'{v:+.0f}' if abs(v)<1e-4 or abs(abs(v)-1)<1e-4 else f'{v:+.3f}' for v in axis)+']\napprox.; exact values in JSON/CSV', transform=ax.transAxes, ha='center', fontsize=9)
    fig.text(.5, .025, 'X red | Y green | Z blue (+joint axis) | orange arc: +X toward +Y (+q). Panel cameras differ.', ha='center', fontsize=12)
    fig.subplots_adjust(left=.015, right=.985, top=.875, bottom=.08, wspace=.03, hspace=.13)
    fig.savefig(output/f'zero_axes_{side}.png', dpi=160)
    fig.savefig(output/f'zero_axes_{side}.pdf', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    output = args.output
    output.mkdir(parents=True)
    (output/'sources').mkdir()
    manifest = build_scene_manifest()
    arms = [m for m in manifest['models'] if m['id'].startswith('arm_')]
    if {m['id'] for m in arms} != {'arm_left', 'arm_right'}:
        raise ValueError('Both current local arm models are required')
    zero = {m['id']: {j['name']: 0. for j in m['joints'] if j['type'] != 'fixed'} for m in arms}
    frames = forward_kinematics(manifest, zero, use_defaults=True)
    root, bodies, joints, initial = build_scene(manifest)
    model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
    data = mujoco.MjData(model)
    for key, joint in joints.items():
        data.qpos[model.joint(joint.full_identifier).qposadr[0]] = (0. if key[0].startswith('arm_') else initial[key])
    mujoco.mj_forward(model, data)
    report = dict(schema='arm_zero_axes.v1', units={'length':'m', 'angle':'rad'},
        zero_definition='URDF joint coordinate q=0 for every arm joint; NOT encoder homing or CAD appearance alignment',
        positive_definition='Right-hand rule about the URDF joint-local axis; all fourteen local axes are [0,0,1]. Viewed from +axis toward origin, +q is counterclockwise.',
        transform_definition='T_A_B maps B coordinates into A; column vectors; parent_T_child(q) = origin @ Rot(axis,q). URDF RPY = Rz(yaw) @ Ry(pitch) @ Rx(roll).',
        world_frame='Current scene manifest base frame, with piano at identity; distinct from each individual arm Base_L/R.',
        current_preview_initial_positions={m['id']:m['initial_positions'] for m in arms},
        zero_positions=zero, current_scene_warnings=manifest['warnings'], hardware_calibrated=False,
        arms={}, validation={}, source_sha256={})
    all_rows = []
    fk_error = axis_error = positive_error = 0.
    delta = 1e-5
    for arm in arms:
        side = arm['id'].removeprefix('arm_')
        path = ROOT/arm['source']['urdf']
        raw = ET.parse(path).getroot()
        base = np.array(arm['root_transform'])
        rows = []
        for i, joint in enumerate([j for j in arm['joints'] if j['type']!='fixed'], 1):
            element = raw.find(f"joint[@name='{joint['urdf_name']}']")
            origin = element.find('origin')
            frame = np.array(frames[arm['id']][joint['child']])
            local_axis = np.array(joint['axis']); local_axis /= np.linalg.norm(local_axis)
            axis_world = frame[:3, :3] @ local_axis
            bid = model.body(bodies[(arm['id'], joint['child'])].full_identifier).id
            actual = np.eye(4); actual[:3,3]=data.xpos[bid];actual[:3,:3]=data.xmat[bid].reshape(3,3)
            fk_error = max(fk_error,float(np.max(np.abs(actual-frame))))
            jid = model.joint(joints[(arm['id'],joint['name'])].full_identifier).id
            axis_error = max(axis_error,float(np.max(np.abs(data.xaxis[jid]-axis_world))))
            moved = copy.deepcopy(zero); moved[arm['id']][joint['name']] = delta
            plus = np.array(forward_kinematics(manifest,moved,True)[arm['id']][joint['child']])
            measured = Rotation.from_matrix(plus[:3,:3]@frame[:3,:3].T).as_rotvec()/delta
            positive_error = max(positive_error,float(np.max(np.abs(measured-axis_world))))
            # Independently check the same positive perturbation in native MuJoCo.
            adr=model.jnt_qposadr[jid];data.qpos[adr]=delta;mujoco.mj_forward(model,data)
            native=np.eye(4);native[:3,3]=data.xpos[bid];native[:3,:3]=data.xmat[bid].reshape(3,3)
            fk_error=max(fk_error,float(np.max(np.abs(native-plus))))
            data.qpos[adr]=0.;mujoco.mj_forward(model,data)
            row=dict(side=side, index=i, urdf_joint=joint['urdf_name'], scene_joint=joint['name'],
                supplied_csv_column=f'marvin_0_arm_joint_{i-1}_q_rad' if side=='left' else None,
                parent=joint['parent'], child=joint['child'], type=joint['type'], zero_rad=0.,
                origin_xyz_parent_m=np.fromstring(origin.get('xyz','0 0 0'),sep=' ').tolist(),
                origin_rpy_parent_rad=np.fromstring(origin.get('rpy','0 0 0'),sep=' ').tolist(),
                axis_joint_local=local_axis.tolist(), axis_arm_base_at_zero=(base[:3,:3].T@axis_world).tolist(),
                axis_world_at_zero=axis_world.tolist(), origin_world_m=frame[:3,3].tolist(),
                origin_arm_base_m=(np.linalg.inv(base)@frame)[:3,3].tolist(),
                T_parent_joint_zero=joint['origin'], T_arm_base_joint_zero=(np.linalg.inv(base)@frame).tolist(),
                T_world_joint_zero=frame.tolist(), limits=joint['limit'], positive_delta_validation_rad=delta)
            rows.append(row);all_rows.append(row)
        hand=next(m for m in manifest['models'] if m['id']==f'hand_{side}')
        report['arms'][side]=dict(urdf_robot_name=raw.get('name'), root_link=arm['root_link'],
            T_world_arm_base=base.tolist(), joints=rows,
            T_world_link7_zero=frames[arm['id']][arm['joints'][-1]['child']],
            T_link7_hand=hand['parent']['transform'],
            T_world_hand_zero=frames[hand['id']][hand['root_link']], fixed_hand_positions=hand['initial_positions'], source=arm['source'])
        figures(side,rows,visual_meshes(model,data,side),output)
        print(f'{side}: zero pose and positive directions exported',flush=True)
    assert fk_error<1e-10 and axis_error<1e-10 and positive_error<1e-8
    report['validation']=dict(independent_fk_max_matrix_error=fk_error,mujoco_axis_max_error=axis_error,
        positive_rotation_vector_max_error=positive_error, positive_delta_rad=delta,joints_checked=len(all_rows),
        no_dynamics_steps=True,render_backend='Matplotlib Agg CPU, MuJoCo GL disabled')
    files=[ROOT/'online_control/models.py',ROOT/'preview/assets/tj_m6/preview_mount.json',
        ROOT/'configs/preview/hand_connector_20260930.json',ROOT/'tsensei_v3_2_pianist/models/hands/tsensei_hand_constants.py',
        ROOT/'online_control/config/preview_pose.json',Path(__file__),ROOT/'preview/scripts/replay_arm_csv.py',
        ROOT/'preview/scripts/draw_arm_coordinate_frames.py']
    files += [ROOT/m['source']['urdf'] for m in manifest['models']]
    for path in files:
        report['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
        target=output/'sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    report['visual_mesh_sha256']={str(Path(path).relative_to(ROOT)):sha(Path(path)) for path in manifest['assets'].values()}
    write_json(output/'arm_zero_axes.json',report)
    write_json(output/'scene_manifest.json',manifest)
    with (output/'joint_axes.csv').open('w',newline='') as stream:
        fields=['side','index','urdf_joint','scene_joint','zero_rad']
        vectors=['origin_xyz_parent_m','origin_rpy_parent_rad','axis_joint_local','axis_arm_base_at_zero','axis_world_at_zero','origin_world_m']
        fields += [f'{key}_{axis}' for key in vectors for axis in 'xyz']+['lower_rad','upper_rad']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in all_rows:
            row={k:r[k] for k in fields[:5]}
            row.update({f'{key}_{axis}':r[key][i] for key in vectors for i,axis in enumerate('xyz')})
            row.update(lower_rad=r['limits']['lower'],upper_rad=r['limits']['upper']);writer.writerow(row)
    (output/'README.md').write_text('''# 当前机械臂零位和正方向

当前本地Marvin M6左右臂，J1→J7顺序。零位定义为七个URDF关节坐标全部0 rad，不代表实机编码器回零或硬件标定。与同事左臂CSV的joint_0…6映射为J1…J7。

每个关节正向均为其局部＋Z=[0,0,1]，按右手定则旋转；从＋轴端朝关节原点看，正转为逆时针。图中红／绿／蓝是局部＋X／＋Y／＋Z，橙色弧线由＋X转向＋Y表示＋q；各子图视角不同，不用屏幕左右判断关节符号。

零姿态下，世界系中的轴向近似：左臂[-Y,-Z,-Y,-Z,-Y,-Z,+X]，右臂[+Y,+Z,+Y,+Z,+Y,+Z,+X]。机械臂自身基座系与世界系不同；不要将此近似列表当作任意姿态下的固定世界轴。URDF的RPY保留原始精度，JSON/CSV中微小非零项未人为归零。

世界系采用当前scene manifest的base，钢琴在该系单位变换；各机械臂自身基座为Base_L/R。左／右基座原点分别为[0.52,-0.0465,0.48]／[0.52,+0.0465,0.48] m，基座绕世界X旋转＋90°／−90°。肩座／法兰变换仍待实机标定。

- zero_axes_left/right.png与PDF：全零姿态和每关节正向示意，左手／右手保留当前固定初始手指姿态。
- arm_zero_axes.json：每关节原始parent origin XYZ/RPY、局部轴、零位基座／世界轴、各4×4变换、限位、安装变换、来源SHA与CPU验证。
- joint_axes.csv：14关节表，长度m、角度rad；origin_rpy的x/y/z列分别代表roll/pitch/yaw。
- scene_manifest.json与sources/：本次实际配置／URDF／导出代码快照；sources未打包网格，并非可独立仿真的模型包。

T_A_B将B坐标映射到A坐标，使用列向量。FK为parent × origin × Rot(axis,q)，RPY为Rz(yaw)Ry(pitch)Rx(roll)。＋1e-5 rad扰动逐关节核对旋转向量符号，并与MuJoCo关节轴及连杆变换比较。全部使用CPU，无动力学推进、GPU、训练或硬件动作。
''')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>机械臂零位和正方向</title><style>body{font:16px sans-serif;max-width:1400px;margin:24px auto}img{width:100%}</style><h1>机械臂零位和正方向</h1><p>J1–J7全为0 rad；正转遵循右手定则绕局部＋Z。红X、绿Y、蓝Z、橙色弧线为＋q。基座／法兰仍待实机标定。</p><p><a href="README.md">完整说明</a> · <a href="arm_zero_axes.json">JSON数据</a> · <a href="joint_axes.csv">关节CSV</a></p>'+''.join(f'<h2>{side}</h2><img src="zero_axes_{side}.png"><a href="zero_axes_{side}.pdf">PDF</a>' for side in ('left','right')))
    deliverables=[p for p in output.rglob('*') if p.is_file()]
    write_json(output/'manifest.json',{str(p.relative_to(output)):sha(p) for p in deliverables})
    with zipfile.ZipFile(output/'arm_zero_axes_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in deliverables+[output/'manifest.json']:archive.write(path,path.relative_to(output))
    print(json.dumps(report['validation'],indent=2),flush=True)


if __name__=='__main__':
    main()
