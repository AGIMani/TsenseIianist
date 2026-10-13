"""Offline recorded-trajectory translation; no model imports or hardware commands."""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile
import numpy as np

ROOT = Path('/home/hj/pianist')
SOURCE = Path('/home/hj/下载/trajectory_bundle.zip')
OUT = ROOT / 'results/right_wrist_compensation_20261013'
DELTA = np.array([.002, .014, -.014])
sha = lambda b: hashlib.sha256(b).hexdigest()
OUT.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(SOURCE) as z:
    source = {n: z.read(n) for n in z.namelist() if not n.endswith('/')}
for name, digest in json.loads(source['manifest.json']).items():
    assert sha(source[name]) == digest, name
# Retain all source artifacts as immutable evidence, clearly outside corrected outputs.
for name, content in source.items():
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    p = OUT / 'original' / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
a = dict(np.load(io.BytesIO(source['trajectory_200Hz.npz']), allow_pickle=False))
T = a['T_piano_right_hand']
corrected = T.copy()
corrected[:, :3, 3] += DELTA
rows = list(csv.reader(io.StringIO(source['both_hand_pose_in_piano_200Hz.csv'].decode())))
header = rows[0]
indices = [header.index('right_' + axis + '_m') for axis in 'xyz']
assert len(rows) - 1 == len(T)
for i, row in enumerate(rows[1:]):
    np.testing.assert_allclose([float(row[j]) for j in indices], T[i, :3, 3], rtol=0, atol=1e-14)
    for axis, j in enumerate(indices):
        row[j] = repr(float(corrected[i, axis, 3]))
with (OUT / 'both_hand_pose_in_piano_200Hz.csv').open('w', newline='') as f:
    csv.writer(f).writerows(rows)
right_indices = [0] + [header.index('right_' + k) for k in ('x_m','y_m','z_m','qw','qx','qy','qz')]
with (OUT / 'right_wrist_pose_in_piano_200Hz.csv').open('w', newline='') as f:
    csv.writer(f).writerows([[r[j] for j in right_indices] for r in rows])
payload = {k: a[k].copy() for k in ('time_s','T_piano_left_hand','left_quaternion_wxyz','right_quaternion_wxyz')}
payload['T_piano_right_hand'] = corrected
payload['compensation_piano_m'] = DELTA
for side in ('left', 'right'):
    mask = np.array(['forearm_' not in n for n in a[side + '_joint_names']])
    assert mask.sum() == 20
    for kind in ('position', 'velocity'):
        payload[side + '_finger_' + kind] = a[side + '_joint_' + kind][:, mask]
    payload[side + '_finger_names'] = a[side + '_joint_names'][mask]
np.savez_compressed(OUT / 'trajectory_compensated_200Hz.npz', **payload)
# Full-series checks: preserve rotation, timestamp, left pose, and every untouched CSV string.
np.testing.assert_array_equal(corrected[:, :3, :3], T[:, :3, :3])
np.testing.assert_allclose(corrected[:, :3, 3] - T[:, :3, 3], np.broadcast_to(DELTA, (len(T), 3)), rtol=0, atol=1e-15)
oldrows = list(csv.reader(io.StringIO(source['both_hand_pose_in_piano_200Hz.csv'].decode())))
assert all(old[j] == new[j] for old,new in zip(oldrows,rows) for j in range(len(header)) if j not in indices)
readback = np.load(OUT / 'trajectory_compensated_200Hz.npz', allow_pickle=False)
for k, v in payload.items():
    if v.dtype.kind in 'f': assert np.array_equal(readback[k], v, equal_nan=True)
    else: assert np.array_equal(readback[k], v)
meta = dict(
    source_zip_sha256=sha(SOURCE.read_bytes()), samples=len(T), sample_hz=200,
    measured_actual_minus_desired_piano_mm=[-2,-14,14], correction_piano_mm=[2,14,-14],
    semantics='derived compensated wrist/base_link pose targets, not new physical measurements',
    assumption='constant translation error in the same piano frame as supplied trajectory, no orientation error',
    mount_rule='Let M=T_wrist_flange be the unchanged physical rigid mount. T_piano_flange=T_piano_wrist @ M. Compensate with D on the LEFT: (D @ T_piano_wrist) @ M = D @ T_piano_flange.',
    installation_numeric_value_required=False,
    installation_note='Pure fixed-frame translation with unchanged orientation is independent of numeric mount transform. No flange trajectories are fabricated; downstream wrist-to-flange conversion must retain the existing actual mount.',
    changed_fields=['right_x_m','right_y_m','right_z_m','T_piano_right_hand[:,:3,3]'],
    raw_joint_records='original/ contains unchanged simulation qpos/qvel including forearm TX/TY/Pitch. These alone do not reproduce the compensated pose.',
    checks=dict(input_manifest_verified=True, all_1831_translations_verified=True, orientations_unchanged=True, left_pose_unchanged=True, timestamps_unchanged=True, npz_roundtrip_exact=True),
    first_right_before_mm=(T[0,:3,3]*1000).tolist(), first_right_after_mm=(corrected[0,:3,3]*1000).tolist(),
    hardware_commands=0, arm_ik_or_path_validation=False)
(OUT / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
(OUT / 'README.txt').write_text('''右手 wrist 轨迹固定平移补偿

测量解释：实际法兰比期望偏移钢琴系[-2,-14,+14]mm，因此命令反向补偿[+2,+14,-14]mm。
适用前提：测量与轨迹使用同一钢琴轴向，误差是固定平移且无姿态误差。单点测量不能证明误差在整条轨迹上恒定。

优先读取 right_wrist_pose_in_piano_200Hz.csv（米，四元数qw,qx,qy,qz）或 trajectory_compensated_200Hz.npz。
both_hand_pose_in_piano_200Hz.csv 保留原双手表结构，只修改右手XYZ。
1831帧，200Hz，0–9.15秒，前1秒准备段也使用相同补偿。朝向、时间、左手和20个手指的位置/速度不变。
NPZ保留双手位姿与20个手指轨迹；原23关节数组及原导出脚本/图只保存在original/，其中TX/TY/Pitch没有被伪造成三维补偿后的关节值。

wrist与法兰不重合：若M为wrist到法兰的固定安装矩阵，T_PF=T_PW*M。
本次T_PW_new=D*T_PW，故T_PF_new=D*T_PW*M=D*T_PF。
因此两原点都平移同一个钢琴系补偿向量，安装偏移和完整朝向仍保留；不能把D右乘到wrist局部坐标，也不能将wrist数值直接用作法兰位置。
数值安装矩阵不影响这次纯平移补偿，故没有用猜测的安装参数生成法兰文件。后续控制器仍须按实机安装矩阵转换wrist到法兰再解IK，避免重复补偿。

输出是离线补偿目标，不是新回放的实际状态，也不是可直接发送的机械臂J1-J7指令。未执行硬件或重新训练/回放，未做全路径可达性/碰撞验证。原包文件完整保留在original/。
''')
manifest = {str(p.relative_to(OUT)):sha(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
archive = OUT.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p, str(p.relative_to(OUT)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(sha(z.read(n)) == h for n,h in manifest.items())
print(json.dumps({'zip':str(archive),'checks':meta['checks'],'first_before_mm':meta['first_right_before_mm'],'first_after_mm':meta['first_right_after_mm']},ensure_ascii=False,indent=2))
