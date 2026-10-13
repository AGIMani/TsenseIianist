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
OUT = ROOT / 'results/left_wrist_compensation_20261013'
DELTA = np.array([.010, .0096, -.0265])
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
T = a['T_piano_left_hand']
corrected = T.copy()
corrected[:, :3, 3] += DELTA
rows = list(csv.reader(io.StringIO(source['both_hand_pose_in_piano_200Hz.csv'].decode())))
header = rows[0]
indices = [header.index('left_' + axis + '_m') for axis in 'xyz']
assert len(rows) - 1 == len(T)
for i, row in enumerate(rows[1:]):
    np.testing.assert_allclose([float(row[j]) for j in indices], T[i, :3, 3], rtol=0, atol=1e-14)
    for axis, j in enumerate(indices):
        row[j] = repr(float(corrected[i, axis, 3]))
with (OUT / 'both_hand_pose_in_piano_200Hz.csv').open('w', newline='') as f:
    csv.writer(f).writerows(rows)
left_indices = [0] + [header.index('left_' + k) for k in ('x_m','y_m','z_m','qw','qx','qy','qz')]
with (OUT / 'left_wrist_pose_in_piano_200Hz.csv').open('w', newline='') as f:
    csv.writer(f).writerows([[r[j] for j in left_indices] for r in rows])
payload = {k: a[k].copy() for k in ('time_s','T_piano_right_hand','right_quaternion_wxyz','left_quaternion_wxyz')}
payload['T_piano_left_hand'] = corrected
payload['compensation_piano_m'] = DELTA
for side in ('right', 'left'):
    mask = np.array(['forearm_' not in n for n in a[side + '_joint_names']])
    assert mask.sum() == 20
    for kind in ('position', 'velocity'):
        payload[side + '_finger_' + kind] = a[side + '_joint_' + kind][:, mask]
    payload[side + '_finger_names'] = a[side + '_joint_names'][mask]
np.savez_compressed(OUT / 'trajectory_compensated_200Hz.npz', **payload)
# Full-series checks: preserve rotation, timestamp, right pose, and every untouched CSV string.
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
    measured_actual_minus_desired_piano_mm=[-10,-9.6,26.5], correction_piano_mm=[10,9.6,-26.5],
    semantics='derived compensated wrist/base_link pose targets, not new physical measurements',
    assumption='constant translation error in the same piano frame as supplied trajectory, no orientation error',
    mount_rule='Let M=T_wrist_flange be the unchanged physical rigid mount. T_piano_flange=T_piano_wrist @ M. Compensate with D on the LEFT: (D @ T_piano_wrist) @ M = D @ T_piano_flange.',
    installation_numeric_value_required=True,
    installation_note='Current left mount and saved SDK flange transform are used in derive_t0.py; the recorded full rotation already includes initial yaw 0.15 rad; do not rotate again.',
    changed_fields=['left_x_m','left_y_m','left_z_m','T_piano_left_hand[:,:3,3]'],
    raw_joint_records='original/ contains unchanged simulation qpos/qvel including forearm TX/TY/Pitch. These alone do not reproduce the compensated pose.',
    checks=dict(input_manifest_verified=True, all_1831_translations_verified=True, orientations_unchanged=True, right_pose_unchanged=True, timestamps_unchanged=True, npz_roundtrip_exact=True),
    first_left_before_mm=(T[0,:3,3]*1000).tolist(), first_left_after_mm=(corrected[0,:3,3]*1000).tolist(),
    hardware_commands=0, arm_ik_or_path_validation=False)
(OUT / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')

print(json.dumps(meta,ensure_ascii=False,indent=2))
