"""Read the bundled M6 SDK reference limits without importing a hardware driver."""
import hashlib
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
M6_LIMITS = ROOT / 'configs/preview/m6_sdk_joint_limits.json'


def load_m6_joint_limits(path=M6_LIMITS):
    """Return SI limits by explicit side/J1-J7 name, with source provenance.

    These are the supplied SDK configuration's nominal planning limits, not
    limits queried from a connected controller or a payload-dependent guarantee.
    """
    path = Path(path)
    raw = path.read_bytes()
    config = json.loads(raw)
    expected = {f'{side}_J{j}' for side in ('right', 'left') for j in range(1, 8)}
    if (config['schema'] != 1 or config['model'] != 'Marvin M6-CCS-696 V4.0' or
            config['units'] != {'velocity': 'deg/s', 'acceleration': 'deg/s^2'} or
            set(config['joints']) != expected):
        raise ValueError('Expected a complete M6 SDK joint limit snapshot in degree units')
    converted = {}
    for name, row in config['joints'].items():
        if set(row) != {'velocity', 'acceleration'} or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0 for value in row.values()):
            raise ValueError('SDK motion limits must be finite and positive: ' + name)
        converted[name] = {kind: math.radians(value) for kind, value in row.items()}
    return dict(schema=1, model=config['model'], source=config['source'],
                interpretation=config['interpretation'],
                snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                reference_kind='SDK configuration nominal planning limits',
                controller_limits_verified=False, ratio_applied=1.0,
                units={'velocity': 'rad/s', 'acceleration': 'rad/s^2'},
                joints=converted, sdk_units=config['units'], sdk_joints=config['joints'])


def limits_table_html(limits, folder):
    """Keep the report's visible values tied to the same data as its plot lines."""
    blocks = ['<p>红色虚线：SDK 配置正负限值；红点：超限采样。'
              '采用原时间轴，不裁剪尖峰。下表为绝对值上限，正负方向相同。'
              '这些是 SDK 规划配置参考值，未读取实物控制器。</p>',
              f'<p><a href="{html.escape(folder)}/joint_limits.json">限值数据与来源</a> · '
              f'<a href="{html.escape(limits["source"]["url"], quote=True)}">SDK 原配置</a></p>',
              '<table><thead><tr><th>关节</th><th>速度 rad/s（°/s）</th>'
              '<th>加速度 rad/s²（°/s²）</th></tr></thead><tbody>']
    for name, row in limits['joints'].items():
        sdk = limits['sdk_joints'][name]
        blocks.append(f'<tr><td>{html.escape(name)}</td>'
                      f'<td>{row["velocity"]:.6f}（{sdk["velocity"]:g}）</td>'
                      f'<td>{row["acceleration"]:.6f}（{sdk["acceleration"]:g}）</td></tr>')
    blocks.append('</tbody></table>')
    return '\n'.join(blocks)
