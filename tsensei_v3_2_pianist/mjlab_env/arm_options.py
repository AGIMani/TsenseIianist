"""Explicit, portable M6 follower inputs; validation needs no CUDA or Torch."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'preview/assets/tj_m6'
SIDES = ('right', 'left')
CONNECTOR = ROOT / 'configs/preview/hand_connector_20260930.json'

# CCS-696 V4.0's SDK couples the nominal J6/J7 position region.  These are
# the SI values reconstructed from ccs_m6_40.MvKDCfg/FxRobot.cpp and are kept
# in the resolved arm contract so online IK and replay share one convention.
ARM_CONTROLLER = 'wuji_cartesian_v1'
DEFAULT_INITIAL_DEG = {
    'right': [-84.27055, -68.29417, 61.39675, -89.05803, -64.834334, 27.92905, -7.30834],
    'left': [84.27055, -68.29417, -61.39675, -89.05803, 64.834334, 27.92905, -7.30834],
}
CCS696_COUPLING = dict(
    constraint='ccs696_j6_j7_absolute',
    j6_index=5,
    j7_index=6,
    coefficient=1.025,
    sum_limit_rad=math.radians(110.5),
    j6_abs_limit_rad=math.radians(60.),
    j7_abs_limit_rad=math.radians(90.),
)


def connector_sources():
    """Pin the approved four-hole assembly, including every displayed part."""
    config = json.loads(CONNECTOR.read_text())
    names = [str(CONNECTOR.relative_to(ROOT))]
    for side in SIDES:
        for name in config['meshes'][side]:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != config['mesh_sha256'][name]:
                raise ValueError('Connector mesh differs from saved four-hole assembly: ' + name)
            names.append(name)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def add_arguments(parser):
    parser.add_argument('--arms', action='store_true', help='Enable ROSELIA-style Cartesian M6 followers')
    for side in SIDES:
        parser.add_argument(f'--arm-initial-{side}-deg', nargs=7, type=float,
                            default=None, metavar='ANGLE',
                            help=f'{side} SDK/URDF J1..J7 initial angles in degrees')
    parser.add_argument('--arm-ik-backend', choices=('torch', 'numba_cpu'), default='torch',
                        help='IK execution backend; same controller and hard constraints')
    parser.add_argument('--arm-mount', type=Path, default=None,
                        help='Explicit link7_to_hand transforms JSON; defaults to the current four-hole connector assembly')


def from_args(args):
    if not args.arms:
        if args.arm_mount is not None:
            raise ValueError('Arm parameters require --arms')
        return None
    mount = args.arm_mount or CONNECTOR
    mount_raw = Path(mount).read_bytes()
    if args.arm_mount is None:
        from ..preview_connector import load_config, tool_transforms
        from scipy.spatial.transform import Rotation
        transforms = tool_transforms(load_config(CONNECTOR))
        tools = {side: dict(xyz_m=matrix[:3, 3].tolist(),
                           rpy_rad=Rotation.from_matrix(matrix[:3, :3]).as_euler('xyz').tolist())
                 for side, matrix in transforms.items()}
    else:
        tools = json.loads(mount_raw)['link7_to_hand']
    limits_raw = (ROOT / 'configs/preview/m6_sdk_joint_limits.json').read_bytes()
    sdk = json.loads(limits_raw)
    result = dict(controller=ARM_CONTROLLER, execution_backend=getattr(args, 'arm_ik_backend', 'torch'), mode='cartesian_palm_position',
                  action_frame='world', action_channels=('TX', 'TY', 'TZ'),
                  position_bounds_m={'x': [-.025, .025], 'y': [-.10, .10], 'z': [-.025, .025]},
                  orientation='fixed_installed_palm_down', initial_q_rad={}, link7_to_hand={},
                  joint_limits={}, source_sha256={}, cartesian_ik=True,
                  coupling=deepcopy(CCS696_COUPLING),
                  failure_policy='bounded_reference', observation='q_v_reference_pose_error_each_side',
                  soft_limit_ratio=.95, full_penalty_excess=.1, maximum_penalty_per_step=5.,
                  penalty_aggregation='physics_substep_max_per_control_step',
                  iterations=12, orientation_weight_m=.15, damping=.008)
    for side in SIDES:
        angles = getattr(args, f'arm_initial_{side}_deg') or DEFAULT_INITIAL_DEG[side]
        if len(angles) != 7 or any(not math.isfinite(v) for v in angles):
            raise ValueError(f'{side}: seven finite initial angles required')
        urdf = ASSETS / side / 'arm.urdf'
        result['source_sha256'][str(urdf.relative_to(ROOT))] = hashlib.sha256(urdf.read_bytes()).hexdigest()
        suffix = side[0].upper()
        joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
        rows = []
        for index, angle in enumerate(angles, 1):
            j = joints[f'Joint{index}_{suffix}'].find('limit')
            low, high = float(j.get('lower')), float(j.get('upper'))
            q = math.radians(angle)
            if not low <= q <= high:
                raise ValueError(f'{side} J{index}: initial angle outside bundled position limits')
            row = sdk['joints'][f'{side}_J{index}']
            rows.append(dict(min=low, max=high, velocity=math.radians(row['velocity']),
                             acceleration=math.radians(row['acceleration'])))
        q6, q7 = math.radians(angles[5]), math.radians(angles[6])
        if (CCS696_COUPLING['coefficient'] * abs(q6) + abs(q7)
                > CCS696_COUPLING['sum_limit_rad'] + 1e-12):
            raise ValueError(f'{side} J6/J7 initial angles violate CCS-696 coupling')
        result['initial_q_rad'][side] = list(map(math.radians, angles))
        result['joint_limits'][side] = rows
        result['link7_to_hand'][side] = deepcopy(tools[side])
    result['source_sha256']['configs/preview/m6_sdk_joint_limits.json'] = hashlib.sha256(limits_raw).hexdigest()
    result['source_sha256'].update(connector_sources())
    result['mount_sha256'] = hashlib.sha256(mount_raw).hexdigest()
    return canonical(result)


def canonical(value):
    if value is None:
        return None
    result = deepcopy(value)
    if result.get('execution_backend', 'torch') not in ('torch', 'numba_cpu'):
        raise ValueError('Unknown arm IK execution backend')
    if (result.get('controller') != ARM_CONTROLLER or result.get('mode') != 'cartesian_palm_position'
            or result.get('action_frame') != 'world'
            or tuple(result.get('action_channels', ())) != ('TX', 'TY', 'TZ')
            or result.get('orientation') != 'fixed_installed_palm_down'
            or result.get('failure_policy') != 'bounded_reference'
            or result.get('cartesian_ik') is not True):
        raise ValueError('Arm follower contract must use wuji_cartesian_v1')
    if result.get('coupling') != CCS696_COUPLING:
        raise ValueError('Arm follower requires the resolved CCS-696 J6/J7 coupling contract')
    if (result.get('soft_limit_ratio') != .95 or result.get('full_penalty_excess') != .1
            or result.get('maximum_penalty_per_step') != 5.
            or result.get('penalty_aggregation') != 'physics_substep_max_per_control_step'
            or result.get('observation') != 'q_v_reference_pose_error_each_side'):
        raise ValueError('Arm follower requires the confirmed 95% / 10% / independent -5 substep-max contract')
    sdk = json.loads((ROOT / 'configs/preview/m6_sdk_joint_limits.json').read_text())
    if sdk['units'] != {'velocity': 'deg/s', 'acceleration': 'deg/s^2'}:
        raise ValueError('Unexpected SDK motion units')
    for side in SIDES:
        q = result['initial_q_rad'].get(side)
        if not isinstance(q, list) or len(q) != 7 or any(type(x) not in (float, int) or not math.isfinite(x) for x in q):
            raise ValueError('无法开启机械臂训练：缺少有效初始关节角 ' + side)
        tool = result['link7_to_hand'][side]
        if set(tool) != {'xyz_m', 'rpy_rad'} or any(len(tool[k]) != 3 or any(
                type(x) not in (float, int) or not math.isfinite(x) for x in tool[k]) for k in tool):
            raise ValueError('Explicit finite Link7-to-wrist installation required: ' + side)
        rows = result['joint_limits'][side]
        if len(rows) != 7:
            raise ValueError('Seven arm joint limits required')
        for angle, row in zip(q, rows):
            if (set(row) != {'min', 'max', 'velocity', 'acceleration'}
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row.values())
                    or not row['min'] <= angle <= row['max'] or row['min'] >= row['max']
                    or row['velocity'] <= 0 or row['acceleration'] <= 0):
                raise ValueError('Invalid initial angle / arm limits')
        joints = {j.get('name'): j for j in ET.parse(ASSETS / side / 'arm.urdf').getroot().findall('joint')}
        for i, row in enumerate(rows, 1):
            position = joints[f'Joint{i}_{side[0].upper()}'].find('limit')
            expected = dict(min=float(position.get('lower')), max=float(position.get('upper')),
                velocity=math.radians(sdk['joints'][f'{side}_J{i}']['velocity']),
                acceleration=math.radians(sdk['joints'][f'{side}_J{i}']['acceleration']))
            if row != expected:
                raise ValueError('Arm limits differ from bundled SDK/URDF: ' + side)
    for name in ('orientation_weight_m', 'damping', 'full_penalty_excess', 'maximum_penalty_per_step'):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError('Invalid arm option ' + name)
    if not 0 < result['soft_limit_ratio'] < 1 or type(result['iterations']) is not int or not 1 <= result['iterations'] <= 100:
        raise ValueError('Invalid arm soft ratio / iterations')
    # A saved recipe must retain its exact FK and SDK source; no fallback imports.
    expected_sources = set(connector_sources()) | {'configs/preview/m6_sdk_joint_limits.json'} | {
        f'preview/assets/tj_m6/{s}/arm.urdf' for s in SIDES}
    if set(result['source_sha256']) != expected_sources:
        raise ValueError('Arm source contract must identify URDFs, SDK and four-hole connector assembly')
    for name, digest in result['source_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Arm source differs from saved contract: ' + name)
    return result


def preflight(argv):
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args, _ = parser.parse_known_args(argv)
    return from_args(args)


if __name__ == '__main__':
    import sys
    try:
        preflight(sys.argv[1:])
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
