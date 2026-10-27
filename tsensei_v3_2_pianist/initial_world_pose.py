"""2026-10-25: initial palm-base positions use world XYZ in metres.

World is the current piano placement frame. Virtual palm_base slide coordinates
remain internal: Tx -> world +Y, Ty -> world -X, Tz -> world +Z. World Z sets
the mount height and Tz starts at zero; palm_base_yaw is a joint angle in rad,
not a world Euler yaw. Preparation lift is applied after this origin pose.
"""
import math


DEFAULT_WORLD_POSE = dict(world_x=.14, right_world_y=.15, left_world_y=-.15,
                          world_z=.10, palm_base_yaw=math.pi / 6)
WORLD_POSE_CONTRACT = 'initial_palm_base_world_xyz_v1'


def validate(values):
    """Validate a complete world pose; never reinterpret legacy joint offsets."""
    if not isinstance(values, dict) or set(values) != set(DEFAULT_WORLD_POSE):
        raise ValueError('Initial pose requires world_x, right_world_y, left_world_y, '
                         'world_z and palm_base_yaw; legacy height/ty/yaw need their matching archive')
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid initial world pose {name}: {value!r}')
    pose = {name: float(value) for name, value in values.items()}
    # Original .18 m X mount and Ty [-.05,.15] preserve the world workspace.
    if not .03 - 1e-12 <= pose['world_x'] <= .23 + 1e-12:
        raise ValueError('world_x must lie in [0.03, 0.23] m (unchanged palm-base travel)')
    if pose['world_z'] <= 0:
        raise ValueError('world_z must be positive, in metres')
    if any(not -.6105 <= pose[side + '_world_y'] <= .6105 for side in ('right', 'left')):
        raise ValueError('Initial world Y must lie in [-0.6105, 0.6105] m')
    if not -math.pi / 6 <= pose['palm_base_yaw'] <= math.pi / 6:
        raise ValueError('palm_base_yaw must lie in [-pi/6, pi/6] rad; it is a joint angle')
    return pose


def slides_for_world(pose, side, mount_position):
    """Convert desired world origin to slides, preserving the installation XY."""
    if side not in ('right', 'left'):
        raise ValueError('Hand side must be right or left')
    return dict(palm_base_tx=pose[side + '_world_y'] - mount_position[1],
                palm_base_ty=mount_position[0] - pose['world_x'], palm_base_tz=0.)


def add_arguments(parser, defaults=None):
    """Add position options shared by current training and preparation scripts."""
    values = validate({**DEFAULT_WORLD_POSE, **(defaults or {})})
    parser.add_argument('--world-x', type=float, default=values['world_x'],
                        help='Initial palm-base world X for both hands, metres, before preparation lift')
    parser.add_argument('--right-world-y', type=float, default=values['right_world_y'],
                        help='Initial right palm-base world Y, metres')
    parser.add_argument('--left-world-y', type=float, default=values['left_world_y'],
                        help='Initial left palm-base world Y, metres')
    parser.add_argument('--world-z', type=float, default=values['world_z'],
                        help='Initial palm-base world Z for both hands, metres, before preparation lift')
    parser.add_argument('--palm-base-yaw', type=float, default=values['palm_base_yaw'],
                        help='Palm_Base_Yaw joint angle, radians; not a world Euler angle')


def reject_legacy_pose_arguments(arguments, parser):
    """Fail before heavy construction instead of silently forwarding old flags."""
    replacements = {'--height': '--world-z', '--ty': '--world-x (position = 0.18 - old Ty)',
                    '--yaw': '--palm-base-yaw', '--height-offset': '--world-z-offset',
                    '--depth-offset': '--world-x-offset'}
    for argument in arguments:
        key = argument.partition('=')[0]
        if key in replacements:
            parser.error(f'{key} was removed; use {replacements[key]}. '
                         'Initial positions are world coordinates in metres.')
