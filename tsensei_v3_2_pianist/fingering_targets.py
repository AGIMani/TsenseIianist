"""Shared geometric definition of piano fingering-reward target points."""
from numbers import Real
import math
from .forearm_collision import collision_scope

DEFAULT_X_OFFSET_HALF_SIZE = 0.35
DEFAULT_Z_OFFSET_HALF_SIZE = 0.5
THUMB_DISTANCE_FROM_POSITIVE_X_END_M = {"white": 0.020, "black": 0.015}
THUMB_GLOBAL_INDICES = (0, 5)


def thumb_target_options(mode='profile', white_distance=None, black_distance=None):
    """Validate optional thumb-only distances, in metres from the +world-X end.

    Profile mode retains each historical task's target unless that key colour
    has an explicit distance. Uniform mode uses the non-thumb formula.
    """
    if mode not in ('profile', 'uniform'):
        raise ValueError('Thumb target mode must be profile or uniform')
    distances = {'white': white_distance, 'black': black_distance}
    for colour, value in distances.items():
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Thumb {colour} target distance must be finite and nonnegative')
            distances[colour] = float(value)
    if mode == 'uniform' and any(value is not None for value in distances.values()):
        raise ValueError('Uniform thumb targets cannot also specify edge distances')
    return {'mode': mode, 'white_distance': distances['white'], 'black_distance': distances['black']}


def validate_thumb_target(options):
    if options is None:
        return None  # A historical caller or source recipe keeps its old profile.
    if not isinstance(options, dict) or set(options) != {'mode', 'white_distance', 'black_distance'}:
        raise ValueError('Invalid thumb target options')
    return thumb_target_options(**options)


def _resolved_distances(task_profile, options):
    collision_scope(task_profile)
    options = validate_thumb_target(options)
    distances = (dict(THUMB_DISTANCE_FROM_POSITIVE_X_END_M)
                 if task_profile == 'tsensei_v1' else {'white': None, 'black': None})
    if options is not None:
        if options['mode'] == 'uniform':
            return {'white': None, 'black': None}
        for colour in ('white', 'black'):
            if options[colour+'_distance'] is not None:
                distances[colour] = options[colour+'_distance']
    return distances


def key_kind(geom_name):
    if "white_key_geom" in geom_name:
        return "white"
    if "black_key_geom" in geom_name:
        return "black"
    raise ValueError(f"Cannot classify piano key geom: {geom_name}")


def target_position(geom_position, geom_size, geom_name, *, thumb=False, task_profile='tsensei_v1',
                    thumb_target=None):
    """Return the target in world axes; +X is the bottom of the keyboard map."""
    # Source recipes and launch wrappers run in the system Python before
    # Composer/Torch environments are selected. Only geometry needs NumPy.
    import numpy as np
    collision_scope(task_profile)  # Validate the shared task contract.
    position = np.asarray(geom_position, dtype=float).copy()
    size = np.asarray(geom_size, dtype=float)
    if position.shape != (3,) or size.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(size).all():
        raise ValueError("Expected finite three-dimensional geom position and half-size")
    distances = _resolved_distances(task_profile, thumb_target)
    distance = distances[key_kind(geom_name)] if thumb else None
    if distance is not None:
        # Start at the key's +world-X end (the bottom in the documented plan
        # view), then move upward toward -X by the user-selected distance.
        if distance > 2 * size[0]:
            raise ValueError('Thumb target distance exceeds the key length')
        position[0] += size[0] - distance
    else:
        position[0] += DEFAULT_X_OFFSET_HALF_SIZE * size[0]
    position[2] += DEFAULT_Z_OFFSET_HALF_SIZE * size[2]
    return position


def manifest_config(task_profile='tsensei_v1', thumb_target=None):
    collision_scope(task_profile)
    options = validate_thumb_target(thumb_target)
    distances = _resolved_distances(task_profile, options)
    result = {
        "schema": 1,
        "coordinate_frame": "world_axes",
        "default": {
            "x_offset_half_size": DEFAULT_X_OFFSET_HALF_SIZE,
            "z_offset_half_size": DEFAULT_Z_OFFSET_HALF_SIZE,
        },
        "thumb": {
            "global_finger_indices": list(THUMB_GLOBAL_INDICES),
            "reference_edge": "positive_world_x_end",
            "distance_toward_negative_world_x_m": distances,
        },
    }
    if all(distance is None for distance in distances.values()):
        result['thumb'] = None
    if options is not None:
        result['schema'] = 2
        result['thumb_target_options'] = options
    return result


def options_from_manifest(manifest, task_profile='tsensei_v1'):
    """Validate a frozen target rule; absent options retain schema-1 semantics."""
    if not isinstance(manifest, dict):
        raise ValueError('Unsupported fingering target geometry')
    options = manifest.get('thumb_target_options') if manifest.get('schema') == 2 else None
    if manifest.get('schema') == 2 and options is None:
        raise ValueError('Fingering target schema 2 requires explicit thumb options')
    if manifest != manifest_config(task_profile, options):
        raise ValueError('Unsupported fingering target geometry')
    return options
