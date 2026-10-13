"""Versioned cross-hand collision coverage; the retired palm-only task profile is rejected."""
from functools import lru_cache
import json
from pathlib import Path

SCOPE = 'cross_hand_palms_and_fingers_v1'
TASK_PROFILES = ('tsensei_v1', 'tsensei_collision_v1')
FINGERS = ('thumb', 'index', 'mid', 'ring', 'pinky')


@lru_cache(maxsize=1)
def _housing_is_base():
    metadata = Path(__file__).resolve().parent / 'assets/model_derivation.json'
    schema = json.loads(metadata.read_text()).get('source_joint_schema')
    # The latest 20-output export puts the physical shell in ``base_link``
    # and keeps the fixed palm link as the mounting segment.  Keep the older
    # with-ID schema readable for archived source bundles.
    return schema in {'a32_with_id_semantic_chain', 'latest_project_20_outputs_withid_v1'}


def palm_housing_body_name(side):
    """Resolve the physical shell, whose vendor link name changed with ID."""
    return 'base_link' if _housing_is_base() else f'{side.capitalize()}_hand_palm_link'


def is_palm_collision_geom(name):
    local = name.rsplit('/', 1)[-1].lower()
    if _housing_is_base():
        return local.startswith('base_link_collision_')
    return '_hand_palm_' in local and '_collision_' in local


def collision_scope(task_profile='tsensei_v1'):
    if task_profile not in TASK_PROFILES:
        raise ValueError(f'Unknown task profile: {task_profile}')
    if task_profile == 'tsensei_collision_v1':
        return 'cross_hand_all_collision_geoms_v1'
    return SCOPE


def is_forearm_collision_geom(name, task_profile='tsensei_v1'):
    """Select physical hand colliders; the new profile also includes every mount segment."""
    collision_scope(task_profile)
    name = name.lower()
    if task_profile == 'tsensei_collision_v1':
        # Every URDF physical collider, including the base shell/mount.
        # Caller supplies geoms from a hand subtree, never piano/visual geoms.
        return '_collision_' in name
    return ('_collision_' in name and
            (is_palm_collision_geom(name) or any(f'_hand_{finger}_' in name for finger in FINGERS)))


def task_collision_groups(manifest):
    """Use the exported full scope; palm-only legacy tasks require archived code."""
    if 'forearm_collision_geoms' not in manifest:
        raise ValueError('Missing full collision scope; replay old tasks with their archived source')
    return manifest['forearm_collision_geoms']


def forearm_contact_reward(collision, task_profile='tsensei_v1'):
    """Shared CPU/Torch base rule; collision may be a bool or a float tensor."""
    collision_scope(task_profile)
    return -.5 * collision if task_profile == 'tsensei_collision_v1' else .5 * (1 - collision)
