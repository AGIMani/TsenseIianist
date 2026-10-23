#!/usr/bin/env python3
"""Retarget an existing frozen music task to a scene with only a changed reset pose."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def derive(source_scene, source_task, target_scene, output):
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
    from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle

    source = SceneBundle(source_scene)
    task = TaskBundle(source_task, source)
    target = SceneBundle(target_scene)
    if source.manifest['gravity_compensation'] is not True or target.manifest['gravity_compensation'] is not True:
        raise ValueError('Both scenes must keep hand gravity compensation enabled')
    for key in ('action_names', 'action_min', 'action_max', 'joint_names',
                'actuator_names', 'physics_dt', 'control_dt', 'key_activation_fraction'):
        if source.manifest[key] != target.manifest[key]:
            raise ValueError(f'Scene contract changed beyond pose: {key}')
    old, new = source.model, target.model
    for key in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'nmesh', 'npair'):
        if getattr(old, key) != getattr(new, key):
            raise ValueError(f'Scene dimension changed: {key}')
    for key in ('body_quat', 'body_mass', 'body_inertia', 'body_gravcomp',
                'jnt_range', 'actuator_ctrlrange', 'actuator_gainprm',
                'actuator_biasprm', 'actuator_forcerange', 'geom_pos',
                'geom_quat', 'geom_size', 'geom_solref', 'geom_solimp',
                'geom_friction', 'dof_armature', 'dof_damping', 'dof_frictionloss'):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key), err_msg=key)
    if source.manifest['physics_options'] != target.manifest['physics_options']:
        raise ValueError('Physics options changed')
    if not np.any(old.body_pos != new.body_pos) or not np.any(source.reset_qpos != target.reset_qpos):
        raise ValueError('Expected mounting and joint reset pose changes')

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in task.manifest['files']:
        if name != 'score.npz':
            raise ValueError(f'Unexpected task asset: {name}')
        (output / name).write_bytes((task.directory / name).read_bytes())
    manifest = copy.deepcopy(task.manifest)
    manifest['scene_manifest_sha256'] = sha256(target.directory / 'manifest.json')
    manifest['files'] = {'score.npz': sha256(output / 'score.npz')}
    manifest['cpu_reference_cases'] = []
    manifest['pose_derivation'] = {
        'source_scene_manifest_sha256': sha256(source.directory / 'manifest.json'),
        'source_task_sha256': sha256(task.directory / 'task.json'),
        'note': 'Score, observation and rewards retained; mounting/reset pose changed.',
    }
    manifest['limitations'].append('Old CPU physical trajectories are invalid under the new reset pose')
    (output / 'task.json').write_text(json.dumps(manifest, indent=2) + '\n')
    TaskBundle(output, target)
    print(f'{output}: task contract preserved for the new gravity-compensated pose')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-scene', 'source-task', 'target-scene', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source_scene, args.source_task, args.target_scene, args.output)
