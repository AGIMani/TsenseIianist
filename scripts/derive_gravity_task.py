#!/usr/bin/env python3
"""Reuse a frozen music task when only hand gravity compensation changes."""
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
    old_task = TaskBundle(source_task, source)
    target = SceneBundle(target_scene)
    if source.manifest['gravity_compensation'] is not False or target.manifest['gravity_compensation'] is not True:
        raise ValueError('Expected uncompensated source and compensated target')
    for key in ('action_names', 'action_min', 'action_max', 'joint_names',
                'actuator_names', 'physics_dt', 'control_dt', 'key_activation_fraction'):
        if source.manifest[key] != target.manifest[key]:
            raise ValueError(f'Scene contract changed beyond gravity compensation: {key}')
    for key in ('qpos', 'qvel', 'ctrl'):
        np.testing.assert_array_equal(source.manifest[key], target.manifest[key], err_msg=key)
    old, new = source.model, target.model
    for key in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'nmesh', 'npair'):
        if getattr(old, key) != getattr(new, key):
            raise ValueError(f'Scene dimension changed: {key}')
    for key in ('body_pos', 'body_quat', 'body_mass', 'body_inertia',
                'jnt_range', 'actuator_ctrlrange', 'actuator_gainprm',
                'actuator_biasprm', 'actuator_forcerange', 'geom_pos',
                'geom_quat', 'geom_size', 'geom_solref', 'geom_solimp',
                'geom_friction', 'dof_armature', 'dof_damping', 'dof_frictionloss'):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key), err_msg=key)
    if old.opt.timestep != new.opt.timestep or old.opt.integrator != new.opt.integrator or old.opt.impratio != new.opt.impratio:
        raise ValueError('Physics options changed')
    increased = (new.body_gravcomp == 1) & (old.body_gravcomp == 0)
    if not increased.any() or not np.array_equal(old.body_gravcomp[~increased], new.body_gravcomp[~increased]):
        raise ValueError('Unexpected body gravity compensation change')

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in old_task.manifest['files']:
        if name == 'cpu_reference.npz':
            continue  # The old physical trajectory is invalid under gravity compensation.
        (output / name).write_bytes((old_task.directory / name).read_bytes())
        copied.append(name)
    manifest = copy.deepcopy(old_task.manifest)
    manifest['scene_manifest_sha256'] = sha256(target.directory / 'manifest.json')
    manifest['scene_gravity_compensation'] = True
    manifest['contract']['environment']['gravity_compensation'] = True
    manifest['cpu_reference_cases'] = []
    manifest['files'] = {name: sha256(output / name) for name in copied}
    manifest['gravity_compensation_derivation'] = {
        'source_scene_manifest_sha256': sha256(source.directory / 'manifest.json'),
        'source_task_sha256': sha256(old_task.directory / 'task.json'),
        'changed_body_count': int(increased.sum()),
        'note': 'Score, observation, rewards and reset contract preserved; CPU physical trajectories must be regenerated.',
    }
    manifest['limitations'].append('CPU reference trajectories from uncompensated scene are not applicable')
    (output / 'task.json').write_text(json.dumps(manifest, indent=2) + '\n')
    TaskBundle(output, target)
    print(f'{output}: gravity compensation on {int(increased.sum())} bodies; score unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-scene', 'source-task', 'target-scene', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source_scene, args.source_task, args.target_scene, args.output)
