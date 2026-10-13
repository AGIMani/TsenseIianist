#!/usr/bin/env python3
"""Freeze hand gravity compensation while preserving an existing scene reset."""
import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def derive(source_scene, compensated_template, output):
    import mujoco
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
    from tsensei_v3_2_pianist.mjlab_env.acceptance_cases import scenarios

    source = SceneBundle(source_scene)
    template = SceneBundle(compensated_template)
    old, new = source.model, template.model
    if source.manifest['gravity_compensation'] is not False or template.manifest['gravity_compensation'] is not True:
        raise ValueError('Expected uncompensated source and compensated template')
    for key in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'nmesh', 'npair'):
        if getattr(old, key) != getattr(new, key):
            raise ValueError(f'Scene dimension changed: {key}')
    for key in ('body_pos', 'body_quat', 'body_mass', 'body_inertia', 'jnt_range',
                'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm',
                'actuator_forcerange', 'geom_pos', 'geom_quat', 'geom_size',
                'geom_solref', 'geom_solimp', 'geom_friction', 'dof_armature',
                'dof_damping', 'dof_frictionloss'):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key), err_msg=key)
    if source.manifest['physics_options'] != template.manifest['physics_options']:
        raise ValueError('Physics options changed')
    changed = (old.body_gravcomp == 0) & (new.body_gravcomp == 1)
    if not changed.any() or not np.array_equal(old.body_gravcomp[~changed], new.body_gravcomp[~changed]):
        raise ValueError('Unexpected gravity compensation values')
    body_names = {mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_BODY, int(i))
                  for i in np.flatnonzero(changed)}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in source.manifest['files']:
        if name not in ('scene.xml', 'cpu_reference.npz'):
            (output / name).write_bytes((source.directory / name).read_bytes())
    tree = ET.parse(source.directory / 'scene.xml')
    seen = set()
    for body in tree.getroot().iter('body'):
        if body.get('name') in body_names:
            body.set('gravcomp', '1')
            seen.add(body.get('name'))
    if seen != body_names:
        raise ValueError(f'Missing gravity-compensated bodies: {body_names - seen}')
    tree.write(output / 'scene.xml')
    manifest = copy.deepcopy(source.manifest)
    manifest['gravity_compensation'] = True
    manifest['gravity_compensation_derivation'] = {
        'source_scene_manifest_sha256': sha256(source.directory / 'manifest.json'),
        'template_scene_manifest_sha256': sha256(template.directory / 'manifest.json'),
        'changed_bodies': sorted(body_names),
        'note': 'Frozen pose, contacts, controls and assets retained; hand gravcomp enabled.',
    }
    manifest['files'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = SceneBundle(output)
    np.testing.assert_array_equal(result.model.body_gravcomp, new.body_gravcomp)
    np.testing.assert_array_equal(result.reset_qpos, source.reset_qpos)
    np.testing.assert_array_equal(result.reset_ctrl, source.reset_ctrl)
    reference = {}
    for case, actions in scenarios(result):
        state = result.reset_cpu()
        qpos = []
        for action in actions:
            state.ctrl[result.action_ids] = action[:-1]
            for _ in range(round(manifest['control_dt'] / manifest['physics_dt'])):
                mujoco.mj_step(result.model, state)
            mujoco.mj_forward(result.model, state)
            qpos.append(state.qpos.copy())
        reference[case + '_qpos'] = np.asarray(qpos)
        reference[case + '_action'] = actions
    np.savez_compressed(output / 'cpu_reference.npz', **reference)
    manifest['files']['cpu_reference.npz'] = sha256(output / 'cpu_reference.npz')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    SceneBundle(output)
    print(f'{output}: hand gravity compensation enabled on {len(body_names)} bodies')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-scene', 'compensated-template', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source_scene, args.compensated_template, args.output)
