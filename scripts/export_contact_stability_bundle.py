#!/usr/bin/env python3
"""Derive a frozen bundle with tested finger inertia/gain; preserve task rules."""
import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(scene_path, task_path, output):
    import numpy as np
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
    from tsensei_v3_2_pianist.mjlab_env.task_bundle import TaskBundle
    from tsensei_v3_2_pianist.models.hands.tsensei_hand_constants import (
        PROVISIONAL_FINGER_ARMATURE, PROVISIONAL_FINGER_KP)

    source = SceneBundle(scene_path)
    task = TaskBundle(task_path, source)
    if 'm6' in source.manifest:
        raise ValueError('Choose a hand-only scene; this export does not change M6 physics')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a frozen bundle: {output}')
    m = source.model
    fingers = [m.joint(int(m.actuator_trnid[a, 0])).name for a in range(m.nu)
               if not m.actuator(a).name.startswith(('R_Wrist_', 'L_Wrist_'))]
    if len(fingers) != 40 or len(set(fingers)) != 40:
        raise ValueError('Expected exactly 40 independent finger hinges')
    if m.npair:
        raise ValueError('Expected default physical contacts; explicit pairs need separate review')
    scene_out, task_out = output/'scene', output/'task'
    scene_out.mkdir(parents=True)
    task_out.mkdir()
    for name in source.manifest['files']:
        if name not in ('scene.xml', 'cpu_reference.npz'):
            (scene_out/name).write_bytes((source.directory/name).read_bytes())
    xml = ET.parse(source.directory/'scene.xml')
    changed = []
    for joint in xml.getroot().find('worldbody').iter('joint'):
        if joint.get('name') in fingers:
            joint.set('armature', str(PROVISIONAL_FINGER_ARMATURE))
            changed.append(joint.get('name'))
    assert set(changed) == set(fingers)
    for actuator in xml.getroot().find('actuator'):
        if actuator.get('joint') in fingers:
            if actuator.tag != 'position':
                raise ValueError('Expected native position actuators')
            actuator.set('kp', str(PROVISIONAL_FINGER_KP))
    xml.write(scene_out/'scene.xml')
    manifest = copy.deepcopy(source.manifest)
    manifest['contact_stability'] = {
        'profile': 'shadow_finger_drive_v1',
        'finger_armature_kg_m2': PROVISIONAL_FINGER_ARMATURE,
        'finger_kp_nm_per_rad': PROVISIONAL_FINGER_KP,
        'joints': fingers,
        'contact_response': 'same_hand_cross_hand_hand_piano_physical',
        'source_scene_manifest_sha256': sha256(source.directory/'manifest.json'),
        'export_script_sha256': sha256(__file__),
        'note': 'Only finger armature and kp changed; pose, wrist gains, contacts and task rules inherited. '
                'V32-04: provisional drive parameters, not hardware calibration.',
    }
    manifest['limitations'].append('Contact stability profile has separate checks; old CPU trajectories not reused')
    manifest['files'] = {p.name: sha256(p) for p in scene_out.iterdir() if p.is_file()}
    (scene_out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    updated = copy.deepcopy(task.manifest)
    for name in updated['files']:
        if name != 'cpu_reference.npz':
            (task_out/name).write_bytes((task.directory/name).read_bytes())
    updated['scene_manifest_sha256'] = sha256(scene_out/'manifest.json')
    updated['source_task_manifest_sha256'] = sha256(task.directory/'task.json')
    updated['cpu_reference_cases'] = []
    updated['limitations'].append('Finger armature changed; historical CPU rollout is not a reference for this scene')
    updated['files'] = {p.name: sha256(p) for p in task_out.iterdir() if p.is_file()}
    (task_out/'task.json').write_text(json.dumps(updated, indent=2)+'\n')
    loaded = SceneBundle(scene_out)
    TaskBundle(task_out, loaded)
    for name in ('body_mass', 'body_inertia',
                 'actuator_forcerange', 'geom_contype', 'geom_conaffinity',
                 'geom_solimp', 'geom_solref', 'geom_friction', 'dof_damping', 'dof_frictionloss'):
        np.testing.assert_array_equal(getattr(loaded.model, name), getattr(m, name))
    np.testing.assert_array_equal(loaded.reset_qpos, source.reset_qpos)
    np.testing.assert_array_equal(loaded.reset_ctrl, source.reset_ctrl)
    for name in fingers:
        assert loaded.model.dof_armature[loaded.model.joint(name).dofadr[0]] == PROVISIONAL_FINGER_ARMATURE
    for aid in range(m.nu):
        if m.actuator(aid).name.startswith(('R_Wrist_', 'L_Wrist_')):
            np.testing.assert_array_equal(loaded.model.actuator_gainprm[aid], m.actuator_gainprm[aid])
            np.testing.assert_array_equal(loaded.model.actuator_biasprm[aid], m.actuator_biasprm[aid])
        else:
            assert loaded.model.actuator_gainprm[aid, 0] == PROVISIONAL_FINGER_KP
            assert loaded.model.actuator_biasprm[aid, 1] == -PROVISIONAL_FINGER_KP
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.scene, args.task, args.output)
