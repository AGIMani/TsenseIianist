"""Create portable hand-only copies of M6 bundles without changing the inputs."""
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from .bundle import SceneBundle, sha256
from .task_bundle import TaskBundle
from ..forearm_collision import collision_scope


def without_arms(scene_directory, task_directory, output):
    scene = SceneBundle(scene_directory)
    task = TaskBundle(task_directory, scene)
    if 'm6' not in scene.manifest:
        if scene.model.nmocap or 'forearm_collision_pairs' in task.manifest:
            raise ValueError('Unknown arm-free bundle layout; refusing to silently retain arm constraints')
        return scene.directory, task.directory
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    scene_out, task_out = out/'scene', out/'task'
    scene_out.mkdir(); task_out.mkdir()
    xml = ET.parse(scene.directory/'scene.xml')
    # M6 exporter appends prefixed assets and top-level mocap/support bodies.
    for section in ('asset', 'worldbody'):
        parent = xml.getroot().find(section)
        for node in list(parent):
            if node.get('name', '').startswith('tj_'):
                parent.remove(node)
    for node in xml.getroot().iter():
        if any(value.startswith('tj_') for value in node.attrib.values()):
            raise ValueError('Unexpected residual M6 XML reference')
    xml.write(scene_out/'scene.xml')
    for node in xml.getroot().iter():
        name = node.get('file')
        if name:
            if Path(name).name != name:
                raise ValueError('Expected flat frozen assets')
            shutil.copy2(scene.directory/name, scene_out/name)
    manifest = copy.deepcopy(scene.manifest)
    for key in ('m6', 'mocap_pos', 'mocap_quat'):
        manifest.pop(key, None)
    manifest['stage'] = 'hand-only scene; M6 removed with --no-arms'
    manifest['arms_removed_from_sha256'] = sha256(scene.directory/'manifest.json')
    manifest['limitations'] = [s for s in manifest['limitations'] if 'M6' not in s and 'TJ-ARM' not in s]
    manifest['files'] = {p.name: sha256(p) for p in scene_out.iterdir()}
    (scene_out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    target = copy.deepcopy(task.manifest)
    target.pop('forearm_collision_pairs', None)
    if len(target.get('forearm_collision_geoms', [])) != 2:
        raise ValueError('Original cross-hand collision groups are required')
    target['forearm_collision_scope'] = collision_scope(target.get('task_profile', 'tsensei_v1'))
    target['scene_manifest_sha256'] = sha256(scene_out/'manifest.json')
    target['cpu_reference_cases'] = []
    target['limitations'].append('Arm removal invalidates M6 rollout references; regenerate for comparisons')
    for name in target['files']:
        if name != 'cpu_reference.npz':
            shutil.copy2(task.directory/name, task_out/name)
    target['files'] = {p.name: sha256(p) for p in task_out.iterdir()}
    (task_out/'task.json').write_text(json.dumps(target, indent=2)+'\n')
    loaded = SceneBundle(scene_out)
    if loaded.model.nmocap:
        raise ValueError('Mocap bodies remain after arm removal')
    TaskBundle(task_out, loaded)
    return scene_out.resolve(), task_out.resolve()
