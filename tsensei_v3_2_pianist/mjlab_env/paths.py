"""New-training defaults plus hash-based lookup for existing frozen recordings."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT/'training_mjlab/outputs'
DEFAULT_SCENE = OUTPUTS/'scene_shadow_40hz400hz'
DEFAULT_TASK = OUTPUTS/'task_twinkle_shadow_40hz400hz'
CURRENT_GRAVCOMP_SCENE = OUTPUTS/'current_h10_ty04_yaw015_20hz200hz_gravcomp_20260924/scene'
CURRENT_GRAVCOMP_TASK = OUTPUTS/'current_h10_ty04_yaw015_20hz200hz_gravcomp_20260924/task'


def resolve_bundle(explicit, digest, filename, default):
    if explicit is not None:
        return Path(explicit)
    if digest is None:
        return default
    candidates = [default/filename, *sorted(OUTPUTS.rglob(filename))]
    for p in candidates:
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest:
            return p.parent
    raise FileNotFoundError(f'No matching frozen {filename}; pass the original --scene/--task explicitly')


def for_checkpoint(checkpoint, scene=None, task=None):
    import torch
    meta = torch.load(checkpoint, map_location='cpu', weights_only=True)['metadata']
    return for_recording(meta, scene, task)


def for_recording(meta, scene=None, task=None):
    if 'source_runtime' in meta:
        if scene is not None or task is not None:
            raise ValueError('Source-based checkpoints rebuild from their config; do not pass scene/task')
        from .source_runtime import build
        scene, task, _ = build(meta['source_runtime'])
        if meta.get('no_arms'):
            from .without_arms import without_arms
            scene, task = without_arms(scene, task, scene.parent/'hand_only_bundle')
        for path, key in ((scene/'manifest.json','scene_sha256'),(task/'task.json','task_sha256')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta[key]:
                raise ValueError('Source reconstruction differs from training: '+key)
        return scene, task
    return (resolve_bundle(scene,meta['scene_sha256'],'manifest.json',DEFAULT_SCENE),
            resolve_bundle(task,meta['task_sha256'],'task.json',DEFAULT_TASK))
