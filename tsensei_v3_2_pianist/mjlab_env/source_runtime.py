"""Rebuild MJLab inputs from source and a JSON recipe; no persistent scene cache.

Composer and Torch live in separate local Python environments. A temporary
compiler workspace bridges them and is deleted on exit. Checkpoints retain
only the complete recipe, dependency versions and source fingerprints.
"""
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
_LIVE_WORKSPACES = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    paths = [p for p in (ROOT/'tsensei_v3_2_pianist').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')]
    paths.extend((ROOT/'configs/actions').glob('*.json'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def dependency_versions():
    result = {}
    for role, relative, names in (
        ('composer', 'training/.venv/bin/python', ['mujoco','dm-control','dm-env','dm-env-wrappers','numpy','mujoco-utils','note-seq','protobuf']),
        ('learner', 'training_mjlab/.venv/bin/python', ['torch','mujoco','mujoco-warp','mjlab','warp-lang','numpy'])):
        code = 'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in '+repr(names)+'}))'
        result[role] = json.loads(subprocess.check_output([str(ROOT/relative), '-c', code], text=True))
    return result


def make_recipe(*, height=.10, ty=.04, yaw=0.5235987755982988, control_hz=20., physics_hz=200.,
                midi='TwinkleTwinkleRousseau', lookahead=None, frame_stack=1,
                initial_buffer_time=1., release_tail_time=0., score_validation='legacy', gravity_compensation=True,
                action_reward_observation=True, disable_forearm_reward=False,
                task_profile='tsensei_v1', collision_profile='convex_hull_v1',
                drive_profile='legacy_v1', preparation_time=0., preparation_lift=.02,
                preparation_mode='fixed_v1', exact_note_timing=True, wrist_pitch_range_scale=1.):
    from ..score_validation import validate_mode, release_tail_steps
    validate_mode(score_validation)
    if (isinstance(wrist_pitch_range_scale, bool) or not math.isfinite(wrist_pitch_range_scale)
            or not 0 < wrist_pitch_range_scale <= 1):
        raise ValueError('wrist_pitch_range_scale must be finite and in (0,1]')
    if type(exact_note_timing) is not bool:
        raise ValueError('exact_note_timing must be true/false')
    if preparation_mode not in ('fixed_v1', 'policy_v1'):
        raise ValueError('Unknown preparation mode')
    if (not math.isfinite(preparation_time) or not math.isfinite(preparation_lift)
            or preparation_time < 0 or preparation_lift < 0):
        raise ValueError('Preparation time and lift must be finite and nonnegative')
    if task_profile not in ('tsensei_v1','tsensei_collision_v1'):
        raise ValueError('Unknown task profile')
    if collision_profile not in ('legacy_mesh_v1','convex_hull_v1','shadow_primitive_v1'):
        raise ValueError('Unknown collision profile')
    if drive_profile not in ('legacy_v1','shadow_v1'):
        raise ValueError('Unknown drive profile')
    if not all(math.isfinite(v) for v in (height,ty,yaw,control_hz,physics_hz,initial_buffer_time)):
        raise ValueError('Pose, frequency and buffer values must be finite')
    if height <= 0 or not -.05 <= ty <= .15 or not -math.pi/6 <= yaw <= math.pi/6:
        raise ValueError('Invalid mounting height or wrist initial position')
    if control_hz <= 0 or physics_hz < control_hz or not math.isclose(physics_hz/control_hz, round(physics_hz/control_hz)):
        raise ValueError('Physics frequency must be an integer multiple of control frequency')
    release_tail_steps(release_tail_time, 1 / control_hz)
    if preparation_time and not math.isclose(preparation_time*control_hz,round(preparation_time*control_hz),abs_tol=1e-10):
        raise ValueError('Preparation time must contain whole control frames')
    if initial_buffer_time < 0 or not math.isclose(initial_buffer_time*control_hz,round(initial_buffer_time*control_hz)):
        raise ValueError('Buffer must be nonnegative and contain whole control frames')
    lookahead = round(control_hz/2) if lookahead is None else lookahead
    if lookahead < 0 or frame_stack < 1:
        raise ValueError('Invalid lookahead or frame stack')
    # External scores travel inside config.json instead of depending on an old path.
    import base64
    path = Path(midi)
    score = ({'name':path.name,'suffix':path.suffix,'sha256':digest(path),
              'base64':base64.b64encode(path.read_bytes()).decode()} if path.is_file() else None)
    return dict(schema=1, method='source_config_v1',
                pose=dict(height=height,ty=ty,yaw=yaw),
                model=dict(collision_profile=collision_profile,drive_profile=drive_profile,
                           **({'wrist_pitch_range_scale': wrist_pitch_range_scale}
                              if wrist_pitch_range_scale != 1 else {})),
                preparation=(dict(duration=preparation_time,lift=preparation_lift,trajectory='quintic_v1',
                                  **({'mode': preparation_mode} if preparation_mode != 'fixed_v1' else {}))
                             if preparation_time else None),
                environment=dict(midi=str(midi) if score is None else score['name'],
                    control_hz=control_hz,physics_hz=physics_hz,lookahead=lookahead,
                    frame_stack=frame_stack,initial_buffer_time=initial_buffer_time,
                    release_tail_time=release_tail_time,score_validation=score_validation,
                    gravity_compensation=gravity_compensation,
                    action_reward_observation=action_reward_observation,
                    disable_forearm_reward=disable_forearm_reward,task_profile=task_profile,
                    **({'exact_note_timing': True} if exact_note_timing else {})),
                external_score=score,source_sha256=source_files(),versions=dependency_versions())


def build(recipe):
    """Return temporary scene/task paths, verifying exact source and dependencies."""
    if recipe.get('schema') != 1 or recipe.get('method') != 'source_config_v1':
        raise ValueError('Unsupported source reconstruction config')
    current = source_files()
    if current != recipe['source_sha256']:
        different = sorted(k for k in current.keys() | recipe['source_sha256'].keys()
                           if current.get(k) != recipe['source_sha256'].get(k))
        raise ValueError('Replay source differs from training; restore the recorded source version: '+', '.join(different[:8]))
    if dependency_versions() != recipe['versions']:
        raise ValueError('Replay dependency versions differ from config.json')
    workspace = tempfile.TemporaryDirectory(prefix='tsensei-v32-runtime-')
    _LIVE_WORKSPACES.append(workspace)
    atexit.register(workspace.cleanup)
    directory = Path(workspace.name)
    config = directory/'source_config.json'
    config.write_text(json.dumps(recipe,indent=2)+'\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    env['TSENSEI_V32_EXPORT_POSE'] = json.dumps(recipe['pose'])
    # Never let a shell override silently change the saved reconstruction recipe.
    model = recipe.get('model', {})
    env['TSENSEI_V32_COLLISION_PROFILE'] = model.get('collision_profile','convex_hull_v1')
    env['TSENSEI_V32_DRIVE_PROFILE'] = model.get('drive_profile','legacy_v1')
    env.setdefault('MUJOCO_GL','egl')
    command = [str(ROOT/'training/.venv/bin/python'), '-m',
               'tsensei_v3_2_pianist.mjlab_env.source_runtime', str(config), str(directory)]
    result = subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:
        workspace.cleanup()
        raise RuntimeError('Source environment construction failed:\n'+result.stdout[-12000:])
    print('Environment rebuilt from source + config (temporary compiler workspace).',flush=True)
    return directory/'scene',directory/'task',config


def _compile(config, output):
    from .bundle import export_scene
    from .task_bundle import export_task
    recipe=json.loads(config.read_text()); e=recipe['environment']
    if any(k.startswith('thumb_target') or k.startswith('thumb_white') or k.startswith('thumb_black') for k in e):
        raise ValueError('Removed thumb target options; use current all-finger geometry')
    midi=e['midi']
    if recipe['external_score']:
        import base64
        score=recipe['external_score']; p=output/('score'+score['suffix'])
        p.write_bytes(base64.b64decode(score['base64'],validate=True))
        if digest(p)!=score['sha256']:raise ValueError('Embedded score checksum mismatch')
        midi=str(p)
    # Compiler subprocess only: never mutate shared source defaults.
    scale = recipe.get('model', {}).get('wrist_pitch_range_scale', 1.)
    if isinstance(scale, bool) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Invalid saved wrist Pitch range scale')
    if scale != 1:
        from ..models.hands import tsensei_hand_constants as hand_consts
        dof = list(hand_consts.FOREARM_DOFS['forearm_pitch'])
        dof[2] = tuple(value * scale for value in dof[2])
        hand_consts.FOREARM_DOFS['forearm_pitch'] = tuple(dof)
    export_scene(output/'scene',gravity_compensation=e['gravity_compensation'],
                 control_timestep=1/e['control_hz'],physics_timestep=1/e['physics_hz'],build_reference=False)
    if scale != 1:
        path = output/'scene/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['action_range_scales'] = {name: scale for name in ('R_Palm_Base_Pitch','L_Palm_Base_Pitch')}
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    preparation = recipe.get('preparation') or {}
    preparation_buffer = preparation.get('duration', 0.) if preparation.get('mode') == 'policy_v1' else 0.
    export_task(output/'task',output/'scene',midi=midi,lookahead=e['lookahead'],frame_stack=e['frame_stack'],
                initial_buffer_time=e['initial_buffer_time'] + preparation_buffer,
                release_tail_time=e.get('release_tail_time', 0.), score_validation=e.get('score_validation', 'legacy'),
                action_reward_observation=e['action_reward_observation'],
                disable_forearm_reward=e['disable_forearm_reward'],
                task_profile=e.get('task_profile','tsensei_v1'),
                build_reference=False,
                exact_note_timing=e.get('exact_note_timing', False))
    if recipe['external_score']:
        # Avoid temporary absolute filenames in the reproducible task checksum.
        p=output/'task/task.json';m=json.loads(p.read_text())
        m['contract']['environment']['midi']='embedded:'+recipe['external_score']['sha256']
        p.write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':
    import sys
    _compile(Path(sys.argv[1]),Path(sys.argv[2]))
