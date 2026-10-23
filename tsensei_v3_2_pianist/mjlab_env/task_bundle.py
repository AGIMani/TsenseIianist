"""Offline Composer music export; runtime loading needs only NumPy and MuJoCo."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .bundle import SceneBundle, sha256
from ..forearm_collision import collision_scope, is_forearm_collision_geom, is_palm_collision_geom
from ..fingering_targets import (manifest_config, options_from_manifest, target_for_key,
                                THUMB_GLOBAL_INDICES)


RESET_ACTION_SEMANTICS = 'scene_neutral_physical_v1'


def _replace_initial_action_observation(observation, contract, physical_action):
    """Make the exported reset reference describe the target applied by physics.

    Composer's action-observation wrapper initializes its history with the
    BoundedArray lower bound.  The frozen scene actually resets actuator
    targets to ``neutral_action``.  Keep the reference useful by correcting
    only the newest reset frame; subsequent frames already contain applied
    controls.
    """
    frames = contract['environment']['frame_stack']
    fields = contract['observation_fields_before_stack']
    base_size = sum(field['size'] for field in fields)
    result = np.asarray(observation).copy().reshape(base_size, frames)
    offset = 0
    for field in fields:
        if field['path'] == 'action':
            if field['size'] != len(physical_action):
                raise ValueError('Action observation and frozen neutral action sizes differ')
            result[offset:offset + field['size'], -1] = physical_action
            return result.reshape(-1)
        offset += field['size']
    return result.reshape(-1)


def export_task(output, scene_directory, *, midi='Toy', lookahead=20, frame_stack=1,
                action_reward_observation=True, disable_forearm_reward=False,
                reference_steps=64, initial_buffer_time=0.0,
                build_reference=True, task_profile='tsensei_v1',
                release_tail_time=0., score_validation='legacy', exact_note_timing=False):
    collision_scope(task_profile)
    if type(exact_note_timing) is not bool:
        raise ValueError('exact_note_timing must be true/false')
    from ..score_validation import validate_mode
    validate_mode(score_validation)
    if isinstance(reference_steps, bool) or not isinstance(reference_steps, int) or reference_steps < 1:
        raise ValueError('reference_steps must be a positive integer')
    # No Composer/music/JAX imports on the GPU runtime side.
    from tsensei_v3_2_pianist.rl.environment import EnvironmentConfig, make_environment
    from tsensei_v3_2_pianist.suite.tasks import piano_with_tsensei_hands as task_source
    from mujoco_utils import collision_utils
    from .acceptance_cases import press_action

    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite task bundle: {out}')
    scene = SceneBundle(scene_directory)
    cfg = EnvironmentConfig(midi=str(midi), n_steps_lookahead=lookahead,
        control_timestep=scene.manifest['control_dt'],
        physics_timestep=scene.manifest['physics_dt'],
        gravity_compensation=scene.manifest['gravity_compensation'],
        initial_buffer_time=initial_buffer_time,
        release_tail_time=release_tail_time, score_validation=score_validation,
        frame_stack=frame_stack, action_reward_observation=action_reward_observation,
        disable_forearm_reward=disable_forearm_reward, task_profile=task_profile)
    env, raw, contract = make_environment(cfg)
    try:
        initial = env.reset()
        task = raw.task
        if task._disable_fingering_reward:
            raise ValueError('This GPU task requires complete fingering; supply an annotated score.')
        m, d = raw.physics.model.ptr, raw.physics.data.ptr
        # An independently frozen scene must describe exactly the same physical setup.
        for name in ('jnt_range', 'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm',
                     'actuator_forcerange', 'body_gravcomp', 'body_mass', 'body_inertia',
                     'geom_size', 'geom_pos', 'geom_quat', 'geom_solref', 'geom_solimp',
                     'geom_friction', 'geom_priority', 'jnt_solref', 'jnt_solimp',
                     'dof_armature', 'dof_damping', 'dof_frictionloss'):
            np.testing.assert_allclose(getattr(m, name), getattr(scene.model, name), atol=1e-10)
        np.testing.assert_allclose(d.qpos, scene.reset_qpos, atol=1e-10)
        np.testing.assert_allclose(d.ctrl, scene.reset_ctrl, atol=1e-10)
        if not np.isclose(m.opt.timestep, scene.manifest['physics_dt']):
            raise ValueError('CPU task and frozen scene physics timestep differ')
        length = len(task._notes)
        if length < 1:
            raise ValueError('Score must contain at least one control frame')
        goal = np.zeros((length, 89), dtype=np.float32)
        fingers = np.zeros((length, 10, 88), dtype=np.float32)
        for t, notes in enumerate(task._notes):
            for note in notes:
                if not 0 <= note.fingering < 10:
                    raise ValueError(f'Missing/invalid fingering at score frame {t}: {note.fingering}')
                goal[t, note.key] = 1
                fingers[t, note.fingering, note.key] += 1
            goal[t, -1] = task._sustains[t]
        score_arrays = dict(goal=goal, fingers=fingers)
        if exact_note_timing:
            from .exact_score import sequence_events, validate_events, physical_coverage
            events, sustains = sequence_events(task._midi.seq, initial_buffer_time,
                scene.manifest['control_dt'], task._music_end_step * scene.manifest['control_dt'])
            events, sustains = validate_events(events, sustains, length * scene.manifest['control_dt'])
            physical_counts = physical_coverage(events, length=length,
                control_dt=scene.manifest['control_dt'], physics_dt=scene.manifest['physics_dt'])
            score_arrays.update(exact_note_events=events, exact_sustain_events=sustains)
            task._score_coverage = dict(schema='exact_note_coverage_v1', valid=True,
                control_frame_coverage=task._score_coverage,
                time_origin='policy_start_including_buffer',
                interval='start_inclusive_end_exclusive',
                source_notes=len(events), covered_notes=len(events),
                control_dt=scene.manifest['control_dt'], physics_dt=scene.manifest['physics_dt'],
                notes=[dict(source_index=i, pitch=int(key) + 21, fingering=int(finger),
                            start_time=float(start), end_time=float(end), physical_target_samples=physical_counts[i])
                       for i, (start, end, key, finger) in enumerate(events)])
        np.savez_compressed(out/'score.npz', **score_arrays)
        if task._score_coverage is not None:
            (out/'score_coverage.json').write_text(json.dumps(task._score_coverage, indent=2)+'\n')
        key_geoms = [key.geom[0].full_identifier for key in task.piano.keys]
        hands = [task.right_hand, task.left_hand]
        joints = [[j.full_identifier for j in hand.joints] for hand in hands]
        sites = [s.full_identifier for hand in hands for s in hand.fingertip_sites]
        palms = [[g.full_identifier for g in hand.mjcf_model.find_all('geom')
                  if is_palm_collision_geom(g.name)] for hand in hands]
        collision_groups = [[g.full_identifier for g in hand.mjcf_model.find_all('geom')
                             if is_forearm_collision_geom(g.name, task_profile)] for hand in hands]
        terms = list(task.reward_fn.reward_fns)
        manifest = dict(schema_version=1, profile='cpu_compatible_fingered_v1',
            task_profile=task_profile,
            scene_manifest_sha256=sha256(scene.directory/'manifest.json'),
            scene_gravity_compensation=scene.manifest['gravity_compensation'],
            control_dt=scene.manifest['control_dt'], physics_dt=scene.manifest['physics_dt'],
            contract=contract, length=length, key_geoms=key_geoms, hand_joints=joints,
            fingertip_sites=sites, palm_geoms=palms, reward_terms=terms,
            fingering_target=manifest_config(task_profile),
            forearm_collision_scope=collision_scope(task_profile), forearm_collision_geoms=collision_groups,
            finger_tolerance=task_source._FINGER_CLOSE_ENOUGH_TO_KEY,
            key_tolerance=task_source._KEY_CLOSE_ENOUGH_TO_PRESSED,
            energy_coefficient=task._energy_penalty_coef,
            sustain_threshold=.5, collision_margin=1e-8,
            # The previous-action observation must match the control target
            # restored by Scene.reset.  The old Composer wrapper used the
            # action lower bound here even though physics reset to neutral.
            reset_action=scene.neutral_action.tolist(),
            reset_action_semantics=RESET_ACTION_SEMANTICS,
            reset_reward=float(raw.reward_spec().generate_value()),
            observation_base_size=sum(f['size'] for f in contract['observation_fields_before_stack']),
            limitations=['Fingered scores only; no OT reward',
                         'No score augmentation or randomized mounting in this profile',
                         'Explicit reset: caller must consume final transition before reset'])
        if exact_note_timing:
            from .exact_score import SCHEMA
            manifest['note_timing'] = SCHEMA
            manifest['limitations'].append('Exact MIDI rewards use physical sample timestamps; Composer key reward remains grid-based, so a full reference is not exported for this mode')
        if task._release_tail_steps:
            manifest['release_tail_steps'] = task._release_tail_steps
            manifest['music_end_step'] = task._music_end_step
        if score_validation != 'legacy':
            manifest['score_validation'] = score_validation
        geom_ids = np.array([m.geom(n).id for n in key_geoms])
        tip_ids = np.array([m.site(n).id for n in sites])
        joint_ids = [[m.joint(n).id for n in names] for names in joints]
        def snapshot():
            # Composer reset can replace MjData; never retain the pre-reset pointer.
            d = raw.physics.data.ptr
            target = np.empty((10, 88, 3), dtype=float)
            for finger in range(10):
                for key, geom_id in enumerate(geom_ids):
                    target[finger, key] = target_for_key(
                        m, d, geom_id,
                        thumb=finger in THUMB_GLOBAL_INDICES,
                        task_profile=task_profile,
                    )
            return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                right_qpos=d.qpos[m.jnt_qposadr[joint_ids[0]]].copy(),
                left_qpos=d.qpos[m.jnt_qposadr[joint_ids[1]]].copy(),
                key_state=task.piano.normalized_state.copy(),
                key_activation=task.piano.activation.copy(),
                sustain=task.piano.sustain_state.copy(),
                fingertips=d.site_xpos[tip_ids].copy(), key_targets=target,
                power=np.array(sum(np.sum(h.observables.actuators_power(raw.physics)) for h in hands)),
                # Keep the snapshot key for old reference/logic compatibility.
                palm_collision=np.array(collision_utils.has_collision(raw.physics, *collision_groups)))
        references = {}
        # Full Toy episodes and bounded long-score rollouts, without forcing key qpos.
        cases = ('neutral', 'press') if build_reference and not exact_note_timing else ()
        for case in cases:
            task._wrong_press_termination = False
            ts = env.reset()
            reset_observation = _replace_initial_action_observation(
                ts.observation, contract, scene.neutral_action)
            rows = [dict(**snapshot(), observation=reset_observation, reward=np.array(0.),
                         terms=np.zeros(len(terms)), discount=np.array(1.), last=np.array(False),
                         failure=np.array(False), completed=np.array(False), index=np.array(0),
                         action=scene.normalize(scene.neutral_action))]
            for t in range(min(length, reference_steps)):
                physical = scene.neutral_action.copy()
                if case != 'neutral' and t % 20 < 12:
                    physical = press_action(scene)
                    physical[-1] = float(t % 3)/2
                action = scene.normalize(physical)
                ts = env.step(action)
                rows.append(dict(**snapshot(), observation=ts.observation.copy(),
                    reward=np.array(ts.reward), terms=np.array([task.reward_fn.reward_terms[k] for k in terms]),
                    discount=np.array(ts.discount), last=np.array(ts.last()),
                    failure=np.array(task._failure_termination), completed=np.array(task._should_terminate),
                    index=np.array(task._t_idx), action=action))
                if ts.last():
                    break
            for key in rows[0]:
                references[case+'_'+key] = np.stack([row[key] for row in rows])
        np.savez_compressed(out/'cpu_reference.npz', **references)
        manifest['cpu_reference_cases'] = list(cases)
        manifest['files'] = {p.name:sha256(p) for p in out.iterdir() if p.is_file()}
        (out/'task.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest
    finally:
        env.close()


class TaskBundle:
    def __init__(self, directory, scene):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory/'task.json').read_text())
        m = self.manifest
        if m.get('schema_version') != 1 or m.get('profile') != 'cpu_compatible_fingered_v1':
            raise ValueError('Unsupported music task profile')
        if sha256(scene.directory/'manifest.json') != m['scene_manifest_sha256']:
            raise ValueError('Music task was exported for a different frozen scene; re-export task')
        for name, expected in m['files'].items():
            p = (self.directory/name).resolve()
            if p.parent != self.directory or sha256(p) != expected:
                raise ValueError(f'Task asset missing or modified: {name}')
        with np.load(self.directory/'score.npz', allow_pickle=False) as score:
            self.goal = score['goal'].copy()
            self.fingers = score['fingers'].copy()
            self.exact_note_events = self.exact_sustain_events = None
            if 'note_timing' in m:
                from .exact_score import SCHEMA, validate_events
                if m['note_timing'] != SCHEMA:
                    raise ValueError('Unsupported exact note timing contract')
                self.exact_note_events, self.exact_sustain_events = validate_events(
                    score['exact_note_events'], score['exact_sustain_events'], m['length'] * m['control_dt'])
            elif 'exact_note_events' in score or 'exact_sustain_events' in score:
                raise ValueError('Exact note events require an explicit timing contract')
        if self.goal.shape != (m['length'],89) or self.fingers.shape != (m['length'],10,88):
            raise ValueError('Invalid score shape')
        if not np.isfinite(self.goal).all() or not np.isfinite(self.fingers).all():
            raise ValueError('Nonfinite score data')
        if not np.isin(self.goal,[0,1]).all() or (self.fingers < 0).any():
            raise ValueError('Invalid key/finger encoding')
        if not np.array_equal(self.fingers.sum(axis=1)>0, self.goal[:,:88]>0):
            raise ValueError('Score has missing or inconsistent finger assignments')
        from ..score_validation import release_tail_steps, validate_mode
        environment = m.get('contract', {}).get('environment', {})
        mode = validate_mode(environment.get('score_validation', 'legacy'))
        if m.get('score_validation', 'legacy') != mode:
            raise ValueError('Score validation mode differs from task contract')
        self.release_tail_steps = release_tail_steps(environment.get('release_tail_time', 0.), m['control_dt'])
        if self.release_tail_steps != m.get('release_tail_steps', 0):
            raise ValueError('Release tail length differs from task contract')
        self.music_end_step = m['length'] - self.release_tail_steps
        if self.music_end_step != m.get('music_end_step', m['length']) or self.music_end_step < 1:
            raise ValueError('Invalid music end boundary')
        if self.release_tail_steps and (self.goal[self.music_end_step:].any() or self.fingers[self.music_end_step:].any()):
            raise ValueError('Release tail must have empty key, fingering and sustain targets')
        if mode == 'strict_v1':
            if 'score_coverage.json' not in m['files']:
                raise ValueError('Strict score validation requires its coverage report')
            report = json.loads((self.directory/'score_coverage.json').read_text())
            expected_schema = ('exact_note_coverage_v1' if self.exact_note_events is not None
                               else 'strict_note_coverage_v1')
            if report.get('schema') != expected_schema or not report.get('valid'):
                raise ValueError('Invalid score coverage report')
            if self.exact_note_events is not None:
                grid = report.get('control_frame_coverage', {})
                if (grid.get('schema') != 'strict_note_coverage_v1' or not grid.get('valid')):
                    raise ValueError('Shadow rewards require validated control-frame coverage; re-export task')
        from .preparation import is_policy
        config = getattr(scene,'preparation_options',None)
        self.preparation_steps = 0
        if is_policy(config):
            self.preparation_steps = round(config['duration']/scene.manifest['control_dt'])
            if (len(self.goal) <= self.preparation_steps
                    or np.any(self.goal[:self.preparation_steps])
                    or np.any(self.fingers[:self.preparation_steps])):
                raise ValueError('Policy preparation requires an exported empty score prefix covering its duration')
        if self.exact_note_events is not None:
            from .exact_score import physical_coverage
            if ((self.exact_note_events[:, 0] < self.preparation_steps * m['control_dt'] - 1e-10).any()
                    or (self.exact_note_events[:, 1] > self.music_end_step * m['control_dt'] + 1e-10).any()):
                raise ValueError('Exact notes must stay outside preparation and release tail')
            physical_coverage(self.exact_note_events, length=m['length'],
                control_dt=m['control_dt'], physics_dt=m['physics_dt'])
        task_profile = m.get('task_profile', 'tsensei_v1')
        collision_scope(task_profile)
        if m.get('contract', {}).get('environment', {}).get('task_profile', 'tsensei_v1') != task_profile:
            raise ValueError('Task profile differs from the exported environment contract')
        options_from_manifest(m.get('fingering_target'), task_profile)
        if 'thumb_target' in m.get('contract', {}).get('environment', {}):
            raise ValueError('Removed thumb target environment contract')
        if task_profile == 'tsensei_collision_v1':
            expected = [[scene.model.geom(i).name for i in range(scene.model.ngeom)
                         if (scene.model.geom(i).name or '').startswith(side + '_/')
                         and is_forearm_collision_geom(scene.model.geom(i).name, task_profile)]
                        for side in ('right', 'left')]
            if (m.get('forearm_collision_scope') != collision_scope(task_profile)
                    or m.get('forearm_collision_geoms') != expected
                    or any(not group for group in expected)):
                raise ValueError('TsenseI collision task requires all physical hand collision geoms')
        reset_action = np.asarray(m['reset_action'])
        if reset_action.shape != scene.neutral_action.shape or not np.isfinite(reset_action).all():
            raise ValueError('Invalid reset action')
        semantics = m.get('reset_action_semantics', 'legacy_action_lower_bound_v1')
        if semantics not in ('legacy_action_lower_bound_v1', RESET_ACTION_SEMANTICS):
            raise ValueError(f'Unsupported reset action semantics: {semantics}')
        if semantics == RESET_ACTION_SEMANTICS:
            # Arm augmentation changes virtual slide coordinates into zero
            # world displacements. Validate the source task against its original
            # frozen scene, never weaken the check to accept either convention.
            expected_reset = getattr(scene, 'task_source_neutral_action', scene.neutral_action)
            if not np.allclose(reset_action, expected_reset, rtol=0, atol=1e-10):
                raise ValueError('Reset action must match frozen scene neutral control')
