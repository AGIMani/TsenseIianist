#!/usr/bin/env python3
"""Read-only archived-policy reward audit, including same-state action branches.

GPU execution must go through scripts/hpc/submit.sh. This script never updates
weights, writes training checkpoints, or changes the archived reward function.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def finger_channels(names, finger):
    side = 'R' if finger < 5 else 'L'
    token = ('Thumb', 'Index', 'Mid', 'Ring', 'Pinky')[finger % 5]
    ids = [i for i, name in enumerate(names) if name.startswith(f'{side}_{token}_')]
    if not ids:
        raise ValueError('No independent policy controls for assigned finger')
    hand = [i for i, name in enumerate(names) if name.startswith(side + '_')]
    return ids, hand


def probe_frames(end, dt):
    """Three controls preceding a real MIDI offset, never a rounded note end."""
    import math
    last = math.ceil(end / dt - 1e-9) - 1
    return list(range(max(0, last - 2), last + 1))


def early_release_mask(active, goal, events, physics_dt):
    """Disjoint sample mask after the final in-target activation of each note."""
    import numpy as np
    active = np.asarray(active, dtype=bool).reshape(-1, 88)
    goal = np.asarray(goal, dtype=bool).reshape(-1, 88)
    times = (np.arange(len(active)) + 1) * physics_dt
    repairs = []
    for i, (start, end, key, finger) in enumerate(events):
        key = int(key)
        target = (times >= start - 1e-10) & (times < end - 1e-10)
        hits = np.flatnonzero(target & active[:, key])
        if not len(hits):
            continue
        mask = target & (np.arange(len(active)) > hits[-1]) & ~active[:, key]
        if mask.any():
            if not goal[mask, key].all():
                raise ValueError('Release repair crossed the actual score boundary')
            repairs.append(dict(note=i, key=key, reference=int(hits[-1]),
                                samples=np.flatnonzero(mask)))
    return repairs


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--source-archive', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--notes', type=int, nargs='+', default=[17, 33, 11, 2, 10, 31])
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--baseline-only', action='store_true',
                   help='Record actual rewards and score-only repair, without action branches')
    p.add_argument('--worker-root', type=Path, help=argparse.SUPPRESS)
    return p.parse_args()


def worker(a):
    sys.path.insert(0, str(a.worker_root))
    import numpy as np
    import torch
    import warp as wp
    from tsensei_v3_2_pianist.mjlab_env.paths import for_checkpoint
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.checkpoint import load_agent, reward_options

    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU diagnostic requires a Slurm allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((a.checkpoint.parent / 'config.json').read_text())
    metadata = torch.load(a.checkpoint, map_location='cpu', weights_only=True)['metadata']
    runpy.run_path(str(a.worker_root / 'scripts/eval_from_source.py'))['validate_config'](config, metadata)
    scene_path, task_path = for_checkpoint(a.checkpoint)
    options = reward_options(a.checkpoint)
    env = PianoTaskEnv(scene_path, task_path, device='cuda:0', num_envs=1,
                       collision_diagnostics=True, **{k: v for k, v in options.items()
                                                    if k != 'collision_diagnostics'})
    scene = env.scene
    # A labelled diagnostic path: no control-level graph pointers are created
    # or changed after a frozen state. Native MJLab step graphs remain enabled.
    scene.physics_graph = False
    load_options = dict(options, action_contract=scene.action_contract)
    for name in ('control_limits', 'finger_clearance', 'finger_coupling'):
        if options.get(name) is not None:
            load_options[name + '_contract'] = getattr(scene, name)
    if options.get('finger_coupling') is not None:
        load_options['coupling_observation_contract'] = env.coupling_observation_contract
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    agent = load_agent(a.checkpoint, scene_sha256=sha(scene_path / 'manifest.json'),
        task_sha256=sha(task_path / 'task.json'), obs_size=env.observation_size,
        action_size=env.action_size, device='cuda:0',
        preparation_contract=scene.bundle.preparation, **load_options)
    if env.exact_score is None:
        raise ValueError('This audit requires exact real MIDI targets')
    dt = float(env.bundle.manifest['control_dt'])
    pdt = float(scene.bundle.manifest['physics_dt'])
    events = env.exact_score.events
    names = config['policy_action_names']
    gamma = config['sac']['gamma']
    targets = {i: dict(index=i, start=float(events[i, 0]), end=float(events[i, 1]),
                     key=int(events[i, 2]), finger=int(events[i, 3])) for i in a.notes}
    scheduled = {}
    for i, note in targets.items():
        for frame in probe_frames(note['end'], dt):
            scheduled.setdefault(frame, []).append(i)

    class Frozen:
        """Save all Warp data arrays and mutable task/scene tensor ownership.

        Array values and the original attribute references are both restored:
        TaskLogic.history and action diagnostics can be rebound during step().
        A duplicate untreated branch is checked before accepting any contrast.
        """
        def __init__(self, observation):
            self.tensors = []
            self.arrays = []
            self.array_ids = set()
            self.seen_nodes = set()
            self.exact_checks = []
            self.expected = dict(qpos=scene.data.qpos[:].clone(),
                qvel=scene.data.qvel[:].clone(), index=env.logic.index.clone(),
                observation=env._with_preparation(env.logic.observation()).clone())
            if not torch.equal(self.expected['observation'], observation):
                raise RuntimeError('Task state does not reconstruct the observation at the branch origin')
            def visit(value, setter, descend=False):
                if isinstance(value, torch.Tensor):
                    self.tensors.append((value, value.clone(), setter))
                elif isinstance(value, wp.array):
                    if id(value) not in self.array_ids:
                        self.array_ids.add(id(value))
                        self.arrays.append((value, wp.clone(value)))
                elif isinstance(value, dict):
                    for k, v in list(value.items()):
                        visit(v, lambda x, d=value, key=k: d.__setitem__(key, x), descend)
                elif isinstance(value, list):
                    for k, v in enumerate(value):
                        visit(v, lambda x, d=value, key=k: d.__setitem__(key, x), descend)
                elif isinstance(value, tuple):
                    for v in value:
                        visit(v, lambda x: None, descend)
                elif descend and hasattr(value, '__dict__') and id(value) not in self.seen_nodes:
                    self.seen_nodes.add(id(value))
                    for k, v in vars(value).items():
                        visit(v, lambda x, obj=value, key=k: setattr(obj, key, x), True)
            torch.cuda.synchronize()
            with scene.scope():
                visit(scene.sim.wp_data, lambda x: None, True)
                if scene.self_query_data is not None:
                    visit(scene.self_query_data, lambda x: None, True)
                owners = [env, env.logic, env.reward_manager, scene]
                owners += [getattr(scene, n, None) for n in ('reward_sampling', 'self_collision',
                    'key_contact_diagnostics', 'control_limiter', 'finger_coupler', 'finger_clearance_guard')]
                for owner in owners:
                    if owner is not None:
                        for k, v in vars(owner).items():
                            visit(v, lambda x, obj=owner, key=k: setattr(obj, key, x))
            torch.cuda.synchronize()

        def restore(self):
            torch.cuda.synchronize()
            with scene.scope():
                for array, saved in self.arrays:
                    wp.copy(array, saved)
                for tensor, saved, setter in self.tensors:
                    tensor.copy_(saved)
                    setter(tensor)
            torch.cuda.synchronize()
            current = dict(qpos=scene.data.qpos[:], qvel=scene.data.qvel[:],
                index=env.logic.index, observation=env._with_preparation(env.logic.observation()))
            checks = {name: torch.equal(value, self.expected[name]) for name, value in current.items()}
            self.exact_checks.append(checks)
            if not all(checks.values()):
                raise RuntimeError('Frozen state was not restored exactly before stepping: ' + str(checks))

    def read_step(result, action, selected=None):
        active = scene.reward_sampling.key_activation_history[0].clone()
        goal = result.info['goal_substeps'][0, :, :88].bool()
        travel = scene.reward_sampling.key_state_history[0].clone()
        row = dict(frame=int(result.info['score_index'][0]),
            reward=float(result.reward[0]),
            terms={k: float(v[0]) for k, v in result.info['reward_terms'].items()},
            tp=int((active & goal).sum()), fp=int((active & ~goal).sum()),
            fn=int((~active & goal).sum()), action=action[0].cpu().tolist())
        if selected is not None:
            k = selected['key']
            row.update(target_key_samples=int(goal[:, k].sum()),
                       target_key_tp=int((active[:, k] & goal[:, k]).sum()),
                       target_key_fp=int((active[:, k] & ~goal[:, k]).sum()),
                       key_active=active[:, k].cpu().tolist(),
                       key_travel=travel[:, k].cpu().tolist(),
                       key_target=goal[:, k].cpu().tolist())
        return row

    def branch(start_obs, note, held, mode):
        obs = start_obs.clone()
        rows = []
        ids, hand_ids = finger_channels(names, note['finger'])
        first_q = None
        for offset in range(a.horizon):
            if bool(env.logic.done[0]):
                break
            time = int(env.logic.index[0]) * dt
            action = agent.act(obs, deterministic=True)
            if (mode == 'finger_one_step' and offset == 0 or
                    mode in ('finger_until_end', 'hand_until_end') and time < note['end'] - 1e-10):
                chosen = hand_ids if mode == 'hand_until_end' else ids
                action[:, chosen] = held[:, chosen]
            if first_q is None:
                first_q = agent.critic(obs, action).flatten().cpu().tolist()
            result = env.step(action)
            rows.append(read_step(result, action, note))
            obs = result.observation
        rewards = [row['reward'] for row in rows]
        term_names = rows[0]['terms'] if rows else {}
        return dict(mode=mode, steps=rows, critic_first_action=first_q,
                    reward_first=rewards[0], reward_sum=sum(rewards),
                    discounted_reward=sum(gamma**i * r for i, r in enumerate(rewards)),
                    term_sums={k: sum(row['terms'][k] for row in rows) for k in term_names},
                    target_tp=sum(r['target_key_tp'] for r in rows),
                    target_fp=sum(r['target_key_fp'] for r in rows),
                    all_tp=sum(r['tp'] for r in rows), all_fp=sum(r['fp'] for r in rows),
                    all_fn=sum(r['fn'] for r in rows),
                    final_qpos=scene.data.qpos[0].cpu().tolist())

    a.output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='same_state_early_release_audit_v1', checkpoint=str(a.checkpoint),
        checkpoint_sha256=sha(a.checkpoint), source_archive_sha256=sha(a.source_archive),
        source_validation=True, execution='single-env diagnostic without control-level physics graph',
        notes=targets, control_dt=dt, physics_dt=pdt, gamma=gamma, horizon_steps=a.horizon,
        baseline_only=a.baseline_only,
        untreated_repeat_tolerances=dict(qpos_absolute=1e-5, reward_absolute=1e-5,
            exact_before_step=['qpos', 'qvel', 'index', 'observation'],
            exact_after_step=['per_step_tp', 'per_step_fp', 'per_step_fn', 'selected_key_activation_bits']),
        policy_action_names=names, reward_config=config['reward_config'], probes=[],
        limitations=['Holding previous targets is an actuator command intervention, not ideal forced key contact.',
                     'Branches have identical saved state; later policy actions respond to altered observations.',
                     'Finite-horizon return and critic estimate are not an infinite-horizon causal training explanation.',
                     'Diagnostic execution may differ from canonical evaluation through floating-point contacts.'])
    baseline = []
    history_actions = []
    history_active = []
    history_travel = []
    history_goals = []
    with torch.no_grad():
        obs = env.reset()
        for frame in range(env.logic.length):
            for note_id in ([] if a.baseline_only else scheduled.get(frame, [])):
                note = targets[note_id]
                # Latest command at least two frames old that actually kept the
                # assigned key on in every physical sample. This prevents using
                # an already released command as the holding intervention.
                candidates = [j for j in range(max(0, frame - 8), max(0, frame - 1))
                    if history_active[j][:, note['key']].all()]
                if not candidates:
                    report['probes'].append(dict(frame=frame, note=note_id,
                        skipped='No fully active prior command in the last 200 ms'))
                    continue
                held_index = candidates[-1]
                held = history_actions[held_index]
                frozen = Frozen(obs)
                values = []
                for mode in ('policy', 'policy_repeat', 'finger_one_step', 'finger_until_end', 'hand_until_end'):
                    frozen.restore()
                    values.append(branch(obs, note, held, 'policy' if mode == 'policy_repeat' else mode))
                first, duplicate = values[:2]
                qpos_error = float(np.max(np.abs(np.asarray(first['final_qpos']) - duplicate['final_qpos'])))
                reward_error = max(abs(x['reward'] - y['reward'])
                    for x, y in zip(first['steps'], duplicate['steps']))
                bits_equal = all(x['key_active'] == y['key_active'] for x, y in zip(first['steps'], duplicate['steps']))
                counts_equal = all(all(x[k] == y[k] for k in ('tp', 'fp', 'fn'))
                    for x, y in zip(first['steps'], duplicate['steps']))
                accepted = qpos_error <= 1e-5 and reward_error <= 1e-5 and bits_equal and counts_equal
                repeat_noise = {name: abs(first[name]-duplicate[name]) for name in
                    ('reward_first', 'reward_sum', 'discounted_reward')}
                for variant in values[2:]:
                    variant['comparison_valid'] = accepted
                    variant['delta_from_policy'] = {name: variant[name]-first[name] for name in
                        ('reward_first', 'reward_sum', 'discounted_reward', 'target_tp', 'target_fp',
                         'all_tp', 'all_fp', 'all_fn')}
                    variant['reward_difference_exceeds_5x_repeat_noise'] = {
                        name: abs(variant[name]-first[name]) > max(1e-5, 5*noise)
                        for name, noise in repeat_noise.items()}
                item = dict(frame=frame, time=frame*dt, note=note_id, hold_reference_frame=held_index,
                    state_restore_validation=dict(qpos_max_difference=qpos_error,
                        reward_max_difference=reward_error, key_bits_equal=bits_equal,
                        all_key_counts_equal=counts_equal, exact_before_step=frozen.exact_checks,
                        passed=accepted, repeat_reward_noise=repeat_noise,
                        rejection_reason=None if accepted else 'Untreated repeat exceeded float tolerance or changed sampled key counts; do not treat this branch contrast as established.',
                        warp_arrays=len(frozen.arrays), tensor_slots=len(frozen.tensors)),
                    variants=[values[0], *values[2:]])
                report['probes'].append(item)
                frozen.restore()
                del frozen
                print('probe', frame, note_id, [(r['mode'], round(r['reward_sum'], 6), r['target_tp'], r['target_fp'])
                                               for r in item['variants']], flush=True)
            action = agent.act(obs, deterministic=True)
            result = env.step(action)
            baseline.append(read_step(result, action))
            history_actions.append(action.clone())
            history_active.append(scene.reward_sampling.key_activation_history[0].cpu().numpy().copy())
            history_travel.append(scene.reward_sampling.key_state_history[0].cpu().numpy().copy())
            history_goals.append(result.info['goal_substeps'][0, :, :88].cpu().numpy().copy())
            obs = result.observation
            if bool(result.done[0]):
                break
    report['baseline'] = baseline
    report['state_restore_all_passed'] = all(p.get('state_restore_validation', {}).get('passed', True)
                                              for p in report['probes'])
    report['totals'] = {k: sum(row[k] for row in baseline) for k in ('tp', 'fp', 'fn')}
    t = report['totals']
    report['exact_diagnostic_f1'] = 2*t['tp'] / (2*t['tp']+t['fp']+t['fn'])
    # An explicit score-only counterfactual isolates the key incentive without
    # pretending the modified key positions are physically executable.
    from tsensei_v3_2_pianist.mjlab_env.rewards import target_key_positive_reward, substep_false_press_penalty
    active_np = np.asarray(history_active)
    travel_np = np.asarray(history_travel)
    goal_np = np.asarray(history_goals)
    repairs = early_release_mask(active_np, goal_np, events, pdt)
    changed_travel = travel_np.copy().reshape(-1, 88)
    changed_active = active_np.copy().reshape(-1, 88)
    for repair in repairs:
        k, ids, previous = repair['key'], repair['samples'], repair['reference']
        changed_travel[ids, k] = travel_np.reshape(-1, 88)[previous, k]
        changed_active[ids, k] = True
    goal_tensor = torch.as_tensor(goal_np, dtype=torch.float32)
    target = goal_tensor[:, -1]
    performance = config['reward_config']['performance']
    def key_scores(travel, activation):
        snapshot = dict(key_state_history=torch.as_tensor(travel.reshape(travel_np.shape)),
                        key_activation_history=torch.as_tensor(activation.reshape(active_np.shape)),
                        exact_goal_history=goal_tensor)
        positive = target_key_positive_reward(snapshot, target, env.bundle.manifest['key_tolerance'],
            substeps=True, activation_weight=performance['target_key_activation_weight'])
        wrong = (snapshot['key_activation_history'] & ~goal_tensor.bool()).any(-1).float().mean(-1)
        scale = dt/config['reward_config']['reference_dt']
        key = (.5*positive + .5*(1-wrong))*scale
        false = substep_false_press_penalty(snapshot['key_state_history'], goal_tensor)*scale*config['reward_config']['terms']['false_press_penalty']['weight']
        return key.numpy(), false.numpy()
    actual_key, actual_false = key_scores(travel_np, active_np)
    repaired_key, repaired_false = key_scores(changed_travel, changed_active)
    actual_logged = np.asarray([row['terms']['key_press_reward'] for row in baseline])
    if not np.allclose(actual_key, actual_logged, atol=1e-7, rtol=0):
        raise ValueError('CPU same-state key reward reconstruction differs from actual GPU reward')
    delta = repaired_key - actual_key
    report['score_only_hold_counterfactual'] = dict(
        scope='Only early-release samples of still-target keys retain last actual active travel/activation; every other reward contribution fixed.',
        executable_policy=False, reconstructed_key_max_error=float(np.max(np.abs(actual_key-actual_logged))),
        early_release_samples=sum(len(r['samples']) for r in repairs),
        affected_notes=[dict(note=r['note'], samples=len(r['samples'])) for r in repairs],
        original_key_reward_sum=float(actual_key.sum()), repaired_key_reward_sum=float(repaired_key.sum()),
        total_reward_gain=float(delta.sum()), min_affected_frame_gain=float(delta[delta>0].min()) if (delta>0).any() else 0.,
        max_affected_frame_gain=float(delta.max()), negative_gain_frames=int((delta < -1e-7).sum()),
        false_press_reward_delta=float((repaired_false-actual_false).sum()),
        changed_frames=[dict(frame=int(i), actual_key=float(actual_key[i]), repaired_key=float(repaired_key[i]),
            total_delta=float(delta[i])) for i in np.flatnonzero(np.abs(delta)>1e-9)])
    np.savez_compressed(a.output / 'baseline_keys.npz',
        key_activation=np.asarray(history_active), key_travel=np.asarray(history_travel),
        exact_goal=np.asarray(history_goals), exact_events=events,
        control_dt=dt, physics_dt=pdt)
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('diagnostic complete', a.output, report['exact_diagnostic_f1'], flush=True)


def main():
    a = arguments()
    a.checkpoint = a.checkpoint.resolve()
    a.source_archive = a.source_archive.resolve()
    a.output = a.output.resolve()
    if a.horizon < 2:
        raise ValueError('Use at least two future steps to observe the release tradeoff')
    if a.worker_root:
        worker(a)
        return
    if (a.output / 'audit.json').exists():
        raise FileExistsError('Use a new output directory')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from eval_from_source_archive import extract_archive, validate_sources
    config = json.loads((a.checkpoint.parent / 'config.json').read_text())
    with tempfile.TemporaryDirectory(prefix='tsensei-early-release-audit-') as temporary:
        archive = Path(temporary)
        extract_archive(a.source_archive, archive)
        validate_sources(archive, config['source_runtime']['source_sha256'])
        for role in ('training', 'training_mjlab'):
            link = archive / role / '.venv'
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((ROOT / role / '.venv').resolve(), target_is_directory=True)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(archive), PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl')
        command = [str(ROOT / 'training_mjlab/.venv/bin/python'), str(Path(__file__).resolve()),
            '--worker-root', str(archive), '--checkpoint', str(a.checkpoint),
            '--source-archive', str(a.source_archive), '--output', str(a.output),
            '--horizon', str(a.horizon), '--notes', *map(str, a.notes)]
        if a.baseline_only:
            command.append('--baseline-only')
        subprocess.run(command, cwd=archive, env=env, check=True)


if __name__ == '__main__':
    main()
