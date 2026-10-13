#!/usr/bin/env python3
"""Four equal-budget, networks-only Twinkle refinements on retained GPU pairs.

This coordinator is CPU-only. GPU execution always uses submit.sh. The complete
R0 smoke config is the runtime contract, checked against A and the pinned archive.
Failures stop dispatch, never retry or release reservations; independent active
stages finish. A signal cancels only this coordinator's recorded numeric steps.
"""
import argparse
from copy import deepcopy
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time

from run_speed_comparison import now, write

ROOT = Path(__file__).resolve().parents[2]
STEPS, ENVS, UPDATES = 4096, 512, 477568
METRIC = 'exact_original_music_micro_f1_min'
STAGES = (('R0_baseline', 1., .5, .75), ('R1_false025', .25, .5, .75),
          ('R2_activation08', 1., .8, .75), ('R3_gamma09', 1., .5, .90))
SOURCE_CHANGES = {'tsensei_v3_2_pianist/rl_torch/train.py',
                  'tsensei_v3_2_pianist/rl_torch/checkpoint.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pinned(path, sha):
    require(re.fullmatch('[0-9a-f]{64}', sha) is not None, 'Invalid SHA256: '+str(path))
    require(digest(path) == sha, 'SHA256 mismatch: '+str(path))


def read(path):
    return json.loads(Path(path).read_text())


def source_contract(template, baseline, archive):
    """Only observer/selection source may differ; verify every recorded hash."""
    tables = [(template['source_sha256'], baseline['source_sha256']),
              (template['source_runtime']['source_sha256'], baseline['source_runtime']['source_sha256'])]
    needed = {}
    for actual, old in tables:
        require(actual.keys() == old.keys(), 'Source manifest file set changed')
        changed = {key for key in actual if actual[key] != old[key]}
        require(changed <= SOURCE_CHANGES, 'Unexpected source changes: '+str(sorted(changed)))
        for key, value in actual.items():
            require(key not in needed or needed[key] == value, 'Conflicting source hashes: '+key)
            needed[key] = value
    with tarfile.open(archive) as packed:
        members = {m.name.removeprefix('./'): m for m in packed.getmembers() if m.isfile()}
        for name, sha in needed.items():
            require(name in members, 'Archive missing contract source: '+name)
            require(hashlib.sha256(packed.extractfile(members[name]).read()).hexdigest() == sha,
                    'Archive source differs from smoke: '+name)


def validate_smoke(template, baseline, parent, parent_sha):
    """Compare the full effective contract; enumerate shared intended changes."""
    expected = deepcopy(baseline)
    expected['sac']['init_temperature'] = .0006
    expected['requested_vector_steps'] = template['requested_vector_steps']
    expected['requested_transitions'] = template['requested_vector_steps'] * ENVS
    expected['checkpoint_retention']['forward_vector_step'] = 200
    expected['evaluation_selection'].update(best_metric=METRIC,
        target_exact_micro_f1=.99, consecutive_evaluations=4098)
    expected['source_sha256'] = template['source_sha256']
    expected['source_runtime']['source_sha256'] = template['source_runtime']['source_sha256']
    expected['origin'] = template['origin']
    expected.pop('cli', None)
    actual = {k: v for k, v in template.items() if k != 'cli'}
    differences = [k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)]
    require(not differences, 'Smoke differs from A outside approved contract: '+str(sorted(differences)))
    origin = template['origin']
    require(origin['checkpoint'] == str(parent) and origin['checkpoint_sha256'] == parent_sha,
            'Smoke parent checkpoint mismatch')
    require(origin['transfer_mode'] == 'finetune_networks_only'
            and origin['fresh_optimizer_alpha_rng'] is True and origin['fresh_replay'] is True,
            'Smoke must use networks only, fresh Adam/alpha/RNG/replay')
    require(template['learning_start_transition_1_based'] == 186881
            and template['warmup_transitions'] == 9999
            and template['observation_size'] == 2297 and template['action_size'] == 31,
            'Smoke prefill, exploration or interface changed')
    for name, value in dict(seed=42, num_envs=512, eval_interval=200, eval_episodes=3,
            eval_num_envs=3, save_interval=0, save_forward=True, allow_reward_change=True,
            allow_control_change=False, warm_start=None, stage2_from=None,
            finetune_from=str(parent), init_temperature=.0006, target_consecutive_evals=4098,
            best_f1_metric=METRIC, target_exact_micro_f1=.99).items():
        require(template['cli'].get(name) == value, 'Smoke CLI mismatch: '+name)
    expected_cli = deepcopy(baseline['cli'])
    expected_cli.update(vector_steps=template['requested_vector_steps'],
        output_dir=template['cli'].get('output_dir'),
        finger_coupling_angle_ranges=template['cli'].get('finger_coupling_angle_ranges'),
        reward_config=template['cli'].get('reward_config'), finetune_from=str(parent),
        init_temperature=.0006, eval_interval=200, allow_reward_change=True,
        target_consecutive_evals=4098, best_f1_metric=METRIC, target_exact_micro_f1=.99)
    differences = [k for k in set(expected_cli) | set(template['cli'])
                   if expected_cli.get(k) != template['cli'].get(k)]
    require(not differences, 'Unexpected smoke CLI changes: '+str(sorted(differences)))


def stage_reward(baseline, stage):
    expected = deepcopy(baseline['reward_config'])
    expected['terms']['false_press_penalty']['weight'] = stage['false_press_weight']
    expected['performance']['target_key_activation_weight'] = stage['activation_weight']
    return expected


def expected_config(template, stage, run, reward_path, parent):
    expected = deepcopy(template)
    expected['requested_vector_steps'] = STEPS
    expected['requested_transitions'] = STEPS * ENVS
    expected['sac']['gamma'] = stage['gamma']
    expected['reward_config'] = stage_reward(template, stage)
    expected['cli'].update(vector_steps=STEPS, output_dir=str(run),
        reward_config=str(reward_path), finetune_from=str(parent),
        target_key_activation_weight=stage['activation_weight'], discount=stage['gamma'])
    return expected


def validate_config(config, template, stage, run, reward_path, parent):
    expected = expected_config(template, stage, run, reward_path, parent)
    # This one CLI path is resolved inside each Slurm extraction. The canonical
    # coupling and every archive file hash are compared independently above.
    key = 'finger_coupling_angle_ranges'
    value = config['cli'].get(key)
    require(isinstance(value, str) and re.fullmatch(
        re.escape(str(ROOT))+r'/logs/hpc/\d+/steps/\d+/source/configs/finger_coupling/angle_ranges_60_80.json', value),
        'Unexpected archive coupling path')
    expected['cli'][key] = value
    differences = [key for key in set(config) | set(expected) if config.get(key) != expected.get(key)]
    require(not differences, 'Unexpected effective-config difference: '+str(sorted(differences)))


def validate_ready_config(config, *args):
    if any(k not in config for k in ('replay_bytes', 'observation_size', 'origin', 'target_entropy')):
        return False  # config.json is written twice during initialization.
    validate_config(config, *args)
    return True


def command(args, stage, allocation):
    reward = args.false025_reward_config if stage['false_press_weight'] == .25 else args.reward_config
    return ['bash', str(ROOT/'scripts/hpc/submit.sh'), '--allocation', str(allocation),
        '--source-archive', str(args.source_archive), 'train', '--seed', '42',
        '--num-envs', '512', '--vector-steps', str(STEPS), '--eval-interval', '200',
        '--eval-episodes', '3', '--eval-num-envs', '3', '--save-interval', '0', '--save-forward',
        '--no-eval-metrics-graph', '--finetune-from', str(args.parent), '--allow-reward-change',
        '--init-temperature', '.0006', '--reward-config', str(reward),
        '--target-key-activation-weight', str(stage['activation_weight']),
        '--discount', str(stage['gamma']), '--best-f1-metric', METRIC,
        '--target-exact-micro-f1', '.99', '--target-consecutive-evals', '4098',
        '--output-dir', str(args.output/stage['label'])]


def dispatchable(stages, allocations):
    if any(s['status'] == 'failed' for s in stages):
        return []
    busy = {s['allocation'] for s in stages if s['status'] == 'running'}
    return list(zip((s for s in stages if s['status'] == 'pending'),
                    (a for a in allocations if a not in busy)))


def numeric_steps(allocation):
    rows = subprocess.check_output(['squeue', '-h', '--steps', '-j', str(allocation), '-o', '%i'], text=True).split()
    return [r for r in rows if re.fullmatch(str(allocation)+r'\.\d+', r)]


def verify_allocation(allocation):
    row = subprocess.check_output(['squeue', '-h', '-j', str(allocation), '-o', '%u %T %j'], text=True).strip()
    require(row == getpass.getuser()+' RUNNING TsenseI-reserve2', 'Not an owned running reserve: '+row)
    require(not numeric_steps(allocation), 'Allocation already has an active workload')


def owned_steps(stage, steps):
    """Attribute steps by the exact submitted output prefix, never by occupancy."""
    expected = stage['command'][stage['command'].index('--output-dir') + 1]
    owned = []
    for step in steps:
        manifest = ROOT/'logs/hpc'/str(stage['allocation'])/'steps'/step.split('.')[-1]/'command.txt'
        if not manifest.is_file():
            continue  # Before job initialization, interrupt only our local srun.
        argv = shlex.split(manifest.read_text())
        outputs = [argv[i+1] for i, arg in enumerate(argv[:-1]) if arg == '--output-dir']
        require(outputs and outputs[-1] == expected, 'Unrelated step occupies pair: '+step)
        owned.append(step)
    return owned


def cancel_stage(stage, child):
    # Only recorded numeric steps from an otherwise empty reserved pair; never
    # pass an allocation, batch/extern step, PID, or another pair to scancel.
    for step in stage.get('owned_step_ids', []):
        if re.fullmatch(str(stage['allocation'])+r'\.\d+', step):
            subprocess.run(['scancel', step], check=False)
    try:
        group = os.getpgid(child.pid)
    except ProcessLookupError:
        # The flock leader may already have exited while its srun survives.
        group = stage.get('submit_pgid')
    safe_group = group == child.pid and group != os.getpgrp()
    if safe_group:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif child.poll() is None:
        child.terminate()  # Never signal an unverified/shared process group.
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if safe_group:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            child.kill()
        child.wait(timeout=10)


def summarize(run, template, stage, reward_path, parent):
    config, summary = read(run/'config.json'), read(run/'summary.json')
    validate_config(config, template, stage, run, reward_path, parent)
    for key, value in dict(status='completed', vector_steps=STEPS, transitions=STEPS*ENVS,
            replay_insertions=STEPS*ENVS, updates_this_run=UPDATES, updates_total=UPDATES,
            stop_reason='requested_budget_exhausted', target_reached=False,
            random_action_transitions=9999).items():
        require(summary.get(key) == value, 'Incomplete/incorrect budget: '+key)
    pipeline = summary['collector_pipeline_state']
    require(pipeline['pending_vector_step'] is None
            and pipeline['learner_updated_through_vector_step'] == STEPS
            and pipeline['collected_vector_steps'] == STEPS, 'Pipeline not fully drained')
    require({p.name for p in run.glob('*.pt')} == {'forward.pt', 'best.pt', 'final.pt'}, 'Unexpected checkpoint files')
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    evaluations = [r for r in rows if 'eval' in r]
    require([r['vector_step'] for r in evaluations] == [0, *range(200, 4001, 200), STEPS], 'Missing evaluation steps')
    from tsensei_v3_2_pianist.rl_torch.checkpoint import exact_music_f1_min
    for row in evaluations:
        evaluation = row['eval']
        _, reasons = exact_music_f1_min(evaluation)
        require(not reasons, 'Invalid exact evaluation: '+str(reasons))
        require(evaluation['note_timing_reference_count'] == 102, 'Incomplete 34-note evaluation')
        for key in ('collision_diagnostics_valid', 'key_contact_diagnostics_valid', 'substep_key_diagnostics_valid'):
            require(evaluation.get(key) is True, 'Invalid diagnostics: '+key)
    best = read(run/'best.json')
    require(best['metric'] == METRIC and best['score'] == exact_music_f1_min(best['evaluation'])[0], 'Incorrect best selection')
    import torch
    parent_state = None
    for name, step, updates in [('forward', 200, 0), ('best', best['vector_step'], best['updates']), ('final', STEPS, UPDATES)]:
        payload = torch.load(run/(name+'.pt'), map_location='cpu', weights_only=True)
        metadata = payload['metadata']
        require(metadata['vector_step'] == step and metadata['updates'] == updates, 'Checkpoint step/update mismatch: '+name)
        require(payload['agent']['updates'] == updates, 'Learner checkpoint update mismatch')
        saved_config = json.loads(json.dumps({k: metadata.get(k) for k in config}))
        require(saved_config == config, 'Checkpoint config mismatch: '+name)
        if name == 'forward':
            parent_state = torch.load(parent, map_location='cpu', weights_only=True)['agent']
            for net in ('actor', 'critic', 'target_critic'):
                require(payload['agent'][net].keys() == parent_state[net].keys()
                        and all(torch.equal(value, parent_state[net][key]) for key, value in payload['agent'][net].items()),
                        'Forward weights differ from parent: '+net)
            require(math.isclose(float(payload['agent']['log_temperature'].exp()), .0006, rel_tol=1e-6), 'Forward alpha changed before learning')
            for optimizer in ('actor_optimizer', 'critic_optimizer', 'temp_optimizer'):
                require(not payload['agent'][optimizer]['state'], 'Forward Adam state is not fresh: '+optimizer)
    final = summary['final_evaluation']
    return dict(run=str(run), steps=STEPS, transitions=STEPS*ENVS, updates=UPDATES,
        seconds=summary['elapsed_s'], best_step=best['vector_step'], best_exact_min=best['score'],
        final_exact_min=exact_music_f1_min(final)[0], final_exact_pooled=final['exact_original_music_micro_f1'],
        final_music_f1_min=final['music_only_f1_min'], diagnostics_verified=True,
        checkpoint_metadata_verified=True, forward_parent_weights_verified=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allocations', type=int, nargs=2, required=True)
    for name in ('source-archive', 'parent', 'baseline-config', 'expected-config', 'reward-config', 'false025-reward-config'):
        p.add_argument('--'+name, type=Path, required=True)
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--inspect-only', action='store_true', help='Validate pinned inputs and print commands; submit nothing')
    return p


def inputs(args):
    for name in ('source_archive', 'parent', 'baseline_config', 'expected_config', 'reward_config', 'false025_reward_config'):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        pinned(path, getattr(args, name+'_sha256'))
    baseline, template = read(args.baseline_config), read(args.expected_config)
    source_contract(template, baseline, args.source_archive)
    validate_smoke(template, baseline, args.parent, args.parent_sha256)
    require(read(args.reward_config) == baseline['reward_config'], 'Baseline reward JSON differs from A')
    expected = deepcopy(baseline['reward_config'])
    expected['terms']['false_press_penalty']['weight'] = .25
    require(read(args.false025_reward_config) == expected, 'False .25 JSON changes additional rewards')
    require(len(set(args.allocations)) == 2 and all(a > 0 for a in args.allocations), 'Need two distinct allocations')
    return template


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    template = inputs(args)
    stages = [dict(label=l, false_press_weight=f, activation_weight=a, gamma=g, status='pending') for l, f, a, g in STAGES]
    if args.inspect_only:
        print(json.dumps([command(args, st, args.allocations[i % 2]) for i, st in enumerate(stages)], indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (args.output/'coordinator.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                *[s for s in sys.argv[1:] if s != '--detach']], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True)
        print(json.dumps(dict(pid=child.pid, output=str(args.output))))
        return
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'runtime.json'
    require(not path.exists(), 'Existing runtime; use a fresh directory, no implicit retry')
    for allocation in args.allocations:
        verify_allocation(allocation)
    state = dict(status='running', pid=os.getpid(), session_id=os.getsid(0), started_at=now(),
        updated_at=now(), allocations=args.allocations, stages=stages,
        inputs={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        budget=dict(vector_steps=STEPS, transitions=STEPS*ENVS, updates=UPDATES),
        stop_policy='fixed budget: 4098 consecutive evaluations required; at most 21 post-initial evaluations',
        forward_note='step200 before prefill365: parent networks, fresh alpha .0006, zero optimizer updates')
    children, logs = {}, {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write(path, state)
    try:
        while True:
            for stage in [s for s in stages if s['status'] == 'running']:
                child = children[stage['label']]
                reward = args.false025_reward_config if stage['false_press_weight'] == .25 else args.reward_config
                try:
                    steps = numeric_steps(stage['allocation'])
                    require(len(steps) <= 1, 'Unexpected concurrent workload in retained allocation')
                    stage['owned_step_ids'] = sorted(set(stage.get('owned_step_ids', [])) | set(owned_steps(stage, steps)))
                    runs = [r for r in args.output.glob(stage['label']+'-*') if (r/'config.json').is_file()]
                    require(len(runs) <= 1, 'Duplicate output directories: '+stage['label'])
                    stage['output_dirs'] = list(map(str, runs))
                    if runs and not stage.get('config_verified'):
                        try:
                            config = read(runs[0]/'config.json')
                        except json.JSONDecodeError:
                            pass
                        else:
                            stage['config_verified'] = validate_ready_config(config, template, stage, runs[0], reward, args.parent)
                    if child.poll() is not None:
                        stage['exit_code'] = child.returncode
                        require(child.returncode == 0, 'Submission failed: '+str(child.returncode))
                        require(len(runs) == 1 and (runs[0]/'summary.json').is_file(), 'Step ended without summary')
                        stage['result'] = summarize(runs[0], template, stage, reward, args.parent)
                        stage.update(status='completed', finished_at=now())
                        logs.pop(stage['label']).close()
                except Exception as error:
                    cancel_stage(stage, child)
                    stage.update(status='failed', error=repr(error), finished_at=now())
                    logs.pop(stage['label']).close()
            for stage, allocation in dispatchable(stages, args.allocations):
                try:
                    verify_allocation(allocation)
                    inputs(args)  # Files may not mutate between stages.
                    argv = command(args, stage, allocation)
                    stage.update(status='running', allocation=allocation, command=argv, started_at=now())
                    write(path, state)
                    log = (args.output/(stage['label']+'.submit.log')).open('w')
                    logs[stage['label']] = log
                    children[stage['label']] = subprocess.Popen(argv, cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    stage['submit_pid'] = children[stage['label']].pid
                    stage['submit_pgid'] = children[stage['label']].pid
                except Exception as error:
                    stage.update(status='failed', error=repr(error), finished_at=now())
                    if stage['label'] in logs:
                        logs.pop(stage['label']).close()
                    break  # Leave independent running stages alone.
            active = [s['label'] for s in stages if s['status'] == 'running']
            state.update(updated_at=now(), active_stages=active)
            if not active:
                state['status'] = 'failed' if any(s['status'] == 'failed' for s in stages) else 'completed'
                write(path, state)
                break
            write(path, state)
            time.sleep(5)
    except BaseException as error:
        for stage in stages:
            if stage['status'] == 'running' and stage['label'] in children:
                cancel_stage(stage, children[stage['label']])
                stage.update(status='failed', error=repr(error), finished_at=now())
        state.update(status='failed', error=repr(error), updated_at=now())
        write(path, state)
        raise
    finally:
        for log in logs.values():
            log.close()
    if state['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
