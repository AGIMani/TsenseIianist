#!/usr/bin/env python3
"""Durable CPU coordinator for fresh 20/200 Hz Twinkle comparisons.

Manifest schema1: source_archive/source_archive_sha256, output_root,
vector_steps(default19532), num_envs(default512), seed(default42),
eval_interval(default500), eval_episodes(default3), common_args(list),
common_expect(dotted effective-config paths), and groups containing label,
reward_config/reward_config_sha256, args(list), expect(dotted paths).
An optional expected_reward_config/expected_reward_config_sha256 pins the fully
resolved reward contract when CLI options override the supplied reward JSON.

The manifest, archive and config files are immutable. --detach survives the
calling shell; rerunning with the same manifest resumes recorded Slurm jobs.
Submissions use durable child receipts: an ambiguous interrupted submission
never causes an automatic duplicate. No GPU/simulation runs in this process.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
METRIC = 'exact_original_music_micro_f1_min'
ACTIVE = {'submitting', 'queued', 'running', 'accounting_wait'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
FORBIDDEN = {'--warm-start', '--finetune-from', '--stage2-from', '--resume',
             '--allow-reward-change', '--allow-control-change', '--source-archive',
             '--output-dir', '--vector-steps', '--num-envs', '--control-hz',
             '--physics-hz', '--lookahead', '--reward-config', '--seed',
             '--best-f1-metric', '--target-consecutive-evals', '--eval-episodes',
             '--eval-num-envs', '--eval-interval', '--target-max-key-underside-substeps',
             '--device', '--learner-device', '--collector-pipeline'}


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, sha):
    require(Path(path).is_absolute(), 'Manifest paths must be absolute: ' + str(path))
    require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), 'Invalid SHA256')
    require(digest(path) == sha, 'Pinned file changed: ' + str(path))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def dotted(value, path):
    for name in path.split('.'):
        value = value[name]
    return value


def archive_hashes(path):
    with tarfile.open(path) as archive:
        return {m.name.removeprefix('./'): hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                for m in archive.getmembers() if m.isfile()}


def load_manifest(path):
    manifest = read(path)
    require(manifest.get('schema') == 1, 'Expected manifest schema1')
    pin(manifest['source_archive'], manifest['source_archive_sha256'])
    require(Path(manifest['output_root']).is_absolute(), 'output_root must be absolute')
    manifest.setdefault('vector_steps', 19532); manifest.setdefault('num_envs', 512)
    manifest.setdefault('seed', 42); manifest.setdefault('eval_interval', 500)
    manifest.setdefault('eval_episodes', 3); manifest.setdefault('common_args', [])
    manifest.setdefault('common_expect', {})
    for key in ('vector_steps', 'num_envs', 'eval_interval', 'eval_episodes'):
        require(type(manifest[key]) is int and manifest[key] > 0, 'Invalid ' + key)
    require(manifest['num_envs'] == 512, 'Comparison uses512 environments')
    groups = manifest['groups']; require(len(groups) >= 2, 'At least2 groups needed for independent top2')
    labels = [g['label'] for g in groups]
    require(len(labels) == len(set(labels)), 'Duplicate group labels')
    for group in groups:
        require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', group['label']), 'Unsafe label')
        group.setdefault('args', []); group.setdefault('expect', {})
        for key in ('reward_config', 'expected_reward_config'):
            if key in group:
                pin(group[key], group[key + '_sha256'])
        require('reward_config' in group, 'Each group needs pinned reward config')
        for args in (manifest['common_args'], group['args']):
            require(isinstance(args, list) and all(isinstance(x, str) for x in args), 'args must be a string list')
            require(not any(x.split('=')[0] in FORBIDDEN for x in args), 'Controlled/warmstart argument in manifest')
    return manifest


def train_command(manifest, group):
    return ['bash', str(ROOT/'scripts/hpc/submit.sh'), '--gpus', '1',
        '--source-archive', manifest['source_archive'], 'train',
        *manifest['common_args'], *group['args'],
        '--control-hz', '20', '--physics-hz', '200', '--lookahead', '10',
        '--seed', str(manifest['seed']), '--num-envs', str(manifest['num_envs']),
        '--vector-steps', str(manifest['vector_steps']), '--reward-config', group['reward_config'],
        '--eval-interval', str(manifest['eval_interval']), '--eval-episodes', str(manifest['eval_episodes']),
        '--eval-num-envs', str(manifest['eval_episodes']), '--save-interval', '0', '--save-forward',
        '--no-eval-metrics-graph', '--best-f1-metric', METRIC,
        '--target-exact-micro-f1', '.99', '--target-consecutive-evals', str(manifest['vector_steps'] + 2),
        '--target-max-key-underside-substeps', '0',
        '--output-dir', str(Path(manifest['output_root'])/group['label'])]


def validate_config(config, manifest, group, hashes):
    # config.json is first written before runtime resolution.
    if any(key not in config for key in ('replay_bytes', 'observation_size', 'source_runtime', 'action_contract')):
        return False
    require(not config.get('origin'), 'Fresh experiment unexpectedly has a checkpoint origin')
    cli = config['cli']
    for key in ('warm_start', 'stage2_from', 'finetune_from'):
        require(cli.get(key) is None, 'Checkpoint transfer forbidden: ' + key)
    env = config['source_runtime']['environment']
    for key, expected in dict(control_hz=20., physics_hz=200., lookahead=10).items():
        require(env.get(key) == expected, 'Runtime timing mismatch: ' + key)
    require(config['requested_vector_steps'] == manifest['vector_steps'], 'Vector budget mismatch')
    require(config['requested_transitions'] == manifest['vector_steps'] * manifest['num_envs'], 'Transition budget mismatch')
    require(config['num_envs'] == manifest['num_envs'] and cli['seed'] == manifest['seed'], 'Seed/environment mismatch')
    require(config['action_size'] == 31 and config['action_contract']['action_size'] == 31, 'Expected31 coupled actions')
    require(config['no_arms'] is True and config['disable_wrist_tz'] is True
            and config['disable_wrist_yaw'] is True, 'Unexpected physical interface')
    require(config['execution_layout']['collector_pipeline'] is False, 'Single-GPU experiment must not pipeline')
    require(config['execution_layout']['collector_device'] == config['execution_layout']['learner_device'] == 'cuda:0',
            'Single-GPU device contract mismatch')
    require(config['evaluation_selection']['best_metric'] == METRIC, 'Wrong best metric')
    require(config['evaluation_selection']['max_key_underside_substeps'] == 0, 'Missing physical best gate')
    require(config['evaluation_selection']['consecutive_evaluations'] == manifest['vector_steps'] + 2, 'Early stopping must be disabled by unreachable consecutive count')
    for key in ('eval_interval', 'eval_episodes'):
        require(cli[key] == manifest[key], 'Evaluation schedule mismatch: ' + key)
    expected_reward = read(group.get('expected_reward_config', group['reward_config']))
    require(config['reward_config'] == expected_reward, 'Resolved reward differs from pinned expected contract')
    for table in (config['source_sha256'], config['source_runtime']['source_sha256']):
        require(bool(table), 'Empty source contract')
        for name, sha in table.items():
            require(hashes.get(name) == sha, 'Runtime source differs from pinned archive: ' + name)
    for path, expected in {**manifest['common_expect'], **group['expect']}.items():
        require(dotted(config, path) == expected, 'Effective override mismatch: ' + path)
    return True


def update_count(config, step):
    ratio = Fraction(*config['scheduling']['update_ratio_fraction'])
    eligible = max(0, step*config['num_envs'] - config['learning_start_transition_1_based'] + 1)
    return (eligible * ratio.numerator) // ratio.denominator


def evaluation_stats(evaluation, expected_episodes=None):
    """Fail closed for missing/nonfinite/incomplete evidence; count F1 independently."""
    n = evaluation.get('evaluation_episodes')
    require(type(n) is int and n > 0 and (expected_episodes is None or n == expected_episodes), 'Wrong evaluation episode count')
    require(evaluation.get('completed_episodes') == n and evaluation.get('truncated_episodes') == 0, 'Incomplete evaluation')
    episodes = evaluation.get('episode_statistics', [])
    require(len(episodes) == n, 'Missing episode statistics')
    require(evaluation.get('note_timing_reference_count') == 34*n, 'Not the full34-note score')
    require(evaluation.get('substep_key_sample_hz') == 200., 'Exact key diagnostic frequency is not200Hz')
    require(evaluation.get('collision_physics_sample_hz') == 200.
            and evaluation.get('key_contact_physics_sample_hz') == 200., 'Contact frequency is not200Hz')
    totals = [0, 0, 0]; scores = []
    for row in [evaluation, *episodes]:
        for key in ('collision_diagnostics_valid', 'key_contact_diagnostics_valid', 'substep_key_diagnostics_valid'):
            require(row.get(key) is True, 'Invalid diagnostics: ' + key)
        for key in ('collision_diagnostics_coverage', 'collision_substep_coverage',
                    'key_contact_diagnostics_coverage', 'key_contact_substep_coverage',
                    'substep_key_diagnostics_coverage', 'substep_key_coverage'):
            require(row.get(key) == 1., 'Incomplete diagnostic coverage: ' + key)
        counts = [row.get('exact_original_music_' + k + '_count') for k in
                  ('true_positive', 'false_positive', 'false_negative')]
        require(all(finite(x) and x >= 0 and int(x) == x for x in counts), 'Invalid exact TP/FP/FN')
        tp, fp, fn = counts; denom = 2*tp + fp + fn
        score = 2*tp/denom if denom else 1.
        require(finite(row.get('exact_original_music_micro_f1')) and
                abs(row['exact_original_music_micro_f1']-score) < 1e-12, 'Exact F1/count mismatch')
        if row is not evaluation:
            require(row.get('note_timing_available') is True
                    and len(row.get('note_timing_records', [])) == 34, 'Incomplete exact note records')
            totals = [a+b for a, b in zip(totals, counts)]; scores.append(score)
    for i, key in enumerate(('true_positive', 'false_positive', 'false_negative')):
        require(totals[i] == evaluation['exact_original_music_' + key + '_count'], 'Pooled count mismatch')
    for key in ('hand_collision_substeps', 'key_side_or_under_substeps', 'key_underside_substeps_max'):
        require(finite(evaluation.get(key)) and evaluation[key] >= 0, 'Invalid contact count: ' + key)
    require(evaluation['key_underside_substeps_max'] == max(ep['key_underside_substeps'] for ep in episodes), 'Underside maximum mismatch')
    return dict(exact_min=min(scores), exact_pooled=evaluation['exact_original_music_micro_f1'],
        tp=totals[0], fp=totals[1], fn=totals[2], episodes=n,
        hand_substeps=evaluation['hand_collision_substeps'],
        side_substeps=evaluation['key_side_or_under_substeps'],
        underside_max=evaluation['key_underside_substeps_max'],
        eligible=evaluation['key_underside_substeps_max'] == 0)


def incremental_metrics(path, stage, config, episodes):
    """Only consume complete lines. A restart keeps the byte offset in state."""
    offset = stage.get('metrics_offset', 0)
    if not path.exists(): return
    require(path.stat().st_size >= offset, 'Metrics file was truncated')
    with path.open('rb') as stream:
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line or not line.endswith(b'\n'): break
            row = json.loads(line)
            step = row['vector_step']
            require(row['transitions'] == step*config['num_envs'], 'Logged transition mismatch')
            # In the enforced single-device loop, commit_vector emits an
            # episode before learn_vector(step). Train/eval rows are emitted
            # after that update block; they must retain the exact current-step
            # check, even if an unexpected mixed record contains episodes.
            before_learning = 'episodes' in row and 'train' not in row and 'eval' not in row
            update_step = step - 1 if before_learning else step
            require(update_step >= 0, 'Episode cannot precede the first vector step')
            require(row['updates'] == update_count(config, update_step), 'Logged new-update mismatch')
            for value in row.get('train', {}).values():
                require(finite(value), 'Nonfinite learner metric')
            if 'eval' in row:
                stats = evaluation_stats(row['eval'], episodes)
                point = dict(vector_step=step, updates=row['updates'], **stats)
                curve = stage.setdefault('curve', [])
                require(not curve or step > curve[-1]['vector_step'], 'Duplicate/out-of-order evaluation')
                curve.append(point)
            stage['latest_vector_step'] = step
            stage['metrics_offset'] = stream.tell()


def checkpoint_check(path, config, step, updates, evaluation):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU coordinator must not initialize CUDA')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta, agent = payload['metadata'], payload['agent']
    require(all(normalize(meta.get(k)) == normalize(v) for k, v in config.items()), 'Checkpoint config mismatch')
    require(meta['vector_step'] == step and meta['updates'] == updates == agent['updates'], 'Checkpoint step/update mismatch')
    require(meta['last_evaluation_vector_step'] == step and meta['last_evaluation'] == evaluation, 'Checkpoint evaluation mismatch')
    require(normalize(agent['config']) == config['sac'], 'Checkpoint SAC mismatch')
    require(agent['obs_size'] == config['observation_size'] and agent['action_size'] == 31, 'Checkpoint interface mismatch')
    def tensors(value):
        if isinstance(value, torch.Tensor):
            require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor')
        elif isinstance(value, dict):
            for item in value.values(): tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value: tensors(item)
    tensors(agent)
    require(not torch.cuda.is_initialized(), 'CUDA initialized during checkpoint check')
    return dict(checkpoint_sha256=digest(path), config_sha256=digest(path.parent/'config.json'),
        finite_tensors=True, checkpoint_metadata_verified=True, cuda_initialized=False)


def complete_training(run, config, manifest, stage):
    summary = read(run/'summary.json'); steps = manifest['vector_steps']
    for key, expected in dict(status='completed', vector_steps=steps,
            transitions=steps*manifest['num_envs'], replay_insertions=steps*manifest['num_envs'],
            updates_this_run=update_count(config, steps), updates_total=update_count(config, steps),
            stop_reason='requested_budget_exhausted', target_reached=False).items():
        require(summary.get(key) == expected, 'Completed budget mismatch: ' + key)
    require(summary['random_action_transitions'] == min(config['warmup_transitions'], summary['transitions']), 'Random exploration budget mismatch')
    require(stage['latest_vector_step'] == steps, 'Final metrics missing')
    expected_steps = [0, *range(manifest['eval_interval'], steps+1, manifest['eval_interval'])]
    if expected_steps[-1] != steps: expected_steps.append(steps)
    require([p['vector_step'] for p in stage.get('curve', [])] == expected_steps, 'Evaluation schedule incomplete')
    candidates = []
    best = read(run/'best.json') if (run/'best.json').is_file() else None
    for role, evaluation, step, updates in [('final', summary['final_evaluation'], steps, summary['updates_total']),
            *([('best', best['evaluation'], best['vector_step'], best['updates'])] if best else [])]:
        stats = evaluation_stats(evaluation, manifest['eval_episodes'])
        require(updates == update_count(config, step), 'Saved update count mismatch')
        logged = [r for r in stage['curve'] if r['vector_step'] == step]
        require(len(logged) == 1 and all(logged[0][k] == v for k, v in stats.items()), 'Saved evaluation differs from stream')
        if role == 'best':
            require(best['metric'] == METRIC and best['score'] == stats['exact_min']
                    and best.get('best_eligible') is True and stats['eligible'], 'Invalid saved best')
            eligible_scores = [r['exact_min'] for r in stage['curve'] if r['eligible']]
            require(best['score'] == max(eligible_scores), 'Saved best is not the highest eligible logged minimum')
        acceptance = checkpoint_check(run/(role+'.pt'), config, step, updates, evaluation)
        candidates.append(dict(group=stage['label'], role=role, path=str(run/(role+'.pt')),
            config=str(run/'config.json'), vector_step=step, updates=updates, **stats, **acceptance))
    stage['candidates'] = candidates
    return dict(vector_steps=steps, transitions=summary['transitions'], new_updates=summary['updates_total'],
                budget_verified=True, saved_checkpoints_checked=len(candidates))


def rank_key(candidate):
    return (-candidate['exact_min'], -candidate['exact_pooled'], candidate['side_substeps'], candidate['hand_substeps'])


def independent_rank_key(candidate):
    return (-candidate['exact_pooled'], -candidate['exact_min'], candidate['side_substeps'], candidate['hand_substeps'])


def replay_command(manifest, selected, destination):
    return ['bash', str(ROOT/'scripts/hpc/submit.sh'), '--gpus', '1',
        '--source-archive', manifest['source_archive'], 'replay',
        '--source-archive', manifest['source_archive'], '--checkpoint', selected['path'],
        '--config', selected['config'], '--mode', 'evaluate', '--episodes', '10',
        '--eval-num-envs', '3', '--output', str(destination)]


def submit_worker(request_path):
    request = read(request_path)
    try:
        result = subprocess.run(request['argv'], cwd=ROOT, capture_output=True, text=True, timeout=120)
        matches = re.findall(r'^([0-9]+)(?:;[^\s]+)?$', result.stdout.strip(), re.M)
        require(result.returncode == 0 and len(matches) == 1, 'Slurm submission did not return exactly one job ID')
        receipt = dict(status='submitted', job_id=matches[0], stdout=result.stdout,
            stderr=result.stderr, returncode=result.returncode, finished_at=now())
    except Exception as error:
        receipt = dict(status='submission_unknown', error=repr(error), finished_at=now())
    write(request['receipt'], receipt)


class Slurm:
    def submit(self, request):
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--submit-worker', str(request)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True).pid

    def states(self, jobs):
        if not jobs: return {}
        ids = ','.join(jobs)
        result = subprocess.run(['squeue', '-h', '-j', ids, '-o', '%i|%T'], capture_output=True, text=True, timeout=20)
        require(result.returncode == 0, 'squeue failed: ' + result.stderr)
        states = {r.split('|')[0]: r.split('|')[1] for r in result.stdout.splitlines() if '|' in r}
        missing = [j for j in jobs if j not in states]
        if missing:
            result = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(missing),
                '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True, timeout=20)
            require(result.returncode == 0, 'sacct failed: ' + result.stderr)
            for line in result.stdout.splitlines():
                fields = line.split('|')
                if len(fields) >= 3 and fields[0] in missing:
                    states[fields[0]] = fields[1].split()[0].rstrip('+')
        return states

    def cancel(self, job):
        require(re.fullmatch('[0-9]+', job), 'Unsafe Slurm job ID')
        subprocess.run(['scancel', job], check=True, timeout=20)


class Coordinator:
    def __init__(self, manifest_path, *, max_parallel=2, top_k=None, backend=None):
        self.path = Path(manifest_path).resolve(); self.manifest = load_manifest(self.path)
        self.root = Path(self.manifest['output_root']); self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root/'coordinator_state.json'
        self.backend = backend or Slurm(); self.hashes = archive_hashes(self.manifest['source_archive'])
        require(type(max_parallel) is int and max_parallel > 0, 'max_parallel must be positive')
        top_k = self.manifest.get('replay_top_k', 0) if top_k is None else top_k
        require(type(top_k) is int and (top_k == 0 or top_k >= 2), 'replay_top_k:0=all, otherwise at least2')
        self.max_parallel, self.top_k = max_parallel, top_k
        if self.state_path.exists():
            self.state = read(self.state_path)
            require(self.state['manifest_sha256'] == digest(self.path), 'Manifest changed since submission')
            require(self.state['coordinator_sha256'] == digest(Path(__file__)), 'Coordinator changed: review before resuming')
            require(self.state['top_k'] == top_k, 'Cannot change saved top_k')
        else:
            require(not any(self.root.glob('*.submit.request.json')), 'Existing submission artifacts without state')
            self.state = dict(schema=1, status='running', started_at=now(),
                manifest=str(self.path), manifest_sha256=digest(self.path),
                coordinator_sha256=digest(Path(__file__)), source_archive=self.manifest['source_archive'],
                source_archive_sha256=self.manifest['source_archive_sha256'], top_k=top_k,
                groups=[dict(label=g['label'], status='pending', kind='train') for g in self.manifest['groups']],
                replays=[], replays_planned=False)
            write(self.root/'manifest.snapshot.json', read(self.path)); self.save()

    def save(self):
        self.state.update(updated_at=now(), pid=os.getpid(), max_parallel=self.max_parallel)
        write(self.state_path, self.state)

    def dispatch(self, stage, argv):
        # Request and intent are durable before the worker can call sbatch.
        load_manifest(self.path)
        require(digest(self.path) == self.state['manifest_sha256'], 'Manifest changed')
        request = self.root/(stage['label']+'.submit.request.json')
        receipt = self.root/(stage['label']+'.submit.receipt.json')
        require(not request.exists() and not receipt.exists(), 'Never overwrite a prior submission')
        write(request, dict(argv=argv, receipt=str(receipt)))
        stage.update(status='submitting', command=argv, receipt=str(receipt), requested_at=now(),
            requested_epoch=time.time())
        self.save()
        stage['submit_pid'] = self.backend.submit(request)
        self.save()

    def update_receipt(self, stage):
        receipt_path = Path(stage['receipt'])
        if receipt_path.exists():
            receipt = read(receipt_path); stage['submission_receipt'] = receipt
            stage.update(status='queued' if receipt['status'] == 'submitted' else 'submission_unknown')
            if receipt['status'] == 'submitted': stage['job_id'] = receipt['job_id']
            return
        if time.time() - stage['requested_epoch'] > 180:
            stage.update(status='submission_unknown', error='No durable submission receipt; inspect Slurm before manual recovery. Not resubmitted.')

    def inspect_train(self, stage, slurm_state):
        group = next(g for g in self.manifest['groups'] if g['label'] == stage['label'])
        runs = [p for p in self.root.glob(stage['label']+'-*') if p.is_dir() and (p/'config.json').exists()]
        require(len(runs) <= 1, 'Duplicate run directories for ' + stage['label'])
        if runs:
            stage['run'] = str(runs[0])
            try: config = read(runs[0]/'config.json')
            except json.JSONDecodeError:
                require(slurm_state not in TERMINAL, 'Invalid terminal config JSON'); return
            if validate_config(config, self.manifest, group, self.hashes):
                stage['config_verified'] = True
                incremental_metrics(runs[0]/'metrics.jsonl', stage, config, self.manifest['eval_episodes'])
                if slurm_state == 'COMPLETED':
                    stage['result'] = complete_training(runs[0], config, self.manifest, stage)
                    stage.update(status='completed', finished_at=now()); return
        if slurm_state in TERMINAL:
            raise ValueError('Slurm ended without validated complete result: ' + str(slurm_state))

    def inspect_replay(self, stage, slurm_state):
        if slurm_state == 'COMPLETED':
            raw = read(stage['output']); selected = stage['selected']
            stats = evaluation_stats(raw['metrics'], 10)
            require(raw.get('episodes') == 10 and Path(raw['checkpoint']).resolve() == Path(selected['path']).resolve(),
                    'Independent replay checkpoint/episode mismatch')
            require(digest(selected['path']) == selected['checkpoint_sha256'], 'Selected checkpoint changed')
            require(digest(selected['config']) == selected['config_sha256'], 'Selected config changed')
            # The replay entry itself validates scene/task/source reconstruction.
            stage.update(status='completed', result=stats, output_sha256=digest(stage['output']), finished_at=now())
        elif slurm_state in TERMINAL:
            raise ValueError('Replay job failed: ' + slurm_state)

    def plan_replays(self):
        if self.state['replays_planned']: return
        if any(s['status'] in ACTIVE or s['status'] == 'pending' for s in self.state['groups']): return
        candidates = []
        for stage in self.state['groups']:
            eligible = [c for c in stage.get('candidates', []) if c['eligible'] and c['updates'] > 0]
            seen_steps = set()
            for candidate in sorted(eligible, key=lambda c: (c['role'] != 'best', rank_key(c))):
                if candidate['vector_step'] not in seen_steps:
                    candidates.append(candidate); seen_steps.add(candidate['vector_step'])
        candidates.sort(key=rank_key)
        self.state['training_ranking'] = candidates
        self.state['replays_planned'] = True
        selected_candidates = candidates if self.top_k == 0 else candidates[:self.top_k]
        for selected in selected_candidates:
            label = selected['group']+'_'+selected['role']+'_independent10'
            destination = self.root/'reevaluation'/(label+'.json')
            require(not destination.exists(), 'Never overwrite independent evaluation')
            self.state['replays'].append(dict(label=label, kind='replay', status='pending',
                selected=selected, output=str(destination)))
        if len({c['group'] for c in selected_candidates}) < 2:
            self.state['insufficient_eligible_groups'] = len(candidates)
        self.save()

    def leaderboard(self):
        rows = []
        for stage in self.state['groups']:
            eligible = [r for r in stage.get('curve', []) if r['eligible'] and r['updates'] > 0]
            rows.append(dict(group=stage['label'], status=stage['status'],
                latest=stage.get('curve', [None])[-1] if stage.get('curve') else None,
                best_logged=sorted(eligible, key=rank_key)[0] if eligible else None,
                run=stage.get('run'), error=stage.get('error')))
        independent = [dict(group=s['selected']['group'], selected=s['selected'], replay_output=s['output'],
            replay_output_sha256=s['output_sha256'], **s['result'])
            for s in self.state['replays'] if s['status'] == 'completed']
        independent.sort(key=independent_rank_key)
        write(self.root/'leaderboard.json', dict(updated_at=now(), frequency_hz=200,
            training=rows, independent10=independent,
            note='Training3ep selection and independent10ep ranking remain separate; counts are physical substeps.'))
        eligible = [r for r in independent if r['eligible']]
        if (eligible and 'insufficient_eligible_groups' not in self.state
                and all(s['status'] == 'completed' for s in self.state['groups'])
                and all(s['status'] == 'completed' for s in self.state['replays'])):
            write(self.root/'best_model.json', dict(schema=1, selected=eligible[0],
                source_archive=self.manifest['source_archive'], source_archive_sha256=self.manifest['source_archive_sha256'],
                manifest=str(self.path), manifest_sha256=self.state['manifest_sha256'],
                criterion='Independent10 pooled exact200HzF1, then worst episode, then side/hand contacts; underside0',
                model_files_copied_or_overwritten=False))

    def tick(self):
        stages = self.state['groups'] + self.state['replays']
        for stage in stages:
            if stage['status'] == 'submitting': self.update_receipt(stage)
        jobs = [s['job_id'] for s in stages if s['status'] in ACTIVE and 'job_id' in s]
        states = self.backend.states(jobs)
        for stage in stages:
            if stage['status'] not in ACTIVE or 'job_id' not in stage: continue
            slurm_state = states.get(stage['job_id'])
            stage['slurm_state'] = slurm_state
            if slurm_state is None:
                stage.setdefault('accounting_missing_since', time.time())
                if time.time()-stage['accounting_missing_since'] > 300:
                    stage.update(status='failed', error='Job missing from queue and accounting for300s; not resubmitted')
                    continue
                stage['status'] = 'accounting_wait'
            else:
                stage.pop('accounting_missing_since', None)
                stage['status'] = 'running' if slurm_state == 'RUNNING' else 'queued'
            try:
                if stage['kind'] == 'train': self.inspect_train(stage, slurm_state)
                else: self.inspect_replay(stage, slurm_state)
            except Exception as error:
                stage.update(status='failed', error=repr(error), finished_at=now())
                if slurm_state not in TERMINAL and slurm_state is not None:
                    self.backend.cancel(stage['job_id']); stage['cancelled_for_contract_failure'] = True
        self.plan_replays()
        stages = self.state['groups'] + self.state['replays']
        capacity = self.max_parallel - sum(s['status'] in ACTIVE for s in stages)
        # Unknown submission may own a GPU. Stop dispatch rather than duplicate/overfill.
        if not any(s['status'] == 'submission_unknown' for s in stages):
            for stage in [s for s in stages if s['status'] == 'pending'][:max(0, capacity)]:
                if stage['kind'] == 'train':
                    group = next(g for g in self.manifest['groups'] if g['label'] == stage['label'])
                    argv = train_command(self.manifest, group)
                else:
                    argv = replay_command(self.manifest, stage['selected'], stage['output'])
                self.dispatch(stage, argv)
        active = any(s['status'] in ACTIVE or s['status'] == 'pending' for s in stages)
        unknown = any(s['status'] == 'submission_unknown' for s in stages)
        if unknown:
            self.state['status'] = 'needs_submission_reconciliation'
        elif not active:
            self.state['status'] = ('completed' if 'insufficient_eligible_groups' not in self.state
                and all(s['status'] == 'completed' for s in stages) else 'completed_with_failures')
        else: self.state['status'] = 'running'
        self.leaderboard(); self.save()
        return self.state['status'] not in ('running',)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--max-parallel', type=int, default=2)
    p.add_argument('--replay-top-k', '--top-k', dest='top_k', type=int,
        help='0(default)=all distinct eligible best/final; otherwise at least2 candidates')
    p.add_argument('--poll-seconds', type=float, default=15.)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--once', action='store_true')
    p.add_argument('--submit-worker', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.submit_worker: submit_worker(args.submit_worker); return
    require(args.manifest is not None, '--manifest is required')
    require(1 <= args.poll_seconds <= 30, 'Poll interval must be1–30seconds')
    manifest = load_manifest(args.manifest.resolve()); root = Path(manifest['output_root'])
    if args.inspect_only:
        print(json.dumps(dict(manifest=str(args.manifest.resolve()),
            commands=[train_command(manifest, g) for g in manifest['groups']]), indent=2)); return
    root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (root/'coordinator.log').open('ab', buffering=0) as log:
            argv = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != '--detach']]
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps(dict(pid=child.pid, output=str(root), detached=True))); return
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinator = Coordinator(args.manifest, max_parallel=args.max_parallel, top_k=args.top_k)
        def stop(signum, frame): raise KeyboardInterrupt('Coordinator stopped; submitted Slurm jobs continue. Resume same manifest.')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            while True:
                done = coordinator.tick()
                if done or args.once: break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt as error:
            coordinator.state['interrupted'] = str(error); coordinator.save()
        print(json.dumps(dict(status=coordinator.state['status'], output=str(root))))


if __name__ == '__main__':
    main()
