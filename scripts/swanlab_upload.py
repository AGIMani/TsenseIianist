#!/usr/bin/env python3
"""Upload existing JSONL metrics, or follow a learner in a separate process.

No MuJoCo/Torch imports; use the isolated monitoring Python environment.
Credentials are read from SWANLAB_API_KEY or an external mode-0600 file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_FILE = Path.home() / '.config/tsensei-pianist/swanlab.key'
DEFAULT_PYTHON = ROOT / 'training_mjlab/monitoring/.venv/bin/python'


def add_arguments(parser):
    parser.add_argument('--swanlab', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--swanlab-project', default='TsenseI_A3.2')
    parser.add_argument('--swanlab-workspace')
    parser.add_argument('--swanlab-run-name')
    parser.add_argument('--swanlab-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--swanlab-key-file', type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument('--swanlab-python', type=Path, default=DEFAULT_PYTHON)


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def iter_records(path, follow_pid=None, poll_seconds=1.):
    """Wait for complete lines; never consume half a concurrent JSON write."""
    while not path.exists():
        if not process_alive(follow_pid):
            raise FileNotFoundError(f'Learner did not create {path.name}')
        time.sleep(poll_seconds)
    with path.open() as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if line.endswith('\n'):
                yield json.loads(line)
                continue
            stream.seek(position)
            if not process_alive(follow_pid):
                # A final valid non-newline record is allowed for imported logs.
                if line.strip():
                    yield json.loads(line)
                break
            time.sleep(poll_seconds)


def grouped_scalars(records, config):
    """A train and eval row can share a step; log that step only once."""
    from swanlab_metrics import scalars_from_record, metric_order
    order = metric_order(config)
    step, pending = None, {}
    for row in records:
        current = row['transitions']
        if type(current) is not int or current < 0:
            raise ValueError('Metric transitions must be a nonnegative integer')
        if step is not None and current < step:
            raise ValueError('Training metrics moved backwards in transitions')
        if step is not None and current != step:
            if pending:
                yield step, {k: pending[k] for k in order if k in pending} | {
                    k: pending[k] for k in sorted(pending) if k not in order}
            pending = {}
        step = current
        pending.update(scalars_from_record(row, config))
    if pending:
        yield step, {k: pending[k] for k in order if k in pending} | {
            k: pending[k] for k in sorted(pending) if k not in order}


def read_config(path, follow_pid):
    while True:
        try:
            config = json.loads(path.read_text())
            # The learner writes an initial config before resolving the model.
            if 'observation_size' in config or not process_alive(follow_pid):
                return config
        except (FileNotFoundError, json.JSONDecodeError):
            if not process_alive(follow_pid):
                raise
        time.sleep(1.)


def upload(args):
    from swanlab_metrics import metric_order, public_config
    run_dir = args.run_dir.resolve()
    config = read_config(run_dir / 'config.json', args.follow_pid)
    state_path = run_dir / 'swanlab_upload.json'
    metrics_path = run_dir / 'metrics.jsonl'
    config_hash = hashlib.sha256(json.dumps(public_config(config), sort_keys=True).encode()).hexdigest()
    if state_path.exists() and not args.force:
        previous = json.loads(state_path.read_text())
        if (previous.get('status') == 'completed' and args.swanlab_mode == 'online'
                and previous.get('project') == args.swanlab_project
                and previous.get('workspace') == args.swanlab_workspace
                and previous.get('config_sha256') == config_hash
                and previous.get('metrics_sha256') == hashlib.sha256(metrics_path.read_bytes()).hexdigest()):
            print('Already uploaded: ' + str(previous.get('url')), flush=True)
            return previous
    key = os.environ.get('SWANLAB_API_KEY', '').strip()
    if not key and args.swanlab_key_file.is_file():
        key = args.swanlab_key_file.read_text().strip()
    state = dict(status='initializing', project=args.swanlab_project,
                 workspace=args.swanlab_workspace, config_sha256=config_hash,
                 mode=args.swanlab_mode, metric_order=metric_order(config),
                 x_axis='collected transitions', records=0,
                 started_at=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)
    sdk = None
    try:
        if args.swanlab_mode == 'online' and not key:
            raise RuntimeError('Set SWANLAB_API_KEY or the external SwanLab key file')
        import swanlab as sdk
        # Upload selected hyperparameters/scalars only, not terminal/source/git.
        settings = sdk.Settings(
            interactive=False, api_key=key or None,
            api_host='https://api.swanlab.cn', web_host='https://swanlab.cn',
            terminal=sdk.Settings.Terminal(proxy_type='none'),
            probe=sdk.Settings.Probe(hardware=False, runtime=False, requirements=False,
                                    conda=False, git=False, monitor=False, swanlab=False))
        run = sdk.init(project=args.swanlab_project, workspace=args.swanlab_workspace,
                       name=args.swanlab_run_name or run_dir.name,
                       public=False, mode=args.swanlab_mode,
                       log_dir=str(run_dir / 'swanlab'), settings=settings,
                       config=public_config(config))
        state.update(status='uploading', sdk_version=sdk.__version__,
                     url=run.url if args.swanlab_mode == 'online' else None,
                     run_id=getattr(run, 'id', None))
        for name in metric_order(config):
            sdk.define_metric(name, section_name='01_metrics')
        atomic_json(state_path, state)
        print('SwanLab run: ' + str(state['url']), flush=True)
        records = iter_records(metrics_path, args.follow_pid)
        for step, values in grouped_scalars(records, config):
            sdk.log(values, step=step)
            state.update(records=state['records'] + 1, last_transition=step)
            if state['records'] % 25 == 0:
                atomic_json(state_path, state)
        sdk.finish()
        state.update(status='completed' if args.swanlab_mode == 'online' else 'offline_ready',
                     metrics_bytes=metrics_path.stat().st_size,
                     metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
                     finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        return state
    except Exception as error:
        message = str(error)
        if key:
            message = message.replace(key, '[REDACTED]')
        state.update(status='failed', error_type=type(error).__name__, error=message[:500])
        atomic_json(state_path, state)
        print(f"SwanLab upload failed ({type(error).__name__}): {message[:500]}", file=sys.stderr)
        # Preserve local metrics; a fresh replay upload can recover without RL.
        return state


def run_with_upload(command, *, root, env, output, options):
    """Preserve learner exit status; reporting errors never fail its training."""
    if not options.swanlab or options.swanlab_mode == 'disabled' or output is None:
        return subprocess.run(command, cwd=root, env=env, check=True)
    output = Path(output)
    output = (output if output.is_absolute() else Path(root) / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    upload_command = [str(options.swanlab_python), str(ROOT/'scripts/swanlab_upload.py'),
                      '--run-dir', str(output), '--swanlab-project', options.swanlab_project,
                      '--swanlab-mode', options.swanlab_mode,
                      '--swanlab-key-file', str(options.swanlab_key_file)]
    for name in ('workspace', 'run_name'):
        value = getattr(options, 'swanlab_' + name)
        if value:
            upload_command += ['--swanlab-' + name.replace('_', '-'), value]
    uploader = None
    log_path = output.with_name(output.name + '.swanlab.log')
    with subprocess.Popen(command, cwd=root, env=env) as learner:
        try:
            with log_path.open('a') as log:
                uploader = subprocess.Popen(upload_command + ['--follow-pid', str(learner.pid)],
                                            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f'SwanLab uploader log: {log_path}', file=sys.stderr, flush=True)
        except OSError as error:
            print(f'SwanLab uploader unavailable ({type(error).__name__}); local training continues. '
                  'Run scripts/setup_swanlab.sh and upload the saved metrics later.', file=sys.stderr)
        try:
            returncode = learner.wait()
        except BaseException:
            learner.terminate()
            try:
                learner.wait(timeout=10)
            except subprocess.TimeoutExpired:
                learner.kill()
                learner.wait()
            raise
        finally:
            if uploader is not None:
                try:
                    uploader.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    uploader.terminate()
                    try:
                        uploader.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        uploader.kill()
                        uploader.wait()
                    print('SwanLab flush timed out; metrics.jsonl is retained for upload retry.', file=sys.stderr)
        if uploader is not None and uploader.returncode:
            print(f'SwanLab upload needs attention; see {log_path}. Local training results are retained.', file=sys.stderr)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return subprocess.CompletedProcess(command, returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--follow-pid', type=int)
    parser.add_argument('--force', action='store_true', help='Create a new upload even if already completed')
    add_arguments(parser)
    args = parser.parse_args()
    if args.swanlab_mode == 'disabled':
        return
    result = upload(args)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
