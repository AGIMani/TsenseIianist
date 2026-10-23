"""Launch and monitor a local worker in a sibling systemd scope.

Only standard-library code runs in the UI/supervisor. Heavy imports take place
after the worker verifies its effective cgroup limits.
"""
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

from fingering_editor_limits import WORKER_HIGH, WORKER_MAX, available_memory, MIN_AVAILABLE, cgroup_directory
from fingering_editor_config import connection_config


class LimitedWorker:
    def __init__(self, root, *, prepare=False):
        self.root, self.prepare = Path(root), prepare
        if available_memory() < MIN_AVAILABLE:
            raise RuntimeError('本地构建需要至少 4 GiB 可用内存；当前不足，未启动模型')
        self.unit = 'pianist-fingering-worker-' + uuid.uuid4().hex[:12] + '.scope'
        folder = self.root / 'logs/fingering_editor'
        folder.mkdir(parents=True, exist_ok=True)
        self.log_path = folder / (self.unit + '.log')
        self.report_path = folder / (self.unit + '.json')
        self.stats = dict(unit=self.unit, started_at=time.time(), reason=None, peak_bytes=0,
                          minimum_available_bytes=available_memory(), max_bytes=WORKER_MAX,
                          high_bytes=WORKER_HIGH, swap_max_bytes=0, samples=[], mode='prepare' if prepare else 'model')
        if prepare:
            python = str(self.root / '.venv/bin/python')
            script = self.root / 'scripts/prepare_local_fingering_editor.py'
            args = []
        else:
            config = json.loads(connection_config(self.root, 'local').read_text())
            runtime = Path(config['runtime_root'])
            import hashlib
            script = runtime / 'scripts/fingering_pose_worker.py'
            if hashlib.sha256(script.read_bytes()).hexdigest() != config['worker_sha256']:
                raise ValueError('Local worker SHA mismatch; rerun preparation')
            helper = runtime / 'scripts/fingering_editor_limits.py'
            if hashlib.sha256(helper.read_bytes()).hexdigest() != config['helper_sha256']:
                raise ValueError('Local helper SHA mismatch; rerun preparation')
            python = config['python']
            args = ['--local', '--cache', str(folder / ('cache-' + self.unit.removesuffix('.scope')))]
        self.command = [
            'systemd-run', '--user', '--scope', '--quiet', '--unit=' + self.unit,
            '-p', 'MemoryHigh=1024M', '-p', 'MemoryMax=1536M', '-p', 'MemorySwapMax=0',
            '-p', 'TasksMax=128',
            '--', 'env', 'CUDA_VISIBLE_DEVICES=', 'MUJOCO_GL=disable',
            'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'NUMEXPR_NUM_THREADS=1', python, str(script), *args]
        parent_scope = cgroup_directory().name
        if parent_scope.startswith('pianist-fingering-ui-') and parent_scope.endswith('.scope'):
            self.command[5:5] = ['-p', 'BindsTo=' + parent_scope, '-p', 'After=' + parent_scope]
        self.stats['command'] = self.command
        self.log = self.log_path.open('w')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        self.stats['launcher_pid'] = self.process.pid
        self.group = None
        self.effective_verified = False
        self.initialized = False
        self.started_monotonic = time.monotonic()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def stop_scope(self, reason):
        self.stats['reason'] = reason
        subprocess.run(['systemctl', '--user', 'stop', self.unit], capture_output=True, timeout=5)

    def monitor(self):
        try:
            for _ in range(100):
                result = subprocess.run(['systemctl', '--user', 'show', self.unit, '-p', 'ControlGroup', '--value'],
                                        capture_output=True, text=True, timeout=3)
                value = result.stdout.strip()
                if value:
                    self.group = Path('/sys/fs/cgroup') / value.lstrip('/')
                    break
                if self.process.poll() is not None:
                    return
                if self.stopping.wait(.05):
                    return
            if not self.group:
                self.stop_scope('Could not verify isolated cgroup')
                return
            (self.group / 'memory.oom.group').write_text('1')
            values = {name: (self.group / name).read_text().strip() for name in
                      ('memory.high', 'memory.max', 'memory.swap.max', 'memory.oom.group')}
            expected = {'memory.high': str(WORKER_HIGH), 'memory.max': str(WORKER_MAX),
                        'memory.swap.max': '0', 'memory.oom.group': '1'}
            self.stats['effective_limits'] = values
            self.stats['cgroup'] = str(self.group)
            if values != expected:
                self.stop_scope('Effective kernel limits do not match requested budget')
                return
            self.effective_verified = True
            while self.process.poll() is None and not self.stopping.wait(.25):
                try:
                    current = int((self.group / 'memory.current').read_text())
                    kernel_peak = int((self.group / 'memory.peak').read_text())
                    available = available_memory()
                    sample = dict(t=time.monotonic(), memory_bytes=current, available_bytes=available)
                    memory_stat = dict(line.split() for line in (self.group/'memory.stat').read_text().splitlines())
                    sample.update(anon_bytes=int(memory_stat['anon']), file_bytes=int(memory_stat['file']))
                    self.stats['samples'].append(sample)
                    # Bounded diagnostics even during a long manual editor session.
                    if len(self.stats['samples']) > 2400:
                        del self.stats['samples'][:1200]
                    self.stats['peak_bytes'] = max(self.stats['peak_bytes'], kernel_peak)
                    self.stats['minimum_available_bytes'] = min(self.stats['minimum_available_bytes'], available)
                    events = dict(line.split() for line in (self.group / 'memory.events').read_text().splitlines())
                    self.stats['memory_events'] = {key: int(value) for key, value in events.items()}
                    if current > int(WORKER_MAX * .80):
                        self.stop_scope('Proactive stop: worker exceeded 80% of 1.5 GiB cap')
                        return
                    if available < 2 * 1024**3:
                        self.stop_scope('Proactive stop: desktop MemAvailable below 2 GiB')
                        return
                    pressure = Path('/proc/pressure/memory').read_text().splitlines()
                    full = float(pressure[1].split('avg10=')[1].split()[0])
                    if full > 1.0:
                        self.stop_scope('Proactive stop: global memory full-pressure avg10 exceeded 1%')
                        return
                    if not self.initialized and time.monotonic()-self.started_monotonic > 180:
                        self.stop_scope('Proactive stop: model initialization exceeded 180 seconds')
                        return
                except FileNotFoundError:
                    break
        except Exception as exc:
            self.stats['monitor_error'] = str(exc)
            try:
                self.stop_scope('Resource monitor failed; refusing unmonitored local work')
            except Exception:
                self.process.terminate()
        finally:
            self.report_path.write_text(json.dumps(self.stats, indent=2) + '\n')

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.stop_scope('Requested session shutdown')
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.stopping.set()
        self.thread.join(timeout=6)
        self.stats['exit_code'] = self.process.poll()
        self.stats['finished_at'] = time.time()
        self.report_path.write_text(json.dumps(self.stats, indent=2) + '\n')
        self.log.close()


def prepare_runtime(root):
    worker = LimitedWorker(root, prepare=True)
    try:
        stdout, _ = worker.process.communicate(timeout=180)
        if worker.process.returncode:
            raise RuntimeError(f'Local preparation failed: {worker.stats.get("reason")}; {worker.log_path}')
        return stdout.strip()
    finally:
        worker.close()
