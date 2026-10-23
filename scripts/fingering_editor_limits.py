"""Fixed desktop budgets; verify kernel cgroup limits before local model imports."""
from pathlib import Path

WORKER_HIGH = 1024 * 1024**2
WORKER_MAX = 1536 * 1024**2
UI_HIGH = 384 * 1024**2
UI_MAX = 512 * 1024**2
MIN_AVAILABLE = 4 * 1024**3


def available_memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def cgroup_directory():
    relative = next(line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    return Path('/sys/fs/cgroup') / relative.lstrip('/')


def verify_limits(*, role='worker', check_available=True):
    group = cgroup_directory()
    cap = WORKER_MAX if role == 'worker' else UI_MAX
    high = WORKER_HIGH if role == 'worker' else UI_HIGH
    values = {name: (group / name).read_text().strip() for name in
              ('memory.max', 'memory.high', 'memory.swap.max', 'memory.oom.group')}
    if (values['memory.max'] == 'max' or not 0 < int(values['memory.max']) <= cap
            or values['memory.high'] == 'max' or not 0 < int(values['memory.high']) <= high
            or values['memory.swap.max'] != '0'):
        raise RuntimeError(f'{role}: missing fixed isolated cgroup limits: {group}: {values}')
    # Only mutate a group AFTER its hard/soft budgets have been verified.
    # Never change the invoking terminal or desktop scope on a refused launch.
    # systemd 255 lacks MemoryOOMGroup= for scopes; use its delegated file.
    (group / 'memory.oom.group').write_text('1')
    values['memory.oom.group'] = (group/'memory.oom.group').read_text().strip()
    if values['memory.oom.group'] != '1':
        raise RuntimeError('Could not enable whole-group OOM containment')
    available = available_memory()
    if check_available and available < MIN_AVAILABLE:
        raise RuntimeError(f'Need 4 GiB MemAvailable before starting; have {available / 1024**3:.2f} GiB')
    return dict(cgroup=str(group), available_bytes=available, **values)
