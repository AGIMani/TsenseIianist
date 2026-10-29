"""Check host replay capacity before committing a large pageable allocation.

No CUDA, model construction or allocation occurs in this module. The pool is
still FP32; Slurm memory is host RAM, independent of the learner's GPU VRAM.
"""
import os
from pathlib import Path


def replay_bytes(state_dim, action_dim, capacity):
    return capacity * (4 * (2 * state_dim + action_dim + 2) + 2)


def host_memory_limits():
    limits = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                limits['host_available_bytes'] = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    # Include every enclosing cgroup v2 limit, not just a possibly unlimited leaf.
    try:
        group = next(line.split(':', 2)[2] for line in
                     Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        base = Path('/sys/fs/cgroup')
        current = base / group.lstrip('/')
        candidates = []
        while current.is_relative_to(base):
            maximum = current / 'memory.max'
            usage = current / 'memory.current'
            if maximum.is_file() and usage.is_file():
                value = maximum.read_text().strip()
                if value != 'max':
                    candidates.append(max(0, int(value) - int(usage.read_text())))
            if current == base:
                break
            current = current.parent
        if candidates:
            limits['cgroup_available_bytes'] = min(candidates)
    except (OSError, ValueError, StopIteration):
        pass
    # The job request remains a useful upper bound when cgroups are not exposed.
    if os.environ.get('SLURM_MEM_PER_NODE', '').isdigit():
        limits['slurm_requested_bytes'] = int(os.environ['SLURM_MEM_PER_NODE']) * 1024**2
    elif (os.environ.get('SLURM_MEM_PER_CPU', '').isdigit()
          and os.environ.get('SLURM_CPUS_PER_TASK', '').isdigit()):
        limits['slurm_requested_bytes'] = (int(os.environ['SLURM_MEM_PER_CPU'])
                                            * int(os.environ['SLURM_CPUS_PER_TASK']) * 1024**2)
    return limits


def check_host_replay_memory(state_dim, action_dim, capacity):
    pool = replay_bytes(state_dim, action_dim, capacity)
    # Do not allow an apparently successful virtual allocation to OOM later in
    # prefill. This headroom is a minimum check, not a peak-RSS guarantee.
    headroom = max(1024**3, pool // 10)
    limits = host_memory_limits()
    required = pool + headroom
    if not limits:
        raise RuntimeError('Cannot verify CPU replay host memory capacity')
    available = min(limits.values())
    if available < required:
        raise MemoryError(
            f'CPU replay needs {pool / 1024**3:.2f} GiB plus '
            f'{headroom / 1024**3:.2f} GiB headroom; available/job limit is '
            f'{available / 1024**3:.2f} GiB. Increase the Slurm host-memory request '
            'through submit.sh --mem-gb; GPU memory cannot supply this pool.')
    return dict(pool_bytes=pool, minimum_headroom_bytes=headroom,
                required_bytes=required, limits=limits,
                scope='minimum_capacity_check_not_peak_memory_validation')
