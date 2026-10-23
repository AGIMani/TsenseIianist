#!/usr/bin/env python3
"""Bounded local preview pressure test; no dynamics, GPU, or intentional OOM.

Runs a cold source rebuild, deterministic solve cases, timed repeated solves,
a mailbox burst, and (optionally) a second cold restart. Records kernel cgroup
memory, PSI-related proactive stops, real latencies, and solver invariants.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import queue
import statistics
import sys
import threading
import time

from fingering_editor_data import Score
from fingering_editor_config import connection_config
from fingering_editor_limits import available_memory, MIN_AVAILABLE
from fingering_local_runner import LimitedWorker, prepare_runtime

ROOT = Path(__file__).resolve().parents[1]


def read_with_timeout(process, seconds):
    mailbox = queue.Queue(maxsize=1)

    def reader():
        try:
            line = process.stdout.readline()
            mailbox.put(json.loads(line) if line else {'type': 'error', 'error': 'worker pipe closed'})
        except Exception as exc:
            mailbox.put({'type': 'error', 'error': str(exc)})

    threading.Thread(target=reader, daemon=True).start()
    return mailbox.get(timeout=seconds)


def invariants(pose):
    import math
    if pose['type'] != 'pose':
        raise RuntimeError(pose)
    assert pose['physical_key_positions_unchanged']
    assert all(59.94 <= angle <= 80.06 for angle in pose['coupling_angles'])
    assert len(pose['tips']) == 10
    assert all(math.isfinite(v) for v in pose['qpos'])
    assert all(math.isfinite(v) for t in pose['transforms'] for v in t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=120, help='Sustained solve duration, after cold rebuild')
    parser.add_argument('--restarts', type=int, choices=(1, 2), default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 600:
        parser.error('--seconds must be 5–600, without changing resource caps')
    out = args.output or ROOT/'results'/('fingering_local_stress_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'results/exper024/input_sources/015-5_fingered.proto'
    report = dict(status='running', started_at=time.time(), output=str(out), stages=[],
                  requested_seconds=args.seconds, requested_restarts=args.restarts,
                  source_proto=str(source), source_proto_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  available_before_bytes=available_memory(), gpu=False, physics=False, intentional_oom=False)
    score = Score(source)
    local_worker = None
    try:
        if report['available_before_bytes'] < MIN_AVAILABLE:
            report['status'] = 'refused_insufficient_memory'
            raise RuntimeError('Need 4 GiB MemAvailable; no heavy process was started')
        print('Preparing SHA-verified local runtime inside 1.5 GiB scope', flush=True)
        prepare_runtime(ROOT)
        report['runtime'] = json.loads(connection_config(ROOT, 'local').read_text())
        timings = []
        poses = []
        cases = [[], *[[dict(finger=f, pitch=69 if f < 5 else 57)] for f in range(10)],
                 [dict(finger=3, pitch=21)], [dict(finger=8, pitch=108)],
                 [dict(finger=1, pitch=60), dict(finger=1, pitch=64)],
                 [dict(finger=f, pitch=48+f*3) for f in range(10)]]
        boundaries = sorted({n.start_time for n in score.seq.notes})
        cases.extend([[dict(finger=n.part, pitch=n.pitch) for _, n in score.active(t)]
                      for t in boundaries[::max(1, len(boundaries)//15)]])
        count = 0
        for restart in range(args.restarts):
            started = time.monotonic()
            local_worker = LimitedWorker(ROOT)
            print(f'Cold start {restart+1}: {local_worker.unit}', flush=True)
            ready = read_with_timeout(local_worker.process, 180)
            if ready['type'] != 'ready':
                raise RuntimeError(ready)
            local_worker.initialized = True
            stage = dict(name='cold_start', index=restart+1, seconds=time.monotonic()-started,
                         pid=ready['pid'], version=ready['version'], source_sha256=ready['source_sha256'])
            report['stages'].append(stage)
            (out/f'ready_{restart}.json').write_text(json.dumps(ready))
            print('Ready', stage, flush=True)
            deadline = time.monotonic() + args.seconds
            i = 0
            while i < len(cases) or time.monotonic() < deadline:
                active = cases[i % len(cases)]
                request = dict(request_id=count, active=active)
                start = time.monotonic()
                local_worker.process.stdin.write(json.dumps(request)+'\n')
                local_worker.process.stdin.flush()
                pose = read_with_timeout(local_worker.process, 15)
                invariants(pose)
                timings.append(time.monotonic()-start)
                count += 1
                if i < len(cases):
                    poses.append(dict(case=i, request=request, pose=pose))
                i += 1
                if local_worker.stats.get('reason'):
                    raise RuntimeError(local_worker.stats['reason'])
                if i % 50 == 0:
                    print(f'{count} solves; peak={local_worker.stats["peak_bytes"]/1024**2:.1f} MiB; '
                          f'min available={local_worker.stats["minimum_available_bytes"]/1024**3:.2f} GiB', flush=True)
            local_worker.close()
            report['stages'].append(dict(name='sustained_solve', count=i, resources=local_worker.stats))
            local_worker = None
        # Mailbox logic is the same one used by actual GUI timeline scrubbing;
        # the burst never submits thousands of pipe requests to the solver.
        from fingering_editor import RemotePreview
        preview = object.__new__(RemotePreview)
        preview.requests = queue.Queue(maxsize=1)
        for i in range(10000):
            preview.submit(dict(request_id=i, active=cases[i % len(cases)]))
        assert preview.requests.qsize() == 1
        assert preview.requests.get_nowait()['request_id'] == 9999
        report['stages'].append(dict(name='timeline_burst', requests=10000, pending_max=1, final_id=9999))
        timings.sort()
        report['latency_seconds'] = dict(count=count, median=statistics.median(timings),
                                        p95=timings[min(len(timings)-1, int(.95*len(timings)))], maximum=max(timings))
        (out/'poses.json').write_text(json.dumps(poses))
        report['status'] = 'passed'
    except Exception as exc:
        if report['status'] == 'running':
            report['status'] = 'failed'
        report['reason'] = str(exc)
    finally:
        if local_worker:
            local_worker.close()
            report['failed_worker_resources'] = local_worker.stats
            if (local_worker.stats.get('reason') or '').startswith('Proactive stop:'):
                report['status'] = 'stopped_by_memory_guard'
        report['finished_at'] = time.time()
        report['available_after_bytes'] = available_memory()
        for file in ('scripts/fingering_pose_worker.py', 'scripts/fingering_editor_limits.py',
                     'scripts/fingering_local_runner.py', 'scripts/stress_fingering_editor.py'):
            report.setdefault('code_sha256', {})[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: v for k,v in report.items() if k not in ('stages', 'runtime', 'code_sha256', 'failed_worker_resources')}, ensure_ascii=False), flush=True)
        print(out/'report.json', flush=True)
    if report['status'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
