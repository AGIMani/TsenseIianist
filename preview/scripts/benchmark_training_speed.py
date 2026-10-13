#!/usr/bin/env python3
"""Time the real training entry point and record GPU telemetry without tuning it.

Run with system Python from the standalone project. Extra arguments after --
are forwarded to run.sh; this script never changes its defaults. Benchmark
checkpoints are short experiments and do not establish policy quality.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vector-steps', type=int, default=240)
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.vector_steps < 1:
        parser.error('--vector-steps must be positive')
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    run_prefix = root/'training_mjlab/outputs'/out.name
    command = ['bash', 'run.sh', '--vector-steps', str(args.vector_steps),
               '--no-progress', '--output-dir', str(run_prefix), *extra]
    fields = ['timestamp', 'utilization.gpu', 'utilization.memory',
              'memory.used', 'power.draw', 'temperature.gpu',
              'clocks.sm', 'clocks.mem', 'pstate']
    monitor_command = ['nvidia-smi', '--query-gpu='+','.join(fields),
                       '--format=csv,nounits', '--loop-ms=200']
    started = time.perf_counter()
    with (out/'gpu.csv').open('w') as gpu, (out/'train.log').open('w') as log:
        monitor = subprocess.Popen(monitor_command, stdout=gpu, stderr=subprocess.STDOUT)
        try:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    runs = sorted(run_prefix.parent.glob(run_prefix.name+'-*'))
    report = dict(command=command, monitor_command=monitor_command,
                  finished_at=datetime.datetime.now().astimezone().isoformat(),
                  wall_seconds=time.perf_counter()-started, returncode=result.returncode,
                  run_dir=str(runs[-1]) if runs else None,
                  note='Wall time includes source rebuild and startup. See summary.json for trainer time.')
    (out/'invocation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
