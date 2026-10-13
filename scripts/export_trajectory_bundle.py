"""Submit recorded hand trajectories through the existing Slurm export entry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory-bundle', type=Path, required=True)
    p.add_argument('--recorded-data', type=Path, required=True,
                   help='Matching physics_trace.npz and original_{config,report}.json / original_scene.xml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    # Validation and scientific libraries use the project environment, not system Python.
    script = ROOT/'preview/scripts/render_trajectory_bundle.py'
    command = [str(ROOT/'training/.venv/bin/python'), str(script), '--validate',
               str(a.trajectory_bundle.resolve()), str(a.recorded_data.resolve())]
    subprocess.run(command, check=True)
    if a.validate_only:
        return
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'; inputs.mkdir()
    files = {'trajectory_bundle.zip': a.trajectory_bundle}
    for name in ('physics_trace.npz', 'original_config.json', 'original_report.json', 'original_scene.xml'):
        files[name] = a.recorded_data/name
    hashes = {}
    for name, source in files.items():
        shutil.copy2(source, inputs/name)
        hashes[name] = hashlib.sha256((inputs/name).read_bytes()).hexdigest()
    plan = dict(kind='recorded_trajectory_current_model', output=str(output), input_sha256=hashes,
                videos=[dict(arms=False)], plots=[], fps=40, hold=1.0,
                scope='Current hand geometry, recorded joint positions and hand-root poses; original recorded piano keys. No new dynamics or policy.')
    path = output/'export_plan.json'; path.write_text(json.dumps(plan, indent=2)+'\n')
    result = subprocess.run(['bash', str(ROOT/'scripts/hpc/submit.sh'), '--gpus', '1',
                             'export', '--plan', str(path)], text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    job = result.stdout.strip().split(';')[0]
    (output/'submission.json').write_text(json.dumps(dict(job_id=job, plan=str(path)), indent=2)+'\n')
    print(json.dumps(dict(job_id=job, output=str(output)), indent=2))


if __name__ == '__main__':
    main()
