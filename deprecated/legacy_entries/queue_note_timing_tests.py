#!/usr/bin/env python3
"""Queue isolated timing experiments behind validation using one pinned source.

Without --submit, only print the plan. Every submitted action uses submit.sh;
the source archive is made before submission and its checksum is carried into
Slurm, so later workspace edits cannot alter queued experimental code.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]


def training_arguments(*, exact, staged=False, permission=0., steps=5860, seed=42):
    # Explicit user-selected controls are identical across all four groups.
    return ['--finger-coupling-angle-ranges','configs/finger_coupling/angle_ranges_60_80.json',
            '--no-coupling-reduce-observation',
            '--reward-config','configs/rewards/coupled_40hz_gesture10_20260930.json',
            '--exact-note-timing' if exact else '--no-exact-note-timing',
            '--staged-fingering-reward' if staged else '--no-staged-fingering-reward',
            '--upcoming-press-permission-time',str(permission),
            '--anticipatory-fingering-time','.15','--no-anticipatory-current-priority',
            '--wrist-tx-max-velocity','1','--wrist-ty-max-velocity','1',
            '--num-envs','512','--vector-steps',str(steps),'--seed',str(seed)]


def make_plan(output, *, steps=5860, seed=42):
    validation=['tests/test_exact_note_timing.py','tests/test_exact_timing_entry.py',
                'tests/test_note_timing_metrics.py','tests/test_batched_evaluation.py',
                'tests/test_upcoming_press_permission.py',
                'tests/test_strike_timing_diagnostic.py','tests/test_exact_timing_gpu.py',
                'tests/test_target_key_reward.py','tests/test_substep_rewards.py',
                'tests/test_fingering_experiments.py','tests/test_reward_time_units.py',
                'tests/test_hpc_40hz_entry.py']
    jobs=[dict(label='validation',mode='validate',arguments=validation),
          dict(label='strike_interventions',mode='strike-diagnostic',arguments=[
              '--recording',str(ROOT/'logs/hpc/1353/export_work/recording'),
              '--run',str(ROOT/'training_mjlab/outputs/hpc/gesture10-comparison-20260930-140658-445247'),
              '--output',str(output/'strike_interventions')]),
          dict(label='functional_smoke',mode='train',arguments=[
              *training_arguments(exact=True,staged=True,permission=.05,steps=420,seed=seed),
              '--num-envs','32','--replay-capacity','65536','--batch-size','256',
              '--eval-interval','200','--eval-episodes','1','--eval-num-envs','1',
              '--no-eval-at-start','--save-interval','1000',
              '--output-dir',str(output/'functional_smoke')])]
    for label,exact,staged,permission in (
        ('A_grid_baseline',False,False,0.),('B_exact_timing',True,False,0.),
        ('C_exact_staged',True,True,0.),('D_exact_upcoming50ms',True,False,.05)):
        jobs.append(dict(label=label,mode='train',arguments=[
            *training_arguments(exact=exact,staged=staged,permission=permission,steps=steps,seed=seed),
            '--output-dir',str(output/label)],transitions=steps*512,
            changed_from=('A_grid_baseline' if label=='B_exact_timing' else
                          'B_exact_timing' if label.startswith(('C_','D_')) else None)))
    return dict(schema='note_timing_comparison_v1',output=str(output),seed=seed,
                source_archive=str(output/'source.tar.gz'),
                common=dict(control_hz=40,physics_hz=400,coupling='range_60_80_deg',
                            gesture_score_multiplier=10,gesture_weight=1,
                            wrist_tx_ty_max_velocity=1.,anticipation_seconds=.15,
                            current_priority=False,num_envs=512),
                note='Functional smoke combines flags only to validate interoperability; A/B/C/D are separate comparisons. No convergence claim from smoke.',
                jobs=jobs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--vector-steps',type=int,default=5860)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--allocation',type=int,help='Wait for and reuse this existing 48-hour Slurm allocation')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args(argv)
    if args.vector_steps<1:parser.error('--vector-steps must be positive')
    if args.allocation is not None and args.allocation<1:parser.error('--allocation must be positive')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    output=(args.output or ROOT/'training_mjlab/outputs'/('note_timing_tests-'+stamp)).resolve()
    plan=make_plan(output,steps=args.vector_steps,seed=args.seed)
    if args.allocation is not None:
        plan['allocation_id']=str(args.allocation)
        plan['execution']='sequential_steps_in_existing_allocation'
    if not args.submit:
        print(json.dumps(plan,indent=2,ensure_ascii=False));return
    output.mkdir(parents=True,exist_ok=False)
    manifest=output/'manifest.json'
    manifest.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    spec=importlib.util.spec_from_file_location('timing_source_package',ROOT/'scripts/hpc/package_a32.py')
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    plan['source']=package.package(ROOT,Path(plan['source_archive']))
    manifest.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    if args.allocation is not None:
        # Freeze the stdlib coordinator as well as the training payload. The
        # detached process only queries Slurm until it can use submit.sh srun.
        runner=output/'allocation_queue.py'
        shutil.copy2(ROOT/'scripts/hpc/run_timing_allocation_queue.py',runner)
        command=[sys.executable,str(runner),'--manifest',str(manifest),'--project-root',str(ROOT)]
        with (output/'coordinator.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (output/'launcher.json').write_text(json.dumps(dict(pid=process.pid,command=command),indent=2)+'\n')
        print(f'Allocation {args.allocation}: queued {len(plan["jobs"])} sequential steps; coordinator PID {process.pid}')
        print('Manifest: '+str(manifest))
        return
    previous=None
    for job in plan['jobs']:
        command=['bash',str(ROOT/'scripts/hpc/submit.sh'),'--source-archive',plan['source_archive']]
        if previous is not None:
            command.extend(['--dependency','afterok:'+previous])
            job['dependency']='afterok:'+previous
        command.extend([job['mode'],*job['arguments']])
        # Persist each accepted job immediately; a later submission failure
        # must not lose IDs of already queued work or submit duplicate jobs.
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        if result.returncode:
            job.update(submission_error=result.stderr,submission_returncode=result.returncode)
            manifest.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
            raise RuntimeError(f'Slurm rejected {job["label"]}: {result.stderr.strip()}')
        job_id=result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            job.update(submission_output=result.stdout,submission_error='Unexpected Slurm response; inspect before resubmitting')
            manifest.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
            raise RuntimeError('Unexpected Slurm job ID: '+result.stdout)
        previous=job_id
        job.update(job_id=job_id,submitted_command=command)
        manifest.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
        print(f'{job["label"]}: {job_id} ({job.get("dependency","ready")})',flush=True)
    print('Manifest: '+str(manifest))


if __name__=='__main__':main()
