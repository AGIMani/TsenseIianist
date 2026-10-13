#!/usr/bin/env python3
"""Run three user-requested ablations against the completed Twinkle A baseline."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_speed_comparison import now, write

ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    ('C_no_anticipation', 0., True),
    ('D_no_false_press', .15, False),
    ('E_no_anticipation_no_false_press', 0., False),
)


def overrides(anticipation, false_press):
    args = ['--no-eval-metrics-graph', '--anticipatory-fingering-time', str(anticipation)]
    if not false_press:
        args += ['--disable-reward', 'false_press_penalty']
    return args


def validate_config(config, baseline, anticipation, false_press):
    expected = deepcopy(baseline)
    expected['reward_config']['performance']['anticipatory_fingering_time'] = anticipation
    expected['reward_config']['terms']['false_press_penalty']['enabled'] = false_press
    # The existing switch removes 50 future-finger features and their replay storage.
    expected['observation_size'] = 2297 if anticipation else 2247
    expected['replay_bytes'] -= 0 if anticipation else 2 * 50 * 4 * baseline['replay_capacity']
    for key in set(config) | set(expected):
        if key != 'cli' and config.get(key) != expected.get(key):
            raise ValueError('Unexpected effective-config difference: '+key)
    for key in ('seed', 'vector_steps', 'num_envs', 'eval_interval', 'eval_episodes',
                'eval_num_envs', 'save_interval', 'save_forward', 'checkpoint', 'resume'):
        if config['cli'].get(key) != baseline['cli'].get(key):
            raise ValueError('Unexpected CLI difference: '+key)


def validate_ready_config(config, baseline, anticipation, false_press):
    if 'replay_bytes' not in config or 'observation_size' not in config:
        return False
    validate_config(config, baseline, anticipation, false_press)
    return True


def summarize(run, baseline, anticipation, false_press):
    config=json.loads((run/'config.json').read_text())
    validate_config(config,baseline,anticipation,false_press)
    summary=json.loads((run/'summary.json').read_text())
    assert summary['vector_steps']==19532 or summary['stop_reason']=='quality_target_reached'
    assert summary['transitions']==summary['vector_steps']*512
    assert summary['replay_insertions']==summary['transitions']
    expected=max(0,summary['transitions']-config['learning_start_transition_1_based']+1)//4
    assert summary['updates_this_run']==expected
    pipeline=summary['collector_pipeline_state']
    assert pipeline['pending_vector_step'] is None
    assert pipeline['learner_updated_through_vector_step']==summary['vector_steps']
    assert (run/'final.pt').is_file() and not list(run.glob('step-*.pt'))
    if summary['vector_steps']>=1000:assert (run/'forward.pt').is_file()
    evaluation=summary['final_evaluation']
    assert evaluation['note_timing_reference_count']==102
    for key in ('collision_diagnostics_valid','key_contact_diagnostics_valid','substep_key_diagnostics_valid'):
        assert evaluation[key],key
    best=json.loads((run/'best.json').read_text())
    return dict(run=str(run),steps=summary['vector_steps'],transitions=summary['transitions'],
        updates=expected,seconds=summary['elapsed_s'],stop_reason=summary['stop_reason'],
        final_exact_f1=evaluation['exact_original_music_micro_f1'],
        final_music_f1_min=evaluation['music_only_f1_min'],best_step=best['vector_step'],
        best_exact_f1=best['evaluation']['exact_original_music_micro_f1'],
        hand_collision_substeps=evaluation['hand_collision_substeps'],
        key_underside_substeps_max=evaluation['key_underside_substeps_max'],
        missed_notes=evaluation['note_timing_missed_count'],full_diagnostics=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation',required=True,type=int)
    parser.add_argument('--baseline-queue',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.detach:
        with (args.output/'coordinator.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                *[x for x in sys.argv[1:] if x!='--detach']],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,output=str(args.output))));return
    lock=(args.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=args.output/'runtime.json'
    if state_path.exists():raise RuntimeError('Existing queue state; no implicit retry')
    old=json.loads(args.baseline_queue.read_text());assert old['status']=='completed'
    ref=next(st for st in old['stages'] if st['label']=='A_baseline')
    baseline_path=Path(ref['result']['run'])/'config.json'
    baseline=json.loads(baseline_path.read_text())
    archive=Path(old['archive']);archive_sha=old['archive_sha256']
    assert baseline['cli']['seed']==42 and baseline['requested_vector_steps']==19532
    assert baseline['reward_config']['performance']['anticipatory_fingering_time']==.15
    assert baseline['reward_config']['terms']['false_press_penalty']['enabled']
    state=dict(status='running',pid=os.getpid(),started_at=now(),updated_at=now(),
        allocation=args.allocation,archive=str(archive),archive_sha256=archive_sha,
        baseline_run=str(baseline_path.parent),baseline_config_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        seed=42,requested_vector_steps=19532,requested_transitions=10000384,
        wrong_key_scope='Only false_press_penalty disabled; key_press_reward unchanged',
        stages=[dict(label=label,anticipation=anticipation,false_press=false_press,status='pending')
            for label,anticipation,false_press in STAGES])
    write(state_path,state)
    child=None
    try:
        for stage in state['stages']:
            assert hashlib.sha256(archive.read_bytes()).hexdigest()==archive_sha
            allocation=subprocess.check_output(['squeue','-h','-j',str(args.allocation),'-o','%u %T %j'],text=True).strip()
            assert allocation==os.environ['USER']+' RUNNING TsenseI-reserve2',allocation
            steps=subprocess.check_output(['squeue','-h','--steps','-j',str(args.allocation),'-o','%i'],text=True).split()
            assert all(x.endswith(('.batch','.extern')) for x in steps),steps
            command=['bash',str(ROOT/'scripts/hpc/submit.sh'),'--allocation',str(args.allocation),
                '--source-archive',str(archive),'train','--seed','42','--vector-steps','19532',
                '--output-dir',str(args.output/stage['label']),
                *overrides(stage['anticipation'],stage['false_press'])]
            stage.update(status='running',started_at=now(),command=command)
            state['active_stage']=stage['label'];write(state_path,state)
            with (args.output/(stage['label']+'.submit.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                stage['submit_pid']=child.pid
                while child.poll() is None:
                    state['updated_at']=now()
                    stage['slurm_steps']=subprocess.check_output(['squeue','-h','--steps','-j',str(args.allocation),'-o','%i'],text=True).split()
                    runs=[r for r in args.output.glob(stage['label']+'-*') if (r/'config.json').exists()]
                    stage['output_dirs']=[str(r) for r in runs]
                    if runs and not stage.get('config_verified'):
                        assert len(runs)==1
                        try:config=json.loads((runs[0]/'config.json').read_text())
                        except json.JSONDecodeError:pass  # The trainer may still be writing config.json.
                        else:
                            # train.py writes an initial source contract, then the
                            # complete config after creating the learner/replay.
                            stage['config_verified']=validate_ready_config(
                                config,baseline,stage['anticipation'],stage['false_press'])
                    write(state_path,state);time.sleep(5)
            stage['exit_code']=child.returncode
            if child.returncode:raise RuntimeError(f'{stage["label"]} failed; no retry, allocation retained')
            runs=[r for r in args.output.glob(stage['label']+'-*') if (r/'summary.json').exists()]
            assert len(runs)==1
            stage['result']=summarize(runs[0],baseline,stage['anticipation'],stage['false_press'])
            stage.update(status='completed',finished_at=now());write(state_path,state)
        state.update(status='completed',active_stage=None,updated_at=now());write(state_path,state)
    except BaseException as error:
        # srun may exit on SIGINT without cancelling its remote step. Only
        # cancel this stage's numeric child steps, never the retained batch.
        for step in stage.get('slurm_steps', []):
            parts=step.split('.')
            if len(parts)==2 and parts[0]==str(args.allocation) and parts[1].isdigit():
                subprocess.run(['scancel',step],check=False)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.terminate()
        stage.update(status='failed',error=repr(error))
        state.update(status='failed',error=repr(error),updated_at=now());write(state_path,state);raise


if __name__=='__main__':main()
