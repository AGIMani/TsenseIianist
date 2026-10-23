#!/usr/bin/env python3
"""Equal-budget Twinkle speed comparisons, sequentially inside a retained allocation.

This CPU coordinator submits all GPU work through submit.sh. It never cancels
an allocation, retries a failed stage, or starts a second concurrent workload.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def summarize(run, steps, graph, seed):
    config = json.loads((run/'config.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    assert config['source_runtime']['environment']['midi'] == 'TwinkleTwinkleRousseau'
    assert config['source_runtime']['model']['wrist_pitch_range_scale'] == .8
    assert config['sac']['gamma'] == .75
    assert config['acceleration']['eval_metrics_graph'] == graph
    assert config['cli']['seed'] == seed
    assert config['num_envs'] == 512
    assert config['source_runtime']['environment']['control_hz'] == 40
    assert config['source_runtime']['environment']['physics_hz'] == 400
    assert config['action_size'] == 31 and config['observation_size'] == 2297
    assert config['requested_vector_steps'] == steps
    assert config['cli']['save_interval'] == 0
    if config['cli'].get('save_forward') and steps >= config['cli']['eval_interval']:
        assert (run/'forward.pt').is_file()
    assert not list(run.glob('step-*.pt'))
    for row in config['action_contract']['channels']:
        if row['name'].endswith('_Palm_Base_Pitch'):
            assert abs(row['physical_min']+.16) < 1e-10 and abs(row['physical_max']-.16) < 1e-10
    fraction = config['scheduling']['update_ratio_fraction']
    expected = max(0, summary['transitions']-config['learning_start_transition_1_based']+1)*fraction[0]//fraction[1]
    assert summary['updates_this_run'] == expected
    assert summary['replay_insertions'] == summary['transitions']
    assert summary['transitions'] == summary['vector_steps']*512
    assert summary['vector_steps'] == steps or summary['stop_reason']=='quality_target_reached'
    pipeline=summary['collector_pipeline_state']
    assert pipeline['pending_vector_step'] is None
    assert pipeline['learner_updated_through_vector_step']==summary['vector_steps']
    assert pipeline['max_actor_lag_at_collection_end_updates'] <= 128
    assert (run/'final.pt').is_file()
    evaluation=summary['final_evaluation']
    assert evaluation['note_timing_reference_count']==102
    for key in ('collision_diagnostics_valid','key_contact_diagnostics_valid','substep_key_diagnostics_valid'):
        assert evaluation[key],key
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    # A shared post-prefill/capture window, including its periodic evaluations.
    records=[r for r in rows if r['vector_step']>=400 and 'loop_transitions_per_second' in r]
    first,last=records[0],records[-1]
    seconds=last['transitions']/last['loop_transitions_per_second']-first['transitions']/first['loop_transitions_per_second']
    evals=[r for r in rows if 'eval' in r]
    return dict(run=str(run),steps=summary['vector_steps'],transitions=summary['transitions'],updates=expected,
        stop_reason=summary['stop_reason'],seconds=summary['elapsed_s'],graph_setup_s=summary['learner_graph_setup_s'],
        evaluation_seconds=summary['evaluation_seconds_total'],evaluation_count=len(evals),
        # Exclude the first call, but graph capture for each real evaluation is included.
        warm_evaluation_mean_s=statistics.mean(r['evaluation_seconds'] for r in evals[1:]),
        steady=dict(first_step=first['vector_step'],last_step=last['vector_step'],seconds=seconds,
            samples_per_second=(last['transitions']-first['transitions'])/seconds),
        final_exact_f1=evaluation['exact_original_music_micro_f1'],
        last5_exact_f1=statistics.mean(r['eval']['exact_original_music_micro_f1'] for r in evals[-5:]),
        key_underside_substeps_max=evaluation['key_underside_substeps_max'],
        hand_collision_frames=evaluation['hand_collision_frames'],full_diagnostics=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allocation',required=True,type=int)
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--phase',choices=['short','full'],required=True)
    p.add_argument('--vector-steps',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args();a.archive=a.archive.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.detach:
        argv=[x for x in sys.argv[1:] if x!='--detach']
        with (a.output/'coordinator.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*argv],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(pid=process.pid,output=str(a.output)),ensure_ascii=False));return
    lock=(a.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=a.output/'runtime.json'
    if state_path.exists():raise RuntimeError('Existing queue state: use a fresh output directory; no implicit retry')
    archive_sha=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    steps=a.vector_steps or (800 if a.phase=='short' else 19532)
    stages=([('A1',False),('B1',True),('B2',True),('A2',False)] if a.phase=='short'
            else [('A_baseline',False),('B_eval_graph',True)])
    state=dict(status='running',pid=os.getpid(),started_at=now(),updated_at=now(),allocation=a.allocation,
        archive=str(a.archive),archive_sha256=archive_sha,phase=a.phase,seed=a.seed,
        requested_vector_steps=steps,stages=[dict(label=l,graph=g,status='pending') for l,g in stages])
    write(state_path,state)
    try:
        for st in state['stages']:
            assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==archive_sha
            allocation=subprocess.check_output(['squeue','-h','-j',str(a.allocation),'-o','%u %T %j'],text=True).strip()
            assert allocation==os.environ['USER']+' RUNNING TsenseI-reserve2',allocation
            active=subprocess.check_output(['squeue','-h','--steps','-j',str(a.allocation),'-o','%i'],text=True).split()
            assert all(x.endswith(('.batch','.extern')) for x in active),active
            prefix=a.output/st['label']
            command=['bash',str(ROOT/'scripts/hpc/submit.sh'),'--allocation',str(a.allocation),
                '--source-archive',str(a.archive),'train','--vector-steps',str(steps),'--seed',str(a.seed),
                '--eval-metrics-graph' if st['graph'] else '--no-eval-metrics-graph',
                '--output-dir',str(prefix)]
            if a.phase=='short':command+=['--eval-interval','200','--save-interval','0','--no-save-forward',
                                        '--target-consecutive-evals',str(steps+2)]
            st.update(status='running',started_at=now(),command=command);state['active_stage']=st['label'];write(state_path,state)
            with (a.output/(st['label']+'.submit.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                st['submit_pid']=child.pid
                while child.poll() is None:
                    state['updated_at']=now()
                    st['slurm_steps']=subprocess.check_output(['squeue','-h','--steps','-j',str(a.allocation),'-o','%i'],text=True).split()
                    runs=sorted(a.output.glob(st['label']+'-*'))
                    st['output_dirs']=[str(x) for x in runs if (x/'config.json').exists()]
                    write(state_path,state);time.sleep(5)
            st['exit_code']=child.returncode
            if child.returncode:raise RuntimeError(f'{st["label"]} failed ({child.returncode}); no retry, allocation retained')
            runs=[x for x in a.output.glob(st['label']+'-*') if (x/'summary.json').exists()]
            assert len(runs)==1,runs
            st['result']=summarize(runs[0],steps,st['graph'],a.seed)
            st.update(status='completed',finished_at=now());write(state_path,state)
            print(st['label'],json.dumps(st['result']),flush=True)
        state.update(status='completed',active_stage=None,updated_at=now())
        grouped={flag:[st['result'] for st in state['stages'] if st['graph']==flag] for flag in (False,True)}
        baseline=statistics.median(r['steady']['samples_per_second'] for r in grouped[False])
        candidate=statistics.median(r['steady']['samples_per_second'] for r in grouped[True])
        state['comparison']=dict(baseline_steady_samples_per_second=baseline,candidate_steady_samples_per_second=candidate,
            throughput_ratio=candidate/baseline,baseline_warm_eval_s=statistics.median(r['warm_evaluation_mean_s'] for r in grouped[False]),
            candidate_warm_eval_s=statistics.median(r['warm_evaluation_mean_s'] for r in grouped[True]))
        write(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),updated_at=now());write(state_path,state);raise


if __name__=='__main__':main()
