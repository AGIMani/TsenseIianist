#!/usr/bin/env python3
"""Remote-only paired scratch-FK check; same two seed sources and CPU caps."""
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,sys,time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
MODEL=None;CONTRACT=None

def init(settings,contract):
    global MODEL,CONTRACT
    from scripts.hpc.piano_fullscore_contact_bank import StaticModel
    MODEL=StaticModel(settings);MODEL.extra=[];CONTRACT=contract

def work(pair):
    from scripts.hpc.piano_fullscore_contact_bank import static_signature
    m=MODEL;results={}
    for label in ('baseline','candidate'):
        original=pair[label];node={k:original[k] for k in ('time_seconds','notes','held_note_indices','onset_note_indices')}
        fingers={n['finger'] for n in node['notes']};sides={f//5 for f in fingers};trials=[]
        for source_label in ('baseline','candidate'):
            source=m.np.asarray(pair[source_label]['action']);seed=m.neutral.copy()
            for side in sides:seed[side*25+20:side*25+25]=source[side*25+20:side*25+25]
            for finger in fingers:
                off=finger//5*25+finger%5*4;seed[off:off+4]=source[off:off+4]
            applied,initial,_=m.inspect(seed,node,True)
            if initial['static_passed']:
                trial=dict(action=applied.tolist(),check=initial,solver_cpu_seconds=0.,solver_statuses=['initial_witness'],objective=0.)
            else:
                m.seed=lambda n,seed=seed:seed.copy()
                trial=m.solve(node,static_signature(node,CONTRACT),CONTRACT)
            trials.append(dict(seed_source=source_label,initial_check=initial,**trial))
        chosen=min(trials,key=lambda r:(not r['check']['static_passed'],r['objective']))
        results[label]=dict(original_static_passed=original['static_passed'],original_signature=original['signature'],trials=trials,
                            static_passed=chosen['check']['static_passed'],best_action=chosen['action'],best_check=chosen['check'])
    return dict(pair_index=pair['pair_index'],occurrence_seconds=pair['occurrence_seconds'],results=results,worker_pid=os.getpid())

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-bank',type=Path,required=True);p.add_argument('--candidate-bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SSH_CONNECTION'):p.error('Remote CPU only; never run locally')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',MUJOCO_GL='disable')
    reports=[json.loads((d/'report.json').read_text()) for d in (a.baseline_bank,a.candidate_bank)]
    assert reports[0]['contract_sha256']==reports[1]['contract_sha256']
    rows=[[json.loads(x) for x in (d/'occurrences.jsonl').read_text().splitlines()] for d in (a.baseline_bank,a.candidate_bank)]
    assert len(rows[0])==len(rows[1])
    unique={}
    for old,new in zip(*rows):
        assert old['time_seconds']==new['time_seconds']
        if old['static_passed'] and not new['static_passed']:
            key=(old['signature'],new['signature'])
            if key not in unique:unique[key]=dict(pair_index=len(unique),baseline=old,candidate=new,occurrence_seconds=[])
            unique[key]['occurrence_seconds'].append(old['time_seconds'])
    a.output.mkdir(parents=True,exist_ok=False)
    settings=reports[1]['arguments'];contract=reports[1]['contract']
    settings=dict(recording=str(Path(settings['recording']).resolve()),geometry=str(Path(settings['geometry']).resolve()),output=str(a.output.resolve()),
        model_sha256=contract['model_sha256'],archive_sha256=contract['archive_sha256'],node_cpu_budget=2.5,max_nfev=60,penetration=contract['target_penetration_m'],seed_banks=[])
    metadata=dict(command=sys.argv,pid=os.getpid(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_bank_report_sha256=hashlib.sha256((a.baseline_bank/'report.json').read_bytes()).hexdigest(),candidate_bank_report_sha256=hashlib.sha256((a.candidate_bank/'report.json').read_bytes()).hexdigest(),
        contract=contract,paired_combinations=len(unique),lost_occurrences=sum(len(p['occurrence_seconds']) for p in unique.values()),
        budget='Each assignment: same baseline/candidate action seed sources, each at most2.5s CPU; static-success seed can stop immediately. Two workers, no physics steps.',
        idle_contract='Each target reinitializes inactive fingers to the identical neutral/pinky parking policy before applying the common seed sources.',physics_steps=0,dynamic_acceptance=None)
    (a.output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(settings,contract)) as pool:
        futures=[pool.submit(work,pair) for pair in unique.values()]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result);(a.output/f'pair_{result["pair_index"]:03}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(completed=len(results),total=len(unique),pair_index=result['pair_index'],baseline=result['results']['baseline']['static_passed'],candidate=result['results']['candidate']['static_passed'])),flush=True)
    results.sort(key=lambda r:r['pair_index']);report={**metadata,'results':results,'exit_status':0,
        'recovered_candidate_combinations':sum(r['results']['candidate']['static_passed'] for r in results),
        'recovered_candidate_occurrences':sum(len(r['occurrence_seconds']) for r in results if r['results']['candidate']['static_passed']),
        'limitations':['Remaining failed searches do not prove kinematic impossibility. A static witness is not load-bearing key activation or a dynamic transition.',
            'The frozen final proto was not changed. Any rollback proposal requires full-score context and dynamic comparison.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('paired_combinations','lost_occurrences','recovered_candidate_combinations','recovered_candidate_occurrences')}),flush=True)
if __name__=='__main__':main()
