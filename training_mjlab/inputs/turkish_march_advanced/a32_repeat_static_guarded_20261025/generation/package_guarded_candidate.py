"""Package frozen DP v7; no model loading, optimization, or score edits."""
from pathlib import Path
import collections
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'logs/turkish_fullscore_20261025/repetition'
BASE = ROOT / 'training_mjlab/inputs/turkish_march_advanced'
OUT = BASE / 'a32_repeat_static_guarded_20261025'
OLD = BASE / 'a32_repeat_refined_20261025'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')

def manifest(folder):
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (folder / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(folder)}\n' for p in files))
    return len(files)

assert sha(OUT/'turkish-march-advanced_fingered.proto') == '66563e8b5e338c1ef3e426233647da2a008ac931939775b1ee87c36857339af0'
assert sha(OUT/'fingering.csv') == '3982adf446ffcf42334191bdce6731a0203d02d19e404121de880c4bc4eff801'
shutil.copytree(OLD/'generation', OUT/'generation', dirs_exist_ok=True)
shutil.copy2(OLD/'original.mid', OUT/'original.mid')
shutil.copy2(WORK/'static_guarded_audit/audit.json', OUT/'repetition_audit.json')
for n in ('reference_repeat_static_guarded_20261025.npz', 'reference_repeat_static_guarded_20261025.json'):
    shutil.copy2(WORK/n, OUT/n)
for n in ('repeat_constraints_v6.json', 'repeat_constraints_v7.json', 'run_repeat_dp_v6.sh',
          'run_repeat_dp_v7.sh', 'run_v6_contact_bank.sh', 'run_v7_contact_bank.sh',
          'protect_static_witnesses.py', 'paired_contact_probe.py', 'run_paired_contact_probe.sh',
          'paired_static_change_review.json', 'package_guarded_candidate.py'):
    shutil.copy2(WORK/n, OUT/'generation'/n)

baseline_bank = ROOT/'logs/turkish_fullscore_20261025/contact_bank/fullscore_static_305_v2'
final_bank = WORK/'v7_static_bank'
def occurrences(folder):
    return {x['time_seconds']:x for x in map(json.loads,(folder/'occurrences.jsonl').read_text().splitlines())}
old, new = occurrences(baseline_bank), occurrences(final_bank)
assert old.keys() == new.keys() and len(new) == 1357
counts = collections.Counter()
changed=[]
for t, a in old.items():
    b=new[t]
    kind=('same' if a['signature']==b['signature'] else 'changed') + '_combination'
    key=f"{kind}_{'pass' if a['static_passed'] else 'fail'}_to_{'pass' if b['static_passed'] else 'fail'}"
    counts[key]+=1
    if a['static_passed'] != b['static_passed']:
        changed.append({'time_seconds':t,'classification':key,'old_signature':a['signature'],'new_signature':b['signature']})
assert not any('pass_to_fail' in k for k in counts)
comparison={'schema':'paired_frozen_static_bank_comparison_v1','baseline_report_sha256':sha(baseline_bank/'report.json'),
 'final_report_sha256':sha(final_bank/'report.json'),'baseline_occurrences_sha256':sha(baseline_bank/'occurrences.jsonl'),
 'final_occurrences_sha256':sha(final_bank/'occurrences.jsonl'),'classification':dict(counts), 'changed_pass_status':changed,
 'baseline_passes_preserved':989,'final_passes':1010,'new_static_passes':21,
 'same_combination_search_recoveries':14,'changed_combination_gains':7,
 'inference':'The net +21 includes 14 unchanged-combination search/cache recoveries; it is not an isolated fingering treatment effect.',
 'physics_steps':0,'dynamic_certification':False}
dump(WORK/'guarded_static_comparison.json',comparison)
dump(OUT/'static_bank_comparison.json',comparison)
shutil.copy2(final_bank/'report.json',OUT/'static_bank_report.json')
shutil.copy2(WORK/'paired_contact/report.json',OUT/'generation/paired_contact_report.json')
provenance=json.loads((OLD/'generation/provenance.json').read_text())
provenance['schema']='a32_guarded_repetition_candidate_provenance_v2'
provenance['geometry_original_model_sha256']='8e91c32ee7174c5b3b0bfdad9cdf49a6bfaa781b4aaae42ace0957343b77afef'
provenance['geometry_role']='Sampled proxy/initializer only; not sampled from the 920475 model. Binary differences are not fully characterized. Every reported native witness is rechecked against the 920475 model.'
for v,pid in ((6,2588838),(7,2594908)):
    provenance['dp_runs'].append({'version':v,'wrapper_pid':pid,'command':f'bash task_scripts/run_repeat_dp_v{v}.sh',
      'script_sha256':sha(WORK/f'run_repeat_dp_v{v}.sh'),'exit_status':int((WORK/f'fingering_repeat_refined_v{v}.exit').read_text()),
      'log':f'task_outputs/fingering_repeat_refined_v{v}.log'})
provenance['selected_version']=7
provenance['validation_command']=provenance['validation_command'].replace('_v5','_v7')
provenance['validation_pid']=2596055
provenance['reference_command']=provenance['reference_command'].replace('a32_repeat_refined_20261025','a32_repeat_static_guarded_20261025').replace('reference_repeat_refined_20261025','reference_repeat_static_guarded_20261025')
provenance['static_bank']={'wrapper_pid':2596053,'pid':2596059,'exit_status':0,'command':'bash task_scripts/run_v7_contact_bank.sh',
 'script_sha256':sha(WORK/'run_v7_contact_bank.sh'),'report_sha256':sha(final_bank/'report.json'),
 'remote_path':'task_outputs/fullscore_static_repeat_v7','local_path':str(final_bank)}
provenance['paired_probe']={'wrapper_pid':2575845,'pid':2575847,'exit_status':0,'command':'bash task_scripts/run_paired_contact_probe.sh',
 'script_sha256':sha(WORK/'paired_contact_probe.py'),'report_sha256':sha(WORK/'paired_contact/report.json')}
provenance['protected_static_constraint_command']='python3 logs/turkish_fullscore_20261025/repetition/protect_static_witnesses.py --baseline training_mjlab/inputs/turkish_march_advanced/a32_contact_refined_20261024 --baseline-bank logs/turkish_fullscore_20261025/contact_bank/fullscore_static_305_v2 --paired-report logs/turkish_fullscore_20261025/repetition/paired_contact/report.json --constraints logs/turkish_fullscore_20261025/repetition/repeat_constraints_v5.json --output logs/turkish_fullscore_20261025/repetition/repeat_constraints_v7.json --additional-baseline-onset 11.75'
provenance['source_note']='Research remains pinned to exper038. Other work updated the default archive; no default source or entry is changed here.'
dump(OUT/'generation/provenance.json',provenance)
report=json.loads((OUT/'report.json').read_text())
report['repetition_refinement']={'version':7,'baseline_thumb_count':305,'thumb_count':312,'baseline_short_reuses':165,'short_reuses':162,
 'changed_notes':69,'new_short_reuses':0,'removed_short_reuses':3,'first20_fingering_identical':True,
 'baseline_inconsistent_groups':69,'inconsistent_groups':50,'baseline_single_hand_inconsistent_groups':37,'single_hand_inconsistent_groups':30,
 'strict_static_passed_onsets':1010,'requested_onsets':1357,'all_baseline_989_static_passes_preserved':True,
 'dynamic_acceptance':None,'static_net_gain_cannot_be_attributed_to_fingering_alone':True}
dump(OUT/'report.json',report)
print(json.dumps({'candidate_files':manifest(OUT),'complete_bank_files':manifest(final_bank),'comparison':dict(counts)},indent=2))
