#!/usr/bin/env python3
"""Enumerate identical, statically checked wide chords without changing MIDI events."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--midi',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    notes=[dict(index=int(r['note_index']),pitch=int(r['pitch']),finger=int(r['finger_id']),
        start=float(r['start_seconds']),end=float(r['end_seconds']),hand=r['hand'])
        for r in csv.DictReader(args.baseline.open())]
    assert len(notes)==2544 and {n['index'] for n in notes}==set(range(2544))
    assignments={};occurrences=[];evidence=[]
    for hand,pitches,fingers in [('right',[73,76,81,85],[0,1,3,4]),('left',[45,49,52,57],[9,8,6,5])]:
        path=args.evidence_root/(hand+'_one_thumb')/'report.json'
        report=json.loads(path.read_text());pose=report['poses'][0]
        subset=[(n,ok) for n,ok in zip(pose['notes'],pose['native_assigned_terminal_contacts'])
                if (n['finger']>=5)==(hand=='left')]
        assert [n['pitch'] for n,ok in subset]==pitches
        assert [n['finger'] for n,ok in subset]==fingers and all(ok for n,ok in subset)
        evidence.append(dict(hand=hand,pitches=pitches,fingers=fingers,source=str(path.resolve()),sha256=sha(path),
            verified_original_time=pose['time_seconds'],scope='One coupled static depressed-key posture with finite top-contact checks; not dynamics.'))
        group=[n for n in notes if n['hand']==hand];seen=set()
        for at in sorted({n['start'] for n in group}|{n['end'] for n in group}):
            active=sorted([n for n in group if n['start']<=at<n['end']],key=lambda n:n['pitch'])
            if [n['pitch'] for n in active]!=pitches:continue
            ids=tuple(n['index'] for n in active)
            if ids in seen:continue
            seen.add(ids)
            for n,finger in zip(active,fingers):
                if n['index'] in assignments:assert assignments[n['index']]==finger
                assignments[n['index']]=finger
            occurrences.append(dict(hand=hand,time_seconds=at,note_indices=ids,pitches=pitches,
                old_fingers=[n['finger'] for n in active],fixed_fingers=fingers,
                original_intervals=[[n['start'],n['end']] for n in active]))
    assert len(occurrences)==7 and len(assignments)==28
    result=dict(schema='a32_static_contact_fixed_chords_v1',source_midi_sha256=sha(args.midi),
        baseline_csv_sha256=sha(args.baseline),generator_sha256=sha(__file__),
        assignments={str(i):f for i,f in sorted(assignments.items())},occurrences=occurrences,evidence=evidence,
        limits=['Only exact pitches and identical verified finger combinations are constrained; no transposition.',
                'Full dynamic programming must replan neighboring notes and preserve held assignments.',
                'The 50ms recovery term remains soft. Static contact evidence does not certify transitions or actual pressing.'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'occurrences':len(occurrences),'fixed_notes':len(assignments),'sha256':sha(args.output)}))

if __name__=='__main__':main()
