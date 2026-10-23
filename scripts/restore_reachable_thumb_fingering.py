#!/usr/bin/env python3
"""Restore a limited set of original thumb assignments for physical validation.

Static feasibility is a candidate filter, never a claim of safe physical play.
All MIDI events and other protobuf fields remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from note_seq.protobuf import music_pb2
import pretty_midi


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--feasibility',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--left-budget',type=int,default=12)
    parser.add_argument('--right-budget',type=int,default=5)
    parser.add_argument('--physical-summary',type=Path,
                        help='Optional prior replay per-note assigned-contact plus activation evidence')
    parser.add_argument('--minimum-contact-activation-recall',type=float,default=.5)
    parser.add_argument('--restore-chord-companions',action='store_true',
                        help='Restore original non-thumb assignments in each selected thumb chord')
    a=parser.parse_args()
    if not 0<a.minimum_contact_activation_recall<=1:
        parser.error('Physical recall threshold must be in (0,1]')
    original=music_pb2.NoteSequence.FromString(a.original.read_bytes())
    seq=music_pb2.NoteSequence.FromString(a.base.read_bytes())
    check=music_pb2.NoteSequence.FromString(seq.SerializeToString())
    if len(check.notes)!=len(original.notes):raise ValueError('Note count mismatch')
    for n,o in zip(check.notes,original.notes):n.part=o.part
    if check.SerializeToString()!=original.SerializeToString():raise ValueError('Base changes original music')
    data=json.loads(a.feasibility.read_text())
    physical=json.loads(a.physical_summary.read_text()) if a.physical_summary else None
    eligible=[]
    for i,n in enumerate(original.notes):
        if n.part not in (0,5):continue
        matches=[r for r in data['results'] if r['finger_part']==n.part and r['pitch']==n.pitch
                 and abs(r['start_s']-n.start_time)<1e-8 and abs(r['end_s']-n.end_time)<1e-8]
        if len(matches)!=1:raise ValueError(f'Missing/duplicate feasibility for note {i}')
        candidates=[c for c in matches[0]['candidates'] if c['static_geometric_pass']]
        if not candidates:continue
        evidence=None
        if physical is not None:
            observed=[r for r in physical['thumb_notes']
                      if r['part']==n.part and r['pitch']==n.pitch
                      and abs(r['start_s']-n.start_time)<1e-8
                      and abs(r['end_s']-n.end_time)<1e-8]
            if len(observed)>1:raise ValueError('Duplicate physical note evidence')
            if not observed or observed[0]['joint_contact_activation_recall']<a.minimum_contact_activation_recall:
                continue
            evidence=observed[0]
        held=[o for o in original.notes if o.part//5==n.part//5
              and max(o.start_time,n.start_time)<min(o.end_time,n.end_time)]
        span=max(o.pitch for o in held)-min(o.pitch for o in held)
        eligible.append(dict(note_index=i,part=n.part,start_s=n.start_time,end_s=n.end_time,
                             pitch=n.pitch,chord_span_semitones=span,
                             prior_physical_evidence=evidence,
                             static_seed=min(candidates,key=lambda c:c['cost'])))
    selected=[]
    for part,budget in [(0,a.right_budget),(5,a.left_budget)]:
        if budget<1:raise ValueError('Keep a nonzero thumb budget for each hand')
        options=sorted([r for r in eligible if r['part']==part],
                       key=lambda r:(-r['chord_span_semitones'],-(r['end_s']-r['start_s']),r['start_s']))
        chosen=options[:budget]
        if not chosen:raise ValueError(f'No statically feasible thumb candidate for hand {part//5}')
        selected.extend(chosen)
    for row in selected:seq.notes[row['note_index']].part=row['part']
    companion_indices=set()
    if a.restore_chord_companions:
        for row in selected:
            for i,n in enumerate(original.notes):
                if (n.part//5==row['part']//5 and n.part not in (0,5)
                    and max(n.start_time,row['start_s'])<min(n.end_time,row['end_s'])):
                    seq.notes[i].part=n.part
                    companion_indices.add(i)
    for i,n in enumerate(seq.notes):
        for m in seq.notes[i+1:]:
            if n.part==m.part and max(n.start_time,m.start_time)<min(n.end_time,m.end_time):
                raise ValueError('Overlapping notes assigned to one finger')
    check=music_pb2.NoteSequence.FromString(seq.SerializeToString())
    for n,o in zip(check.notes,original.notes):n.part=o.part
    assert check.SerializeToString()==original.SerializeToString()
    a.output.mkdir(parents=True,exist_ok=False)
    proto=a.output/'015-5_fingered.proto';proto.write_bytes(seq.SerializeToString())
    lines=['// Selective thumb candidate; original MIDI times unchanged; physics validation required.']
    for i,n in enumerate(seq.notes):
        f=(n.part%5+1)*(1 if n.part<5 else -1)
        lines.append(f'{i}\t{n.start_time:.12g}\t{n.end_time:.12g}\t{pretty_midi.note_number_to_name(n.pitch)}\t{n.velocity}\t80\t{n.part//5}\t{f}')
    (a.output/'015-5_fingering.txt').write_text('\n'.join(lines)+'\n')
    report=dict(kind='limited_original_thumb_restoration_candidate',source_sha256=sha(__file__),
                original_score_sha256=sha(a.original),base_score_sha256=sha(a.base),
                feasibility_sha256=sha(a.feasibility),revised_score_sha256=sha(proto),notes=len(seq.notes),
                original_thumb_notes=sum(n.part in (0,5) for n in original.notes),
                thumb_assignments=len(selected),right_thumb_notes=sum(r['part']==0 for r in selected),
                left_thumb_notes=sum(r['part']==5 for r in selected),selected=selected,
                restored_chord_companions=sorted(companion_indices),
                changed_notes=sum(n.part!=o.part for n,o in zip(seq.notes,original.notes)),
                preserved='All original protobuf fields except note.part',physical_validation='pending')
    if a.physical_summary:
        report['physical_summary_sha256']=sha(a.physical_summary)
        report['minimum_contact_activation_recall']=a.minimum_contact_activation_recall
        report['physical_evidence_scope']='Prior replay only; revised score and new trajectory require fresh validation'
    (a.output/'revision_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='selected'},indent=2))


if __name__=='__main__':main()
