#!/usr/bin/env python3
"""Audit exact MIDI-tick repeats and propose context-checked fingering constraints.

Pure functions need only the standard library. MIDI parsing is imported by the
CLI, which reads events and never loads a robot model or integrates physics.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict,deque
from dataclasses import dataclass
import hashlib,json,math,os
from pathlib import Path
import csv,time


@dataclass(frozen=True)
class Note:
    index:int;hand:int;pitch:int;start:int;end:int;start_seconds:float;end_seconds:float;finger:int;velocity:int


@dataclass(frozen=True)
class Window:
    start:int;end:int;hand:int;width:int;notes:tuple[Note,...]

    def signature(self,mode='exact'):
        base=min((n.pitch for n in self.notes),default=0) if mode=='transposed' else 0
        events=tuple((n.hand,n.start-self.start,n.pitch-base) if mode=='release_variant'
                     else (n.hand,n.start-self.start,n.end-n.start,n.pitch-base) for n in self.notes)
        return self.hand,self.width,self.end-self.start,events


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def short_reuses(notes,assignments,recovery=.05):
    result=[]
    for finger in range(10):
        group=sorted((n for n in notes if assignments[n.index]==finger),key=lambda n:(n.start_seconds,n.index))
        for a,b in zip(group,group[1:]):
            gap=b.start_seconds-a.end_seconds
            if gap<recovery:
                result.append(dict(finger=finger,previous_note=a.index,next_note=b.index,
                    previous_pitch=a.pitch,next_pitch=b.pitch,onset_seconds=b.start_seconds,
                    gap_seconds=gap,same_pitch=a.pitch==b.pitch,
                    normalized_cost=max(0.,1.-gap/recovery)**2))
    return sorted(result,key=lambda r:(r['onset_seconds'],r['finger']))


def assignment_issue(notes,assignments,fixed):
    if any(assignments[i]!=finger for i,finger in fixed.items()):return 'fixed constraint mismatch'
    for finger in range(10):
        group=sorted((n for n in notes if assignments[n.index]==finger),key=lambda n:(n.start,n.index))
        if any(a.end>b.start for a,b in zip(group,group[1:])):return 'same-finger interval overlap'
    for hand in (0,1):
        events=defaultdict(lambda:[[],[]])
        for n in notes:
            if n.hand==hand:events[n.start][0].append(n);events[n.end][1].append(n)
        active={}
        for at,(starts,ends) in sorted(events.items()):
            for n in ends:active.pop(n.index,None)
            for n in starts:active[n.index]=n
            fingers=[assignments[n.index] for n in sorted(active.values(),key=lambda n:(n.pitch,n.index))]
            thumb=5*hand
            if any(f//5!=hand for f in fingers):return 'wrong hand'
            if thumb in fingers and fingers.index(thumb) not in (0,len(fingers)-1):return 'thumb crosses natural order'
            other=[f for f in fingers if f!=thumb]
            if other!=sorted(other,reverse=bool(hand)):return 'nonthumb natural order mismatch'
    return None


def parse_midi(midi,csv_path):
    import mido
    parsed=mido.MidiFile(midi)
    metadata=[];tempos=[];signatures=[];raw=[]
    for track_id,track in enumerate(parsed.tracks):
        at=0;open_notes=defaultdict(deque)
        for message in track:
            at+=message.time
            if message.is_meta and message.type in ('set_tempo','time_signature','key_signature'):
                metadata.append(dict(track=track_id,tick=at,message=str(message)))
                if message.type=='set_tempo':tempos.append((at,message.tempo,track_id))
                if message.type=='time_signature':signatures.append((at,message.numerator,message.denominator,track_id))
            if message.type=='note_on' and message.velocity:
                open_notes[(message.channel,message.note)].append((at,message.velocity))
            elif message.type=='note_off' or (message.type=='note_on' and not message.velocity):
                key=(message.channel,message.note)
                if open_notes[key]:
                    start,velocity=open_notes[key].popleft();raw.append((track_id,message.note,start,at,velocity))
        if any(open_notes.values()):raise ValueError('Unterminated raw MIDI notes')
    tempo_by_tick={0:500000}
    for tick,tempo,track in tempos:
        if tick in tempo_by_tick and tempo_by_tick[tick]!=tempo and tick!=0:raise ValueError('Conflicting tempo events')
        tempo_by_tick[tick]=tempo
    ordered_tempos=sorted(tempo_by_tick.items())
    def seconds(tick):
        total=0.
        for j,(start,tempo) in enumerate(ordered_tempos):
            end=ordered_tempos[j+1][0] if j+1<len(ordered_tempos) else tick
            if tick>start:total+=(min(end,tick)-start)*tempo/1e6/parsed.ticks_per_beat
        return total
    by_signature=defaultdict(deque)
    key=lambda hand,pitch,start,end,velocity:(hand,pitch,round(start,9),round(end,9),velocity)
    for hand,pitch,start,end,velocity in raw:
        by_signature[key(hand,pitch,seconds(start),seconds(end),velocity)].append((start,end))
    notes=[]
    for row in csv.DictReader(Path(csv_path).open()):
        hand=int(row['instrument']);pitch=int(row['pitch']);start=float(row['start_seconds']);end=float(row['end_seconds']);velocity=int(row['velocity'])
        match=by_signature[key(hand,pitch,start,end,velocity)]
        if not match:raise ValueError(f'CSV note {row["note_index"]} does not match raw MIDI')
        first,last=match.popleft()
        assert abs(seconds(first)-start)<1e-10 and abs(seconds(last)-end)<1e-10
        notes.append(Note(int(row['note_index']),hand,pitch,first,last,start,end,int(row['finger_id']),velocity))
    if any(by_signature.values()):raise ValueError('Raw MIDI notes omitted from CSV')
    signature_by_tick={}
    for at,num,den,track in signatures:
        if at in signature_by_tick and signature_by_tick[at]!=(num,den):raise ValueError('Conflicting time signatures')
        signature_by_tick[at]=(num,den)
    if 0 not in signature_by_tick:signature_by_tick[0]=(4,4)
    signature_events=sorted(signature_by_tick.items());bars=[];at=0;end=max(n.end for n in notes)
    while at<end:
        sig=max((s for s in signature_events if s[0]<=at),key=lambda s:s[0]);num,den=sig[1]
        length=parsed.ticks_per_beat*4*num/den
        if not length.is_integer():raise ValueError('Nonintegral MIDI tick measure length')
        boundary=min([x[0] for x in signature_events if at<x[0]<at+length],default=at+int(length))
        bars.append(dict(index=len(bars),start_tick=at,end_tick=boundary,start_seconds=seconds(at),end_seconds=seconds(boundary),numerator=num,denominator=den))
        at=boundary
    return notes,bars,dict(ticks_per_quarter=parsed.ticks_per_beat,raw_note_count=len(raw),metadata=metadata,
        tempo_events=[dict(tick=t,tempo_microseconds_per_quarter=v,quarter_notes_per_minute=60e6/v) for t,v in ordered_tempos],
        exact_tick_to_original_second_alignment=True),seconds


def windows_for(notes,bars):
    windows=[]
    for width in (1,2,4):
        for j in range(len(bars)-width+1):
            start,end=bars[j]['start_tick'],bars[j+width-1]['end_tick']
            for hand in (0,1,2):
                selected=tuple(sorted((n for n in notes if (hand==2 or n.hand==hand) and n.start<end and n.end>start),key=lambda n:(n.start,n.hand,n.pitch,n.end,n.index)))
                if selected:windows.append(Window(start,end,hand,width,selected))
    return windows


def group_windows(windows,mode):
    groups=defaultdict(list)
    for window in windows:groups[window.signature(mode)].append(window)
    return {key:values for key,values in groups.items() if len(values)>1}


def pattern_record(key,values,assignments,gaps,seconds):
    variants=defaultdict(list)
    for w in values:variants[tuple(assignments[n.index] for n in w.notes)].append(w)
    onset_ids={n.index for w in values for n in w.notes if w.start<=n.start<w.end}
    return dict(id=hashlib.sha256(repr(key).encode()).hexdigest()[:12],hand=('right','left','both')[key[0]],bars=key[1],
        occurrence_count=len(values),notes_per_window=len(values[0].notes),covered_onset_note_count=len(onset_ids),
        onset_ticks=[w.start for w in values],onset_seconds=[seconds(w.start) for w in values],
        fingering_variant_count=len(variants),fingering_consistent=len(variants)==1,
        fingering_variants=[dict(fingers=list(fingers),occurrences_seconds=[seconds(w.start) for w in ws]) for fingers,ws in variants.items()],
        short_reuses_with_next_onset_in_group=sum(g['next_note'] in onset_ids for g in gaps),
        note_indices_by_occurrence=[[n.index for n in w.notes] for w in values])


def propose(notes,groups,baseline,fixed,seconds,max_groups):
    current=dict(baseline);constraints=dict(fixed);selected=[];rejected=Counter()
    candidates=[(k,v) for k,v in groups.items() if k[0]!=2 and len({tuple(current[n.index] for n in w.notes) for w in v})>1]
    gaps=short_reuses(notes,current)
    candidates.sort(key=lambda kv:(-sum(any(n.index==g['next_note'] and w.start<=n.start<w.end for w in kv[1] for n in w.notes) for g in gaps),-kv[0][1],-len(kv[1])))
    candidates=candidates[:80]
    def inconsistent(mapping):return sum(len({tuple(mapping[n.index] for n in w.notes) for w in values})>1 for key,values in groups.items() if key[0]!=2)
    def score(mapping):
        rows=short_reuses(notes,mapping)
        return len(rows),round(sum(r['normalized_cost'] for r in rows),10),sum(f%5==0 for f in mapping.values()),inconsistent(mapping)
    current_score=score(current)
    for iteration in range(max_groups):
        best=None
        for key,values in candidates:
            variants={tuple(current[n.index] for n in w.notes):w for w in values}
            if len(variants)<2:continue
            for fingers,origin in variants.items():
                patch={};conflict=False
                for w in values:
                    for n,f in zip(w.notes,fingers):
                        if (n.index in patch and patch[n.index]!=f) or (n.index in constraints and constraints[n.index]!=f):conflict=True;break
                        patch[n.index]=f
                    if conflict:break
                if conflict:rejected['overlapping_repeat_or_fixed_constraint']+=1;continue
                changed={**current,**patch};issue=assignment_issue(notes,changed,constraints)
                if issue:rejected[issue]+=1;continue
                new_score=score(changed)
                if new_score>=current_score:rejected['no_lexicographic_reuse_thumb_consistency_improvement']+=1;continue
                if best is None or new_score<best[0]:best=(new_score,key,values,fingers,origin,patch,changed)
        if best is None:break
        new_score,key,values,fingers,origin,patch,changed=best
        selected.append(dict(pattern_id=hashlib.sha256(repr(key).encode()).hexdigest()[:12],hand=('right','left')[key[0]],bars=key[1],
            template_time_seconds=seconds(origin.start),occurrence_seconds=[seconds(w.start) for w in values],
            fingers=list(fingers),fixed_note_indices=sorted(patch),changed_note_indices=[i for i in patch if current[i]!=patch[i]],
            before_score=current_score,after_score=new_score))
        current=changed;constraints.update(patch);current_score=new_score
    return dict(assignments={str(i):f for i,f in sorted(constraints.items())},selected_patterns=selected,
        pre_dp_candidate_assignments={str(i):f for i,f in sorted(current.items())},pre_dp_score=current_score,
        proposal_rejection_counts=dict(rejected),scope='Exact-pitch, exact-onset and exact-duration repeats only; whole-score interval/order/fixed constraints checked. DP must still replan unconstrained neighbors.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--midi',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--propose',action='store_true');p.add_argument('--max-groups',type=int,default=8)
    args=p.parse_args();start=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    notes,bars,metadata,seconds=parse_midi(args.midi,args.plan/'fingering.csv');assignments={n.index:n.finger for n in notes}
    fixed_source=json.loads((args.plan/'fixed_fingering.json').read_text());fixed={int(i):f for i,f in fixed_source['assignments'].items()}
    issue=assignment_issue(notes,assignments,fixed)
    if issue:raise ValueError(issue)
    windows=windows_for(notes,bars);exact=group_windows(windows,'exact');transposed=group_windows(windows,'transposed');near=group_windows(windows,'release_variant')
    gaps=short_reuses(notes,assignments)
    records=[pattern_record(k,v,assignments,gaps,seconds) for k,v in exact.items()]
    records.sort(key=lambda r:(-r['short_reuses_with_next_onset_in_group'],-r['covered_onset_note_count'],-r['bars']))
    transposed_records=[]
    for key,values in transposed.items():
        bases=[min(n.pitch for n in w.notes) for w in values]
        if len(set(bases))<2:continue
        transposed_records.append(dict(hand=('right','left','both')[key[0]],bars=key[1],occurrences_seconds=[seconds(w.start) for w in values],
            absolute_bases=bases,relative_semitone_shifts=[b-bases[0] for b in bases],only_octave_transposition=all((b-bases[0])%12==0 for b in bases),finger_copy_allowed=False))
    near_records=[dict(hand=('right','left','both')[key[0]],bars=key[1],occurrences_seconds=[seconds(w.start) for w in values],
        exact_release_variants=len({w.signature('exact') for w in values}),finger_copy_allowed=False)
        for key,values in near.items() if len({w.signature('exact') for w in values})>1]
    covered={n.index for values in exact.values() for w in values for n in w.notes if w.start<=n.start<w.end}
    report=dict(schema='exact_tick_repetition_audit_v1',pid=os.getpid(),script_sha256=sha(__file__),source_midi_sha256=sha(args.midi),
        source_csv_sha256=sha(args.plan/'fingering.csv'),source_proto_sha256=sha(args.plan/'turkish-march-advanced_fingered.proto'),
        note_count=len(notes),duration_seconds=max(n.end_seconds for n in notes),midi=metadata,bars=bars,
        strict_repeated_groups=len(records),strict_repeated_groups_with_inconsistent_fingering=sum(not r['fingering_consistent'] for r in records),
        notes_covered_by_strict_repeats=len(covered),transposed_group_count=len(transposed_records),release_variant_group_count=len(near_records),
        counts_by_hand={('right','left')[h]:sum(n.hand==h for n in notes) for h in (0,1)},thumb_count=sum(n.finger%5==0 for n in notes),
        short_reuse_count=len(gaps),short_reuse_same_pitch_count=sum(g['same_pitch'] for g in gaps),short_reuses=gaps,
        strict_groups=records,transposed_groups=transposed_records,near_same_onset_pitch_release_variants=near_records,
        limitations=['Repeat-window coverage overlaps and group counts must not be summed as independent measures.',
            'Exact patterns retain original note releases and carry-in/carry-out notes; near repeats and transpositions are explicitly excluded from automatic copying.',
            'Repeated fingering is not proof of dynamic feasibility, contact forces, or cross-hand collision clearance. No model or physics is loaded.'])
    if args.propose:
        proposal=propose(notes,exact,assignments,fixed,seconds,args.max_groups)
        proposal.update(schema='a32_context_checked_exact_repetition_constraints_v1',source_midi_sha256=sha(args.midi),
            source_baseline_csv_sha256=sha(args.plan/'fingering.csv'),preserved_static_contact_constraints_sha256=sha(args.plan/'fixed_fingering.json'))
        (args.output/'repeat_constraints.json').write_text(json.dumps(proposal,indent=2)+'\n')
        report['proposal_summary']={k:v for k,v in proposal.items() if k not in ('assignments','pre_dp_candidate_assignments')}
    report['wall_seconds']=time.monotonic()-start
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('note_count','duration_seconds','strict_repeated_groups','strict_repeated_groups_with_inconsistent_fingering','notes_covered_by_strict_repeats','transposed_group_count','release_variant_group_count','short_reuse_count','thumb_count','wall_seconds')}),flush=True)
    if args.propose:print(json.dumps(report['proposal_summary']),flush=True)

if __name__=='__main__':main()
