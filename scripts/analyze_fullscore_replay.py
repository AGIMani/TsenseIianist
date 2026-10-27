#!/usr/bin/env python3
"""Exact recorded-key analysis by MIDI bar, passage and repeated note pattern.

CLI runs remotely on CPU; pure functions support small-array local tests.
Repeated sections are compared, never substituted into the physical trace.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+'\n')


def window_metrics(trace,start,end,*,include_end=False):
    time=np.asarray(trace['time'])
    first=int(np.searchsorted(time,start-1e-10));last=int(np.searchsorted(time,end+1e-10 if include_end else end-1e-10))
    actual=np.asarray(trace['actual_keys'][first:last],bool);target=np.asarray(trace['target_keys'][first:last],bool)
    tp=int(np.count_nonzero(actual&target));fp=int(np.count_nonzero(actual&~target));fn=int(np.count_nonzero(~actual&target))
    count=last-first;contacts=np.asarray(trace['hand_contacts'][first:last])
    result=dict(start_s=float(start),end_s=float(end),terminal_sample_included=include_end,samples=count,
        tp=tp,fp=fp,fn=fn,micro_f1=2*tp/max(2*tp+fp+fn,1),
        target_key_samples=int(target.sum()),actual_key_samples=int(actual.sum()),
        target_sample_coverage=tp/max(tp+fn,1),extra_pitch_count=int((actual&~target).any(0).sum()),
        extra_pitches=(np.flatnonzero((actual&~target).any(0))+21).tolist(),
        same_hand_contact_samples=int(np.count_nonzero(contacts[:,0])),
        cross_hand_contact_samples=int(np.count_nonzero(contacts[:,1])))
    for name in ('max_hand_contact_force_n','max_hand_contact_penetration_m'):
        if name in trace:result[name]=float(np.max(trace[name][first:last],initial=0.))
    if 'hand_contact_categories' in trace:
        categories=np.asarray(trace['hand_contact_categories'][first:last])
        for i,name in enumerate(('different_finger','finger_palm','same_finger','cross_hand')):
            result[name+'_samples']=int(np.count_nonzero(categories[:,i]))
    if 'hand_key_contacts' in trace:
        result['non_target_hand_key_contact_samples']=int(np.count_nonzero((trace['hand_key_contacts'][first:last]&~target).any(1)))
    return result


def note_diagnostics(time,actual,events):
    time=np.asarray(time);actual=np.asarray(actual,bool)
    rising=np.zeros_like(actual);rising[1:]=actual[1:]&~actual[:-1]
    notes=[]
    for index,(start,end,key,finger) in enumerate(events):
        key,finger=int(key),int(finger)
        first=int(np.searchsorted(time,start-1e-10));last=int(np.searchsorted(time,end-1e-10))
        active=np.flatnonzero(actual[first:last,key]);edges=np.flatnonzero(rising[first:last,key])
        count=last-first
        notes.append(dict(event_index=index,start_s=float(start),end_s=float(end),pitch=key+21,finger=finger,
            sample_count=count,recording_covers_full_note=bool(time[0]<=start and time[-1]>=end),
            any_activation_in_note=bool(len(active)) if count else None,
            new_activation_edge_in_note=bool(len(edges)) if count else None,
            active_at_recording_start=bool(first==0 and actual[0,key]),
            already_active_before_first_note_sample=bool(actual[first-1,key]) if first>0 and first<len(time) else False,
            active_sample_fraction=float(len(active)/count) if count else None,
            first_active_sample_s=float(time[first+active[0]]) if len(active) else None,
            first_new_activation_s=float(time[first+edges[0]]) if len(edges) else None,
            onset_delay_s=float(time[first+edges[0]]-start) if len(edges) else None))
    return notes


def repeated_patterns(bars,raw_events):
    groups={}
    for bar in bars:
        start,end=bar['start_s'],bar['end_s']
        events=[[round(s-start,9),round(e-start,9),int(p),int(v),int(track)]
                for s,e,p,v,track in raw_events if e>start+1e-10 and s<end-1e-10]
        if not events:continue
        key=json.dumps([round(end-start,9),events],separators=(',',':'))
        groups.setdefault(key,[]).append(bar)
    return [dict(pattern_sha256=hashlib.sha256(key.encode()).hexdigest(),bars=[r['bar'] for r in rows],
                 occurrences=rows,physical_trajectories_reused=False,
                 identity='Equal pitches, velocities and tracks; relative raw note boundaries and bar length rounded to 1 ns only for pattern matching; includes sustained notes crossing boundaries; excludes fingering/pedal')
            for key,rows in groups.items() if len(rows)>1]


def detail_plots(output,trace,original_events,source_hash,seconds=20.):
    """Readable excerpts of the same continuous recording, retaining MIDI time."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from scripts.package_planned_replay import actual_intervals
    time=trace['time'];actual=np.asarray(trace['actual_keys'],bool);target=np.asarray(trace['target_keys'],bool)
    original=[[float(s),float(e),int(k)+21] for s,e,k,_ in original_events]
    layers=[(actual_intervals(mask,time),color,label) for mask,color,label in
            [(actual&target,'#26956b','correct'),(actual&~target,'#da4b48','extra'),(~actual&target,'#edc76a','missed')]]
    folder=output/'detail_pages';folder.mkdir();pages=[]
    def draw(axis,events,color,start,end):
        polygons=[[(max(s,start),p-.4),(min(e,end),p-.4),(min(e,end),p+.4),(max(s,start),p+.4)]
                  for s,e,p in events if e>start and s<end]
        axis.add_collection(PolyCollection(polygons,facecolors=color,edgecolors='none'))
    for index,start in enumerate(np.arange(0,float(time[-1]),seconds)):
        start=float(start);end=min(start+seconds,float(time[-1]));last=end==float(time[-1])
        metrics=window_metrics(trace,start,end,include_end=last)
        fig,axes=plt.subplots(2,1,figsize=(17,7),sharex=True,sharey=True,layout='constrained')
        draw(axes[0],original,'#2475bd',start,end)
        for events,color,_ in layers:draw(axes[1],events,color,start,end)
        visible=[p for s,e,p in original if e>start and s<end]
        for events,_,_ in layers:visible.extend(p for s,e,p in events if e>start and s<end)
        low,high=(max(21,min(visible)-2),min(108,max(visible)+2)) if visible else (21,108)
        for axis in axes:
            axis.set_xlim(start,end);axis.set_ylim(low-.5,high+.5);axis.set_ylabel('MIDI pitch');axis.grid(axis='x',alpha=.2)
        axes[0].set_title('Original MIDI: unmodified timing',loc='left')
        axes[1].set_title('Same recorded physical keys: actual angle reaches 100% travel',loc='left')
        axes[1].legend(handles=[Patch(color=color,label=label) for _,color,label in layers],loc='upper right')
        axes[1].set_xlabel('Original MIDI time (seconds); preparation excluded; consecutive excerpts of one continuous recording')
        fig.suptitle(f'Continuous recording excerpt {start:.3f}–{end:.3f}s | local micro-F1 {metrics["micro_f1"]:.6f}\n'
                     f'1000Hz actual activation | trace SHA256 {source_hash[:16]} | no checkpoint or safety claim')
        name=f'comparison_{index+1:02d}_{start:07.3f}_{end:07.3f}.png'
        fig.savefig(folder/name,dpi=160);plt.close(fig)
        pages.append(dict(file='detail_pages/'+name,**metrics))
    write_json(folder/'index.json',dict(source_trace_sha256=source_hash,pages=pages,physical_segments_stitched=False))
    return pages


def analyze(recording,original_midi,output,*,allow_partial=False):
    if not os.environ.get('SSH_CONNECTION'):
        raise RuntimeError('Run recorded full-array analysis remotely; local use is limited to small pure-function tests')
    import pretty_midi
    from scripts.package_planned_replay import validate_trace
    recording,original_midi,output=map(lambda p:Path(p).resolve(),(recording,original_midi,output))
    if output.exists():raise FileExistsError(output)
    source=json.loads((recording/'replay_report.json').read_text())
    if sha(original_midi)!=source['source']['original_midi_sha256']:raise ValueError('Original MIDI SHA differs')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');midi=pretty_midi.PrettyMIDI(str(original_midi))
    parser_warnings=[str(w.message) for w in captured]
    raw=sorted([n.start,n.end,n.pitch,n.velocity,i] for i,inst in enumerate(midi.instruments) for n in inst.notes)
    score_end=float(max(row[1] for row in raw))
    fields=('time','qpos','qvel','ctrl','key_qpos','key_limits','key_activation_fraction','key_qpos_addresses',
            'actual_keys','target_keys','assignment','original_events','physics_dt','hand_contacts',
            'hand_contact_categories','hand_key_contacts','max_hand_contact_force_n','max_hand_contact_penetration_m')
    with np.load(recording/'physical_trace.npz',allow_pickle=False) as stored:
        trace={name:stored[name] for name in fields if name in stored}
    checked=validate_trace(trace);time=trace['time'];end=float(time[-1])
    if end<score_end and not allow_partial:raise ValueError('Incomplete continuous score recording; use --allow-partial only for diagnostic tool checks')
    expected=np.asarray(sorted((s,e,p-21) for s,e,p,*_ in raw));got=np.asarray(sorted(trace['original_events'][:,:3].tolist()))
    if expected.shape!=got.shape or not np.allclose(expected,got,rtol=0,atol=1e-9):raise ValueError('Trace note events differ from raw MIDI')
    whole=window_metrics(trace,0,end,include_end=True)
    for k in ('tp','fp','fn'):
        if whole[k]!=source[k]:raise ValueError('Recorded global metric mismatch: '+k)
    if not np.isclose(whole['micro_f1'],source['original_midi_micro_f1'],rtol=0,atol=1e-12):raise ValueError('Global F1 mismatch')
    # Free bulky fields after complete array validation; metrics do not infer keys from commands.
    for key in ('qpos','qvel','ctrl','assignment','key_qpos'):trace.pop(key,None)
    notes=note_diagnostics(time,checked['actual'],trace['original_events'])
    downbeats=np.asarray(midi.get_downbeats(),float)
    starts=np.unique(np.r_[0.,downbeats[(downbeats>0)&(downbeats<end-1e-10)]])
    bounds=np.r_[starts,end];bars=[]
    for index,(s,e) in enumerate(zip(bounds[:-1],bounds[1:])):
        row=window_metrics(trace,float(s),float(e),include_end=index==len(bounds)-2)
        onset_notes=[n for n in notes if n['start_s']>=s-1e-10 and n['start_s']<e-1e-10]
        row.update(bar=index+1,note_onsets=len(onset_notes),
            notes_without_any_activation=sum(n['any_activation_in_note'] is False for n in onset_notes),
            notes_without_new_activation_edge=sum(n['new_activation_edge_in_note'] is False for n in onset_notes),
            unsampled_notes=sum(n['sample_count']==0 for n in onset_notes))
        bars.append(row)
    for name in ('samples','tp','fp','fn','same_hand_contact_samples','cross_hand_contact_samples'):
        if sum(row[name] for row in bars)!=whole[name]:raise AssertionError('Bar partition does not reconstruct '+name)
    repeats=repeated_patterns(bars,raw)
    regions=[]
    for label,s,e in [('first_20s',0,min(20,end)),('after_20s',min(20,end),end),
                      ('first_half',0,end/2),('second_half',end/2,end),('last_20s',max(0,end-20),end)]:
        if e>s:regions.append(dict(label=label,**window_metrics(trace,s,e,include_end=e==end)))
    output.mkdir(parents=True)
    summary=dict(kind='continuous_full_score_analysis_v1',source_recording=str(recording),
        physical_trace_sha256=sha(recording/'physical_trace.npz'),source_report_sha256=sha(recording/'replay_report.json'),
        original_midi_sha256=sha(original_midi),analyzer_sha256=sha(__file__),model_sha256=source['model_sha256'],
        archive_sha256=source['source']['archive_sha256'],score_sha256=source['source']['score_sha256'],
        requested_duration_s=source.get('requested_duration_s'),recorded_end_s=end,original_score_end_s=score_end,
        entire_score_recorded=end>=score_end,continuous_recording=True,physical_segments_stitched=False,
        physics_hz=1/checked['physics_dt'],control_hz=source['control_hz'],
        excluded_preparation_seconds=source.get('excluded_preparation_seconds',0),
        global_metrics=whole,note_count=len(notes),notes_without_any_activation=sum(n['any_activation_in_note'] is False for n in notes),
        notes_without_new_activation_edge=sum(n['new_activation_edge_in_note'] is False for n in notes),
        unsampled_notes=sum(n['sample_count']==0 for n in notes),bar_count=len(bars),repeated_pattern_group_count=len(repeats),
        regions=regions,worst_20_bars=sorted(bars,key=lambda b:(b['micro_f1'],-b['fn']))[:20],
        time_signatures=[dict(time_s=float(t.time),numerator=t.numerator,denominator=t.denominator) for t in midi.time_signature_changes],
        midi_parser_warnings=parser_warnings,bar_definition='MIDI downbeats (including the encoded pickup); final interval stops at the recorded endpoint. Raw note times unchanged.',
        sample_partition='Half-open bars, last bar includes terminal sample; sums of bar TP/FP/FN/contact samples equal global totals.',
        f1_definition='2TP/(2TP+FP+FN); all-empty is 0 as in planner micro-F1; never average bar F1 for global score.',
        note_definition='Any sampled 100% key angle during original note interval is coverage. New observed rising edge is separate articulation evidence; active initial state is not an observed edge. Actual assigned-finger contact is not inferred.',
        region_definition='Overlapping first/last/half/after-20s views; not additive partitions.',
        collision_definition='Saved force-bearing same/cross-hand contacts from the same physical samples; occurrence counts, not static held-pose reward estimates.',
        limitations=['No repeated-passage dynamics reuse; matching musical patterns can have different continuous entry states and outcomes.',
                     'A complete recording or high score is not physical safety/hardware qualification.',
                     'Repeated-pattern matching excludes fingering and pedal; raw pitch/time/velocity/track identity is explicit.',
                     'Sub-sample contact/activation cannot be reconstructed; note timing uses existing 1e-10s numerical tolerance.'])
    write_json(output/'report.json',summary);write_json(output/'bars.json',bars);write_json(output/'notes.json',notes)
    write_json(output/'repeated_patterns.json',repeats)
    with (output/'bars.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bars[0]));writer.writeheader();writer.writerows(bars)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(18,9),sharex=True,layout='constrained')
    centers=[(b['start_s']+b['end_s'])/2 for b in bars]
    axes[0].plot(centers,[b['micro_f1'] for b in bars],label='Per MIDI bar micro-F1');axes[0].set_ylim(0,1);axes[0].set_ylabel('F1');axes[0].legend()
    for name,color in [('fn','#bd7e08'),('fp','#bd3333')]:axes[1].plot(centers,[b[name] for b in bars],label=name.upper()+' key-samples',color=color)
    axes[1].legend();axes[1].set_ylabel('Key-samples')
    for name in ('same_hand_contact_samples','cross_hand_contact_samples'):axes[2].plot(centers,[b[name] for b in bars],label=name)
    axes[2].legend();axes[2].set_ylabel('Physical samples');axes[2].set_xlabel('Original MIDI time (seconds); preparation excluded')
    for axis in axes:axis.grid(alpha=.2);axis.set_xlim(0,end)
    fig.suptitle(f'Continuous recorded score — {end:.3f}s / {score_end:.6f}s | global micro-F1 {whole["micro_f1"]:.6f}\n'
                 f'{1/checked["physics_dt"]:g}Hz actual 100% key activation; no checkpoint; no safety claim')
    fig.savefig(output/'bars_overview.png',dpi=140);plt.close(fig)
    pages=detail_plots(output,trace,trace['original_events'],summary['physical_trace_sha256'])
    summary['detail_page_count']=len(pages);summary['detail_page_index']='detail_pages/index.json'
    write_json(output/'report.json',summary)
    write_json(output/'manifest.json',dict(files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('recording','original-midi','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();result=analyze(args.recording,args.original_midi,args.output,allow_partial=args.allow_partial)
    print(json.dumps({k:result[k] for k in ('entire_score_recorded','recorded_end_s','bar_count','repeated_pattern_group_count','global_metrics')}))
