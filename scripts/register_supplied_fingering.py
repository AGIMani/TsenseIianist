#!/usr/bin/env python3
"""Attach supplied single-finger annotations without changing MIDI notes or timing."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import numpy as np,pretty_midi
 from tsensei_v3_2_pianist.music.midi_file import MidiFile,NoteTrajectory
 from tsensei_v3_2_pianist.mjlab_env.exact_score import validate_events,physical_coverage
 p=argparse.ArgumentParser();p.add_argument('--midi',type=Path,required=True);p.add_argument('--fingering',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 midi=MidiFile.from_file(a.midi);rows=[]
 for line in a.fingering.read_text().splitlines():
  if not line.strip() or line.startswith('//'):continue
  v=line.split();assert len(v)==8,v
  finger=int(v[7]);assert 1<=abs(finger)<=5
  rows.append(dict(id=int(v[0]),start=float(v[1]),end=float(v[2]),pitch=pretty_midi.note_name_to_number(v[3]),velocity=int(v[4]),channel=int(v[6]),part=abs(finger)-1+(5 if finger<0 else 0)))
 assert len(rows)==len(midi.seq.notes),(len(rows),len(midi.seq.notes))
 pending=set(range(len(rows)));maxerror=0
 for note in midi.seq.notes:
  eligible=[i for i in pending if rows[i]['pitch']==note.pitch and rows[i]['velocity']==note.velocity]
  assert eligible,('Unmatched MIDI',note)
  i=min(eligible,key=lambda i:abs(rows[i]['start']-note.start_time)+abs(rows[i]['end']-note.end_time));r=rows[i]
  error=max(abs(r['start']-note.start_time),abs(r['end']-note.end_time));assert error<=.0001,('Annotation mismatch',note,r,error)
  maxerror=max(error,maxerror);note.part=r['part'];pending.remove(i)
 ordered=sorted(midi.seq.notes,key=lambda n:(n.start_time,n.end_time,n.pitch,n.instrument));copies=[type(n)().FromString(n.SerializeToString()) for n in ordered];del midi.seq.notes[:];midi.seq.notes.extend(copies)
 assert not pending;assert midi.seq.notes[-1].end_time==max(n.end_time for n in midi.seq.notes)
 trimmed=midi.trim_silence();traj=NoteTrajectory.from_midi(trimmed,.05)
 conflicts=[i for i,chord in enumerate(traj.notes) if len({n.fingering for n in chord})!=len(chord)]
 events=np.array([[n.start_time,n.end_time,n.pitch-21,n.part] for n in trimmed.seq.notes]);duration=math.ceil(events[:,1].max()/.05)*.05+.25
 exact_error=None
 try:validate_events(events,np.array([[duration,0.]]),duration)
 except ValueError as e:exact_error=str(e)
 coverage={}
 for hz in (200,400,500,1000,2000,4000):
  try:
   counts=physical_coverage(events,length=round(duration/.05),control_dt=.05,physics_dt=1/hz);coverage[str(hz)]=dict(valid=True,min_samples=min(counts))
  except ValueError as e:coverage[str(hz)]=dict(valid=False,error=str(e))
 a.output.mkdir(parents=True,exist_ok=False);proto=a.output/'015-5_fingered.proto';midi.save(proto)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 report=dict(notes=len(rows),original_duration_s=max(n.end_time for n in midi.seq.notes),trim_origin_s=min(n.start_time for n in midi.seq.notes),trimmed_duration_s=float(events[:,1].max()),episode_duration_s=duration,control_frames=len(traj.notes),control_finger_conflict_frames=conflicts,exact_event_error=exact_error,physical_coverage=coverage,annotation_max_rounding_error_s=maxerror,midi_sha256=sha(a.midi),annotation_sha256=sha(a.fingering),proto_sha256=sha(proto),fingering_source='User supplied signed 1..5 right / -1..-5 left; no optimization or timing changes')
 (a.output/'registration_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
