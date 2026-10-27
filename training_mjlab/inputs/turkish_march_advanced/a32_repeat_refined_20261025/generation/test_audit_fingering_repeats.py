"""Pure event tests: no MIDI dependency, model, or simulation is loaded."""
import importlib.util
from pathlib import Path
import sys


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_fingering_repeats.py'
    spec=importlib.util.spec_from_file_location('repeat_audit_test',path)
    result=importlib.util.module_from_spec(spec);sys.modules[spec.name]=result;spec.loader.exec_module(result)
    return result


def test_exact_transposed_and_release_variant_patterns_are_distinct():
    m=module()
    a=m.Note(0,0,60,0,90,0.,.09,1,70)
    b=m.Note(1,0,62,100,190,.1,.19,2,70)
    c=m.Note(2,0,60,200,295,.2,.295,1,70)
    wa=m.Window(0,100,0,1,(a,));wb=m.Window(100,200,0,1,(b,));wc=m.Window(200,300,0,1,(c,))
    assert wa.signature('exact')!=wb.signature('exact')
    assert wa.signature('transposed')==wb.signature('transposed')
    assert wa.signature('exact')!=wc.signature('exact')
    assert wa.signature('release_variant')==wc.signature('release_variant')


def test_full_interval_guard_rejects_cross_bar_held_finger_collision():
    m=module()
    notes=[m.Note(0,0,60,0,200,0.,.2,1,70),m.Note(1,0,64,100,150,.1,.15,2,70)]
    assert m.assignment_issue(notes,{0:1,1:2},{}) is None
    assert 'overlap' in m.assignment_issue(notes,{0:1,1:1},{})
    assert 'fixed' in m.assignment_issue(notes,{0:2,1:3},{0:1})


def test_reuse_audit_includes_external_neighbor_and_same_key_retrigger():
    m=module()
    notes=[m.Note(0,0,60,0,100,0.,.1,1,70),m.Note(1,0,60,105,200,.105,.2,2,70)]
    assert m.short_reuses(notes,{0:1,1:2})==[]
    result=m.short_reuses(notes,{0:1,1:1})
    assert len(result)==1 and result[0]['same_pitch']
    assert abs(result[0]['gap_seconds']-.005)<1e-12


def test_near_pitch_edit_audit_excludes_transpositions_and_changed_rhythm():
    m=module()
    def window(start,pitches,duration=90):
        return m.Window(start,start+400,0,1,tuple(m.Note(i,0,p,start+i*100,start+i*100+duration,0,1,1,70) for i,p in enumerate(pitches)))
    a=window(0,[60,64,67,69]);b=window(400,[60,64,67,70]);c=window(800,[62,66,69,71]);d=window(1200,[60,64,67,70],80)
    records=m.near_pitch_edits([a,b,c,d],float)
    assert len(records)==1
    assert records[0]['changed_note_count']==1
    assert records[0]['finger_copy_allowed'] is False
