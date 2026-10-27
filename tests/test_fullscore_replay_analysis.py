import numpy as np
import json

from scripts.analyze_fullscore_replay import window_metrics, note_diagnostics, repeated_patterns


def test_report_serializes_numpy_metadata_without_stringifying_boolean(tmp_path):
    from scripts.analyze_fullscore_replay import write_json
    output=tmp_path/'report.json'
    write_json(output,dict(complete=222.474>=np.float64(222.47395833333334),samples=np.int64(222475)))
    assert json.loads(output.read_text())==dict(complete=True,samples=222475)


def test_bar_partition_counts_boundary_once_and_keeps_final_sample():
    time=np.arange(5)*.1
    actual=np.zeros((5,88),bool);actual[2:,0]=True
    target=np.zeros_like(actual);target[1:4,0]=True
    trace=dict(time=time,actual_keys=actual,target_keys=target,
               hand_contacts=np.array([[0,0],[0,0],[1,0],[0,0],[0,1]]))
    whole=window_metrics(trace,0,.4,include_end=True)
    left=window_metrics(trace,0,.2);right=window_metrics(trace,.2,.4,include_end=True)
    assert (whole['tp'],whole['fp'],whole['fn'])==(2,1,1)
    for key in ('samples','tp','fp','fn','same_hand_contact_samples','cross_hand_contact_samples'):
        assert whole[key]==left[key]+right[key]
    assert whole['micro_f1']==2/3


def test_held_key_is_activation_but_not_a_new_note_trigger():
    time=np.arange(5)*.1;actual=np.zeros((5,88),bool);actual[1:4,0]=True
    events=np.array([[.1,.2,0,0],[.2,.4,0,1],[0,.1,1,2]])
    notes=note_diagnostics(time,actual,events)
    assert notes[0]['new_activation_edge_in_note'] is True
    assert notes[1]['any_activation_in_note'] is True
    assert notes[1]['new_activation_edge_in_note'] is False
    assert notes[1]['already_active_before_first_note_sample'] is True
    assert notes[2]['any_activation_in_note'] is False


def test_equal_musical_patterns_keep_separate_physical_results():
    bars=[dict(bar=i+1,start_s=float(i),end_s=float(i+1),micro_f1=f)
          for i,f in enumerate([.2,.8,.1])]
    raw=[[.1,.2,60,100,0],[1.1,1.2,60,100,0],[2.1,2.2,61,100,0]]
    groups=repeated_patterns(bars,raw)
    assert len(groups)==1
    assert groups[0]['bars']==[1,2]
    assert groups[0]['occurrences'][0]['micro_f1']==.2
    assert groups[0]['occurrences'][1]['micro_f1']==.8
    assert groups[0]['physical_trajectories_reused'] is False


def test_unsampled_note_is_unknown_not_a_claimed_miss():
    notes=note_diagnostics(np.array([0.,.1]),np.zeros((2,88),bool),np.array([[.02,.03,0,0]]))
    assert notes[0]['sample_count']==0
    assert notes[0]['any_activation_in_note'] is None


def test_active_first_recorded_state_does_not_invent_an_observed_rising_edge():
    actual=np.zeros((3,88),bool);actual[:2,0]=True
    note=note_diagnostics(np.array([0.,.1,.2]),actual,np.array([[0.,.2,0,0]]))[0]
    assert note['any_activation_in_note'] is True
    assert note['new_activation_edge_in_note'] is False
    assert note['active_at_recording_start'] is True
