"""Array-free scheduling/cache contracts; never load MuJoCo."""
import importlib
import pytest


def api():
    return importlib.import_module('scripts.hpc.piano_fullscore_contact_bank')


def note(index, pitch, finger, start, end):
    return dict(index=index, pitch=pitch, finger=finger, start=start, end=end)


def test_onset_nodes_keep_held_context_and_release_before_same_time_onset():
    rows=[note(0,60,1,0.,1.),note(1,64,2,.5,1.),note(2,65,2,1.,2.)]
    nodes=api().onset_nodes(rows)
    assert [n['time_seconds'] for n in nodes]==[0.,.5,1.]
    assert nodes[1]['held_note_indices']==[0]
    assert nodes[1]['onset_note_indices']==[1]
    assert [n['index'] for n in nodes[2]['notes']]==[2]
    with pytest.raises(ValueError,match='overlap'):
        api().onset_nodes(rows+[note(3,67,1,.75,1.25)])


def test_static_signature_keeps_both_hands_and_contract_but_not_duration():
    m=api();c={'model':'abc','geometry':'def','idle':'fixed'}
    a={'notes':[note(0,60,1,0,1),note(1,48,8,0,1)]}
    b={'notes':[note(8,48,8,4,6),note(9,60,1,4,5)]}
    assert m.static_signature(a,c)==m.static_signature(b,c)
    assert m.static_signature(a,c)!=m.static_signature(a,{**c,'model':'changed'})
    assert m.static_signature(a,c)!=m.static_signature({'notes':[a['notes'][0]]},c)
    assert m.static_signature(a,c)!=m.static_signature({'notes':[note(0,72,1,0,1),a['notes'][1]]},c)


def test_cache_is_only_finite_action_seed_with_matching_signature_and_contract():
    m=api();c={'model':'abc'};n={'notes':[note(0,60,1,0,1)]};signature=m.static_signature(n,c)
    row=dict(kind='static_candidate_v1',signature=signature,contract_sha256=m.digest(c),action=[0.]*51)
    assert m.cached_action(row,signature,c)==[0.]*51
    for changed in ({**row,'contract_sha256':'wrong'}, {**row,'action':[float('nan')]*51},
                    {**row,'kind':'physical_success'}, {**row,'signature':'another'}):
        with pytest.raises(ValueError):m.cached_action(changed,signature,c)


def test_native_finite_top_requires_terminal_normal_and_interior_not_distance_proxy():
    check=api().finite_top_contact
    assert check([0.,0.,.01],[.05,.01,.01],.9,True,-.0005)
    assert not check([0.,0.,.01],[.05,.01,.01],.3,True,-.0005)
    assert not check([.049,0.,.01],[.05,.01,.01],.9,True,-.0005)
    assert not check([0.,0.,.01],[.05,.01,.01],.9,False,-.0005)
    assert not check([0.,0.,.01],[.05,.01,.01],.9,True,.00001)


def test_occurrence_requires_new_check_and_does_not_copy_physical_result():
    m=api();node=dict(time_seconds=1.,notes=[note(7,60,1,1,2)],held_note_indices=[],onset_note_indices=[7])
    candidate=dict(signature='s',action=[0.]*51)
    with pytest.raises(ValueError,match='revalidation'):
        m.occurrence_record(node,candidate,None)
    row=m.occurrence_record(node,candidate,dict(static_passed=True,coupling_converged=True))
    assert row['revalidated'] and row['static_passed']
    assert row['physics_steps']==0 and row['dynamic_acceptance'] is None
    assert row['held_note_indices']==[] and row['onset_note_indices']==[7]
