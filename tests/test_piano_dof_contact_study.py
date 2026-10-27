"""Pure study-selection and canonical mapping tests; no model imports."""
import importlib
import pytest


def api():
    return importlib.import_module('scripts.hpc.piano_dof_contact_study')


def test_actual_names_resolve_live_tz_without_49_channel_heuristic():
    names=['f'+str(i) for i in range(51)]
    names[22]='R_TZ';names[24]='R_Yaw';names[47]='L_TZ';names[49]='L_Yaw'
    live=[n for i,n in enumerate(names[:50]) if i not in (24,49)]
    ids=api().canonical_indices(names,live)
    assert len(ids)==48 and 22 in ids and 47 in ids and 24 not in ids and 49 not in ids
    with pytest.raises(ValueError):api().canonical_indices(names,live+['unknown'])


def test_selection_pairs_whole_bilateral_combination_and_covers_boundaries():
    def row(s,t,r,l,fingers,pitches,passed=False):
        a=[0.]*51;a[23]=r;a[48]=l
        return dict(signature=s,time_seconds=t,action=a,static_passed=passed,
                    notes=[dict(finger=f,pitch=p) for f,p in zip(fingers,pitches)])
    rows=[row('upper',0,.5,.2,[0,4,6],[60,72,48]),row('lower',1,.0,.2,[1,7],[65,55]),
          row('wide',2,.2,.2,[0,1,2,4],[60,64,67,72]),row('control',3,.2,.2,[1,7],[64,48],True)]
    selected=api().select_cases(rows,maximum=4)
    assert {x['signature'] for x in selected}=={x['signature'] for x in rows}
    assert selected[0]['notes']==rows[0]['notes']
    assert {'upper_pitch','lower_pitch','wide_four','static_pass_control'} <= {v for r in selected for v in r['selection_reasons']}


def test_variant_ranges_do_not_allow_expanding_finger_contract():
    channels=[dict(name='R_Index_MCP',physical_min=0.,physical_max=1.4),
              dict(name='R_Palm_Base_Tz',physical_min=-.025,physical_max=0.),
              dict(name='L_Palm_Base_Pitch',physical_min=0.,physical_max=.5)]
    patched=api().patch_channels(channels,-.1,.7)
    assert patched[0]==channels[0] and channels[1]['physical_min']==-.025
    assert (patched[1]['physical_min'],patched[1]['physical_max'])==(0.,.03)
    assert (patched[2]['physical_min'],patched[2]['physical_max'])==(-.1,.7)
    with pytest.raises(ValueError):api().patch_channels(channels,-.5,1.5)
