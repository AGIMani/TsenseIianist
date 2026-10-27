"""Pure contact contracts; no model loading, FK or physics."""
import ast
import importlib
from pathlib import Path

import pytest


def api():
    return importlib.import_module('scripts.self_collision_metrics')


def body(side, finger, link):
    return f'{side}_/{side}_hand_{finger}_{link}'


def test_all_same_finger_links_and_palm_are_excluded():
    m=api()
    for a,b in [('MCP_link','DIP_link'),('swing_link','PIP_link')]:
        c=m.classify_contact(body('right','index',a),body('right','index',b),-.001)
        assert c['category']=='same_finger' and not c['reward_counted']
    for link in ('rota_block_link','rotaback_link2','MCP_link','PIP_link'):
        assert m.finger_identity(body('left','thumb',link))==('left','thumb')
    assert not m.classify_contact('left_/base_link',body('left','index','DIP_link'),-.002)['reward_counted']
    assert m.finger_identity('right_/right_hand_index_unknown_link') is None


def test_distinct_body_pairs_deduplicate_points_but_not_separate_links():
    m=api();a=body('right','index','DIP_link');b=body('right','mid','PIP_link')
    rows=[m.classify_contact(a,b,-.001),m.classify_contact(b,a,-.002),
          m.classify_contact(a,body('right','mid','DIP_link'),0.),
          m.classify_contact(a,body('left','index','DIP_link'),-.004),
          m.classify_contact(a,b,.00001)]
    score=m.score_contacts(rows,physics_hz=1000,control_hz=20)
    assert score['eligible_body_pair_count']==2
    assert score['instant_substep_penalty']==pytest.approx(-.2)
    assert score['held_control_step_penalty']==-5
    assert score['substeps_per_control_step']==50
    assert m.score_contacts(rows,physics_hz=200,control_hz=20)['held_control_step_penalty']==-2


def test_weld_and_explicit_reward_exclusions_are_respected():
    m=api();a=body('right','index','DIP_link');b=body('right','mid','DIP_link')
    assert not m.classify_contact(a,b,-.001,same_weld=True)['reward_counted']
    assert not m.classify_contact(a,b,-.001,ignored=True)['reward_counted']
    assert m.classify_contact(a,b,0.)['reward_counted']  # No force threshold in training.


def test_classification_matches_runtime_name_contract_without_importing_runtime():
    path=Path(__file__).resolve().parents[1]/'tsensei_v3_2_pianist/mjlab_env/self_collision.py'
    tree=ast.parse(path.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'finger_identity','is_distinct_finger_pair'}]
    namespace={};exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),namespace)
    names=[body(s,f,k) for s in ('left','right') for f in ('thumb','index','mid','ring','pinky')
           for k in ('rota_block_link','rotaback_link2','MCP_link','PIP_link','DIP_link','swing_link')]
    names+=['left_/base_link','right_/palm_base_pitch','piano/base']
    assert [api().finger_identity(n) for n in names]==[namespace['finger_identity'](n) for n in names]


def test_requests_reject_followers_nonfinite_and_out_of_bounds_without_mutation():
    m=api();controls=[dict(name='R_Index_MCP',min=0.,max=1.4)]
    assert m.validate_request({'id':3,'values':{'R_Index_MCP':.5}},controls)=={'R_Index_MCP':.5}
    for values in ({'R_Index_PIP':.5},{'R_Index_MCP':float('nan')},{'R_Index_MCP':1.5}):
        with pytest.raises(ValueError):m.validate_request({'values':values},controls)
    with pytest.raises(ValueError):m.validate_request({'reset':'yes'},controls)


def test_probe_rejects_a_diagnostic_with_changed_coupling_range():
    import copy
    options=dict(schema=1,mode='distal_angle_range_v1',reduce_observation=False,
        angle_ranges_deg={s+'_'+f:[10.,80.] for s in ('left','right') for f in ('index','mid','ring','pinky')})
    api().validate_coupling_contract(options)
    wrong=copy.deepcopy(options);wrong['angle_ranges_deg']['right_index']=[60.,80.]
    with pytest.raises(ValueError,match='10.*80'):api().validate_coupling_contract(wrong)
