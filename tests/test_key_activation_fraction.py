"""Cross-consumer activation contract, no model loading in these unit tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tsensei_v3_2_pianist.key_activation import KEY_ACTIVATION_FRACTION, KEY_PRESS_TOLERANCE, key_active
from tsensei_v3_2_pianist.preview_audio import events_from_states
from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight


def test_numpy_audio_highlight_and_white_black_boundaries():
    from tsensei_v3_2_pianist.models.piano import piano_constants as pc
    upper=np.array([pc.WHITE_KEY_JOINT_MAX_ANGLE, pc.BLACK_KEY_JOINT_MAX_ANGLE])
    limits=np.column_stack([np.zeros(2),upper])
    fractions=np.array([0.,.86891,.90157,.949,.95,.951,1.,1.1,0.])
    q=fractions[:,None]*upper
    expected=np.tile((fractions>=1.)[:,None],(1,2))
    np.testing.assert_array_equal(key_active(q,limits),expected)
    assert KEY_ACTIVATION_FRACTION == 1.
    assert KEY_PRESS_TOLERANCE == .05
    highlight=ActiveKeyHighlight.__new__(ActiveKeyHighlight)
    highlight.qadr=np.array([0,1]);highlight.maximum=upper
    for row,want in zip(q,expected):
        np.testing.assert_array_equal(highlight.active(SimpleNamespace(qpos=row)),want)
    full=np.zeros((len(q),88));full[:,:2]=q
    full_limits=np.tile(limits[0],(88,1));full_limits[1]=limits[1]
    events=events_from_states(full,full_limits,np.zeros(len(q)),.005,1.)
    assert [(e['type'],e['note'],e['time']) for e in events]==[
        ('note_on',21,.035),('note_on',22,.035),('note_off',21,.045),('note_off',22,.045)]


def test_old_threshold_and_reward_contract_are_rejected(tmp_path):
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    from tsensei_v3_2_pianist.mjlab_env.reward_config import validate
    for fraction in (.00872665, .9, .95, float('nan')):
        with pytest.raises(ValueError):key_active(np.zeros(1),np.array([[0.,1.]]),fraction)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema_version=1,key_threshold=.00872665)))
    with pytest.raises(ValueError,match='Old key activation'):SceneBundle(tmp_path)
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/rewards/default_20261006.json').read_text())
    cfg['contract']='physical_key_fingering_exact_v1'
    with pytest.raises(ValueError,match='activation100'):validate(cfg)


def test_composer_piano_physical_state_matches_audio_threshold():
    import os, subprocess
    if not os.environ.get('SLURM_JOB_ID'):
        pytest.skip('Remote Slurm model construction only')
    root=Path(__file__).resolve().parents[1]
    code='''
import numpy as np
from dm_control import mjcf
from tsensei_v3_2_pianist.models.piano.piano import Piano
from tsensei_v3_2_pianist.key_activation import key_active
p=Piano()
physics=mjcf.Physics.from_mjcf_model(p.mjcf_model)
p.initialize_episode(physics,np.random.RandomState(0))
limits=physics.bind(p.joints).range.copy()
for fraction,active in [(0.,False),(.90,False),(.949,False),(.95,False),(.951,False),(1.,True),(1.1,True)]:
    physics.bind(p.joints).qpos=limits[:,1]*fraction
    p._update_key_state(physics)
    expected=key_active(physics.bind(p.joints).qpos,limits)
    np.testing.assert_array_equal(p.activation,expected)
    assert bool(p.activation.all())==active
print('Composer 88-key normalized activation boundaries passed')
'''
    subprocess.run([str(root/'training/.venv/bin/python'),'-c',code],cwd=root,check=True)
