"""Explicit per-axis servos must retain their selected damping and mass checks."""
import copy
import json
import pytest
from tsensei_v3_2_pianist import drive_settings
from tsensei_v3_2_pianist.drive_settings import load, translation_damping

@pytest.mark.parametrize('joint,damping',[('forearm_tx',34.15259872981849),('forearm_ty',34.15259872981849)])
def test_explicit_damping_is_not_replaced_by_critical_value(joint,damping):
    settings=load('shadow_v1')['translation']
    assert translation_damping(settings,.972,joint)==damping
    assert settings['axes'][joint]['kp']==300.
    assert settings['axes'][joint]['force_limit']==50.

@pytest.mark.parametrize('mass',[.952,3.952,0.,float('nan')])
def test_stale_or_invalid_compiled_mass_is_rejected(mass):
    with pytest.raises(ValueError,match='compiled inertia'):
        translation_damping(load('shadow_v1')['translation'],mass,'forearm_tx')

@pytest.mark.parametrize('bad',[True,0.,-1.,float('nan'),float('inf')])
def test_invalid_axis_values_are_rejected(tmp_path,monkeypatch,bad):
    doc=json.loads(drive_settings.SETTINGS_PATH.read_text())
    doc['profiles']['shadow_v1']['translation']['axes']['forearm_ty']['damping']=bad
    path=tmp_path/'drive.json';path.write_text(json.dumps(doc));monkeypatch.setattr(drive_settings,'SETTINGS_PATH',path)
    with pytest.raises(ValueError,match='Invalid translation drive value'):load('shadow_v1')

def test_missing_axis_and_legacy_shared_fields_are_rejected(tmp_path,monkeypatch):
    original=json.loads(drive_settings.SETTINGS_PATH.read_text());path=tmp_path/'drive.json'
    monkeypatch.setattr(drive_settings,'SETTINGS_PATH',path)
    for kind in ('missing','shared'):
        doc=copy.deepcopy(original);settings=doc['profiles']['shadow_v1']['translation']
        if kind=='missing':del settings['axes']['forearm_ty']
        else:settings['kp']=300.
        path.write_text(json.dumps(doc))
        with pytest.raises(ValueError,match='Unsupported translation drive contract'):load('shadow_v1')
