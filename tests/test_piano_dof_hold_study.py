import importlib
import numpy as np


def test_hold_request_keeps_wrist_goal_and_smoothly_introduces_finger_goal():
    m=importlib.import_module('scripts.hpc.piano_dof_hold_study')
    neutral=np.zeros(51);goal=np.ones(51)
    before=m.hold_request(neutral,goal,.5)
    assert np.all(before[[20,21,22,23,45,46,47,48]]==1)
    assert np.all(before[:20]==0)
    mid=m.hold_request(neutral,goal,1.25)
    assert np.allclose(mid[:20],.5)
    assert np.all(m.hold_request(neutral,goal,1.5)[:20]==1)
    assert np.all(m.hold_request(neutral,goal,3.)[:20]==1)


def test_hold_interval_uses_1000_poststep_samples_and_excludes_left_endpoint():
    m=importlib.import_module('scripts.hpc.piano_dof_hold_study')
    t=np.arange(3001)/1000
    mask=m.hold_window(t)
    assert mask.sum()==1000 and not mask[2000] and mask[3000]
