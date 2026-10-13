"""Check the intervention, not MuJoCo's solver (rollouts belong in Slurm)."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location('strike_diagnostic', Path(__file__).resolve().parents[1]/'scripts/hpc/strike_timing_diagnostic.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_lead_is_bitwise_baseline():
    ctrl=np.arange(120,dtype=float).reshape(20,6)
    result=MODULE.shifted_targets(ctrl,[1,2,3],onset=5,release=10,lead=0)
    np.testing.assert_array_equal(result,ctrl)


def test_advance_holds_press_tuple_and_preserves_release_and_other_channels():
    ctrl=np.arange(120,dtype=float).reshape(20,6)
    result=MODULE.shifted_targets(ctrl,[1,2,3],onset=5,release=10,lead=2)
    np.testing.assert_array_equal(result[:3],ctrl[:3])
    np.testing.assert_array_equal(result[10:],ctrl[10:])
    np.testing.assert_array_equal(result[:,[0,4,5]],ctrl[:,[0,4,5]])
    np.testing.assert_array_equal(result[3:10,1:4],ctrl[[5,6,7,8,9,9,9],1:4])


def test_donor_preserves_complete_coupled_tuple():
    ctrl=np.arange(180,dtype=float).reshape(30,6)
    result=MODULE.shifted_targets(ctrl,[1,2,3],onset=5,release=9,lead=1,
                                  donor_onset=20,donor_release=23)
    np.testing.assert_array_equal(result[4:9,1:4],ctrl[[20,21,22,22,22],1:4])
    np.testing.assert_array_equal(result[9:],ctrl[9:])
    np.testing.assert_array_equal(result[:,[0,4,5]],ctrl[:,[0,4,5]])


def test_activation_time_is_post_integration_and_half_open_note():
    times=np.array([.100,.1025,.105,.1075])
    note=dict(start_time=.1,end_time=.106,target_start_time=.1,target_end_time=.125)
    event=MODULE.observed_note(np.array([False,True,True,False]),False,times,note)
    assert np.isclose(event['onset_vs_original_ms'],2.5)
    assert np.isclose(event['offset_vs_original_ms'],1.5)


def test_release_locator_preserves_early_learned_release():
    ctrl=np.zeros((15,2));ctrl[5:9,0]=1
    assert MODULE.release_frame(ctrl,0,5,10)==9
