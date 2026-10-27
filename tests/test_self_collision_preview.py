"""Pure display/protocol checks; no MuJoCo import or model work."""
import numpy as np
import pytest
from scripts.self_collision_preview import check_pose, selected_controls, slider_value


def test_drag_is_clamped_and_mirrored_hands_never_share_controls():
    assert slider_value(-10,0,100,0,1.4)==0
    assert slider_value(120,0,100,0,1.4)==1.4
    assert slider_value(25,0,100,-.5,.5)==-.25
    rows=[{'name':n} for n in ('R_Index_MCP','L_Index_MCP','R_Palm_Base_Pitch','R_Pinky_MCP')]
    assert selected_controls(rows,'L','Index')==[rows[1]]
    assert selected_controls(rows,'R','Wrist')==[rows[2]]


def test_display_refuses_invalid_remote_state_or_positive_penalty():
    response=dict(qpos=[0,1],score=dict(instant_substep_penalty=-.1,held_control_step_penalty=-5.))
    np.testing.assert_array_equal(check_pose(response,2),[0,1])
    with pytest.raises(ValueError,match='pose'):
        check_pose(response,3)
    response['qpos']=[0,float('nan')]
    with pytest.raises(ValueError,match='pose'):
        check_pose(response,2)
    response['qpos']=[0,1]
    response['score']['held_control_step_penalty']=.1
    with pytest.raises(ValueError,match='penalty'):
        check_pose(response,2)
