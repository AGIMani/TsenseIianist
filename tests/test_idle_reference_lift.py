"""Rebuild the user-selected reset reference and check actual fingertip FK."""
import json
from pathlib import Path
import numpy as np

def test_rebuilt_reset_reference_lifts_all_eight_tips(monkeypatch):
    monkeypatch.setenv('TSENSEI_V32_EXPORT_POSE', '{"height":0.06,"ty":0.04,"yaw":0.5235987755982988}')
    monkeypatch.setenv('TSENSEI_V32_COLLISION_PROFILE', 'convex_hull_v1')
    monkeypatch.setenv('TSENSEI_V32_DRIVE_PROFILE', 'shadow_v1')
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as c
    c._apply_export_pose_overrides()
    env=suite.load(task_kwargs=dict(control_timestep=.05,physics_timestep=.005,
        gravity_compensation=True,task_profile='tsensei_collision_v1',initial_buffer_time=0.))
    try:
        env.reset()
        old_z=dict(index=.08547007653,mid=.09135827101,ring=.08404048387,pinky=.07597382848)
        for side in ('right','left'):
            for finger,z in old_z.items():
                tip=env.physics.named.data.site_xpos[f'{side}_/{finger}_tip']
                np.testing.assert_allclose(tip[2]-z,.008,atol=1e-9,rtol=0)
                assert c.INITIAL_ANGLES_BY_SIDE[side][finger+'_swing']==0.
            assert c.INITIAL_ANGLES_BY_SIDE[side]['thumb_MCP']==.241250147
    finally:
        env.close()
