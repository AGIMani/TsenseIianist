import json, math
from pathlib import Path
import numpy as np
import mujoco
from dm_control import mjcf
from tsensei_v3_2_pianist.models.piano.piano import Piano
from tsensei_v3_2_pianist.models.piano import piano_mjcf
from tsensei_v3_2_pianist.key_activation import key_active, KEY_PRESS_TOLERANCE
p=Piano();physics=mjcf.Physics.from_mjcf_model(p.mjcf_model)
p.initialize_episode(physics,np.random.RandomState(0))
limits=physics.bind(p.joints).range.copy()
black=np.array([(i+21)%12 in (1,3,6,8,10) for i in range(88)])
expected=np.where(black,math.atan(.010/.090),math.atan(.012/.150))
np.testing.assert_allclose(limits[:,1],expected,atol=1e-14,rtol=0)
for fraction in [0.,.95,.999999,1.,1.1]:
 physics.bind(p.joints).qpos=limits[:,1]*fraction
 p._update_key_state(physics)
 np.testing.assert_array_equal(p.activation,np.full(88,fraction>=1))
 np.testing.assert_array_equal(p.activation,key_active(physics.bind(p.joints).qpos,limits))
assert KEY_PRESS_TOLERANCE==.05
actuated=mjcf.Physics.from_mjcf_model(piano_mjcf.build(add_actuators=True))
np.testing.assert_allclose(actuated.model.actuator_ctrlrange[:,1],expected,atol=1e-14,rtol=0)
physics.reset()
for _ in range(20):physics.step()
assert np.isfinite(physics.data.qpos).all() and not np.any(physics.data.warning.number)
# Small, standalone piano snapshots, never substituted into old recorded replay.
Path('../validation/piano.xml').write_text(p.mjcf_model.to_xml_string())
mujoco.mj_saveModel(physics.model.ptr,'../validation/piano.mjb',None)
result={'keys':88,'white_keys':52,'black_keys':36,'nominal_travel_mm':{'white':12,'black':10},'joint_max_rad':{'white':float(expected[0]),'black':float(expected[1])},'joint_max_deg':{'white':math.degrees(expected[0]),'black':math.degrees(expected[1])},'activation_fraction':1,'continuous_reward_full_fraction':.95,'boundary_checks':'passed','actuator_ranges':'passed','steps':20,'warnings':0,'mujoco_version':mujoco.__version__}
Path('../validation/travel_report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
