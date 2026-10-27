"""CPU/GPU resistance parity against an independently compiled current scene."""
import os
from pathlib import Path
import numpy as np
import pytest

def test_current_resistance_cuda():
    import mujoco
    import torch
    import warp as wp
    from mjlab.sim import SimulationCfg
    from mjlab.sim.sim import Simulation
    from tsensei_v3_2_pianist.mjlab_env.scene import ImportedMujocoCfg
    from tsensei_v3_2_pianist.finger_resistance import Resistance
    from tsensei_v3_2_pianist.mjlab_env.finger_resistance import BatchedResistance
    assert os.environ.get('SLURM_JOB_ID'), 'GPU validation requires Slurm'
    folder=Path(os.environ['PIANIST_RESISTANCE_VALIDATION'])
    m=mujoco.MjModel.from_xml_path(str(folder/'compiled_xml/scene.xml'))
    r=Resistance(m);assert len(r.rows)==40
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        sim=Simulation(num_envs=2,model=m,device='cuda:0',cfg=SimulationCfg(mujoco=ImportedMujocoCfg(timestep=.005),nconmax=1024,njmax=4096))
        b=BatchedResistance(m,sim.wp_model,sim.wp_data,2,'cuda:0')
        sim.create_graph()
        qs=[];expected=[]
        for angle in [.1,.5]:
            d=mujoco.MjData(m)
            for j,_,_,_ in r.rows:
                d.qpos[m.jnt_qposadr[j]]=np.clip(angle,*m.jnt_range[j])
            r.update(d);qs.append(d.qpos.copy());expected.append(m.dof_frictionloss.copy())
        sim.data.qpos[:]=torch.tensor(np.array(qs),device='cuda:0',dtype=torch.float32)
        b.update();wp.synchronize()
        np.testing.assert_allclose(sim.wp_model.dof_frictionloss.numpy(),expected,atol=2e-5,rtol=2e-5)
        assert not np.allclose(expected[0],expected[1])
        for _ in range(20):
            b.update();sim.step()
        wp.synchronize()
        assert torch.isfinite(sim.data.qpos).all()
        (folder/'gpu_passed.txt').write_text('40 joints; 2 distinct poses/worlds; CPU-GPU friction parity; 20 GPU steps passed\n')
