"""Opt-in Slurm physical smoke/throughput check, using a recorded scene."""
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('COLLISION_REPAIR_MODEL'), reason='Requires explicit repair scene in a Slurm allocation')
def test_compound_scene_512_worlds_without_overflow():
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    assert os.environ.get('SLURM_JOB_ID'), 'GPU validation must run through Slurm'
    wp.init()
    model=mujoco.MjModel.from_binary_path(os.environ['COLLISION_REPAIR_MODEL'])
    with np.load(os.environ['COLLISION_REPAIR_TRACE']) as z:q=z['qpos'][0]
    with wp.ScopedDevice('cuda:0'):
        gpu=mw.put_model(model)
        data=mw.make_data(model,nworld=512,nconmax=1024,njmax=4096)
        data.qpos.assign(np.tile(q,(512,1)).astype(np.float32))
        ctrl=np.clip(q[model.jnt_qposadr[model.actuator_trnid[:,0]]],
                     model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
        data.ctrl.assign(np.tile(ctrl,(512,1)).astype(np.float32))
        mw.forward(gpu,data)
        for _ in range(10):mw.step(gpu,data)
        wp.synchronize()
        started=time.perf_counter()
        for _ in range(100):mw.step(gpu,data)
        wp.synchronize();elapsed=time.perf_counter()-started
        assert np.isfinite(data.qpos.numpy()).all()
        assert np.isfinite(data.qvel.numpy()).all()
        assert not data.overflow.numpy().any()
        report=dict(job_id=os.environ['SLURM_JOB_ID'],nworld=512,physics_steps=110,
                    timed_physics_steps=100,seconds=elapsed,ngeom=model.ngeom,
                    overflow=False,finite=True,model=os.environ['COLLISION_REPAIR_MODEL'],
                    scope='Constant-position physical smoke test; not a trained policy or training throughput')
        Path(os.environ['COLLISION_REPAIR_REPORT']).write_text(json.dumps(report,indent=2)+'\n')
