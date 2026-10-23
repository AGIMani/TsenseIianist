"""Reference Composer callback parity; run on the remote host with Composer Python."""
import os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.skipif(not os.environ.get('TSENSEI_REMOTE_CPU_TEST'),reason='Remote model/physics test only')
def test_composer_substeps_and_reset():
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.fingering_targets import target_position, target_for_key
    import mujoco
    env=suite.load(str(Path(__file__).parent/'fixtures/exact_fingering_015.proto'),task_kwargs=dict(
        control_timestep=.05,physics_timestep=.005,trim_silence=True,
        task_profile='tsensei_collision_v1',initial_buffer_time=0.))
    env.reset();task=env.task
    original=task.after_substep
    expected=[]
    def collect(physics,random_state):
        original(physics,random_state)
        t=physics.time();samples=[]
        # Independent native kinematics from the physical state at this sample.
        state=mujoco.MjData(physics.model.ptr);state.qpos[:]=physics.data.qpos
        mujoco.mj_kinematics(physics.model.ptr,state)
        for note in task._midi.seq.notes:
            if note.start_time-1e-10<=t<note.end_time-1e-10:
                hand=task.right_hand if note.part<5 else task.left_hand
                site=hand.fingertip_sites[note.part%5]
                geom=task.piano.keys[note.pitch-21].geom[0]
                si=physics.model.name2id(site.full_identifier,'site');gi=physics.model.name2id(geom.full_identifier,'geom')
                target=target_for_key(physics.model.ptr,state,gi,task_profile=task._task_profile)
                distance=np.linalg.norm(state.site_xpos[si]-target)
                samples.append(np.exp(np.log(.1)*(max(distance-.01,0)/.1)**2))
        expected.append(float(np.mean(samples)) if samples else 0.)
    # Composer caches hooks at environment construction; use its task callback
    # wrapper table by rebuilding through the task, then verify samples itself.
    task.after_substep=collect
    action=np.zeros(env.action_spec().shape)
    env.step(action)
    assert len(task._finger_substep_scores)==10
    assert np.isfinite(task._compute_fingering_reward(env.physics))
    assert len(expected)==10
    np.testing.assert_allclose(task._finger_substep_scores,expected,atol=1e-10)
    assert any(x>0 for x in task._finger_substep_scores)
    env.reset()
    assert task._compute_fingering_reward(env.physics)==0
