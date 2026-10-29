"""Opt-in Slurm two-world range regression; no learner or replay allocation."""
import json
import os


def test_current_pitch_range(monkeypatch):
    import pytest
    if 'PIANIST_ROLL_ACCEPTANCE_DIR' not in os.environ:
        pytest.skip('Explicit remote Slurm validation only')
    from tests import test_roll_capacity_acceptance as harness

    def exercise(args, destination, report):
        import numpy as np
        import torch
        from tsensei_v3_2_pianist.rl_torch.train import effective_config
        from tsensei_v3_2_pianist.mjlab_env.preview_contract import task_kwargs
        from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
        torch.set_num_threads(2)
        config = effective_config(args)
        config['source_runtime'] = json.loads(args.source_config.read_text())
        report.update(scope='2 worlds, 128 real control steps each; first-note preparation; no replay/learner',
                      stage='building', num_envs=2, learner_updates=0, replay_allocated=False)
        env = PianoTaskEnv(args.scene, args.task, num_envs=2, device='cuda:0', **task_kwargs(config))
        assert env.action_size == 33 and env.observation_size == 1410
        assert env.scene.decimation == 10 and env.logic.length == 11137
        config.update(action_contract=env.scene.action_contract, control_limits_contract=env.scene.control_limits,
                      preparation_contract=env.scene.bundle.preparation)
        harness.save(destination/'config.json', config)
        rows = [r for r in env.scene.action_contract['channels'] if r['name'].endswith('_Pitch')]
        assert len(rows) == 2
        full = [r['full_index'] for r in rows]
        for side, sign in [('right',1),('left',-1)]:
            model = env.scene.model
            joint = model.joint(side+'_/palm_base_roll')
            assert np.allclose(joint.range, [-.3,.5])
            assert np.allclose(joint.axis, [0,0,sign])
            np.testing.assert_allclose(model.joint(side+'_/palm_base_pitch').range,[-.3,.6])
            for axis in ('Pitch','Roll'):
                actuator=model.actuator(side[0].upper()+'_Palm_Base_'+axis)
                assert np.allclose(actuator.forcerange,[-18,18])
        for r in rows:
            assert (r['physical_min'],r['physical_max']) == (-.3,.6)
        with torch.no_grad():
            obs=env.reset()
            previous=env.scene.control_reset_action[full].double().expand(2,-1).clone()
            velocity=torch.zeros_like(previous)
            peaks=[0.,0.];lo=1.;hi=-1.
            # Explicit physical zero must survive the asymmetric affine mapping.
            zero=env.scene.control_reset_action.clone();zero[full]=0
            policy=env.scene.policy_action(zero)
            for r in rows: assert abs(policy[r['index']].item()-2/3)<1e-6
            for step in range(128):
                action=env.scene.policy_action(env.scene.control_reset_action).expand(2,-1).clone()
                for hand,r in enumerate(rows):
                    action[:,r['index']] = r['policy_max'] if (step//32+hand)%2 else r['policy_min']
                result=env.step(action)
                assert not env.scene.data.overflow[:].any()
                for x in (result.observation,result.reward,env.scene.data.qpos,env.scene.data.qvel): assert torch.isfinite(x).all()
                q=env.logic.last_action[:,full].double();v=(q-previous)/.02;a=(v-velocity)/.02
                assert (q>=-.3-1e-7).all() and (q<=.6+1e-7).all()
                assert v.abs().max()<=3.1416+1e-8 and a.abs().max()<=15.708+1e-7
                peaks=[max(peaks[0],v.abs().max().item()),max(peaks[1],a.abs().max().item())]
                lo=min(lo,q.min().item());hi=max(hi,q.max().item());previous=q.clone();velocity=v.clone()
            assert abs(lo+.3)<1e-6 and abs(hi-.6)<1e-6
            assert torch.isfinite(env.reset([0])).all()
        report.update(passed=True,status='passed',stage='completed',pitch_target_observed_range=[lo,hi],
                      pitch_target_speed_max=peaks[0],pitch_target_acceleration_max=peaks[1],
                      action_size=env.action_size,observation_size=env.observation_size,
                      control_steps_per_world=128,physical_seconds_per_world=2.56,zero_policy_coordinate=2/3)
    monkeypatch.setattr(harness,'exercise',exercise)
    harness.test_joint_capacity_physics_default()
