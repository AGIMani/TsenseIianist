"""Opt-in Slurm two-world range regression; no learner or replay allocation."""
import json
import os


def test_current_palm_ranges(monkeypatch):
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
        from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
        raw=SceneBundle(args.scene,preparation=None)
        for side in ('right','left'):
            for axis,bounds in [('ty',[-.05,.05]),('tz',[-.6,.46])]:
                np.testing.assert_allclose(raw.model.joint(side+'_/palm_base_'+axis).range,bounds)
                np.testing.assert_allclose(raw.model.actuator(side[0].upper()+'_Palm_Base_'+axis.title()).ctrlrange,bounds)
        del raw
        import gc;gc.collect()
        env = PianoTaskEnv(args.scene, args.task, num_envs=2, device='cuda:0', **task_kwargs(config))
        assert env.action_size == 33 and env.observation_size == 1410
        assert env.scene.decimation == 10 and env.logic.length == 11137
        config.update(action_contract=env.scene.action_contract, control_limits_contract=env.scene.control_limits,
                      preparation_contract=env.scene.bundle.preparation)
        harness.save(destination/'config.json', config)
        rows = [r for r in env.scene.action_contract['channels'] if r['name'].endswith(('_Pitch','_Ty'))]
        assert len(rows) == 4
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
        lows=torch.tensor([-.3 if r['name'].endswith('_Pitch') else -.05 for r in rows],device='cuda:0',dtype=torch.float64)
        highs=torch.tensor([.6 if r['name'].endswith('_Pitch') else .05 for r in rows],device='cuda:0',dtype=torch.float64)
        vmax=torch.tensor([3.1416 if r['name'].endswith('_Pitch') else 1.99 for r in rows],device='cuda:0')
        amax=torch.tensor([15.708 if r['name'].endswith('_Pitch') else 1.99 for r in rows],device='cuda:0')
        for i,r in enumerate(rows):
            assert np.allclose([r['physical_min'],r['physical_max']],[lows[i].item(),highs[i].item()])
        assert not any(r['name'].endswith('_Tz') for r in env.scene.action_contract['channels'])
        with torch.no_grad():
            obs=env.reset()
            previous=env.scene.control_reset_action[full].double().expand(2,-1).clone()
            velocity=torch.zeros_like(previous)
            peaks=[torch.zeros_like(lows),torch.zeros_like(lows)];lo=torch.ones_like(lows);hi=-torch.ones_like(lows)
            # Explicit physical zero must survive the asymmetric affine mapping.
            zero=env.scene.control_reset_action.clone();zero[full]=0
            policy=env.scene.policy_action(zero)
            for r in rows: assert abs(policy[r['index']].item()-(2/3 if r['name'].endswith('_Pitch') else 1.))<1e-6
            for step in range(128):
                action=env.scene.policy_action(env.scene.control_reset_action).expand(2,-1).clone()
                for hand,r in enumerate(rows):
                    action[:,r['index']] = r['policy_max'] if (step//32+hand)%2 else r['policy_min']
                result=env.step(action)
                assert not env.scene.data.overflow[:].any()
                for x in (result.observation,result.reward,env.scene.data.qpos,env.scene.data.qvel): assert torch.isfinite(x).all()
                q=env.logic.last_action[:,full].double();v=(q-previous)/.02;a=(v-velocity)/.02
                assert (q>=lows-1e-7).all() and (q<=highs+1e-7).all()
                assert (v.abs()<=vmax+1e-7).all() and (a.abs()<=amax+1e-6).all()
                peaks=[torch.maximum(peaks[0],v.abs().amax(0)),torch.maximum(peaks[1],a.abs().amax(0))]
                lo=torch.minimum(lo,q.amin(0));hi=torch.maximum(hi,q.amax(0));previous=q.clone();velocity=v.clone()
            assert torch.allclose(lo,lows,atol=1e-6,rtol=0) and torch.allclose(hi,highs,atol=1e-6,rtol=0)
            assert torch.isfinite(env.reset([0])).all()
        report.update(passed=True,status='passed',stage='completed',tested_channels=[r["name"] for r in rows],target_observed_low=lo.tolist(),target_observed_high=hi.tolist(),
                      target_speed_max=peaks[0].tolist(),target_acceleration_max=peaks[1].tolist(),
                      action_size=env.action_size,observation_size=env.observation_size,
                      control_steps_per_world=128,physical_seconds_per_world=2.56,tz_active=False,source_tz_range=[-.6,.46])
    monkeypatch.setattr(harness,'exercise',exercise)
    harness.test_joint_capacity_physics_default()
