"""Opt-in Slurm joint capacity/physics acceptance from the actual default entry.

No formal training: real transitions seed a fully resident 48M CPU allocation;
only eight bounded learner blocks are executed. Never fills replay with fake
training experience, changes score time, or projects physical joint states.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def exercise(args, destination, report):
    import torch
    from tsensei_v3_2_pianist.rl_torch.train import effective_config, resolve_replay_prefill
    from tsensei_v3_2_pianist.mjlab_env.preview_contract import task_kwargs
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
    from tsensei_v3_2_pianist.rl_torch.host_memory import check_host_replay_memory
    from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig
    from tsensei_v3_2_pianist.rl_torch.collector_pipeline import CollectorPolicy, OneBatchPipeline
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
    helpers = runpy.run_path(str(ROOT/'tests/test_cpu_replay_scale.py'))
    host_memory, gpu_memory = helpers['_host_memory'], helpers['_gpu_memory']
    assert torch.cuda.device_count() == 2
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = effective_config(args)
    recipe = json.loads(args.source_config.read_text())
    assert (args.num_envs, args.replay_capacity, args.batch_size) == (4096, 48_000_000, 1024)
    assert (recipe['environment']['control_hz'], recipe['environment']['physics_hz']) == (50, 500)
    assert recipe['environment']['score_validation'] == 'legacy'
    assert recipe['environment']['exact_note_timing'] is True
    assert args.collector_pipeline and args.no_arms
    config['source_runtime'] = recipe
    print('当前验收奖励：'+json.dumps(config['reward_config'], ensure_ascii=False), flush=True)
    report.update(host_initial=host_memory(), gpu_initial=gpu_memory(torch), stage='building_4096_worlds')
    save(destination/'acceptance.json', report)
    with torch.cuda.device(0):
        env = PianoTaskEnv(args.scene, args.task, num_envs=4096, device='cuda:0', **task_kwargs(config))
    assert env.scene.decimation == 10 and env.logic.length == 11137
    resolve_replay_prefill(args, config, env.logic.length)
    config.update(observation_size=env.observation_size, action_size=env.action_size,
                  action_contract=env.scene.action_contract,
                  control_limits_contract=env.scene.control_limits,
                  preparation_contract=env.scene.bundle.preparation)
    save(destination/'config.json', config)
    # This is an audit of the current source, not a checkpoint compatibility test.
    channels = env.scene.action_contract['channels']
    roll = [row for row in channels if row['name'].endswith('_Palm_Base_Roll')]
    expected_roll = int(os.environ.get('PIANIST_EXPECT_ROLL', '2'))
    assert expected_roll in (0, 2) and len(roll) == expected_roll
    report['roll_enabled'] = bool(roll)
    for row in roll:
        assert (row['physical_min'], row['physical_max']) == (-.3, .5)
        limits = env.scene.control_limits['channels'][row['name']]
        assert (limits['max_velocity'], limits['max_acceleration']) == (3.1416, 15.708)
    if roll:
        import numpy as np
        model = env.scene.model
        drive_rows = []
        for side, prefix, sign in (('right', 'R', 1), ('left', 'L', -1)):
            # The fixed initial Yaw is baked into the root, including its axes.
            # Both local Roll axes must remain unit length; source-axis mirror
            # is checked separately in the CPU contract test.
            for axis in ('Pitch', 'Roll'):
                aid = model.actuator(f'{prefix}_Palm_Base_{axis}').id
                jid = int(model.actuator_trnid[aid, 0])
                assert np.allclose(model.actuator_forcerange[aid], [-18., 18.])
                assert bool(model.actuator_forcelimited[aid])
                assert np.isclose(model.actuator_gainprm[aid, 0], 8.)
                assert np.isclose(model.dof_damping[model.jnt_dofadr[jid]], .5)
                assert np.allclose(model.jnt_range[jid], [-.3, .5 if axis == 'Roll' else .6])
                drive_rows.append(dict(name=model.actuator(aid).name,
                    axis=model.jnt_axis[jid].tolist(), torque_nm=model.actuator_forcerange[aid].tolist()))
            assert f'{side}_/palm_base_roll' in env.bundle.manifest['hand_joints'][0 if side == 'right' else 1]
            assert np.allclose(model.joint(f'{side}_/palm_base_roll').axis, [0,0,sign])
        report['compiled_wrist_drives'] = drive_rows
        assert env.action_size == 33
        assert env.observation_size == 1410
    report.update(action_size=env.action_size, observation_size=env.observation_size,
                  roll_channels=roll, gpu_after_environment=gpu_memory(torch))
    memory_check = check_host_replay_memory(env.observation_size, env.action_size, args.replay_capacity)
    report['memory_admission'] = memory_check
    with torch.cuda.device(1):
        agent = SAC(env.observation_size, env.action_size, SACConfig(**config['sac']),
                    device='cuda:1', seed=args.seed, compile_learner=args.compile_learner,
                    fused_optimizer=args.fused_optimizer, action_profile=args.action_profile)
    collector = CollectorPolicy(agent, 'cuda:0')
    replay = Replay(env.observation_size, env.action_size, args.replay_capacity, 1024,
                    device='cpu', seed=args.seed, action_profile=args.action_profile)
    before = host_memory()
    report.update(stage='touching_all_replay_pages', replay_bytes=replay.bytes_allocated,
                  replay_gib=replay.bytes_allocated/GIB, host_before_touch=before)
    save(destination/'acceptance.json', report)
    # Touch every byte, but keep size=0. These zeros are storage capacity only
    # and are never sampled as synthetic transitions or counted as prefill.
    for name, storage in zip(Transition._fields, replay.data):
        storage.zero_()
        print('Touched full replay field: '+name, flush=True)
    touched = host_memory()
    assert touched['VmRSS'] >= before['VmRSS'] + .97*replay.bytes_allocated
    assert touched.get('VmSwap', 0) == 0
    assert replay.size == replay.total_inserted == 0
    report.update(host_full_pool=touched, gpu_full_pool=gpu_memory(torch), stage='real_physics',
                  full_pool_resident_with_4096_worlds=True, fake_transitions_inserted=0)
    save(destination/'acceptance.json', report)
    with torch.no_grad(), torch.cuda.device(0):
        obs = env.reset()
    rows = []
    peak_used = [0, 0]
    previous_target = env.scene.control_reset_action[None, :].expand(4096, -1).double().clone()
    previous_velocity = torch.zeros_like(previous_target)
    full_roll = [row['full_index'] for row in roll]
    actual_roll_qadr = [int(env.scene.model.jnt_qposadr[env.scene.model.joint(side+'_/palm_base_roll').id])
                       for side in (('right', 'left') if roll else ())]

    def collect(step, actor_version):
        nonlocal previous_target, previous_velocity
        with torch.no_grad(), torch.cuda.device(0):
            previous = obs.clone()
            action = collector.act(obs)
            # After the learner check, deliberately request both range endpoints.
            # Only target commands change; never overwrite qpos/qvel.
            if step > 8:
                for side, row in enumerate(roll):
                    sign = ((step-9)//32 + side) % 2
                    action[:, row['index']] = row['policy_max'] if sign else row['policy_min']
            started = time.perf_counter()
            result = env.step(action)
            torch.cuda.synchronize(0)
            assert not result.done.any()
            assert not env.scene.data.overflow[:].any()
            for value in (result.observation, result.reward, env.scene.data.qpos, env.scene.data.qvel):
                assert bool(torch.isfinite(value).all())
            info = result.info
            assert bool((info['substep_count'] == 10).all())
            assert info['goal_substeps'].shape == (4096, 10, 89)
            assert bool((info['score_index'] == step).all())
            assert torch.equal(info['goal_substeps'], env.exact_score.goal_history[info['score_index']])
            for group in ('reward_raw_terms', 'reward_weighted_terms'):
                assert info[group] and all(bool(torch.isfinite(v).all()) for v in info[group].values())
            applied = env.logic.last_action.double()
            velocity = (applied-previous_target)/.02
            acceleration = (velocity-previous_velocity)/.02
            assert bool(((applied[:, full_roll] >= -.3-1e-7) & (applied[:, full_roll] <= .5+1e-7)).all())
            assert bool((velocity[:, full_roll].abs() <= 3.1416+1e-5).all())
            assert bool((acceleration[:, full_roll].abs() <= 15.708+1e-4).all())
            previous_target, previous_velocity = applied.clone(), velocity.clone()
            memory = gpu_memory(torch)
            for device in range(2):
                item = memory[str(device)]
                peak_used[device] = max(peak_used[device], item['total_bytes']-item['free_bytes'])
            row = dict(step=step, seconds=time.perf_counter()-started, actor_version=actor_version,
                reward_mean=result.reward.mean().item(),
                roll_target_velocity_max=velocity[:, full_roll].abs().max().item() if roll else None,
                roll_target_acceleration_max=acceleration[:, full_roll].abs().max().item() if roll else None,
                actual_roll_abs_max=env.scene.data.qpos[:, actual_roll_qadr].abs().max().item() if roll else None,
                actual_roll_range=[env.scene.data.qpos[:, actual_roll_qadr].min().item(),
                                   env.scene.data.qpos[:, actual_roll_qadr].max().item()] if roll else None,
                self_contact_substeps=info['self_collision_substeps'].sum().item(),
                cross_hand_contact_substeps=info['inter_hand_collision_substeps'].sum().item())
            return Transition(previous, action, result.reward, result.discount,
                              result.observation, result.terminated, result.truncated), row

    def commit(batch):
        nonlocal obs
        transition, row = batch
        first = replay.total_inserted
        replay.insert(transition)
        assert all(torch.equal(storage[first:first+4096], value.cpu())
                   for storage, value in zip(replay.data, transition))
        obs = transition.next_state
        rows.append(row)
        if row['step'] % 16 == 0:
            report.update(steps=rows, gpu_sampled_peak_used_bytes=peak_used)
            save(destination/'acceptance.json', report)
            print('Physical control step '+str(row['step']), flush=True)

    commit(collect(0, 0))
    with torch.cuda.device(1):
        learner = CUDALearner(agent, replay, optimizer_layout=True, matmul_precision='tf32')
    assert not torch.backends.cuda.matmul.allow_tf32
    report['staging_contract'] = learner.staging_contract
    def update(step):
        with torch.cuda.device(1):
            metrics = learner.update(1024)
            assert all(bool(torch.isfinite(v).all()) for v in metrics.values())
    with OneBatchPipeline(snapshot=lambda: collector.sync_from(agent), collect=collect,
                          commit=commit, update=update, learner_version=lambda: agent.updates) as pipeline:
        for step in range(1, 9):
            pipeline.step(step)
        pipeline.flush()
        report['pipeline_state'] = pipeline.state()
    assert agent.updates == 8192 and report['pipeline_state']['pending_vector_step'] is None
    for step in range(9, 257):
        commit(collect(step, collector.version))
    with torch.no_grad(), torch.cuda.device(0):
        ids = torch.tensor([0, 2048, 4095], device='cuda:0')
        saved = replay.data.next_state[256*4096+ids.cpu()].clone()
        indexes = env.logic.index.clone()
        reset = env.reset(ids)
        assert bool(torch.isfinite(reset).all())
        indexes[ids] = 0
        assert torch.equal(indexes, env.logic.index)
        assert torch.equal(saved, replay.data.next_state[256*4096+ids.cpu()])
    assert replay.total_inserted == 257*4096
    assert all(value.device.type == 'cpu' for value in replay.data)
    batch = replay.sample(device='cuda:1')
    assert all(bool(torch.isfinite(value).all()) for value in batch)
    report.update(stage='completed', status='passed', passed=True, steps=rows,
                  physics_control_steps=257, seconds_per_world=257/50,
                  total_physics_steps=257*10*4096, learner_updates=agent.updates,
                  replay_insertions=replay.total_inserted, partial_reset_passed=True,
                  host_final=host_memory(), gpu_final=gpu_memory(torch),
                  gpu_sampled_peak_used_bytes=peak_used, full_pool_still_allocated=replay.bytes_allocated)


def test_joint_capacity_physics_default():
    import pytest
    if 'PIANIST_ROLL_ACCEPTANCE_DIR' not in os.environ:
        pytest.skip('Explicit Slurm capacity/physics acceptance only')
    assert os.environ.get('SLURM_JOB_ID')
    destination = Path(os.environ['PIANIST_ROLL_ACCEPTANCE_DIR']).resolve()
    assert not (destination/'acceptance.json').exists(), 'Use a fresh acceptance directory'
    expected = (destination/'source_archive_sha256.txt').read_text().split()[0]
    assert os.environ['TSENSEI_JOB_SOURCE_SHA256'] == expected
    command = json.loads((destination/'launch_command.json').read_text())
    original_root = str(Path(command[0]).parents[2])
    arguments = [arg.replace(original_root, str(ROOT)) for arg in command[command.index('train')+1:]]
    # Preserve all default settings; inspect-only avoids creating a formal run.
    arguments[arguments.index('--midi')+1] = str(destination/'original_score.proto')
    assert hashlib.sha256((destination/'original_score.proto').read_bytes()).hexdigest() == '66563e8b5e338c1ef3e426233647da2a008ac931939775b1ee87c36857339af0'
    arguments += ['--inspect-only']
    report = dict(status='running', passed=False, formal_training=False,
                  job_id=os.environ['SLURM_JOB_ID'], source_sha256=expected,
                  host=os.uname().nodename, pid=os.getpid(),
                  scope='257 real control steps/world; eight diagnostic learner blocks; fully resident 48M CPU replay')
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT/'scripts'))
    import swanlab_upload
    original = swanlab_upload.run_with_upload
    def intercept(command, **kwargs):
        from tsensei_v3_2_pianist.rl_torch.train import parser
        args = parser().parse_args(command[command.index('tsensei_v3_2_pianist.rl_torch.train')+1:])
        exercise(args, destination, report)
    swanlab_upload.run_with_upload = intercept
    try:
        runpy.run_path(str(ROOT/'scripts/train_from_source.py'))['main'](arguments)
    except BaseException as error:
        report.update(status='failed', passed=False, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        swanlab_upload.run_with_upload = original
        report['elapsed_seconds'] = time.perf_counter()-started
        save(destination/'acceptance.json', report)
