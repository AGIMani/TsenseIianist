"""Ordering, bounded collection, update accounting and failure propagation."""
import threading

import pytest
import torch

from tsensei_v3_2_pianist.rl_torch.collector_pipeline import MultiBatchPipeline, OneBatchPipeline, execution_layout
from tsensei_v3_2_pianist.rl_torch.train import effective_config, parser, updates_this_step


def test_collection_and_learning_overlap_but_replay_writes_do_not():
    collecting_second = threading.Event()
    learner_running = threading.Event()
    replay = []
    versions = []
    updates = []
    caller = threading.get_ident()

    def collect(step, version):
        assert threading.get_ident() != caller
        versions.append(version)
        if step == 2:
            collecting_second.set()
            assert learner_running.wait(3), 'learner did not overlap collection'
        return step

    def update(step):
        assert threading.get_ident() == caller
        assert replay == list(range(1, step + 1)), 'future transition entered the update replay'
        if step == 1:
            assert collecting_second.wait(3), 'collection did not run concurrently'
            learner_running.set()
        updates.append(step)

    def commit(step):
        assert threading.get_ident() == caller
        assert updates == list(range(1, step)), 'commit preceded the prior update'
        replay.append(step)

    with OneBatchPipeline(snapshot=lambda: len(updates), collect=collect, update=update,
                          commit=commit, learner_version=lambda: len(updates)) as pipe:
        pipe.step(1)
        pipe.step(2)
        assert pipe.pending == 2
        assert pipe.updated_through_step == 1
        pipe.flush()
        pipe.flush()
        assert updates == [1, 2]
        assert pipe.state()['host_overlap_seconds'] > 0
        assert pipe.state()['max_actor_lag_at_collection_end_updates'] == 1
    assert replay == [1, 2]
    assert versions == [0, 0]
    assert pipe.closed


def test_flush_at_evaluation_save_and_early_stop_does_not_duplicate_updates():
    replay = []
    learned = []
    versions = []
    with OneBatchPipeline(snapshot=lambda: len(learned),
                          collect=lambda step, version: versions.append(version) or step,
                          update=learned.append, commit=replay.append) as pipe:
        pipe.step(1)
        pipe.step(2)
        pipe.flush()  # An evaluation/checkpoint boundary.
        pipe.flush()  # Saving again is idempotent.
        pipe.step(3)
        pipe.flush()  # Early stop must not prefetch step 4.
        assert pipe.state()['pending_vector_step'] is None
        assert pipe.state()['learner_updated_through_vector_step'] == 3
    assert replay == learned == [1, 2, 3]
    assert versions == [0, 0, 2]


def test_fractional_budget_and_replay_readiness_are_unchanged():
    args = parser().parse_args(['--num-envs', '3', '--batch-size', '5',
                               '--learning-start-transitions', '1', '--updates-per-transition', '.1'])
    effective_config(args)
    insertions = 0
    updates = 0
    visited = []

    def commit(step):
        nonlocal insertions
        insertions += args.num_envs

    def learn(step):
        nonlocal updates
        assert insertions == step * args.num_envs
        if insertions >= args.batch_size:
            updates += updates_this_step(args, step)
        visited.append(step)

    with OneBatchPipeline(snapshot=lambda: updates, collect=lambda step, version: step,
                          update=learn, commit=commit) as pipe:
        for step in range(1, 101):
            pipe.step(step)
            if step % 7 == 0:
                pipe.flush()
    expected = sum(updates_this_step(args, step) for step in range(1, 101)
                   if step * args.num_envs >= args.batch_size)
    assert updates == expected == 30
    assert insertions == 300
    assert visited == list(range(1, 101))


def test_collector_error_propagates_without_committing_bad_batch():
    replay, learned = [], []

    def collect(step, version):
        if step == 2:
            raise FloatingPointError('nonfinite physics')
        return step

    with pytest.raises(FloatingPointError, match='nonfinite physics'):
        with OneBatchPipeline(snapshot=lambda: 0, collect=collect, update=learned.append,
                              commit=replay.append) as pipe:
            pipe.step(1)
            pipe.step(2)
    assert replay == learned == [1]
    assert pipe.failed and pipe.closed
    with pytest.raises(RuntimeError, match='closed or failed'):
        pipe.step(2)


def test_learner_error_joins_worker_and_never_commits_or_retries():
    second_started = threading.Event()
    learner_failed = threading.Event()
    worker_finished = threading.Event()
    replay, attempts = [], []

    def collect(step, version):
        if step == 2:
            second_started.set()
            assert learner_failed.wait(3)
            worker_finished.set()
        return step

    def learn(step):
        attempts.append(step)
        assert second_started.wait(3)
        learner_failed.set()
        raise FloatingPointError('nonfinite learner')

    with pytest.raises(FloatingPointError, match='nonfinite learner'):
        with OneBatchPipeline(snapshot=lambda: 0, collect=collect, update=learn,
                              commit=replay.append) as pipe:
            pipe.step(1)
            pipe.step(2)
    assert worker_finished.is_set()
    assert replay == attempts == [1]
    assert pipe.closed and pipe.failed


def test_pending_final_block_failure_is_not_swallowed():
    def learn(step):
        raise RuntimeError('final flush failed')
    with pytest.raises(RuntimeError, match='final flush failed'):
        with OneBatchPipeline(snapshot=lambda: 0, collect=lambda step, version: step,
                              update=learn, commit=lambda batch: None) as pipe:
            pipe.step(1)
    assert pipe.closed and pipe.failed


def test_nonconsecutive_steps_are_rejected_before_collection():
    with OneBatchPipeline(snapshot=lambda: 0, collect=lambda step, version: step,
                          update=lambda step: None, commit=lambda batch: None) as pipe:
        with pytest.raises(ValueError, match='consecutive'):
            pipe.step(2)
        assert pipe.collected_steps == 0
        pipe.step(1)
        with pytest.raises(ValueError, match='consecutive'):
            pipe.step(1)


def test_cli_layout_preserves_learning_budget_and_opt_in_semantics():
    base_args = parser().parse_args([])
    split_args = parser().parse_args(['--learner-device', 'cuda:1', '--collector-pipeline'])
    base, split = effective_config(base_args), effective_config(split_args)
    assert base.pop('execution_layout')['mode'] == 'single_device_v1'
    layout = split.pop('execution_layout')
    assert base == split
    assert layout['max_pending_vector_batches'] == layout['max_actor_lag_vector_batches'] == 1
    assert layout['sampling_order_changed']
    assert execution_layout('cuda', 'cuda:1')['mode'] == 'synchronous_split_v1'
    assert execution_layout('cuda')['collector_device'] == 'cuda:0'
    for collector, learner, pipeline in [('cuda', 'cuda:0', True), ('cpu', 'cuda:1', False),
                                         ('cuda:0', 'cpu', False)]:
        with pytest.raises(ValueError):
            execution_layout(collector, learner, pipeline)


def test_commit_failure_prevents_learning_on_uncommitted_batch():
    learned = []
    def commit(batch):
        raise ValueError('invalid replay transition')
    with pytest.raises(ValueError, match='invalid replay transition'):
        with OneBatchPipeline(snapshot=lambda: 0, collect=lambda step, version: step,
                              update=learned.append, commit=commit) as pipe:
            pipe.step(1)
    assert not learned
    assert pipe.failed and pipe.closed


def test_multi_collector_pipeline_preserves_slice_order_and_update_barrier():
    replay = []
    learned = []
    collected = []

    def collect(step, version, index):
        collected.append((step, index, version))
        return f'{step}:{index}'

    def update(step):
        assert replay == list(range(1, step + 1))
        learned.append(step)

    def commit(batch):
        assert batch == [f'{len(replay) + 1}:0', f'{len(replay) + 1}:1', f'{len(replay) + 1}:2']
        replay.append(len(replay) + 1)

    with MultiBatchPipeline(collector_count=3,
                            collector_devices=['cuda:0', 'cuda:1', 'cuda:2'],
                            snapshot=lambda: tuple(0 for _ in range(3)), collect=collect,
                            update=update, commit=commit,
                            learner_version=lambda: len(learned)) as pipe:
        pipe.step(1)
        pipe.step(2)
        assert pipe.pending == 2
        assert pipe.updated_through_step == 1
        pipe.flush()

    assert replay == learned == [1, 2]
    assert sorted(collected) == [(1, 0, (0, 0, 0)), (1, 1, (0, 0, 0)), (1, 2, (0, 0, 0)),
                                 (2, 0, (0, 0, 0)), (2, 1, (0, 0, 0)), (2, 2, (0, 0, 0))]
    state = pipe.state()
    assert state['collector_count'] == 3
    assert state['collector_devices'] == ['cuda:0', 'cuda:1', 'cuda:2']
    assert state['max_actor_lag_at_collection_end_updates'] == 1


def test_multi_collector_layout_requires_pipeline_and_distinct_learner():
    layout = execution_layout('cuda:0', 'cuda:3', True,
                               collector_devices='cuda:0,cuda:1,cuda:2')
    assert layout['mode'] == 'multi_vector_pipeline_v1'
    assert layout['collector_count'] == 3
    assert layout['collector_devices'] == ['cuda:0', 'cuda:1', 'cuda:2']
    with pytest.raises(ValueError, match='require --collector-pipeline'):
        execution_layout('cuda:0', 'cuda:3', False,
                         collector_devices='cuda:0,cuda:1,cuda:2')
    with pytest.raises(ValueError, match='distinct learner'):
        execution_layout('cuda:0', 'cuda:1', True,
                         collector_devices='cuda:0,cuda:1,cuda:2')


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires two actual CUDA devices')
@pytest.mark.parametrize('use_graph', [False, True])
def test_two_cuda_devices_transfer_actor_rng_replay_and_update(use_graph):
    from tsensei_v3_2_pianist.rl_torch.collector_pipeline import CollectorPolicy
    from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
    from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
    from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig

    with torch.cuda.device(1):
        agent = SAC(13, 47, SACConfig(actor_hidden=(16,), critic_hidden=(16,)),
                    device='cuda:1', fused_optimizer=True)
        collector = CollectorPolicy(agent, 'cuda:0')
        replay = Replay(13, 47, 64, 8, device='cuda:1')
        graph = None

        def collect(step, version):
            with torch.cuda.device(0):
                obs = torch.full((8, 13), step * .01, device='cuda:0')
                action = collector.act(obs)
                result = Transition(obs, action, torch.zeros(8, device='cuda:0'),
                                    torch.ones(8, device='cuda:0'), obs + .01,
                                    torch.zeros(8, device='cuda:0', dtype=torch.bool),
                                    torch.zeros(8, device='cuda:0', dtype=torch.bool))
                torch.cuda.synchronize(0)
                return result

        def learn(step):
            nonlocal graph
            assert replay.total_inserted == step * 8
            if use_graph:
                if graph is None:
                    graph = CUDALearner(agent, replay)
                graph.update(2)
            else:
                for _ in range(2):
                    agent.update(replay.sample())

        with OneBatchPipeline(snapshot=lambda: collector.sync_from(agent), collect=collect,
                              update=learn, commit=replay.insert,
                              learner_version=lambda: agent.updates) as pipe:
            pipe.step(1)
            pipe.flush()  # Complete first CUDA graph capture without worker CUDA activity.
            for step in range(2, 5):
                pipe.step(step)
        assert agent.updates == 8
        assert replay.total_inserted == 32
        assert pipe.max_actor_lag_updates == 2
        collector.sync_from(agent)
        for first, second in zip(agent.actor.parameters(), collector.actor.parameters()):
            torch.testing.assert_close(first.cpu(), second.cpu(), rtol=0, atol=0)
        collector.copy_action_rng_to(agent)
        assert torch.equal(collector.action_rng.get_state(), agent.action_rng.get_state())
