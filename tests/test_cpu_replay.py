"""CPU replay ownership and CUDA staging preserve the SAC update contract.

GPU cases run only inside the project's remote Slurm validation allocation.
CPU sampling is compared with CPU sampling, never with CUDA's different RNG.
"""
from copy import deepcopy

import pytest
import torch

from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig


def rows(count, offset=0, *, state_size=13, action_size=47):
    generator = torch.Generator().manual_seed(102 + offset)
    return Transition(
        torch.randn((count, state_size), generator=generator) * .2,
        torch.rand((count, action_size), generator=generator) * 2 - 1,
        torch.arange(offset, offset + count, dtype=torch.float32) * .01,
        torch.ones(count),
        torch.randn((count, state_size), generator=generator) * .2,
        torch.zeros(count, dtype=torch.bool),
        torch.arange(count).remainder(3) == 0)


def cpu_replay(*, capacity=128, batch_size=8, seed=46, fill=32):
    replay = Replay(13, 47, capacity, batch_size, device='cpu', seed=seed)
    if fill:
        replay.insert(rows(fill))
    return replay


def test_cpu_pool_remains_pageable_and_accounts_all_storage():
    replay = cpu_replay()
    assert all(value.device.type == 'cpu' and not value.is_pinned() for value in replay.data)
    assert replay.bytes_allocated == 128 * (4 * (13 * 2 + 47 + 2) + 2)
    assert replay.storage_contract == dict(
        device='cpu', storage_dtype='float32_with_bool_end_flags',
        bytes_allocated=replay.bytes_allocated, pool_pinned=False,
        sampling='uniform_with_replacement', sampling_rng_device='cpu',
        sampling_rng_backend='pytorch_cpu', cross_device_rng_bitwise_equivalent=False)


def test_cpu_sampling_matches_uniform_indices_and_owns_its_data():
    replay = cpu_replay(capacity=16, batch_size=16, fill=16)
    expected_rng = torch.Generator().set_state(replay.generator.get_state())
    expected_indices = torch.randint(16, (16,), generator=expected_rng)
    expected = Transition(*(value[expected_indices].clone() for value in replay.data))
    actual = replay.sample(device='cpu')
    assert torch.equal(replay.generator.get_state(), expected_rng.get_state())
    assert expected_indices.unique().numel() < 16  # Sampling is with replacement.
    for one, two in zip(expected, actual):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    replay.insert(rows(25, offset=100))
    for one, two in zip(expected, actual):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert replay.size == 16 and replay.total_inserted == 41 and replay.pointer == 9
    # Terminal next observations and flags survive the circular overwrite.
    source = rows(25, offset=100)
    retained = torch.arange(9, 25)
    positions = (retained + 16) % 16
    for stored, incoming in zip(replay.data, source):
        torch.testing.assert_close(stored[positions], incoming[retained], rtol=0, atol=0)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote Slurm CUDA validation')


@cuda
def test_cpu_sample_can_feed_cuda_without_a_graph():
    first = cpu_replay()
    second = cpu_replay()
    actual = first.sample(device='cuda:0')
    expected = second.sample()
    assert all(value.device == torch.device('cuda:0') for value in actual)
    for one, two in zip(actual, expected):
        torch.testing.assert_close(one.cpu(), two, rtol=0, atol=0)


def agents():
    config = SACConfig(actor_hidden=(16, 16, 16), critic_hidden=(16, 16, 16),
                       actor_lr=3e-4, critic_lr=3e-4, temp_lr=3e-4,
                       gamma=.8, init_temperature=1.)
    return [SAC(13, 47, config, seed=41, compile_learner=True, fused_optimizer=True)
            for _ in range(2)]


def compare_agents(first, second, *, atol=2e-6):
    for name in ('actor', 'critic', 'target_critic'):
        for one, two in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
            torch.testing.assert_close(one, two, rtol=1e-6, atol=atol)
    torch.testing.assert_close(first.log_temperature, second.log_temperature, rtol=0, atol=atol)
    assert first.updates == second.updates
    assert torch.equal(first.action_rng.get_state(), second.action_rng.get_state())
    assert torch.equal(first.update_rng.get_state(), second.update_rng.get_state())
    for name in ('actor_optimizer', 'critic_optimizer', 'temp_optimizer'):
        for one, two in zip(getattr(first, name).state.values(), getattr(second, name).state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                tolerance = 0 if key == 'step' else atol
                torch.testing.assert_close(one[key], two[key], rtol=1e-6 if tolerance else 0,
                                           atol=tolerance)


@cuda
@pytest.mark.parametrize('optimizer_layout', [False, True])
def test_cpu_graph_matches_eager_sampling_after_capture_growth_and_wrap(optimizer_layout, monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eager, graphed = agents()
    eager_replay, replay = cpu_replay(), cpu_replay()
    original = deepcopy(graphed.state_dict())
    replay_rng = replay.generator.get_state().clone()
    graph = CUDALearner(graphed, replay, optimizer_layout=optimizer_layout)
    assert graphed.updates == 0
    assert torch.equal(graphed.update_rng.get_state(), original['update_rng'])
    assert torch.equal(replay.generator.get_state(), replay_rng)
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in getattr(graphed, name).state_dict().items():
            assert torch.equal(value, original[name][key])
    assert all(state['step'].item() == 0 for optimizer in
               (graphed.actor_optimizer, graphed.critic_optimizer, graphed.temp_optimizer)
               for state in optimizer.state.values())
    assert all(value.is_pinned() for batch in graph._host_batches for value in batch)
    assert all(not value.is_pinned() for value in replay.data)
    minibatch_bytes = (4 * (13 * 2 + 47 + 2) + 2) * 8
    assert graph.staging_contract['host_pinned_bytes'] == 2 * minibatch_bytes
    assert graph.staging_contract['device_batch_bytes'] == minibatch_bytes
    addresses = graph._staging_storage()
    # More than two updates exercise both host slots repeatedly, including on
    # a nondefault stream. A smaller ready size also tests the randint bound.
    stream = torch.cuda.Stream()
    for count, additional in ((1, 0), (7, 17), (17, 113)):
        if additional:
            incoming = rows(additional, offset=additional)
            replay.insert(incoming)
            eager_replay.insert(incoming)
        for _ in range(count):
            expected = eager.update(eager_replay.sample(device=eager.device))
        def reject_device_sync(*args, **kwargs):
            raise AssertionError('Per-update CPU staging must not globally synchronize CUDA')
        with monkeypatch.context() as patch, torch.cuda.stream(stream):
            patch.setattr(torch.cuda, 'synchronize', reject_device_sync)
            actual = graph.update(count)
        torch.cuda.current_stream().wait_stream(stream)
        compare_agents(eager, graphed)
        assert torch.equal(replay.generator.get_state(), eager_replay.generator.get_state())
        assert graph._staging_storage() == addresses
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=1e-4)


@cuda
def test_cpu_staging_bounds_do_not_scale_with_replay_capacity():
    # No physics is initialized; this is the actual 1406/31 batch payload only.
    small = Replay(1406, 31, 1024, 1024, device='cpu', action_profile='direct_joint_v1')
    large = Replay(1406, 31, 3072, 1024, device='cpu', action_profile='direct_joint_v1')
    config = SACConfig(actor_hidden=(16,), critic_hidden=(16,), critic_dropout=0.)
    for replay in (small, large):
        transition = rows(1024, state_size=1406, action_size=31)
        transition = transition._replace(action=(transition.action + 1) * .5
            * (replay.action_high - replay.action_low) + replay.action_low)
        replay.insert(transition)
        agent = SAC(1406, 31, config, device='cuda:0', fused_optimizer=True,
                    action_profile='direct_joint_v1')
        graph = CUDALearner(agent, replay)
        assert graph.staging_contract['host_pinned_bytes'] == 23_310_336
        assert graph.staging_contract['device_batch_bytes'] == 11_655_168
        assert not replay.storage_contract['pool_pinned']
        graph.update(3)


@cuda
def test_cpu_graph_rejects_replaced_storage_and_preserves_failure():
    _, agent = agents()
    replay = cpu_replay()
    graph = CUDALearner(agent, replay)
    original = graph._host_batches[0]
    graph._host_batches[0] = Transition(*(value.clone() for value in original))
    with pytest.raises(ValueError, match='storage changed'):
        graph.update(1)
    graph._host_batches[0] = original
    replay.data.reward.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='Nonfinite SAC update'):
        graph.update(5)
    replay.data.reward.zero_()
    with pytest.raises(FloatingPointError, match='previously failed'):
        graph.update(1)


@cuda
def test_cpu_graph_preserves_nonempty_optimizer_state_on_capture():
    first, second = agents()
    first_replay, second_replay = cpu_replay(), cpu_replay()
    for _ in range(3):
        first.update(first_replay.sample(device=first.device))
    second.load_state_dict(deepcopy(first.state_dict()))
    second_replay.generator.set_state(first_replay.generator.get_state())
    graph = CUDALearner(second, second_replay, optimizer_layout=True)
    compare_agents(first, second, atol=0)
    for _ in range(9):
        first.update(first_replay.sample(device=first.device))
    graph.update(9)
    compare_agents(first, second)
