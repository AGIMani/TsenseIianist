"""CUDA execution must preserve SAC budgets, RNG, state and error handling."""
from copy import deepcopy

import pytest
import torch

from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph validation')


def objects(*, seed=41, fill=32):
    config = SACConfig(actor_hidden=(16, 16, 16), critic_hidden=(16, 16, 16),
                       actor_lr=3e-4, critic_lr=3e-4, temp_lr=3e-4,
                       gamma=.8, init_temperature=1.)
    agent = SAC(13, 47, config, seed=seed, compile_learner=True,
                fused_optimizer=True, action_profile='legacy')
    replay = Replay(13, 47, 128, 8, seed=seed+5, action_profile='legacy')
    if fill:
        replay.insert(batch(fill))
    return agent, replay


def batch(n, offset=0):
    generator = torch.Generator(device='cuda').manual_seed(102+offset)
    return Transition(torch.randn((n, 13), device='cuda', generator=generator)*.2,
                      torch.rand((n, 47), device='cuda', generator=generator)*2-1,
                      torch.rand(n, device='cuda', generator=generator),
                      torch.ones(n, device='cuda'),
                      torch.randn((n, 13), device='cuda', generator=generator)*.2,
                      torch.zeros(n, device='cuda', dtype=torch.bool),
                      torch.zeros(n, device='cuda', dtype=torch.bool))


def compare_agents(first, second, *, atol=0):
    for name in ('actor', 'critic', 'target_critic'):
        for p, q in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
            torch.testing.assert_close(p, q, rtol=1e-6 if atol else 0, atol=atol)
    torch.testing.assert_close(first.log_temperature, second.log_temperature, rtol=0, atol=atol)
    assert first.updates == second.updates
    assert torch.equal(first.action_rng.get_state(), second.action_rng.get_state())
    assert torch.equal(first.update_rng.get_state(), second.update_rng.get_state())
    for name in ('actor_optimizer', 'critic_optimizer', 'temp_optimizer'):
        one, two = getattr(first, name), getattr(second, name)
        for p, q in zip(one.state.values(), two.state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                # A compiled optimizer can differ by roundoff in moments. Counts
                # remain exactly equal, including a warm restart's prior steps.
                tol = 0 if key == 'step' else atol
                torch.testing.assert_close(p[key], q[key], rtol=1e-6 if tol else 0, atol=tol)


@pytest.mark.parametrize('optimizer_layout', [False, True])
def test_capture_preserves_rng_and_dynamic_replay_sequence(optimizer_layout):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eager, old_replay = objects()
    graphed, replay = objects()
    original = deepcopy(graphed.state_dict())
    replay_rng = replay.generator.get_state().clone()
    graph = CUDALearner(graphed, replay, optimizer_layout=optimizer_layout)
    assert graphed.updates == 0
    assert torch.equal(graphed.update_rng.get_state(), original['update_rng'])
    assert torch.equal(graphed.action_rng.get_state(), original['action_rng'])
    assert torch.equal(replay.generator.get_state(), replay_rng)
    for name in ('actor', 'critic', 'target_critic'):
        for key, value in getattr(graphed, name).state_dict().items():
            assert torch.equal(value, original[name][key])
    assert all(value['step'].item() == 0 for optimizer in
               (graphed.actor_optimizer, graphed.critic_optimizer, graphed.temp_optimizer)
               for value in optimizer.state.values())
    for count, additional in ((1, 0), (3, 17), (5, 113)):
        if additional:
            incoming = batch(additional, offset=additional)
            replay.insert(incoming)
            old_replay.insert(incoming)
        for _ in range(count):
            expected = eager.update(old_replay.sample())
        actual = graph.update(count)
        compare_agents(eager, graphed, atol=2e-6 if optimizer_layout else 0)
        assert torch.equal(replay.generator.get_state(), old_replay.generator.get_state())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-6 if optimizer_layout else 0,
                                       atol=1e-4 if optimizer_layout else 0)


@pytest.mark.parametrize('optimizer_layout', [False, True])
def test_warm_restart_nonempty_adam_state_and_checkpoint_continuation(optimizer_layout):
    trained, replay0 = objects()
    for _ in range(4):
        trained.update(replay0.sample())
    saved = deepcopy(trained.state_dict())
    first, replay1 = objects()
    second, replay2 = objects()
    # Separate deserializations own separate Adam tensors. Passing the very same
    # CUDA state dict twice aliases optimizer tensors in PyTorch load_state_dict.
    first.load_state_dict(deepcopy(saved))
    second.load_state_dict(deepcopy(saved))
    graph = CUDALearner(second, replay2, optimizer_layout=optimizer_layout)
    compare_agents(first, second)
    for _ in range(7):
        first.update(replay1.sample())
    graph.update(7)
    compare_agents(first, second, atol=2e-6 if optimizer_layout else 0)
    # Graph-produced state remains loadable by the ordinary eager SAC path.
    restored, _ = objects()
    restored.load_state_dict(deepcopy(second.state_dict()))
    compare_agents(second, restored)
    incoming = replay1.sample()
    second.update(incoming)
    restored.update(incoming)
    compare_agents(second, restored)


def test_invalid_counts_nonready_and_changed_adam_storage_are_rejected():
    agent, replay = objects(fill=0)
    with pytest.raises(ValueError, match='not ready'):
        CUDALearner(agent, replay)
    replay.insert(batch(32))
    graph = CUDALearner(agent, replay)
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            graph.update(count)
    saved_size = replay.size
    replay.size = 0
    with pytest.raises(ValueError, match='not ready'):
        graph.update(1)
    replay.size = saved_size
    agent.load_state_dict(deepcopy(agent.state_dict()))
    with pytest.raises(ValueError, match='storage changed'):
        graph.update(1)


@pytest.mark.parametrize('optimizer_layout', [False, True])
def test_nonfinite_any_update_fails_before_returning(optimizer_layout):
    agent, replay = objects()
    graph = CUDALearner(agent, replay, optimizer_layout=optimizer_layout)
    replay.data.reward.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='Nonfinite SAC update'):
        graph.update(3)
    # It is forbidden to continue using poisoned parameters, even if the caller
    # later repairs the replay input. Finite history is checked inside every step.
    replay.data.reward.fill_(0.)
    with pytest.raises(FloatingPointError, match='previously failed'):
        graph.update(1)


def test_finite_accumulator_remembers_earlier_failure():
    agent, replay = objects()
    flag = torch.tensor(False, device='cuda')
    agent.update(replay.sample(), _finite_accumulator=flag)
    assert not flag.item()
