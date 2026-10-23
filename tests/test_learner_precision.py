"""Learner TF32 is captured once, never a process-wide inference default."""
import pytest
import torch

from tsensei_v3_2_pianist.rl_torch.learner_graph import CUDALearner
from tsensei_v3_2_pianist.rl_torch.replay import Replay, Transition
from tsensei_v3_2_pianist.rl_torch.sac import SAC, SACConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Slurm CUDA validation")


def objects():
    agent = SAC(1406, 31, SACConfig(actor_hidden=(256, 256, 256),
                                  critic_hidden=(256, 256, 256),
                                  critic_lr=.0003, temp_lr=.0003, gamma=.8,
                                  init_temperature=.5),
                seed=42, compile_learner=True, fused_optimizer=True,
                action_profile="direct_joint_v1")
    replay = Replay(1406, 31, 2048, 1024, seed=123,
                    action_profile="direct_joint_v1")
    g = torch.Generator(device="cuda").manual_seed(32)
    replay.insert(Transition(
        torch.randn(2048, 1406, device="cuda", generator=g)*.2,
        replay.action_low + torch.rand(2048, 31, device="cuda", generator=g) *
        (replay.action_high-replay.action_low),
        torch.rand(2048, device="cuda", generator=g)-3,
        torch.ones(2048, device="cuda"),
        torch.randn(2048, 1406, device="cuda", generator=g)*.2,
        torch.zeros(2048, dtype=torch.bool, device="cuda"),
        torch.zeros(2048, dtype=torch.bool, device="cuda")))
    return agent, replay


def test_capture_precision_is_restored_and_replay_is_independent(monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    first, replay1 = objects()
    second, replay2 = objects()
    observed = []
    original = CUDALearner._capture

    def capture(self):
        observed.append(torch.backends.cuda.matmul.allow_tf32)
        original(self)

    monkeypatch.setattr(CUDALearner, "_capture", capture)
    one = CUDALearner(first, replay1, optimizer_layout=True, matmul_precision="tf32")
    assert not torch.backends.cuda.matmul.allow_tf32
    two = CUDALearner(second, replay2, optimizer_layout=True, matmul_precision="tf32")
    assert observed == [True, True]
    assert not torch.backends.cuda.matmul.allow_tf32
    one.update(128)
    # Once captured, global inference precision must not change learner math.
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        two.update(128)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    for name in ("actor", "critic", "target_critic"):
        for p, q in zip(getattr(first, name).parameters(), getattr(second, name).parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert torch.equal(first.update_rng.get_state(), second.update_rng.get_state())
    assert first.updates == second.updates == 128


def test_capture_failure_restores_inference_precision(monkeypatch):
    torch.backends.cuda.matmul.allow_tf32 = False
    agent, replay = objects()

    def broken(self):
        assert torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("intentional capture failure")

    monkeypatch.setattr(CUDALearner, "_capture", broken)
    with pytest.raises(RuntimeError, match="intentional"):
        CUDALearner(agent, replay, matmul_precision="tf32")
    assert not torch.backends.cuda.matmul.allow_tf32
