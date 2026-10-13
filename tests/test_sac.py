"""SAC parity, transition boundaries and adapter integration.

Reference data were generated from the original Shadow trainer, not this module.
The train extra is optional; CPU environment-only installations may skip these.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("dm_env_wrappers")
from dm_env import specs
from tsensei_v3_2_pianist.rl import checkpoint
from tsensei_v3_2_pianist.rl.environment import EnvironmentConfig, make_environment
from tsensei_v3_2_pianist.rl.replay import Buffer, Transition
from tsensei_v3_2_pianist.rl.sac import SAC, SACConfig
from tsensei_v3_2_pianist.rl.specs import EnvironmentSpec


def small_agent(target_entropy=None):
    spec = EnvironmentSpec(specs.Array((7,), np.float32),
                           specs.BoundedArray((3,), np.float32, -1, 1))
    config = SACConfig(hidden_dims=(16, 16), critic_dropout_rate=.01,
                       critic_layer_norm=True, target_entropy=target_entropy)
    return SAC.initialize(spec, config, seed=12, discount=.8), config


def test_shadow_reference_three_updates():
    reference = np.load(Path(__file__).parent / "data/shadow_sac_reference.npz")
    agent, _ = small_agent()
    batch = Transition(*(reference[field] for field in Transition._fields))
    agent, actions = agent.sample_actions_device(batch.state)
    assert isinstance(actions, jax.Array)
    np.testing.assert_allclose(actions, reference["sampled_actions"], rtol=2e-5, atol=2e-6)
    for step in range(3):
        agent, metrics = agent.update(batch)
        for key, value in metrics.items():
            np.testing.assert_allclose(value, reference[f"metric_{step}_{key}"], rtol=2e-5, atol=2e-6)
    leaves = jax.tree.leaves(agent)
    assert len(leaves) == len([key for key in reference if key.startswith("leaf_")])
    for i, leaf in enumerate(leaves):
        np.testing.assert_allclose(leaf, reference[f"leaf_{i}"], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(agent.eval_actions(batch.state), reference["eval_actions"],
                               rtol=2e-5, atol=2e-6)


def test_replay_terminal_boundaries_and_wrap():
    replay = Buffer(2, 1, max_size=3, batch_size=3, seed=8)
    with pytest.raises(ValueError, match="not ready"):
        replay.sample()
    replay.insert([0, 0], [0], 1, 0, [9, 9])  # terminal before reset
    replay.insert([100, 100], [0], 2, 1, [101, 101])
    replay.insert([101, 101], [0], 3, 1, [102, 102])
    np.testing.assert_array_equal(replay._data.next_state[0], [9, 9])
    for _ in range(5):
        batch = replay.sample()
        for state, discount, following in zip(batch.state, batch.discount, batch.next_state):
            np.testing.assert_array_equal(following, [9, 9] if discount == 0 else state + 1)
    replay.insert([200, 200], [0], 4, 1, [201, 201])
    assert len(replay) == 3
    np.testing.assert_array_equal(replay._data.state[0], [200, 200])
    before = [v.copy() for v in replay._data]
    with pytest.raises(ValueError):
        replay.insert([200, 200], [np.nan], 4, 1, [201, 201])
    for actual, expected in zip(replay._data, before):
        np.testing.assert_array_equal(actual, expected)


def test_checkpoint_optimizer_rng_and_contract(tmp_path):
    reference = np.load(Path(__file__).parent / "data/shadow_sac_reference.npz")
    agent, config = small_agent()
    batch = Transition(*(reference[field] for field in Transition._fields))
    agent, _ = agent.update(batch)
    metadata = {"sac_config": asdict(config), "contract": {"actions": ["a", "b", "c"]},
                "discount": .8, "updates": 1}
    path = tmp_path / "state.msgpack"
    checkpoint.save(path, agent, metadata)
    restored, _ = checkpoint.load(path, small_agent()[0], metadata)
    next_agent, _ = agent.update(batch)
    next_restored, _ = restored.update(batch)
    for actual, expected in zip(jax.tree.leaves(next_restored), jax.tree.leaves(next_agent)):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="contract"):
        checkpoint.load(path, small_agent()[0], {**metadata, "contract": {"actions": ["b", "a", "c"]}})


def test_explicit_zero_entropy_and_invalid_settings():
    assert small_agent(target_entropy=0.0)[0].target_entropy == 0.0
    for values in ({"tau": 0}, {"critic_dropout_rate": 1}, {"num_min_qs": 3}, {"actor_lr": float("nan")}):
        with pytest.raises(ValueError):
            SACConfig(**values)


@pytest.mark.parametrize("frame_stack,fingering", [(1, True), (2, False)])
def test_v32_adapter_contract_and_action_endpoints(frame_stack, fingering):
    env, raw, contract = make_environment(EnvironmentConfig(
        midi="Toy", frame_stack=frame_stack, disable_fingering_reward=not fingering))
    try:
        step = env.reset()
        assert step.observation.dtype == np.float32
        assert step.observation.shape == env.observation_spec().shape
        expected = (1180 if fingering else 1170) * frame_stack
        assert contract["observation_size"] == expected
        assert contract["action_size"] == 51
        assert len(set(contract["action_names"])) == 51
        assert contract["action_names"][:2] == [a.name for a in raw.task.right_hand.actuators[:2]]
        np.testing.assert_array_equal(env.action_spec().minimum, -np.ones(51))
        lo, hi = raw.action_spec().minimum, raw.action_spec().maximum
        for normalized in (-1.0, 0.0, 1.0):
            env.reset()
            step = env.step(np.full(51, normalized, np.float32))
            expected_ctrl = lo + (normalized + 1) * .5 * (hi - lo)
            np.testing.assert_allclose(raw.physics.bind(raw.task.right_hand.actuators).ctrl, expected_ctrl[:25])
            np.testing.assert_allclose(raw.physics.bind(raw.task.left_hand.actuators).ctrl, expected_ctrl[25:50])
            np.testing.assert_allclose(raw.task.piano.sustain_state, expected_ctrl[50:])
            assert np.isfinite(step.reward)
        with pytest.raises(ValueError, match="Invalid normalized"):
            env.step(np.full(51, np.nan))
    finally:
        env.close()


def test_evaluation_does_not_change_training_state(tmp_path, monkeypatch):
    import dm_env
    from types import SimpleNamespace
    from tsensei_v3_2_pianist.rl import train

    class FakeEnv(dm_env.Environment):
        def __init__(self):
            self.task = SimpleNamespace(_goal_current=np.array([1, 0]),
                                        piano=SimpleNamespace(activation=np.array([True]),
                                                              sustain_activation=np.array([False])))
        def reset(self):
            self.t = 0
            return dm_env.restart(np.zeros(7, np.float32))
        def step(self, action):
            self.t += 1
            state = np.full(7, self.t + float(np.sum(action)), np.float32)
            if self.t == 3:
                return dm_env.termination(1., state)
            return dm_env.transition(1., state)
        def action_spec(self):
            return specs.BoundedArray((3,), np.float32, -1, 1)
        def observation_spec(self):
            return specs.Array((7,), np.float32)
        def close(self):
            pass

    def factory(*args):
        env = FakeEnv()
        return env, env, {"test": "fake"}
    monkeypatch.setattr(train, "make_environment", factory)
    payloads = []
    from flax import serialization
    for eval_interval in [0, 4]:
        output = tmp_path / str(eval_interval)
        args = train.parser().parse_args([
            "--max-steps", "12", "--warmstart-steps", "4", "--batch-size", "4",
            "--replay-capacity", "16", "--hidden-dims", "16", "16",
            "--eval-interval", str(eval_interval), "--save-interval", "0",
            "--output-dir", str(output)])
        result = train.run(args)
        assert result["updates"] == 9 and result["episodes"] == 4
        payloads.append(serialization.msgpack_restore((output / "final.msgpack").read_bytes())["agent"])
    for a, b in zip(jax.tree.leaves(payloads[0]), jax.tree.leaves(payloads[1])):
        np.testing.assert_array_equal(a, b)


def test_evaluation_metric_matches_shadow_convention():
    from sklearn.metrics import precision_recall_fscore_support
    from tsensei_v3_2_pianist.rl.train import binary_metrics
    for goal in ([0, 0], [1, 0], [1, 1]):
        for actual in ([0, 0], [0, 1], [1, 1]):
            expected = precision_recall_fscore_support(goal, actual, average="binary", zero_division=1)[:3]
            np.testing.assert_allclose(binary_metrics(np.array(goal, bool), np.array(actual, bool)), expected)
