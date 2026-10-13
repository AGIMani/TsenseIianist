"""Single-environment SAC baseline for V3.2; no GPU physics or multi-GPU runner."""
import argparse
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import time

import jax
import numpy as np

from . import checkpoint
from .environment import EnvironmentConfig, make_environment
from .replay import Buffer
from .sac import SAC, SACConfig
from .specs import EnvironmentSpec


def binary_metrics(goal, actual):
    """Upstream per-frame binary precision/recall/F1 with zero_division=1."""
    tp = int(np.sum(goal & actual))
    fp = int(np.sum(~goal & actual))
    fn = int(np.sum(goal & ~actual))
    return (tp / (tp + fp) if tp + fp else 1.0,
            tp / (tp + fn) if tp + fn else 1.0,
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0)


def evaluate(agent, env, raw, episodes=1, max_steps=20000):
    """Independent evaluation; retain upstream per-frame-then-episode averaging."""
    returns, lengths = [], []
    key_metrics, sustain_metrics = [], []
    tp = fp = fn = completed = 0
    for _ in range(episodes):
        evaluation_step = env.reset()
        reward = 0.0
        key_frames, sustain_frames = [], []
        for length in range(1, max_steps + 1):
            evaluation_step = env.step(agent.eval_actions(evaluation_step.observation))
            reward += float(evaluation_step.reward)
            goal = np.asarray(raw.task._goal_current[:-1], dtype=bool)
            actual = np.asarray(raw.task.piano.activation, dtype=bool)
            tp += int(np.sum(goal & actual))
            fp += int(np.sum(~goal & actual))
            fn += int(np.sum(goal & ~actual))
            key_frames.append(binary_metrics(goal, actual))
            sustain_frames.append(binary_metrics(
                np.asarray(raw.task._goal_current[-1:], dtype=bool),
                np.asarray(raw.task.piano.sustain_activation, dtype=bool)))
            if evaluation_step.last():
                completed += 1
                break
        returns.append(reward)
        lengths.append(length)
        key_metrics.append(np.mean(key_frames, axis=0))
        sustain_metrics.append(np.mean(sustain_frames, axis=0))
    musical_metrics = dict(zip(("precision", "recall", "f1"), map(float, np.mean(key_metrics, axis=0))))
    musical_metrics.update(zip(("sustain_precision", "sustain_recall", "sustain_f1"),
                              map(float, np.mean(sustain_metrics, axis=0))))
    return {**musical_metrics, "return": float(np.mean(returns)), "length": float(np.mean(lengths)),
            "ended_episodes": completed, "capped_episodes": episodes - completed,
            "key_frame_micro_f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--midi", default="TwinkleTwinkleRousseau")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=5_000_000)
    p.add_argument("--warmstart-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=1_000_000)
    p.add_argument("--discount", type=float, default=0.8)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256])
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--temp-lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--target-entropy", type=float)
    p.add_argument("--critic-dropout-rate", type=float, default=0.01)
    p.add_argument("--critic-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-steps-lookahead", type=int, default=10)
    p.add_argument("--control-timestep", type=float, default=0.05)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--action-reward-observation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trim-silence", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gravity-compensation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wrong-press-termination", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--disable-fingering-reward", action="store_true")
    p.add_argument("--disable-forearm-reward", action="store_true")
    p.add_argument("--disable-hand-collisions", action="store_true")
    p.add_argument("--eval-interval", type=int, default=10000, help="0 disables evaluation")
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--eval-max-steps", type=int, default=20000)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--save-interval", type=int, default=10000)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--warm-start", type=Path, help="Load SAC state, start fresh replay and episodes")
    p.add_argument("--inspect-only", action="store_true", help="Print actual environment contract; no training")
    return p


def run(args):
    if min(args.max_steps, args.warmstart_steps, args.eval_interval, args.save_interval) < 0:
        raise ValueError("Step counts and intervals must be nonnegative")
    if min(args.log_interval, args.eval_episodes, args.eval_max_steps) < 1:
        raise ValueError("Log interval and evaluation counts must be positive")
    if not 0 <= args.discount <= 1:
        raise ValueError("discount must be in [0,1]")
    if args.batch_size < 1 or args.replay_capacity < args.batch_size:
        raise ValueError("Require replay_capacity >= batch_size > 0")
    config = EnvironmentConfig(**{key: getattr(args, key) for key in EnvironmentConfig.__dataclass_fields__})
    sac_config = SACConfig(
        hidden_dims=tuple(args.hidden_dims), actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        temp_lr=args.temp_lr, tau=args.tau, target_entropy=args.target_entropy,
        critic_dropout_rate=args.critic_dropout_rate, critic_layer_norm=args.critic_layer_norm,
    )
    env = eval_env = None
    try:
        env, raw, contract = make_environment(config, args.seed)
        timestep = env.reset()
        metadata = {
            "contract": contract, "sac_config": asdict(sac_config), "discount": args.discount,
            "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "versions": {name: importlib.metadata.version(name) for name in
                         ("jax", "jaxlib", "flax", "optax", "distrax", "mujoco", "dm-env-wrappers")},
            "checkpoint_kind": "SAC state only; warm restart with fresh replay and physics",
        }
        if args.inspect_only:
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            return metadata
        output = args.output_dir or Path("runs") / datetime.now().strftime("sac-%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True, exist_ok=False)
        spec = EnvironmentSpec.make(env)
        agent = SAC.initialize(spec, sac_config, args.seed, args.discount)
        updates = 0
        if args.warm_start:
            agent, previous = checkpoint.load(args.warm_start, agent, metadata)
            updates = previous["updates"]
            metadata["warm_start_origin"] = str(args.warm_start.resolve())
        metadata["devices"] = [str(device) for device in jax.devices()]
        (output / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        replay = Buffer(spec.observation_dim, spec.action_dim, args.replay_capacity,
                        args.batch_size, seed=args.seed)
        if args.eval_interval:
            eval_env, eval_raw, _ = make_environment(config, args.seed + 1)
        # Dedicated exploration RNG prevents evaluation/reset from consuming training RNG.
        exploration_rng = np.random.RandomState(args.seed)
        started = time.monotonic()
        episode_return = 0.0
        episode_length = episodes = 0
        metrics = {}
        with (output / "metrics.jsonl").open("w") as log:
            def emit(record):
                record = {"step": step, "updates": updates, **record}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
            for step in range(1, args.max_steps + 1):
                if step < args.warmstart_steps:
                    action = spec.sample_action(exploration_rng)
                else:
                    agent, action = agent.sample_actions(timestep.observation)
                next_step = env.step(action)
                replay.insert(timestep.observation, action, next_step.reward,
                              next_step.discount, next_step.observation)
                episode_return += float(next_step.reward)
                episode_length += 1
                timestep = next_step
                if next_step.last():
                    episodes += 1
                    emit({"episode_return": episode_return, "episode_length": episode_length})
                    timestep = env.reset()
                    episode_return, episode_length = 0.0, 0
                if step >= args.warmstart_steps and replay.is_ready():
                    agent, metrics = agent.update(replay.sample())
                    updates += 1
                if step % args.log_interval == 0:
                    emit({"train": {k: float(v) for k, v in metrics.items()},
                          "transitions_per_second": step / (time.monotonic() - started)})
                if args.eval_interval and step % args.eval_interval == 0:
                    emit({"eval": evaluate(agent, eval_env, eval_raw,
                                            args.eval_episodes, args.eval_max_steps)})
                if args.save_interval and step % args.save_interval == 0:
                    checkpoint.save(output / f"step-{step}.msgpack", agent,
                                    {**metadata, "step": step, "updates": updates})
        checkpoint.save(output / "final.msgpack", agent,
                        {**metadata, "step": args.max_steps, "updates": updates})
        summary = {"steps": args.max_steps, "updates": updates, "episodes": episodes,
                   "replay_size": len(replay), "output_dir": str(output.resolve()),
                   "metrics": {k: float(v) for k, v in metrics.items()}}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
