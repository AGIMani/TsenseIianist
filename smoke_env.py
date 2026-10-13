#!/usr/bin/env python3
"""Exercise reset, rewards, position actions and termination; does not train a policy."""
import argparse
import json
import numpy as np
from tsensei_v3_2_pianist import suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    env = suite.load()
    timestep = env.reset()
    spec = env.action_spec()
    action = np.concatenate((env.task.right_hand.neutral_control, env.task.left_hand.neutral_control, [0.]))
    observation_size = sum(np.asarray(v).size for v in timestep.observation.values())
    episodes = 0
    for _ in range(args.steps):
        timestep = env.step(action)
        assert np.isfinite(timestep.reward)
        assert np.isfinite(env.physics.data.qpos).all()
        assert all(np.isfinite(v).all() for v in timestep.observation.values())
        if timestep.last():
            episodes += 1
            timestep = env.reset()
    print(json.dumps({"steps": args.steps, "completed_episodes": episodes,
                      "action_size": spec.shape[0], "observation_size": observation_size,
                      "nq": env.physics.model.nq, "nv": env.physics.model.nv,
                      "nu": env.physics.model.nu, "status": "passed"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
