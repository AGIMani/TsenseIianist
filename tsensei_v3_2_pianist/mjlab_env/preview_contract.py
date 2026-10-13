"""Reconstruct the exact MJLab training contract for preview and audits.

The interactive preview historically built a hand-only ``PianoScene`` and the
old arm preview used a separate IK implementation.  This module is the small
bridge used by the MJLab preview: a saved training ``config.json`` is the
single source for the source recipe, reward, controls, preparation and arm
contract.  It deliberately keeps source-runtime verification enabled so a
preview cannot silently use a different checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .arm_options import canonical as canonical_arm
from .source_runtime import build as build_source


def load_training_config(path: str | Path) -> dict:
    """Load and validate the subset of a training config needed by MJLab.

    A checkpoint/config from an older source archive must fail at
    ``source_runtime.build`` rather than being silently adapted.  Keeping the
    validation here also gives the viewer and the CPU reachability scanner the
    same input contract.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Training config must contain a JSON object")
    recipe = config.get("source_runtime")
    if not isinstance(recipe, dict):
        raise ValueError("Training config is missing source_runtime")
    arm = config.get("arm_follow")
    if arm is None:
        raise ValueError("Training config is missing the explicit arm_follow contract")
    # This checks schema, mount hashes, connector hashes and the seven-angle
    # posture before any GPU is allocated.
    config["arm_follow"] = canonical_arm(arm)
    config["_preview_config_path"] = str(path)
    return config


def build_source_from_config(config: dict):
    """Build the temporary scene/task from the recorded source recipe."""

    return build_source(config["source_runtime"])


def task_kwargs(config: dict) -> dict:
    """Return the constructor kwargs shared by training and MJLab preview."""

    acceleration = config.get("acceleration") or {}
    return dict(
        max_episode_steps=config.get("max_episode_steps"),
        physics_graph=bool(acceleration.get("physics_graph", True)),
        self_collision_weight=float(config.get("self_collision_weight", 0.0)),
        reward_config=config.get("reward_config"),
        action_profile=config.get("action_profile", "legacy"),
        disable_wrist_tz=bool(config.get("disable_wrist_tz", False)),
        disable_wrist_yaw=bool(config.get("disable_wrist_yaw", False)),
        self_collision_detection_only=bool(config.get("self_collision_detection_only", False)),
        hand_collision_detection_only=bool(config.get("hand_collision_detection_only", False)),
        preparation=config.get("preparation"),
        collision_diagnostics=bool(config.get("collision_diagnostics", False)),
        control_limits=config.get("control_limits"),
        finger_clearance=config.get("finger_clearance"),
        finger_coupling=config.get("finger_coupling"),
        arm_follow=config["arm_follow"],
    )


def build_task_from_config(config: dict, *, device: str = "cuda:0", num_envs: int = 1):
    """Build ``PianoTaskEnv`` with the same options as the recorded run."""

    scene_directory, task_directory, compiler_config = build_source_from_config(config)
    from .task import PianoTaskEnv

    env = PianoTaskEnv(
        scene_directory,
        task_directory,
        num_envs=num_envs,
        device=device,
        **task_kwargs(config),
    )
    return env, scene_directory, task_directory, compiler_config


def build_bundle_from_config(config: dict):
    """Build a CPU ``SceneBundle`` for static arm workspace auditing.

    The bundle path follows the same preparation and arm augmentation as the
    GPU scene, while avoiding MJLab/Warp so an audit can run as ordinary CPU
    analysis.  No policy or reward is evaluated in this path.
    """

    from .bundle import SceneBundle

    # Composer's MuJoCo import eagerly initializes a GL backend.  The scan is
    # intentionally CPU-only, so disable GL for this temporary compile even
    # when the surrounding Slurm job exported its normal EGL setting.
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "disable"
    try:
        scene_directory, _, compiler_config = build_source_from_config(config)
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl
    bundle = SceneBundle(
        scene_directory,
        disable_wrist_tz=bool(config.get("disable_wrist_tz", False)),
        disable_wrist_yaw=bool(config.get("disable_wrist_yaw", False)),
        preparation=config.get("preparation"),
        arm_follow=config["arm_follow"],
    )
    return bundle, scene_directory, compiler_config
