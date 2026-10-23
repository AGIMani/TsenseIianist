"""V3.2 adapter using the same wrapper order as the Shadow SAC trainer."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import dm_env_wrappers as wrappers
import numpy as np
import tree

from tsensei_v3_2_pianist import suite


class CanonicalActionWrapper(wrappers.CanonicalSpecWrapper):
    def step(self, action):
        # V3.2 validates strict physical limits: scale in the physical spec dtype
        # to avoid float32 rounding a valid endpoint outside a float64 limit.
        action = np.asarray(action, dtype=self._action_spec.dtype)
        if action.shape != self._action_spec.shape or not np.isfinite(action).all():
            raise ValueError("Invalid normalized action")
        return super().step(action)


class SinglePrecisionWrapper(wrappers.SinglePrecisionWrapper):
    def _convert_timestep(self, timestep):
        # dm-env-wrappers 0.0.13 uses np.array(copy=False) when casting dtypes,
        # which NumPy 2 rejects. Preserve the original dtype conversion policy.
        def convert(value):
            if value is None:
                return None
            value = np.asarray(value)
            dtype = {np.dtype("float64"): np.float32,
                     np.dtype("int64"): np.int32}.get(value.dtype, value.dtype)
            return np.asarray(value, dtype=dtype)
        return timestep._replace(**{
            field: tree.map_structure(convert, getattr(timestep, field))
            for field in ("observation", "reward", "discount")
        })


@dataclass(frozen=True)
class EnvironmentConfig:
    midi: str = "TwinkleTwinkleRousseau"
    n_steps_lookahead: int = 20
    control_timestep: float = 0.025
    physics_timestep: float | None = None
    trim_silence: bool = True
    gravity_compensation: bool = True
    initial_buffer_time: float = 0.0
    release_tail_time: float = 0.0
    score_validation: str = 'legacy'
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_hand_collisions: bool = False
    frame_stack: int = 1
    action_reward_observation: bool = True
    task_profile: str = 'tsensei_v1'

    def __post_init__(self):
        from tsensei_v3_2_pianist.forearm_collision import collision_scope
        collision_scope(self.task_profile)
        if self.n_steps_lookahead < 0 or self.frame_stack < 1:
            raise ValueError("Invalid lookahead or frame_stack")
        if not np.isfinite(self.control_timestep) or self.control_timestep <= 0:
            raise ValueError("control_timestep must be positive")
        if not np.isfinite(self.initial_buffer_time) or self.initial_buffer_time < 0:
            raise ValueError('initial_buffer_time must be finite and nonnegative')
        from tsensei_v3_2_pianist.score_validation import validate_mode, release_tail_steps
        validate_mode(self.score_validation)
        release_tail_steps(self.release_tail_time, self.control_timestep)


def make_environment(config=EnvironmentConfig(), seed=0):
    kwargs = asdict(config)
    for name in ("midi", "frame_stack", "action_reward_observation"):
        kwargs.pop(name)
    if kwargs["physics_timestep"] is None:
        kwargs.pop("physics_timestep")
    raw = suite.load(config.midi, seed=seed, task_kwargs=kwargs)
    try:
        env = raw
        if config.action_reward_observation:
            env = wrappers.ObservationActionRewardWrapper(env)
        fields = [
            {"path": "/".join(map(str, path)), "shape": list(spec.shape),
             "size": int(np.prod(spec.shape))}
            for path, spec in tree.flatten_with_path(env.observation_spec())
        ]
        env = wrappers.ConcatObservationWrapper(env)
        if config.frame_stack > 1:
            env = wrappers.FrameStackingWrapper(env, num_frames=config.frame_stack, flatten=True)
        env = CanonicalActionWrapper(env, clip=True)
        env = SinglePrecisionWrapper(env)
        env = wrappers.DmControlWrapper(env)
        raw_spec = raw.action_spec()
        contract = {
            "environment": asdict(config),
            "observation_fields_before_stack": fields,
            "observation_size": int(np.prod(env.observation_spec().shape)),
            "action_size": int(np.prod(raw_spec.shape)),
            "action_names": [a.name for h in (raw.task.right_hand, raw.task.left_hand)
                             for a in h.actuators] + ["sustain"],
            "physical_action_minimum": raw_spec.minimum.tolist(),
            "physical_action_maximum": raw_spec.maximum.tolist(),
            "normalized_action_minimum": env.action_spec().minimum.tolist(),
            "normalized_action_maximum": env.action_spec().maximum.tolist(),
        }
        package = Path(__file__).resolve().parents[1]
        tracked = ["assets/model_derivation.json", "assets/urdf/TsenseI_left.urdf",
                   "assets/urdf/TsenseI_right.urdf", "models/hands/tsensei_hand_constants.py",
                   "models/hands/tsensei_hand.py", "models/hands/urdf_model.py",
                   "suite/tasks/base.py", "suite/tasks/piano_with_tsensei_hands.py",
                   "models/piano/piano.py", "models/piano/piano_mjcf.py",
                   "models/piano/piano_constants.py", "music/library.py"]
        contract["source_sha256"] = {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in tracked
        }
        if Path(config.midi).is_file():
            contract["midi_sha256"] = hashlib.sha256(Path(config.midi).read_bytes()).hexdigest()
        return env, raw, contract
    except BaseException:
        raw.close()
        raise
