"""Pure, stdlib-only conversion of local training records to public scalars.

No SDK, filesystem, network, environment, credentials, or training code is used.
The first three curves use the existing full-episode, control-frame evaluation
averages (including preparation). They must not be mixed with music-only or
micro-averaged scores. Missing/non-finite values are omitted, never filled in.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


PRIMARY = ("metrics/01_F1", "metrics/02_recall", "metrics/03_precision")
LEGACY_REWARDS = (
    "energy_reward", "fingering_reward", "forearm_reward", "key_press_reward",
    "sustain_reward",
)
REWARD_NAMES = frozenset((*LEGACY_REWARDS, "self_collision_reward",
    "false_press_penalty", "humanoid_finger_gesture_reward", "finger_speed_reward", "joint_motion_limit_penalty",
    "wrist_pitch_reward",
    "forward_press_reward", *(f"{group}_{kind}_reward"
        for group in ("forearm", "wrist", "finger")
        for kind in ("power", "velocity", "acceleration"))))
TRAIN_FIELDS = ("actor_loss", "critic_loss", "entropy", "q", "temperature",
                "temperature_loss")
EVAL_FIELDS = (
    "return", "length", "completed_episodes", "truncated_episodes",
    "evaluation_episodes", "metric_sample_hz", "music_frames",
    "music_only_f1", "music_only_f1_min", "music_only_micro_f1",
    "music_only_micro_f1_min", "key_frame_micro_f1", "sustain_f1",
    "sustain_recall", "sustain_precision", "preparation_frames",
    "preparation_wrong_key_frames", "preparation_wrong_key_frames_max",
    "preparation_wrong_key_count", "extra_buffer_frames",
    "collision_diagnostics_coverage", "collision_substep_coverage",
    "key_contact_diagnostics_coverage", "key_contact_substep_coverage",
    "substep_key_diagnostics_coverage", "substep_key_coverage",
    "substep_key_sample_hz", "motion_sample_hz",
    "music_end_frame", "release_tail_steps", "release_tail_frames",
    "release_tail_substep_key_micro_f1", "release_tail_substep_key_micro_f1_min",
    *(name + maximum
      for name in ("release_tail_wrong_key_frames", "release_tail_wrong_key_count",
                   "release_tail_end_active_keys", "release_tail_wrong_key_substeps",
                   "release_tail_wrong_key_key_substeps")
      for maximum in ("", "_max")),
    *(prefix + suffix + maximum
      for prefix in ("", "music_only_", "preparation_")
      for suffix in ("wrong_key_substeps", "wrong_key_key_substeps", "substep_key_micro_f1")
      for maximum in ("", "_max" if suffix != "substep_key_micro_f1" else "_min")),
    *(name + maximum
      for name in ("hand_collision_frames", "self_collision_frames", "inter_hand_collision_frames",
                   "self_collision_pair_frames", "inter_hand_collision_pair_frames",
                   "self_collision_substeps", "inter_hand_collision_substeps", "hand_collision_substeps",
                   "key_underside_frames", "key_side_or_under_frames", "key_non_distal_frames",
                   "key_underside_substeps", "key_side_or_under_substeps", "key_non_distal_substeps",
                   "control_clipped_frames", "control_slew_limited_frames",
                   "control_clipped_fraction", "control_slew_limited_fraction",
                   "finger_clearance_actual_deficit_frames", "finger_clearance_applied_deficit_frames",
                   "finger_clearance_actual_max_deficit_m", "finger_clearance_requested_max_deficit_m",
                   "finger_clearance_applied_max_deficit_m")
      for maximum in ("", "_max")),
    *("motion_max_" + name for name in (
        "wrist_tx_speed_m_s", "wrist_tx_target_jump_m", "wrist_ty_speed_m_s",
        "wrist_ty_target_jump_m", "finger_speed_rad_s", "finger_target_jump_rad",
        "wrist_pitch_speed_rad_s", "wrist_pitch_target_jump_rad")),
)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _finite(value):
    # JSON booleans are not numeric observations; numeric strings are not trusted.
    return type(value) in (int, float) and math.isfinite(value)


def _reward_names(config):
    terms = _mapping(_mapping(config).get("reward_config")).get("terms")
    if not isinstance(terms, Mapping):
        return list(LEGACY_REWARDS)
    return sorted(name for name, term in terms.items()
                  if name in REWARD_NAMES and _mapping(term).get("enabled") is True)


def metric_order(config: Mapping) -> list[str]:
    """Return stable dashboard order: F1, recall, precision, weighted rewards.

    Only enabled, recognized reward terms receive numbered main charts. The
    numbers keep their order when the dashboard sorts chart names. Legacy
    configurations without a reward manifest use the five original rewards.
    """
    return [*PRIMARY, *(f"metrics/{index:02d}_{name}"
                       for index, name in enumerate(_reward_names(config), 4))]


def _music_micro(evaluation, *, substeps=False):
    rows = evaluation.get("episode_statistics")
    if not isinstance(rows, list) or not rows:
        return {}
    prefix = "music_only_substep_" if substeps else "music_"
    fields = [prefix + name + "_count" for name in
              ("true_positive", "false_positive", "false_negative")]
    # All episodes must have all counts. Do not create an optimistic partial sum.
    if any(not isinstance(row, Mapping) or any(
            not _finite(row.get(name)) or row[name] < 0 or row[name] % 1
            for name in fields) for row in rows):
        return {}
    tp, fp, fn = (sum(row[name] for row in rows) for name in fields)
    label = "music_only_substep_key_micro_" if substeps else "music_only_micro_"
    result = {}
    for name, numerator, denominator in (
        ("f1", 2 * tp, 2 * tp + fp + fn),
        ("recall", tp, tp + fn), ("precision", tp, tp + fp),
    ):
        if denominator:
            result["evaluation/" + label + name] = numerator / denominator
    return result


def scalars_from_record(record: Mapping, config: Mapping) -> dict[str, float]:
    """Convert one JSONL row; preserve chart order and omit unavailable values.

    Main rewards are already weighted contributions, without multiplying their
    weight again. Historical ``reward_terms`` is a fallback only when the
    explicit weighted dictionary is absent. Raw terms are a separate group.
    Stochastic training-episode F1 never fills a deterministic evaluation chart.
    """
    record = _mapping(record)
    evaluation = _mapping(record.get("eval"))
    result = {}
    for key, field in zip(PRIMARY, ("f1", "recall", "precision")):
        if _finite(evaluation.get(field)):
            result[key] = float(evaluation[field])
    weighted = record.get("reward_weighted_terms")
    if not isinstance(weighted, Mapping):
        weighted = _mapping(record.get("reward_terms"))
    for key, name in zip(metric_order(config)[3:], _reward_names(config)):
        if _finite(weighted.get(name)):
            result[key] = float(weighted[name])
    extra = {}
    for field in EVAL_FIELDS:
        if _finite(evaluation.get(field)):
            extra["evaluation/" + field] = float(evaluation[field])
    if type(evaluation.get('release_tail_complete')) is bool:
        extra['evaluation/release_tail_complete'] = float(evaluation['release_tail_complete'])
    extra.update(_music_micro(evaluation))
    extra.update(_music_micro(evaluation, substeps=True))
    for group, data, fields in (
        ("reward_raw", record.get("reward_raw_terms"), _reward_names(config)),
        ("train", record.get("train"), TRAIN_FIELDS),
        ("train_episodes", record.get("episodes"), ("count", "mean_return", "mean_f1")),
        ("training", record, ("vector_step", "transitions", "updates", "replay_size",
                               "loop_transitions_per_second")),
    ):
        data = _mapping(data)
        for field in fields:
            if _finite(data.get(field)):
                extra[f"{group}/{field}"] = float(data[field])
    return {**result, **dict(sorted(extra.items()))}


def merge_same_step_records(records: Iterable[Mapping], config: Mapping) -> list[tuple[int, dict[str, float]]]:
    """Group rows by vector step before calling SDK log once per step.

    Training, episode, and evaluation events can be separate JSONL rows at the
    same step. Merge all scalar keys; a later *present* value replaces an older
    value of the same key. Missing/invalid later values do not erase valid data.
    Invalid/missing steps raise ValueError rather than inventing a step number.
    Streaming callers must wait until a newer step or end of run before flushing.
    """
    grouped = {}
    for record in records:
        step = _mapping(record).get("vector_step")
        if type(step) is not int or step < 0:
            raise ValueError("Each metrics record requires a nonnegative integer vector_step")
        grouped.setdefault(step, {}).update(scalars_from_record(record, config))
    primary = metric_order(config)
    return [(step, {**{key: data[key] for key in primary if key in data},
                    **{key: data[key] for key in sorted(data) if key not in primary}})
            for step, data in sorted(grouped.items())]


# Explicit allowlists are used rather than secret-key blacklists: new local
# metadata, source archives, paths, scores, and credentials stay local by default.
_TOP_NUMERIC = ("stage", "num_envs", "replay_capacity", "batch_size", "warmup_vector_steps",
    "warmup_transitions", "learning_starts_vector_steps", "learning_start_transition_1_based",
    "requested_learning_start_transition_1_based",
    "gradient_steps_per_vector_step", "steady_updates_per_transition", "steady_sample_ratio",
    "requested_vector_steps", "requested_transitions", "observation_size", "action_size",
    "target_entropy", "self_collision_weight")
_TOP_BOOL = ("no_arms", "disable_wrist_tz", "disable_wrist_yaw",
    "hand_collision_detection_only", "self_collision_detection_only", "collision_diagnostics")
_ENUMS = {
    "action_profile": {"legacy", "mixed_v1", "direct_joint_v1", "shadow_v1"},
    "weight_stage": {"current", "stage1", "stage2_canon", "stage2_other"},
    "action_units": {"physical", "normalized"}, "regularization_shaping": {"identity"},
    "mode": {"legacy", "paper", "fixed_v1", "policy_v1", "profile", "uniform"},
    "trajectory": {"quintic_v1"}, "task_profile": {"tsensei_v1", "tsensei_collision_v1"},
    "collision_profile": {"legacy_mesh_v1", "convex_hull_v1", "shadow_primitive_v1"},
    "drive_profile": {"legacy_v1", "shadow_v1"},
    "best_metric": {"music_only_f1_min"},
    "chord_scope": {"same_hand", "both_hands"},
    "score_validation": {"legacy", "strict_v1"},
}


def _pick(value, numeric=(), boolean=(), enums=()):
    source, result = _mapping(value), {}
    for key in numeric:
        if _finite(source.get(key)):
            result[key] = source[key]
    for key in boolean:
        if type(source.get(key)) is bool:
            result[key] = source[key]
    for key in enums:
        if key in source and (source[key] is None or
                isinstance(source[key], str) and source[key] in _ENUMS[key]):
            result[key] = source[key]
    return result


def public_config(config: Mapping) -> dict:
    """Return only public hyperparameters and scalar reward settings.

    Never copies arbitrary dictionaries/strings: CLI arguments, source hashes,
    source/runtime paths, host metadata, external scores and model maps are
    excluded even if nested under an otherwise accepted configuration section.
    Recognized safe environment hyperparameters are extracted separately.
    """
    config = _mapping(config)
    result = _pick(config, _TOP_NUMERIC, _TOP_BOOL, ("action_profile",))
    sections = {
        "sac": (("actor_lr", "critic_lr", "temp_lr", "gamma", "tau", "init_temperature",
                 "target_entropy", "critic_dropout"), ("critic_layer_norm",), ()),
        "preparation": (("duration", "lift"), (), ("mode", "trajectory")),
        "acceleration": ((), ("physics_graph", "compile_learner", "fused_optimizer",
                              "learner_graph", "optimizer_layout"), ()),
        "replay_prefill": (("episodes", "episode_steps", "required_vector_steps", "required_transitions",
            "requested_learning_start_transition_1_based", "effective_learning_start_transition_1_based"),
            ("resolved",), ()),
        "evaluation_selection": (("target_music_f1", "target_micro_f1", "consecutive_evaluations",
            "max_preparation_wrong_frames", "max_hand_collision_frames", "max_key_underside_substeps"),
            ("save_best", "eval_at_start", "require_complete_episodes", "require_full_collision_coverage"),
            ("best_metric",)),
    }
    for key, specification in sections.items():
        if key in config:
            result[key] = _pick(config[key], *specification)
    for key in ("actor_hidden", "critic_hidden"):
        value = _mapping(config.get("sac")).get(key)
        if isinstance(value, (list, tuple)) and all(type(x) is int and x > 0 for x in value):
            result.setdefault("sac", {})[key] = list(value)
    reward = _mapping(config.get("reward_config"))
    if reward:
        clean = _pick(reward, ("schema", "reference_dt"), ("scale_by_dt",),
                      ("weight_stage", "action_units", "regularization_shaping"))
        clean["terms"] = {}
        for name, term in sorted(_mapping(reward.get("terms")).items()):
            if name not in REWARD_NAMES or not isinstance(term, Mapping):
                continue
            clean_term = _pick(term, ("weight",), ("enabled",))
            clean_term["params"] = _pick(term.get("params"),
                ("coefficient", "limit", "bonus", "threshold", "maximum_per_finger", "sigma",
                 "maximum_total", "lookahead_seconds", "missed_key_fraction", "clearance_scale",
                 "posture_fraction", "score_multiplier"), ("chord_bonus",), ("mode", "chord_scope"))
            clean["terms"][name] = clean_term
        if "performance" in reward:
            clean["performance"] = _pick(reward["performance"],
                ("anticipatory_fingering_time", "target_key_activation_weight"),
                ("staged_fingering_reward", "substep_rewards", "target_key_substep_reward",
                 "anticipatory_current_priority"))
        result["reward_config"] = clean
    runtime = _mapping(config.get("source_runtime"))
    if runtime:
        result["environment"] = _pick(runtime.get("environment"),
            ("control_hz", "physics_hz", "lookahead", "frame_stack", "initial_buffer_time", "release_tail_time"),
            ("gravity_compensation", "action_reward_observation", "disable_forearm_reward"),
            ("task_profile", "score_validation"))
        result["model"] = _pick(runtime.get("model"), (), (), ("collision_profile", "drive_profile"))
        result["pose"] = _pick(runtime.get("pose"), ("height", "ty", "yaw"))
    seed = _mapping(config.get("cli")).get("seed")
    if type(seed) is int:
        result["seed"] = seed
    return result
