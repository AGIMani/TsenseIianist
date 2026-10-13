"""MJLab + SAC with optional one-vector collector/learner overlap on two GPUs."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from tqdm import tqdm

from .sac import SAC, SACConfig
from .replay import Replay, Transition
from .collector_pipeline import CollectorPolicy, OneBatchPipeline, execution_layout
from .coupling_evaluation import CouplingMetrics
from . import checkpoint
from ..mjlab_env import reward_config as rewards
from ..mjlab_env.action_mapping import PROFILES

ROOT = Path(__file__).resolve().parents[2]
from ..mjlab_env.paths import CURRENT_GRAVCOMP_SCENE, CURRENT_GRAVCOMP_TASK


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=CURRENT_GRAVCOMP_SCENE)
    p.add_argument('--task', type=Path, default=CURRENT_GRAVCOMP_TASK)
    p.add_argument('--hand-collision-detection-only', action='store_true',
                   help='Detect same/cross-hand collisions for rewards without physical reaction forces')
    p.add_argument('--self-collision-detection-only', action='store_true',
                   help='Keep same-hand collision penalties but remove their physical reaction forces')
    p.add_argument('--disable-wrist-yaw', action='store_true',
                   help='With --disable-wrist-tz: 47 actions, physically fix yaw at frozen reset angles')
    p.add_argument('--disable-wrist-tz', action='store_true',
                   help='49-action policy: omit both TZ joints and actuators before physics compilation (physical lock at zero)')
    p.add_argument('--no-arms', action='store_true',
                   help='Remove M6 bodies, IK and arm collision pairs in a private frozen copy; retain hand collisions')
    from ..mjlab_env.arm_options import add_arguments as add_arm_arguments
    add_arm_arguments(p)
    p.add_argument('--device', default='cuda:0', help='Collector/physics CUDA device')
    p.add_argument('--learner-device', help='SAC/replay CUDA device; omitted uses --device')
    p.add_argument('--collector-pipeline', action=argparse.BooleanOptionalAction, default=False,
                   help='Overlap collection with the preceding vector update on another GPU; actor lags at most one update block')
    p.add_argument('--action-profile', choices=PROFILES, default='legacy')
    p.add_argument('--finger-coupling', action=argparse.BooleanOptionalAction, default=False,
                   help='Four fingers per hand: MCP leads physical PIP/DIP servos; thumbs remain independent')
    p.add_argument('--finger-coupling-angle', type=float,
                   help='Explicit legacy fixed distal angle in degrees; overrides angle ranges')
    p.add_argument('--finger-coupling-angle-ranges', type=Path,
                   help='JSON mapping all eight non-thumb finger names to [lower, upper] degrees; coupling defaults to 50..80 each')
    p.add_argument('--coupling-reduce-observation', action=argparse.BooleanOptionalAction, default=True,
                   help='With coupling, omit follower positions/previous targets from policy observation')
    p.add_argument('--eval-metrics-graph', action=argparse.BooleanOptionalAction, default=False,
                   help='CUDA graph for read-only evaluation statistics; keep all frames and diagnostics')
    p.add_argument('--physics-graph', action=argparse.BooleanOptionalAction, default=True,
                   help='Launch all physics substeps as one CUDA graph')
    p.add_argument('--compile-learner', action=argparse.BooleanOptionalAction, default=True,
                   help='Compile SAC tensor objectives and backward; first update incurs compilation')
    p.add_argument('--fused-optimizer', action=argparse.BooleanOptionalAction, default=True,
                   help='Use CUDA fused Adam without changing optimizer hyperparameters')
    p.add_argument('--learner-graph', action=argparse.BooleanOptionalAction, default=False,
                   help='Capture full sequential SAC updates on CUDA; requires fused optimizer, preserves update budget')
    p.add_argument('--optimizer-layout', action=argparse.BooleanOptionalAction, default=False,
                   help='Optimize CUDA Adam execution layout using original Torch math; requires learner graph and local CUDA compiler')
    p.add_argument('--self-collision-weight', type=float, default=0.,
                   help='Opt-in same-hand collision penalty; review proposal is 0.2, default remains 0')
    p.add_argument('--reward-config', type=Path, help='JSON with independent enabled/weight/params for every reward')
    p.add_argument('--joint-motion-limit-penalty', action=argparse.BooleanOptionalAction, default=None,
                   help='Penalize excessive physical target velocity/acceleration; does not clamp motion, omitted preserves reward JSON')
    p.add_argument('--finger-target-velocity-limit', type=float, default=None,
                   help='Positive finger target velocity threshold in rad/s; overrides reward JSON without enabling the penalty')
    p.add_argument('--finger-target-acceleration-limit', type=float, default=None,
                   help='Positive finger target acceleration threshold in rad/s^2; overrides reward JSON without enabling the penalty')
    p.add_argument('--wrist-target-velocity-limit', type=float, default=None,
                   help='Explicit wrist rotation target threshold in rad/s; requires an acceleration threshold in CLI or JSON')
    p.add_argument('--wrist-target-acceleration-limit', type=float, default=None,
                   help='Explicit wrist rotation target threshold in rad/s^2; requires a velocity threshold in CLI or JSON')
    p.add_argument('--joint-motion-full-penalty-excess', type=float, default=None,
                   help='Positive relative threshold excess reaching the full motion penalty, e.g. 0.2 for 20 percent')
    p.add_argument('--tcp-target-velocity-limit', type=float, default=None,
                   help='Raw TX/TY demand threshold in m/s; paired with TCP acceleration threshold')
    p.add_argument('--tcp-target-acceleration-limit', type=float, default=None,
                   help='Raw TX/TY demand threshold in m/s^2; paired with TCP velocity threshold')
    p.add_argument('--joint-motion-max-penalty', type=float, default=None,
                   help='Maximum motion penalty magnitude per control step; overrides reward JSON without enabling the penalty')
    p.add_argument('--anticipatory-fingering-time', type=float,
                   help='Future-fingering positioning horizon in seconds; 0 disables, omitted preserves reward JSON')
    p.add_argument('--anticipatory-fingering-reward', action=argparse.BooleanOptionalAction, default=None,
                   help='Include future-finger positioning in reward; disabling retains the configured future observation')
    p.add_argument('--anticipatory-current-priority', action=argparse.BooleanOptionalAction, default=None,
                   help='Preserve current fingering mean; future fingers fill only bounded residual credit, omitted preserves JSON')
    p.add_argument('--staged-fingering-reward', action=argparse.BooleanOptionalAction, default=None,
                   help='Use approach/alignment and actual key-travel fingering phases; overrides reward JSON')
    p.add_argument('--exact-note-timing', action=argparse.BooleanOptionalAction, default=None,
                   help='Required for physical-substep key reward; also retains original MIDI observations/diagnostics')
    p.add_argument('--upcoming-press-permission-time', type=float, default=None,
                   help='Seconds before an assigned onset to relax only safe distal requested-target geometry cost; default 0')
    p.add_argument('--humanoid-finger-gesture', action='store_true',
                   help='Reward idle fingers for matching their 20-joint-per-hand scene reset pose')
    p.add_argument('--enable-reward', action='append', default=[], choices=list(rewards.template()['terms']))
    p.add_argument('--disable-reward', action='append', default=[], choices=list(rewards.template()['terms']))
    p.add_argument('--allow-reward-change', action='store_true',
                   help='Allow different rewards only with --stage2-from/--finetune-from; fresh optimizer/replay/alpha')
    p.add_argument('--collision-diagnostics', action=argparse.BooleanOptionalAction, default=False,
                   help='Measure same/cross-hand collisions across physics substeps; no reward or physics change')
    p.add_argument('--control-limits', type=Path,
                   help='Explicit software target range/rate limits JSON; default retains original control')
    from ..mjlab_env.control_limits import add_rate_arguments
    add_rate_arguments(p)
    p.add_argument('--finger-clearance-config', type=Path,
                   help='Whole-finger geometry settings for diagnostics/reward and optional control; default is absent')
    p.add_argument('--idle-finger-protection', action=argparse.BooleanOptionalAction, default=None,
                   help='Explicitly enable/disable idle whole-finger target clearance; overrides geometry JSON')
    p.add_argument('--active-press-permission', action=argparse.BooleanOptionalAction, default=None,
                   help='Explicitly enable/disable actual-position permission for active finger commands; overrides geometry JSON')
    p.add_argument('--allow-control-change', action='store_true',
                   help='Allow changed software limits/geometry guard only with --finetune-from; fresh optimizer/replay/alpha')
    p.add_argument('--stage', type=int, choices=(1, 2), default=1)
    p.add_argument('--init-temperature', type=float, default=1.)
    p.add_argument('--actor-hidden', type=int, nargs='+', default=[256, 256, 256])
    p.add_argument('--discount', type=float, default=.8, help='SAC gamma; environment terminal discount is separate')
    p.add_argument('--actor-lr', type=float, help='Override the selected SAC preset actor learning rate')
    p.add_argument('--critic-lr', type=float, help='Override the selected SAC preset critic learning rate')
    p.add_argument('--temp-lr', type=float, help='Override the selected SAC preset temperature learning rate')
    p.add_argument('--source-config', type=Path, help='Complete source reconstruction recipe, provided by source launcher')
    p.add_argument('--preparation-time', type=float, default=0.,
                   help='Raised initial to origin height transition seconds')
    p.add_argument('--preparation-mode', choices=('fixed_v1','policy_v1'), default='fixed_v1',
                   help='policy_v1 includes controlled preparation in training; fixed_v1 preserves historical reset')
    p.add_argument('--preparation-lift', type=float, default=.02, help='World-Z initial lift in metres')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--vector-steps', type=int, default=20000)
    p.add_argument('--replay-vector-slots', type=int, default=1024)
    p.add_argument('--replay-capacity', type=int,
                   help='Exact transition capacity; overrides replay-vector-slots times num-envs')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--warmup-vector-steps', type=int, help='Default 300 in Stage1, 0 in Stage2')
    p.add_argument('--warmup-transitions', type=int,
                   help='Exact number of random action rows; overrides warmup-vector-steps, splitting a boundary vector')
    p.add_argument('--learning-starts', type=int, default=1000, help='Vector steps')
    p.add_argument('--learning-start-transitions', type=int,
                   help='First eligible transition (1-based); overrides learning-starts, with no updates before this row')
    p.add_argument('--replay-prefill-episodes', type=int, default=0,
                   help='Delay updates until each environment samples this many full prepared scores; retains random warmup and total sample budget')
    p.add_argument('--gradient-steps', type=int, default=32, help='Complete SAC updates per eligible vector step')
    p.add_argument('--updates-per-transition', type=float,
                   help='Override gradient-steps with a cumulative update budget per eligible transition')
    p.add_argument('--eval-interval', type=int, default=1000, help='Vector steps; 0 disables periodic evaluation')
    p.add_argument('--eval-episodes', type=int, default=1)
    p.add_argument('--eval-num-envs', type=int, default=1,
                   help='Independent evaluation worlds per batch; episodes and worst-episode gates are unchanged')
    p.add_argument('--eval-at-start', action=argparse.BooleanOptionalAction, default=False,
                   help='Evaluate transferred/random networks before training; save baseline best, do not count target streak')
    p.add_argument('--log-interval', type=int, default=10, help='Vector steps')
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True,
                   help='Show sample progress, throughput and ETA; --no-progress disables the progress display')
    p.add_argument('--save-interval', type=int, default=1000, help='Vector steps; final checkpoint always saved')
    p.add_argument('--save-forward', action=argparse.BooleanOptionalAction, default=False,
                   help='Save forward.pt once at the first periodic evaluation (not the initial/final evaluation)')
    p.add_argument('--save-best', action=argparse.BooleanOptionalAction, default=True,
                   help='Retain best.pt by worst-episode music F1; explicit key-underside limit also gates eligibility')
    p.add_argument('--best-f1-metric', choices=('music_only_f1_min', checkpoint.EXACT_MUSIC_F1_MIN),
                   default='music_only_f1_min',
                   help='Best-checkpoint criterion; exact original-clock mode requires complete physical key diagnostics')
    p.add_argument('--target-music-f1', type=float,
                   help='Opt-in stopping target for worst episode music-only frame F1; default disables stopping')
    p.add_argument('--target-micro-f1', type=float,
                   help='Worst episode music-only micro F1 target; defaults to target-music-f1 when stopping is enabled')
    p.add_argument('--target-exact-micro-f1', type=float,
                   help='Additional worst-episode original-clock physical-sample F1 stopping gate; requires target-music-f1')
    p.add_argument('--target-consecutive-evals', type=int, default=3,
                   help='Require at least this many consecutive complete qualifying evaluations (minimum 2)')
    p.add_argument('--target-max-preparation-wrong-frames', type=float, default=0.,
                   help='Maximum wrong-key preparation frames in any evaluated episode')
    p.add_argument('--target-max-hand-collision-frames', type=float, default=0.,
                   help='Maximum control frames with any same/cross-hand physics-substep contact in any episode')
    p.add_argument('--target-max-key-underside-substeps', type=int,
                   help='Optional maximum hand/key underside physics substeps in any episode for best eligibility and target stopping')
    p.add_argument('--max-episode-steps', type=int, help='Optional explicit episode cutoff')
    p.add_argument('--output-dir', type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--warm-start', type=Path, help='Restore learner; start fresh replay and physics')
    group.add_argument('--stage2-from', type=Path, help='Transfer Stage1 networks only; fresh optimizer/replay/alpha')
    group.add_argument('--finetune-from', type=Path,
                       help='Transfer same-contract networks at either stage; fresh optimizer/replay/alpha/RNG')
    p.add_argument('--inspect-only', action='store_true', help='Print effective configuration without allocating GPU physics')
    return p


def effective_config(args):
    import math
    layout = execution_layout(args.device, args.learner_device, args.collector_pipeline)
    from ..mjlab_env.finger_coupling import canonical as canonical_coupling, MODE, RANGE_MODE
    coupling_options = None
    if args.finger_coupling:
        coupling_options = dict(schema=1, mode=RANGE_MODE, reduce_observation=args.coupling_reduce_observation)
        if args.finger_coupling_angle is not None:
            coupling_options.update(mode=MODE, target_angle_deg=args.finger_coupling_angle)
        elif args.finger_coupling_angle_ranges is not None:
            coupling_options['angle_ranges_deg'] = json.loads(args.finger_coupling_angle_ranges.read_text())
    finger_coupling = canonical_coupling(coupling_options)
    if args.save_forward and args.eval_interval < 1:
        raise ValueError('--save-forward requires a positive --eval-interval')
    if args.learner_graph and not args.fused_optimizer:
        raise ValueError('--learner-graph requires --fused-optimizer; disable both to use unfused Adam')
    if args.optimizer_layout and not args.learner_graph:
        raise ValueError('--optimizer-layout requires --learner-graph')
    if any(not math.isfinite(v) or v < 0 for v in (args.preparation_time, args.preparation_lift)):
        raise ValueError('Preparation time and lift must be finite and nonnegative')
    from ..mjlab_env.arm_options import from_args as arm_from_args
    arm_follow = arm_from_args(args)
    if arm_follow is not None and not (args.disable_wrist_tz and args.disable_wrist_yaw):
        raise ValueError('--arms requires --disable-wrist-tz --disable-wrist-yaw for the current command layout')
    if args.preparation_time and not (args.disable_wrist_tz and args.disable_wrist_yaw and (args.no_arms or arm_follow is not None)):
        raise ValueError('Preparation currently requires --no-arms --disable-wrist-tz --disable-wrist-yaw')
    if args.disable_wrist_yaw and not args.disable_wrist_tz:
        raise ValueError("--disable-wrist-yaw requires --disable-wrist-tz")
    if not math.isfinite(args.self_collision_weight) or args.self_collision_weight < 0:
        raise ValueError('self_collision_weight must be finite and nonnegative')
    reward = rewards.validate(
        json.loads(args.reward_config.read_text()) if args.reward_config else rewards.template())
    if args.reward_config and args.self_collision_weight:
        raise ValueError('Use --reward-config OR --self-collision-weight, not both')
    if args.self_collision_weight:
        reward['terms']['self_collision_reward'].update(enabled=True,weight=args.self_collision_weight)
    if arm_follow is not None:
        reward['terms']['arm_motion_limit_penalty']['enabled'] = True
    motion = reward['terms']['joint_motion_limit_penalty']
    if args.joint_motion_limit_penalty is not None:
        motion['enabled'] = args.joint_motion_limit_penalty
    for argument, parameter in (
            ('finger_target_velocity_limit', 'finger_velocity_limit'),
            ('finger_target_acceleration_limit', 'finger_acceleration_limit'),
            ('wrist_target_velocity_limit', 'wrist_velocity_limit'),
            ('wrist_target_acceleration_limit', 'wrist_acceleration_limit'),
            ('tcp_target_velocity_limit', 'tcp_velocity_limit'),
            ('tcp_target_acceleration_limit', 'tcp_acceleration_limit'),
            ('joint_motion_full_penalty_excess', 'full_penalty_excess'),
            ('joint_motion_max_penalty', 'maximum_penalty_per_step')):
        value = getattr(args, argument)
        if value is not None:
            motion['params'][parameter] = value
    # Generic term switches take precedence over the dedicated default switch.
    if set(args.enable_reward) & set(args.disable_reward):
        raise ValueError('A reward cannot be both enabled and disabled on the command line')
    for name in args.enable_reward:
        reward['terms'][name]['enabled'] = True
    if args.humanoid_finger_gesture:
        if 'humanoid_finger_gesture_reward' in args.disable_reward:
            raise ValueError('--humanoid-finger-gesture conflicts with --disable-reward humanoid_finger_gesture_reward')
        reward['terms']['humanoid_finger_gesture_reward']['enabled'] = True
    for name in args.disable_reward:
        reward['terms'][name]['enabled'] = False
    performance_overrides = {name: getattr(args, name) for name in
        ('anticipatory_fingering_time', 'staged_fingering_reward', 'anticipatory_current_priority',
         'exact_note_timing', 'upcoming_press_permission_time',
         'anticipatory_fingering_reward')
        if getattr(args, name) is not None}
    if performance_overrides:
        from ..mjlab_env import performance_options
        # Preserve absent optional fields in old saved contracts; only explicit
        # CLI choices introduce new reward semantics into the serialized config.
        reward['performance'] = performance_options.canonical(reward.get('performance')) | performance_overrides
    reward = rewards.validate(reward)
    if args.exact_note_timing is False and reward['terms']['key_press_reward']['enabled']:
        raise ValueError('Current key reward requires --exact-note-timing; endpoint scoring was removed')
    if reward['terms']['arm_motion_limit_penalty']['enabled'] and arm_follow is None:
        raise ValueError('arm_motion_limit_penalty requires --arms')
    if args.allow_reward_change and not (args.stage2_from or args.finetune_from):
        raise ValueError('--allow-reward-change requires --stage2-from or --finetune-from (networks only)')
    if args.allow_control_change and not args.finetune_from:
        raise ValueError('--allow-control-change requires --finetune-from (networks only, fresh optimizer/replay)')
    from ..mjlab_env.control_limits import rate_overrides
    control_limits = rate_overrides(json.loads(args.control_limits.read_text()) if args.control_limits else None, args)
    from ..mjlab_env.finger_clearance import canonical as canonical_finger_clearance
    finger_clearance_options = (
        json.loads(args.finger_clearance_config.read_text()) if args.finger_clearance_config else None)
    protection_overrides = {name: getattr(args, name) for name in
                            ('idle_finger_protection', 'active_press_permission')
                            if getattr(args, name) is not None}
    if protection_overrides:
        # No implicit companion protection: a standalone CLI switch enables
        # only its named control. Existing JSON keeps its historical defaults.
        if finger_clearance_options is None:
            finger_clearance_options = dict(schema=1, mode='full_finger_clearance_v1',
                active_command_mode='active_contact_v1', idle_finger_protection=False,
                active_press_permission=False)
        finger_clearance_options.update(protection_overrides)
    finger_clearance = canonical_finger_clearance(finger_clearance_options)
    gesture = reward['terms']['humanoid_finger_gesture_reward']
    if reward.get('performance', {}).get('upcoming_press_permission_time', 0.):
        if (finger_clearance is None or finger_clearance.get('active_command_mode') != 'active_contact_v1'
                or finger_clearance.get('idle_finger_protection', True)
                or finger_clearance.get('active_press_permission', True)):
            raise ValueError('Upcoming press permission requires active_contact_v1 geometry with both physical projections disabled')
        if not gesture['enabled']:
            raise ValueError('Upcoming press permission requires the active humanoid gesture reward')
    if args.target_micro_f1 is not None and args.target_music_f1 is None:
        raise ValueError('--target-micro-f1 requires --target-music-f1')
    if args.target_exact_micro_f1 is not None and args.target_music_f1 is None:
        raise ValueError('--target-exact-micro-f1 requires --target-music-f1')
    if args.best_f1_metric == checkpoint.EXACT_MUSIC_F1_MIN or args.target_exact_micro_f1 is not None:
        performance = reward.get('performance', {})
        if not performance.get('exact_note_timing', False):
            raise ValueError('Exact F1 selection requires physical key sampling: enable exact-note-timing')
    if args.target_max_key_underside_substeps is not None:
        if args.target_max_key_underside_substeps < 0:
            raise ValueError('--target-max-key-underside-substeps must be nonnegative')
        if args.target_music_f1 is None:
            raise ValueError('--target-max-key-underside-substeps requires --target-music-f1')
        if finger_clearance is None or not args.collision_diagnostics:
            raise ValueError('Key underside target requires --finger-clearance-config and --collision-diagnostics')
    for key in ('target_music_f1', 'target_micro_f1', 'target_exact_micro_f1'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(key+' must be finite and within [0, 1]')
    for key in ('target_max_preparation_wrong_frames', 'target_max_hand_collision_frames'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(key+' must be finite and nonnegative')
    if args.target_consecutive_evals < 2:
        raise ValueError('target_consecutive_evals must be at least 2')
    if args.target_music_f1 is not None and (not args.eval_interval or not args.collision_diagnostics):
        raise ValueError('Target stopping requires periodic evaluation and --collision-diagnostics')
    if args.warmup_vector_steps is None:
        args.warmup_vector_steps = 300 if args.stage == 1 else 0
    for key in ('num_envs', 'vector_steps', 'replay_vector_slots', 'batch_size', 'gradient_steps', 'eval_episodes', 'eval_num_envs', 'log_interval'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('warmup_vector_steps', 'learning_starts', 'eval_interval', 'save_interval', 'replay_prefill_episodes'):
        if getattr(args, key) < 0:
            raise ValueError(f'{key} cannot be negative')
    for key in ('replay_capacity', 'learning_start_transitions'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if args.warmup_transitions is not None and args.warmup_transitions < 0:
        raise ValueError('warmup_transitions cannot be negative')
    if args.updates_per_transition is not None and (
            not math.isfinite(args.updates_per_transition) or args.updates_per_transition <= 0):
        raise ValueError('updates_per_transition must be positive and finite')
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        raise ValueError('max_episode_steps must be positive')
    capacity = args.replay_capacity if args.replay_capacity is not None else args.replay_vector_slots * args.num_envs
    if capacity < args.batch_size:
        raise ValueError('Replay transition capacity must be at least batch_size')
    if args.stage2_from and args.stage != 2:
        raise ValueError('--stage2-from requires --stage 2')
    if args.stage == 2 and not (args.stage2_from or args.finetune_from or args.warm_start or args.inspect_only):
        raise ValueError('Stage2 requires --stage2-from, --finetune-from or a Stage2 --warm-start')
    c = replace(SACConfig.paper(args.stage), actor_hidden=tuple(args.actor_hidden),
                gamma=args.discount, init_temperature=args.init_temperature)
    c = replace(c, **{name: getattr(args, name) for name in ('actor_lr', 'critic_lr', 'temp_lr')
                      if getattr(args, name) is not None})
    warmup = args.warmup_transitions if args.warmup_transitions is not None else args.warmup_vector_steps * args.num_envs
    start, ratio = _update_schedule(args)
    requested_start = (args.learning_start_transitions if args.learning_start_transitions is not None
                       else (max(1, args.learning_starts)-1)*args.num_envs+1)
    return dict(sac=asdict(c), stage=args.stage, num_envs=args.num_envs,
                execution_layout=layout,
                preparation=(dict(duration=args.preparation_time,lift=args.preparation_lift,trajectory='quintic_v1',
                                  **({'mode': args.preparation_mode} if args.preparation_mode != 'fixed_v1' else {}))
                             if args.preparation_time else None),
                action_profile=args.action_profile, no_arms=args.no_arms, disable_wrist_tz=args.disable_wrist_tz, disable_wrist_yaw=args.disable_wrist_yaw,
                hand_collision_detection_only=args.hand_collision_detection_only,
                self_collision_detection_only=args.self_collision_detection_only,
                wrist_tz_physics="joint_removed_v1" if args.disable_wrist_tz else "enabled",
                acceleration=dict(physics_graph=args.physics_graph, compile_learner=args.compile_learner,
                                  fused_optimizer=args.fused_optimizer, learner_graph=args.learner_graph,
                                  optimizer_layout=args.optimizer_layout, eval_metrics_graph=args.eval_metrics_graph),
                self_collision_weight=args.self_collision_weight,
                collision_diagnostics=args.collision_diagnostics,
                control_limits=control_limits,
                finger_coupling=finger_coupling,
                arm_follow=arm_follow,
                finger_clearance=finger_clearance,
                evaluation_execution=dict(num_envs=min(args.eval_num_envs, args.eval_episodes),
                    requested_num_envs=args.eval_num_envs, episodes=args.eval_episodes,
                    mode='independent_world_batches_v1'),
                evaluation_selection=dict(
                    save_best=args.save_best, best_metric=args.best_f1_metric,
                    eval_at_start=args.eval_at_start,
                    target_music_f1=args.target_music_f1,
                    target_micro_f1=(args.target_micro_f1 if args.target_micro_f1 is not None else args.target_music_f1),
                    **({'target_exact_micro_f1': args.target_exact_micro_f1}
                       if args.target_exact_micro_f1 is not None else {}),
                    consecutive_evaluations=args.target_consecutive_evals,
                    max_preparation_wrong_frames=args.target_max_preparation_wrong_frames,
                    max_hand_collision_frames=args.target_max_hand_collision_frames,
                    max_key_underside_substeps=args.target_max_key_underside_substeps,
                    require_complete_episodes=True, require_full_collision_coverage=True,
                    scope='Worst episode, excluding all preparation/buffer frames from music F1; empirical target, no convergence guarantee'),
                checkpoint_retention=dict(periodic_interval=args.save_interval, save_best=args.save_best,
                    forward_vector_step=args.eval_interval if args.save_forward else None, final=True),
                reward_config=reward,
                replay_capacity=capacity, batch_size=args.batch_size,
                warmup_vector_steps=warmup // args.num_envs,
                warmup_transitions=warmup, warmup_boundary_random_rows=warmup % args.num_envs,
                learning_starts_vector_steps=(start + args.num_envs - 1) // args.num_envs,
                learning_start_transition_1_based=start,
                requested_learning_start_transition_1_based=requested_start,
                replay_prefill=dict(episodes=args.replay_prefill_episodes,
                    episode_steps=None, required_vector_steps=None,
                    required_transitions=None, resolved=not bool(args.replay_prefill_episodes)),
                gradient_steps_per_vector_step=float(ratio * args.num_envs),
                steady_updates_per_transition=float(ratio),
                steady_sample_ratio=float(ratio * args.batch_size),
                scheduling=dict(
                    transition_overrides=dict(replay_capacity=args.replay_capacity,
                                              warmup_transitions=args.warmup_transitions,
                                              learning_start_transitions=args.learning_start_transitions,
                                              updates_per_transition=args.updates_per_transition),
                    update_ratio_fraction=[ratio.numerator, ratio.denominator],
                    update_rule='floor(eligible_rows_after * ratio) - floor(eligible_rows_before * ratio); updates require replay readiness',
                    parallel_caveat='All worlds are collected before the vector update block; this is not single-environment interleaving'),
                requested_vector_steps=args.vector_steps, requested_transitions=args.vector_steps * args.num_envs,
                scope='SAC stage preset with explicit user CLI overrides; user-selected reward formulas. Missing mappings must be supplied explicitly; paper Stage2 shaping is undisclosed.',
                critic_architecture='256x3 retained from baseline; paper does not fully specify critic hidden layers',
                checkpoint_kind='warm restart: learner/optimizer/RNG only; fresh replay, physics, score and scheduling')


def _update_schedule(args):
    """Return the first eligible row and an exact rational update ratio."""
    start = (args.learning_start_transitions if args.learning_start_transitions is not None
             else (max(1, args.learning_starts) - 1) * args.num_envs + 1)
    start = max(start, getattr(args, '_prefill_learning_start', 1))
    ratio = (Fraction(str(args.updates_per_transition)) if args.updates_per_transition is not None
             else Fraction(args.gradient_steps, args.num_envs))
    return start, ratio


def resolve_replay_prefill(args, config, episode_steps):
    """Resolve full-score coverage from the actual prepared runtime task.

    The first eligible row follows the entire prefill, so its transitions do
    not accrue a deferred update backlog. Random action count stays unchanged;
    all subsequent prefill rows come from the stochastic actor.
    """
    if not args.replay_prefill_episodes:
        return
    if not isinstance(episode_steps, int) or isinstance(episode_steps, bool) or episode_steps < 1:
        raise ValueError('Replay prefill requires a positive full-score episode length')
    if args.max_episode_steps is not None and args.max_episode_steps < episode_steps:
        raise ValueError('Replay prefill requires complete scores; max_episode_steps truncates the score')
    if config['replay_capacity'] < episode_steps * args.num_envs:
        raise ValueError('Replay capacity must retain at least one full score per environment for replay prefill')
    steps = args.replay_prefill_episodes * episode_steps
    transitions = steps * args.num_envs
    args._prefill_learning_start = transitions + 1
    start, _ = _update_schedule(args)
    config['learning_start_transition_1_based'] = start
    config['learning_starts_vector_steps'] = (start + args.num_envs - 1)//args.num_envs
    config['replay_prefill'] = dict(episodes=args.replay_prefill_episodes,
        episode_steps=episode_steps, required_vector_steps=steps,
        required_transitions=transitions, resolved=True,
        requested_learning_start_transition_1_based=config['requested_learning_start_transition_1_based'],
        effective_learning_start_transition_1_based=start)


def updates_this_step(args, step):
    """Difference of cumulative integer budgets, avoiding floating-point drift."""
    start, ratio = _update_schedule(args)
    before = max(0, (step - 1) * args.num_envs - start + 1)
    after = max(0, step * args.num_envs - start + 1)
    return (after * ratio.numerator // ratio.denominator
            - before * ratio.numerator // ratio.denominator)


def warmup_action_count(args, step):
    """Random rows first, policy rows second when warmup ends inside a vector."""
    count = (args.warmup_transitions if args.warmup_transitions is not None
             else args.warmup_vector_steps * args.num_envs)
    return min(args.num_envs, max(0, count - (step - 1) * args.num_envs))


@torch.no_grad()
def training_action(agent, scene, obs, args, step, generator):
    count = warmup_action_count(args, step)
    if not count:
        return agent.act(obs)
    random_action = scene.action_low + torch.rand(
        (count, scene.action_low.numel()), device=obs.device, generator=generator
    ) * (scene.action_high - scene.action_low)
    if count == args.num_envs:
        return random_action
    return torch.cat((random_action, agent.act(obs[count:])), dim=0)


class MotionMetrics:
    """Observe control-cycle endpoints only, with no changes to the simulation."""
    def __init__(self, scene):
        self.scene = scene
        self.groups = []
        self.names = []
        names = scene.bundle.manifest['action_names']
        active = list(zip(scene.bundle.active_action_indices, scene.bundle.action_ids))
        selectors = (
            ('wrist_tx', lambda name: name.endswith('_Wrist_TX'), 'm_s', 'm'),
            ('wrist_ty', lambda name: name.endswith('_Wrist_TY'), 'm_s', 'm'),
            ('finger', lambda name: '_Wrist_' not in name, 'rad_s', 'rad'),
            ('wrist_pitch', lambda name: name.endswith('_Wrist_Pitch'), 'rad_s', 'rad'),
        )
        for label, select, speed_unit, target_unit in selectors:
            actuator_ids = [int(act) for full, act in active if select(names[full])]
            if not actuator_ids:
                continue
            # V3.2 is directly joint-actuated. Do not interpret tendon IDs as
            # joint IDs if a future morphology adds a different transmission.
            if any(int(scene.model.actuator_trntype[act]) != 0 for act in actuator_ids):
                raise ValueError('Motion metrics require direct joint actuators')
            dofs = [int(scene.model.jnt_dofadr[scene.model.actuator_trnid[act, 0]])
                    for act in actuator_ids]
            self.groups.append((torch.tensor(dofs, device=scene.device),
                                torch.tensor(actuator_ids, device=scene.device)))
            self.names.extend((f'motion_max_{label}_speed_{speed_unit}',
                               f'motion_max_{label}_target_jump_{target_unit}'))
        self.maximum = torch.zeros(len(self.names), device=scene.device)
        self.previous_ctrl = None

    def reset(self):
        # Compare each episode's first target against the actual reset control.
        self.previous_ctrl = self.scene.data.ctrl.clone()

    def sample(self, active_worlds=None):
        data = self.scene.data
        difference = (data.ctrl - self.previous_ctrl).abs()
        values = []
        for dofs, actuators in self.groups:
            worlds = slice(None, active_worlds)
            values.extend((data.qvel[worlds][:, dofs].abs().amax(), difference[worlds][:, actuators].amax()))
        if values:
            self.maximum.copy_(torch.maximum(self.maximum, torch.stack(values)))
        self.previous_ctrl.copy_(data.ctrl)

    def result(self):
        return dict(zip(self.names, self.maximum.cpu().tolist())) | {
            'motion_sample_hz': 1.0 / self.scene.bundle.manifest['control_dt']}


class SubstepKeyMetrics:
    """Keep 200 Hz event counts separate from the historical endpoint F1."""

    keys = ('wrong_key_substeps', 'wrong_key_key_substeps', 'substep_key_counts', 'substep_count')

    def __init__(self, device, preparation_steps, music_start_steps, decimation, music_end_step=None, num_envs=1):
        self.device = device
        self.num_envs = num_envs
        self.preparation_steps = preparation_steps
        self.music_start_steps = music_start_steps
        self.music_end_step = music_end_step
        self.decimation = decimation
        # Per phase: covered control frames, physics samples, wrong substeps,
        # wrong key-substeps, TP, FP, FN. Never infer missing samples as zero.
        self.sums = torch.zeros((num_envs, 4, 7), device=device, dtype=torch.float64)
        self.valid = torch.ones(num_envs, device=device, dtype=torch.bool)

    def sample(self, info):
        if not all(key in info for key in (*self.keys, 'score_index')):
            return
        wrong, key_wrong, counts, sampled = [torch.as_tensor(info[key], device=self.device)
                                            for key in self.keys]
        wrong, key_wrong, sampled = (value.reshape(-1) for value in (wrong, key_wrong, sampled))
        counts = counts.reshape(-1, 3)
        values = torch.cat((torch.ones((self.num_envs, 1), device=self.device),
                            torch.stack((sampled, wrong, key_wrong), dim=-1), counts), dim=-1).double()
        valid = (torch.isfinite(values).all(-1) & (values >= 0).all(-1)
                 & (values == values.floor()).all(-1) & (sampled > 0)
                 & (wrong <= sampled) & (key_wrong >= wrong) & (key_wrong <= 88*sampled)
                 & (counts[:, 1] == key_wrong) & (counts <= 88*sampled[:, None]).all(-1))
        if self.decimation is not None:
            valid &= sampled == self.decimation
        self.valid &= valid
        index = info['score_index'].reshape(-1)
        phase = ((index >= self.preparation_steps).long() + (index >= self.music_start_steps).long())
        if self.music_end_step is not None:
            phase = torch.where(index >= self.music_end_step, 3, phase)
        if 'substep_phase_counts' in info:
            # Exact MIDI mode assigns each post-integration sample to its true
            # phase, including the sample on the preparation/music boundary.
            phase_counts = info['substep_phase_counts'].double()
            if phase_counts.shape != (self.num_envs, 4, 6):
                raise ValueError('Invalid exact substep phase counts')
            self.valid &= (torch.isfinite(phase_counts).all((1, 2))
                           & (phase_counts >= 0).all((1, 2))
                           & (phase_counts == phase_counts.floor()).all((1, 2))
                           & (phase_counts.sum(1) == values[:, 1:]).all(-1))
            self.sums[:, :, 0].scatter_add_(1, phase[:, None], torch.ones_like(phase[:, None], dtype=torch.float64))
            self.sums[:, :, 1:] += torch.nan_to_num(phase_counts)
            return
        # Invalid counters stay invalid without allowing NaNs into JSON reports.
        self.sums.scatter_add_(1, phase[:, None, None].expand(-1, 1, 7),
                               torch.nan_to_num(values)[:, None, :])

    def result(self, length, world=0):
        phases = self.sums[world].cpu().tolist()
        total = [sum(phase[i] for phase in phases) for i in range(7)]
        valid = bool(self.valid[world].item())
        covered = valid and total[0] == length
        output = dict(substep_key_diagnostics_coverage=total[0]/max(length, 1),
                      substep_key_diagnostics_valid=valid and covered,
                      substep_key_coverage=(total[1]/(length*self.decimation)
                                            if length and self.decimation else None))
        for prefix, values in (('', total), ('music_only_', phases[2]), ('preparation_', phases[0]),
                               ('release_tail_', phases[3])):
            _, samples, wrong, key_wrong, tp, fp, fn = values
            output.update({prefix+'wrong_key_substeps': wrong if covered else None,
                           prefix+'wrong_key_key_substeps': key_wrong if covered else None,
                           prefix+'substep_key_micro_f1': 2*tp/max(2*tp+fp+fn, 1) if covered and samples else None,
                           prefix+'substep_true_positive_count': tp if covered else None,
                           prefix+'substep_false_positive_count': fp if covered else None,
                           prefix+'substep_false_negative_count': fn if covered else None})
        return output


@torch.no_grad()
def evaluate(agent, env, episodes=1, *, metrics_graph=False):
    num_envs = getattr(env, 'num_envs', 1)
    if episodes < 1 or num_envs < 1:
        raise ValueError('Evaluation requires positive episodes and independent worlds')
    from ..mjlab_env.task_logic import binary_metrics
    from .note_timing_metrics import NoteTimingMetrics
    rows = []
    total_counts = torch.zeros(3, device=agent.device, dtype=torch.int64)
    motion = MotionMetrics(env.scene) if hasattr(env, 'scene') else None
    preparation_steps = getattr(env, 'preparation_steps', 0)
    manifest = getattr(getattr(env, 'bundle', None), 'manifest', {})
    music_start_steps = max(preparation_steps, round(
        manifest.get('contract', {}).get('environment', {}).get('initial_buffer_time', 0)
        / manifest.get('control_dt', 1)))
    release_tail_steps = getattr(env, 'release_tail_steps', manifest.get('release_tail_steps', 0))
    music_end_step = (getattr(env, 'music_end_step',
                     getattr(getattr(env, 'bundle', None), 'music_end_step', env.logic.length))
                     if release_tail_steps else None)
    control_hz = 1 / manifest['control_dt'] if manifest.get('control_dt') else None
    physics_hz = 1 / manifest['physics_dt'] if manifest.get('physics_dt') else None
    decimation = getattr(getattr(env, 'scene', None), 'decimation', None)
    if decimation is None and control_hz and physics_hz:
        decimation = round(physics_hz/control_hz)
    diagnostic_keys = ('self_collision_pairs', 'inter_hand_collision_pairs',
                       'self_collision_substeps', 'inter_hand_collision_substeps',
                       'collision_physics_substeps', 'hand_collision_substeps')
    key_contact_keys = ('key_underside_substeps', 'key_side_or_under_substeps',
                        'key_non_distal_substeps', 'key_contact_physics_substeps')
    controls = getattr(getattr(env, 'scene', None), 'control_limits', None)
    controlled_channel_count = len(controls['channels']) if controls else 0
    clearance = getattr(getattr(env, 'scene', None), 'finger_clearance', None)
    coupling = getattr(getattr(env, 'scene', None), 'finger_coupling', None)
    clearance_keys = ('finger_projection_mask', 'finger_fallback_mask', 'finger_actual_deficit_m',
                      'finger_requested_deficit_m', 'finger_applied_deficit_m')
    for batch_start in range(0, episodes, num_envs):
        batch_episodes = min(num_envs, episodes - batch_start)
        obs = env.reset()
        substep_metrics = SubstepKeyMetrics(agent.device, preparation_steps, music_start_steps,
                                           decimation, music_end_step, num_envs=num_envs)
        coupling_metrics = CouplingMetrics(agent.device, num_envs, bool(coupling))
        note_timing_metrics = NoteTimingMetrics(env)
        batch_phase_sums = torch.zeros((num_envs, 4, 7), device=agent.device)
        batch_collisions = torch.zeros((num_envs, 10), device=agent.device)
        batch_diagnostic_valid = torch.ones(num_envs, device=agent.device, dtype=torch.bool)
        batch_key_contacts = torch.zeros((num_envs, 8), device=agent.device)
        batch_key_contact_valid = torch.ones(num_envs, device=agent.device, dtype=torch.bool)
        batch_control_counts = torch.zeros((num_envs, 5), device=agent.device)
        batch_clearance_counts = torch.zeros((num_envs, 7), device=agent.device)
        batch_clearance_maxima = torch.zeros((num_envs, 3), device=agent.device)
        batch_clearance_valid = torch.ones(num_envs, device=agent.device, dtype=torch.bool)
        ones = torch.ones(num_envs, device=agent.device)
        have_frame_metrics = False
        if motion is not None:
            motion.reset()
        def sample_metrics(info):
            nonlocal have_frame_metrics
            nonlocal batch_control_counts
            nonlocal batch_clearance_valid
            nonlocal batch_clearance_counts
            nonlocal batch_diagnostic_valid
            nonlocal batch_collisions
            nonlocal batch_key_contact_valid
            nonlocal batch_key_contacts
            substep_metrics.sample(info)
            coupling_metrics.sample(info)
            if motion is not None:
                motion.sample(active_worlds=batch_episodes)
            if controlled_channel_count and all(key in info for key in ('control_clip_mask', 'control_slew_mask')):
                clip, slew = info['control_clip_mask'], info['control_slew_mask']
                batch_control_counts += torch.stack((ones, clip.any(-1).float(), slew.any(-1).float(),
                                                     clip.sum(-1).float(), slew.sum(-1).float()), dim=-1)
            if clearance and all(key in info for key in clearance_keys):
                projection, fallback, actual, requested, applied = [info[key] for key in clearance_keys]
                deficits = torch.stack((actual, requested, applied), dim=1)
                batch_clearance_valid &= torch.isfinite(deficits).all(dim=(1, 2)) & (deficits >= 0).all(dim=(1, 2))
                batch_clearance_counts += torch.stack((ones,
                    projection.any(-1).float(), fallback.any(-1).float(), projection.sum(-1).float(), fallback.sum(-1).float(),
                    (actual > 1e-6).any(-1).float(), (applied > 1e-6).any(-1).float()), dim=-1)
                batch_clearance_maxima.copy_(torch.maximum(batch_clearance_maxima, deficits.amax(dim=-1)))
            frame_keys = ('goal_applied', 'key_activation', 'score_index')
            if all(key in info for key in frame_keys):
                have_frame_metrics = True
                frame_metrics, counts = binary_metrics(info['goal_applied'][:, :88], info['key_activation'])
                # Frame metrics follow the same pre-action grid as rewards.
                phase_index = info['score_index']
                phase = ((phase_index >= preparation_steps).long()
                         + (phase_index >= music_start_steps).long())
                if music_end_step is not None:
                    phase = torch.where(phase_index >= music_end_step, 3, phase)
                tp, fp, fn = counts.unbind(-1)
                values = torch.stack((ones, frame_metrics[:, 2], (fp > 0).float(), fp.float(),
                                      tp.float(), fp.float(), fn.float()), dim=-1)
                batch_phase_sums.scatter_add_(1, phase[:, None, None].expand(-1, 1, 7), values[:, None, :])
            elif music_start_steps or release_tail_steps:
                raise ValueError('Cannot calculate music-only F1 without frame-level target/activation/index')
            if all(key in info for key in diagnostic_keys):
                own, cross, own_steps, cross_steps, sampled, hand_steps = [
                    torch.as_tensor(info[key], device=agent.device).reshape(-1) for key in diagnostic_keys]
                valid = (torch.isfinite(torch.stack((own, cross, own_steps, cross_steps, sampled, hand_steps), dim=-1)).all(-1)
                         & (own >= 0) & (cross >= 0) & (own_steps >= 0) & (cross_steps >= 0)
                         & (sampled > 0) & (own_steps <= sampled) & (cross_steps <= sampled)
                         & (hand_steps >= own_steps) & (hand_steps >= cross_steps) & (hand_steps <= sampled))
                if decimation is not None:
                    valid &= sampled == decimation
                batch_diagnostic_valid &= valid
                batch_collisions += torch.stack((ones,
                    ((own > 0) | (cross > 0)).float(), (own > 0).float(), (cross > 0).float(),
                    own.float(), cross.float(), own_steps.float(), cross_steps.float(), sampled.float(), hand_steps.float()), dim=-1)
            if all(key in info for key in key_contact_keys):
                under, side, non_distal, sampled = [torch.as_tensor(info[key], device=agent.device).reshape(-1)
                                                   for key in key_contact_keys]
                values = torch.stack((under, side, non_distal, sampled), dim=-1)
                valid = (torch.isfinite(values).all(-1) & (values >= 0).all(-1) & (sampled > 0)
                         & (under <= side) & (side <= sampled) & (non_distal <= sampled)
                         & (values == values.floor()).all(-1))
                if decimation is not None:
                    valid &= sampled == decimation
                batch_key_contact_valid &= valid
                batch_key_contacts += torch.stack((ones,
                    (under > 0).float(), (side > 0).float(), (non_distal > 0).float(),
                    under.float(), side.float(), non_distal.float(), sampled.float()), dim=-1)
        statistic_graph = None
        graph_keys = (*diagnostic_keys, *key_contact_keys, *clearance_keys,
            *SubstepKeyMetrics.keys, *CouplingMetrics.value_keys, 'coupling_active',
            'control_clip_mask', 'control_slew_mask', 'goal_applied', 'key_activation',
            'score_index', 'substep_phase_counts')
        graph_state = [batch_phase_sums, batch_collisions, batch_diagnostic_valid,
            batch_key_contacts, batch_key_contact_valid, batch_control_counts,
            batch_clearance_counts, batch_clearance_maxima, batch_clearance_valid,
            substep_metrics.sums, substep_metrics.valid, coupling_metrics.frames,
            coupling_metrics.valid, coupling_metrics.counts, coupling_metrics.sums,
            coupling_metrics.error_maxima]
        if motion is not None:
            graph_state += [motion.maximum, motion.previous_ctrl]
        for _ in range(env.logic.length):
            transition = env.step(agent.act(obs, deterministic=True))
            obs = transition.observation
            info = transition.info
            if metrics_graph:
                if statistic_graph is None:
                    from .evaluation_graph import EvaluationMetricGraph
                    statistic_graph = EvaluationMetricGraph(sample_metrics, info, graph_state, keys=graph_keys)
                statistic_graph.sample(info)
            else:
                sample_metrics(info)
            sampling = getattr(getattr(env, 'scene', None), 'reward_sampling', None)
            note_timing_metrics.sample(getattr(sampling, 'key_activation_history', None))
            # PianoTaskEnv ends on score completion or a shared time limit; wrong
            # keys never end an episode. Reject changed termination semantics
            # instead of stepping a completed world or counting it twice.
            if transition.done.any():
                if not transition.done.all():
                    raise ValueError('Batched evaluation requires synchronized score completion/time limits')
                break
        for world in range(batch_episodes):
            phase_sums = batch_phase_sums[world]
            collisions = batch_collisions[world]
            diagnostic_valid = batch_diagnostic_valid[world]
            key_contacts = batch_key_contacts[world]
            key_contact_valid = batch_key_contact_valid[world]
            control_counts = batch_control_counts[world]
            clearance_counts = batch_clearance_counts[world]
            clearance_maxima = batch_clearance_maxima[world]
            clearance_valid = batch_clearance_valid[world]
            names = ('precision', 'recall', 'f1', 'sustain_precision', 'sustain_recall', 'sustain_f1')
            row = dict(zip(names, info['episode_metrics'][world].cpu().tolist()))
            row.update(episode=batch_start + world, return_=info['episode_return'][world].item(),
                       length=info['episode_steps'][world].item(), completed=int(transition.terminated[world].item()),
                       truncated=int(transition.truncated[world].item()),
                       key_frame_micro_f1=info['key_frame_micro_f1'][world].item())
            total_counts += env.logic.key_counts[world]
            row['return'] = row.pop('return_')
            row.update(substep_metrics.result(row['length'], world))
            row.update(coupling_metrics.result(row['length'], world))
            row.update(note_timing_metrics.result(world))
            control_frames, clip_frames, slew_frames, clipped, slewed = control_counts.cpu().tolist()
            control_covered = control_frames == row['length']
            row.update(control_limits_enabled=bool(controls),
                       control_limits_diagnostics_coverage=control_frames/max(row['length'], 1) if controls else None,
                       control_clipped_frames=clip_frames if control_covered else None,
                       control_slew_limited_frames=slew_frames if control_covered else None,
                       control_clipped_fraction=clipped/(row['length']*controlled_channel_count) if control_covered and controlled_channel_count else None,
                       control_slew_limited_fraction=slewed/(row['length']*controlled_channel_count) if control_covered and controlled_channel_count else None)
            guard_frames, projected_frames, fallback_frames, projected, fallen_back, actual_bad, applied_bad = clearance_counts.cpu().tolist()
            guard_covered = guard_frames == row['length'] and bool(clearance_valid.item())
            actual_max, requested_max, applied_max = clearance_maxima.cpu().tolist()
            row.update(finger_clearance_enabled=bool(clearance),
                       finger_clearance_diagnostics_valid=bool(clearance_valid.item()) if clearance else None,
                       finger_clearance_diagnostics_coverage=guard_frames/max(row['length'], 1) if clearance else None,
                       finger_clearance_projection_frames=projected_frames if guard_covered else None,
                       finger_clearance_fallback_frames=fallback_frames if guard_covered else None,
                       finger_clearance_projection_fraction=projected/(row['length']*10) if guard_covered else None,
                       finger_clearance_fallback_fraction=fallen_back/(row['length']*10) if guard_covered else None,
                       finger_clearance_actual_deficit_frames=actual_bad if guard_covered else None,
                       finger_clearance_applied_deficit_frames=applied_bad if guard_covered else None,
                       finger_clearance_actual_max_deficit_m=actual_max if guard_covered else None,
                       finger_clearance_requested_max_deficit_m=requested_max if guard_covered else None,
                       finger_clearance_applied_max_deficit_m=applied_max if guard_covered else None)
            prep, extra, music, tail = phase_sums.cpu().tolist()
            if not have_frame_metrics:
                # Compatibility with minimal non-prefixed evaluation adapters only.
                music = [row['length'], row['f1'] * row['length'], 0, 0,
                         *env.logic.key_counts[world].cpu().tolist()]
            row.update(preparation_frames=prep[0], preparation_wrong_key_frames=prep[2],
                       preparation_wrong_key_count=prep[3], extra_buffer_frames=extra[0], music_frames=music[0],
                       music_only_f1=music[1]/music[0] if music[0] else None,
                       music_only_micro_f1=2*music[4]/max(2*music[4]+music[5]+music[6], 1) if music[0] else None,
                       music_true_positive_count=music[4], music_false_positive_count=music[5],
                       music_false_negative_count=music[6])
            tail_complete = bool(row['completed'] and tail[0] == release_tail_steps)
            row.update(release_tail_frames=tail[0], release_tail_wrong_key_frames=tail[2],
                       release_tail_wrong_key_count=tail[3],
                       release_tail_complete=tail_complete if release_tail_steps else None,
                       release_tail_end_active_keys=(int(info['key_activation'][world].sum().item())
                           if release_tail_steps and tail_complete and have_frame_metrics else None))
            sampled_frames, hand, own, cross, own_pairs, cross_pairs, own_steps, cross_steps, sampled_steps, hand_steps = collisions.cpu().tolist()
            covered = bool(diagnostic_valid.item()) and sampled_frames == row['length']
            row.update(collision_diagnostics_coverage=sampled_frames/max(row['length'], 1),
                       collision_diagnostics_valid=bool(diagnostic_valid.item()),
                       collision_substep_coverage=(sampled_steps/(row['length']*decimation)
                                                  if decimation and row['length'] else None),
                       hand_collision_frames=hand if covered else None,
                       self_collision_frames=own if covered else None,
                       inter_hand_collision_frames=cross if covered else None,
                       self_collision_pair_frames=own_pairs if covered else None,
                       inter_hand_collision_pair_frames=cross_pairs if covered else None,
                       self_collision_substeps=own_steps if covered else None,
                       inter_hand_collision_substeps=cross_steps if covered else None,
                       hand_collision_substeps=hand_steps if covered else None)
            sampled_frames,under_frames,side_frames,non_distal_frames,under_steps,side_steps,non_distal_steps,sampled_steps=key_contacts.cpu().tolist()
            covered=bool(key_contact_valid.item()) and sampled_frames==row['length']
            row.update(key_contact_diagnostics_coverage=sampled_frames/max(row['length'],1),
                       key_contact_diagnostics_valid=bool(key_contact_valid.item()),
                       key_contact_substep_coverage=(sampled_steps/(row['length']*decimation)
                                                    if decimation and row['length'] else None),
                       key_underside_frames=under_frames if covered else None,
                       key_side_or_under_frames=side_frames if covered else None,
                       key_non_distal_frames=non_distal_frames if covered else None,
                       key_underside_substeps=under_steps if covered else None,
                       key_side_or_under_substeps=side_steps if covered else None,
                       key_non_distal_substeps=non_distal_steps if covered else None)
            rows.append(row)
    result = {key: sum(row[key] for row in rows)/episodes for key in
              (*names, 'return', 'length', 'completed', 'truncated', 'preparation_frames',
               'preparation_wrong_key_frames', 'preparation_wrong_key_count', 'extra_buffer_frames', 'music_frames',
               'release_tail_frames', 'release_tail_wrong_key_frames', 'release_tail_wrong_key_count')}
    tp, fp, fn = total_counts.cpu().tolist()
    result['key_frame_micro_f1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    result.update(completed_episodes=sum(row['completed'] for row in rows),
                  truncated_episodes=sum(row['truncated'] for row in rows), evaluation_episodes=episodes,
                  episode_statistics=rows, evaluation_action='deterministic_actor_mean',
                  evaluation_num_envs=num_envs, evaluation_batches=(episodes + num_envs - 1)//num_envs,
                  music_start_frame=music_start_steps, music_end_frame=music_end_step,
                  release_tail_steps=release_tail_steps, metric_sample_hz=control_hz,
                  collision_control_sample_hz=control_hz, collision_physics_sample_hz=physics_hz,
                  collision_diagnostics_coverage=min(row['collision_diagnostics_coverage'] for row in rows),
                  collision_diagnostics_valid=all(row['collision_diagnostics_valid'] for row in rows),
                  collision_substep_coverage=(min(row['collision_substep_coverage'] for row in rows)
                      if all(row['collision_substep_coverage'] is not None for row in rows) else None))
    result.update(control_limits_enabled=bool(controls), control_limited_channel_count=controlled_channel_count)
    result.update(substep_key_diagnostics_coverage=min(row['substep_key_diagnostics_coverage'] for row in rows),
                  substep_key_diagnostics_valid=all(row['substep_key_diagnostics_valid'] for row in rows),
                  substep_key_coverage=(min(row['substep_key_coverage'] for row in rows)
                      if all(row['substep_key_coverage'] is not None for row in rows) else None),
                  substep_key_sample_hz=physics_hz)
    for prefix in ('', 'music_only_', 'preparation_', 'release_tail_'):
        for suffix in ('wrong_key_substeps', 'wrong_key_key_substeps'):
            key = prefix + suffix
            present = all(row[key] is not None for row in rows)
            result[key] = sum(row[key] for row in rows)/episodes if present else None
            result[key+'_max'] = max(row[key] for row in rows) if present else None
        key = prefix+'substep_key_micro_f1'
        present = all(row[key] is not None for row in rows)
        counts = [sum(row[prefix+'substep_'+name+'_count'] for row in rows)
                  for name in ('true_positive', 'false_positive', 'false_negative')] if present else None
        result[key] = 2*counts[0]/max(2*counts[0]+counts[1]+counts[2], 1) if present else None
        result[key+'_min'] = min(row[key] for row in rows) if present else None
    result['release_tail_complete'] = (all(row['release_tail_complete'] for row in rows)
                                       if release_tail_steps else None)
    for key in ('release_tail_wrong_key_frames', 'release_tail_wrong_key_count', 'release_tail_end_active_keys'):
        present = all(row[key] is not None for row in rows)
        result[key] = sum(row[key] for row in rows)/episodes if present else None
        result[key+'_max'] = max(row[key] for row in rows) if present else None
    result.update(key_contact_diagnostics_coverage=min(row['key_contact_diagnostics_coverage'] for row in rows),
                  key_contact_diagnostics_valid=all(row['key_contact_diagnostics_valid'] for row in rows),
                  key_contact_substep_coverage=(min(row['key_contact_substep_coverage'] for row in rows)
                      if all(row['key_contact_substep_coverage'] is not None for row in rows) else None),
                  key_contact_physics_sample_hz=physics_hz,
                  key_contact_diagnostics_contract=getattr(getattr(getattr(env,'scene',None),
                      'key_contact_diagnostics',None),'contract',None))
    for key in ('key_underside_frames','key_side_or_under_frames','key_non_distal_frames',
                'key_underside_substeps','key_side_or_under_substeps','key_non_distal_substeps'):
        present=all(row[key] is not None for row in rows)
        result[key]=sum(row[key] for row in rows)/episodes if present else None
        result[key+'_max']=max(row[key] for row in rows) if present else None
    result.update(finger_clearance_enabled=bool(clearance),
                  finger_clearance_diagnostics_valid=(all(row['finger_clearance_diagnostics_valid'] for row in rows)
                                                     if clearance else None))
    for key in ('finger_clearance_diagnostics_coverage', 'finger_clearance_projection_frames',
                'finger_clearance_fallback_frames', 'finger_clearance_projection_fraction',
                'finger_clearance_fallback_fraction', 'finger_clearance_actual_deficit_frames',
                'finger_clearance_applied_deficit_frames', 'finger_clearance_actual_max_deficit_m',
                'finger_clearance_requested_max_deficit_m', 'finger_clearance_applied_max_deficit_m'):
        present = all(row[key] is not None for row in rows)
        result[key] = sum(row[key] for row in rows)/episodes if present else None
        result[key+'_max'] = max(row[key] for row in rows) if present else None
    for key in ('control_limits_diagnostics_coverage', 'control_clipped_frames', 'control_slew_limited_frames',
                'control_clipped_fraction', 'control_slew_limited_fraction'):
        present = all(row[key] is not None for row in rows)
        result[key] = sum(row[key] for row in rows)/episodes if present else None
        result[key+'_max'] = max(row[key] for row in rows) if present else None
    music_frames = sum(row['music_frames'] for row in rows)
    result['music_only_f1'] = (sum(row['music_only_f1'] * row['music_frames'] for row in rows)/music_frames
                              if all(row['music_only_f1'] is not None for row in rows) and music_frames else None)
    mtp, mfp, mfn = (sum(row[key] for row in rows) for key in
                     ('music_true_positive_count', 'music_false_positive_count', 'music_false_negative_count'))
    result['music_only_micro_f1'] = 2*mtp/max(2*mtp+mfp+mfn, 1) if music_frames else None
    for key in ('music_only_f1', 'music_only_micro_f1'):
        result[key+'_min'] = min(row[key] for row in rows) if all(row[key] is not None for row in rows) else None
    result['preparation_wrong_key_frames_max'] = max(row['preparation_wrong_key_frames'] for row in rows)
    for key in ('hand_collision_frames', 'self_collision_frames', 'inter_hand_collision_frames',
                'self_collision_pair_frames', 'inter_hand_collision_pair_frames',
                'self_collision_substeps', 'inter_hand_collision_substeps', 'hand_collision_substeps'):
        present = all(row[key] is not None for row in rows)
        result[key] = sum(row[key] for row in rows)/episodes if present else None
        result[key+'_max'] = max(row[key] for row in rows) if present else None
    if motion is not None:
        result.update(motion.result())
    result.update(CouplingMetrics.aggregate(rows))
    result.update(NoteTimingMetrics.aggregate(rows))
    result['note_timing_contract'] = getattr(env, 'note_timing_contract', None)
    if coupling:
        result.update(finger_coupling_sample_hz=control_hz,
                      finger_coupling_diagnostics_scope=(
                          'control_cycle_endpoints; active means assigned by current score, '
                          'not measured physical contact; not physics-substep angle coverage'))
    return result


def run(args):
    config = effective_config(args)
    if args.inspect_only:
        if args.replay_prefill_episodes:
            # The source launcher has already exported the full score including
            # policy preparation and release tail. No GPU allocation is needed.
            manifest = json.loads((args.task/'task.json').read_text())
            resolve_replay_prefill(args, config, manifest['length'])
        print(json.dumps(config, indent=2))
        return config
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The MJLab training runner requires CUDA')
    print('当前训练奖励（已应用 JSON 和末尾参数覆盖）：\n' +
          json.dumps(config['reward_config'], ensure_ascii=False, indent=2), flush=True)
    print('当前关节状态限位：' + str((config.get('control_limits') or {}).get(
        'joint_state_limits', '历史软限位')), flush=True)
    layout = config['execution_layout']
    collector_device = torch.device(layout['collector_device'])
    learner_device = torch.device(layout['learner_device'])
    for device in (collector_device, learner_device):
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f'Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible')
    # CUDA graph construction and native kernels may consult the current device.
    # Physics is initialized under its own device; the main learner runs below
    # with the learner device current. The collector worker sets its device too.
    torch.cuda.set_device(collector_device)
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    # FP32 migration baseline. Do not silently introduce TF32 or AMP differences.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output_dir or ROOT/'training_mjlab/outputs'/datetime.now().strftime('sac-%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    if args.no_arms:
        from ..mjlab_env.without_arms import without_arms
        build_root = args.scene.parent if args.source_config else output
        args.scene, args.task = without_arms(args.scene, args.task, build_root/'hand_only_bundle')
    metadata = {**config, 'scene_sha256': sha256(args.scene/'manifest.json'), 'task_sha256': sha256(args.task/'task.json'),
                'cli': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'versions': {n: importlib.metadata.version(n) for n in ('torch', 'mujoco', 'mujoco-warp', 'mjlab')},
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                  list(Path(__file__).parent.glob('*.py')) + list((ROOT/'tsensei_v3_2_pianist/mjlab_env').glob('*.py'))
                                  + [ROOT/'tsensei_v3_2_pianist/forearm_collision.py']}}
    metadata['execution_environment'] = dict(
        torch_cuda=torch.version.cuda, capability=list(torch.cuda.get_device_capability(args.device)),
        optimizer_implementation=('torch_header_adam_math_layout_v1' if args.optimizer_layout else
                                  'torch_adam'),
        ninja=importlib.metadata.version('ninja') if args.optimizer_layout else None,
        fp32_matmul='ieee', tf32=False, mixed_precision=False)
    metadata['execution_environment']['devices'] = {
        str(device): dict(name=torch.cuda.get_device_name(device),
                          capability=list(torch.cuda.get_device_capability(device)))
        for device in (collector_device, learner_device)}
    metadata['execution_environment']['peer_access'] = (
        dict(collector_to_learner=torch.cuda.can_device_access_peer(collector_device, learner_device),
             learner_to_collector=torch.cuda.can_device_access_peer(learner_device, collector_device))
        if collector_device != learner_device else None)
    if args.source_config:
        metadata['source_runtime'] = json.loads(args.source_config.read_text())
        if metadata['source_runtime'].get('preparation') != config['preparation']:
            raise ValueError('Preparation differs between source recipe and learner configuration')
        # Temporary compiler products are never part of the replay recipe.
        for name in ('scene', 'task', 'source_config'):
            metadata['cli'][name] = None
    (output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    started = time.perf_counter()
    if args.progress:
        print(f'Initializing GPU training: {output}\n'
              'The first physics step and learner update may take longer to compile.',
              file=sys.stderr, flush=True)
    env = PianoTaskEnv(args.scene, args.task, num_envs=args.num_envs, device=args.device,
                       arm_follow=config['arm_follow'],
                       preparation=config['preparation'],
                       finger_coupling=config['finger_coupling'],
                       collision_diagnostics=args.collision_diagnostics,
                       control_limits=config['control_limits'],
                       finger_clearance=config['finger_clearance'],
                       max_episode_steps=args.max_episode_steps, physics_graph=args.physics_graph,
                       reward_config=config['reward_config'], action_profile=args.action_profile, disable_wrist_tz=args.disable_wrist_tz, disable_wrist_yaw=args.disable_wrist_yaw, self_collision_detection_only=args.self_collision_detection_only, hand_collision_detection_only=args.hand_collision_detection_only)
    resolve_replay_prefill(args, config, env.logic.length)
    metadata.update(config)
    if args.replay_prefill_episodes:
        print('Replay prefill: '+json.dumps(config['replay_prefill'])+
              f'; random action rows unchanged: {config["warmup_transitions"]}; '
              'remaining prefill uses stochastic policy; no deferred updates.', flush=True)
    metadata['preparation_contract'] = env.scene.bundle.preparation
    metadata['control_limits_contract'] = env.scene.control_limits
    metadata['finger_clearance_contract'] = env.scene.finger_clearance
    metadata['finger_coupling_contract'] = env.scene.finger_coupling
    metadata['coupling_observation_contract'] = env.coupling_observation_contract
    metadata['arm_follow_contract'] = (env.scene.arm_follower.contract if env.scene.arm_follower is not None else None)
    metadata['action_contract'] = env.scene.action_contract
    metadata['score_timing_contract'] = dict(
        reward_target='original_midi_physical_substeps', reward_state='post_physics_substep_mean',
        key_activation_weight=0.5, other_rewards='shadow_control_frame_before_action_endpoint',
        score_validation=env.bundle.manifest.get('score_validation', 'legacy'),
        music_end_step=env.bundle.music_end_step,
        release_tail_steps=env.bundle.release_tail_steps)
    if env.note_timing_contract is not None:
        metadata['score_timing_contract']['exact_events'] = env.note_timing_contract
    score_coverage = args.task/'score_coverage.json'
    if score_coverage.is_file():
        shutil.copyfile(score_coverage, output/'score_coverage.json')
    metadata['key_contact_diagnostics_contract'] = getattr(
        getattr(env.scene,'key_contact_diagnostics',None),'contract',None)
    if config['reward_config']['terms']['humanoid_finger_gesture_reward']['enabled']:
        metadata['humanoid_finger_gesture_pose'] = env.reward_manager.gesture_pose
    torch.cuda.set_device(learner_device)
    agent = SAC(env.observation_size, env.action_size, SACConfig(**config['sac']), device=str(learner_device), seed=args.seed,
                compile_learner=args.compile_learner, fused_optimizer=args.fused_optimizer, action_profile=args.action_profile)
    exploration = torch.Generator(device=env.device).manual_seed(args.seed + 3)
    parent_checkpoint = args.warm_start or args.stage2_from or args.finetune_from
    if parent_checkpoint:
        networks_only = bool(args.stage2_from or args.finetune_from)
        previous = checkpoint.load(parent_checkpoint, agent, metadata,
                                   weights_only=networks_only, exploration_rng=exploration,
                                   allow_reward_change=args.allow_reward_change,
                                   allow_control_change=args.allow_control_change,
                                   transfer_mode='finetune' if args.finetune_from else 'stage2')
        metadata['origin'] = dict(checkpoint=str(parent_checkpoint.resolve()),
                                  checkpoint_sha256=sha256(parent_checkpoint),
                                  transfer_mode=('finetune_networks_only' if args.finetune_from else
                                                 'stage2_networks_only' if args.stage2_from else 'learner_warm_restart'),
                                  fresh_optimizer_alpha_rng=networks_only, fresh_replay=True,
                                  previous_vector_step=previous.get('vector_step'), previous_updates=previous.get('updates'),
                                  previous_reward_config=checkpoint.reward_contract(previous),
                                  allow_control_change=args.allow_control_change,
                                  previous_control_limits=previous.get('control_limits'),
                                  previous_control_limits_contract=previous.get('control_limits_contract'),
                                  current_control_limits=config['control_limits'],
                                  current_control_limits_contract=metadata['control_limits_contract'],
                                  previous_finger_clearance=previous.get('finger_clearance'),
                                  previous_finger_clearance_contract=previous.get('finger_clearance_contract'),
                                  current_finger_clearance=config['finger_clearance'],
                                  current_finger_clearance_contract=metadata['finger_clearance_contract'],
                                  original_metadata=previous)
    replay = Replay(env.observation_size, env.action_size, config['replay_capacity'], args.batch_size,
                    device=str(learner_device), seed=args.seed + 4, action_profile=args.action_profile)
    collector = (CollectorPolicy(agent, collector_device)
                 if collector_device != learner_device else agent)
    # Independent simulation: deterministic evaluation cannot reset the training worlds.
    torch.cuda.set_device(collector_device)
    eval_env = PianoTaskEnv(args.scene, args.task, device=args.device,
                           arm_follow=config['arm_follow'],
                           num_envs=config['evaluation_execution']['num_envs'],
                           preparation=config['preparation'],
                           finger_coupling=config['finger_coupling'],
                           collision_diagnostics=args.collision_diagnostics,
                           control_limits=config['control_limits'],
                           finger_clearance=config['finger_clearance'],
                           max_episode_steps=args.max_episode_steps, physics_graph=args.physics_graph,
                           reward_config=config['reward_config'], action_profile=args.action_profile, disable_wrist_tz=args.disable_wrist_tz, disable_wrist_yaw=args.disable_wrist_yaw, self_collision_detection_only=args.self_collision_detection_only, hand_collision_detection_only=args.hand_collision_detection_only) if args.eval_interval or args.eval_at_start else None
    metadata.update(observation_size=env.observation_size, action_size=env.action_size,
                    policy_action_names=[env.scene.bundle.manifest["action_names"][i] for i in env.scene.policy_ids.cpu().tolist()],
                    reset_contract=dict(
                        mode='explicit_partial_after_terminal_transition_v1',
                        # Limited actions start from actual reset ctrl, stored in
                        # the latest history frame (51 slots; sustain is zero).
                        previous_action=('actual_reset_actuator_ctrl_v1' if env.scene.has_action_transform
                                         else env.logic.reset_action_semantics),
                        physics_state=('policy_preparation_initial_v1' if config['preparation'] and
                                       config['preparation'].get('mode') == 'policy_v1' else
                                       'fixed_preparation_terminal_v1' if config['preparation'] else
                                       'source_config_reset_v1' if args.source_config else 'frozen_scene_reset_v1')),
                    target_entropy=agent.target_entropy, replay_bytes=replay.bytes_allocated,
                    device=torch.cuda.get_device_name(env.device))
    (output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    obs = env.reset()
    torch.cuda.synchronize(collector_device)
    torch.cuda.set_device(learner_device)
    completed = truncated = 0
    initial_updates = agent.updates
    metrics = {}
    learner_graph = None
    learner_graph_setup_s = 0.
    selection = checkpoint.EvaluationTracker(config['evaluation_selection'])
    last_evaluation = None
    last_evaluation_step = None
    target_reached = False
    evaluation_seconds_total = 0.
    evaluation_count = 0
    checkpoint_seconds_total = 0.
    torch.cuda.synchronize(env.device)
    loop_start = time.perf_counter()
    pipeline = None
    collector_version = agent.updates
    with (output/'metrics.jsonl').open('w') as stream, tqdm(
            total=args.vector_steps * args.num_envs, desc='Training', unit='sample',
            dynamic_ncols=True, mininterval=1., disable=not args.progress) as progress, ExitStack() as runtime:
        latest_f1 = None
        def emit(record):
            stream.write(json.dumps(dict(vector_step=step, transitions=step * args.num_envs,
                                         updates=agent.updates,
                                         **({'collector_pipeline': pipeline.state()} if pipeline else {}),
                                         **record), allow_nan=False)+'\n')
            stream.flush()
        def snapshot_metadata():
            return {**metadata, 'vector_step': step, 'updates': agent.updates,
                    **({'collector_pipeline_state': pipeline.state()} if pipeline else {}),
                    'evaluation_selection_state': selection.state_dict(),
                    'last_evaluation': last_evaluation, 'last_evaluation_vector_step': last_evaluation_step}
        def drain():
            if pipeline is not None:
                pipeline.flush()
        def save_checkpoint(path):
            nonlocal checkpoint_seconds_total
            drain()
            save_started = time.perf_counter()
            if collector is not agent:
                collector.copy_action_rng_to(agent)
            checkpoint.save(path, agent, snapshot_metadata(), exploration)
            checkpoint_seconds_total += time.perf_counter() - save_started
        def sync_collector():
            nonlocal collector_version
            collector_version = (collector.sync_from(agent) if collector is not agent else agent.updates)
            return collector_version
        def collect_vector(vector_step, version):
            # PyTorch CUDA current devices/streams are thread-local. The worker
            # never touches the learner, replay or output streams.
            with torch.cuda.device(collector_device):
                action = training_action(collector, env.scene, obs, args, vector_step, exploration)
                result = env.step(action)
                if collector is not agent:
                    torch.cuda.synchronize(collector_device)
                return obs, action, result, version
        def commit_vector(batch):
            nonlocal obs, completed, truncated
            previous_obs, action, result, _ = batch
            # A pipeline invokes this only after the previous update block is
            # complete. Copy terminal observations before resetting any world.
            replay.insert(Transition(previous_obs, action, result.reward, result.discount, result.observation,
                                     result.terminated, result.truncated))
            obs = result.observation
            with torch.cuda.device(collector_device):
                ids = result.done.nonzero(as_tuple=False).flatten()
                if ids.numel():
                    completed += int(result.terminated.sum().item())
                    truncated += int(result.truncated.sum().item())
                    emit(dict(episodes=dict(count=ids.numel(), mean_return=result.info['episode_return'][ids].mean().item(),
                                            mean_f1=result.info['episode_metrics'][ids, 2].mean().item())))
                    obs = env.reset(ids)
                if collector is not agent:
                    # Reset writes must be visible before next worker launch.
                    torch.cuda.synchronize(collector_device)
        def learn_vector(vector_step):
            nonlocal learner_graph, learner_graph_setup_s, metrics
            update_count = updates_this_step(args, vector_step)
            if update_count and replay.is_ready():
                if agent.updates == initial_updates and args.compile_learner:
                    progress.set_description_str('SAC compile/update')
                if args.learner_graph:
                    if learner_graph is None:
                        from .learner_graph import CUDALearner
                        graph_start = time.perf_counter()
                        learner_graph = CUDALearner(agent, replay, optimizer_layout=args.optimizer_layout)
                        learner_graph_setup_s = time.perf_counter() - graph_start
                    metrics = learner_graph.update(update_count)
                else:
                    for _ in range(update_count):
                        metrics = agent.update(replay.sample())
        if args.collector_pipeline:
            pipeline = runtime.enter_context(OneBatchPipeline(
                snapshot=sync_collector, collect=collect_vector, update=learn_vector, commit=commit_vector,
                learner_version=lambda: agent.updates))
        def run_evaluation(*, baseline=False):
            nonlocal latest_f1, last_evaluation, last_evaluation_step
            nonlocal evaluation_seconds_total, evaluation_count
            drain()
            evaluation_started = time.perf_counter()
            sync_collector()
            with torch.cuda.device(collector_device):
                evaluation = evaluate(collector, eval_env, args.eval_episodes, metrics_graph=args.eval_metrics_graph)
                torch.cuda.synchronize(collector_device)
            evaluation_seconds = time.perf_counter() - evaluation_started
            evaluation_seconds_total += evaluation_seconds
            evaluation_count += 1
            latest_f1 = evaluation['music_only_f1_min']
            last_evaluation, last_evaluation_step = evaluation, step
            decision = selection.observe(evaluation, vector_step=step, updates=agent.updates,
                                         count_for_target=not baseline)
            previous_checkpoint_seconds = checkpoint_seconds_total
            if decision['improved'] and args.save_best:
                save_checkpoint(output/'best.pt')
                (output/'best.json').write_text(json.dumps(selection.best, indent=2)+'\n')
            if args.save_forward and not baseline and step == args.eval_interval:
                save_checkpoint(output/'forward.pt')
            emit(dict(eval=evaluation, checkpoint_selection=decision, baseline=baseline,
                      evaluation_seconds=evaluation_seconds,
                      checkpoint_seconds=checkpoint_seconds_total-previous_checkpoint_seconds))
            return decision['target_reached']
        step = 0
        if args.eval_at_start:
            progress.set_description_str('Initial evaluation')
            run_evaluation(baseline=True)
        for step in range(1, args.vector_steps + 1):
            if pipeline is not None:
                batch = pipeline.step(step)
                if (agent.updates == initial_updates and updates_this_step(args, step)
                        and replay.is_ready()):
                    # First compile/capture runs without concurrent CUDA work
                    # from the collector. Later blocks reuse the captured graph.
                    drain()
            else:
                batch = collect_vector(step, sync_collector())
                commit_vector(batch)
                learn_vector(step)
            result = batch[2]
            if (step == args.vector_steps
                    or (args.eval_interval and step % args.eval_interval == 0)
                    or (args.save_interval and step % args.save_interval == 0)):
                drain()
            if step % args.log_interval == 0:
                torch.cuda.synchronize(env.device)
                emit(dict(train={k: v.item() for k, v in metrics.items()}, replay_size=len(replay),
                          reward_terms={k: v.mean().item() for k, v in result.info['reward_terms'].items()},
                          reward_raw_terms={k: v.mean().item() for k, v in result.info.get('reward_raw_terms',{}).items()},
                          reward_weighted_terms={k: v.mean().item() for k, v in result.info.get('reward_weighted_terms',{}).items()},
                          loop_transitions_per_second=step * args.num_envs / (time.perf_counter() - loop_start)))
            if eval_env is not None and args.eval_interval and step % args.eval_interval == 0:
                progress.set_description_str('Evaluating')
                target_reached = run_evaluation()
            if args.save_interval and step % args.save_interval == 0:
                progress.set_description_str('Saving checkpoint')
                save_checkpoint(output/f'step-{step}.pt')
            progress.set_description_str('Training', refresh=False)
            postfix = dict(vstep=f'{step}/{args.vector_steps}', updates=agent.updates, replay=len(replay))
            if latest_f1 is not None:
                postfix['music_F1_min'] = f'{latest_f1:.4f}'
            progress.set_postfix(postfix, refresh=False)
            progress.update(args.num_envs)
            if target_reached:
                break
        drain()
        if eval_env is not None and last_evaluation_step != step:
            progress.set_description_str('Final evaluation')
            target_reached = run_evaluation()
            progress.set_postfix(vstep=f'{step}/{args.vector_steps}', updates=agent.updates,
                                 replay=len(replay), music_F1_min=latest_f1)
        save_checkpoint(output/'final.pt')
    torch.cuda.synchronize(env.device)
    torch.cuda.synchronize(learner_device)
    summary = dict(status='completed', vector_steps=step, transitions=step * args.num_envs,
                   stop_reason='quality_target_reached' if target_reached else 'requested_budget_exhausted',
                   target_reached=target_reached, evaluation_selection_state=selection.state_dict(),
                   best_checkpoint=str((output/'best.pt').resolve()) if args.save_best and selection.best else None,
                   final_evaluation=last_evaluation,
                   updates_this_run=agent.updates - initial_updates, updates_total=agent.updates,
                   random_action_transitions=min(config['warmup_transitions'], step * args.num_envs),
                   replay_prefill=config['replay_prefill'],
                   completed_episodes=completed, truncated_episodes=truncated, replay_size=len(replay),
                   replay_insertions=replay.total_inserted, replay_bytes=replay.bytes_allocated,
                   elapsed_s=time.perf_counter() - started,
                   learner_graph_setup_s=learner_graph_setup_s,
                   evaluation_seconds_total=evaluation_seconds_total,
                   evaluation_timing=dict(count=evaluation_count, total_seconds=evaluation_seconds_total,
                       mean_seconds=evaluation_seconds_total/evaluation_count if evaluation_count else None,
                       num_envs=config['evaluation_execution']['num_envs'], episodes=args.eval_episodes,
                       scope='actor_sync_and_evaluation_excluding_pipeline_drain_and_checkpoint_save'),
                   checkpoint_seconds_total=checkpoint_seconds_total,
                   execution_layout=layout,
                   collector_pipeline_state=pipeline.state() if pipeline else None,
                   torch_peak_allocated_bytes_by_device={
                       str(device): torch.cuda.max_memory_allocated(device)
                       for device in (collector_device, learner_device)},
                   torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(env.device),
                   note='Torch memory excludes some Warp allocations; short runs do not demonstrate learning quality',
                   output_dir=str(output.resolve()))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
