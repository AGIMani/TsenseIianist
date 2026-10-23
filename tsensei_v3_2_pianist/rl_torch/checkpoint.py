"""Atomic learner snapshots; warm restart, never an exact physics/replay resume."""
import os
import math
from copy import deepcopy
from pathlib import Path
import tempfile

import torch
from ..mjlab_env.reward_config import canonical


def check_tz_physics(metadata):
    if metadata.get('disable_wrist_tz', False) and metadata.get('wrist_tz_physics') != 'joint_removed_v1':
        raise ValueError('Legacy TZ projection checkpoint is incompatible with physical joint removal; start a fresh run')


def reward_contract(metadata):
    config = metadata.get('reward_config')
    return canonical(config) if config is not None else canonical(
        self_collision_weight=metadata.get('self_collision_weight',0.))


def control_contract(metadata):
    """Legacy missing controls mean identity; new limits require resolved units."""
    from ..mjlab_env.control_limits import canonical
    options = canonical(metadata.get('control_limits'))
    resolved = metadata.get('control_limits_contract')
    if (options is None) != (resolved is None):
        raise ValueError('Control limits require both options and the resolved control contract')
    return options, resolved


def clearance_contract(metadata):
    """Legacy missing guards mean identity; enabled guards need full geometry."""
    from ..mjlab_env.finger_clearance import canonical
    options = canonical(metadata.get('finger_clearance'))
    resolved = metadata.get('finger_clearance_contract')
    if (options is None) != (resolved is None):
        raise ValueError('Finger clearance requires both options and the resolved geometry contract')
    return options, resolved


def coupling_contract(metadata):
    """Follower geometry and observation layout are part of the policy identity."""
    from ..mjlab_env.finger_coupling import canonical
    options = canonical(metadata.get('finger_coupling'))
    resolved = metadata.get('finger_coupling_contract')
    observation = metadata.get('coupling_observation_contract')
    if (options is None) != (resolved is None) or (options is None) != (observation is None):
        raise ValueError('Finger coupling requires options, resolved geometry and observation contracts')
    return options, resolved, observation


def arm_contract(metadata):
    from ..mjlab_env.arm_options import canonical
    options = canonical(metadata.get('arm_follow'))
    resolved = metadata.get('arm_follow_contract')
    if (options is None) != (resolved is None):
        raise ValueError('Arm follower requires options and its resolved base/observation/control contract')
    if resolved is not None and resolved.get('options') != options:
        raise ValueError('Arm follower options differ from resolved contract')
    return options, resolved


def save(path, agent, metadata, exploration_rng):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=1, kind='learner_warm_restart', agent=agent.state_dict(),
                   metadata=metadata, exploration_rng=exploration_rng.get_state())
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = stream.name
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def load(path, agent, expected, *, weights_only=False, exploration_rng=None, allow_reward_change=False,
         transfer_mode='stage2', allow_control_change=False):
    if allow_reward_change and not weights_only:
        raise ValueError('Reward changes require networks-only transfer with fresh optimizer/replay')
    if allow_control_change and not (weights_only and transfer_mode == 'finetune'):
        raise ValueError('Control changes require explicit finetune networks-only transfer with fresh optimizer/replay')
    payload = torch.load(path, map_location=agent.device, weights_only=True)
    check_tz_physics(payload['metadata'])
    if payload.get('schema') != 1 or payload.get('kind') != 'learner_warm_restart':
        raise ValueError('Unsupported learner checkpoint')
    if coupling_contract(payload['metadata']) != coupling_contract(expected):
        raise ValueError('Checkpoint finger coupling/observation contract mismatch; start a matching fresh policy')
    if arm_contract(payload['metadata']) != arm_contract(expected):
        raise ValueError('Checkpoint arm follower/observation contract mismatch; start a matching fresh policy')
    previous_controls, next_controls = control_contract(payload['metadata']), control_contract(expected)
    if previous_controls != next_controls and not allow_control_change:
        raise ValueError('Checkpoint control limits contract mismatch; explicit networks-only --allow-control-change required')
    if clearance_contract(payload['metadata']) != clearance_contract(expected) and not allow_control_change:
        raise ValueError('Checkpoint finger clearance contract mismatch; explicit networks-only --allow-control-change required')
    for key in ('scene_sha256', 'task_sha256'):
        if payload['metadata'][key] != expected[key]:
            raise ValueError(f'Checkpoint task/scene contract mismatch: {key}')
    if payload['metadata'].get('action_profile','legacy') != expected.get('action_profile','legacy'):
        raise ValueError('Checkpoint action profile mismatch')
    if payload['metadata'].get('action_contract') != expected.get('action_contract'):
        raise ValueError('Checkpoint action mapping range mismatch')
    for key in ('preparation', 'preparation_contract'):
        if payload['metadata'].get(key) != expected.get(key):
            raise ValueError('Checkpoint preparation contract mismatch: '+key)
    if payload['metadata'].get('disable_wrist_tz', False) != expected.get('disable_wrist_tz', False):
        raise ValueError('Checkpoint Palm_Base_Tz control contract mismatch')
    if payload['metadata'].get('hand_collision_detection_only', False) != expected.get('hand_collision_detection_only', False):
        raise ValueError('Checkpoint cross-hand physics mismatch')
    if payload['metadata'].get('self_collision_detection_only', False) != expected.get('self_collision_detection_only', False):
        raise ValueError('Checkpoint self-collision physics mismatch')
    if payload['metadata'].get('disable_wrist_yaw', False) != expected.get('disable_wrist_yaw', False):
        raise ValueError('Checkpoint Palm_Base_Yaw control contract mismatch')
    if reward_contract(payload['metadata']) != reward_contract(expected) and not allow_reward_change:
        raise ValueError('Checkpoint reward contract mismatch; use a fresh run or explicit networks-only reward transfer')
    if transfer_mode not in ('stage2', 'finetune') or (transfer_mode == 'finetune' and not weights_only):
        raise ValueError('Finetune transfer must explicitly load networks only')
    if weights_only and transfer_mode == 'stage2' and payload['metadata'].get('stage') != 1:
        raise ValueError('Stage2 transfer requires a Stage1 checkpoint')
    if weights_only:
        if agent.updates or any(getattr(agent, name).state for name in
                                ('actor_optimizer', 'critic_optimizer', 'temp_optimizer')):
            raise ValueError('Networks-only transfer requires a fresh learner and optimizer')
        # Validate every network before copying any weights. A failure must not
        # leave only the actor loaded when the critic architecture differs.
        for key in ('obs_size', 'action_size'):
            if payload['agent'][key] != getattr(agent, key):
                raise ValueError('Network transfer shape mismatch: '+key)
        for name in ('actor', 'critic', 'target_critic'):
            current, saved = getattr(agent, name).state_dict(), payload['agent'][name]
            if current.keys() != saved.keys() or any(current[k].shape != saved[k].shape for k in current):
                raise ValueError('Network transfer architecture mismatch: '+name)
    agent.load_state_dict(payload['agent'], weights_only=weights_only)
    if exploration_rng is not None and not weights_only:
        exploration_rng.set_state(payload['exploration_rng'].cpu())
    return payload['metadata']


class EvaluationTracker:
    """Track best music score and optional consecutive complete-evaluation gates.

    Gate inputs are per-episode worst cases, never frame scores averaged with
    the silent preparation prefix. Missing collision coverage fails closed.
    No training state, reward or scheduling is changed by this helper.
    """
    def __init__(self, config):
        self.config = deepcopy(config)
        self.metric = config.get('best_metric', 'f1')
        if self.metric not in ('f1',):
            raise ValueError('Unsupported best F1 metric: ' + str(self.metric))
        self.best = None
        self.consecutive = 0
        self.evaluations = 0
        self.last = None

    @staticmethod
    def _finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def _key_contact_failures(self, evaluation):
        """The optional underside criterion applies to best candidates and stop."""
        threshold=self.config.get('max_key_underside_substeps')
        if threshold is None:return []
        reasons=[]
        value=evaluation.get('key_underside_substeps_max')
        if (not self._finite(value) or value<0 or value!=math.floor(value) or value>threshold):
            reasons.append('key_underside_substeps_max')
        for key in ('key_contact_diagnostics_coverage','key_contact_substep_coverage'):
            if evaluation.get(key)!=1.0:reasons.append(key)
        if evaluation.get('key_contact_diagnostics_valid') is not True:
            reasons.append('key_contact_diagnostics_valid')
        hz=evaluation.get('key_contact_physics_sample_hz')
        if not self._finite(hz) or hz<=0:reasons.append('key_contact_physics_sample_hz')
        return reasons

    def observe(self, evaluation, *, vector_step, updates, count_for_target=True):
        self.evaluations += 1
        expected = evaluation.get('evaluation_episodes')
        complete = (isinstance(expected, int) and expected > 0
                    and evaluation.get('completed_episodes') == expected
                    and evaluation.get('truncated_episodes') == 0)
        score = evaluation.get(self.metric)
        best_reasons=[]
        if not complete:best_reasons.append('evaluation_incomplete')
        if not self._finite(score):best_reasons.append(self.metric)
        key_contact_reasons=self._key_contact_failures(evaluation)
        best_reasons.extend(key_contact_reasons)
        best_eligible=not best_reasons
        improved = (best_eligible
                    and (self.best is None or score > self.best['score']))
        if improved:
            self.best = dict(metric=self.metric, score=score, vector_step=vector_step,
                             updates=updates, evaluation=deepcopy(evaluation),best_eligible=True,
                             best_ineligible_reasons=[],
                             max_key_underside_substeps=self.config.get('max_key_underside_substeps'))
        # Extra metric stopping gates were removed with the auxiliary F1
        # evaluations.  Keep the decision shape for checkpoint consumers.
        reasons = []
        enabled = False
        self.consecutive = 0
        stop = enabled and self.consecutive >= self.config['consecutive_evaluations']
        self.last = dict(vector_step=vector_step, updates=updates, improved=improved,
                         best_eligible=best_eligible,best_ineligible_reasons=best_reasons,
                         target_enabled=enabled, target_passed=enabled and not reasons,
                         counted_for_target=count_for_target,
                         target_failed_checks=reasons, consecutive=self.consecutive,
                         target_reached=stop)
        return deepcopy(self.last)

    def state_dict(self):
        return dict(config=deepcopy(self.config), evaluations=self.evaluations,
                    consecutive=self.consecutive, best=deepcopy(self.best), last=deepcopy(self.last))


def reward_options(path):
    """Read checkpoint task additions before constructing evaluation physics."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    check_tz_physics(payload['metadata'])
    if payload.get('schema') != 1 or payload.get('kind') != 'learner_warm_restart':
        raise ValueError('Unsupported learner checkpoint')
    options = ({'reward_config':reward_contract(payload['metadata'])}
               if payload['metadata'].get('reward_config') is not None else
               {'self_collision_weight':payload['metadata'].get('self_collision_weight',0.)})
    profile = payload['metadata'].get('action_profile','legacy')
    if profile != 'legacy':
        options['action_profile'] = profile
    if payload["metadata"].get("disable_wrist_tz", False):
        options["disable_wrist_tz"] = True
    if payload["metadata"].get("self_collision_detection_only", False):
        options["self_collision_detection_only"] = True
    if payload["metadata"].get("hand_collision_detection_only", False):
        options["hand_collision_detection_only"] = True
    if payload["metadata"].get("disable_wrist_yaw", False):
        options["disable_wrist_yaw"] = True
    if payload['metadata'].get('preparation') is not None:
        options['preparation'] = payload['metadata']['preparation']
    if payload['metadata'].get('collision_diagnostics', False):
        options['collision_diagnostics'] = True
    controls, _ = control_contract(payload['metadata'])
    if controls is not None:
        options['control_limits'] = controls
    clearance, _ = clearance_contract(payload['metadata'])
    if clearance is not None:
        options['finger_clearance'] = clearance
    coupling, _, _ = coupling_contract(payload['metadata'])
    if coupling is not None:
        options['finger_coupling'] = coupling
    arms, _ = arm_contract(payload['metadata'])
    if arms is not None:
        options['arm_follow'] = arms
    return options



def load_agent(path, *, scene_sha256, task_sha256, obs_size, action_size, device='cuda:0',
               self_collision_weight=0., reward_config=None, compile_learner=False, fused_optimizer=False,
               action_profile='legacy', action_contract=None, disable_wrist_tz=False, disable_wrist_yaw=False, self_collision_detection_only=False, hand_collision_detection_only=False,
               preparation=None, preparation_contract=None, collision_diagnostics=False,
               control_limits=None, control_limits_contract=None,
               finger_clearance=None, finger_clearance_contract=None,
               finger_coupling=None, finger_coupling_contract=None, coupling_observation_contract=None,
               arm_follow=None, arm_follow_contract=None):
    """Reconstruct a learner for deterministic evaluation/playback, verify task identity."""
    # Collision diagnostics are observer-only, never a different policy/physics
    # contract. Accept the common environment options without blocking rechecks.
    from .sac import SAC, SACConfig
    payload = torch.load(path, map_location='cpu', weights_only=True)
    check_tz_physics(payload['metadata'])
    if payload.get('schema') != 1 or payload.get('kind') != 'learner_warm_restart':
        raise ValueError('Unsupported learner checkpoint')
    if arm_contract(payload['metadata']) != arm_contract(dict(arm_follow=arm_follow, arm_follow_contract=arm_follow_contract)):
        raise ValueError('Checkpoint arm follower/observation contract mismatch')
    if control_contract(payload['metadata']) != control_contract(dict(
            control_limits=control_limits, control_limits_contract=control_limits_contract)):
        raise ValueError('Checkpoint control limits contract mismatch')
    if clearance_contract(payload['metadata']) != clearance_contract(dict(
            finger_clearance=finger_clearance, finger_clearance_contract=finger_clearance_contract)):
        raise ValueError('Checkpoint finger clearance contract mismatch')
    if coupling_contract(payload['metadata']) != coupling_contract(dict(
            finger_coupling=finger_coupling, finger_coupling_contract=finger_coupling_contract,
            coupling_observation_contract=coupling_observation_contract)):
        raise ValueError('Checkpoint finger coupling/observation contract mismatch')
    for key, expected in (('scene_sha256', scene_sha256), ('task_sha256', task_sha256)):
        if payload['metadata'][key] != expected:
            raise ValueError(f'Checkpoint task/scene contract mismatch: {key}')
    if payload['metadata'].get('action_profile','legacy') != action_profile:
        raise ValueError('Checkpoint action profile mismatch')
    if payload['metadata'].get('action_contract') != action_contract:
        raise ValueError('Checkpoint action mapping range mismatch')
    if payload['metadata'].get('preparation') != preparation:
        raise ValueError('Checkpoint preparation contract mismatch')
    if payload['metadata'].get('preparation_contract') != preparation_contract:
        raise ValueError('Checkpoint preparation contract mismatch: preparation_contract')
    if preparation is not None and preparation_contract is None:
        raise ValueError('Prepared checkpoints require the reconstructed preparation_contract')
    if payload['metadata'].get('disable_wrist_tz', False) != disable_wrist_tz:
        raise ValueError('Checkpoint Palm_Base_Tz control contract mismatch')
    if payload['metadata'].get('hand_collision_detection_only', False) != hand_collision_detection_only:
        raise ValueError('Checkpoint cross-hand physics mismatch')
    if payload['metadata'].get('self_collision_detection_only', False) != self_collision_detection_only:
        raise ValueError('Checkpoint self-collision physics mismatch')
    if payload['metadata'].get('disable_wrist_yaw', False) != disable_wrist_yaw:
        raise ValueError('Checkpoint Palm_Base_Yaw control contract mismatch')
    config = SACConfig(**payload['agent']['config'])
    if reward_contract(payload['metadata']) != canonical(reward_config,self_collision_weight):
        raise ValueError('Checkpoint reward contract mismatch')
    agent = SAC(obs_size, action_size, config, device=device,
                compile_learner=compile_learner, fused_optimizer=fused_optimizer, action_profile=action_profile)
    agent.load_state_dict(payload['agent'])
    return agent
