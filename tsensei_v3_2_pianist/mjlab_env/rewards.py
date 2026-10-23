"""Torch implementations of paper pp.10–12, with explicit TsenseI mappings.

No hidden Stage2 shaping or guessed wrist/press-angle/contact mapping. Returned
terms are final weighted contributions. Idle-posture and joint-motion-limit
costs use per-control-step units; other terms retain the configured dt scaling.
"""
import torch

from .reward_config import validate, GROUPS, FINGER_NAMES, finite_number
from .task_logic import key_press_substep_reward, gaussian_tolerance, fingering_substep_reward
from . import performance_options


def false_press_penalty(key_state, target):
    """Daxian per-step wrong-key cost, summed over 88 keys (pedal excluded).

    x = clip(normalized key travel, 0, 1). Each non-target key costs zero
    when x <= 0.001; otherwise 0.01 + (0.15 - 0.01) * x. This uses travel,
    not key activation, and applies to every key during a silent score frame.
    """
    travel = key_state.clamp(0., 1.)
    cost = torch.where(travel > 1e-3, .01 + (.15 - .01) * travel, 0.)
    return -(cost * ~target[:, :88].bool()).sum(-1)


def gesture_protection_mask(fingers, lookahead_steps):
    """Current and near-future assigned fingers, with zero padding at score end."""
    if (type(lookahead_steps) is not int or lookahead_steps < 0
            or fingers.ndim != 3 or fingers.shape[1:] != (10,88)):
        raise ValueError('Gesture protection requires (T,10,88) fingering and nonnegative integer steps')
    active=fingers.bool().any(-1)
    protected=active.clone()
    for offset in range(1,min(lookahead_steps+1,len(active))):
        protected[:-offset] |= active[offset:]
    return protected


class ArmTrackingErrorAccumulator:
    """Peak raw-request error across both palms and every physical substep.

    Inputs are world positions in metres. The requested position must precede
    TCP and IK limiting; the separate IK target is diagnostic only. Actual
    position comes from the rigid hand body after the scene's forward update.
    No endpoint-only resampling, target projection, averaging or dt scaling.
    """
    def __init__(self, *, num_envs, device):
        self.requested_peak = torch.zeros((num_envs, 2), device=device, dtype=torch.float64)
        self.ik_target_peak = torch.zeros_like(self.requested_peak)
        self.over_1mm_side_substeps = torch.zeros((num_envs, 2), device=device, dtype=torch.int64)
        self.over_1mm_substeps = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self.over_1mm_events = torch.zeros_like(self.over_1mm_substeps)
        self.sampled_substeps = torch.zeros_like(self.over_1mm_substeps)
        self.previous_over_1mm = torch.zeros((num_envs, 2), device=device, dtype=torch.bool)
        self.sample_count = 0

    def begin_control(self):
        self.requested_peak.zero_()
        self.ik_target_peak.zero_()
        self.over_1mm_side_substeps.zero_()
        self.over_1mm_substeps.zero_()
        self.over_1mm_events.zero_()
        self.sampled_substeps.zero_()
        # Keep the previous substep's state across control-frame boundaries.
        # A continuous excursion is one entry event, not one per frame.
        self.sample_count = 0

    def reset(self, ids):
        self.requested_peak[ids] = 0
        self.ik_target_peak[ids] = 0
        self.over_1mm_side_substeps[ids] = 0
        self.over_1mm_substeps[ids] = 0
        self.over_1mm_events[ids] = 0
        self.sampled_substeps[ids] = 0
        self.previous_over_1mm[ids] = False

    def sample(self, requested, actual, ik_target):
        expected = (*self.requested_peak.shape, 3)
        if any(tuple(value.shape) != expected for value in (requested, actual, ik_target)):
            raise ValueError('Arm tracking requires requested/actual/IK world positions [env, 2, 3]')
        actual = actual.double()
        # torch.maximum intentionally propagates nonfinite input. The task's
        # final finite-state guard rejects that transition instead of hiding it.
        error = torch.linalg.vector_norm(requested.double() - actual, dim=-1)
        self.requested_peak.copy_(torch.maximum(self.requested_peak, error))
        self.ik_target_peak.copy_(torch.maximum(self.ik_target_peak,
            torch.linalg.vector_norm(ik_target.double() - actual, dim=-1)))
        over = error > .001
        any_over = over.any(-1)
        self.over_1mm_side_substeps.add_(over.long())
        self.over_1mm_substeps.add_(any_over.long())
        self.over_1mm_events.add_((any_over & ~self.previous_over_1mm.any(-1)).long())
        self.sampled_substeps.add_(1)
        self.previous_over_1mm.copy_(over)
        self.sample_count += 1

    def snapshot(self):
        requested = self.requested_peak.amax(-1)
        return dict(arm_tracking_requested_error_m=requested,
                    arm_tracking_requested_error_by_side_m=self.requested_peak,
                    arm_tracking_ik_target_error_m=self.ik_target_peak.amax(-1),
                    arm_tracking_within_1mm=requested <= .001,
                    arm_tracking_over_1mm_substeps=self.over_1mm_substeps,
                    arm_tracking_over_1mm_side_substeps=self.over_1mm_side_substeps,
                    arm_tracking_over_1mm_events=self.over_1mm_events,
                    arm_tracking_sampled_substeps=self.sampled_substeps)


class ArmTrackingMetrics:
    """Persistent counts and peaks over all supplied control transitions.

    Substeps mean world-times, with either hand over 1 mm counted once. Side
    counts remain separate. The fraction uses substeps, not control frames.
    """
    def __init__(self):
        self.counts = None
        self.peaks = None
        self.side_counts = None

    def sample(self, info, active_worlds=None):
        if 'arm_tracking_requested_error_m' not in info:
            return
        def values(name):
            return info['arm_tracking_' + name][:active_worlds]
        requested = values('requested_error_m')
        counts = torch.stack((values('sampled_substeps').sum(), values('over_1mm_substeps').sum(),
            values('over_1mm_events').sum(), (~values('within_1mm')).sum(),
            torch.full((), requested.numel(), device=requested.device, dtype=torch.int64)))
        peaks = torch.stack((requested.amax(), values('ik_target_error_m').amax()))
        sides = values('over_1mm_side_substeps').sum(0)
        if self.counts is None:
            self.counts, self.peaks, self.side_counts = counts, peaks, sides
        else:
            self.counts.add_(counts)
            self.peaks.copy_(torch.maximum(self.peaks, peaks))
            self.side_counts.add_(sides)

    def result(self):
        if self.counts is None:
            return None
        sampled, over, events, control_over, controls = self.counts.cpu().tolist()
        requested, ik_target = self.peaks.cpu().tolist()
        return dict(requested_peak_m=requested, ik_target_peak_m=ik_target,
            over_1mm_substeps=over, sampled_substeps=sampled,
            within_1mm_substep_fraction=1 - over / sampled if sampled else None,
            over_1mm_entry_events=events, over_1mm_control_steps=control_over,
            sampled_control_steps=controls, over_1mm_side_substeps=dict(
                zip(('right', 'left'), self.side_counts.cpu().tolist())),
            scope='all_collected_world_control_steps; substep counts are world-times with either hand over 1mm')


class RewardManager:
    def __init__(self, config, scene_manifest, task_manifest, *, num_envs, device):
        self.config = c = validate(config)
        self.performance = performance_options.get(c)
        self.terms = c['terms']
        self.names = [n for n,t in self.terms.items() if t['enabled']]
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.dt = float(scene_manifest['control_dt'])
        finite_number(self.dt,'control_dt',positive=True)
        self.task_manifest = task_manifest
        self.actions = scene_manifest['action_names'][:-1]
        if len(self.actions)!=50 or len(set(self.actions))!=50 or scene_manifest['action_names'][-1]!='sustain':
            raise ValueError('Reward mapping requires 50 uniquely named hand actuators and sustain')
        self.low=self.tensor(scene_manifest['action_min'][:-1])
        self.high=self.tensor(scene_manifest['action_max'][:-1])
        self.neutral=self.tensor(scene_manifest['neutral_action'][:-1])
        if not torch.isfinite(self.neutral).all() or not (self.high>self.low).all():
            raise ValueError('Invalid neutral action or action range')
        index={n:i for i,n in enumerate(self.actions)}
        # Action names use the canonical mixed-case Palm_Base_Tx/Ty/Tz labels.
        translation=[f'{s}_Palm_Base_T{axis}' for s in ('R','L') for axis in ('x','y','z')]
        wrist=[f'{s}_Palm_Base_{axis}' for s in ('R','L') for axis in ('Pitch','Yaw')]
        if not set(translation+wrist)<=set(index):
            raise ValueError('Missing named TsenseI wrist actions')
        finger=[n for n in self.actions if n not in translation+wrist]
        self.groups={g:torch.tensor([index[n] for n in names],device=device)
                     for g,names in zip(GROUPS,(translation,wrist,finger))}
        self.non_forearm=torch.cat((self.groups['wrist'],self.groups['finger']))
        self.tcp_indices=torch.tensor([index[f'{side}_Palm_Base_T{axis}']
            for side in ('R','L') for axis in ('x','y')], device=device)
        self.previous=self.neutral.expand(num_envs,-1).clone()
        self.previous2=self.previous.clone()
        self.gesture_active = self.enabled('humanoid_finger_gesture_reward')
        self.gesture_lookahead_steps = 0
        self.gesture_target_indices = None
        self.gesture_pose = {}
        if self.gesture_active:
            params = self.terms['humanoid_finger_gesture_reward']['params']
            self.gesture_lookahead_steps = params['lookahead_steps']
            # Full physical actuator targets include coupled PIP/DIP followers.
            groups = [[index[name] for name in self.actions if name.startswith(finger+'_')]
                      for finger in FINGER_NAMES]
            if any(len(ids) != 4 for ids in groups):
                raise ValueError('Humanoid gesture requires four command joints per finger')
            self.gesture_target_indices = torch.tensor(groups, device=device)
            self.gesture_joint_scales = self.tensor([[params['swing_scale']
                if self.actions[i].endswith('_SW') else 1. for i in ids] for ids in groups])
            self.gesture_pose = {self.actions[i]: float(self.neutral[i]) for ids in groups for i in ids}
        self.pitch_indices=torch.tensor([index[n] for n in ('R_Palm_Base_Pitch','L_Palm_Base_Pitch')],device=device)
        if self.enabled('wrist_pitch_reward'):
            signs=self.terms['wrist_pitch_reward']['params']['directions']
            self.pitch_sign=self.tensor([signs[n] for n in ('R_Palm_Base_Pitch','L_Palm_Base_Pitch')])
        if self.enabled('forward_press_reward'):
            mapping=self.terms['forward_press_reward']['params']['angle_map']
            matrix=torch.zeros((10,50),device=device)
            offsets=[]
            for row,name in enumerate(FINGER_NAMES):
                offsets.append(mapping[name]['offset'])
                for joint,coefficient in mapping[name]['joints'].items():
                    if joint not in index:
                        raise ValueError(f'Unknown forward-press action: {joint}')
                    matrix[row,index[joint]]=coefficient
            self.press_matrix=matrix
            self.press_offset=self.tensor(offsets)
        self.raw_terms={}
        self.weighted_terms={}

    def tensor(self, x):
        return torch.as_tensor(x,device=self.device,dtype=torch.float32)

    def enabled(self,name):
        return self.terms[name]['enabled']

    def reset(self,ids):
        # Do not use legacy observation reset_action: it contains lower bounds.
        self.previous[ids]=self.neutral
        self.previous2[ids]=self.neutral
        self.raw_terms={}
        self.weighted_terms={}

    def coordinates(self,physical):
        if self.config['action_units']=='normalized':
            return 2*(physical-self.low)/(self.high-self.low)-1
        return physical

    def __call__(self, base, action, snapshot, target, fingers):
        physical=action[:,:50]
        u=self.coordinates(physical)
        previous=self.coordinates(self.previous)
        previous2=self.coordinates(self.previous2)
        velocity=(u-previous)/self.dt
        acceleration=(u-2*previous+previous2)/(self.dt*self.dt)
        raw={}
        for name in self.names:
            params=self.terms[name]['params']
            if name=='key_press_reward':
                value = key_press_substep_reward(snapshot, target,
                    self.task_manifest['key_tolerance'],
                    chord_bonus=params['chord_bonus'], chord_scope=params['chord_scope'])
            elif name=='fingering_reward':
                value = fingering_substep_reward(snapshot, self.task_manifest['finger_tolerance'])
            elif name in ('key_press_reward','sustain_reward','energy_reward','fingering_reward','forearm_reward'):
                if name not in base:
                    raise ValueError(f'{name} was disabled in the frozen task; re-export or disable it here')
                value=base[name]
            elif name=='self_collision_reward':
                value=-snapshot['self_collision_pair_substeps'].float()
            elif name=='false_press_penalty':
                value=false_press_penalty(snapshot['key_state'], target)
                # Signed negative weight multiplies a nonnegative travel cost.
                # Existing positive-magnitude configurations keep their formula.
                if self.terms[name]['weight'] < 0:
                    value=-value
            elif name=='finger_speed_reward':
                value=params['coefficient']*(velocity[:,self.groups['finger']].abs()-params['limit']).clamp_min(0).sum(-1)
            elif name=='humanoid_finger_gesture_reward':
                # The task precomputes current + the next N score frames, with
                # no wrap at the score end. This is the sole active humanoid
                # gesture formula: idle target L1 relative to scene reset.
                protected=snapshot.get('gesture_protected_fingers')
                if (not isinstance(protected,torch.Tensor) or protected.dtype != torch.bool
                        or protected.shape != fingers.shape[:2]):
                    raise ValueError('Humanoid gesture requires boolean gesture_protected_fingers [env, 10]')
                inactive=~(protected | fingers.bool().any(-1))
                deviation=((physical-self.neutral)[:,self.gesture_target_indices].abs()
                           *self.gesture_joint_scales).sum(-1)
                value=params['coefficient']*(deviation*inactive).sum(-1)
            elif name=='wrist_pitch_reward':
                value=params['bonus']*(physical[:,self.pitch_indices]*self.pitch_sign>0).float().sum(-1)
            elif name=='forward_press_reward':
                angle=physical@self.press_matrix.T+self.press_offset
                value=params['coefficient']*(angle>params['threshold']).float().sum(-1)
            elif name=='arm_motion_limit_penalty':
                excess = snapshot.get('arm_motion_excess')
                if not isinstance(excess, torch.Tensor) or excess.shape != (self.num_envs,):
                    raise ValueError('Arm penalty requires per-control maximum physical-substep excess')
                value = -params['maximum_penalty_per_step'] * (excess / params['full_penalty_excess']).clamp(0, 1)
            elif name=='arm_tracking_error_penalty':
                error = snapshot.get('arm_tracking_requested_error_m')
                if not isinstance(error, torch.Tensor) or error.shape != (self.num_envs,):
                    raise ValueError('Arm tracking penalty requires maximum raw-request error over physical substeps')
                ratio = error / params['error_scale_m']
                # Preserve the quadratic near zero; the logarithmic tail keeps
                # larger raw-request errors distinguishable without a plateau.
                # clamp_min only makes the unused log branch finite at zero.
                value = -torch.where(ratio <= 1., ratio.square(),
                                     1. + 2. * ratio.clamp_min(1.).log())
                value = torch.where(torch.isfinite(error), value, torch.full_like(value, float('nan'))).float()
            elif name=='joint_motion_limit_penalty':
                # Final expanded actuator targets (including coupled followers),
                # never normalized policy outputs or measured qvel/qacc. The
                # first target is compared against reset's held target at rest.
                target_velocity=(physical-self.previous)/self.dt
                target_acceleration=(physical-2*self.previous+self.previous2)/(self.dt*self.dt)
                finger_ids=self.groups['finger']
                excess=torch.maximum(
                    (target_velocity[:,finger_ids].abs()/params['finger_velocity_limit']-1).amax(-1),
                    (target_acceleration[:,finger_ids].abs()/params['finger_acceleration_limit']-1).amax(-1))
                if params['wrist_velocity_limit'] is not None:
                    wrist_ids=self.groups['wrist']
                    wrist_excess=torch.maximum(
                        (target_velocity[:,wrist_ids].abs()/params['wrist_velocity_limit']-1).amax(-1),
                        (target_acceleration[:,wrist_ids].abs()/params['wrist_acceleration_limit']-1).amax(-1))
                    excess=torch.maximum(excess,wrist_excess)
                if params['tcp_velocity_limit'] is not None:
                    # Same demand convention as the isolated arm IK limiter:
                    # raw candidate against previously ISSUED targets/velocity.
                    requested=snapshot.get('requested_physical_action')
                    if not isinstance(requested,torch.Tensor) or requested.shape != (self.num_envs,51):
                        raise ValueError('TCP motion penalty requires full requested_physical_action (num_envs, 51)')
                    ids=self.tcp_indices
                    previous=self.previous[:,ids].double()
                    previous_velocity=(previous-self.previous2[:,ids].double())/self.dt
                    tcp_velocity=(requested[:,ids].double()-previous)/self.dt
                    tcp_acceleration=(tcp_velocity-previous_velocity)/self.dt
                    tcp_excess=torch.maximum(
                        (tcp_velocity.abs()/params['tcp_velocity_limit']-1).amax(-1),
                        (tcp_acceleration.abs()/params['tcp_acceleration_limit']-1).amax(-1))
                    excess=torch.maximum(excess,tcp_excess.to(excess.dtype))
                # One shared budget across every joint and both derivatives.
                value=-params['maximum_penalty_per_step']*(excess/params['full_penalty_excess']).clamp(0,1)
            else:
                group,kind,_=name.split('_')
                ids=self.groups[group]
                values = (snapshot['signed_power'][:,ids].clamp_min(0) if kind=='power'
                          else velocity[:,ids].abs() if kind=='velocity' else acceleration[:,ids].abs())
                value=params['coefficient']*values.sum(-1)
            raw[name]=value
        weighted={name:value*self.terms[name]['weight'] for name,value in raw.items()}
        if 'self_collision_reward' in weighted:
            # The budget is in final reward units: weight first, then cap once
            # across all pairs and substeps, never once per pair or substep.
            weighted['self_collision_reward'] = weighted['self_collision_reward'].clamp_min(
                -self.terms['self_collision_reward']['params']['maximum_penalty_per_step'])
        # Optional reference units preserve the reward rate of an existing
        # control frequency. Missing reference_dt retains historical seconds.
        scale=(self.dt/self.config.get('reference_dt', 1.)
               if self.config['scale_by_dt'] else 1.)
        contributions={name:value*(1. if name in ('joint_motion_limit_penalty', 'arm_motion_limit_penalty',
                                                'arm_tracking_error_penalty',
                                                'self_collision_reward','humanoid_finger_gesture_reward') else scale)
            for name,value in weighted.items()}
        self.previous2.copy_(self.previous)
        self.previous.copy_(physical)
        self.raw_terms=raw
        self.weighted_terms=weighted
        return contributions
