"""GPU physical scene + fingered music rules, ready for an off-policy runner."""
import mujoco
import numpy as np
import torch
import warp as wp

from .scene import PianoScene
from .task_bundle import TaskBundle
from .task_logic import MusicTaskLogic
from .reward_config import canonical, contact_options
from .rewards import RewardManager, gesture_protection_mask, ArmTrackingErrorAccumulator
from .performance_options import get as performance_options
from .anticipation import anticipation_fields
from ..forearm_collision import task_collision_groups
from ..fingering_targets import target_position, key_reference, options_from_manifest


class PianoTaskEnv:
    def __init__(self, scene_directory, task_directory, *, num_envs=1, device='cuda:0',
                 max_episode_steps=None, physics_graph=True, self_collision_weight=0., reward_config=None,
                 action_profile='legacy', disable_wrist_tz=False, disable_wrist_yaw=False, self_collision_detection_only=False, hand_collision_detection_only=False,
                 preparation=None, collision_diagnostics=False, control_limits=None, finger_clearance=None, finger_coupling=None, arm_follow=None):
        self.reward_config = canonical(reward_config,self_collision_weight)
        self.performance = performance_options(self.reward_config)
        self_term = self.reward_config['terms']['self_collision_reward']
        self.self_collision_weight = self_term['weight'] if self_term['enabled'] else 0.
        self.scene = s = PianoScene(scene_directory,num_envs=num_envs,device=device,
                                    physics_graph=physics_graph, action_profile=action_profile, disable_wrist_tz=disable_wrist_tz, disable_wrist_yaw=disable_wrist_yaw,
                                    self_collision_detection_only=self_collision_detection_only, hand_collision_detection_only=hand_collision_detection_only,
                                    preparation=preparation,
                                    collision_diagnostics=collision_diagnostics,
                                    control_limits=control_limits,
                                    finger_clearance=finger_clearance,
                                    finger_coupling=finger_coupling,
                                    arm_follow=arm_follow,
                                    **contact_options(self.reward_config))
        for name in ('arm_motion_limit_penalty', 'arm_tracking_error_penalty'):
            if self.reward_config['terms'][name]['enabled'] and arm_follow is None:
                raise ValueError(f'{name} requires --arms and explicit initial angles')
        self.bundle = b = TaskBundle(task_directory,s.bundle)
        self.preparation_steps = b.preparation_steps
        self.release_tail_steps = b.release_tail_steps
        self.music_end_step = b.music_end_step
        self.collision_diagnostics=collision_diagnostics
        self.device, self.num_envs = s.device,num_envs
        self.arm_tracking = (ArmTrackingErrorAccumulator(num_envs=num_envs, device=device)
                             if getattr(s, 'arm_follower', None) is not None else None)
        self.logic = MusicTaskLogic(b,num_envs=num_envs,device=device,max_episode_steps=max_episode_steps)
        self.exact_score = None
        self.note_timing_contract = None
        if self.performance.get('exact_note_timing', False) != (b.exact_note_events is not None):
            raise ValueError('Reward exact_note_timing and exported score timing must match')
        if self.performance.get('exact_note_timing', False):
            from .exact_score import ExactScore
            self.exact_score = ExactScore(b.exact_note_events, b.exact_sustain_events,
                length=self.logic.length, control_dt=b.manifest['control_dt'],
                physics_dt=b.manifest['physics_dt'],
                lookahead_steps=b.manifest['contract']['environment']['n_steps_lookahead'],
                anticipation_horizon=max(self.performance['anticipatory_fingering_time'],
                    self.performance.get('upcoming_press_permission_time', 0.)), device=device)
            self.note_timing_contract = self.exact_score.contract
            # Current observations are point samples of the exact score;
            # future onsets remain in their own fields, never simultaneous goals.
            self.logic.observation_goal = self.exact_score.goal
            self.logic.observation_fingers = self.exact_score.fingers
            decimation = self.exact_score.decimation
            sample_steps = torch.arange(1, self.logic.length * decimation + 1, device=device)
            music_start = max(self.preparation_steps, round(b.manifest['contract']['environment'].get(
                'initial_buffer_time', 0.) / b.manifest['control_dt']))
            phase = ((sample_steps >= self.preparation_steps * decimation).long()
                     + (sample_steps >= music_start * decimation).long())
            if self.release_tail_steps:
                phase = torch.where(sample_steps >= self.music_end_step * decimation, 3, phase)
            self.exact_substep_phase = torch.nn.functional.one_hot(phase, 4).bool().reshape(
                self.logic.length, decimation, 4)
        from .finger_coupling import observation_contract
        self.coupling_observation_contract = observation_contract(
            b.manifest, s.bundle.manifest['action_names'], s.finger_coupling_options)
        self._coupling_observation_ids = (None if self.coupling_observation_contract is None else
            torch.tensor(self.coupling_observation_contract['retained_base_indices'], device=device))
        self.future_assignments = self.future_seconds = None
        if self.performance['anticipatory_fingering_time'] > 0:
            self.future_assignments = self.exact_score.future
            self.future_seconds = self.exact_score.future_seconds
        if s.control_limiter is not None:
            if (any(row['max_velocity'] is not None for row in s.control_limits['channels'].values())
                    and not any(field['path']=='action' for field in
                                b.manifest['contract']['observation_fields_before_stack'])):
                raise ValueError('Target slew requires the applied previous-action observation')
        if s.has_action_transform:
            # reset() fills only the newest history frame from this real target.
            self.logic.reset_action=s.control_reset_action.clone()
        self.reward_manager = RewardManager(
            self.reward_config,s.bundle.manifest,b.manifest,num_envs=num_envs,device=device)
        if s.has_action_transform:
            self.reward_manager.neutral=(s.tensor(s.bundle.neutral_action[:50]).clone()
                if (s.bundle.preparation_options or {}).get('mode') == 'first_note_v1'
                else s.control_reset_action[:50].clone())
            if self.reward_manager.gesture_active:
                manager=self.reward_manager
                manager.gesture_pose={name:float(manager.neutral[i])
                                      for i,name in enumerate(manager.actions)
                                      if '_Palm_Base_' not in name}
        if self.performance.get('upcoming_press_permission_time', 0.):
            options = s.finger_clearance_options
            if (s.finger_clearance_guard is None or options['active_command_mode'] != 'active_contact_v1'
                    or options.get('idle_finger_protection', True)
                    or options.get('active_press_permission', True)):
                raise ValueError('Upcoming press permission requires active_contact_v1 with both physical projections disabled')
        self.gesture_protected_fingers = None
        if self.reward_manager is not None and self.reward_manager.gesture_active:
            self.gesture_protected_fingers=gesture_protection_mask(
                torch.as_tensor(b.fingers, device=device), self.reward_manager.gesture_lookahead_steps)
        m = b.manifest
        self.tip_ids = torch.as_tensor(s.bundle.ids(mujoco.mjtObj.mjOBJ_SITE,m['fingertip_sites']),device=device)
        self.key_geoms = torch.as_tensor(s.bundle.ids(mujoco.mjtObj.mjOBJ_GEOM,m['key_geoms']),device=device)
        options_from_manifest(m.get('fingering_target'), m.get('task_profile', 'tsensei_v1'))
        references = [key_reference(s.model, gid) for gid in self.key_geoms.cpu().numpy()]
        self.key_site_ids = torch.as_tensor([r[0] for r in references], device=device)
        half_sizes = np.array([r[1] for r in references])
        offsets = np.array([target_position(np.zeros(3), size,
                            s.model.geom(gid).name, task_profile=m.get('task_profile', 'tsensei_v1'))
                            for gid, size in zip(self.key_geoms.cpu().numpy(), half_sizes)], dtype=np.float32)
        self.key_offsets = s.tensor(offsets)
        self.finger_key_offsets = s.tensor(np.repeat(offsets[None], 10, axis=0))
        self.hand_qadr = []
        self.hand_present = []
        self.hand_fixed = []
        for names in m['hand_joints']:
            present = [n not in s.bundle.fixed_joint_values for n in names]
            # Removed joints have no physics address; restore their fixed values in observations.
            ids = [s.model.joint(n).qposadr[0] if keep else 0 for n, keep in zip(names, present)]
            self.hand_qadr.append(torch.tensor(ids, device=device))
            self.hand_present.append(s.tensor(present))
            self.hand_fixed.append(s.tensor([s.bundle.fixed_joint_values.get(n, 0.) for n in names]))
        self.palms = []
        # Historical tasks lack the new field and remain palm-only.
        for names in task_collision_groups(m):
            ids=s.bundle.ids(mujoco.mjtObj.mjOBJ_GEOM,names)
            mask=torch.zeros(s.model.ngeom,device=device,dtype=torch.bool)
            mask[torch.as_tensor(ids,device=device)]=True
            self.palms.append(mask)
        self.forearm_pairs = None
        self.forearm_pair_ids = None
        if 'forearm_collision_pairs' in m:
            self.forearm_pairs = torch.zeros((s.model.ngeom,s.model.ngeom),device=device,dtype=torch.bool)
            for first,second in m['forearm_collision_pairs']:
                a,b=s.model.geom(first).id,s.model.geom(second).id
                self.forearm_pairs[a,b]=self.forearm_pairs[b,a]=True
            self.forearm_pair_ids=torch.tensor([[s.model.geom(a).id,s.model.geom(b).id]
                for a,b in m['forearm_collision_pairs']],device=device)
            # Strictly inscribed spheres cover the coincident-center CCD degeneracy.
            # They cannot report a collision outside the original convex hulls.
            from scipy.spatial import ConvexHull
            radii=np.zeros(s.model.ngeom)
            for gid in torch.unique(self.forearm_pair_ids).cpu().tolist():
                mesh=int(s.model.geom_dataid[gid])
                if s.model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                start=int(s.model.mesh_vertadr[mesh]);count=int(s.model.mesh_vertnum[mesh])
                equations=ConvexHull(s.model.mesh_vert[start:start+count]).equations
                radii[gid]=max(0,float(np.min(-equations[:,3])))
            self.forearm_inner_radius=s.tensor(radii)
        self.contact_slots = torch.arange(s.sim.wp_data.naconmax,device=device)
        self.rest_key_top_z = (s.data.site_xpos[0,self.key_site_ids,2]
            + s.tensor(half_sizes[:,2])).clone()
        if self.exact_score is None:
            raise ValueError('Current key/fingering rewards require exact_note_timing and original score events')
        # Base diagnostics compute the same key formula even when its weight is
        # disabled, so history must not disappear with the reward switch.
        pairs = self.forearm_pairs
        if pairs is None:
            pairs = ((self.palms[0][:,None] & self.palms[1][None,:])
                     | (self.palms[1][:,None] & self.palms[0][None,:]))
        s.configure_reward_sampling(pairs, collision_margin=m['collision_margin'],
            tip_ids=self.tip_ids.cpu().numpy(), key_site_ids=self.key_site_ids.cpu().numpy(), key_offsets=offsets,
            key_hold_geom_ids=(self.key_geoms.cpu().numpy()
                if self.reward_config['terms']['key_hold_lateral_motion_penalty']['enabled'] else None))
        self.reset()

    @property
    def observation_size(self):
        base = (self.logic.observation_size if self.coupling_observation_contract is None else
                self.coupling_observation_contract['reduced_base_size'])
        return (base + int(self.scene.policy_preparation)
                + (50 if self.future_assignments is not None else 0)
                + (176 if self.exact_score is not None else 0)
                + (self.scene.arm_follower.observation_size if getattr(self.scene, 'arm_follower', None) is not None else 0))

    def _with_preparation(self, observation):
        if self._coupling_observation_ids is not None:
            observation = observation[:, self._coupling_observation_ids]
        phase = self.scene.preparation_state()
        if phase is not None:
            observation = torch.cat((observation,phase),dim=-1)
        if self.future_assignments is not None:
            _, future_observation = self._anticipation()
            observation = torch.cat((observation, future_observation), dim=-1)
        if self.exact_score is not None:
            index = self.logic.index.clamp_max(self.logic.length - 1)
            observation = torch.cat((observation, self.exact_score.observation[index]), dim=-1)
        if getattr(self.scene, 'arm_follower', None) is not None:
            observation = torch.cat((observation, self.scene.arm_follower.observation()), dim=-1)
        return observation

    def _key_targets(self):
        d = self.scene.data
        return (d.site_xpos[:,None,self.key_site_ids]+self.finger_key_offsets[None]
                if self.finger_key_offsets is not None else
                d.site_xpos[:,self.key_site_ids]+self.key_offsets)

    def _anticipation(self):
        index = self.logic.index.clamp_max(self.logic.length-1)
        assignments, seconds = self.future_assignments, self.future_seconds
        # A longer request-permission window must not extend the explicit reward horizon.
        visible = seconds[index] <= self.performance['anticipatory_fingering_time'] + 1e-7
        return anticipation_fields(assignments[index] * visible[..., None], seconds[index] * visible,
            self._key_targets(), self.scene.data.site_xpos[:,self.tip_ids], self.rest_key_top_z)

    @property
    def action_size(self):
        return self.scene.action_size

    @property
    def reward_term_names(self):
        if self.reward_manager is not None:
            return self.reward_manager.names
        return list(self.bundle.manifest['reward_terms']) + (
            ['self_collision_reward'] if self.self_collision_weight else [])

    def snapshot(self, *, endpoint=False):
        s=self.scene
        d=s.data
        raw=d.qpos[:,s.key_qadr]
        keys=raw.clamp(s.key_limits[:,0],s.key_limits[:,1])
        contacts = s.self_query_data.contact if s.hand_collision_detection_only else d.contact
        if s.hand_collision_detection_only:
            geom = wp.to_torch(contacts.geom).long().clamp(0,s.model.ngeom-1)
            dist = wp.to_torch(contacts.dist)
            nacon = wp.to_torch(s.self_query_data.nacon)[0]
            world = wp.to_torch(contacts.worldid).long().clamp(0,self.num_envs-1)
        else:
            geom = contacts.geom[:].long().clamp(0,s.model.ngeom-1)
            dist, nacon = contacts.dist[:], d.nacon[0]
            world = contacts.worldid[:].long().clamp(0,self.num_envs-1)
        valid=(torch.arange(len(dist),device=self.device)<nacon) & (dist<=self.bundle.manifest['collision_margin'])
        palm=valid & ((self.palms[0][geom[:,0]] & self.palms[1][geom[:,1]])
                     | (self.palms[1][geom[:,0]] & self.palms[0][geom[:,1]]))
        if self.forearm_pairs is not None:
            palm=valid & self.forearm_pairs[geom[:,0],geom[:,1]]
        collisions=torch.zeros(self.num_envs,device=self.device,dtype=torch.int64)
        collisions.scatter_add_(0,world,palm.long())
        if self.forearm_pair_ids is not None:
            first,second=self.forearm_pair_ids.T
            distance=(d.geom_xpos[:,first]-d.geom_xpos[:,second]).norm(dim=-1)
            radius=self.forearm_inner_radius[first]+self.forearm_inner_radius[second]
            collisions+=(distance < radius-1e-7).any(-1).long()
        arm_collision_any = (s.arm_collision_tracker.count_control_substeps() > 0
                             if s.arm_collision_tracker is not None
                             else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        palm_collision_substeps = None
        if getattr(s, 'reward_sampling', None) is not None:
            palm_collision_substeps = s.reward_sampling.palm_collision_substeps.clone()
            # Detection-only arm contacts are sampled at the same physical
            # cadence.  Count a control substep once even if several arm pairs
            # are touching during that substep.
            if s.arm_collision_tracker is not None:
                palm_collision_substeps = torch.minimum(
                    palm_collision_substeps + s.arm_collision_tracker.count_control_substeps(),
                    torch.full_like(palm_collision_substeps, s.decimation))
        result = {'right_qpos' :d.qpos[:,self.hand_qadr[0]]*self.hand_present[0]+self.hand_fixed[0],'left_qpos':d.qpos[:,self.hand_qadr[1]]*self.hand_present[1]+self.hand_fixed[1],
                'key_state':keys/s.key_limits[:,1],
                'key_activation':keys>=s.key_limits[:,1]*s.bundle.manifest['key_activation_fraction'],
                'sustain':s.sustain[:,None],
                'fingertips':d.site_xpos[:,self.tip_ids],
                'key_targets':self._key_targets(),
                # Composer legacy_step refreshes velocity/positions with mj_step1,
                # while force remains from the final integrated substep.
                'power':(s.last_step_actuator_force[:,s.action_ids]*d.actuator_velocity[:,s.action_ids]).abs().sum(-1),
                # Arm conflicts are detection-only and share the forearm
                # reward signal without entering the force solver.
                'palm_collision':(collisions>0) | arm_collision_any,
                'arm_collision_any':arm_collision_any}
        if palm_collision_substeps is not None:
            result['palm_collision_substeps'] = palm_collision_substeps
        if self.future_assignments is not None:
            result.update(self._anticipation()[0])
        sampling = getattr(s, 'reward_sampling', None)
        if sampling is not None:
            history = sampling.key_state_history
            activation = sampling.key_activation_history
            result.update(key_state_history=history, key_activation_history=activation,
                          fingertip_history=sampling.fingertip_history,
                          key_target_history=sampling.key_target_history,
                          key_activation_any=activation.any(1), power_mean=sampling.power_mean,
                          palm_collision_any=sampling.palm_collision_any | arm_collision_any)
        if sampling is not None and sampling.key_hold_lateral is not None:
            result.update(key_hold_lateral_velocity_history=sampling.key_hold_lateral.velocity_history,
                          key_hold_contact_history=sampling.key_hold_lateral.contact_history)
        if self.exact_score is not None and endpoint:
            index = self.logic.index.clamp_max(self.logic.length - 1)
            result['exact_goal_history'] = self.exact_score.goal_history[index]
            result['exact_finger_key_history'] = self.exact_score.finger_key_history[index]
        if self.reward_manager is not None:
            power = s.last_step_actuator_force[:,s.action_ids]*d.actuator_velocity[:,s.action_ids]
            if s.disable_wrist_tz:
                full_power = torch.zeros((self.num_envs, 50), device=self.device)
                full_power[:,s.physical_ids] = power
                power = full_power
            result['signed_power'] = power
            if s.self_collision is not None and self.reward_config['terms']['self_collision_reward']['enabled']:
                result['self_collision_pairs'] = s.self_collision.count(('self',))
                if s.self_collision.accumulate_pair_substeps:
                    result['self_collision_pair_substeps'] = s.self_collision.count_finger_pair_substeps()
        if self.gesture_protected_fingers is not None:
            result['gesture_protected_fingers']=self.gesture_protected_fingers[
                self.logic.index.clamp_max(self.logic.length-1)]
        if s.finger_clearance_guard is not None:
            result.update(s.finger_clearance_metrics)
        if getattr(s, 'arm_follower', None) is not None:
            result['arm_motion_excess'] = s.arm_follower.excess.float()
            result.update(self.arm_tracking.snapshot())
        if s.arm_collision_tracker is not None:
            result['arm_collision_substeps'] = s.arm_collision_tracker.count_substeps()
            for category in s.arm_collision_tracker.CATEGORIES:
                result[f'{category}_collision_substeps'] = s.arm_collision_tracker.count_substeps(category)
        return result

    def _update_timing_clearance(self):
        s = self.scene
        guard = s.finger_clearance_guard
        window = self.performance.get('upcoming_press_permission_time', 0.)
        if not window:
            return
        index = self.logic.index.clamp_max(self.logic.length - 1)
        end_keys = self.exact_score.fingers[index]
        with s.scope():
            # Geometry and current fingering use the action's control frame.
            requested = s.finger_clearance_metrics['finger_requested_deficit_m']
            permitted = guard.upcoming_request_metrics(s.last_applied_physical_action,
                s.last_requested_clearance_support, self.exact_score.fingers[index],
                self.future_assignments[index], self.future_seconds[index], window, requested)
            s.finger_clearance_metrics.update(permitted)
            s.finger_clearance_metrics['finger_clearance_deficit_m'] = guard.reward_deficit(
                {'finger_clearance_deficit_m': s.finger_clearance_metrics['finger_actual_deficit_m']},
                permitted['finger_reward_requested_deficit_m'], end_keys)

    def _sample_arm_tracking(self, scene):
        follower = scene.arm_follower
        # Same preparation-time convention as the controller, but the policy
        # request is deliberately read BEFORE TCP target/IK limiting. Using
        # follower.last.target_position here would conceal requested-target lag.
        times = (scene.control_steps * scene.decimation + self.arm_tracking.sample_count + 1) * follower.dt
        requested = follower.requests(scene.last_requested_physical_action, times)
        sides = ('right', 'left')
        self.arm_tracking.sample(
            torch.stack([requested[side][0] for side in sides], dim=1),
            torch.stack([scene.data.xpos[:, follower.wrists[side]['body_id']] for side in sides], dim=1),
            torch.stack([follower.last[side]['target_position'] for side in sides], dim=1))

    def reset(self, env_ids=None):
        ids=self.scene.env_ids(env_ids)
        if ids.numel()==0:return self._with_preparation(self.logic.observation())
        self.scene.reset(ids)
        if self.arm_tracking is not None:
            self.arm_tracking.reset(ids)
        if self.reward_manager is not None:
            self.reward_manager.reset(ids)
        return self._with_preparation(self.logic.reset(ids,self.snapshot()))

    @torch.no_grad()
    def step(self, action):
        if self.logic.done.any():
            raise RuntimeError('Reset completed env_ids before stepping; consume the final transition first')
        action=self.scene.tensor(action)
        if self.arm_tracking is not None:
            self.arm_tracking.begin_control()
        # PianoScene validates shape, finite values and normalized bounds before mutation.
        self.scene.step(action,substep_callback=(self._sample_arm_tracking if self.arm_tracking is not None else None),
            assigned_keys=(None if self.scene.finger_clearance_guard is None else
            self.exact_score.fingers[self.logic.index.clamp_max(self.logic.length-1)]))
        if self.arm_tracking is not None and self.arm_tracking.sample_count != self.scene.decimation:
            raise RuntimeError('Arm tracking requires exactly one sample per physical substep')
        if self.scene.data.overflow[:].any():
            raise RuntimeError('Physics contact/constraint capacity overflow; abort rollout')
        physical=(self.scene.physical_action(action) if not self.scene.has_action_transform else
                  self.scene.last_applied_physical_action)
        self._update_timing_clearance()
        snapshot=self.snapshot(endpoint=True)
        # Raw TX/TY demand must survive target slew/acceleration projection.
        # With no transforms the physical action itself is the raw request.
        snapshot['requested_physical_action']=(self.scene.last_requested_physical_action
            if self.scene.has_action_transform else physical)
        result=self.logic.advance(physical,snapshot,reward_transform=self.reward_manager)
        result.observation = self._with_preparation(result.observation)
        if getattr(self.scene, 'arm_follower', None) is not None:
            follower = self.scene.arm_follower
            result.info['arm_motion_excess'] = follower.excess.float().clone()
            result.info.update({name: value.clone() for name, value in self.arm_tracking.snapshot().items()})
            for side, a in follower.sides.items():
                result.info[f'arm_{side}_q_rad'] = a['q'].clone()
                result.info[f'arm_{side}_velocity_rad_s'] = a['v'].clone()
                result.info[f'arm_{side}_acceleration_rad_s2'] = a['acceleration'].clone()
                for name, value in follower.last.get(side, {}).items():
                    result.info[f'arm_{side}_{name}'] = value.clone()
        if self.scene.arm_collision_tracker is not None:
            tracker = self.scene.arm_collision_tracker
            result.info['arm_collision_substeps'] = tracker.count_substeps().clone()
            result.info['arm_collision_pairs'] = tracker.count().clone()
            for category in tracker.CATEGORIES:
                result.info[f'{category}_collision_substeps'] = (
                    tracker.count_substeps(category).clone())
        if 'key_hold_contact_history' in snapshot:
            held = snapshot['key_hold_contact_history'].bool()
            speed = snapshot['key_hold_lateral_velocity_history'].abs()
            result.info['key_hold_contact_substeps_by_hand'] = held.sum(1)
            result.info['key_hold_lateral_peak_m_s_by_hand'] = (speed * held).amax(1)
        if 'key_activation_history' in snapshot:
            active = snapshot['key_activation_history']
            goal = snapshot.get('exact_goal_history', result.info['goal_applied'][:,None,:])[:, :, :88].bool()
            wrong = active & ~goal
            result.info.update(wrong_key_substeps=wrong.any(-1).sum(-1),
                wrong_key_key_substeps=wrong.sum((-1,-2)),
                substep_key_counts=torch.stack(((active & goal).sum((-1,-2)),
                    wrong.sum((-1,-2)), (~active & goal).sum((-1,-2))), -1),
                substep_count=torch.full((self.num_envs,), self.scene.decimation,
                    device=self.device, dtype=torch.int64))
        if self.exact_score is not None:
            result.info['goal_at_control_start'] = self.exact_score.goal[result.info['score_index']].clone()
            result.info['goal_substeps'] = snapshot['exact_goal_history'].clone()
            result.info['finger_keys_substeps'] = snapshot['exact_finger_key_history'].clone()
            phase = self.exact_substep_phase[result.info['score_index']]
            sample_values = torch.stack((torch.ones_like(wrong.any(-1), dtype=torch.int64),
                wrong.any(-1).long(), wrong.sum(-1), (active & goal).sum(-1),
                wrong.sum(-1), (~active & goal).sum(-1)), -1)
            result.info['substep_phase_counts'] = (sample_values[:, :, None, :] * phase[..., None]).sum(1)
        if self.scene.policy_preparation:
            result.info['preparation_active'] = result.info['score_index'] < self.preparation_steps
            result.info['preparation_remaining_fraction'] = self.scene.preparation_state()[:,0].clone()
        result.info['reward_raw_terms'] = self.reward_manager.raw_terms
        result.info['reward_weighted_terms'] = self.reward_manager.weighted_terms
        for name in ('self_collision_pairs','self_collision_pair_substeps','undesired_collision_pairs'):
            if name in snapshot:
                result.info[name] = snapshot[name]
        if self.collision_diagnostics:
            tracker=self.scene.self_collision
            result.info['self_collision_pairs']=tracker.count(('self',))
            result.info['inter_hand_collision_pairs']=tracker.count(('inter_hand',))
            for column,name in enumerate(('self_collision_substeps','inter_hand_collision_substeps',
                                          'hand_collision_substeps','collision_physics_substeps')):
                result.info[name]=tracker.substep_counts[:,column].clone()
            key_tracker=self.scene.key_contact_diagnostics
            if key_tracker is not None:
                for column,name in enumerate(key_tracker.COLUMNS):
                    result.info[name]=key_tracker.substep_counts[:,column].clone()
        if self.gesture_protected_fingers is not None:
            result.info['gesture_protected_fingers']=snapshot['gesture_protected_fingers'].clone()
        if self.scene.finger_clearance_guard is not None:
            result.info.update({name:value.clone() for name,value in self.scene.finger_clearance_metrics.items()})
        if self.scene.finger_coupler is not None:
            result.info.update({name:value.clone() for name,value in self.scene.finger_coupling_metrics.items()})
            fingers = self.logic.fingers[result.info['score_index']][:, [1, 2, 3, 4, 6, 7, 8, 9]]
            result.info['coupling_active'] = fingers.bool().any(-1)
        if self.scene.has_action_transform:
            result.info.update(raw_policy_action=action.clone(),
                requested_physical_action=self.scene.last_requested_physical_action.clone(),
                applied_physical_action=physical.clone(),
                control_clip_mask=self.scene.last_control_clip_mask.clone(),
                control_slew_mask=self.scene.last_control_slew_mask.clone())
        if self.scene.joint_limiter is not None:
            result.info['joint_limit_corrections_since_reset'] = wp.to_torch(
                self.scene.joint_limiter.counts).clone()
        if (not torch.isfinite(result.observation).all() or not torch.isfinite(result.reward).all()
                or not torch.isfinite(self.scene.data.qvel[:]).all()):
            raise FloatingPointError('Nonfinite task state; abort rollout rather than insert invalid replay data')
        return result
