"""Batched MJLab Simulation adapter for the frozen, complete piano scene.

This is a physical scene API, not an RL environment: no music reward, termination
or SAC observation is supplied by this layer; see PianoTaskEnv for music rules.
Native XML servos act directly, with no
additional MJLab PD controller or modifications to imported MuJoCo options.
"""
from contextlib import contextmanager
import gc

import numpy as np
import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg, Simulation

from .bundle import SceneBundle


class ImportedMujocoCfg(MujocoCfg):
    def apply(self, model):
        # Simulation normally overwrites XML options. The frozen scene is the
        # authoritative physics configuration at this migration stage.
        if model.opt.timestep != self.timestep:
            raise ValueError("Frozen scene timestep mismatch")


class PianoScene:
    def __init__(self, directory, *, device='cuda:0', num_envs=1, physics_graph=True,
                 track_self_collision=False, collision_categories=(), ignored_body_pairs=(), action_profile='legacy', disable_wrist_tz=False, disable_wrist_yaw=False, self_collision_detection_only=False, hand_collision_detection_only=False, preparation=None, collision_diagnostics=False, control_limits=None, finger_clearance=None, finger_coupling=None, self_collision_substep_sum=False, arm_follow=None):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('MJLab scene acceptance requires a CUDA device')
        self.bundle = SceneBundle(directory, disable_wrist_tz=disable_wrist_tz, disable_wrist_yaw=disable_wrist_yaw, self_collision_detection_only=self_collision_detection_only, hand_collision_detection_only=hand_collision_detection_only, preparation=preparation, arm_follow=arm_follow)
        self.arm_follow_options = arm_follow
        self.arm_follower = None
        self.hand_collision_detection_only = hand_collision_detection_only
        self.disable_wrist_tz = disable_wrist_tz
        from .finger_coupling import canonical as canonical_coupling, FingerCoupler
        self.finger_coupling_options = canonical_coupling(finger_coupling)
        self.action_size = len(getattr(self.bundle, 'policy_action_indices', self.bundle.active_action_indices)) + 1
        if self.finger_coupling_options is not None:
            self.action_size -= 16
        self.device = torch.device(device)
        wp.init()
        # A Warp-owned stream supports CUDA graph capture; Torch's default
        # stream is not a capturable Warp stream on all supported versions.
        self.stream = wp.get_stream(str(self.device))
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.num_envs = num_envs
        dt = self.bundle.manifest['physics_dt']
        self.decimation = round(self.bundle.manifest['control_dt'] / dt)
        if not np.isclose(self.decimation * dt, self.bundle.manifest['control_dt']):
            raise ValueError('Control timestep must be an integer multiple of physics dt')
        physics_model = self.bundle.model
        self.arm_collision_model = None
        self.arm_collision_data = None
        self.arm_collision_tracker = None
        self._arm_collision_graph = None
        self._arm_collision_step_graph = None
        query_model = getattr(self.bundle, 'arm_collision_query_model', None)
        if query_model is not None:
            # A separate collision-only model cannot accidentally add arm forces
            # to CPU stepping, the main GPU model, or its captured physics graph.
            with self.scope():
                self.arm_collision_model = mjwarp.put_model(query_model)
        with self.scope():
            self.sim = Simulation(num_envs=num_envs, model=physics_model, device=str(self.device),
                                  cfg=SimulationCfg(mujoco=ImportedMujocoCfg(timestep=dt),
                                                    nconmax=1024, njmax=4096))
        self.self_query_model = self.self_query_data = None
        if self.bundle.self_collision_query_model is not None:
            with self.scope():
                query = self.bundle.self_collision_query_model
                self.self_query_model = mjwarp.put_model(query)
                self.self_query_data = mjwarp.make_data(query, nworld=num_envs, nconmax=1024, njmax=4096)
                # Only transforms are shared. Contacts/constraints have separate storage,
                # and the query model is never stepped or passed to the force solver.
                for field in ('geom_xpos', 'geom_xmat', 'xpos', 'xquat', 'xmat'):
                    setattr(self.self_query_data, field, getattr(self.sim.wp_data, field))
        self.model = self.sim.mj_model
        self.data = self.sim.data
        if query_model is not None:
            from .arm_collision import ArmCollisionTracker
            with self.scope():
                self.arm_collision_data = mjwarp.make_data(
                    query_model, nworld=num_envs, nconmax=8192, njmax=4096)
                # Only kinematic transforms are shared.  Query contacts and
                # capacity remain private to the detection-only model.
                for field in ('geom_xpos', 'geom_xmat', 'xpos', 'xquat', 'xmat'):
                    setattr(self.arm_collision_data, field, getattr(self.sim.wp_data, field))
            self.arm_collision_tracker = ArmCollisionTracker(
                query_model, self.bundle.arm_collision_query_pairs,
                num_envs=num_envs, device=self.device)
        from .action_mapping import active_indices, physical_bounds, policy_bounds, resolved_contract
        self.action_profile = action_profile
        self.policy_ids = torch.tensor(active_indices(self.action_size, action_profile), device=self.device)
        self.physical_ids = torch.tensor(list(self.bundle.active_action_indices), device=self.device)
        self.low = self.tensor(self.bundle.low)
        self.high = self.tensor(self.bundle.high)
        policy_low, policy_high = policy_bounds(action_profile, self.action_size)
        self.action_low, self.action_high = self.tensor(policy_low), self.tensor(policy_high)
        # Kept as a center alias for old diagnostics; direct_joint_v1 does not
        # use a grouped shift to define its mapping.
        self.action_shift = (self.action_low + self.action_high) * .5
        physical_low, physical_high = physical_bounds(
            action_profile, self.bundle.manifest['action_names'], self.bundle.low, self.bundle.high,
            range_scales=self.bundle.manifest.get('action_range_scales'))
        self.physical_low, self.physical_high = self.tensor(physical_low), self.tensor(physical_high)
        self.action_contract = resolved_contract(
            action_profile, self.action_size, self.bundle.manifest['action_names'],
            self.bundle.low, self.bundle.high, range_scales=self.bundle.manifest.get('action_range_scales'))
        self.action_ids = torch.as_tensor(self.bundle.action_ids, device=self.device)
        self.finger_coupler = (None if self.finger_coupling_options is None else FingerCoupler(
            self.bundle, device=self.device, options=self.finger_coupling_options,
            physical_low=physical_low, physical_high=physical_high))
        self.finger_coupling = None if self.finger_coupler is None else self.finger_coupler.contract
        self.finger_coupling_metrics = {}
        self._coupling_graph = None
        self._coupling_substep_actual_error_max = None
        self.key_qadr = torch.as_tensor(self.bundle.key_qadr, device=self.device)
        self.key_limits = self.tensor(self.model.jnt_range[self.bundle.key_ids])
        self.sustain = torch.zeros(num_envs, device=self.device)
        from .control_limits import canonical, resolve, TargetLimiter
        self.control_limits_options=canonical(control_limits)
        reset_action=self.bundle.neutral_action.copy()
        reset_action[self.bundle.active_action_indices]=self.bundle.reset_ctrl[self.bundle.action_ids]
        reset_action[-1]=0.
        self.control_limits=resolve(self.control_limits_options,
            action_names=self.bundle.manifest['action_names'],active_indices=getattr(
                self.bundle, 'policy_action_indices', self.bundle.active_action_indices),
            physical_low=physical_low,physical_high=physical_high,reset_action=reset_action,
            control_dt=self.bundle.manifest['control_dt'])
        if self.finger_coupler is not None and self.control_limits is not None:
            followers = set(self.finger_coupling['follower_full_indices'])
            if any(row['index'] in followers for row in self.control_limits['channels'].values()):
                raise ValueError('Explicit PIP/DIP limits or slew are unsupported with follower coupling')
        self.control_limiter=(None if self.control_limits is None else TargetLimiter(self.control_limits,device=self.device))
        self.joint_limiter = None
        if (self.control_limits_options or {}).get('joint_state_limits') == 'hard_v1':
            from .joint_limits import resolve_robot_limits, inward_bounds
            from .joint_limits_warp import HardJointLimits
            contract = resolve_robot_limits(self.model, key_ids=self.bundle.key_ids,
                action_ids=self.bundle.action_ids, active_indices=self.bundle.active_action_indices,
                physical_low=physical_low, physical_high=physical_high, control_contract=self.control_limits)
            self.control_limits['joint_state_contract'] = contract
            with self.scope():
                self.joint_limiter = HardJointLimits(contract, num_envs=num_envs, device=str(self.device))
            self.hard_target_low, self.hard_target_high = map(self.tensor, inward_bounds(physical_low, physical_high))
            joint_rows = {row['joint_id']: row for row in contract['joints']}
            for aid, index in zip(self.bundle.action_ids, self.bundle.active_action_indices):
                row = joint_rows.get(int(self.model.actuator_trnid[aid, 0]))
                if row is not None:
                    self.hard_target_low[index] = row['float32_min']
                    self.hard_target_high[index] = row['float32_max']
        from .finger_clearance import canonical as canonical_clearance, FingerClearanceGuard
        self.finger_clearance_options=canonical_clearance(finger_clearance)
        if (self.finger_coupler is not None and self.finger_clearance_options is not None
                and any(self.finger_clearance_options.get(name, True) for name in
                        ('idle_finger_protection', 'active_press_permission'))):
            raise ValueError('Coupling requires both finger clearance projection controls disabled')
        self.finger_clearance_guard=(None if self.finger_clearance_options is None else
            FingerClearanceGuard(self.bundle,device=self.device,options=self.finger_clearance_options))
        self.finger_clearance=(None if self.finger_clearance_guard is None else self.finger_clearance_guard.contract)
        self.has_action_transform=(self.control_limiter is not None or self.finger_clearance_guard is not None
                                   or self.finger_coupler is not None or arm_follow is not None)
        self.finger_clearance_metrics={}
        self._clearance_filter_graph=None
        self._clearance_actual_graph=None
        self.target_motion_limiter = None
        if self.has_action_transform:
            self.control_reset_action=self.tensor(reset_action)
            if self.joint_limiter is not None:
                self.control_reset_action = self.control_reset_action.clamp(self.hard_target_low, self.hard_target_high)
            self.last_applied_physical_action=self.control_reset_action.expand(num_envs,-1).clone()
            self.last_requested_physical_action=self.last_applied_physical_action.clone()
            self.last_control_clip_mask=torch.zeros_like(self.last_applied_physical_action,dtype=torch.bool)
            self.last_control_slew_mask=torch.zeros_like(self.last_applied_physical_action,dtype=torch.bool)
            if self.control_limits is not None and any(
                    'max_acceleration' in row for row in self.control_limits['channels'].values()):
                from .target_motion_limits import TargetMotionLimiter
                self.target_motion_limiter = TargetMotionLimiter(self.control_limits,
                    num_envs=num_envs, device=self.device, reset_action=self.control_reset_action,
                    physical_low=self.hard_target_low if self.joint_limiter is not None else self.physical_low,
                    physical_high=self.hard_target_high if self.joint_limiter is not None else self.physical_high)
                self.control_limits['target_motion_contract'] = self.target_motion_limiter.contract
        self.control_steps = torch.zeros(num_envs, device=self.device, dtype=torch.int64)
        from .preparation import is_policy, MOUNTS, support_kernel
        self.policy_preparation = is_policy(self.bundle.preparation_options)
        self.preparation_steps = 0
        if self.policy_preparation:
            config = self.bundle.preparation_options
            self.preparation_steps = round(config['duration']/self.bundle.manifest['control_dt'])
            self.preparation_physics_steps = self.preparation_steps*self.decimation
            if arm_follow is None:
                self._preparation_kernel = support_kernel()
                self._preparation_mount_ids = [int(self.model.body(name).mocapid[0]) for name in MOUNTS.values()]
                with self.scope():
                    self._preparation_counter = wp.from_torch(self.control_steps,dtype=wp.int64)
                    self._preparation_origins = wp.array(self.bundle.origin_mocap_pos,dtype=wp.vec3,device=str(self.device))
        self.last_step_actuator_force = torch.zeros((num_envs, self.model.nu), device=self.device)
        self.physics_graph = physics_graph
        self.control_graph = None
        self._captured_step_graph = None
        self.reward_sampling = None
        self.self_collision = None
        categories = set(collision_categories) | ({'self'} if track_self_collision else set())
        if type(collision_diagnostics) is not bool:
            raise ValueError('collision_diagnostics must be true/false')
        self.collision_diagnostics = collision_diagnostics
        self.key_contact_diagnostics=None
        if collision_diagnostics and self.finger_clearance_guard is not None:
            from .key_contact_diagnostics import KeyContactDiagnostics
            self.key_contact_diagnostics=KeyContactDiagnostics(self.model,num_envs,self.device)
            with self.scope():
                self.key_contact_diagnostics.sample(self.sim)
        if collision_diagnostics:
            categories |= {'self','inter_hand'}
        if categories:
            from .self_collision import SelfCollisionTracker
            self.self_collision = SelfCollisionTracker(self.model, num_envs, self.device,
                                                        categories=categories, ignored_body_pairs=ignored_body_pairs,
                                                        diagnostics=collision_diagnostics,
                                                        accumulate_pair_substeps=self_collision_substep_sum)
            # Compile the small Warp kernel before CUDA graph capture begins.
            with self.scope():
                self._sample_self_collision()
                if collision_diagnostics or self_collision_substep_sum:
                    self._sample_self_collision(count_substep=True)
        self.arm_ik = None
        if arm_follow is not None:
            from .arm_scene import SceneArmFollower
            follower_type = SceneArmFollower
            if arm_follow.get('execution_backend', 'torch') == 'numba_cpu':
                from .arm_native_scene import NativeSceneArmFollower
                follower_type = NativeSceneArmFollower
            self.arm_follower = follower_type(self.bundle, num_envs=num_envs, device=self.device)
        if 'm6' in self.bundle.manifest:
            from .arm_ik import GPUArmIK
            self.arm_ik = GPUArmIK(self)
        self.reset()
        if self.policy_preparation:
            # Compile before recording the control graph; no physical step.
            with self.scope():
                self._advance_preparation(0)

    def _advance_preparation(self, substep):
        if self.policy_preparation and self.arm_follow_options is None:
            wp.launch(self._preparation_kernel,dim=self.num_envs,
                      inputs=[self._preparation_counter,self.sim.wp_data.mocap_pos,
                              self._preparation_origins,*self._preparation_mount_ids,
                              self.decimation,substep,self.preparation_physics_steps,
                              self.bundle.preparation_options['lift']],device=str(self.device))

    def configure_reward_sampling(self, geom_pair_mask, *, collision_margin=0.0,
                                  tip_ids=None, key_site_ids=None, key_offsets=None):
        """Collect post-integration samples for key reward and diagnostics.

        Current rigid-wrist arm followers step alongside physics and use this
        same collector. The obsolete endpoint-only arm IK is unsupported.
        """
        if geom_pair_mask is None:
            self.reward_sampling = None
        else:
            from .substep_rewards import SubstepRewardCollector
            if isinstance(geom_pair_mask, torch.Tensor):
                geom_pair_mask = geom_pair_mask.detach().cpu().numpy()
            if np.shape(geom_pair_mask) != (self.model.ngeom, self.model.ngeom):
                raise ValueError('Substep reward geom-pair mask must match the compiled scene')
            with self.scope():
                self.reward_sampling = SubstepRewardCollector(
                    num_envs=self.num_envs, decimation=self.decimation, device=self.device,
                    key_qadr=self.bundle.key_qadr,
                    key_limits=self.model.jnt_range[self.bundle.key_ids],
                    action_ids=self.bundle.action_ids, geom_pair_mask=geom_pair_mask,
                    collision_margin=collision_margin,
                    key_activation_fraction=self.bundle.manifest['key_activation_fraction'],
                    tip_ids=tip_ids, key_site_ids=key_site_ids, key_offsets=key_offsets)
                # Compile only read-only kernels before capture; never step physics.
                self._sample_reward_substep(0)
                self.reward_sampling.clear()
        self.control_graph = None
        self._captured_step_graph = None

    def _sample_reward_substep(self, substep):
        # Euler leaves transforms at the pre-integration pose unless a hard
        # joint limiter/arm follower already refreshed them. Kinematics alone
        # refreshes post-step sites/geoms, without another contact solve/step.
        if self.reward_sampling.fingertip_history is not None and self.joint_limiter is None and self.arm_follower is None:
            mjwarp.kinematics(self.sim.wp_model, self.sim.wp_data)
        contact_data = self.self_query_data if self.hand_collision_detection_only else None
        if contact_data is not None and self.self_collision is None:
            mjwarp.collision(self.self_query_model, contact_data)
        self.reward_sampling.sample(self.sim, substep, contact_data=contact_data)

    def _sample_arm_collision(self):
        if self.arm_collision_tracker is not None:
            self.arm_collision_tracker.sample(self.arm_collision_data, device=self.sim.wp_device)

    def _query_and_sample_arm_collision(self):
        # Called on the scene's shared Torch/Warp stream, after forward. Keep
        # tracker sampling outside capture: it owns Torch temporaries and must
        # accumulate exactly once per physical substep, including the first.
        if (self.physics_graph and self.sim.use_cuda_graph
                and self.sim.step_graph is not None):
            if (self._arm_collision_graph is None
                    or self._arm_collision_step_graph is not self.sim.step_graph):
                # MJLab recreates its graphs after replacing simulation arrays.
                # Refresh the query aliases too before capturing their pointers.
                self._arm_collision_graph = None
                for field in ('geom_xpos', 'geom_xmat', 'xpos', 'xquat', 'xmat'):
                    setattr(self.arm_collision_data, field, getattr(self.sim.wp_data, field))
                # Compile/allocate outside capture without sampling or stepping
                # physics. The actual substep is queried again by graph replay.
                mjwarp.collision(self.arm_collision_model, self.arm_collision_data)
                # As in MJLab graph creation, do not let unrelated Warp array
                # destructors get recorded by a collection during capture.
                gc_enabled = gc.isenabled()
                if gc_enabled:
                    gc.disable()
                try:
                    with wp.ScopedCapture() as capture:
                        mjwarp.collision(self.arm_collision_model, self.arm_collision_data)
                finally:
                    if gc_enabled:
                        gc.enable()
                self._arm_collision_graph = capture.graph
                self._arm_collision_step_graph = self.sim.step_graph
            wp.capture_launch(self._arm_collision_graph)
        else:
            mjwarp.collision(self.arm_collision_model, self.arm_collision_data)
        self._sample_arm_collision()

    def preparation_state(self):
        """One observed phase determines support height/velocity from config."""
        if not self.policy_preparation:
            return None
        return (1-self.control_steps.to(torch.float32)/self.preparation_steps).clamp(0,1)[:,None]

    def _sample_self_collision(self, *, count_substep=False):
        if self.self_query_model is not None:
            mjwarp.collision(self.self_query_model, self.self_query_data)
        self.self_collision.sample(self.sim, data=self.self_query_data,count_substep=count_substep)

    def _ensure_control_graph(self):
        # Rebuild if MJLab reallocated arrays/recreated its graph. Record native
        # kernels directly: this Warp version cannot nest MJLab step graph
        # launches inside stream capture (CUDA error 900).
        if self._captured_step_graph is not self.sim.step_graph:
            with wp.ScopedCapture() as capture:
                for substep in range(self.decimation):
                    self._advance_preparation(substep+1)
                    mjwarp.step(self.sim.wp_model, self.sim.wp_data)
                    self._enforce_joint_limits()
                    if self.self_collision is not None:
                        self._sample_self_collision(count_substep=True)
                    if self.key_contact_diagnostics is not None:
                        self.key_contact_diagnostics.sample(self.sim)
                    if self.reward_sampling is not None:
                        self._sample_reward_substep(substep)
            self.control_graph = capture.graph
            self._captured_step_graph = self.sim.step_graph

    def _enforce_joint_limits(self, *, use_cached_forward=None):
        # ``use_cached_forward`` was an internal diagnostic switch.  Keep the
        # keyword accepted for callers from the GPU regression suite, but do
        # not reintroduce the unsafe eager-forward branch.
        if self.joint_limiter is not None:
            self.joint_limiter.apply(self.sim.wp_data)
            # Refresh geometry and contacts before all per-substep consumers.
            # Reuse MJLab's already-captured forward graph whenever it is
            # available.  Calling mjwarp.forward directly after sim.step()
            # would allocate a fresh collision context on every substep while
            # the step graph may still be in flight; with large batched
            # collectors that can surface as Warp CUDA error 700.  Simulation
            # falls back to eager forward on configurations without a graph.
            # The call remains a native kernel launch when the surrounding
            # control graph is being captured.
            self.sim.forward()

    def tensor(self, array):
        return torch.as_tensor(array, device=self.device, dtype=torch.float32)

    @contextmanager
    def scope(self):
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            yield
        caller.wait_stream(self.torch_stream)

    def env_ids(self, env_ids=None):
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
            if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError('env_ids must be a 1D integer sequence')
            if torch.any(ids < 0) or torch.any(ids >= self.num_envs) or ids.unique().numel() != ids.numel():
                raise ValueError('env_ids must be unique and in range')
        return ids.to(torch.int64)

    def reset(self, env_ids=None):
        ids = self.env_ids(env_ids)
        if ids.numel() == 0:
            return self.state()
        with self.scope():
            self.sim.reset(ids)
            self.data.qpos[ids] = self.tensor(self.bundle.reset_qpos)
            self.data.qvel[ids] = self.tensor(self.bundle.reset_qvel)
            self.data.ctrl[ids] = self.tensor(self.bundle.reset_ctrl)
            # Preparation is generated on a separate native world exactly once.
            # Never step this batched simulation during a subset reset: doing so
            # would advance every continuing episode without storing transitions.
            self.data.qacc_warmstart[ids] = self.tensor(self.bundle.reset_qacc_warmstart)
            if self.model.nmocap:
                self.data.mocap_pos[ids] = self.tensor(self.bundle.reset_mocap_pos)
                self.data.mocap_quat[ids] = self.tensor(self.bundle.reset_mocap_quat)
            if self.arm_ik is not None:
                self.arm_ik.reset(ids)
            if self.arm_follower is not None:
                self.arm_follower.reset(ids)
                self.arm_follower.publish(self.data)
            if self.joint_limiter is not None:
                wp.to_torch(self.joint_limiter.counts)[ids] = 0
                self.joint_limiter.apply(self.sim.wp_data, wp.from_torch(ids.to(torch.int32)))
                self.data.ctrl[ids[:, None], self.action_ids] = self.data.ctrl[ids[:, None], self.action_ids].clamp(
                    self.hard_target_low[self.physical_ids], self.hard_target_high[self.physical_ids])
            self.sustain[ids] = 0
            if self.has_action_transform:
                self.last_applied_physical_action[ids]=self.control_reset_action
                self.last_requested_physical_action[ids]=self.control_reset_action
                self.last_control_clip_mask[ids]=False
                self.last_control_slew_mask[ids]=False
            if self.target_motion_limiter is not None:
                self.target_motion_limiter.reset(ids)
            for values in self.finger_clearance_metrics.values():
                values[ids]=False if values.dtype==torch.bool else 0
            for values in self.finger_coupling_metrics.values():
                values[ids]=False if values.dtype==torch.bool else 0
            self.control_steps[ids] = 0
            if self.self_collision is not None:
                self.self_collision.clear(ids)
            if self.key_contact_diagnostics is not None:
                self.key_contact_diagnostics.clear(ids)
            if self.reward_sampling is not None:
                self.reward_sampling.clear(ids)
            if self.arm_collision_tracker is not None:
                self.arm_collision_tracker.clear(ids)
            self.sim.forward()
            if self.self_query_model is not None:
                mjwarp.collision(self.self_query_model, self.self_query_data)
            self.last_step_actuator_force[ids] = self.data.actuator_force[ids]
        return self.state()

    def physical_action(self, action):
        fraction = ((action-self.action_low)/(self.action_high-self.action_low)).clamp(0, 1)
        active_low = self.physical_low[self.policy_ids]
        active_high = self.physical_high[self.policy_ids]
        active = active_low + fraction*(active_high-active_low)
        if self.action_size != 51:
            physical = self.physical_low.expand(*active.shape[:-1], 51).clone()
            physical[..., self.policy_ids] = active
            fixed_indices = {
                'right_/forearm_tz': 22, 'left_/forearm_tz': 47,
                'right_/forearm_pitch': 23, 'left_/forearm_pitch': 48,
                'right_/forearm_yaw': 24, 'left_/forearm_yaw': 49,
            }
            for name, index in fixed_indices.items():
                # Cartesian TZ is a live request even though its obsolete
                # physical slide was removed by rigid arm augmentation.
                if (name in self.bundle.fixed_joint_values
                        and index not in getattr(self.bundle, 'policy_action_indices', ())):
                    physical[..., index] = self.bundle.fixed_joint_values[name]
            return physical
        return active

    def policy_action(self, physical):
        """Convert full physical targets to this scene's active policy coordinates."""
        physical = self.tensor(physical)
        if physical.shape[-1] != 51:
            raise ValueError('Expected full 51-channel physical action')
        selected = physical[..., self.policy_ids]
        low = self.physical_low[self.policy_ids]
        high = self.physical_high[self.policy_ids]
        fraction = ((selected-low)/(high-low)).clamp(0, 1)
        return self.action_low + fraction*(self.action_high-self.action_low)

    def _couple_finger_targets(self, physical):
        coupler = self.finger_coupler
        def run(inputs):
            return coupler.apply(inputs[0], actual_qpos=inputs[1], body_xmat=inputs[2])
        sources = (physical, self.data.qpos[:], self.data.xmat[:])
        if not self.physics_graph:
            return run(sources)
        if self._coupling_graph is None:
            inputs = tuple(source.clone() for source in sources)
            for _ in range(2):
                run(inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.torch_stream):
                outputs = run(inputs)
            self._coupling_inputs, self._coupling_outputs, self._coupling_graph = inputs, outputs, graph
        for dest, source in zip(self._coupling_inputs, sources):
            dest.copy_(source)
        self._coupling_graph.replay()
        physical, info = self._coupling_outputs
        return physical, {name: value.clone() for name, value in info.items()}

    def _filter_finger_targets(self, physical, assigned_keys):
        """Capture only stateless geometry arithmetic; keep targets and role inputs live."""
        guard=self.finger_clearance_guard
        support = guard.support(self.data.mocap_pos[:], self.data.xpos[:], self.data.xmat[:])
        self.last_requested_clearance_support = guard.projection_support(support).clone()
        if not self.physics_graph:
            return guard.filter(physical,actual_qpos=self.data.qpos[:],
                                mocap_pos=self.data.mocap_pos[:],assigned_keys=assigned_keys,support=support)
        if self._clearance_filter_graph is None:
            inputs=(physical.clone(),self.data.qpos[:].clone(),self.data.mocap_pos[:].clone(),assigned_keys.clone(),support.clone())
            def run():
                return guard.filter(inputs[0],actual_qpos=inputs[1],mocap_pos=inputs[2],assigned_keys=inputs[3],support=inputs[4])
            # No physics, randomness, counters or optimizer state is touched by warmup.
            for _ in range(2):run()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=self.torch_stream):
                outputs=run()
            self._clearance_filter_inputs=inputs
            self._clearance_filter_outputs=outputs
            self._clearance_filter_graph=graph
        for dest,source in zip(self._clearance_filter_inputs,
                              (physical,self.data.qpos[:],self.data.mocap_pos[:],assigned_keys,support)):
            dest.copy_(source)
        self._clearance_filter_graph.replay()
        projected,metrics=self._clearance_filter_outputs
        # Metrics can be reset for selected worlds without modifying graph outputs.
        return projected,{name:values.clone() for name,values in metrics.items()}

    def _actual_finger_clearance(self, assigned_keys):
        guard=self.finger_clearance_guard
        def evaluate(qpos,lift,keys):
            physical=guard.physical_qpos(qpos)
            result=guard.evaluate(physical,lift,keys)
            reference=physical.clone()
            reference[:,guard.finger_indices]=guard.origin_angles
            result['finger_pose_reference_feasible']=(
                guard.evaluate(reference,lift,keys)['finger_clearance_deficit_m']<=1e-6)
            return result
        support = guard.support(self.data.mocap_pos[:], self.data.xpos[:], self.data.xmat[:])
        if not self.physics_graph:
            return evaluate(self.data.qpos[:],support,assigned_keys)
        if self._clearance_actual_graph is None:
            inputs=(self.data.qpos[:].clone(),support.clone(),assigned_keys.clone())
            def run():
                return evaluate(*inputs)
            for _ in range(2):run()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=self.torch_stream):
                outputs=run()
            self._clearance_actual_inputs=inputs
            self._clearance_actual_outputs=outputs
            self._clearance_actual_graph=graph
        for dest,source in zip(self._clearance_actual_inputs,
                              (self.data.qpos[:],support,assigned_keys)):
            dest.copy_(source)
        self._clearance_actual_graph.replay()
        return self._clearance_actual_outputs

    def step(self, action, *, substep_callback=None, assigned_keys=None, assigned_end_keys=None):
        """Accept configured policy (num_envs, 51) action; return diagnostic physical state."""
        action = self.tensor(action)
        if action.shape != (self.num_envs, self.action_size) or not torch.isfinite(action).all():
            raise ValueError(f'Expected finite normalized action shape ({self.num_envs}, {self.action_size})')
        if torch.any((action < self.action_low-1e-6) | (action > self.action_high+1e-6)):
            raise ValueError('Actions outside configured policy bounds')
        if self.finger_clearance_guard is not None and assigned_keys is None:
            raise ValueError('The finger clearance guard requires current score finger/key assignments')
        with self.scope():
            physical = self.physical_action(action)
            if self.has_action_transform:
                self.last_requested_physical_action=physical.clone()
            if self.control_limiter is not None:
                previous=self.last_applied_physical_action.clone()
                previous[:,self.physical_ids]=self.data.ctrl[:,self.action_ids]
                previous[:,-1]=self.sustain
                physical,self.last_control_clip_mask,self.last_control_slew_mask=self.control_limiter.apply(physical,previous)
            if self.finger_coupler is not None:
                physical,self.finger_coupling_metrics=self._couple_finger_targets(physical)
                follower_ids = self.finger_coupler.indices[:, 1:].flatten()
                self.last_requested_physical_action[:, follower_ids] = physical[:, follower_ids]
                self._coupling_substep_actual_error_max = torch.zeros(
                    (self.num_envs, len(self.finger_coupler.groups)), device=self.device)
            if self.finger_clearance_guard is not None:
                physical,self.finger_clearance_metrics=self._filter_finger_targets(physical,assigned_keys)
            if self.joint_limiter is not None:
                physical = physical.clamp(self.hard_target_low, self.hard_target_high)
            if self.target_motion_limiter is not None:
                physical, motion_limited = self.target_motion_limiter.apply(physical)
                self.last_control_slew_mask |= motion_limited
            if self.has_action_transform:
                self.last_applied_physical_action=physical.clone()
            self.data.ctrl[:, self.action_ids] = physical[:, self.physical_ids]
            self.sustain[:] = physical[:, -1]
            if self.arm_follower is not None:
                self.arm_follower.begin_control()
            if self.self_collision is not None:
                self.self_collision.clear()
            if self.key_contact_diagnostics is not None:
                self.key_contact_diagnostics.clear()
            if self.reward_sampling is not None:
                self.reward_sampling.clear()
            if self.arm_collision_tracker is not None:
                self.arm_collision_tracker.clear_step()
            if (self.arm_follower is None and self.arm_collision_tracker is None
                    and self.finger_coupler is None
                    and self.physics_graph and substep_callback is None
                    and self.sim.step_graph is not None and not self.sim.nan_guard.enabled):
                self._ensure_control_graph()
                wp.capture_launch(self.control_graph)
            else:
                # Audio/contact diagnostics must still observe every substep.
                for substep in range(self.decimation):
                    self._advance_preparation(substep+1)
                    if self.finger_coupler is not None:
                        # Re-solve against the current physical qpos/xmat at
                        # 200Hz. This removes control-frame target hold while
                        # retaining real servo lag and contact.
                        physical, coupling_metrics = self._couple_finger_targets(physical)
                        self.finger_coupling_metrics.update(coupling_metrics)
                        self.data.ctrl[:, self.action_ids] = physical[:, self.physical_ids]
                    if self.arm_follower is not None:
                        # Wrist commands lead IK. The only actual hand root is
                        # rigidly attached to Link7; physics and rewards use it.
                        times = (self.control_steps*self.decimation + substep+1)*self.arm_follower.dt
                        self.arm_follower.step(self.arm_follower.requests(physical, times))
                        self.arm_follower.publish(self.data)
                        self.sim.forward()
                    self.sim.step()
                    self._enforce_joint_limits()
                    if self.arm_follower is not None:
                        self.sim.forward()
                    if self.finger_coupler is not None:
                        actual = self.finger_coupler.actual_diagnostics(self.data.xmat[:])
                        self._coupling_substep_actual_error_max = torch.maximum(
                            self._coupling_substep_actual_error_max,
                            actual['coupling_actual_error_deg'])
                    if self.arm_collision_model is not None:
                        self._query_and_sample_arm_collision()
                    if self.self_collision is not None:
                        self._sample_self_collision(count_substep=True)
                    if self.key_contact_diagnostics is not None:
                        self.key_contact_diagnostics.sample(self.sim)
                    if self.reward_sampling is not None:
                        self._sample_reward_substep(substep)
                    if substep_callback is not None:
                        substep_callback(self)
            self.control_steps += 1
            if self.finger_coupler is not None:
                # The final substep target is the applied previous-action
                # observation for the next control frame.
                self.last_applied_physical_action = physical.clone()
                follower_ids = self.finger_coupler.indices[:, 1:].flatten()
                self.last_requested_physical_action[:, follower_ids] = physical[:, follower_ids]
            self.last_step_actuator_force[:] = self.data.actuator_force[:]
            if self.arm_ik is not None:
                self.arm_ik.update()
                if self.joint_limiter is not None:
                    self.joint_limiter.apply(self.sim.wp_data)
            # Refresh positions/contact information at the returned qpos.
            self.sim.forward()
            if self.finger_coupler is not None:
                self.finger_coupling_metrics.update(self.finger_coupler.actual_diagnostics(self.data.xmat[:]))
                self.finger_coupling_metrics['coupling_substep_actual_error_deg_max'] = (
                    self._coupling_substep_actual_error_max.clone())
            if self.self_query_model is not None:
                mjwarp.collision(self.self_query_model, self.self_query_data)
            if self.self_collision is not None:
                self._sample_self_collision()
            # Arm query contacts were sampled after every physical substep;
            # do not sample the endpoint again or double-count the last sample.
            if self.reward_sampling is not None:
                self.reward_sampling.sample_contacts(
                    self.sim, contact_data=(self.self_query_data if self.hand_collision_detection_only else None))
            if self.finger_clearance_guard is not None:
                reward_keys = assigned_keys if assigned_end_keys is None else assigned_end_keys
                actual=self._actual_finger_clearance(reward_keys)
                self.finger_clearance_metrics['finger_actual_deficit_m']=actual['finger_clearance_deficit_m'].clone()
                self.finger_clearance_metrics['finger_pose_reference_feasible']=actual['finger_pose_reference_feasible'].clone()
                self.finger_clearance_metrics['finger_terminal_allowed']=actual['finger_terminal_allowed'].clone()
                self.finger_clearance_metrics['finger_clearance_deficit_m']=self.finger_clearance_guard.reward_deficit(
                    actual,self.finger_clearance_metrics['finger_requested_deficit_m'],reward_keys)
                for name in ('finger_active_force_permitted','finger_assigned_support',
                             'finger_key_bottom_deficit_m','finger_actual_pad_plane_m'):
                    if name in actual:
                        self.finger_clearance_metrics[name]=actual[name].clone()
        return self.state()

    def state(self):
        q = self.data.qpos[:, self.key_qadr].clamp(self.key_limits[:, 0], self.key_limits[:, 1])
        return {'qpos': self.data.qpos[:], 'qvel': self.data.qvel[:],
                'key_activation': q >= self.key_limits[:, 1] * self.bundle.manifest['key_activation_fraction'],
                'sustain': self.sustain, 'control_steps': self.control_steps}

    def cpu_snapshot(self, env_id=0):
        """Copy for diagnostics/rendering only; never used to advance GPU physics."""
        import mujoco
        if not 0 <= env_id < self.num_envs:
            raise ValueError("env_id out of range")
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos[env_id].cpu().numpy()
        data.qvel[:] = self.data.qvel[env_id].cpu().numpy()
        data.ctrl[:] = self.data.ctrl[env_id].cpu().numpy()
        if self.model.nmocap:
            data.mocap_pos[:] = self.data.mocap_pos[env_id].cpu().numpy()
            data.mocap_quat[:] = self.data.mocap_quat[env_id].cpu().numpy()
        mujoco.mj_forward(self.model, data)
        return data
