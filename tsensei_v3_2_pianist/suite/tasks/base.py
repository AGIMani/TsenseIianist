# 2026-10-25: Virtual palm_base naming unified; axis values and order unchanged.
# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base piano composer task."""

from typing import Sequence

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils

from tsensei_v3_2_pianist.models.hands import HandSide, tsensei_hand
from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as hand_consts
from tsensei_v3_2_pianist.models.piano import piano

# Timestep of the physics simulation, in seconds.
from tsensei_v3_2_pianist.physics_settings import load as load_physics_settings, apply_mjcf_piano

_PHYSICS_TIMESTEP = load_physics_settings()["physics_dt"]

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = load_physics_settings()["control_dt"]  # 40 Hz by default.

# Default position and orientation of the hands.
_LEFT_HAND_POSITION = hand_consts.LEFT_POSITION
_LEFT_HAND_QUATERNION = hand_consts.ATTACHMENT_QUATERNION
_RIGHT_HAND_POSITION = hand_consts.RIGHT_POSITION
_RIGHT_HAND_QUATERNION = hand_consts.ATTACHMENT_QUATERNION

_ATTACHMENT_YAW = 0  # Degrees.

# Keep the existing wrist workspace independent of the piano layout revision.
# The previous 1221 mm keyboard supplied this half-span; changing key geometry
# must not also change joint limits, actuator ranges or action normalization.
_FOREARM_LATERAL_HALF_RANGE = 1.221 / 2


class PianoOnlyTask(composer.Task):
    """Piano task with no hands."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        change_color_on_activation: bool = False,
        add_piano_actuators: bool = False,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self._piano = piano.Piano(
            change_color_on_activation=change_color_on_activation,
            add_actuators=add_piano_actuators,
        )
        arena.attach(self._piano)

        settings = load_physics_settings()
        # Shadow's piano rule is solref=(2*physics_dt, 1), now 5 ms at 400 Hz.
        # Explicit user contact settings are kept when using the configured dt.
        if physics_timestep != settings['physics_dt']:
            settings['piano']['solref'][0] = 2 * physics_timestep
        apply_mjcf_piano(self._piano.mjcf_model, settings)
        self._arena.mjcf_model.option.integrator = settings['integrator']
        self._arena.mjcf_model.option.impratio = settings['impratio']

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    @property
    def piano(self) -> piano.Piano:
        return self._piano

    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0


class PianoTask(PianoOnlyTask):
    def before_substep(self, physics, action, random_state):
        from tsensei_v3_2_pianist.finger_resistance import Resistance
        model = physics.model.ptr
        if getattr(self, '_resistance_model', None) is not model:
            self._resistance_model = model
            self._finger_resistance = Resistance(model)
        self._finger_resistance.update(physics.data.ptr)


    """Base class for piano tasks."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        gravity_compensation: bool = True,
        change_color_on_activation: bool = False,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        attachment_yaw: float = _ATTACHMENT_YAW,
        palm_base_dofs: Sequence[str] = hand_consts.DEFAULT_PALM_BASE_DOFS,
        exclude_assembly_collisions: bool = True,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._exclude_assembly_collisions = exclude_assembly_collisions
        if primitive_fingertip_collisions or reduced_action_space:
            raise ValueError("Shadow-specific collision/action options are unsupported for TsenseI")
        super().__init__(
            arena=arena,
            change_color_on_activation=change_color_on_activation,
            add_piano_actuators=False,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )

        self._right_hand = self._add_hand(
            hand_side=HandSide.RIGHT,
            position=_RIGHT_HAND_POSITION,
            quaternion=_RIGHT_HAND_QUATERNION,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            palm_base_dofs=palm_base_dofs,
        )
        self._left_hand = self._add_hand(
            hand_side=HandSide.LEFT,
            position=_LEFT_HAND_POSITION,
            quaternion=_LEFT_HAND_QUATERNION,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            palm_base_dofs=palm_base_dofs,
        )

    # Accessors.

    @property
    def left_hand(self) -> tsensei_hand.TsenseiHand:
        return self._left_hand

    @property
    def right_hand(self) -> tsensei_hand.TsenseiHand:
        return self._right_hand

    # Helper methods.

    def _add_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        palm_base_dofs: Sequence[str],
    ) -> tsensei_hand.TsenseiHand:
        joint_range = [-_FOREARM_LATERAL_HALF_RANGE, _FOREARM_LATERAL_HALF_RANGE]

        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand = tsensei_hand.TsenseiHand(
            side=hand_side,
            palm_base_dofs=palm_base_dofs,
            exclude_assembly_collisions=self._exclude_assembly_collisions,
        )
        hand.root_body.pos = position

        # Slightly rotate the palm bases inwards (Z-axis) to mimic human posture.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if hand_side == HandSide.LEFT else 1
        angle = np.radians(sign * attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override palm_base translation joint range.
        palm_base_tx_joint = hand.mjcf_model.find("joint", "palm_base_tx")
        if palm_base_tx_joint is not None:
            palm_base_tx_joint.range = joint_range
        palm_base_tx_actuator = hand.actuator_for_joint("palm_base_tx")
        if palm_base_tx_actuator is not None:
            palm_base_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        # Composer prefixes elements owned by attached models (for example,
        # right_/R_Index_SW). Root-owned actuators keep the compact Control
        # labels while still targeting namespaced hand joints.
        hand.promote_actuators(self._arena.mjcf_model)
        return hand
