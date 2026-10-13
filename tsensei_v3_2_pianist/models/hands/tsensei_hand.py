"""TsenseI Composer entity, implemented from the bundled URDF models."""
import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from mujoco_utils import spec_utils
from tsensei_v3_2_pianist.models.hands import base, urdf_model
from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as consts
from tsensei_v3_2_pianist.models.hands.fingertip_geometry import measure_distal_tip
from tsensei_v3_2_pianist.drive_settings import load as load_drive_settings
from tsensei_v3_2_pianist.forearm_collision import palm_housing_body_name

FINGERTIP_SITE_GROUP = 5  # Group 4 is already used by the piano's sensor sites.


class TsenseiHand(base.Hand):
    def _build(self, side=base.HandSide.RIGHT, forearm_dofs=consts.DEFAULT_FOREARM_DOFS,
               exclude_assembly_collisions=True):
        self._hand_side = side
        label = side.name.lower()
        self._prefix = label + "_hand_"
        self._mjcf_root, self._root_body = urdf_model.build(
            consts.ASSETS / "urdf" / f"TsenseI_{label}.urdf", label + "_")
        self._finger_joints = tuple(self._mjcf_root.find_all("joint"))
        self._drive_settings = load_drive_settings()
        if self._drive_settings['profile'] == 'shadow_v1':
            settings = self._drive_settings
            finger = settings['finger']
            for joint in self._finger_joints:
                joint.damping = finger['damping']
                joint.armature = finger['armature']
                joint.frictionloss = finger['frictionloss']
            for actuator in self._mjcf_root.find_all('actuator'):
                actuator.kp = finger['kp']
                actuator.forcerange = (-finger['force_limit'], finger['force_limit'])
        # Preserve the existing shell/thumb assembly exclusion by physical
        # role: with-ID uses base_link for the shell, not the fixed palm link.
        # TODO(V32-06): Validate/decompose the assembly geometry. This exclusion
        # applies throughout motion; disable it to inspect the original contacts.
        if exclude_assembly_collisions:
            self._mjcf_root.contact.add(
                "exclude",
                body1=self._mjcf_root.find("body", palm_housing_body_name(label)),
                body2=self._mjcf_root.find("body", f"{self._prefix}thumb_rotaback_link2"))
        if len(set(forearm_dofs)) != len(forearm_dofs):
            raise ValueError("Duplicate forearm DOFs")
        for dof in forearm_dofs:
            kind, axis, bounds, kp, damping, force = consts.FOREARM_DOFS[dof]
            if self._drive_settings['profile'] == 'shadow_v1':
                if dof in self._drive_settings['translation']['joints']:
                    kp = self._drive_settings['translation']['kp']
                    force = self._drive_settings['translation']['force_limit']
                elif dof == 'forearm_pitch':
                    # Numeric WRJ1 reference; preserve the user's V3.2 axis.
                    pitch = self._drive_settings['pitch']
                    kp, damping, force = pitch['kp'], pitch['damping'], pitch['force_limit']
            # Equal targets: yaw rotates both wrists in the same direction;
            # pitch rotates them in opposite directions from aligned frames.
            if side == base.HandSide.LEFT and dof == "forearm_pitch":
                axis = tuple(-a for a in axis)
            j = self.root_body.add("joint", name=dof, type=kind, axis=axis,
                                   range=bounds, limited=True, damping=damping)
            if (self._drive_settings['profile'] == 'shadow_v1'
                    and dof in self._drive_settings['translation']['joints']):
                # The current A3.2 hand has no modeled upstream forearm. Reflect
                # only the explicitly configured small residual translation
                # armature into the two orthogonal V3.2 slides; do not attach a
                # rigid mass below Pitch.
                j.armature = self._drive_settings['translation']['armature']
            self._mjcf_root.actuator.add("position", name=consts.control_label(side, dof), joint=j, kp=kp,
                                         ctrlrange=bounds, ctrllimited=True,
                                         forcerange=(-force,force), forcelimited=True)
        if self._drive_settings['profile'] == 'shadow_v1':
            # Like Shadow's _add_dofs, compile inertias and write damping back
            # into MJCF, so exported GPU scenes receive the identical values.
            from dm_control import mjcf
            import math
            physics = mjcf.Physics.from_mjcf_model(self._mjcf_root)
            try:
                for name in self._drive_settings['translation']['joints']:
                    joint = self._mjcf_root.find('joint', name)
                    if joint is not None:
                        joint_id = physics.model.name2id(joint.full_identifier, 'joint')
                        dof_id = int(physics.model.jnt_dofadr[joint_id])
                        inertia = float(physics.model.dof_M0[dof_id])
                        joint.damping = 2 * math.sqrt(inertia * self._drive_settings['translation']['kp'])
            finally:
                physics.free()
        self._joints = tuple(self._mjcf_root.find_all("joint"))
        self._actuators = tuple(self._mjcf_root.find_all("actuator"))
        self._fingertip_bodies, self._fingertip_sites = [], []
        for finger in consts.FINGERS:
            distal = "PIP" if finger == "thumb" else "DIP"
            body = self._mjcf_root.find("body", f"{self._prefix}{finger}_{distal}_link")
            # Provisional distal-face landmark, measured after URDF collision scale/origin transforms.
            # TODO(V32-07): Real hardware pad deformation still needs calibration.
            pos = measure_distal_tip(
                consts.ASSETS / "urdf" / f"TsenseI_{label}.urdf", body.name)
            site = body.add("site", name=finger+"_tip", pos=pos, type="sphere", size=(.004,),
                            rgba=(0, 0, 0, 1), group=FINGERTIP_SITE_GROUP)
            self._fingertip_bodies.append(body)
            self._fingertip_sites.append(site)

    def _build_observables(self):
        return TsenseiObservables(self)

    @property
    def mjcf_model(self): return self._mjcf_root
    @property
    def name(self): return self._mjcf_root.model
    @property
    def hand_side(self): return self._hand_side
    @property
    def root_body(self): return self._root_body
    @property
    def joints(self): return self._joints
    @property
    def actuators(self): return self._actuators

    def actuator_for_joint(self, joint_name):
        """Resolve an actuator by its stable source joint name."""
        return next((actuator for actuator in self.actuators
                     if actuator.joint.name == joint_name), None)

    def promote_actuators(self, root_model):
        """Move controls to the scene root so Composer adds no hand namespace.

        The actuator-to-joint references remain unchanged. This only controls
        the names presented by MuJoCo's Control panel after model composition.
        """
        promoted = []
        for actuator in self._actuators:
            attributes = actuator.get_attributes()
            actuator_type = actuator.tag
            actuator.remove()
            promoted.append(root_model.actuator.add(actuator_type, **attributes))
        self._actuators = tuple(promoted)
    @property
    def fingertip_bodies(self): return tuple(self._fingertip_bodies)
    @property
    def fingertip_sites(self): return tuple(self._fingertip_sites)

    def _initial(self, joint):
        short = joint.name.removeprefix(self._prefix).removesuffix("_joint")
        overrides = consts.INITIAL_ANGLES_BY_SIDE.get(self.hand_side.name.lower(), {})
        value = overrides.get(short, consts.INITIAL_ANGLES.get(short, 0.0))
        if not joint.range[0] <= value <= joint.range[1]:
            raise ValueError(f"Initial angle outside limits: {joint.name}={value}")
        return value

    @property
    def neutral_control(self):
        return np.array([self._initial(a.joint) for a in self.actuators])

    def initialize_episode(self, physics, random_state):
        del random_state
        physics.bind(self.joints).qpos = [self._initial(j) for j in self.joints]
        physics.bind(self.actuators).ctrl = self.neutral_control

    def action_spec(self, physics):
        return spec_utils.create_action_spec(physics, self.actuators, prefix=self.name)

    def apply_action(self, physics, action, random_state):
        del random_state
        action = np.asarray(action)
        spec = self.action_spec(physics)
        if action.shape != spec.shape or not np.isfinite(action).all():
            raise ValueError(f"Invalid action for {self.name}: {action}")
        if np.any(action < spec.minimum) or np.any(action > spec.maximum):
            raise ValueError(f"Action outside joint target limits for {self.name}")
        physics.bind(self.actuators).ctrl = action


class TsenseiObservables(base.HandObservables):
    @composer.observable
    def actuators_power(self):
        def power(physics):
            return np.abs(physics.bind(self._entity.actuators).force *
                          physics.bind(self._entity.actuators).velocity)
        return observable.Generic(power)

    @composer.observable
    def fingertip_positions(self):
        return observable.Generic(lambda physics: physics.bind(self._entity.fingertip_sites).xpos.ravel())

    @composer.observable
    def joints_torque(self):
        return observable.MJCFFeature("qfrc_actuator", self._entity.joints)
