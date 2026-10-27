"""2026-10-25: virtual forearm joint names are now palm_base_*; physical axes unchanged.

Portable frozen scene: MJCF/assets, named reset state and provenance."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) for i in range(count)]


def option_values(option):
    result = {}
    for name in dir(option):
        if name.startswith('_'):
            continue
        value = getattr(option, name)
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (int, float)):
            result[name] = value
    return result


def export_scene(output, *, gravity_compensation=True, control_timestep=.05, physics_timestep=.005,
                 build_reference=True):
    # Composer belongs only to the export side; the MJLab loader needs no dm_control.
    from dm_control import mjcf
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.models.piano.piano import _KEY_ACTIVATION_FRACTION
    from tsensei_v3_2_pianist.models.hands import tsensei_hand_constants as hand_consts
    import tsensei_v3_2_pianist as package

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a frozen scene: {output}; use a new directory")
    kwargs = {"gravity_compensation": gravity_compensation}
    for name, value in (("control_timestep", control_timestep), ("physics_timestep", physics_timestep)):
        if value is not None:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            kwargs[name] = value
    env = suite.load(task_kwargs=kwargs)
    try:
        env.reset()
        m, d = env.physics.model.ptr, env.physics.data.ptr
        mjcf.export_with_assets(env.task.root_entity.mjcf_model, str(output), "scene.xml")
        actuator_names = names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu)
        action_names = [a.name for h in (env.task.right_hand, env.task.left_hand) for a in h.actuators]
        action_ids = [actuator_names.index(n) for n in action_names]
        joint_names = names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt)
        assert m.nq == m.nv == m.njnt == 138 and m.nu == 50
        hands = {}
        for side, hand in (("right", env.task.right_hand), ("left", env.task.left_hand)):
            root = env.physics.bind(hand.root_body)
            hands[side] = {
                "body": hand.root_body.full_identifier,
                "position": root.xpos.tolist(), "rotation": root.xmat.tolist(),
                "sites": [s.full_identifier for s in hand.fingertip_sites],
                "fingertips": env.physics.bind(hand.fingertip_sites).xpos.tolist(),
            }
        spec = env.action_spec()
        from tsensei_v3_2_pianist.physics_settings import load as load_contact_settings
        settings = load_contact_settings()
        if float(m.opt.timestep) != settings['physics_dt']:
            settings['piano']['solref'][0] = 2 * float(m.opt.timestep)
        settings.update(physics_dt=float(m.opt.timestep), control_dt=float(env.control_timestep()))
        data = {
            "contact_settings": settings,
            "schema_version": 1, "stage": "physical scene only; no music task or SAC",
            "gravity_compensation": gravity_compensation,
            "mujoco_export_version": mujoco.__version__,
            "physics_options": option_values(m.opt),
            "exporter_default_options": option_values(mujoco.MjOption()),
            "nq": m.nq, "nv": m.nv, "nu": m.nu,
            "physics_dt": float(m.opt.timestep), "control_dt": float(env.control_timestep()),
            "joint_names": joint_names,
            "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(),
            "actuator_names": actuator_names, "ctrl": d.ctrl.tolist(),
            "action_names": action_names + ["sustain"],
            "action_min": spec.minimum.tolist(), "action_max": spec.maximum.tolist(),
            "neutral_action": d.ctrl[action_ids].tolist() + [0.0],
            "piano_joints": [j.full_identifier for j in env.task.piano.joints],
            "key_activation_fraction": float(_KEY_ACTIVATION_FRACTION), "hands": hands,
            "source_pose": {
                "contract": "initial_palm_base_world_xyz_v1",
                "world_position_unit": "m",
                "right_mount_world_xyz_m": hand_consts.RIGHT_POSITION,
                "left_mount_world_xyz_m": hand_consts.LEFT_POSITION,
                "right_initial_world_xyz_m": hands['right']['position'],
                "left_initial_world_xyz_m": hands['left']['position'],
                "palm_base_slide_axes_world": {"tx": [0, 1, 0], "ty": [-1, 0, 0], "tz": [0, 0, 1]},
                **{side + '_initial_palm_base_joint_q': {
                    name: hand_consts.INITIAL_ANGLES_BY_SIDE[side][name]
                    for name in hand_consts.PALM_BASE_DOFS} for side in ('right', 'left')},
            },
            "source_hashes": {},
            "limitations": ["V32-02 virtual wrist", "V32-04/05 provisional drive limits",
                            "V32-06 collision exclusions", "V32-07 provisional fingertips",
                            "V32-10 supplied right hand; manufacturer provenance unverified"],
        }
        root = Path(package.__file__).parent
        for pattern in ("assets/**/*", "models/**/*.py", "suite/**/*.py"):
            for path in sorted(root.glob(pattern)):
                if path.is_file() and "__pycache__" not in path.parts:
                    data["source_hashes"][str(path.relative_to(root))] = sha256(path)
        # Round-trip the exported file before accepting it as a portable bundle.
        loaded = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
        state = mujoco.MjData(loaded)
        state.qpos[:] = d.qpos
        state.qvel[:] = d.qvel
        state.ctrl[:] = d.ctrl
        mujoco.mj_forward(loaded, state)
        np.testing.assert_allclose(state.site_xpos, d.site_xpos, atol=1e-10, rtol=0)
        data["files"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        (output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
        if not build_reference:
            return data
        # Freeze CPU trajectories in the original environment's MuJoCo version,
        # so a newer/older MJLab dependency cannot silently change the reference.
        from .acceptance_cases import scenarios
        bundle = SceneBundle(output)
        reference = {}
        decimation = round(data['control_dt'] / data['physics_dt'])
        for name, actions in scenarios(bundle):
            native = bundle.reset_cpu()
            qpos = []
            for action in actions:
                native.ctrl[bundle.action_ids] = action[:-1]
                for _ in range(decimation):
                    mujoco.mj_step(bundle.model, native)
                mujoco.mj_forward(bundle.model, native)
                qpos.append(native.qpos.copy())
            reference[name+'_qpos'] = np.asarray(qpos)
            reference[name+'_action'] = actions
        np.savez_compressed(output/'cpu_reference.npz', **reference)
        data['files']['cpu_reference.npz'] = sha256(output/'cpu_reference.npz')
        (output/'manifest.json').write_text(json.dumps(data, indent=2)+'\n')
        return data
    finally:
        env.close()


class SceneBundle:
    def __init__(self, directory, *, disable_wrist_tz=False, disable_wrist_yaw=False, self_collision_detection_only=False, hand_collision_detection_only=False, preparation=None, arm_follow=None, preparation_context=None):
        from .preparation import options
        self.preparation_options = options(preparation)
        first_note = (self.preparation_options or {}).get('mode') == 'first_note_v1'
        if first_note and (arm_follow is not None or not disable_wrist_tz or not disable_wrist_yaw):
            raise ValueError('First-note preparation requires hand-only fixed TZ/Yaw')
        self.preparation = self.preparation_trace = None
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        from ..key_activation import validate_fraction
        if 'key_threshold' in self.manifest or 'key_activation_fraction' not in self.manifest:
            raise ValueError('Old key activation scene is unsupported; rebuild for normalized fraction 1.0')
        validate_fraction(self.manifest['key_activation_fraction'])
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported scene schema")
        for name, expected in self.manifest["files"].items():
            path = (self.directory / name).resolve()
            if path.parent != self.directory or sha256(path) != expected:
                raise ValueError(f"Scene asset missing or modified: {name}")
        if disable_wrist_tz and 'm6' in self.manifest:
            raise ValueError('--disable-wrist-tz requires a hand-only scene; use --no-arms')
        # Cartesian arm mode keeps the three translation joints and removes
        # only the two orientation joints.  Hand-only legacy scenes may still
        # request TZ removal explicitly.
        self.disable_wrist_tz = disable_wrist_tz
        self.fixed_joint_values = {}
        if disable_wrist_tz:
            self.fixed_joint_values.update({'right_/palm_base_tz': 0., 'left_/palm_base_tz': 0.})
        if disable_wrist_yaw:
            for side in ('right', 'left'):
                name = f'{side}_/palm_base_yaw'
                self.fixed_joint_values[name] = self.manifest['qpos'][self.manifest['joint_names'].index(name)]
        removed_actions = ({22,47} if disable_wrist_tz else set()) | ({24,49} if disable_wrist_yaw else set())
        self.active_action_indices = np.array([i for i in range(50)
            if i not in removed_actions])
        joint_keep = [i for i, n in enumerate(self.manifest['joint_names'])
                      if n not in self.fixed_joint_values]
        actuator_keep = [i for i, n in enumerate(self.manifest['actuator_names'])
                         if n not in [self.manifest['action_names'][i] for i in removed_actions]]
        if disable_wrist_tz or disable_wrist_yaw or self.preparation_options is not None:
            # Exclude joints BEFORE compiling: no hidden TZ dynamics or projection.
            import xml.etree.ElementTree as ET
            xml = ET.parse(self.directory / 'scene.xml').getroot()
            if disable_wrist_yaw:
                for body in xml.iter('body'):
                    yaw = next((j for j in body.findall('joint') if j.get('name', '').endswith('/palm_base_yaw')), None)
                    if yaw is None:
                        continue
                    joints = body.findall('joint')
                    if any(np.any(np.fromstring(j.get('pos','0 0 0'), sep=' ')) or float(j.get('ref','0')) != 0 for j in joints):
                        raise ValueError('Yaw baking requires zero joint anchors/reference')
                    angle = self.fixed_joint_values[yaw.get('name')]
                    axis = np.fromstring(yaw.get('axis'), sep=' ')
                    rotation = np.empty(4); mujoco.mju_axisAngle2Quat(rotation, axis, angle)
                    initial = np.fromstring(body.get('quat','1 0 0 0'), sep=' ')
                    combined = np.empty(4); mujoco.mju_mulQuat(combined, initial, rotation)
                    body.set('quat', ' '.join(map(str, combined)))
                    matrix = np.empty(9); mujoco.mju_quat2Mat(matrix, rotation)
                    # All wrist joints share a body; preserve the remaining
                    # translation axes
                    # axes when baking the final yaw into its rest orientation.
                    for joint in joints:
                        if joint is not yaw:
                            axis = matrix.reshape(3,3).T @ np.fromstring(joint.get('axis','0 0 1'), sep=' ')
                            joint.set('axis', ' '.join(map(str, axis)))
            removed = []
            for parent in xml.iter():
                for node in list(parent):
                    name = node.get('name', '')
                    if ((node.tag == 'joint' and name in self.fixed_joint_values)
                            or (parent.tag == 'actuator' and name in [self.manifest['action_names'][i] for i in removed_actions])):
                        removed.append(name)
                        parent.remove(node)
            if len(removed) != 2 * len(removed_actions):
                raise ValueError('Unexpected number of removed wrist joints/actuators')
            if self.preparation_options is not None and not first_note:
                from .preparation import wrap_hands
                wrap_hands(xml, self.manifest)
            assets = {name: (self.directory/name).read_bytes() for name in self.manifest['files']
                      if name not in ('scene.xml', 'cpu_reference.npz')}
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'), assets)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(self.directory / "scene.xml"))
        self.self_collision_query_model = None
        # Optional rigid-arm augmentation installs a separate detection-only
        # model here.  It is deliberately kept out of the force solver.
        self.arm_collision_query_model = None
        self.arm_collision_query_pairs = None
        if self_collision_detection_only or hand_collision_detection_only:
            if 'm6' in self.manifest:
                raise ValueError('Detection-only self contacts currently require --no-arms')
            import copy
            from ..detection_only_collision import disable_same_hand_response
            self.self_collision_query_model = copy.copy(self.model)
            disable_same_hand_response(self.model, include_cross_hand=hand_collision_detection_only)
        m = self.model
        self.unsupported_default_options = []
        for name, value in self.manifest['physics_options'].items():
            if not hasattr(m.opt, name):
                if value != self.manifest['exporter_default_options'].get(name):
                    raise ValueError(f'Runtime MuJoCo does not support nondefault option {name}')
                self.unsupported_default_options.append(name)
                continue
            current = getattr(m.opt, name)
            if isinstance(current, np.ndarray):
                current[:] = value
            else:
                setattr(m.opt, name, value)
        if (m.nq, m.nv, m.nu) != (138-len(removed_actions), 138-len(removed_actions), 50-len(removed_actions)):
            raise ValueError("Unexpected V3.2 scene dimensions")
        self.joint_ids = self.ids(mujoco.mjtObj.mjOBJ_JOINT, [self.manifest["joint_names"][i] for i in joint_keep])
        if not np.all(np.isin(m.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE])):
            raise ValueError("Stage 1 requires scalar joints")
        self.qadr = m.jnt_qposadr[self.joint_ids]
        self.vadr = m.jnt_dofadr[self.joint_ids]
        self.ctrl_ids = self.ids(mujoco.mjtObj.mjOBJ_ACTUATOR, [self.manifest["actuator_names"][i] for i in actuator_keep])
        self.action_ids = self.ids(mujoco.mjtObj.mjOBJ_ACTUATOR, [self.manifest["action_names"][i] for i in self.active_action_indices])
        self.key_ids = self.ids(mujoco.mjtObj.mjOBJ_JOINT, self.manifest["piano_joints"])
        self.key_qadr = m.jnt_qposadr[self.key_ids]
        self.reset_qpos = np.zeros(m.nq)
        self.reset_qvel = np.zeros(m.nv)
        self.reset_ctrl = np.zeros(m.nu)
        self.reset_qpos[self.qadr] = np.asarray(self.manifest["qpos"])[joint_keep]
        self.reset_qvel[self.vadr] = np.asarray(self.manifest["qvel"])[joint_keep]
        self.reset_ctrl[self.ctrl_ids] = np.asarray(self.manifest["ctrl"])[actuator_keep]
        self.neutral_action = np.asarray(self.manifest["neutral_action"])
        if disable_wrist_tz:
            self.neutral_action[[22, 47]] = 0
        self.low = np.asarray(self.manifest["action_min"])
        self.high = np.asarray(self.manifest["action_max"])
        # Keep the complete original posture separately from the actually reached
        # preparation terminal state. Neither is silently projected onto the other.
        self.origin_qpos = self.reset_qpos.copy()
        self.origin_qvel = self.reset_qvel.copy()
        self.origin_ctrl = self.reset_ctrl.copy()
        origin = mujoco.MjData(m)
        self.origin_mocap_pos = origin.mocap_pos.copy()
        self.origin_mocap_quat = origin.mocap_quat.copy()
        if 'mocap_pos' in self.manifest:
            if self.preparation_options is not None:
                raise ValueError('Preparation does not support scenes with existing mocap states')
            self.origin_mocap_pos[:] = self.manifest['mocap_pos']
            self.origin_mocap_quat[:] = self.manifest['mocap_quat']
        self.reset_mocap_pos = self.origin_mocap_pos.copy()
        self.reset_mocap_quat = self.origin_mocap_quat.copy()
        self.reset_qacc_warmstart = np.zeros(m.nv)
        if self.preparation_options is not None:
            from .preparation import generate, is_policy, policy_initial
            prepare = policy_initial if is_policy(self.preparation_options) else generate
            if first_note:
                from .first_note_preparation import generate as prepare_first_note
                self.preparation, self.preparation_trace, terminal = prepare_first_note(
                    self, self.preparation_options, preparation_context)
            else:
                self.preparation, self.preparation_trace, terminal = prepare(self, self.preparation_options)
            for name, value in terminal.items():
                setattr(self, 'reset_'+name, value)
        if arm_follow is not None:
            from .arm_scene import augment_bundle
            augment_bundle(self, arm_follow)
            if self_collision_detection_only or hand_collision_detection_only:
                raise ValueError('Arm follower currently requires physical hand contacts')

    def ids(self, kind, values):
        result = np.array([mujoco.mj_name2id(self.model, kind, n) for n in values])
        if np.any(result < 0) or len(np.unique(result)) != len(result):
            raise ValueError(f"Missing or duplicate names: {values}")
        return result

    def reset_cpu(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.reset_qpos
        data.qvel[:] = self.reset_qvel
        data.ctrl[:] = self.reset_ctrl
        data.mocap_pos[:] = self.reset_mocap_pos
        data.mocap_quat[:] = self.reset_mocap_quat
        data.qacc_warmstart[:] = self.reset_qacc_warmstart
        mujoco.mj_forward(self.model, data)
        return data

    def normalize(self, physical):
        return 2 * (np.asarray(physical) - self.low) / (self.high - self.low) - 1

    def key_activation(self, qpos):
        limits = self.model.jnt_range[self.key_ids]
        state = np.clip(np.asarray(qpos)[..., self.key_qadr], limits[:, 0], limits[:, 1])
        return state >= limits[:, 1] * self.manifest["key_activation_fraction"]
