"""Preview-only M6 differential IK. Arm visuals never enter hand physics/actions.

TJ-ARM-02/03: shoulder pose and tool adapter are provisional, configurable.
Source URDF joints/axes/limits are preserved; meshes are bundled locally.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .interactive_ik import damped_least_squares_step

ASSETS = Path(__file__).parent / 'assets' / 'tj_m6'
if not ASSETS.is_dir():
    # Optional arm assets live with the preview, outside A3.2 training assets.
    ASSETS = Path(__file__).resolve().parents[1] / 'preview/assets/tj_m6'
DEFAULT_CONFIG = ASSETS / 'preview_mount.json'


def quat(rpy):
    return np.roll(Rotation.from_euler('xyz', rpy).as_quat(), 1)


def origin(element):
    node = element.find('origin')
    return (np.fromstring(node.get('xyz', '0 0 0'), sep=' '),
            np.fromstring(node.get('rpy', '0 0 0'), sep=' ')) if node is not None else (np.zeros(3), np.zeros(3))


class ArmPreview:
    """Append inert mocap visuals before compilation; bind/follow after reset."""
    def __init__(self, root, config_path=DEFAULT_CONFIG, *, tool_overrides=None):
        self.config_path = Path(config_path)
        self.config = c = json.loads(self.config_path.read_text())
        if tool_overrides is not None:
            for side, transform in tool_overrides.items():
                c['link7_to_hand'][side] = {'xyz_m': transform[:3, 3].tolist(),
                    'rpy_rad': Rotation.from_matrix(transform[:3, :3]).as_euler('xyz').tolist()}
        center = np.asarray(c['shoulder_center_xyz_m'], dtype=float)
        spacing = float(c['shoulder_spacing_m'])
        if center.shape != (3,) or not np.isfinite(center).all() or not 0 < spacing < 2:
            raise ValueError('Invalid shoulder center/spacing')
        base_xyz = c.get('base_xyz_m', {})
        if not isinstance(base_xyz, dict) or set(base_xyz) - {'left', 'right'}:
            raise ValueError('base_xyz_m must map left/right to piano-frame positions')
        for side, value in base_xyz.items():
            if np.asarray(value).shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid base position for {side}')
        for key in ('orientation_weight_m','damping'):
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if type(c['iterations_per_frame']) is not int or not 1 <= c['iterations_per_frame'] <= 100:
            raise ValueError('iterations_per_frame must be an integer from 1 to 100')
        for side in ('right','left'):
            for value, size in ((c['base_rpy_rad'][side],3),(c['seed_q_rad'][side],7),
                                (c['link7_to_hand'][side]['xyz_m'],3),
                                (c['link7_to_hand'][side]['rpy_rad'],3)):
                if np.asarray(value).shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid mounting/seed vector for {side}')
        from dm_control import mjcf
        fk = mjcf.RootElement(model='m6_ik')
        fk.compiler.angle = 'radian'
        self.visuals = []
        self.sides = {}
        self.handles = {}
        for side, sign in (('right', 1), ('left', -1)):
            source = ET.parse(ASSETS / side / 'arm.urdf').getroot()
            links = {n.get('name'): n for n in source.findall('link')}
            children = {}
            child_names = set()
            for j in source.findall('joint'):
                children.setdefault(j.find('parent').get('link'), []).append(j)
                child_names.add(j.find('child').get('link'))
            base = (set(links)-child_names).pop()
            position = np.asarray(base_xyz.get(side, center + (0, sign*spacing/2, 0)), dtype=float)
            rotation = quat(c['base_rpy_rad'][side])
            joint_names, velocities = [], []

            def add_link(name, parent, joint=None):
                xyz, rpy = origin(joint) if joint is not None else (position, None)
                body = parent.add('body', name=f'{side}_{name}', pos=xyz,
                                  quat=quat(rpy) if rpy is not None else rotation)
                # FK-only model: no simulation or torques use this inertial proxy.
                body.add('inertial', pos=[0,0,0], mass=1, diaginertia=[.01]*3)
                if joint is not None:
                    limit = joint.find('limit')
                    jname = f'{side}_{joint.get("name")}'
                    body.add('joint', name=jname, type='hinge',
                             axis=np.fromstring(joint.find('axis').get('xyz'), sep=' '),
                             range=[float(limit.get('lower')), float(limit.get('upper'))], limited=True)
                    joint_names.append(jname)
                    velocities.append(float(limit.get('velocity')))
                visual = root.worldbody.add('body', name=f'tj_{side}_{name}', mocap=True)
                for k, item in enumerate(links[name].findall('visual')):
                    mesh = item.find('geometry/mesh')
                    asset = root.asset.add('mesh', name=f'tj_{side}_{name}_{k}',
                        file=str(ASSETS / side / 'meshes' / Path(mesh.get('filename')).name))
                    p, r = origin(item)
                    visual.add('geom', type='mesh', mesh=asset, pos=p, quat=quat(r),
                               contype=0, conaffinity=0, group=1, rgba=c['arm_rgba'])
                for k, item in enumerate(links[name].findall('collision')):
                    mesh = item.find('geometry/mesh')
                    asset = root.asset.add('mesh', name=f'tj_collision_mesh_{side}_{name}_{k}',
                        file=str(ASSETS / side / 'meshes' / Path(mesh.get('filename')).name))
                    p, r = origin(item)
                    visual.add('geom', name=f'tj_collision_{side}_{name}_{k}', type='mesh',
                        mesh=asset, pos=p, quat=quat(r), contype=0, conaffinity=0, group=3,
                        rgba=[1,.2,.15,.6] if side == 'right' else [.1,.5,1,.6])
                self.visuals.append((visual, body.full_identifier))
                for child in children.get(name, []):
                    add_link(child.find('child').get('link'), body, child)
                return body

            add_link(base, fk.worldbody)
            suffix = 'R' if side == 'right' else 'L'
            end = fk.find('body', f'{side}_Link7_{suffix}')
            adapter = c['link7_to_hand'][side]
            site = end.add('site', name=f'{side}_wrist', pos=adapter['xyz_m'],
                           quat=quat(adapter['rpy_rad']), size=.005)
            # Visible configurable adapter, beginning at the provisional flange.
            tip_visual = next(v for v, name in self.visuals if name == end.full_identifier)
            if tool_overrides is None and np.linalg.norm(np.asarray(adapter['xyz_m'])-[0,-.095,0]) > 1e-6:
                tip_visual.add('geom', type='cylinder', size=[.028],
                               fromto=[0,-.095,0]+list(adapter['xyz_m']), contype=0, conaffinity=0,
                               group=1, rgba=[.70,.73,.77,1])
            self.sides[side] = dict(joints=joint_names, velocity=np.asarray(velocities), site=site.full_identifier)
        # Mounting support is only a visual aid, not claimed to be manufacturer CAD.
        support = root.worldbody.add('body', name='tj_preview_support', pos=center)
        support.add('geom', type='box', size=[.045, spacing/2, .045], group=1,
                    contype=0, conaffinity=0, rgba=[.40,.43,.47,1])
        self.model = mujoco.MjModel.from_xml_string(fk.to_xml_string(), fk.get_assets())
        self.data = mujoco.MjData(self.model)
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)
        for side, arm in self.sides.items():
            ids = [self.model.joint(n).id for n in arm['joints']]
            arm.update(qadr=self.model.jnt_qposadr[ids], dof=self.model.jnt_dofadr[ids],
                       limits=self.model.jnt_range[ids], sid=self.model.site(arm['site']).id)
        self.targets = {}
        self.initial_q = None
        for side in self.sides:
            handle = root.worldbody.add('body', name=f'tj_{side}_wrist_drag', mocap=True)
            handle.add('geom', type='sphere', size=[.014], contype=0, conaffinity=0,
                       group=2, rgba=[.15,.9,.55,.65])
            self.handles[side] = handle

    def bind(self, physics, hands):
        self.physics = physics
        self.hand_ids = {side: physics.model.ptr.body(hand.root_body.full_identifier).id
                         for side, hand in zip(('right','left'), hands)}
        self.mapping = [(int(physics.model.ptr.body_mocapid[physics.model.ptr.body(v.full_identifier).id]),
                         self.model.body(name).id) for v, name in self.visuals]
        m = physics.model.ptr
        self.handle_ids = {side:int(m.body_mocapid[m.body(body.full_identifier).id])
                           for side,body in self.handles.items()}
        self.wrist_actuators = {side: [i for i in range(m.nu)
                                      if f'{side[0].upper()}_Palm_Base_' in (m.actuator(i).name or '')]
                               for side in self.sides}
        self.handle_last = {}
        self.hand_work = mujoco.MjData(m)
        self._read_targets()
        self.initial_targets = {s: (p.copy(), r.copy()) for s, (p, r) in self.targets.items()}
        rng = np.random.default_rng(17)
        for side, arm in self.sides.items():
            low, high = arm['limits'].T
            def error(q):
                self.data.qpos[arm['qadr']] = q
                mujoco.mj_forward(self.model, self.data)
                return self._error(side)
            seed = np.asarray(self.config['seed_q_rad'][side])
            best = None
            for candidate in [seed] + [rng.uniform(low*.8, high*.8) for _ in range(8)]:
                fit = least_squares(error, np.clip(candidate, low+1e-5, high-1e-5),
                                    bounds=(low, high), max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                    best = fit
                if np.linalg.norm(fit.fun) < 1e-6:
                    break
            if np.linalg.norm(best.fun) > 1e-5:
                raise ValueError(f'M6 {side}: initial wrist pose is unreachable; adjust mount or initial hand pose')
            self.data.qpos[arm['qadr']] = best.x
        self.initial_q = self.data.qpos.copy()
        self.accepted_wrist = {}
        self.request_errors = {}
        for side, aids in self.wrist_actuators.items():
            jids = m.actuator_trnid[aids, 0]
            self.accepted_wrist[side] = physics.data.qpos[m.jnt_qposadr[jids]].copy()
        self._publish()
        from .preview_arm_collision import ArmForearmCollision
        self.collision = ArmForearmCollision(physics, self)

    def set_tool_transforms(self, transforms):
        """Refit arm IK after an explicit connector adjustment, with rollback."""
        self._read_targets()
        current_targets = {s: (p.copy(), r.copy()) for s, (p, r) in self.targets.items()}
        old_q = self.data.qpos.copy()
        old_initial = self.initial_q.copy()
        old_sites = {s: (self.model.site_pos[a['sid']].copy(), self.model.site_quat[a['sid']].copy())
                     for s, a in self.sides.items()}
        try:
            for side, transform in transforms.items():
                sid = self.sides[side]['sid']
                self.model.site_pos[sid] = transform[:3, 3]
                self.model.site_quat[sid] = np.roll(Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)
            solutions = []
            for targets, seed in ((self.initial_targets, old_initial), (current_targets, old_q)):
                self.targets = targets
                self.data.qpos[:] = seed
                for side, arm in self.sides.items():
                    def residual(q):
                        self.data.qpos[arm['qadr']] = q
                        mujoco.mj_forward(self.model, self.data)
                        return self._error(side)
                    fit = least_squares(residual, np.clip(seed[arm['qadr']], arm['limits'][:, 0]+1e-10,
                                                        arm['limits'][:, 1]-1e-10),
                                        bounds=arm['limits'].T, max_nfev=120,
                                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
                    if np.linalg.norm(fit.fun) > 1e-5:
                        raise ValueError(f'{side} 机械臂无法到达该安装偏移')
                    self.data.qpos[arm['qadr']] = fit.x
                solutions.append(self.data.qpos.copy())
            self.initial_q = solutions[0]
            self.data.qpos[:] = solutions[1]
            for side, transform in transforms.items():
                self.config['link7_to_hand'][side] = {'xyz_m': transform[:3, 3].tolist(),
                    'rpy_rad': Rotation.from_matrix(transform[:3, :3]).as_euler('xyz').tolist()}
        except Exception:
            self.data.qpos[:] = old_q
            self.initial_q = old_initial
            for side, (position, rotation) in old_sites.items():
                sid = self.sides[side]['sid']
                self.model.site_pos[sid], self.model.site_quat[sid] = position, rotation
            raise
        finally:
            self.targets = current_targets
            mujoco.mj_forward(self.model, self.data)
        self._publish()

    def _read_targets(self):
        d = self.physics.data.ptr
        self.targets = {s: (d.xpos[i].copy(), d.xmat[i].reshape(3,3).copy()) for s,i in self.hand_ids.items()}

    def _error(self, side):
        sid = self.sides[side]['sid']
        pos, rotation = self.targets[side]
        angular = Rotation.from_matrix(rotation @ self.data.site_xmat[sid].reshape(3,3).T).as_rotvec()
        return np.r_[pos-self.data.site_xpos[sid], float(self.config['orientation_weight_m'])*angular]

    def apply_drag_handles(self):
        """Convert viewer mocap-handle drags to the existing five wrist controls.

        Roll/unreachable requests project onto the hand's permitted joint ranges.
        Arm IK subsequently follows the actual hand, not an unattainable handle.
        """
        m, actual = self.physics.model.ptr, self.physics.data.ptr
        for side, mid in self.handle_ids.items():
            position, quaternion = actual.mocap_pos[mid].copy(), actual.mocap_quat[mid].copy()
            oldpos, oldquat = self.handle_last[side]
            if np.allclose(position, oldpos, atol=1e-7) and abs(np.dot(quaternion,oldquat)) > 1-1e-9:
                continue
            work = self.hand_work
            mujoco.mj_copyData(work,m,actual)
            aids = self.wrist_actuators[side]
            jids = m.actuator_trnid[aids,0]
            adr, dof = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
            bid = self.hand_ids[side]
            target_rotation = Rotation.from_quat(np.roll(quaternion,-1)).as_matrix()
            jp,jr = np.zeros((3,m.nv)),np.zeros((3,m.nv))
            for _ in range(16):
                mujoco.mj_forward(m,work)
                error=np.r_[position-work.xpos[bid], .15*Rotation.from_matrix(
                    target_rotation@work.xmat[bid].reshape(3,3).T).as_rotvec()]
                if np.linalg.norm(error)<1e-6: break
                mujoco.mj_jacBody(m,work,jp,jr,bid)
                jac=np.vstack((jp[:,dof], .15*jr[:,dof]))
                delta=damped_least_squares_step(jac,error,1e-3)
                work.qpos[adr]=np.clip(work.qpos[adr]+delta,m.jnt_range[jids,0],m.jnt_range[jids,1])
            actual.ctrl[aids]=work.qpos[adr]

    def update(self, dt=1/60, reset=False):
        """Accept only wrist motion reachable by the arm within this frame.

        Both targets stay in the original five-dimensional hand joint manifold.
        Unreachable/too-fast requests are shortened, never displayed detached.
        This is a preview kinematic constraint, not a physical weld.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('IK frame dt must be finite and positive')
        self._read_targets()
        m, actual = self.physics.model.ptr, self.physics.data.ptr
        if reset:
            self.data.qpos[:] = self.initial_q
        for side, arm in self.sides.items():
            aids = self.wrist_actuators[side]
            jids = m.actuator_trnid[aids, 0]
            adr, dof = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
            requested = actual.qpos[adr].copy()
            requested_target = self.targets[side]
            previous = requested.copy() if reset else self.accepted_wrist[side].copy()
            start = self.data.qpos[arm['qadr']].copy()
            low = np.maximum(arm['limits'][:, 0], start-arm['velocity']*dt)
            high = np.minimum(arm['limits'][:, 1], start+arm['velocity']*dt)
            mujoco.mj_copyData(self.hand_work, m, actual)
            chosen = None
            def residual(q):
                self.data.qpos[arm['qadr']] = q
                mujoco.mj_forward(self.model, self.data)
                return self._error(side)
            # Bounded backtracking also handles unreachable targets and singularities.
            for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
                candidate = previous + fraction*(requested-previous)
                self.hand_work.qpos[adr] = candidate
                mujoco.mj_forward(m, self.hand_work)
                bid = self.hand_ids[side]
                self.targets[side] = (self.hand_work.xpos[bid].copy(),
                                      self.hand_work.xmat[bid].reshape(3,3).copy())
                q = start.copy()
                for _ in range(self.config['iterations_per_frame']):
                    error = residual(q)
                    if np.linalg.norm(error) < 1e-5:
                        break
                    mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, arm['sid'])
                    jac = np.vstack((self.jacp[:,arm['dof']],
                        self.config['orientation_weight_m']*self.jacr[:,arm['dof']]))
                    delta = damped_least_squares_step(jac, error, self.config['damping'])
                    q = np.clip(q+delta, low, high)
                if np.linalg.norm(residual(q)) < 1e-5:
                    chosen = candidate
                    break
                fit = least_squares(residual, np.clip(q, low+1e-12, high-1e-12),
                                    bounds=(low, high), max_nfev=25,
                                    ftol=1e-8, xtol=1e-8, gtol=1e-9)
                if np.linalg.norm(fit.fun) < 1e-5:
                    chosen = candidate
                    self.data.qpos[arm['qadr']] = fit.x
                    break
            if chosen is None:
                chosen = previous
                self.data.qpos[arm['qadr']] = start
            actual.qpos[adr] = chosen
            if not np.allclose(chosen, requested, atol=1e-9):
                actual.qvel[dof] = 0
            self.accepted_wrist[side] = chosen.copy()
            mujoco.mj_forward(self.model, self.data)
            self.targets[side] = requested_target
            error = self._error(side)
            self.request_errors[side] = dict(position_mm=float(np.linalg.norm(error[:3])*1000),
                orientation_deg=float(np.linalg.norm(error[3:])/self.config['orientation_weight_m']*180/np.pi),
                limited=not np.allclose(chosen, requested, atol=1e-9))
        self.physics.forward()
        self._read_targets()
        self._publish()

    def _publish(self):
        mujoco.mj_forward(self.model, self.data)
        for mocap, body in self.mapping:
            self.physics.data.ptr.mocap_pos[mocap] = self.data.xpos[body]
            self.physics.data.ptr.mocap_quat[mocap] = self.data.xquat[body]
        # Green wrist handles remain attached to the actual hand after each frame.
        for side, mid in self.handle_ids.items():
            pos, rot = self.targets[side]
            q = np.roll(Rotation.from_matrix(rot).as_quat(),1)
            self.physics.data.ptr.mocap_pos[mid] = pos
            self.physics.data.ptr.mocap_quat[mid] = q
            self.handle_last[side] = (pos.copy(),q.copy())
        self.physics.forward()

    def report(self):
        self._read_targets()
        return dict(mode='preview_only_kinematic_follower', config=self.config,
                    participates_in_training=False, arm_contacts_enabled=False,
                    forearm_collision=self.collision.evaluate(),
                    arms={s: dict(q_rad=self.data.qpos[a['qadr']].tolist(),
                                  requested_target_error=self.request_errors.get(s),
                                  position_error_mm=float(np.linalg.norm(self._error(s)[:3])*1000),
                                  orientation_error_deg=float(np.linalg.norm(self._error(s)[3:])/
                                      self.config['orientation_weight_m']*180/np.pi))
                          for s,a in self.sides.items()})

    @staticmethod
    def camera(camera):
        camera.lookat[:] = (.22, 0, .28)
        camera.distance = 1.65
        camera.azimuth = 0
        camera.elevation = -25
