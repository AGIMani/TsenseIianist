#!/usr/bin/env python3
"""Hold colliding fingers/hands/keys; measure motion and verify physical forces.

No policy, viewer, interpolation, or task reset is involved. CPU samples every
physics step. GPU uses the control graph, then samples its physical substeps in
the final two seconds so control-rate aliasing cannot hide rapid vibration.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ('same_hand', 'cross_hand', 'hand_key')


def targets(model, reset_ctrl):
    controls = np.tile(reset_ctrl, (3, 1))
    fingers = [a for a in range(model.nu) if not model.actuator(a).name.startswith(('R_Palm_Base_', 'L_Palm_Base_'))]
    for aid in fingers:
        name = model.actuator(aid).name
        if '_SW' in name:
            controls[0, aid] = model.actuator_ctrlrange[aid, int('Index' in name or 'Ring' in name)]
        elif 'Thumb' not in name:
            controls[0, aid] = .6
        if 'MCP' in name or 'PIP' in name:
            controls[2, aid] = .85
    controls[1, model.actuator('R_Palm_Base_Tx').id] = -.14
    controls[1, model.actuator('L_Palm_Base_Tx').id] = .14
    # At h=12 cm, move over the keybed before pressing; TY=0 can miss the keys.
    for side in ('R', 'L'):
        controls[2, model.actuator(side+'_Palm_Base_Ty').id] = .04
    joints = model.actuator_trnid[fingers, 0]
    return controls, model.jnt_qposadr[joints], model.jnt_dofadr[joints]


def category(model, geoms):
    names = [model.geom(int(g)).name or '' for g in geoms]
    sides = [next((s for s in ('right_/', 'left_/') if n.startswith(s)), None) for n in names]
    if sides[0] and sides[0] == sides[1]:
        return 'same_hand'
    if all(sides):
        return 'cross_hand'
    if any(sides) and any('key_geom_' in n for n in names):
        return 'hand_key'
    return None


def metrics(q, v, dt):
    return dict(peak_to_peak_deg=float(np.rad2deg(np.ptp(q, axis=0)).max()),
                velocity_rms_rad_s=float(np.sqrt(np.mean(v*v))),
                acceleration_rms_rad_s2=float(np.sqrt(np.mean((np.diff(v, axis=0)/dt)**2))))


def cpu(args):
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle
    bundle = SceneBundle(args.scene, disable_wrist_tz=args.disable_wrist_tz)
    m = bundle.model
    controls, qa, va = targets(m, bundle.reset_ctrl)
    rows, traces = [], {}
    for name, ctrl in zip(CASES, controls):
        d = bundle.reset_cpu()
        d.ctrl[:] = ctrl
        qs, vs, forces = [], [], {key: 0. for key in CASES}
        for i in range(round(args.seconds/m.opt.timestep)):
            mujoco.mj_step(m, d)
            if (i+1)*m.opt.timestep < args.seconds-2:
                continue
            qs.append(d.qpos[qa].copy()); vs.append(d.qvel[va].copy())
            for j, contact in enumerate(d.contact):
                cat = category(m, contact.geom)
                if cat and contact.efc_address >= 0:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, j, force)
                    forces[cat] = max(forces[cat], abs(float(force[0])))
        qs, vs = np.asarray(qs), np.asarray(vs)
        rows.append(dict(case=name, **metrics(qs, vs, m.opt.timestep),
                         peak_normal_force_N=forces))
        traces[name+'_qpos'] = qs
    return rows, traces


def gpu(args):
    import torch
    import warp as wp
    from tsensei_v3_2_pianist.mjlab_env.task import PianoTaskEnv
    if args.task is None:
        raise ValueError('--task is required for GPU reward checks')
    env = PianoTaskEnv(args.scene, args.task, num_envs=3, physics_graph=True,
                       disable_wrist_tz=args.disable_wrist_tz, action_profile='mixed_v1',
                       self_collision_weight=.2)
    s, m = env.scene, env.scene.model
    assert s.self_query_model is None  # All contact categories use real solver constraints.
    controls, qa, va = targets(m, s.bundle.reset_ctrl)
    physical = np.tile(s.bundle.neutral_action, (3, 1))
    physical[:, s.bundle.active_action_indices] = controls[:, s.bundle.action_ids]
    action = s.policy_action(physical)
    qs, vs = [], []
    forces = [{key: 0. for key in CASES} for _ in CASES]

    def sample(scene):
        qs.append(scene.data.qpos[:, qa].clone())
        vs.append(scene.data.qvel[:, va].clone())
        # Inspect actual GPU constraint forces, not recomputed CPU contact forces.
        raw = scene.sim.wp_data
        n = int(wp.to_torch(raw.nacon)[0].item())
        geoms = wp.to_torch(raw.contact.geom)[:n].cpu().numpy()
        worlds = wp.to_torch(raw.contact.worldid)[:n].cpu().numpy()
        addresses = wp.to_torch(raw.contact.efc_address)[:n].cpu().numpy()
        values = wp.to_torch(raw.efc.force).cpu().numpy()
        for geoms, world, adr in zip(geoms, worlds, addresses):
            cat = category(m, geoms)
            valid = adr[adr >= 0]
            if cat and len(valid):
                # With the default pyramidal cone, normal force is the sum of edges.
                normal = float(np.abs(values[int(world), valid]).sum())
                forces[int(world)][cat] = max(forces[int(world)][cat], normal)

    control_dt = s.bundle.manifest['control_dt']
    for i in range(round(args.seconds/control_dt)):
        callback = sample if i*control_dt >= args.seconds-2-1e-9 else None
        s.step(action, substep_callback=callback)
    qs, vs = torch.stack(qs).cpu().numpy(), torch.stack(vs).cpu().numpy()
    assert np.isfinite(qs).all() and np.isfinite(vs).all()
    result = env.step(action)
    terms = result.info['reward_terms']
    torch.testing.assert_close(terms['self_collision_reward'], -.2*result.info['self_collision_pairs'].float())
    torch.testing.assert_close(terms['forearm_reward'], .5*(~env.snapshot()['palm_collision']).float())
    assert s.control_graph is not None
    rows = [dict(case=name, **metrics(qs[:, i], vs[:, i], m.opt.timestep),
                 peak_normal_force_N=forces[i],
                 self_collision_reward=float(terms['self_collision_reward'][i]),
                 forearm_reward=float(terms['forearm_reward'][i])) for i, name in enumerate(CASES)]
    return rows, {name+'_qpos': qs[:, i] for i, name in enumerate(CASES)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--task', type=Path)
    p.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    p.add_argument('--disable-wrist-tz', action='store_true')
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not np.isfinite(a.seconds) or a.seconds < 3:
        p.error('--seconds must be finite and >= 3')
    rows, traces = (cpu if a.backend == 'cpu' else gpu)(a)
    report = dict(backend=a.backend, mujoco=mujoco.__version__, scene=str(a.scene.resolve()),
                  seconds=a.seconds, sample_window_seconds=2, disable_wrist_tz=a.disable_wrist_tz, rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(a.output.with_suffix('.npz'), **traces)
    print(json.dumps(report, indent=2))
    for row in rows:
        assert row['peak_normal_force_N'][row['case']] > .01, row
        assert row['velocity_rms_rad_s'] < .02, row


if __name__ == '__main__':
    main()
