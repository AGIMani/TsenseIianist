#!/usr/bin/env python3
"""Accept the stage-1 MJLab scene against native CPU MuJoCo, not a policy."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def render(model, data, output):
    import mujoco
    from PIL import Image
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (.03, 0, .06)
    camera.distance, camera.azimuth, camera.elevation = 1.05, 155, -50
    with mujoco.Renderer(model, height=720, width=1120) as renderer:
        renderer.update_scene(data, camera=camera)
        from tsensei_v3_2_pianist.preview_keys import ActiveKeyHighlight
        ActiveKeyHighlight(model).color_render_scene(renderer.scene, data)
        Image.fromarray(renderer.render()).save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'training_mjlab/outputs/scene')
    parser.add_argument('--output', type=Path, default=ROOT/'training_mjlab/outputs/acceptance')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--render', action='store_true', help='Save initial and pressing GPU-state PNGs (EGL)')
    args = parser.parse_args()
    import mujoco
    import numpy as np
    import torch
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
    from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
    from tsensei_v3_2_pianist.mjlab_env.acceptance_cases import scenarios

    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'stage': 'single physical scene; not training', 'checks': {}, 'cases': {}}
    started = time.perf_counter()
    try:
        scene = PianoScene(args.scene, device=args.device)
        bundle, model = scene.bundle, scene.model
        report.update(device=torch.cuda.get_device_name(scene.device),
                      manifest_sha256=sha256(args.scene/'manifest.json'),
                      versions={n: importlib.metadata.version(n) for n in ('mjlab','mujoco','mujoco-warp','warp-lang','torch')},
                      gravity_compensation=bundle.manifest['gravity_compensation'],
                      dimensions={'nq': model.nq, 'nv': model.nv, 'nu': model.nu, 'actions': 51},
                      physics_dt=model.opt.timestep, decimation=scene.decimation)
        report['unsupported_exporter_default_options'] = bundle.unsupported_default_options
        report['load_seconds'] = time.perf_counter() - started
        def check(name, ok, **values):
            report['checks'][name] = {'passed': bool(ok), **values}
        initial = scene.data.qpos[0].cpu().numpy().copy()
        initial_error = float(np.max(np.abs(initial-bundle.reset_qpos)))
        check('reset_qpos', initial_error < 1e-6, max_error=initial_error, tolerance=1e-6)
        for side, hand in bundle.manifest['hands'].items():
            body = model.body(hand['body']).id
            error = float(np.max(np.abs(scene.data.xpos[0, body].cpu().numpy()-hand['position'])))
            ids = bundle.ids(mujoco.mjtObj.mjOBJ_SITE, hand['sites'])
            tips_error = float(np.max(np.abs(scene.data.site_xpos[0, ids].cpu().numpy()-hand['fingertips'])))
            rotation_error = float(np.max(np.abs(scene.data.xmat[0, body].cpu().numpy().reshape(-1)-hand['rotation'])))
            check(side+'_initial_pose', max(error, tips_error, rotation_error) < 1e-6,
                  root_error_m=error, fingertip_error_m=tips_error, rotation_error=rotation_error)
        check('reset_ctrl', np.allclose(scene.data.ctrl[0].cpu().numpy(), bundle.reset_ctrl, atol=1e-6))
        roots = [model.body(bundle.manifest['hands'][s]['body']).id for s in ('right', 'left')]
        for axis in ('Yaw', 'Pitch'):
            for angle in (-.1, .1):
                scene.reset()
                # Zero both rotation coordinates before checking a single local axis.
                with scene.scope():
                    for side in ('R', 'L'):
                        for suffix in ('Yaw', 'Pitch'):
                            act = model.actuator(f'{side}_Palm_Base_{suffix}').id
                            adr = model.jnt_qposadr[model.actuator_trnid[act, 0]]
                            scene.data.qpos[0, int(adr)] = 0.
                    scene.sim.forward()
                    baseline = scene.data.xmat[0, roots].cpu().numpy().reshape(2, 3, 3).copy()
                    for side in ('R', 'L'):
                        act = model.actuator(f'{side}_Palm_Base_{axis}').id
                        adr = model.jnt_qposadr[model.actuator_trnid[act, 0]]
                        scene.data.qpos[0, int(adr)] = angle
                    scene.sim.forward()
                current = scene.data.xmat[0, roots].cpu().numpy().reshape(2, 3, 3)
                delta = current @ baseline.transpose(0, 2, 1)
                expected = delta[1] if axis == 'Yaw' else delta[1].T
                check(f'{axis}_world_direction_{angle}', np.allclose(delta[0], expected, atol=1e-6)
                      and not np.allclose(delta[0], np.eye(3), atol=1e-3))
        scene.reset()
        if args.render:
            render(model, scene.cpu_snapshot(), args.output/'initial.png')
        traces = {}
        frozen = np.load(args.scene/'cpu_reference.npz', allow_pickle=False)
        for name, actions in scenarios(bundle):
            scene.reset()
            cpu = bundle.reset_cpu()
            cq, gq, active_cpu, active_gpu, gpu_contacts = [], [], set(), set(), set()
            max_error, max_slide, max_hinge = 0., 0., 0.
            max_contacts, max_constraints = 0, 0
            press_image = None
            for i, physical in enumerate(actions):
                cpu.ctrl[bundle.action_ids] = physical[:-1]
                for _ in range(scene.decimation):
                    mujoco.mj_step(model, cpu)
                mujoco.mj_forward(model, cpu)
                state = scene.step(bundle.normalize(physical)[None])
                q = state['qpos'][0].cpu().numpy().copy()
                v = state['qvel'][0].cpu().numpy()
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise AssertionError(f'Nonfinite state in {name} step {i}')
                delta = np.abs(q-cpu.qpos)
                slide = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE]
                hinge = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
                max_error = max(max_error, float(delta.max()))
                max_slide = max(max_slide, float(delta[slide].max()))
                max_hinge = max(max_hinge, float(delta[hinge].max()))
                active_cpu.update(np.flatnonzero(bundle.key_activation(cpu.qpos)).tolist())
                active_gpu.update(np.flatnonzero(state['key_activation'][0].cpu().numpy()).tolist())
                cq.append(cpu.qpos.copy()); gq.append(q)
                ncon = int(scene.sim.wp_data.nacon.numpy()[0])
                nefc = int(scene.sim.wp_data.nefc.numpy()[0])
                max_contacts, max_constraints = max(max_contacts, ncon), max(max_constraints, nefc)
                if ncon >= scene.sim.wp_data.naconmax or nefc >= scene.sim.wp_data.njmax:
                    raise AssertionError(f'Contact/constraint capacity reached in {name}')
                if name == 'press_release':
                    # Read contacts produced by Warp, not CPU-renderer reconstructed contacts.
                    geom = scene.sim.wp_data.contact.geom.numpy()[:ncon]
                    for g1, g2 in geom:
                        a, b = model.geom(int(g1)).name, model.geom(int(g2)).name
                        if (('key_geom_' in a and b.startswith(('right_/', 'left_/')))
                                or ('key_geom_' in b and a.startswith(('right_/', 'left_/')))):
                            gpu_contacts.add((a, b))
                    if i == 11:
                        press_image = scene.cpu_snapshot()
            cq, gq = np.array(cq), np.array(gq)
            np.testing.assert_array_equal(actions, frozen[name+'_action'])
            traces[name+'_cpu_qpos'] = cq; traces[name+'_gpu_qpos'] = gq; traces[name+'_physical_action'] = actions
            # Free motion has a tighter numerical criterion than contact trajectories.
            contact_case = name == 'press_release'
            ok = max_slide < (.002 if contact_case else .0005) and max_hinge < (.05 if contact_case else .01)
            check(name+'_trajectory', ok, max_slide_error_m=max_slide, max_hinge_error_rad=max_hinge,
                  slide_tolerance_m=.002 if contact_case else .0005,
                  hinge_tolerance_rad=.05 if contact_case else .01)
            original_delta = np.abs(gq-frozen[name+'_qpos'])
            original_slide = float(original_delta[:, slide].max())
            original_hinge = float(original_delta[:, hinge].max())
            check(name+'_original_cpu', original_slide < (.002 if contact_case else .0005)
                  and original_hinge < (.05 if contact_case else .01),
                  max_slide_error_m=original_slide, max_hinge_error_rad=original_hinge,
                  exporter_mujoco_version=bundle.manifest['mujoco_export_version'])
            report['cases'][name] = {'control_steps': len(actions), 'cpu_active_keys': sorted(active_cpu),
                                     'gpu_active_keys': sorted(active_gpu), 'max_qpos_error': max_error,
                                     'max_contacts_at_control_boundary': max_contacts,
                                     'max_constraints_at_control_boundary': max_constraints}
            if name.startswith('wrist_') or name.endswith('_finger'):
                changed = np.flatnonzero(np.abs(actions[-1, :-1]-bundle.neutral_action[:-1]) > 1e-8)
                for index in changed:
                    act = bundle.action_ids[index]
                    adr = model.jnt_qposadr[model.actuator_trnid[act, 0]]
                    direction = np.sign(actions[-1, index]-bundle.neutral_action[index])
                    displacement = float((gq[-1, adr]-initial[adr])*direction)
                    check(name+'_'+bundle.manifest['action_names'][index]+'_responds', displacement > .0005,
                          signed_displacement=displacement)
            if contact_case:
                check('hand_key_contacts', bool(gpu_contacts), pairs=sorted(gpu_contacts))
                check('keys_activated', bool(active_gpu) and active_cpu == active_gpu,
                      cpu=sorted(active_cpu), gpu=sorted(active_gpu))
                check('keys_released', not state['key_activation'].any().item())
                if args.render and press_image is not None:
                    render(model, press_image, args.output/'press.png')
            if name == 'sustain':
                check('sustain_action', abs(scene.sustain.item()-1) < 1e-6)
        scene.reset()
        check('reset_after_motion', np.allclose(scene.data.qpos[0].cpu().numpy(), initial, atol=1e-7)
              and np.allclose(scene.data.qvel[0].cpu().numpy(), bundle.reset_qvel, atol=1e-7)
              and np.allclose(scene.data.ctrl[0].cpu().numpy(), bundle.reset_ctrl, atol=1e-6)
              and scene.sustain.item() == 0)
        report['status'] = 'passed' if all(v['passed'] for v in report['checks'].values()) else 'failed'
        np.savez_compressed(args.output/'trajectories.npz', **traces)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f"Acceptance: {report['status']}; {args.output.resolve()/'report.json'}", flush=True)
    if report['status'] != 'passed':
        failed = [k for k,v in report['checks'].items() if not v['passed']]
        raise SystemExit('Failed checks: '+', '.join(failed))


if __name__ == '__main__':
    main()
