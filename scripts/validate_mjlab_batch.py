#!/usr/bin/env python3
"""Validate 16/64 independent Warp worlds, hard stops and partial resets."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=ROOT/'training_mjlab/outputs/scene')
    p.add_argument('--num-envs', type=int, choices=(1,16,64), default=16)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = args.output or ROOT/f'training_mjlab/outputs/batch_{args.num_envs}'
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'num_envs': args.num_envs, 'checks': {}}
    started = time.perf_counter()
    try:
        import mujoco
        import numpy as np
        import torch
        from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
        from tsensei_v3_2_pianist.mjlab_env.diagnostics import PhysicsAudit
        from tsensei_v3_2_pianist.mjlab_env.bundle import sha256
        from tsensei_v3_2_pianist.mjlab_env.acceptance_cases import press_action
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.device)
        scene = PianoScene(args.scene, num_envs=args.num_envs, device=args.device)
        b = scene.bundle
        report.update(gpu=torch.cuda.get_device_name(scene.device), manifest_sha256=sha256(args.scene/'manifest.json'),
                      load_s=time.perf_counter()-started, physics_dt=scene.model.opt.timestep, decimation=scene.decimation)
        def check(name, passed, **info):
            report['checks'][name] = {'passed': bool(passed), **info}
        check('hard_contact_profile', scene.model.opt.timestep <= .0005
              and np.all(scene.model.jnt_solimp[b.key_ids, 0] >= .999))
        check('wrist_25mm_range', all(np.allclose(
            scene.model.actuator_ctrlrange[scene.model.actuator(side+'_Wrist_TZ').id], [-.025, 0])
            and np.allclose(b.low[b.manifest['action_names'].index(side+'_Wrist_TZ')], -.025)
            for side in ('R', 'L')))
        check('initial_state_all_worlds', np.allclose(scene.data.qpos[:].cpu().numpy(), b.reset_qpos, atol=1e-6))
        n = args.num_envs
        # Four action groups; neighbouring worlds deliberately do different things.
        pattern = np.arange(n)%4 if n > 1 else np.array([3])
        actions = np.tile(b.neutral_action, (n,1))
        for i, group in enumerate(pattern):
            sides = [side for side, groups in [('R',(1,3)),('L',(2,3))] if group in groups]
            actions[i] = press_action(b, sides)
            actions[i,-1] = group/3
        audit = PhysicsAudit(scene)
        # CPU checks one representative per distinct action group.
        representatives = [int(np.flatnonzero(pattern==g)[0]) for g in np.unique(pattern)]
        cpu = {i:b.reset_cpu() for i in representatives}
        trajectory_errors = []
        slide = scene.model.jnt_qposadr[scene.model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE]
        hinge = scene.model.jnt_qposadr[scene.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
        cpu_slide_error = cpu_hinge_error = 0.
        for step in range(30):
            state = scene.step(b.normalize(actions), substep_callback=audit)
            check(f'finite_{step}', torch.isfinite(state['qpos']).all().item() and torch.isfinite(state['qvel']).all().item())
            for i, d in cpu.items():
                d.ctrl[b.action_ids] = actions[i,:-1]
                for _ in range(scene.decimation):mujoco.mj_step(scene.model,d)
                mujoco.mj_forward(scene.model,d)
                delta = np.abs(d.qpos-scene.data.qpos[i].cpu().numpy())
                trajectory_errors.append(float(delta.max()))
                cpu_slide_error = max(cpu_slide_error, float(delta[slide].max()))
                cpu_hinge_error = max(cpu_hinge_error, float(delta[hinge].max()))
        active = state['key_activation'].cpu().numpy()
        check('independent_actions', all((not active[i].any()) if g==0 else active[i].any() for i,g in enumerate(pattern)))
        check('independent_sustain', np.allclose(scene.sustain.cpu().numpy(), pattern/3, atol=1e-6))
        positions = scene.data.qpos[:].cpu().numpy()
        np.savez_compressed(output/'pressure_states.npz', qpos=positions, actions=actions, groups=pattern)
        differences = np.array([np.abs(positions[i]-positions[np.flatnonzero(pattern==g)[0]]) for i,g in enumerate(pattern)])
        # Contact ordering and float32 reductions need physical-unit tolerances;
        # unlike a reset, a contact rollout is not guaranteed bitwise identical.
        check('same_actions_same_state', differences[:,slide].max()<.00005 and differences[:,hinge].max()<.002
              and all(np.array_equal(active[i],active[np.flatnonzero(pattern==g)[0]]) for i,g in enumerate(pattern)),
              max_slide_m=float(differences[:,slide].max()), max_hinge_rad=float(differences[:,hinge].max()),
              slide_tolerance_m=.00005, hinge_tolerance_rad=.002)
        check('cpu_trajectory', cpu_slide_error<.002 and cpu_hinge_error<.05,
              max_slide_error_m=cpu_slide_error, max_hinge_error_rad=cpu_hinge_error,
              slide_tolerance_m=.002, hinge_tolerance_rad=.05)
        report['cpu_max_qpos_difference_mixed_units'] = max(trajectory_errors)
        # After sustained pressure, key 33/54 is at its upper stop while the middle MCP target is unmet.
        for side, key, groups in [('R',54,(1,3)),('L',33,(2,3))]:
            selected = np.flatnonzero(np.isin(pattern,groups))
            jid = b.key_ids[key]; adr = scene.model.jnt_qposadr[jid]
            key_error = np.abs(scene.data.qpos[selected,int(adr)].cpu().numpy()-scene.model.jnt_range[jid,1])
            actuator = scene.model.actuator(side+'_Mid_MCP').id
            finger_adr = int(scene.model.jnt_qposadr[scene.model.actuator_trnid[actuator,0]])
            gap = actions[selected,b.manifest['action_names'].index(side+'_Mid_MCP')]-scene.data.qpos[selected,finger_adr].cpu().numpy()
            check(side+'_bottom_stop', bool(len(selected)) and np.all(key_error < .001), max_angle_error=float(key_error.max()))
            check(side+'_blocked_target', np.all(gap > .05), minimum_target_gap_rad=float(gap.min()))
        report['physics_audit'] = audit.report()
        a = report['physics_audit']
        check('no_overflow_any_substep', not any(a['overflow_bits']) and a['max_total_contacts'] < a['contact_capacity_total'] and max(a['max_constraints_per_world']) < a['constraint_capacity_per_world'])
        check('contact_penetration', max(a['max_hand_key_penetration_m']) < .0005, tolerance_m=.0005)
        check('key_stop_overtravel', max(a['max_key_front_overtravel_m']) < .0001, tolerance_m=.0001)
        # Save every non-reset world's live state, including time and warm-start state.
        ids = torch.arange(0,n,2,device=scene.device)
        keep = torch.arange(1,n,2,device=scene.device)
        fields = ('qpos','qvel','ctrl','time','qacc_warmstart')
        before = {f:getattr(scene.data,f)[:].clone() for f in fields}
        sustain, counters = scene.sustain.clone(), scene.control_steps.clone()
        scene.reset(ids)
        check('partial_reset_selected', np.allclose(scene.data.qpos[ids].cpu().numpy(),b.reset_qpos,atol=1e-6)
              and np.allclose(scene.data.qvel[ids].cpu().numpy(),b.reset_qvel,atol=1e-7)
              and np.allclose(scene.data.ctrl[ids].cpu().numpy(),b.reset_ctrl,atol=1e-6)
              and (scene.control_steps[ids]==0).all().item() and (scene.sustain[ids]==0).all().item())
        check('partial_reset_unselected_unchanged', all(torch.equal(getattr(scene.data,f)[keep],v[keep]) for f,v in before.items())
              and torch.equal(scene.sustain[keep],sustain[keep]) and torch.equal(scene.control_steps[keep],counters[keep]))
        # Compare the next step of non-reset worlds to continuously stepped CPU refs.
        for i,d in cpu.items():
            if i%2==0:cpu[i]=b.reset_cpu()
        scene.step(b.normalize(actions))
        next_errors=[]
        for i,d in cpu.items():
            d.ctrl[b.action_ids]=actions[i,:-1]
            for _ in range(scene.decimation):mujoco.mj_step(scene.model,d)
            next_errors.append(float(np.max(np.abs(d.qpos-scene.data.qpos[i].cpu().numpy()))))
        check('next_step_after_partial_reset', max(next_errors)<.05, max_qpos_error=max(next_errors))
        for _ in range(20):scene.step(b.normalize(np.tile(b.neutral_action,(n,1))))
        check('release_under_physics', not scene.state()['key_activation'].any().item())
        scene.reset(torch.arange(n-1,-1,-1,device=scene.device))
        check('repeated_reset_reverse_ids', np.allclose(scene.data.qpos[:].cpu().numpy(), b.reset_qpos, atol=1e-6))
        report['active_keys_before_reset']=[np.flatnonzero(row).tolist() for row in active]
        report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated(scene.device)
        report['status']='passed' if all(x['passed'] for x in report['checks'].values()) else 'failed'
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        report['elapsed_s']=time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"Batch acceptance: {report['status']}; {output/'report.json'}",flush=True)
    if report['status']!='passed':raise SystemExit('Failed checks: '+','.join(k for k,v in report['checks'].items() if not v['passed']))


if __name__=='__main__':main()
