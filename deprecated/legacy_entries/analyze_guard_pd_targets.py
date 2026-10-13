#!/usr/bin/env python3
"""CPU-only frozen-trajectory FK diagnosis: PD target geometry is not actual pose.

Rebuilds the parent physical scene/score with strict hashes. Does not step physics,
execute a policy, update a learner, or claim counterfactual rollout performance.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True, help='Baseline variant directory')
    p.add_argument('--guard-config', type=Path, required=True)
    p.add_argument('--reward-config', type=Path, help='Optional reward-only observation sensitivity')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    from tsensei_v3_2_pianist.mjlab_env.source_runtime import make_recipe, build
    from tsensei_v3_2_pianist.mjlab_env.without_arms import without_arms
    from tsensei_v3_2_pianist.mjlab_env.bundle import SceneBundle, sha256
    from tsensei_v3_2_pianist.mjlab_env.finger_clearance import FingerClearanceGuard
    torch.set_num_threads(1)
    config = json.loads((a.run/'config.json').read_text())
    saved = config['source_runtime']; prep = saved.get('preparation') or {}
    recipe = make_recipe(**saved['pose'], **saved['model'], **saved['environment'],
        preparation_time=prep.get('duration', 0), preparation_lift=prep.get('lift', .02),
        preparation_mode=prep.get('mode', 'fixed_v1'))
    recipe['external_score'] = deepcopy(saved.get('external_score'))
    recipe['environment']['midi'] = saved['environment']['midi']
    scene, task, _ = build(recipe)
    if config.get('cli', {}).get('no_arms'):
        scene, task = without_arms(scene, task, scene.parent/'hand_only_bundle')
    if (sha256(scene/'manifest.json') != config['scene_sha256']
            or sha256(task/'task.json') != config['task_sha256']):
        raise ValueError('Rebuilt physical scene or score differs from baseline parent')
    bundle = SceneBundle(scene, disable_wrist_tz=config['disable_wrist_tz'],
        disable_wrist_yaw=config['disable_wrist_yaw'], preparation=config['preparation'],
        self_collision_detection_only=config.get('self_collision_detection_only', False),
        hand_collision_detection_only=config.get('hand_collision_detection_only', False))
    options = json.loads(a.guard_config.read_text())
    options['active_command_mode'] = 'strict_v1'
    guard = FingerClearanceGuard(bundle, device='cpu', options=options)
    detail = a.audit/'best'/'detail-0'
    rollout = np.load(detail/'rollout.npz'); physics = np.load(detail/'physics_200hz.npz')
    diag = np.load(detail/'control_diagnostics.npz'); score = np.load(a.audit/'score.npz')
    initial = np.load(a.audit/'initial_geometry.npz')
    # Baseline applied controls already include its saved ordinary wrist limits.
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    request = t(diag['applied_physical_action']); assigned = t(score['fingers'])
    pre_qpos = t(np.concatenate((initial['qpos'][None], rollout['qpos'][:-1])))
    post_qpos = t(rollout['qpos'])
    post_mocap = t(physics['mocap_pos'][9::10])
    pre_mocap = t(np.concatenate((initial['mocap_pos'][None], physics['mocap_pos'][9:-10:10])))
    actual = guard.physical_qpos(post_qpos); lift = guard.lift(post_mocap)
    with torch.no_grad():
        projected, metrics = guard.filter(request, actual_qpos=pre_qpos,
            mocap_pos=pre_mocap, assigned_keys=assigned)
        req = guard.evaluate(request, torch.zeros_like(lift), assigned)
        act = guard.evaluate(actual, lift, assigned)
        active = assigned.bool().any(-1)
        successful = (assigned.bool() & torch.as_tensor(rollout['actual'])[:, None, :]).any(-1)
        def pad_plane(physical, support):
            endpoints = guard.fk(physical, support)[0][:, :, -1]
            key = assigned.argmax(-1)
            normal = guard.pressed_normal[key]
            required = (normal*guard.pressed_center[key]).sum(-1)+guard.key_half_z[key]
            # Unconditional assigned-key full-press plane, independent of XY fits.
            return (endpoints*normal[:, :, None, :]).sum(-1).amin(-1)-guard.radii[:, -1]-required
        req_pad = pad_plane(request, torch.zeros_like(lift))
        act_pad = pad_plane(actual, lift)
        delta = (projected[:, guard.finger_indices]-request[:, guard.finger_indices]).abs().amax(-1)
        tracking_error = (request[:, guard.finger_indices]-actual[:, guard.finger_indices]).abs().amax(-1)
    arrays = {**{f'metric_{k}': v.numpy() for k, v in metrics.items()},
        'active': active.numpy(), 'successful_assigned': successful.numpy(),
        'requested_capsule_clearance_m': req['capsule_clearance_m'].numpy(),
        'actual_capsule_clearance_m': act['capsule_clearance_m'].numpy(),
        'requested_terminal_allowed': req['finger_terminal_allowed'].numpy(),
        'actual_terminal_allowed': act['finger_terminal_allowed'].numpy(),
        'requested_pad_fullpress_plane_clearance_m': req_pad.numpy(),
        'actual_pad_fullpress_plane_clearance_m': act_pad.numpy(),
        'projection_max_joint_delta_rad': delta.numpy(),
        'tracking_max_joint_error_rad': tracking_error.numpy(),
        'projected_physical_action': projected.numpy(),
        'baseline_applied_physical_action': request.numpy()}
    def describe(values):
        values = np.asarray(values)
        return dict(count=int(values.size), min=float(values.min()), median=float(np.median(values)),
            p95=float(np.quantile(values, .95)), max=float(values.max())) if values.size else dict(count=0)
    def section(mask):
        mask = np.asarray(mask); valid = int(mask.sum())
        selected = lambda key: arrays[key][mask]
        req_caps = arrays['requested_capsule_clearance_m']
        act_caps = arrays['actual_capsule_clearance_m']
        return dict(finger_frames=valid,
            projected=int((arrays['metric_finger_projection_mask'] & mask).sum()),
            fallback=int((arrays['metric_finger_fallback_mask'] & mask).sum()),
            request_terminal_allowed=int((arrays['requested_terminal_allowed'] & mask).sum()),
            actual_terminal_allowed=int((arrays['actual_terminal_allowed'] & mask).sum()),
            request_nonterminal_violation=int(((req_caps[:, :, :-1].min(-1)<-1e-6)&mask).sum()),
            request_terminal_violation=int(((req_caps[:, :, -1]<-1e-6)&mask).sum()),
            actual_nonterminal_violation=int(((act_caps[:, :, :-1].min(-1)<-1e-6)&mask).sum()),
            actual_terminal_violation=int(((act_caps[:, :, -1]<-1e-6)&mask).sum()),
            request_pad_more_than_1mm_below_fullpress=int(((arrays['requested_pad_fullpress_plane_clearance_m']<-.001)&mask).sum()),
            request_pad_plane_m=describe(selected('requested_pad_fullpress_plane_clearance_m')),
            actual_pad_plane_m=describe(selected('actual_pad_fullpress_plane_clearance_m')),
            projection_max_joint_delta_rad=describe(selected('projection_max_joint_delta_rad')),
            tracking_max_joint_error_rad=describe(selected('tracking_max_joint_error_rad')))
    active_np = arrays['active']; good = arrays['successful_assigned']
    result = dict(method='Fixed recorded baseline trajectory; CPU FK only; not a closed-loop counterfactual',
        baseline=str(a.audit.resolve()), config=guard.options,
        geometry_source_sha256=sha256(ROOT/'tsensei_v3_2_pianist/mjlab_env/finger_clearance.py'),
        physical_scene_sha256=config['scene_sha256'], task_sha256=config['task_sha256'],
        active=section(active_np), successful_assigned=section(good), idle=section(~active_np),
        fingers={group['name']: section(good & (np.arange(10)[None, :]==i)) for i,group in enumerate(guard.groups)},
        notes=['Strict guard compares commanded FK at origin support height; actual is post-control endpoint.',
            'PD commands can legitimately lie beyond a contact surface to produce pressing force.',
            'Full-press-plane comparison ignores target XY eligibility deliberately to isolate servo overdrive.',
            'A command below the plane is not proof of a physical collision or of sufficient force.',
            'Inactive assigned-key plane numbers are not meaningful (no assigned key); use capsule clearances there.'])
    # Change one input field at a time on identical recorded observations. This
    # measures immediate actor sensitivity; it cannot reconstruct feedback over
    # a new physical trajectory and is not an F1 estimate.
    from tsensei_v3_2_pianist.rl_torch.sac import Actor
    state = torch.load(a.run/'best.pt', map_location='cpu', weights_only=True)['agent']
    actor = Actor(state['obs_size'], state['action_size'], state['config']['actor_hidden'], state['action_profile'])
    actor.load_state_dict(state['actor']); actor.eval()
    observation = t(rollout['observation'])
    manifest = json.loads((task/'task.json').read_text())
    fields = manifest['contract']['observation_fields_before_stack']
    base_size = sum(field['size'] for field in fields)
    if base_size+1 != observation.shape[1]:
        raise ValueError('Sensitivity probe currently requires one frame plus preparation phase')
    offsets = {}; start = 0
    for field in fields:
        offsets[field['path']] = slice(start, start+field['size']); start += field['size']
    changed_action = observation.clone()
    changed_action[1:, offsets['action']] = projected[:-1]
    with torch.no_grad():
        policy = actor.deterministic(observation)
        changed = actor.deterministic(changed_action)
    result['actor_sensitivity'] = dict(
        method='One-step CPU actor on fixed baseline observations, perturb prior field only; no dynamics feedback',
        cpu_vs_saved_gpu_policy_max_abs=float((policy-t(rollout['actions'])).abs().max()),
        projected_previous_action_policy_max_abs_per_frame=describe((changed-policy).abs().amax(-1).numpy()),
        projected_previous_action_policy_rms=float((changed-policy).square().mean().sqrt()))
    arrays['policy_action_previous_action_only_delta'] = (changed-policy).numpy()
    if a.reward_config:
        from tsensei_v3_2_pianist.mjlab_env.rewards import geometry_clearance_penalty
        reward_config = json.loads(a.reward_config.read_text())
        terms = reward_config['terms']; gesture = terms['humanoid_finger_gesture_reward']
        report = json.loads((detail/'report.json').read_text())
        old_gesture = t(rollout['reward_terms'][:, report['reward_term_names'].index('humanoid_finger_gesture_reward')])
        reference = actual.clone(); reference[:, guard.finger_indices] = guard.origin_angles
        with torch.no_grad():
            feasible = guard.evaluate(reference, lift, assigned)['finger_clearance_deficit_m'] <= 1e-6
            posture_mse = (actual[:, guard.finger_indices]-guard.origin_angles).square().mean(-1)
            deficit = torch.maximum(act['finger_clearance_deficit_m'], req['finger_clearance_deficit_m'])
            new_gesture = geometry_clearance_penalty(deficit, posture_mse, ~active, feasible,
                t(score['goal']), gesture['params'], terms['key_press_reward']['weight'])*gesture['weight']
            changed_reward = observation.clone()
            changed_reward[1:, offsets['reward']] += (new_gesture-old_gesture)[:-1, None]
            reward_policy = actor.deterministic(changed_reward)
            combined = changed_action.clone()
            combined[1:, offsets['reward']] = changed_reward[1:, offsets['reward']]
            both_policy = actor.deterministic(combined)
        result['actor_sensitivity'].update(
            baseline_geometry_penalty=describe(new_gesture.numpy()),
            previous_reward_delta=describe((new_gesture-old_gesture).numpy()),
            previous_reward_only_policy_max_abs_per_frame=describe((reward_policy-policy).abs().amax(-1).numpy()),
            previous_reward_only_policy_rms=float((reward_policy-policy).square().mean().sqrt()),
            both_fields_policy_max_abs_per_frame=describe((both_policy-policy).abs().amax(-1).numpy()),
            both_fields_policy_rms=float((both_policy-policy).square().mean().sqrt()))
        arrays.update(policy_action_previous_reward_only_delta=(reward_policy-policy).numpy(),
            policy_action_both_fields_delta=(both_policy-policy).numpy(),
            baseline_new_geometry_penalty=new_gesture.numpy())
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output/'fixed_baseline_fk.npz', **arrays)
    (a.output/'fixed_baseline_fk.json').write_text(json.dumps(result, indent=2)+'\n')
    (a.output/'source_recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('active','successful_assigned','idle')}, indent=2))


if __name__ == '__main__':
    main()
