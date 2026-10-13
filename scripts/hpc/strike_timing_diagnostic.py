#!/usr/bin/env python3
"""Same-state, open-loop strike interventions on an archived physical recording.

Actual MuJoCo stepping is permitted only inside a Slurm allocation. This loads
recorded physical targets and the saved compiled model; it never imports a newer
policy/coupler or alters a checkpoint. CPU solver results are compared against a
repeated CPU baseline, with drift from the original GPU recording reported apart.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDING = ROOT / 'logs/hpc/1353/export_work/recording'
DEFAULT_RUN = ROOT / 'training_mjlab/outputs/hpc/gesture10-comparison-20260930-140658-445247'
FINGERS = ('Thumb', 'Index', 'Mid', 'Ring', 'Pinky')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shifted_targets(ctrl, channels, *, onset, release, lead, donor_onset=None,
                    donor_release=None):
    """Advance only a finger's press phase, keeping its original release intact.

    Before onset-lead and from release onward, all targets are unmodified. The
    final pre-release tuple is held when advancing would cross the release.
    Donor tuples are copied whole: no independent PIP/DIP scaling is introduced.
    """
    if lead < 0 or onset < lead or not onset < release <= len(ctrl):
        raise ValueError('Invalid onset/release/lead interval')
    if donor_onset is None:
        donor_onset, donor_release = onset, release
    if donor_release is None or not 0 <= donor_onset < donor_release <= len(ctrl):
        raise ValueError('Invalid donor interval')
    result = ctrl.copy()
    dst = np.arange(onset - lead, release)
    src = np.minimum(donor_onset + dst - (onset - lead), donor_release - 1)
    result[np.ix_(dst, channels)] = ctrl[np.ix_(src, channels)]
    return result


def key_active(qpos, limits, threshold):
    return np.abs(np.clip(qpos, limits[:, 0], limits[:, 1]) - limits[:, 1]) <= threshold


def activation_events(mask, initial, times):
    changes = np.diff(np.r_[bool(initial), np.asarray(mask, bool)].astype(int))
    return times[changes == 1], times[changes == -1]


def observed_note(mask, initial, times, note, next_onset=float('inf')):
    ons, offs = activation_events(mask, initial, times)
    lo = note['start_time'] - .075
    hi = min(note['end_time'] + .075, next_onset)
    candidates = ons[(ons >= lo - 1e-10) & (ons < hi - 1e-10)]
    onset = float(candidates[0]) if len(candidates) else None
    end_candidates = offs[offs > onset + 1e-10] if onset is not None else []
    offset = float(end_candidates[0]) if len(end_candidates) else None
    return dict(onset_s=onset, offset_s=offset,
                onset_vs_original_ms=None if onset is None else (onset-note['start_time'])*1000,
                onset_vs_grid_ms=None if onset is None else (onset-note['target_start_time'])*1000,
                offset_vs_original_ms=None if offset is None else (offset-note['end_time'])*1000,
                offset_vs_grid_ms=None if offset is None else (offset-note['target_end_time'])*1000)


def release_frame(ctrl, mcp, onset, end):
    # Preserve the large release already learned by the policy, including the
    # frequent release one control frame before the quantized note end.
    low, high = max(onset+1, end-2), min(len(ctrl)-1, end+1)
    candidates = np.arange(low, high+1)
    differences = ctrl[candidates, mcp] - ctrl[candidates-1, mcp]
    chosen = int(candidates[np.argmin(differences)])
    return chosen if differences.min() < -.1 else end


def hand_side(name):
    return 1 if name.startswith('right_/') else 2 if name.startswith('left_/') else 0


def prepare(args):
    recording, run = args.recording.resolve(), args.run.resolve()
    config = json.loads((run/'config.json').read_text())
    recording_report = json.loads((recording/'report.json').read_text())
    for option in ('finger_coupling', 'preparation', 'control_limits'):
        if config.get(option) != recording_report['reward_options'].get(option):
            raise ValueError(f'Run config and recorded replay disagree on {option}')
    coverage = json.loads((run/'score_coverage.json').read_text())
    trace = dict(np.load(recording/'physics_trace.npz'))
    rollout = dict(np.load(recording/'rollout.npz'))
    audio = dict(np.load(recording/'audio_substeps.npz'))
    notes = coverage['notes']
    pdt, cdt = float(trace['physics_dt']), float(coverage['control_dt'])
    dec = round(cdt/pdt)
    if not np.isclose(dec*pdt, cdt) or len(trace['qpos']) != len(rollout['ctrl'])*dec:
        raise ValueError('Recording physics/control timebases are inconsistent')
    music_frames = np.flatnonzero(rollout['phase'] == 'music')
    if not len(music_frames):
        raise ValueError('Recording has no music phase')
    prefix_frames = int(music_frames[0])
    prefix_seconds = prefix_frames*cdt
    reconstructed_grid = np.zeros_like(rollout['goals'][:,:88], dtype=bool)
    for note in notes:
        reconstructed_grid[prefix_frames+np.asarray(note['retained_frames']), note['pitch']-21] = True
    if not np.array_equal(reconstructed_grid, rollout['goals'][:,:88].astype(bool)):
        raise ValueError('Run score coverage does not reproduce recorded target frames')
    if np.any(np.diff(rollout['policy_time']) <= 0):
        raise ValueError('Recording time must be strictly increasing')
    times = (np.arange(len(trace['qpos']))+1)*pdt-prefix_seconds
    original_active = key_active(audio['key_qpos'], audio['key_limits'], audio['key_threshold'])
    for i, note in enumerate(notes):
        note['note_index'] = i+1
        interval = (times >= note['start_time']) & (times < note['end_time'])
        note['recorded_success'] = bool(np.any(original_active[interval, note['pitch']-21]))
    selected = notes if not args.notes else [notes[i-1] for i in args.notes]
    if not selected:
        raise ValueError('No notes selected')
    plan = dict(schema='strike_intervention_v1', recording=str(recording), run=str(run),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), backend='CPU MuJoCo',
                control_dt=cdt, physics_dt=pdt, preparation_seconds=prefix_seconds,
                finger_coupling=config['finger_coupling'], lead_ms=args.lead_ms,
                selected_note_indices=[n['note_index'] for n in selected],
                checkpoint_sha256=recording_report['checkpoint_sha256'],
                model_sha256=sha256(recording/'kinematic_scene/model.mjb'),
                config_sha256=sha256(run/'config.json'),
                recording_sha256={name: sha256(recording/name) for name in
                                  ('physics_trace.npz', 'rollout.npz', 'audio_substeps.npz')},
                command_rule='Advance complete MCP/PIP/DIP tuples; thumb uses MCP/PIP only. Preserve original release frame; hold final pre-release tuple when needed.',
                amplitude_rule='For recorded misses shorter than 150 ms, transplant the nearest successful same-finger/same-pitch note lasting >=150 ms, with identical advance and recipient release boundary.',
                reset_rule='Recorded qpos/qvel/mocap at the shared pre-intervention tick; original ctrl; no actuator activation exists in this model. Unrecorded solver warmstart/forces set to zero. Clone full mjSTATE_INTEGRATION state for all cases.',
                limitations=[
                    'Open-loop CPU interventions are not policy evaluations or evidence of trained F1 improvement.',
                    'Saved GPU qvel has float32 precision and contains no GPU solver warmstart; reconstructed CPU baseline can diverge from the original GPU recording.',
                    'PIP/DIP targets replay archived fixed-60-degree tuples; they are not recomputed using current source or the recipient wrist. Dynamic distal angle is not guaranteed.',
                    'Donor amplitude changes a complete learned flexion tuple, not MCP amplitude alone; wrist, finger swing and all other actuator commands remain unchanged.',
                    'Contacts/forces are solved CPU contacts before each integration step; key activation is sampled after that step.',
                    'A local window can end with a held key; missing offset then means censored, not sustained forever.',
                ])
    return plan, config, notes, selected, trace, rollout, audio, original_active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, default=DEFAULT_RECORDING)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lead-ms', type=float, nargs='+', default=[0,25,50])
    parser.add_argument('--notes', type=int, nargs='+', help='One-based score note indices; default all 34')
    parser.add_argument('--dry-run', action='store_true', help='Validate recording and write a plan without physics')
    args = parser.parse_args()
    if any(v < 0 for v in args.lead_ms) or 0 not in args.lead_ms:
        parser.error('--lead-ms must be nonnegative and include 0')
    if args.notes and any(v < 1 or v > 34 for v in args.notes):
        parser.error('--notes must be between 1 and 34 for this recording')
    if not args.dry_run and not os.environ.get('SLURM_JOB_ID'):
        parser.error('Physics rollouts require a Slurm allocation; use scripts/hpc/submit.sh strike-diagnostic')
    plan, config, notes, selected, trace, rollout, audio, recorded_active = prepare(args)
    pdt, cdt = plan['physics_dt'], plan['control_dt']
    dec, prefix = round(cdt/pdt), round(plan['preparation_seconds']/cdt)
    for ms in args.lead_ms:
        if not np.isclose(round(ms/1000/cdt)*cdt*1000, ms):
            parser.error('--lead-ms must be an integer number of control intervals')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return

    os.environ['MUJOCO_GL'] = 'disable'
    import mujoco
    m = mujoco.MjModel.from_binary_path(str(args.recording/'kinematic_scene/model.mjb'))
    if m.na:
        raise ValueError('Recording has no actuator state; cannot reconstruct a model with na > 0')
    if not np.isclose(m.opt.timestep, pdt):
        raise ValueError('Saved model physics_dt differs from trace')
    model_names = [m.geom(i).name for i in range(m.ngeom)]
    sides = np.array([hand_side(name) for name in model_names])
    key_by_geom = {}
    key_qpos = np.empty(88, dtype=int)
    for gid, name in enumerate(model_names):
        match = re.fullmatch(r'piano/(?:white|black)_key_geom_(\d+)', name)
        if match:
            key = int(match[1]); key_by_geom[gid] = key
            body = m.geom_bodyid[gid]
            key_qpos[key] = m.jnt_qposadr[m.body_jntadr[body]]
    if len(key_by_geom) != 88 or m.nu != rollout['ctrl'].shape[1]:
        raise ValueError('Saved physical model indices do not match recording')
    if not np.allclose(trace['qpos'][:,key_qpos], audio['key_qpos'], atol=2e-8):
        raise ValueError('Physical trace and audio disagree on piano key positions')
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    rows, details, traces, repeatability = [], [], {}, []
    global_times = (np.arange(len(trace['qpos']))+1)*pdt-plan['preparation_seconds']
    goals = np.zeros((len(trace['qpos']),88), bool)
    for note in notes:
        goals[(global_times >= note['start_time']) & (global_times < note['end_time']), note['pitch']-21] = True

    for note in selected:
        idx, finger = note['note_index'], int(note['fingering'])
        side = 'R' if finger < 5 else 'L'
        fname = FINGERS[finger % 5]
        channel_names = [f'{side}_{fname}_{joint}' for joint in
                         (('MCP','PIP') if finger % 5 == 0 else ('MCP','PIP','DIP'))]
        channels = np.array([m.actuator(name).id for name in channel_names])
        mcp = int(channels[0])
        onset = prefix+int(note['quantized_start_frame'])
        end = prefix+int(note['quantized_end_frame'])
        release = release_frame(rollout['ctrl'], mcp, onset, end)
        lead_frames = [round(ms/1000/cdt) for ms in args.lead_ms]
        start_frame = max(0, onset-max(lead_frames)-1)
        end_frame = min(len(rollout['ctrl']), max(end,release)+4)
        lo, hi = start_frame*dec, end_frame*dec
        if lo == 0:
            raise ValueError('Initial qvel was not recorded; select a note with a pre-state at tick > 0')
        ticks = np.arange(lo, hi)
        times = global_times[ticks]
        next_same_pitch = min((n['start_time'] for n in notes if n['pitch']==note['pitch']
                               and n['start_time'] > note['start_time']), default=float('inf'))
        key = note['pitch']-21
        saved = mujoco.MjData(m)
        mujoco.mj_resetData(m, saved)
        saved.time = lo*pdt
        saved.qpos[:] = trace['qpos'][lo-1]
        saved.qvel[:] = trace['qvel'][lo-1]
        saved.ctrl[:] = rollout['ctrl'][start_frame-1]
        saved.mocap_pos[:] = trace['mocap_pos'][lo-1]
        saved.mocap_quat[:] = trace['mocap_quat'][lo-1]
        saved.qacc_warmstart[:] = 0
        saved.qfrc_applied[:] = 0
        saved.xfrc_applied[:] = 0
        initial_state = np.empty(mujoco.mj_stateSize(m, spec))
        mujoco.mj_getState(m, saved, initial_state, spec)
        initial_active = key_active(saved.qpos[key_qpos], audio['key_limits'], audio['key_threshold'])
        saved_initial_key_qpos = saved.qpos[key_qpos].copy()

        def simulate(commands):
            d = mujoco.MjData(m)
            mujoco.mj_setState(m, d, initial_state, spec)
            positions = np.empty((len(ticks),m.nq))
            active = np.empty((len(ticks),88), bool)
            flags = {k: np.zeros(len(ticks),bool) for k in
                     ('assigned_contact','assigned_loaded_contact','any_key_contact',
                      'same_hand','cross_hand','key_side_or_under','key_underside')}
            peak_force = 0.
            max_penetration = 0.
            for j, tick in enumerate(ticks):
                d.ctrl[:] = commands[tick//dec]
                d.mocap_pos[:] = trace['mocap_pos'][tick]
                d.mocap_quat[:] = trace['mocap_quat'][tick]
                mujoco.mj_step(m, d)
                positions[j] = d.qpos
                active[j] = key_active(d.qpos[key_qpos],audio['key_limits'],audio['key_threshold'])
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    if c.dist > 1e-6 or c.efc_address < 0:
                        continue
                    g0,g1 = int(c.geom1),int(c.geom2)
                    s0,s1 = sides[g0],sides[g1]
                    if s0 and s1:
                        flags['same_hand' if s0==s1 else 'cross_hand'][j] = True
                        continue
                    kg = g0 if g0 in key_by_geom else g1 if g1 in key_by_geom else -1
                    hg = g1 if kg==g0 else g0
                    if kg < 0 or not sides[hg]:
                        continue
                    flags['any_key_contact'][j] = True
                    # Contact normal points geom0 -> geom1; turn it outward
                    # from the piano key before classifying its local surface.
                    outward = c.frame[:3]*(1 if kg==g0 else -1)
                    local_z = float((d.geom_xmat[kg].reshape(3,3).T@outward)[2])
                    flags['key_side_or_under'][j] |= local_z < .7
                    flags['key_underside'][j] |= local_z <= -.7
                    max_penetration = max(max_penetration,float(-c.dist))
                    finger_token = f'_hand_{fname.lower()}_'
                    if key_by_geom[kg] == key and finger_token in model_names[hg] and sides[hg] == (1 if side=='R' else 2):
                        flags['assigned_contact'][j] = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(m,d,ci,force)
                        peak_force = max(peak_force,float(force[0]))
                        flags['assigned_loaded_contact'][j] |= force[0] > 1e-6
            if not np.all(np.isfinite(positions)):
                raise RuntimeError('Non-finite CPU rollout')
            return positions,active,flags,peak_force,max_penetration

        donor = None
        if not note['recorded_success'] and note['end_time']-note['start_time'] < .15:
            candidates = [n for n in notes if n['pitch']==note['pitch'] and n['fingering']==finger
                          and n['recorded_success'] and n['end_time']-n['start_time'] >= .15]
            if candidates:
                donor = min(candidates,key=lambda n:abs(n['start_time']-note['start_time']))
        variants = [('recorded',None)] + ([('donor_press_tuple',donor)] if donor else [])
        baseline = None
        original_event = observed_note(recorded_active[lo:hi,key],initial_active[key],times,note,next_same_pitch)
        for amplitude, source_note in variants:
            donor_onset = prefix+source_note['quantized_start_frame'] if source_note else None
            donor_end = prefix+source_note['quantized_end_frame'] if source_note else None
            donor_release = release_frame(rollout['ctrl'],mcp,donor_onset,donor_end) if source_note else None
            for ms,lead in zip(args.lead_ms,lead_frames):
                commands = shifted_targets(rollout['ctrl'],channels,onset=onset,release=release,lead=lead,
                                           donor_onset=donor_onset,donor_release=donor_release)
                other = np.setdiff1d(np.arange(m.nu),channels)
                assert np.array_equal(commands[:,other],rollout['ctrl'][:,other])
                assert np.array_equal(commands[release:],rollout['ctrl'][release:])
                if amplitude=='recorded' and lead==0:
                    assert np.array_equal(commands,rollout['ctrl'])
                result = simulate(commands)
                positions,active,flags,peak_force,penetration = result
                if amplitude=='recorded' and lead==0:
                    repeated = simulate(commands)
                    max_error = float(np.max(np.abs(repeated[0]-positions)))
                    exact = bool(np.array_equal(repeated[0],positions) and np.array_equal(repeated[1],active)
                                 and all(np.array_equal(repeated[2][k],flags[k]) for k in flags))
                    repeatability.append(dict(note_index=idx,bitwise_equal=exact,max_qpos_error=max_error))
                    if not exact:
                        raise RuntimeError(f'CPU baseline reset was not deterministic for note {idx}')
                    baseline = active
                event = observed_note(active[:,key],initial_active[key],times,note,next_same_pitch)
                expected = goals[lo:hi]
                fp = active & ~expected
                fn = ~active & expected
                tp = active & expected
                contacts = np.flatnonzero(flags['assigned_contact'])
                first_contact = float(ticks[contacts[0]]*pdt-plan['preparation_seconds']) if len(contacts) else None
                press_interval = (times >= note['start_time']) & (times < note['end_time'])
                fractional = np.clip((positions[:,key_qpos[key]]-audio['key_limits'][key,0]) /
                                     np.diff(audio['key_limits'][key])[0],0,1)
                identity = f'note{idx:02d}_{amplitude}_lead{ms:g}ms'
                row = dict(case=identity,note_index=idx,pitch=note['pitch'],finger=f'{side}_{fname}',
                           original_start_s=note['start_time'],original_end_s=note['end_time'],
                           grid_start_s=note['target_start_time'],grid_end_s=note['target_end_time'],
                           recording_note_success=note['recorded_success'],
                           amplitude=amplitude,donor_note_index=source_note['note_index'] if source_note else None,
                           lead_ms=ms,actual_first_command_s=(onset-lead)*cdt-plan['preparation_seconds'],
                           preserved_release_command_s=release*cdt-plan['preparation_seconds'],
                           **event,assigned_first_contact_s=first_contact,
                           assigned_contact_vs_original_ms=None if first_contact is None else (first_contact-note['start_time'])*1000,
                           assigned_peak_contact_force_N=peak_force,max_key_penetration_mm=penetration*1000,
                           note_active_substeps=int(active[press_interval,key].sum()),
                           note_target_substeps=int(press_interval.sum()),
                           note_max_travel_fraction=float(fractional[press_interval].max()) if press_interval.any() else None,
                           wrong_key_substeps=int(fp.any(axis=1).sum()),wrong_key_samples=int(fp.sum()),
                           missed_key_samples=int(fn.sum()),true_key_samples=int(tp.sum()),
                           early_target_key_samples=int((active[:,key] & (times < note['start_time'])).sum()),
                           late_target_key_samples=int((active[:,key] & (times >= note['end_time'])).sum()),
                           activation_difference_samples_from_recording=int((active != recorded_active[lo:hi]).sum()),
                           max_qpos_difference_from_recording=float(np.abs(positions-trace['qpos'][lo:hi]).max()),
                           **{f'{k}_substeps':int(v.sum()) for k,v in flags.items()})
                rows.append(row)
                details.append(dict(**row,window_start_s=lo*pdt-plan['preparation_seconds'],window_end_s=hi*pdt-plan['preparation_seconds'],
                                    original_gpu_event=original_event,changed_actuators=channel_names,
                                    initial_integration_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
                                    wrist_and_other_command_arrays_unchanged=True))
                traces[identity+'__time'] = times
                traces[identity+'__key_qpos'] = positions[:,key_qpos]
                traces[identity+'__ctrl'] = commands[ticks//dec][:,channels]
                traces[identity+'__finger_qpos'] = positions[:,m.jnt_qposadr[m.actuator_trnid[channels,0]]]
                traces[identity+'__contact'] = flags['assigned_contact']
                print(json.dumps({k:row[k] for k in ('case','onset_vs_original_ms','assigned_contact_substeps','note_active_substeps','wrong_key_samples','same_hand_substeps','cross_hand_substeps')}),flush=True)
        (args.output/'progress.json').write_text(json.dumps(dict(completed_notes=idx,cases=len(rows)),indent=2)+'\n')

    with (args.output/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(args.output/'traces.npz',**traces)
    groups=[]
    for amplitude in ('recorded','donor_press_tuple'):
        for lead in args.lead_ms:
            chosen=[r for r in rows if r['amplitude']==amplitude and r['lead_ms']==lead]
            if not chosen:continue
            errors=[r['onset_vs_original_ms'] for r in chosen if r['onset_vs_original_ms'] is not None]
            groups.append(dict(amplitude=amplitude,lead_ms=lead,cases=len(chosen),
                               cases_with_activation=sum(r['note_active_substeps']>0 for r in chosen),
                               mean_onset_vs_original_ms=float(np.mean(errors)) if errors else None,
                               mean_absolute_onset_error_ms=float(np.mean(np.abs(errors))) if errors else None,
                               wrong_key_samples=sum(r['wrong_key_samples'] for r in chosen),
                               missed_key_samples=sum(r['missed_key_samples'] for r in chosen),
                               same_hand_substeps=sum(r['same_hand_substeps'] for r in chosen),
                               cross_hand_substeps=sum(r['cross_hand_substeps'] for r in chosen),
                               key_underside_substeps=sum(r['key_underside_substeps'] for r in chosen)))
    report=dict(**plan,mujoco_version=mujoco.__version__,model_opt=dict(timestep=m.opt.timestep,integrator=int(m.opt.integrator),solver=int(m.opt.solver)),
                cases=details,baseline_repeatability=repeatability,groups=groups,
                summary_scope='Sums over independent overlapping local windows; not whole-song F1 or unique whole-song collision totals.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),groups=groups),indent=2),flush=True)


if __name__=='__main__':
    main()
