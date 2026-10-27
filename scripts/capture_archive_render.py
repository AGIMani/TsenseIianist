#!/usr/bin/env python3
"""Passively capture full physics states while running a verified archived renderer.

This observer runs outside the fingerprinted archive. Its callback only copies
state; archived action, reward, physics, and checkpoint checks remain unchanged.
"""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys


def run_and_flush(run, flush):
    try:
        result=run()
    except BaseException as error:
        flush(error)
        raise
    flush(None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    parser2 = argparse.ArgumentParser(add_help=False)
    parser2.add_argument('--output', type=Path, required=True)
    output, _ = parser2.parse_known_args(remaining)
    sys.path.insert(0, str(args.archive_root))
    import mujoco
    import numpy as np
    import torch
    from tsensei_v3_2_pianist.mjlab_env.scene import PianoScene
    original_step = PianoScene.step
    records = {key: [] for key in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat', 'sustain')}
    captured = {}

    def observed_step(scene, *step_args, substep_callback=None, **kwargs):
        if not captured:
            folder = output.output / 'kinematic_scene'
            folder.mkdir(parents=True)
            # Preserve the compiled, structurally locked model for offline FK.
            for name in scene.bundle.manifest['files']:
                if name != 'scene.xml' and name != 'cpu_reference.npz':
                    shutil.copy2(scene.bundle.directory / name, folder / name)
            mujoco.mj_saveLastXML(str(folder / 'scene.xml'), scene.model)
            mujoco.mj_saveModel(scene.model, str(folder / 'model.mjb'))
            captured.update(physics_dt=float(scene.model.opt.timestep),
                            hands=scene.bundle.manifest['hands'],
                            initial_qpos=scene.data.qpos[0].detach().cpu().numpy().copy(),
                            initial_mocap_pos=scene.data.mocap_pos[0].detach().cpu().numpy().copy(),
                            initial_mocap_quat=scene.data.mocap_quat[0].detach().cpu().numpy().copy())

        def observe(current):
            if substep_callback is not None:
                substep_callback(current)
            for key in records:
                value = current.sustain[0] if key == 'sustain' else getattr(current.data, key)[0]
                records[key].append(value.detach().clone())

        return original_step(scene, *step_args, substep_callback=observe, **kwargs)

    PianoScene.step = observed_step
    sys.argv = [str(args.archive_root / 'scripts/eval_from_source.py'), *remaining]
    def flush(error):
        PianoScene.step = original_step
        if not records['qpos']:
            return
        torch.cuda.synchronize()
        arrays = {key: torch.stack(value).cpu().numpy() for key, value in records.items()}
        # Save policy data before any tail validation: a bad seam cannot erase it.
        if error is not None:
            np.savez_compressed(output.output/'failed_policy_physics.npz',**arrays,
                **{key:captured[key] for key in ('initial_qpos','initial_mocap_pos','initial_mocap_quat','physics_dt')})
        ending_samples = 0
        ending_path = output.output / 'ending.npz'
        report_path = output.output/'report.json'
        report_data = json.loads(report_path.read_text()) if report_path.exists() else {}
        if report_data.get('ending_seconds',0) or (error is not None and ending_path.exists()):
            from tsensei_v3_2_pianist.mjlab_env.ending import append_trace
            with np.load(ending_path,allow_pickle=False) as ending:
                if error is None and not np.isclose(ending['time'][-1],report_data['ending_seconds']):
                    raise ValueError('Ending report duration mismatch')
                arrays=append_trace(arrays,ending,captured['physics_dt'])
                ending_samples=len(ending['time'])-1
        arrays.update({key: captured[key] for key in ('initial_qpos', 'initial_mocap_pos', 'initial_mocap_quat')})
        arrays['physics_dt'] = captured['physics_dt']
        np.savez_compressed(output.output / 'physics_trace.npz', **arrays)
        report = {key: captured[key] for key in ('physics_dt', 'hands')}
        report.update(complete=error is None, error=None if error is None else str(error), samples=len(arrays['qpos']), ending_samples=ending_samples, observer='read-only state copies after each physics substep',
                      source='strictly verified archived checkpoint rollout',
                      times='state i corresponds to (i+1)*physics_dt; initial state is saved separately')
        (output.output / 'physics_trace.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Captured {report["samples"]} complete physics states for arm IK.', flush=True)
    try:
        run_and_flush(lambda:runpy.run_path(str(args.archive_root/'scripts/eval_from_source.py'),run_name='__main__'),flush)
    finally:
        PianoScene.step=original_step


if __name__ == '__main__':
    main()
