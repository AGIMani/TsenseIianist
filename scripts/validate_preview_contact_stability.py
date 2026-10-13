#!/usr/bin/env python3
"""Exercise current Composer preview contacts at both supported physics rates."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsensei_v3_2_pianist import suite
from scripts.validate_contact_stability import CASES, category, metrics, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dt in (.0025, .005):
        env = suite.load(task_kwargs={'physics_timestep': dt, 'control_timestep': 10*dt})
        try:
            env.reset()
            controls, qa, va = targets(env.physics.model.ptr, env.physics.data.ctrl)
            for case, ctrl in zip(CASES, controls):
                env.reset()
                # Composer can rebuild Physics at reset; never reuse its old pointer.
                physics = env.physics
                m, d = physics.model.ptr, physics.data.ptr
                assert m.npair == 0
                d.ctrl[:] = ctrl
                qs, vs, peak = [], [], 0.
                for step in range(round(20/dt)):
                    physics.step()
                    # Same extra forward cadence as preview_runtime, without a GUI.
                    if (step+1) % max(1, round((1/60)/dt)) == 0:
                        physics.forward()
                    if (step+1)*dt < 18:
                        continue
                    qs.append(d.qpos[qa].copy()); vs.append(d.qvel[va].copy())
                    for index, contact in enumerate(d.contact):
                        if contact.efc_address >= 0 and category(m, contact.geom) == case:
                            force = np.zeros(6)
                            mujoco.mj_contactForce(m, d, index, force)
                            peak = max(peak, abs(float(force[0])))
                row = dict(case=case, physics_dt=dt,
                           **metrics(np.asarray(qs), np.asarray(vs), dt), peak_normal_force_N=peak)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    assert all(r['peak_normal_force_N'] > .01 and r['velocity_rms_rad_s'] < .02 for r in rows)


if __name__ == '__main__':
    main()
