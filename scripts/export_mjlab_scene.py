#!/usr/bin/env python3
"""Freeze current V3.2 physical scene and reset pose for MJLab."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'training_mjlab/outputs/scene_source_20hz200hz_gravcomp')
    parser.add_argument('--control-hz', type=float, default=20., help='Control frequency (default: 20 Hz)')
    parser.add_argument('--physics-hz', type=float, default=200., help='Physics frequency (default: 200 Hz)')
    args = parser.parse_args()
    import math
    for hz in (args.control_hz, args.physics_hz):
        if hz is not None and (not math.isfinite(hz) or hz <= 0):
            parser.error('frequencies must be finite and positive')
    from tsensei_v3_2_pianist.mjlab_env.bundle import export_scene
    info = export_scene(args.output, gravity_compensation=True,
                        control_timestep=1 / args.control_hz,
                        physics_timestep=1 / args.physics_hz)
    print(f'Exported {args.output.resolve()}: nq={info["nq"]}, nu={info["nu"]}, actions=51')


if __name__ == '__main__':
    main()
