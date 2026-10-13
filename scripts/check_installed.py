#!/usr/bin/env python3
"""Verify a wheel installed with pip --target, blocking source-tree file access.

Run with the project venv Python and -I. Third-party dependencies under
sys.prefix remain available; the wheel target must be outside --forbid-root.
"""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('--forbid-root', type=Path, required=True)
    args = parser.parse_args()
    target, forbidden = args.target.resolve(), args.forbid_root.resolve()
    if target.is_relative_to(forbidden):
        parser.error('Install the wheel target outside the forbidden source tree')
    sys.path.insert(0, str(target))
    dependency_root = Path(sys.prefix).resolve()

    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            value = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
            path = Path(value).resolve()
            if path.is_relative_to(forbidden) and not path.is_relative_to(dependency_root):
                raise RuntimeError(f'Source tree access forbidden: {path}')
    sys.addaudithook(audit)

    class BlockOriginal(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'robopianist', 'komopianist', 'tsensei_pianist', 'tsensei_v3_pianist'}:
                raise RuntimeError(f'Original project import forbidden: {fullname}')
    sys.meta_path.insert(0, BlockOriginal())
    import numpy as np
    from tsensei_v3_2_pianist import suite
    from tsensei_v3_2_pianist.models.hands.tsensei_hand_constants import ASSETS
    assert ASSETS.is_relative_to(target)
    results = []
    for midi in ('Toy', str(ASSETS.parent / 'music/data/rousseau/twinkle-twinkle-trimmed.mid')):
        env = suite.load(midi)
        try:
            env.reset()
            for _ in range(40):
                step = env.step(np.r_[env.task.right_hand.neutral_control,
                                      env.task.left_hand.neutral_control, 0.])
                assert np.isfinite(step.reward)
                assert all(np.isfinite(v).all() for v in step.observation.values())
                if step.last():
                    env.reset()
            results.append({'midi': Path(midi).name, 'steps': 40, 'action_size': env.action_spec().shape[0]})
        finally:
            env.close()
    assert all(Path(module.__file__).is_relative_to(target)
               for name, module in sys.modules.items()
               if name.startswith('tsensei_v3_2_pianist') and getattr(module, '__file__', None))
    print(json.dumps({'installed_wheel_isolation': 'passed', 'runs': results}, indent=2))


if __name__ == '__main__':
    main()
