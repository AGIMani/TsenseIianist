"""Keep frozen source first and only reuse this deployment's optional IK deps."""
import os
from pathlib import Path


def source_pythonpath(source_root, *, deployment_root=None):
    source_root = Path(source_root).resolve()
    deployment_root = Path(deployment_root or source_root).resolve()
    paths = [str(source_root)]
    dependencies = deployment_root / 'training_mjlab' / 'native_ik_deps'
    if dependencies.is_dir():
        paths.append(str(dependencies.resolve()))
    # Deliberately ignore inherited PYTHONPATH: it may contain another source.
    return os.pathsep.join(paths)
