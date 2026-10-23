import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('source_pythonpath',
    Path(__file__).resolve().parents[1] / 'scripts/source_pythonpath.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourcePythonPathTest(unittest.TestCase):
    def test_does_not_inherit_another_source(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, PYTHONPATH='/old/project'):
            root = Path(temp)
            self.assertEqual(module.source_pythonpath(root), str(root.resolve()))

    def test_archive_uses_only_explicit_deployment_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, deployment = root/'frozen', root/'deployment'
            deps = deployment/'training_mjlab/native_ik_deps'
            deps.mkdir(parents=True)
            self.assertEqual(module.source_pythonpath(source, deployment_root=deployment),
                             os.pathsep.join((str(source), str(deps))))

    def test_slurm_source_dependency_symlink_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root/'frozen'
            deps = root/'dependencies'
            deps.mkdir()
            link = frozen/'training_mjlab/native_ik_deps'
            link.parent.mkdir(parents=True)
            link.symlink_to(deps, target_is_directory=True)
            self.assertEqual(module.source_pythonpath(frozen),
                             os.pathsep.join((str(frozen), str(deps))))
