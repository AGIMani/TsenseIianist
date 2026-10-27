"""Lightweight isolated Git checks; no physics or GPU dependencies."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('release_tracking', Path(__file__).resolve().parents[1] / 'scripts/check_release_tracking.py')
release_tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_tracking)


class ReleaseTrackingTests(unittest.TestCase):
    def test_missing_and_mismatched_committed_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', directory], check=True)
            sha = 'a' * 64
            path = 'logs/new_release/source_release.tar.gz'
            (root / 'all_use_this_train.sh').write_text(f'training_archive="$ROOT/{path}"\ntraining_archive_sha={sha}\n')
            subprocess.run(['git', '-C', directory, 'add', '.'], check=True)
            with patch.object(release_tracking, 'ROOT', root), patch('sys.argv', ['check', '--ref', 'INDEX']):
                with self.assertRaises(subprocess.CalledProcessError):
                    release_tracking.main()
                archive = root / path
                archive.parent.mkdir(parents=True)
                archive.write_text(f'version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize 123\n')
                sidecar = root / (path + '.sha256')
                sidecar.write_text('b' * 64)
                subprocess.run(['git', '-C', directory, 'add', '.'], check=True)
                with self.assertRaises(ValueError):
                    release_tracking.main()
                sidecar.write_text(sha)
                subprocess.run(['git', '-C', directory, 'add', '.'], check=True)
                release_tracking.main()

    def test_register_is_idempotent_and_preserves_evidence_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = 'logs/new/source_release.tar.gz'
            (root / path).parent.mkdir(parents=True)
            (root / path).touch()
            (root / (path + '.sha256')).touch()
            (root / 'all_use_this_train.sh').write_text(f'training_archive="$ROOT/{path}"\ntraining_archive_sha={"a" * 64}\n')
            ignore = root / '.gitignore'
            ignore.write_text('/logs/*\n!/logs/new/\n/logs/new/*\n!/logs/new/validation.json\n')
            with patch.object(release_tracking, 'ROOT', root), patch('sys.argv', ['check', '--register']):
                release_tracking.main()
                first = ignore.read_text()
                release_tracking.main()
                self.assertEqual(first, ignore.read_text())
                self.assertEqual(first.count('/logs/new/*'), 1)
