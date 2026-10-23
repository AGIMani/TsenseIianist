"""Compile Composer drive contracts with the deployment's source-build Python."""
from pathlib import Path
import subprocess

def test_composer_drive_contracts():
    root=Path(__file__).resolve().parents[1]
    subprocess.run([str(root/'training/.venv/bin/python'), '-m', 'pytest', '-q',
                    'tests/test_drive_profiles.py'], cwd=root, check=True)
