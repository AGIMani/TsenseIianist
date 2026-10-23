"""Explicit Slurm export acceptance; inactive in ordinary test runs."""
import os
from pathlib import Path
import pytest


def test_recorded_ik_video_export():
    source=os.environ.get('IK_EXPORT_RECORDING')
    if not source:pytest.skip('Set IK_EXPORT_RECORDING for explicit recorded-IK export')
    from scripts.hpc.export_ik_recording import export
    result=export(Path(source),Path(os.environ['IK_EXPORT_OUTPUT']))
    assert result['decoded_video_and_audio'] and result['actual_key_highlight']
    assert result['physics_rerun'] is False
