"""Run the real entry preflight while intercepting its final submission command."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def outer_entry_available():
    entry = ROOT / 'all_use_this_train.sh'
    if not entry.is_file():
        return False
    match = re.search(r'training_archive="\$ROOT/([^\"]+)"', entry.read_text())
    return bool(match and (ROOT / match.group(1)).is_file())


pytestmark = pytest.mark.skipif(not outer_entry_available(),
    reason='Outer submission entry and pinned archive are verified in the project checkout, outside the extracted source bundle')


def run_entry(tmp_path, *arguments):
    capture = tmp_path / 'submission.json'
    binary = tmp_path / 'bin'
    binary.mkdir(exist_ok=True)
    fake_bash = binary / 'bash'
    fake_bash.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'open(os.environ["ENTRY_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake_bash.chmod(0o755)
    env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'], 'ENTRY_CAPTURE': str(capture)}
    result = subprocess.run(['/bin/bash', str(ROOT / 'all_use_this_train.sh'), *arguments],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    return result, json.loads(capture.read_text()) if capture.exists() else None


def test_entry_passes_world_positions_and_allows_last_override(tmp_path):
    result, command = run_entry(tmp_path, '--world-x', '.20', '--left-world-y', '-.12')
    assert result.returncode == 0, result.stderr
    assert command[0] == str(ROOT / 'scripts/hpc/submit.sh')
    for key, value in [('--world-x', '0.16'), ('--world-z', '0.09'),
                        ('--right-world-y', '0.15'), ('--left-world-y', '-0.15')]:
        assert command[command.index(key) + 1] == value
    assert command[-4:] == ['--world-x', '.20', '--left-world-y', '-.12']
    assert not set(command) & {'--height', '--ty', '--yaw'}


@pytest.mark.parametrize('arguments', [('--height', '.09'), ('--ty=.02',), ('--yaw', '.1'),
                                     ('--world-x', '.24'), ('--world-z', 'nan')])
def test_ambiguous_or_invalid_flags_do_not_reach_submission(tmp_path, arguments):
    result, command = run_entry(tmp_path, *arguments)
    assert result.returncode == 2
    assert command is None
