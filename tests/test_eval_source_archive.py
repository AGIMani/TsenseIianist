"""Archive replay must not write through links or accept mismatched source."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/eval_from_source_archive.py'
SPEC = importlib.util.spec_from_file_location('archive_replay', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('name,link', [('../outside', False), ('/absolute', False),
                                     ('innocent', True), ('training/.venv/bin/python', False)])
def test_rejects_unsafe_archive_before_extraction(tmp_path, name, link):
    archive = tmp_path/'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        good = tarfile.TarInfo('ordinary.txt')
        good.size = 1
        handle.addfile(good, io.BytesIO(b'x'))
        bad = tarfile.TarInfo(name)
        if link:
            bad.type, bad.linkname = tarfile.SYMTYPE, '../outside'
        handle.addfile(bad)
    destination = tmp_path/'unpacked'
    destination.mkdir()
    with pytest.raises(ValueError, match='archive member'):
        MODULE.extract_archive(archive, destination)
    assert list(destination.iterdir()) == []


def test_hash_requires_exact_file_set(tmp_path):
    package = tmp_path/'tsensei_v3_2_pianist'
    package.mkdir()
    source = package/'one.py'
    source.write_bytes(b'original')
    expected = {'tsensei_v3_2_pianist/one.py': hashlib.sha256(b'original').hexdigest()}
    MODULE.validate_sources(tmp_path, expected)
    (package/'extra.py').write_text('extra')
    with pytest.raises(ValueError, match='source differs'):
        MODULE.validate_sources(tmp_path, expected)


def test_action_contract_is_checked_for_new_archives(tmp_path):
    package = tmp_path/'tsensei_v3_2_pianist'
    package.mkdir()
    (package/'one.py').write_bytes(b'original')
    actions = tmp_path/'configs/actions'
    actions.mkdir(parents=True)
    contract = actions/'direct_joint_v1.json'
    contract.write_bytes(b'{"physical_max":1.3}')
    expected = {
        'tsensei_v3_2_pianist/one.py': hashlib.sha256(b'original').hexdigest(),
        'configs/actions/direct_joint_v1.json': hashlib.sha256(contract.read_bytes()).hexdigest(),
    }
    MODULE.validate_sources(tmp_path, expected)
    contract.write_bytes(b'{"physical_max":1.2}')
    with pytest.raises(ValueError, match='direct_joint_v1.json'):
        MODULE.validate_sources(tmp_path, expected)


def test_render_rejects_missing_archive_preview_script(tmp_path):
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/eval_from_source.py').write_text('')
    MODULE.validate_scripts(tmp_path, 'evaluate')
    with pytest.raises(ValueError, match='current-source scripts are not substituted'):
        MODULE.validate_scripts(tmp_path, 'render')
