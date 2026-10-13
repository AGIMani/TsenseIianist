"""Migration archives must exclude outputs and preserve reproducible source bytes."""
import importlib.util
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_a32", ROOT / "scripts/hpc/package_a32.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def fixture_source(root):
    names = (*pack.ESSENTIAL_FILES,
             "configs/preview/m6_sdk_joint_limits.json",
             "tsensei_v3_2_pianist/assets/meshes_left/base_link.STL",
             "tsensei_v3_2_pianist/assets/meshes_right/base_link.STL",
             "scripts/hpc/launch.sbatch", "docs/a32_migration_bundle_20260928.md")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + name + "\n")


def test_selection_excludes_old_models_credentials_and_outputs(tmp_path):
    root = tmp_path / "source"
    fixture_source(root)
    excluded = (
        "tsensei_v3_2_pianist/TsenseI_V3.2_hand_withid_fixed.zip",
        "tsensei_v3_2_pianist/assets/tj_m6/left/arm.urdf",
        "tsensei_v3_2_pianist/__pycache__/module.pyc",
        "scripts/hpc/.env.json", "scripts/hpc/secrets.json",
        "scripts/hpc/swanlab_api_key.txt", "docs/chat_answers_20260924.md",
        "configs/archive/old.json", "docs/references/paper.md",
        "training_mjlab/outputs/run/config.json",
        "training_mjlab/.venv/site-packages/example.py",
        "training_mjlab/source_versions/old.tar.gz",
        "tests/test_preview_arm_ik.py", "tests/test_m6_gpu.py",
    )
    for name in excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must remain local")
    files = pack.collect_files(root)
    assert not set(excluded) & files.keys()
    assert "scripts/hpc/launch.sbatch" in files
    assert "docs/a32_migration_bundle_20260928.md" in files
    assert "configs/preview/m6_sdk_joint_limits.json" in files


def test_archive_reproduces_bytes_and_audits_sha(tmp_path):
    root = tmp_path / "source"
    fixture_source(root)
    first, second = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    a, b = pack.package(root, first), pack.package(root, second)
    assert first.read_bytes() == second.read_bytes()
    assert a["sha256"] == b["sha256"]
    manifest = pack.audit_archive(first)
    assert manifest["includes_training_artifacts"] is False
    with tarfile.open(first) as bundle:
        assert len(bundle.getmembers()) == manifest["file_count"] + 1
        assert all(member.isfile() for member in bundle)
    with pytest.raises(FileExistsError):
        pack.package(root, first)


def test_reject_symlink_and_incomplete_source(tmp_path):
    root = tmp_path / "source"
    fixture_source(root)
    link = root / "scripts/hpc/escaped.py"
    link.symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="non-regular"):
        pack.collect_files(root)
    link.unlink()
    (root / "run.sh").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        pack.collect_files(root)


def test_audit_rejects_changed_content_and_unsafe_members(tmp_path):
    root = tmp_path / "source"
    fixture_source(root)
    files = pack.collect_files(root)
    manifest = pack.build_manifest(files)
    altered = dict(files, **{"run.sh": b"modified after checksum"})
    corrupt = tmp_path / "corrupt.tar.gz"
    pack.write_archive(corrupt, altered, manifest)
    with pytest.raises(ValueError, match="differs from its manifest"):
        pack.audit_archive(corrupt)
    unsafe = tmp_path / "unsafe.tar.gz"
    pack.write_archive(unsafe, {**files, "../outside.py": b"escape"}, manifest)
    with pytest.raises(ValueError, match="Unexpected archive member"):
        pack.audit_archive(unsafe)


def test_reject_source_change_during_packaging(tmp_path, monkeypatch):
    root = tmp_path / "source"
    fixture_source(root)
    original = pack.write_archive

    def change_source(*args):
        original(*args)
        (root / "run.sh").write_text("concurrent edit")

    monkeypatch.setattr(pack, "write_archive", change_source)
    output = tmp_path / "changing.tar.gz"
    with pytest.raises(RuntimeError, match="Sources changed"):
        pack.package(root, output)
    assert not output.exists()


def test_optional_preview_assets_are_explicit_and_audited(tmp_path):
    root = tmp_path/'source'
    fixture_source(root)
    names = ('preview/assets/tj_m6/left/arm.urdf', 'preview/assets/tj_m6/LICENSE',
             'preview/assets/hand_connector/left_01.stl')
    for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    assert not set(names) & pack.collect_files(root).keys()
    assert set(names) <= pack.collect_files(root, include_preview_assets=True).keys()
    output = tmp_path/'preview.tar.gz'
    pack.package(root, output, include_preview_assets=True)
    assert pack.audit_archive(output)['includes_m6_assets'] is True
    assert not pack.preview_asset('preview/assets/tj_m6_extra/arm.urdf')
    assert not pack.preview_asset('preview/assets/tj_m6/secrets/key.json')


def test_online_export_uses_exact_files_without_traversing_web_runtime(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    fixture_source(root)
    rejected = ('online_control/app.py', 'online_control/backends.py', 'online_control/vendor/sdk.py',
                'online_control/runtime/experiment.py', 'online_control/.browser-libs/library.py',
                'online_control/static/app.js', 'online_control/config/preview_pose.json',
                'online_control/tests/test_device.py')
    for name in (*pack.OFFLINE_EXPORT_FILES, *rejected):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture ' + name)
    walk = pack.os.walk
    def no_web_walk(directory, *args, **kwargs):
        assert 'online_control' not in Path(directory).parts, 'online_control must not be recursively scanned'
        return walk(directory, *args, **kwargs)
    monkeypatch.setattr(pack.os, 'walk', no_web_walk)
    files = pack.collect_files(root)
    assert set(pack.OFFLINE_EXPORT_FILES) <= files.keys()
    assert not set(rejected) & files.keys()
    assert {name for name in files if name.startswith('online_control/')} == set(pack.OFFLINE_EXPORT_FILES)
