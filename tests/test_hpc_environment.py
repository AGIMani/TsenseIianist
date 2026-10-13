"""CPU checks for refusing accidental login-node GPU use and venv overwrite."""

import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/hpc/probe_gpus.py"
SETUP = ROOT / "scripts/hpc/setup_envs.sh"
CUDA_SETUP = ROOT / "scripts/hpc/setup_cuda.sh"
EGL_SETUP = ROOT / "scripts/hpc/setup_egl.sh"
allocation_environment = runpy.run_path(str(PROBE))["allocation_environment"]


@pytest.mark.parametrize("args,expected", [([], 2), (["--expected-gpus", "1"], 1),
                                         (["--expected-gpus", "2"], 2)])
def test_probe_expected_gpu_count_without_loading_cuda(tmp_path, monkeypatch, args, expected):
    main = runpy.run_path(str(PROBE))["main"]
    namespace = main.__globals__
    observed = []
    monkeypatch.setitem(namespace, "allocation_environment", lambda env: {"SLURM_JOB_ID": "123"})
    monkeypatch.setitem(namespace, "package_versions", lambda: {})
    monkeypatch.setitem(namespace, "command_info", lambda argv: {})
    def fake_torch_checks(report, count):
        observed.append(count)
        return object()
    monkeypatch.setitem(namespace, "torch_checks", fake_torch_checks)
    monkeypatch.setitem(namespace, "peer_checks", lambda *unused: None)
    monkeypatch.setitem(namespace, "warp_checks", lambda *unused: None)
    report = tmp_path/'probe.json'
    assert main(["--output", str(report), *args]) == 0
    assert observed == [expected]
    data = json.loads(report.read_text())
    assert data["expected_gpus"] == expected
    assert data["status"] == "passed_basic_gpu_checks"


def test_single_gpu_peer_probe_performs_no_transfers():
    peer_checks = runpy.run_path(str(PROBE))["peer_checks"]
    # Any attempted CUDA operation besides device_count would fail on this stub.
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
    report = {}
    peer_checks(torch, report, transfer_mib=16, repeats=20)
    assert report["peer_copies"] == []


@pytest.mark.parametrize("count", ["0", "-1"])
def test_probe_rejects_non_positive_expected_gpu_count(tmp_path, count):
    main = runpy.run_path(str(PROBE))["main"]
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path/'probe.json'), "--expected-gpus", count])
    assert exc.value.code == 2
    assert not (tmp_path/'probe.json').exists()


@pytest.mark.parametrize("environ", [
    {},
    {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1"},
    {"SLURM_JOB_ID": "123", "SLURM_GPUS_ON_NODE": "0", "CUDA_VISIBLE_DEVICES": "0,1"},
    {"SLURM_JOB_ID": "123", "SLURM_GPUS_ON_NODE": "2"},
    {"SLURM_JOB_ID": "123", "SLURM_GPUS_ON_NODE": "2", "CUDA_VISIBLE_DEVICES": "-1"},
])
def test_probe_requires_an_allocated_visible_gpu(environ):
    with pytest.raises(RuntimeError):
        allocation_environment(environ)


@pytest.mark.parametrize("gpu_environment", [
    {"SLURM_JOB_GPUS": "0"},
    {"SLURM_STEP_GPUS": "0,1"},
    {"SLURM_GPUS_ON_NODE": "2"},
])
def test_allocation_identifiers_are_preserved(gpu_environment):
    environ = {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1", **gpu_environment}
    result = allocation_environment(environ)
    for name, value in environ.items():
        assert result[name] == value


def test_probe_failure_report_does_not_import_gpu_libraries(tmp_path):
    report = tmp_path / "probe.json"
    environ = {name: value for name, value in os.environ.items()
               if not name.startswith("SLURM_") and name != "CUDA_VISIBLE_DEVICES"}
    # -S removes site-packages: the allocation rejection must work without Torch/Warp.
    result = subprocess.run([sys.executable, "-S", str(PROBE), "--output", str(report)],
                            env=environ, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data["status"] == "failed"
    assert data["stage"] == "allocation"
    assert "Slurm job allocation is required" in data["error"]
    assert "torch" not in data


def copy_setup(tmp_path):
    path = tmp_path / "scripts/hpc/setup_envs.sh"
    path.parent.mkdir(parents=True)
    shutil.copyfile(SETUP, path)
    return path


def test_setup_plan_does_not_create_environments(tmp_path):
    setup = copy_setup(tmp_path)
    result = subprocess.run(["bash", str(setup), "--print-plan"], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Python: 3.12" in result.stdout
    assert not (tmp_path / "training").exists()
    assert not (tmp_path / "training_mjlab").exists()


def test_setup_refuses_existing_environment_before_installing(tmp_path):
    setup = copy_setup(tmp_path)
    existing = tmp_path / "training/.venv"
    existing.mkdir(parents=True)
    sentinel = existing / "unrelated.txt"
    sentinel.write_text("keep this environment")
    result = subprocess.run(["bash", str(setup)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Existing unmanaged environment" in result.stderr
    assert sentinel.read_text() == "keep this environment"
    assert not (tmp_path / "training_mjlab").exists()


def test_setup_refuses_environment_symlink(tmp_path):
    setup = copy_setup(tmp_path)
    target = tmp_path / "unrelated"
    target.mkdir()
    (tmp_path / "training").mkdir()
    (tmp_path / "training/.venv").symlink_to(target, target_is_directory=True)
    result = subprocess.run(["bash", str(setup)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "Unsafe environment path" in result.stderr
    assert list(target.iterdir()) == []


def copy_cuda_setup(tmp_path):
    script = tmp_path / "scripts/hpc/setup_cuda.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(CUDA_SETUP, script)
    python = tmp_path / "training_mjlab/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    return script


@pytest.mark.parametrize("with_compat", [False, True])
def test_private_cuda_plan_is_pinned_and_does_not_install(tmp_path, with_compat):
    script = copy_cuda_setup(tmp_path)
    result = subprocess.run(["bash", str(script), "--print-plan", *(["--with-compat"] if with_compat else [])],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan["sources"]) == 3
    assert all(len(source["sha256"]) == 64 and source["url"].startswith(
        "https://developer.download.nvidia.com/") for source in plan["sources"])
    assert (plan["compat"] is not None) == with_compat
    assert not (tmp_path / "training_mjlab/hpc").exists()


def test_private_cuda_refuses_unknown_installation(tmp_path):
    script = copy_cuda_setup(tmp_path)
    existing = tmp_path / "training_mjlab/hpc/cuda-12.8.1"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("do not modify")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "without our ownership marker" in result.stderr
    assert sentinel.read_text() == "do not modify"
    assert not (tmp_path / "training_mjlab/hpc/cuda-downloads").exists()


def test_egl_plan_does_not_install(tmp_path):
    script = tmp_path / "scripts/hpc/setup_egl.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(EGL_SETUP, script)
    result = subprocess.run(["bash", str(script), "--print-plan"], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "https://archive.ubuntu.com/" in result.stdout
    assert "libegl1_1.4.0-1_amd64.deb" in result.stdout
    assert not (tmp_path / "training_mjlab").exists()


def test_egl_refuses_existing_unmanaged_directory(tmp_path):
    script = tmp_path / "scripts/hpc/setup_egl.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(EGL_SETUP, script)
    existing = tmp_path / "training_mjlab/hpc/egl-jammy-1.4.0-1"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep the old loader")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Refusing unmanaged EGL directory" in result.stderr
    assert sentinel.read_text() == "keep the old loader"
    assert not (tmp_path / "training_mjlab/hpc/egl-downloads").exists()
