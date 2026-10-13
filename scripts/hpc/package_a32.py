#!/usr/bin/env python3
"""Build and audit a reproducible, standalone TsenseI_A3.2 source archive.

Only source, current hand assets, score inputs, tests and maintenance documents
are selected. No virtual environment, old hand archive or training output is
copied. This script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
SCHEMA = "tsensei_a32_source_bundle_v1"
TOP_LEVEL = (
    "AGENTS.md", "README.md", "LICENSE.robopianist", "LICENSE.robopianist-rl",
    "pyproject.toml", "requirements-tested.txt", "requirements-sac-tested.txt",
    "eval.sh", "export_video.sh", "smoke_env.py", "preview_pose.py",
)
TREE_SUFFIXES = {
    "tsensei_v3_2_pianist": {
        ".py", ".json", ".urdf", ".stl", ".obj", ".mid", ".proto", ".npz", ".md",
    },
    "scripts": {".py", ".sh", ".json", ".toml", ".txt", ".md", ".sbatch", ".cmd"},
    "configs": {".json", ".md"},
    "tests": {".py", ".json", ".npz"},
    "preview/scripts": {".py", ".sh"},
    "docs": {".md", ".json", ".csv"},
    "training_mjlab/inputs": {".mid", ".proto", ".csv", ".json", ".md"},
}
DEPENDENCY_FILES = (
    "training_mjlab/setup.sh", "training_mjlab/requirements-tested.txt",
    "training_mjlab/monitoring/requirements.txt",
    "preview/requirements-replay.txt",
)
# Select exact offline entry files. Never traverse the web application's SDK,
# browser runtime, recordings or environment when building a training archive.
OFFLINE_EXPORT_FILES = (
    "online_control/__init__.py", "online_control/export.py",
    "online_control/models.py", "online_control/trajectory.py",
)
EXCLUDED_COMPONENTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", "outputs", "runs", "source_versions",
    "model_backups", "archive", "references", "tj_m6", "external_never_copy_only_reference",
    ".ssh", ".aws", ".config", "node_modules", "credentials", "secrets",
}
EXCLUDED_FILES = {
    "tests/test_preview_arm_ik.py", "tests/test_m6_gpu.py",
    # Conversation exports can contain account details and are not run documentation.
    "docs/chat_answers_20260924.md",
}
ESSENTIAL_FILES = (
    "eval.sh", "pyproject.toml", "AGENTS.md",
    "scripts/train_from_source.py", "scripts/eval_from_source.py",
    "scripts/swanlab_upload.py", "scripts/swanlab_metrics.py",
    "tsensei_v3_2_pianist/mjlab_env/source_runtime.py",
    "tsensei_v3_2_pianist/rl_torch/train.py",
    "tsensei_v3_2_pianist/assets/source/left_original.urdf",
    "tsensei_v3_2_pianist/assets/urdf/TsenseI_left.urdf",
    "tsensei_v3_2_pianist/assets/urdf/TsenseI_right.urdf",
    "tsensei_v3_2_pianist/assets/model_derivation.json",
    "tsensei_v3_2_pianist/assets/contact_settings.json",
    "tsensei_v3_2_pianist/assets/collision_proxies.json",
    "tsensei_v3_2_pianist/assets/drive_settings.json",
    "tsensei_v3_2_pianist/music/data/rousseau/twinkle-twinkle-trimmed.mid",
    "configs/actions/direct_joint_v1.json",
    "configs/control_limits/generic_wrist_rate_20260927.json",
    "configs/finger_clearance/control_disabled_20260927.json",
    "configs/rewards/default_20261006.json",
    "requirements-tested.txt", "training_mjlab/requirements-tested.txt",
)


def excluded(relative: str) -> bool:
    path = PurePosixPath(relative)
    lower = path.name.lower()
    return (
        bool(set(path.parts) & EXCLUDED_COMPONENTS)
        or any(part.endswith(".egg-info") for part in path.parts)
        or relative in EXCLUDED_FILES
        or lower.startswith((".env", "id_rsa", "id_ed25519", "id_ecdsa", "credentials", "secrets"))
        or lower.endswith((".key", ".pem", ".p12", ".pfx", ".pub"))
        or "api_key" in lower or "apikey" in lower
    )


def _candidate_paths(root: Path):
    for name in (*TOP_LEVEL, *DEPENDENCY_FILES, *OFFLINE_EXPORT_FILES):
        path = root / name
        if path.exists() or path.is_symlink():
            yield path
    for name, suffixes in TREE_SUFFIXES.items():
        base = root / name
        if base.is_symlink():
            raise ValueError(f"Refusing source directory symlink: {name}")
        if not base.is_dir():
            continue
        for directory, children, filenames in os.walk(base, followlinks=False):
            kept = []
            for child in sorted(children):
                path = Path(directory) / child
                relative = path.relative_to(root).as_posix()
                if excluded(relative):
                    continue
                if path.is_symlink():
                    raise ValueError(f"Refusing source directory symlink: {relative}")
                kept.append(child)
            children[:] = kept
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix.lower() in suffixes:
                    yield path


def preview_asset(relative: str) -> bool:
    path = PurePosixPath(relative)
    return (path.parts[:3] in {('preview', 'assets', 'tj_m6'), ('preview', 'assets', 'hand_connector')}
            and not (set(path.parts[3:]) & EXCLUDED_COMPONENTS)
            and (path.suffix.lower() in {'.json', '.stl', '.urdf'} or path.name == 'LICENSE'))


def collect_files(root: Path, *, include_preview_assets=False) -> dict[str, bytes]:
    """Snapshot selected regular files in memory; never follow a source symlink."""
    root = root.resolve()
    result = {}
    for path in _candidate_paths(root):
        relative = path.relative_to(root).as_posix()
        if excluded(relative):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Refusing non-regular source file: {relative}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source file escapes project: {relative}")
        result[relative] = path.read_bytes()
    if include_preview_assets:
        for directory in ('preview/assets/tj_m6', 'preview/assets/hand_connector'):
            base = root / directory
            if not base.is_dir() or base.is_symlink():
                raise ValueError('Missing optional preview assets: ' + directory)
            for path in sorted(base.rglob('*')):
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    raise ValueError('Refusing preview asset symlink: ' + relative)
                if path.is_file() and preview_asset(relative):
                    result[relative] = path.read_bytes()
    missing = sorted(set(ESSENTIAL_FILES) - result.keys())
    if missing:
        raise ValueError("Incomplete A3.2 source tree: " + ", ".join(missing))
    for side in ("left", "right"):
        prefix = f"tsensei_v3_2_pianist/assets/meshes_{side}/"
        if not any(name.startswith(prefix) for name in result):
            raise ValueError(f"Missing current A3.2 {side} meshes")
    return dict(sorted(result.items()))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def file_mode(name: str) -> int:
    return 0o755 if name.endswith(".sh") else 0o644


def build_manifest(files: dict[str, bytes]) -> dict:
    return {
        "schema": SCHEMA,
        "model": "TsenseI_A3.2",
        "entrypoint": "scripts/train_from_source.py (Slurm internal; submit with all_use_this_train.sh)",
        "includes_training_artifacts": False,
        "includes_m6_assets": any(name.startswith('preview/assets/tj_m6/') for name in files),
        "file_count": len(files),
        "total_file_bytes": sum(map(len, files.values())),
        "files": {
            name: {"sha256": sha256(data), "size": len(data), "mode": file_mode(name)}
            for name, data in files.items()
        },
        "exclusion_policy": {
            "directory_components": sorted(EXCLUDED_COMPONENTS),
            "specific_files": sorted(EXCLUDED_FILES),
            "selection": "Only the top-level allowlist and allowed extensions in package_a32.py",
            "credentials": "Exclude credential filenames; never include home directories or runtime environments",
            "old_assets": ("Optional local M6 and connector preview meshes only; no historical archives"
                           if any(preview_asset(name) for name in files)
                           else "No historical compressed archives or M6 arm assets"),
        },
    }


def write_archive(path: Path, files: dict[str, bytes], manifest: dict) -> None:
    contents = {**files, MANIFEST_NAME: canonical_json(manifest)}
    # Fixed gzip header, metadata, permissions and order make content reproducible
    # across paths and source checkout mtimes. Exclusive create avoids replacement.
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                for name, data in sorted(contents.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = file_mode(name)
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    bundle.addfile(info, io.BytesIO(data))


def audit_archive(path: Path) -> dict:
    """Verify exact archive membership and hashes without extracting anything."""
    contents = {}
    with tarfile.open(path, "r:gz") as bundle:
        for member in bundle:
            name = member.name
            pure = PurePosixPath(name)
            if (not member.isfile() or pure.is_absolute() or ".." in pure.parts
                    or name in contents or name != pure.as_posix()
                    or (name != MANIFEST_NAME and excluded(name) and not preview_asset(name))):
                raise ValueError(f"Unexpected archive member: {name}")
            if (member.mode != file_mode(name) or member.uid or member.gid or member.mtime
                    or member.uname or member.gname):
                raise ValueError(f"Non-reproducible archive metadata: {name}")
            contents[name] = bundle.extractfile(member).read()
    if MANIFEST_NAME not in contents:
        raise ValueError("Archive has no bundle manifest")
    manifest = json.loads(contents.pop(MANIFEST_NAME))
    if manifest.get("schema") != SCHEMA or manifest != build_manifest(dict(sorted(contents.items()))):
        raise ValueError("Archive content differs from its manifest")
    missing = sorted(set(ESSENTIAL_FILES) - contents.keys())
    if missing:
        raise ValueError("Archive is missing runtime files: " + ", ".join(missing))
    return manifest


def package(root: Path, output: Path, *, include_preview_assets=False) -> dict:
    output = output.resolve()
    if not output.name.endswith(".tar.gz"):
        raise ValueError("--output must end in .tar.gz")
    manifest_path = output.with_name(output.name + ".manifest.json")
    checksum_path = output.with_name(output.name + ".sha256")
    if any(path.exists() or path.is_symlink() for path in (output, manifest_path, checksum_path)):
        raise FileExistsError("Output already exists; choose a fresh archive name")
    files = collect_files(root, include_preview_assets=include_preview_assets)
    manifest = build_manifest(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_archive(output, files, manifest)
    # Reject a mixed snapshot if another editor changed any selected input.
    if collect_files(root, include_preview_assets=include_preview_assets) != files:
        output.unlink()
        raise RuntimeError("Sources changed during packaging; finish edits and retry")
    audit_archive(output)
    archive_sha = sha256(output.read_bytes())
    external = dict(manifest, archive_sha256=archive_sha, archive_bytes=output.stat().st_size)
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json(external))
    with checksum_path.open("x") as handle:
        handle.write(f"{archive_sha}  {output.name}\n")
        handle.write(f"{sha256(manifest_path.read_bytes())}  {manifest_path.name}\n")
    return {
        "archive": str(output), "sha256": archive_sha,
        "manifest": str(manifest_path), "checksum_file": str(checksum_path),
        "file_count": manifest["file_count"], "source_bytes": manifest["total_file_bytes"],
        "archive_bytes": output.stat().st_size, "audit": "passed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Standalone project root")
    parser.add_argument('--include-preview-assets', action='store_true', help='Bundle optional M6 and connector meshes for preview jobs')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="New .tar.gz path; existing outputs are refused")
    action.add_argument("--audit", type=Path, help="Verify an existing archive without extracting it")
    args = parser.parse_args(argv)
    try:
        if args.audit:
            manifest = audit_archive(args.audit)
            result = dict(archive=str(args.audit.resolve()), audit="passed",
                          file_count=manifest["file_count"], sha256=sha256(args.audit.read_bytes()))
        else:
            result = package(args.root, args.output, include_preview_assets=args.include_preview_assets)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
        parser.exit(1, f"A3.2 bundle: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
