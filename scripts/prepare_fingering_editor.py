#!/usr/bin/env python3
"""Prepare an isolated, SHA-verified remote CPU preview deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from fingering_editor_config import connection_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-base", default="/home/houj/projects/pianist_model_replace_20261020",
                        help="Verified deployment providing the pinned archive and dependency environments")
    parser.add_argument('--output', type=Path, default=connection_config(ROOT, 'remote', migrate=False))
    args = parser.parse_args()
    archive = ROOT / "logs/model_replace_20261020/source_supplied_hand_v6.tar.gz"
    expected = "9299ab01d69563092b65cd4d91b09429ae99aa501cedbb797b9956bd098fece8"
    if archive.is_file():
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError("Preview requires the verified supplied_hand_v6 archive")
    else:
        # The editor may be opened from a clean checkout without ignored logs.
        # Reuse only the exact remotely pinned preview archive, whose contents
        # are checked below; never substitute the remote working-tree source.
        digest = expected
    worker = ROOT / "scripts/fingering_pose_worker.py"
    worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    remote = f"/home/houj/projects/pianist_fingering_editor_{digest[:8]}_{worker_sha[:8]}"
    logdir = ROOT / "logs/fingering_editor"
    logdir.mkdir(parents=True, exist_ok=True)
    payload = dict(base=args.remote_base, remote=remote, archive_sha=digest, worker_sha=worker_sha)
    code = """
import hashlib,json,os,pathlib,shutil,sys,tarfile
p=json.loads(sys.argv[1]); base=pathlib.Path(p['base']); dest=pathlib.Path(p['remote'])
archive=base/'logs/model_replace_20261020/source_supplied_hand_v6.tar.gz'
assert archive.is_file(), archive
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==p['archive_sha']
available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
assert available>8*1024*1024, 'Remote requires 8 GiB available memory'
assert shutil.disk_usage(dest.parent).free>3*1024**3, 'Remote requires 3 GiB free disk'
for role in ['training','training_mjlab']:
    assert (base/role/'.venv/bin/python').is_file(), role
if not dest.exists():
    dest.mkdir()
    with tarfile.open(archive) as tf: tf.extractall(dest,filter='data')
    (dest/'EDITOR_SOURCE_SHA256').write_text(p['archive_sha']+'\\n')
else:
    assert (dest/'EDITOR_SOURCE_SHA256').read_text().strip()==p['archive_sha']
for role in ['training','training_mjlab']:
    (dest/role).mkdir(exist_ok=True)
    env=dest/role/'.venv'
    if not env.exists(): env.symlink_to((base/role/'.venv').resolve(),target_is_directory=True)
print(json.dumps(dict(host=os.uname().nodename,remote=str(dest),available_kib=available,archive_sha=p['archive_sha'])))
"""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "volc-a32",
               shlex.quote(args.remote_base + "/training_mjlab/.venv/bin/python") +
               " -c " + shlex.quote(code) + " " + shlex.quote(json.dumps(payload))]
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    (logdir / "deployment.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Remote preflight/deployment failed:\n{result.stderr}\nSee {logdir / 'deployment.log'}")
    # New worker content has its own deployment directory. Never overwrite an
    # existing experiment or another version of the editor.
    subprocess.run(["rsync", "-a", str(worker), f"volc-a32:{remote}/scripts/"], check=True)
    verify = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "volc-a32",
                                     "sha256sum " + shlex.quote(remote + "/scripts/" + worker.name)], text=True)
    if verify.split()[0] != worker_sha:
        raise ValueError("Remote worker SHA mismatch")
    config = dict(host="volc-a32", remote_root=remote, source_sha256=digest, worker_sha256=worker_sha,
                  python=f"{remote}/training_mjlab/.venv/bin/python",
                  created_at=time.time(), preflight=json.loads(result.stdout),
                  command=command, exit_code=result.returncode, gpu=False)
    config['local_archive_present'] = archive.is_file()
    config['preview_scope'] = 'Frozen supplied_hand_v6 editor; not the current training runtime'
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
