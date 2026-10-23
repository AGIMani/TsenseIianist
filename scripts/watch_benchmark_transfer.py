"""Poll an isolated Slurm benchmark and return its artifacts with SHA checks."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()
    root = args.submission.resolve().parent
    lock = (root / "transfer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    submission = json.loads(args.submission.read_text())
    host, job = submission["host"], str(submission["job_id"])
    remote = submission["remote_project"]
    output = submission["remote_output"]
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
    while True:
        try:
            query = subprocess.check_output(
                ["ssh", "-o", "ConnectTimeout=15", host,
                 f"sacct -X -n -P -j {shlex.quote(job)} --format=JobIDRaw,State,ExitCode"],
                text=True, timeout=45).strip()
            row = next((x.split("|") for x in query.splitlines()
                        if x.split("|")[0] == job), None)
            status = dict(checked_at=datetime.now().astimezone().isoformat(),
                          job_id=job, state=row[1] if row else "UNKNOWN",
                          exit_code=row[2] if row else None)
            (root / "sync_status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(json.dumps(status), flush=True)
            if row and row[1].split()[0].rstrip("+") in terminal:
                # Generate the manifest on the remote host; never call local
                # hashes alone a verification of successful transmission.
                code = (
                    "import hashlib,json,pathlib;"
                    f"r=pathlib.Path({output!r});"
                    "files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() "
                    "for p in r.iterdir() if p.is_file() and p.name!='manifest.json'};"
                    "(r/'manifest.json').write_text(json.dumps(files,indent=2)+'\\n')")
                subprocess.run(["ssh", host, "python3 -c " + shlex.quote(code)],
                               check=True, timeout=120)
                destination = root / "returned"
                destination.mkdir(exist_ok=True)
                subprocess.run(["rsync", "-az", host + ":" + output + "/",
                                str(destination) + "/"], check=True, timeout=1800)
                manifest = json.loads((destination / "manifest.json").read_text())
                for name, expected in manifest.items():
                    digest = digest_file(destination / name)
                    if digest != expected:
                        raise ValueError("Returned SHA mismatch: " + name)
                for suffix in (".out", ".err"):
                    subprocess.run(["rsync", "-a",
                                    f"{host}:{remote}/logs/hpc/{job}{suffix}",
                                    str(root / ("slurm" + suffix))], check=True, timeout=120)
                status.update(transfer_verified=True, files_verified=len(manifest),
                              benchmark_passed=row[1] == "COMPLETED" and row[2] == "0:0",
                              report_present=(destination / "report.json").exists())
                (root / "sync_status.json").write_text(json.dumps(status, indent=2) + "\n")
                print(json.dumps(status), flush=True)
                return
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print("Will retry: " + repr(error), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
