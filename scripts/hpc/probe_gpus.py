#!/usr/bin/env python3
"""Check allocated GPUs without changing dependencies or driver settings.

Run with training_mjlab/.venv/bin/python inside a Slurm job (two GPUs by default).
A successful report covers basic Torch, CUDA graph, peer copies when multiple
GPUs are allocated, and Warp kernel execution.
It does not validate the custom Adam extension, MJLab physics, or training.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


def allocation_environment(environ):
    """Reject login nodes before importing libraries that may initialize CUDA."""
    names = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES",
             "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE",
             "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES")
    result = {name: environ.get(name) for name in names}
    if not str(result["SLURM_JOB_ID"] or "").isdigit():
        raise RuntimeError("A Slurm job allocation is required; do not run GPU probes on the login node.")
    # A JOB_GPUS/STEP_GPUS value of "0" means physical GPU zero, while
    # GPUS_ON_NODE="0" is an empty allocation.
    gpu_ids = result["SLURM_JOB_GPUS"] or result["SLURM_STEP_GPUS"]
    gpu_count = result["SLURM_GPUS_ON_NODE"]
    if not gpu_ids and (not gpu_count or gpu_count == "0"):
        raise RuntimeError("Slurm reports no GPU allocation; request GPUs in sbatch/srun.")
    if result["CUDA_VISIBLE_DEVICES"] in (None, "", "-1", "NoDevFiles"):
        raise RuntimeError("Slurm must set CUDA_VISIBLE_DEVICES to the allocated GPUs.")
    return result


def command_info(argv):
    executable = shutil.which(argv[0])
    if executable is None:
        return {"available": False, "command": argv}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"available": True, "path": executable, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "path": executable, "error": str(exc)}


def package_versions():
    versions = {}
    for name in ("torch", "mujoco", "mujoco-warp", "mjlab", "warp-lang", "numpy", "ninja"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_checks(report, expected_gpus):
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    count = torch.cuda.device_count()
    report["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                       "visible_device_count": count, "devices": []}
    if count != expected_gpus:
        raise RuntimeError(f"Expected exactly {expected_gpus} allocated visible GPUs, found {count}.")
    for index in range(count):
        with torch.cuda.device(index):
            props = torch.cuda.get_device_properties(index)
            device = {"index": index, "name": props.name,
                      "compute_capability": [props.major, props.minor],
                      "total_memory_bytes": props.total_memory}
            report["torch"]["devices"].append(device)
            matrix = torch.full((256, 256), 0.25, device=index)
            product = matrix @ matrix
            torch.cuda.synchronize(index)
            if not torch.equal(product, torch.full_like(product, 16.0)):
                raise RuntimeError(f"Torch matrix multiplication failed on cuda:{index}.")
            device["matmul_passed"] = True

            x = torch.arange(1024, dtype=torch.float32, device=index)
            stream = torch.cuda.Stream(device=index)
            stream.wait_stream(torch.cuda.current_stream(index))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = x * 2.0 + 1.0
            torch.cuda.current_stream(index).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = x * 2.0 + 1.0
            x.add_(1.0)
            graph.replay()
            torch.cuda.synchronize(index)
            if not torch.equal(output, x * 2.0 + 1.0):
                raise RuntimeError(f"CUDA graph replay failed on cuda:{index}.")
            device["cuda_graph_passed"] = True
    return torch


def peer_checks(torch, report, transfer_mib, repeats):
    report["peer_copies"] = []
    count = torch.cuda.device_count()
    for source_index in range(count):
        for target_index in range(count):
            if source_index == target_index:
                continue
            peer = {"source": source_index, "target": target_index,
                    "can_access_peer": torch.cuda.can_device_access_peer(source_index, target_index)}
            report["peer_copies"].append(peer)
            if not peer["can_access_peer"]:
                raise RuntimeError(f"Peer access is unavailable: cuda:{source_index} -> cuda:{target_index}.")
            # Integers modulo 1024 are exactly representable in float32.
            elements = transfer_mib * 1024 * 1024 // 4
            source = torch.arange(elements, dtype=torch.float32, device=source_index).remainder_(1024)
            target = torch.empty(elements, dtype=torch.float32, device=target_index)
            torch.cuda.synchronize(source_index)
            torch.cuda.synchronize(target_index)
            for _ in range(3):
                target.copy_(source, non_blocking=True)
            torch.cuda.synchronize(source_index)
            torch.cuda.synchronize(target_index)
            start = time.perf_counter()
            for _ in range(repeats):
                target.copy_(source, non_blocking=True)
            torch.cuda.synchronize(source_index)
            torch.cuda.synchronize(target_index)
            elapsed = time.perf_counter() - start
            if not torch.equal(source.cpu(), target.cpu()):
                raise RuntimeError(f"Peer copy data mismatch: {source_index} -> {target_index}.")
            byte_count = source.numel() * source.element_size()
            peer.update(data_verified=True, transfer_bytes=byte_count, repeats=repeats,
                        elapsed_seconds=elapsed, gigabytes_per_second=byte_count * repeats / elapsed / 1e9,
                        timing="host wall clock including final synchronization; not peak link bandwidth")


def warp_checks(torch, report):
    # wp must be global so Warp's source compiler can resolve kernel symbols.
    global wp
    import warp as wp

    report["warp"] = {"version": wp.__version__, "devices": []}
    with tempfile.TemporaryDirectory(prefix="tsensei-hpc-warp-probe-") as cache:
        # Force this tiny kernel to compile, without invalidating another job's cache.
        wp.config.kernel_cache_dir = cache
        wp.init()
        report["warp"]["toolkit_version"] = wp.get_cuda_toolkit_version()
        report["warp"]["driver_version"] = wp.get_cuda_driver_version()

        @wp.kernel
        def affine_kernel(values: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
            index = wp.tid()
            output[index] = values[index] * 2.0 + 1.0

        for index in range(torch.cuda.device_count()):
            alias = f"cuda:{index}"
            device = wp.get_device(alias)
            if not device.is_cuda:
                raise RuntimeError(f"Warp did not resolve a CUDA device for {alias}.")
            item = {"device": alias, "name": device.name, "arch": device.arch}
            report["warp"]["devices"].append(item)
            values = torch.arange(4096, dtype=torch.float32, device=index)
            output = torch.empty_like(values)
            torch.cuda.synchronize(index)
            start = time.perf_counter()
            wp.launch(affine_kernel, dim=values.numel(),
                      inputs=[wp.from_torch(values), wp.from_torch(output)], device=device)
            wp.synchronize_device(device)
            elapsed = time.perf_counter() - start
            if not torch.equal(output, values * 2.0 + 1.0):
                raise RuntimeError(f"Warp compiled kernel data mismatch on {alias}.")
            item.update(kernel_passed=True, torch_interop_passed=True, first_launch_seconds=elapsed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report, including errors on failure")
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--transfer-mib", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if args.expected_gpus < 1 or not 1 <= args.transfer_mib <= 1024 or not 1 <= args.repeats <= 1000:
        parser.error("expected-gpus must be >=1, transfer-mib 1..1024, repeats 1..1000")
    report = {"schema": 1, "status": "failed", "stage": "allocation", "hostname": platform.node(),
              "expected_gpus": args.expected_gpus,
              "time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "limits": ["Does not compile or validate the project's Adam extension.",
                         "Does not run MJLab physics, full training, or performance tuning.",
                         "Peer-copy measurements do not establish peak NVLink bandwidth."]}
    try:
        report["slurm"] = allocation_environment(os.environ)
        report["stage"] = "metadata"
        report["packages"] = package_versions()
        report["driver"] = command_info(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"])
        report["topology"] = command_info(["nvidia-smi", "topo", "-m"])
        report["toolchain"] = {name: command_info([name, "--version"]) for name in ("nvcc", "c++", "ninja")}
        report["cuda_home"] = os.environ.get("CUDA_HOME")
        report["compat_enabled_by_job"] = os.environ.get("TSENSEI_CUDA_COMPAT") == "1"
        report["stage"] = "torch_and_graph"
        torch = torch_checks(report, args.expected_gpus)
        report["stage"] = "peer_copies"
        peer_checks(torch, report, args.transfer_mib, args.repeats)
        report["stage"] = "warp_compile_and_launch"
        warp_checks(torch, report)
        report["stage"] = "complete"
        report["status"] = "passed_basic_gpu_checks"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    # nvidia-smi reports the kernel driver even when a private libcuda is loaded.
    # Record actual mapped CUDA driver/JIT library paths to distinguish the two.
    if report["stage"] != "allocation" and Path("/proc/self/maps").exists():
        report["loaded_driver_libraries"] = sorted({
            row.split(maxsplit=5)[-1] for row in Path("/proc/self/maps").read_text().splitlines()
            if any(name in row for name in ("/libcuda.so", "/libnvidia-ptxjitcompiler.so", "/libnvidia-nvvm.so"))
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                     prefix=f".{args.output.name}.", delete=False) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(f"{report['status']}: {args.output.resolve()}", flush=True)
    if report["status"] == "failed":
        print(f"Stage {report['stage']}: {report['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
