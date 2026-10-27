"""Remote array/file checks of an exchange-fingering build; no model import."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not os.environ.get("SSH_CONNECTION"):
        raise RuntimeError("Run remotely on volc-a32")
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    folder, plan, reference = args.build_output, args.plan, args.reference
    diagnostic = json.loads((folder / "model_diagnostic.json").read_text())
    recipe = json.loads((folder / "source_config.json").read_text())
    baseline = json.loads(Path("task_outputs/controller_check4/source_config.json").read_text())
    planning = json.loads((plan / "report.json").read_text())
    metadata = json.loads(reference.with_suffix(".json").read_text())
    process = json.loads((folder / "process.json").read_text())
    assert recipe["source_sha256"] == baseline["source_sha256"]
    assert recipe["environment"]["physics_hz"] == 1000
    assert recipe["environment"]["control_hz"] == 20
    assert sha(folder / "build/score.proto") == sha(plan / "turkish-march-advanced_fingered.proto") == metadata["score_sha256"] == diagnostic["score_sha256"]
    assert diagnostic["archive_sha256"] == "1d4a67f15c796a849387ccc967c4960f4ee12d0b18bde2ff81f27a591c23d560"
    assert diagnostic["fitted_resistance_joints"] == 40
    assert diagnostic["disabled_wrist_axes"] == ["TZ", "Yaw"]
    assert diagnostic["coupling_enabled"]
    assert sha(reference) == metadata["reference_sha256"]
    with np.load(reference, allow_pickle=False) as saved:
        shape = saved["actions"].shape
        assert shape == (len(saved["time"]), 51)
        assert np.isfinite(saved["actions"]).all() and np.isfinite(saved["all_commands"]).all()
        assert np.allclose(np.diff(saved["time"]), .05, rtol=0, atol=1e-12)
    original = list(csv.DictReader(Path("task_outputs/fingering_reach_cost200/fingering.csv").open()))
    rows = list(csv.DictReader((plan / "fingering.csv").open()))
    fields = ["note_index", "instrument", "pitch", "start_seconds", "end_seconds", "velocity"]
    assert [[x[k] for k in fields] for x in original] == [[x[k] for k in fields] for x in rows]
    weight = planning["parameters"]["exchange_weight_per_reused_finger_onset"]
    recovery = planning["parameters"]["recovery_time_seconds"]
    last, exchange = {}, [0., 0.]
    for row in sorted(rows, key=lambda x: (float(x["start_seconds"]), int(x["note_index"]))):
        finger, at = int(row["finger_id"]), float(row["start_seconds"])
        if finger in last:
            exchange[finger // 5] += weight * max(0, 1 - (at - last[finger]) / recovery) ** 2
        last[finger] = float(row["end_seconds"])
    for index, side in enumerate(["right", "left"]):
        assert np.isclose(exchange[index], planning["hands"][side]["costs"]["finger_exchange"])
    report = dict(kind="exchange_candidate_bundle_validation_v1", host="volc-a32",
        validation_pid=os.getpid(), validation_script_sha256=sha(__file__), exit_status=0,
        physics_model_loaded=False, physics_steps=0, build_pid=process["pid"],
        build_exit_status=int(folder.with_suffix(".exit").read_text()),
        build=str(folder.resolve() / "build"), score=str((plan / "turkish-march-advanced_fingered.proto").resolve()),
        reference=str(reference.resolve()), mjb_sha256=sha(folder / "model.mjb"),
        dependency_versions={"mujoco": importlib.metadata.version("mujoco")},
        score_sha256=diagnostic["score_sha256"], original_midi_sha256=diagnostic["original_midi_sha256"],
        archive_sha256=diagnostic["archive_sha256"], recipe_sha256=sha(folder / "source_config.json"),
        source_runner_sha256=diagnostic["source_runner_sha256"], model_nq=diagnostic["nq"], model_nu=diagnostic["nu"],
        source_core_matches_original_pinned_build=True, source_file_count=len(recipe["source_sha256"]),
        exact_note_fields_preserved=True, independent_exchange_cost=exchange,
        reference_metadata=metadata, reference_shape=list(shape), reference_is_final_limited_trajectory=False)
    assert report["build_exit_status"] == 0
    output = folder / "bundle_validation.json"
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
