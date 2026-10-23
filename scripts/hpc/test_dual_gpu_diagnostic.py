"""Separate trajectory repeatability from fixed-input coupling graph math."""
import json
import os
from pathlib import Path
import sys


def test_fixed_input_graph_and_baseline_repeat():
    import pytest
    if "DUAL_GPU_DIAG_DIR" not in os.environ:
        pytest.skip("Opt-in Slurm graph diagnosis")
    assert os.environ.get("SLURM_JOB_ID")
    sys.path.insert(0, str(Path.cwd()))
    import torch
    from tsensei_v3_2_pianist.mjlab_env.preview_contract import build_task_from_config

    output = Path(os.environ["DUAL_GPU_DIAG_DIR"])
    reference_root = output.parent / "dual_gpu_20261020"
    config = json.loads((reference_root / "config.json").read_text())
    config["acceleration"]["physics_graph"] = False
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = torch.load(reference_root / "reference_rollout.pt", weights_only=True)
    env, *_ = build_task_from_config(config, device="cuda:0", num_envs=512)
    env.reset()
    rows, coupling = [], []
    with torch.no_grad(), torch.cuda.device(0):
        for step in range(128):
            action = reference["action"][step].to("cuda:0")
            if step in (0, 1, 2, 10, 30, 60, 100, 127):
                scene = env.scene
                with scene.scope():
                    physical = scene.physical_action(action)
                    scene.physics_graph = False
                    a, ai = scene._couple_finger_targets(physical)
                    scene.physics_graph = True
                    b, bi = scene._couple_finger_targets(physical)
                    scene.physics_graph = False
                    row = dict(step=step, target_max_abs=(a-b).abs().max().item(),
                               info_equal={k: bool(torch.equal(ai[k], bi[k])) for k in ai})
                coupling.append(row)
            result = env.step(action)
            values = dict(qpos=env.scene.data.qpos[:].cpu(),
                          qvel=env.scene.data.qvel[:].cpu(),
                          reward=result.reward.cpu())
            errors = {key: (value-reference[key][step]).abs().max().item()
                      for key, value in values.items()}
            rows.append(dict(step=step, **errors))
    result = dict(job_id=os.environ["SLURM_JOB_ID"],
                  gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  scope="Same-action baseline repeat for 128 control steps; graph math compared on identical live inputs",
                  first_nonzero=next((r for r in rows if any(r[k] != 0
                                                           for k in ("qpos", "qvel", "reward"))), None),
                  max_errors={k: max(r[k] for r in rows) for k in ("qpos", "qvel", "reward")},
                  fixed_input_coupling=coupling, trajectory_errors=rows)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory_errors"}), flush=True)
    # Independent chaotic trajectories are diagnostic, not a graph-math oracle.
    assert all(row["target_max_abs"] == 0 and all(row["info_equal"].values())
               for row in coupling), "Identical-input coupling graph arithmetic differs"
