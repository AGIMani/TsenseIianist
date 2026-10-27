# Key Hold Lateral Motion Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this authorized implementation.

**Goal:** Add the approved per-hand physical lateral-motion penalty to the default frozen training source.
**Architecture:** A read-only Warp substep collector records per-hand active-key contact and actual palm-origin Y velocity; RewardManager reduces with the approved bounded formula. Existing model/solver/budgets stay unchanged.
**Tech Stack:** Python, Torch, MuJoCo/Warp, Slurm.
**Spec:** docs/superpowers/specs/2026-10-26-key-hold-lateral-design.md

## Global Constraints
- tolerance=.01m/s; full=.10m/s; cap=.5; weight=1; activated at100%; contact dist<=0.
- Actual worldY velocity, per-side contact, max sides then mean substeps, no dt multiplier.
- Source parent SHA63d4224c3a29807a6026d391967297ead55f163f221f8d29cc86f1b116bcb992; overlay only reviewed paths.
- All physics/model/GPU tests remote volc-a32 via scripts/hpc/submit.sh. No training submission.

## Review Focus
- One hand holding while the other moves: no cross-side penalty (Task1/2).
- Rotated parent, rigid arms and body-origin offsets: Jacobian velocity vs MuJoCo (Task2).
- Duplicate contacts and near-but-separated geometries: deduplicate, strict dist gate (Task2).
- Reset/endpoint forward and graph capture: no stale state/double samples (Task2/3).
- Different physical rates and old reward JSON: frequency-invariant score, explicit missing-term rejection (Task1/3).

### Task1: Formula and strict config
Files: reward_config.py, rewards.py, configs/rewards/default_20261006.json; tests/test_key_hold_lateral_reward.py.
- [x] Write tests for boundaries/signs, contact gating, max vs sum, averaging, finite validation and missing terms.
- [x] Run RED; implement config and RewardManager branch; run GREEN.
### Task2: Physical collector
Files: new mjlab_env/key_hold_lateral.py; scene.py/task.py integration; tests/test_key_hold_lateral_sampling.py.
- [x] Test activated-key/contact attribution and actual velocity against MuJoCo; add graph/reset tests.
- [x] Implement fixed-size buffers, ancestor Jacobian and substep-only sampling; expose diagnostic histories.
### Task3: Frozen runtime validation and release
Files: tests/test_key_hold_lateral_physical.py; default entry, reward master table, README/change_log, logs/key_hold_lateral_20261026/build_archive.py.
- [x] Freeze reviewed overlay in isolated remote project, record SHA/job/logs, run relevant reward/regression/physical tests in Slurm.
- [x] Review implementation independently; fix findings and revalidate.
- [x] Freeze final source, verify returned SHA/manifest, point default script to it, verify effective default and docs.

## Ledger
User explicitly authorized implementation and remote release; no further design/plan permission needed. Current working tree contains unrelated uncommitted work: preserve it; isolation is a fresh remote deployment of the pinned source archive plus enumerated overlay, not a checkout of stale Git HEAD. Record task-only before snapshots for review; do not commit other work.

Task1/2 implementation complete; independent review findings fixed. Candidatev3 Slurm10140:73 passed+114 subtests,2 launcher checks skipped (external script),1 test harness constructor-argument error before arm sampling. Fixed test to read its exported key-geometry manifest directly. No production change afterv3; final freeze/verification pending. Full-arm MJLab zero-mocap initialization overflow is a separate existing limitation, recorded in change_log/review.

Release complete: Slurm10141 passed 74 tests +114 subtests; 2 external-launcher skips covered by local captured actual launcher and two-stage parser validation. Tested archive SHA a3668ebb939292e08e93e5627d5b9c5d14243242e759ddb379c8e7d2c5808e88 is now the default; no training started. Final changelog/plan bookkeeping occurs after immutable source freeze; runtime, tests, default JSON and reward master table match the tested archive.
