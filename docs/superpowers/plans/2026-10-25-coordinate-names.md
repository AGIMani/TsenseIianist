# Coordinate Names Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; independent rename audit/implementation and final review are delegated within the existing shared workspace.

**Goal:** Make initial position options world coordinates and remove virtual forearm/palm_base ambiguity without changing the no-arm physical task.

**Architecture:** A standard-library pose contract validates world positions and converts them to existing virtual slide coordinates. Joint names change throughout active readers; archived sources remain immutable. Arm assembly uses the requested world palm position for its fixed-base calculation.

**Tech Stack:** Python, Bash, JSON, pure CPU contract tests; remote MuJoCo verification.

**Spec:** docs/superpowers/specs/2026-10-25-coordinate-names.md

## Global Constraints

- No training; no local MuJoCo model construction/loading.
- Preserve all unrelated dirty changes, historical archives, URDF bytes and reward formulas.
- Current entry world XYZ: R(.16,.15,.09), L(.16,-.15,.09) m.
- New archive derives from f9741e1f68b6253909011edde71993bc252a6829f61fda4840c095154467bccf.

## Review Focus

- Old flags rejected before model/GPU allocation; no accidental pixel-height replacement.
- Nondefault world position changes actual roots with correct signs, not simply option labels.
- Palm prefix classification still separates fingers; control channels/order unchanged.
- Arm Cartesian zero is installed at requested world position; world displacement semantics unchanged.
- Historical recipes cannot silently rebuild with changed source.

### Task 1: Initial world pose and CLI
- [x] Add failing pure tests for world-position validation/conversion and early old-option rejection.
- [x] Implement shared pose contract; update recipe/constants/CLI/callers/manifest and code headers.
- [x] Verify pure tests and syntax, including nondefault asymmetric world Y.

### Task 2: Palm names and remaining aliases
- [x] Rename all five generated joints and prefix classifiers, preserving action channel order and physical parameters.
- [x] Update current displays/metadata and distinguish actual, target, mount and body positions.
- [x] Add static invariance checks; review other names and document exclusions.

### Task 3: Freeze and remote verification
- [x] Build immutable overlay archive with manifest/SHA; sync independent remote directory and verify hashes.
- [x] Run remote no-arm/arm root-position and structural regression tests (no training).
- [x] Review implementation independently; fix material findings and rerun affected checks.
- [x] Update entry archive and world flags, README/current coordinate guide/change log; capture effective entry without submitting.

Validation: 47 local pure cases; 4 actual world-pose cases; 38 remote regressions plus corrected stale-count case passing on rerun. Default fixed-TZ/Yaw 495/495 at 2e-12; auxiliary full-model derived-state exceptions remain disclosed in logs. No training.
