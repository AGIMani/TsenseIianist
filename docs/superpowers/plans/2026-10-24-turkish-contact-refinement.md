# Turkish Contact Refinement Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans / systematic-debugging; authorized continuation, no training.

**Goal:** Improve current A3.2 Turkish fingering and physical paths, collision behavior, and assess upward TZ0–30mm.
**Architecture:** Independent model contract and contact probes → corrected constrained contact IK → time-aware path/fingering refinement → same-trace physical comparison and interactive delivery.
**Tech Stack:** Existing Python/NumPy/SciPy/MuJoCo/current coupled drives on remote CPU.
**Spec:** docs/superpowers/specs/2026-10-24-turkish-contact-refinement-design.md

## Global Constraints
- Original2544 events unchanged,20Hz control/1000Hz physics, no training.
- Current fitted model/keys/100%activation; TZ trial separately hashed and never silently mixed with existing checkpoints.
- Local dirty working copy retained, no bulk commits or reset; remote variants isolated.
- Source/contact/target/actual pose and policy contract distinguished.

## Review Focus
- TZ49-action count ambiguous with old yaw-only shape heuristic; never declare training-compatible without explicit mask audit.
- Idle fingers already high must not be pulled down; frozen wrist rows must stay frozen through coupledIK.
- Real contact top is not task reward target; key travel from physical state only.
- Avoid different-finger, palm, cross-hand and wrong-key collisions, not only sampled tip separation.
- 1000Hz recording source model/config/MIDI same through audio/graph/viewer.

### Task1: Diagnose and isolate
- [x] Audit TZ axis/ranges/drive/action/observation and prepare separately hashed variant. Mapping ambiguity and unvalidated drive keep default off.
- [x] Test single-note steady pressure with current fitted physics and classify failures.
- [x] Probe difficult chords: fixed targets, real top, free longitudinal top coordinates, TZ0 geometry and separately verified 0–30mm travel. Dynamic TZ/chord advantage remains unproven.

### Task2: Correct contact solver and schedule
- [x] Add failing pure tests for idle one-sided errors, bounded coupledIK, key-top domains, event release/transfer deadlines.
- [x] Implement minimal shared contact residuals and constrained solve in scripts/hpc/piano_contact_planner.py; integrate current planner without fake key motion.
- [x] Refine assignments and press targets; test lift-transfer-press with collision checks. Diagnostic chord hold succeeds, original-tempo dynamic path remains unaccepted.
- [x] Run controlled comparisons and record the tradeoff: improved raw-MIDI F1/cross-hand contact, worse same-hand contact. Deliver candidate for inspection, not overall physical acceptance.

### Task3: Delivery and consistency
- [x] Save new fingered proto/reference paths and old baseline; verify all non-part data.
- [x] Keep TZ off and research variant explicit; no runtime archive change.
- [x] Package same physical trace with sound, PNG, events, MJB/sourceSHA; protected local playback check. Raw-MIDI20s and separate chord diagnostic each validated.
- [x] Update README/change_log; independent review; no training submitted.

## 2026-10-25 added authorized scope: collision reward inspector
- [x] Audit current frozen same-hand body-pair filter and exclusion semantics; no reward runtime change.
- [x] Remote 253-pose bending/swing sweep, exact coupling/model/source verification.
- [x] Add remote JSONL pose recorder and local protected MuJoCo playback with joint sliders and live static penalty estimate; visual/hull toggle.
- [x] Verify zero/positive/reset cases, scene rendering, fixed memory budgets, no local physics/contact solve.
- [x] Finish matching-final-score replay and independent chord diagnostic delivery, update change log and remaining limitations.

Remaining physical acceptance: full-score original-tempo execution, release/next-note transition from wide chord, all target activation and collision reduction together, actuator load margin; TZ dynamic value and complete CAD-shell false-contact classification remain unproved. These are not marked passed by packaging or UI tests.
