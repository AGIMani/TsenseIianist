# Turkish hand fingering Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to execute and verify the following tasks. The user has authorized generation, default-input update and a planning preview.

**Goal:** New A3.2-derived full-score fingering with reduced nonzero thumbs and a reviewable interactive physical preview.
**Architecture:** Verified frozen source on isolated remote CPU workspace; sampled coupled geometry feeds exact-event dynamic programming; current-contract actuator IK records physics; existing MJB viewer displays that same trace.
**Tech Stack:** Python/NumPy/SciPy, NoteSequence, MuJoCo, existing CPU coupling and Slurm rendering tools.
**Spec:** docs/superpowers/specs/2026-10-24-turkish-hand-fingering-design.md

## Global Constraints
- Preserve original 2544 note events and all non-fingering sequence fields. No local model compilation or physics.
- Current pinned archive, model geometry and controller contracts; no arm or reward algorithm changes.
- 1000Hz is the pre-existing authorized original-score Turkish exception; 20Hz control, 512 environments and learning budget unchanged.
- Local work tree is already dirty and contains latest assets; keep unrelated edits and avoid bulk commits/resets. Remote isolation comes from verified archive, not stale git HEAD.

## Review Focus
- Exact repeated-note gaps and sustained-note occupancy, including events within one control frame.
- Thumb soft cost must not silently ban thumbs or force a conflicting assignment.
- New black-key targets must use reference sites; current MCP/coupling bounds must govern geometry.
- Physical key audio/highlight must come from recorded qpos, never requested target.
- Default entry must resolve the delivered proto, preserve source SHA and fit full-episode prefill.

### Task 1: Extract and plan
- [x] Independently verify raw MIDI identity, tracks, event counts and minimum physical sampling coverage.
- [x] Extract current coupled model geometry in remote isolated source; save source/config and joint constraints.
- [x] Test exact sustained-note assignments and thumb-use selection on synthetic event cases; generate full-score proto/CSV preserving every non-part field.
- [x] Review reported reach/transition residuals and ensure both hands retain useful thumb assignments.

### Task 2: Record and package preview
- [x] Update offline planner for current recipe/shared key references/coupling/fitted resistance/target limits; static tests guard contracts.
- [x] Rebuild with the delivered score and run a ~20 second physical preview remotely at 1000Hz/20Hz; retain logs/status.
- [x] Package same trace model.mjb, audio, exact annotations, comparison PNG/arrays, SHA manifest and existing interactive launcher. Report all remaining misses/collisions.
- [x] Transfer and verify SHA; local playback validation only inside existing protected scope.

### Task 3: Select default and document
- [x] Update score/physics/replay-capacity and total steps needed for this MIDI; 22402 steps offset the longer prefill and preserve 2297216 optimizer updates; source archive unchanged.
- [x] Validate entry arguments without submitting training, full sequence compatibility and manifests.
- [x] Update current song README, root README and docs/change_log.md; independent review before final delivery.

Physical artifacts are delivered and validated; physical performance remains failed (20s F1=.14205), explicitly reported rather than treated as an acceptance pass.
