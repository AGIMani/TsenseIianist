# Turkish Full-Score Verification Implementation Plan

> Execute the user's explicit full-score continuation using diagnostic evidence, parallel independent work, and verification before completion. No training.

**Goal:** Inspect and refine all 2544 original notes, reuse compatible repeated patterns, and deliver a continuous full-score physical replay with per-section evidence.

**Architecture:** Exact MIDI repeat/context audit and continuous baseline run proceed independently. Verified repeat constraints feed full-context fingering DP. Targeted controller corrections require reproducible evidence. Final candidate receives its own continuous full-song recording and paired interactive/PNG/audio package.

**Tech Stack:** Existing Python/NumPy/SciPy/MuJoCo3.11 remote CPU, protected saved-MJB local renderer.

**Spec:** docs/superpowers/specs/2026-10-25-turkish-fullscore-design.md

## Global constraints
- Keep original2544 event attributes except part; source MIDI SHA83707e4f40347d0a51402349a8ef8af7cf59c357656b50432591b3e8fceb88eb.
- Fixed archive1d4a67f15c796a849387ccc967c4960f4ee12d0b18bde2ff81f27a591c23d560 and model920475c61dc0469779d9f64769c75044140163151461201609ec181a7509fd4f, TZ off; no reward/drive/runtime archive change implied.
- 20Hz commands,1000Hz physical steps; all simulation/FK/collision remote. Never concatenate reset segments as continuous performance.
- Preserve dirty user workspace and prior results; no destructive cleanup or training submission.

## Review focus
- Pickup/signature/tempo events across MIDI tracks; time signature is not guessed from a rounded second grid.
- Repeated notes crossing a boundary remain held; repeated patterns with different context may need different fingering.
- Transposition changes black/white geometry except octave-equivalent shapes; recurrence is not physical reachability evidence.
- Terminal MIDI event can fall between physical samples; duration/source comparisons use documented endpoint rule.
- Full-score coverage must be evidenced by all notes, last timestamp, every repeated occurrence and trace continuity.

### Task1: Whole-score repeat audit and fingering
- [x] Test strict repeated measures, held boundary notes, transposed patterns, durations/velocities and mismatched context.
- [x] Implement scripts/audit_fingering_repeats.py with original-time repeat groups, per-note coverage and transition risk output.
- [x] Run full audit on305 candidate; generate only evidence-based repeat constraints, rerun full-contextDP remotely.
- [x] Verify no non-part changes/overlapping assignments; compare every group and all short-gap events, retain rejected candidates.

### Task2: Continuous physics and controller diagnosis
- [x] Start current305 continuous full-score baseline in unique remote output; record source/command/PID/resources.
- [x] Inspect scheduler/IK/drive trace for reproducible causes; implement only necessary corrections with focused tests.
- [x] If candidate/controller changes, run matching continuous full-score trajectory; no copied dynamics across repetitions.
- [x] Produce per-measure and per-repeat occurrence actual-key/contact report and full-score coverage assertions.

### Task3: Deliver and review
- [x] Package complete matching trajectory, MJB, audio, comparisonPNG/arrays, provenance/SHA and jump-to-section index.
- [x] Update full-score interactive fingering inspection to final candidate and show repeated/problem sections without implying physical success.
- [x] Validate local protected render/playback, complete source/trace/audio/time/SHA consistency and tests.
- [x] Update default external input only for selected candidate; update README/change_log and independently review full-coverage claims.

### Task4: Model-specific freedom and range study (user follow-up)
- [x] Report current A3.2 independent axes, coupled followers, bounds and actual source geometry used by the fingering planner.
- [x] Use failed full-score contact patterns and measured boundary occupancy to compare TZ off with upward 0–30mm and Pitch against its current 0–0.5rad range.
- [x] Build explicit isolated model variants remotely; keep canonical axis mapping, source fingerprints and drive/velocity limits visible. Do not infer enabled axes from action dimension.
- [x] Recheck whole-hand/key contacts and representative physical pressing/transitions for useful range changes. A static candidate or virtual-stage bound is not hardware calibration.
- [x] Decide and document which ranges have evidence for change; preserve training defaults until evidence supports a complete compatible contract. No training.

### Task5: Initial bilateral Palm_Base_Ty (user follow-up)
- [x] Verify the actual axis direction and mapping between --ty, initial root position and wrist joint coordinates; do not assume the sign of moving backward.
- [x] Compare current0.04m and modest backward offsets with the current frozen first_note_v1 source8ed0d50…121a3, same score and preparation settings, in a separate remote CPU deployment.
- [x] Record unprojected physical approach trajectories, root travel/peak velocity/acceleration, finger hover errors, clearances, hand/key contacts and position excursions. Do not reuse clipped full-song trajectories as startup safety evidence.
- [x] If a backward start gives a useful non-regressing result, set the verified default --ty directly as authorized; preserve other settings and the frozen archive. Otherwise retain0.04m and explain the evidence.
- [x] Deliver source/command/PID/exit/SHA records, update the combined result and change log, distinguish startup improvement from mid-song transitions.

## Rulings
- User explicitly authorized full-score continuation and repeat reuse; proceed without another design confirmation.
- Reuse geometric/fingering solutions, never physical states or measured scores. Boundary contexts remain independently checked.
- Prior branch/dirty working copy is the ongoing authorized workspace; no bulk commit, reset or unrelated changes.
- The user explicitly requested model-specific range exploration and singled out Pitch as well as TZ. The previous TZ-off conclusion is a baseline to test, not a reason to omit the new comparison.

## Final evidence
- Complete 222.474s candidate physical recording and 223-bar analysis delivered; F1 .1455738164 but acceptance false and collision counts higher. No successful-performance claim.
- New Ty=.020 validated separately for 2.55s current-source preparation; old-Ty full score remains explicitly separate.
- Full/preparation packages 123/48 files hashed; protected local render/viewer checks passed with OOM0; 101 pure tests passed.
- Default external score66563… and Ty=.02 captured through intercepted submission; no training.
