# Ending return implementation plan
Spec: docs/superpowers/specs/2026-10-25-ending-return.md
1. Add explicit ending contract and bounded release/return planner; test no wrist travel during release, origin endpoint, finite limits and invalid config rejection.
2. Continue native physics from actual final state, save raw diagnostics and failure traces. Integrate renderer, audio, physics observer, empty score tail and packaging. Test seam and sample/time accounting.
3. Deploy isolated current archive plus task overlay on volc-a32; run physical and Slurm GPU integration checks. Preserve any excursions and provenance.
4. Review implementation, update docs/default entry and publish descendant source archive/SHA after checks.
Constraints: preserve working changes and experiments; no new training; no local model loads or physics. Work in existing feature workspace, backup current files; isolation for heavy verification uses a fresh remote directory.
Review focus: end target is original pre-preparation pose, last policy state continuity, missing/duplicate samples, score/audio duration, failures must persist raw evidence, no policy experience/budget changes.
