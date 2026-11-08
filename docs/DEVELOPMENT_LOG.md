# Development Log

## 30 September 2026: earlier prototype

CLBI prototype and API (`archive/clbi-prototype`). Its data was simulated, so it is not part of Project 17.

## 24 to 28 October 2026: Phase 1

- Architecture documents, contributor roadmap and NDWS data audit.
- NDWS pipeline in `src/aml/`, terrain-fingerprint split audit, IEEE report and slides.
- First leakage-safe U-Net.
- Dependency, test and API fixes on `main`.

## 8 November 2026: final submission branch

**Audit.** The real pipeline was on `phase1-final-submission`, and `main` mostly presented CLBI. Findings included a spatial split that still mixed weeks and no position-leak test for tabular models.

**Team decisions.** Phase 1 is frozen. Add the strict split and leak test to Phase 2, and state what changed. Work on a new branch with a pull request into `main`, reviewed and merged by Vedant. Move CLBI to an archive branch. Report three splits, stay on NDWS, and skip WildfireSpreadTS and ConvLSTM.

**Cleanup.** Removed CLBI code, simulated data, the superseded NDWS pipeline, stale reports, duplicate scripts and tracked weights. The graded Phase 1 report, slides and metrics moved to `phase1_submission/`.

**Repairs.**
- A crash from OpenMP loading order on macOS was fixed by importing PyTorch first (`tests/conftest.py`).
- A CNN receptive-field test used an incorrect figure and zero input. It now uses random input and the correct 29 pixels.
- Six `test_ui_contract` tests were removed along with the dashboard they tested. They were not fixed.
- Ruff is clean at a project line length of 150.

**New work.** Strict and dev splits with programmatic checks, Dummy baseline, position-leak tests, compact CNN, greedy tuning sweep, ablations, late fusion, unified comparison, generated report tables and final deliverables.

**Incidents during the full run.**
- Low battery and heavy swap slowed training for a short time. Charging restored normal speed.
- The first 31-channel U-Net froze the 8 GB machine because loading made several in-memory copies. The run was stopped, the loader changed to process chunks, and the run resumed from completed results. The new loader matched the old one exactly on three sample cases.
- The resume marker for the ablation stage was changed so that a partial run is no longer treated as complete.

**Outcome.** No network or hybrid exceeded the best tree on any split. Results are stated in the report without adjustment.
