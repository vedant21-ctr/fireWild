# Progress

Short status of the final submission. Branch: `phase2/final-submission`. Last updated 8 November 2026.

## Done

- CLBI moved to `archive/clbi-prototype`; simulated data removed from this branch
- `main`'s October fixes merged into the working branch
- Old NDWS pipeline, stale reports, old slide sources and tracked model weights removed
- Graded Phase 1 report, slides and metrics grouped in `phase1_submission/`
- NDWS downloaded; split manifest regenerated and matches Phase 1 on every row
- Strict and tuning splits added, with programmatic checks
- Position-leak test, Compact CNN, loss and regularisation sweep, late fusion and unified comparison written
- Test crash fixed; unit tests and Ruff pass
- Methodology, architecture, project plan, dataset card and literature review updated; citations verified

## In progress

- Full experiment run (`scripts/run_all.sh`): tabular training on all three splits

## Left

- Deep-learning sweep, ablations and final runs; hybrid; final comparison
- README with re-run results
- Final IEEE report, including what changed since Phase 1
- Final slides, executed notebooks and viva guide
- `CONTRIBUTIONS.md` matching the real history, with suggested tasks for teammates
- `docs/DEVELOPMENT_LOG.md`, `docs/EXPERIMENT_TRACKER.md`, `docs/REQUIREMENTS_CHECKLIST.md`
- Make `run_all.sh` treat a partial ablation run as unfinished (after the current run ends)
- Fresh-clone check, final review, pull request into `main` for Vedant to review
