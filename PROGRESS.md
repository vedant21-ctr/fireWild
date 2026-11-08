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
- Tabular models and leak tests re-run on all three splits; deep-learning sweep and ablations finished
- Memory fix for the 31-channel U-Net; `run_all.sh` resumes interrupted final and ablation stages correctly
- `CONTRIBUTIONS.md` rewritten from the real history, with suggested tasks for teammates
- Viva guide rewritten
- Generator for report tables and numbers (`src/evaluation/report_assets.py`)
- All 30 final deep-learning runs and the hybrid finished

## In progress

- Final comparison with confidence intervals (`src.evaluation.compare`): official done, spatial running, strict next
- IEEE report: drafted up to Evaluation Metrics, plus Limitations and Future Work

## Left

- Report abstract, results, ablation studies, discussion and conclusion; compile to PDF
- README with re-run results
- Final slides and executed notebooks
- `docs/DEVELOPMENT_LOG.md`, `docs/EXPERIMENT_TRACKER.md`, `docs/REQUIREMENTS_CHECKLIST.md`
- Fresh-clone check, final review, pull request into `main` for Vedant to review
