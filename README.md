# fireWild: Next-Day Wildfire Spread Prediction

**CSAI Project 17:** *Next Day Wildfire Spread Prediction via Tabular AML and Spatial Deep Learning.*

For every 1 km pixel of a 64 x 64 km region with an active fire on day *t*, predict the probability that it is burning on
day *t*+1. The project compares engineered tabular models, spatial deep learning and a hybrid of the two.

> Results are being regenerated for the final submission. This README will list them once every experiment has been re-run.
> Current status: [PROGRESS.md](PROGRESS.md).

## Documentation

| Topic | File |
|---|---|
| Dataset, splits and limits | [docs/DATASET_CARD.md](docs/DATASET_CARD.md) |
| Methodology | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Literature review | [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) |
| Graded Phase 1 report and slides (frozen) | [phase1_submission/](phase1_submission/) |

## Reproducing

Requires Python 3.12. On macOS, run `brew install libomp` first.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHON=python bash scripts/run_all.sh
python -m pytest
```
