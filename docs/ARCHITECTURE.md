# Architecture

## Data flow

```text
Kaggle TFRecords
  -> src/aml/ndws.py        NumPy cache of raw patches (data/ndws/, gitignored)
  -> src/aml/splits.py      terrain fingerprint, four protocols, manifest + audit (data/splits/, committed)
  -> src/aml/eda.py         figures and data-quality audit

Tabular branch                               Spatial branch
  src/aml/features.py   32 features            src/dl/data.py   channel cache, train-only normalisation, crops
  src/aml/dataset.py    pixel sampling         src/dl/model.py  CompactCNN, UNet, losses
  src/aml/train.py      5 models, selection    src/dl/train.py  training loop, inference
  src/aml/leak_test.py  position leak tests    src/dl/run_experiments.py  sweep, ablations, final runs
  src/aml/evaluate.py   tabular figures

Both branches write predictions to data/ndws/preds/
  -> src/hybrid/fusion.py        late fusion, hybrid selection
  -> src/evaluation/compare.py   metrics, significance, maps, error analysis, report tables
```

`scripts/run_all.sh` runs every stage in order and can resume after an interruption. Settings are in `configs/`.

## Spatial models

**Compact CNN** (`src/dl/model.py::CompactCNN`). No down-sampling, so it keeps full resolution.

| Stage | Layers | Output channels | Dilation |
|---|---|---|---|
| Block 1 | 2 x (3x3 conv, batch norm, ReLU) | base | 1 |
| Block 2 | 2 x (3x3 conv, batch norm, ReLU) | 2 x base | 2 |
| Block 3 | 2 x (3x3 conv, batch norm, ReLU) | 2 x base | 4 |
| Head | spatial dropout, 1x1 conv | 1 logit per pixel | |

Each 3x3 convolution with dilation *d* widens the receptive field by 2*d*, so the total is 1 + 2 x 2 x (1 + 2 + 4) = 29 pixels.

**U-Net** (`src/dl/model.py::UNet`). Spatial size is shown for a 64x64 input.

| Stage | Operation | Channels | Spatial size |
|---|---|---|---|
| Encoder 1 | double conv | base | 64 |
| Encoder 2 | max pool, double conv | 2 x base | 32 |
| Encoder 3 | max pool, double conv | 4 x base | 16 |
| Bottleneck | max pool, double conv, spatial dropout | 8 x base | 8 |
| Decoder 3 | transposed conv, concat encoder 3, double conv | 4 x base | 16 |
| Decoder 2 | transposed conv, concat encoder 2, double conv | 2 x base | 32 |
| Decoder 1 | transposed conv, concat encoder 1, double conv | base | 64 |
| Head | 1x1 conv | 1 logit per pixel | 64 |

Both networks are fully convolutional, so they train on 32x32 crops and predict on any input size.

**Output.** A sigmoid turns each logit into P(pixel burns at day *t*+1). These are per-pixel statistical probabilities
given the inputs. They are not a causal simulation of fire propagation.

## Hybrid models

| Option | What is combined | Where |
|---|---|---|
| A. Late fusion | Logits of the best tree model and a network, combined by logistic regression fitted on held-out validation patches | `src/hybrid/fusion.py` |
| B. Feature-assisted network | Engineered channels added to the U-Net input | `channels: selected` in `configs/dl.yaml` |

Validation selects the reported hybrid among these candidates.
