#!/usr/bin/env bash
# Reproduce every experiment from scratch. Run from the repository root.
#   PYTHON=path/to/python bash scripts/run_all.sh
# Each stage logs to experiments/ndws/logs/. A stage is skipped when its output already exists,
# so an interrupted run can be resumed. Delete the output to force a stage to run again.
set -euo pipefail

PYTHON="${PYTHON:-python}"
LOGS=experiments/ndws/logs
mkdir -p "$LOGS"
export PYTHONDONTWRITEBYTECODE=1

stage() {  # stage <output that marks completion> <log name> <python module and arguments...>
  local done_marker="$1" name="$2"
  shift 2
  if [ -e "$done_marker" ]; then
    echo "skip $name ($done_marker exists)"
    return
  fi
  echo "run  $name"
  "$PYTHON" -m "$@" 2>&1 | tee -a "$LOGS/$name.log"
}

stage data/ndws/manifest.json data "src.aml.ndws"
stage data/splits/split_audit.json splits "src.aml.splits"
stage experiments/ndws/eda/eda_summary.json eda "src.aml.eda"

for protocol in official spatial strict; do
  ablation=""
  [ "$protocol" != official ] && ablation="--ablation"
  stage "experiments/ndws/results/$protocol/results.json" "train_$protocol" src.aml.train --protocol "$protocol" $ablation
done
stage experiments/ndws/RESULTS.md evaluate_tabular "src.aml.evaluate"
for protocol in official spatial strict; do
  stage "experiments/ndws/results/$protocol/leak_test.json" "leak_test_$protocol" src.aml.leak_test --protocol "$protocol"
done

stage experiments/ndws/dl/dev/selected.json dl_sweep src.dl.run_experiments sweep
stage experiments/ndws/dl/dev/ablation/no_terrain_vegetation/results.json dl_ablations src.dl.run_experiments ablate
stage experiments/ndws/dl/strict/unet_fullpatch/results.json dl_final src.dl.run_experiments final
stage experiments/ndws/hybrid/strict/fusion.json hybrid src.hybrid.fusion
stage experiments/ndws/FINAL_RESULTS.md compare src.evaluation.compare
stage report/generated_results.tex report_assets src.evaluation.report_assets
