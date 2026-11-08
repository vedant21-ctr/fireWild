"""
run_experiments.py - Hyperparameter sweep, ablations and final training for the spatial models.

    sweep     Greedy one-factor-at-a-time search on the dev protocol. Each stage in configs/dl.yaml varies one
              setting; the value with the best validation PR-AUC is carried into the next stage. The
              architecture and feature-channel stages are recorded but do not change the base U-Net.
    ablate    Single-change runs against the selected U-Net on the dev protocol.
    final     Train cnn, unet, unet_features and the unet_fullpatch diagnostic on every reported protocol,
              using the selected settings, and save validation and test predictions.

Writes experiments/ndws/dl/dev/{sweep.csv, selected.json, ablations.csv}.

Usage:
    python -m src.dl.run_experiments sweep
    python -m src.dl.run_experiments ablate
    python -m src.dl.run_experiments final
"""

import argparse
import json
import os

import pandas as pd

from src.aml.dataset import REPORTED_PROTOCOLS

from .train import CFG, RESULTS_DIR, log, resolve_config, results_path, train

DEV = "dev"
DEV_DIR = os.path.join(RESULTS_DIR, DEV)
SELECTED_PATH = os.path.join(DEV_DIR, "selected.json")
# Stages that are reported but do not change the base U-Net carried forward.
RECORD_ONLY_STAGES = {"architecture", "feature_channels"}


def summarise(stage, name, cfg, result):
    return {"stage": stage, "name": name, **{k: cfg[k] for k in CFG["defaults"] if k not in ("seed", "epochs", "patience")},
            "val_pr_auc": result["val"]["pr_auc"], "val_roc_auc": result["val"]["roc_auc"], "val_brier": result["val"]["brier"],
            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "seconds": sum(h["seconds"] for h in result["history"])}


def option_key(cfg):
    return json.dumps(cfg, sort_keys=True)


def run_sweep():
    current, rows, seen, feature_choice = resolve_config({}), [], {}, "base"
    os.makedirs(DEV_DIR, exist_ok=True)
    for stage, options in CFG["sweep"].items():
        best = None
        for i, option in enumerate(options):
            cfg = {**current, **option}
            key = option_key(cfg)
            name = f"{stage}_{i}"
            if key in seen:
                result = seen[key]
                log(f"[sweep] {name}: reusing an identical earlier run")
            else:
                result = train(DEV, name, cfg, group=f"{DEV}/sweep")
                seen[key] = result
            rows.append(summarise(stage, name, cfg, result))
            pd.DataFrame(rows).to_csv(os.path.join(DEV_DIR, "sweep.csv"), index=False)
            if best is None or result["val"]["pr_auc"] > best[0]:
                best = (result["val"]["pr_auc"], cfg, option)
        log(f"[sweep] stage {stage}: best {best[2]} val PR-AUC={best[0]:.4f}")
        if stage == "feature_channels":
            feature_choice = best[1]["channels"]
        if stage not in RECORD_ONLY_STAGES:
            current = best[1]
    with open(SELECTED_PATH, "w") as f:
        json.dump({"unet": current, "feature_channels": feature_choice}, f, indent=2)
    log(f"[sweep] selected: {current}; feature channels: {feature_choice}")


def load_selected():
    with open(SELECTED_PATH) as f:
        return json.load(f)


def run_ablations():
    sel = load_selected()
    rows = []
    baseline = train(DEV, "baseline", sel["unet"], group=f"{DEV}/ablation")
    rows.append(summarise("ablation", "baseline", sel["unet"], baseline))
    for name, change in CFG["ablations"].items():
        cfg = {**sel["unet"], **change}
        result = train(DEV, name, cfg, group=f"{DEV}/ablation")
        rows.append(summarise("ablation", name, cfg, result))
        pd.DataFrame(rows).to_csv(os.path.join(DEV_DIR, "ablations.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(DEV_DIR, "ablations.csv"), index=False)


def final_config(name, spec, sel, seed):
    settings = dict(spec)
    if settings.get("channels") == "selected":
        settings["channels"] = sel["feature_channels"]
    return {**sel["unet"], **settings, "seed": seed}


def run_final():
    sel = load_selected()
    for protocol in REPORTED_PROTOCOLS:
        for name, spec in CFG["final"].items():
            seeds = CFG["final_seeds"] if name != "unet_fullpatch" else CFG["final_seeds"][:1]
            for seed in seeds:
                # the results file is written last, so its presence means the run and its predictions are complete
                if os.path.exists(results_path(protocol, name, seed)):
                    log(f"[{protocol}/{name}] seed {seed} already complete, skipping")
                    continue
                train(protocol, name, final_config(name, spec, sel, seed), save_predictions=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "ablate", "final"])
    stage = ap.parse_args().stage
    {"sweep": run_sweep, "ablate": run_ablations, "final": run_final}[stage]()


if __name__ == "__main__":
    main()
