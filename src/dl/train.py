"""
train.py - Train one spatial model configuration on one protocol.

Leakage controls (same rules as the tabular pipeline):
  * inputs are day-t rasters only; the day t+1 mask is used only as the label
  * normalisation statistics come from the protocol's training split only
  * early stopping and the F1 threshold use the validation split only
  * the test split is predicted once, after training, with the selected weights
  * unknown labels (-1) are masked out of the loss and metrics
  * position leak: training uses random 32x32 crops; crop=0 trains on full patches and exists only as a leak diagnostic

The training target is next-day ACTIVE fire (FireMask at t+1 on every known pixel), as in the tabular models.
Inference runs on the full 64x64 patch (the networks are fully convolutional), and again on the fixed 32x32
windows from src/aml/leak_test.py for position-controlled scoring, with engineered channels recomputed from the window.

Usage (normally driven by src/dl/run_experiments.py):
    python -m src.dl.train --protocol spatial --name unet
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.aml.dataset import iter_patch_batches, patch_refs
from src.aml.leak_test import CROP, crop_raw, crop_windows
from src.aml.metrics import best_f1_threshold, compute_metrics
from src.aml.ndws import BASE_DIR, CACHE_DIR, CH
from src.config import load_config

from .data import (
    CHANNEL_SETS,
    augment,
    batch_all_channels,
    channel_indices,
    compute_norm,
    load_channels,
    normalise,
    random_crop,
    vector_index_pairs,
)
from .model import build_model, make_loss

CFG = load_config("dl")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ndws_dl")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "ndws", "dl")
PRED_DIR = os.path.join(CACHE_DIR, "preds")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_config(overrides):
    return {**CFG["defaults"], **overrides}


def run_channels(cfg):
    drop = set(cfg.get("drop_channels", []))
    return [c for c in CHANNEL_SETS[cfg["channels"]] if c not in drop]


@torch.no_grad()
def predict(model, X, dev, batch=128):
    """Sigmoid probability maps (N, H, W) float32 for normalised inputs X."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(dev)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)


def known_pixels(y, p):
    m = y >= 0
    return y[m].astype(np.int8), p[m]


def crop_inputs(protocol, split, channels, norm):
    """Normalised inputs and labels for the fixed 32x32 windows, channels recomputed from the window."""
    refs = patch_refs(protocol, split)
    rows, cols = crop_windows(len(refs))
    Xs, ys, offset = [], [], 0
    cols_idx = channel_indices(channels)
    for raw in iter_patch_batches(refs):
        b = len(raw)
        cropped = crop_raw(raw, rows[offset:offset + b], cols[offset:offset + b], CROP)
        Xs.append(batch_all_channels(cropped)[:, cols_idx])
        ys.append(cropped[:, CH["FireMask"]].astype(np.int8))
        offset += b
    return normalise(np.concatenate(Xs), norm), np.concatenate(ys)


def train(protocol, name, cfg, group=None, save_predictions=False):
    """Train one configuration. Results go to experiments/ndws/dl/<group or protocol>/<name>/results.json."""
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    channels = run_channels(cfg)
    out_dir = os.path.join(RESULTS_DIR, group or protocol, name)
    for d in (out_dir, MODEL_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)

    raw_train, ytr = load_channels(protocol, "train", channels)
    norm = compute_norm(raw_train, channels)
    Xtr = normalise(raw_train, norm)
    del raw_train
    raw_val, yva = load_channels(protocol, "val", channels)
    Xva = normalise(raw_val, norm)
    del raw_val
    log(f"[{protocol}/{name}] seed {seed} train {Xtr.shape}, val {Xva.shape}, device {dev}")

    model = build_model(cfg, len(channels)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = make_loss(cfg, dev)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = vector_index_pairs(channels)
    batch, crop = cfg["batch"], cfg["crop"]

    best_ap, best_state, best_epoch, history, stale = -1.0, None, -1, [], 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), batch):
            idx = np.sort(order[i:i + batch])
            xb = torch.from_numpy(np.asarray(Xtr[idx], dtype=np.float32)).to(dev)
            yb = torch.from_numpy(np.asarray(ytr[idx], dtype=np.int64)).to(dev)
            if crop:
                xb, yb = random_crop(xb, yb, crop, gen)
            if cfg["augment"]:
                xb, yb = augment(xb, yb, gen, pairs)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        yv, pv = known_pixels(np.asarray(yva), predict(model, Xva, dev))
        ap = float(average_precision_score(yv, pv))
        sched.step(ap)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pr_auc": ap,
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        log(f"[{protocol}/{name}] epoch {epoch:2d} loss={np.mean(losses):.4f} val PR-AUC={ap:.4f} ({time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                log(f"[{protocol}/{name}] early stop (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    tag = f"{protocol}_{name}" + (f"_seed{seed}" if seed != CFG["defaults"]["seed"] else "")
    torch.save({"state_dict": best_state, "config": cfg, "channels": channels, "norm": norm},
               os.path.join(MODEL_DIR, f"{tag}.pt"))

    p_val = predict(model, Xva, dev)
    yv, pv = known_pixels(np.asarray(yva), p_val)
    threshold = best_f1_threshold(yv, pv)
    results = {
        "protocol": protocol, "name": name, "target": "next-day active fire (FireMask at t+1, known pixels)",
        "config": cfg, "channels": channels, "n_channels": len(channels), "parameters": n_params, "device": str(dev),
        "epochs_run": len(history), "best_epoch": best_epoch, "history": history, "threshold_from_val": threshold,
        "val": compute_metrics(yv, pv, threshold),
    }

    if save_predictions:
        np.save(os.path.join(PRED_DIR, f"{tag}_val.npy"), p_val.astype(np.float32))
        raw_test, yte = load_channels(protocol, "test", channels)
        p_test = predict(model, normalise(raw_test, norm), dev)
        np.save(os.path.join(PRED_DIR, f"{tag}_test.npy"), p_test.astype(np.float32))
        yt, pt = known_pixels(np.asarray(yte), p_test)
        results["test"] = compute_metrics(yt, pt, threshold)
        log(f"[{protocol}/{name}] TEST PR-AUC={results['test']['pr_auc']:.4f} ROC-AUC={results['test']['roc_auc']:.4f}")
        for split in ("val", "test"):
            Xc, _ = crop_inputs(protocol, split, channels, norm)
            np.save(os.path.join(PRED_DIR, f"{tag}_{split}_crop.npy"), predict(model, Xc, dev).astype(np.float32))

    results_path = os.path.join(out_dir, "results.json" if seed == CFG["defaults"]["seed"] else f"results_seed{seed}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    del model, opt, best_state
    if dev.type == "mps":
        torch.mps.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--architecture", default=None)
    ap.add_argument("--channels", default=None)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("protocol", "name") and v is not None}
    train(args.protocol, args.name, resolve_config(overrides), save_predictions=True)


if __name__ == "__main__":
    main()
