"""
train.py - Train the spatial U-Net on one protocol and score its test split once.

Leakage controls (same rules as the tabular pipeline):
  * inputs are day-t rasters only; the day t+1 mask is used only as the label
  * normalisation statistics come from the training split only
  * early stopping and the F1 threshold use the validation split only
  * the test split is predicted once, after training, with the selected weights
  * unknown labels (-1) are masked out of the loss and metrics

  * position leak: NDWS patches are centred on (next-day) fire, so training uses random
    32x32 crops by default; the model cannot learn where the patch centre is (--crop 0
    trains the position-exploitable full-patch variant, kept only as a leak diagnostic)

The training target is next-day ACTIVE fire (FireMask at t+1 on every known pixel), as in
the tabular models. Transition (new-fire) metrics are computed afterwards by
src/dl/evaluate.py on the pixels that were not burning at day t. Inference always runs on
the full 64x64 patch (the network is fully convolutional).

Usage:
    python -m src.dl.train --protocol spatial              # variant "unet" (32x32 crops)
    python -m src.dl.train --protocol spatial --crop 0     # variant "unet_fullpatch"
"""

import os
import json
import time
import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.aml.ndws import BASE_DIR, CACHE_DIR
from src.aml.metrics import best_f1_threshold, compute_metrics
from .data import DL_CHANNELS, load_split_tensors, augment, random_crop
from .model import UNet, masked_bce

SEED = 42
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


@torch.no_grad()
def predict(model, X, dev, batch=128):
    """Sigmoid probability maps (N, 64, 64) float32 for normalised inputs X."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(dev)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)


def known_pixels(y, p):
    m = y >= 0
    return y[m].astype(np.int8), p[m]


def variant_name(crop):
    return "unet" if crop else "unet_fullpatch"


def train(protocol, epochs=25, batch=32, lr=1e-3, weight_decay=1e-4, pos_weight=1.0, patience=5, base=24, crop=32):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = device()
    variant = variant_name(crop)
    out_dir = os.path.join(RESULTS_DIR, protocol, variant)
    for d in (out_dir, MODEL_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)

    log(f"[{protocol}] loading tensors")
    Xtr, ytr, norm = load_split_tensors(protocol, "train")
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr)  # ~1.6 GB in RAM; random reads from the memmap are 4x slower
    Xva, yva, _ = load_split_tensors(protocol, "val", norm)
    log(f"[{protocol}] train {Xtr.shape}, val {Xva.shape}, device {dev}")

    model = UNet(len(DL_CHANNELS), base=base).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    pw = torch.tensor(pos_weight, device=dev) if pos_weight != 1.0 else None
    gen = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    best_ap, best_state, best_epoch, history, stale = -1.0, None, -1, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), batch):
            idx = np.sort(order[i:i + batch])
            xb = torch.from_numpy(np.asarray(Xtr[idx], dtype=np.float32)).to(dev)
            yb = torch.from_numpy(np.asarray(ytr[idx], dtype=np.int64)).to(dev)
            if crop:
                xb, yb = random_crop(xb, yb, crop, gen)
            xb, yb = augment(xb, yb, gen)
            loss = masked_bce(model(xb), yb, pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        yv, pv = known_pixels(np.asarray(yva), predict(model, Xva, dev))
        ap = float(average_precision_score(yv, pv))
        sched.step(ap)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pr_auc": ap,
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        log(f"[{protocol}] epoch {epoch:2d} loss={np.mean(losses):.4f} val PR-AUC={ap:.4f} ({time.time() - t0:.0f}s)")
        if ap > best_ap:
            best_ap, best_epoch, stale = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log(f"[{protocol}] early stop (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "base": base, "channels": DL_CHANNELS, "norm": norm},
               os.path.join(MODEL_DIR, f"{protocol}_{variant}.pt"))

    yv, pv = known_pixels(np.asarray(yva), predict(model, Xva, dev))
    threshold = best_f1_threshold(yv, pv)
    val_metrics = compute_metrics(yv, pv, threshold)

    log(f"[{protocol}] scoring the test split")
    Xte, yte, _ = load_split_tensors(protocol, "test", norm)
    p_test = predict(model, Xte, dev)
    np.save(os.path.join(PRED_DIR, f"{protocol}_{variant}_test.npy"), p_test.astype(np.float16))
    yt, pt = known_pixels(np.asarray(yte), p_test)
    test_metrics = compute_metrics(yt, pt, threshold)
    log(f"[{protocol}] TEST {variant} PR-AUC={test_metrics['pr_auc']:.4f} ROC-AUC={test_metrics['roc_auc']:.4f} F1={test_metrics['f1']:.4f}")

    results = {
        "protocol": protocol, "variant": variant, "target": "next-day active fire (FireMask at t+1, known pixels)",
        "model": {"architecture": "U-Net", "base_channels": base, "parameters": n_params, "inputs": DL_CHANNELS},
        "training": {"epochs_run": len(history), "best_epoch": best_epoch, "batch": batch, "lr": lr,
                     "weight_decay": weight_decay, "pos_weight": pos_weight, "patience": patience,
                     "train_crop": crop or 64,
                     "augmentation": "random dihedral flips/transposes with wind vector transformed", "seed": SEED,
                     "device": str(dev)},
        "history": history, "threshold_from_val": threshold, "val": val_metrics, "test": test_metrics,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"[{protocol}] wrote {os.path.relpath(out_dir, BASE_DIR)}/results.json")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["official", "spatial"], required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--crop", type=int, default=32, help="random training crop size; 0 = full 64x64 patches")
    args = ap.parse_args()
    train(args.protocol, epochs=args.epochs, batch=args.batch, lr=args.lr, pos_weight=args.pos_weight, base=args.base,
          crop=args.crop)


if __name__ == "__main__":
    main()
