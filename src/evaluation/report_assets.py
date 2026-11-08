"""
report_assets.py - Turn the results files into the tables, figures and numbers used by the report and slides.

No number in report/main.tex or the slides is typed by hand: tables are \\input from report/tables/, and values in the
text are macros from report/generated_results.tex.

Reads experiments/ndws/ and phase1_submission/metrics/. Writes experiments/ndws/final/numbers.json,
experiments/ndws/final/{sweep,dl_ablations}.png, report/tables/*.tex and report/generated_results.tex.

Usage:
    python -m src.evaluation.report_assets
"""

import json
import os
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.aml.dataset import REPORTED_PROTOCOLS
from src.aml.ndws import BASE_DIR

from .compare import EXP_DIR, LABELS, OUT_DIR, read_json

REPORT_DIR = os.path.join(BASE_DIR, "report")
TABLE_DIR = os.path.join(REPORT_DIR, "tables")
PHASE1_DIR = os.path.join(BASE_DIR, "phase1_submission", "metrics")
DEV_DIR = os.path.join(EXP_DIR, "dl", "dev")
RESULT_MODELS = ["dummy", "persistence", "logistic_regression", "random_forest", "xgboost", "lightgbm", "cnn", "unet", "unet_features", "hybrid"]
CROP_RESULT_MODELS = ["position_prior", "lightgbm_full", "lightgbm_crop", "cnn", "unet", "unet_features", "hybrid"]
PHASE1_MODELS = ["persistence", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
FUSION_MODELS = ["fusion_tree_only", "fusion_unet_only", "fusion_tree_unet", "fusion_tree_cnn", "fusion_tree_unet_features", "unet_features"]
SWEEP_FIELDS = {"architecture": "architecture", "loss": "loss", "dropout": "dropout", "weight_decay": "weight_decay", "width": "base",
                "feature_channels": "channels"}
ABLATION_LABELS = {
    "baseline": "Selected U-Net", "no_augmentation": "No flips or rotations", "no_dropout": "No dropout", "no_weight_decay": "No weight decay",
    "no_regularisation": "None of the three", "no_skip_connections": "No skip connections", "no_prev_fire_channel": "No day-$t$ fire channels",
    "no_wind_channels": "No wind channels", "no_weather_channels": "No weather channels", "no_terrain_vegetation": "No elevation or NDVI",
}
TABULAR_ABLATION_LABELS = {
    "raw_only": "Raw values only", "fire_only": "Fire context only", "raw+fire": "Raw + fire", "raw+fire+wind": "Raw + fire + wind",
    "raw+fire+wind+terrain": "Raw + fire + wind + terrain", "all_features": "All features", "all_minus_raw": "All minus raw",
    "all_minus_fire": "All minus fire", "all_minus_wind": "All minus wind", "all_minus_terrain": "All minus terrain",
    "all_minus_weather": "All minus weather", "all_minus_vegetation": "All minus vegetation",
}
MACROS = ("pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc", "pr_auc_lift", "precision", "recall", "f1")


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(name, columns, rows, align):
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    lines += [r"\midrule" if row is None else " & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(os.path.join(TABLE_DIR, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def num(x, digits=3):
    return "--" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def with_ci(row):
    return f"{row['pr_auc']:.3f} [{row['pr_auc_ci_low']:.3f}, {row['pr_auc_ci_high']:.3f}]"


def result_tables(table):
    """One table per protocol and regime for the active target, plus a transition table of PR-AUCs."""
    for protocol in REPORTED_PROTOCOLS:
        sub = table[(table.protocol == protocol) & (table.target == "active")]
        full = sub[sub.regime == "full"].set_index("model")
        columns = ("roc_auc", "precision", "recall", "f1", "recall_at_1pct_fpr", "recall_at_5pct_fpr")
        write_table(f"results_{protocol}_full", ["Model", "PR-AUC [95\\% CI]", "Lift", "ROC-AUC", "Prec.", "Rec.", "F1", "R@1\\%", "R@5\\%"],
                    [[LABELS[m], with_ci(full.loc[m]), num(full.loc[m, "pr_auc_lift"], 1), *[num(full.loc[m, c]) for c in columns]]
                     for m in RESULT_MODELS], "lcccccccc")
        crop = sub[sub.regime == "crop"].set_index("model")
        write_table(f"results_{protocol}_crop", ["Model", "PR-AUC [95\\% CI]", "Lift", "ROC-AUC"],
                    [[LABELS[m], with_ci(crop.loc[m]), num(crop.loc[m, "pr_auc_lift"], 1), num(crop.loc[m, "roc_auc"])]
                     for m in CROP_RESULT_MODELS], "lccc")
    wide = table[(table.regime == "full") & (table.target == "transition")].pivot(index="model", columns="protocol", values="pr_auc")
    write_table("results_transition", ["Model", *REPORTED_PROTOCOLS],
                [[LABELS[m], *[num(wide.loc[m, p]) for p in REPORTED_PROTOCOLS]] for m in RESULT_MODELS], "lccc")


def seed_table(seeds):
    write_table("seed_stability", ["Protocol", "Model", "Mean", "SD", "Min", "Max"],
                [[r["protocol"], LABELS[r["model"]], *[num(r[f"{s}_test_pr_auc"]) for s in ("mean", "std", "min", "max")]]
                 for r in seeds.to_dict("records")], "llcccc")


def significance_table(sigs, regime="full", name="significance"):
    """Paired patch-bootstrap differences for the main questions on the active target."""
    rows = []
    for protocol in REPORTED_PROTOCOLS:
        items = sigs[protocol][regime]["active"]
        for key, s in items.items():
            if not key.startswith(("unet", "cnn", "hybrid")):
                continue
            rows.append([protocol, tex_escape(key), f"{s['delta']:+.4f}", f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]", num(s["p_le_zero"])])
        rows.append(None)
    write_table(name, ["Protocol", "Comparison", "$\\Delta$PR-AUC", "95\\% CI", "$P(\\Delta \\le 0)$"], rows[:-1], "llccc")


def fusion_table(fusions):
    rows = []
    for protocol in REPORTED_PROTOCOLS:
        report = fusions[protocol]["full"]
        for name in FUSION_MODELS:
            c = report["candidates"][name]
            coefficients = ", ".join(f"{tex_escape(k)}={v:.2f}" for k, v in c.get("coefficients", {}).items()) or "--"
            mark = " (selected)" if name == report["selected_hybrid"] else ""
            rows.append([protocol, LABELS[name] + mark, coefficients, num(c["select_half_pr_auc"]), num(c["test"]["pr_auc"])])
        rows.append(None)
    write_table("fusion", ["Protocol", "Candidate", "Weights on logits", "Val. PR-AUC", "Test PR-AUC"], rows[:-1], "llccc")


def sweep_outputs():
    """Sweep table and figure. The stage's best value is carried forward, except the two record-only stages."""
    sweep = pd.read_csv(os.path.join(DEV_DIR, "sweep.csv"))
    rows = []
    for stage, field in SWEEP_FIELDS.items():
        sub = sweep[sweep.stage == stage]
        best = sub.val_pr_auc.max()
        for r in sub.to_dict("records"):
            value = f"{r[field]} (weight {r['pos_weight']:g})" if field == "loss" and r["loss"] == "weighted_bce" else r[field]
            mark = r"\checkmark" if r["val_pr_auc"] == best else ""
            rows.append([tex_escape(stage), tex_escape(value), num(r["val_pr_auc"], 4), num(r["best_epoch"], 0), mark])
        rows.append(None)
    write_table("sweep", ["Stage", "Option", "Val. PR-AUC", "Best epoch", "Best"], rows[:-1], "llccc")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [f"{r['stage']}: {r[SWEEP_FIELDS[r['stage']]]}" for r in sweep.to_dict("records")]
    ax.barh(range(len(sweep)), sweep.val_pr_auc, color="#30638e")
    ax.set_yticks(range(len(sweep)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Validation PR-AUC (dev protocol)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def dl_ablation_outputs():
    ablations = pd.read_csv(os.path.join(DEV_DIR, "ablations.csv")).set_index("name")
    baseline = ablations.loc["baseline", "val_pr_auc"]
    write_table("dl_ablations", ["Change", "Val. PR-AUC", "Change vs selected"],
                [[ABLATION_LABELS[n], num(r["val_pr_auc"], 4), "--" if n == "baseline" else f"{r['val_pr_auc'] - baseline:+.4f}"]
                 for n, r in ablations.iterrows()], "lcc")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(range(len(ablations)), ablations.val_pr_auc, color=["#6a4c93" if n == "baseline" else "#adb5bd" for n in ablations.index])
    ax.set_yticks(range(len(ablations)), [re.sub(r"\$t\$", "t", ABLATION_LABELS[n]) for n in ablations.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Validation PR-AUC (dev protocol)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dl_ablations.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ablations


def tabular_ablation_table():
    results = {p: read_json(EXP_DIR, "results", p, "results.json")["ablation"] for p in ("spatial", "strict")}
    write_table("tabular_ablations", ["Feature groups", "Spatial val.", "Spatial test", "Strict val.", "Strict test"],
                [[label, *[num(results[p][key][f"{s}_pr_auc"]) for p in results for s in ("val", "test")]]
                 for key, label in TABULAR_ABLATION_LABELS.items()], "lcccc")
    return results


def leak_table():
    tests = {p: read_json(EXP_DIR, "results", p, "leak_test.json") for p in REPORTED_PROTOCOLS}
    columns = ["Test", *REPORTED_PROTOCOLS]
    measures = [
        ("Features only, PR-AUC", lambda t: t["add_position"]["test"]["pr_auc_features"]),
        ("Features + position, PR-AUC", lambda t: t["add_position"]["test"]["pr_auc_features_plus_position"]),
        ("Recover position from features, ROC-AUC", lambda t: t["recover_position"]["test_roc_auc"]),
        ("Window: full-patch model, PR-AUC", lambda t: t["crop_control"]["test"]["pr_auc_full_patch_model"]),
        ("Window: window-trained model, PR-AUC", lambda t: t["crop_control"]["test"]["pr_auc_crop_trained"]),
        ("Window: position prior, PR-AUC", lambda t: t["crop_control"]["test"]["pr_auc_position_prior"]),
        ("Window: positive rate", lambda t: t["crop_control"]["test"]["positive_rate"]),
    ]
    write_table("leak_test", columns, [[label, *[num(f(tests[p])) for p in REPORTED_PROTOCOLS]] for label, f in measures], "lccc")
    return tests


def phase1_change_table():
    """Test PR-AUC submitted at Phase 1 against the re-run on the same splits, and the new strict split."""
    phase1 = {p: pd.read_csv(os.path.join(PHASE1_DIR, f"{p}_metrics_table.csv")).set_index("model") for p in ("official", "spatial")}
    now = {p: pd.read_csv(os.path.join(EXP_DIR, "results", p, "metrics_table.csv")).set_index("model") for p in REPORTED_PROTOCOLS}
    rows, out = [], {}
    for m in PHASE1_MODELS:
        out[m] = {"phase1_official": float(phase1["official"].loc[m, "pr_auc"]), "final_official": float(now["official"].loc[m, "pr_auc"]),
                  "phase1_spatial": float(phase1["spatial"].loc[m, "pr_auc"]), "final_spatial": float(now["spatial"].loc[m, "pr_auc"]),
                  "final_strict": float(now["strict"].loc[m, "pr_auc"])}
        rows.append([LABELS[m], *[num(v) for v in out[m].values()]])
    write_table("phase1_changes", ["Model", "Official (P1)", "Official (final)", "Spatial (P1)", "Spatial (final)", "Strict (new)"], rows, "lccccc")
    return out


def macro_name(*parts):
    return "r" + "".join(re.sub(r"[^A-Za-z]", "", p.title()) for p in parts)


def data_macros():
    """Dataset, split and model-size numbers quoted in the text."""
    audit = read_json(BASE_DIR, "data", "splits", "split_audit.json")
    eda = read_json(EXP_DIR, "eda", "eda_summary.json")
    balance = {r["split"]: r for r in eda["class_balance"]}
    wind = eda["wind_orientation"]["rows S->N, cols W->E (adopted)"]
    test_overlap = audit["overlap"]["test"]
    macros = {
        "rPatches": f"{audit['n_patches']:,}", "rLocationGroups": str(audit["n_location_groups"]),
        "rTwoLargestGroups": f"{100 * sum(audit['largest_groups'][:2]) / audit['n_patches']:.0f}",
        "rOfficialOverlap": f"{100 * test_overlap['share_terrain_with_train'] / test_overlap['n_patches']:.0f}",
        "rPersistenceRecall": f"{100 * balance['train']['persistence_recall']:.0f}",
        "rNewFireShare": f"{100 * balance['train']['frac_next_day_fire_new']:.0f}",
        "rTrainPositiveRate": f"{100 * balance['train']['positive_rate']:.2f}", "rUnknownRate": f"{100 * balance['train']['unknown_label_rate']:.1f}",
        "rWindRatio": f"{wind['ratio']:.2f}", "rWindLow": f"{wind['ratio_ci95'][0]:.2f}", "rWindHigh": f"{wind['ratio_ci95'][1]:.2f}",
        "rStrictTrain": f"{audit['strict_split_sizes']['train']:,}", "rStrictVal": str(audit["strict_split_sizes"]["val"]),
        "rStrictTest": str(audit["strict_split_sizes"]["test"]), "rDevTrain": f"{audit['dev_split_sizes']['train']:,}",
        "rDevVal": f"{audit['dev_split_sizes']['val']:,}", "rSpatialTrain": f"{audit['spatial_split_sizes']['train']:,}",
        "rSpatialVal": f"{audit['spatial_split_sizes']['val']:,}", "rSpatialTest": f"{audit['spatial_split_sizes']['test']:,}",
    }
    for split, name in (("train", "Train"), ("eval", "Val"), ("test", "Test")):
        macros[f"rOfficial{name}"] = f"{balance[split]['patches']:,}"
    for name in ("cnn", "unet", "unet_features"):
        macros[macro_name(name, "parameters")] = f"{read_json(EXP_DIR, 'dl', 'strict', name, 'results.json')['parameters']:,}"
    return macros


def comparison_macros(sigs):
    macros = {}
    for protocol, regimes in sigs.items():
        for regime, targets in regimes.items():
            for target, items in targets.items():
                for key, s in items.items():
                    a, b = key.split(" - ")
                    base = macro_name(protocol, regime, target, a, "minus", b)
                    macros[f"{base}Delta"], macros[f"{base}Low"], macros[f"{base}High"] = (f"{s[k]:+.3f}" for k in ("delta", "ci_low", "ci_high"))
    return macros


def distance_macros():
    """Recall of the validation-selected threshold by distance to the nearest day-t fire, for XGBoost."""
    macros = {}
    bins = {"0.5-1.5 km": "Near", "1.5-3.5 km": "Mid", "3.5-7 km": "Far"}
    for protocol in REPORTED_PROTOCOLS:
        by_distance = read_json(OUT_DIR, protocol, "error_analysis.json")["by_distance"]["xgboost"]
        for key, name in bins.items():
            macros[macro_name(protocol, "recall", name)] = f"{by_distance[key]['recall']:.2f}"
    return macros


def numbers_and_macros(table, fusions, leak, seeds, ablations, sigs, tabular):
    numbers, macros = {}, {**data_macros(), **comparison_macros(sigs), **distance_macros()}
    active = table[table.target == "active"]
    for r in table.to_dict("records"):
        slot = numbers.setdefault(r["protocol"], {}).setdefault(r["regime"], {}).setdefault(r["target"], {})
        slot[r["model"]] = {k: (None if pd.isna(v) else v) for k, v in r.items() if k not in ("protocol", "regime", "target", "model")}
    for r in active.to_dict("records"):
        for key in MACROS:
            if key in r and pd.notna(r[key]):
                digits = 1 if key == "pr_auc_lift" else 3
                suffix = key.replace("pr_auc_", "") if key != "pr_auc" else "pr"
                macros[macro_name(r["protocol"], r["regime"], r["model"], suffix)] = f"{r[key]:.{digits}f}"
    for protocol in REPORTED_PROTOCOLS:
        full = active[(active.protocol == protocol) & (active.regime == "full")].iloc[0]
        macros[macro_name(protocol, "positive", "rate")] = f"{100 * full.positive_rate:.2f}"
        macros[macro_name(protocol, "hybrid", "selected")] = LABELS[fusions[protocol]["full"]["selected_hybrid"]]
        macros[macro_name(protocol, "best", "tree")] = LABELS[fusions[protocol]["best_tree"]]
        t = leak[protocol]
        macros[macro_name(protocol, "leak", "features")] = f"{t['add_position']['test']['pr_auc_features']:.3f}"
        macros[macro_name(protocol, "leak", "position")] = f"{t['add_position']['test']['pr_auc_features_plus_position']:.3f}"
        macros[macro_name(protocol, "leak", "recover")] = f"{t['recover_position']['test_roc_auc']:.3f}"
    for r in seeds.to_dict("records"):
        macros[macro_name(r["protocol"], r["model"], "seed", "mean")] = f"{r['mean_test_pr_auc']:.3f}"
    macros["rSeedSdMin"], macros["rSeedSdMax"] = f"{seeds.std_test_pr_auc.min():.3f}", f"{seeds.std_test_pr_auc.max():.3f}"
    for name, value in ablations.val_pr_auc.items():
        macros[macro_name("ablation", name)] = f"{value:.3f}"
    for protocol, results in tabular.items():
        for key in ("raw_only", "fire_only", "all_features", "all_minus_fire", "all_minus_wind"):
            macros[macro_name(protocol, "tabular", key)] = f"{results[key]['test_pr_auc']:.3f}"
    numbers["leak"] = leak
    numbers["seed_stability"] = seeds.to_dict("records")
    with open(os.path.join(OUT_DIR, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    with open(os.path.join(REPORT_DIR, "generated_results.tex"), "w") as f:
        f.write("% Generated by python -m src.evaluation.report_assets. Do not edit by hand.\n")
        f.writelines(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
    return numbers


def main():
    table = pd.read_csv(os.path.join(OUT_DIR, "metrics.csv"))
    seeds = pd.read_csv(os.path.join(OUT_DIR, "seed_stability.csv"))
    fusions = {p: read_json(EXP_DIR, "hybrid", p, "fusion.json") for p in REPORTED_PROTOCOLS}
    result_tables(table)
    seed_table(seeds)
    sigs = read_json(OUT_DIR, "significance.json")
    significance_table(sigs)
    significance_table(sigs, regime="crop", name="significance_crop")
    fusion_table(fusions)
    sweep_outputs()
    ablations = dl_ablation_outputs()
    tabular = tabular_ablation_table()
    leak = leak_table()
    phase1 = phase1_change_table()
    numbers = numbers_and_macros(table, fusions, leak, seeds, ablations, sigs, tabular)
    numbers["phase1_changes"] = phase1
    with open(os.path.join(OUT_DIR, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)


if __name__ == "__main__":
    main()
