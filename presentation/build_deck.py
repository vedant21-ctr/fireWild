"""
build_deck.py - Render the final slide deck to PDF and PPTX.

Every result on the slides is read from report/generated_results.tex, which src/evaluation/report_assets.py writes
from the results files. Figures come from experiments/ndws/. Headless Google Chrome renders the slides to:
    presentation/fireWild_final.pdf    vector PDF, one 16:9 page per slide
    presentation/fireWild_final.pptx   16:9 PowerPoint, one full-slide image per slide, speaker notes included

Usage (macOS, Google Chrome installed):
    python presentation/build_deck.py
"""

import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
W, H = 1920, 1080
FONTS = "https://fonts.googleapis.com/css2?family=EB+Garamond:wght@500;600&family=IBM+Plex+Sans:wght@400;600&display=swap"
TITLE = "Next-Day Wildfire Spread Prediction"
INK, ACCENT, MUTED, PAPER, RULE = "#1f2a33", "#b0432c", "#6a7178", "#f7f5f0", "#d6d1c4"

CSS = f"""
@page {{ size: {W}px {H}px; margin: 0 }}
html, body {{ margin: 0; padding: 0; background: {INK} }}
section {{ position: relative; width: {W}px; height: {H}px; box-sizing: border-box; overflow: hidden; display: flex; flex-direction: column;
          gap: 32px; padding: 72px 112px 96px; background: {PAPER}; color: {INK}; font: 36px/1.4 'IBM Plex Sans', Arial, sans-serif;
          page-break-after: always; break-after: page }}
section * {{ box-sizing: border-box }}
h1, h2, p, ul, table {{ margin: 0 }}
.kicker {{ font-size: 24px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; color: {ACCENT} }}
h1 {{ font: 600 104px/1.05 'EB Garamond', Georgia, serif }}
h2 {{ font: 600 62px/1.1 'EB Garamond', Georgia, serif }}
.head {{ display: flex; flex-direction: column; gap: 10px }}
.row {{ display: flex; gap: 56px; align-items: center; flex: 1; min-height: 0 }}
.col {{ display: flex; flex-direction: column; gap: 22px; flex: 1; min-width: 0 }}
.fig {{ flex: 1; min-width: 0; min-height: 0; max-height: 100%; object-fit: contain; background: #fff; border: 1px solid {RULE}; padding: 12px }}
.full {{ flex: 1; min-height: 0; width: 100%; object-fit: contain; background: #fff; border: 1px solid {RULE}; padding: 12px }}
ul {{ padding-left: 1.1em; display: flex; flex-direction: column; gap: 16px }}
table {{ border-collapse: collapse; width: 100% }}
th, td {{ padding: 9px 18px; border-bottom: 1px solid {RULE}; text-align: left; font-size: 31px }}
th {{ font-weight: 600; color: {MUTED}; font-size: 26px }}
td.best {{ font-weight: 600; color: {ACCENT} }}
.big {{ font: 600 84px/1 'EB Garamond', Georgia, serif; color: {ACCENT} }}
.foot {{ position: absolute; left: 112px; right: 112px; bottom: 40px; font-size: 22px; color: {MUTED} }}
aside {{ display: none }}
"""


def load_numbers():
    with open(os.path.join(BASE_DIR, "report", "generated_results.tex")) as f:
        return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", f.read()))


def figure(path, css="fig"):
    return f'<img class="{css}" src="file://{os.path.join(BASE_DIR, "experiments", "ndws", path)}">'


def bullets(*items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def table(header, rows):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def score_rows(n, models):
    """PR-AUC [95% CI] per protocol, marking the highest score in each column."""
    protocols = ("Official", "Spatial", "Strict")
    best = {p: max(float(n[f"r{p}Full{m}Pr"]) for _, m in models) for p in protocols}
    rows = []
    for label, m in models:
        cells = []
        for p in protocols:
            text = f"{n[f'r{p}Full{m}Pr']} [{n[f'r{p}Full{m}CiLow']}, {n[f'r{p}Full{m}CiHigh']}]"
            cells.append(f'<td class="best">{text}</td>' if float(n[f"r{p}Full{m}Pr"]) == best[p] else text)
        rows.append([label, *cells])
    return rows


def slides(n):
    s = []

    def add(kicker, title, body, notes, source=""):
        s.append((kicker, title, body, notes, source))

    add("The task", "Which pixels will be burning tomorrow?",
        '<div class="row"><div class="col">' + bullets(
            "For each 1 km pixel around an active fire, estimate the chance it burns on day <i>t</i>+1",
            "Inputs: 11 environmental rasters and the fire map from day <i>t</i>",
            f"Only {n['rTrainPositiveRate']}% of training pixels are positive, so accuracy is useless. We use PR-AUC",
            "Two families compared on the same pixels: trees on hand-made features, and CNN and U-Net on images",
        ) + "</div></div>",
        "State the question simply. A model that always says no fire would be about 99% accurate and useless, which is why PR-AUC is the main metric.")

    add("Data", "NDWS: one day of inputs, the next day's fire",
        '<div class="row">' + figure("eda/example_patches.png") + '<div class="col" style="flex:.55">' + bullets(
            f"{n['rPatches']} patches of 64 x 64 km, US, 2012 to 2020",
            "Labels: fire, no fire, or unknown (excluded)",
            "No coordinates, dates or event IDs in the release",
        ) + "</div></div>",
        "Walk through one row of the figure. Stress that the release has no coordinates, dates or fire IDs, which limits what we can claim.",
        "experiments/ndws/eda/example_patches.png")

    add("What the data shows", "Fire mostly appears close to fire",
        '<div class="row">' + figure("eda/spread_vs_distance.png") + figure("eda/wind_orientation.png") + "</div>"
        f"<p>{n['rNewFireShare']}% of next-day fire pixels were not burning on day <i>t</i>. Persistence alone recalls {n['rPersistenceRecall']}% of them. "
        f"New fire is {n['rWindRatio']} times as likely downwind as upwind (95% CI {n['rWindLow']} to {n['rWindHigh']}).</p>",
        "The left chart motivates the distance features. The right chart is how we inferred image orientation, since the data has no georeferencing.",
        "experiments/ndws/eda/")

    add("Honest evaluation", "Three test splits, three different questions",
        '<div class="row"><div class="col">' + table(
            ["Protocol", "Weeks kept apart", "Terrain kept apart", "Train / val / test patches"],
            [["Official", "Yes", "No", f"{n['rOfficialTrain']} / {n['rOfficialVal']} / {n['rOfficialTest']}"],
             ["Spatial", "No", "Yes", f"{n['rSpatialTrain']} / {n['rSpatialVal']} / {n['rSpatialTest']}"],
             ["Strict", "Yes", "Yes", f"{n['rStrictTrain']} / {n['rStrictVal']} / {n['rStrictTest']}"]]) +
        f"<p>Identical terrain shows about <b>{n['rOfficialOverlap']}%</b> of official test patches overlap training locations. "
        f"The strict test set is small ({n['rStrictTest']} patches), so its intervals are wide.</p></div></div>",
        "Explain the terrain fingerprint: elevation barely changes between days, so identical terrain means the same place. Strict is the hardest test, and the smallest.",
        "data/splits/split_audit.json")

    add("A hidden shortcut", "Patches are centred on fire, so position leaks the answer",
        '<div class="row"><div class="col">' + bullets(
            f"On full patches, a prior using only position scores {n['rStrictFullCentrePriorPr']} on strict, above every valid model",
            f"A U-Net trained on full patches gains {n['rOfficialFullActiveUnetFullpatchMinusUnetDelta']}, "
            f"{n['rSpatialFullActiveUnetFullpatchMinusUnetDelta']} and {n['rStrictFullActiveUnetFullpatchMinusUnetDelta']} over random crops",
            "So networks train on random 32 x 32 crops",
            "Every model is also scored on fixed random windows where the centre is not special",
        ) + '</div><div class="col">' + table(
            ["LightGBM test PR-AUC", "Official", "Spatial", "Strict"],
            [["Features", n["rOfficialLeakFeatures"], n["rSpatialLeakFeatures"], n["rStrictLeakFeatures"]],
             ["+ position", n["rOfficialLeakPosition"], n["rSpatialLeakPosition"], n["rStrictLeakPosition"]]]) +
        "<p>Adding position helps a lot, so the tabular features do not already contain all of it.</p></div></div>",
        "Full-patch results are diagnostics and not valid forecasts. In a real forecast over a fixed map, the centre of an image means nothing.",
        "src/aml/leak_test.py, experiments/ndws/results/*/leak_test.json")

    add("Models", "Trees, networks and hybrids",
        '<div class="row"><div class="col">' + table(["Family", "What it sees", "Models"], [
            ["Baselines", "Trivial rules", "Dummy, persistence, neighbourhood persistence"],
            ["Tabular", "32 features per pixel", "Logistic regression, random forest, XGBoost, LightGBM"],
            ["Spatial", "Image patches", f"Compact CNN ({n['rCnnParameters']} params), U-Net ({n['rUnetParameters']} params)"],
            ["Hybrid", "Both", "U-Net with 31 channels, and late fusion of tree and network"]]) +
        "<p>U-Net settings were chosen on a separate tuning split: weighted cross-entropy (weight 10), dropout 0.2. "
        "Fusion weights use half of validation, and the other half picks the hybrid.</p></div></div>",
        "Emphasise that tuning and hybrid selection never see test data.",
        "configs/dl.yaml, experiments/ndws/dl/dev/selected.json")

    models = [("Dummy", "Dummy"), ("Random forest", "RandomForest"), ("XGBoost", "Xgboost"), ("LightGBM", "Lightgbm"), ("Compact CNN", "Cnn"),
              ("U-Net", "Unet"), ("U-Net + 31 channels", "UnetFeatures"), ("Hybrid (selected on validation)", "Hybrid")]
    add("Results", "No network or hybrid beats the best tree",
        '<div class="col">' + table(["Test PR-AUC [95% CI]", "Official", "Spatial", "Strict"], score_rows(n, models)) + "</div>"
        f"<p>Positive rates differ across splits ({n['rOfficialPositiveRate']}%, {n['rSpatialPositiveRate']}%, {n['rStrictPositiveRate']}%), so compare within a column. "
        "Network rows use the first of three training seeds.</p>",
        "Give the headline first: the best tree wins every column. The intervals overlap, so the next slide uses paired tests. Report this honestly, as the brief asks.",
        "experiments/ndws/final/metrics.csv")

    add("Are the gaps real?", "Paired tests against the validation-selected tree (XGBoost)",
        '<div class="row"><div class="col">' + table(["Difference in PR-AUC", "Official", "Spatial", "Strict"], [
            ["U-Net - XGBoost", n["rOfficialFullActiveUnetMinusXgboostDelta"], n["rSpatialFullActiveUnetMinusXgboostDelta"], n["rStrictFullActiveUnetMinusXgboostDelta"]],
            ["CNN - XGBoost", n["rOfficialFullActiveCnnMinusXgboostDelta"], n["rSpatialFullActiveCnnMinusXgboostDelta"], n["rStrictFullActiveCnnMinusXgboostDelta"]],
            ["Hybrid - XGBoost", n["rOfficialFullActiveHybridMinusXgboostDelta"], n["rSpatialFullActiveHybridMinusXgboostDelta"], n["rStrictFullActiveHybridMinusXgboostDelta"]],
            ["Hybrid - U-Net", n["rOfficialFullActiveHybridMinusUnetDelta"], n["rSpatialFullActiveHybridMinusUnetDelta"], n["rStrictFullActiveHybridMinusUnetDelta"]]]) +
        bullets("U-Net and CNN are significantly below XGBoost on all three splits",
                "The hybrid is significantly below XGBoost on official, and indistinguishable on spatial and strict",
                "The hybrid significantly improves on the U-Net on spatial and strict") + "</div></div>",
        "Paired bootstrap on the same resampled patches. Intervals for each difference are in report tables. Strict's best test tree is LightGBM, but the test uses the tree chosen on validation.",
        "experiments/ndws/final/significance.json")

    add("Without the centre", "Fixed windows close the gap on official and spatial, not on strict",
        '<div class="row"><div class="col">' + table(["Test PR-AUC on fixed windows", "Official", "Spatial", "Strict"], [
            ["LightGBM (window-trained)", n["rOfficialCropLightgbmCropPr"], n["rSpatialCropLightgbmCropPr"], n["rStrictCropLightgbmCropPr"]],
            ["U-Net", n["rOfficialCropUnetPr"], n["rSpatialCropUnetPr"], n["rStrictCropUnetPr"]],
            ["Hybrid", n["rOfficialCropHybridPr"], n["rSpatialCropHybridPr"], n["rStrictCropHybridPr"]],
            ["Position prior only", n["rOfficialCropPositionPriorPr"], n["rSpatialCropPositionPriorPr"], n["rStrictCropPositionPriorPr"]]]) +
        bullets(f"Official: the hybrid beats the window-trained tree by {n['rOfficialCropActiveHybridMinusLightgbmCropDelta']}, the only significant hybrid win over a tree",
                f"Strict: the U-Net trails by {n['rStrictCropActiveUnetMinusLightgbmCropDelta']} and the hybrid by {n['rStrictCropActiveHybridMinusLightgbmCropDelta']}") + "</div></div>",
        "Here position carries almost no information. On official the U-Net and hybrid match or beat the window-trained tree, on spatial they are within its interval, and on strict the tree is still ahead.",
        "experiments/ndws/final/metrics.csv")

    add("Stability", "Three seeds: small gaps are within training noise",
        '<div class="row"><div class="col">' + table(["Mean test PR-AUC", "Official", "Spatial", "Strict"], [
            ["Compact CNN", n["rOfficialCnnSeedMean"], n["rSpatialCnnSeedMean"], n["rStrictCnnSeedMean"]],
            ["U-Net", n["rOfficialUnetSeedMean"], n["rSpatialUnetSeedMean"], n["rStrictUnetSeedMean"]],
            ["U-Net + 31 channels", n["rOfficialUnetFeaturesSeedMean"], n["rSpatialUnetFeaturesSeedMean"], n["rStrictUnetFeaturesSeedMean"]]]) +
        f"<p>Seed standard deviation runs from {n['rSeedSdMin']} to {n['rSeedSdMax']}. The 31-channel U-Net's lead on strict has the largest spread, so we treat it as suggestive only.</p></div></div>",
        "These are means of three training runs, separate from the bootstrap on test patches.",
        "experiments/ndws/final/seed_stability.csv")

    add("What matters", "Today's fire map dominates",
        '<div class="row">' + figure("final/dl_ablations.png") + '<div class="col" style="flex:.7">' + bullets(
            f"Without day-<i>t</i> fire channels: validation PR-AUC {n['rAblationBaseline']} falls to {n['rAblationNoPrevFireChannel']}",
            f"Tabular on strict: raw values only {n['rStrictTabularRawOnly']}, fire context only {n['rStrictTabularFireOnly']}",
            f"Removing augmentation gave {n['rAblationNoAugmentation']}, so we do not claim it helped",
        ) + "</div></div>",
        "Single-run ablations on the tuning split. Only the fire channels produce a large effect, so do not over-read the smaller bars.",
        "experiments/ndws/dl/dev/ablations.csv")

    add("Prediction maps", "Actual, trees, networks, hybrid, and errors (strict test)",
        figure("final/strict/prediction_maps.png", "full"),
        "Read one row from left to right. Trees and hybrid stay close to fire. Networks give smoother, wider areas. False negatives are mostly new fire away from the perimeter.",
        "experiments/ndws/final/strict/prediction_maps.png")

    add("Where models fail", "Fire far from today's perimeter is missed",
        '<div class="row">' + figure("final/strict/error_by_distance.png") + '<div class="col" style="flex:.6">' + table(["XGBoost recall", "Under 1.5 km", "1.5-3.5", "3.5-7"], [
            ["Official", n["rOfficialRecallNear"], n["rOfficialRecallMid"], n["rOfficialRecallFar"]],
            ["Spatial", n["rSpatialRecallNear"], n["rSpatialRecallMid"], n["rSpatialRecallFar"]],
            ["Strict", n["rStrictRecallNear"], n["rStrictRecallMid"], n["rStrictRecallFar"]]]) + "</div></div>",
        "Recall at the validation-chosen threshold, by distance to the nearest day-t fire pixel.",
        "experiments/ndws/final/*/error_analysis.json")

    add("Since Phase 1", "What changed, and why",
        '<div class="row"><div class="col">' + bullets(
            "A full re-run reproduces the graded Phase 1 official and spatial scores to three decimals",
            "New <b>strict split</b>: the Phase 1 spatial split still mixed weeks between training and test",
            "New <b>position-leak test</b>: Phase 1 did not test whether tabular models exploit position",
            "New <b>Dummy baseline, CNN, tuned U-Net and hybrids</b> to complete the comparison",
        ) + "</div></div>",
        "The graders asked for this. The rerun matching Phase 1 shows the pipeline is reproducible.",
        "report/tables/phase1_changes.tex")

    add("Limits", "What this work cannot claim",
        '<div class="row"><div class="col">' + bullets(
            "Weeks are randomly assigned, so this is not forecasting into future seasons",
            "No coordinates, dates, fuel moisture or suppression data",
            "Terrain matching only finds exact overlaps, so nearby fires may still cross splits",
            f"The strict test has only {n['rStrictTest']} patches",
            "Samples exist only where fire is present, so this is not monitoring of all land",
        ) + "</div></div>",
        "Say these before the examiner does. Associations do not prove how a fire physically spread.")

    add("Conclusion", "Well-built features still lead on this benchmark",
        '<div class="row"><div class="col">' + bullets(
            f"Best test PR-AUC: XGBoost {n['rOfficialFullXgboostPr']} (official) and {n['rSpatialFullXgboostPr']} (spatial), LightGBM {n['rStrictFullLightgbmPr']} (strict)",
            "The hybrid improved on the U-Net but did not beat the best tree",
            "Benchmark scores depend strongly on how data is split and sampled",
            "Next: WildfireSpreadTS, for real dates and multi-day models",
        ) + "</div></div>",
        "End with the honest conclusion. Offer the viva guide and repository for detail.")
    return s


def render(slide_list):
    pages = [f"""<section><div style="flex:1;display:flex;flex-direction:column;justify-content:center;gap:28px">
<p class="kicker">Project 17 &middot; Final submission</p><h1>{TITLE}</h1>
<p style="font-size:38px;color:{MUTED}">Tabular machine learning and spatial deep learning, compared with care for leakage</p>
<p>Vedant Satbhai &middot; Pratik Agade &middot; Lakshay Saharan &middot; Raghvendra Singh &middot; Aaryan Krishna</p>
<p style="color:{MUTED}">CSAI, Newton School of Technology, Rishihood University</p></div>
<aside>Introduce the team and the single question: can we tell where a wildfire will be tomorrow, and how far can we trust the score?</aside></section>"""]
    for i, (kicker, title, body, notes, source) in enumerate(slide_list, start=2):
        foot = f"{i}" + (f" &middot; {source}" if source else "")
        pages.append(f'<section><div class="head"><p class="kicker">{kicker}</p><h2>{title}</h2></div>{body}<p class="foot">{foot}</p><aside>{notes}</aside></section>')
    return pages


def document(sections):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{TITLE}</title><link rel="stylesheet" href="{FONTS}">'
            f"<style>{CSS}</style></head><body>{''.join(sections)}</body></html>")


def chrome(*args):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    "--virtual-time-budget=8000", *args], check=True, capture_output=True)


def main():
    sections = render(slides(load_numbers()))
    out_pdf, out_pptx = os.path.join(HERE, "fireWild_final.pdf"), os.path.join(HERE, "fireWild_final.pptx")
    tmp = tempfile.mkdtemp()
    try:
        full = os.path.join(tmp, "deck.html")
        with open(full, "w") as f:
            f.write(document(sections))
        chrome("--no-pdf-header-footer", f"--print-to-pdf={out_pdf}", "file://" + full)

        from pptx import Presentation
        from pptx.util import Emu
        prs = Presentation()
        prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 13.333 x 7.5 in (16:9)
        for i, section in enumerate(sections):
            one, png = os.path.join(tmp, f"{i}.html"), os.path.join(tmp, f"{i}.png")
            with open(one, "w") as f:
                f.write(document([section]))
            chrome(f"--window-size={W},{H}", f"--screenshot={png}", "file://" + one)
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            slide.shapes.add_picture(png, 0, 0, width=prs.slide_width, height=prs.slide_height)
            slide.notes_slide.notes_text_frame.text = re.search(r"<aside>(.*?)</aside>", section, re.S).group(1)
        prs.save(out_pptx)
        print(f"wrote {out_pdf} and {out_pptx} ({len(sections)} slides)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
