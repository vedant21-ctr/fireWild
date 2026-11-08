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
TITLE = "Predicting Tomorrow's Wildfire"
INK, ACCENT, MUTED, PAPER, RULE, CARD = "#1f2a33", "#b0432c", "#6a7178", "#f7f5f0", "#d6d1c4", "#ffffff"

CSS = f"""
@page {{ size: {W}px {H}px; margin: 0 }}
html, body {{ margin: 0; padding: 0; background: {INK} }}
section {{ position: relative; width: {W}px; height: {H}px; box-sizing: border-box; overflow: hidden; display: flex; flex-direction: column;
          gap: 44px; padding: 80px 112px 96px; background: {PAPER}; color: {INK}; font: 42px/1.35 'IBM Plex Sans', Arial, sans-serif;
          page-break-after: always; break-after: page }}
section * {{ box-sizing: border-box }}
h1, h2, h3, p, ul, table {{ margin: 0 }}
h1 {{ font: 600 124px/1.05 'EB Garamond', Georgia, serif }}
h2 {{ font: 600 76px/1.1 'EB Garamond', Georgia, serif }}
h3 {{ font: 600 46px/1.2 'IBM Plex Sans', Arial, sans-serif }}
.row {{ display: flex; gap: 40px; align-items: stretch; flex: 1; min-height: 0 }}
.center {{ align-items: center }}
.card {{ flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 22px; justify-content: center; background: {CARD}; border: 1px solid {RULE}; padding: 44px 40px }}
.card p {{ font-size: 37px; color: #3e4a54 }}
.arrow {{ align-self: center; font-size: 84px; color: {ACCENT} }}
.big {{ font: 600 132px/1 'EB Garamond', Georgia, serif; color: {ACCENT} }}
.note {{ color: #3e4a54 }}
.full {{ flex: 1; min-height: 0; width: 100%; object-fit: contain; background: #fff; border: 1px solid {RULE}; padding: 12px }}
ul {{ padding-left: 1.1em; display: flex; flex-direction: column; gap: 26px }}
table {{ border-collapse: collapse; width: 100% }}
th, td {{ padding: 26px 24px; border-bottom: 1px solid {RULE}; text-align: left; font-size: 46px }}
th {{ font-weight: 600; color: {MUTED}; font-size: 34px }}
td.best {{ font-weight: 600; color: {ACCENT} }}
.foot {{ position: absolute; right: 112px; bottom: 36px; font-size: 26px; color: {MUTED} }}
aside {{ display: none }}
"""


def load_numbers():
    with open(os.path.join(BASE_DIR, "report", "generated_results.tex")) as f:
        return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", f.read()))


def figure(path):
    return f'<img class="full" src="file://{os.path.join(BASE_DIR, "experiments", "ndws", path)}">'


def card(title, text):
    return f'<div class="card"><h3>{title}</h3><p>{text}</p></div>'


def stat(number, text):
    return f'<div class="card"><div class="big">{number}</div><p>{text}</p></div>'


def cards(*items):
    return '<div class="row">' + "".join(items) + "</div>"


def bullets(*items):
    return '<div class="row center"><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>"


def percent(value):
    return f"{100 * float(value):.0f}%"


def best_tree(n, protocol):
    return max(n[f"r{protocol}Full{m}Pr"] for m in ("RandomForest", "Xgboost", "Lightgbm"))


def results_table(n):
    protocols = ("Official", "Spatial", "Strict")
    head = "<tr><th></th><th>New weeks</th><th>New places</th><th>New weeks and places</th></tr>"
    tree = "".join(f'<td class="best">{best_tree(n, p)}</td>' for p in protocols)
    rows = f"<tr><td>Best tree model</td>{tree}</tr>"
    for label, model in (("U-Net", "Unet"), ("Hybrid", "Hybrid")):
        rows += f"<tr><td>{label}</td>" + "".join(f"<td>{n[f'r{p}Full{model}Pr']}</td>" for p in protocols) + "</tr>"
    return f'<div class="row center"><table>{head}{rows}</table></div>'


def slides(n):
    accuracy = f"{100 - float(n['rTrainPositiveRate']):.0f}%"
    return [
        ("Can we predict where a wildfire will burn tomorrow?",
         cards(card("Today", "Where the fire is, the weather, the land and the vegetation"), '<div class="arrow">&rarr;</div>',
               card("Our models", "Learn from thousands of past fires"), '<div class="arrow">&rarr;</div>',
               card("Tomorrow", "The chance that each 1 km square is on fire")),
         "Say: Our question is simple. If we know where a fire is today, along with the weather and land, can we predict which squares will be burning tomorrow?"),
        ("Our data: real wildfires seen from satellites",
         cards(stat(n["rPatches"], "snapshots of US wildfires, 2012 to 2020"), stat("64 km", "wide area in each snapshot, split into 1 km squares"),
               stat(f"{float(n['rTrainPositiveRate']):.0f}%", "of squares are burning the next day")),
         "Say: Each snapshot shows one fire today and the same area tomorrow. Fire is rare: only about one square in a hundred is burning next day."),
        ("Fire is rare, so accuracy would fool us",
         cards(stat(accuracy, "accuracy for a model that always says &ldquo;no fire&rdquo;. It sounds great, but it is useless."),
               card("So we use PR-AUC", f"It measures how well a model finds real fires without too many false alarms. Higher is better. Random guessing scores about {n['rOfficialFullDummyPr']}.")),
         "Say: A lazy model that always says no fire is almost always right, and helps nobody. PR-AUC rewards a model only for finding the rare fires."),
        ("We compared three kinds of model",
         cards(card("Tree models", "Look at one square at a time, using 32 facts we designed, like distance to the fire and wind direction."),
               card("Neural networks", "A CNN and a U-Net look at the whole map and learn patterns themselves."),
               card("Hybrid", "Combines a tree model and a network, to see if each can help the other.")),
         "Say: Trees depend on the facts we give them. Networks work out their own patterns from the map. The hybrid tries to get the best of both."),
        ("How the U-Net reads a map",
         cards(card("1. Shrink", "Compress the map to understand the big picture."), '<div class="arrow">&rarr;</div>',
               card("2. Grow back", "Expand to full size, giving an answer for every square."), '<div class="arrow">&rarr;</div>',
               card("3. Keep the detail", "Skip connections carry fine detail, like the fire's edge, straight across.")),
         "Say: The U-Net first zooms out to see the whole fire, then zooms back in. Shortcuts in the network stop it losing the sharp detail at the fire's edge."),
        ("We tested on data the models had never seen",
         cards(card("New weeks", "The standard test. But about " + n["rOfficialOverlap"] + "% of its areas also appear in training."),
               card("New places", "Test areas never appear in training."), card("New weeks and places", "The hardest test, but also the smallest, with only " + n["rStrictTest"] + " snapshots.")),
         "Say: We found most areas in the standard test also appear in training. So we added two harder tests where the model has never seen that land."),
        ("We blocked a hidden shortcut",
         cards(stat(n["rOfficialFullUnetPr"], "U-Net, trained on random crops so it cannot use position"), stat(n["rOfficialFullUnetFullpatchPr"], "U-Net on full images. Higher, but only because it learns &ldquo;the middle burns&rdquo;.")),
         "Say: Every snapshot is centred on a fire, so a model can cheat by learning that the middle usually burns. That would not work in real life, so we train on random crops, and we report the fair score. Numbers are for the new-weeks test."),
        ("Result: tree models won all three tests", results_table(n) + '<p class="note">Higher is better. Gaps are small. The U-Net is really behind; on the two harder tests the hybrid is level with XGBoost.</p>',
         "Say: This is the main result. The best tree model scored highest on every test. The networks were close, but did not beat it. We report this honestly, even though networks are the fancier approach."),
        ("The hybrid helped the U-Net, but not enough",
         cards(stat(n["rSpatialFullActiveHybridMinusUnetDelta"], "better than the U-Net on new places"), stat(n["rStrictFullActiveHybridMinusUnetDelta"], "better than the U-Net on new weeks and places"),
               card("But", "It did not beat the best tree model on any test.")),
         "Say: Combining trees and networks made the network better on the harder tests. But it still did not beat the tree models, so we do not claim that the hybrid is the winner."),
        ("Today's fire map matters most",
         cards(stat(n["rAblationBaseline"], "U-Net score with every input"), '<div class="arrow">&rarr;</div>', stat(n["rAblationNoPrevFireChannel"], "after removing today's fire map")),
         "Say: We removed inputs one at a time to see what matters. Without today's fire map the model almost stops working. Removing other inputs changed the score only a little."),
        ("One example: the fire moved",
         figure("final/strict/prediction_map_example.png") + '<p class="note">The models expect fire to stay close to where it is today, so they miss fire that jumps further away.</p>',
         "Say: Read this from left to right: today's fire, tomorrow's real fire, then each model's guess, then the mistakes. Models are good near today's fire and weak when fire moves far."),
        ("What we cannot claim",
         bullets("The data has no real dates, so this does not prove we can forecast future seasons", "Every snapshot already contains a fire, so this is not a search for new fires",
                 "It shows patterns in the data, not why a fire spread"),
         "Say: We want to be clear about the limits. These are honest boundaries of the dataset, and the report covers them in detail."),
        ("What we learned",
         bullets("Well-designed features can match or beat neural networks on this problem", "How you test a model changes what you can conclude", "Combining models helped the network, but the simple tree models stayed on top"),
         "Say: Three things to remember. Good features still matter. Fair testing matters. And a more complex model is not automatically a better one. Thank you."),
    ]


def render(slide_list):
    pages = [f"""<section><div style="flex:1;display:flex;flex-direction:column;justify-content:center;gap:36px">
<h1>{TITLE}</h1>
<p style="color:{MUTED}">Comparing tree models and neural networks on real wildfire data</p>
<p style="font-size:36px">Vedant Satbhai &middot; Pratik Agade &middot; Lakshay Saharan &middot; Raghvendra Singh &middot; Aaryan Krishna</p>
<p style="font-size:32px;color:{MUTED}">CSAI, Newton School of Technology, Rishihood University</p></div>
<aside>Say: We are presenting our project on predicting where a wildfire will burn tomorrow.</aside></section>"""]
    for i, (title, body, notes) in enumerate(slide_list, start=2):
        pages.append(f'<section><h2>{title}</h2>{body}<p class="foot">{i}</p><aside>{notes}</aside></section>')
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
