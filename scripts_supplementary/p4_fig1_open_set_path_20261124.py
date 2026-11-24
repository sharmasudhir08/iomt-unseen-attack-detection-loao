"""Paper 4 Fig. 1, alternative drawing (no data, no numbers): the leave-one-attack-out open-set decision path.
Differs from Paper 1 Fig. 1 (end-to-end methodology with data preparation, statistics and the H1 to H7 verdict panel):
left-to-right three-step flow of this paper's evaluation (training without the held-out class, scores of each test window, open-set decision),
the model collapsed to one box, outline boxes with a teal / amber / slate palette instead of filled blue, purple and green cards.
Facts (all from Sections III-A, III-B and III-D of the manuscript): window of 16 flows x 45 features; top-1 mixture of experts (4 experts);
2-, 6- and 19-class heads; energy s_E = logsumexp(z) and MSP from the 19-class logits; threshold = 5th percentile of validation energy
(known classes only); caught-any = flagged unknown OR binary head says Attack. Print width 17.8 cm, all text 8 pt.
Output: documents_v2/paper4/figures_v6/fig1_open_set_path.pdf (+ preview PNG). CPU only."""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents_v2" / "paper4" / "figures_v6"
FS = 8.0
W, H = 17.8 / 2.54, 3.55
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": FS, "pdf.fonttype": 42})
TEAL, TEAL_F = "#1b7f79", "#eaf4f3"
AMB, AMB_F = "#b8860b", "#fdf3dc"
SLATE, SLATE_F = "#53616f", "#f0f2f5"
GREY = "#8a8f94"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
boxes = []


def box(x, y, w, h, text, edge, face, bold=False, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", linewidth=1.0, edgecolor=edge, facecolor=face, linestyle="--" if dashed else "-")
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, fontweight="bold" if bold else "normal", linespacing=1.25)
    boxes.append((t, (x, y, w, h)))
    return p


def arrow(x0, y0, x1, y1, color=SLATE, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=color, linestyle=ls, shrinkA=0, shrinkB=0))


c1, c2, c3 = 0.05, 2.32, 4.52
cw1, cw2, cw3 = 2.0, 1.95, 2.43
assert c3 + cw3 <= W - 0.04, (c3 + cw3, W)
for x, w, s in ((c1, cw1, "1  Training (leave-one-attack-out)"), (c2, cw2, "2  Scores of each test window"), (c3, cw3, "3  Open-set decision")):
    ax.text(x, H - 0.12, s, ha="left", va="center", fontsize=FS, fontweight="bold", color="black")
    ax.plot([x, x + w], [H - 0.27, H - 0.27], color=GREY, lw=0.8)

# column 1
box(c1, 2.3, cw1, 0.87, "Known classes only: training\nand validation windows; the\nheld-out attack class is removed", SLATE, SLATE_F)
box(c1, 0.08, cw1, 2.0, "Model, trained once per held-out\nclass and seed: window encoder\n(16 flows, 45 features), mixture-of-\nexperts layer (4 experts, top-1)\nand three heads (2, 6 and 19\nclasses)\n\nThe two baselines follow the\nsame training protocol", TEAL, TEAL_F)
arrow(c1 + cw1 / 2, 2.3, c1 + cw1 / 2, 2.08, SLATE)
# column 2
box(c2, 1.78, cw2, 0.62, "Energy score\ns$_E$ = logsumexp(z)", AMB, AMB_F, bold=True)
box(c2, 0.98, cw2, 0.62, "MSP baseline\nmax softmax(z)", GREY, "white")
box(c2, 0.08, cw2, 0.7, "Binary head\nP(Attack) > 0.5", TEAL, TEAL_F)
ax.text(c2 + cw2 / 2, 2.62, "z: 19-class logits", ha="center", va="center", fontsize=FS, style="italic")
arrow(c1 + cw1, 2.09, c2, 2.09, TEAL)
arrow(c1 + cw1, 1.29, c2, 1.29, TEAL)
arrow(c1 + cw1, 0.43, c2, 0.43, TEAL)
# column 3
box(c3, 2.45, cw3, 0.72, "Held-out attack class: absent from\ntraining and validation, present\nonly in the test files", AMB, "white", dashed=True)
box(c3, 1.78, cw3, 0.62, "Threshold: 5th percentile of the\nvalidation energy (known classes)", AMB, AMB_F)
box(c3, 0.98, cw3, 0.62, "Flag as unknown if\ns$_E$ is below the threshold", AMB, AMB_F)
box(c3, 0.08, cw3, 0.7, "caught-any: flagged unknown\nOR binary head says Attack", AMB, AMB_F, bold=True)
arrow(c2 + cw2, 2.09, c3, 2.09, AMB)
arrow(c3 + cw3 / 2, 1.78, c3 + cw3 / 2, 1.6, AMB)
arrow(c3 + cw3 / 2, 0.98, c3 + cw3 / 2, 0.78, AMB)
arrow(c2 + cw2, 0.43, c3, 0.43, TEAL)

# checks
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bad = []
for t, (x, y, w, h) in boxes:
    bb = t.get_window_extent(r)
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    if x0 < x + 0.04 or x1 > x + w - 0.04 or y0 < y + 0.02 or y1 > y + h - 0.02:
        bad.append((t.get_text().split("\n")[0], round(x0 - x, 2), round(x + w - x1, 2), round(y0 - y, 2), round(y + h - y1, 2)))
sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
print("text outside its box:", bad if bad else "none")
print("smallest text", min(sizes), "pt;", len(sizes), "texts; width", round(W * 2.54, 2), "cm; height", round(H * 2.54, 2), "cm")
assert not bad and min(sizes) >= FS - 1e-6
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "preview").mkdir(exist_ok=True)
fig.savefig(OUT / "fig1_open_set_path.pdf")
fig.savefig(OUT / "preview" / "fig1_open_set_path.png", dpi=300)
print("SHA-256 script", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
