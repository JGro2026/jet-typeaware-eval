#!/usr/bin/env python3
"""Figures for the ESWA manuscript (type-aware evaluation of journal entry anomaly detectors).

Every number drawn here is parsed from the input files named in the functions below
or from the analysis outputs (outputs/public/ and restricted/outputs/). Nothing is typed in by hand.

Output per figure: <name>.pdf (vector, fonts embedded as TrueType, fonttype 42), <name>.eps and a
PNG preview <name>.png, all in paper/figures.

Width: the elsarticle preprint text width measured with \\showthe\\textwidth is 390.0 pt
(documentclass[preprint,12pt,authoryear]{elsarticle}), i.e. 5.40 in; every figure is set at that width
and is meant to be included with width=\\textwidth (scale 1), so the 8 to 9 pt font sizes are final.

Run: python3 paper/make_figures.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import paths  # noqa: E402

import math
import sys
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = paths.ROOT
OUT = paths.FIGURES

# ----------------------------------------------------------------------------------------------
# Page geometry and style
# ----------------------------------------------------------------------------------------------
TEXTWIDTH_PT = 390.0                      # measured, see module docstring
FULL = TEXTWIDTH_PT / 72.27               # 5.396 in
HALF = (TEXTWIDTH_PT - 12.0) / 2 / 72.27  # 2.615 in (two figures side by side with a 12 pt gap)

FS = 8          # base font size (pt) at final size
FS_PANEL = 9    # panel letters

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.cal": "Liberation Sans",
    "mathtext.tt": "Liberation Sans",
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#4d4d4d",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
    "xtick.labelcolor": "#1a1a1a",
    "ytick.labelcolor": "#1a1a1a",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "lines.linewidth": 1.1,
    "lines.markersize": 4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": "#1a1a1a",
})

INK = "#1a1a1a"
INK2 = "#4d4d4d"
MUTED = "#8c8c8c"
GRID = "#e3e3e3"

# Okabe-Ito hues, validated with the dataviz validator (all adjacent and all-pairs CVD checks pass).
FAM_COLOR = {"nb": "#0072B2", "iso": "#E69F00", "rec": "#009E73", "emb": "#D55E00", "ceil": "#000000"}
FAM_LABEL = {
    "nb": "Neighbour-based (kNN, LOF)",
    "iso": "Isolation and per-field (IF, HBOS, ECOD)",
    "rec": "Projection and reconstruction (PCA, AE, VAE)",
    "emb": "Embedding autoencoder (AE-emb)",
    "ceil": "LightGBM reference (supervised)",
}
DETS = ["IF", "LOF", "kNN", "HBOS", "ECOD", "PCA", "AE", "VAE", "AE-emb"]
CEIL = "LightGBM ceiling"
FAMILY = {"kNN": "nb", "LOF": "nb", "IF": "iso", "HBOS": "iso", "ECOD": "iso",
          "PCA": "rec", "AE": "rec", "VAE": "rec", "AE-emb": "emb", CEIL: "ceil"}
MARKER = {"kNN": "o", "LOF": "s", "IF": "^", "HBOS": "v", "ECOD": "D",
          "PCA": "^", "AE": "v", "VAE": "D", "AE-emb": "o", CEIL: "*"}
LSTYLE = {"kNN": "-", "LOF": (0, (4, 1.5)), "IF": "-", "HBOS": (0, (4, 1.5)), "ECOD": (0, (1, 1.2)),
          "PCA": "-", "AE": (0, (4, 1.5)), "VAE": (0, (1, 1.2)), "AE-emb": "-", CEIL: "-"}
# difficulty: one-hue ordinal ramp (validated: monotone L, light end 2.25:1) plus line style and marker
DIFF_STYLE = {
    "easy": dict(color="#56B4E9", ls="-", marker="o"),
    "medium": dict(color="#0072B2", ls=(0, (4, 1.5)), marker="s"),
    "hard": dict(color="#003A5D", ls=(0, (1, 1.2)), marker="^"),
}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"])

NAME_MAP = {"AE-pyod": "AE", "VAE-pyod": "VAE", "LGBM-ceiling": CEIL, "LightGBM ceiling": CEIL}


def norm_name(s: str) -> str:
    s = s.strip()
    return NAME_MAP.get(s, s)


def f2(x: float, nd: int = 2) -> str:
    """Round half up on the shortest decimal representation, as in tables/make_tables.py."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def fmt_int(x: float) -> str:
    from decimal import Decimal, ROUND_HALF_UP
    return f"{int(Decimal(repr(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}"


def panel_label(ax, text, x=-0.02, y=1.02, ha="left", fig_coords=False, **kw):
    """Panel identifier such as '(a) Ledger C', set above the axes (not a chart title)."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom", fontsize=FS, color=INK, **kw)


def bold_letter(letter: str, rest: str = "") -> str:
    return r"$\mathbf{(" + letter + r")}$" + ((" " + rest) if rest else "")


def text_color_for(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.5 else INK


def save(fig, name: str):
    """Save at exactly the text width: the horizontal extent is fixed to [0, W]; only empty vertical margins
    are cropped. Content that would fall outside the canvas is reported, so that the layout can be fixed
    instead of being shrunk at inclusion (which would push the fonts below 8 pt)."""
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    over = dict(left=-tb.x0, right=tb.x1 - W, bottom=-tb.y0, top=tb.y1 - H)
    bad = {k: round(v * 72, 1) for k, v in over.items() if v > 0.01}
    if bad:
        print(f"WARNING {name}: content outside the canvas (pt): {bad}")
    y0 = max(0.0, tb.y0 - 0.03)
    y1 = min(H, tb.y1 + 0.03)
    box = matplotlib.transforms.Bbox([[0, y0], [W, y1]])
    for ext in ("pdf", "eps"):
        fig.savefig(OUT / f"{name}.{ext}", format=ext, bbox_inches=box)
    fig.savefig(OUT / f"{name}.png", format="png", dpi=200, bbox_inches=box)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------------------------

def md_section(path: Path, start_pat: str) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(start_pat, text, flags=re.M)
    if not m:
        raise ValueError(f"section {start_pat!r} not found in {path}")
    rest = text[m.start():]
    nxt = re.search(r"^## ", rest[3:], flags=re.M)
    return rest if not nxt else rest[: nxt.start() + 3]


def md_blocks(section: str) -> dict[str, list[list[str]]]:
    """Split a markdown section into **Table-X** blocks of table rows (header row first)."""
    blocks: dict[str, list[list[str]]] = {}
    cur = None
    for line in section.splitlines():
        m = re.match(r"^\*\*Table-([A-D])\*\*", line.strip())
        if m:
            cur = m.group(1)
            blocks[cur] = []
            continue
        if cur and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells):
                continue
            blocks[cur].append(cells)
    return blocks


def runB_sweep() -> pd.DataFrame:
    """Run B T3: P@100 by contamination and difficulty for ledgers C, A, B (DECOMP and AVG dropped)."""
    sec = md_section(paths.CLIENT_RESULTS, r"^## T3\.")
    rows = []
    for led, tab in md_blocks(sec).items():
        header = tab[0]
        for r in tab[1:]:
            det = norm_name(r[0])
            if det in ("DECOMP", "AVG"):
                continue
            for h, v in zip(header[1:], r[1:]):
                m = re.match(r"c=([0-9.]+)\s+(\w+)", h)
                rows.append(dict(ledger=led, detector=det, c=float(m.group(1)), difficulty=m.group(2),
                                 P100=float(v)))
    df = pd.DataFrame(rows)
    assert set(df.detector) == set(DETS), set(df.detector)
    return df


def runB_core_header() -> dict[str, dict]:
    """Rows and injected anomalies per ledger from the run B T1 block headers."""
    sec = md_section(paths.CLIENT_RESULTS, r"^## T1\.")
    out = {}
    for m in re.finditer(r"\*\*Table-([A-D])\*\* \(([\d,]+) rows, ([\d,]+) injected anomalies, (\d+) seeds\)", sec):
        out[m.group(1)] = dict(rows=int(m.group(2).replace(",", "")), injected=int(m.group(3).replace(",", "")),
                               seeds=int(m.group(4)))
    return out


def n_anom_protocol() -> dict[str, list[int]]:
    """n_anom per ledger for c = 0.005 / 0.015 / 0.03 from harness/protocol.md, Section 3."""
    txt = (paths.PROTOCOL).read_text(encoding="utf-8")
    m = re.search(r"resulting counts are (.*?)\n\s*for c = 0\.005/0\.015/0\.03", txt, flags=re.S)
    seg = m.group(1).replace("\n", " ")
    out = {}
    for led, a, b, c in re.findall(r"([A-D]) ([\d,]+)/([\d,]+)/([\d,]+)", seg):
        out[led] = [int(x.replace(",", "")) for x in (a, b, c)]
    return out


def dib_header() -> dict:
    """Posting lines, journal entries and anomalous entries of the public ledger, from the published file."""
    sys.path.insert(0, str(paths.PIPELINE))
    import dib_data  # noqa: E402
    df = dib_data.load()
    return dict(lines=int(len(df)), entries=int(df.posting_id.nunique()),
                anomalous=int(df.groupby("posting_id").label.max().sum()))


def wp0_profile() -> pd.DataFrame:
    txt = (paths.CLIENT_AUDIT).read_text(encoding="utf-8")
    sec = txt[txt.index("## 1. Profile verification"):]
    sec = sec[: sec.index("\n## ", 5)]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"^\| ([A-D]) \| ([\d,]+) \|", line)
        if m:
            rows.append(dict(ledger=m.group(1), rows=int(m.group(2).replace(",", ""))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Label placement for scatter points (greedy, collision-aware)
# ----------------------------------------------------------------------------------------------

def _overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _seg_hits_box(p0, p1, bb, n=12) -> bool:
    for t in np.linspace(0.15, 0.95, n):
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1:
            return True
    return False


def place_labels(ax, xs, ys, labels, fontsize=FS, marker_r_pt=3.0, extra_obstacles=None):
    """Place one text label per point, avoiding other labels, markers and the axes frame.

    Candidates: 12 directions at increasing distances (points). Labels placed further than 8 pt from
    their point get a thin leader line; leader lines that would cross another marker or label are
    penalised.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    sc = fig.dpi / 72.0
    pts = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    r = marker_r_pt * sc
    mboxes = [matplotlib.transforms.Bbox([[px - r, py - r], [px + r, py + r]]) for px, py in pts]
    obstacles = list(extra_obstacles or [])
    axbb = ax.get_window_extent(rend)
    angles = np.deg2rad([0, 180, 30, 150, -30, -150, 60, 120, -60, -120, 90, -90])
    dists = [4.0, 8.0, 13.0, 19.0, 26.0]
    order = sorted(range(len(xs)), key=lambda i: -sum(
        1 for j in range(len(xs)) if j != i and np.hypot(*(np.subtract(pts[i], pts[j]))) < 45 * sc))
    placed = []
    for i in order:
        best = None
        for di, d in enumerate(dists):
            for a in angles:
                dx, dy = math.cos(a) * d, math.sin(a) * d
                ha = "left" if math.cos(a) > 0.3 else ("right" if math.cos(a) < -0.3 else "center")
                va = "bottom" if math.sin(a) > 0.3 else ("top" if math.sin(a) < -0.3 else "center")
                t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                                ha=ha, va=va, fontsize=fontsize, color=INK, zorder=6)
                bb = t.get_window_extent(rend)
                t.remove()
                bbx = bb.expanded(1.06, 1.12)
                cost = 3.0 * sum(_overlap(bbx, p) for p in placed)
                cost += 2.0 * sum(_overlap(bbx, m) for k, m in enumerate(mboxes) if k != i)
                cost += 2.0 * sum(_overlap(bbx, o) for o in obstacles)
                inside = matplotlib.transforms.Bbox.intersection(bbx, axbb)
                cost += 3.0 * (bbx.width * bbx.height - (inside.width * inside.height if inside is not None else 0))
                if d > 8.0:
                    # leader from the point to the nearest point of the label box
                    cx = min(max(pts[i][0], bb.x0), bb.x1)
                    cy = min(max(pts[i][1], bb.y0), bb.y1)
                    for k, m in enumerate(mboxes):
                        if k != i and _seg_hits_box(pts[i], (cx, cy), m):
                            cost += 150.0 * sc * sc
                    for p in placed:
                        if _seg_hits_box(pts[i], (cx, cy), p):
                            cost += 150.0 * sc * sc
                # ambiguity: another marker must not sit closer to the label than its own marker
                def _bdist(pt):
                    ddx = max(bb.x0 - pt[0], 0, pt[0] - bb.x1)
                    ddy = max(bb.y0 - pt[1], 0, pt[1] - bb.y1)
                    return math.hypot(ddx, ddy)
                own = _bdist(pts[i]) if d <= 8.0 else 0.0
                for k, pk in enumerate(pts):
                    if k != i and _bdist(pk) < max(own * 1.6, own + 4.0 * sc):
                        cost += 120.0 * sc * sc
                cost += di * 25.0 * sc * sc
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, (dx, dy), ha, va, bbx, d)
            if best[0] <= di * 25.0 * sc * sc + 1e-6:
                break
        cost, off, ha, va, bbx, d = best
        arrow = dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2.2) if d > 8.0 else None
        ax.annotate(labels[i], (xs[i], ys[i]), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=fontsize, color=INK, zorder=6, arrowprops=arrow)
        placed.append(bbx)


def family_legend_handles(fams=("nb", "iso", "rec", "emb"), with_ceiling=False):
    hs = []
    for f in fams + (("ceil",) if with_ceiling else tuple()):
        hs.append(Line2D([0], [0], marker="s", ls="none", markersize=6, markerfacecolor=FAM_COLOR[f],
                         markeredgecolor="white", markeredgewidth=0.6, label=FAM_LABEL[f]))
    return hs


def det_handles(dets, with_line=True):
    hs = []
    for d in dets:
        f = FAMILY[d]
        name = "LightGBM reference" if d == CEIL else d
        hs.append(Line2D([0], [0], color=FAM_COLOR[f], ls=LSTYLE[d] if with_line else "none",
                         lw=1.1 if d != CEIL else 1.3, marker=MARKER[d],
                         markersize=4.2 if d != CEIL else 6, markerfacecolor=FAM_COLOR[f],
                         markeredgecolor="white", markeredgewidth=0.5, label=name))
    return hs


# ----------------------------------------------------------------------------------------------
# Figure 1: framework schematic
# ----------------------------------------------------------------------------------------------

def fig_framework():
    prof = wp0_profile()
    rmin, rmax = int(prof.rows.min()), int(prof.rows.max())
    dib = dib_header()
    ta = pd.read_csv(paths.find("datev_typeaware_k100.csv"))
    types_real = len([c for c in ta.columns if re.fullmatch(r"R_T\d", c)])
    mk = pd.read_csv(paths.find("a6_public_marking_recall.csv"))
    n_mark = mk.marking.nunique()
    n_det = pd.read_csv(paths.find("a9_fig3_hits_vs_fsr.csv")).detector.nunique()

    W, H = FULL, 2.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    Y = 100 * H / W
    ax.set_ylim(0, Y)
    ax.axis("off")
    checks = []  # (text artist, box patch) pairs verified after drawing

    def box(x0, y0, w, h, fc="#f4f4f4", ec="#8c8c8c", lw=0.6):
        p = mpatches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=lw, zorder=1)
        ax.add_patch(p)
        return p

    def txt(p, x, y, s, bold=False, color=INK):
        t = ax.text(x, y, s, ha="left", va="top", fontsize=FS, color=color, linespacing=1.3, zorder=3,
                    fontweight="bold" if bold else "normal")
        checks.append((t, p))
        return t

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7), zorder=2)

    y0, y1 = 0.8, Y - 0.8
    # 1 ledgers
    x1, w1 = 0.4, 21.6
    b1 = box(x1, y0, w1, y1 - y0)
    txt(b1, x1 + 1.2, y1 - 1.2, "1  Ledgers", bold=True)
    s1 = box(x1 + 1.0, 21.6, w1 - 2.0, 19.8, fc="white", ec="#b3b3b3", lw=0.5)
    txt(s1, x1 + 1.9, 40.4, f"Four client\nledgers C, A, B, D\n{rmin:,} to {rmax:,}\npostings\n"
        f"injected types\nT1 to T{types_real}")
    s2 = box(x1 + 1.0, 1.8, w1 - 2.0, 18.4, fc="white", ec="#b3b3b3", lw=0.5)
    txt(s2, x1 + 1.9, 19.2, f"Public synthetic\nledger\n{dib['entries']:,} entries\n"
        f"generator markings\nM1 to M{n_mark}")
    # 2 detectors
    x2, w2 = 25.0, 19.6
    b2 = box(x2, y0, w2, y1 - y0)
    txt(b2, x2 + 1.2, y1 - 1.2, "2  Detectors", bold=True)
    txt(b2, x2 + 1.2, y1 - 6.4, f"{n_det} unsupervised\n(no labels):\nIF, LOF, kNN,\nHBOS, ECOD,\n"
        "PCA, AE, VAE,\nAE-emb")
    txt(b2, x2 + 1.2, 13.0, "LightGBM reference\n(supervised,\nuses labels)", color=INK2)
    # 3 ranking with budget
    x3, w3 = 47.6, 22.4
    b3 = box(x3, y0, w3, y1 - y0)
    txt(b3, x3 + 1.2, y1 - 1.2, "3  Ranking", bold=True)
    lists = [["T4", "T4", "", "T4", "T4", "", "", ""], ["T1", "T4", "", "T5", "T2", "", "", ""]]
    cw, ch, gap = 5.4, 3.5, 0.6
    kk = 5
    ytop = y1 - 10.4
    for li, lst in enumerate(lists):
        cx = x3 + 7.2 + li * 7.9
        ax.text(cx + cw / 2, y1 - 6.2, f"list {li + 1}", ha="center", va="top", fontsize=FS, color=INK2)
        for r, t in enumerate(lst):
            yy = ytop - r * (ch + gap)
            fc = "#0072B2" if t else "white"
            ec = "#0072B2" if t else ("#9a9a9a" if r < kk else "#d0d0d0")
            ax.add_patch(mpatches.Rectangle((cx, yy - ch), cw, ch, fc=fc, ec=ec, lw=0.5, zorder=3))
            if t:
                ax.text(cx + cw / 2, yy - ch / 2, t, ha="center", va="center", fontsize=FS,
                        color="white", zorder=4)
    yb1 = ytop - kk * (ch + gap) + gap
    bx = x3 + 5.4
    ax.plot([bx + 0.8, bx, bx, bx + 0.8], [ytop, ytop, yb1, yb1], color=INK, lw=0.7, zorder=3)
    ax.text(bx - 0.9, (ytop + yb1) / 2, "top $k$ reviewed", rotation=90, ha="center", va="center",
            fontsize=FS, color=INK)
    # 4 readouts
    x4, w4 = 73.0, 26.6
    hh = (y1 - y0 - 1.4) / 2
    b4a = box(x4, y0 + hh + 1.4, w4, hh, fc="#f4f4f4")
    txt(b4a, x4 + 1.2, y1 - 1.2, "4a  Hit-based view", bold=True)
    txt(b4a, x4 + 1.2, y1 - 6.2, "P@$k$, AP, ROC-AUC\nhow many anomalies?\nlists 1 and 2: equal")
    b4b = box(x4, y0, w4, hh, fc="#e6f0f8", ec="#0072B2")
    txt(b4b, x4 + 1.2, y0 + hh - 1.2, "4b  Type-aware view", bold=True)
    txt(b4b, x4 + 1.2, y0 + hh - 6.2, "$R_t$@$k$, TC@$k$, FSR@$k$,\nFHR$_t$: which types?\n"
        "list 1: one type,\nlist 2: four types")
    ymid = Y / 2
    arrow(x1 + w1 + 0.2, ymid, x2 - 0.3, ymid)
    arrow(x2 + w2 + 0.2, ymid, x3 - 0.3, ymid)
    arrow(x3 + w3 + 0.2, ymid + 2.0, x4 - 0.3, y0 + hh + 1.4 + hh / 2)
    arrow(x3 + w3 + 0.2, ymid - 2.0, x4 - 0.3, y0 + hh / 2)
    # verify that every text block sits inside its box
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t, p in checks:
        tb, pb = t.get_window_extent(rend), p.get_window_extent(rend)
        if tb.x1 > pb.x1 - 2 or tb.y0 < pb.y0 + 2 or tb.x0 < pb.x0:
            print("WARNING framework text outside box:", repr(t.get_text()[:30]), tb, pb)
    save(fig, "fig_framework")


# ----------------------------------------------------------------------------------------------
# Figure 2: per-type recall at 100 on the real ledgers
# ----------------------------------------------------------------------------------------------

def fig_pertype_real():
    ta = pd.read_csv(paths.find("datev_typeaware_k100.csv"))
    nb = pd.read_csv(paths.find("a2_found_bounds.csv"))[["ledger", "type", "n_t"]].drop_duplicates()
    ledgers = ["C", "A", "B", "D"]
    types = [f"T{i}" for i in range(1, 6)]
    fig = plt.figure(figsize=(FULL, 5.0))
    norm = Normalize(0, 1)
    nrows = len(DETS) + 2  # detectors, gap, ceiling
    pw, ph = 0.385, 0.335
    pos = {"C": (0.105, 0.565), "A": (0.575, 0.565), "B": (0.105, 0.1), "D": (0.575, 0.1)}
    for li, led in enumerate(ledgers):
        x0, y0 = pos[led]
        ax = fig.add_axes([x0, y0, pw, ph])
        sub = ta[ta.ledger == led].set_index("detector").loc[DETS]
        for ri, det in enumerate(DETS):
            for ci, t in enumerate(types):
                v = float(sub.loc[det, f"R_{t}"])
                col = SEQ(norm(v))
                ax.add_patch(mpatches.Rectangle((ci, ri), 1, 1, fc=col, ec="white", lw=1.2))
                ax.text(ci + 0.5, ri + 0.5, f2(v), ha="center", va="center", fontsize=FS,
                        color=(MUTED if v == 0 else text_color_for(col)))
        rr = len(DETS) + 1
        for ci, t in enumerate(types):
            v = float(sub[f"ceil_{t}"].iloc[0])
            ax.add_patch(mpatches.Rectangle((ci + 0.05, rr + 0.07), 0.9, 0.86, fc="white", ec="#7f7f7f", lw=0.6))
            ax.text(ci + 0.5, rr + 0.5, f2(v), ha="center", va="center", fontsize=FS, color=INK,
                    fontstyle="italic")
        ax.set_xlim(0, 5)
        ax.set_ylim(nrows, 0)
        ax.set_xticks(np.arange(5) + 0.5)
        nts = nb[nb.ledger == led].set_index("type").loc[types, "n_t"].astype(int).tolist()
        ax.set_xticklabels([f"{t}\n{n:,}" for t, n in zip(types, nts)], fontsize=FS, linespacing=1.15)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(list(np.arange(len(DETS)) + 0.5) + [rr + 0.5])
        if led in ("C", "B"):
            ax.set_yticklabels(DETS + ["Budget\nceiling"], fontsize=FS, linespacing=1.0)
            ax.get_yticklabels()[-1].set_fontstyle("italic")
            ax.text(-0.02, 1.0, r"$n_t$", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.02, 1.2, bold_letter("abcd"[li], f"Ledger {led}"), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=FS)
    cax = fig.add_axes([0.575, 0.04, 0.385, 0.014])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])
    cb.ax.tick_params(labelsize=FS, length=2, width=0.5)
    fig.text(0.56, 0.047, r"$R_t$@100", ha="right", va="center", fontsize=FS)
    save(fig, "fig_pertype_real")


# ----------------------------------------------------------------------------------------------
# Figure 3: hits versus fair-share type recall
# ----------------------------------------------------------------------------------------------

def _merge_identical(df, xcol, ycol):
    """Group detectors with identical coordinates so that they share one label."""
    groups = []
    used = set()
    for i, r in df.iterrows():
        if i in used:
            continue
        same = df[(np.isclose(df[xcol], r[xcol], atol=1e-9)) & (np.isclose(df[ycol], r[ycol], atol=1e-9))]
        used.update(same.index)
        groups.append(list(same.detector))
    return groups


def _nice_lim(v):
    for lim in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        if v <= lim * 0.93:
            return lim
    return 1.0


def _ticks(lim):
    step = 0.1 if lim <= 0.5 else 0.2
    if lim <= 0.25:
        step = 0.05
    return np.round(np.arange(0, lim + 1e-9, step), 2)


def fig_hits_vs_types():
    d = pd.read_csv(paths.find("a9_fig3_hits_vs_fsr.csv"))
    panels = [("C", 100, "Ledger C"), ("A", 100, "Ledger A"), ("B", 100, "Ledger B"),
              ("D", 100, "Ledger D, FSR bounds"), ("public", 100, "Public ledger"), ("public", 1000, "Public ledger")]
    fig, axes = plt.subplots(3, 2, figsize=(FULL, 6.6))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.965, bottom=0.135, wspace=0.26, hspace=0.42)
    for pi, (led, k, title) in enumerate(panels):
        ax = axes.flat[pi]
        sub = d[(d.ledger == led) & (d.k == k)].copy().reset_index(drop=True)
        assert len(sub) == len(DETS), (led, k, len(sub))
        xlim = _nice_lim(sub.P.max())
        ylim = _nice_lim(max(sub.FSR.max(), sub.FSR_hi.max()))
        m = min(xlim, ylim)
        ax.plot([0, m], [0, m], color="#c8c8c8", lw=0.6, zorder=0)
        if led == "D":
            for _, r in sub.iterrows():
                if r.FSR_hi > r.FSR_lo:
                    ax.plot([r.P, r.P], [r.FSR_lo, r.FSR_hi], color=FAM_COLOR[FAMILY[r.detector]], lw=1.3,
                            solid_capstyle="butt", zorder=2)
                    for yy in (r.FSR_lo, r.FSR_hi):
                        ax.plot([r.P - xlim * 0.012, r.P + xlim * 0.012], [yy, yy],
                                color=FAM_COLOR[FAMILY[r.detector]], lw=0.9, zorder=2)
        for _, r in sub.iterrows():
            f = FAMILY[r.detector]
            ax.plot(r.P, r.FSR, marker=MARKER[r.detector], ls="none", markersize=4.4,
                    markerfacecolor=FAM_COLOR[f], markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_xlim(-0.015 * xlim, xlim)
        ax.set_ylim(-0.015 * ylim, ylim)
        xt, yt = _ticks(xlim), _ticks(ylim)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels([f"{t:g}" for t in xt])
        ax.set_yticklabels([f"{t:g}" for t in yt])
        kl = f"{k:,}"
        ax.set_xlabel(f"P@{kl}", labelpad=1.5)
        ax.set_ylabel(f"FSR@{kl}", labelpad=1.5)
        # points closer than 7 pt on the page (single linkage) share one label
        fig.canvas.draw()
        disp = {r.detector: ax.transData.transform((r.P, r.FSR)) for _, r in sub.iterrows()}
        sc = fig.dpi / 72.0
        parent = {dd: dd for dd in disp}

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a
        names = list(disp)
        for i1 in range(len(names)):
            for i2 in range(i1 + 1, len(names)):
                if np.hypot(*(disp[names[i1]] - disp[names[i2]])) < 7.0 * sc:
                    parent[find(names[i2])] = find(names[i1])
        clusters = {}
        for dd in DETS:
            clusters.setdefault(find(dd), []).append(dd)
        xs, ys, labs = [], [], []
        for g in clusters.values():
            gs = sub[sub.detector.isin(g)]
            xs.append(float(gs.P.mean()))
            ys.append(float(gs.FSR.mean()))
            labs.append(", ".join(g))
        extra = []
        rr = 2.6 * sc
        for g in clusters.values():
            if len(g) > 1:
                for dd in g:
                    px, py = disp[dd]
                    extra.append(matplotlib.transforms.Bbox([[px - rr, py - rr], [px + rr, py + rr]]))
        if led == "D":
            fig.canvas.draw()
            for _, r in sub.iterrows():
                p0 = ax.transData.transform((r.P, r.FSR_lo))
                p1 = ax.transData.transform((r.P, r.FSR_hi))
                extra.append(matplotlib.transforms.Bbox([[p0[0] - 2, p0[1]], [p1[0] + 2, p1[1]]]))
        place_labels(ax, xs, ys, labs, marker_r_pt=2.6, extra_obstacles=extra)
        ttl = f"{title}, k = {kl}" if led == "public" else title
        ax.text(-0.02, 1.04, bold_letter("abcdef"[pi], ttl), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS)
    hs = family_legend_handles(("nb", "iso", "emb", "rec"))
    hs.append(Line2D([0], [0], color="#c8c8c8", lw=0.8, label="FSR = P (diagonal)"))
    fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5,
               handletextpad=0.4, fontsize=FS)
    save(fig, "fig_hits_vs_types")


# ----------------------------------------------------------------------------------------------
# Figure 4: budget on the public ledger
# ----------------------------------------------------------------------------------------------

def fig_budget_public():
    m = pd.read_csv(paths.find("dib_typeaware_mean.csv"))
    m["detector"] = m.detector.map(norm_name)
    fh = pd.read_csv(paths.find("a4_first_hit.csv"))
    fh["detector"] = fh.detector.map(norm_name)
    fps = pd.read_csv(paths.find("a4_first_hit_perseed.csv"))
    fps["detector"] = fps.detector.map(norm_name)
    mk = pd.read_csv(paths.find("a6_public_marking_recall.csv"))
    n_anom = int(mk.drop_duplicates("marking").n_marking.sum())
    order = DETS + [CEIL]
    ks = sorted(m.k.unique())

    fig = plt.figure(figsize=(FULL, 5.35))
    axP = fig.add_axes([0.085, 0.63, 0.32, 0.33])
    axF = fig.add_axes([0.585, 0.63, 0.32, 0.33])
    axH = fig.add_axes([0.13, 0.035, 0.71, 0.34])
    for ax, col, lab in [(axP, "P", "P@$k$"), (axF, "FSR", "FSR@$k$")]:
        if col == "P":
            kk = np.array(ks, float)
            ax.plot(kk, np.minimum(1.0, n_anom / kk), color="#b0b0b0", lw=0.8, ls=(0, (1, 1.2)), zorder=1)
            ax.text(kk[0] * 1.05, 0.975, "maximum attainable", ha="left", va="top",
                    fontsize=FS, color=MUTED)
        ax.axvline(1000, color=GRID, lw=0.8, zorder=0)
        for det in order:
            s = m[m.detector == det].sort_values("k")
            f = FAMILY[det]
            ax.plot(s.k, s[col], color=FAM_COLOR[f], ls=LSTYLE[det], lw=1.3 if det in ("kNN", "AE-emb", CEIL) else 0.9,
                    marker=MARKER[det], markersize=3.6 if det != CEIL else 5.5, markerfacecolor=FAM_COLOR[f],
                    markeredgecolor="white", markeredgewidth=0.4, zorder=3 if det in ("kNN", "AE-emb", CEIL) else 2)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k:,}" for k in ks], rotation=40, ha="right", rotation_mode="anchor")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(ks[0] / 1.15, ks[-1] * 1.15)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Review budget $k$ (entries, log scale)", labelpad=2)
        ax.set_ylabel(lab, labelpad=2)
        ax.tick_params(axis="x", labelsize=FS)
        # direct labels for the three highlighted series at the right end
        ends = []
        for det, name in [("kNN", "kNN"), ("AE-emb", "AE-emb"), (CEIL, "LightGBM")]:
            s = m[m.detector == det].sort_values("k")
            ends.append((float(s[col].iloc[-1]), name, det))
        ends.sort()
        ylast = -1
        for y, name, det in ends:
            yy = max(y, ylast + 0.07)
            ax.annotate(name, (ks[-1], y), xytext=(ks[-1] * 1.22, yy), textcoords="data", ha="left", va="center",
                        fontsize=FS, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2))
            ylast = yy
    axP.text(-0.02, 1.03, bold_letter("a", "Hits"), transform=axP.transAxes, ha="left", va="bottom", fontsize=FS)
    axF.text(-0.02, 1.03, bold_letter("b", "Fair-share type recall"), transform=axF.transAxes, ha="left",
             va="bottom", fontsize=FS)
    for ax in (axP, axF):
        ax.text(1000 / 1.06, 0.5, "$k$ = 1,000", ha="right", va="center", rotation=90, fontsize=FS, color=MUTED)
    fig.legend(handles=det_handles(order), loc="center", ncol=5, bbox_to_anchor=(0.5, 0.479),
               columnspacing=1.0, handletextpad=0.4, handlelength=2.4, fontsize=FS)

    # (c) first-hit rank heatmap, log colour scale
    marks = [f"M{i}" for i in range(1, 9)]
    vals = fh.pivot(index="detector", columns="marking", values="first_hit_mean").loc[order, marks]
    per_seed_max = fps.set_index(["detector", "seed"])[marks].max(axis=1).groupby("detector").mean()
    allc = per_seed_max.loc[order]
    vmax = float(max(vals.values.max(), allc.max()))
    lnorm = LogNorm(vmin=1, vmax=10 ** math.ceil(math.log10(vmax)))
    cmap_r = SEQ.reversed()
    ncol = len(marks) + 1
    for ri, det in enumerate(order):
        for ci in range(ncol):
            x = ci + (0.35 if ci == len(marks) else 0)
            v = float(vals.loc[det, marks[ci]]) if ci < len(marks) else float(allc.loc[det])
            col = cmap_r(lnorm(v))
            axH.add_patch(mpatches.Rectangle((x, ri), 1, 1, fc=col, ec="white", lw=1.2))
            weight = "bold" if v <= 1000 else "normal"
            axH.text(x + 0.5, ri + 0.5, fmt_int(v), ha="center", va="center", fontsize=FS,
                     color=text_color_for(col), fontweight=weight)
    axH.set_xlim(0, ncol + 0.35)
    axH.set_ylim(len(order), 0)
    axH.set_xticks([i + 0.5 for i in range(len(marks))] + [len(marks) + 0.85])
    nmk = mk.drop_duplicates("marking").set_index("marking").loc[marks, "n_marking"].astype(int)
    axH.set_xticklabels([f"{mm}\n{n}" for mm, n in zip(marks, nmk)] + ["All\n" + f"{n_anom}"], linespacing=1.1)
    axH.xaxis.tick_top()
    axH.tick_params(axis="both", length=0, pad=2)
    axH.set_yticks(np.arange(len(order)) + 0.5)
    axH.set_yticklabels([d if d != CEIL else "LightGBM" for d in order])
    axH.get_yticklabels()[-1].set_fontstyle("italic")
    for sp in axH.spines.values():
        sp.set_visible(False)
    axH.text(-0.155, 1.135, bold_letter("c", "First-hit rank per marking (bold: within $k$ = 1,000)"),
             transform=axH.transAxes, ha="left", va="bottom", fontsize=FS)
    axH.text(-0.155, 1.02, r"$n_t$", transform=axH.transAxes, ha="left", va="bottom", fontsize=FS)
    cax = fig.add_axes([0.865, 0.035, 0.014, 0.28])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=lnorm, cmap=cmap_r), cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([1, 10, 100, 1000, 10000])
    cb.set_ticklabels(["1", "10", "100", "1,000", "10,000"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FS, length=2, width=0.5)
    cb.ax.axhline(1000, color=INK, lw=0.8)
    cb.set_label(r"FHR$_t$ (log scale)", fontsize=FS, labelpad=2)
    save(fig, "fig_budget_public")


# ----------------------------------------------------------------------------------------------
# Figure 5: contamination and difficulty sweep plus critical-difference diagram
# ----------------------------------------------------------------------------------------------

def cd_cliques(ranks: pd.Series, cd: float):
    """Maximal groups of detectors (consecutive by mean rank) whose rank range is below the CD."""
    s = ranks.sort_values()
    names, vals = list(s.index), list(s.values)
    cl = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i:
            cl.append((i, j))
    maximal = [c for c in cl if not any((o[0] <= c[0] and o[1] >= c[1] and o != c) for o in cl)]
    return names, vals, maximal


def fig_sweep():
    sw = runB_sweep()
    nan = n_anom_protocol()
    fr = pd.read_csv(paths.find("a8_friedman_check.csv")).set_index("detector")
    ranks = fr.loc[[d for d in fr.index if d in DETS or norm_name(d) in DETS], "mean_rank"].astype(float)
    ranks.index = [norm_name(i) for i in ranks.index]
    cd = float(fr.loc["CD Nemenyi", "mean_rank"])
    chi2 = float(fr.loc["Friedman chi2", "mean_rank"])
    ncell = int(float(fr.loc["n cells", "mean_rank"]))

    dets = ["kNN", "VAE", "AE", "PCA", "IF"]
    ledgers = ["C", "A", "B"]
    cs = sorted(sw.c.unique())
    fig = plt.figure(figsize=(FULL, 6.1))
    L, R, T = 0.10, 0.985, 0.875
    gh = 0.178
    hs = 0.026
    ws = 0.022
    pw = (R - L - 4 * ws) / 5
    for li, led in enumerate(ledgers):
        for di, det in enumerate(dets):
            ax = fig.add_axes([L + di * (pw + ws), T - (li + 1) * gh - li * hs, pw, gh])
            cap = [min(1.0, n / 100) for n in nan[led]]
            xpos = np.arange(len(cs))
            ax.plot(xpos, cap, color="#b0b0b0", lw=0.8, ls=(0, (1, 1.2)), zorder=1)
            for dif in ["easy", "medium", "hard"]:
                s = sw[(sw.ledger == led) & (sw.detector == det) & (sw.difficulty == dif)].sort_values("c")
                st = DIFF_STYLE[dif]
                ax.plot([cs.index(c) for c in s.c], s.P100, color=st["color"], ls=st["ls"], marker=st["marker"], markersize=3.4,
                        markerfacecolor=st["color"], markeredgecolor="white", markeredgewidth=0.4, lw=1.1, zorder=3)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-0.35, len(cs) - 0.65)
            ax.set_xticks(xpos)
            ax.set_yticks([0, 0.5, 1.0])
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            if di == 0:
                ax.set_yticklabels(["0", "0.5", "1"])
                ax.set_ylabel(f"Ledger {led}\nP@100", labelpad=2)
            else:
                ax.set_yticklabels([])
            if li == len(ledgers) - 1:
                ax.set_xticklabels([f"{c * 100:g}" for c in cs])
            else:
                ax.set_xticklabels([])
            if li == 0:
                ax.text(0.5, 1.05, det, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS,
                        fontweight="bold")
    fig.text(L + (R - L) / 2, T - 3 * gh - 2 * hs - 0.038, "Contamination $c$ (percent of rows)", ha="center",
             va="top", fontsize=FS)
    fig.text(0.01, 0.975, bold_letter("a", "P@100 over the contamination and difficulty sweep (mean over five seeds)"),
             ha="left", va="top", fontsize=FS)
    hd = [Line2D([0], [0], color=DIFF_STYLE[d]["color"], ls=DIFF_STYLE[d]["ls"], marker=DIFF_STYLE[d]["marker"],
                 markersize=3.6, markerfacecolor=DIFF_STYLE[d]["color"], markeredgecolor="white",
                 markeredgewidth=0.4, lw=1.1, label=d) for d in ["easy", "medium", "hard"]]
    hd.append(Line2D([0], [0], color="#b0b0b0", ls=(0, (1, 1.2)), lw=0.8, label="maximum attainable"))
    fig.legend(handles=hd, loc="upper left", ncol=4, bbox_to_anchor=(L - 0.012, 0.955), columnspacing=1.4,
               handletextpad=0.4, fontsize=FS)

    # (b) critical-difference diagram
    axc = fig.add_axes([L, 0.008, R - L, 0.175])
    names, vals, cliques = cd_cliques(ranks, cd)
    k = len(names)
    axc.set_xlim(-1.4, k + 2.4)
    axc.set_ylim(-4.8, 1.7)
    axc.axis("off")
    axc.plot([1, k], [0, 0], color=INK, lw=0.7)
    for t in range(1, k + 1):
        axc.plot([t, t], [0, 0.18], color=INK, lw=0.7)
        axc.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=FS)
    axc.plot([1, 1 + cd], [1.25, 1.25], color=INK, lw=1.0)
    for x in (1, 1 + cd):
        axc.plot([x, x], [1.1, 1.4], color=INK, lw=1.0)
    axc.text(1 + cd / 2, 1.45, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=FS)
    half = math.ceil(k / 2)
    for i, (n, v) in enumerate(zip(names, vals)):
        if i < half:
            y = -1.25 - i * 0.72
            xt = 0.75
            axc.plot([v, v, xt + 0.05], [0, y, y], color=INK2, lw=0.6)
            axc.text(xt, y, f"{n} ({v:.2f})", ha="right", va="center", fontsize=FS)
        else:
            j = k - 1 - i
            y = -1.25 - j * 0.72
            xt = k + 0.25
            axc.plot([v, v, xt - 0.05], [0, y, y], color=INK2, lw=0.6)
            axc.text(xt, y, f"{n} ({v:.2f})", ha="left", va="center", fontsize=FS)
    for ci, (a, b) in enumerate(cliques):
        y = -0.35 - ci * 0.2
        axc.plot([vals[a] - 0.03, vals[b] + 0.03], [y, y], color=INK, lw=2.2, solid_capstyle="butt")
    fig.text(0.01, 0.222, bold_letter("b", f"Mean rank by P@100, {ncell} seed-averaged cells of ledgers C, A, B "
             f"(exploratory; Friedman $\\chi^2$ = {chi2:.1f})"), ha="left", va="top", fontsize=FS)
    save(fig, "fig_sweep")


# ----------------------------------------------------------------------------------------------
# Figure 6: markings and rule populations on the public ledger
# ----------------------------------------------------------------------------------------------

def fig_public_types():
    mk = pd.read_csv(paths.find("a6_public_marking_recall.csv"))
    mk["detector"] = mk.detector.map(norm_name)
    rp = pd.read_csv(paths.find("a6_public_rule_populations.csv"))
    rp["detector"] = rp.detector.map(norm_name)
    order = DETS + [CEIL]
    marks = [f"M{i}" for i in range(1, 9)]
    fig = plt.figure(figsize=(FULL, 5.6))
    axH = fig.add_axes([0.13, 0.595, 0.74, 0.335])
    norm = Normalize(0, 1)
    piv = mk.pivot(index="detector", columns="marking", values="R1000_mean").loc[order, marks]
    for ri, det in enumerate(order):
        for ci, mm in enumerate(marks):
            v = float(piv.loc[det, mm])
            col = SEQ(norm(v))
            axH.add_patch(mpatches.Rectangle((ci, ri), 1, 1, fc=col, ec="white", lw=1.2))
            axH.text(ci + 0.5, ri + 0.5, f2(v), ha="center", va="center", fontsize=FS,
                     color=(MUTED if v == 0 else text_color_for(col)))
    axH.set_xlim(0, len(marks))
    axH.set_ylim(len(order), 0)
    nmk = mk.drop_duplicates("marking").set_index("marking").loc[marks]
    short = {r.Index: r.marking_name.split(" ", 1)[1] for r in nmk.itertuples()}
    axH.set_xticks(np.arange(len(marks)) + 0.5)
    axH.set_xticklabels([f"{mm}\n{int(nmk.loc[mm, 'n_marking'])}" for mm in marks], linespacing=1.1)
    axH.xaxis.tick_top()
    axH.tick_params(axis="both", length=0, pad=2)
    axH.set_yticks(np.arange(len(order)) + 0.5)
    axH.set_yticklabels([d if d != CEIL else "LightGBM" for d in order])
    axH.get_yticklabels()[-1].set_fontstyle("italic")
    for sp in axH.spines.values():
        sp.set_visible(False)
    axH.text(-0.155, 1.135, bold_letter("a", "Recall within the top 1,000 entries per marking"),
             transform=axH.transAxes, ha="left", va="bottom", fontsize=FS)
    axH.text(-0.155, 1.02, r"$n_t$", transform=axH.transAxes, ha="left", va="bottom", fontsize=FS)
    cax = fig.add_axes([0.9, 0.595, 0.014, 0.335])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=FS, length=2, width=0.5)
    cb.set_label(r"$R_t$@1,000", fontsize=FS, labelpad=2)

    # (b) P@100 inside rule populations against rule precision
    axD = fig.add_axes([0.335, 0.165, 0.62, 0.34])
    pops = rp.drop_duplicates("population").sort_values("rule_precision", ascending=False)
    offsets = np.linspace(-0.3, 0.3, len(order))
    for pi, pr in enumerate(pops.itertuples()):
        y0 = pi
        axD.plot([pr.rule_precision, pr.rule_precision], [y0 - 0.42, y0 + 0.42], color=INK, lw=1.4,
                 solid_capstyle="butt", zorder=2)
        for di, det in enumerate(order):
            r = rp[(rp.population == pr.population) & (rp.detector == det)].iloc[0]
            f = FAMILY[det]
            axD.plot(r["P@100_mean"], y0 + offsets[di], marker=MARKER[det], ls="none",
                     markersize=4.2 if det != CEIL else 6.2, markerfacecolor=FAM_COLOR[f], markeredgecolor="white",
                     markeredgewidth=0.5, zorder=3)
        if pi == 0:
            axD.annotate("rule precision", (pr.rule_precision, y0 - 0.42), xytext=(pr.rule_precision - 0.02, y0 - 0.62),
                         ha="right", va="center", fontsize=FS, color=INK,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=1))
    axD.set_ylim(len(pops) - 0.45, -0.75)
    axD.set_yticks(range(len(pops)))
    axD.set_yticklabels([f"{p.population}\n{p.entries:,} entries, {p.anomalies} anomalies\nrule precision "
                         f"{p.rule_precision:.3f}" for p in pops.itertuples()], fontsize=FS, linespacing=1.15)
    axD.tick_params(axis="y", length=0, pad=4)
    axD.set_xlim(0, 1.0)
    axD.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axD.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    axD.xaxis.grid(True, color=GRID, lw=0.5)
    axD.set_axisbelow(True)
    axD.set_xlabel("P@100 inside the rule population (entries ranked by detector score)", labelpad=2)
    for i in range(1, len(pops)):
        axD.axhline(i - 0.5, color=GRID, lw=0.5)
    axD.text(-0.52, 1.03, bold_letter("b", "Ranking inside the rule populations"), transform=axD.transAxes,
             ha="left", va="bottom", fontsize=FS)
    fig.legend(handles=det_handles(order, with_line=False), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3, fontsize=FS)
    save(fig, "fig_public_types")


# ----------------------------------------------------------------------------------------------
# Supplementary figure: bootstrap intervals on the public ledger
# ----------------------------------------------------------------------------------------------

def fig_bootstrap():
    ci = pd.read_csv(paths.find("a5_bootstrap_ci.csv"))
    ci["detector"] = ci.detector.map(norm_name)
    pdiff = pd.read_csv(paths.find("a5_bootstrap_paired_kNN.csv"))
    metrics = ["P@100", "FSR@100", "P@1000", "FSR@1000"]
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 4.6), sharey="row")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.93, bottom=0.08, wspace=0.12, hspace=0.42)
    order = DETS
    others = [d for d in DETS if d != "kNN"]
    for mi, met in enumerate(metrics):
        ax = axes[0, mi]
        for yi, det in enumerate(order):
            r = ci[(ci.detector == det) & (ci.metric == met)].iloc[0]
            f = FAMILY[det]
            ax.plot([r.ci_lo_2_5, r.ci_hi_97_5], [yi, yi], color=FAM_COLOR[f], lw=1.3, solid_capstyle="butt")
            ax.plot(r.point_seed0, yi, marker=MARKER[det], ls="none", markersize=4.2, markerfacecolor=FAM_COLOR[f],
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlim(0, 0.35)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(met.replace("1000", "1,000"), labelpad=2)
        ax.tick_params(axis="y", length=0)
        if mi == 0:
            ax.text(-0.36, 1.05, bold_letter("a", "Point estimate (seed 0) and 95 percent interval"),
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=FS)
        ax2 = axes[1, mi]
        ax2.axvline(0, color=INK2, lw=0.6)
        for yi, det in enumerate(others):
            r = pdiff[(pdiff.comparison == f"kNN minus {det}") & (pdiff.metric == met)].iloc[0]
            f = FAMILY[det]
            ax2.plot([r.ci_lo_2_5, r.ci_hi_97_5], [yi, yi], color=FAM_COLOR[f], lw=1.3, solid_capstyle="butt")
            ax2.plot(r.point_difference, yi, marker=MARKER[det], ls="none", markersize=4.2,
                     markerfacecolor=(FAM_COLOR[f] if r.excludes_zero else "white"),
                     markeredgecolor=(FAM_COLOR[f] if not r.excludes_zero else "white"),
                     markeredgewidth=(0.9 if not r.excludes_zero else 0.5))
        ax2.set_ylim(len(others) - 0.4, -0.6)
        ax2.set_yticks(range(len(others)))
        ax2.set_yticklabels([f"kNN \u2212 {d}" for d in others])
        ax2.set_xlim(-0.15, 0.3)
        ax2.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
        ax2.set_xticklabels(["\u22120.1", "0", "0.1", "0.2", "0.3"])
        ax2.xaxis.grid(True, color=GRID, lw=0.5)
        ax2.set_axisbelow(True)
        ax2.set_xlabel(met.replace("1000", "1,000"), labelpad=2)
        ax2.tick_params(axis="y", length=0)
        if mi == 0:
            ax2.text(-0.36, 1.05, bold_letter("b", "Paired difference kNN minus detector (open: interval "
                     "contains 0)"), transform=ax2.transAxes, ha="left", va="bottom", fontsize=FS)
    save(fig, "fig_bootstrap")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_framework()
    fig_pertype_real()
    fig_hits_vs_types()
    fig_budget_public()
    fig_sweep()
    fig_public_types()
    fig_bootstrap()
