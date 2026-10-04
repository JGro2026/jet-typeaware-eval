"""WP4 analysis: tables T1 to T7, the DECOMP component ablation, and figures.

  python -m src.analyze

Reads results/metrics.parquet, results/ceiling.parquet, results/feedback.parquet
(run the two `collect` commands first) plus the per-cell DECOMP component files.
Writes reports/tables.md (all tables, markdown), reports/tables/*.csv and
*.tex, and reports/figures/*.pdf and *.png. Works on partial results and skips
tables whose inputs are missing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detectors import ALL_DETECTORS, UNSUPERVISED
from .evaluate import friedman_nemenyi, precision_at_k, ranking, recall_at_k_by_type
from .inject import TYPES
from .load import ROOT

RESULTS = ROOT / "results"
REP = ROOT / "reports"
FIG = REP / "figures"
TAB = REP / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)

TABLE_ORDER = ["C", "A", "B", "D"]
TABLE_ROWS = {"C": 3461, "A": 5080, "B": 17505, "D": 279975}
DET_ORDER = ALL_DETECTORS
FEEDBACK_ORDER = ["STATIC-AVG", "LinTS", "GREEDY-LR", "DeepSAD", "LinTS-spec", "GREEDY-LR-spec"]

# Validated palette (dataviz reference instance), light mode
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": SURF,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42,
})

MD: list[str] = []


def md(text: str = "") -> None:
    MD.append(text)


def fmt_pm(m, s, nd=2):
    if pd.isna(m):
        return "n/a"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if not pd.isna(s) else f"{m:.{nd}f}"


def to_markdown(df: pd.DataFrame, index_name: str = "") -> str:
    d = df.copy()
    d.index.name = index_name
    cols = [index_name] + [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in d.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, index_name: str = "", caption: str = "") -> None:
    df.to_csv(TAB / f"{name}.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tex = df.to_latex(escape=True, caption=caption, label=f"tab:{name}", index_names=bool(index_name))
    (TAB / f"{name}.tex").write_text(tex)
    md(to_markdown(df, index_name))
    md()


def load() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    m = pd.read_parquet(RESULTS / "metrics.parquet")
    ceil = pd.read_parquet(RESULTS / "ceiling.parquet") if (RESULTS / "ceiling.parquet").exists() else None
    fb = pd.read_parquet(RESULTS / "feedback.parquet") if (RESULTS / "feedback.parquet").exists() else None
    return m, ceil, fb


def wide(m: pd.DataFrame, metric: str) -> pd.DataFrame:
    """rows = (table, c, d, seed), columns = detectors."""
    sub = m[m.metric == metric]
    return sub.pivot_table(index=["table", "c", "d", "seed"], columns="detector", values="value")


def core_mask(m: pd.DataFrame) -> pd.Series:
    return (m.c == 0.015) & (m.d == "medium")


# ---------------------------------------------------------------- T1

def table_t1(m: pd.DataFrame) -> None:
    md("## T1. Core grid (c = 1.5 percent, medium): P@25, P@50, P@100 and AP per detector and table, mean ± sd over seeds\n")
    core = m[core_mask(m)]
    for t in TABLE_ORDER:
        sub = core[core.table == t]
        if sub.empty:
            continue
        n_seeds = sub.seed.nunique()
        rows = {}
        for det in DET_ORDER:
            s = sub[sub.detector == det]
            if s.empty:
                continue
            g = s.groupby("metric").value
            rows[det] = {k: fmt_pm(g.mean().get(k), g.std().get(k)) for k in ["p_at_25", "p_at_50", "p_at_100", "ap", "roc_auc"]}
        df = pd.DataFrame(rows).T
        df.columns = ["P@25", "P@50", "P@100", "AP", "ROC-AUC"]
        n_anom = int(sub[sub.metric == "n_anom"].value.iloc[0])
        md(f"**Table-{t}** ({TABLE_ROWS[t]:,} rows, {n_anom} injected anomalies, {n_seeds} seeds)\n")
        save_table(df, f"T1_core_{t}", "Detector", f"Core grid, Table-{t}: mean and standard deviation over {n_seeds} seeds.")


# ---------------------------------------------------------------- T2

def table_t2(m: pd.DataFrame) -> None:
    md("## T2. Per-type recall at 100 (core grid), detectors by anomaly type, mean over seeds\n")
    core = m[core_mask(m)]
    panels = []
    for t in TABLE_ORDER:
        sub = core[(core.table == t) & core.metric.str.startswith("r100_")]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="detector", columns="metric", values="value", aggfunc="mean")
        piv = piv.reindex([d for d in DET_ORDER if d in piv.index])
        piv.columns = [c.replace("r100_", "") for c in piv.columns]
        piv = piv[TYPES]
        panels.append((t, piv))
        md(f"**Table-{t}**\n")
        save_table(piv.round(2), f"T2_types_{t}", "Detector", f"Per-type recall at 100, Table-{t}, core grid.")
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 1.2, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (t, piv) in zip(axes, panels):
        im = ax.imshow(piv.to_numpy(), cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_title(f"Table-{t} ({TABLE_ROWS[t]:,} rows)", fontsize=9, color=INK)
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=("white" if v > 0.6 else INK))
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label("Recall at 100", color=INK2)
    fig.suptitle("Per-type recall at 100 on the core grid (c = 1.5 percent, medium)", fontsize=10, color=INK)
    fig.savefig(FIG / "fig_T2_type_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_T2_type_heatmap.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- T3

def table_t3(m: pd.DataFrame) -> None:
    md("## T3. Sweep grid: P@100 by contamination and difficulty (A, B, C), mean over seeds\n")
    sweep = m[m.table.isin(["A", "B", "C"]) & (m.metric == "p_at_100") & m.detector.isin(DET_ORDER)]
    if sweep.empty:
        return
    cs, ds = [0.005, 0.015, 0.03], ["easy", "medium", "hard"]
    for t in ["C", "A", "B"]:
        sub = sweep[sweep.table == t]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="detector", columns=["c", "d"], values="value", aggfunc="mean")
        cols = [(c, d) for c in cs for d in ds if (c, d) in piv.columns]
        piv = piv.reindex([d for d in DET_ORDER if d in piv.index])[cols]
        piv.columns = [f"c={c:.3f} {d}" for c, d in cols]
        md(f"**Table-{t}**\n")
        save_table(piv.round(2), f"T3_sweep_{t}", "Detector", f"Sweep grid, Table-{t}: P@100 by contamination and difficulty.")
    # figure: P@100 versus contamination, one panel per table, lines per difficulty, for a subset of detectors
    show = [d for d in ["DECOMP", "kNN", "VAE-pyod", "AE-pyod", "PCA", "AVG", "AE-emb", "IF"] if d in sweep.detector.unique()]
    tables = [t for t in ["C", "A", "B"] if not sweep[sweep.table == t].empty]
    fig, axes = plt.subplots(len(tables), 3, figsize=(8.2, 2.3 * len(tables)), sharey=True, squeeze=False)
    for i, t in enumerate(tables):
        for j, d in enumerate(ds):
            ax = axes[i, j]
            for k, det in enumerate(show):
                s = sweep[(sweep.table == t) & (sweep.d == d) & (sweep.detector == det)].groupby("c").value
                mu, sd = s.mean(), s.std()
                if mu.empty:
                    continue
                ax.plot(mu.index * 100, mu.values, marker="o", color=CAT[k % len(CAT)], label=det)
                ax.fill_between(mu.index * 100, (mu - sd).values, (mu + sd).values, color=CAT[k % len(CAT)], alpha=0.12, linewidth=0)
            ax.set_ylim(0, 1)
            ax.set_xticks([0.5, 1.5, 3.0])
            if i == 0:
                ax.set_title(d, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"Table-{t}\nP@100", color=INK2)
            if i == len(tables) - 1:
                ax.set_xlabel("contamination (percent)", color=INK2)
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle("P@100 across contamination and difficulty (sweep grid, mean ± sd over 5 seeds)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_T3_sweep_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_T3_sweep_curves.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- T4

def table_t4(m: pd.DataFrame) -> None:
    md("## T4. Frequency-augmented encoding (E_freq) versus E_base on the sweep grid, mean over all cells of a table\n")
    sub = m[m.table.isin(["A", "B", "C"]) & m.metric.isin(["p_at_100", "ap"])]
    if not sub.detector.str.endswith("+freq").any():
        return
    rows = []
    for t in ["C", "A", "B"]:
        for det in ["IF", "LOF", "kNN", "ECOD"]:
            base = sub[(sub.table == t) & (sub.detector == det)]
            freq = sub[(sub.table == t) & (sub.detector == det + "+freq")]
            if base.empty or freq.empty:
                continue
            # align on cells that have both
            cells = set(map(tuple, freq[["c", "d", "seed"]].drop_duplicates().to_numpy().tolist()))
            base = base[base[["c", "d", "seed"]].apply(tuple, axis=1).isin(cells)]
            r = {"Table": t, "Detector": det, "cells": len(cells)}
            for met, lab in [("p_at_100", "P@100"), ("ap", "AP")]:
                b = base[base.metric == met].value.mean()
                f = freq[freq.metric == met].value.mean()
                r[f"{lab} E_base"] = round(b, 3)
                r[f"{lab} E_freq"] = round(f, 3)
                r[f"{lab} delta"] = round(f - b, 3)
            rows.append(r)
    if rows:
        df = pd.DataFrame(rows).set_index(["Table", "Detector"])
        save_table(df, "T4_freq_ablation", "Table, Detector", "E_freq ablation versus E_base on the sweep grid.")


# ---------------------------------------------------------------- T5

def table_t5(fb: pd.DataFrame | None) -> None:
    if fb is None or fb.empty:
        return
    md("## T5. Feedback protocol: cumulative anomalies found after 25/50/75/100 reviews and final P@k on the unreviewed remainder, mean ± sd over seeds\n")
    tables = [t for t in TABLE_ORDER if t in fb.table.unique()]
    for t in tables:
        sub = fb[fb.table == t]
        rows = {}
        for meth in FEEDBACK_ORDER:
            s = sub[sub.method == meth]
            if s.empty:
                continue
            r = {}
            for rr in [1, 2, 3, 4]:
                g = s[s["round"] == rr].anomalies_found
                r[f"found@{25 * rr}"] = fmt_pm(g.mean(), g.std(), 1)
            last = s[s["round"] == 4]
            for k in [25, 50, 100]:
                r[f"final P@{k}"] = fmt_pm(last[f"p_at_{k}"].mean(), last[f"p_at_{k}"].std())
            r["P@100 adaptive"] = fmt_pm(last.anomalies_found.mean() / 100, last.anomalies_found.std() / 100)
            rows[meth] = r
        df = pd.DataFrame(rows).T
        md(f"**Table-{t}** ({sub.seed.nunique()} seeds)\n")
        save_table(df, f"T5_feedback_{t}", "Method", f"Feedback protocol, Table-{t}.")
    # discovery curves
    fig, axes = plt.subplots(1, len(tables), figsize=(2.6 * len(tables) + 0.8, 2.9), sharey=False, squeeze=False)
    for ax, t in zip(axes[0], tables):
        sub = fb[fb.table == t]
        for k, meth in enumerate(FEEDBACK_ORDER):
            s = sub[sub.method == meth].groupby("round").anomalies_found
            if s.mean().empty:
                continue
            x = [0] + (s.mean().index * 25).tolist()
            y = [0] + s.mean().tolist()
            ls = "--" if meth.endswith("-spec") else "-"
            ax.plot(x, y, marker="o", color=CAT[k % len(CAT)], linestyle=ls, label=meth)
        ax.set_title(f"Table-{t}", fontsize=9, color=INK)
        ax.set_xlabel("reviews", color=INK2)
        ax.set_xticks([0, 25, 50, 75, 100])
    axes[0, 0].set_ylabel("injected anomalies found", color=INK2)
    axes[0, -1].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Discovery curves under the 4 x 25 review protocol (mean over seeds)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_T5_discovery.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_T5_discovery.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- T6

def table_t6(m: pd.DataFrame, ceil: pd.DataFrame | None, fb: pd.DataFrame | None) -> None:
    if ceil is None or ceil.empty:
        return
    md("## T6. Supervised ceiling alongside the best unsupervised detector and the best feedback method (core grid, mean over seeds)\n")
    core = m[core_mask(m) & m.detector.isin(DET_ORDER)]
    rows = []
    for t in TABLE_ORDER:
        c = ceil[(ceil.table == t) & core_mask(ceil)]
        if c.empty:
            continue
        cg = c.groupby("metric").value.mean()
        u = core[core.table == t]
        up = u[u.metric == "p_at_100"].groupby("detector").value.mean()
        best_u = up.idxmax()
        ug = u[u.detector == best_u].groupby("metric").value.mean()
        r = {"Table": t,
             "Ceiling P@25": round(cg["p_at_25"], 2), "Ceiling P@100": round(cg["p_at_100"], 2), "Ceiling AP": round(cg["ap"], 2),
             "Best unsupervised": best_u, "P@25": round(ug["p_at_25"], 2), "P@100": round(ug["p_at_100"], 2), "AP": round(ug["ap"], 2)}
        if fb is not None and not fb[fb.table == t].empty:
            f = fb[(fb.table == t) & (fb["round"] == 4) & ~fb.method.str.endswith("-spec")]
            fg = f.groupby("method").anomalies_found.mean()
            best_f = fg.idxmax()
            r["Best feedback"] = best_f
            r["found@100 / 100"] = round(fg.max() / 100, 2)
            r["found@25 / 25"] = round(fb[(fb.table == t) & (fb.method == best_f) & (fb["round"] == 1)].anomalies_found.mean() / 25, 2)
        rows.append(r)
    if rows:
        df = pd.DataFrame(rows).set_index("Table")
        save_table(df, "T6_ceiling", "Table", "Supervised ceiling versus the best unsupervised and the best feedback method.")


# ---------------------------------------------------------------- T7

def table_t7(m: pd.DataFrame) -> None:
    md("## T7. Friedman test and Nemenyi post-hoc across detectors on the sweep grid (A, B, C; one observation per cell)\n")
    sweep = m[m.table.isin(["A", "B", "C"]) & m.detector.isin(DET_ORDER)]
    if sweep.empty:
        return
    for metric, lab in [("p_at_100", "P@100"), ("ap", "AP")]:
        W = wide(sweep, metric)
        W = W[[d for d in DET_ORDER if d in W.columns]].dropna()
        if W.shape[0] < 3 or W.shape[1] < 3:
            continue
        res = friedman_nemenyi(W)
        ranks = res["mean_ranks"]
        best = ranks.index[0]
        nem = res["nemenyi_p"]
        df = pd.DataFrame({"mean rank": ranks.round(2),
                           "Nemenyi p vs best": [("best" if d == best else f"{nem.loc[best, d]:.3f}") for d in ranks.index],
                           "different from best (CD)": [("" if d == best else ("yes" if ranks[d] - ranks[best] > res["critical_difference"] else "no")) for d in ranks.index]})
        md(f"**{lab}**: Friedman chi2 = {res['statistic']:.1f}, p = {res['p_value']:.2e}, n = {res['n_cells']} cells, k = {res['n_detectors']} detectors, "
           f"critical difference (alpha 0.05) = {res['critical_difference']:.2f}\n")
        save_table(df, f"T7_friedman_{metric}", "Detector", f"Friedman and Nemenyi on the sweep grid, {lab}.")
        nem.round(3).to_csv(TAB / f"T7_nemenyi_matrix_{metric}.csv")


# ---------------------------------------------------------------- DECOMP ablation

def decomp_ablation() -> None:
    files = sorted((RESULTS / "scores").glob("*_c0.015_medium_s*_decomp_components.parquet"))
    if not files:
        return
    md("## T8. DECOMP leave-one-component-out ablation (core grid): per-type recall at 100 when one component is removed, mean over seeds\n")
    rows = []
    for f in files:
        cell = f.name.replace("_decomp_components.parquet", "")
        table = cell.split("_")[0]
        comp = pd.read_parquet(f)
        man = pd.read_parquet(RESULTS / "manifests" / f"{cell}.parquet")
        assert (comp.row_id.to_numpy() == man.row_id.to_numpy()).all()
        types = man.anom_type.to_numpy(dtype=object)
        labels = man.label.to_numpy()
        comps = [c for c in comp.columns if c != "row_id"]
        P = comp[comps].to_numpy()
        variants = {"full": P.max(axis=1)}
        for j, c in enumerate(comps):
            variants[f"without {c}"] = np.delete(P, j, axis=1).max(axis=1)
        for name, sc in variants.items():
            r = recall_at_k_by_type(sc, types, 100)
            r.update(table=table, variant=name, p_at_100=precision_at_k(sc, labels, 100))
            rows.append(r)
    df = pd.DataFrame(rows)
    for t in TABLE_ORDER:
        sub = df[df.table == t]
        if sub.empty:
            continue
        piv = sub.groupby("variant")[["p_at_100"] + TYPES].mean().round(2)
        piv = piv.reindex(["full"] + [v for v in piv.index if v != "full"])
        piv.columns = ["P@100"] + TYPES
        md(f"**Table-{t}**\n")
        save_table(piv, f"T8_decomp_ablation_{t}", "Variant", f"DECOMP component ablation, Table-{t}.")


# ---------------------------------------------------------------- post-hoc pair score

def posthoc_pair_score() -> None:
    """Post-hoc diagnostic, outside the frozen protocol: DECOMP with its pair
    component replaced by the smoothed conditional surprise -log P(g | a).
    Explains the T5 result; not a benchmark entry."""
    from .detectors import decomp_components
    from .encode import _standardise  # noqa: F401  (import check)
    from .inject import inject
    from .load import ACCOUNT, CONTRA, load_table
    from .evaluate import rank_normalise

    files = sorted((RESULTS / "manifests").glob("*_c0.015_medium_s*.parquet"))
    if not files:
        return
    md("## T9. Post-hoc diagnostic (outside the frozen protocol): DECOMP with the pair component replaced by the smoothed conditional surprise -log P(contra | account)\n")
    rows = []
    cache = {}
    for f in files:
        cell = f.stem
        table, c, d, seed = cell.split("_")[0], 0.015, "medium", int(cell.split("_s")[-1])
        if table not in cache:
            cache[table] = load_table(table)
        inj = inject(cache[table], table, c, d, seed)
        man = pd.read_parquet(f)
        assert inj.meta["dataset_hash"] == man.dataset_hash.iloc[0]
        df = inj.data
        comp = decomp_components(df)
        ca = df[ACCOUNT].map(df[ACCOUNT].value_counts()).to_numpy(dtype=float)
        cag = df.groupby([ACCOUNT, CONTRA], sort=False)[ACCOUNT].transform("size").to_numpy(dtype=float)
        G = df[CONTRA].nunique()
        alt = -np.log((cag + 0.5) / (ca + 0.5 * G))
        comp_alt = comp.copy()
        comp_alt["s_pair"] = alt
        types = man.anom_type.to_numpy(dtype=object)
        labels = man.label.to_numpy()
        for name, cc in [("DECOMP (protocol)", comp), ("DECOMP with -log P(g|a)", comp_alt)]:
            sc = np.column_stack([rank_normalise(cc[k].to_numpy()) for k in cc.columns]).max(axis=1)
            r = recall_at_k_by_type(sc, types, 100)
            r.update(table=table, variant=name, p_at_100=precision_at_k(sc, labels, 100))
            rows.append(r)
        # the pair component alone, for reference
        for name, v in [("pair component alone (protocol)", comp["s_pair"].to_numpy()), ("pair component alone (-log P(g|a))", alt)]:
            r = recall_at_k_by_type(v, types, 100)
            r.update(table=table, variant=name, p_at_100=precision_at_k(v, labels, 100))
            rows.append(r)
    df = pd.DataFrame(rows)
    piv = df.groupby(["table", "variant"])[["p_at_100"] + TYPES].mean().round(2)
    piv = piv.reindex([(t, v) for t in TABLE_ORDER for v in ["DECOMP (protocol)", "DECOMP with -log P(g|a)", "pair component alone (protocol)", "pair component alone (-log P(g|a))"] if (t, v) in piv.index])
    piv.columns = ["P@100"] + TYPES
    save_table(piv, "T9_posthoc_pair", "Table, Variant", "Post-hoc pair-score diagnostic on the core grid.")


# ---------------------------------------------------------------- scale figure

def figure_scale(m: pd.DataFrame) -> None:
    core = m[core_mask(m) & (m.metric == "p_at_100") & m.detector.isin(DET_ORDER)]
    tables = [t for t in TABLE_ORDER if t in core.table.unique()]
    if len(tables) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    highlight = ["DECOMP", "kNN", "VAE-pyod", "AE-pyod", "AE-emb", "AVG", "IF", "PCA"]
    x = [TABLE_ROWS[t] for t in tables]
    for det in DET_ORDER:
        s = core[core.detector == det].groupby("table").value
        mu = s.mean().reindex(tables)
        if mu.isna().all():
            continue
        if det in highlight:
            k = highlight.index(det)
            ax.plot(x, mu.values, marker="o", color=CAT[k], label=det, zorder=3)
        else:
            ax.plot(x, mu.values, marker="o", color=MUTED, alpha=0.5, linewidth=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{t}\n{TABLE_ROWS[t]:,}" for t in tables])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P@100 (c = 1.5 percent, medium)", color=INK2)
    ax.set_xlabel("table (rows)", color=INK2)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax.set_title("Scale effect of P@100 across the four tables", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scale_p100.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_scale_p100.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def runtime_table(m: pd.DataFrame) -> None:
    md("## Runtimes: mean wall-clock seconds per detector fit on the core grid (2 CPU cores)\n")
    core = m[core_mask(m) & (m.metric == "p_at_100")]
    piv = core.pivot_table(index="detector", columns="table", values="runtime_s", aggfunc="mean")
    piv = piv.reindex([d for d in DET_ORDER if d in piv.index])[[t for t in TABLE_ORDER if t in piv.columns]].round(1)
    save_table(piv, "runtimes_core", "Detector", "Mean runtime per detector and table on the core grid.")


def main() -> None:
    m, ceil, fb = load()
    md("# Result tables (generated by src/analyze.py)\n")
    md(f"Cells with metrics: {m[['table','c','d','seed']].drop_duplicates().shape[0]}; "
       f"feedback cells: {0 if fb is None else fb[['table','c','d','seed']].drop_duplicates().shape[0]}.\n")
    table_t1(m)
    table_t2(m)
    table_t3(m)
    table_t4(m)
    table_t5(fb)
    table_t6(m, ceil, fb)
    table_t7(m)
    decomp_ablation()
    posthoc_pair_score()
    figure_scale(m)
    runtime_table(m)
    (REP / "tables.md").write_text("\n".join(MD))
    print("\n".join(MD))


if __name__ == "__main__":
    main()
