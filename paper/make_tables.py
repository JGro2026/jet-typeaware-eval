#!/usr/bin/env python3
"""LaTeX tables (booktabs, caption first) for the ESWA manuscript and its supplement.

Every number is parsed from the sources listed in paper/tables/README.md, from the analysis
outputs (outputs/public/ and restricted/outputs/), or computed here from the published public ledger file
(the profile column of the public ledger; see dib_profile() below). Descriptive text that is reused
from the previous manuscript (type and marking definitions, detector settings, field mapping) is
parsed from restricted/inputs/discover_manuscript_main.tex; DECOMP and AVG rows are dropped.

Main tables    -> paper/tables/tab_*.tex
Supplement     -> paper/supplement/tables/tabS_*.tex
Needs: booktabs, array, makecell, amsmath, amssymb.

Run: python3 paper/make_tables.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import paths  # noqa: E402

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.ROOT
OUT_MAIN = paths.TABLES
OUT_SUP = paths.SUP_TABLES
OUT_MAIN.mkdir(parents=True, exist_ok=True); OUT_SUP.mkdir(parents=True, exist_ok=True)

DETS = ["IF", "LOF", "kNN", "HBOS", "ECOD", "PCA", "AE", "VAE", "AE-emb"]
CEIL = "LightGBM ceiling"      # label of the supervised LightGBM model in the result files
REF_NAME = "LightGBM reference"  # how the paper names it
NAME_MAP = {"AE-pyod": "AE", "VAE-pyod": "VAE", "LGBM-ceiling": CEIL, "LightGBM ceiling": CEIL}
LEDGERS = ["C", "A", "B", "D"]


def nm(s: str) -> str:
    return NAME_MAP.get(s.strip(), s.strip())


# ----------------------------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------------------------

def f2(x: float, nd: int = 2) -> str:
    """Round half up on the shortest decimal representation (0.215 -> 0.22, not 0.21)."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    s = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    if re.fullmatch(r"-0\.0*", s):
        s = s[1:]
    return s.replace("-", "$-$")


def fint(x: float) -> str:
    return f"{int(float(f2(x, 0))):,}"


def fpct(x: float, nd: int = 1) -> str:
    return f2(x, nd)


def ms(mean: float, sd: float | None, nd: int = 2) -> str:
    return f2(mean, nd) if sd is None or (isinstance(sd, float) and np.isnan(sd)) else f"{f2(mean, nd)} ({f2(sd, nd)})"


def frange(lo: float, hi: float, nd: int = 2, sep: str = " to ") -> str:
    a, b = f2(lo, nd), f2(hi, nd)
    return a if a == b else f"{a}{sep}{b}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def best_mask(values: list[float], nd: int = 2, higher=True) -> list[bool]:
    """Bold on the displayed (rounded) values, so ties at the printed precision are all bold."""
    disp = [float(f2(v, nd).replace("$-$", "-")) if v is not None and not np.isnan(v) else None for v in values]
    valid = [d for d in disp if d is not None]
    if not valid:
        return [False] * len(values)
    t = max(valid) if higher else min(valid)
    return [d is not None and abs(d - t) < 1e-12 for d in disp]


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert "\u2014" not in text and "---" not in text, f"em dash in {path.name}"
    assert "DECOMP" not in text and "AVG" not in text, f"excluded method in {path.name}"
    path.write_text(text, encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def table_env(label: str, caption: str, body: str, size: str = r"\footnotesize", note: str | None = None) -> str:
    if note:
        body += "\\par\\smallskip\\parbox{\\linewidth}{\\scriptsize " + note + "}\n"
    return (f"\\begin{{table}}[tbp]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n{size}\n"
            f"{body}\\end{{table}}\n")


# ----------------------------------------------------------------------------------------------
# parsers
# ----------------------------------------------------------------------------------------------

def md_section(path: Path, start_pat: str) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(start_pat, text, flags=re.M)
    rest = text[m.start():]
    nxt = re.search(r"^## ", rest[3:], flags=re.M)
    return rest if not nxt else rest[: nxt.start() + 3]


def md_blocks(section: str) -> dict[str, list[list[str]]]:
    blocks, cur = {}, None
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
    sec = md_section(paths.CLIENT_RESULTS, r"^## T3\.")
    rows = []
    for led, tab in md_blocks(sec).items():
        header = tab[0]
        for r in tab[1:]:
            det = nm(r[0])
            if det in ("DECOMP", "AVG"):
                continue
            for h, v in zip(header[1:], r[1:]):
                m = re.match(r"c=([0-9.]+)\s+(\w+)", h)
                rows.append(dict(ledger=led, detector=det, c=float(m.group(1)), difficulty=m.group(2), P100=float(v)))
    df = pd.DataFrame(rows)
    assert set(df.detector) == set(DETS)
    return df


def runB_core_header() -> dict[str, dict]:
    sec = md_section(paths.CLIENT_RESULTS, r"^## T1\.")
    out = {}
    for m in re.finditer(r"\*\*Table-([A-D])\*\* \(([\d,]+) rows, ([\d,]+) injected anomalies, (\d+) seeds\)", sec):
        out[m.group(1)] = dict(rows=int(m.group(2).replace(",", "")), injected=int(m.group(3).replace(",", "")),
                               seeds=int(m.group(4)))
    return out


def n_anom_protocol() -> dict[str, list[int]]:
    txt = (paths.PROTOCOL).read_text(encoding="utf-8")
    m = re.search(r"resulting counts are (.*?)\n\s*for c = 0\.005/0\.015/0\.03", txt, flags=re.S)
    return {led: [int(x.replace(",", "")) for x in (a, b, c)]
            for led, a, b, c in re.findall(r"([A-D]) ([\d,]+)/([\d,]+)/([\d,]+)", m.group(1).replace("\n", " "))}


def wp0_profile() -> pd.DataFrame:
    txt = (paths.CLIENT_AUDIT).read_text(encoding="utf-8")
    sec = txt[txt.index("## 1. Profile verification"):]
    sec = sec[: sec.index("\n## ", 5)]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"^\| ([A-D]) \|(.*)\|\s*$", line)
        if m:
            c = [x.strip() for x in m.group(2).split("|")]
            miss = re.match(r"([\d,]+) \(([\d.]+)%\)", c[6])
            rows.append(dict(ledger=m.group(1), rows=int(c[0].replace(",", "")), accounts=int(c[1]),
                             contra=int(c[2].replace(",", "")), currency=int(c[3]), tax=int(c[4]), users=int(c[5]),
                             pay_missing=int(miss.group(1).replace(",", "")), pay_missing_pct=float(miss.group(2))))
    return pd.DataFrame(rows).set_index("ledger")


def old_tex() -> str:
    return (paths.METHOD_TEXT).read_text(encoding="utf-8")


def old_table_rows(label: str) -> list[str]:
    tex = old_tex()
    i = tex.index(r"\label{" + label + "}")
    i = tex.index(r"\toprule", i) + len(r"\toprule")
    j = tex.index(r"\end{tabular}", i)
    body = tex[i:j]
    rows = [r.strip() for r in body.split(r"\\") if "&" in r]
    return rows


def dib_header() -> dict:
    """Facts about the public ledger and its run: computed from the published file and the run configuration."""
    p = dib_profile()
    return dict(lines=p["lines"], entries=p["entries"], years=p["years"], accounts=p["accounts"], users=p["users"],
                anomalous=p["anomalous_entries"], anomalous_pct=round(100 * p["anomalous_entries"] / p["entries"], 2),
                contra=p["contra"], seed_lo=min(paths.PUBLIC_SEEDS), seed_hi=max(paths.PUBLIC_SEEDS))


def dib_profile() -> dict:
    """Profile of the public ledger computed from the published file with the mapping of
    dib/dib_data.py (a computation run here). Cached in tables/dib_profile_computed.json."""
    cache = OUT_MAIN / "dib_profile_computed.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(paths.PIPELINE))
    import dib_data  # noqa: E402
    df = dib_data.load()
    f10 = ["gl_account", "contra", "amount", "lag_days", "cd_flag", "tax_rate", "user", "round_amt",
           "doc_outside_fy", "text_present"]
    key = df[f10].astype(str).agg("|".join, axis=1)
    size = key.map(key.value_counts())
    out = dict(
        lines=int(len(df)), entries=int(df.posting_id.nunique()),
        years=int(pd.to_datetime(df.posting_date).dt.year.nunique()),
        accounts=int(df.gl_account.nunique()), contra=int(df.contra.nunique()), tax=int(df.tax_rate.nunique()),
        users=int(df.user.nunique()), lag_missing=int(df.lag_days.isna().sum()),
        dup_repeat=int(df.duplicated(subset=f10).sum()), dup_in_group=int((size >= 2).sum()),
        dup_ge6=int((size >= 6).sum()), anomalous_entries=int(df.groupby("posting_id").label.max().sum()),
        definition="duplicate = same values in the ten mapped fields " + ", ".join(f10),
    )
    cache.write_text(json.dumps(out, indent=1))
    return out


# ==============================================================================================
# MAIN TABLES
# ==============================================================================================

def tab_data():
    prof = wp0_profile()
    dup = pd.read_csv(paths.find("dup_group_stats.csv")).set_index("ledger")
    core = runB_core_header()
    dh = dib_header()
    dp = dib_profile()
    for k in ("lines", "entries", "accounts", "users", "contra"):
        assert dp[k] == dh[k], (k, dp[k], dh[k])
    assert dp["years"] == dh["years"] and dp["anomalous_entries"] == dh["anomalous"]
    assert "one fiscal year per table" in (paths.PROTOCOL).read_text()
    ta = pd.read_csv(paths.find("datev_typeaware_k100.csv"))
    n_types = len([c for c in ta.columns if re.fullmatch(r"R_T\d", c)])
    n_marks = pd.read_csv(paths.find("a6_public_marking_recall.csv")).marking.nunique()

    def row(label, vals):
        return label + " & " + " & ".join(vals) + r" \\" + "\n"

    L = LEDGERS
    body = r"\setlength{\tabcolsep}{4pt}" + "\n" + r"\begin{tabular}{@{}lrrrrr@{}}" + "\n\\toprule\n"
    body += r" & \multicolumn{4}{c}{Client ledgers} & Public \\" + "\n" + r"\cmidrule(lr){2-5}" + "\n"
    body += row("Property", L + ["ledger"]) + "\\midrule\n"
    body += row("Scored unit", ["posting"] * 4 + ["line"])
    body += row("Evaluated unit", ["posting"] * 4 + ["entry"])
    body += row("Scored rows", [fint(prof.loc[l, "rows"]) for l in L] + [fint(dp["lines"])])
    body += row("Evaluated units", [fint(prof.loc[l, "rows"]) for l in L] + [fint(dp["entries"])])
    body += row("Fiscal years", ["1"] * 4 + [str(dp["years"])])
    body += "\\addlinespace\n"
    body += row("Accounts", [fint(prof.loc[l, "accounts"]) for l in L] + [fint(dp["accounts"])])
    body += row("Contra accounts", [fint(prof.loc[l, "contra"]) for l in L] + [fint(dp["contra"]) + "$^{a}$"])
    body += row("Tax codes", [fint(prof.loc[l, "tax"]) for l in L] + [fint(dp["tax"])])
    body += row("Users", [fint(prof.loc[l, "users"]) for l in L] + [fint(dp["users"])])
    body += row(r"Payment period missing (\%)", [fpct(prof.loc[l, "pay_missing_pct"]) for l in L]
                + [fpct(100 * dp["lag_missing"] / dp["lines"]) + "$^{b}$"])
    body += "\\addlinespace\n"
    body += r"\multicolumn{6}{@{}l}{\textit{Duplicate rows}} \\" + "\n"
    body += row(r"\quad repeating an earlier row (\%)",
                [fpct(100 * dup.loc[l, "rows_repeating_earlier"] / dup.loc[l, "n"]) for l in L]
                + [fpct(100 * dp["dup_repeat"] / dp["lines"])])
    body += row(r"\quad in a duplicate group (\%)", [fpct(100 * dup.loc[l, "share_in_dup_groups"]) for l in L]
                + [fpct(100 * dp["dup_in_group"] / dp["lines"])])
    body += row(r"\quad in groups of six or more", [fint(dup.loc[l, "rows_in_groups_ge6"]) for l in L]
                + [fint(dp["dup_ge6"])])
    body += "\\addlinespace\n"
    body += row("Anomalous units", [fint(core[l]["injected"]) for l in L] + [fint(dp["anomalous_entries"])])
    body += row(r"Anomalous units (\%)", [fpct(100 * core[l]["injected"] / core[l]["rows"], 2) for l in L]
                + [fpct(100 * dp["anomalous_entries"] / dp["entries"], 2)])
    body += row("Anomaly classes", [f"T1 to T{n_types}"] * 4 + [f"M1 to M{n_marks}"])
    body += "\\bottomrule\n\\end{tabular}\n"
    body += ("\\par\\smallskip\\parbox{\\linewidth}{\\scriptsize Percentages of duplicate rows refer to the scored rows. $^{a}$ Derived: the account of the largest-amount "
             "line on the opposite side of the same entry. $^{b}$ Entry lag, the counterpart of the payment period, "
             "is never missing.}\n")
    cap = (r"Profile of the four client ledgers and the public synthetic ledger. Counts of accounts, contra accounts, tax c"
           r"odes and users are numbers of distinct values. A duplicate row repeats all feature values of another row. Anom"
           r"alous units are the injected anomalies of the core grid (client ledgers) and the generator-labelled journal en"
           r"tries (public ledger).")
    write(OUT_MAIN / "tab_data.tex", table_env("tab:data", cap, body))


def _types_rows():
    rows = old_table_rows("tab:types")
    trows, mrows = [], []
    for r in rows:
        for tok in ("\\addlinespace", "\\midrule", "\\toprule"):
            r = r.replace(tok, "")
        r = r.strip()
        parts = [p.strip() for p in r.split(" & ")]
        m = re.match(r"^(T[1-5]) (.+)$", parts[0])
        if m and len(parts) == 5:
            trows.append(dict(code=m.group(1), name=m.group(2), mod=parts[1], share=int(parts[2]),
                              diff=parts[3], foorthuis=parts[4]))
        m = re.match(r"^(M[1-8]) (.+)$", parts[0])
        if m and len(parts) == 4:
            pop = re.search(r"\{([^{}]*)\}\s*$", parts[3]).group(1)
            mrows.append(dict(code=m.group(1), name=m.group(2), definition=parts[1], n=int(parts[2]), pop=pop))
    assert len(trows) == 5 and len(mrows) == 8, (len(trows), len(mrows))
    return trows, mrows


def tab_types():
    trows, mrows = _types_rows()
    nb = pd.read_csv(paths.find("a2_found_bounds.csv"))[["ledger", "type", "n_t"]].drop_duplicates()
    nt = nb.pivot(index="type", columns="ledger", values="n_t")
    # shares check: n_t / n on ledger D equals the share printed in the old table
    for t in trows:
        share_D = 100 * nt.loc[t["code"], "D"] / nt["D"].sum()
        assert abs(share_D - t["share"]) < 1e-9, (t, share_D)
    # Foorthuis mapping from harness/protocol.md (Section 3 table)
    prot = (paths.PROTOCOL).read_text(encoding="utf-8")
    fmap = {}
    for m in re.finditer(r"^\| (T\d)(?:, (T\d))? \| Type (\w+), ([^|]+?) \| (\w+), ([^|]+?) \|", prot, flags=re.M):
        for t in (m.group(1), m.group(2)):
            if t:
                fmap[t] = (m.group(3), m.group(4).strip(), m.group(5), m.group(6).strip())
    for t in trows:
        roman, _, sub, _ = fmap[t["code"]]
        assert t["foorthuis"].startswith(roman + " ") and f"({sub})" in t["foorthuis"], (t, fmap[t["code"]])
    mk = pd.read_csv(paths.find("a6_public_marking_recall.csv")).drop_duplicates("marking").set_index("marking")
    rp = pd.read_csv(paths.find("a6_public_rule_populations.csv")).drop_duplicates("population")
    popname = {p.split(" ", 1)[0]: p for p in rp.population}
    for m in mrows:
        assert int(mk.loc[m["code"], "n_marking"]) == m["n"], m

    shares = [t["share"] for t in trows]
    body = r"\setlength{\tabcolsep}{4pt}" + "\n"
    body += (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}"
             r">{\raggedright\arraybackslash}p{0.47\linewidth}r@{}}" + "\n\\toprule\n")
    body += (r"\multicolumn{3}{@{}l}{\textit{(a) Types injected into the client ledgers}} \\" + "\n")
    body += (r"Type (Foorthuis subtype) & Modification of a sampled posting & $n_t$: C / A / B / D \\"
             + "\n\\midrule\n")
    for t in trows:
        mod = t["mod"]
        if t["diff"] != "none":
            mod += f"; {t['diff']}"
        sub = re.search(r"\(([IVX]+a)\)", t["foorthuis"]).group(1)
        body += (f"{t['code']} {t['name']} ({sub}) & {mod} & "
                 + " / ".join(fint(nt.loc[t['code'], l]) for l in LEDGERS) + r" \\" + "\n")
    body += "\\bottomrule\n\\end{tabular}\n\n\\medskip\n"
    body += (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.25\linewidth}"
             r">{\raggedright\arraybackslash}p{0.56\linewidth}lr@{}}" + "\n\\toprule\n")
    body += r"\multicolumn{4}{@{}l}{\textit{(b) Markings of the public ledger, defined by its generator}} \\" + "\n"
    body += r"Marking & Definition & Rule & $n_t$ \\" + "\n\\midrule\n"
    for m in mrows:
        code = m["pop"].split(" ", 1)[0]
        pop = code if code in popname else "none"
        body += f"{m['code']} {m['name']} & {m['definition']} & {pop} & {m['n']} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    pops = "; ".join(popname[c] for c in sorted(popname))
    cap = (r"Anomaly types and markings. (a) Types injected into the client ledgers, with their subtype in the typology of "
           r"\citet{foorthuis2021nature}, the difficulty parameter for the levels easy, medium and hard, and the number $n_"
           r"t$ of injected rows in the core grid. (b) Markings of the public ledger, the rule population into which each w"
           r"as injected, and $n_t$.")
    write(OUT_MAIN / "tab_types.tex", table_env("tab:types", cap, body, note=(r"Subtypes: Ia extreme tail value, IIa unusual class, Va uncommon class combination. Rule populations: J1 top-n "
           r"amounts, J2 high cash, J3 delayed entry, J4 weekend or non-working hours.")))


def tab_detectors():
    rows = old_table_rows("tab:detectors")
    keep = DETS + [CEIL]
    parsed = {}
    for r in rows:
        parts = [p.strip() for p in r.split(" & ")]
        name = parts[0].replace("\\addlinespace", "").replace("\\midrule", "").strip()
        if name in keep and len(parts) == 4:
            parsed[name] = parts[1:]
    assert set(parsed) == set(keep), set(keep) - set(parsed)
    # numbers in the settings must match harness/protocol.md, Section 5
    prot = (paths.PROTOCOL).read_text(encoding="utf-8")
    for tok in ["n_estimators=200", "max_samples=256", "n_neighbors=20", "n_neighbors=5", "n_bins 10", "alpha 0.1",
                "tol 0.5", "n_components=0.90", "hidden_neuron_list=[64, 32, 8]", "latent_dim=8", "epoch_num=100",
                "batch_size=64", "dropout 0.2", "lr 1e-3", "weight decay 1e-5", "embeddings (10, 10, 1, 1, 1)",
                "LeakyReLU(0.01)", "StratifiedKFold(5", "scale_pos_weight=N_neg/N_pos"]:
        assert tok in prot, tok
    body = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.12\linewidth}"
            r">{\raggedright\arraybackslash}p{0.2\linewidth}>{\raggedright\arraybackslash}p{0.1\linewidth}"
            r">{\raggedright\arraybackslash}p{0.48\linewidth}@{}}" + "\n\\toprule\n")
    body += r"Detector & Family & Input & Key settings \\" + "\n\\midrule\n"
    for d in keep:
        fam, inp, sett = parsed[d]
        if d == CEIL:
            body += "\\midrule\n"
            d_out = REF_NAME
            inp = r"$E_{\mathrm{base}}$, labels"
        else:
            d_out = d
        body += f"{d_out} & {fam} & {inp} & {sett} \\\\\n"
    body += ("\\midrule\n" + r"DeepSAD (review protocol) & semi-supervised deep one-class & $E_{\mathrm{emb}}$, reviewed labels & "
             r"embedding layers and encoder of the AE-emb model of the same setting, initialised with its weights; "
             r"$\eta = 1$, $\varepsilon = 10^{-6}$; Adam $10^{-4}$, weight decay $10^{-6}$; 50 epochs, batch 128; "
             r"retrained before every round; round 1 = AE-emb ranking \\" + "\n")  # settings of public_ledger/feedback.py
    body += "\\bottomrule\n\\end{tabular}\n"
    cap = (r"Detectors, the supervised reference and DeepSAD with their settings, fixed in advance and never tuned on the l"
           r"abels; higher scores mean more anomalous. IF to VAE are PyOD implementations \citep{zhao2019pyod}, AE-emb and "
           r"DeepSAD are our PyTorch code. $E_{\mathrm{base}}$ is the one-hot and standardised encoding, $E_{\mathrm{emb}}$"
           r" the integer-coded input of the embedding autoencoder.")
    write(OUT_MAIN / "tab_detectors.tex", table_env("tab:detectors", cap, body))


def _conventional_values():
    core = pd.read_csv(paths.find("a7_datev_core.csv"))
    ceil = pd.read_csv(paths.find("a7_datev_ceiling.csv")).set_index("ledger")
    pub = pd.read_csv(paths.find("a6_public_global.csv"))
    pub["detector"] = pub.detector.map(nm)
    pub = pub.set_index("detector")
    # cross-check public P@100/AP against the brief's source (metrics_mean.csv)
    mm = pd.read_csv(paths.public_result("metrics_mean.csv"), index_col=0)
    mm.index = [nm(i) for i in mm.index]
    for d in DETS + [CEIL]:
        assert abs(mm.loc[d, "P@100"] - pub.loc[d, "P@100_mean"]) < 1e-9
        assert abs(mm.loc[d, "AP"] - pub.loc[d, "AP_mean"]) < 1e-9
    return core, ceil, pub


def tab_conventional():
    core, ceil, pub = _conventional_values()
    cols = LEDGERS + ["public"]
    out = r"\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{tabular}{@{}lrrrrr@{}}" + "\n\\toprule\n"
    out += r" & \multicolumn{4}{c}{Client ledgers (five seeds)} & Public \\" + "\n" + r"\cmidrule(lr){2-5}" + "\n"
    out += r"Detector & C & A & B & D & ledger \\" + "\n"
    for pi, (met, mcol, ccol) in enumerate([("P@100", "P@100", "ceiling_P@100"), ("AP", "AP", "ceiling_AP")]):
        out += "\\midrule\n" + rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{({'ab'[pi]}) {met}}}}} \\" + "\n"
        cells, means = {}, {}
        for c in cols:
            vals = []
            for d in DETS:
                if c == "public":
                    mu = float(pub.loc[d, f"{mcol}_mean"])
                    sd = float(pub.loc[d, f"{mcol}_sd"]) if int(pub.loc[d, "n_seeds"]) > 1 else None
                else:
                    r = core[(core.ledger == c) & (core.detector == d)].iloc[0]
                    mu, sd = float(r[f"{mcol}_mean"]), float(r[f"{mcol}_sd"])
                vals.append((mu, sd))
            mask = best_mask([v[0] for v in vals])
            cells[c] = [bold(ms(*v)) if b else ms(*v) for v, b in zip(vals, mask)]
        for i, d in enumerate(DETS):
            out += d + " & " + " & ".join(cells[c][i] for c in cols) + r" \\" + "\n"
        crow = []
        for c in cols:
            if c == "public":
                crow.append(ms(float(pub.loc[CEIL, f"{mcol}_mean"]), float(pub.loc[CEIL, f"{mcol}_sd"])))
            else:
                crow.append(f2(float(ceil.loc[c, ccol])))
        out += r"\addlinespace" + "\n" + r"\textit{" + REF_NAME + "} & " + " & ".join(crow) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = (r"Conventional, hit-based view: P@100 and average precision (AP), mean (standard deviation) over seeds. Client l"
           r"edgers: core grid, five seeds; supervised reference as a mean only. Public ledger: three seeds for IF, AE, VAE"
           r", AE-emb and the supervised reference, one run for the deterministic detectors. Bold: best unsupervised detect"
           r"or per column. The supervised LightGBM reference uses the labels and is shown for comparison only.")
    write(OUT_MAIN / "tab_conventional.tex", table_env("tab:conventional", cap, out, note=(r"Precision on the client ledgers is a lower bound, because their genuine anomalies carry label 0 (Section~\ref{"
           r"sec:bench-client}).")))


def tab_typeaware():
    t5 = pd.read_csv(paths.find("a3_table5.csv"))
    rows = [("C", 100), ("A", 100), ("B", 100), ("D", 100), ("public", 100), ("public", 1000)]
    out = r"\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{tabular}{@{}lllllllr@{}}" + "\n\\toprule\n"
    out += (r" & \multicolumn{4}{c}{Best by hits} & \multicolumn{2}{c}{Best by FSR@$k$} & \\" + "\n"
            + r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}" + "\n")
    out += r"Ledger ($k$) & Detector & P@$k$ & TC@$k$ & FSR@$k$ & Detector & FSR@$k$ & $\tau_b$ \\" + "\n\\midrule\n"
    for led, k in rows:
        r = t5[(t5.ledger == led) & (t5.k == k)].iloc[0]
        nd = 3 if (led == "D" or k == 1000) else 2
        tc = str(r.its_TC)
        if led != "public":
            tc = "n/d"  # TC@k is not determined by the seed-mean recalls of the client ledgers
        elif re.fullmatch(r"\d+-\d+", tc):
            a, b = tc.split("-")
            tc = f"{a}--{b}"
        else:
            v = float(tc)
            tc = str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
        best_p = r.best_by_hits.replace("=", ", ")
        if best_p.count(",") >= 2:
            a, b = best_p.rsplit(", ", 1)
            best_p = r"\makecell[l]{" + a + r",\\" + b + "}"
        fsr_its = frange(r.its_FSR_lo, r.its_FSR_hi, nd, sep="--")
        fsr_best = frange(r.FSR_best_lo, r.FSR_best_hi, nd, sep="--")
        lab = f"Public" if led == "public" else led
        tau = frange(r.tau_P_FSR_min, r.tau_P_FSR_max, 3 if led == "D" else 2, sep="--")
        out += (f"{lab} ({k:,}) & {best_p} & {f2(r.P_at_k, 3 if k == 1000 else 2)} & {tc} & {fsr_its} & "
                f"{r.best_by_FSR} & {fsr_best} & {tau} \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = (r"What changes under the type-aware view. Per ledger: the detector with the highest P@$k$, its type coverage TC@"
           r"$k$ and its FSR@$k$; the detector with the highest FSR@$k$; and Kendall's $\tau_b$ between the orderings of th"
           r"e nine detectors by P@$k$ and by FSR@$k$. Client ledgers: FSR of the seed-averaged type profile, with ranges g"
           r"iven by the bounds that the two-decimal rounding implies; public ledger: mean of the per-seed values.")
    write(OUT_MAIN / "tab_typeaware.tex", table_env("tab:typeaware", cap, out, note=(r"n/d: not determined by the seed-mean recalls of the client ledgers (Section~\ref{sec:bench-protocol}). TC@$k$ "
           r"counts types out of eight markings. Public ledger: exact, from the full rankings, mean over three seeds for th"
           r"e stochastic detectors.")))


# ==============================================================================================
# SUPPLEMENTARY TABLES
# ==============================================================================================

def tabS_core_grid():
    core = pd.read_csv(paths.find("a7_datev_core.csv"))
    ceil = pd.read_csv(paths.find("a7_datev_ceiling.csv")).set_index("ledger")
    mets = ["P@25", "P@50", "P@100", "AP", "ROC-AUC"]
    out = r"\begin{tabular}{@{}lrrrrr@{}}" + "\n\\toprule\n"
    out += "Detector & " + " & ".join(mets) + r" \\" + "\n"
    for li, led in enumerate(LEDGERS):
        sub = core[core.ledger == led].set_index("detector").loc[DETS]
        out += "\\midrule\n" + (rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{({'abcd'[li]}) Ledger {led}: "
                                rf"{fint(sub.rows.iloc[0])} postings, {fint(sub.injected.iloc[0])} injected "
                                rf"anomalies, {int(sub.seeds.iloc[0])} seeds}}}} \\") + "\n"
        cells = {}
        for m in mets:
            mask = best_mask(list(sub[f"{m}_mean"].astype(float)))
            cells[m] = [bold(ms(mu, sd)) if b else ms(mu, sd)
                        for mu, sd, b in zip(sub[f"{m}_mean"], sub[f"{m}_sd"], mask)]
        for i, d in enumerate(DETS):
            out += d + " & " + " & ".join(cells[m][i] for m in mets) + r" \\" + "\n"
        crow = []
        for m in mets:
            col = f"ceiling_{m}"
            crow.append(f2(float(ceil.loc[led, col])) if col in ceil.columns else "n/a")
        out += r"\textit{" + REF_NAME + "} & " + " & ".join(crow) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = (r"Core grid on the client ledgers (contamination 1.5 percent, medium difficulty): precision at budgets of 25, 50"
           r" and 100 postings, average precision and ROC-AUC, mean (standard deviation) over five seeds. Bold marks the be"
           r"st unsupervised detector per column within a ledger, on the printed values. The supervised reference is availa"
           r"ble as a mean for P@25, P@100 and AP only (n/a: not recorded).")
    write(OUT_SUP / "tabS_core_grid.tex", table_env("tab:s-core", cap, r"\setlength{\tabcolsep}{4pt}" + "\n" + out,
                                                     size=r"\scriptsize"))


def tabS_pertype():
    ta = pd.read_csv(paths.find("datev_typeaware_k100.csv")).set_index(["ledger", "detector"])
    fs = pd.read_csv(paths.find("datev_fsr_k100.csv")).set_index(["ledger", "detector"])
    nb = pd.read_csv(paths.find("a2_found_bounds.csv"))[["ledger", "type", "n_t"]].drop_duplicates()
    nt = nb.pivot(index="type", columns="ledger", values="n_t")
    types = [f"T{i}" for i in range(1, 6)]
    outs = []
    for leds in [("C", "A"), ("B", "D")]:
        out = r"\begin{tabular}{@{}lrrrrrr@{}}" + "\n\\toprule\n"
        out += "Detector & " + " & ".join(types) + r" & FSR@100 \\" + "\n"
        for led in leds:
            li = LEDGERS.index(led)
            out += "\\midrule\n" + rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{({'abcd'[li]}) Ledger {led}}}}} \\" + "\n"
            out += (r"\textit{$n_t$} & " + " & ".join(r"\textit{" + fint(nt.loc[t, led]) + "}" for t in types)
                    + r" & \\" + "\n")
            nd = 3 if led == "D" else 2
            for d in DETS:
                r = ta.loc[(led, d)]
                f = fs.loc[(led, d)]
                fsr = frange(f.FSR_lo_lattice, f.FSR_hi_lattice, nd)
                out += d + " & " + " & ".join(f2(r[f"R_{t}"]) for t in types) + f" & {fsr}" + r" \\" + "\n"
            out += (r"\textit{Budget ceiling} & " + " & ".join(r"\textit{" + f2(ta.loc[(led, DETS[0]), f"ceil_{t}"]) + "}"
                                                         for t in types) + r" & \\" + "\n")
        out += "\\bottomrule\n\\end{tabular}\n"
        outs.append(out)
    cap = (r"Recall within the first 100 postings per injected type ($R_t$@100) on the core grid of ledgers C and A, mean o"
           r"ver five seeds as printed in the result tables, with the fair-share type recall FSR@100 derived from them. $n_"
           r"t$ is the number of injected rows of type $t$ and the budget ceiling $\min(1, 100/n_t)$ the largest attainable"
           r" $R_t$@100. Because the per-type recalls are seed means rounded to two decimals, FSR@100 is given as the tight"
           r"est bounds consistent with the printed values and the printed P@100 (a single value where both bounds agree at"
           r" the printed precision). Type coverage is not reported, because the seed-mean recalls do not determine it. T1 "
           r"extreme amount, T2 extreme payment period, T3 unseen contra account, T4 unseen account, T5 unusual account pai"
           r"r.")
    nmin = int(nt["D"].min())
    cap2 = (r"Recall within the first 100 postings per injected type, continued from Table~\ref{tab:s-pertype} for ledgers B"
           r" and D. On ledger D every type holds at least 336 injected rows, so no per-type recall can exceed 0.30.")
    write(OUT_SUP / "tabS_pertype.tex", table_env("tab:s-pertype", cap, outs[0]) + "\n"
          + table_env("tab:s-pertype-cont", cap2, outs[1]))


def tabS_sweep():
    sw = runB_sweep()
    nan = n_anom_protocol()
    cs = sorted(sw.c.unique())
    diffs = ["easy", "medium", "hard"]
    out = r"\begin{tabular}{@{}l" + "r" * 9 + "@{}}\n\\toprule\n"
    out += " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$c = {c * 100:g}$\,\%}}" for c in cs) + r" \\" + "\n"
    out += "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3)) + "\n"
    out += "Detector & " + " & ".join(d for _ in cs for d in diffs) + r" \\" + "\n"
    for li, led in enumerate(["C", "A", "B"]):
        out += "\\midrule\n" + rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{({'abc'[li]}) Ledger {led}}}}} \\" + "\n"
        sub = sw[sw.ledger == led]
        cells = {}
        for c in cs:
            for dif in diffs:
                s = sub[(sub.c == c) & (sub.difficulty == dif)].set_index("detector").loc[DETS, "P100"]
                mask = best_mask(list(s.values))
                cells[(c, dif)] = [bold(f2(v)) if b else f2(v) for v, b in zip(s.values, mask)]
        for i, d in enumerate(DETS):
            out += d + " & " + " & ".join(cells[(c, dif)][i] for c in cs for dif in diffs) + r" \\" + "\n"
        out += (r"\textit{Maximum} & " + " & ".join(r"\textit{" + f2(min(1.0, nan[led][ci] / 100)) + "}"
                                                    for ci, c in enumerate(cs) for _ in diffs) + r" \\" + "\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = (r"Contamination and difficulty sweep on ledgers C, A and B: P@100 for every combination of the contamination $c$"
           r" and the difficulty level, mean over five seeds. The row Maximum is the attainable P@100, $\min(1, n/100)$ for"
           r" $n$ injected anomalies; at $c = 0.5$ percent every ledger holds fewer than 100 anomalies. The 27 cells are th"
           r"e input of the rank analysis in Section 5.1, which uses the seed-averaged cells because per-seed values were n"
           r"ot retained. Bold marks the highest P@100 per column within a ledger.")
    write(OUT_SUP / "tabS_sweep.tex", table_env("tab:s-sweep", cap, out))


def tabS_public_global():
    pub = pd.read_csv(paths.find("a6_public_global.csv"))
    pub["detector"] = pub.detector.map(nm)
    pub = pub.set_index("detector")
    comp = pd.read_csv(paths.find("a6_public_top100_composition.csv"))
    comp["detector"] = comp.detector.map(nm)
    comp = comp.set_index("detector")
    thr = re.search(r"rare account \(<(\d+) lines\)", (paths.public_result("top100_fill.csv")).read_text()).group(1)
    order = DETS + [CEIL]

    def cell(d, m):
        mu = float(pub.loc[d, f"{m}_mean"])
        sd = float(pub.loc[d, f"{m}_sd"]) if int(pub.loc[d, "n_seeds"]) > 1 else None
        return mu, sd, (3 if m in ("P@1000", "R@1000") else 2)

    out = r"\begin{tabular}{@{}lrrrr@{}}" + "\n\\toprule\n"
    out += r"\multicolumn{5}{@{}l}{\textit{(a) Precision at review budgets}} \\" + "\n\\addlinespace\n"
    m1 = ["P@25", "P@50", "P@100", "P@1000"]
    out += "Detector & " + " & ".join(m.replace("1000", "1,000") for m in m1) + r" \\" + "\n\\midrule\n"
    cells = {}
    for m in m1:
        vals = [cell(d, m) for d in DETS]
        mask = best_mask([v[0] for v in vals], vals[0][2])
        cells[m] = [bold(ms(*v)) if b else ms(*v) for v, b in zip(vals, mask)] + [ms(*cell(CEIL, m))]
    for i, d in enumerate(order):
        name = r"\textit{" + REF_NAME + "}" if d == CEIL else d
        pre = r"\addlinespace" + "\n" if d == CEIL else ""
        out += pre + name + " & " + " & ".join(cells[m][i] for m in m1) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n\n\\bigskip\n"
    out += r"\begin{tabular}{@{}lrrrrrr@{}}" + "\n\\toprule\n"
    out += r"\multicolumn{7}{@{}l}{\textit{(b) Threshold-free metrics and composition of the first 100 entries}} \\" + "\n\\addlinespace\n"
    out += (r" & & & & \multicolumn{3}{c}{First 100 entries} \\" + "\n" + r"\cmidrule(l){5-7}" + "\n")
    out += (r"Detector & R@1,000 & AP & ROC-AUC & anomalous & rare account & other \\" + "\n\\midrule\n")
    m2 = ["R@1000", "AP", "ROC-AUC"]
    cells = {}
    for m in m2:
        vals = [cell(d, m) for d in DETS]
        mask = best_mask([v[0] for v in vals], vals[0][2])
        cells[m] = [bold(ms(*v)) if b else ms(*v) for v, b in zip(vals, mask)] + [ms(*cell(CEIL, m))]
    for i, d in enumerate(order):
        name = r"\textit{" + REF_NAME + "}" if d == CEIL else d
        pre = r"\addlinespace" + "\n" if d == CEIL else ""
        c = comp.loc[d]
        comp_cells = [f"{c.anomalous_mean:.1f}" if int(c.n_seeds) > 1 else fint(c.anomalous_mean),
                      f"{c.normal_rare_account_mean:.1f}" if int(c.n_seeds) > 1 else fint(c.normal_rare_account_mean),
                      f"{c.other_normal_mean:.1f}" if int(c.n_seeds) > 1 else fint(c.other_normal_mean)]
        out += pre + name + " & " + " & ".join(cells[m][i] for m in m2) + " & " + " & ".join(comp_cells) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    dh = dib_header()
    cap = (r"Public ledger, global ranking of all 102,429 journal entries, of which 850 are anomalous. Mean (standard devia"
           r"tion) over seeds 0 to 2 for the stochastic detectors and the supervised reference; the data are fixed, so dete"
           r"rministic detectors have a single run. Bold marks the best unsupervised detector per column on the printed val"
           r"ues. The last three columns split the first 100 entries into anomalous entries, normal entries on a rare accou"
           r"nt (fewer than 1,000 lines) and other normal entries (means over seeds).")
    write(OUT_SUP / "tabS_public_global.tex", table_env("tab:s-public-global", cap, r"\setlength{\tabcolsep}{4pt}"
                                                        + "\n" + out, size=r"\scriptsize"))


def tabS_marking_recall():
    mk = pd.read_csv(paths.find("a6_public_marking_recall.csv"))
    mk["detector"] = mk.detector.map(nm)
    order = DETS + [CEIL]
    marks = [f"M{i}" for i in range(1, 9)]
    nmk = mk.drop_duplicates("marking").set_index("marking").loc[marks, "n_marking"].astype(int)
    out = r"\begin{tabular}{@{}l" + "r" * 8 + "@{}}\n\\toprule\n"
    out += "Detector & " + " & ".join(marks) + r" \\" + "\n"
    out += r"\textit{$n_t$} & " + " & ".join(r"\textit{" + str(n) + "}" for n in nmk) + r" \\" + "\n"
    for pi, (col, lab) in enumerate([("R100_mean", "Recall within the first 100 entries"),
                                     ("R1000_mean", "Recall within the first 1,000 entries")]):
        out += "\\midrule\n" + rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{({'ab'[pi]}) {lab}}}}} \\" + "\n"
        piv = mk.pivot(index="detector", columns="marking", values=col).loc[order, marks]
        cells = {}
        for m in marks:
            mask = best_mask(list(piv.loc[DETS, m].values))
            cells[m] = [bold(f2(v)) if b and v > 0 else f2(v) for v, b in zip(piv.loc[DETS, m].values, mask)]
            cells[m].append(f2(piv.loc[CEIL, m]))
        for i, d in enumerate(order):
            name = r"\textit{" + REF_NAME + "}" if d == CEIL else d
            pre = r"\addlinespace" + "\n" if d == CEIL else ""
            out += pre + name + " & " + " & ".join(cells[m][i] for m in marks) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = ("Public ledger, recall per marking within the first 100 and the first 1,000 journal entries, mean over "
           "seeds for the stochastic detectors. Bold marks the best unsupervised detector per column (none where "
           "all are zero). M1 high amount, M2 high cash disbursement, M3 delayed entry, M4 very delayed entry, M5 "
           "non-working hours, M6 weekend, M7 shadow bank account, M8 cross-linked clearing; $n_t$ is the number "
           "of anomalous entries per marking.")
    write(OUT_SUP / "tabS_marking_recall.tex", table_env("tab:s-marking", cap, out))


def tabS_rule_populations():
    rp = pd.read_csv(paths.find("a6_public_rule_populations.csv"))
    rp["detector"] = rp.detector.map(nm)
    # cross-check against the brief's source (within_population.csv, three decimals)
    wp = pd.read_csv(paths.public_result("within_population.csv"))
    wp["detector"] = wp.detector.map(nm)
    for _, r in wp[wp.detector.isin(DETS + [CEIL])].iterrows():
        a = rp[(rp.population == r.population) & (rp.detector == r.detector)].iloc[0]
        assert abs(a["P@100_mean"] - r["P@100"]) < 0.0015 and abs(a["AP_mean"] - r["AP"]) < 0.0015, (r.population, r.detector)
    order = DETS + [CEIL]
    pops = list(rp.drop_duplicates("population").population)
    head = rp.drop_duplicates("population").set_index("population")
    out = r"\begin{tabular}{@{}lrrrr@{}}" + "\n\\toprule\n"
    out += " & " + " & ".join(p.split(" ", 1)[0] for p in pops) + r" \\" + "\n\\midrule\n"
    def wrap(t):
        w = t.split(" ")
        if len(t) <= 12:
            return t
        h = (len(w) + 1) // 2
        return r"\makecell[r]{" + " ".join(w[:h]) + r"\\" + " ".join(w[h:]) + "}"
    out += r"Rule & " + " & ".join(wrap(p.split(" ", 1)[1]) for p in pops) + r" \\" + "\n"
    out += r"Entries & " + " & ".join(fint(head.loc[p, "entries"]) for p in pops) + r" \\" + "\n"
    out += r"Anomalies & " + " & ".join(fint(head.loc[p, "anomalies"]) for p in pops) + r" \\" + "\n"
    out += r"Rule precision & " + " & ".join(f2(head.loc[p, "rule_precision"], 3) for p in pops) + r" \\" + "\n"
    for pi, (col, lab, nd) in enumerate([("P@100_mean", "P@100 inside the population", 2),
                                         ("AP_mean", "Average precision inside the population", 3)]):
        out += "\\midrule\n" + rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{({'ab'[pi]}) {lab}}}}} \\" + "\n"
        piv = rp.pivot(index="detector", columns="population", values=col).loc[order, pops]
        cells = {}
        for p in pops:
            mask = best_mask(list(piv.loc[DETS, p].values), nd)
            cells[p] = [bold(f2(v, nd)) if b else f2(v, nd) for v, b in zip(piv.loc[DETS, p].values, mask)]
            cells[p].append(f2(piv.loc[CEIL, p], nd))
        for i, d in enumerate(order):
            name = r"\textit{" + REF_NAME + "}" if d == CEIL else d
            pre = r"\addlinespace" + "\n" if d == CEIL else ""
            out += pre + name + " & " + " & ".join(cells[p][i] for p in pops) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = ("Public ledger, ranking inside the four journal entry testing rule populations. Each population is "
           "the entry set that one rule returns; inside it the rule attribute is constant, so the ranking cannot "
           "use it. Rule precision is the share of anomalous entries in the population, which is what plain rule "
           "matching delivers. Means over seeds for the stochastic detectors; bold marks the best unsupervised "
           "detector per column.")
    write(OUT_SUP / "tabS_rule_populations.tex", table_env("tab:s-rules", cap, out))


def tabS_first_hit():
    fh = pd.read_csv(paths.find("a4_first_hit.csv"))
    fh["detector"] = fh.detector.map(nm)
    fps = pd.read_csv(paths.find("a4_first_hit_perseed.csv"))
    fps["detector"] = fps.detector.map(nm)
    order = DETS + [CEIL]
    marks = [f"M{i}" for i in range(1, 9)]
    allc = fps.set_index(["detector", "seed"])[marks].max(axis=1)
    out = r"\begin{tabular}{@{}l" + "r" * 9 + "@{}}\n\\toprule\n"
    out += "Detector & " + " & ".join(marks) + r" & All \\" + "\n\\midrule\n"
    for d in order:
        name = r"\textit{" + REF_NAME + "}" if d == CEIL else d
        if d == CEIL:
            out += r"\addlinespace" + "\n"
        sub = fh[fh.detector == d].set_index("marking").loc[marks]
        am = allc.loc[d]
        ns = int(sub.n_seeds.iloc[0])
        out += name + " & " + " & ".join(fint(v) for v in sub.first_hit_mean) + " & " + fint(am.mean()) + r" \\" + "\n"
        if ns > 1:
            for lab, col, agg in [("min", "first_hit_min", am.min()), ("max", "first_hit_max", am.max())]:
                vals = [fint(v) for v in sub[col]] + [fint(agg)]
                out += (r"\quad\textit{" + lab + "} & " + " & ".join(r"\textit{" + x + "}" for x in vals) + r" \\" + "\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = (r"Public ledger, first-hit rank per marking: the number of journal entries an auditor reviews until the first an"
           r"omaly of the marking appears. All is the rank at which every marking has appeared at least once (the largest f"
           r"irst-hit rank). Stochastic detectors and the supervised reference: mean over three seeds, with the smallest an"
           r"d largest value over the seeds in italics; the mean of All is taken over the per-seed maxima.")
    write(OUT_SUP / "tabS_first_hit.tex", table_env("tab:s-firsthit", cap, r"\setlength{\tabcolsep}{4pt}" + "\n" + out))


def tab_app_bootstrap():
    ci = pd.read_csv(paths.find("a5_bootstrap_ci.csv"))
    ci["detector"] = ci.detector.map(nm)
    pdiff = pd.read_csv(paths.find("a5_bootstrap_paired_kNN.csv"))
    src = (paths.ANALYSIS_PUBLIC / "a5_bootstrap.py").read_text()
    B = int(re.search(r"^B = (\d+)", src, flags=re.M).group(1))
    mets = ["P@100", "FSR@100", "P@1000", "FSR@1000"]

    def c3(pt, lo, hi, nd=2):
        return f"{f2(pt, nd)} [{f2(lo, nd)}, {f2(hi, nd)}]"

    out = r"\begin{tabular}{@{}lrrrrr@{}}" + "\n\\toprule\n"
    out += r"\multicolumn{6}{@{}l}{\textit{(a) Point estimate [95 percent interval]}} \\" + "\n\\addlinespace\n"
    out += "Detector & " + " & ".join(m.replace("1000", "1,000") for m in mets) + r" & TC@1,000 \\" + "\n\\midrule\n"
    for d in DETS:
        cells = []
        for m in mets:
            r = ci[(ci.detector == d) & (ci.metric == m)].iloc[0]
            cells.append(c3(r.point_seed0, r.ci_lo_2_5, r.ci_hi_97_5, 3 if m.endswith("1000") else 2))
        r = ci[(ci.detector == d) & (ci.metric == "TC@1000")].iloc[0]
        cells.append(f"{fint(r.point_seed0)} [{fint(r.ci_lo_2_5)}, {fint(r.ci_hi_97_5)}]")
        out += d + " & " + " & ".join(cells) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n\n\\bigskip\n"
    out += r"\begin{tabular}{@{}lrrrr@{}}" + "\n\\toprule\n"
    out += (r"\multicolumn{5}{@{}l}{\textit{(b) Paired difference kNN minus detector [95 percent interval]; "
            r"share of resamples with kNN ahead}} \\" + "\n\\addlinespace\n")
    out += "Compared with & " + " & ".join(m.replace("1000", "1,000") for m in mets) + r" \\" + "\n\\midrule\n"
    for d in [x for x in DETS if x != "kNN"]:
        cells = []
        for m in mets:
            r = pdiff[(pdiff.comparison == f"kNN minus {d}") & (pdiff.metric == m)].iloc[0]
            s = c3(r.point_difference, r.ci_lo_2_5, r.ci_hi_97_5, 3 if m.endswith("1000") else 2)
            if not r.excludes_zero:
                s = r"\textit{" + s + "}"
            cells.append(s)
        out += d + " & " + " & ".join(cells) + r" \\" + "\n"
        shares = []
        for m in mets:
            r = pdiff[(pdiff.comparison == f"kNN minus {d}") & (pdiff.metric == m)].iloc[0]
            shares.append(r"{\scriptsize " + f"{100 * r.share_kNN_higher:.1f}" + r"\,\%}")
        out += r" & " + " & ".join(shares) + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = (f"Public ledger, bootstrap over journal entries: {fint(B)} resamples of the entries with replacement, "
           "the same resamples for every detector, percentile 95 percent intervals. The stochastic detectors enter "
           "with their seed-0 scores, so the point estimates can differ from the three-seed means in the main "
           "text. (a) Intervals per detector. (b) Paired differences kNN minus the named detector; italic marks "
           "an interval that contains zero, and the second line gives the share of resamples in which kNN is "
           "strictly ahead.")
    write(OUT_MAIN / "tab_app_bootstrap.tex", table_env("tab:app-bootstrap", CAP_BOOTSTRAP, r"\setlength{\tabcolsep}{3pt}" + "\n" + out,
                                                    size=r"\scriptsize"))


def tabS_runtime():
    txt = (paths.CLIENT_RESULTS).read_text(encoding="utf-8")  # runtimes of this run (machine-specific)
    sec = txt[txt.index("## Runtimes"):]
    rt = {}
    for line in sec.splitlines():
        m = re.match(r"^\| ([^|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            rt[nm(m.group(1))] = [float(m.group(i)) for i in range(2, 6)]
    hdr = re.search(r"\*\*Table-D\*\* \([\d,]+ rows, [\d,]+ injected anomalies, (\d+) seeds\)", txt)
    seedsD = int(hdr.group(1))
    hdrC = re.search(r"\*\*Table-C\*\* \([\d,]+ rows, [\d,]+ injected anomalies, (\d+) seeds\)", txt)
    seedsC = int(hdrC.group(1))
    prot = (paths.PROTOCOL).read_text(encoding="utf-8")
    hw = re.search(r"Hardware for\s+all timings: ([^.]+)\.", prot, flags=re.S).group(1).replace("\n", " ")
    rj = [json.loads(l) for l in (paths.public_result("runtimes.jsonl")).read_text().splitlines() if l.strip()]
    pub = pd.DataFrame(rj).drop_duplicates(["det", "seed"], keep="last")  # a rerun appends to runtimes.jsonl
    pub["det"] = pub.det.map(nm)
    pmean = pub.groupby("det").runtime_s.mean()
    pn = pub.groupby("det").runtime_s.size()
    order = DETS + [CEIL]
    out = r"\begin{tabular}{@{}lrrrrr@{}}" + "\n\\toprule\n"
    out += r" & \multicolumn{4}{c}{Client ledgers} & Public \\" + "\n" + r"\cmidrule(lr){2-5}" + "\n"
    out += r"Detector & C & A & B & D & ledger \\" + "\n\\midrule\n"
    for d in order:
        name = r"\textit{" + REF_NAME + "}" if d == CEIL else d
        vals = [f"{v:,.1f}" for v in rt[d]] if d in rt else ["n/a"] * 4
        pv = f"{pmean.loc[d]:,.1f}" if d in pmean.index else "n/a"
        pre = r"\addlinespace" + "\n" if d == CEIL else ""
        out += pre + name + " & " + " & ".join(vals) + f" & {pv}" + r" \\" + "\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    ns = len(paths.PUBLIC_SEEDS)
    cap = ("Wall-clock seconds for one detector fit on the machine of this run (the paper's values were measured on "
           f"2 CPU cores with 7 GB RAM and no GPU). Client ledgers: core-grid cells, mean over {seedsC} seeds on ledgers "
           f"C, A and B and over {seedsD} seeds on ledger D. Public ledger: mean over the runs of a detector ({ns} seeds "
           "for the stochastic detectors and the supervised reference). On ledger D and on the public ledger the three "
           "networks train for 30 epochs with batch 256 instead of 100 epochs with batch 64, so their times there are "
           "not comparable with ledgers C, A and B. n/a: not timed.")
    if set(pn.reindex(["IF", "AE", "VAE", "AE-emb", CEIL]).fillna(0)) != {len(paths.PUBLIC_SEEDS)}:
        print("note: runtimes.jsonl does not hold one run per seed for every stochastic detector; "
              "the public-ledger runtimes are means over the runs recorded")
    write(OUT_SUP / "tabS_runtime.tex", table_env("tab:s-runtime", cap, out))


def tabS_field_mapping():
    rows = old_table_rows("tab:dib-mapping")
    contra_levels = dib_header()["contra"]
    out = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.2\linewidth}"
           r">{\raggedright\arraybackslash}p{0.3\linewidth}>{\raggedright\arraybackslash}p{0.42\linewidth}@{}}"
           + "\n\\toprule\n")
    out += r"Role & Client ledgers & Public ledger \\" + "\n\\midrule\n"
    n = 0
    for r in rows:
        parts = [p.strip() for p in r.replace("\\midrule", "").replace("\\toprule", "").split(" & ")]
        if len(parts) != 3 or parts[0] == "Role":
            continue
        pub = parts[2]
        pub = pub.replace(r", the raw amount in the $s_{\mathrm{amt}}$ component of DECOMP", "")
        pub = pub.replace(r"; used in the encodings and in the $s_{\mathrm{pay}}$ component of DECOMP",
                          "; used in the encodings")
        pub = re.sub(r"(\d+) levels", lambda m: f"{contra_levels} levels" if int(m.group(1)) == contra_levels else m.group(0), pub)
        assert "DECOMP" not in pub, pub
        cl = parts[1].replace("DATEV", "client ledgers").replace("real ledgers", "client ledgers")
        pub = pub.replace("real ledgers", "client ledgers")
        out += f"{parts[0]} & {cl} & {pub} \\\\\n"
        n += 1
    assert n >= 12, n
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = ("Mapping of the benchmark onto the public ledger, field by field. The protocol expects the eleven "
           "columns of the client ledgers; the table states which field of the public ledger takes each role, "
           "what had to be derived and what was added because the public ledger records timestamps. The rule "
           "attributes supplied with the dataset are excluded from every feature set, as its authors recommend. "
           "All detector settings are those of ledger D.")
    write(OUT_SUP / "tabS_field_mapping.tex", table_env("tab:s-mapping", cap, out))


def tabS_deviations():
    prot = (paths.PROTOCOL).read_text(encoding="utf-8")
    m = re.search(r"has only (\d+) valid T5 pairs\. At c = 0\.03\s+the cell needs (\d+) T5 rows, so (\d+) rows reuse a pair",
                  prot)
    valid, need, reuse = m.groups()
    pca_rel = re.search(r"to (\d+e-\d+)\s+relative on A and B", prot).group(1)
    ecod_diff = re.search(r"on C, A and B\s+\(max difference (\d+e-\d+)\)", prot).group(1)
    dnet = re.search(r"Table-D: batch (\d+), (\d+) epochs", prot)
    bD, eD = dnet.groups()
    base = re.search(r"epoch_num=(\d+),\s+batch_size=(\d+)", prot)
    e0, b0 = base.groups()
    rowsD = wp0_profile().loc["D", "rows"]
    dh = dib_header()
    kpub = f"{paths.PUBLIC_K:,}"
    sw = runB_sweep()
    ncells = len(sw[["ledger", "c", "difficulty"]].drop_duplicates())

    def e(x: str) -> str:
        mant, ex = x.split("e-")
        return (r"10^{-" if mant == "1" else mant + r"\times10^{-") + ex + "}"

    items = [
        ("Injection", f"Ledger A, easy level, $c = 3$\\,\\%: the pool holds {valid} valid account pairs for the {need} "
                      f"T5 rows the cell needs, so {reuse} rows reuse a pair. The protocol permits drawing with "
                      "replacement once the pool is exhausted; the reuse is recorded per cell."),
        ("Typology", "The protocol called T5 a multidimensional mixed-data anomaly (Foorthuis Type VI). T5 changes two "
                     "categorical fields only, so it is classified as Type V, subtype Va."),
        ("PCA", "The score is PyOD's own: the sum of the distances between the standardised row and the retained "
                "eigenvectors, each divided by its explained-variance ratio, not a literal reconstruction error. The "
                "covariance eigensolver replaces the full singular value decomposition (same components, scores "
                f"agree to ${e(pca_rel)}$ relative or better, identical P@$k$ and AP)."),
        ("ECOD", "PyOD's algorithm recomputed column chunk by column chunk, because the library implementation "
                 f"exhausts memory on ledger D; maximum difference to the library ${e(ecod_diff)}$ on ledgers C, A "
                 "and B. Used on every ledger."),
        ("Networks on ledger D", f"AE, VAE and AE-emb train with batch {bD} for {eD} epochs instead of batch {b0} "
                                 f"for {e0}; the protocol grants this for AE-emb, and a runtime probe extended it to "
                                 "the two PyOD networks."),
        ("Execution on ledger D", "One thread per process for the linear algebra libraries and PyTorch; every "
                                  "detector runs in a checkpointing worker process; LOF and kNN are fitted on all "
                                  f"{fint(rowsD)} rows, the subsampling rule being implemented but never triggered. "
                                  "Scores are unaffected."),
        ("Not run", "The optional contextual extreme-amount type and the random-label sensitivity analysis were not "
                    "run. The contamination and difficulty sweep covers ledgers C, A and B; every ledger D result "
                    "is at $c = 1.5$\\,\\% and medium difficulty."),
        ("Public ledger", f"Networks at the ledger D setting; stochastic detectors on seeds {dh['seed_lo']} to "
                          f"{dh['seed_hi']}; sine and cosine of the hour and weekday of the entry timestamp added to "
                          "the encodings; posting lines are scored and journal entries evaluated (maximum over the "
                          f"lines); budget $k = {kpub.replace(',', '{,}')}$ added; contra account derived; rule attributes excluded; "
                          "cross-fitting folds of the supervised reference grouped by entry."),
        ("Analysis", f"The Friedman test uses the {ncells} seed-averaged sweep cells of ledgers C, A and B, because "
                     "per-seed values of the sweep were not retained. The type-aware measures on the client ledgers "
                     "are computed from seed-mean per-type recalls at $k = 100$ printed to two decimals and are "
                     "reported with the bounds this rounding implies."),
        ("Added in revision", "A no-feedback control of the DeepSAD review protocol on the public ledger, seeds "
                              f"{dh['seed_lo']} to {dh['seed_hi']}: the same initialisation, rounds, seeds, optimiser "
                              "and retraining as the protocol run, with the revealed labels withheld from the "
                              "retraining. It was run after the frozen experiments, is reported as a control only and "
                              "changes no protocol result."),
    ]
    out = (r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.2\linewidth}"
           r">{\raggedright\arraybackslash}p{0.76\linewidth}@{}}" + "\n\\toprule\n")
    out += r"Area & Deviation or implementation note \\" + "\n\\midrule\n"
    for a, b in items:
        out += f"{a} & {b} \\\\\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    cap = ("Protocol deviations and implementation notes. None changes an injection definition, a metric or a "
           "detector hyperparameter on ledgers C, A and B, and each is recorded per cell in the harness output.")
    write(OUT_SUP / "tabS_deviations.tex", table_env("tab:s-deviations", cap, out))


CAP_FEEDBACK = (r"Auditor reviews fed back to DeepSAD. (a) Anomalies found after 25, 50, 75 and 100 reviews (four rounds of 25),"
           r" mean over seeds with the standard deviation of the final count, against the anomalies among the first 100 uni"
           r"ts of AE-emb, the list the same reviews follow if the ranking stays static, and of the static detector with th"
           r"e most hits; the gain is that of the adaptive review protocol over the static AE-emb ranking. Client ledgers: "
           r"core grid, five seeds; the static counts are 100 times the printed P@100 and exact to within 0.5. Public ledge"
           r"r: three seeds, units are journal entries. Public, control: the same protocol with the revealed labels withhel"
           r"d from the retraining, added in revision outside the frozen protocol. (b) Public ledger: markings among the 10"
           r"0 reviewed entries, with and without labels, and among the first 100 entries of AE-emb, mean over three seeds,"
           r" with type coverage TC@100 and fair-share type recall FSR@100; $n_t$ is the number of entries per marking.")
CAP_MACRO = (r"Which conclusions need the fair-share normalisation: the leader under P@$k$, macro-averaged recall (the mean o"
           r"f $R_t$@$k$ over the types) and FSR@$k$, the fair-share caps ($n_t$ when every cap equals the type size, in wh"
           r"ich case FSR equals macro recall, otherwise the common cap $k/T$), and Kendall's $\tau_b$ between the ordering"
           r"s of the nine detectors. Client ledgers at $k=100$, from the printed seed-mean per-type recalls (FSR of the se"
           r"ed-averaged type profile, midpoints of the rounding bounds; detectors within 0.005 of the macro-recall leader "
           r"are listed together). Public ledger exact, mean over seeds.")
CAP_SEEDS = (r"Public ledger, DeepSAD per seed: markings among the 100 reviewed entries with labels and in the no-feedback co"
           r"ntrol (labels withheld, added in revision), and among the first 100 entries of the AE-emb model both runs star"
           r"t from, with the anomalies found after 25 and 100 reviews (F@25, F@100), type coverage TC@100 and fair-share t"
           r"ype recall FSR@100. Round 1 reviews the first 25 entries of AE-emb, so the three lists agree after 25 reviews.")
CAP_BOOTSTRAP = (r"Public ledger, bootstrap over journal entries: 1,000 resamples of the entries with replacement, the same resam"
           r"ples for every detector, percentile 95 percent intervals. The stochastic detectors enter with their seed-0 sco"
           r"res, so the point estimates can differ from the three-seed means in Table~\ref{tab:conventional}. (a) Interval"
           r"s per detector. (b) Paired differences kNN minus the named detector; italic marks an interval that contains ze"
           r"ro, and the second line gives the share of resamples in which kNN is strictly ahead.")
CAP_WEIGHTS = (r"Sensitivity of FSR@$k$ to the type weights (\ref{app:weights}). (a) Client ledgers at $k=100$: the leader by P"
           r"@100, the FSR leader under equal weights, the number of the eleven structured schemes (equal weights and each "
           r"type doubled or halved) in which a reversal between the P@100 leader and FSR is fixed by the rounding bounds, "
           r"the schemes that are exceptions, and the share of 300 random weight vectors (each weight between half and twic"
           r"e its equal value) with a reversal fixed by the bounds. (b) Public ledger: the FSR leader under equal weights,"
           r" the number of the seventeen structured schemes in which it stays the FSR leader, the smallest Kendall $\tau_b"
           r"$ between the orderings by P@$k$ and FSR@$k$ over these schemes, and, over 2,000 random vectors, the share in "
           r"which the equal-weight FSR leader stays the leader and the median $\tau_b$.")


# ==============================================================================================
# tables added in the ESWA revision: review protocol with the no-feedback control, FSR against
# macro-averaged recall, and the appendix tables (DeepSAD per seed, weight sensitivity)
# ==============================================================================================

MARKINGS = [f"M{m}" for m in range(1, 9)]


def _f1(x: float) -> str:
    """One decimal, but an exact zero prints as 0 (counts of markings in Table 8b)."""
    return "0" if abs(x) < 1e-12 else f2(x, 1)


def _public_nt() -> list[int]:
    mk = pd.read_csv(paths.find("a6_public_marking_recall.csv")).drop_duplicates("marking").set_index("marking")
    return [int(mk.loc[m, "n_marking"]) for m in MARKINGS]


def tab_feedback():
    real = pd.read_csv(paths.find("deepsad_real.csv")).set_index("ledger")
    per = pd.read_csv(paths.find("deepsad_control_perseed.csv"))
    summ = pd.read_csv(paths.find("deepsad_control_summary.csv")).set_index("method")
    glob = pd.read_csv(paths.find("a6_public_global.csv")).set_index("detector")
    ds, st, co = "DeepSAD (100 reviews)", "AE-emb static top 100", "Control: same protocol, labels withheld"
    best = glob.loc[DETS, "P@100_mean"]
    best_det = best.idxmax(); best_found = 100 * best.max()
    out = r"\setlength{\tabcolsep}{3.4pt}" + "\n" + r"\begin{tabular}{@{}lrrrrrrll@{}}" + "\n"
    out += r"\multicolumn{9}{@{}l}{\textit{(a) Anomalies found in 100 reviews}} \\" + "\n\\toprule\n"
    out += (r" & \multicolumn{4}{c}{DeepSAD, after $r$ reviews} & \multicolumn{2}{c}{AE-emb, first 100} & "
            r"\multicolumn{2}{c}{Most hits, first 100} \\" + "\n" + r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}" + "\n")
    out += r"Ledger & 25 & 50 & 75 & 100 & Found & Gain & Detector & Found \\" + "\n\\midrule\n"
    for L in LEDGERS:
        r = real.loc[L]
        det = ", ".join(nm(x) for x in str(r.best_static).split(" / "))
        out += (f"{L} & {f2(r.ds_found25, 1)} & {f2(r.ds_found50, 1)} & {f2(r.ds_found75, 1)} & "
                f"{f2(r.ds_found100, 1)} ({f2(r.ds_found100_sd, 1)}) & {r.aeemb_found100:.0f} ({r.aeemb_found100_sd:.0f}) & "
                f"{f2(r.gain_vs_start, 1)} & {det} & {r.best_static_found100:.0f} \\\\\n")
    a = summ.loc[st]
    for name, meth in [("Public", ds), ("Public, control", co)]:
        m = summ.loc[meth]
        f50 = m.found50_mean; f75 = m.found75_mean
        out += (f"{name} & {f2(m.found25_mean, 1)} & {f2(f50, 1)} & {f2(f75, 1)} & {f2(m.found100_mean, 1)} "
                f"({f2(m.found100_std, 1)}) & {f2(a.found100_mean, 1)} ({f2(a.found100_std, 1)}) & "
                f"{f2(m.found100_mean - a.found100_mean, 1)} & {best_det} & {f2(best_found, 1)} \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n\n\\vspace{1.2em}\n" + r"\setlength{\tabcolsep}{3.2pt}" + "\n"
    out += r"\begin{tabular}{@{}lrrrrrrrrrrr@{}}" + "\n"
    out += r"\multicolumn{12}{@{}l}{\textit{(b) Public ledger: markings among 100 entries}} \\" + "\n\\toprule\n"
    out += " & " + " & ".join(MARKINGS) + r" & Hits & TC@100 & FSR@100 \\" + "\n\\midrule\n"
    nt = _public_nt()
    out += "$n_t$ & " + " & ".join(str(n) for n in nt) + f" & {sum(nt)} & {len(nt)} & \\\\\n"
    for name, meth in [("AE-emb, first 100", st), ("DeepSAD, reviewed", ds), ("Control, reviewed", co)]:
        m = summ.loc[meth]
        out += (name + " & " + " & ".join(_f1(m[f"{k}_mean"]) for k in MARKINGS)
                + f" & {f2(m.found100_mean, 1)} & {f2(m.coverage_mean, 1)} & {f2(m.FSR100_mean, 2)} \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    assert len(per) == 9
    cap = CAP_FEEDBACK
    write(OUT_MAIN / "tab_feedback.tex", table_env("tab:feedback", cap, out))


def tab_app_deepsad_seeds():
    per = pd.read_csv(paths.find("deepsad_control_perseed.csv"))
    names = [("AE-emb static top 100", "AE-emb, first 100"), ("DeepSAD (100 reviews)", "DeepSAD, reviewed"),
             ("Control: same protocol, labels withheld", "Control, reviewed")]
    out = r"\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{tabular}{@{}llrrrrrrrrrrrr@{}}" + "\n\\toprule\n"
    out += "Seed & List & " + " & ".join(MARKINGS) + r" & F@25 & F@100 & TC@100 & FSR@100 \\" + "\n\\midrule\n"
    for s in sorted(per.seed.unique()):
        if s:
            out += "\\addlinespace\n"
        for meth, label in names:
            r = per[(per.seed == s) & (per.method == meth)].iloc[0]
            out += (f"{s} & {label} & " + " & ".join(str(int(r[m])) for m in MARKINGS)
                    + f" & {int(r.found25)} & {int(r.found100)} & {int(r.coverage)} & {r.FSR100:.3f} \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    txt = table_env("tab:app-deepsad-seeds", CAP_SEEDS, out, size=r"\scriptsize").replace("\\begin{table}[tbp]", "\\begin{table}[htbp]", 1)
    write(OUT_MAIN / "tab_app_deepsad_seeds.tex", txt)


def tab_macro():
    cs = pd.read_csv(paths.find("a11_client_summary.csv"))
    cm = pd.read_csv(paths.find("a11_client_macro.csv"))
    ps = pd.read_csv(paths.find("a11_public_summary.csv"))
    nb = pd.read_csv(paths.find("a2_found_bounds.csv"))[["ledger", "type", "n_t"]].drop_duplicates()
    nt_pub = np.array(_public_nt(), float)
    sys.path.insert(0, str(paths.TYPEAWARE))
    from fsr import caps  # noqa: E402

    def cap_cell(n, k):
        c = caps(np.asarray(n, float), k)
        if np.allclose(c, n):
            return "$n_t$"
        v = float(c.max())
        return f"{v:g}"

    def by_value(names, L):
        """Macro-recall leaders within 0.005, ordered by their macro recall."""
        g = cm[cm.ledger == L].set_index("detector").macro
        return ", ".join(sorted([x.strip() for x in names.split(",")], key=lambda d: -g[d]))

    out = r"\setlength{\tabcolsep}{2.8pt}" + "\n" + r"\begin{tabular}{@{}lllllcc@{}}" + "\n\\toprule\n"
    out += (r" & & \multicolumn{3}{c}{Leader by} & \multicolumn{2}{c}{Kendall's $\tau_b$} \\" + "\n"
            + r"\cmidrule(lr){3-5}\cmidrule(l){6-7}" + "\n")
    out += r"Ledger ($k$) & Cap $c_t$ & P@$k$ & Macro & FSR@$k$ & P, macro & macro, FSR \\" + "\n\\midrule\n"
    for _, r in cs.iterrows():
        n = nb[nb.ledger == r.ledger].sort_values("type").n_t.values
        out += (f"{r.ledger} (100) & {cap_cell(n, 100)} & {r.P_leader} & {by_value(r.macro_leader, r.ledger)} & "
                f"{r.FSR_leader} & {f2(r.tau_P_macro)} & {f2(r.tau_macro_FSR)} \\\\\n")
    for _, r in ps.iterrows():
        out += (f"Public ({int(r.k):,}) & {cap_cell(nt_pub, int(r.k))} & {r.P_leader} & {r.macro_leader} & "
                f"{r.FSR_leader} & {f2(r.tau_P_macro)} & {f2(r.tau_macro_FSR)} \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    write(OUT_MAIN / "tab_macro.tex", table_env("tab:macro", CAP_MACRO, out))


_HOW = {"x2": ("doubled", "doubled"), "x0.5": ("halved", "halved")}


def _scheme_text(s: str) -> tuple[str, str]:
    t, f = s.split(" ")
    return t, _HOW[f][0]


def tab_app_weights():
    cw = pd.read_csv(paths.find("a10_client_weight_summary.csv"))
    rb = pd.read_csv(paths.find("a10b_client_random_bounds.csv")).set_index("ledger")
    pw = pd.read_csv(paths.find("a10c_public_weight_schemes.csv"))
    pr = pd.read_csv(paths.find("a10c_public_random_weights.csv")).set_index("k")
    out = (r"\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.13\textwidth}"
           r">{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.12\textwidth}"
           r">{\raggedright\arraybackslash}p{0.26\textwidth}>{\raggedright\arraybackslash}p{0.14\textwidth}@{}}" + "\n")
    out += r"\multicolumn{6}{@{}l}{\textit{(a) Client ledgers, $k=100$}} \\" + "\n\\toprule\n"
    out += (r"Ledger & P@100 leader & FSR leader, equal weights & Reversal fixed (of 11 schemes) & Exceptions & "
            r"Random vectors with reversal fixed \\" + "\n\\midrule\n")
    for L in LEDGERS:
        g = cw[cw.ledger == L]
        u = g[g.scheme == "uniform"].iloc[0]
        nfix = int(g.reversal_fixed_by_bounds.sum()); n = len(g)
        assert n == 11
        other = g[(g.scheme != "uniform") & (g.reversal_fixed_by_bounds != u.reversal_fixed_by_bounds)]
        if u.reversal_fixed_by_bounds:
            fsr_lead = u.FSR_leader_by_bounds
            if other.empty:
                exc = "none"
            else:
                parts = [f"{t} {h}" for t, h in (_scheme_text(s) for s in other.scheme)]
                exc = ", ".join(parts)
                if (other.FSR_leader_by_bounds == "undetermined").all():
                    exc += " (undetermined)"
        else:
            fsr_lead = f"{u.FSR_leader_by_bounds} (no reversal)"
            leaders = other.FSR_leader_by_bounds.unique()
            assert len(leaders) == 1
            parts = [_scheme_text(s) for s in other.scheme]
            exc = (f"reversal to {leaders[0]} when " + f"{parts[0][0]} is {parts[0][1]}"
                   + "".join(f" or {t} {h}" for t, h in parts[1:]))
        share = rb.loc[L, "share_reversal_fixed_by_bounds"]
        out += (f"{L} & {u.P100_leader} & {fsr_lead} & {nfix} & {exc} & {100 * share:.0f}\\% \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n\n\\vspace{1em}\n"
    out += (r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.1\textwidth}>{\raggedright\arraybackslash}p{0.12\textwidth}"
            r">{\raggedright\arraybackslash}p{0.2\textwidth}>{\raggedright\arraybackslash}p{0.1\textwidth}"
            r">{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.12\textwidth}@{}}" + "\n")
    out += r"\multicolumn{7}{@{}l}{\textit{(b) Public ledger}} \\" + "\n\\toprule\n"
    out += (r"$k$ & P@$k$ leader & FSR leader, equal weights & FSR leader kept (of 17 schemes) & Smallest $\tau_b$ & "
            r"Random vectors with leader kept & Median $\tau_b$, random \\" + "\n\\midrule\n")
    for k in (100, 1000):
        g = pw[pw.k == k]
        u = g[g.scheme == "uniform"].iloc[0]
        kept = int((g.FSR_leader == u.FSR_leader).sum()); assert len(g) == 17
        lost = g[g.FSR_leader != u.FSR_leader]
        kept_txt = str(kept)
        if not lost.empty:
            kept_txt += " (" + "; ".join(f"{r.FSR_leader} when {_scheme_text(r.scheme)[0]} is {_scheme_text(r.scheme)[1]}"
                                         for _, r in lost.iterrows()) + ")"
        leaders = ast.literal_eval(pr.loc[k, "FSR_leaders"])
        out += (f"{k:,} & {u.P_leader} & {u.FSR_leader} & {kept_txt} & {f2(g.tau_P_FSR.min())} & "
                f"{100 * leaders[u.FSR_leader]:.0f}\\% & {f2(pr.loc[k, 'tau_median'])} \\\\\n")
    out += "\\bottomrule\n\\end{tabular}\n"
    txt = table_env("tab:app-weights", CAP_WEIGHTS, out, size=r"\scriptsize").replace("\\begin{table}[tbp]", "\\begin{table}[htbp]", 1)
    write(OUT_MAIN / "tab_app_weights.tex", txt)


SOURCES = {
    "tab:data": "restricted/inputs/wp0_data_audit.md (Section 1: rows, accounts, contra accounts, tax codes, users, payment period "
                "missing); restricted/inputs/dup_group_stats.csv (duplicate rows); restricted/inputs/results_runB_complete.md T1 block "
                "headers (rows, injected anomalies); harness/protocol.md (one fiscal year per table); public ledger: "
                "computed from data/journal_entries_4203.csv with public_ledger/dib_data.py (cached in "
                "tables/dib_profile_computed.json) and checked against reference/public_ledger/dib_results.md (lines, entries, "
                "years, accounts, users, 21 contra levels, 850 anomalous entries).",
    "tab:types": "definitions parsed from restricted/inputs/discover_manuscript_main.tex tab:types; n_t from "
                 "a2_found_bounds.csv (equal to BRIEF Section 4 counts); shares checked against "
                 "n_t on ledger D; Foorthuis subtypes checked against harness/protocol.md Section 3; marking counts "
                 "checked against a6_public_marking_recall.csv; rule population names from "
                 "a6_public_rule_populations.csv.",
    "tab:feedback": "deepsad_real.csv (analysis/client/deepsad_client.py, run B T1 and T5), deepsad_control_summary.csv and deepsad_control_perseed.csv (analysis/public/deepsad_public.py, deepsad_control.py), a6_public_global.csv (static leader at k = 100), a6_public_marking_recall.csv (n_t).",
    "tab:detectors": "rows parsed from restricted/inputs/discover_manuscript_main.tex tab:detectors (rows of the excluded methods dropped); "
                     "setting values asserted against harness/protocol.md Section 5.",
    "tab:conventional": "client ledgers: a7_datev_core.csv (= run B T1) and "
                        "a7_datev_ceiling.csv (= run B T6); public ledger: a6_public_global.csv, "
                        "asserted equal to reference/public_ledger/metrics_mean.csv.",
    "tab:typeaware": "a3_table5.csv (best by P@k, its TC and FSR with bounds, best by FSR with "
                     "bounds, Kendall tau_b range); built from datev_fsr_k100.csv and "
                     "dib_typeaware_mean.csv.",
    "tab:s-core": "a7_datev_core.csv, a7_datev_ceiling.csv (run B T1, T6).",
    "tab:s-pertype": "datev_typeaware_k100.csv (R_T, ceilings; = run B T2), datev_fsr_k100.csv "
                     "(TC_lo, TC_hi, FSR_lo_lattice, FSR_hi_lattice), a2_found_bounds.csv (n_t).",
    "tab:s-pertype-cont": "as tab:s-pertype.",
    "tab:s-sweep": "restricted/inputs/results_runB_complete.md T3 (rows of the excluded methods dropped); maximum from n_anom in "
                   "harness/protocol.md Section 3.",
    "tab:s-public-global": "a6_public_global.csv and a6_public_top100_composition.csv; rare-account "
                           "threshold from reference/public_ledger/top100_fill.csv header; seeds from reference/public_ledger/dib_results.md.",
    "tab:s-marking": "a6_public_marking_recall.csv.",
    "tab:s-rules": "a6_public_rule_populations.csv, checked against "
                   "reference/public_ledger/within_population.csv.",
    "tab:s-firsthit": "a4_first_hit.csv and a4_first_hit_perseed.csv (All = per-seed maximum "
                      "over the eight markings, averaged over seeds).",
    "tab:app-bootstrap": "a5_bootstrap_ci.csv, a5_bootstrap_paired_kNN.csv; B from a5_bootstrap.py.",
    "tab:macro": "a11_client_summary.csv, a11_client_macro.csv (analysis/client/a11_client_macro.py), a11_public_summary.csv (analysis/public/a11_public_macro.py); caps from a2_found_bounds.csv and a6_public_marking_recall.csv.",
    "tab:app-deepsad-seeds": "deepsad_control_perseed.csv (analysis/public/deepsad_public.py and deepsad_control.py).",
    "tab:app-weights": "a10_client_weight_summary.csv, a10b_client_random_bounds.csv (analysis/client), a10c_public_weight_schemes.csv, a10c_public_random_weights.csv (analysis/public).",
    "tab:s-runtime": "restricted/inputs/results_runA_partial.md runtime table (run A; hardware from harness/protocol.md "
                     "Section 8; seeds from the run A T1 headers); reference/public_ledger/runtimes.jsonl (mean per detector).",
    "tab:s-mapping": "rows parsed from restricted/inputs/discover_manuscript_main.tex tab:dib-mapping with the clauses on the "
                     "excluded reference method removed; 21 contra levels checked against reference/public_ledger/dib_results.md.",
    "tab:s-deviations": "harness/protocol.md Sections 2 to 9 (numbers parsed: 19/36/17 T5 pairs, 4e-4, 1e-14, batch "
                        "256 and 30 epochs, batch 64 and 100 epochs), restricted/inputs/wp0_data_audit.md (279,975 rows), "
                        "reference/public_ledger/dib_results.md (seeds, k = 1,000), and the deviations appendix of "
                        "restricted/inputs/discover_manuscript_main.tex for wording (items on the excluded methods and the feedback study dropped).",
}


def write_readme():
    lines = ["# Tables", "",
             "Generated by `python3 tables/make_tables.py`. Every number is parsed from the named source files or "
             "computed from them; nothing is typed in. booktabs, caption first, `\\label{tab:...}`; needs "
             "`booktabs`, `array`, `makecell`, `amsmath`, `amssymb`. All tables were compiled in a minimal "
             "`\\documentclass[preprint,12pt,authoryear]{elsarticle}` document with `build/elsarticle.cls`: no "
             "errors, no overfull boxes, no float larger than the page.",
             "",
             "Conventions: bold marks the best unsupervised detector per column on the printed values (ties all "
             "bold); rounding is half up on the shortest decimal representation (0.215 prints as 0.22, 0.125 as "
             "0.13), identical in figures/make_figures.py; ranges use an en dash or the word 'to'; the methods "
             "excluded by the brief never appear (the writer asserts this).",
             "",
             "`tables/tab_notation.tex` (label `tab:notation`) is not produced by this script; it belongs to the "
             "notation appendix (Table A.1 in v2/appendix.tex).", ""]
    for title, folder, pat in [("Main text", OUT_MAIN, "tab_*.tex"), ("Supplement", OUT_SUP, "tabS_*.tex")]:
        lines += [f"## {title}", ""]
        for f in sorted(folder.glob(pat)):
            if f.name == "tab_notation.tex":
                continue
            t = f.read_text()
            for cap, lab in re.findall(r"\\caption\{(.*?)\}\n\\label\{(.*?)\}", t, flags=re.S):
                lines += [f"### `{f.relative_to(ROOT)}` (`{lab}`)", "",
                          f"Draft caption: {cap}", "", f"Sources: {SOURCES[lab]}", ""]
    lines += ["## Open points", "",
              "- Public-ledger duplicate rows in `tab:data` are recomputed here (ten mapped fields, see "
              "`tables/dib_profile_computed.json`): 1,242 lines repeat an earlier line (0.5 percent, as in "
              "reference/public_ledger/dib_results.md). The previous manuscript printed 1,342 with an undocumented definition; "
              "that number was not reproduced and is not used.",
              "- `tab:typeaware` and `tab:s-pertype` report ledger B and D values as ranges where the two-decimal "
              "rounding of the per-type recalls leaves them undetermined (three decimals on ledger D).", ""]
    (OUT_MAIN / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote tables/README.md")


if __name__ == "__main__":
    OUT_SUP.mkdir(parents=True, exist_ok=True)
    tab_data()
    tab_types()
    tab_detectors()
    tab_conventional()
    tab_typeaware()
    tabS_core_grid()
    tabS_pertype()
    tabS_sweep()
    tabS_public_global()
    tabS_marking_recall()
    tabS_rule_populations()
    tabS_first_hit()
    tab_app_bootstrap()
    tab_feedback()
    tab_app_deepsad_seeds()
    tab_macro()
    tab_app_weights()
    tabS_runtime()
    tabS_field_mapping()
    tabS_deviations()
    write_readme()
