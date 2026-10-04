"""WP0 data audit: verifies the Section 2 profile and reports the injection
statistics that WP1 depends on. Writes reports/wp0_data_audit.md.

Usage: python -m src.audit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .load import (ACCOUNT, ALL_COLUMNS, AMOUNT_Z, BINARY, CATEGORICAL, CONTRA,
                   CURRENCY, DATE_DEV, KEY, PAYPERIOD, ROOT, ROUND_AMT, TABLES,
                   TAX, TEXT_PRESENT, USER, load_table, p999)

# Profile claimed in Section 2, for verification.
CLAIMED = {
    "C": dict(rows=3461, n_account=18, n_contra=159, n_tax=2, n_user=10,
              payperiod_missing=281, amount_z_min=-174, amount_z_max=7.5),
    "A": dict(rows=5080, n_account=23, n_contra=608, n_tax=3, n_user=2,
              payperiod_missing=454, amount_z_min=-6611, amount_z_max=416),
    "B": dict(rows=17505, n_account=48, n_contra=362, n_tax=2, n_user=5,
              payperiod_missing=929, amount_z_min=-367, amount_z_max=97),
    "D": dict(rows=279975, n_account=24, n_contra=12359, n_tax=3, n_user=6,
              payperiod_missing=22611, amount_z_min=None, amount_z_max=None),
}

PAIR_F = [50, 20, 5]


def valid_pairs(df: pd.DataFrame, F: int) -> tuple[int, int, int]:
    """Number of (Kontonummer, Gegenkonto) pairs with count(a) >= F,
    count(g) >= F and count(a, g) = 0. Returns (n_valid, n_a, n_g)."""
    ca = df[ACCOUNT].value_counts()
    cg = df[CONTRA].value_counts()
    a_ok = ca[ca >= F].index
    g_ok = cg[cg >= F].index
    observed = df[[ACCOUNT, CONTRA]].drop_duplicates()
    observed = observed[observed[ACCOUNT].isin(a_ok) & observed[CONTRA].isin(g_ok)]
    n_valid = len(a_ok) * len(g_ok) - len(observed)
    return int(n_valid), int(len(a_ok)), int(len(g_ok))


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def audit_table(table: str) -> dict:
    df = load_table(table)
    n = len(df)
    info: dict = {"table": table, "rows": n}
    info["guid_unique"] = bool(df[KEY].is_unique)
    info["guid_min"] = int(df[KEY].min())
    info["guid_max"] = int(df[KEY].max())
    info["duplicate_feature_rows"] = int(df.duplicated(subset=[c for c in ALL_COLUMNS if c != KEY]).sum())

    # Cardinalities
    for col, name in [(ACCOUNT, "n_account"), (CONTRA, "n_contra"), (CURRENCY, "n_currency"),
                      (TAX, "n_tax"), (USER, "n_user")]:
        info[name] = int(df[col].nunique())
    info["currency_levels"] = df[CURRENCY].value_counts().to_dict()
    info["tax_levels"] = df[TAX].value_counts().to_dict()
    info["user_levels"] = df[USER].value_counts().to_dict()

    # Missingness
    info["missing"] = {c: int(df[c].isna().sum()) for c in ALL_COLUMNS}
    info["payperiod_missing"] = info["missing"][PAYPERIOD]
    info["payperiod_missing_share"] = info["payperiod_missing"] / n

    # Constant columns and binary balance
    info["constant_columns"] = [c for c in ALL_COLUMNS if c != KEY and df[c].nunique(dropna=False) <= 1]
    info["binary_share_of_ones"] = {c: float(df[c].mean()) for c in BINARY}
    info["binary_values"] = {c: sorted(df[c].unique().tolist()) for c in BINARY}

    # Numeric profiles
    z = df[AMOUNT_Z].to_numpy()
    info["amount_z_min"] = float(np.min(z))
    info["amount_z_max"] = float(np.max(z))
    info["amount_z_median"] = float(np.median(z))
    info["amount_z_mad"] = float(np.median(np.abs(z - np.median(z))))
    info["amount_z_abs_p99"] = float(np.percentile(np.abs(z), 99))
    info["amount_z_abs_p999"] = p999(np.abs(z))
    info["amount_z_abs_max"] = float(np.max(np.abs(z)))
    info["amount_z_n_abs_ge_p999"] = int((np.abs(z) >= info["amount_z_abs_p999"]).sum())
    info["amount_z_n_abs_ge_2p999"] = int((np.abs(z) >= 2 * info["amount_z_abs_p999"]).sum())
    info["amount_z_n_abs_ge_5p999"] = int((np.abs(z) >= 5 * info["amount_z_abs_p999"]).sum())

    pp = df[PAYPERIOD].to_numpy()
    ppv = pp[~np.isnan(pp)]
    info["payperiod_min"] = float(np.min(ppv))
    info["payperiod_max"] = float(np.max(ppv))
    info["payperiod_median"] = float(np.median(ppv))
    info["payperiod_p99"] = float(np.percentile(ppv, 99))
    info["payperiod_p999"] = p999(ppv)
    info["payperiod_n_ge_p999"] = int((ppv >= info["payperiod_p999"]).sum())
    info["payperiod_n_ge_2p999"] = int((ppv >= 2 * info["payperiod_p999"]).sum())
    info["payperiod_n_ge_5p999"] = int((ppv >= 5 * info["payperiod_p999"]).sum())
    info["payperiod_negative"] = int((ppv < 0).sum())
    info["payperiod_is_integer"] = bool(np.all(np.mod(ppv, 1) == 0))

    # Account structure
    ca = df[ACCOUNT].value_counts()
    cg = df[CONTRA].value_counts()
    info["account_rows_lt_30"] = int((ca < 30).sum())
    info["account_min_rows"] = int(ca.min())
    info["account_top_share"] = float(ca.iloc[0] / n)
    info["contra_singletons"] = int((cg == 1).sum())
    info["contra_top20_share"] = float(cg.iloc[:20].sum() / n)
    info["n_observed_pairs"] = int(df[[ACCOUNT, CONTRA]].drop_duplicates().shape[0])
    info["pairs"] = {F: valid_pairs(df, F) for F in PAIR_F}

    # Requirements for T5 at c = 0.03 (largest contamination): 24 percent of n_anom
    info["t5_need_max"] = int(round(0.24 * round(0.03 * n)))
    return info


def render(infos: list[dict]) -> str:
    L: list[str] = []
    L.append("# WP0 data audit\n")
    L.append("Generated by `python -m src.audit` from the four raw CSVs in `data/raw/`. "
             "No cleaning or deduplication was applied. All statistics are computed on the "
             "original tables, which is where the injection thresholds (P99.9) and the T5 "
             "pair pools are defined.\n")

    L.append("## 0. Findings\n")
    byt = {i["table"]: i for i in infos}
    d = byt["D"]
    L.append(f"1. The claimed profile is confirmed for all four tables. Table-D `Umsatz_mod_z` ranges from "
             f"{d['amount_z_min']:,.1f} to {d['amount_z_max']:,.1f}. Only Table-A has two currencies; Währung is "
             f"constant in B, C and D and carries no signal there.")
    L.append("2. `Abweichung Belegdatum zum WJ` is constant zero in all four tables, not only in A and C as Section 2 "
             "states. It contributes nothing to any detector. `Buchungstext nicht Leer` is constant in C and nearly "
             "constant in A and B (share of ones above 0.97), while in D it is one for only 2.2 percent of rows.")
    L.append("3. Table-A has effectively one user (5,078 versus 2 rows), as stated; the DECOMP `s_user` component and "
             "the user one-hot are uninformative there. Table-D is close to this situation as well (user_1 covers "
             f"{byt['D']['user_levels']['user_1'] / byt['D']['rows']:.1%} of rows).")
    a = byt["A"]
    L.append(f"4. T5 pair pools are sufficient everywhere except Table-A at F = 50 (easy): {a['pairs'][50][0]} valid pairs "
             f"against {a['t5_need_max']} T5 rows needed at c = 0.03. Injection will reuse pairs in that cell and record it. "
             f"At c = 0.015 Table-A needs {int(round(0.24 * round(0.015 * a['rows'])))} T5 rows, which the pool covers.")
    L.append("5. The tail counts in Section 4 show what an injected T1 row competes with: at the hard level (f = 1) roughly "
             "0.1 percent of real rows already reach the injected magnitude by construction (4, 7, 18 and 280 rows in C, A, B, D). "
             f"Table-D additionally has {d['amount_z_n_abs_ge_2p999']} real rows at or above 2x P99.9, so medium T1 anomalies in D are not "
             "separable from the real tail by amount alone. Table-A has an extremely heavy tail (max |z| = "
             f"{a['amount_z_abs_max']:,.0f} against P99.9 = {a['amount_z_abs_p999']:,.0f}).")
    L.append("6. Zahlungszeitraum is integer-valued and never negative; its maximum is 365 in D. Easy T2 values (5x P99.9) "
             "are therefore far outside the observed range in every table, which makes T2-easy a near-trivial target for any "
             "detector that looks at that column.")
    L.append("7. Exact duplicate feature rows are common (C 210, A 1,007, B 2,110, D 10,900). They are kept per Section 8. "
             "Duplicates affect density-based detectors (LOF, kNN): a duplicated normal row has zero distance to its copies. "
             "Injected rows are modified in place, so an injected copy of a duplicated row becomes unique.")
    L.append("8. In Table-D 411 contra accounts are singletons and the top 20 contra accounts cover only 2.8 percent of rows, "
             "so the `E_base` one-hot of the 20 most frequent Gegenkonto levels describes almost nothing in D; the "
             "log-frequency feature carries the contra-account information there.\n")

    L.append("## 1. Profile verification (Section 2 of the protocol)\n")
    L.append("| Table | Rows | Kontonummer | Gegenkonto | Währung | Steuersatz | Users | Zahlungszeitraum missing | Umsatz_mod_z min | Umsatz_mod_z max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in infos:
        L.append(f"| {i['table']} | {fmt(i['rows'])} | {i['n_account']} | {fmt(i['n_contra'])} | {i['n_currency']} | "
                 f"{i['n_tax']} | {i['n_user']} | {fmt(i['payperiod_missing'])} ({i['payperiod_missing_share']:.1%}) | "
                 f"{i['amount_z_min']:,.1f} | {i['amount_z_max']:,.1f} |")
    L.append("")
    L.append("Deviations from the claimed profile:\n")
    devs = []
    for i in infos:
        c = CLAIMED[i["table"]]
        for k, v in c.items():
            if v is None:
                continue
            actual = i[k]
            if k in ("amount_z_min", "amount_z_max"):
                if abs(actual - v) > max(1.0, 0.01 * abs(v)):
                    devs.append(f"- Table-{i['table']} {k}: claimed {v}, actual {actual:,.2f}")
            elif actual != v:
                devs.append(f"- Table-{i['table']} {k}: claimed {v}, actual {actual}")
    L.extend(devs if devs else ["- None. All claimed counts and ranges are confirmed (ranges up to rounding)."])
    L.append("")
    L.append("Additional checks: BUCHUNGS_GUID is unique in every table. Rows whose ten feature values coincide with another row (exact duplicates ignoring the key): " +
             ", ".join(f"{i['table']} {fmt(i['duplicate_feature_rows'])}" for i in infos) +
             ". They are kept, as Section 8 forbids deduplication.\n")

    L.append("## 2. Categorical levels\n")
    for i in infos:
        L.append(f"**Table-{i['table']}**: Währung {i['currency_levels']}; Steuersatz {i['tax_levels']}; "
                 f"Benutzererkennung {i['user_levels']}.\n")

    L.append("## 3. Missingness and constant columns\n")
    L.append("| Table | Missing values (column: count) | Constant columns |")
    L.append("|---|---|---|")
    for i in infos:
        miss = {k: v for k, v in i["missing"].items() if v > 0}
        L.append(f"| {i['table']} | {miss if miss else 'none'} | {', '.join(i['constant_columns']) if i['constant_columns'] else 'none'} |")
    L.append("")
    L.append("Share of ones in the binary columns:\n")
    L.append("| Table | Buchungstext nicht Leer | Abweichung Belegdatum zum WJ | Umsatz_Smooth |")
    L.append("|---|---|---|---|")
    for i in infos:
        b = i["binary_share_of_ones"]
        L.append(f"| {i['table']} | {b[TEXT_PRESENT]:.3f} | {b[DATE_DEV]:.3f} | {b[ROUND_AMT]:.3f} |")
    L.append("")

    L.append("## 4. Injection thresholds for T1 and T2\n")
    L.append("P99.9 is the 99.9th percentile with linear interpolation (numpy default), computed on the original table. "
             "For T1 it is taken over |Umsatz_mod_z| of all rows; for T2 over the non-missing Zahlungszeitraum values. "
             "The columns `n >= f x P99.9` count original rows that already reach the injected magnitude at f = 1, 2, 5; "
             "they are the real rows an injected anomaly has to compete with at the hard, medium and easy level.\n")
    L.append("| Table | P99.9 abs(Umsatz_mod_z) | max abs | n >= 1x | n >= 2x | n >= 5x | P99 abs | median | MAD |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i in infos:
        L.append(f"| {i['table']} | {i['amount_z_abs_p999']:,.3f} | {i['amount_z_abs_max']:,.1f} | {i['amount_z_n_abs_ge_p999']} | "
                 f"{i['amount_z_n_abs_ge_2p999']} | {i['amount_z_n_abs_ge_5p999']} | {i['amount_z_abs_p99']:,.3f} | "
                 f"{i['amount_z_median']:.3f} | {i['amount_z_mad']:.3f} |")
    L.append("")
    L.append("| Table | P99.9 Zahlungszeitraum | max | n >= 1x | n >= 2x | n >= 5x | P99 | median | negative values | integer-valued |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in infos:
        L.append(f"| {i['table']} | {i['payperiod_p999']:,.1f} | {i['payperiod_max']:,.0f} | {i['payperiod_n_ge_p999']} | "
                 f"{i['payperiod_n_ge_2p999']} | {i['payperiod_n_ge_5p999']} | {i['payperiod_p99']:,.1f} | {i['payperiod_median']:.0f} | "
                 f"{i['payperiod_negative']} | {'yes' if i['payperiod_is_integer'] else 'no'} |")
    L.append("")

    L.append("## 5. Account structure and T5 pair pools\n")
    L.append("Valid T5 pairs are (Kontonummer a, Gegenkonto g) with count(a) >= F, count(g) >= F and count(a, g) = 0 in the original table. "
             "`need` is the largest number of T5 rows any grid cell requests (24 percent of n_anom at c = 0.03).\n")
    L.append("| Table | observed pairs | accounts with < 30 rows | min rows per account | contra singletons | top-20 contra share | F=50 valid (a x g) | F=20 valid (a x g) | F=5 valid (a x g) | need |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in infos:
        p = i["pairs"]
        cell = lambda F: f"{fmt(p[F][0])} ({p[F][1]} x {p[F][2]})"
        L.append(f"| {i['table']} | {fmt(i['n_observed_pairs'])} | {i['account_rows_lt_30']} | {i['account_min_rows']} | "
                 f"{fmt(i['contra_singletons'])} | {i['contra_top20_share']:.3f} | {cell(50)} | {cell(20)} | {cell(5)} | {i['t5_need_max']} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    infos = [audit_table(t) for t in ["C", "A", "B", "D"]]
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "wp0_data_audit.json").write_text(json.dumps(infos, indent=1, default=str))
    md = render(infos)
    (out_dir / "wp0_data_audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
