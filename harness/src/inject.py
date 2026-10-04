"""Typed anomaly injection (Section 3.2 of the protocol).

Rows are modified in place; nothing is appended. Each row is modified by at
most one type. A manifest records every modified field with its original and
new value. Every random choice depends only on (table, c, d, seed) through a
single numpy Generator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .load import (ACCOUNT, AMOUNT_Z, CONTRA, KEY, PAYPERIOD, dataset_hash,
                   p999)

TYPES = ["T1", "T2", "T3", "T4", "T5"]
PROPORTIONS = {"T1": 0.40, "T2": 0.20, "T3": 0.08, "T4": 0.08, "T5": 0.24}
CONTAMINATIONS = [0.005, 0.015, 0.03]
DIFFICULTIES = ["easy", "medium", "hard"]
F_MULT = {"easy": 5.0, "medium": 2.0, "hard": 1.0}   # T1, T2 multiplier of P99.9
F_PAIR = {"easy": 50, "medium": 20, "hard": 5}       # T5 minimum marginal count
FRESH_ID_SPAN = 1_000_000                            # range width for fresh ids

# Foorthuis (2021) typology, verified against the paper (Fig. 3 and Section 3.2)
FOORTHUIS = {
    "T1": "Type I uncommon number anomaly, subtype Ia extreme tail value",
    "T2": "Type I uncommon number anomaly, subtype Ia extreme tail value",
    "T3": "Type II uncommon class anomaly, subtype IIa unusual class",
    "T4": "Type II uncommon class anomaly, subtype IIa unusual class",
    "T5": "Type V multidimensional categorical anomaly, subtype Va uncommon class combination",
}


def allocate_counts(n_anom: int) -> dict[str, int]:
    """Fixed proportions, each rounded to the nearest integer; the remainder
    (positive or negative) goes to T1 so that the counts sum to n_anom."""
    counts = {t: int(round(PROPORTIONS[t] * n_anom)) for t in TYPES if t != "T1"}
    counts["T1"] = n_anom - sum(counts.values())
    if counts["T1"] < 0:
        raise ValueError(f"n_anom={n_anom} too small for the fixed proportions")
    return {t: counts[t] for t in TYPES}


def n_anomalies(n_rows: int, c: float) -> int:
    return int(round(c * n_rows))


@dataclass
class InjectionResult:
    data: pd.DataFrame            # injected table, same rows and order as the input
    manifest: pd.DataFrame        # one row per table row (Section 5 schema)
    meta: dict = field(default_factory=dict)

    @property
    def labels(self) -> np.ndarray:
        return self.manifest["label"].to_numpy()

    @property
    def types(self) -> np.ndarray:
        return self.manifest["anom_type"].to_numpy(dtype=object)


def _fresh_ids(rng: np.random.Generator, existing: set[int], n: int, max_id: int) -> np.ndarray:
    """n distinct integer ids drawn from (max_id, max_id + FRESH_ID_SPAN] that are
    absent from `existing`. Seed-dependent."""
    out: list[int] = []
    seen: set[int] = set()
    while len(out) < n:
        cand = rng.integers(max_id + 1, max_id + 1 + FRESH_ID_SPAN, size=max(4 * (n - len(out)), 8))
        for v in cand.tolist():
            if v in existing or v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) == n:
                break
    return np.asarray(out, dtype=np.int64)


def valid_pair_pool(df: pd.DataFrame, F: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (a, g) with count(a) >= F, count(g) >= F, count(a, g) = 0 in `df`.
    Returns (a_levels, g_levels, boolean matrix of validity) in sorted level order."""
    ca = df[ACCOUNT].value_counts()
    cg = df[CONTRA].value_counts()
    a_levels = np.sort(ca[ca >= F].index.to_numpy())
    g_levels = np.sort(cg[cg >= F].index.to_numpy())
    valid = np.ones((len(a_levels), len(g_levels)), dtype=bool)
    if len(a_levels) and len(g_levels):
        obs = df[[ACCOUNT, CONTRA]].drop_duplicates()
        ai = np.searchsorted(a_levels, obs[ACCOUNT].to_numpy())
        gi = np.searchsorted(g_levels, obs[CONTRA].to_numpy())
        ok = (ai < len(a_levels)) & (gi < len(g_levels))
        ai, gi = ai[ok], gi[ok]
        ok2 = (a_levels[ai] == obs[ACCOUNT].to_numpy()[ok]) & (g_levels[gi] == obs[CONTRA].to_numpy()[ok])
        valid[ai[ok2], gi[ok2]] = False
    return a_levels, g_levels, valid


def inject(df: pd.DataFrame, table: str, c: float, d: str, seed: int) -> InjectionResult:
    """Inject typed anomalies into a copy of `df` (Section 3.2)."""
    if d not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {d!r}")
    rng = np.random.default_rng([int(seed), 0xDA4])  # single generator for everything
    orig = df.reset_index(drop=True)
    N = len(orig)
    n_anom = n_anomalies(N, c)
    counts = allocate_counts(n_anom)
    f = F_MULT[d]
    F = F_PAIR[d]

    # Thresholds on the original table
    p999_amt = p999(np.abs(orig[AMOUNT_Z].to_numpy()))
    p999_pay = p999(orig[PAYPERIOD].to_numpy())

    # --- Row sampling: one random permutation, T2 restricted to non-missing
    # Zahlungszeitraum, all types disjoint, at most one type per row.
    perm = rng.permutation(N)
    pay_ok = orig[PAYPERIOD].notna().to_numpy()
    assigned = np.full(N, "", dtype=object)
    t2_rows = [i for i in perm if pay_ok[i]][: counts["T2"]]
    if len(t2_rows) < counts["T2"]:
        raise ValueError("not enough rows with non-missing Zahlungszeitraum for T2")
    assigned[t2_rows] = "T2"
    rest = [i for i in perm if assigned[i] == ""]
    pos = 0
    for t in ["T1", "T3", "T4", "T5"]:
        rows = rest[pos: pos + counts[t]]
        assigned[rows] = t
        pos += counts[t]
    assert pos == n_anom - counts["T2"]

    out = orig.copy()
    mod_fields: list[list[str]] = [[] for _ in range(N)]
    mod_orig: list[list] = [[] for _ in range(N)]
    mod_new: list[list] = [[] for _ in range(N)]

    def record(i: int, col: str, new):
        mod_fields[i].append(col)
        old = orig.at[i, col]
        mod_orig[i].append(None if pd.isna(old) else (old.item() if hasattr(old, "item") else old))
        mod_new[i].append(new.item() if hasattr(new, "item") else new)
        out.at[i, col] = new

    # --- T1 extreme amount
    t1 = np.flatnonzero(assigned == "T1")
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(t1))
    for i, s in zip(t1, signs):
        record(int(i), AMOUNT_Z, float(s * f * p999_amt))

    # --- T2 extreme payment period
    for i in np.flatnonzero(assigned == "T2"):
        record(int(i), PAYPERIOD, float(f * p999_pay))

    # --- T3 unseen contra account, T4 unseen account (fresh id per row)
    t3 = np.flatnonzero(assigned == "T3")
    t4 = np.flatnonzero(assigned == "T4")
    contra_existing = set(orig[CONTRA].unique().tolist())
    account_existing = set(orig[ACCOUNT].unique().tolist())
    ids3 = _fresh_ids(rng, contra_existing, len(t3), int(orig[CONTRA].max()))
    ids4 = _fresh_ids(rng, account_existing, len(t4), int(orig[ACCOUNT].max()))
    for i, v in zip(t3, ids3):
        record(int(i), CONTRA, int(v))
    for i, v in zip(t4, ids4):
        record(int(i), ACCOUNT, int(v))

    # --- T5 unusual account pair
    t5 = np.flatnonzero(assigned == "T5")
    a_levels, g_levels, valid = valid_pair_pool(orig, F)
    pool = np.flatnonzero(valid.ravel())
    n_valid = int(len(pool))
    pairs_reused = False
    if len(t5):
        if n_valid == 0:
            raise ValueError(f"no valid T5 pairs at F={F} for table {table}")
        if n_valid >= len(t5):
            chosen = rng.choice(pool, size=len(t5), replace=False)
        else:
            pairs_reused = True
            chosen = np.concatenate([rng.permutation(pool),
                                     rng.choice(pool, size=len(t5) - n_valid, replace=True)])
        ai, gi = np.unravel_index(chosen, valid.shape)
        for i, a, g in zip(t5, a_levels[ai], g_levels[gi]):
            record(int(i), ACCOUNT, int(a))
            record(int(i), CONTRA, int(g))

    # --- Manifest
    label = (assigned != "").astype(np.int8)
    h = dataset_hash(out)
    manifest = pd.DataFrame({
        "row_id": orig[KEY].to_numpy(),
        "table": table,
        "c": float(c),
        "d": d,
        "seed": int(seed),
        "label": label,
        "anom_type": [t if t else None for t in assigned],
        "difficulty": [(d if t in ("T1", "T2", "T5") else ("single" if t else None)) for t in assigned],
        "modified_fields": [json.dumps(m) if m else None for m in mod_fields],
        "original_values": [json.dumps(m) if m else None for m in mod_orig],
        "new_values": [json.dumps(m) if m else None for m in mod_new],
        "dataset_hash": h,
    })
    meta = {
        "table": table, "c": float(c), "d": d, "seed": int(seed), "N": N,
        "n_anom": n_anom, "counts": counts, "f_mult": f, "F_pair": F,
        "p999_abs_amount_z": p999_amt, "p999_payperiod": p999_pay,
        "n_valid_pairs": n_valid, "pairs_reused": pairs_reused,
        "dataset_hash": h,
    }
    return InjectionResult(data=out, manifest=manifest, meta=meta)
