"""Rank-based evaluation (Section 3.6).

Every detector emits a continuous score, higher is more anomalous. Rankings
are obtained with a stable descending sort, so ties are broken by ascending
row position. Injected rows are sampled uniformly over positions, so this tie
rule is not systematically favourable or unfavourable to any detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from .inject import TYPES

KS = [25, 50, 100]


def ranking(scores: np.ndarray) -> np.ndarray:
    """Row positions ordered from most to least anomalous."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Average-rank percentile in (0, 1], higher is more anomalous."""
    s = np.asarray(scores, dtype=np.float64)
    return stats.rankdata(s, method="average") / len(s)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = ranking(scores)[:k]
    return float(np.asarray(labels)[order].sum() / k)


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    labels = np.asarray(labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    order = ranking(scores)[:k]
    return float(labels[order].sum() / n_pos)


def recall_at_k_by_type(scores: np.ndarray, types: np.ndarray, k: int) -> dict[str, float]:
    order = ranking(scores)[:k]
    types = np.asarray(types, dtype=object)
    found = types[order]
    out = {}
    for t in TYPES:
        n_t = int((types == t).sum())
        out[t] = float((found == t).sum() / n_t) if n_t > 0 else float("nan")
    return out


def compute_metrics(scores: np.ndarray, labels: np.ndarray, types: np.ndarray) -> dict[str, float]:
    """All Section 3.6 metrics for one score vector. Keys are metric names."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    m: dict[str, float] = {}
    for k in KS:
        m[f"p_at_{k}"] = precision_at_k(scores, labels, k)
        m[f"r_at_{k}"] = recall_at_k(scores, labels, k)
    for t, v in recall_at_k_by_type(scores, types, 100).items():
        m[f"r100_{t}"] = v
    if 0 < labels.sum() < len(labels):
        m["ap"] = float(average_precision_score(labels, scores))
        m["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        m["ap"] = float("nan")
        m["roc_auc"] = float("nan")
    m["n_anom"] = float(labels.sum())
    return m


def check_scores(scores: np.ndarray, name: str = "") -> None:
    """Raise if a score vector is not finite or is constant (Section 6)."""
    s = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{name}: non-finite scores")
    if np.nanmax(s) == np.nanmin(s):
        raise ValueError(f"{name}: constant score vector")


def mean_ranks(metric_table: pd.DataFrame) -> pd.Series:
    """metric_table: rows = cells (table, c, d, seed), columns = detectors,
    values = a metric where higher is better. Returns mean rank per detector
    (1 = best), average ranks for ties."""
    ranks = metric_table.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def friedman_nemenyi(metric_table: pd.DataFrame) -> dict:
    """Friedman test across detectors (columns) over cells (rows) and Nemenyi
    post-hoc, as in ADBench. Returns statistic, p-value, mean ranks, the
    Nemenyi p-value matrix and the critical difference at alpha = 0.05."""
    import scikit_posthocs as sp

    M = metric_table.dropna(axis=0, how="any")
    stat, p = stats.friedmanchisquare(*[M[c].to_numpy() for c in M.columns])
    ranks = mean_ranks(M)
    nemenyi = sp.posthoc_nemenyi_friedman(M.to_numpy())
    nemenyi.index = M.columns
    nemenyi.columns = M.columns
    k, n = M.shape[1], M.shape[0]
    q_alpha = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"statistic": float(stat), "p_value": float(p), "mean_ranks": ranks,
            "nemenyi_p": nemenyi, "critical_difference": float(cd), "n_cells": int(n), "n_detectors": int(k)}
