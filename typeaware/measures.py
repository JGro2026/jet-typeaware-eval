"""Type-aware budget evaluation of a ranking (the measures of Section 3 of the paper).

All functions take a score vector (higher = more anomalous) and, where needed, an integer type label per unit
(-1 or any value outside `types` for normal units). Ties are broken by ascending position (stable sort).
"""
import numpy as np
from fsr import caps, fsr


def ranking(scores):
    """Units ordered by descending score, ties by ascending position."""
    return np.argsort(-np.asarray(scores, float), kind='stable')


def counts_at_k(scores, type_of, types, k):
    """f_t(k): number of anomalies of each type among the first k units."""
    top = ranking(scores)[:k]; t = np.asarray(type_of)[top]
    return np.array([(t == ty).sum() for ty in types], float)


def precision_at_k(scores, type_of, types, k):
    return counts_at_k(scores, type_of, types, k).sum() / k


def recall_per_type(scores, type_of, types, k):
    """R_t@k and its budget ceiling min(1, k / n_t)."""
    n = np.array([(np.asarray(type_of) == ty).sum() for ty in types], float)
    return counts_at_k(scores, type_of, types, k) / n, np.minimum(1.0, k / n)


def type_coverage(scores, type_of, types, k):
    return int((counts_at_k(scores, type_of, types, k) >= 1).sum())


def fair_share_type_recall(scores, type_of, types, k, weights=None):
    n = np.array([(np.asarray(type_of) == ty).sum() for ty in types], float)
    return fsr(counts_at_k(scores, type_of, types, k), n, k, weights)


def first_hit_ranks(scores, type_of, types):
    """FHR_t (1-based) and the full-coverage budget max_t FHR_t."""
    t = np.asarray(type_of)[ranking(scores)]
    fhr = np.array([int(np.flatnonzero(t == ty)[0]) + 1 for ty in types])
    return fhr, int(fhr.max())
