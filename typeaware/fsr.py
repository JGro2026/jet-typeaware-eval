"""Fair-share type recall (FSR@k) with water-filling caps.

caps(n, k): real-valued fair shares c_t = min(n_t, s), with the water level s chosen so that
sum_t c_t = min(k, sum_t n_t). Computed in closed form (sort the type sizes, fill the smallest
types completely while they are below the equal share of the remaining budget, split the rest
equally). Types with n_t = 0 get c_t = 0.

fsr(found, n, k, w): sum_t w_t * min(found_t, c_t) / c_t over the types with n_t > 0, weights
renormalised over those types (equal weights by default). FSR lies in [0, 1].

The earlier bisection version (caps_bisect) is kept for the regression check in
robustness/a1_check.py; both agree to within 1e-9 on every input used in the paper.
"""
import numpy as np


def caps(n, k):
    """Water-filling fair shares: c_t = min(n_t, s) with sum c_t = min(k, sum n_t)."""
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    budget = float(min(k, n.sum()))
    idx = [i for i in np.argsort(n, kind='stable') if n[i] > 0]
    m = len(idx)
    for j, i in enumerate(idx):
        share = budget / (m - j)
        if n[i] <= share:
            out[i] = n[i]; budget -= n[i]
        else:
            for i2 in idx[j:]:
                out[i2] = share
            break
    return out


def caps_bisect(n, k):
    """Previous implementation (bisection on the water level); kept for the regression check."""
    n = np.asarray(n, float); k = min(k, n.sum())
    lo, hi = 0.0, n.max()
    for _ in range(200):
        s = (lo + hi) / 2
        if np.minimum(n, s).sum() < k: lo = s
        else: hi = s
    return np.minimum(n, hi)


def fsr(found, n, k, w=None):
    n = np.asarray(n, float); c = caps(n, k); found = np.asarray(found, float)
    act = n > 0
    w = np.ones(len(n)) if w is None else np.asarray(w, float)
    w = np.where(act, w, 0.0); w = w / w.sum()
    ratio = np.zeros(len(n))
    ratio[act] = np.minimum(found[act], c[act]) / c[act]
    return float((w * ratio).sum())
