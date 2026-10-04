"""Checks of the fair-share caps and FSR properties stated in the paper (run: python3 test_measures.py)."""
import numpy as np
from fsr import caps, fsr
from measures import precision_at_k, type_coverage, fair_share_type_recall, first_hit_ranks

# Section 3.2 example: k = 100, types of 10, 30, 200 and 400 anomalies -> caps 10, 30, 30, 30
n = np.array([10, 30, 200, 400], float)
assert np.allclose(caps(n, 100), [10, 30, 30, 30])
assert np.isclose(fsr([0, 0, 0, 100], n, 100), 0.25)      # budget spent on the largest type
assert np.isclose(fsr([10, 30, 30, 30], n, 100), 1.0)      # every cap filled
# caps sum to min(k, sum n) and never exceed n
for k in (5, 50, 100, 640, 1000):
    c = caps(n, k); assert np.isclose(c.sum(), min(k, n.sum())) and (c <= n + 1e-12).all()
# monotone in every count; concave (Jensen): FSR(mean counts) >= mean FSR
rng = np.random.default_rng(0)
for _ in range(200):
    f = rng.integers(0, 60, (5, 4)).astype(float); f = np.minimum(f, n)
    assert fsr(f.mean(0), n, 100) >= np.mean([fsr(x, n, 100) for x in f]) - 1e-12
# ranking helpers on a toy ledger
scores = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], float)
types = np.array([0, 0, -1, 1, -1, -1, 2, -1, -1, -1])
assert precision_at_k(scores, types, [0, 1, 2], 4) == 0.75
assert type_coverage(scores, types, [0, 1, 2], 4) == 2
fhr, full = first_hit_ranks(scores, types, [0, 1, 2]); assert list(fhr) == [1, 4, 7] and full == 7
print('all checks passed')
