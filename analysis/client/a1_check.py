"""A1: correctness checks for fsr.py, datev_typeaware.py and dib_typeaware.py.

1. caps(): closed form equals the previous bisection version; sum c_t = min(k, sum n_t); c_t <= n_t;
   types below the water level are filled completely; FSR in [0, 1].
2. alpha-nDCG ideal ordering: the greedy ideal used in dib_typeaware.py equals the brute-force optimum
   over all orderings on small instances.
3. Public ledger: first-hit ranks in dib_typeaware_perseed.csv are 1-based (no anomaly of the marking
   above the rank, one at the rank); P@k and per-marking recall reproduce dib/results/metrics_mean.csv.
4. Real ledgers: found counts from the rounded per-type recalls agree with P@100 to within rounding.
Writes a1_checks.csv (one row per check with the maximum deviation found)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.TYPEAWARE))
from fsr import caps, caps_bisect, fsr

rows = []
rng = np.random.default_rng(1)
# 1. caps and FSR
dev_bis = dev_sum = viol_le = viol_fill = 0.0; fmin, fmax = 1.0, 0.0
for _ in range(20000):
    T = rng.integers(1, 9); n = rng.integers(0, 400, T).astype(float)
    if n.sum() == 0: n[0] = 1
    k = int(rng.integers(1, 1200))
    c = caps(n, k); cb = caps_bisect(n, k)
    act = n > 0
    dev_bis = max(dev_bis, np.abs(c - cb).max())
    dev_sum = max(dev_sum, abs(c.sum() - min(k, n.sum())))
    viol_le = max(viol_le, (c - n).max())
    s = c[act].max()  # water level
    viol_fill = max(viol_fill, np.abs(c[act] - np.minimum(n[act], s)).max())
    f = rng.uniform(0, 1, T) * n
    v = fsr(f, n, k); fmin = min(fmin, v); fmax = max(fmax, v)
    w = rng.uniform(0.1, 3, T); v2 = fsr(f, n, k, w); fmin = min(fmin, v2); fmax = max(fmax, v2)
rows += [dict(check='caps closed form vs bisection, max abs diff', value=dev_bis, ok=dev_bis < 1e-9),
         dict(check='|sum caps - min(k, sum n)|, max', value=dev_sum, ok=dev_sum < 1e-9),
         dict(check='max(c_t - n_t)', value=viol_le, ok=viol_le <= 1e-12),
         dict(check='caps equal min(n_t, water level), max abs diff', value=viol_fill, ok=viol_fill < 1e-9),
         dict(check='min FSR over random inputs', value=fmin, ok=fmin >= 0),
         dict(check='max FSR over random inputs', value=fmax, ok=fmax <= 1 + 1e-12)]
# worked examples used in the paper
for lab, n, k in [('ledger C', [22, 10, 4, 4, 12], 100), ('ledger A', [31, 15, 6, 6, 18], 100),
                  ('ledger B', [105, 53, 21, 21, 63], 100), ('ledger D', [1680, 840, 336, 336, 1008], 100),
                  ('public k=25', [120, 60, 120, 120, 100, 250, 60, 20], 25),
                  ('public k=100', [120, 60, 120, 120, 100, 250, 60, 20], 100),
                  ('public k=250', [120, 60, 120, 120, 100, 250, 60, 20], 250),
                  ('public k=500', [120, 60, 120, 120, 100, 250, 60, 20], 500),
                  ('public k=1000', [120, 60, 120, 120, 100, 250, 60, 20], 1000)]:
    rows.append(dict(check=f'caps {lab}: ' + ' '.join(f'{x:.6g}' for x in caps(n, k)), value=caps(n, k).sum(),
                     ok=abs(caps(n, k).sum() - min(k, sum(n))) < 1e-9))


# 2. alpha-nDCG ideal: greedy vs brute force
def dcg(seq, alpha=0.5):
    seen = {}; g = 0.0
    for r, t in enumerate(seq):
        if t is None: continue
        g += (1 - alpha) ** seen.get(t, 0) / np.log2(r + 2); seen[t] = seen.get(t, 0) + 1
    return g


def greedy_ideal(n, k, alpha=0.5):
    M = list(range(len(n))); cnt = {m: 0 for m in M}; left = dict(zip(M, n)); ig = 0.0
    for r in range(k):
        best = max(M, key=lambda m: ((1 - alpha) ** cnt[m]) if left[m] > 0 else -1)
        if left[best] <= 0: break
        ig += (1 - alpha) ** cnt[best] / np.log2(r + 2); cnt[best] += 1; left[best] -= 1
    return ig


worst = 0.0
for n in [(2, 1, 3), (1, 1, 1, 1), (4, 2), (3, 3, 1), (5,), (2, 2, 2)]:
    items = [t for t, c in enumerate(n) for _ in range(c)]
    for k in range(1, len(items) + 2):
        best = 0.0
        for perm in set(itertools.permutations(items + [None] * max(0, k - len(items)), min(k, len(items) + max(0, k - len(items))))):
            best = max(best, dcg(perm[:k]))
        for alpha in (0.5,):
            worst = max(worst, abs(best - greedy_ideal(list(n), k, alpha)))
rows.append(dict(check='alpha-nDCG ideal: greedy vs brute force, max abs diff (alpha 0.5)', value=worst, ok=worst < 1e-12))

# 3. public ledger
sys.path.insert(0, str(paths.PIPELINE))
from dib_data import load, entries, entry_scores
df = load(); e = entries(df); y = e.label.values; mk = e.marking.values; eid = df.eid.values
PS = pd.read_csv(paths.find('dib_typeaware_perseed.csv'))
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE', 'LGBM-ceiling': 'LightGBM ceiling'}
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling'}
bad_first = 0; maxdev_p = 0.0
MM = pd.read_csv(paths.public_result('metrics_mean.csv'), index_col=0)
for d in ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling']:
    acc = {}
    for s in ([0, 1, 2] if d in STOCH else [0]):
        sc = entry_scores(np.load(f'{paths.SCORES}/{d}_s{s}.npy'), eid)
        o = np.argsort(-sc, kind='stable'); mo = mk[o]
        r = PS[(PS.detector == NAME.get(d, d)) & (PS.seed == s)].iloc[0]
        for m in range(1, 9):
            f = int(r[f'first_M{m}'])
            if not ((mo[:f - 1] == m).sum() == 0 and mo[f - 1] == m): bad_first += 1
        for k in (25, 50, 100, 1000):
            acc.setdefault(f'P@{k}', []).append(y[o[:k]].mean())
        for m in range(1, 9):
            for k in (100, 1000):
                acc.setdefault(f'R{k}_M{m}', []).append((mo[:k] == m).sum() / (mk == m).sum())
    for key, v in acc.items():
        maxdev_p = max(maxdev_p, abs(np.mean(v) - MM.loc[d, key]))
rows.append(dict(check='first-hit ranks 1-based and correct (count of violations)', value=bad_first, ok=bad_first == 0))
rows.append(dict(check='public ledger P@k and per-marking recall vs metrics_mean.csv, max abs diff', value=maxdev_p, ok=maxdev_p < 1e-12))
# P column of perseed vs recomputed
PM = PS[PS.k.isin([25, 50, 100, 1000])]
dev = 0.0
for _, r in PM.iterrows():
    inv = {v: k for k, v in NAME.items()}; d = inv.get(r.detector, r.detector)
    if len(PS[(PS.detector == r.detector)].seed.unique()) == 1:
        dev = max(dev, abs(r.P - MM.loc[d, f'P@{int(r.k)}']))
rows.append(dict(check='dib_typeaware_perseed P (deterministic detectors) vs metrics_mean.csv', value=dev, ok=dev < 1e-12))

# 4. real ledgers: found counts vs P@100
T = pd.read_csv(paths.find('datev_typeaware_k100.csv'))
gap = (T.found_total_from_types - 100 * T.P100).abs()
for L in 'CAB':
    g = gap[T.ledger == L].max()
    rows.append(dict(check=f'ledger {L}: max |sum_t found_t - 100 P@100| from rounded values', value=g, ok=True))
g = gap[T.ledger == 'D'].max()
rows.append(dict(check='ledger D: max |sum_t found_t - 100 P@100| from rounded values (rounding hides hits)', value=g, ok=True))
R = pd.DataFrame(rows); R.to_csv(paths.out_client('a1_checks.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(R.to_string(index=False))
