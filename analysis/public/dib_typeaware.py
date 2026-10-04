"""Exact type-aware metrics on the public (DIB) ledger, entry level, from the saved per-line scores."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.PIPELINE)); sys.path.insert(0, str(paths.TYPEAWARE))
from dib_data import load, entries, entry_scores
from fsr import fsr, caps
SC = str(paths.SCORES)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE', 'LGBM-ceiling': 'LightGBM ceiling'}
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling'}
KS = [25, 50, 100, 250, 500, 1000, 2000]
df = load(); e = entries(df); y = e.label.values; mk = e.marking.values; eid = df.eid.values
M = list(range(1, 9)); n = np.array([(mk == m).sum() for m in M])
def order(s): return np.argsort(-np.asarray(s), kind='stable')
def alpha_ndcg(o, k, alpha=0.5):
    # alpha-nDCG over marking types (Clarke et al. 2008): gain of an anomaly of type t at rank r is (1-alpha)^(hits of t before r)
    seen = {m: 0 for m in M}; g = 0.0
    for r, i in enumerate(o[:k]):
        if y[i]:
            m = mk[i]; g += (1 - alpha) ** seen[m] / np.log2(r + 2); seen[m] += 1
    # ideal: greedy, which is optimal for this per-type diminishing gain with equal-length lists
    cnt = {m: 0 for m in M}; left = dict(zip(M, n)); ig = 0.0
    for r in range(k):
        best = max(M, key=lambda m: ((1 - alpha) ** cnt[m]) if left[m] > 0 else -1)
        if left[best] <= 0: break
        ig += (1 - alpha) ** cnt[best] / np.log2(r + 2); cnt[best] += 1; left[best] -= 1
    return g / ig
rows = []
for d in DETS:
    for s in ([0, 1, 2] if d in STOCH else [0]):
        sc = entry_scores(np.load(f'{SC}/{d}_s{s}.npy'), eid); o = order(sc)
        first = {m: int(np.where(mk[o] == m)[0][0]) + 1 for m in M}
        for k in KS:
            top = o[:k]; found = np.array([(mk[top] == m).sum() for m in M])
            rows.append(dict(detector=NAME.get(d, d), seed=s, k=k, P=found.sum() / k, coverage=int((found > 0).sum()),
                             FSR=fsr(found, n, k), alpha_nDCG=alpha_ndcg(o, k),
                             **{f'first_M{m}': first[m] for m in M}))
R = pd.DataFrame(rows); R.to_csv(paths.out_public('dib_typeaware_perseed.csv'), index=False)
A = R.groupby(['detector', 'k']).mean(numeric_only=True).drop(columns='seed').reset_index()
A['n_seeds'] = R.groupby(['detector', 'k']).size().values
A.to_csv(paths.out_public('dib_typeaware_mean.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for k in [100, 1000]:
    a = A[A.k == k].sort_values('FSR', ascending=False)
    print(f'--- k = {k}  caps:', np.round(caps(n, k), 1))
    print(a[['detector', 'P', 'coverage', 'FSR', 'alpha_nDCG']].round(3).to_string(index=False))
F = A[A.k == 100][['detector'] + [f'first_M{m}' for m in M]].round(0)
print('--- rank of the first anomaly of each marking (mean over seeds)'); print(F.to_string(index=False))
