"""Public ledger: leaders and Kendall tau_b under P@k, macro-averaged recall (mean of R_t@k over the markings) and FSR@k,
at k = 100 and k = 1,000, exact per seed from the saved entry scores and averaged over seeds (public-ledger rows of
Table 7). Outputs: a11_public_macro.csv, a11_public_summary.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import sys
sys.path.insert(0, str(paths.TYPEAWARE)); sys.path.insert(0, str(paths.PIPELINE))
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from fsr import fsr
from dib_data import load, entries, entry_scores

DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb']; NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE'}
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb'}
df = load(); e = entries(df); mk = e.marking.values; eid = df.eid.values
M = list(range(1, 9)); n = np.array([(mk == m).sum() for m in M], float)
pr = []
for det in DETS:
    for s in ([0, 1, 2] if det in STOCH else [0]):
        o = np.argsort(-np.asarray(entry_scores(np.load(paths.SCORES / f'{det}_s{s}.npy'), eid)), kind='stable')
        for k in (100, 1000):
            ff = np.array([(mk[o[:k]] == m).sum() for m in M], float)
            pr.append(dict(detector=NAME.get(det, det), seed=s, k=k, P=ff.sum() / k, macro=(ff / n).mean(), FSR=fsr(ff, n, k)))
P = pd.DataFrame(pr).groupby(['detector', 'k']).mean(numeric_only=True).drop(columns='seed').reset_index()
P.to_csv(paths.out_public('a11_public_macro.csv'), index=False)
rows = []
def lead(g, col):
    m = g[col].max(); return ', '.join(g.detector[g[col] >= m - 1e-9])
for k in (100, 1000):
    g = P[P.k == k]
    rows.append(dict(ledger='public', k=k, P_leader=lead(g, 'P'), macro_leader=lead(g, 'macro'), FSR_leader=lead(g, 'FSR'),
                     tau_P_macro=kendalltau(g.P, g.macro).statistic, tau_macro_FSR=kendalltau(g.macro, g.FSR).statistic,
                     tau_P_FSR=kendalltau(g.P, g.FSR).statistic))
S = pd.DataFrame(rows); S.to_csv(paths.out_public('a11_public_summary.csv'), index=False)
print(S.round(2).to_string(index=False))
