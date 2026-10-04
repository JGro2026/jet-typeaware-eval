"""Type-aware metrics on the real ledgers from run B aggregates (core grid, k = 100).
Point values from the two-decimal per-type recalls (seed means). coverage_types counts types with a
non-zero rounded seed-mean recall; on ledgers B and D rounding can hide hits (see robustness/a2_bounds.py
for exact bounds). Full precision since the A1 review (previously found counts rounded to 0.1)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, re, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = str(paths.CLIENT_RESULTS)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE'}
# injected counts per type at c = 1.5 percent (protocol: T2..T5 round(p*n), T1 remainder; Python round)
NA = {'C': 52, 'A': 76, 'B': 263, 'D': 4200}
P = {'T2': 0.20, 'T3': 0.08, 'T4': 0.08, 'T5': 0.24}
def counts(n):
    c = {t: round(p * n) for t, p in P.items()}; c['T1'] = n - sum(c.values())
    return {t: c[t] for t in ['T1', 'T2', 'T3', 'T4', 'T5']}
txt = open(SRC).read()
sec = txt.split('## T2.')[1].split('## T3.')[0]
T1sec = txt.split('## T1.')[1].split('## T2.')[0]
rows = []
for L in ['C', 'A', 'B', 'D']:
    blk = sec.split(f'**Table-{L}**')[1].split('**Table-')[0]
    b1 = T1sec.split(f'**Table-{L}**')[1].split('**Table-')[0]
    nt = counts(NA[L])
    for line in blk.splitlines():
        m = re.match(r'\|\s*([^|]+?)\s*\|(.*)\|', line)
        if not m or m.group(1) not in DETS: continue
        det = m.group(1); r = [float(x) for x in m.group(2).split('|')]
        p100 = None
        for l1 in b1.splitlines():
            if l1.startswith(f'| {det} |'):
                p100 = float(l1.split('|')[4].split('±')[0])
        found = np.array([r[i] * nt[t] for i, t in enumerate(nt)])
        lo = np.array([max(0, r[i] - 0.005) * nt[t] for i, t in enumerate(nt)])
        hi = np.array([(r[i] + 0.005) * nt[t] for i, t in enumerate(nt)])
        cov = int((np.array(r) > 0).sum())
        p = found / found.sum() if found.sum() > 0 else found
        ent = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if found.sum() > 0 else 0.0
        ceil = np.array([min(1.0, 100 / nt[t]) for t in nt])
        rows.append(dict(ledger=L, detector=NAME.get(det, det), P100=p100, **{f'R_{t}': r[i] for i, t in enumerate(nt)},
                         **{f'found_{t}': found[i] for i, t in enumerate(nt)},
                         found_total_from_types=found.sum(), coverage_types=cov,
                         eff_types=ent, **{f'ceil_{t}': ceil[i] for i, t in enumerate(nt)}))
D = pd.DataFrame(rows); D.to_csv(paths.out_client('datev_typeaware_k100.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print('type counts:', {L: counts(n) for L, n in NA.items()})
print(D[['ledger', 'detector', 'P100', 'R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5', 'found_total_from_types', 'coverage_types', 'eff_types']].to_string(index=False))
# rank changes: P@100 vs coverage and effective types
from scipy.stats import kendalltau
for L in ['C', 'A', 'B', 'D']:
    d = D[D.ledger == L]
    t1, _ = kendalltau(d.P100, d.eff_types); t2, _ = kendalltau(d.P100, d.coverage_types)
    best_p = d.sort_values('P100', ascending=False).detector.iloc[0]; best_e = d.sort_values(['eff_types', 'P100'], ascending=False).detector.iloc[0]
    print(L, 'tau(P100, eff_types)=%.2f tau(P100, coverage)=%.2f' % (t1, t2), 'best by P100:', best_p, '| best by eff types:', best_e)
