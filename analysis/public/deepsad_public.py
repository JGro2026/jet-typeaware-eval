"""DeepSAD under the review protocol on the public ledger (Table 8b, Table B.1): markings among the 100 reviewed
entries against the first 100 entries of the AE-emb model the run starts from, type coverage and FSR@100.
Outputs: deepsad_public_perseed.csv, deepsad_public_summary.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, str(paths.TYPEAWARE)); sys.path.insert(0, str(paths.PIPELINE))
from fsr import fsr, caps

from dib_data import load, entries, entry_scores
df = load(); e = entries(df); mk = e.marking.values; eid = df.eid.values
M = list(range(1, 9)); n = np.array([(mk == m).sum() for m in M])
print('n per marking:', n.tolist(), 'caps@100:', np.round(caps(n, 100), 2).tolist())
prow = []
for s in (0, 1, 2):
    d = json.load(open(paths.public_result(f'feedback_deepsad_s{s}.json')))['DeepSAD']
    rev = np.array([d[f'rev_M{m}'] for m in M])
    assert rev.sum() == d['found@100'], (s, rev.sum(), d['found@100'])
    sc = entry_scores(np.load(paths.SCORES / f'AE-emb_s{s}.npy'), eid)
    o = np.argsort(-np.asarray(sc), kind='stable'); top = o[:100]
    st = np.array([(mk[top] == m).sum() for m in M])
    top25 = o[:25]; st25 = int(np.isin(mk[top25], M).sum())
    prow.append(dict(seed=s, method='DeepSAD (100 reviews)', found25=d['found@25'], found100=d['found@100'],
                     coverage=int((rev > 0).sum()), FSR100=fsr(rev, n, 100), **{f'M{m}': int(rev[m - 1]) for m in M}))
    prow.append(dict(seed=s, method='AE-emb static top 100', found25=st25, found100=int(st.sum()),
                     coverage=int((st > 0).sum()), FSR100=fsr(st, n, 100), **{f'M{m}': int(st[m - 1]) for m in M}))
P = pd.DataFrame(prow); P.to_csv(paths.out_public('deepsad_public_perseed.csv'), index=False)
print(P.round(3).to_string(index=False))
S = P.groupby('method').agg(['mean', 'std']).drop(columns='seed')
S.columns = ['_'.join(c) for c in S.columns]
S.round(3).to_csv(paths.out_public('deepsad_public_summary.csv'))
print(S.round(3).T.to_string())
g = P[P.method.str.startswith('DeepSAD')].found100.values - P[P.method.str.startswith('AE-emb')].found100.values
print('per-seed gain over AE-emb start:', g.tolist(), 'mean', g.mean().round(2))
