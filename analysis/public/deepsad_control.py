"""No-feedback control for DeepSAD on the public ledger (added in revision, outside the frozen protocol).

Input: reference/public_ledger/feedback_deepsad-nolabel-control_s{0,1,2}.json, written by
public_ledger/feedback_nolabel_control.py (same review loop, AE-emb initialisation, rounds, seeds and retraining
as feedback.py deepsad, but the revealed labels are withheld, so every unit enters the retraining unlabelled).
Compared with deepsad_public_perseed.csv (DeepSAD with labels, and the AE-emb static top 100).
Outputs: deepsad_control_perseed.csv, deepsad_control_summary.csv.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.PIPELINE)); sys.path.insert(0, str(paths.TYPEAWARE))
from fsr import fsr
from dib_data import load, entries

df = load(); e = entries(df); mk = e.marking.values
M = list(range(1, 9)); n = np.array([(mk == m).sum() for m in M])
rows = []
for s in (0, 1, 2):
    d = json.load(open(paths.public_result(f'feedback_deepsad-nolabel-control_s{s}.json')))['DeepSAD-nolabel']
    rev = np.array([d[f'rev_M{m}'] for m in M])
    assert rev.sum() == d['found@100'], (s, rev.sum(), d['found@100'])
    rows.append(dict(seed=s, method='Control: same protocol, labels withheld', found25=d['found@25'], found50=d['found@50'],
                     found75=d['found@75'], found100=d['found@100'], coverage=int((rev > 0).sum()),
                     FSR100=fsr(rev, n, 100), **{f'M{m}': int(rev[m - 1]) for m in M}, runtime_s=d['runtime_s']))
C = pd.DataFrame(rows)
P = pd.read_csv(paths.find('deepsad_public_perseed.csv'))
for s in (0, 1, 2):
    d = json.load(open(paths.public_result(f'feedback_deepsad_s{s}.json')))['DeepSAD']
    P.loc[(P.seed == s) & P.method.str.startswith('DeepSAD'), ['found50', 'found75']] = [d['found@50'], d['found@75']]
A = pd.concat([P, C], ignore_index=True)
A.to_csv(paths.out_public('deepsad_control_perseed.csv'), index=False)
print(A.round(3).to_string(index=False))
S = A.drop(columns=['runtime_s']).groupby('method').agg(['mean', 'std']).drop(columns='seed')
S.columns = ['_'.join(c) for c in S.columns]
S.round(3).to_csv(paths.out_public('deepsad_control_summary.csv'))
print(S.round(3).T.to_string())
ds = P[P.method.str.startswith('DeepSAD')].sort_values('seed')
st = P[P.method.str.startswith('AE-emb')].sort_values('seed')
cs = C.sort_values('seed')
print('found@100  labels:', ds.found100.tolist(), ' control:', cs.found100.tolist(), ' AE-emb:', st.found100.tolist())
print('gain of labels over control per seed:', (ds.found100.values - cs.found100.values).tolist(),
      'mean', round(float((ds.found100.values - cs.found100.values).mean()), 2))
print('gain of control over AE-emb per seed:', (cs.found100.values - st.found100.values).tolist(),
      'mean', round(float((cs.found100.values - st.found100.values).mean()), 2))
