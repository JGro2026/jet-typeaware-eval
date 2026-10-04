"""A10c: FSR@k under alternative marking weights on the public ledger, exact per seed from the saved entry scores
(existing results only). Schemes: uniform; each marking doubled; each marking halved; 2,000 random vectors with every
weight within [0.5, 2] x 1/8. Reports the FSR leader and Kendall tau_b(P@k, FSR@k) over the nine unsupervised detectors."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, str(paths.TYPEAWARE)); sys.path.insert(0, str(paths.PIPELINE))
from fsr import caps
from dib_data import load, entries, entry_scores
SC = str(paths.SCORES)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE'}
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb'}
df = load(); e = entries(df); mk = e.marking.values; eid = df.eid.values
M = list(range(1, 9)); n = np.array([(mk == m).sum() for m in M], float)
cnt = {}
for d in DETS:
    per = {100: [], 1000: []}
    for s in ([0, 1, 2] if d in STOCH else [0]):
        o = np.argsort(-np.asarray(entry_scores(np.load(f'{SC}/{d}_s{s}.npy'), eid)), kind='stable')
        for k in (100, 1000):
            top = o[:k]; per[k].append([(mk[top] == m).sum() for m in M])
    cnt[NAME.get(d, d)] = {k: np.array(v, float) for k, v in per.items()}
dets = list(cnt)
def evaluate(w, k):
    c = caps(n, k)
    F = np.array([np.mean([(w * np.minimum(f, c) / c).sum() for f in cnt[d][k]]) for d in dets])
    P = np.array([np.mean([f.sum() / k for f in cnt[d][k]]) for d in dets])
    return F, P
rng = np.random.default_rng(11)
def rand_w(T, N):
    out = []
    while len(out) < N:
        w = rng.dirichlet(np.ones(T))
        if (w >= 0.5 / T).all() and (w <= 2 / T).all(): out.append(w)
    return out
sch = {'uniform': np.full(8, 1 / 8)}
for i in range(8):
    v = np.ones(8); v[i] = 2; sch[f'M{i+1} x2'] = v / v.sum()
    v = np.ones(8); v[i] = 0.5; sch[f'M{i+1} x0.5'] = v / v.sum()
rows = []
for k in (100, 1000):
    for name, w in sch.items():
        F, P = evaluate(w, k)
        rows.append(dict(k=k, scheme=name, P_leader=dets[P.argmax()], FSR_leader=dets[F.argmax()], FSR_leader_value=F.max(),
                         tau_P_FSR=kendalltau(P, F).statistic))
S = pd.DataFrame(rows); S.to_csv(paths.out_public('a10c_public_weight_schemes.csv'), index=False)
pd.set_option('display.width', 200); print(S.round(3).to_string(index=False))
R = []
for k in (100, 1000):
    leads = []; taus = []
    for w in rand_w(8, 2000):
        F, P = evaluate(w, k); leads.append(dets[F.argmax()]); taus.append(kendalltau(P, F).statistic)
    R.append(dict(k=k, draws=2000, FSR_leaders=pd.Series(leads).value_counts(normalize=True).round(3).to_dict(),
                  tau_min=np.min(taus), tau_median=np.median(taus), share_tau_eq_1=np.mean(np.isclose(taus, 1))))
RR = pd.DataFrame(R); RR.to_csv(paths.out_public('a10c_public_random_weights.csv'), index=False); print(RR.to_string(index=False))
