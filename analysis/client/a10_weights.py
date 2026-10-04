"""A10: sensitivity of FSR@k to the type weights w_t (existing results only, no new runs).
Client ledgers: FSR@100 under alternative weights with exact lattice bounds (same constraints as a2_bounds.py:
seed-mean per-type recalls printed to two decimals, P@100 printed to two decimals, 5 seeds).
Public ledger: exact per-seed FSR@100 and FSR@1000 from the saved entry scores, averaged over seeds.
Weight schemes: uniform; each type doubled (2 : 1 : ... : 1, normalised); each type halved; and 2,000 random vectors
drawn uniformly from the simplex subject to every weight lying within [0.5, 2] times the uniform weight."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, str(paths.TYPEAWARE)); sys.path.insert(0, HERE)
from fsr import caps, fsr
import common_datev as cd
from a2_bounds import boxes, solve, K, S

def fsr_bounds_w(R, n, P, w):
    T = len(n); c = caps(n, K); lo, hi, Hlo, Hhi = boxes(R, n, P, True); cc = c * S
    cobj = np.r_[np.zeros(T), -w / cc]
    A = np.zeros((T + 1, 2 * T)); A[:T, :T] = -np.eye(T); A[:T, T:] = np.eye(T); A[T, :T] = 1
    r = solve(cobj, A, np.r_[np.full(T, -np.inf), Hlo], np.r_[np.zeros(T), Hhi], np.r_[lo, np.zeros(T)], np.r_[hi, cc],
              np.r_[np.ones(T), np.zeros(T)])
    fmax = -r.fun if r.success else np.nan
    fmin = np.inf
    for mask in itertools.product([0, 1], repeat=T):
        m = np.array(mask, bool); l2 = lo.copy(); h2 = hi.copy()
        l2[m] = np.maximum(l2[m], cc[m]); h2[~m] = np.minimum(h2[~m], cc[~m])
        if (l2 > h2 + 1e-12).any(): continue
        rr = solve(np.where(m, 0.0, w / cc), np.ones((1, T)), [Hlo], [Hhi], l2, h2, np.ones(T))
        if rr.success: fmin = min(fmin, rr.fun + w[m].sum())
    return fmin, fmax

def schemes(T, labels):
    out = {'uniform': np.full(T, 1 / T)}
    for i, t in enumerate(labels):
        v = np.ones(T); v[i] = 2; out[f'{t} x2'] = v / v.sum()
        v = np.ones(T); v[i] = 0.5; out[f'{t} x0.5'] = v / v.sum()
    return out

def random_weights(T, n, rng):
    ws = []
    while len(ws) < n:
        w = rng.dirichlet(np.ones(T))
        if (w >= 0.5 / T).all() and (w <= 2 / T).all(): ws.append(w)
    return np.array(ws)

rng = np.random.default_rng(20260923)
# ---------------- client ledgers
T1 = cd.t1(); T2 = cd.t2()
D = T2.merge(T1[['ledger', 'detector', 'P@100_mean']], on=['ledger', 'detector'])
rows = []
for L in cd.LEDGERS:
    sub = D[D.ledger == L]; n = cd.type_counts(cd.N_ANOM[L])
    for name, w in schemes(5, cd.TYPES).items():
        for _, r in sub.iterrows():
            R = np.array([r[f'R_{t}'] for t in cd.TYPES]); lo, hi = fsr_bounds_w(R, n, r['P@100_mean'], w)
            rows.append(dict(ledger=L, scheme=name, detector=r.detector, P100=r['P@100_mean'], FSR_lo=lo, FSR_hi=hi))
C = pd.DataFrame(rows); C.to_csv(paths.out_client('a10_client_weight_bounds.csv'), index=False)
summ = []
for (L, sch), g in C.groupby(['ledger', 'scheme'], sort=False):
    pwin = g.loc[g.P100 == g.P100.max(), 'detector'].tolist()
    # FSR leader determined by bounds: a detector whose lower bound exceeds every other upper bound
    best = [d for d, lo in zip(g.detector, g.FSR_lo) if all(lo > h + 1e-12 for d2, h in zip(g.detector, g.FSR_hi) if d2 != d)]
    # does every P@100 leader lie strictly below some other detector (reversal fixed by the bounds)?
    rev_fixed = all(any(g2lo > g.loc[g.detector == pw, 'FSR_hi'].item() + 1e-12 for g2lo, d2 in zip(g.FSR_lo, g.detector) if d2 != pw) for pw in pwin)
    summ.append(dict(ledger=L, scheme=sch, P100_leader=', '.join(pwin), FSR_leader_by_bounds=', '.join(best) or 'undetermined',
                     reversal_fixed_by_bounds=rev_fixed))
CS = pd.DataFrame(summ); CS.to_csv(paths.out_client('a10_client_weight_summary.csv'), index=False)
pd.set_option('display.width', 200); print(CS.to_string(index=False))
# random weights on client ledgers, point estimate = FSR of the rounded recalls (midpoint reading)
rw = random_weights(5, 2000, rng); rr = []
for L in cd.LEDGERS:
    sub = D[D.ledger == L]; n = cd.type_counts(cd.N_ANOM[L]); c = caps(n, K)
    pw = sub.loc[sub['P@100_mean'] == sub['P@100_mean'].max(), 'detector'].tolist()
    Rm = np.array([[r[f'R_{t}'] for t in cd.TYPES] for _, r in sub.iterrows()]); dets = sub.detector.tolist()
    f = np.minimum(Rm * n, c) / c                       # detectors x types
    F = rw @ f.T                                        # draws x detectors
    lead = np.array(dets)[F.argmax(1)]
    rr.append(dict(ledger=L, P100_leader=', '.join(pw), share_draws_FSR_leader_differs=float(np.mean([l not in pw for l in lead])),
                   FSR_leaders=pd.Series(lead).value_counts(normalize=True).round(3).to_dict()))
RR = pd.DataFrame(rr); RR.to_csv(paths.out_client('a10_client_random_weights.csv'), index=False); print(RR.to_string(index=False))
