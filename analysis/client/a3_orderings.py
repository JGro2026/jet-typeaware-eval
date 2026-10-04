"""A3: hit-based versus type-aware orderings of the nine unsupervised detectors (LightGBM ceiling excluded).

Real ledgers (k = 100): P@100 as printed in run B T1 (seed mean, two decimals; equal printed values are ties;
different printed values are strictly ordered because P@100 is a multiple of 1/500 and the rounding boundaries
are not). FSR@100 point = midpoint of the lattice bounds (a2_bounds.py). TC@100 = types hit in at least one of
the five seeds (exact on C and A; interval on B and D).
For every ledger: Kendall tau-b (scipy) between P@100 and FSR@100, and between P@100 and TC@100, at the point
values, plus the range of tau-b over every weak order of the values that is consistent with the bounds
(exact enumeration: relations of undetermined pairs in {<, =, >}, realisability by LP; integer TC vectors).
Public ledger (k = 100 and k = 1,000): exact values from ../dib_typeaware_mean.csv (stochastic detectors
averaged over seeds 0 to 2); sensitivity with the seed-0 values from ../dib_typeaware_perseed.csv.
A flip is a pair strictly ordered one way by P@k and strictly the other way by the type-aware criterion.
Outputs: a3_tau.csv, a3_best.csv, a3_flips.csv, a3_table5.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.dirname(HERE)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE', 'VAE', 'AE-emb']
EPS = 1e-7


def tau(x, y):
    return float(kendalltau(x, y).statistic)


def tau_range_intervals(x, lo, hi):
    """min and max tau-b(x, v) over weak orders of v realisable with v_i in [lo_i, hi_i]."""
    n = len(x); U = [(i, j) for i, j in itertools.combinations(range(n), 2)
                     if not (lo[i] > hi[j] + 1e-12 or lo[j] > hi[i] + 1e-12)]
    vals = []
    for rel in itertools.product([-1, 0, 1], repeat=len(U)):
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for (i, j), r in zip(U, rel):
            row = np.zeros(n)
            if r == 0:
                row[i], row[j] = 1, -1; A_eq.append(row); b_eq.append(0)
            else:  # r = 1: v_i - v_j >= EPS  ->  v_j - v_i <= -EPS
                row[j], row[i] = r, -r; A_ub.append(row); b_ub.append(-EPS)
        res = linprog(np.zeros(n), A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub if b_ub else None,
                      A_eq=np.array(A_eq) if A_eq else None, b_eq=b_eq if b_eq else None,
                      bounds=list(zip(lo, hi)), method='highs')
        if res.status == 0:
            v = res.x.copy()
            for (i, j), r in zip(U, rel):   # make intended ties exact
                if r == 0: v[j] = v[i]
            vals.append(tau(x, v))
    return min(vals), max(vals), len(U)


def tau_range_integer(x, lo, hi):
    lo = np.asarray(lo, int); hi = np.asarray(hi, int); n = len(x)
    grids = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing='ij'), -1).reshape(-1, len(lo)).astype(np.int8)
    x = np.asarray(x, float); I, J = np.triu_indices(n, 1)
    sx = np.sign(x[I] - x[J])
    sy = np.sign(grids[:, I] - grids[:, J]).astype(np.int8)
    num = (sx * sy).sum(1)
    n0 = len(I); n1 = (sx == 0).sum(); n2 = (sy == 0).sum(1)
    den = np.sqrt((n0 - n1) * (n0 - n2).astype(float))
    t = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return float(np.nanmin(t)), float(np.nanmax(t)), len(grids)


def order_str(vals, names):
    s = sorted(zip(vals, names), key=lambda z: -z[0])
    out, i = [], 0
    while i < len(s):
        grp = [s[i][1]]; j = i + 1
        while j < len(s) and abs(s[j][0] - s[i][0]) < 1e-12: grp.append(s[j][1]); j += 1
        out.append('='.join(grp)); i = j
    return ' > '.join(out)


def best(vals, names):
    m = max(vals); return '='.join(n for v, n in zip(vals, names) if abs(v - m) < 1e-12), m


tau_rows, best_rows, flip_rows, t5 = [], [], [], []
# ---------------- real ledgers ----------------
F = pd.read_csv(paths.find('datev_fsr_k100.csv'))
F['detector'] = pd.Categorical(F.detector, DETS, ordered=True)
for L in ['C', 'A', 'B', 'D']:
    f = F[F.ledger == L].sort_values('detector')
    names = list(f.detector.astype(str)); P = f.P100.values; FS = f.FSR.values
    flo, fhi = f.FSR_lo_lattice.values, f.FSR_hi_lattice.values
    clo, chi = f.FSR_lo.values, f.FSR_hi.values
    tlo, thi = f.TC_lo.values, f.TC_hi.values
    t_pf = tau(P, FS); a1, b1, u1 = tau_range_intervals(P, flo, fhi); a2, b2, u2 = tau_range_intervals(P, clo, chi)
    tc_exact = bool((tlo == thi).all())
    if tc_exact:
        t_pt = tau(P, tlo); c1 = d1 = t_pt; ng = 1
    else:
        t_pt = tau(P, tlo); c1, d1, ng = tau_range_integer(P, tlo, thi)
    tau_rows += [dict(ledger=L, k=100, criterion='FSR', tau_b_point=t_pf, tau_b_min_lattice=a1, tau_b_max_lattice=b1,
                      undetermined_pairs_lattice=u1, tau_b_min_cont=a2, tau_b_max_cont=b2, undetermined_pairs_cont=u2,
                      point_basis='FSR midpoint of lattice bounds'),
                 dict(ledger=L, k=100, criterion='TC', tau_b_point=t_pt, tau_b_min_lattice=c1, tau_b_max_lattice=d1,
                      undetermined_pairs_lattice=np.nan, tau_b_min_cont=np.nan, tau_b_max_cont=np.nan, undetermined_pairs_cont=np.nan,
                      point_basis='TC exact' if tc_exact else f'TC lower bound (TC_lo); range over {ng} integer TC vectors')]
    bp, vp = best(P, names); bf, vf = best(FS, names)
    # is the FSR winner determined? its lattice lower bound exceeds every other upper bound
    i = int(np.argmax(FS)); det_best = all(flo[i] > fhi[j] + 1e-12 for j in range(len(names)) if j != i)
    bt, vt = best(tlo if tc_exact else tlo, names)
    best_rows.append(dict(ledger=L, k=100, best_by_P=bp, P_best=vp, best_by_FSR=bf, FSR_best=vf, FSR_best_determined=det_best,
                          best_by_TC=bt if tc_exact else 'undetermined', TC_best=vt if tc_exact else np.nan,
                          order_P=order_str(P, names), order_FSR=order_str(FS, names),
                          order_TC=order_str(tlo, names) if tc_exact else 'intervals: ' + ', '.join(f'{n} [{a},{b}]' for n, a, b in zip(names, tlo, thi))))
    for i, j in itertools.combinations(range(len(names)), 2):
        sp = np.sign(P[i] - P[j])
        for crit, v, lo_, hi_ in [('FSR', FS, flo, fhi), ('TC', tlo, tlo, thi)]:
            sv = np.sign(v[i] - v[j])
            if sp != 0 and sv != 0 and sp != sv:
                w, l = (i, j) if sp > 0 else (j, i)          # w wins on P
                determined = lo_[l] > hi_[w] + (1e-12 if crit == 'FSR' else 0)
                det_c = (clo[l] > chi[w] + 1e-12) if crit == 'FSR' else determined
                flip_rows.append(dict(ledger=L, k=100, criterion=crit, higher_by_P=names[w], P_higher=P[w],
                                      higher_by_criterion=names[l], P_lower=P[l], crit_of_P_winner=v[w], crit_of_criterion_winner=v[l],
                                      determined_by_bounds=bool(determined), determined_by_cont_bounds=bool(det_c)))
    ib = int(np.argmax(P)) if (P == P.max()).sum() == 1 else [k for k in range(len(names)) if P[k] == P.max()]
    idxs = ib if isinstance(ib, list) else [ib]
    t5.append(dict(ledger=L, k=100, best_by_hits=bp, P_at_k=vp,
                   its_TC=('/'.join(str(int(tlo[k])) if tlo[k] == thi[k] else f'{int(tlo[k])}-{int(thi[k])}' for k in idxs[:1])),
                   its_FSR=FS[idxs[0]], its_FSR_lo=flo[idxs[0]], its_FSR_hi=fhi[idxs[0]],
                   best_by_FSR=bf, FSR_of_best=vf, FSR_best_lo=flo[int(np.argmax(FS))], FSR_best_hi=fhi[int(np.argmax(FS))],
                   P_of_FSR_best=P[int(np.argmax(FS))], tau_P_FSR=t_pf, tau_P_FSR_min=a1, tau_P_FSR_max=b1))

# ---------------- public ledger ----------------
M = pd.read_csv(paths.find('dib_typeaware_mean.csv'))
PS = pd.read_csv(paths.find('dib_typeaware_perseed.csv'))
for variant, src in [('mean over seeds', M), ('seed 0', PS[PS.seed == 0])]:
    for k in [100, 1000]:
        d = src[(src.k == k) & (src.detector.isin(DETS))].set_index('detector').loc[DETS]
        names = DETS; P = d.P.values; FS = d.FSR.values; TC = d.coverage.values
        for crit, v in [('FSR', FS), ('TC', TC)]:
            tau_rows.append(dict(ledger=f'public ({variant})', k=k, criterion=crit, tau_b_point=tau(P, v),
                                 tau_b_min_lattice=np.nan, tau_b_max_lattice=np.nan, undetermined_pairs_lattice=0,
                                 tau_b_min_cont=np.nan, tau_b_max_cont=np.nan, undetermined_pairs_cont=0, point_basis='exact'))
        bp, vp = best(P, names); bf, vf = best(FS, names); bt, vt = best(TC, names)
        best_rows.append(dict(ledger=f'public ({variant})', k=k, best_by_P=bp, P_best=vp, best_by_FSR=bf, FSR_best=vf,
                              FSR_best_determined=True, best_by_TC=bt, TC_best=vt, order_P=order_str(P, names),
                              order_FSR=order_str(FS, names), order_TC=order_str(TC, names)))
        for i, j in itertools.combinations(range(len(names)), 2):
            sp = np.sign(P[i] - P[j])
            for crit, v in [('FSR', FS), ('TC', TC)]:
                sv = np.sign(v[i] - v[j])
                if sp != 0 and sv != 0 and sp != sv:
                    w, l = (i, j) if sp > 0 else (j, i)
                    flip_rows.append(dict(ledger=f'public ({variant})', k=k, criterion=crit, higher_by_P=names[w], P_higher=P[w],
                                          higher_by_criterion=names[l], P_lower=P[l], crit_of_P_winner=v[w],
                                          crit_of_criterion_winner=v[l], determined_by_bounds=True, determined_by_cont_bounds=True))
        if variant == 'mean over seeds':
            ib = int(np.argmax(P)); ibf = int(np.argmax(FS))
            t5.append(dict(ledger='public', k=k, best_by_hits=bp, P_at_k=vp, its_TC=str(TC[ib]), its_FSR=FS[ib], its_FSR_lo=FS[ib],
                           its_FSR_hi=FS[ib], best_by_FSR=bf, FSR_of_best=vf, FSR_best_lo=vf, FSR_best_hi=vf, P_of_FSR_best=P[ibf],
                           tau_P_FSR=tau(P, FS), tau_P_FSR_min=tau(P, FS), tau_P_FSR_max=tau(P, FS)))

T = pd.DataFrame(tau_rows); B = pd.DataFrame(best_rows); FL = pd.DataFrame(flip_rows); T5 = pd.DataFrame(t5)
T.to_csv(paths.out_client('a3_tau.csv'), index=False); B.to_csv(paths.out_client('a3_best.csv'), index=False)
FL.to_csv(paths.out_client('a3_flips.csv'), index=False); T5.to_csv(paths.out_client('a3_table5.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 200)
print(T.round(3).to_string(index=False)); print(B.to_string(index=False)); print(T5.round(3).to_string(index=False))
print(FL.groupby(['ledger', 'k', 'criterion']).size())
