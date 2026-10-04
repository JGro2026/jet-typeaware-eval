"""A2: exact bounds on FSR@100 and type coverage TC@100 on the real ledgers, given that run B prints
per-type recalls at 100 (seed means over 5 seeds) and P@100 (seed mean) to two decimals.

Unknowns: seed-mean found counts f_t of each type in the top 100.
Continuous program (as specified):
    max(0, R_t - 0.005) n_t <= f_t <= min(n_t, (R_t + 0.005) n_t)
    100 (P - 0.005) <= sum_t f_t <= 100 (P + 0.005),  f_t >= 0.
Lattice program (uses the design): 5 seeds, n_t identical in every seed, types disjoint, so
F_t = 5 f_t is an integer and sum_t F_t = 500 P@100 is an integer; same interval constraints.
FSR@100 = (1/5) sum_t min(f_t, c_t)/c_t with water-filling caps c_t (fsr.caps). FSR is computed on the
seed-averaged type profile (per-seed values are not available); for ledgers C and A the caps equal n_t,
FSR is linear and this equals the seed mean of per-seed FSR.
Max FSR: LP/MILP with g_t <= f_t, g_t <= c_t. Min FSR (concave objective): minimum over the 2^5 regions
{f_t >= c_t or f_t <= c_t}, linear inside each region.
TC@100 in two readings. (a) Union over seeds: types with f_t > 0, i.e. hit in at least one seed: lattice MILP with binaries z_t
(min: F_t <= UB_t z_t; max: F_t >= z_t). The continuous variant uses the same threshold of one hit in one
seed (f_t >= 1/5) for positivity. (b) Mean per-seed TC (the reading used on the public ledger): bounds from
tcseed_bounds(); the union reading overstates coverage for detectors with few hits.
Outputs: a2_fsr_tc_bounds.csv, a2_found_bounds.csv, a2_pairwise_determinacy.csv, and the refreshed
../datev_fsr_k100.csv (same columns as before plus new ones; FSR_lo/FSR_hi now tight)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, itertools
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, str(paths.TYPEAWARE)); sys.path.insert(0, HERE)
from fsr import caps, fsr
import common_datev as cd

S = cd.SEEDS; K = 100; E = 0.005


def boxes(R, n, P, lattice):
    lo = np.maximum(0, R - E) * n; hi = np.minimum(n, (R + E) * n)
    Hlo, Hhi = K * (P - E), K * (P + E)
    if lattice:  # in units of F = S f
        return np.ceil(S * lo - 1e-9), np.floor(S * hi + 1e-9), np.ceil(S * Hlo - 1e-9), np.floor(S * Hhi + 1e-9)
    return lo, hi, Hlo, Hhi


def solve(c_obj, A, lb_A, ub_A, lb, ub, integ):
    res = milp(c=c_obj, constraints=[LinearConstraint(A, lb_A, ub_A)], bounds=Bounds(lb, ub), integrality=integ,
               options={'mip_rel_gap': 0, 'presolve': True})
    return res


def fsr_bounds(R, n, P, lattice):
    T = len(n); c = caps(n, K); lo, hi, Hlo, Hhi = boxes(R, n, P, lattice)
    scale = S if lattice else 1.0           # variable x = scale * f
    cc = c * scale
    if Hlo > Hhi or lo.sum() > Hhi or hi.sum() < Hlo:
        return np.nan, np.nan, False
    # max: variables [x(T), g(T)], maximise sum g/(T cc)
    cobj = np.r_[np.zeros(T), -1.0 / (T * cc)]
    A = np.zeros((T + 1, 2 * T)); A[:T, :T] = -np.eye(T); A[:T, T:] = np.eye(T); A[T, :T] = 1
    lbA = np.r_[np.full(T, -np.inf), Hlo]; ubA = np.r_[np.zeros(T), Hhi]
    integ = np.r_[np.full(T, 1 if lattice else 0), np.zeros(T)]
    r = solve(cobj, A, lbA, ubA, np.r_[lo, np.zeros(T)], np.r_[hi, cc], integ)
    fmax = -r.fun if r.success else np.nan
    # min over regions
    fmin = np.inf
    for mask in itertools.product([0, 1], repeat=T):
        m = np.array(mask, bool)
        l2 = lo.copy(); h2 = hi.copy()
        l2[m] = np.maximum(l2[m], cc[m]); h2[~m] = np.minimum(h2[~m], cc[~m])
        if (l2 > h2 + 1e-12).any(): continue
        cobj = np.where(m, 0.0, 1.0 / (T * cc))
        A1 = np.ones((1, T))
        r = solve(cobj, A1, [Hlo], [Hhi], l2, h2, np.full(T, 1 if lattice else 0))
        if r.success: fmin = min(fmin, r.fun + m.sum() / T)
    return fmin, fmax, True


def tc_bounds(R, n, P, lattice):
    T = len(n); lo, hi, Hlo, Hhi = boxes(R, n, P, lattice)
    one = 1.0 if lattice else 1.0 / S       # positivity threshold: one hit in one seed
    integ = np.r_[np.full(T, 1 if lattice else 0), np.ones(T)]
    # min coverage: x_t <= hi_t z_t, minimise sum z
    A = np.zeros((T + 1, 2 * T)); A[:T, :T] = np.eye(T); A[:T, T:] = -np.diag(hi); A[T, :T] = 1
    r = solve(np.r_[np.zeros(T), np.ones(T)], A, np.r_[np.full(T, -np.inf), Hlo], np.r_[np.zeros(T), Hhi],
              np.r_[lo, np.zeros(T)], np.r_[hi, np.ones(T)], integ)
    tmin = r.fun if r.success else np.nan
    # max coverage: x_t >= one * z_t
    A = np.zeros((T + 1, 2 * T)); A[:T, :T] = np.eye(T); A[:T, T:] = -one * np.eye(T); A[T, :T] = 1
    r = solve(np.r_[np.zeros(T), -np.ones(T)], A, np.r_[np.zeros(T), Hlo], np.r_[np.full(T, np.inf), Hhi],
              np.r_[lo, np.zeros(T)], np.r_[hi, np.ones(T)], integ)
    tmax = -r.fun if r.success else np.nan
    return tmin, tmax


def found_bounds(R, n, P, lattice):
    T = len(n); lo, hi, Hlo, Hhi = boxes(R, n, P, lattice); scale = S if lattice else 1.0
    out = []
    for t in range(T):
        e = np.zeros(T); e[t] = 1
        integ = np.full(T, 1 if lattice else 0)
        r1 = solve(e, np.ones((1, T)), [Hlo], [Hhi], lo, hi, integ)
        r2 = solve(-e, np.ones((1, T)), [Hlo], [Hhi], lo, hi, integ)
        out.append((r1.fun / scale + 0.0 if r1.success else np.nan, -r2.fun / scale + 0.0 if r2.success else np.nan))
    return out


def tcseed_bounds(R, n, P):
    """Mean over the 5 seeds of the per-seed type coverage. With F_t the total hits of type t over the seeds and
    h_t the number of seeds in which type t is hit: ceil(F_t / n_t) <= h_t <= min(5, F_t) (at most n_t hits of a
    type per seed). Mean per-seed TC = sum_t h_t / 5. Lattice MILP over (F, h)."""
    T = len(n); lo, hi, Hlo, Hhi = boxes(R, n, P, True)
    integ = np.ones(2 * T)
    A = np.zeros((2 * T + 1, 2 * T))
    A[:T, :T] = -np.eye(T); A[:T, T:] = np.eye(T)                 # h - F <= 0
    A[T:2 * T, :T] = -np.eye(T); A[T:2 * T, T:] = np.diag(n)      # n h - F >= 0
    A[2 * T, :T] = 1
    lbA = np.r_[np.full(T, -np.inf), np.zeros(T), Hlo]; ubA = np.r_[np.zeros(T), np.full(T, np.inf), Hhi]
    bl = np.r_[lo, np.zeros(T)]; bu = np.r_[hi, np.full(T, S)]
    r1 = solve(np.r_[np.zeros(T), np.ones(T)], A, lbA, ubA, bl, bu, integ)
    r2 = solve(np.r_[np.zeros(T), -np.ones(T)], A, lbA, ubA, bl, bu, integ)
    return (r1.fun / S if r1.success else np.nan), (-r2.fun / S if r2.success else np.nan)


T1 = cd.t1(); T2 = cd.t2()
D = T2.merge(T1[['ledger', 'detector', 'P@100_mean']], on=['ledger', 'detector'])
rows, frows = [], []
for _, r in D.iterrows():
    L = r.ledger; n = cd.type_counts(cd.N_ANOM[L]); R = np.array([r[f'R_{t}'] for t in cd.TYPES]); P = r['P@100_mean']
    c = caps(n, K)
    point = fsr(R * n, n, K)
    tc_point = int((R > 0).sum())
    lo_c, hi_c, feas_c = fsr_bounds(R, n, P, False)
    lo_i, hi_i, feas_i = fsr_bounds(R, n, P, True)
    tlo_c, thi_c = tc_bounds(R, n, P, False)
    tlo_i, thi_i = tc_bounds(R, n, P, True)
    tslo, tshi = tcseed_bounds(R, n, P)
    rows.append(dict(ledger=L, detector=r.detector, P100=P, P100_lo=P - E, P100_hi=P + E,
                     **{f'R_{t}': R[i] for i, t in enumerate(cd.TYPES)},
                     FSR_point_rounded=point, FSR_lo_cont=lo_c, FSR_hi_cont=hi_c, FSR_lo_lattice=lo_i, FSR_hi_lattice=hi_i,
                     point_inside_cont=bool(lo_c - 1e-12 <= point <= hi_c + 1e-12),
                     TC_point_rounded=tc_point, TC_lo_cont=tlo_c, TC_hi_cont=thi_c, TC_lo_lattice=tlo_i, TC_hi_lattice=thi_i,
                     TCseed_lo=tslo, TCseed_hi=tshi,
                     feasible_cont=feas_c, feasible_lattice=feas_i))
    for lat in (False, True):
        fb = found_bounds(R, n, P, lat)
        for i, t in enumerate(cd.TYPES):
            frows.append(dict(ledger=L, detector=r.detector, program='lattice' if lat else 'continuous', type=t,
                              n_t=n[i], cap_t=c[i], R_t_printed=R[i], found_lo=fb[i][0], found_hi=fb[i][1]))
B = pd.DataFrame(rows); F = pd.DataFrame(frows)
B.to_csv(paths.out_client('a2_fsr_tc_bounds.csv'), index=False)
F.to_csv(paths.out_client('a2_found_bounds.csv'), index=False)

# pairwise determinacy of FSR and TC orders (bounds of different detectors are independent)
prs = []
for L in cd.LEDGERS:
    b = B[B.ledger == L].set_index('detector')
    for a, bb in itertools.combinations(b.index, 2):
        x, y = b.loc[a], b.loc[bb]
        for prog in ('cont', 'lattice'):
            if x[f'FSR_lo_{prog}'] > y[f'FSR_hi_{prog}'] + 1e-12: o = f'{a} > {bb}'
            elif y[f'FSR_lo_{prog}'] > x[f'FSR_hi_{prog}'] + 1e-12: o = f'{bb} > {a}'
            else: o = 'undetermined'
            if x[f'TC_lo_lattice'] > y[f'TC_hi_lattice']: ot = f'{a} > {bb}'
            elif y[f'TC_lo_lattice'] > x[f'TC_hi_lattice']: ot = f'{bb} > {a}'
            elif x.TC_lo_lattice == x.TC_hi_lattice == y.TC_lo_lattice == y.TC_hi_lattice: ot = 'equal'
            else: ot = 'undetermined'
            prs.append(dict(ledger=L, det_a=a, det_b=bb, program=prog, P100_a=x.P100, P100_b=y.P100,
                            P100_order=(f'{a} > {bb}' if x.P100 > y.P100 else f'{bb} > {a}' if y.P100 > x.P100 else 'tie (printed)'),
                            FSR_order=o, TC_order_lattice=ot))
PR = pd.DataFrame(prs); PR.to_csv(paths.out_client('a2_pairwise_determinacy.csv'), index=False)

# refreshed datev_fsr_k100.csv (leading columns as before; values now consistent with P@100)
# FSR point = midpoint of the lattice bounds (exact on C; width at most 0.0013 on A, 0.008 on B, 0.06 on D).
# FSR_lo/FSR_hi = continuous bounds (conservative), *_lattice = tight bounds. coverage = TC_lo (guaranteed
# minimum); equals the rounded-recall count on C, A and B, and TC is undetermined on D (see TC_lo, TC_hi).
out = B[['ledger', 'detector', 'P100']].copy()
out['coverage'] = B.TC_lo_lattice.astype(int)
out['FSR'] = (B.FSR_lo_lattice + B.FSR_hi_lattice) / 2
out['FSR_lo'] = B.FSR_lo_cont; out['FSR_hi'] = B.FSR_hi_cont
out['FSR_lo_lattice'] = B.FSR_lo_lattice; out['FSR_hi_lattice'] = B.FSR_hi_lattice
out['TC_lo'] = B.TC_lo_lattice.astype(int); out['TC_hi'] = B.TC_hi_lattice.astype(int)
out['TCseed_lo'] = B.TCseed_lo; out['TCseed_hi'] = B.TCseed_hi
out['FSR_from_rounded_recalls'] = B.FSR_point_rounded; out['coverage_from_rounded_recalls'] = B.TC_point_rounded
out.to_csv(paths.out_client('datev_fsr_k100.csv'), index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(B[['ledger', 'detector', 'P100', 'FSR_point_rounded', 'FSR_lo_cont', 'FSR_hi_cont', 'FSR_lo_lattice', 'FSR_hi_lattice',
         'point_inside_cont', 'TC_point_rounded', 'TC_lo_lattice', 'TC_hi_lattice', 'TCseed_lo', 'TCseed_hi', 'feasible_lattice']].round(4).to_string(index=False))
print('max width cont by ledger:', (B.FSR_hi_cont - B.FSR_lo_cont).groupby(B.ledger).max().round(4).to_dict())
print('max width lattice by ledger:', (B.FSR_hi_lattice - B.FSR_lo_lattice).groupby(B.ledger).max().round(4).to_dict())
