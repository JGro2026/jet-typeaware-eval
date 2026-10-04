"""A5: public ledger robustness.
Entry bootstrap: B = 1000 resamples of the 102,429 journal entries with replacement,
numpy.random.default_rng(20260923), idx = rng.integers(0, N, N); the same resamples for every detector, so
paired differences are valid. Seed-0 scores for the stochastic detectors. In a resample an entry drawn c times
occupies c consecutive places at its original rank (ties keep the original entry order, as the stable sort);
P@k, per-marking found counts, the marking sizes n_t and hence the FSR caps are recomputed on the resample.
Metrics: P@100, P@1000, FSR@100, FSR@1000, TC@1000. Percentile 95 percent intervals.
Paired differences kNN minus each other detector for P@1000 and FSR@1000.
Seed range for the stochastic detectors (IF, AE, VAE, AE-emb; LightGBM ceiling as reference) from the exact
per-seed values.
Outputs: a5_bootstrap_ci.csv, a5_bootstrap_paired_kNN.csv, a5_seed_range.csv, a5_bootstrap_draws.npz."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.dirname(HERE)
sys.path.insert(0, str(paths.PIPELINE)); sys.path.insert(0, str(paths.TYPEAWARE))
from dib_data import load, entries, entry_scores
from fsr import fsr

SC = str(paths.SCORES)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE', 'LGBM-ceiling': 'LightGBM ceiling'}
B = 1000; SEED = 20260923; KS = (100, 1000); LPOS = 6000
COLS = ['P@100', 'P@1000', 'FSR@100', 'FSR@1000', 'TC@1000']
df = load(); e = entries(df); mk = e.marking.values.astype(np.int64); eid = df.eid.values; N = len(mk)

pre = {}
for d in DETS:
    sc = entry_scores(np.load(f'{SC}/{d}_s0.npy'), eid); o = np.argsort(-sc, kind='stable')
    pre[d] = (o[:LPOS], mk[o[:LPOS]])


def rep(d, c, nt):
    o, mo = pre[d]; co = c[o]; cum = np.cumsum(co)
    out = {}
    for k in KS:
        i = int(np.searchsorted(cum, k, side='left'))
        if i >= len(cum): raise RuntimeError('increase LPOS')
        before = cum[i - 1] if i > 0 else 0.0
        f = np.bincount(mo[:i], weights=co[:i], minlength=9).astype(float)
        f[mo[i]] += k - before
        found = f[1:]
        out[f'P@{k}'] = found.sum() / k
        out[f'FSR@{k}'] = fsr(found, nt, k)
        if k == 1000: out['TC@1000'] = int((found > 0).sum())
    return out


# point estimates with unit counts must equal the exact seed-0 values
ones = np.ones(N); nt1 = np.bincount(mk, minlength=9)[1:].astype(float)
point = {d: rep(d, ones, nt1) for d in DETS}
PS = pd.read_csv(paths.find('dib_typeaware_perseed.csv'))
for d in DETS:
    r = PS[(PS.detector == NAME.get(d, d)) & (PS.seed == 0)].set_index('k')
    ref = {'P@100': r.loc[100, 'P'], 'P@1000': r.loc[1000, 'P'], 'FSR@100': r.loc[100, 'FSR'], 'FSR@1000': r.loc[1000, 'FSR'],
           'TC@1000': r.loc[1000, 'coverage']}
    for c_ in COLS:
        assert abs(point[d][c_] - ref[c_]) < 1e-12, (d, c_, point[d][c_], ref[c_])
print('point estimates reproduce dib_typeaware_perseed.csv (seed 0)', flush=True)

draws = {d: np.empty((B, len(COLS))) for d in DETS}
rng = np.random.default_rng(SEED); t0 = time.time()
for b in range(B):
    c = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
    nt = np.bincount(mk, weights=c, minlength=9)[1:]
    for d in DETS:
        m = rep(d, c, nt); draws[d][b] = [m[x] for x in COLS]
    if (b + 1) % 250 == 0: print(f'{b + 1}/{B} {time.time() - t0:.0f}s', flush=True)

rows = []
for d in DETS:
    for j, c_ in enumerate(COLS):
        v = draws[d][:, j]
        rows.append(dict(detector=NAME.get(d, d), metric=c_, point_seed0=point[d][c_], boot_mean=v.mean(), boot_sd=v.std(ddof=1),
                         ci_lo_2_5=np.percentile(v, 2.5), ci_hi_97_5=np.percentile(v, 97.5)))
CI = pd.DataFrame(rows); CI.to_csv(paths.out_public('a5_bootstrap_ci.csv'), index=False)

rows = []
for d in DETS:
    if d == 'kNN': continue
    for c_ in ['P@1000', 'FSR@1000', 'P@100', 'FSR@100']:   # the first two are the requested ones
        j = COLS.index(c_); v = draws['kNN'][:, j] - draws[d][:, j]
        lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
        rows.append(dict(comparison=f'kNN minus {NAME.get(d, d)}', metric=c_, point_difference=point['kNN'][c_] - point[d][c_],
                         boot_mean_difference=v.mean(), ci_lo_2_5=lo, ci_hi_97_5=hi, share_kNN_higher=float((v > 0).mean()),
                         share_tied=float((v == 0).mean()), excludes_zero=bool(lo > 0 or hi < 0)))
PD = pd.DataFrame(rows); PD.to_csv(paths.out_public('a5_bootstrap_paired_kNN.csv'), index=False)
np.savez_compressed(paths.out_public('a5_bootstrap_draws.npz'), cols=np.array(COLS), **{NAME.get(d, d): draws[d] for d in DETS})

# seed range
rows = []
for det in ['IF', 'AE', 'VAE', 'AE-emb', 'LightGBM ceiling']:
    r = PS[PS.detector == det]
    for k, col, lab in [(100, 'P', 'P@100'), (1000, 'P', 'P@1000'), (100, 'FSR', 'FSR@100'), (1000, 'FSR', 'FSR@1000'),
                        (1000, 'coverage', 'TC@1000'), (100, 'coverage', 'TC@100')]:
        v = r[r.k == k].sort_values('seed')[col].values
        rows.append(dict(detector=det, metric=lab, s0=v[0], s1=v[1], s2=v[2], mean=v.mean(), sd=v.std(ddof=1), min=v.min(), max=v.max(),
                         range=v.max() - v.min(), reference_only=det == 'LightGBM ceiling'))
SR = pd.DataFrame(rows); SR.to_csv(paths.out_public('a5_seed_range.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(CI.round(4).to_string(index=False)); print(PD.round(4).to_string(index=False)); print(SR.round(4).to_string(index=False))
print('done', round(time.time() - t0, 1), 's')
