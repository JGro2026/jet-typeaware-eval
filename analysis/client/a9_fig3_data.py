"""Tidy data for the hits-versus-FSR figure: P@k and FSR@k (with bounds) and TC@k per detector for the four real
ledgers (k = 100; FSR point = midpoint of the lattice bounds, interval = lattice bounds, see a2_bounds.py) and
the public ledger (k = 100 and k = 1,000; exact, mean over seeds for stochastic detectors, seed min and max).
Output: a9_fig3_hits_vs_fsr.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.dirname(HERE)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE', 'VAE', 'AE-emb']
F = pd.read_csv(paths.find('datev_fsr_k100.csv'))
rows = []
for r in F.itertuples():
    rows.append(dict(ledger=r.ledger, k=100, detector=r.detector, P=r.P100, P_lo=r.P100 - 0.005, P_hi=r.P100 + 0.005,
                     FSR=r.FSR, FSR_lo=r.FSR_lo_lattice, FSR_hi=r.FSR_hi_lattice, TC=r.TC_lo if r.TC_lo == r.TC_hi else np.nan,
                     TC_lo=r.TC_lo, TC_hi=r.TC_hi, exact=bool(r.ledger in 'CA' and r.FSR_hi_lattice - r.FSR_lo_lattice < 0.002),
                     basis='run B seed means (5 seeds), two-decimal rounding bounds'))
PS = pd.read_csv(paths.find('dib_typeaware_perseed.csv'))
for k in (100, 1000):
    for d in DETS:
        v = PS[(PS.detector == d) & (PS.k == k)]
        rows.append(dict(ledger='public', k=k, detector=d, P=v.P.mean(), P_lo=v.P.min(), P_hi=v.P.max(), FSR=v.FSR.mean(),
                         FSR_lo=v.FSR.min(), FSR_hi=v.FSR.max(), TC=v.coverage.mean(), TC_lo=v.coverage.min(), TC_hi=v.coverage.max(),
                         exact=True, basis=f'exact; mean over {len(v)} seed(s); lo/hi = seed min/max'))
pd.DataFrame(rows).to_csv(paths.out_client('a9_fig3_hits_vs_fsr.csv'), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
