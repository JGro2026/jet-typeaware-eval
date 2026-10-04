"""A10b: random weight vectors (uniform on the simplex, each weight within [0.5, 2] x 1/T) with exact lattice bounds on
the client ledgers (300 draws), and exact per-seed values on the public ledger (2,000 draws). Existing results only."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, str(paths.TYPEAWARE)); sys.path.insert(0, HERE)
import builtins, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from a10_weights import fsr_bounds_w, random_weights, schemes
import common_datev as cd
from fsr import caps
rng = np.random.default_rng(7)
T1 = cd.t1(); T2 = cd.t2()
D = T2.merge(T1[['ledger', 'detector', 'P@100_mean']], on=['ledger', 'detector'])
W = random_weights(5, 300, rng); t0 = time.time(); out = []
for L in cd.LEDGERS:
    sub = D[D.ledger == L].reset_index(drop=True); n = cd.type_counts(cd.N_ANOM[L])
    pw = sub.loc[sub['P@100_mean'] == sub['P@100_mean'].max(), 'detector'].tolist()
    fixed = 0; leaders = []
    for w in W:
        b = [fsr_bounds_w(np.array([r[f'R_{t}'] for t in cd.TYPES]), n, r['P@100_mean'], w) for _, r in sub.iterrows()]
        lo = np.array([x[0] for x in b]); hi = np.array([x[1] for x in b])
        # reversal fixed: every P@100 leader has some other detector whose lower bound exceeds its upper bound
        ok = all(any(lo[j] > hi[i] + 1e-12 for j in range(len(sub)) if j != i) for i in sub.index[sub.detector.isin(pw)])
        fixed += ok
        best = [sub.detector[i] for i in range(len(sub)) if all(lo[i] > hi[j] + 1e-12 for j in range(len(sub)) if j != i)]
        leaders.append(best[0] if best else 'undetermined')
    out.append(dict(ledger=L, P100_leader=', '.join(pw), draws=len(W), share_reversal_fixed_by_bounds=fixed / len(W),
                    FSR_leader_by_bounds=pd.Series(leaders).value_counts(normalize=True).round(3).to_dict()))
    print(L, round(time.time() - t0), 's', out[-1], flush=True)
pd.DataFrame(out).to_csv(paths.out_client('a10b_client_random_bounds.csv'), index=False)
