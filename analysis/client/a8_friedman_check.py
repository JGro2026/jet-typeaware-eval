"""Check of the Friedman test without DECOMP and AVG quoted in the brief: run B T3 sweep (P@100, seed means,
two decimals), ledgers C, A, B, 3 contamination x 3 difficulty = 27 cells, 9 detectors. Ranks per cell with
ties averaged (rank 1 = highest P@100). Nemenyi critical difference at alpha 0.05 with q from the studentized
range distribution divided by sqrt(2) (Demsar 2006). Output: a8_friedman_check.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys, re
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common_datev as cd

txt = open(cd.SRC).read(); sec = cd._section(txt, '## T3.', '## T4.')
rows = []
for L in ['C', 'A', 'B']:
    R = cd._rows(cd._block(sec, L))
    for d in cd.DETS:
        rows.append([float(x) for x in R[d]])
X = np.array(rows).reshape(3, 9, 9)            # ledger, detector, cell
M = np.concatenate([X[i].T for i in range(3)])  # 27 cells x 9 detectors
chi2, p = friedmanchisquare(*[M[:, j] for j in range(9)])
ranks = np.array([rankdata(-r) for r in M]).mean(0)
k, n = 9, M.shape[0]
q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
cd_ = q * np.sqrt(k * (k + 1) / (6 * n))
out = pd.DataFrame({'detector': [cd.NAME.get(d, d) for d in cd.DETS], 'mean_rank': ranks})
out = pd.concat([out, pd.DataFrame([{'detector': 'Friedman chi2', 'mean_rank': chi2}, {'detector': 'p value', 'mean_rank': p},
                                    {'detector': 'n cells', 'mean_rank': n}, {'detector': 'k detectors', 'mean_rank': k},
                                    {'detector': 'q_alpha (0.05)', 'mean_rank': q}, {'detector': 'CD Nemenyi', 'mean_rank': cd_}])])
out.to_csv(paths.out_client('a8_friedman_check.csv'), index=False)
print(out.to_string(index=False))
# which detectors differ from kNN by more than CD
r = dict(zip([cd.NAME.get(d, d) for d in cd.DETS], ranks))
print('within CD of kNN:', [d for d in r if r[d] - r['kNN'] <= cd_])
