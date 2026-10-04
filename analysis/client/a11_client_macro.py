"""Client ledgers: leaders and Kendall tau_b under P@100, macro-averaged recall (mean of R_t@100 over the types) and FSR@100
(midpoint of the lattice bounds), from the printed seed-mean per-type recalls (client rows of Table 7).
Needs datev_typeaware.py and a2_bounds.py. Outputs: a11_client_macro.csv, a11_client_summary.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import pandas as pd
from scipy.stats import kendalltau

d = pd.read_csv(paths.find('datev_typeaware_k100.csv')); f = pd.read_csv(paths.find('datev_fsr_k100.csv'))
d = d.merge(f[['ledger', 'detector', 'FSR_lo_lattice', 'FSR_hi_lattice']], on=['ledger', 'detector'])
d['macro'] = d[['R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5']].mean(axis=1)
d['FSR_mid'] = (d.FSR_lo_lattice + d.FSR_hi_lattice) / 2
d[['ledger', 'detector', 'P100', 'macro', 'FSR_lo_lattice', 'FSR_hi_lattice', 'FSR_mid']].to_csv(paths.out_client('a11_client_macro.csv'), index=False)

def lead(g, col, tol):
    m = g[col].max(); return ', '.join(g.detector[g[col] >= m - tol])

rows = []
for L in ['C', 'A', 'B', 'D']:
    g = d[d.ledger == L]
    rows.append(dict(ledger=L, k=100, P_leader=lead(g, 'P100', 1e-9), macro_leader=lead(g, 'macro', 0.005), FSR_leader=lead(g, 'FSR_mid', 1e-9),
                     tau_P_macro=kendalltau(g.P100, g.macro).statistic, tau_macro_FSR=kendalltau(g.macro, g.FSR_mid).statistic,
                     tau_P_FSR=kendalltau(g.P100, g.FSR_mid).statistic))
S = pd.DataFrame(rows); S.to_csv(paths.out_client('a11_client_summary.csv'), index=False)
print(S.round(2).to_string(index=False))
