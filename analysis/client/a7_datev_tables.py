"""A7: real-ledger core-grid tables without DECOMP and AVG (run B, c = 1.5 percent, medium, 5 seeds).
- a7_datev_core.csv: P@25, P@50, P@100, AP, ROC-AUC (mean and sd, two decimals as printed in run B T1) for the
  nine unsupervised detectors on ledgers C, A, B, D.
- a7_datev_ceiling.csv: LightGBM ceiling P@25, P@100, AP from run B T6.
- a7_datev_best.csv: best unsupervised detector per ledger by P@25, P@100, AP and ROC-AUC, recomputed without
  DECOMP (ties at the printed precision are listed jointly), with the ceiling alongside.
- a7_datev_P100_AP_wide.csv: compact P@100 and AP per detector and ledger (main-text Table 4 layout), with the
  public ledger (P@100, AP, P@1000; mean over seeds; run a6_public_tables.py first)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common_datev as cd

T1 = cd.t1(); T6 = cd.t6()
T1.insert(1, 'rows', T1.ledger.map(cd.N_ROWS)); T1.insert(2, 'injected', T1.ledger.map(cd.N_ANOM)); T1.insert(3, 'seeds', cd.SEEDS)
T1.to_csv(paths.out_client('a7_datev_core.csv'), index=False)
T6.to_csv(paths.out_client('a7_datev_ceiling.csv'), index=False)
rows = []
for L in cd.LEDGERS:
    t = T1[T1.ledger == L]; c6 = T6[T6.ledger == L].iloc[0]
    rec = dict(ledger=L)
    for m in ['P@25', 'P@50', 'P@100', 'AP', 'ROC-AUC']:
        v = t[f'{m}_mean']; mx = v.max()
        rec[f'best_by_{m}'] = '='.join(t.detector[v == mx]); rec[f'{m}_best'] = mx
        second = v[v < mx].max(); rec[f'{m}_second'] = second
        rec[f'second_by_{m}'] = '='.join(t.detector[v == second])
    rec.update({'ceiling_P@25': c6['ceiling_P@25'], 'ceiling_P@100': c6['ceiling_P@100'], 'ceiling_AP': c6['ceiling_AP']})
    rows.append(rec)
BEST = pd.DataFrame(rows); BEST.to_csv(paths.out_client('a7_datev_best.csv'), index=False)
W = T1.pivot(index='detector', columns='ledger', values=['P@100_mean', 'AP_mean'])
W = W.reindex(['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE', 'VAE', 'AE-emb'])
W.columns = [f'{a.replace("_mean", "")}_{b}' for a, b in W.columns]
W = W[[f'{m}_{L}' for L in cd.LEDGERS for m in ['P@100', 'AP']]]
ceil = pd.DataFrame([{**{f'P@100_{L}': T6.set_index('ledger').loc[L, 'ceiling_P@100'] for L in cd.LEDGERS},
                      **{f'AP_{L}': T6.set_index('ledger').loc[L, 'ceiling_AP'] for L in cd.LEDGERS}}], index=['LightGBM ceiling'])
W = pd.concat([W, ceil[W.columns]]); W.index.name = 'detector'
# public ledger columns (mean over seeds; from a6_public_global.csv, produced by a6_public_tables.py)
G = pd.read_csv(paths.find('a6_public_global.csv')).set_index('detector')
W['P@100_public'] = G.loc[W.index, 'P@100_mean'].values; W['AP_public'] = G.loc[W.index, 'AP_mean'].values
W['P@1000_public'] = G.loc[W.index, 'P@1000_mean'].values
W.to_csv(paths.out_client('a7_datev_P100_AP_wide.csv'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(BEST.to_string(index=False)); print(W.to_string())
