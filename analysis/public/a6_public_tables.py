"""A6: public ledger tables for the nine unsupervised detectors and the LightGBM ceiling (DECOMP, DECOMP+ctx,
AVG and LINK dropped), at full precision.
- a6_public_global.csv: P@25, P@50, P@100, P@1000, R@1000, AP, ROC-AUC (mean and sd over seeds; n_seeds) taken
  from dib/results/metrics_mean.csv and metrics_sd.csv, and recomputed from the saved scores as a check.
- a6_public_marking_recall.csv: recall per marking within the top 100 and top 1,000 entries (mean and sd).
- a6_public_rule_populations.csv: P@25, P@50, P@100 and AP inside the four rule populations (J1 top-n amounts,
  J2 high cash, J3 delayed entry, J4 weekend or non-working hours), recomputed at full precision from the
  scores with the definitions of dib/analyze.py (the results file is rounded to three decimals; the
  recomputation is checked against it).
- a6_public_top100_composition.csv: composition of the top 100 entries (anomalous; normal touching a rare
  account with fewer than 1,000 lines; other normal), seed 0 as in top100_fill.csv (checked) and the mean over
  seeds for the stochastic detectors."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.PIPELINE))
from dib_data import load, entries, entry_scores

SC = str(paths.SCORES)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE', 'LGBM-ceiling': 'LightGBM ceiling'}
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling'}
MKN = {1: 'M1 high amount', 2: 'M2 high cash disbursement', 3: 'M3 delayed entry', 4: 'M4 very delayed entry',
       5: 'M5 non-working hours', 6: 'M6 weekend', 7: 'M7 shadow bank account', 8: 'M8 cross-linked clearing'}
MM = pd.read_csv(paths.public_result('metrics_mean.csv'), index_col=0).loc[DETS]
SD = pd.read_csv(paths.public_result('metrics_sd.csv'), index_col=0).loc[DETS]
WP = pd.read_csv(paths.public_result('within_population.csv')); TF = pd.read_csv(paths.public_result('top100_fill.csv'), index_col=0)

df = load(); e = entries(df); y = e.label.values; mk = e.marking.values; eid = df.eid.values
S = {d: {s: entry_scores(np.load(f'{SC}/{d}_s{s}.npy'), eid) for s in ([0, 1, 2] if d in STOCH else [0])} for d in DETS}
order = lambda s: np.argsort(-np.asarray(s), kind='stable')

# global
gcols = ['P@25', 'P@50', 'P@100', 'P@1000', 'R@1000', 'AP', 'AUC']
rows, maxdev = [], 0.0
for d in DETS:
    rec = dict(detector=NAME.get(d, d), n_seeds=int(MM.loc[d, 'n_seeds']))
    for c in gcols:
        rec[f'{c}_mean'] = MM.loc[d, c]; rec[f'{c}_sd'] = SD.loc[d, c]
    # recompute
    vals = {c: [] for c in gcols}
    for s, sc in S[d].items():
        o = order(sc)
        for k in (25, 50, 100, 1000): vals[f'P@{k}'].append(y[o[:k]].mean())
        vals['R@1000'].append(y[o[:1000]].sum() / y.sum())
        vals['AP'].append(average_precision_score(y, sc)); vals['AUC'].append(roc_auc_score(y, sc))
    for c in gcols:
        maxdev = max(maxdev, abs(np.mean(vals[c]) - MM.loc[d, c]))
        sdv = np.std(vals[c], ddof=1) if len(vals[c]) > 1 else 0.0
        maxdev = max(maxdev, abs(sdv - SD.loc[d, c]))
    rows.append(rec)
G = pd.DataFrame(rows).rename(columns={'AUC_mean': 'ROC-AUC_mean', 'AUC_sd': 'ROC-AUC_sd'})
G['reference_only'] = G.detector == 'LightGBM ceiling'
G.to_csv(paths.out_public('a6_public_global.csv'), index=False)
print('global table: max |recomputed - results file| =', maxdev)

# per-marking recall
rows = []
for d in DETS:
    for m in range(1, 9):
        rows.append(dict(detector=NAME.get(d, d), marking=f'M{m}', marking_name=MKN[m], n_marking=int((mk == m).sum()),
                         R100_mean=MM.loc[d, f'R100_M{m}'], R100_sd=SD.loc[d, f'R100_M{m}'],
                         R1000_mean=MM.loc[d, f'R1000_M{m}'], R1000_sd=SD.loc[d, f'R1000_M{m}']))
MR = pd.DataFrame(rows); MR.to_csv(paths.out_public('a6_public_marking_recall.csv'), index=False)

# rule populations
pops = {'J1 top-n amounts': e.top_n.values == 1, 'J2 high cash': e.high_cash.values == 1,
        'J3 delayed entry': e.promptly.values >= 2, 'J4 weekend or non-working hours': (e.weekend.values >= 1) | (e.nwh.values == 1)}
rows, maxdev_w = [], 0.0
for pn, msk in pops.items():
    yy = y[msk]
    for d in DETS:
        recs = []
        for s, sc in S[d].items():
            ss = sc[msk]; o = order(ss)
            recs.append({'P@25': yy[o[:25]].mean(), 'P@50': yy[o[:50]].mean(), 'P@100': yy[o[:100]].mean(),
                         'AP': average_precision_score(yy, ss)})
        R = pd.DataFrame(recs)
        rec = dict(population=pn, entries=int(msk.sum()), anomalies=int(yy.sum()), rule_precision=yy.mean(),
                   detector=NAME.get(d, d), n_seeds=len(R))
        for c in ['P@25', 'P@50', 'P@100', 'AP']:
            rec[f'{c}_mean'] = R[c].mean(); rec[f'{c}_sd'] = R[c].std(ddof=1) if len(R) > 1 else 0.0
            ref = WP[(WP.population == pn) & (WP.detector == d)][c].iloc[0]
            maxdev_w = max(maxdev_w, abs(round(rec[f'{c}_mean'], 3) - ref))
        rec['reference_only'] = d == 'LGBM-ceiling'
        rows.append(rec)
W = pd.DataFrame(rows); W.to_csv(paths.out_public('a6_public_rule_populations.csv'), index=False)
print('rule populations: max |round(recomputed, 3) - results file| =', maxdev_w)
best = W[~W.reference_only].loc[lambda w: w.groupby('population')['P@100_mean'].transform('max') == w['P@100_mean']]
print(best[['population', 'detector', 'P@100_mean', 'rule_precision']].to_string(index=False))

# top-100 composition
vc = df.gl_account.value_counts(); rare_acc = set(vc[vc < 1000].index)
has_rare = df.assign(r=df.gl_account.isin(rare_acc)).groupby('eid').r.max().values
rows, dev_t = [], 0
for d in DETS:
    comp = []
    for s, sc in S[d].items():
        o = order(sc)[:100]
        comp.append((int(y[o].sum()), int(((y[o] == 0) & has_rare[o]).sum()), int(((y[o] == 0) & ~has_rare[o]).sum())))
    comp = np.array(comp)
    dev_t = max(dev_t, int(np.abs(comp[0] - TF.loc[d].values).max()))
    rows.append(dict(detector=NAME.get(d, d), anomalous_seed0=comp[0, 0], normal_rare_account_seed0=comp[0, 1], other_normal_seed0=comp[0, 2],
                     anomalous_mean=comp[:, 0].mean(), normal_rare_account_mean=comp[:, 1].mean(), other_normal_mean=comp[:, 2].mean(),
                     n_seeds=len(comp), reference_only=d == 'LGBM-ceiling'))
TC = pd.DataFrame(rows); TC.to_csv(paths.out_public('a6_public_top100_composition.csv'), index=False)
print('top-100 composition seed 0: max |recomputed - results file| =', dev_t)
print('rare accounts (<1000 lines):', sorted(rare_acc), '; normal entries touching them:', int((has_rare & (y == 0)).sum()),
      '; all entries touching them:', int(has_rare.sum()), 'of', len(y))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(G.round(4).to_string(index=False)); print(TC.to_string(index=False))
