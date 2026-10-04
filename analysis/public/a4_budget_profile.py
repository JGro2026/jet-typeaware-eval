"""A4: public ledger budget profile, from the saved per-line scores (entry score = max over lines, stable
descending sort, ties by entry order; exactly as dib_typeaware.py).
- markings covered at k in {25, 50, 100, 250, 500, 1000, 2000}: mean, min, max over seeds
- first-hit rank per marking (1-based): mean, min, max over seeds
- full-coverage budget: per seed, the set S of markings whose first hit is within the first 2,000 entries and
  the rank at which the last of them is first hit (max first-hit rank over S); mean, min, max over seeds
- per-marking recall at every k (mean over seeds), and Kendall tau-b of P@k against FSR@k, TC@k and alpha-nDCG@k
  over the nine unsupervised detectors at every k.
Outputs: a4_coverage_by_k.csv, a4_first_hit.csv, a4_full_coverage_budget.csv, a4_marking_recall_by_k.csv,
a4_public_tau_by_k.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from scipy.stats import kendalltau
HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.dirname(HERE)
sys.path.insert(0, str(paths.PIPELINE)); sys.path.insert(0, str(paths.TYPEAWARE))
from dib_data import load, entries, entry_scores
from fsr import fsr

SC = str(paths.SCORES)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE', 'LGBM-ceiling': 'LightGBM ceiling'}
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling'}
KS = [25, 50, 100, 250, 500, 1000, 2000]
M = list(range(1, 9))
df = load(); e = entries(df); y = e.label.values; mk = e.marking.values; eid = df.eid.values
n = np.array([(mk == m).sum() for m in M])
cov_rows, fh_rows, fb_rows, rec_rows = [], [], [], []
for d in DETS:
    for s in ([0, 1, 2] if d in STOCH else [0]):
        sc = entry_scores(np.load(f'{SC}/{d}_s{s}.npy'), eid); o = np.argsort(-sc, kind='stable'); mo = mk[o]
        first = np.array([int(np.flatnonzero(mo == m)[0]) + 1 for m in M])
        fh_rows.append(dict(detector=NAME.get(d, d), seed=s, **{f'M{m}': first[i] for i, m in enumerate(M)}))
        Sset = [m for i, m in enumerate(M) if first[i] <= 2000]
        fb_rows.append(dict(detector=NAME.get(d, d), seed=s, markings_within_2000=len(Sset),
                            markings_list=' '.join(f'M{m}' for m in Sset),
                            budget_all_reached=int(max(first[i] for i, m in enumerate(M) if first[i] <= 2000)) if Sset else np.nan,
                            markings_never_within_2000=' '.join(f'M{m}' for i, m in enumerate(M) if first[i] > 2000)))
        for k in KS:
            found = np.array([(mo[:k] == m).sum() for m in M])
            cov_rows.append(dict(detector=NAME.get(d, d), seed=s, k=k, P=found.sum() / k, coverage=int((found > 0).sum()),
                                 FSR=fsr(found, n, k)))
            rec_rows.append(dict(detector=NAME.get(d, d), seed=s, k=k, **{f'R_M{m}': found[i] / n[i] for i, m in enumerate(M)}))
C = pd.DataFrame(cov_rows); FH = pd.DataFrame(fh_rows); FB = pd.DataFrame(fb_rows); RR = pd.DataFrame(rec_rows)
order = [NAME.get(d, d) for d in DETS]
# coverage by k
g = C.groupby(['detector', 'k']).coverage
CK = pd.DataFrame({'coverage_mean': g.mean(), 'coverage_min': g.min(), 'coverage_max': g.max(), 'n_seeds': g.size()}).reset_index()
cov2000 = CK[CK.k == 2000].set_index('detector').coverage_mean
CK['reached_final_2000_level'] = [abs(r.coverage_mean - cov2000[r.detector]) < 1e-12 for r in CK.itertuples()]
CK['detector'] = pd.Categorical(CK.detector, order, ordered=True); CK = CK.sort_values(['detector', 'k'])
CK.to_csv(paths.out_public('a4_coverage_by_k.csv'), index=False)
# first hit
FHL = FH.melt(id_vars=['detector', 'seed'], var_name='marking', value_name='first_hit_rank')
g = FHL.groupby(['detector', 'marking']).first_hit_rank
FHS = pd.DataFrame({'first_hit_mean': g.mean(), 'first_hit_min': g.min(), 'first_hit_max': g.max(), 'n_seeds': g.size()}).reset_index()
FHS['detector'] = pd.Categorical(FHS.detector, order, ordered=True); FHS = FHS.sort_values(['detector', 'marking'])
FHS.to_csv(paths.out_public('a4_first_hit.csv'), index=False)
FH.to_csv(paths.out_public('a4_first_hit_perseed.csv'), index=False)
# full-coverage budget
g = FB.groupby('detector')
FBS = pd.DataFrame({'markings_within_2000_mean': g.markings_within_2000.mean(), 'markings_within_2000_min': g.markings_within_2000.min(),
                    'markings_within_2000_max': g.markings_within_2000.max(),
                    'budget_all_reached_mean': g.budget_all_reached.mean(), 'budget_all_reached_min': g.budget_all_reached.min(),
                    'budget_all_reached_max': g.budget_all_reached.max(),
                    'markings_per_seed': g.markings_list.apply(lambda s: ' | '.join(s)),
                    'never_within_2000_per_seed': g.markings_never_within_2000.apply(lambda s: ' | '.join(s)),
                    'budget_per_seed': g.budget_all_reached.apply(lambda s: ' | '.join(str(int(v)) for v in s))}).reset_index()
# smallest grid k at which the (mean) coverage reaches its k = 2000 level
FBS['grid_k_reaching_2000_coverage'] = [int(CK[(CK.detector == d) & CK.reached_final_2000_level].k.min()) for d in FBS.detector]
FBS['detector'] = pd.Categorical(FBS.detector, order, ordered=True); FBS = FBS.sort_values('detector')
FBS.to_csv(paths.out_public('a4_full_coverage_budget.csv'), index=False)
FB.to_csv(paths.out_public('a4_full_coverage_budget_perseed.csv'), index=False)
# per-marking recall by k
RM = RR.groupby(['detector', 'k']).mean(numeric_only=True).drop(columns='seed').reset_index()
RM['detector'] = pd.Categorical(RM.detector, order, ordered=True); RM = RM.sort_values(['detector', 'k'])
RM.to_csv(paths.out_public('a4_marking_recall_by_k.csv'), index=False)
# tau by k (nine unsupervised detectors, mean over seeds; alpha-nDCG from ../dib_typeaware_mean.csv)
A = pd.read_csv(paths.find('dib_typeaware_mean.csv'))
nine = order[:-1]; rows = []
for k in KS:
    a = A[A.k == k].set_index('detector').loc[nine]
    cm = C[C.k == k].groupby('detector')[['P', 'coverage', 'FSR']].mean().loc[nine]
    assert np.allclose(cm.P.values, a.P.values) and np.allclose(cm.FSR.values, a.FSR.values)
    rows.append(dict(k=k, tau_P_FSR=kendalltau(a.P, a.FSR).statistic, tau_P_TC=kendalltau(a.P, a.coverage).statistic,
                     tau_P_alphaNDCG=kendalltau(a.P, a.alpha_nDCG).statistic,
                     best_P=a.P.idxmax(), best_FSR=a.FSR.idxmax(), best_TC='='.join(a.index[a.coverage == a.coverage.max()]),
                     best_alphaNDCG=a.alpha_nDCG.idxmax()))
TK = pd.DataFrame(rows); TK.to_csv(paths.out_public('a4_public_tau_by_k.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 80)
print(CK.pivot(index='detector', columns='k', values='coverage_mean').round(2).to_string())
print(FHS.pivot(index='detector', columns='marking', values='first_hit_mean').round(0).to_string())
print(FBS.drop(columns=['markings_per_seed']).to_string(index=False))
print(TK.round(3).to_string(index=False))
