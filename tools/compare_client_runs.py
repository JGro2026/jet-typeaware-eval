"""Compare two executions of the client-ledger benchmark (for example your rerun with the run reported in the paper):
R6 cell-by-cell reproduction, R7 conclusion stability and R8 seed variability. Run A and run B are the tables.md files
that harness/src/analyze.py writes.
Only cells that both runs contain are compared, and only ledgers A, B and C (Table-D was run with
3 seeds in run A and 5 in run B, so its aggregates are not comparable cell by cell)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdatev as rd
if len(sys.argv) != 4:
    sys.exit('usage: python tools/compare_client_runs.py RUN_A_tables.md RUN_B_tables.md OUTPUT_FOLDER')
rd.FILES.update({'A': sys.argv[1], 'B': sys.argv[2]})
OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
DETERMINISTIC = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'DECOMP']
NEURAL = ['AE-pyod', 'VAE-pyod', 'AE-emb']
LEDGERS = ['C', 'A', 'B']
EXCL_T3 = ('B', 'c=0.030 hard')        # run A averaged this sweep cell over 3 seeds, run B over 5


def group_of(name):
    if name in DETERMINISTIC: return 'deterministic'
    if name in NEURAL: return 'neural'
    if name == 'AVG': return 'AVG'
    return 'feedback' if name in ('STATIC-AVG', 'LinTS', 'GREEDY-LR', 'DeepSAD',
                                  'LinTS-spec', 'GREEDY-LR-spec') else 'other'


def compare(sec, part='mean', ledgers=LEDGERS, excl=None, label=None):
    a = rd.long_metric('A', sec); b = rd.long_metric('B', sec)
    key = ['ledger', 'row', 'metric']
    m = a.merge(b, on=key, suffixes=('_A', '_B'))
    m = m[m.ledger.isin(ledgers)]
    if excl is not None:
        m = m[~((m.ledger == excl[0]) & (m.metric == excl[1]))]
    m = m.dropna(subset=[f'{part}_A', f'{part}_B'])
    m['section'] = label or sec
    m['value_A'] = m[f'{part}_A']; m['value_B'] = m[f'{part}_B']
    m['abs_diff'] = (m.value_B - m.value_A).abs()
    m['identical'] = m.abs_diff < 1e-12
    m['group'] = m.row.map(group_of)
    return m[['section', 'ledger', 'row', 'metric', 'group', 'value_A', 'value_B', 'abs_diff', 'identical']]


blocks = []
blocks.append(compare('T1', 'mean', label='T1 core grid (means)'))
blocks.append(compare('T3', 'mean', excl=EXCL_T3, label='T3 sweep grid'))
main = pd.concat(blocks, ignore_index=True)

FOUND = ['found@25', 'found@50', 'found@75', 'found@100']
t5m = compare('T5', 'mean', label='T5 feedback')
t5s = compare('T5', 'sd', label='T5 feedback')
t5m['section'] = np.where(t5m.metric.isin(FOUND), 'T5 feedback, anomalies found (counts, means)',
                          'T5 feedback, final and adaptive precision (means)')
t5s['section'] = np.where(t5s.metric.isin(FOUND), 'T5 feedback, anomalies found (counts, sd)',
                          'T5 feedback, final and adaptive precision (sd)')
extra = [compare('T1', 'sd', label='T1 core grid (sd over seeds)'),
         compare('T2', 'mean', label='T2 per-type recall at 100'),
         compare('T8', 'mean', label='T8 DECOMP ablation'),
         t5m, t5s]
# T9 rows carry the ledger inside the row label, so they need their own merge
t9 = rd.long_metric('A', 'T9').merge(rd.long_metric('B', 'T9'), on=['row', 'metric'],
                                     suffixes=('_A', '_B'))
t9['ledger'] = t9.row.str.extract(r"\('([ABCD])'")[0]
t9 = t9[t9.ledger.isin(LEDGERS)].dropna(subset=['mean_A', 'mean_B'])
t9 = pd.DataFrame({'section': 'T9 pair diagnostic (post-hoc)', 'ledger': t9.ledger, 'row': t9.row,
                   'metric': t9.metric, 'group': 'deterministic',
                   'value_A': t9['mean_A'], 'value_B': t9['mean_B']})
t9['abs_diff'] = (t9.value_B - t9.value_A).abs(); t9['identical'] = t9.abs_diff < 1e-12
extra.append(t9)
# T6 holds the supervised ceiling; its rows are ledgers, not detectors
t6a = rd.long_metric('A', 'T6'); t6b = rd.long_metric('B', 'T6')
t6 = t6a.merge(t6b, on=['ledger', 'row', 'metric'], suffixes=('_A', '_B'))
t6 = t6[t6.row.isin(LEDGERS) & t6.metric.str.startswith('Ceiling')].dropna(subset=['mean_A', 'mean_B'])
t6 = pd.DataFrame({'section': 'T6 supervised ceiling', 'ledger': t6.row, 'row': 'LightGBM ceiling',
                   'metric': t6.metric, 'group': 'deterministic',
                   'value_A': t6['mean_A'], 'value_B': t6['mean_B']})
t6['abs_diff'] = (t6.value_B - t6.value_A).abs(); t6['identical'] = t6.abs_diff < 1e-12
extra.append(t6)
allc = pd.concat([main] + extra, ignore_index=True)
allc.round(6).to_csv(f'{OUT}/r6_datev_all_compared_cells.csv', index=False)
allc[~allc.identical].round(6).to_csv(f'{OUT}/r6_datev_differing_cells.csv', index=False)


def summarise(d, by):
    g = d.groupby(by).agg(cells=('identical', 'size'), identical=('identical', 'sum'),
                          max_abs_diff=('abs_diff', 'max')).reset_index()
    g['share_identical'] = (g.identical / g.cells).round(4)
    return g


summarise(main, ['ledger', 'row']).round(6).to_csv(f'{OUT}/r6_datev_by_ledger_detector.csv', index=False)
summarise(main, ['group']).round(6).to_csv(f'{OUT}/r6_datev_group_summary_T1T3.csv', index=False)
summarise(allc, ['section', 'group']).round(6).to_csv(f'{OUT}/r6_datev_section_group_summary.csv', index=False)
summarise(allc, ['row']).rename(columns={'row': 'detector or method'}).round(6).to_csv(
    f'{OUT}/r6_datev_by_detector_all_sections.csv', index=False)
print('== T1 means + T3 sweep, ledgers C/A/B'); print(summarise(main, ['group']).to_string(index=False))
print('total cells', len(main), 'identical', int(main.identical.sum()))
print('== by section and group'); print(summarise(allc, ['section', 'group']).to_string(index=False))

# T4 frequency ablation: only ledger A is like for like (40 cells in both runs)
t4a = rd.long_metric('A', 'T4'); t4b = rd.long_metric('B', 'T4')
t4 = t4a.merge(t4b, on=['ledger', 'row', 'metric'], suffixes=('_A', '_B'))
t4['ledger'] = t4.row.str.extract(r"\('([ABC])'")[0]; t4['det'] = t4.row.str.extract(r"'([^']+)'\)")[0]
t4['abs_diff'] = (t4['mean_B'] - t4['mean_A']).abs(); t4['identical'] = t4.abs_diff < 1e-12
t4 = t4[['ledger', 'det', 'metric', 'mean_A', 'mean_B', 'abs_diff', 'identical']]
t4.round(6).to_csv(f'{OUT}/r6_datev_T4_comparison.csv', index=False)
print('== T4 sweep-cell counts per (ledger, detector)')
print(t4[t4.metric == 'cells'].to_string(index=False))
print('== T4 metric cells by ledger (the count column excluded)')
print(t4[t4.metric != 'cells'].groupby('ledger').agg(cells=('identical', 'size'),
      identical=('identical', 'sum'), max_abs_diff=('abs_diff', 'max')).to_string())

# ---------------- R7: conclusion stability ----------------
def t1(run):
    d = rd.long_metric(run, 'T1'); d['run'] = run; return d


T1 = pd.concat([t1('A'), t1('B')], ignore_index=True)
claims = []


def add(claim, run, holds, detail):
    claims.append({'claim': claim, 'run': run, 'holds': holds, 'numbers': detail})


for run in ['A', 'B']:
    for led in ['C', 'A', 'B']:
        d = T1[(T1.run == run) & (T1.ledger == led) & (T1.metric == 'P@100')]
        d = d.set_index('row')['mean']
        best = d.idxmax(); top = d.max(); second = d.drop(best).max(); sec_name = d.drop(best).idxmax()
        add(f'DECOMP has the highest P@100 on ledger {led}', f'run {run}', bool(best == 'DECOMP'),
            f'DECOMP {d["DECOMP"]:.2f}, next {sec_name} {second:.2f}')
        lo = d[['IF', 'LOF', 'HBOS', 'ECOD']]
        add(f'IF, LOF, HBOS and ECOD stay below 0.10 P@100 on ledger {led}', f'run {run}',
            bool((lo < 0.10).all()), ', '.join(f'{k} {v:.2f}' for k, v in lo.items()))
    dD = T1[(T1.run == run) & (T1.ledger == 'D')]
    p = dD[dD.metric == 'P@100'].set_index('row')['mean']
    ap = dD[dD.metric == 'AP'].set_index('row')['mean']
    rec = p[['PCA', 'AE-pyod', 'VAE-pyod']]
    add('the three reconstruction detectors reach P@100 0.98 on ledger D', f'run {run}',
        bool((rec == 0.98).all()), ', '.join(f'{k} {v:.2f}' for k, v in rec.items()))
    add('DECOMP has the highest AP on ledger D', f'run {run}', bool(ap.idxmax() == 'DECOMP'),
        f'DECOMP {ap["DECOMP"]:.2f}, next {ap.drop("DECOMP").idxmax()} {ap.drop("DECOMP").max():.2f}')

for run in ['A', 'B']:
    sec = dict(rd.parse(run)['T7'])
    for metric, T in rd.parse(run)['T7']:
        mr = T.set_index('Detector')['mean rank'].astype(float)
        diff = T.set_index('Detector')['different from best (CD)']
        nm = 'P@100' if 'P@100' in str(metric) else 'AP'
        add(f'DECOMP and kNN are not separable on the sweep grid ({nm})', f'run {run}',
            bool(mr.idxmin() == 'DECOMP' and str(diff.get('kNN', '')).strip() == 'no'),
            f'mean ranks DECOMP {mr["DECOMP"]}, kNN {mr["kNN"]}, Nemenyi p = '
            f'{T.set_index("Detector").loc["kNN", "Nemenyi p vs best"]}, all others differ')

T5 = pd.concat([rd.long_metric(r, 'T5').assign(run=r) for r in ['A', 'B']], ignore_index=True)
for run in ['A', 'B']:
    for led in ['C', 'A', 'B', 'D']:
        d = T5[(T5.run == run) & (T5.ledger == led) & (T5.metric == 'found@100')]
        if d.empty: continue
        v = d.set_index('row')['mean']
        add(f'the textbook linear feedback settings stay below STATIC-AVG on ledger {led}', f'run {run}',
            bool(v['LinTS-spec'] < v['STATIC-AVG'] and v['GREEDY-LR-spec'] < v['STATIC-AVG']),
            f'STATIC-AVG {v["STATIC-AVG"]}, LinTS-textbook {v["LinTS-spec"]}, '
            f'GREEDY-LR-textbook {v["GREEDY-LR-spec"]} found after 100 reviews')
        if led in ('A', 'B', 'D'):
            add(f'DeepSAD finds more than STATIC-AVG after 100 reviews on ledger {led}', f'run {run}',
                bool(v['DeepSAD'] > v['STATIC-AVG']),
                f'DeepSAD {v["DeepSAD"]} vs STATIC-AVG {v["STATIC-AVG"]}')
R7 = pd.DataFrame(claims)
R7.to_csv(f'{OUT}/r7_datev_conclusion_stability.csv', index=False)
print('== R7'); print(R7.to_string(index=False))

# ---------------- R8: seed variability in run B ----------------
sd = rd.long_metric('B', 'T1').dropna(subset=['sd'])
piv = sd.pivot_table(index='row', columns=['metric'], values='sd', aggfunc='max')
piv['largest_sd_any_metric'] = piv.max(axis=1)
w = sd.loc[sd.groupby(['row', 'metric']).sd.idxmax()][['row', 'metric', 'ledger', 'sd', 'mean']]
piv = piv.join(w[w.metric == 'P@100'].set_index('row')[['ledger', 'sd']]
               .rename(columns={'ledger': 'ledger_of_largest_P@100_sd', 'sd': 'largest_P@100_sd'}))
idx = sd.groupby('row').sd.idxmax()
top = sd.loc[idx].set_index('row')[['ledger', 'metric', 'sd', 'mean']]
piv['metric_of_largest_sd'] = top['metric']; piv['ledger_of_largest_sd'] = top['ledger']
piv['mean_at_largest_sd'] = top['mean']
piv.round(6).to_csv(f'{OUT}/r8_datev_seed_sd.csv')
print('== R8 largest sd per detector (run B, over ledgers C/A/B/D)')
print(piv[['P@25', 'P@50', 'P@100', 'AP', 'ROC-AUC', 'largest_sd_any_metric',
           'ledger_of_largest_P@100_sd']].to_string())

rows = []
for led in ['C', 'A', 'B', 'D']:
    d = rd.long_metric('B', 'T1'); d = d[(d.ledger == led) & (d.metric == 'P@100')].set_index('row')
    best = d['mean'].idxmax(); second = d['mean'].drop(best).idxmax()
    gap = d.loc[best, 'mean'] - d.loc[second, 'mean']
    pooled = float(np.hypot(d.loc[best, 'sd'], d.loc[second, 'sd']))
    rows.append({'ledger': led, 'best': best, 'best_P@100': d.loc[best, 'mean'], 'best_sd': d.loc[best, 'sd'],
                 'second': second, 'second_P@100': d.loc[second, 'mean'], 'second_sd': d.loc[second, 'sd'],
                 'gap': round(gap, 6), 'sd_of_gap_if_independent': round(pooled, 6),
                 'gap_exceeds_one_sd_of_gap': bool(gap > pooled)})
pd.DataFrame(rows).to_csv(f'{OUT}/r8_datev_gap_vs_sd.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))

# adjacent pairs in the P@100 ranking that are not separated by one sd of their difference
pairs = []
for led in ['C', 'A', 'B', 'D']:
    d = rd.long_metric('B', 'T1'); d = d[(d.ledger == led) & (d.metric == 'P@100')]
    d = d.sort_values('mean', ascending=False).reset_index(drop=True)
    for i in range(len(d) - 1):
        hi, lo = d.loc[i], d.loc[i + 1]
        gap = hi['mean'] - lo['mean']; sg = float(np.hypot(hi['sd'], lo['sd']))
        pairs.append({'ledger': led, 'rank': i + 1, 'higher': hi['row'], 'lower': lo['row'],
                      'P@100_higher': hi['mean'], 'P@100_lower': lo['mean'], 'gap': round(gap, 6),
                      'sd_of_gap_if_independent': round(sg, 6),
                      'separated_by_more_than_one_sd': bool(gap > sg)})
P = pd.DataFrame(pairs)
P.to_csv(f'{OUT}/r8_datev_adjacent_pairs.csv', index=False)
print('== adjacent P@100 pairs not separated by one sd of the difference (run B)')
print(P[~P.separated_by_more_than_one_sd].to_string(index=False))
print('done')
