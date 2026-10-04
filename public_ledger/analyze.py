import os, json, glob
import numpy as np, pandas as pd
from dib_data import *
from evalx import metrics, order, MK
from feedback import DETS, det_file

from dibpaths import SCORES as SC, RESULTS as RES
SEEDS = [0, 1, 2]
STOCH = {'IF', 'AE-pyod', 'VAE-pyod', 'AE-emb', 'LGBM-ceiling'}
ALL = DETS + ['AVG', 'DECOMP+ctx', 'LINK (post-hoc)', 'LGBM-ceiling']


def entry_score_table(df):
    eid = df.eid.values; S = {}
    for seed in SEEDS:
        S[seed] = {}
        for d in DETS + ['DECOMP+ctx', 'LGBM-ceiling']:
            f = det_file(d, seed)
            if os.path.exists(f): S[seed][d] = entry_scores(np.load(f), eid)
        S[seed]['LINK (post-hoc)'] = np.load(f'{SC}/LINK_entry.npy')
        if all(d in S[seed] for d in DETS):
            S[seed]['AVG'] = np.mean([pct(S[seed][d]) for d in DETS], 0)
    return S


def agg(recs):
    D = pd.DataFrame(recs)
    return D.mean(), (D.std(ddof=1) if len(D) > 1 else D.std().fillna(0) * 0).fillna(0)


if __name__ == '__main__':
    df = load(); e = entries(df); y = e.label.values; mk = e.marking.values
    S = entry_score_table(df)
    rows, sds = {}, {}
    for d in ALL:
        recs = [metrics(S[s][d], y, mk) for s in SEEDS if d in S[s]]
        if not recs: continue
        if d not in STOCH and d != 'AVG': recs = recs[:1]
        if d == 'AVG': recs = [metrics(S[s][d], y, mk) for s in SEEDS if d in S[s]]
        m, sd = agg(recs); rows[d] = m; sds[d] = sd; rows[d]['n_seeds'] = len(recs)
    M = pd.DataFrame(rows).T; SD = pd.DataFrame(sds).T
    M.to_csv(f'{RES}/metrics_mean.csv'); SD.to_csv(f'{RES}/metrics_sd.csv')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    cols = ['P@25', 'P@50', 'P@100', 'P@1000', 'R@1000', 'AP', 'AUC']
    T = M[cols].round(3).astype(str) + ' ± ' + SD[cols].round(3).astype(str)
    print('== Global ranking over 102,429 entries (850 anomalous), mean ± sd over seeds'); print(T)
    print('== Per-marking recall at 1000 entries (at 100 in brackets)')
    PM = pd.DataFrame({MK[m]: M[f'R1000_M{m}'].round(2).astype(str) + ' (' + M[f'R100_M{m}'].round(2).astype(str) + ')' for m in MK})
    print(PM)
    # within JET populations
    pops = {'J1 top-n amounts': e.top_n.values == 1, 'J2 high cash': e.high_cash.values == 1,
            'J3 delayed entry': e.promptly.values >= 2, 'J4 weekend or non-working hours': (e.weekend.values >= 1) | (e.nwh.values == 1)}
    W = []
    for pn, msk in pops.items():
        yy = y[msk]
        for d in ALL:
            recs = []
            for s in SEEDS:
                if d not in S[s]: continue
                ss = S[s][d][msk]; o = order(ss); r = {}
                for k in (25, 50, 100): r[f'P@{k}'] = yy[o[:k]].mean()
                from sklearn.metrics import average_precision_score
                r['AP'] = average_precision_score(yy, ss); recs.append(r)
                if d not in STOCH and d != 'AVG': break
            if recs:
                m = pd.DataFrame(recs).mean(); W.append({'population': pn, 'entries': int(msk.sum()), 'anomalies': int(yy.sum()),
                                                        'rule precision': round(yy.mean(), 3), 'detector': d, **m.round(3).to_dict()})
    W = pd.DataFrame(W); W.to_csv(f'{RES}/within_population.csv', index=False)
    anyrule = (e.top_n.values == 1) | (e.high_cash.values == 1) | (e.promptly.values >= 2) | (e.weekend.values >= 1) | (e.nwh.values == 1)
    print(f'== Union of the four JET rules: {anyrule.sum()} entries, {y[anyrule].sum()} of {y.sum()} anomalies, precision {y[anyrule].mean():.3f}')
    for pn in pops:
        w = W[W.population == pn]; print(f'== {pn}: {w.entries.iloc[0]} entries, {w.anomalies.iloc[0]} anomalous, rule precision {w["rule precision"].iloc[0]}')
        print(w[['detector', 'P@25', 'P@50', 'P@100', 'AP']].to_string(index=False))
    # what fills the top 100: rare-account (legitimate recurring / period-end) entries
    vc = df.gl_account.value_counts(); rare_acc = set(vc[vc < 1000].index)
    has_rare = df.assign(r=df.gl_account.isin(rare_acc)).groupby('eid').r.max().values
    fill = {}
    for d in ALL:
        if d not in S[0]: continue
        o = order(S[0][d])[:100]
        fill[d] = {'anomalous': int(y[o].sum()), 'normal with rare account (<1000 lines)': int(((y[o] == 0) & has_rare[o]).sum()),
                   'other normal': int(((y[o] == 0) & ~has_rare[o]).sum())}
    F = pd.DataFrame(fill).T; F.to_csv(f'{RES}/top100_fill.csv'); print('== Composition of the top 100 entries (seed 0)'); print(F)
    print('rare accounts:', sorted(rare_acc), 'normal entries touching them:', int((has_rare & (y == 0)).sum()))
