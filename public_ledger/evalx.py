import numpy as np, pandas as pd, glob, os
from sklearn.metrics import average_precision_score, roc_auc_score
from dib_data import load, entries, entry_scores

KS = [25, 50, 100, 1000]
MK = {1: 'M1 high amount', 2: 'M2 high cash', 3: 'M3 delayed', 4: 'M4 very delayed',
      5: 'M5 non-working hours', 6: 'M6 weekend', 7: 'M7 shadow bank', 8: 'M8 cross-linked'}


def order(s):
    # stable descending sort, ties by ascending position
    return np.argsort(-np.asarray(s), kind='stable')


def metrics(s, y, mk=None, ks=KS):
    o = order(s); r = {}
    for k in ks:
        top = y[o[:k]]; r[f'P@{k}'] = top.mean(); r[f'R@{k}'] = top.sum() / max(y.sum(), 1)
    r['AP'] = average_precision_score(y, s); r['AUC'] = roc_auc_score(y, s)
    if mk is not None:
        for m in MK:
            tot = (mk == m).sum()
            for k in (100, 1000):
                r[f'R{k}_M{m}'] = (mk[o[:k]] == m).sum() / tot if tot else np.nan
    return r
