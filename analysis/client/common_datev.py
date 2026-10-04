"""Parsing helpers for run B (restricted/inputs/results_runB_complete.md): T1 core grid, T2 per-type recall at 100,
T6 ceiling. DECOMP and AVG rows are dropped. Values are the two-decimal numbers printed in the file."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import re
import numpy as np, pandas as pd

SRC = str(paths.CLIENT_RESULTS)
DETS = ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb']
NAME = {'AE-pyod': 'AE', 'VAE-pyod': 'VAE'}
LEDGERS = ['C', 'A', 'B', 'D']
N_ROWS = {'C': 3461, 'A': 5080, 'B': 17505, 'D': 279975}
N_ANOM = {'C': 52, 'A': 76, 'B': 263, 'D': 4200}
SEEDS = 5
TYPES = ['T1', 'T2', 'T3', 'T4', 'T5']
P_T = {'T2': 0.20, 'T3': 0.08, 'T4': 0.08, 'T5': 0.24}


def type_counts(n):
    c = {t: round(p * n) for t, p in P_T.items()}; c['T1'] = n - sum(c.values())
    return np.array([c[t] for t in TYPES], float)


def _section(txt, a, b):
    return txt.split(a)[1].split(b)[0]


def _block(sec, L):
    return sec.split(f'**Table-{L}**')[1].split('**Table-')[0]


def _rows(blk):
    out = {}
    for line in blk.splitlines():
        m = re.match(r'\|\s*([^|]+?)\s*\|(.*)\|\s*$', line)
        if not m: continue
        out[m.group(1)] = [c.strip() for c in m.group(2).split('|')]
    return out


def t1():
    """Core grid: mean and sd of P@25, P@50, P@100, AP, ROC-AUC."""
    txt = open(SRC).read(); sec = _section(txt, '## T1.', '## T2.')
    rows = []
    for L in LEDGERS:
        R = _rows(_block(sec, L))
        for d in DETS:
            cells = R[d]; rec = dict(ledger=L, detector=NAME.get(d, d))
            for name, c in zip(['P@25', 'P@50', 'P@100', 'AP', 'ROC-AUC'], cells):
                mu, sd = [float(x) for x in c.split('±')]
                rec[f'{name}_mean'] = mu; rec[f'{name}_sd'] = sd
            rows.append(rec)
    return pd.DataFrame(rows)


def t2():
    """Per-type recall at 100 (seed means, two decimals)."""
    txt = open(SRC).read(); sec = _section(txt, '## T2.', '## T3.')
    rows = []
    for L in LEDGERS:
        R = _rows(_block(sec, L))
        for d in DETS:
            rows.append(dict(ledger=L, detector=NAME.get(d, d), **{f'R_{t}': float(x) for t, x in zip(TYPES, R[d])}))
    return pd.DataFrame(rows)


def t6():
    txt = open(SRC).read(); sec = _section(txt, '## T6.', '## T7.')
    rows = []
    for line in sec.splitlines():
        m = re.match(r'\|\s*([CABD])\s*\|(.*)\|\s*$', line)
        if not m: continue
        c = [x.strip() for x in m.group(2).split('|')]
        rows.append({'ledger': m.group(1), 'ceiling_P@25': float(c[0]), 'ceiling_P@100': float(c[1]),
                     'ceiling_AP': float(c[2])})
    return pd.DataFrame(rows)
