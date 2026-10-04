"""Parser for the client-ledger result tables of the two executions: results_runA_partial.md (first execution)
and results_runB_complete.md (second, complete execution, reported in the paper)."""
import re
import numpy as np, pandas as pd

FILES = {}   # set by compare_client_runs.py: 'A' and 'B' -> the tables.md of the two executions
LEDGER = {'Table-C': 'C', 'Table-A': 'A', 'Table-B': 'B', 'Table-D': 'D'}


def _num(tok):
    tok = tok.strip()
    if tok in ('', 'nan', 'n/a', 'best'): return np.nan
    try: return float(tok)
    except ValueError: return np.nan


def parse_cell(v):
    """Return (mean, sd). '0.04 ± 0.03' -> (0.04, 0.03); '0.04' -> (0.04, nan)."""
    v = v.strip()
    if '±' in v:
        a, b = v.split('±'); return _num(a), _num(b)
    return _num(v), np.nan


def _tables(block):
    """Yield (subtitle, DataFrame of raw strings) for every markdown table in a block."""
    sub = None; hdr = None; rows = []
    for line in block.split('\n'):
        s = line.strip()
        m = re.match(r'^\*\*(.+?)\*\*', s)
        if m:
            if hdr and rows: yield sub, pd.DataFrame(rows, columns=hdr); hdr, rows = None, []
            sub = m.group(1); continue
        if s.startswith('|'):
            cells = [c.strip() for c in s.strip('|').split('|')]
            if set(''.join(cells)) <= set('-: '):
                continue
            if hdr is None:
                hdr = cells
            else:
                while len(cells) > len(hdr):      # a literal '|' inside the first cell, e.g. -log P(g|a)
                    cells = ['|'.join(cells[:2])] + cells[2:]
                rows.append(cells)
        elif hdr and rows:
            yield sub, pd.DataFrame(rows, columns=hdr); hdr, rows = None, []
    if hdr and rows: yield sub, pd.DataFrame(rows, columns=hdr)


def parse(run):
    txt = open(FILES[run]).read()
    parts = re.split(r'\n## ', txt)
    out = {}
    for p in parts:
        key = p.split('.')[0].strip()
        if not re.match(r'^(T\d|Runtimes)', key): continue
        key = 'Runtimes' if key.startswith('Runtimes') else key
        out[key] = list(_tables(p))
    return out


def long_metric(run, sec, value_cols=None, row_key='Detector'):
    """Flatten a section into rows (ledger, row, column, mean, sd)."""
    res = []
    for sub, T in parse(run)[sec]:
        led = None
        if sub:
            m = re.match(r'^(Table-[ABCD])', sub)
            if m: led = LEDGER[m.group(1)]
        keycol = T.columns[0]
        for _, r in T.iterrows():
            for c in T.columns[1:]:
                if value_cols and c not in value_cols: continue
                mu, sd = parse_cell(r[c])
                res.append({'ledger': led, 'row': r[keycol], 'metric': c, 'mean': mu, 'sd': sd})
    return pd.DataFrame(res)
