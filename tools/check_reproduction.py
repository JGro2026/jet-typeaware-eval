"""Compare a full run with the values reported in the paper and write output/check_report.md.

1. Public-ledger results (metrics, review protocol) against reference/public_ledger.
2. Public-ledger analysis outputs against reference/analysis_public.
3. Every generated table of the paper and its supplement against reference/paper (the tables as printed).

Deterministic detectors (IF, LOF, kNN, HBOS, ECOD, PCA and the LightGBM reference) reproduce exactly. The neural
detectors (AE, VAE, AE-emb) and DeepSAD depend on the CPU and the BLAS library and vary across machines (the paper
reports this); the runtime table is machine-specific by nature. Differences are listed, not treated as errors.
"""
import re, sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402

lines = ['# Reproduction check', '', f'Output folder: `{paths.OUTPUT}`', '']
NEURAL = ('AE', 'VAE', 'AE-emb', 'AE-pyod', 'VAE-pyod', 'DeepSAD', 'AVG')


def csv_dev(a, b, index_col=None):
    A = pd.read_csv(a, index_col=index_col); B = pd.read_csv(b, index_col=index_col)
    if A.shape != B.shape:
        return None, f'shape {A.shape} vs {B.shape}'
    num = [c for c in A.columns if pd.api.types.is_numeric_dtype(A[c]) and c in B.columns and pd.api.types.is_numeric_dtype(B[c])
           and 'runtime' not in c]  # runtimes are machine-specific
    if not num:
        return 0.0, 'identical' if A.astype(str).equals(B.astype(str)) else 'text differs'
    d = (A[num].astype(float) - B[num].astype(float)).abs().max().max()
    return float(np.nan_to_num(d)), ''


# 1. public-ledger results
lines += ['## 1. Public-ledger results', '', '| File | max abs deviation |', '|---|---|']
for f in sorted(paths.REF_PUBLIC.glob('*.csv')):
    g = paths.RESULTS_RUN / f.name
    if not g.exists():
        lines.append(f'| {f.name} | not produced |'); continue
    d, note = csv_dev(g, f, index_col=0)
    lines.append(f'| {f.name} | {"n/a " + note if d is None else f"{d:.4f}"} |')
for f in sorted(paths.REF_PUBLIC.glob('feedback_deepsad*.json')):
    g = paths.RESULTS_RUN / f.name
    if not g.exists():
        lines.append(f'| {f.name} | not produced |'); continue
    A, B = json.load(open(g)), json.load(open(f))
    m = next(iter(B))
    lines.append(f'| {f.name} | found@100 {A[m]["found@100"]} (paper {B[m]["found@100"]}) |')
if (paths.RESULTS_RUN / 'metrics_mean.csv').exists():
    A = pd.read_csv(paths.RESULTS_RUN / 'metrics_mean.csv', index_col=0)
    B = pd.read_csv(paths.REF_PUBLIC / 'metrics_mean.csv', index_col=0)
    det = [d for d in ('IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'LGBM-ceiling') if d in A.index]
    num = A.columns.intersection(B.columns)
    dd = (A.loc[det, num] - B.loc[det, num]).abs().max().max()
    lines += ['', f'Deterministic detectors and the supervised reference: max abs deviation {dd:.2e} '
              f'({"exact" if dd < 1e-9 else "NOT exact"}).']

# 2. public-ledger analyses
lines += ['', '## 2. Public-ledger analyses', '', '| File | max abs deviation |', '|---|---|']
for f in sorted(paths.REF_OUT_PUBLIC.glob('*.csv')):
    g = paths.OUT_PUBLIC / f.name
    if not g.exists():
        lines.append(f'| {f.name} | not produced |'); continue
    d, note = csv_dev(g, f)
    lines.append(f'| {f.name} | {"n/a " + note if d is None else f"{d:.4f}"} |')

# 3. tables of the paper
num_re = re.compile(r'(?<![A-Za-z{\\])-?\d[\d,]*\.?\d*')


def table_rows(t):
    body = re.sub(r'\\caption\{.*?\}\n\\label', r'\\label', t, flags=re.S)
    return [r for r in body.split('\n') if '&' in r]


lines += ['', '## 3. Tables of the paper', '',
          '| Table | Result | Rows with different numbers |', '|---|---|---|']
n_ident = 0
for sub, gen in (('tables', paths.TABLES), ('supplement_tables', paths.SUP_TABLES)):
    for f in sorted((paths.REF_PAPER / sub).glob('*.tex')):
        g = gen / f.name
        if not g.exists():
            lines.append(f'| {f.stem} | not produced | |'); continue
        a, b = g.read_text(), f.read_text()
        if a == b:
            n_ident += 1; lines.append(f'| {f.stem} | identical | |'); continue
        ra, rb = table_rows(a), table_rows(b)
        diff = []
        for x, y in zip(ra, rb):
            if num_re.findall(x) != num_re.findall(y):
                diff.append(x.split('&')[0].strip().replace('\\textit{', '').replace('}', ''))
        kind = 'machine-specific (runtimes)' if 'runtime' in f.stem else (
            'numbers differ' if diff else 'text differs, numbers identical')
        lines.append(f'| {f.stem} | {kind} | {", ".join(diff[:12])}{" ..." if len(diff) > 12 else ""} |')
lines += ['', f'{n_ident} tables identical to the paper. Rows that differ should belong to the neural detectors '
          f'({", ".join(NEURAL)}), DeepSAD or quantities derived from them; see the paper, Section 4.5, on their '
          'variation across machines.']
out = paths.OUTPUT / 'check_report.md'
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines)); print(f'\nwritten to {out}')
