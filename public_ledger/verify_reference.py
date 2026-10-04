"""Compare the results of a rerun (work/public_ledger/results) with the results reported in the paper
(reference/public_ledger).

Usage: python3 public_ledger/verify_reference.py
Prints, per file, the largest absolute deviation between the reproduction and the reference.
Deterministic detectors should agree to floating-point precision on the same machine and to
the second decimal across machines; the neural detectors (AE, VAE, AE-emb, DeepSAD) and
everything derived from them vary with the CPU and the BLAS library.
"""
import os, sys, json, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dibpaths import RESULTS, REFERENCE as REF


def compare_csv(name, index_col=0):
    a, b = os.path.join(RESULTS, name), os.path.join(REF, name)
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f'{name}: skipped (not produced yet)'); return
    A = pd.read_csv(a, index_col=index_col); B = pd.read_csv(b, index_col=index_col)
    common_rows = A.index.intersection(B.index); common_cols = A.columns.intersection(B.columns)
    num = A.loc[common_rows, common_cols].select_dtypes('number')
    dif = (num - B.loc[common_rows, num.columns]).abs()
    worst = dif.stack().sort_values(ascending=False)
    print(f'{name}: {len(common_rows)} rows compared, max |deviation| = {worst.iloc[0]:.4f} '
          f'at {worst.index[0] if len(worst) else "-"}')


def compare_feedback():
    for f in sorted(glob.glob(os.path.join(REF, 'feedback_*_s*.json'))):
        name = os.path.basename(f)
        mine = os.path.join(RESULTS, name)
        if not os.path.exists(mine):
            print(f'{name}: skipped (not produced yet)'); continue
        A, B = json.load(open(mine)), json.load(open(f))
        d = max(abs(A[m][k] - B[m][k]) for m in B for k in B[m]
                if isinstance(B[m][k], (int, float)) and k != 'runtime_s' and m in A)
        print(f'{name}: max |deviation| = {d:.3f}')


if __name__ == '__main__':
    for n in ['metrics_mean.csv', 'metrics_sd.csv', 'within_population.csv', 'top100_fill.csv']:
        compare_csv(n)
    compare_feedback()
