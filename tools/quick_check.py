"""Quick check (run_all.py --quick): the deterministic public-ledger detectors computed in the quick run reproduce
the published values exactly (P@25, P@50, P@100, P@1000, AP), which confirms the data file, the encodings and the
installation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
sys.path.insert(0, str(paths.PIPELINE))
import numpy as np, pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from dib_data import load, entries, entry_scores  # noqa: E402

df = load(); e = entries(df); y = e.label.values; eid = df.eid.values
ref = pd.read_csv(paths.REF_PUBLIC / 'metrics_mean.csv', index_col=0)
worst = 0.0
for d in ('HBOS', 'ECOD', 'PCA'):
    sc = entry_scores(np.load(paths.SCORES / f'{d}_s0.npy'), eid); o = np.argsort(-sc, kind='stable')
    got = {f'P@{k}': y[o[:k]].mean() for k in (25, 50, 100, 1000)}; got['AP'] = average_precision_score(y, sc)
    dev = max(abs(got[k] - ref.loc[d, k]) for k in got); worst = max(worst, dev)
    print(f'{d:5s} ' + '  '.join(f'{k} {got[k]:.4f} (paper {ref.loc[d, k]:.4f})' for k in got) + f'  max |diff| {dev:.1e}')
if worst > 1e-9:
    sys.exit(f'deterministic detectors differ from the published values (max |diff| {worst:.2e})')
print('quick check passed: deterministic public-ledger detectors reproduce the published values exactly')
