"""Duplicate structure of the four client ledgers, measured on the raw files (Table 2): a duplicate row repeats all
eleven feature fields of another row. Output: dup_group_stats.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import sys
import pandas as pd
sys.path.insert(0, str(paths.HARNESS))
from src.load import load_table, FEATURE_COLUMNS  # noqa: E402  (the harness loader with its enforced schema)

rows = []
for t in ['C', 'A', 'B', 'D']:
    df = load_table(t, paths.CLIENT_DATA)
    key = pd.util.hash_pandas_object(df[FEATURE_COLUMNS], index=False)
    size = key.map(key.value_counts())
    g = key.value_counts(); g = g[g >= 2]
    rows.append(dict(ledger=t, n=len(df), rows_repeating_earlier=int(key.duplicated().sum()),
                     rows_in_dup_groups=int((size >= 2).sum()), share_in_dup_groups=round((size >= 2).mean(), 4),
                     groups=int(len(g)), max_group=int(g.max()), rows_in_groups_ge6=int((size >= 6).sum()),
                     share_ge6=round((size >= 6).mean(), 4), rows_in_groups_ge21=int((size >= 21).sum()),
                     share_ge21=round((size >= 21).mean(), 4)))
D = pd.DataFrame(rows); D.to_csv(paths.out_client('dup_group_stats.csv'), index=False)
print(D.to_string(index=False))
