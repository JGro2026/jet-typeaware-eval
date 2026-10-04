"""DeepSAD under the review protocol on the client ledgers (Table 8a): run B tables T1 (static detectors, core grid)
and T5 (review protocol) from the aggregated result table. Output: deepsad_real.csv."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2])); import paths  # noqa: E402
import re
import pandas as pd

MD = open(paths.CLIENT_RESULTS).read()

def block(section, table):
    sec = MD.split(f'## {section}.')[1].split('\n## ')[0]
    part = sec.split(f'**Table-{table}**')[1].split('**Table-')[0]
    rows = [l for l in part.splitlines() if l.startswith('|') and not l.startswith('|---')]
    hdr = [h.strip() for h in rows[0].strip('|').split('|')]
    out = {}
    for l in rows[1:]:
        c = [x.strip() for x in l.strip('|').split('|')]
        out[c[0]] = dict(zip(hdr[1:], c[1:]))
    return out

def mean_sd(s):
    m = re.match(r'([\d.]+)\s*±\s*([\d.]+)', s)
    return float(m.group(1)), float(m.group(2))

rows = []
for L in ['C', 'A', 'B', 'D']:
    t1 = block('T1', L); t5 = block('T5', L)
    ds = {k: mean_sd(t5['DeepSAD'][f'found@{k}']) for k in (25, 50, 75, 100)}
    ae25 = mean_sd(t1['AE-emb']['P@25']); ae100 = mean_sd(t1['AE-emb']['P@100'])
    static = {d: mean_sd(t1[d]['P@100'])[0] * 100 for d in ['IF', 'LOF', 'kNN', 'HBOS', 'ECOD', 'PCA', 'AE-pyod', 'VAE-pyod', 'AE-emb'] if d in t1}
    best = max(static.values()); bestd = [d for d, v in static.items() if abs(v - best) < 1e-9]
    rows.append(dict(ledger=L, ds_found25=ds[25][0], ds_found50=ds[50][0], ds_found75=ds[75][0],
                     ds_found100=ds[100][0], ds_found100_sd=ds[100][1],
                     aeemb_found25=round(ae25[0] * 25, 2), aeemb_found100=round(ae100[0] * 100, 1), aeemb_found100_sd=round(ae100[1] * 100, 1),
                     gain_vs_start=round(ds[100][0] - ae100[0] * 100, 1),
                     best_static=' / '.join(bestd), best_static_found100=round(best, 1),
                     diff_vs_best_static=round(ds[100][0] - best, 1),
                     ds_final_P100=t5['DeepSAD']['final P@100']))
R = pd.DataFrame(rows)
R.to_csv(paths.out_client('deepsad_real.csv'), index=False)
print(R.to_string(index=False))
# consistency: DeepSAD round 1 equals the AE-emb top 25 (its starting list)
print('round-1 check (DeepSAD found@25 vs 25 x AE-emb P@25):', list(zip(R.ds_found25, R.aeemb_found25)))

