# Client-ledger benchmark harness (frozen protocol)

This is the benchmark code that produced the client-ledger results of the paper, unchanged. `protocol.md` is the protocol
frozen before the experiments: typed injection of five anomaly types at three contamination and three difficulty
levels, the detectors with their settings, the review protocol with auditor feedback and the supervised reference.

`run_all.py` runs it from a working copy in `output/client_harness/` (the raw ledgers are linked into
`output/client_harness/data/raw/`), so this folder stays untouched and the code fingerprint that the harness writes into
every result (`code_hash` = `7835deeb3b75`, a hash of `src/*.py`) identifies the code version. The steps are:

```bash
python -m pytest -q tests                           # sanity tests
python -m src.audit                                 # data audit -> reports/wp0_data_audit.md
python -m src.run grid --part core_ABC              # core grid, ledgers A, B, C (5 seeds)
python -m src.run_feedback grid --part core_ABC     # review protocol on the core grid
python -m src.run grid --part sweep                 # contamination x difficulty sweep, 135 cells
python -m src.run grid --part core_D                # ledger D (5 seeds)
python -m src.run_feedback grid --part core_D
python -m src.run collect && python -m src.run_feedback collect
python -m src.analyze                               # reports/tables.md, read by analysis/client/
```

Each cell is skipped when its result exists. Cell names are `<ledger>_c<contamination>_<difficulty>_s<seed>`. The
protocol contains methods that the paper does not report (a transparent per-field reference DECOMP, the rank-average
ensemble AVG and linear feedback methods); they run with the benchmark and the analyses drop them. The runtime heading
of `src/analyze.py` always says "2 CPU cores", the hardware of the first execution.
