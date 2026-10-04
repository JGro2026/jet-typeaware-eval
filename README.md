# jet-typeaware-eval

Reproduction code for **"Which anomalies does an auditor see? Type-aware evaluation of journal entry anomaly detectors under a review budget"** (anonymized for review).
The repository takes the raw ledgers and produces every number, table and figure of the paper and its supplement with one command. It contains:
- the type-aware measures (per-type recall against the budget ceiling, type coverage TC@k, fair-share type recall FSR@k, first-hit ranks);
- the two benchmarks: four client ledgers with injected, typed anomalies and a public synthetic ledger;
- the analyses;
- the generators of all tables and figures.

Nothing is taken from an earlier run. The only inputs are the raw data files.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # Python 3.11, CPU only
# put journal_entries_4203.csv into data/ (see data/README.md)
python run_all.py --quick                               # smoke test of every stage, a few minutes
python run_all.py                                       # the full reproduction
```

The client ledgers are confidential and not part of this repository. Place `tableA.csv` to `tableD.csv` in `data/client/`, or point the variable `JET_CLIENT_DATA` to their folder, and `run_all.py` runs the client-ledger part as well. Without them it runs the public-ledger part and says what it skipped.

## What `run_all.py` does

| Stage | What | Time on 2 CPU cores |
|---|---|---|
| `test` | Unit tests of the measures (`typeaware/test_measures.py`) | seconds |
| `public` | Public-ledger benchmark (`public_ledger/`): nine unsupervised detectors, the supervised LightGBM reference, the DeepSAD review protocol and its no-feedback control, three seeds each. Then the public-ledger analyses (`analysis/public/`). | about 3 h |
| `client` | Client-ledger benchmark with the frozen protocol (`harness/`): data audit, typed injection, detectors, review protocol and supervised reference on the core grid, the contamination and difficulty sweep. Then the client-ledger analyses (`analysis/client/`). Runs only with the client ledgers. | about 8 h for ledgers A, B, C and 12 h for ledger D. With `--parallel` the two parts run side by side. |
| `paper` | All tables and figures (`paper/make_tables.py`, `paper/make_figures.py`), the manuscript and supplement (`paper/assemble.py`) and the anonymized versions (`paper/make_variants.py`). Compiled with LaTeX when `pdflatex` is installed. | minutes |
| `check` | Compares everything with the values printed in the paper (`reference/`) and writes `output/check_report.md` | seconds |

All results go to `output/` (`output_quick/` for `--quick`), with one log per step in `output/logs/`.

**Resuming.** Every step is skipped when its outputs exist, and the benchmark harness skips finished cells, so an interrupted run resumes where it stopped. `--force` reruns everything. `--only public`, `--only client-analyses,paper,check` and similar select stages.

## What to expect

- **Deterministic detectors.** IF, LOF, kNN, HBOS, ECOD, PCA and the supervised LightGBM reference reproduce the published values. So do the typed injection (the harness checks every dataset hash) and every quantity computed only from them.
- **Neural models.** AE, VAE, AE-emb and DeepSAD depend on the CPU and the BLAS library. On another machine their values move, as the paper reports in Section 4.5. The public-ledger rerun on a second machine is included in `reference/public_ledger_second_machine/` as an example.
- **Runtimes.** The runtime table is machine-specific by nature.
- **Where differences show.** `output/check_report.md` lists, for every table, whether it is identical to the paper and which rows differ.

**Verification before release.** With the result files of the runs reported in the paper, this code regenerates:
- all nineteen other tables of the paper and its supplement identically; the runtime table is the exception, since its values depend on the machine;
- a manuscript source byte-identical to the submitted one.

A fresh run of the client-ledger harness on the same data reproduces the dataset hashes and every deterministic detector.

## Data

| File | Where | Source |
|---|---|---|
| `journal_entries_4203.csv` | `data/` or `JET_PUBLIC_CSV` | Zenodo, https://doi.org/10.5281/zenodo.22014653, CC BY 4.0 |
| `tableA.csv` to `tableD.csv` | `data/client/` or `JET_CLIENT_DATA` | Client ledgers provided by DATEV eG under a research agreement; not public |

`run_all.py` checks the SHA-256 of every input against the files used for the paper (`paths.py`) and warns when a file differs.

## Repository layout

| Path | Content |
|---|---|
| `run_all.py` | The single entry point |
| `paths.py` | Every input and output location; override with `JET_PUBLIC_CSV`, `JET_CLIENT_DATA`, `JET_OUTPUT` |
| `typeaware/` | The measures (`fsr.py`: water-filling caps and FSR; `measures.py`: P@k, R_t@k with ceiling, TC@k, FSR@k, first-hit ranks) and their tests |
| `public_ledger/` | Public-ledger pipeline: data mapping, detectors, supervised reference, DeepSAD review protocol, no-feedback control, metrics |
| `harness/` | Client-ledger benchmark harness of the frozen protocol (`harness/protocol.md`), used unchanged; see `harness/README.md` |
| `analysis/public/`, `analysis/client/` | Type-aware analyses of the two benchmarks |
| `paper/` | Table and figure generators, manuscript and supplement sources, assembler; see `paper/README.md` |
| `reference/` | Values reported in the paper, used only by the `check` stage |
| `tools/` | `check_reproduction.py`, `quick_check.py`, `compare_client_runs.py` (compares two client-ledger runs cell by cell, as in Supplementary Section S3) and `make_review_copy.py` (anonymized copy for double-blind review) |

## Where each result comes from

| Paper | Generator (`paper/`) | Analyses |
|---|---|---|
| Table 2 (data) | `make_tables.tab_data` | harness data audit, `client/duplicates.py`, `client/datev_typeaware.py`, public ledger profile |
| Table 3 (types, markings) | `make_tables.tab_types` | `client/a2_bounds.py`, `public/a6_public_tables.py`, `harness/protocol.md` |
| Table 4 (detectors) | `make_tables.tab_detectors` | `harness/protocol.md`, `paper/text/method_tables.tex` |
| Table 5 (hit-based view) | `make_tables.tab_conventional` | `client/a7_datev_tables.py`, `public/a6_public_tables.py` |
| Table 6 (type-aware view) | `make_tables.tab_typeaware` | `client/a3_orderings.py` (with `a2_bounds.py`, `public/dib_typeaware.py`) |
| Table 7 (FSR and macro-averaged recall) | `make_tables.tab_macro` | `client/a11_client_macro.py`, `public/a11_public_macro.py` |
| Table 8 (review protocol) | `make_tables.tab_feedback` | `client/deepsad_client.py`, `public/deepsad_public.py`, `public/deepsad_control.py` |
| Tables B.1, C.1, D.1 | `make_tables.tab_app_deepsad_seeds`, `tab_app_bootstrap`, `tab_app_weights` | `public/deepsad_*.py`, `public/a5_bootstrap.py`, `client/a10_weights.py`, `client/a10b_weights_random.py`, `public/a10c_public_weights.py` |
| Supplementary Tables S1 to S11 | `make_tables.tabS_*` | as listed in `output/paper/tables/README.md` after a run |
| Figures 1 to 6, C.1 | `make_figures.fig_framework`, `fig_pertype_real`, `fig_hits_vs_types`, `fig_budget_public`, `fig_public_types`, `fig_sweep`, `fig_bootstrap` | `client/a9_fig3_data.py`, `client/a8_friedman_check.py`, `public/a4_budget_profile.py`, `public/a5_bootstrap.py`, `public/a6_public_tables.py` |

Table 1 (the measures compared) and Table A.1 (notation) contain no data and are part of the manuscript text.

## Notes

- **Thread settings.** The public-ledger steps run with one BLAS thread, as for the paper, and the harness sets its own thread counts. Changing them can change the neural results slightly.
- **Harness fingerprint.** The harness records a fingerprint of its code in every result (`code_hash`, here `7835deeb3b75`). `run_all.py` runs it from a copy in `output/client_harness/` without modifying it.
- **Not reported in the paper.** The frozen protocol also contains methods the paper leaves out: a transparent per-field reference, a rank-average ensemble and linear feedback methods. They run with the benchmark, and the analyses drop them.

## Note on this copy

This anonymized copy for review leaves out the manuscript and supplement sources, so the `paper` stage writes all tables and figures but does not assemble the manuscript.
