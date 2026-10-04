# Reference values (used only by `run_all.py check`)

| Folder | Content |
|---|---|
| `paper/tables/`, `paper/supplement_tables/` | The tables as printed in the paper and its supplement |
| `public_ledger/` | Public-ledger results of the run reported in the paper (metrics, review protocol, no-feedback control) |
| `analysis_public/` | Public-ledger analysis outputs of that run |
| `public_ledger_second_machine/` | The same public-ledger benchmark rerun on a second machine (AMD EPYC 7302, Python 3.12, same package versions): results, environment and `pip freeze`. Deterministic detectors agree; the neural detectors and DeepSAD differ, which illustrates the cross-machine variation the paper reports. |

Nothing in this folder is read by the analyses; they start from the raw data.
