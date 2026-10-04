# Data

The repository contains no data. `run_all.py` reads two kinds of input.

## Public ledger (required)

Download `journal_entries_4203.csv` from Zenodo and place it in this folder (or set `JET_PUBLIC_CSV`):

- Gronewald J, Stephan S, Rombach A, Fettke P. DIB General Ledger Data. Zenodo. https://doi.org/10.5281/zenodo.22014653 (CC BY 4.0)
- Described in: Generating synthetic journal entries for audit analytics: method and dataset. Data in Brief 69 (2026) 113257. https://doi.org/10.1016/j.dib.2026.113257

The file used for the paper has 256,232 posting lines in 102,429 journal entries, 31,127,872 bytes and SHA-256
`ec7a9424645ec2446387aa994486844dfb13b11762a5a8c488964b5850ae0da3` (`sha256sum data/journal_entries_4203.csv`).

## Client ledgers (optional, confidential)

The four client ledgers were provided by DATEV eG under a research agreement and cannot be shared. With access, put
`tableA.csv`, `tableB.csv`, `tableC.csv` and `tableD.csv` into `data/client/` (or set `JET_CLIENT_DATA` to their folder).
They are the general-ledger exports described in the paper (Section 4.1): 5,080, 17,505, 3,461 and 279,975 postings with
the eleven columns that `harness/src/load.py` enforces. SHA-256 of the files used for the paper:

| File | SHA-256 |
|---|---|
| tableA.csv | db227c4dac6df559ad4d5bd22d90192b2f3ca5aec558da963d266413cf3ec9e1 |
| tableB.csv | 0b4bd7a016ab7ece616347cf207e05ee9e75f02a6c67db15c45d512f6afc6321 |
| tableC.csv | 32de10d4693aa449792c42275c06eff1b6ef701e5c50375f5963a9d0d8f40c10 |
| tableD.csv | 132d852542f907f15184b6d9727214ac3c404a14d394b9404a41b0d4a8ee3690 |

`.gitignore` keeps every file in this folder out of version control.
