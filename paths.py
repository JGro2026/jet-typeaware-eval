"""Every location the code reads or writes, in one place.

Inputs are raw data only: the public ledger (journal_entries_4203.csv, Zenodo) and, optionally, the four client ledgers
(tableA.csv to tableD.csv). Everything else is produced by the code under output/. Paths can be overridden with
environment variables, so the repository runs from any folder on Linux, macOS and Windows:

    JET_PUBLIC_CSV    the public ledger                                (default: data/journal_entries_4203.csv)
    JET_CLIENT_DATA   folder with tableA.csv, tableB.csv, tableC.csv, tableD.csv  (default: data/client/)
    JET_OUTPUT        folder for everything the code produces          (default: output/)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_env = os.environ.get

# ---------------------------------------------------------------- code
TYPEAWARE = ROOT / 'typeaware'               # the type-aware measures (fsr.py, measures.py)
PIPELINE = ROOT / 'public_ledger'            # public-ledger benchmark pipeline
HARNESS = ROOT / 'harness'                   # client-ledger benchmark harness (frozen protocol)
ANALYSIS_PUBLIC = ROOT / 'analysis' / 'public'
ANALYSIS_CLIENT = ROOT / 'analysis' / 'client'
PAPER = ROOT / 'paper'
MANUSCRIPT = PAPER / 'manuscript'            # hand-written sections, figure floats, class files, bibliography
SUPPLEMENT = PAPER / 'supplement'            # hand-written supplement text
METHOD_TEXT = PAPER / 'text' / 'method_tables.tex'   # descriptive text of types, detectors and field mapping
PROTOCOL = HARNESS / 'protocol.md'           # frozen benchmark protocol

# ---------------------------------------------------------------- raw data (never part of the repository)
PUBLIC_CSV = Path(_env('JET_PUBLIC_CSV') or ROOT / 'data' / 'journal_entries_4203.csv')
CLIENT_DATA = Path(_env('JET_CLIENT_DATA') or ROOT / 'data' / 'client')
CLIENT_FILES = [CLIENT_DATA / f'table{t}.csv' for t in 'ABCD']
# SHA-256 of the files used for the paper; run_all.py warns when a file differs
SHA256 = {
    'journal_entries_4203.csv': 'ec7a9424645ec2446387aa994486844dfb13b11762a5a8c488964b5850ae0da3',
    'tableA.csv': 'db227c4dac6df559ad4d5bd22d90192b2f3ca5aec558da963d266413cf3ec9e1',
    'tableB.csv': '0b4bd7a016ab7ece616347cf207e05ee9e75f02a6c67db15c45d512f6afc6321',
    'tableC.csv': '32de10d4693aa449792c42275c06eff1b6ef701e5c50375f5963a9d0d8f40c10',
    'tableD.csv': '132d852542f907f15184b6d9727214ac3c404a14d394b9404a41b0d4a8ee3690',
}

# ---------------------------------------------------------------- run configuration of the public ledger
PUBLIC_SEEDS = (0, 1, 2)                     # seeds of the stochastic detectors, the supervised reference and DeepSAD
PUBLIC_K = 1000                              # size-scaled budget (about 1 percent of the entries)

# ---------------------------------------------------------------- everything the code produces
OUTPUT = Path(_env('JET_OUTPUT') or ROOT / 'output')
SCORES = OUTPUT / 'public_ledger' / 'scores'          # per-line detector scores, trained AE-emb models
RESULTS_RUN = OUTPUT / 'public_ledger' / 'results'    # metrics and review-protocol results of the public ledger
CLIENT_RUN = OUTPUT / 'client_harness'                # working copy of the harness with its results and reports
CLIENT_RESULTS = CLIENT_RUN / 'reports' / 'tables.md' # aggregated client-ledger results written by the harness
CLIENT_AUDIT = CLIENT_RUN / 'reports' / 'wp0_data_audit.md'
OUT_PUBLIC = OUTPUT / 'analysis' / 'public'           # public-ledger analyses
OUT_CLIENT = OUTPUT / 'analysis' / 'client'           # analyses of the client-ledger results
TABLES = OUTPUT / 'paper' / 'tables'                  # generated main-text and appendix tables
SUP_TABLES = OUTPUT / 'paper' / 'supplement_tables'   # generated supplementary tables
FIGURES = OUTPUT / 'paper' / 'figures'                # generated figures
BUILD = OUTPUT / 'paper' / 'build'                    # assembled manuscript and supplement
ANON = OUTPUT / 'paper' / 'anon'                      # anonymized manuscript, title page and supplement

# ---------------------------------------------------------------- published values, for run_all.py check
REFERENCE = ROOT / 'reference'
REF_PAPER = REFERENCE / 'paper'                       # the tables as printed in the paper and its supplement
REF_PUBLIC = REFERENCE / 'public_ledger'              # public-ledger results of the run reported in the paper
REF_OUT_PUBLIC = REFERENCE / 'analysis_public'        # public-ledger analysis outputs of that run


def public_result(name):
    return RESULTS_RUN / name


def has_client_data():
    return all(f.exists() for f in CLIENT_FILES)


def out_public(name):
    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    return OUT_PUBLIC / name


def out_client(name):
    OUT_CLIENT.mkdir(parents=True, exist_ok=True)
    return OUT_CLIENT / name


def find(name):
    """An analysis output, wherever the analysis that writes it puts it."""
    for d in (OUT_PUBLIC, OUT_CLIENT):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f'{name}: not in {OUT_PUBLIC} or {OUT_CLIENT}; run the earlier stages first (python run_all.py)')
