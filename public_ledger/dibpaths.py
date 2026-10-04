"""Paths of the public-ledger pipeline, taken from the repository-wide paths.py."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

ROOT = str(P.ROOT)
DATA_CSV = str(P.PUBLIC_CSV)
SCORES = str(P.SCORES)          # per-line scores and trained AE-emb models
RESULTS = str(P.RESULTS_RUN)    # results of a rerun
REFERENCE = str(P.REF_PUBLIC)   # results of the run reported in the paper
SRC = os.path.dirname(os.path.abspath(__file__))

for _d in (SCORES, RESULTS):
    os.makedirs(_d, exist_ok=True)
