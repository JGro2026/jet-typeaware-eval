"""Single-detector worker used by run_cell for Table-D: runs one detector for
one cell in a fresh process and writes the per-detector checkpoint
results/partial/<cell>/<detector>.npz (plus the AE-emb state dict).

  python -m src.worker --table D --c 0.015 --d medium --seed 1 --detector AE-emb [--save-model]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .detectors import run_ae_emb, run_pyod
from .encode import encode_base, encode_emb
from .inject import inject
from .load import ROOT, load_table

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--d", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--subsample", action="store_true")
    a = ap.parse_args()
    from .run import AE_EMB_LARGE, cell_name

    name = cell_name(a.table, a.c, a.d, a.seed)
    part = RESULTS / "partial" / name
    part.mkdir(parents=True, exist_ok=True)
    inj = inject(load_table(a.table), a.table, a.c, a.d, a.seed)
    (part / "dataset_hash.txt").write_text(inj.meta["dataset_hash"])
    big = a.table == "D"
    if a.detector == "AE-emb":
        emb = encode_emb(inj.data)
        kw = dict(AE_EMB_LARGE) if big else {}
        save = str(RESULTS / "models" / f"{name}_ae_emb.pt") if a.save_model else None
        det = run_ae_emb(emb, a.seed, save_path=save, **kw)
    else:
        base = encode_base(inj.data)
        det = run_pyod(a.detector, base, a.seed, "E_base", subsample=a.subsample, big=big)
    info = {k: v for k, v in det.info.items() if k != "component_percentiles"}
    np.savez(part / f"{det.name}.npz", scores=det.scores, runtime_s=det.runtime_s, encoding=det.encoding,
             info=json.dumps(info, default=str))
    print(f"[worker] {name} {det.name} {det.runtime_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
