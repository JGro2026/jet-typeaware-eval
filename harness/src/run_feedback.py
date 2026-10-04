"""Feedback protocol runner (WP3) on top of the WP2 artifacts of a cell.

  python -m src.run_feedback cell --table C --c 0.015 --d medium --seed 0
  python -m src.run_feedback grid --part core
  python -m src.run_feedback collect

Reads results/scores/<cell>.parquet, results/manifests/<cell>.parquet and
results/models/<cell>_ae_emb.pt; writes results/feedback/<cell>.parquet
(long: table, c, d, seed, method, round, reviews_cumulative, anomalies_found,
p_at_25, p_at_50, p_at_100, runtime_s, code_hash) and a JSON with details.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from .detectors import UNSUPERVISED, context_matrix, DetectorOutput
from .encode import encode_emb
from .feedback import DeepSAD, GreedyLR, LinTS, Oracle, StaticAvg, run_rounds
from .inject import inject
from .load import code_hash
from .run import CONFIG, RESULTS, _table, cell_name, grid_cells

METHODS = ["STATIC-AVG", "LinTS", "GREEDY-LR", "LinTS-spec", "GREEDY-LR-spec", "DeepSAD"]


def feedback_settings() -> dict:
    """Method settings from configs/grid.yaml (section `feedback`)."""
    cfg = yaml.safe_load(CONFIG.read_text())["feedback"]
    return {
        "lints": dict(cfg.get("lints", {})),
        "greedy_lr": dict(cfg.get("greedy_lr", {})),
        "lints_spec": dict(cfg.get("lints_spec", {})),
        "greedy_lr_spec": dict(cfg.get("greedy_lr_spec", {})),
        "deepsad": dict(cfg.get("deepsad", {})),
        "deepsad_D": dict(cfg.get("deepsad_D", {})),
    }


def run_feedback_cell(table: str, c: float, d: str, seed: int, methods: list[str] | None = None,
                      overwrite: bool = False, verbose: bool = True, deepsad_kw: dict | None = None) -> dict:
    import torch

    methods = list(methods or METHODS)
    name = cell_name(table, c, d, seed)
    out_dir = RESULTS / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"[skip] feedback {name} exists")
        return {"cell": name, "skipped": True}

    scores = pd.read_parquet(RESULTS / "scores" / f"{name}.parquet")
    manifest = pd.read_parquet(RESULTS / "manifests" / f"{name}.parquet")
    assert (scores["row_id"].to_numpy() == manifest["row_id"].to_numpy()).all()
    labels = manifest["label"].to_numpy().astype(int)
    outputs = {k: DetectorOutput(k, scores[k].to_numpy(), 0.0, "") for k in UNSUPERVISED}
    X = context_matrix(outputs)
    chash = code_hash()
    cfg = feedback_settings()

    records, details = [], {}
    for m in methods:
        t0 = time.perf_counter()
        oracle = Oracle(labels)
        if m == "STATIC-AVG":
            method = StaticAvg(scores["AVG"].to_numpy())
        elif m == "LinTS":
            method = LinTS(X, seed, **cfg["lints"])
        elif m == "GREEDY-LR":
            method = GreedyLR(X, seed, **cfg["greedy_lr"])
        elif m == "LinTS-spec":
            method = LinTS(X, seed, **cfg["lints_spec"]); method.name = m
        elif m == "GREEDY-LR-spec":
            method = GreedyLR(X, seed, **cfg["greedy_lr_spec"]); method.name = m
        elif m == "DeepSAD":
            inj = inject(_table(table), table, c, d, seed)
            assert inj.meta["dataset_hash"] == manifest["dataset_hash"].iloc[0], "re-injection hash mismatch"
            emb = encode_emb(inj.data)
            ckpt = torch.load(RESULTS / "models" / f"{name}_ae_emb.pt", weights_only=False)
            assert ckpt["vocab_sizes"] == emb.vocab_sizes
            kw = dict(cfg["deepsad"])
            if table == "D":
                kw.update(cfg["deepsad_D"])
            kw.update(deepsad_kw or {})
            method = DeepSAD(emb, ckpt["state_dict"], scores["AE-emb"].to_numpy(), seed, **kw)
        else:
            raise ValueError(m)
        recs = run_rounds(method, oracle)
        rt = time.perf_counter() - t0
        for r in recs:
            records.append(dict(table=table, c=float(c), d=d, seed=int(seed), method=m, round=r.round,
                                reviews_cumulative=r.reviews_cumulative, anomalies_found=r.anomalies_found,
                                p_at_25=r.p_at_25, p_at_50=r.p_at_50, p_at_100=r.p_at_100,
                                runtime_s=rt, code_hash=chash))
        details[m] = {"runtime_s": rt, "found_per_round": [r.anomalies_found for r in recs],
                      "final": {k: getattr(recs[-1], k) for k in ("p_at_25", "p_at_50", "p_at_100")},
                      "remaining_anomalies": recs[-1].extra.get("remaining_anomalies")}
        if m.startswith(("GREEDY-LR", "LinTS")):
            details[m]["settings"] = method.settings
        if m.startswith("GREEDY-LR"):
            details[m]["fallback_rounds"] = method.fallback_rounds
        if m == "DeepSAD":
            details[m]["round_runtimes"] = method.round_runtimes
            details[m]["params"] = {k: getattr(method, k) for k in ("eta", "lr", "wd", "epochs", "bs")}
        if verbose:
            print(f"  {m:<11s} {rt:8.1f}s  found@25/50/75/100 = {details[m]['found_per_round']}  "
                  f"final P@25={recs[-1].p_at_25:.2f} P@100={recs[-1].p_at_100:.2f}", flush=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)
    (out_dir / f"{name}.json").write_text(json.dumps(details, indent=1, default=str))
    if verbose:
        print(f"[done] feedback {name}, n_anom={labels.sum()}")
    return details


def collect() -> None:
    parts = sorted((RESULTS / "feedback").glob("*.parquet"))
    if not parts:
        print("no feedback results yet")
        return
    f = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    f.to_parquet(RESULTS / "feedback.parquet", index=False)
    print(f"feedback.parquet: {len(f)} rows, {f[['table','c','d','seed']].drop_duplicates().shape[0]} cells")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("cell")
    pc.add_argument("--table", required=True)
    pc.add_argument("--c", type=float, required=True)
    pc.add_argument("--d", required=True)
    pc.add_argument("--seed", type=int, required=True)
    pc.add_argument("--methods", nargs="*", default=None)
    pc.add_argument("--overwrite", action="store_true")
    pg = sub.add_parser("grid")
    pg.add_argument("--part", default="core")
    pg.add_argument("--overwrite", action="store_true")
    sub.add_parser("collect")
    a = ap.parse_args(argv)
    if a.cmd == "cell":
        run_feedback_cell(a.table, a.c, a.d, a.seed, a.methods, a.overwrite)
    elif a.cmd == "grid":
        cfg = yaml.safe_load(CONFIG.read_text())
        for cell in grid_cells(a.part, cfg):
            run_feedback_cell(cell["table"], cell["c"], cell["d"], cell["seed"],
                              cfg[a.part].get("feedback", METHODS), a.overwrite)
        collect()
    elif a.cmd == "collect":
        collect()


if __name__ == "__main__":
    main()
