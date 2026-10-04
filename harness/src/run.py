"""Per-cell pipeline and grid driver.

  python -m src.run cell --table C --c 0.015 --d medium --seed 0 [--ablation] [--ceiling] [--save-model]
  python -m src.run grid --part core|sweep|sweep_D [--dry-run]
  python -m src.run collect

Per cell the pipeline writes
  results/manifests/<cell>.parquet   Section 5 manifest, one row per table row
  results/scores/<cell>.parquet      wide: row_id plus one column per detector
  results/scores/<cell>_decomp_components.parquet   the five DECOMP component percentiles
  results/metrics/<cell>.parquet     long: table, c, d, seed, detector, encoding, metric, value, runtime_s, code_hash
  results/models/<cell>_ae_emb.pt    AE-emb state dict (core grid)
and `collect` concatenates results/metrics/*.parquet into results/metrics.parquet
(ceiling rows go to results/ceiling.parquet).
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .detectors import (ABLATION, ALL_DETECTORS, CLASSICAL, UNSUPERVISED,
                        DetectorOutput, run_ae_emb, run_avg, run_ceiling,
                        run_decomp, run_pyod)
from .encode import encode_base, encode_emb, encode_freq
from .evaluate import check_scores, compute_metrics
from .inject import inject
from .load import KEY, ROOT, code_hash, load_table

RESULTS = ROOT / "results"
CONFIG = ROOT / "configs" / "grid.yaml"

AE_EMB_LARGE = {"batch_size": 256, "epochs": 30}   # Table-D deviation, Section 3.4


def cell_name(table: str, c: float, d: str, seed: int) -> str:
    return f"{table}_c{c:.3f}_{d}_s{seed}"


@lru_cache(maxsize=4)
def _table(table: str) -> pd.DataFrame:
    return load_table(table)


def metrics_rows(table, c, d, seed, det: DetectorOutput, labels, types, chash, dhash) -> list[dict]:
    m = compute_metrics(det.scores, labels, types)
    rows = []
    for k, v in m.items():
        rows.append(dict(table=table, c=float(c), d=d, seed=int(seed), detector=det.name,
                         encoding=det.encoding, metric=k, value=float(v),
                         runtime_s=float(det.runtime_s), code_hash=chash, dataset_hash=dhash))
    return rows


def run_cell(table: str, c: float, d: str, seed: int, detectors: list[str] | None = None,
             ablation: bool = False, ceiling: bool = False, save_model: bool = False,
             subsample_lof_knn: bool = False, overwrite: bool = False, verbose: bool = True) -> dict:
    detectors = list(detectors or ALL_DETECTORS)
    name = cell_name(table, c, d, seed)
    paths = {k: RESULTS / k / f"{name}.parquet" for k in ("manifests", "scores", "metrics")}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    if paths["metrics"].exists() and not overwrite:
        if verbose:
            print(f"[skip] {name} exists")
        return {"cell": name, "skipped": True}

    t_cell = time.perf_counter()
    df = _table(table)
    inj = inject(df, table, c, d, seed)
    inj.manifest.to_parquet(paths["manifests"], index=False)
    labels, types = inj.labels, inj.types
    chash, dhash = code_hash(), inj.meta["dataset_hash"]

    outputs: dict[str, DetectorOutput] = {}
    base = encode_base(inj.data)
    timings: dict[str, float] = {}

    # Per-detector checkpoints so that an interrupted cell resumes at the next detector.
    part = RESULTS / "partial" / name
    part.mkdir(parents=True, exist_ok=True)
    hash_file = part / "dataset_hash.txt"
    if hash_file.exists() and hash_file.read_text() != dhash:
        for f in part.glob("*"):
            f.unlink()
    hash_file.write_text(dhash)

    def cached(det_name: str) -> DetectorOutput | None:
        f = part / f"{det_name}.npz"
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=True)
        info = json.loads(str(z["info"]))
        return DetectorOutput(det_name, z["scores"], float(z["runtime_s"]), str(z["encoding"]), info)

    def log(det: DetectorOutput):
        check_scores(det.scores, det.name)
        outputs[det.name] = det
        timings[det.name] = det.runtime_s
        f = part / f"{det.name}.npz"
        if not f.exists():
            info = {k: v for k, v in det.info.items() if k != "component_percentiles"}
            np.savez(f, scores=det.scores, runtime_s=det.runtime_s, encoding=det.encoding,
                     info=json.dumps(info, default=str))
        if verbose:
            m = compute_metrics(det.scores, labels, types)
            print(f"  {det.name:<12s} {det.runtime_s:8.1f}s  P@100={m['p_at_100']:.3f}  AP={m['ap']:.3f}", flush=True)

    big = table == "D"     # AE-pyod and VAE-pyod: batch 256, 30 epochs on Table-D (recorded deviation)

    def via_worker(det_name: str) -> DetectorOutput:
        """Heavy Table-D detectors run in a fresh process (memory isolation and
        checkpointing); the worker writes the same npz that `cached` reads."""
        import subprocess, sys
        cmd = [sys.executable, "-m", "src.worker", "--table", table, "--c", str(c), "--d", d, "--seed", str(seed),
               "--detector", det_name] + (["--save-model"] if save_model else []) + (["--subsample"] if subsample_lof_knn else [])
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        hit = cached(det_name)
        if hit is None:
            raise RuntimeError(f"worker produced no checkpoint for {det_name}")
        return hit

    HEAVY = set(CLASSICAL)   # every PyOD detector on Table-D runs in its own process
    for det in CLASSICAL:
        if det in detectors:
            hit = cached(det)
            if hit is None:
                hit = via_worker(det) if (big and det in HEAVY) else run_pyod(det, base, seed, "E_base", subsample=subsample_lof_knn, big=big)
            log(hit)
    if "AE-emb" in detectors:
        save = RESULTS / "models" / f"{name}_ae_emb.pt"
        hit = cached("AE-emb")
        if hit is None or (save_model and not save.exists()):
            if big:
                hit = via_worker("AE-emb")
            else:
                emb = encode_emb(inj.data)
                hit = run_ae_emb(emb, seed, save_path=str(save) if save_model else None)
        log(hit)
    if "DECOMP" in detectors:
        dec = run_decomp(inj.data)
        log(dec)
        comp = pd.DataFrame(dec.info["component_percentiles"], columns=dec.info["components"])
        comp.insert(0, "row_id", inj.manifest["row_id"].to_numpy())
        comp.to_parquet(RESULTS / "scores" / f"{name}_decomp_components.parquet", index=False)
    if "AVG" in detectors:
        log(run_avg(outputs))
    if ablation:
        freq = encode_freq(inj.data)
        for det in ABLATION:
            hit = cached(f"{det}+freq")
            if hit is None:
                hit = run_pyod(det, freq, seed, "E_freq", subsample=subsample_lof_knn)
                hit.name = f"{det}+freq"
            log(hit)
    if ceiling:
        log(cached("LGBM-ceiling") or run_ceiling(base, labels, seed))

    # scores: wide parquet
    wide = pd.DataFrame({"row_id": inj.manifest["row_id"].to_numpy()})
    for k, v in outputs.items():
        wide[k] = v.scores.astype(np.float64)
    wide.to_parquet(paths["scores"], index=False)

    rows = []
    for det in outputs.values():
        rows += metrics_rows(table, c, d, seed, det, labels, types, chash, dhash)
    pd.DataFrame(rows).to_parquet(paths["metrics"], index=False)

    meta = dict(inj.meta)
    meta.update(cell=name, code_hash=chash, timings=timings, total_s=time.perf_counter() - t_cell,
                detector_info={k: {kk: vv for kk, vv in v.info.items() if kk != "component_percentiles"}
                               for k, v in outputs.items()},
                subsample_lof_knn=subsample_lof_knn)
    (RESULTS / "metrics" / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    for f in part.glob("*"):
        f.unlink()
    part.rmdir()
    if verbose:
        print(f"[done] {name} in {meta['total_s']:.1f}s, n_anom={meta['n_anom']}, pairs_reused={meta['pairs_reused']}")
    return meta


def grid_cells(part: str, cfg: dict) -> list[dict]:
    g = cfg[part]
    cells = []
    for table in g["tables"]:
        for c in g["contamination"]:
            for d in g["difficulty"]:
                for seed in g["seeds"]:
                    cells.append(dict(table=table, c=float(c), d=d, seed=int(seed),
                                      detectors=g.get("detectors", ALL_DETECTORS),
                                      ablation=bool(g.get("ablation", False)),
                                      ceiling=bool(g.get("ceiling", False)),
                                      save_model=bool(g.get("save_models", False)),
                                      subsample_lof_knn=bool(g.get("subsample_lof_knn", {}).get(table, False))
                                      if isinstance(g.get("subsample_lof_knn"), dict) else bool(g.get("subsample_lof_knn", False))))
    return cells


def collect() -> None:
    parts = sorted((RESULTS / "metrics").glob("*.parquet"))
    if not parts:
        print("no metrics yet")
        return
    m = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    ceiling = m[m["detector"] == "LGBM-ceiling"]
    m = m[m["detector"] != "LGBM-ceiling"]
    m.to_parquet(RESULTS / "metrics.parquet", index=False)
    if len(ceiling):
        ceiling.to_parquet(RESULTS / "ceiling.parquet", index=False)
    print(f"metrics.parquet: {len(m)} rows from {m[['table','c','d','seed']].drop_duplicates().shape[0]} cells; "
          f"ceiling.parquet: {len(ceiling)} rows")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("cell")
    pc.add_argument("--table", required=True)
    pc.add_argument("--c", type=float, required=True)
    pc.add_argument("--d", required=True)
    pc.add_argument("--seed", type=int, required=True)
    pc.add_argument("--detectors", nargs="*", default=None)
    pc.add_argument("--ablation", action="store_true")
    pc.add_argument("--ceiling", action="store_true")
    pc.add_argument("--save-model", action="store_true")
    pc.add_argument("--subsample-lof-knn", action="store_true")
    pc.add_argument("--overwrite", action="store_true")
    pg = sub.add_parser("grid")
    pg.add_argument("--part", required=True)
    pg.add_argument("--dry-run", action="store_true")
    pg.add_argument("--overwrite", action="store_true")
    sub.add_parser("collect")
    a = ap.parse_args(argv)

    if a.cmd == "cell":
        run_cell(a.table, a.c, a.d, a.seed, a.detectors, a.ablation, a.ceiling, a.save_model,
                 a.subsample_lof_knn, a.overwrite)
    elif a.cmd == "grid":
        cfg = yaml.safe_load(CONFIG.read_text())
        cells = grid_cells(a.part, cfg)
        print(f"{a.part}: {len(cells)} cells")
        for cell in cells:
            if a.dry_run:
                print(cell_name(cell["table"], cell["c"], cell["d"], cell["seed"]), cell["detectors"], cell["ablation"], cell["ceiling"])
                continue
            run_cell(overwrite=a.overwrite, **cell)
        collect()
    elif a.cmd == "collect":
        collect()


if __name__ == "__main__":
    main()
