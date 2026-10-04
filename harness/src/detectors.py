"""Unsupervised detectors (Section 3.4), the naive AVG ensemble, the E_freq
ablation and the supervised LightGBM ceiling. Every function returns a float64
score vector over all rows, higher is more anomalous, plus fit metadata.

Hyperparameters are fixed here and must not be changed on the basis of the
injected labels.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .encode import EMB_COLUMNS, EMB_DIMS, Encoded, EncodedEmb
from .evaluate import rank_normalise
from .load import ACCOUNT, AMOUNT_Z, CATEGORICAL, CONTRA, PAYPERIOD, USER

CLASSICAL = ["IF", "LOF", "kNN", "HBOS", "ECOD", "PCA", "AE-pyod", "VAE-pyod"]
UNSUPERVISED = CLASSICAL + ["AE-emb", "DECOMP"]          # detectors 1 to 10
ALL_DETECTORS = UNSUPERVISED + ["AVG"]                    # detectors 1 to 11
ABLATION = ["IF", "LOF", "kNN", "ECOD"]                   # run on E_freq as X+freq
SUBSAMPLE_N = 50_000                                      # Section 3.7, LOF and kNN on Table-D only


def seed_everything(seed: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # One intra-op thread for every torch model: the per-step tensors are small, and two
    # processes (Table-D and the A/B/C sweep) share the 2-core sandbox without oversubscription.
    torch.set_num_threads(1)


@dataclass
class DetectorOutput:
    name: str
    scores: np.ndarray
    runtime_s: float
    encoding: str
    info: dict = field(default_factory=dict)
    model: object = None


# ---------------------------------------------------------------- PyOD / sklearn

PYOD_NET_DEFAULT = {"epochs": 100, "batch_size": 64}
PYOD_NET_LARGE = {"epochs": 30, "batch_size": 256}   # Table-D deviation, decided at the WP2 checkpoint


def _pyod_model(name: str, seed: int, n_features: int, big: bool):
    net = PYOD_NET_LARGE if big else PYOD_NET_DEFAULT
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.pca import PCA
    from pyod.models.vae import VAE

    if name == "IF":
        return IForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=1)
    if name == "LOF":
        return LOF(n_neighbors=20, algorithm="auto", n_jobs=1)
    if name == "kNN":
        return KNN(n_neighbors=5, method="largest", algorithm="auto", n_jobs=1)
    if name == "HBOS":
        return HBOS()
    if name == "ECOD":
        return ECOD(n_jobs=1)
    if name == "PCA":
        # n_components as a variance fraction; the covariance eigensolver gives the
        # same components as the full SVD (verified) without materialising U
        return PCA(n_components=0.90, svd_solver="covariance_eigh", random_state=seed)
    if name == "AE-pyod":
        return AutoEncoder(hidden_neuron_list=[64, 32, 8], epoch_num=net["epochs"], batch_size=net["batch_size"],
                           random_state=seed, verbose=0)
    if name == "VAE-pyod":
        return VAE(encoder_neuron_list=[64, 32], decoder_neuron_list=[32, 64], latent_dim=8,
                   epoch_num=net["epochs"], batch_size=net["batch_size"], random_state=seed, verbose=0)
    raise ValueError(name)


def ecod_scores(X: np.ndarray, chunk: int = 32) -> np.ndarray:
    """PyOD ECOD (n_jobs=1) computed column-chunk by column-chunk. Identical
    scores to `pyod.models.ecod.ECOD().fit(X).decision_scores_`, but with a
    memory footprint of a few columns instead of five copies of X. Used for
    every table so that all cells share one implementation."""
    from pyod.utils.stat_models import column_ecdf
    from scipy.stats import skew

    X = np.asarray(X, dtype=np.float64)
    scores = np.zeros(X.shape[0], dtype=np.float64)
    for j in range(0, X.shape[1], chunk):
        C = np.ascontiguousarray(X[:, j: j + chunk])
        U_l = -1 * np.log(column_ecdf(C))
        U_r = -1 * np.log(column_ecdf(-C))
        skewness = np.sign(np.nan_to_num(skew(C, axis=0)))
        U_skew = U_l * -1 * np.sign(skewness - 1) + U_r * np.sign(skewness + 1)
        O = np.maximum(np.maximum(U_l, U_r), U_skew)
        scores += O.sum(axis=1)
    return scores


def pca_scores(X: np.ndarray, variance: float = 0.90, chunk: int = 20_000, copy: bool = True) -> tuple[np.ndarray, int]:
    """PyOD PCA (standardization=True, weighted=True, n_selected_components=None)
    reproduced with one standardised copy of X: components from the eigen-
    decomposition of the covariance matrix (sklearn's covariance_eigh path,
    same sign convention), the number of components chosen as in sklearn for a
    variance fraction, and PyOD's score sum_k ||x - v_k|| / w_k computed in
    row chunks. Verified identical to PyOD on C, A and B."""
    from scipy.spatial.distance import cdist

    X = np.array(X, dtype=np.float64, copy=copy)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0                      # sklearn StandardScaler semantics
    X -= mu
    X /= sd
    n = X.shape[0]
    C = (X.T @ X) / (n - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.maximum(evals[order], 0.0), evecs[:, order]
    Vt = evecs.T
    # svd_flip(u_based_decision=False): largest-magnitude entry of each component positive
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    signs[signs == 0] = 1.0
    Vt = Vt * signs[:, None]
    ratio = evals / evals.sum()
    k = int(np.searchsorted(np.cumsum(ratio), variance, side="right") + 1)
    comps, w = Vt[:k], ratio[:k]
    scores = np.empty(n, dtype=np.float64)
    for s0 in range(0, n, chunk):
        scores[s0: s0 + chunk] = (cdist(X[s0: s0 + chunk], comps) / w).sum(axis=1)
    return scores, k


def run_pyod(name: str, enc: Encoded, seed: int, encoding: str = "E_base",
             subsample: bool = False, big: bool = False) -> DetectorOutput:
    """Fit a PyOD detector on enc.X and score every row.

    subsample=True (LOF and kNN on Table-D only, Section 3.7): fit on a seeded
    random subsample of SUBSAMPLE_N rows, score the fitted rows with their
    in-sample scores (self excluded from the neighbourhood) and the remaining
    rows with decision_function against the fitted rows.
    """
    X = enc.X.astype(np.float64)
    t0 = time.perf_counter()
    info: dict = {}
    if name in ("AE-pyod", "VAE-pyod"):
        seed_everything(seed)
    if name == "ECOD":
        scores = ecod_scores(X)
        return DetectorOutput(name, scores, time.perf_counter() - t0, encoding, {"impl": "column-chunked, identical to PyOD"})
    if name == "PCA":
        scores, k = pca_scores(X, copy=False)   # X is already a private float64 copy
        return DetectorOutput(name, scores, time.perf_counter() - t0, encoding, {"impl": "covariance eigensolver, identical to PyOD", "n_components": k})
    model = _pyod_model(name, seed, X.shape[1], big=big)
    if subsample and X.shape[0] > SUBSAMPLE_N and name in ("LOF", "kNN"):
        rng = np.random.default_rng([seed, 0x5AB])
        idx = np.sort(rng.choice(X.shape[0], size=SUBSAMPLE_N, replace=False))
        model.fit(X[idx])
        scores = np.empty(X.shape[0], dtype=np.float64)
        scores[idx] = model.decision_scores_
        rest = np.setdiff1d(np.arange(X.shape[0]), idx)
        scores[rest] = model.decision_function(X[rest])
        info["subsample_fit_rows"] = int(SUBSAMPLE_N)
    else:
        model.fit(X)
        scores = np.asarray(model.decision_scores_, dtype=np.float64)
    if name == "PCA":
        info["n_components"] = int(model.n_components_)
    if name in ("AE-pyod", "VAE-pyod"):
        info["epochs"] = int(model.epoch_num)
        info["batch_size"] = int(model.batch_size)
        info["large_table_setting"] = bool(big)
    return DetectorOutput(name, scores, time.perf_counter() - t0, encoding, info)


# ---------------------------------------------------------------- AE-emb (PyTorch)

def build_ae_emb(vocab_sizes: list[int], n_num: int):
    import torch
    from torch import nn

    class AEEmb(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(v, EMB_DIMS[c]) for v, c in zip(vocab_sizes, EMB_COLUMNS)])
            d_in = sum(EMB_DIMS[c] for c in EMB_COLUMNS) + n_num
            act = nn.LeakyReLU(0.01)
            self.encoder = nn.Sequential(nn.Linear(d_in, 64), act, nn.Linear(64, 32), act, nn.Linear(32, 8))
            self.decoder = nn.Sequential(nn.Linear(8, 32), act, nn.Linear(32, 64), act)
            self.num_head = nn.Linear(64, n_num)
            self.cat_heads = nn.ModuleList([nn.Linear(64, v) for v in vocab_sizes])
            self.d_in = d_in

        def embed(self, cat, num):
            e = [emb(cat[:, j]) for j, emb in enumerate(self.embs)]
            return torch.cat(e + [num], dim=1)

        def encode(self, cat, num):
            return self.encoder(self.embed(cat, num))

        def forward(self, cat, num):
            z = self.encode(cat, num)
            h = self.decoder(z)
            return self.num_head(h), [head(h) for head in self.cat_heads], z

        def row_loss(self, cat, num):
            """Per-row total reconstruction loss: mean squared error over the
            numeric and binary columns plus the sum of the cross-entropies of
            the categorical columns (weight 1.0 each)."""
            num_hat, logits, _ = self(cat, num)
            mse = ((num_hat - num) ** 2).mean(dim=1)
            ce = torch.zeros_like(mse)
            for j, lg in enumerate(logits):
                ce = ce + nn.functional.cross_entropy(lg, cat[:, j], reduction="none")
            return mse + 1.0 * ce

    return AEEmb()


def run_ae_emb(enc: EncodedEmb, seed: int, epochs: int = 100, batch_size: int = 64,
               lr: float = 1e-3, save_path: str | None = None) -> DetectorOutput:
    import torch

    t0 = time.perf_counter()
    seed_everything(seed)
    # One intra-op thread: the per-step tensors are small and OpenMP oversubscription
    # on the 2-core sandbox made 2 threads up to 10x slower (measured 51 vs 556 ms/step).
    torch.set_num_threads(1)
    cat = torch.as_tensor(enc.cat_codes, dtype=torch.long)
    num = torch.as_tensor(enc.num, dtype=torch.float32)
    model = build_ae_emb(enc.vocab_sizes, num.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = cat.shape[0]
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for s in range(0, N, batch_size):
            b = perm[s: s + batch_size]
            opt.zero_grad()
            loss = model.row_loss(cat[b], num[b]).mean()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model.row_loss(cat[s: s + 8192], num[s: s + 8192])
                            for s in range(0, N, 8192)]).numpy().astype(np.float64)
    if save_path is not None:
        torch.save({"state_dict": model.state_dict(), "vocab_sizes": enc.vocab_sizes,
                    "n_num": int(num.shape[1]), "epochs": epochs, "batch_size": batch_size,
                    "seed": seed}, save_path)
    info = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "vocab_sizes": enc.vocab_sizes}
    return DetectorOutput("AE-emb", scores, time.perf_counter() - t0, "E_emb", info, model=model)


# ---------------------------------------------------------------- DECOMP

def _robust_z_within(df: pd.DataFrame, col: str, group: str, min_rows: int = 30) -> np.ndarray:
    """|x - median| / MAD within `group`; groups with fewer than min_rows
    non-missing values use the global median and MAD. A zero MAD falls back
    to the global MAD, and to 1 if that is zero as well. Missing values get 0."""
    x = df[col].to_numpy(dtype=np.float64)
    ok = ~np.isnan(x)
    g_med = np.median(x[ok])
    g_mad = np.median(np.abs(x[ok] - g_med))
    if g_mad == 0:
        g_mad = 1.0
    med = np.full(len(x), g_med)
    mad = np.full(len(x), g_mad)
    sub = df.loc[ok, [group]].assign(_x=x[ok])
    grp = sub.groupby(group, sort=False)["_x"]
    n = grp.transform("size").to_numpy()
    m = grp.transform("median").to_numpy()
    d = sub.assign(_d=np.abs(sub["_x"].to_numpy() - m)).groupby(group, sort=False)["_d"].transform("median").to_numpy()
    use = n >= min_rows
    idx = np.flatnonzero(ok)
    med[idx[use]] = m[use]
    d_use = np.where(d[use] > 0, d[use], g_mad)
    mad[idx[use]] = d_use
    z = np.zeros(len(x))
    z[ok] = np.abs(x[ok] - med[ok]) / mad[ok]
    return z


def decomp_components(df: pd.DataFrame) -> pd.DataFrame:
    """The five raw component scores of DECOMP on the (injected) table."""
    N = len(df)
    comp = pd.DataFrame(index=df.index)
    comp["s_amt"] = _robust_z_within(df, AMOUNT_Z, ACCOUNT)
    comp["s_pay"] = _robust_z_within(df, PAYPERIOD, ACCOUNT)
    cat = np.column_stack([-np.log(df[c].map(df[c].value_counts()).to_numpy(dtype=np.float64) / N)
                           for c in CATEGORICAL])
    comp["s_cat"] = cat.max(axis=1)
    ca = df[ACCOUNT].map(df[ACCOUNT].value_counts()).to_numpy(dtype=np.float64)
    cg = df[CONTRA].map(df[CONTRA].value_counts()).to_numpy(dtype=np.float64)
    cag = df.groupby([ACCOUNT, CONTRA], sort=False)[ACCOUNT].transform("size").to_numpy(dtype=np.float64)
    expected = ca * cg / N
    comp["s_pair"] = np.maximum(0.0, -np.log((cag + 0.5) / (expected + 0.5)))
    K = df[ACCOUNT].nunique()
    cu = df[USER].map(df[USER].value_counts()).to_numpy(dtype=np.float64)
    cua = df.groupby([USER, ACCOUNT], sort=False)[ACCOUNT].transform("size").to_numpy(dtype=np.float64)
    comp["s_user"] = -np.log((cua + 0.5) / (cu + 0.5 * K))
    return comp


def run_decomp(df: pd.DataFrame) -> DetectorOutput:
    t0 = time.perf_counter()
    comp = decomp_components(df)
    pct = np.column_stack([rank_normalise(comp[c].to_numpy()) for c in comp.columns])
    scores = pct.max(axis=1)
    info = {"components": list(comp.columns)}
    out = DetectorOutput("DECOMP", scores, time.perf_counter() - t0, "raw", info)
    out.info["component_percentiles"] = pct
    return out


# ---------------------------------------------------------------- AVG

def run_avg(outputs: dict[str, DetectorOutput]) -> DetectorOutput:
    t0 = time.perf_counter()
    missing = [d for d in UNSUPERVISED if d not in outputs]
    if missing:
        raise ValueError(f"AVG needs detectors 1 to 10, missing {missing}")
    R = np.column_stack([rank_normalise(outputs[d].scores) for d in UNSUPERVISED])
    return DetectorOutput("AVG", R.mean(axis=1), time.perf_counter() - t0, "ranks",
                          {"members": list(UNSUPERVISED)})


def context_matrix(outputs: dict[str, DetectorOutput]) -> np.ndarray:
    """Rank-normalised scores of detectors 1 to 10 plus an intercept, the
    context x_i of the feedback methods (Section 3.5)."""
    R = np.column_stack([rank_normalise(outputs[d].scores) for d in UNSUPERVISED])
    return np.column_stack([R, np.ones(R.shape[0])])


# ---------------------------------------------------------------- supervised ceiling

def run_ceiling(enc: Encoded, labels: np.ndarray, seed: int) -> DetectorOutput:
    """LightGBM with default parameters and scale_pos_weight = N_neg / N_pos,
    5-fold stratified cross-fitting, out-of-fold probabilities as scores."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    t0 = time.perf_counter()
    X = enc.X.astype(np.float32)
    y = np.asarray(labels).astype(int)
    spw = float((y == 0).sum() / max(1, (y == 1).sum()))
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = lgb.LGBMClassifier(scale_pos_weight=spw, random_state=seed, verbose=-1, n_jobs=1)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return DetectorOutput("LGBM-ceiling", oof, time.perf_counter() - t0, "E_base",
                          {"scale_pos_weight": spw, "folds": 5})
