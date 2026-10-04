"""The three encodings of Section 3.3. All statistics are fit on the table that
is passed in (the injected table, transductive setting).

E_base : classical and PyOD detectors, supervised ceiling
E_freq : E_base plus four log1p count features (ablation only)
E_emb  : integer codes per categorical column plus the numeric block, for the
         embedding autoencoder and DeepSAD
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .load import (ACCOUNT, AMOUNT_Z, BINARY, CONTRA, CURRENCY, PAYPERIOD,
                   TAX, USER)

ONEHOT_ALL = [CURRENCY, TAX, USER, ACCOUNT]
TOP_CONTRA = 20
EMB_DIMS = {ACCOUNT: 10, CONTRA: 10, CURRENCY: 1, TAX: 1, USER: 1}
EMB_COLUMNS = [ACCOUNT, CONTRA, CURRENCY, TAX, USER]


@dataclass
class Encoded:
    X: np.ndarray                 # float32, rows in table order
    names: list[str]


@dataclass
class EncodedEmb:
    cat_codes: np.ndarray         # int64, shape (N, 5), column order EMB_COLUMNS
    vocab_sizes: list[int]
    num: np.ndarray               # float32, numeric block (scaled) and binaries
    num_names: list[str]
    n_scaled: int                 # first n_scaled columns of num are MSE targets on a z-scale


def _standardise(x: np.ndarray) -> np.ndarray:
    """StandardScaler semantics: (x - mean) / std with population std; a
    constant column becomes all zeros."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def numeric_block(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Umsatz_mod_z scaled; Zahlungszeitraum median-imputed and scaled plus a
    0/1 missing indicator; the three binaries as 0/1. Only the two numeric
    columns are standardised; the indicator and the binaries stay 0/1."""
    pay = df[PAYPERIOD].to_numpy(dtype=np.float64)
    miss = np.isnan(pay)
    med = np.nanmedian(pay)
    pay_imp = np.where(miss, med, pay)
    cols = [
        _standardise(df[AMOUNT_Z].to_numpy(dtype=np.float64)),
        _standardise(pay_imp),
        miss.astype(np.float64),
    ] + [df[b].to_numpy(dtype=np.float64) for b in BINARY]
    names = [f"num:{AMOUNT_Z}", f"num:{PAYPERIOD}", f"ind:{PAYPERIOD}_missing"] + [f"bin:{b}" for b in BINARY]
    return np.column_stack(cols), names


def onehot(values: pd.Series, name: str, levels: list | None = None) -> tuple[np.ndarray, list[str]]:
    if levels is None:
        levels = sorted(pd.unique(values).tolist())
    idx = {v: j for j, v in enumerate(levels)}
    M = np.zeros((len(values), len(levels)), dtype=np.float64)
    codes = values.map(idx)
    ok = codes.notna().to_numpy()
    M[np.flatnonzero(ok), codes[ok].astype(int).to_numpy()] = 1.0
    return M, [f"oh:{name}={lvl}" for lvl in levels]


def contra_block(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """log1p(frequency of the row's Gegenkonto) plus one-hot of the 20 most
    frequent levels (ties broken by ascending id) plus an 'other' indicator."""
    counts = df[CONTRA].value_counts()
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [k for k, _ in order[:TOP_CONTRA]]
    freq = df[CONTRA].map(counts).to_numpy(dtype=np.float64)
    M, names = onehot(df[CONTRA], CONTRA, levels=top)
    other = (M.sum(axis=1) == 0).astype(np.float64)
    X = np.column_stack([np.log1p(freq), M, other])
    return X, [f"freq:{CONTRA}_log1p"] + names + [f"oh:{CONTRA}=other"]


def encode_base(df: pd.DataFrame) -> Encoded:
    blocks, names = [], []
    X, n = numeric_block(df)
    blocks.append(X); names += n
    for col in ONEHOT_ALL:
        M, n = onehot(df[col], col)
        blocks.append(M); names += n
    X, n = contra_block(df)
    blocks.append(X); names += n
    return Encoded(np.column_stack(blocks).astype(np.float32), names)


def count_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    def cnt(cols: list[str]) -> np.ndarray:
        return df.groupby(cols, sort=False, dropna=False)[cols[0]].transform("size").to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.log1p(cnt([ACCOUNT])),
        np.log1p(cnt([CONTRA])),
        np.log1p(cnt([ACCOUNT, CONTRA])),
        np.log1p(cnt([USER, ACCOUNT])),
    ])
    return X, [f"cnt:{ACCOUNT}", f"cnt:{CONTRA}", f"cnt:{ACCOUNT}x{CONTRA}", f"cnt:{USER}x{ACCOUNT}"]


def encode_freq(df: pd.DataFrame) -> Encoded:
    base = encode_base(df)
    X, n = count_features(df)
    return Encoded(np.column_stack([base.X, X.astype(np.float32)]), base.names + n)


def encode_emb(df: pd.DataFrame) -> EncodedEmb:
    codes, sizes = [], []
    for col in EMB_COLUMNS:
        levels = sorted(pd.unique(df[col]).tolist())
        idx = {v: j for j, v in enumerate(levels)}
        codes.append(df[col].map(idx).to_numpy(dtype=np.int64))
        sizes.append(len(levels))
    num, names = numeric_block(df)
    return EncodedEmb(cat_codes=np.column_stack(codes), vocab_sizes=sizes,
                      num=num.astype(np.float32), num_names=names, n_scaled=2)
