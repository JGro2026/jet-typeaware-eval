"""Loading of the four raw client tables with an enforced schema.

All column names are exposed as constants so that the umlaut columns are
never typed by hand elsewhere in the code base.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
SRC_DIR = ROOT / "src"

TABLES = ["A", "B", "C", "D"]

# Column constants
KEY = "BUCHUNGS_GUID"
ACCOUNT = "Kontonummer"
CONTRA = "Gegenkonto"
CURRENCY = "Währung"
TAX = "Steuersatz"
PAYPERIOD = "Zahlungszeitraum"
USER = "Benutzererkennung"
TEXT_PRESENT = "Buchungstext nicht Leer"
DATE_DEV = "Abweichung Belegdatum zum WJ"
AMOUNT_Z = "Umsatz_mod_z"
ROUND_AMT = "Umsatz_Smooth"

CATEGORICAL = [ACCOUNT, CONTRA, CURRENCY, TAX, USER]
NUMERIC = [PAYPERIOD, AMOUNT_Z]
BINARY = [TEXT_PRESENT, DATE_DEV, ROUND_AMT]
FEATURE_COLUMNS = CATEGORICAL + NUMERIC + BINARY
ALL_COLUMNS = [KEY] + [ACCOUNT, CONTRA, CURRENCY, TAX, PAYPERIOD, USER,
                       TEXT_PRESENT, DATE_DEV, AMOUNT_Z, ROUND_AMT]

DTYPES = {
    KEY: "int64",
    ACCOUNT: "int64",
    CONTRA: "int64",
    CURRENCY: "string",
    TAX: "string",
    PAYPERIOD: "float64",
    USER: "string",
    TEXT_PRESENT: "int64",
    DATE_DEV: "int64",
    AMOUNT_Z: "float64",
    ROUND_AMT: "int64",
}


def table_path(table: str, data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    return data_dir / f"table{table}.csv"


def load_table(table: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one raw table with the fixed schema and a clean RangeIndex.

    Rows keep the file order. No deduplication or cleaning is applied.
    """
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of {TABLES}")
    df = pd.read_csv(table_path(table, data_dir), dtype=DTYPES, encoding="utf-8")
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"table {table}: missing columns {missing}")
    df = df[ALL_COLUMNS].copy()
    # Use plain Python str objects for categoricals; StringDtype is not needed
    # downstream and object dtype hashes and groups deterministically.
    for col in (CURRENCY, TAX, USER):
        df[col] = df[col].astype(object)
    if not df[KEY].is_unique:
        raise ValueError(f"table {table}: {KEY} is not unique")
    df = df.reset_index(drop=True)
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """SHA-256 of a canonical serialisation of a table.

    Rows are sorted by the key and columns are written in the fixed order, so
    the hash is independent of row order and of the pandas index.
    """
    canon = df[ALL_COLUMNS].sort_values(KEY).reset_index(drop=True)
    payload = canon.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_hash() -> str:
    """Content hash of src/*.py, used in place of a git commit."""
    h = hashlib.sha256()
    for p in sorted(SRC_DIR.glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def summarise(df: pd.DataFrame) -> dict:
    """Small profile used by the WP0 audit and the run logs."""
    out = {
        "rows": int(len(df)),
        "n_account": int(df[ACCOUNT].nunique()),
        "n_contra": int(df[CONTRA].nunique()),
        "n_currency": int(df[CURRENCY].nunique()),
        "n_tax": int(df[TAX].nunique()),
        "n_user": int(df[USER].nunique()),
        "payperiod_missing": int(df[PAYPERIOD].isna().sum()),
        "amount_z_min": float(df[AMOUNT_Z].min()),
        "amount_z_max": float(df[AMOUNT_Z].max()),
    }
    return out


def p999(values: np.ndarray) -> float:
    """99.9th percentile with numpy's default linear interpolation."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.percentile(values, 99.9))
