"""Sanity checks of Section 6, plus a few structural checks of the encodings
and metrics. Run with `pytest -q tests`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detectors import (ALL_DETECTORS, CLASSICAL, UNSUPERVISED, run_avg,
                           run_decomp, run_pyod)
from src.encode import (EMB_COLUMNS, ONEHOT_ALL, encode_base, encode_emb,
                        encode_freq)
from src.evaluate import (check_scores, compute_metrics, precision_at_k,
                          rank_normalise, recall_at_k, recall_at_k_by_type)
from src.feedback import LabelLeak, Oracle, StaticAvg, run_rounds
from src.inject import (CONTAMINATIONS, DIFFICULTIES, F_MULT, F_PAIR,
                        PROPORTIONS, TYPES, allocate_counts, inject,
                        n_anomalies)
from src.load import (ACCOUNT, ALL_COLUMNS, AMOUNT_Z, CONTRA, KEY, PAYPERIOD,
                      ROOT, load_table, p999)

ROOT_RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def tables():
    return {t: load_table(t) for t in ["C", "A"]}


def _cells():
    for table in ["C", "A"]:
        for c in CONTAMINATIONS:
            for d in DIFFICULTIES:
                yield table, c, d


@pytest.fixture(scope="module")
def injected(tables):
    out = {}
    for table, c, d in _cells():
        out[(table, c, d)] = inject(tables[table], table, c, d, seed=0)
    return out


# ---------------------------------------------------------------- Section 6

def test_label_count_and_type_proportions(injected):
    for (table, c, d), r in injected.items():
        n = len(r.data)
        n_anom = n_anomalies(n, c)
        assert int(r.labels.sum()) == n_anom == r.meta["n_anom"]
        counts = pd.Series(r.types[r.labels == 1]).value_counts().to_dict()
        assert counts == allocate_counts(n_anom)
        for t in TYPES:
            expected = PROPORTIONS[t] * n_anom
            tol = 0.5 if t != "T1" else 0.5 * (len(TYPES) - 1)  # T1 absorbs the rounding remainder
            assert abs(counts.get(t, 0) - expected) <= tol + 1e-9, (table, c, d, t)


def test_fresh_ids_absent_from_original(tables, injected):
    for (table, c, d), r in injected.items():
        orig = tables[table]
        m = r.manifest
        t3 = m[m.anom_type == "T3"]
        t4 = m[m.anom_type == "T4"]
        contra = set(orig[CONTRA].tolist())
        account = set(orig[ACCOUNT].tolist())
        new3 = [json.loads(v)[0] for v in t3.new_values]
        new4 = [json.loads(v)[0] for v in t4.new_values]
        assert all(v not in contra for v in new3)
        assert all(v not in account for v in new4)
        assert len(set(new3)) == len(new3) and len(set(new4)) == len(new4)  # one fresh id per row


def test_t5_pairs_unseen_and_frequent(tables, injected):
    for (table, c, d), r in injected.items():
        orig = tables[table]
        F = F_PAIR[d]
        ca = orig[ACCOUNT].value_counts()
        cg = orig[CONTRA].value_counts()
        seen = set(map(tuple, orig[[ACCOUNT, CONTRA]].drop_duplicates().to_numpy().tolist()))
        m = r.manifest
        t5 = m[m.anom_type == "T5"]
        pairs = []
        for fields, new in zip(t5.modified_fields, t5.new_values):
            f, v = json.loads(fields), json.loads(new)
            assert f == [ACCOUNT, CONTRA]
            a, g = v
            assert (a, g) not in seen
            assert ca[a] >= F and cg[g] >= F
            pairs.append((a, g))
        if r.meta["pairs_reused"]:
            assert len(set(pairs)) == r.meta["n_valid_pairs"] < len(pairs)
        else:
            assert len(set(pairs)) == len(pairs)


def test_table_a_easy_reuses_pairs(injected):
    r = injected[("A", 0.03, "easy")]
    assert r.meta["pairs_reused"] is True
    assert r.meta["n_valid_pairs"] == 19


def test_t1_t2_exceed_multiple_of_p999(tables, injected):
    for (table, c, d), r in injected.items():
        orig = tables[table]
        f = F_MULT[d]
        thr_amt = f * p999(np.abs(orig[AMOUNT_Z].to_numpy()))
        thr_pay = f * p999(orig[PAYPERIOD].to_numpy())
        m = r.manifest
        z = r.data[AMOUNT_Z].to_numpy()
        pay = r.data[PAYPERIOD].to_numpy()
        t1 = np.flatnonzero(m.anom_type.to_numpy() == "T1")
        t2 = np.flatnonzero(m.anom_type.to_numpy() == "T2")
        assert np.all(np.abs(z[t1]) >= thr_amt * (1 - 1e-12))
        assert np.all(pay[t2] >= thr_pay * (1 - 1e-12))
        assert len(set(np.sign(z[t1]))) <= 2
        # T2 only on rows whose original payment period was present
        assert orig[PAYPERIOD].to_numpy()[t2].size == t2.size and not np.isnan(orig[PAYPERIOD].to_numpy()[t2]).any()


def test_same_seed_same_hash_different_seed_different_hash(tables):
    df = tables["C"]
    a = inject(df, "C", 0.015, "medium", 0)
    b = inject(df, "C", 0.015, "medium", 0)
    c = inject(df, "C", 0.015, "medium", 1)
    assert a.meta["dataset_hash"] == b.meta["dataset_hash"]
    assert a.meta["dataset_hash"] != c.meta["dataset_hash"]
    pd.testing.assert_frame_equal(a.data, b.data)
    pd.testing.assert_frame_equal(a.manifest, b.manifest)
    assert not a.manifest.label.equals(c.manifest.label)


def test_row_count_and_guid_set_unchanged(tables, injected):
    for (table, c, d), r in injected.items():
        orig = tables[table]
        assert len(r.data) == len(orig)
        assert set(r.data[KEY]) == set(orig[KEY])
        assert list(r.data[KEY]) == list(orig[KEY])            # order preserved
        assert list(r.data.columns) == ALL_COLUMNS
        # at most one type per row, unmodified rows identical to the original
        assert r.manifest.anom_type.notna().sum() == r.labels.sum()
        keep = r.labels == 0
        pd.testing.assert_frame_equal(r.data[keep].reset_index(drop=True), orig[keep].reset_index(drop=True))


def test_feedback_label_mask_is_enforced(injected):
    r = injected[("C", 0.015, "medium")]
    oracle = Oracle(r.labels)
    with pytest.raises(LabelLeak):
        oracle.labels_for(np.array([0, 1, 2]))
    with pytest.raises(LabelLeak):
        oracle.assert_mask(np.array([5]))
    rng = np.random.default_rng(0)
    scores = rng.random(len(r.labels))

    class Spy(StaticAvg):
        seen: list = []

        def update(self, rows, labels, round_idx):
            oracle.assert_mask(rows)
            assert len(rows) == 25 * round_idx
            Spy.seen.append(rows.copy())
            return self.scores

    recs = run_rounds(Spy(scores), oracle)
    assert [x.reviews_cumulative for x in recs] == [25, 50, 75, 100]
    assert oracle.reviewed.sum() == 100
    assert all(len(np.unique(h)) == 25 for h in oracle.history)
    assert len(set(np.concatenate(oracle.history).tolist())) == 100
    with pytest.raises(LabelLeak):
        oracle.review(oracle.history[0][:3])                    # re-presenting reviewed rows
    # the final P@k is computed on the unreviewed remainder only
    assert recs[-1].extra["remaining_anomalies"] == r.labels.sum() - oracle.found_so_far()


def test_scores_finite_and_not_constant_fast_detectors(injected):
    r = injected[("C", 0.015, "medium")]
    base = encode_base(r.data)
    for det in ["IF", "LOF", "kNN", "HBOS", "ECOD", "PCA"]:
        out = run_pyod(det, base, seed=0)
        check_scores(out.scores, det)
        assert out.scores.shape == (len(r.data),)
    dec = run_decomp(r.data)
    check_scores(dec.scores, "DECOMP")
    with pytest.raises(ValueError):
        check_scores(np.zeros(10), "const")
    with pytest.raises(ValueError):
        check_scores(np.array([1.0, np.nan]), "nan")


def test_scores_finite_and_not_constant_in_saved_results():
    files = sorted((ROOT_RESULTS / "scores").glob("*.parquet"))
    if not files:
        pytest.skip("no saved score matrices yet")
    for p in files:
        sc = pd.read_parquet(p)
        assert sc["row_id"].is_unique
        for col in sc.columns:
            if col == "row_id":
                continue
            check_scores(sc[col].to_numpy(), f"{p.name}:{col}")


# ---------------------------------------------------------------- structure

def test_encodings_structure(injected):
    r = injected[("A", 0.015, "medium")]
    base = encode_base(r.data)
    assert base.X.dtype == np.float32 and np.isfinite(base.X).all()
    for col in ONEHOT_ALL:
        cols = [j for j, n in enumerate(base.names) if n.startswith(f"oh:{col}=")]
        assert len(cols) == r.data[col].nunique()
        assert np.allclose(base.X[:, cols].sum(axis=1), 1.0)
    top = [j for j, n in enumerate(base.names) if n.startswith(f"oh:{CONTRA}=") and not n.endswith("=other")]
    other = base.names.index(f"oh:{CONTRA}=other")
    assert len(top) == 20
    assert np.allclose(base.X[:, top].sum(axis=1) + base.X[:, other], 1.0)
    miss = base.names.index(f"ind:{PAYPERIOD}_missing")
    assert np.array_equal(base.X[:, miss] == 1.0, r.data[PAYPERIOD].isna().to_numpy())
    zcol = base.names.index(f"num:{AMOUNT_Z}")
    assert abs(base.X[:, zcol].mean()) < 1e-4 and abs(base.X[:, zcol].std() - 1) < 1e-3
    freq = encode_freq(r.data)
    assert freq.X.shape[1] == base.X.shape[1] + 4
    emb = encode_emb(r.data)
    assert emb.cat_codes.shape == (len(r.data), 5)
    assert emb.vocab_sizes == [r.data[c].nunique() for c in EMB_COLUMNS]
    assert (emb.cat_codes.max(axis=0) < np.array(emb.vocab_sizes)).all()
    assert emb.num.shape[1] == 6


def test_metric_definitions():
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    types = np.array(["T1", None, "T5", None, None, "T2", None, None, None, None], dtype=object)
    assert precision_at_k(scores, labels, 2) == 0.5
    assert recall_at_k(scores, labels, 2) == pytest.approx(1 / 3)
    assert precision_at_k(scores, labels, 100) == pytest.approx(3 / 100)     # capped by n_anom < k
    assert recall_at_k(scores, labels, 100) == 1.0
    r = recall_at_k_by_type(scores, types, 3)
    assert r["T1"] == 1.0 and r["T5"] == 1.0 and r["T2"] == 0.0 and np.isnan(r["T3"])
    m = compute_metrics(scores, labels, types)
    assert m["n_anom"] == 3 and 0 < m["ap"] <= 1 and 0 < m["roc_auc"] <= 1
    rn = rank_normalise(np.array([3.0, 1.0, 2.0, 2.0]))
    assert rn.tolist() == [1.0, 0.25, 0.625, 0.625]
    # ties broken by ascending row position
    assert precision_at_k(np.array([1.0, 1.0, 1.0]), np.array([0, 1, 1]), 1) == 0.0
