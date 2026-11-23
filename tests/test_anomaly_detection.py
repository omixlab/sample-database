"""
Tests for the Isolation Forest anomaly detector's feature assembly.

Weighted towards what the detector *reports about itself*, because the failure this guards
against is not a crash: the embedding block is resolved by column prefix, so a dataset built
before step 28 fits a silently narrower matrix and still writes a plausible-looking
`is_anomaly` column. Only the report distinguishes the two.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from sample.analysis import (ANOMALY_DESCRIPTOR_COLUMNS, _resolve_prefixed,
                             run_anomaly_detection)


def _frame(n: int = 60, embeddings: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"entry_id": [f"SAMPLE{i:04d}" for i in range(n)],
            "sequence": ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 12)) for _ in range(n)],
            "is_decoy": rng.integers(0, 2, n)}
    for col in ANOMALY_DESCRIPTOR_COLUMNS:
        data[col] = rng.normal(size=n)
    for i in range(20):
        data[f"aa_freq_{i}"] = rng.random(n)
    for i in range(8):
        data[f"aa_group_freq_{i}"] = rng.random(n)
    for i in range(embeddings):
        data[f"esm2_dim_{i}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def _run(tmp_path, df, **kw):
    src = tmp_path / "dataset.csv"
    out = tmp_path / "out.csv"
    rep = tmp_path / "report.json"
    df.to_csv(src, index=False)
    ok = run_anomaly_detection(str(src), str(out), str(rep), **kw)
    report = json.loads(rep.read_text()) if rep.exists() else {}
    return ok, report, (pd.read_csv(out) if out.exists() else None)


class TestFeatureBlocks:
    def test_the_embedding_block_is_fitted_and_reported(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame(embeddings=8))
        assert ok
        assert report["embedding_used"] is True
        assert report["feature_blocks"]["embedding"]["n_features"] == 8
        assert report["feature_blocks"]["embedding"]["n_source_columns"] == 8

    def test_every_declared_descriptor_is_fitted_not_just_the_first_six(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame())
        assert ok
        block = report["feature_blocks"]["physicochemical"]
        assert block["n_features"] == len(ANOMALY_DESCRIPTOR_COLUMNS)
        # the four the manuscript names must all actually be in there
        for named in ("molecular_weight", "logp", "isoelectric_point", "net_charge_ph7"):
            assert named in block["columns"]

    def test_total_feature_count_is_the_sum_of_the_blocks(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame(embeddings=8))
        assert ok
        assert report["n_features_total"] == sum(
            b["n_features"] for b in report["feature_blocks"].values())

    def test_missing_embeddings_are_recorded_rather_than_passed_off_as_a_full_fit(self, tmp_path):
        df = _frame(embeddings=0)
        ok, report, out = _run(tmp_path, df)
        assert ok                                  # still usable, but
        assert report["embedding_used"] is False   # never claims otherwise
        assert "embedding" not in report["feature_blocks"]
        assert report["embedding_feature_share"] == 0.0
        assert "is_anomaly" in out.columns         # the column looks identical: hence the flag

    def test_missing_embeddings_are_warned_about_loudly(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            _run(tmp_path, _frame(embeddings=0))
        assert any("esm2_dim_" in r.getMessage() for r in caplog.records)

    def test_absent_descriptors_are_named_in_a_warning(self, tmp_path, caplog):
        df = _frame().drop(columns=["boman_index", "hydrophobic_moment"])
        with caplog.at_level("WARNING"):
            ok, report, _ = _run(tmp_path, df)
        assert ok
        assert report["feature_blocks"]["physicochemical"]["n_features"] == \
            len(ANOMALY_DESCRIPTOR_COLUMNS) - 2
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "boman_index" in joined and "hydrophobic_moment" in joined

    def test_no_features_at_all_returns_false(self, tmp_path):
        df = pd.DataFrame({"entry_id": ["A", "B"], "sequence": ["AAAA", "CCCC"]})
        ok, _, _ = _run(tmp_path, df)
        assert ok is False


class TestNaNAccounting:
    def test_filled_embedding_values_are_counted_not_silently_zeroed(self, tmp_path):
        df = _frame(embeddings=8)
        df.loc[0:4, "esm2_dim_0"] = np.nan      # 5 rows, inclusive label slice
        ok, report, _ = _run(tmp_path, df)
        assert ok
        assert report["feature_values_filled"]["embedding"] == 5
        assert report["feature_blocks"]["embedding"]["values_filled"] == 5

    def test_a_clean_frame_reports_zero_fills(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame(embeddings=8))
        assert ok
        assert sum(report["feature_values_filled"].values()) == 0


class TestEsmComponents:
    def test_pca_reduces_the_block_and_records_retained_variance(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame(embeddings=8), esm_components=3)
        assert ok
        block = report["feature_blocks"]["embedding"]
        assert block["n_source_columns"] == 8
        assert block["n_features"] == 3
        assert block["pca_components"] == 3
        assert 0.0 < block["explained_variance_ratio"] <= 1.0

    def test_reduction_lowers_the_embedding_share_of_the_matrix(self, tmp_path):
        _, raw, _ = _run(tmp_path, _frame(embeddings=8))
        _, cut, _ = _run(tmp_path, _frame(embeddings=8), esm_components=2)
        assert cut["embedding_feature_share"] < raw["embedding_feature_share"]

    def test_zero_components_keeps_the_raw_dimensions(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame(embeddings=8), esm_components=0)
        assert ok
        assert report["feature_blocks"]["embedding"]["pca_components"] is None
        assert report["feature_blocks"]["embedding"]["n_features"] == 8

    def test_asking_for_more_components_than_dimensions_is_clamped(self, tmp_path):
        ok, report, _ = _run(tmp_path, _frame(embeddings=8), esm_components=999)
        assert ok
        assert report["feature_blocks"]["embedding"]["n_features"] <= 8


class TestColumnOrdering:
    def test_embedding_columns_sort_numerically_not_lexicographically(self):
        df = pd.DataFrame({f"esm2_dim_{i}": [0.0] for i in range(12)})
        cols = _resolve_prefixed(df, ("esm2_dim_",))
        assert cols[:4] == ["esm2_dim_0", "esm2_dim_1", "esm2_dim_2", "esm2_dim_3"]
        assert cols[-1] == "esm2_dim_11"      # lexicographic order would put esm2_dim_9 last

    def test_prefixes_are_grouped_in_the_order_given(self):
        df = pd.DataFrame({"protbert_dim_0": [0.0], "esm2_dim_0": [0.0], "esm2_dim_1": [0.0]})
        assert _resolve_prefixed(df, ("esm2_dim_", "protbert_dim_")) == \
            ["esm2_dim_0", "esm2_dim_1", "protbert_dim_0"]
