"""Tests for src/struct_devpred/error_analysis.py."""

import numpy as np
import pandas as pd

from struct_devpred.error_analysis import (
    _cdr_h3_bucket,
    _v_gene_family,
    choose_best_per_endpoint,
    per_slice_metrics,
)


class TestBuckets:
    def test_cdr_h3_bucket(self):
        assert _cdr_h3_bucket(5) == "short (≤12)"
        assert _cdr_h3_bucket(12) == "short (≤12)"
        assert _cdr_h3_bucket(13) == "medium (13-17)"
        assert _cdr_h3_bucket(18) == "long (18-22)"
        assert _cdr_h3_bucket(25) == "very long (≥23)"
        assert _cdr_h3_bucket(float("nan")) == "unknown"

    def test_v_gene_family(self):
        assert _v_gene_family("IGHV3-23") == "IGHV3"
        assert _v_gene_family("IGHV1-69") == "IGHV1"
        assert _v_gene_family("") == "unknown"
        assert _v_gene_family(None) == "unknown"


class TestChooseBest:
    def test_picks_highest_rho(self):
        df = pd.DataFrame([
            {"endpoint": "e1", "descriptor": "d1", "model": "m1", "rho_point": 0.1},
            {"endpoint": "e1", "descriptor": "d2", "model": "m2", "rho_point": 0.3},
            {"endpoint": "e2", "descriptor": "d3", "model": "m3", "rho_point": 0.5},
        ])
        best = choose_best_per_endpoint(df)
        assert best == {"e1": ("d2", "m2"), "e2": ("d3", "m3")}


class TestPerSliceMetrics:
    def _make_oof(self):
        rng = np.random.default_rng(0)
        n = 60
        return pd.DataFrame({
            "descriptor": ["A"] * n,
            "endpoint": ["hic_rt"] * n,
            "model": ["ridge"] * n,
            "ab_id": [f"ab{i}" for i in range(n)],
            "source": ["jain_2017"] * 30 + ["shehata_2019"] * 30,
            "cdr_h3_length": list(range(5, 65)),
            "v_gene_heavy": ["IGHV3-23"] * 30 + ["IGHV1-69"] * 30,
            "y_true": rng.normal(size=n),
            "y_pred_oof": rng.normal(size=n),
        })

    def test_produces_expected_slices(self):
        oof = self._make_oof()
        best = {"hic_rt": ("A", "ridge")}
        result = per_slice_metrics(oof, best)

        slices = set(result["slice_column"].unique())
        assert "all" in slices
        assert "source" in slices
        assert "cdr_h3_length_bucket" in slices
        assert "v_gene_heavy_family" in slices

    def test_whole_endpoint_row_present(self):
        oof = self._make_oof()
        result = per_slice_metrics(oof, {"hic_rt": ("A", "ridge")})
        whole = result[result["slice_column"] == "all"]
        assert len(whole) == 1
        assert whole.iloc[0]["n"] == 60

    def test_skips_missing_best_entry(self):
        oof = self._make_oof()
        result = per_slice_metrics(oof, {"nonexistent_endpoint": ("A", "ridge")})
        assert len(result) == 0
