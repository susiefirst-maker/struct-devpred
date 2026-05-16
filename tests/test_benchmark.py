"""Tests for src/struct_devpred/benchmark.py."""

import numpy as np
import pandas as pd
import pytest

from struct_devpred.benchmark import (
    RunResult,
    _r2_ci,
    _rmse_ci,
    _spearman_ci,
    evaluate_combination,
    results_to_dataframe,
    run_ablation,
)


class TestCIs:
    def test_spearman_positive_on_correlated(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=100)
        y = x + rng.normal(scale=0.2, size=100)
        rho, lo, hi, n = _spearman_ci(x, y, seed=0)
        assert rho > 0.9
        assert lo > 0.8
        assert n == 100

    def test_rmse_positive(self):
        y = np.arange(40.0)
        rmse, lo, hi = _rmse_ci(y, y + 1, seed=0)
        assert rmse == pytest.approx(1.0, abs=0.05)

    def test_r2_perfect_fit(self):
        y = np.arange(50.0)
        r2, lo, hi = _r2_ci(y, y.copy(), seed=0)
        assert r2 > 0.99


class TestEvaluateCombination:
    def _synth(self, n=80, d=5, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, d)).astype(np.float32)
        beta = rng.normal(size=d)
        y = X @ beta + rng.normal(size=n) * 0.3
        groups = rng.integers(0, 10, size=n)
        return X, y, groups

    def test_ridge_recovers_signal(self):
        X, y, g = self._synth()
        res, oof = evaluate_combination(
            X, y, g, descriptor="synth", endpoint="synth_y", model="ridge",
            n_splits=5, bootstrap_seed=0,
        )
        assert isinstance(res, RunResult)
        assert res.n == 80
        assert res.n_features == 5
        assert res.n_folds <= 5
        assert res.rho_point > 0.3
        assert oof.shape == (80,)

    def test_random_forest_runs(self):
        X, y, g = self._synth()
        res, oof = evaluate_combination(
            X, y, g, descriptor="synth", endpoint="synth_y", model="random_forest",
            n_splits=5, bootstrap_seed=0,
        )
        assert not np.isnan(res.rho_point)

    def test_gradient_boosting_runs(self):
        X, y, g = self._synth()
        res, oof = evaluate_combination(
            X, y, g, descriptor="synth", endpoint="synth_y", model="gradient_boosting",
            n_splits=5, bootstrap_seed=0,
        )
        assert not np.isnan(res.rho_point)


class TestRunAblation:
    def test_empty_endpoints_returns_empty(self):
        df = pd.DataFrame([
            {"ab_id_canonical": f"ab{i}", "vh": "A" * 110, "vl": "D" * 110}
            for i in range(20)
        ])
        results, oof = run_ablation(df, endpoints=[], descriptor_families=["composition"])
        assert results == []

    def test_endpoint_with_too_few_rows_skipped(self):
        df = pd.DataFrame([
            {"ab_id_canonical": f"ab{i}", "vh": "A" * 110, "vl": "D" * 110,
             "rare_endpoint": float(i) if i < 5 else np.nan}
            for i in range(25)
        ])
        results, _ = run_ablation(
            df, endpoints=["rare_endpoint"],
            descriptor_families=["composition"], models=["ridge"],
        )
        # Fewer than 20 rows have the endpoint, so skipped.
        assert results == []


class TestRunResult:
    def test_ci_excludes_zero_flag(self):
        r = RunResult(
            descriptor="d", endpoint="e", model="m", n=100, n_clusters=20,
            n_features=10, n_folds=5,
            rho_point=0.3, rho_low=0.1, rho_high=0.5,
            rmse_point=1.0, rmse_low=0.8, rmse_high=1.2,
            r2_point=0.1, r2_low=0, r2_high=0.2,
        )
        assert r.rho_ci_excludes_zero

    def test_ci_includes_zero(self):
        r = RunResult(
            descriptor="d", endpoint="e", model="m", n=100, n_clusters=20,
            n_features=10, n_folds=5,
            rho_point=0.1, rho_low=-0.1, rho_high=0.3,
            rmse_point=1.0, rmse_low=0.8, rmse_high=1.2,
            r2_point=0.1, r2_low=0, r2_high=0.2,
        )
        assert not r.rho_ci_excludes_zero


class TestResultsToDataframe:
    def test_shape(self):
        results = [
            RunResult(
                descriptor="d", endpoint="e", model="m", n=10, n_clusters=5,
                n_features=3, n_folds=5,
                rho_point=0.2, rho_low=0.0, rho_high=0.4,
                rmse_point=1.0, rmse_low=0.8, rmse_high=1.2,
                r2_point=0.1, r2_low=0, r2_high=0.2,
            )
        ]
        df = results_to_dataframe(results)
        assert len(df) == 1
        assert "rho_point" in df.columns
        assert "n_clusters" in df.columns
