"""Benchmark loop — descriptor × endpoint × model grid.

Runs 5-fold GroupKFold (groups from `ab_benchmark.eval.splits.assign_clusters`
when available, else `ab_id_canonical`), collects out-of-fold predictions,
and reports Spearman ρ + RMSE + R² each with a bootstrap 95% CI.

Handles input feature standardization per model family; RF and gradient
boosting skip it, ridge uses it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from struct_devpred.models import MODEL_FACTORIES, STANDARDIZE_INPUTS


# --- bootstrap CI utilities (self-contained fallback) ----------------------


def _bootstrap_ci(stat_fn, n: int, n_boot: int = 2000, alpha: float = 0.05,
                  random_state: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(idx)
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def _spearman_ci(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0):
    from scipy.stats import spearmanr

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), n
    rho = float(spearmanr(y_true, y_pred).statistic)
    low, high = _bootstrap_ci(
        lambda idx: float(spearmanr(y_true[idx], y_pred[idx]).statistic),
        n=n, random_state=seed,
    )
    return rho, low, high, n


def _rmse_ci(y_true, y_pred, seed: int = 0):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    r = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    low, high = _bootstrap_ci(
        lambda idx: float(np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))),
        n=n, random_state=seed,
    )
    return r, low, high


def _r2_ci(y_true, y_pred, seed: int = 0):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 4:
        return float("nan"), float("nan"), float("nan")

    def r2(ys: np.ndarray, yp: np.ndarray) -> float:
        ss_res = float(((ys - yp) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_hat = r2(y_true, y_pred)
    low, high = _bootstrap_ci(lambda idx: r2(y_true[idx], y_pred[idx]), n=n, random_state=seed)
    return r2_hat, low, high


# --- groups (leakage-resistant clustering) ---------------------------------

_PROJECTS_ROOT = Path(__file__).resolve().parents[3]
_AB_BENCHMARK = Path(
    os.environ.get("AB_BENCHMARK_PATH", str(_PROJECTS_ROOT / "ab-benchmark"))
)


def _assign_groups(df: pd.DataFrame) -> np.ndarray:
    """Use ab_benchmark.eval.splits.assign_clusters when available."""
    required = {"ab_id_canonical", "v_gene_heavy", "cdr_h3", "cdr_h3_length"}
    if _AB_BENCHMARK.is_dir() and str(_AB_BENCHMARK) not in sys.path:
        sys.path.append(str(_AB_BENCHMARK))
    try:
        from ab_benchmark.eval.splits import assign_clusters  # type: ignore

        if required.issubset(df.columns):
            return assign_clusters(df, identity_threshold=0.5).to_numpy()
    except ImportError:
        pass
    missing = sorted(required - set(df.columns))
    if missing:
        print(
            "[warn] leakage-resistant cluster columns missing; falling back to "
            f"ab_id_canonical grouping. Missing: {', '.join(missing)}",
            file=sys.stderr,
        )
    else:
        print(
            "[warn] ab_benchmark cluster helper unavailable; falling back to "
            "ab_id_canonical grouping.",
            file=sys.stderr,
        )
    codes, _ = pd.factorize(df["ab_id_canonical"])
    return codes


# --- single evaluation -----------------------------------------------------


@dataclass
class RunResult:
    descriptor: str
    endpoint: str
    model: str
    n: int
    n_clusters: int
    n_features: int
    n_folds: int
    rho_point: float
    rho_low: float
    rho_high: float
    rmse_point: float
    rmse_low: float
    rmse_high: float
    r2_point: float
    r2_low: float
    r2_high: float

    @property
    def rho_ci_excludes_zero(self) -> bool:
        return not (self.rho_low <= 0 <= self.rho_high)


def _groupkfold_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    model_name: str, n_splits: int = 5,
) -> np.ndarray:
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)

    n_unique = int(len(np.unique(groups)))
    effective = min(n_splits, n_unique)
    if effective < 2:
        raise ValueError(f"need ≥2 unique groups, got {n_unique}")

    gkf = GroupKFold(n_splits=effective)
    factory = MODEL_FACTORIES[model_name]
    need_scale = STANDARDIZE_INPUTS[model_name]

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        if need_scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        mdl = factory()
        mdl.fit(X_tr, y_tr)
        oof[test_idx] = mdl.predict(X_te)
    return oof


def evaluate_combination(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    descriptor: str, endpoint: str, model: str,
    n_splits: int = 5, bootstrap_seed: int = 0,
) -> tuple[RunResult, np.ndarray]:
    """Return (summary, oof_predictions) for one descriptor × endpoint × model cell."""
    oof = _groupkfold_oof(X, y, groups, model, n_splits=n_splits)
    rho, rho_lo, rho_hi, _ = _spearman_ci(y, oof, seed=bootstrap_seed)
    rmse, rmse_lo, rmse_hi = _rmse_ci(y, oof, seed=bootstrap_seed)
    r2, r2_lo, r2_hi = _r2_ci(y, oof, seed=bootstrap_seed)
    res = RunResult(
        descriptor=descriptor,
        endpoint=endpoint,
        model=model,
        n=int(len(y)),
        n_clusters=int(len(np.unique(groups))),
        n_features=int(X.shape[1]),
        n_folds=min(n_splits, int(len(np.unique(groups)))),
        rho_point=rho, rho_low=rho_lo, rho_high=rho_hi,
        rmse_point=rmse, rmse_low=rmse_lo, rmse_high=rmse_hi,
        r2_point=r2, r2_low=r2_lo, r2_high=r2_hi,
    )
    return res, oof


# --- ablation driver -------------------------------------------------------


def run_ablation(
    df: pd.DataFrame,
    endpoints: list[str],
    descriptor_families: list[str],
    models: list[str] | None = None,
    esm_cache: dict[str, np.ndarray] | None = None,
    hybrid_families: dict[str, list[str]] | None = None,
    n_splits: int = 5,
) -> tuple[list[RunResult], dict[tuple[str, str, str], np.ndarray]]:
    """Full descriptor × endpoint × model grid.

    Returns (results list, oof map indexed by (descriptor, endpoint, model)).
    """
    from struct_devpred.descriptors import build_descriptor_matrix, build_hybrid_matrix

    models = models or list(MODEL_FACTORIES.keys())
    results: list[RunResult] = []
    oof_map: dict[tuple[str, str, str], np.ndarray] = {}

    groups_all = _assign_groups(df)

    single_family_matrices: dict[str, tuple[np.ndarray, list[str], list[str]]] = {}
    for fam in descriptor_families:
        X, names, ids = build_descriptor_matrix(df, fam, esm_cache=esm_cache)
        single_family_matrices[fam] = (X, names, ids)

    # Hybrid compositions.
    hybrid_families = hybrid_families or {}
    for label, members in hybrid_families.items():
        X, names, ids = build_hybrid_matrix(df, members, esm_cache=esm_cache)
        single_family_matrices[label] = (X, names, ids)

    for endpoint in endpoints:
        if endpoint not in df.columns:
            continue
        mask = df[endpoint].notna().to_numpy()
        if mask.sum() < 20:
            continue
        y_full = df[endpoint].to_numpy(dtype=float)

        for descriptor, (X_full, _, _) in single_family_matrices.items():
            X_ep = X_full[mask]
            y_ep = y_full[mask]
            g_ep = groups_all[mask]
            for model in models:
                try:
                    res, oof = evaluate_combination(
                        X_ep, y_ep, g_ep,
                        descriptor=descriptor, endpoint=endpoint, model=model,
                        n_splits=n_splits,
                    )
                except Exception as e:
                    # Never kill the whole run on one bad combination —
                    # record a NaN row instead.
                    res = RunResult(
                        descriptor=descriptor, endpoint=endpoint, model=model,
                        n=int(mask.sum()),
                        n_clusters=int(len(np.unique(g_ep))),
                        n_features=int(X_ep.shape[1]),
                        n_folds=0,
                        rho_point=float("nan"), rho_low=float("nan"), rho_high=float("nan"),
                        rmse_point=float("nan"), rmse_low=float("nan"), rmse_high=float("nan"),
                        r2_point=float("nan"), r2_low=float("nan"), r2_high=float("nan"),
                    )
                    oof = np.full(int(mask.sum()), np.nan)
                    print(f"[warn] {descriptor}/{endpoint}/{model} failed: {e}", file=sys.stderr)
                results.append(res)
                oof_map[(descriptor, endpoint, model)] = oof

    return results, oof_map


def results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
