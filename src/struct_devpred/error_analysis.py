"""Per-slice error analysis on OOF predictions.

Consumes `data/processed/oof_predictions.parquet` (produced by cli.py) and
computes per-slice Spearman ρ + residual stats. Slices:

    - Source dataset (jain_2017, sabdab_thera, ...)
    - CDR-H3 length bucket (≤12, 13-17, 18-22, ≥23)
    - V-gene-heavy family (IGHV1, IGHV3, IGHV4, ...)

A "best descriptor" per endpoint is chosen based on overall ρ point estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class SliceMetric:
    descriptor: str
    endpoint: str
    model: str
    slice_column: str
    slice_value: str
    n: int
    rho: float


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 4:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _cdr_h3_bucket(length: int) -> str:
    if length is None or np.isnan(length):
        return "unknown"
    try:
        n = int(length)
    except (TypeError, ValueError):
        return "unknown"
    if n <= 12:
        return "short (≤12)"
    if n <= 17:
        return "medium (13-17)"
    if n <= 22:
        return "long (18-22)"
    return "very long (≥23)"


def _v_gene_family(v_gene_heavy: str) -> str:
    if not v_gene_heavy or not isinstance(v_gene_heavy, str):
        return "unknown"
    # Keep everything before the dash: "IGHV3-23" → "IGHV3".
    prefix = v_gene_heavy.split("-")[0].strip().upper()
    return prefix or "unknown"


# ---------------------------------------------------------------------------


def per_slice_metrics(
    oof_df: pd.DataFrame,
    best_per_endpoint: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """For the "best descriptor + model" per endpoint, slice OOF by source,
    CDR-H3 length, and V-gene family, and compute per-slice Spearman ρ.
    """
    rows: list[SliceMetric] = []

    for endpoint, (descriptor, model) in best_per_endpoint.items():
        sub = oof_df[
            (oof_df["endpoint"] == endpoint)
            & (oof_df["descriptor"] == descriptor)
            & (oof_df["model"] == model)
        ]
        if sub.empty:
            continue

        # Whole-endpoint baseline.
        rho = _spearman(sub["y_true"].to_numpy(), sub["y_pred_oof"].to_numpy())
        rows.append(SliceMetric(descriptor, endpoint, model, "all", "all",
                                len(sub), rho))

        # By source.
        for src, sub_src in sub.groupby("source"):
            rho = _spearman(sub_src["y_true"].to_numpy(), sub_src["y_pred_oof"].to_numpy())
            rows.append(SliceMetric(descriptor, endpoint, model, "source",
                                    str(src), len(sub_src), rho))

        # By CDR-H3 length bucket.
        if "cdr_h3_length" in sub.columns:
            sub_c = sub.copy()
            sub_c["_bucket"] = sub_c["cdr_h3_length"].map(_cdr_h3_bucket)
            for bucket, sub_b in sub_c.groupby("_bucket"):
                rho = _spearman(sub_b["y_true"].to_numpy(), sub_b["y_pred_oof"].to_numpy())
                rows.append(SliceMetric(descriptor, endpoint, model,
                                        "cdr_h3_length_bucket", str(bucket),
                                        len(sub_b), rho))

        # By V-gene family.
        if "v_gene_heavy" in sub.columns:
            sub_v = sub.copy()
            sub_v["_vfam"] = sub_v["v_gene_heavy"].map(_v_gene_family)
            for vfam, sub_vg in sub_v.groupby("_vfam"):
                rho = _spearman(sub_vg["y_true"].to_numpy(), sub_vg["y_pred_oof"].to_numpy())
                rows.append(SliceMetric(descriptor, endpoint, model,
                                        "v_gene_heavy_family", str(vfam),
                                        len(sub_vg), rho))

    out = pd.DataFrame([
        {
            "endpoint": r.endpoint,
            "descriptor": r.descriptor,
            "model": r.model,
            "slice_column": r.slice_column,
            "slice_value": r.slice_value,
            "n": r.n,
            "rho": r.rho,
        }
        for r in rows
    ])
    return out


def choose_best_per_endpoint(
    ablation_df: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    best: dict[str, tuple[str, str]] = {}
    for endpoint, sub in ablation_df.groupby("endpoint"):
        idx = sub["rho_point"].idxmax()
        best[str(endpoint)] = (str(sub.loc[idx, "descriptor"]),
                               str(sub.loc[idx, "model"]))
    return best


# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation", type=Path, default=Path("reports/ablation_results.csv"))
    ap.add_argument("--oof", type=Path, default=Path("data/processed/oof_predictions.parquet"))
    ap.add_argument("--out", type=Path, default=Path("reports/error_analysis.csv"))
    args = ap.parse_args(argv)

    ablation = pd.read_csv(args.ablation)
    oof = pd.read_parquet(args.oof)
    best = choose_best_per_endpoint(ablation)

    result = per_slice_metrics(oof, best)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    print(f"Wrote {args.out} — {len(result)} slice rows")
    print()
    print("Per-endpoint headline (whole-slice ρ):")
    for endpoint, (descriptor, model) in best.items():
        whole = result[
            (result["endpoint"] == endpoint)
            & (result["slice_column"] == "all")
        ]
        if not whole.empty:
            print(f"  {endpoint:18s}  {descriptor:22s}  {model:18s}  "
                  f"ρ={whole.iloc[0]['rho']:+.3f}  (n={whole.iloc[0]['n']})")

    print("\nCDR-H3 length effect on HIC RT (where we expect an effect):")
    sub = result[
        (result["endpoint"] == "hic_rt")
        & (result["slice_column"] == "cdr_h3_length_bucket")
    ]
    for _, r in sub.iterrows():
        print(f"  {r['slice_value']:22s}  n={int(r['n']):3d}  ρ={r['rho']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
