"""struct-devpred CLI — run the descriptor × endpoint × model ablation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from struct_devpred.benchmark import results_to_dataframe, run_ablation
from struct_devpred.descriptors import (
    DESCRIPTOR_FAMILIES,
    load_esm2_cache,
)


def _get_h3_extractor():
    """Return a function mapping (vh, vl) -> CDR-H3 length; empty if ab-benchmark unavailable."""
    projects_root = Path(__file__).resolve().parents[3]
    _AB_BENCHMARK = Path(
        os.environ.get("AB_BENCHMARK_PATH", str(projects_root / "ab-benchmark"))
    )
    if _AB_BENCHMARK.is_dir() and str(_AB_BENCHMARK) not in sys.path:
        sys.path.append(str(_AB_BENCHMARK))
    try:
        from ab_benchmark.seqprops import extract_cdrs  # type: ignore

        def h3_len(vh: str, vl: str) -> int:
            return len(extract_cdrs(str(vh), str(vl)).get("h3", ""))

        return h3_len
    except ImportError:
        return lambda vh, vl: 0


_PROJECTS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HARMONIZED = Path(
    os.environ.get(
        "AB_BENCHMARK_WIDE",
        str(_PROJECTS_ROOT / "ab-benchmark" / "data" / "processed" / "harmonized_antibody_dev_wide.parquet"),
    )
)
DEFAULT_ENDPOINTS = [
    "tm_onset_c", "hic_rt", "ac_sins",
    "bvp_score", "psr_score", "expression_mgl",
]
DEFAULT_FAMILIES = list(DESCRIPTOR_FAMILIES.keys()) + ["esm2_t12"]
DEFAULT_MODELS = ["ridge", "random_forest", "gradient_boosting"]
DEFAULT_HYBRIDS = {
    "hybrid_seq_struct": ["composition", "physicochem", "cdr_lengths",
                          "structure_informed", "aggregation"],
    "hybrid_esm_struct": ["esm2_t12", "structure_informed"],
    "hybrid_all": ["composition", "physicochem", "cdr_lengths",
                   "structure_informed", "humanness", "aggregation", "esm2_t12"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harmonized", type=Path, default=DEFAULT_HARMONIZED)
    ap.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", type=Path, default=Path("reports/ablation_results.csv"))
    ap.add_argument("--oof-out", type=Path, default=Path("data/processed/oof_predictions.parquet"))
    ap.add_argument("--skip-hybrids", action="store_true",
                    help="Skip hybrid concatenated descriptor families.")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args(argv)

    if not args.harmonized.exists():
        print(f"ERROR: harmonized parquet not found: {args.harmonized}", file=sys.stderr)
        print("Run the ab-benchmark Phase 0 harmonizer first:", file=sys.stderr)
        print("  cd ../ab-benchmark", file=sys.stderr)
        print("  .venv/bin/python -m ab_benchmark.data.build_harmonized --config configs/datasets.yaml", file=sys.stderr)
        print("Or pass --harmonized /path/to/harmonized_antibody_dev_wide.parquet", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.harmonized)
    # Restrict to rows with both VH and VL sequences.
    df = df[df["vh"].astype(str).str.len() > 0]
    df = df[df["vl"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"Harmonized rows: {len(df)}  ({df['source'].value_counts().to_dict()})")

    esm_cache = load_esm2_cache()
    if esm_cache is None and "esm2_t12" in args.families:
        print("[warn] ESM-2 cache not found — skipping esm2_t12 family", file=sys.stderr)
        args.families = [f for f in args.families if f != "esm2_t12"]
    else:
        print(f"ESM-2 cache loaded: {len(esm_cache) if esm_cache else 0} embeddings")

    hybrids = {} if args.skip_hybrids else {
        k: [f for f in v if f in args.families or f == "esm2_t12"]
        for k, v in DEFAULT_HYBRIDS.items()
    }

    results, oof_map = run_ablation(
        df, endpoints=args.endpoints,
        descriptor_families=args.families,
        models=args.models,
        esm_cache=esm_cache,
        hybrid_families=hybrids,
        n_splits=args.n_splits,
    )

    # Results table.
    res_df = results_to_dataframe(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.out, index=False)

    # OOF predictions — long format for error analysis. Extract CDR-H3
    # length on the fly when the harmonized parquet has it unpopulated.
    h3_len_fn = _get_h3_extractor()
    rows = []
    for (descriptor, endpoint, model), oof in oof_map.items():
        mask = df[endpoint].notna().to_numpy() if endpoint in df.columns else None
        if mask is None:
            continue
        rel_df = df.loc[mask].reset_index(drop=True)
        y_true = rel_df[endpoint].to_numpy(dtype=float)
        for i in range(len(oof)):
            row = rel_df.iloc[i]
            h3_len = row.get("cdr_h3_length", 0)
            if not h3_len:
                h3_len = h3_len_fn(str(row.get("vh", "")), str(row.get("vl", "")))
            rows.append({
                "descriptor": descriptor,
                "endpoint": endpoint,
                "model": model,
                "ab_id": row.get("ab_id", ""),
                "ab_id_canonical": row.get("ab_id_canonical", ""),
                "source": row.get("source", ""),
                "cdr_h3_length": int(h3_len),
                "v_gene_heavy": row.get("v_gene_heavy", ""),
                "y_true": float(y_true[i]),
                "y_pred_oof": float(oof[i]),
            })
    oof_df = pd.DataFrame(rows)
    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(args.oof_out, index=False)

    # JSON summary sidecar.
    summary = {
        "n_rows_harmonized": int(len(df)),
        "endpoints": args.endpoints,
        "descriptor_families": args.families + list(hybrids.keys()),
        "models": args.models,
        "n_splits": args.n_splits,
        "n_results": len(results),
        "output_csv": str(args.out),
        "output_oof_parquet": str(args.oof_out),
        "headline_per_endpoint": {},
    }
    for endpoint in args.endpoints:
        sub = res_df[res_df["endpoint"] == endpoint]
        if sub.empty:
            continue
        best_idx = sub["rho_point"].idxmax()
        best = sub.loc[best_idx]
        summary["headline_per_endpoint"][endpoint] = {
            "descriptor": best["descriptor"],
            "model": best["model"],
            "rho": float(best["rho_point"]),
            "ci_low": float(best["rho_low"]),
            "ci_high": float(best["rho_high"]),
        }
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)

    # Print headline.
    print("\n" + "=" * 70)
    print("Ablation complete")
    print("=" * 70)
    print(f"Rows evaluated: {res_df.shape[0]}")
    print("Best-per-endpoint (Spearman ρ, out-of-fold, grouped-CV):")
    for endpoint, info in summary["headline_per_endpoint"].items():
        print(f"  {endpoint:18s}  {info['descriptor']:22s}  {info['model']:18s}  "
              f"ρ={info['rho']:+.3f}  [{info['ci_low']:+.3f}, {info['ci_high']:+.3f}]")
    print(f"\nWritten: {args.out}")
    print(f"         {args.oof_out}")
    print(f"         {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
