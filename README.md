# struct-devpred

> Systematic ablation of sequence descriptors vs. structure-informed baselines for antibody developability prediction under leakage-resistant cross-validation.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)

## Key results

| Descriptor family | Dim | Best endpoint | Best rho | Model |
|-------------------|-----|---------------|----------|-------|
| Physicochemical | 7 | HIC-RT | **0.41** | Random Forest |
| ESM-2 t12 embeddings | 960 | HIC-RT | 0.35 | Ridge |
| Structure-informed (TAP/DI/CamSol) | 3 | Tm onset | 0.28 | Gradient Boosting |
| Composition | 20 | HIC-RT | 0.33 | Random Forest |
| Hybrid (seq + struct) | varies | overall | 0.33 | Ridge |

180 experiments (10 descriptor families x 6 endpoints x 3 models) on 1,243 antibodies. 5-fold GroupKFold with V-gene x CDR-H3 length x 50% identity clustering. Bootstrap 95% CIs on all Spearman correlations.

**Key finding:** Structure-informed descriptors (TAP, DI, CamSol) do not uniformly beat simple physicochemical baselines. A 7D physicochemical feature set outperforms 960D ESM-2 embeddings on HIC retention time. Null and weak results are reported honestly.

## Problem

Antibody developability prediction papers commonly report results under random train/test splits that leak information through shared V-gene families and similar CDR-H3 loops. No open benchmark systematically compares descriptor families under leakage-resistant CV on the same clinical-stage dataset.

## Approach

Computes descriptor families on antibodies in the configured harmonized panel, trains three model families (Ridge, Random Forest, Gradient Boosting) per endpoint, and produces a full ablation table with out-of-fold Spearman rho and bootstrap 95% CIs.

Six biophysical endpoints from Jain et al. 2017: Tm onset, HIC retention time, AC-SINS, BVP score, PSR score, HEK expression.

## Quick start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    python -m struct_devpred.cli \
      --endpoints tm_onset_c hic_rt ac_sins bvp_score psr_score expression_mgl \
      --out reports/ablation_results.csv \
      --oof-out data/processed/oof_predictions.parquet

## Reproduction

    # 1. Run the full ablation grid
    python -m struct_devpred.cli \
      --endpoints tm_onset_c hic_rt ac_sins bvp_score psr_score expression_mgl \
      --out reports/ablation_results.csv \
      --oof-out data/processed/oof_predictions.parquet
    # 2. Per-slice error analysis
    python -m struct_devpred.error_analysis \
      --ablation reports/ablation_results.csv \
      --oof data/processed/oof_predictions.parquet \
      --out reports/error_analysis.csv

Requires sibling repos by default: `../ab-benchmark/` for harmonized data and `../ProtePilot/` for optional ESM-2 embeddings. Override with `AB_BENCHMARK_PATH`, `AB_BENCHMARK_WIDE`, `PROTEPILOT_PATH`, or explicit `--harmonized`.

## Citation

    @software{wu2026structdevpred,
      author = {Wu, Di},
      title  = {struct-devpred: Descriptor Ablation for Antibody Developability Prediction},
      year   = {2026},
      url    = {https://github.com/diwuhub/struct-devpred}
    }

## References

- Jain et al. 2017 PNAS -- Clinical-stage antibody biophysical panel (DOI: 10.1073/pnas.1616408114)
- Raybould et al. 2019 PNAS -- TAP (Therapeutic Antibody Profiler)
- Lauer et al. 2012 J. Pharm. Sci. -- Developability Index
- Sormanni et al. 2015 J. Mol. Biol. -- CamSol-intrinsic
- Lin et al. 2023 Science -- ESM-2 protein language model

## License

MIT. See LICENSE.
