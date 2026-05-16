"""Descriptor-family computation for struct-devpred.

Seven families; every function returns a dict {ab_id_canonical: np.ndarray}
and a list of feature names — kept symmetric so the benchmark loop can
stack them uniformly.

Families:
    1. composition        — AA frequencies (20) + length (1)  → 21 dims
    2. physicochemistry   — pI, MW, GRAVY, net charge pH 7.4, pH 6.0,
                            aromatic fraction, polar fraction            →  7 dims
    3. cdr_lengths        — H1/H2/H3/L3 lengths (missing → 0)            →  4 dims
    4. esm2_t12           — 960-dim VH+VL mean-pool from ProtePilot cache → 960 dims
    5. structure_informed — TAP risk-flag metrics + DI-seq + CamSol-intrinsic → ~15 dims
    6. humanness          — Hamming distance to IGHV3-23 + IGKV1-39 germlines
                            (simple proxy; BioPhi/OASis optional)        →  2 dims
    7. aggregation        — mean/max/longest-run Kyte-Doolittle + hydro-patch count → 4 dims

A composite "hybrid_all" concatenation is built by the benchmark loop
on demand — no need to pre-compute.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --- import shims for sibling repos ----------------------------------------

_PROJECTS_ROOT = Path(__file__).resolve().parents[3]
_AB_BENCHMARK = Path(
    os.environ.get("AB_BENCHMARK_PATH", str(_PROJECTS_ROOT / "ab-benchmark"))
)
_PROTEPILOT = Path(
    os.environ.get("PROTEPILOT_PATH", str(_PROJECTS_ROOT / "ProtePilot"))
)


def _ensure_sibling_on_path(path: Path) -> None:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.append(str(path))


# --- canonical AA tables ---------------------------------------------------

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle
_KD = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Monoisotopic residue masses (Da) — enough precision for proxy MW.
_MW = {
    "A": 71.04, "R": 156.10, "N": 114.04, "D": 115.03, "C": 103.01,
    "Q": 128.06, "E": 129.04, "G": 57.02, "H": 137.06, "I": 113.08,
    "L": 113.08, "K": 128.09, "M": 131.04, "F": 147.07, "P": 97.05,
    "S": 87.03, "T": 101.05, "W": 186.08, "Y": 163.06, "V": 99.07,
}

_AROMATIC = set("FWY")
_POLAR = set("STNQCH")
_POSITIVE = set("KR")
_NEGATIVE = set("DE")

# Canonical germlines for humanness proxy.
IGHV3_23 = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTIS"
            "RDNSKNTLYLQMNSLRAEDTAVYYCAK")
IGKV1_39 = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQ"
            "SYSTP")


# --- helpers ---------------------------------------------------------------


def _net_charge(seq: str, ph: float) -> float:
    """Henderson-Hasselbalch net charge estimate."""
    pKa = {"D": 3.65, "E": 4.25, "H": 6.0, "K": 10.53, "R": 12.48, "C": 8.33, "Y": 10.07}
    q = 1.0 / (1.0 + 10 ** (ph - 8.0))  # N-terminal
    q -= 1.0 / (1.0 + 10 ** (3.1 - ph))  # C-terminal
    for a in seq:
        pk = pKa.get(a)
        if pk is None:
            continue
        if a in {"K", "R", "H"}:
            q += 1.0 / (1.0 + 10 ** (ph - pk))
        else:
            q -= 1.0 / (1.0 + 10 ** (pk - ph))
    return q


def _pi(seq: str) -> float:
    """Binary search isoelectric point."""
    lo, hi = 0.0, 14.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mw(seq: str) -> float:
    return sum(_MW.get(a, 110.0) for a in seq) + 18.02


def _hamming_identity(a: str, b: str) -> float:
    """Hamming identity on the shorter length; ignores insertions/deletions."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


# ---------------------------------------------------------------------------
# Per-antibody feature builders
# ---------------------------------------------------------------------------


def composition_features(vh: str, vl: str) -> tuple[np.ndarray, list[str]]:
    seq = (vh or "") + (vl or "")
    names = [f"comp_{aa}" for aa in STANDARD_AA] + ["length"]
    if not seq:
        return np.zeros(len(names), dtype=np.float32), names
    counts = np.array([seq.count(aa) for aa in STANDARD_AA], dtype=np.float32)
    fracs = counts / max(len(seq), 1)
    vals = np.concatenate([fracs, [len(seq) / 300.0]]).astype(np.float32)
    return vals, names


def physicochem_features(vh: str, vl: str) -> tuple[np.ndarray, list[str]]:
    seq = (vh or "") + (vl or "")
    names = [
        "pc_pi", "pc_mw_kda", "pc_gravy",
        "pc_net_charge_ph74", "pc_net_charge_ph60",
        "pc_aromatic_frac", "pc_polar_frac",
    ]
    if not seq:
        return np.zeros(len(names), dtype=np.float32), names
    pi = _pi(seq)
    mw = _mw(seq) / 1000.0
    gravy = sum(_KD.get(a, 0) for a in seq) / len(seq)
    q74 = _net_charge(seq, 7.4)
    q60 = _net_charge(seq, 6.0)
    arom = sum(1 for a in seq if a in _AROMATIC) / len(seq)
    polar = sum(1 for a in seq if a in _POLAR) / len(seq)
    return np.array([pi, mw, gravy, q74, q60, arom, polar], dtype=np.float32), names


def cdr_length_features(vh: str, vl: str) -> tuple[np.ndarray, list[str]]:
    """CDR lengths via ab_benchmark.seqprops regex extractor."""
    names = ["cdr_h1_len", "cdr_h2_len", "cdr_h3_len", "cdr_l3_len"]
    _ensure_sibling_on_path(_AB_BENCHMARK)
    try:
        from ab_benchmark.seqprops import extract_cdrs  # type: ignore
    except ImportError:
        return np.zeros(len(names), dtype=np.float32), names
    cdrs = extract_cdrs(vh or "", vl or "")
    vals = np.array(
        [len(cdrs.get("h1", "")), len(cdrs.get("h2", "")),
         len(cdrs.get("h3", "")), len(cdrs.get("l3", ""))],
        dtype=np.float32,
    )
    return vals, names


def structure_informed_features(vh: str, vl: str) -> tuple[np.ndarray, list[str]]:
    """TAP + Developability Index + CamSol-intrinsic sequence-proxy outputs."""
    _ensure_sibling_on_path(_AB_BENCHMARK)
    try:
        from ab_benchmark.baselines.camsol import compute_camsol_intrinsic
        from ab_benchmark.baselines.developability_index import compute_developability_index
        from ab_benchmark.baselines.tap import compute_tap
        from ab_benchmark.schema import AntibodyRecord, SourceDataset
    except ImportError:
        names = ["si_unavailable"]
        return np.zeros(1, dtype=np.float32), names

    r = AntibodyRecord(ab_id="_x", source=SourceDataset.JAIN_2017, vh=vh or "", vl=vl or "")
    vals: list[float] = []
    names: list[str] = []

    tap_res = compute_tap(r)
    tap_keys = [
        "tap_h3_length", "tap_cdr_total_length", "tap_cdr_mean_hydrophobicity",
        "tap_cdr_net_pos_count", "tap_cdr_net_neg_count", "tap_cdr_his_count",
        "tap_vh_charge", "tap_vl_charge", "tap_fv_charge_asymmetry",
        "tap_risk_flag_count",
    ]
    for k in tap_keys:
        vals.append(float(tap_res.metrics.get(k, 0.0)) if tap_res.available else 0.0)
        names.append(f"si_{k}")

    di_res = compute_developability_index(r)
    di_keys = ["di_sfvcsp_seq", "di_mean_hydrophobicity", "di_hydro_patch_length", "di_seq_proxy"]
    for k in di_keys:
        vals.append(float(di_res.metrics.get(k, 0.0)) if di_res.available else 0.0)
        names.append(f"si_{k}")

    camsol_res = compute_camsol_intrinsic(r)
    camsol_keys = [
        "camsol_intrinsic_mean", "camsol_intrinsic_min",
        "camsol_intrinsic_std", "camsol_intrinsic_frac_negative",
    ]
    for k in camsol_keys:
        vals.append(float(camsol_res.metrics.get(k, 0.0)) if camsol_res.available else 0.0)
        names.append(f"si_{k}")

    return np.asarray(vals, dtype=np.float32), names


def humanness_features(vh: str, vl: str) -> tuple[np.ndarray, list[str]]:
    """Simple germline-identity proxy. A deeper BioPhi/OASis layer is Phase 5+."""
    names = ["human_vh_igv3_23_id", "human_vl_igk1_39_id"]
    h = _hamming_identity(vh or "", IGHV3_23)
    light_id = _hamming_identity(vl or "", IGKV1_39)
    return np.array([h, light_id], dtype=np.float32), names


def aggregation_features(vh: str, vl: str) -> tuple[np.ndarray, list[str]]:
    """Sequence-level aggregation proxies (Kyte-Doolittle patches)."""
    names = ["agg_mean_kd", "agg_max_kd_window5", "agg_longest_hydro_run", "agg_frac_exposed_hydro"]
    seq = (vh or "") + (vl or "")
    if not seq:
        return np.zeros(len(names), dtype=np.float32), names
    kd = np.array([_KD.get(a, 0.0) for a in seq], dtype=np.float32)
    mean_kd = float(kd.mean())
    # max KD over a sliding 5-residue window
    if len(kd) >= 5:
        win = np.convolve(kd, np.ones(5) / 5, mode="valid")
        max_kd = float(win.max())
    else:
        max_kd = mean_kd
    # longest run above hydrophobicity threshold
    run = best = 0
    for v in kd:
        if v > 1.0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    frac_exposed_hydro = float((kd > 1.0).mean())
    return np.array([mean_kd, max_kd, best, frac_exposed_hydro], dtype=np.float32), names


# ---------------------------------------------------------------------------
# ESM-2 t12 features (loaded from ProtePilot cache)
# ---------------------------------------------------------------------------


_ESM_CACHE_NPZ = _PROTEPILOT / "data" / "esm2_cache_ab_benchmark.npz"
_ESM_CACHE_PT = _PROTEPILOT / "data" / "esm2_cache_ab_benchmark.pt"


def _seq_hash(vh: str, vl: str = "") -> str:
    """SHA-256 of (VH | VL) — must match the convention in ProtePilot/src/esm2_features.py."""
    import hashlib
    h = hashlib.sha256()
    h.update((vh or "").strip().upper().encode())
    h.update(b"|")
    h.update((vl or "").strip().upper().encode())
    return h.hexdigest()


def load_esm2_cache() -> dict[str, np.ndarray] | None:
    """Return {seq_hash: embedding}; None if cache isn't on disk.

    Prefers the torch-free .npz format (produced by
    `scripts/export_esm_cache_npz.py` in the sibling ProtePilot repo).
    Falls back to the .pt file only if torch is importable.
    """
    if _ESM_CACHE_NPZ.exists():
        with np.load(_ESM_CACHE_NPZ, allow_pickle=False) as blob:
            emb = blob["embeddings"]
            hashes = blob["hashes"]
        return {str(h): emb[i] for i, h in enumerate(hashes)}

    if _ESM_CACHE_PT.exists():
        try:
            import torch  # type: ignore
        except ImportError:
            return None
        blob = torch.load(_ESM_CACHE_PT, weights_only=False)
        emb = blob["embeddings"].cpu().numpy()
        hash_to_row = blob["hash_to_row"]
        return {h: emb[row] for h, row in hash_to_row.items()}

    return None


def esm2_features(vh: str, vl: str, cache: dict[str, np.ndarray] | None) -> tuple[np.ndarray, list[str]]:
    names = [f"esm2_{i}" for i in range(960)]
    if cache is None:
        return np.zeros(960, dtype=np.float32), names
    key = _seq_hash(vh, vl)
    vec = cache.get(key)
    if vec is None:
        return np.zeros(960, dtype=np.float32), names
    return vec.astype(np.float32), names


# ---------------------------------------------------------------------------
# Family registry + matrix builder
# ---------------------------------------------------------------------------


DESCRIPTOR_FAMILIES = {
    "composition": composition_features,
    "physicochem": physicochem_features,
    "cdr_lengths": cdr_length_features,
    "structure_informed": structure_informed_features,
    "humanness": humanness_features,
    "aggregation": aggregation_features,
    # esm2_t12 is handled separately because it loads a shared cache.
}


def build_descriptor_matrix(
    df: pd.DataFrame,
    family: str,
    esm_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build an (N, D) matrix for one descriptor family, plus feature names + IDs.

    Returns (X, feature_names, ab_ids) where ab_ids is the per-row canonical ID.
    """
    if family == "esm2_t12":
        rows: list[np.ndarray] = []
        feat_names = [f"esm2_{i}" for i in range(960)]
        ab_ids: list[str] = []
        for _, r in df.iterrows():
            vec, _ = esm2_features(str(r["vh"]), str(r["vl"]), esm_cache)
            rows.append(vec)
            ab_ids.append(str(r["ab_id_canonical"]))
        return np.vstack(rows), feat_names, ab_ids

    if family not in DESCRIPTOR_FAMILIES:
        raise ValueError(
            f"Unknown descriptor family {family!r}. "
            f"Available: {list(DESCRIPTOR_FAMILIES)} + ['esm2_t12']"
        )

    fn = DESCRIPTOR_FAMILIES[family]
    rows = []
    ab_ids = []
    feat_names: list[str] | None = None
    for _, r in df.iterrows():
        vec, names = fn(str(r["vh"]), str(r["vl"]))
        rows.append(vec)
        ab_ids.append(str(r["ab_id_canonical"]))
        if feat_names is None:
            feat_names = names
    return np.vstack(rows), (feat_names or []), ab_ids


def build_hybrid_matrix(
    df: pd.DataFrame,
    families: list[str],
    esm_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Concatenate multiple families into one feature matrix."""
    if not families:
        raise ValueError("build_hybrid_matrix requires at least one family")
    Xs: list[np.ndarray] = []
    all_names: list[str] = []
    ab_ids: list[str] | None = None
    for fam in families:
        X, names, ids = build_descriptor_matrix(df, fam, esm_cache=esm_cache)
        Xs.append(X)
        all_names.extend([f"{fam}.{n}" for n in names])
        if ab_ids is None:
            ab_ids = ids
        elif ids != ab_ids:
            raise RuntimeError(
                f"ID mismatch between family {fam!r} and first family"
            )
    return np.concatenate(Xs, axis=1), all_names, (ab_ids or [])
