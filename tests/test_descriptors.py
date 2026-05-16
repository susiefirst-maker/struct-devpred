"""Tests for src/struct_devpred/descriptors.py."""

import numpy as np
import pandas as pd
import pytest

from struct_devpred.descriptors import (
    DESCRIPTOR_FAMILIES,
    _net_charge,
    _pi,
    aggregation_features,
    build_descriptor_matrix,
    build_hybrid_matrix,
    cdr_length_features,
    composition_features,
    humanness_features,
    physicochem_features,
    structure_informed_features,
)

TRAS_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRAS_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


class TestNetCharge:
    def test_poly_lysine_positive(self):
        assert _net_charge("KKKKKKKK", 7.0) > 5

    def test_poly_glutamate_negative(self):
        assert _net_charge("EEEEEEEE", 7.0) < -5

    def test_charge_decreases_with_rising_ph(self):
        for seq in [TRAS_VH, TRAS_VL]:
            low = _net_charge(seq, 4.0)
            high = _net_charge(seq, 10.0)
            assert low > high


class TestPi:
    def test_trastuzumab_vh_in_plausible_range(self):
        # Therapeutic mAb variable regions typically have pI 7-9.
        pi = _pi(TRAS_VH)
        assert 5.0 < pi < 11.0

    def test_poly_lys_high_pi(self):
        assert _pi("KKKKKKKKKK") > 9

    def test_poly_glu_low_pi(self):
        assert _pi("EEEEEEEEEE") < 5


class TestPerFamilyFeatures:
    def test_composition(self):
        v, names = composition_features(TRAS_VH, TRAS_VL)
        assert v.shape == (21,)
        assert len(names) == 21
        assert abs(v[:20].sum() - 1.0) < 1e-4  # AA fractions sum to 1
        assert v[20] > 0  # length normalized > 0

    def test_physicochem(self):
        v, names = physicochem_features(TRAS_VH, TRAS_VL)
        assert v.shape == (7,)
        # pI in plausible range
        assert 5 < v[0] < 11
        # MW in kDa — combined VH+VL ≈ 25 kDa
        assert 20 < v[1] < 30

    def test_cdr_lengths(self):
        v, names = cdr_length_features(TRAS_VH, TRAS_VL)
        assert v.shape == (4,)
        # h3 should be nonzero if ab-benchmark importable
        # (and plausibly short, ≤ 25)
        assert v[2] == 0 or 5 < v[2] < 25

    def test_structure_informed_returns_finite(self):
        v, names = structure_informed_features(TRAS_VH, TRAS_VL)
        assert v.ndim == 1
        assert np.all(np.isfinite(v))
        # Should be > 1 if ab-benchmark available; = 1 otherwise.
        assert len(v) >= 1

    def test_humanness(self):
        v, names = humanness_features(TRAS_VH, TRAS_VL)
        assert v.shape == (2,)
        assert 0 <= v[0] <= 1
        assert 0 <= v[1] <= 1

    def test_aggregation(self):
        v, names = aggregation_features(TRAS_VH, TRAS_VL)
        assert v.shape == (4,)
        # mean KD is in [-4.5, 4.5]
        assert -4.5 <= v[0] <= 4.5

    def test_empty_seq_returns_zeros_not_nan(self):
        for fn in [composition_features, physicochem_features, cdr_length_features,
                   humanness_features, aggregation_features]:
            v, _ = fn("", "")
            assert np.all(np.isfinite(v))


class TestBuildMatrix:
    def test_single_family_shape(self):
        df = pd.DataFrame([
            {"ab_id_canonical": "a1", "vh": TRAS_VH, "vl": TRAS_VL},
            {"ab_id_canonical": "a2", "vh": TRAS_VH, "vl": TRAS_VL},
        ])
        for fam in DESCRIPTOR_FAMILIES:
            X, names, ids = build_descriptor_matrix(df, fam)
            assert X.shape[0] == 2
            assert X.shape[1] == len(names)
            assert ids == ["a1", "a2"]

    def test_hybrid_matrix_concatenates(self):
        df = pd.DataFrame([
            {"ab_id_canonical": "a1", "vh": TRAS_VH, "vl": TRAS_VL},
        ])
        X, names, ids = build_hybrid_matrix(df, ["composition", "physicochem"])
        assert X.shape == (1, 21 + 7)
        assert len(names) == 28
        assert all(n.startswith("composition.") or n.startswith("physicochem.") for n in names)

    def test_unknown_family_raises(self):
        df = pd.DataFrame([{"ab_id_canonical": "a1", "vh": TRAS_VH, "vl": TRAS_VL}])
        with pytest.raises(ValueError, match="Unknown"):
            build_descriptor_matrix(df, "fictional_family")

    def test_empty_hybrid_raises(self):
        df = pd.DataFrame([{"ab_id_canonical": "a1", "vh": TRAS_VH, "vl": TRAS_VL}])
        with pytest.raises(ValueError, match="at least one"):
            build_hybrid_matrix(df, [])


class TestEsm2Fallback:
    def test_esm2_with_no_cache_returns_zeros(self):
        df = pd.DataFrame([{"ab_id_canonical": "a1", "vh": TRAS_VH, "vl": TRAS_VL}])
        X, names, ids = build_descriptor_matrix(df, "esm2_t12", esm_cache=None)
        assert X.shape == (1, 960)
        assert np.all(X == 0)
        assert len(names) == 960
