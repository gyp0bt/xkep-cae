"""テーブル補間型硬化則のテスト.

検証項目:
  - TabularIsotropicHardening の区分線形補間
  - Plasticity1D + TabularIsotropicHardening の return mapping
  - PlaneStrainPlasticity + TabularIsotropicHardening の return mapping
  - AbaqusMaterial → 構成則オブジェクト変換
  - 線形硬化テーブルと IsotropicHardening の等価性
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.core.state import PlasticState1D, PlasticState3D
from xkep_cae.io.abaqus_inp import AbaqusMaterial
from xkep_cae.io.material_converter import (
    abaqus_material_to_plane_strain_plasticity,
    abaqus_material_to_plasticity_1d,
)
from xkep_cae.materials.plasticity_1d import (
    IsotropicHardening,
    Plasticity1D,
    TabularIsotropicHardening,
)
from xkep_cae.materials.plasticity_3d import (
    IsotropicHardening3D,
    PlaneStrainPlasticity,
)

# ===================================================================
# TabularIsotropicHardening 単体テスト
# ===================================================================


class TestTabularIsotropicHardening:
    """テーブル補間の基本動作テスト."""

    def test_sigma_y0(self):
        """初期降伏応力."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.1)])
        assert tab.sigma_y0 == 250.0

    def test_sigma_y_at_table_points(self):
        """テーブル上の点で正確な値."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (300.0, 0.05), (350.0, 0.10)])
        assert tab.sigma_y(0.0) == pytest.approx(250.0)
        assert tab.sigma_y(0.05) == pytest.approx(300.0)
        assert tab.sigma_y(0.10) == pytest.approx(350.0)

    def test_sigma_y_interpolation(self):
        """テーブル間の線形補間."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (300.0, 0.10)])
        assert tab.sigma_y(0.05) == pytest.approx(275.0)
        assert tab.sigma_y(0.025) == pytest.approx(262.5)

    def test_sigma_y_extrapolation_beyond_table(self):
        """テーブル範囲外は最終値で一定."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.10)])
        assert tab.sigma_y(0.20) == pytest.approx(350.0)
        assert tab.sigma_y(1.0) == pytest.approx(350.0)

    def test_dR_dalpha_within_table(self):
        """テーブル内の硬化係数."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (300.0, 0.05), (350.0, 0.10)])
        # 第1区間: (300-250)/(0.05-0) = 1000
        assert tab.dR_dalpha(0.0) == pytest.approx(1000.0)
        assert tab.dR_dalpha(0.03) == pytest.approx(1000.0)
        # 第2区間: (350-300)/(0.10-0.05) = 1000
        assert tab.dR_dalpha(0.07) == pytest.approx(1000.0)

    def test_dR_dalpha_variable_slope(self):
        """区間ごとに異なる勾配."""
        tab = TabularIsotropicHardening(table=[(200.0, 0.0), (300.0, 0.05), (320.0, 0.10)])
        # 第1区間: (300-200)/(0.05) = 2000
        assert tab.dR_dalpha(0.02) == pytest.approx(2000.0)
        # 第2区間: (320-300)/(0.05) = 400
        assert tab.dR_dalpha(0.07) == pytest.approx(400.0)

    def test_dR_dalpha_beyond_table(self):
        """テーブル範囲外は硬化なし."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.10)])
        assert tab.dR_dalpha(0.15) == pytest.approx(0.0)

    def test_single_point_table(self):
        """1点テーブル（完全弾塑性）."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0)])
        assert tab.sigma_y0 == 250.0
        assert tab.sigma_y(0.0) == 250.0
        assert tab.sigma_y(0.1) == 250.0
        assert tab.dR_dalpha(0.0) == 0.0
        assert tab.dR_dalpha(0.1) == 0.0

    def test_invalid_non_monotonic(self):
        """eps_p 非単調で ValueError."""
        with pytest.raises(ValueError, match="単調増加"):
            TabularIsotropicHardening(table=[(250.0, 0.0), (300.0, 0.1), (350.0, 0.05)])

    def test_invalid_empty_table(self):
        """空テーブルで ValueError."""
        with pytest.raises(ValueError, match="最低1点"):
            TabularIsotropicHardening(table=[])


# ===================================================================
# Plasticity1D + TabularIsotropicHardening テスト
# ===================================================================


class TestPlasticity1DTabular:
    """Plasticity1D にテーブル硬化を適用するテスト."""

    def test_elastic_range(self):
        """降伏前は弾性応答."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.1)])
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()
        result = mat.return_mapping(0.001, state)
        assert result.stress == pytest.approx(210.0)
        assert result.tangent == pytest.approx(210000.0)

    def test_yielding(self):
        """降伏後の応力が降伏応力以上."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.1)])
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()
        # 降伏応力 250 → eps_y = 250/210000 ≈ 0.00119
        result = mat.return_mapping(0.01, state)
        assert result.stress > 250.0
        assert result.state_new.alpha > 0.0
        assert result.state_new.eps_p > 0.0

    def test_consistent_tangent_positive(self):
        """硬化中の consistent tangent は正."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.1)])
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()
        result = mat.return_mapping(0.01, state)
        assert result.tangent > 0.0
        assert result.tangent < 210000.0  # 弾性剛性より小

    def test_perfect_plasticity(self):
        """完全弾塑性（1点テーブル）."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0)])
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()
        result = mat.return_mapping(0.01, state)
        assert result.stress == pytest.approx(250.0, abs=0.01)
        assert result.tangent == pytest.approx(0.0, abs=1e-6)

    def test_equivalence_with_linear_hardening(self):
        """線形テーブルと IsotropicHardening の等価性検証."""
        E = 210000.0
        sigma_y0 = 250.0
        H_iso = 1000.0

        # 線形硬化（既存）
        iso_linear = IsotropicHardening(sigma_y0=sigma_y0, H_iso=H_iso)
        mat_linear = Plasticity1D(E=E, iso=iso_linear)

        # テーブル（等価な2点線形）
        # eps_p=0: sigma_y=250, eps_p=1.0: sigma_y=250+1000*1.0=1250
        tab = TabularIsotropicHardening(table=[(sigma_y0, 0.0), (sigma_y0 + H_iso * 1.0, 1.0)])
        mat_tabular = Plasticity1D(E=E, iso=tab)

        state = PlasticState1D()
        strains = [0.001, 0.005, 0.01, 0.02, 0.05]

        for eps in strains:
            r_lin = mat_linear.return_mapping(eps, state)
            r_tab = mat_tabular.return_mapping(eps, state)
            assert r_tab.stress == pytest.approx(r_lin.stress, rel=1e-8), (
                f"eps={eps}: stress mismatch"
            )
            assert r_tab.tangent == pytest.approx(r_lin.tangent, rel=1e-6), (
                f"eps={eps}: tangent mismatch"
            )
            assert r_tab.state_new.eps_p == pytest.approx(r_lin.state_new.eps_p, rel=1e-8)
            assert r_tab.state_new.alpha == pytest.approx(r_lin.state_new.alpha, rel=1e-8)

    def test_incremental_loading(self):
        """増分載荷で応力が単調増加."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (300.0, 0.05), (350.0, 0.10)])
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()

        stresses = []
        for eps in [0.002, 0.005, 0.01, 0.02, 0.05]:
            result = mat.return_mapping(eps, state)
            stresses.append(result.stress)
            state = result.state_new

        # 応力は単調増加
        for i in range(len(stresses) - 1):
            assert stresses[i + 1] >= stresses[i]

    def test_unloading_elastic(self):
        """除荷は弾性的（逆降伏しない範囲）."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.10)])
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()

        # 載荷 → 塑性
        result = mat.return_mapping(0.01, state)
        state1 = result.state_new
        sigma_max = result.stress

        # わずかに除荷（逆降伏しない範囲: eps_p ± sigma_y/E 以内）
        eps_unload = state1.eps_p + (sigma_max - 10.0) / 210000.0
        result2 = mat.return_mapping(eps_unload, state1)
        assert result2.tangent == pytest.approx(210000.0)
        assert result2.stress < sigma_max

    def test_multilinear_stress_path(self):
        """多直線テーブルでの応力経路."""
        # 3区間: 急な硬化 → 緩い硬化 → 完全塑性的
        tab = TabularIsotropicHardening(
            table=[
                (200.0, 0.0),
                (400.0, 0.01),  # H=20000
                (450.0, 0.05),  # H=1250
                (460.0, 0.10),  # H=200
            ]
        )
        mat = Plasticity1D(E=210000.0, iso=tab)
        state = PlasticState1D()

        # 大きな歪みで降伏後の応力を確認
        result = mat.return_mapping(0.10, state)
        # 応力は初期降伏応力より大きい
        assert result.stress > 200.0
        # テーブル最大値以下
        assert result.stress <= 460.0 + 1.0  # 少し余裕


# ===================================================================
# PlaneStrainPlasticity + TabularIsotropicHardening テスト
# ===================================================================


class TestPlaneStrainPlasticityTabular:
    """PlaneStrainPlasticity にテーブル硬化を適用するテスト."""

    def test_elastic_range(self):
        """降伏前は弾性."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.1)])
        mat = PlaneStrainPlasticity(E=210000.0, nu=0.3, iso=tab)
        state = PlasticState3D()
        eps = np.array([0.0005, -0.0005 * 0.3, 0.0])
        result = mat.return_mapping(eps, state)
        assert result.state_new.alpha == pytest.approx(0.0)

    def test_yielding_uniaxial(self):
        """単軸引張で降伏."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (350.0, 0.1)])
        mat = PlaneStrainPlasticity(E=210000.0, nu=0.3, iso=tab)
        state = PlasticState3D()
        # 大きな歪みで降伏を確認
        eps = np.array([0.01, 0.0, 0.0])
        result = mat.return_mapping(eps, state)
        assert result.state_new.alpha > 0.0

    def test_equivalence_linear_hardening_3d(self):
        """線形テーブルと IsotropicHardening3D の等価性."""
        E = 210000.0
        nu = 0.3
        sigma_y0 = 250.0
        H_iso = 1000.0

        iso_linear = IsotropicHardening3D(sigma_y0=sigma_y0, H_iso=H_iso)
        mat_linear = PlaneStrainPlasticity(E=E, nu=nu, iso=iso_linear)

        tab = TabularIsotropicHardening(table=[(sigma_y0, 0.0), (sigma_y0 + H_iso * 1.0, 1.0)])
        mat_tabular = PlaneStrainPlasticity(E=E, nu=nu, iso=tab)

        state = PlasticState3D()
        eps = np.array([0.01, 0.0, 0.0])

        r_lin = mat_linear.return_mapping(eps, state)
        r_tab = mat_tabular.return_mapping(eps, state)

        assert r_tab.stress == pytest.approx(r_lin.stress, rel=1e-6)
        assert r_tab.state_new.alpha == pytest.approx(r_lin.state_new.alpha, rel=1e-6)

    def test_consistent_tangent_symmetry(self):
        """consistent tangent の対称性."""
        tab = TabularIsotropicHardening(table=[(250.0, 0.0), (300.0, 0.05), (350.0, 0.10)])
        mat = PlaneStrainPlasticity(E=210000.0, nu=0.3, iso=tab)
        state = PlasticState3D()
        eps = np.array([0.01, 0.0, 0.0])
        result = mat.return_mapping(eps, state)
        D = result.tangent
        assert D == pytest.approx(D.T, abs=1e-6)


# ===================================================================
# AbaqusMaterial → 構成則変換テスト
# ===================================================================


class TestMaterialConverter:
    """AbaqusMaterial からの変換テスト."""

    def test_to_plasticity_1d_with_table(self):
        """テーブル硬化 → Plasticity1D."""
        mat = AbaqusMaterial(
            name="steel",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0), (350.0, 0.1)],
        )
        p1d = abaqus_material_to_plasticity_1d(mat)
        assert isinstance(p1d, Plasticity1D)
        assert p1d.E == 210000.0
        assert isinstance(p1d.iso, TabularIsotropicHardening)
        assert p1d.iso.sigma_y0 == 250.0

    def test_to_plasticity_1d_elastic_only(self):
        """弾性のみ → Plasticity1D（降伏しない）."""
        mat = AbaqusMaterial(
            name="elastic_mat",
            elastic=(210000.0, 0.3),
        )
        p1d = abaqus_material_to_plasticity_1d(mat)
        assert isinstance(p1d, Plasticity1D)
        # 弾性範囲内
        state = PlasticState1D()
        result = p1d.return_mapping(0.01, state)
        assert result.tangent == pytest.approx(210000.0)

    def test_to_plasticity_1d_no_elastic_error(self):
        """*ELASTIC なしで ValueError."""
        mat = AbaqusMaterial(name="no_elastic")
        with pytest.raises(ValueError, match="ELASTIC.*未定義"):
            abaqus_material_to_plasticity_1d(mat)

    def test_to_plasticity_1d_kinematic_error(self):
        """KINEMATIC 硬化で ValueError."""
        mat = AbaqusMaterial(
            name="kin_mat",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0)],
            plastic_hardening="KINEMATIC",
        )
        with pytest.raises(ValueError, match="KINEMATIC.*未対応"):
            abaqus_material_to_plasticity_1d(mat)

    def test_to_plane_strain_plasticity(self):
        """テーブル硬化 → PlaneStrainPlasticity."""
        mat = AbaqusMaterial(
            name="steel",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0), (300.0, 0.05), (350.0, 0.10)],
        )
        ps = abaqus_material_to_plane_strain_plasticity(mat)
        assert isinstance(ps, PlaneStrainPlasticity)
        assert ps.E == 210000.0
        assert ps.nu == 0.3

    def test_to_plane_strain_no_elastic_error(self):
        """*ELASTIC なしで ValueError."""
        mat = AbaqusMaterial(name="no_elastic")
        with pytest.raises(ValueError, match="ELASTIC.*未定義"):
            abaqus_material_to_plane_strain_plasticity(mat)

    def test_roundtrip_return_mapping(self):
        """パーサー → コンバータ → return mapping のラウンドトリップ."""
        mat = AbaqusMaterial(
            name="mild_steel",
            elastic=(210000.0, 0.3),
            plastic=[
                (250.0, 0.0),
                (300.0, 0.05),
                (350.0, 0.10),
            ],
        )
        p1d = abaqus_material_to_plasticity_1d(mat)

        state = PlasticState1D()
        # 載荷
        result = p1d.return_mapping(0.01, state)
        assert result.stress > 250.0
        assert result.state_new.alpha > 0.0

        # さらに載荷
        result2 = p1d.return_mapping(0.05, result.state_new)
        assert result2.stress > result.stress
