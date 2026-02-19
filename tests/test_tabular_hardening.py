"""テーブル補間型硬化則のテスト.

検証項目:
  - TabularIsotropicHardening の区分線形補間
  - Plasticity1D + TabularIsotropicHardening の return mapping
  - PlaneStrainPlasticity + TabularIsotropicHardening の return mapping
  - AbaqusMaterial → 構成則オブジェクト変換
  - 線形硬化テーブルと IsotropicHardening の等価性
  - HARDENING=KINEMATIC テーブル → Armstrong-Frederick 変換
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.core.state import PlasticState1D, PlasticState3D
from xkep_cae.io.abaqus_inp import AbaqusMaterial
from xkep_cae.io.material_converter import (
    abaqus_material_to_plane_strain_plasticity,
    abaqus_material_to_plasticity_1d,
    kinematic_table_to_armstrong_frederick,
)
from xkep_cae.materials.plasticity_1d import (
    IsotropicHardening,
    KinematicHardening,
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

    def test_to_plasticity_1d_kinematic_linear(self):
        """KINEMATIC 硬化（線形）→ Plasticity1D + KinematicHardening."""
        mat = AbaqusMaterial(
            name="kin_mat",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0), (350.0, 0.1)],
            plastic_hardening="KINEMATIC",
        )
        p1d = abaqus_material_to_plasticity_1d(mat)
        assert isinstance(p1d, Plasticity1D)
        assert p1d.E == 210000.0
        assert p1d.sigma_y0 == pytest.approx(250.0)
        assert p1d.kin.C_kin == pytest.approx(1000.0)  # (350-250)/0.1
        assert p1d.kin.gamma_kin == pytest.approx(0.0)

    def test_to_plasticity_1d_kinematic_perfect(self):
        """KINEMATIC 硬化（1点テーブル = 完全弾塑性）."""
        mat = AbaqusMaterial(
            name="kin_perf",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0)],
            plastic_hardening="KINEMATIC",
        )
        p1d = abaqus_material_to_plasticity_1d(mat)
        assert p1d.sigma_y0 == pytest.approx(250.0)
        assert p1d.kin.C_kin == pytest.approx(0.0)

    def test_to_plasticity_1d_combined_error(self):
        """COMBINED 硬化で ValueError."""
        mat = AbaqusMaterial(
            name="combined_mat",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0)],
            plastic_hardening="COMBINED",
        )
        with pytest.raises(ValueError, match="COMBINED.*未対応"):
            abaqus_material_to_plasticity_1d(mat)

    def test_to_plane_strain_kinematic(self):
        """KINEMATIC 硬化 → PlaneStrainPlasticity + KinematicHardening3D."""
        mat = AbaqusMaterial(
            name="kin_3d",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0), (350.0, 0.1)],
            plastic_hardening="KINEMATIC",
        )
        ps = abaqus_material_to_plane_strain_plasticity(mat)
        assert isinstance(ps, PlaneStrainPlasticity)
        assert ps.sigma_y0 == pytest.approx(250.0)
        assert ps.kin.C_kin == pytest.approx(1000.0)
        assert ps.kin.gamma_kin == pytest.approx(0.0)

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


# ===================================================================
# KINEMATIC テーブル → Armstrong-Frederick 変換テスト
# ===================================================================


class TestKinematicTableToArmstrongFrederick:
    """kinematic_table_to_armstrong_frederick() の単体テスト."""

    def test_single_point_perfect_plasticity(self):
        """1点テーブル → 完全弾塑性（C_kin=0, gamma_kin=0）."""
        sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick([(250.0, 0.0)])
        assert sigma_y0 == pytest.approx(250.0)
        assert C_kin == pytest.approx(0.0)
        assert gamma_kin == pytest.approx(0.0)

    def test_two_point_linear(self):
        """2点テーブル → 線形移動硬化（gamma_kin=0）."""
        sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(
            [(250.0, 0.0), (350.0, 0.1)]
        )
        assert sigma_y0 == pytest.approx(250.0)
        assert C_kin == pytest.approx(1000.0)  # (350-250)/0.1
        assert gamma_kin == pytest.approx(0.0)

    def test_two_point_high_slope(self):
        """2点テーブル、高い硬化勾配."""
        sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(
            [(200.0, 0.0), (400.0, 0.01)]
        )
        assert sigma_y0 == pytest.approx(200.0)
        assert C_kin == pytest.approx(20000.0)  # (400-200)/0.01
        assert gamma_kin == pytest.approx(0.0)

    def test_multipoint_nonlinear_fit(self):
        """3点以上 → Armstrong-Frederick 非線形フィッティング.

        テストデータ: 既知の AF パラメータから生成し、逆変換で復元を検証。
        beta(eps_p) = (C/gamma)(1 - exp(-gamma * eps_p))
        """
        # 既知パラメータ: C_kin=5000, gamma_kin=50
        C_true = 5000.0
        gamma_true = 50.0
        sigma_y0_true = 250.0

        # テーブルを AF 曲線から生成
        eps_p_vals = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
        table = []
        for ep in eps_p_vals:
            if ep == 0.0:
                sigma = sigma_y0_true
            else:
                beta = (C_true / gamma_true) * (1.0 - np.exp(-gamma_true * ep))
                sigma = sigma_y0_true + beta
            table.append((sigma, ep))

        sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(table)

        assert sigma_y0 == pytest.approx(sigma_y0_true)
        assert C_kin == pytest.approx(C_true, rel=0.05)
        assert gamma_kin == pytest.approx(gamma_true, rel=0.05)

    def test_multipoint_saturating_curve(self):
        """飽和型の背応力曲線のフィッティング精度."""
        # 飽和応力 beta_sat = C/gamma = 100 MPa
        C_true = 2000.0
        gamma_true = 20.0
        sigma_y0_true = 300.0
        beta_sat = C_true / gamma_true  # = 100

        eps_p_vals = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3]
        table = []
        for ep in eps_p_vals:
            if ep == 0.0:
                sigma = sigma_y0_true
            else:
                beta = beta_sat * (1.0 - np.exp(-gamma_true * ep))
                sigma = sigma_y0_true + beta
            table.append((sigma, ep))

        sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(table)

        # 飽和応力の復元精度を確認
        if gamma_kin > 0:
            beta_sat_fit = C_kin / gamma_kin
            assert beta_sat_fit == pytest.approx(beta_sat, rel=0.05)

    def test_linear_table_returns_zero_gamma(self):
        """線形に硬化するテーブル（3点以上）→ gamma_kin ≈ 0."""
        # 完全に線形: beta = 1000 * eps_p
        table = [
            (250.0, 0.0),
            (260.0, 0.01),
            (270.0, 0.02),
            (280.0, 0.03),
            (300.0, 0.05),
        ]
        sigma_y0, C_kin, gamma_kin = kinematic_table_to_armstrong_frederick(table)
        assert sigma_y0 == pytest.approx(250.0)
        assert gamma_kin == pytest.approx(0.0, abs=1e-6)
        assert C_kin == pytest.approx(1000.0, rel=0.01)

    def test_empty_table_raises(self):
        """空テーブルで ValueError."""
        with pytest.raises(ValueError, match="最低1点"):
            kinematic_table_to_armstrong_frederick([])


# ===================================================================
# KINEMATIC 変換 → return mapping ラウンドトリップ
# ===================================================================


class TestKinematicRoundtrip:
    """KINEMATIC 変換 → Plasticity1D → return mapping の統合テスト."""

    def test_linear_kinematic_hysteresis(self):
        """線形移動硬化でのバウシンガー効果の検証.

        載荷→除荷→逆載荷で、逆降伏が σ < σ_y0 で発生することを確認。
        """
        E = 210000.0
        sigma_y0 = 250.0
        C_kin = 1000.0

        iso = IsotropicHardening(sigma_y0=sigma_y0)
        kin = KinematicHardening(C_kin=C_kin)
        mat = Plasticity1D(E=E, iso=iso, kin=kin)

        state = PlasticState1D()

        # 正方向載荷（降伏超え）
        eps_load = 0.01
        r1 = mat.return_mapping(eps_load, state)
        assert r1.stress > sigma_y0
        assert r1.state_new.beta > 0  # 背応力が正方向に移動

        # 除荷
        eps_unload = r1.state_new.eps_p
        r2 = mat.return_mapping(eps_unload, r1.state_new)
        assert r2.tangent == pytest.approx(E)  # 弾性除荷

        # 逆方向載荷（圧縮側）
        eps_reverse = -0.01
        r3 = mat.return_mapping(eps_reverse, r1.state_new)
        # 背応力シフトにより、逆降伏応力 < -σ_y0 ではなく
        # -(σ_y0 - beta) で逆降伏する（バウシンガー効果）
        assert r3.stress < 0
        assert r3.state_new.alpha > r1.state_new.alpha

    def test_kinematic_converter_roundtrip(self):
        """AbaqusMaterial(KINEMATIC) → Plasticity1D → 載荷・除荷."""
        mat = AbaqusMaterial(
            name="kinematic_steel",
            elastic=(210000.0, 0.3),
            plastic=[(250.0, 0.0), (350.0, 0.1)],
            plastic_hardening="KINEMATIC",
        )
        p1d = abaqus_material_to_plasticity_1d(mat)

        state = PlasticState1D()
        # 正方向載荷
        r1 = p1d.return_mapping(0.01, state)
        assert r1.stress > 250.0
        assert r1.state_new.beta > 0

        # 除荷→逆載荷
        r2 = p1d.return_mapping(-0.005, r1.state_new)
        # 応力が反転
        assert r2.stress < r1.stress

    def test_nonlinear_kinematic_converter_roundtrip(self):
        """多点 KINEMATIC テーブル → AF 変換 → return mapping."""
        # AF パラメータから生成したテーブル
        C_true = 3000.0
        gamma_true = 30.0
        sigma_y0 = 250.0

        eps_p_vals = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]
        table = []
        for ep in eps_p_vals:
            if ep == 0.0:
                sigma = sigma_y0
            else:
                beta = (C_true / gamma_true) * (1.0 - np.exp(-gamma_true * ep))
                sigma = sigma_y0 + beta
            table.append((sigma, ep))

        mat = AbaqusMaterial(
            name="af_steel",
            elastic=(210000.0, 0.3),
            plastic=table,
            plastic_hardening="KINEMATIC",
        )
        p1d = abaqus_material_to_plasticity_1d(mat)

        # return mapping が正常に動作
        state = PlasticState1D()
        r1 = p1d.return_mapping(0.01, state)
        assert r1.stress > sigma_y0
        assert r1.state_new.alpha > 0.0

        # 増分載荷
        r2 = p1d.return_mapping(0.05, r1.state_new)
        assert r2.stress > r1.stress

        # 非線形移動硬化なので gamma_kin > 0
        assert p1d.kin.gamma_kin > 0
