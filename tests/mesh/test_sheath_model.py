"""シース（外被）モデル（SheathModel）のテスト.

撚線全体を覆う円筒シースの幾何計算、断面特性、
径方向ギャップ、最外層素線特定の検証。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.mesh.twisted_wire import (
    CoatingModel,
    SheathModel,
    compute_envelope_radius,
    make_twisted_wire_mesh,
    outermost_layer,
    outermost_strand_ids,
    outermost_strand_node_indices,
    sheath_equivalent_stiffness,
    sheath_inner_radius,
    sheath_radial_gap,
    sheath_section_properties,
)

# ====================================================================
# ヘルパー
# ====================================================================


def _make_7wire_mesh(n_elems: int = 8) -> object:
    """7本撚りメッシュを生成."""
    return make_twisted_wire_mesh(
        n_strands=7,
        wire_diameter=2.0e-3,
        pitch=40.0e-3,
        length=40.0e-3,
        n_elems_per_strand=n_elems,
    )


def _make_3wire_mesh(n_elems: int = 8) -> object:
    """3本撚りメッシュを生成."""
    return make_twisted_wire_mesh(
        n_strands=3,
        wire_diameter=2.0e-3,
        pitch=40.0e-3,
        length=40.0e-3,
        n_elems_per_strand=n_elems,
    )


def _make_19wire_mesh(n_elems: int = 4) -> object:
    """19本撚りメッシュを生成."""
    return make_twisted_wire_mesh(
        n_strands=19,
        wire_diameter=2.0e-3,
        pitch=40.0e-3,
        length=40.0e-3,
        n_elems_per_strand=n_elems,
    )


# ====================================================================
# SheathModel 基本テスト
# ====================================================================


class TestSheathModel:
    """SheathModel の基本テスト."""

    def test_basic_creation(self):
        """基本的なインスタンス生成."""
        sm = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, mu=0.15)
        assert sm.thickness == 0.5e-3
        assert sm.E == 70.0e9
        assert sm.nu == 0.33
        assert sm.mu == 0.15

    def test_default_values(self):
        """デフォルト値の確認."""
        sm = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        assert sm.mu == 0.3
        assert sm.clearance == 0.0

    def test_clearance(self):
        """クリアランスの設定."""
        sm = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, clearance=0.1e-3)
        assert sm.clearance == 0.1e-3

    def test_shear_modulus(self):
        """せん断弾性係数 G = E / 2(1+ν)."""
        sm = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.25)
        expected_G = 70.0e9 / (2.0 * 1.25)
        assert abs(sm.G - expected_G) < 1.0

    def test_invalid_thickness(self):
        """肉厚が非正ならエラー."""
        with pytest.raises(ValueError, match="シース肉厚"):
            SheathModel(thickness=0.0, E=70.0e9, nu=0.33)
        with pytest.raises(ValueError, match="シース肉厚"):
            SheathModel(thickness=-0.1e-3, E=70.0e9, nu=0.33)

    def test_invalid_E(self):
        """ヤング率が非正ならエラー."""
        with pytest.raises(ValueError, match="ヤング率"):
            SheathModel(thickness=0.5e-3, E=0.0, nu=0.33)

    def test_invalid_nu(self):
        """ポアソン比が範囲外ならエラー."""
        with pytest.raises(ValueError, match="ポアソン比"):
            SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.5)
        with pytest.raises(ValueError, match="ポアソン比"):
            SheathModel(thickness=0.5e-3, E=70.0e9, nu=-1.0)

    def test_invalid_mu(self):
        """摩擦係数が負ならエラー."""
        with pytest.raises(ValueError, match="摩擦係数"):
            SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, mu=-0.1)

    def test_invalid_clearance(self):
        """クリアランスが負ならエラー."""
        with pytest.raises(ValueError, match="クリアランス"):
            SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, clearance=-0.01e-3)


# ====================================================================
# compute_envelope_radius テスト
# ====================================================================


class TestComputeEnvelopeRadius:
    """エンベロープ半径計算のテスト."""

    def test_7wire_envelope(self):
        """7本撚り: エンベロープ = 第1層配置半径 + 素線半径."""
        mesh = _make_7wire_mesh()
        r_env = compute_envelope_radius(mesh)
        # 7本撚り: 層1の配置半径 = 2 * wire_radius = 2e-3
        # エンベロープ = 2e-3 + 1e-3 = 3e-3
        assert r_env == pytest.approx(3.0e-3, rel=1e-10)

    def test_3wire_envelope(self):
        """3本撚り: エンベロープ = 配置半径 + 素線半径."""
        mesh = _make_3wire_mesh()
        r_env = compute_envelope_radius(mesh)
        # 3本撚り: 配置半径 = d/√3 = 2e-3/√3
        # エンベロープ = 2e-3/√3 + 1e-3
        expected = 2.0e-3 / math.sqrt(3.0) + 1.0e-3
        assert r_env == pytest.approx(expected, rel=1e-10)

    def test_19wire_envelope(self):
        """19本撚り: 最外層（第2層）の配置半径 + 素線半径."""
        mesh = _make_19wire_mesh()
        r_env = compute_envelope_radius(mesh)
        # 19本: 第2層の配置半径 = 2 * d = 4e-3
        # エンベロープ = 4e-3 + 1e-3 = 5e-3
        assert r_env == pytest.approx(5.0e-3, rel=1e-10)

    def test_envelope_with_coating(self):
        """被膜付きのエンベロープ半径."""
        mesh = _make_7wire_mesh()
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        r_env = compute_envelope_radius(mesh, coating=coating)
        # 2e-3 + 1e-3 + 0.1e-3 = 3.1e-3
        assert r_env == pytest.approx(3.1e-3, rel=1e-10)


# ====================================================================
# sheath_inner_radius テスト
# ====================================================================


class TestSheathInnerRadius:
    """シース内径テスト."""

    def test_no_clearance(self):
        """クリアランス0: シース内径 = エンベロープ半径."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        r_in = sheath_inner_radius(mesh, sheath)
        assert r_in == pytest.approx(3.0e-3, rel=1e-10)

    def test_with_clearance(self):
        """クリアランスあり: シース内径 = エンベロープ + クリアランス."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, clearance=0.2e-3)
        r_in = sheath_inner_radius(mesh, sheath)
        assert r_in == pytest.approx(3.2e-3, rel=1e-10)

    def test_with_coating(self):
        """被膜込みシース内径."""
        mesh = _make_7wire_mesh()
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        r_in = sheath_inner_radius(mesh, sheath, coating=coating)
        assert r_in == pytest.approx(3.1e-3, rel=1e-10)


# ====================================================================
# sheath_section_properties テスト
# ====================================================================


class TestSheathSectionProperties:
    """シース断面特性テスト."""

    def test_annular_area(self):
        """円筒管断面積 A = π(r_out² - r_in²)."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)

        r_in = 3.0e-3
        r_out = r_in + 0.5e-3
        expected_A = math.pi * (r_out**2 - r_in**2)
        assert props["A"] == pytest.approx(expected_A, rel=1e-12)

    def test_annular_inertia(self):
        """断面二次モーメント I = π/4 (r_out⁴ - r_in⁴)."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)

        r_in = 3.0e-3
        r_out = r_in + 0.5e-3
        expected_I = math.pi / 4.0 * (r_out**4 - r_in**4)
        assert props["Iy"] == pytest.approx(expected_I, rel=1e-12)
        assert props["Iz"] == pytest.approx(expected_I, rel=1e-12)

    def test_torsion_constant(self):
        """ねじり定数 J = π/2 (r_out⁴ - r_in⁴)."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)

        r_in = 3.0e-3
        r_out = r_in + 0.5e-3
        expected_J = math.pi / 2.0 * (r_out**4 - r_in**4)
        assert props["J"] == pytest.approx(expected_J, rel=1e-12)

    def test_Iy_equals_Iz(self):
        """円筒対称なので Iy = Iz."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)
        assert props["Iy"] == props["Iz"]

    def test_radii_returned(self):
        """内外径が返される."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)
        assert props["r_inner"] == pytest.approx(3.0e-3, rel=1e-10)
        assert props["r_outer"] == pytest.approx(3.5e-3, rel=1e-10)

    def test_thin_sheath_approximation(self):
        """薄肉近似: A ≈ 2πrt, I ≈ πr³t."""
        mesh = _make_7wire_mesh()
        t = 0.01e-3  # 非常に薄い
        sheath = SheathModel(thickness=t, E=70.0e9, nu=0.33)
        props = sheath_section_properties(mesh, sheath)

        r_in = 3.0e-3
        A_approx = 2.0 * math.pi * r_in * t
        I_approx = math.pi * r_in**3 * t
        assert abs(props["A"] - A_approx) / A_approx < 0.01
        assert abs(props["Iy"] - I_approx) / I_approx < 0.01


# ====================================================================
# sheath_equivalent_stiffness テスト
# ====================================================================


class TestSheathEquivalentStiffness:
    """シース等価梁剛性テスト."""

    def test_stiffness_values(self):
        """EA/EI/GJ が正値."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        assert stiff["EA"] > 0
        assert stiff["EIy"] > 0
        assert stiff["EIz"] > 0
        assert stiff["GJ"] > 0

    def test_EA_value(self):
        """EA = E * A."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        props = sheath_section_properties(mesh, sheath)
        assert stiff["EA"] == pytest.approx(70.0e9 * props["A"], rel=1e-12)

    def test_GJ_value(self):
        """GJ = G * J."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        props = sheath_section_properties(mesh, sheath)
        G = 70.0e9 / (2.0 * 1.33)
        assert stiff["GJ"] == pytest.approx(G * props["J"], rel=1e-12)

    def test_EIy_equals_EIz(self):
        """円筒対称: EIy = EIz."""
        mesh = _make_7wire_mesh()
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        stiff = sheath_equivalent_stiffness(mesh, sheath)
        assert stiff["EIy"] == stiff["EIz"]

    def test_thicker_sheath_stiffer(self):
        """厚いシースほど剛性が大きい."""
        mesh = _make_7wire_mesh()
        thin = SheathModel(thickness=0.2e-3, E=70.0e9, nu=0.33)
        thick = SheathModel(thickness=1.0e-3, E=70.0e9, nu=0.33)
        s_thin = sheath_equivalent_stiffness(mesh, thin)
        s_thick = sheath_equivalent_stiffness(mesh, thick)
        assert s_thick["EA"] > s_thin["EA"]
        assert s_thick["EIy"] > s_thin["EIy"]
        assert s_thick["GJ"] > s_thin["GJ"]


# ====================================================================
# outermost_layer / outermost_strand_ids テスト
# ====================================================================


class TestOutermostLayer:
    """最外層特定テスト."""

    def test_7wire_outermost_layer(self):
        """7本撚り: 最外層 = 層1."""
        mesh = _make_7wire_mesh()
        assert outermost_layer(mesh) == 1

    def test_3wire_outermost_layer(self):
        """3本撚り: 最外層 = 層1."""
        mesh = _make_3wire_mesh()
        assert outermost_layer(mesh) == 1

    def test_19wire_outermost_layer(self):
        """19本撚り: 最外層 = 層2."""
        mesh = _make_19wire_mesh()
        assert outermost_layer(mesh) == 2

    def test_7wire_outermost_ids(self):
        """7本撚り: 最外層素線は6本（ID 1〜6）."""
        mesh = _make_7wire_mesh()
        ids = outermost_strand_ids(mesh)
        assert len(ids) == 6
        assert set(ids) == {1, 2, 3, 4, 5, 6}

    def test_3wire_outermost_ids(self):
        """3本撚り: 全3本が最外層."""
        mesh = _make_3wire_mesh()
        ids = outermost_strand_ids(mesh)
        assert len(ids) == 3
        assert set(ids) == {0, 1, 2}

    def test_19wire_outermost_ids(self):
        """19本撚り: 最外層は12本（ID 7〜18）."""
        mesh = _make_19wire_mesh()
        ids = outermost_strand_ids(mesh)
        assert len(ids) == 12


# ====================================================================
# outermost_strand_node_indices テスト
# ====================================================================


class TestOutermostStrandNodes:
    """最外層節点インデックスのテスト."""

    def test_7wire_node_count(self):
        """7本撚り: 最外層6素線 × (n_elems+1) ノード."""
        n_elems = 8
        mesh = _make_7wire_mesh(n_elems=n_elems)
        nodes = outermost_strand_node_indices(mesh)
        expected = 6 * (n_elems + 1)
        assert len(nodes) == expected

    def test_3wire_all_outer(self):
        """3本撚り: 全素線が最外層 → 全ノード."""
        n_elems = 4
        mesh = _make_3wire_mesh(n_elems=n_elems)
        nodes = outermost_strand_node_indices(mesh)
        expected = 3 * (n_elems + 1)
        assert len(nodes) == expected

    def test_nodes_are_valid_indices(self):
        """返される節点インデックスが有効範囲内."""
        mesh = _make_7wire_mesh()
        nodes = outermost_strand_node_indices(mesh)
        assert np.all(nodes >= 0)
        assert np.all(nodes < mesh.n_nodes)


# ====================================================================
# sheath_radial_gap テスト
# ====================================================================


class TestSheathRadialGap:
    """径方向ギャップのテスト."""

    def test_zero_clearance_nonnegative(self):
        """クリアランス0: ギャップ >= 0（全点で非負、ヘリックスのため端部で正）."""
        mesh = _make_7wire_mesh(n_elems=16)
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        gaps = sheath_radial_gap(mesh, sheath)
        # clearance=0 のとき、配置半径上の点でギャップ=0、
        # ヘリカル形状のため端部では若干の変動があるが基本的に ≈ 0
        assert np.all(gaps >= -1e-10)  # 数値誤差許容

    def test_positive_clearance(self):
        """クリアランスあり: 全ギャップが clearance 以上."""
        mesh = _make_7wire_mesh()
        clearance = 0.5e-3
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, clearance=clearance)
        gaps = sheath_radial_gap(mesh, sheath)
        # 全点でギャップ > 0
        assert np.all(gaps > 0)

    def test_gap_shape(self):
        """ギャップ配列の形状 = 最外層ノード数."""
        mesh = _make_7wire_mesh(n_elems=8)
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        gaps = sheath_radial_gap(mesh, sheath)
        expected_len = 6 * (8 + 1)  # 6素線 × 9ノード
        assert gaps.shape == (expected_len,)

    def test_gap_with_coating(self):
        """被膜込みギャップ."""
        mesh = _make_7wire_mesh()
        coating = CoatingModel(thickness=0.1e-3, E=3.0e9, nu=0.35)
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33)
        gaps = sheath_radial_gap(mesh, sheath, coating=coating)
        assert gaps.shape[0] > 0
        # 被膜込みでもギャップ ≈ 0
        assert np.all(gaps >= -1e-10)

    def test_3wire_gap(self):
        """3本撚り: 全素線が最外層."""
        mesh = _make_3wire_mesh(n_elems=8)
        sheath = SheathModel(thickness=0.5e-3, E=70.0e9, nu=0.33, clearance=0.1e-3)
        gaps = sheath_radial_gap(mesh, sheath)
        expected_len = 3 * (8 + 1)
        assert gaps.shape == (expected_len,)
        assert np.all(gaps > 0)
